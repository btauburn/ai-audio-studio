"""Qwen identities: generated or explicitly eligible reference -> Base reuse.

External references require an owner-created, hash-bound acquisition receipt in
this project's fixed Voice Lab directory. No caller-supplied path or URL is used.
The NPZ contains numeric tensors and JSON bytes only; pickle is never enabled.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import re
import uuid
import time
from collections import OrderedDict
from threading import Lock
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from dubbing_service.app.core.config import safe_project_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoped_file(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("voice artifact must be project-relative")
    candidate = root / path
    resolved = candidate.resolve()
    resolved.relative_to(root.resolve())
    # Reject links even when they point within the project: identities must be
    # materialized, portable project assets rather than mutable cache aliases.
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("linked voice artifact is forbidden")
        cursor = cursor.parent
    if must_exist and not resolved.is_file():
        raise ValueError("voice artifact is missing")
    return resolved


def identity_directory(project_id: str, identity_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", identity_id):
        raise ValueError("invalid character voice identity id")
    root = safe_project_root(project_id).resolve()
    folder = scoped_file(root, f"casting/voices/{identity_id}", must_exist=False)
    folder.mkdir(parents=True, exist_ok=True)
    return root, folder


def save_prompt(path: Path, item: Any, metadata: dict[str, Any]) -> None:
    def array(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy()
    ref_code = array(item.ref_code) if item.ref_code is not None else np.empty((0,), dtype=np.int64)
    embedding = array(item.ref_spk_embedding.float())
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite speaker embedding")
    payload = dict(metadata, ref_text=item.ref_text, x_vector_only_mode=bool(item.x_vector_only_mode), icl_mode=bool(item.icl_mode))
    if path.exists():
        raise ValueError("persistent prompt already exists")
    # An interrupted writer leaves only an unreferenced staging artifact; the
    # immutable published prompt is either complete or absent on restart.
    staged = path.with_name(path.name + ".stage-" + uuid.uuid4().hex)
    with staged.open("xb") as stream:
        np.savez_compressed(stream, ref_code=ref_code, ref_spk_embedding=embedding,
                            metadata=np.frombuffer(json.dumps(payload, sort_keys=True).encode("utf-8"), dtype=np.uint8))
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise ValueError("persistent prompt already exists")
    os.replace(staged, path)


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    staged = path.with_name(path.name + ".stage-" + uuid.uuid4().hex)
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise ValueError("immutable voice metadata already exists")
    os.replace(staged, path)


def load_prompt(path: Path, expected_hash: str, *, identity_id: str, reference_hash: str) -> tuple[Any, dict[str, Any]]:
    if not expected_hash or sha256(path) != expected_hash:
        raise ValueError("persistent voice prompt checksum mismatch")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("persistent voice prompt exceeds size limit")
    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"ref_code", "ref_spk_embedding", "metadata"}:
            raise ValueError("invalid persistent voice prompt schema")
        meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
        embedding = data["ref_spk_embedding"].copy()
        codes = data["ref_code"].copy()
    if meta.get("schema") != "qwen-original-prompt-v1" or meta.get("identity_id") != identity_id or meta.get("reference_audio_sha256") != reference_hash:
        raise ValueError("persistent voice prompt provenance mismatch")
    if embedding.ndim != 1 or not np.isfinite(embedding).all() or codes.dtype.kind not in "iu":
        raise ValueError("invalid persistent voice prompt tensors")
    item = VoiceClonePromptItem(
        ref_code=torch.from_numpy(codes) if codes.size else None,
        ref_spk_embedding=torch.from_numpy(embedding),
        x_vector_only_mode=bool(meta["x_vector_only_mode"]),
        icl_mode=bool(meta["icl_mode"]), ref_text=meta["ref_text"],
    )
    return item, meta


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _eligible_acquisition(root: Path, acquisition_id: str, acquisition_sha256: str, *, allow_draft: bool = False) -> tuple[dict, Path, Path]:
    """Revalidate the owner receipt and actual audio before trusting eligibility."""
    from dubbing_service.app.audiobook.performer_discovery import reference_eligibility, validate_audio_review
    if not re.fullmatch(r"[a-f0-9]{32}", acquisition_id) or not re.fullmatch(r"[a-f0-9]{64}", acquisition_sha256):
        raise ValueError("invalid reference acquisition identity")
    prefix = "voice_lab/discovery/"
    receipt_path = scoped_file(root, prefix + f"acquisitions/{acquisition_id}/acquisition.json")
    if receipt_path.stat().st_size > 256_000 or sha256(receipt_path) != acquisition_sha256:
        raise ValueError("reference acquisition receipt checksum mismatch")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if receipt.get("schema") != "voice-lab-acquisition-v1" or receipt.get("acquisition_id") != acquisition_id:
        raise ValueError("invalid reference acquisition receipt schema")
    eligibility = reference_eligibility(receipt.get("source", {}),
        intended_use_evidence=receipt.get("eligibility", {}).get("intended_use_evidence"))
    if not eligibility["acquisition_eligible"] or receipt.get("clone_ready") is not True or receipt.get("technical_qa", {}).get("status") != "pass":
        raise ValueError("reference acquisition lacks eligible use or passing technical QA")
    original_relative = receipt.get("original_path", "")
    reference_relative = receipt.get("reference_path", "")
    if not re.fullmatch(r"downloads/[a-f0-9]{64}\.audio", original_relative) or reference_relative != f"acquisitions/{acquisition_id}/reference.wav":
        raise ValueError("reference acquisition paths are outside their fixed scope")
    original = scoped_file(root, prefix + original_relative)
    reference = scoped_file(root, prefix + reference_relative)
    if original.stat().st_size > (32_000_000 if receipt.get("source_decode") else 16_000_000) or reference.stat().st_size > 16_000_000:
        raise ValueError("reference recording exceeds byte limit")
    if sha256(original) != receipt.get("original_sha256") or sha256(reference) != receipt.get("reference_audio_sha256"):
        raise ValueError("acquired reference recording checksum mismatch")
    # Recheck the actual original against the reviewed registry too. Receipt
    # hashes alone cannot establish that an official recording is this source.
    from dubbing_service.app.audiobook.official_recording_sources import verify_original
    verify_original(receipt.get("source", {}), original.read_bytes())
    transcript = receipt.get("reference_text", "")
    if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 1000 or hashlib.sha256(transcript.encode()).hexdigest() != receipt.get("transcript_sha256"):
        raise ValueError("reference transcript checksum mismatch")
    start, end = receipt.get("start_seconds"), receipt.get("end_seconds")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (start, end)) or start < 0 or not 3 <= end - start <= 30:
        raise ValueError("invalid reference crop boundaries")
    if validate_audio_review(receipt.get("audio_review"), original_sha256=receipt["original_sha256"],
                             transcript=transcript, start_seconds=start, end_seconds=end,
                             source_id=receipt["source"]["source_id"]):
        raise ValueError("reference transcript/speaker review is missing or stale")
    import soundfile as sf

    from dubbing_service.app.audiobook.reference_codec import decode_reference
    decoded = decode_reference(original, start, end, expected_decoder=receipt.get("source_decode"))
    if decoded.decoder is not None and receipt.get("source_decode") != decoded.decoder:
        raise ValueError("WebM reference lacks decoder provenance")
    expected, expected_rate = decoded.wave, decoded.sample_rate
    source_processing = receipt.get("source_processing")
    if source_processing is not None:
        if not isinstance(source_processing, dict):
            raise ValueError("invalid reference preprocessing recipe")
        from dubbing_service.app.audiobook.reference_separation import isolate_speech
        expected, expected_rate, replayed = isolate_speech(expected, expected_rate)
        if (source_processing != replayed
                or receipt.get("audio_review", {}).get("source_processing") != replayed
                or {"operation": "reference_vocals_isolation", "recipe": replayed} not in receipt.get("preprocessing", [])):
            raise ValueError("speech-isolation recipe or audio review failed exact replay")
    elif decoded.channels == 2:
        if {"operation": "stereo_to_mono_mean", "input_channels": 2} not in receipt.get("preprocessing", []):
            raise ValueError("Stereo reference conversion lacks recorded provenance")
        expected = expected.mean(axis=1)
    info = sf.info(str(reference))
    if (info.channels != 1 or info.samplerate != expected_rate
            or not 3 <= info.duration <= 30 + 1 / info.samplerate):
        raise ValueError("reference audio format or duration is unsuitable")
    samples, _rate = sf.read(str(reference), dtype="float32")
    if samples.shape != expected.shape or not np.array_equal(samples, expected):
        raise ValueError("reference audio does not match the claimed contiguous original crop and preprocessing")
    if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1 or np.mean(np.abs(samples) >= .999) > .001 or np.mean(np.abs(samples) < .001) > .65 or np.sqrt(np.mean(samples ** 2)) < .005:
        raise ValueError("reference audio failed clipping/silence checks")
    if (receipt.get("automatic_preparation") is not None or "narration_delivery_fit" in receipt
            or receipt["source"].get("requires_narration_delivery_review")):
        # New automatic/corpus references must retain narration fitness.
        # Explicit historical recipes are unchanged. No canonical clone-ready
        # automatic receipt existed when this gate was introduced.
        from dubbing_service.app.audiobook.reference_delivery_fit import evaluate
        stored = receipt.get("narration_delivery_fit")
        if not isinstance(stored, dict):
            raise ValueError("automatic reference lacks narration delivery evidence")
        assessed = evaluate(samples, _rate, source=receipt["source"],
            start_seconds=start, end_seconds=end, technical_status="pass", transcript=transcript,
            audio_assessment=(receipt.get("audio_review") or {}).get("narration_delivery"))
        if assessed != stored:
            raise ValueError("reference narration delivery evidence failed exact audio replay")
        if assessed.get("selected_reference") is not True:
            if not allow_draft or assessed.get("delivery_decision") != "review" or assessed.get("technical_decision") != "pass":
                raise ValueError("reference narration delivery evidence failed exact audio replay")
            from dubbing_service.app.audiobook.recorded_audition_policy import verify_policy
            verify_policy(receipt.get("audition_draft_policy"), samples, _rate, source=receipt["source"],
                original_sha256=receipt["original_sha256"], transcript=transcript,
                start_seconds=start, end_seconds=end)
    return receipt, original, reference


def _copy_immutable(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists():
        if sha256(target) != expected_hash:
            raise ValueError("immutable identity reference already has different bytes")
        return
    staged = target.with_name(target.name + ".stage-" + uuid.uuid4().hex)
    with source.open("rb") as inp, staged.open("xb") as out:
        while block := inp.read(1024 * 1024):
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    if sha256(staged) != expected_hash or target.exists():
        raise ValueError("reference changed during immutable import")
    os.replace(staged, target)


ACOUSTICS_SCHEMA = "identity-acoustics-v1"

#: Conditioning keys a request may carry. Anything else is refused rather than
#: stored, so a caller cannot smuggle an arbitrary blob into an identity receipt.
_CONDITIONING_KEYS = ("schema", "target_profile", "parameters", "instruction", "bias",
                      "shaping_bounds", "characterization", "reference")


def _conditioning(value: Any) -> dict[str, Any] | None:
    """Validate a conditioning payload down to numbers and reference provenance."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("conditioning must be an object")
    unknown = sorted(set(value) - set(_CONDITIONING_KEYS))
    if unknown:
        raise ValueError("conditioning carries unsupported keys: " + ", ".join(unknown))
    profile = value.get("target_profile")
    parameters = value.get("parameters")
    if not isinstance(profile, dict) or not profile or not isinstance(parameters, dict) or not parameters:
        raise ValueError("conditioning requires a target profile and derived parameters")
    numeric = {"f0_median_hz", "words_per_minute", "mean_pause_ms", "rms_dbfs", "dynamic_range_db",
               "brightness_hz", "breathiness", "rasp", "articulation_index", "sentence_end_f0_slope",
               "pitch_target_hz", "rate_words_per_minute", "pause_ms", "energy_dbfs", "rasp"}
    if not any(profile.get(key) is not None for key in numeric):
        raise ValueError("conditioning target profile carries no acoustic value")
    return json.loads(json.dumps(value))


def _reference_for(root: Path, folder: Path, receipt: dict[str, Any]) -> tuple[Path, str, dict[str, Any] | None]:
    """The reference the prompt is built from, plus its acoustics receipt if shaped.

    Once a conditioning pass has been applied, the shaped reference is the
    identity's reference: it is what the Base prompt encodes, and the original
    design output is kept beside it as the pre-conditioning record.
    """
    acoustics_path = folder / "acoustics.json"
    if acoustics_path.is_file():
        acoustics = json.loads(acoustics_path.read_text(encoding="utf-8"))
        if acoustics.get("identity_id") != receipt.get("identity_id"):
            raise ValueError("acoustics receipt belongs to a different identity")
        shaped = scoped_file(root, acoustics["reference_audio"])
        shaped_hash = sha256(shaped)
        if shaped_hash != acoustics.get("reference_audio_sha256"):
            raise ValueError("conditioned reference checksum mismatch")
        if acoustics.get("source_reference_audio_sha256") != receipt.get("reference_audio_sha256"):
            raise ValueError("conditioned reference does not descend from this identity's design output")
        return shaped, shaped_hash, acoustics
    reference = scoped_file(root, receipt["reference_audio"])
    reference_hash = sha256(reference)
    if reference_hash != receipt.get("reference_audio_sha256"):
        raise ValueError("original reference checksum mismatch")
    return reference, reference_hash, None


class CharacterVoiceRuntime:
    def __init__(self, project_root: Path, load_model: Callable[[str], Any]):
        self.project_root = project_root
        self.load_model = load_model
        # CPU-only prompts, bounded independently of the model's GPU lifecycle.
        # Cached tensors never escape: a generator receives its own copy.
        self._prompt_cache = OrderedDict()
        self._prompt_cache_bytes = 0
        self._prompt_cache_lock = Lock()

    def _cached_prompt(self, path: Path, expected_hash: str, *, identity_id: str,
                       reference_hash: str) -> tuple[Any, dict[str, Any]]:
        key = (path.resolve(), expected_hash, identity_id, reference_hash)
        with self._prompt_cache_lock:
            cached = self._prompt_cache.get(key)
            if cached is not None:
                # Verify actual bytes on EVERY hit, including same-size edits.
                # File timestamps are not an integrity or provenance decision.
                if sha256(path) != expected_hash:
                    self._prompt_cache_bytes -= self._prompt_cache.pop(key)[2]
                    raise ValueError("persistent voice prompt checksum mismatch")
                self._prompt_cache.move_to_end(key)
                return copy.deepcopy(cached[:2])
            item, metadata = load_prompt(path, expected_hash, identity_id=identity_id,
                                         reference_hash=reference_hash)
            size = len(json.dumps(metadata).encode("utf-8")) + sum(
                tensor.detach().cpu().numpy().nbytes
                for tensor in (item.ref_code, item.ref_spk_embedding) if tensor is not None)
            # Bound decompressed tensors, not ZIP size (zeros compress heavily).
            if size <= 32 * 1024 * 1024:
                while self._prompt_cache and (len(self._prompt_cache) >= 16
                        or self._prompt_cache_bytes + size > 32 * 1024 * 1024):
                    _, evicted = self._prompt_cache.popitem(last=False)
                    self._prompt_cache_bytes -= evicted[2]
                self._prompt_cache[key] = (item, metadata, size)
                self._prompt_cache_bytes += size
            return copy.deepcopy((item, metadata))

    def _model_metadata(self, kind: str) -> dict[str, Any]:
        manifest = self.project_root / "models" / "manifests" / f"qwen3_tts_{kind}.json"
        model = json.loads(manifest.read_text(encoding="utf-8"))
        return {"model": model["sources"]["model"]["model_id"],
                "model_revision": sha256(manifest), "runtime_version": importlib.metadata.version("qwen-tts")}

    def design(self, project_id: str, identity_id: str, instruction: str, text: str, seed: int,
               conditioning: dict[str, Any] | None = None) -> dict[str, Any]:
        root, folder = identity_directory(project_id, identity_id)
        receipt_path = folder / "reference.json"
        conditioning = _conditioning(conditioning)
        request_hash = hashlib.sha256(json.dumps([instruction, text, seed, conditioning],
                                                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("request_hash") != request_hash:
                raise ValueError("identity already designed with different inputs; use a new identity id")
            reference = scoped_file(root, receipt["reference_audio"])
            if sha256(reference) != receipt["reference_audio_sha256"]:
                raise ValueError("original reference checksum mismatch")
            return receipt
        if not instruction.strip() or not text.strip():
            raise ValueError("original design requires traits and reference text")
        intent_path = folder / "intent.json"
        if intent_path.exists():
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if intent.get("request_hash") != request_hash:
                raise ValueError("pending identity has different inputs; use a new identity id")
        else:
            _publish_json(intent_path, {"request_hash": request_hash, "identity_id": identity_id,
                                       "instruction": instruction, "reference_text": text, "seed": seed,
                                       "conditioning": conditioning})
        import soundfile as sf
        settings = {"seed": seed, "max_new_tokens": 384, "do_sample": True, "temperature": 0.8,
                    "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
                    "subtalker_dosample": True, "subtalker_top_k": 50,
                    "subtalker_top_p": 1.0, "subtalker_temperature": 0.9}
        reference = folder / "reference.wav"
        if not reference.exists():
            import torch
            model = self.load_model("voice_design")
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                torch.manual_seed(seed)
                waves, rate = model.generate_voice_design(text=text, language="English", instruct=instruction,
                                                         **{k: v for k, v in settings.items() if k != "seed"})
            if not waves or not len(waves[0]) or not np.isfinite(waves[0]).all():
                raise ValueError("voice design returned no valid audio")
            staged = folder / ("reference.stage-" + uuid.uuid4().hex + ".wav")
            sf.write(str(staged), waves[0], rate)
            if reference.exists():
                raise ValueError("original reference already exists")
            os.replace(staged, reference)
        if sf.info(str(reference)).frames <= 0:
            raise ValueError("original reference has no audio frames")
        receipt = {"identity_id": identity_id, "mechanism": "qwen_reference", "backend": "qwen3tts",
                   "reference_audio": reference.relative_to(root).as_posix(), "reference_text": text,
                   "reference_audio_sha256": sha256(reference), "design_instruction": instruction,
                   "request_hash": request_hash, "generation_settings": settings,
                   "conditioning": conditioning,
                   "provenance": {"origin": "generated_original_voice", "actor_audio_used": False,
                                  "design_model": self._model_metadata("voicedesign"),
                                  "attention": "sdpa", "dtype": "bfloat16",
                                  "created_at": datetime.now(UTC).isoformat()}}
        _publish_json(receipt_path, receipt)
        return receipt

    def acoustics(self, project_id: str, identity_id: str, conditioning: dict[str, Any] | None = None,
                  text: str | None = None) -> dict[str, Any]:
        """Move one generated original reference onto its reference acoustic target.

        The conditioning numbers — pitch, range, rate, pause, level, brightness,
        texture, accent and delivery anchors — are applied to the *audio* the
        identity is built from, not merely described to the design model: the
        reference is measured, a bounded shaping plan is derived against the
        target, the plan is applied, and the result is re-measured so the
        receipt can state what was actually achieved. The Base prompt is then
        extracted from this conditioned reference, which is why the reference
        layer changes the sound rather than the prose.

        Immutable and idempotent: the shaped reference and its receipt are
        published once and re-verified on every later call.
        """
        from service import voice_acoustics

        root, folder = identity_directory(project_id, identity_id)
        receipt_path = folder / "reference.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("identity_id") != identity_id:
            raise ValueError("design receipt belongs to a different identity")
        if receipt.get("provenance", {}).get("origin") != "generated_original_voice":
            raise ValueError("acoustic conditioning applies to generated original references only")
        target = _conditioning(conditioning if conditioning is not None else receipt.get("conditioning"))
        if target is None:
            raise ValueError("acoustic conditioning requires a target profile and parameters")

        acoustics_path = folder / "acoustics.json"
        shaped_path = folder / "reference-shaped.wav"
        if acoustics_path.exists():
            existing = json.loads(acoustics_path.read_text(encoding="utf-8"))
            if existing.get("conditioning_hash") != _digest_json(target):
                raise ValueError("identity already conditioned with a different target; use a new identity id")
            reference = scoped_file(root, existing["reference_audio"])
            if sha256(reference) != existing["reference_audio_sha256"]:
                raise ValueError("conditioned reference checksum mismatch")
            return existing

        source = scoped_file(root, receipt["reference_audio"])
        if sha256(source) != receipt["reference_audio_sha256"]:
            raise ValueError("original reference checksum mismatch")
        before = voice_acoustics.measure(source)
        bounds = target.get("shaping_bounds") or {}
        plan = voice_acoustics.plan_shaping(
            target["target_profile"], before,
            max_semitones=float(bounds.get("max_semitones", 3.0)),
            rate_range=tuple(bounds.get("rate_range", (0.85, 1.18))))
        staged = folder / ("reference-shaped.stage-" + uuid.uuid4().hex + ".wav")
        shaping = voice_acoustics.shape(source, staged, semitones=plan["semitones"], rate=plan["rate"])
        if shaped_path.exists():
            # Another pass published this reference between our read and write:
            # accept it only if it is the same audio we just produced.
            same = sha256(shaped_path) == sha256(staged)
            staged.unlink(missing_ok=True)
            if not same:
                raise ValueError("conditioned reference already published with different audio")
        else:
            os.replace(staged, shaped_path)
        after_measured = voice_acoustics.measure(shaped_path)
        after, reconciliation = voice_acoustics.reconcile_after(before, after_measured, shaping)
        comparison = voice_acoustics.compare(target["target_profile"], after)
        reach = voice_acoustics.achievement(comparison)
        moved = voice_acoustics.compare(before, after)
        receipt_out = {
            "schema": ACOUSTICS_SCHEMA, "identity_id": identity_id,
            "conditioning_hash": _digest_json(target),
            "source_reference_audio": receipt["reference_audio"],
            "source_reference_audio_sha256": receipt["reference_audio_sha256"],
            "reference_audio": shaped_path.relative_to(root).as_posix(),
            "reference_audio_sha256": sha256(shaped_path),
            "reference_text": receipt["reference_text"],
            "conditioning": target,
            "shaping_plan": plan,
            "shaping": shaping,
            "profile_before": before, "profile_after": after,
            "profile_after_measured": after_measured, "reconciliation": reconciliation,
            "target_comparison": comparison, "change": moved,
            # Three-valued on purpose: a key the bounded actuator cannot move is
            # neither a reach nor a shortfall, and reporting it as either turns
            # the actuator's scope into a verdict about the run.
            "reach": reach,
            "achieved": reach["reached_target"],
            "achieved_all_keys": not comparison["exceeded"],
            "engine_hints": voice_acoustics.parameters(after)["engine_hints"],
            "preprocessing": [{"operation": "bounded_pitch_shift", "semitones": plan["semitones"]},
                              {"operation": "bounded_rate_change", "rate": plan["rate"]}],
            "provenance": {"origin": "conditioned_generated_original_voice",
                           "actor_audio_used": False, "conditioned_at": datetime.now(UTC).isoformat(),
                           "method": before.get("method"), "text": text},
        }
        _publish_json(acoustics_path, receipt_out)
        return receipt_out

    def prepare(self, project_id: str, identity_id: str) -> dict[str, Any]:
        root, folder = identity_directory(project_id, identity_id)
        receipt_path = scoped_file(root, (folder / "reference.json").relative_to(root).as_posix())
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        origin = receipt.get("provenance", {}).get("origin")
        if receipt.get("identity_id") != identity_id or origin not in {"generated_original_voice", "eligible_reference_clone"}:
            raise ValueError("only generated original or verified eligible acquired references may become character voices")
        if origin == "eligible_reference_clone":
            acquired, _, _ = _eligible_acquisition(root, receipt["provenance"]["acquisition_id"], receipt["provenance"]["acquisition_sha256"], allow_draft=True)
            if acquired["reference_audio_sha256"] != receipt["reference_audio_sha256"] or acquired["reference_text"] != receipt["reference_text"]:
                raise ValueError("imported identity differs from its acquired source")
        reference, reference_hash, acoustics = _reference_for(root, folder, receipt)
        if origin == "generated_original_voice" and _conditioning(receipt.get("conditioning")) and acoustics is None:
            # A conditioned candidate without its conditioning pass is a silent
            # downgrade to a performer-name-only prompt, which is exactly the
            # outcome this stage exists to prevent.
            raise ValueError("conditioned identity must apply its acoustic conditioning before it can be prepared")
        prompt_path = folder / "prompt.npz"
        manifest = self._model_metadata("base")
        if not prompt_path.exists():
            model = self.load_model("base")
            items = model.create_voice_clone_prompt(ref_audio=str(reference), ref_text=receipt["reference_text"], x_vector_only_mode=False)
            save_prompt(prompt_path, items[0], {"schema": "qwen-original-prompt-v1", "identity_id": identity_id,
                                              "reference_audio_sha256": reference_hash,
                                              "origin": origin,
                                              **({"acoustics_sha256": _digest_json(acoustics)} if acoustics else {}),
                                              **({"acquisition_id": receipt["provenance"]["acquisition_id"],
                                                  "acquisition_sha256": receipt["provenance"]["acquisition_sha256"]} if origin == "eligible_reference_clone" else {}),
                                              **({"audition_draft_policy": receipt["provenance"]["audition_draft_policy"]} if receipt["provenance"].get("audition_draft_policy") else {}),
                                              **manifest})
        checksum = sha256(prompt_path)
        _, prompt_meta = self._cached_prompt(prompt_path, checksum, identity_id=identity_id, reference_hash=reference_hash)
        return {**receipt, "reference_audio": reference.relative_to(root).as_posix(),
                "reference_audio_sha256": reference_hash, "acoustics": acoustics,
                **{k: prompt_meta[k] for k in manifest},
                "generation_settings": {**receipt["generation_settings"], "render": {
                    "inference_policy_revision": "qwen-base-sampled-v1",
                    "do_sample": True, "subtalker_dosample": True, "temperature": 0.9,
                    "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
                    "subtalker_top_k": 50, "subtalker_top_p": 1.0, "subtalker_temperature": 0.9,
                    "seed_rule": "sha256(identity_id + text) first 8 hex digits",
                    "non_streaming_mode": True,
                    "token_budget": "max(128, ceil((word_count * 0.7 + 4) * 12.5)), capped at 2048",
                }},
                "performance_support": {"emotion": False, "intensity": False, "emphasis": False,
                                        "delivery": False, "pace": "project_postprocessing", "volume": "project_postprocessing"},
                "prompt_artifact": prompt_path.relative_to(root).as_posix(), "prompt_sha256": checksum}

    def import_reference(self, project_id: str, identity_id: str, acquisition_id: str,
                         acquisition_sha256: str) -> dict[str, Any]:
        """Import an eligible project acquisition; never fetch or synthesize here."""
        root = safe_project_root(project_id).resolve()
        acquired, original, reference = _eligible_acquisition(root, acquisition_id, acquisition_sha256, allow_draft=True)
        root, folder = identity_directory(project_id, identity_id)
        receipt_path = folder / "reference.json"
        request_hash = hashlib.sha256(json.dumps([acquisition_id, acquisition_sha256]).encode()).hexdigest()
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt.get("request_hash") != request_hash or receipt.get("provenance", {}).get("origin") != "eligible_reference_clone":
                raise ValueError("identity already has a different reference; use a new identity id")
            if sha256(scoped_file(root, receipt["reference_audio"])) != acquired["reference_audio_sha256"]:
                raise ValueError("imported reference checksum mismatch")
            return receipt
        intent_path = folder / "intent.json"
        if intent_path.exists():
            if json.loads(intent_path.read_text("utf-8")).get("request_hash") != request_hash:
                raise ValueError("pending identity has different inputs; use a new identity id")
        else:
            _publish_json(intent_path, {"identity_id": identity_id, "request_hash": request_hash,
                                       "origin": "eligible_reference_clone", "acquisition_id": acquisition_id})
        _copy_immutable(original, folder / "original.audio", acquired["original_sha256"])
        _copy_immutable(reference, folder / "reference.wav", acquired["reference_audio_sha256"])
        receipt = {"identity_id": identity_id, "mechanism": "qwen_reference", "backend": "qwen3tts",
                   "reference_audio": (folder / "reference.wav").relative_to(root).as_posix(),
                   "reference_audio_sha256": acquired["reference_audio_sha256"], "reference_text": acquired["reference_text"],
                   "original_audio": (folder / "original.audio").relative_to(root).as_posix(),
                   "original_audio_sha256": acquired["original_sha256"], "design_instruction": "",
                   "request_hash": request_hash, "generation_settings": {},
                   "provenance": {"origin": "eligible_reference_clone", "actor_audio_used": True,
                                  "source": acquired["source"], "eligibility": acquired["eligibility"],
                                  "acquisition_id": acquisition_id, "acquisition_sha256": acquisition_sha256,
                                  "preprocessing": acquired["preprocessing"], "technical_qa": acquired["technical_qa"],
                                  **({"audition_draft_policy": acquired["audition_draft_policy"], "human_fit": "unknown", "production_approved": False} if acquired.get("audition_draft_policy") else {}),
                                  "created_at": datetime.now(UTC).isoformat()}}
        _publish_json(receipt_path, receipt)
        return receipt

    def synthesize(self, project_id: str, identity_id: str, prompt_artifact: str, prompt_sha256: str,
                   reference_audio: str, reference_audio_sha256: str, text: str, language: str,
                   performance: dict[str, Any] | None = None) -> dict[str, Any]:
        synthesis_started = time.perf_counter()
        root = safe_project_root(project_id).resolve()
        expected_prefix = f"casting/voices/{identity_id}/"
        if not prompt_artifact.startswith(expected_prefix) or not reference_audio.startswith(expected_prefix):
            raise ValueError("voice artifact does not belong to the identity")
        reference = scoped_file(root, reference_audio)
        if sha256(reference) != reference_audio_sha256:
            raise ValueError("persistent voice reference checksum mismatch")
        prompt = scoped_file(root, prompt_artifact)
        item, meta = self._cached_prompt(prompt, prompt_sha256, identity_id=identity_id, reference_hash=reference_audio_sha256)
        from dubbing_service.app.audiobook import recorded_audition_policy as draft
        import soundfile as sf
        reference_samples, reference_rate = sf.read(str(reference), dtype="float32")
        draft_record = draft.reference_record(reference_samples, reference_rate)
        draft_allowed = False
        if draft_record is not None or meta.get("audition_draft_policy"):
            server_reference = json.loads(scoped_file(root, expected_prefix + "reference.json").read_text("utf-8"))
            provenance = server_reference.get("provenance", {})
            if (meta.get("origin") != "eligible_reference_clone" or provenance.get("origin") != "eligible_reference_clone"
                    or meta.get("audition_draft_policy") != provenance.get("audition_draft_policy")
                    or not provenance.get("audition_draft_policy")
                    or meta.get("acquisition_id") != provenance.get("acquisition_id")
                    or meta.get("acquisition_sha256") != provenance.get("acquisition_sha256")
                    or meta.get("ref_text") != server_reference.get("reference_text")
                    or server_reference.get("identity_id") != identity_id
                    or server_reference.get("reference_audio_sha256") != reference_audio_sha256
                    or meta.get("x_vector_only_mode") is not False or meta.get("icl_mode") is not True):
                raise ValueError("recorded draft provenance cannot be removed or replaced")
            with np.load(prompt, allow_pickle=False) as tensors:
                if not tensors["ref_code"].size:
                    raise ValueError("recorded draft requires full reference codes")
            draft_allowed = (draft.audition_context(root, identity_id, meta["acquisition_sha256"], text)
                or draft.human_casting_context(root, identity_id, meta["acquisition_sha256"]))
            if not draft_allowed:
                raise ValueError("unapproved recorded draft is restricted to its exact audition; human selection required for production")
        external = meta.get("origin") == "eligible_reference_clone"
        if external:
            acquired, _, _ = _eligible_acquisition(root, meta["acquisition_id"], meta["acquisition_sha256"], allow_draft=draft_allowed)
            if acquired["reference_audio_sha256"] != reference_audio_sha256:
                raise ValueError("persistent external reference differs from acquisition")
        current_metadata = self._model_metadata("base")
        if meta["model_revision"] != current_metadata["model_revision"] or meta["runtime_version"] != current_metadata["runtime_version"]:
            raise ValueError("Base model provenance changed; explicit voice migration is required")
        conditioning_seconds = time.perf_counter() - synthesis_started
        model_started = time.perf_counter()
        model = self.load_model("base")
        model_load_seconds = time.perf_counter() - model_started
        # Natural-language emotion is not a supported public Base argument.
        # Upstream recommends sampling: greedy Base decoding can repeat or fail
        # to terminate. A stable per-identity/text seed keeps rerenders reusable.
        import torch
        seed = int(hashlib.sha256((identity_id + text).encode("utf-8")).hexdigest()[:8], 16)
        token_budget = min(2048, max(128, math.ceil((len(text.split()) * 0.7 + 4) * 12.5)))
        render_settings = {"max_new_tokens": token_budget, "do_sample": True, "subtalker_dosample": True,
                           "temperature": 0.9, "non_streaming_mode": True,
                           "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
                           "subtalker_top_k": 50, "subtalker_top_p": 1.0, "subtalker_temperature": 0.9}
        generation_started = time.perf_counter()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(seed)
            waves, rate = model.generate_voice_clone(text=text, language=language, voice_clone_prompt=[item], **render_settings)
        model_generate_seconds = time.perf_counter() - generation_started
        if not waves or not len(waves[0]) or not np.isfinite(waves[0]).all():
            raise ValueError("persistent voice synthesis returned no valid audio")
        duration = len(waves[0]) / rate
        # A codec sequence reaching its finite budget is not a successful
        # audition: it may be truncated or looping. Preserve the identity and
        # flag the sample instead of accepting obviously overlong dialogue.
        if duration >= (token_budget / 12.5) * 0.95 or duration > max(10.0, len(text.split()) * 0.75 + 4):
            raise ValueError(f"persistent voice sample is overlong or reached its generation budget ({duration:.2f}s, {token_budget} tokens); review required")
        import soundfile as sf
        output = self.project_root / "outputs" / f"qwen_identity_{uuid.uuid4().hex}.wav"
        output.parent.mkdir(exist_ok=True)
        audio_write_started = time.perf_counter()
        sf.write(str(output), waves[0], rate)
        audio_write_seconds = time.perf_counter() - audio_write_started
        unsupported = [key for key in ("emotion", "intensity", "emphasis", "delivery") if (performance or {}).get(key)]
        return {"success": True, "message": "persistent eligible reference clone" if external else "persistent original character voice", "audio_file": str(output),
                "sample_rate": rate, "duration_seconds": len(waves[0]) / rate, "identity_id": identity_id,
                "prompt_sha256": prompt_sha256, "performance": performance or {},
                "stage_timings": {"schema": "qwen-character-stage-timings-v1", "clock": "time.perf_counter",
                                  "conditioning_seconds": round(conditioning_seconds, 6),
                                  "model_load_seconds": round(model_load_seconds, 6),
                                  "model_generate_seconds": round(model_generate_seconds, 6),
                                  "audio_write_seconds": round(audio_write_seconds, 6),
                                  "synthesis_seconds": round(time.perf_counter() - synthesis_started, 6)},
                "inference_policy_revision": "qwen-base-sampled-v1",
                "generation_settings": {**render_settings, "seed": seed, "inference_policy_revision": "qwen-base-sampled-v1"},
                "execution_profile": {"backend": "qwen3tts", "model_id": current_metadata["model"],
                                      "model_revision": current_metadata["model_revision"], "qwen_tts_version": current_metadata["runtime_version"],
                                      "attention_resolved": "sdpa", "dtype": "bfloat16", "mechanism": "persistent_eligible_reference" if external else "persistent_original_reference"},
                "unsupported_performance_fields": unsupported,
                "performance_note": "Base has no supported natural-language emotion instruction; pace and volume use existing project postprocessing."}
