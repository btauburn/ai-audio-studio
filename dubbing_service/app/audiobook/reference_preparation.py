"""Bounded automatic reference preparation; preserves source and existing identities."""
from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf

from . import reference_quality, reference_separation, reference_source_qc
from .reference_codec import _run

_LOCK = Lock()
_PATH = Path(__file__).resolve()
_SOURCE_SHA = hashlib.sha256(_PATH.read_bytes()).hexdigest()
_CACHE_BYTES = 16_000_000
_REQUIRED = {"decode", "clipping", "signal", "speech", "asr", "transcript_alignment", "crop_coverage"}
_RAW_REQUIRED = _REQUIRED - {"transcript_alignment"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _child(args):
    python = _PATH.parents[3] / "environments/faster_qwen_base032/.venv/Scripts/python.exe"
    if not python.is_file() or python.is_symlink():
        raise ValueError("Existing source-QC environment unavailable; no installation attempted")
    return json.loads(_run([str(python), str(Path(reference_source_qc.__file__)), *args],
                           timeout_seconds=90, max_stdout_bytes=32_000))


def _manifest():
    if _sha(_PATH.read_bytes()) != _SOURCE_SHA:
        raise ValueError("Automatic preparation changed; restart Studio")
    if _sha(Path(reference_source_qc.__file__).read_bytes()) != reference_source_qc.SOURCE_SHA:
        raise ValueError("Source classifier changed; restart Studio before preparation")
    separator_path = Path(reference_separation.__file__)
    if _sha(separator_path.read_bytes()) != reference_separation._PIPELINE_SHA256:
        raise ValueError("Reference separation changed; restart Studio")
    weights = reference_separation._weights()
    weight_sha = _sha(weights.read_bytes()) if weights.is_file() and not weights.is_symlink() else None
    return {"schema": "automatic-reference-preparation-v1", "source_sha256": _SOURCE_SHA,
            "classifier": _child(["--manifest"]),
            "separator_source_sha256": reference_separation._PIPELINE_SHA256,
            "separator_weights_sha256": weight_sha,
            "quality": reference_quality._pipeline()}


def _classify(wave, rate):
    with tempfile.TemporaryDirectory(prefix="reference-source-qc-") as folder:
        path = Path(folder) / "input.npz"
        np.savez(path, wave=wave, sample_rate=np.asarray(rate, dtype=np.int64))
        report = _child([str(path)])
    if (report.get("status") not in {"pass", "contaminated", "review"}
            or not isinstance(report.get("windows"), list) or not report["windows"]
            or not isinstance(report.get("contaminated"), bool)
            or report.get("input_pcm_sha256") != _sha(wave.tobytes())
            or report.get("sample_rate") != rate):
        raise ValueError("Invalid source-QC result")
    return report


def _quality(directory, wave, rate, text, language, origin):
    raw = io.BytesIO()
    sf.write(raw, wave.mean(1) if wave.ndim == 2 else wave, rate, format="WAV", subtype="FLOAT")
    evidence = reference_quality.verify_bytes(directory, raw.getvalue(), text,
                                               language=language, expected_origin=origin)
    evidence.pop("cache_hit", None)
    evidence["input_pcm_sha256"] = _sha((wave.mean(1) if wave.ndim == 2 else wave).tobytes())
    evidence["sample_rate"] = rate
    return evidence


def _technical_pass(report, required=_REQUIRED):
    return all(report.get("checks", {}).get(k, {}).get("status") == "pass" for k in required)


def _quality_binding(report, request, pcm_sha, rate):
    expected = request["text"].strip() if request["text"] else None
    return (isinstance(report, dict) and report.get("input_pcm_sha256") == pcm_sha
            and report.get("sample_rate") == rate and report.get("pipeline") == request["pipeline"]["quality"]
            and report.get("expected_text") == expected
            and report.get("expected_text_sha256") == (_sha(expected.encode()) if expected else None)
            and report.get("language") == request["language"]
            and report.get("expected_text_origin") == request["expected_origin"])


def _window_binding(report, duration):
    rows = report.get("windows", [])
    return (bool(rows) and abs(min(r["start_seconds"] for r in rows)) < .001
            and abs(max(r["start_seconds"] + r["duration_seconds"] for r in rows) - duration) < .001)


def _validate_record(record, request, output, rate):
    evidence, processing = record["evidence"], record["processing"]
    if (not isinstance(evidence, dict) or (processing is not None and not isinstance(processing, dict))
            or not isinstance(evidence.get("raw_source_qc"), dict) or not isinstance(evidence.get("raw_quality"), dict)):
        raise ValueError("Cached automatic evidence schema invalid")
    if evidence.get("pipeline") != request["pipeline"]:
        raise ValueError("Cached automatic evidence pipeline mismatch")
    raw_qc = evidence["raw_source_qc"]
    if (raw_qc.get("pipeline") != request["pipeline"]["classifier"]
            or raw_qc.get("input_pcm_sha256") != request["input_pcm_sha256"]
            or raw_qc.get("sample_rate") != request["sample_rate"]
            or not _window_binding(raw_qc, request["shape"][0] / request["sample_rate"])):
        raise ValueError("Cached raw source-QC binding mismatch")
    for field in ("raw_source_qc", "post_source_qc"):
        if field in evidence:
            classified = evidence[field]
            if not isinstance(classified, dict) or not isinstance(classified.get("windows"), list):
                raise ValueError("Cached classifier evidence schema invalid")
            derived = reference_source_qc.evaluate_windows(classified["windows"])
            if any(classified.get(k) != v for k, v in derived.items()):
                raise ValueError("Cached classifier decision differs from measured windows")
    raw_quality = evidence["raw_quality"]
    if not _quality_binding(raw_quality, request, request["input_mono_pcm_sha256"], request["sample_rate"]):
        raise ValueError("Cached raw technical evidence binding mismatch")
    attempted = evidence.get("isolation_attempts")
    should_isolate = raw_qc["status"] != "review" and _technical_pass(raw_quality, _RAW_REQUIRED) and raw_qc["contaminated"]
    failed = evidence.get("processing_failed") is True
    if attempted != int(should_isolate) or evidence.get("processed_reference") is not bool(should_isolate and not failed):
        raise ValueError("Cached isolation decision differs from raw checks")
    if failed:
        if (not should_isolate or processing is not None or evidence.get("status") != "hold"
                or evidence.get("failure_stage") not in {"separation", "post_source_qc", "post_quality"}
                or rate != request["sample_rate"] or _sha(output.tobytes()) != request["input_pcm_sha256"]
                or evidence.get("human_verified") is not False or evidence.get("production_approved") is not False
                or evidence.get("automatic_source_approval") is not False):
            raise ValueError("Cached failure evidence malformed")
        return
    if should_isolate:
        post_qc, post_quality = evidence["post_source_qc"], evidence["post_quality"]
        if not isinstance(processing, dict) or not isinstance(post_qc, dict) or not isinstance(post_quality, dict):
            raise ValueError("Cached processing evidence schema invalid")
        out_sha = _sha(output.tobytes())
        if (post_qc.get("pipeline") != request["pipeline"]["classifier"]
                or post_qc.get("input_pcm_sha256") != out_sha or post_qc.get("sample_rate") != rate
                or not _window_binding(post_qc, len(output) / rate)
                or not _quality_binding(post_quality, request, out_sha, rate)
                or processing.get("input_pcm_sha256") != request["input_pcm_sha256"]
                or processing.get("output_pcm_sha256") != out_sha
                or processing.get("input_sample_rate") != request["sample_rate"]
                or processing.get("output_sample_rate") != rate
                or processing.get("output_samples") != len(output)
                or processing.get("pipeline_sha256") != request["pipeline"]["separator_source_sha256"]
                or processing.get("model_sha256") != request["pipeline"]["separator_weights_sha256"]):
            raise ValueError("Cached processed reference provenance mismatch")
        ready = post_qc["status"] == "pass" and _technical_pass(post_quality) and post_quality.get("status") == "pass"
    else:
        if processing is not None or rate != request["sample_rate"] or _sha(output.tobytes()) not in {
                request["input_pcm_sha256"], request["input_mono_pcm_sha256"]}:
            raise ValueError("Cached unprocessed PCM changed")
        ready = raw_qc["status"] == "pass" and _technical_pass(raw_quality) and raw_quality.get("status") == "pass"
        if ready and (output.ndim != 1 or _sha(output.tobytes()) != request["input_mono_pcm_sha256"]):
            raise ValueError("Cached clean reference must retain exact mono frames")
    ready = ready and request["expected_origin"] in {"provided", "user_correction"}
    if (evidence.get("status") != ("pass" if ready else "hold")
            or evidence.get("human_verified") is not False
            or evidence.get("automatic_source_approval") is not False
            or evidence.get("production_approved") is not False):
        raise ValueError("Cached reference decision differs from required checks")


def _prepare(directory, wave, rate, text, language, origin, pipeline):
    raw_qc = _classify(wave, rate)
    if raw_qc.get("pipeline") != pipeline["classifier"]:
        raise ValueError("Source-QC provenance changed")
    raw_quality = _quality(directory, wave, rate, text, language, origin)
    evidence = {"schema": pipeline["schema"], "pipeline": pipeline, "raw_source_qc": raw_qc,
                "raw_quality": raw_quality, "isolation_attempts": 0, "processed_reference": False,
                "human_verified": False, "automatic_source_approval": False, "production_approved": False}
    if raw_qc["status"] == "review":
        return wave, rate, None, evidence | {"status": "hold", "reason": "Ambiguous speakers, overlap or speech; no isolation retry"}
    if not _technical_pass(raw_quality, _RAW_REQUIRED):
        return wave, rate, None, evidence | {"status": "hold", "reason": "ASR/alignment or technical source checks require review"}
    processing = None
    if raw_qc["contaminated"]:
        if pipeline["separator_weights_sha256"] != reference_separation.MODEL_SHA256:
            raise ValueError("Pinned separation model unavailable or changed")
        original_wave, original_rate = wave, rate
        evidence["isolation_attempts"] = 1
        stage = "separation"
        try:
            wave, rate, processing = reference_separation.isolate_speech(wave, rate)
            evidence.update(processed_reference=True, processing_completed=True)
            stage = "post_source_qc"
            post_qc = _classify(wave, rate)
            stage = "post_quality"
            post_quality = _quality(directory, wave, rate, text, language, origin)
            if post_qc.get("pipeline") != pipeline["classifier"]:
                raise ValueError("Post-isolation source-QC provenance changed")
            evidence.update(post_source_qc=post_qc, post_quality=post_quality)
            ready = (post_qc["status"] == "pass" and _technical_pass(post_quality)
                     and post_quality.get("status") == "pass")
        except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
            return original_wave, original_rate, None, evidence | {
                "status": "hold", "processing_failed": True, "failure_stage": stage,
                "processed_reference": False,
                "reason": "Single isolation attempt unavailable/failed; original retained: " + str(exc)[:250]}
    else:
        if wave.ndim == 2:
            wave = wave.mean(1)
        ready = raw_qc["status"] == "pass" and raw_quality.get("status") == "pass"
    ready = ready and origin in {"provided", "user_correction"}
    return wave, rate, processing, evidence | {
        "status": "pass" if ready else "hold",
        "reason": "Machine source checks passed; human voice acceptance still required" if ready
                  else "Post-isolation checks require review; no second pass or regeneration"}


def _cached_pcm(raw):
    """Our writer uses stored ZIP members; reject compressed/oversized payloads before NumPy allocation."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != 2 or {m.filename for m in members} != {"wave.npy", "sample_rate.npy"}
                or any(m.compress_type != zipfile.ZIP_STORED for m in members)
                or sum(m.file_size for m in members) > 12_000_000):
            raise ValueError("Cached PCM ZIP schema/compression/size invalid")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError("Unsupported cached PCM NPY header")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                if member.filename == "wave.npy":
                    if (dtype != np.dtype("float32") or fortran or len(shape) not in (1, 2)
                            or any(not isinstance(n, int) or n < 1 for n in shape)
                            or (len(shape) == 2 and shape[1] != 2)
                            or shape[0] > 30 * 96000):
                        raise ValueError("Cached PCM shape/dtype exceeds reference bounds")
                elif shape != () or dtype != np.dtype("int64"):
                    raise ValueError("Cached sample rate malformed")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return data["wave"].copy(), int(data["sample_rate"])


def prepare(directory, wave, sample_rate, expected_text, *, language="en", expected_origin="provided",
            storage_budget_bytes=24_000_000):
    """One serial pass. Unavailable checks hold; cached artifacts never imply human acceptance."""
    if (not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000
            or wave.dtype != np.float32 or wave.ndim not in (1, 2)
            or (wave.ndim == 2 and wave.shape[1] != 2)
            or not 3 <= len(wave) / sample_rate <= 30 or not np.isfinite(wave).all()):
        raise ValueError("Invalid automatic reference crop")
    if expected_text is not None and (not isinstance(expected_text, str) or not 1 <= len(expected_text.strip()) <= 1000):
        raise ValueError("Invalid automatic reference transcript")
    if (isinstance(storage_budget_bytes, bool) or not isinstance(storage_budget_bytes, int)
            or not 1024 <= storage_budget_bytes <= 100_000_000):
        raise ValueError("Invalid automatic preparation storage budget")
    directory = Path(directory)
    if len(wave) / sample_rate > 10:
        return wave, sample_rate, None, {
            "schema": "automatic-reference-preparation-v1", "status": "hold",
            "reason": "Global speaker consistency unavailable for crops longer than10s; explicit speaker review required",
            "isolation_attempts": 0, "processed_reference": False, "human_verified": False,
            "automatic_source_approval": False, "production_approved": False}, False
    with _LOCK:
        evidence = None
        stage = "manifest"
        try:
            pipeline = _manifest()
            request = {"pipeline": pipeline, "input_pcm_sha256": _sha(wave.tobytes()),
                       "input_mono_pcm_sha256": _sha((wave.mean(1) if wave.ndim == 2 else wave).tobytes()),
                       "shape": list(wave.shape), "sample_rate": sample_rate, "text": expected_text,
                       "language": language, "expected_origin": expected_origin}
            key = _sha(_json(request))
            cache = directory / "automatic-preparation"
            cache.mkdir(parents=True, exist_ok=True)
            if cache.is_symlink():
                raise ValueError("Automatic preparation cache must be a local directory")
            meta, audio = cache / (key + ".json"), cache / (key + ".npz")
            if meta.exists():
                if meta.is_symlink() or audio.is_symlink() or meta.stat().st_size > 500_000 or audio.stat().st_size > 12_000_000:
                    raise ValueError("Automatic preparation cache invalid or oversized")
                record = json.loads(meta.read_bytes())
                record_sha = record.pop("record_sha256")
                if _sha(_json(record)) != record_sha:
                    raise ValueError("Automatic preparation cached evidence changed")
                if record["request"] != request or _sha(audio.read_bytes()) != record["output_sha256"]:
                    raise ValueError("Automatic preparation cache provenance mismatch")
                output, rate = _cached_pcm(audio.read_bytes())
                if (output.dtype != np.float32 or output.ndim not in (1, 2) or not np.isfinite(output).all()
                        or not 8000 <= rate <= 96000 or not 3 <= len(output) / rate <= 30
                        or _sha(output.tobytes()) != record["output_pcm_sha256"]):
                    raise ValueError("Automatic preparation cached PCM changed")
                _validate_record(record, request, output, rate)
                return output, rate, record["processing"], record["evidence"], True
            outside = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and p.parent != cache)
            reserve = min(_CACHE_BYTES, max(0, storage_budget_bytes - outside - 2_000_000))
            if len(wave) * 2 * 4 + 500_000 > reserve:
                raise ValueError("Storage budget cannot retain preparation evidence; no isolation attempted")
            stage = "preparation"
            output, rate, processing, evidence = _prepare(directory, wave, sample_rate, expected_text,
                                                          language, expected_origin, pipeline)
            stage = "final_provenance"
            if _manifest() != pipeline:
                raise ValueError("Automatic preparation provenance changed during operation")
            buf = io.BytesIO()
            np.savez(buf, wave=output, sample_rate=np.asarray(rate, dtype=np.int64))
            raw = buf.getvalue()
            record = {"request": request, "output_sha256": _sha(raw), "output_pcm_sha256": _sha(output.tobytes()),
                      "processing": processing, "evidence": evidence}
            stage = "cache_validation"
            _validate_record(record, request, output, rate)
            record["record_sha256"] = _sha(_json(record))
            json_raw = _json(record)
            if len(raw) > 12_000_000 or len(json_raw) > 500_000:
                raise ValueError("Automatic preparation output exceeds storage bounds")
            # Keep cache bounded; remove only named digest artifacts in this fixed directory.
            entries = [p for p in cache.iterdir() if re.fullmatch(r"[a-f0-9]{64}\.(json|npz)", p.name) and not p.is_symlink()]
            # Honor the owner's total discovery budget as well as the subcache cap.
            maximum = storage_budget_bytes
            outside_bytes = sum(p.stat().st_size for p in directory.rglob("*")
                                if p.is_file() and p.parent != cache)
            budget = min(_CACHE_BYTES, max(0, maximum - outside_bytes - 2_000_000))
            if len(raw) + len(json_raw) > budget:
                # Valid uncached evidence is useful even when the storage budget is full.
                return output, rate, processing, evidence, False
            while sum(p.stat().st_size for p in entries) + len(raw) + len(json_raw) > budget and entries:
                oldest = min(entries, key=lambda p: p.stat().st_mtime)
                # Remove metadata first, then its paired PCM. No half-entry survives eviction.
                for suffix in (".json", ".npz"):
                    paired = cache / (oldest.stem + suffix)
                    if paired.exists() and not paired.is_symlink():
                        paired.unlink()
                    entries = [p for p in entries if p != paired]
            stage = "cache_write"
            # Metadata is the commit marker. A crash leaves an ignored orphan that is overwritten.
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                staged_path = Path(stream.name)
                stream.write(raw)
            try:
                staged_path.replace(audio)
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                    staged_path = Path(stream.name)
                    stream.write(json_raw)
                staged_path.replace(meta)
            finally:
                staged_path.unlink(missing_ok=True)
            return output, rate, processing, evidence, False
        except (ValueError, OSError, KeyError, TypeError, AttributeError, ImportError, RuntimeError, zipfile.BadZipFile) as exc:
            return wave, sample_rate, None, {
                  **(evidence if isinstance(evidence, dict) else {}),
                  "schema": "automatic-reference-preparation-v1",
                  "status": "hold", "reason": "Automatic preparation unavailable: " + str(exc)[:300],
                  "isolation_attempts": evidence.get("isolation_attempts", 0) if isinstance(evidence, dict) else 0,
                  "failure_stage": stage, "original_retained": True,
                  "processed_reference": False, "human_verified": False,
                  "automatic_source_approval": False, "production_approved": False}, False


def transcript_origin(receipt, expected, correction=None):
    """Same trusted-origin rules as the existing acquisition-quality owner."""
    embedded = receipt.get("transcript_evidence") or {}
    provided = (receipt.get("reference_text") or "").strip()
    origin = "provided" if provided and expected == provided and not embedded.get("machine_generated") else "unverified_entry"
    if embedded.get("machine_generated") and expected == (embedded.get("text") or "").strip():
        origin = "machine_draft"
    if correction and expected == correction["text"].strip():
        origin = "user_correction"
    return origin


def prepare_acquisition(project_id, acquisition_id, acquisition_sha256, expected_text=None):
    """Owner tool: derive a new acquisition; never assign, rate, favorite, or edit an identity."""
    from dubbing_service.app.core.config import safe_project_root

    from .performer_discovery import DiscoveryLimits, acquire_reference
    from .reference_transcripts import acquisition_audio
    from .voice_lab_audition import _read_verified, _safe_path
    acquisition_audio(project_id, acquisition_id, acquisition_sha256)
    directory = safe_project_root(project_id) / "voice_lab/discovery"
    receipt = json.loads(_read_verified(_safe_path(directory, f"acquisitions/{acquisition_id}/acquisition.json"), acquisition_sha256))
    if (receipt.get("clone_ready") is True or receipt.get("source_processing")
            or receipt.get("automatic_preparation", {}).get("isolation_attempts") == 1):
        return {"status": "preserved", "acquisition_id": acquisition_id,
                "acquisition_sha256": acquisition_sha256,
                "reason": "Existing approved or processed reference retained; no processing or identity change"}
    original = _safe_path(directory, receipt["original_path"])
    _read_verified(original, receipt["original_sha256"])
    from .reference_transcripts import _correction
    expected = expected_text.strip() if expected_text is not None else (receipt.get("reference_text") or "").strip() or None
    correction = _correction(directory, acquisition_id, acquisition_sha256,
                             receipt["reference_audio_sha256"], None, require_asr_match=False)
    origin = transcript_origin(receipt, expected, correction)
    # Missing originals must never cause a download from this preparation tool.
    result = acquire_reference(directory, receipt["source"],
        intended_use_evidence=receipt["eligibility"].get("intended_use_evidence") or {},
        transcript=expected, transcript_origin=origin,
        start_seconds=receipt["start_seconds"], end_seconds=receipt["end_seconds"],
        automatic_preparation=True, cached_original_only=True, limits=DiscoveryLimits(max_requests=0, max_retries=0, max_download_bytes=32_000_000, max_storage_bytes=48_000_000))
    return {"status": "prepared" if result["clone_ready"] else "hold", "source_acquisition_id": acquisition_id,
            "source_acquisition_sha256": acquisition_sha256, **result,
            "identity_changed": False, "selection_changed": False, "audio_generation_triggered": False}

