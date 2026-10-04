"""One source-bound listening excerpt through Studio's existing Take pipeline.

Only isolated listening-test projects may use this explicit operation. Review
actions never call it. It shares the production TTS lease and durable job owner,
requires all final conditioning identities in advance, and renders exactly the
manifest's utterances. No auditions, design references or additional editions
are generated here. Files are immutable and project publication is conditional
on the same manifest, source, cast and selected Takes remaining current.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import wave
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.storage.jobs import get_job, mutate_job, start_job, upsert_job, mark_job_active
from dubbing_service.app.storage.projects import load_project, project_meta_lock, save_project
from dubbing_service.app.storage.segments import load_segments, project_segments_lock
from . import pipeline, qa, release, renderer as renderer_module, takes
from .fingerprint import file_sha256
from .listening_excerpt import (ChapterIdentity, ExcerptUtterance, ExcerptValidationError,
                                build_excerpt_manifest, text_hash, validate_rendered_excerpt,
                                adjust_excerpt_to_measured_duration)
from .listening_excerpt import validate_source_bytes


class ListeningRenderError(ValueError):
    pass


class _TimingAdjustmentNeeded(ListeningRenderError):
    def __init__(self, manifest: dict, mapping: list[dict], records: Sequence):
        self.manifest = manifest
        self.mapping = mapping
        self.records = records
        super().__init__("Actual decoded duration is outside 285–315 seconds; only bounded adjacent source adjustment is allowed")


def _path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ListeningRenderError("Listening artifact escapes its canonical project")
    return result


def _immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ListeningRenderError("Existing immutable artifact has different bytes")
        return
    # Stage complete bytes before the exclusive publish. Hard-link creation is
    # atomic and fails if the destination exists; it never replaces a completed
    # manifest/receipt. An interrupted stage is noncanonical and harmless.
    from uuid import uuid4
    staged = path.with_name(f".{path.name}.{uuid4().hex}.stage")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(staged, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ListeningRenderError("Concurrent immutable artifact has different bytes")
    # Keep the bounded staging hard link as crash evidence. It is never read as
    # project state; avoiding cleanup also preserves every pre-existing file.


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _validate_manifest(manifest: dict, source_text: str) -> dict:
    allowed = {field.name for field in fields(ExcerptUtterance)}
    original_rows = manifest.get("chapter_utterances", manifest["utterances"])
    rows = [ExcerptUtterance(**{k: v for k, v in row.items() if k in allowed},
                            ) for row in original_rows]
    # Preserve source aliases when revalidating the normalized manifest.
    rows = [ExcerptUtterance(**{**row.__dict__, "role_id": old.get("source_role_id", row.role_id)})
            for row, old in zip(rows, original_rows)]
    validated = build_excerpt_manifest(
        chapter=ChapterIdentity(**manifest["chapter"]), source_text=source_text,
        utterances=rows, selected_utterance_ids=manifest["utterance_ids"],
        character_ids=manifest.get("chapter_character_ids", manifest["character_ids"]), narrator_role_id=manifest["narrator_role_id"],
        role_aliases=manifest.get("role_aliases"),
        full_chapter_inspected_sha256=manifest["full_chapter_inspected_sha256"],
        selection_rationale=manifest["selection_rationale"],
    )
    if validated["manifest_id"] != manifest.get("manifest_id"):
        raise ListeningRenderError("Manifest content changed after source validation")
    return validated


def _canonical(project_id: str, project: Any, manifest: dict) -> list:
    if project.voice_lab.scope != "isolated_listening_test":
        raise ListeningRenderError("Automatic excerpt casting/rendering is restricted to an isolated listening-test project")
    root = safe_project_root(project_id)
    chapter = manifest["chapter"]
    revision = next((r for r in project.source_revisions if r.id == chapter["source_revision_id"]), None)
    if revision is None or revision.sha256 != chapter["source_sha256"] or not revision.snapshot_file:
        raise ListeningRenderError("Canonical source revision differs from the excerpt manifest")
    snapshot = _path(root, revision.snapshot_file)
    if not snapshot.is_file() or file_sha256(snapshot) != revision.sha256:
        raise ListeningRenderError("Canonical source snapshot is missing or hash-mismatched")
    validate_source_bytes(ChapterIdentity(**chapter), snapshot.read_bytes())
    live_chapter = next((c for c in project.chapters if c.id == chapter["chapter_id"]), None)
    if live_chapter is None or live_chapter.title != chapter["chapter_title"]:
        raise ListeningRenderError("Canonical chapter identity changed")
    by_id = {s.id: s for s in load_segments(project_id)}
    ordered = []
    for row in manifest["utterances"]:
        segment = by_id.get(row["utterance_id"])
        if segment is None or not release.is_live(segment):
            raise ListeningRenderError("Selected source utterance is missing or retired")
        anchor = segment.source_anchor
        if (not anchor or anchor.source_revision_id != chapter["source_revision_id"]
                or anchor.start_offset != row["start_offset"] or anchor.end_offset != row["end_offset"]
                or segment.chapter_id != chapter["chapter_id"] or text_hash(segment.text) != row["source_text_sha256"]
                or segment.spoken_text_revision_id != row.get("spoken_text_revision_id")):
            raise ListeningRenderError("Canonical source text, spoken-text revision or exact offsets changed")
        if segment.role_id != row["role_id"]:
            raise ListeningRenderError("Canonical speaker assignment differs from resolved source attribution")
        attribution = segment.attribution
        if not attribution or attribution.source == "unknown" or attribution.speaker_id != segment.role_id:
            raise ListeningRenderError("Canonical source attribution is unresolved")
        if segment.playback_speed != 1.0 or segment.edit_state is not None:
            raise ListeningRenderError("Listening excerpt requires complete unedited utterances at natural 1x playback")
        selected = takes.selected_take(segment)
        if selected and (selected.playback_speed != 1.0 or selected.baked_speed != 1.0 or selected.edit_state is not None):
            raise ListeningRenderError("Selected Take has a trim or speed edit; an explicit clean source version is required")
        ordered.append(segment)
    return ordered


def persist_excerpt_manifest(project_id: str, manifest: dict, *, source_text: str,
                             expected_manifest_id: str | None = None) -> dict:
    """Persist source proof before GPU work; never imports or mutates source/cast."""
    manifest = _validate_manifest(manifest, source_text)
    root = safe_project_root(project_id)
    relative = f"voice_lab/excerpts/{manifest['manifest_id']}"
    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            current = project.voice_lab.excerpt or {}
            if current.get("manifest_id") != expected_manifest_id:
                raise ListeningRenderError("Excerpt selection changed; refusing a stale manifest")
            _canonical(project_id, project, manifest)
            _immutable(_path(root, relative + "/manifest.json"), _json(manifest))
            _immutable(_path(root, relative + "/chapter.txt"), source_text.encode("utf-8"))
            project.voice_lab.excerpt = {
                "status": "selected_pending_synthesis", "manifest_id": manifest["manifest_id"],
                "manifest_file": relative + "/manifest.json", "chapter_text_file": relative + "/chapter.txt",
                "chapter_title": manifest["chapter"]["chapter_title"],
                "estimated_duration_seconds": manifest["estimated_duration_seconds"],
                "duration_seconds": None, "audio_url": None, "audio_sha256": None,
                "technical_qa": "pending", "human_review": "pending", "map": [], "comparisons": [],
            }
            project.voice_lab.revision += 1
            save_project(project)
            return dict(project.voice_lab.excerpt)


def load_excerpt_manifest(project_id: str, expected_manifest_id: str) -> tuple[dict, str]:
    root = safe_project_root(project_id)
    project = load_project(project_id)
    excerpt = project.voice_lab.excerpt or {}
    if not expected_manifest_id or excerpt.get("manifest_id") != expected_manifest_id:
        raise ListeningRenderError("A current source-validated persisted excerpt must be selected before rendering")
    if not excerpt.get("manifest_file") or not excerpt.get("chapter_text_file"):
        raise ListeningRenderError("Source manifest and complete local chapter extraction have not been persisted")
    manifest = json.loads(_path(root, excerpt["manifest_file"]).read_text(encoding="utf-8"))
    source_text = _path(root, excerpt["chapter_text_file"]).read_bytes().decode("utf-8")
    if manifest.get("manifest_id") != expected_manifest_id:
        raise ListeningRenderError("Persisted manifest does not match current excerpt selection")
    validated = _validate_manifest(manifest, source_text)
    _canonical(project_id, project, validated)
    return validated, source_text


def _identity_for_take(take: Any) -> tuple[str, str, str]:
    identity = take.generation_metadata.get("character_voice_identity") or {}
    identity_id = identity.get("id")
    version = (take.fingerprint_components or {}).get("voice_revision")
    prompt = identity.get("prompt_sha256")
    reference = identity.get("reference_audio_sha256")
    calls = take.generation_metadata.get("provider_calls") or []
    if identity.get("mechanism") != "qwen_reference" or not identity_id or not version or not prompt or not reference:
        raise ListeningRenderError("Take lacks a versioned final Qwen Base reference identity")
    if not calls or any(call.get("identity_id") != identity_id or call.get("prompt_sha256") != prompt for call in calls):
        raise ListeningRenderError("Provider generation metadata does not prove this Take used its declared final conditioning identity")
    return identity_id, version, reference


def _take_audio(root: Path, segments: Sequence, manifest: dict) -> tuple[list[Path], list[dict], list[dict]]:
    coverage = release.check_coverage(segments, project_root=root, verify_bytes=True, require_qa=True)
    if not coverage.ready:
        raise ListeningRenderError("Selected Takes fail existing blocking release QA: " + json.dumps(coverage.as_dict()))
    paths, mapping, selections = [], [], []
    cursor = 0.0
    for segment, source in zip(segments, manifest["utterances"]):
        take = takes.selected_take(segment)
        if take is None or take.stale or take.role_id != segment.role_id or take.source_text_hash != source["source_text_sha256"]:
            raise ListeningRenderError("Selected Take is stale or bound to different source text/speaker")
        if take.playback_speed != 1.0 or take.baked_speed != 1.0 or take.edit_state is not None:
            raise ListeningRenderError("Listening test cannot pad, trim or change playback speed")
        path = _path(root, take.audio_file)
        facts = qa.read_audio_facts(path)
        if not facts or not facts.decodable or facts.duration_ms <= 0:
            raise ListeningRenderError("Selected audio cannot be decoded")
        identity_id, version, reference = _identity_for_take(take)
        end = cursor + _decoded_duration(path)
        mapping.append({"utterance_id": segment.id, "role_id": segment.role_id, "voice_id": identity_id,
                        "voice_version": version, "conditioning_sha256": reference, "take_id": take.id,
                        "audio_sha256": take.audio_sha256, "start_seconds": cursor, "end_seconds": end,
                        "rms_dbfs": 20 * math.log10(max(float(facts.rms), 1e-6)) if hasattr(facts, "rms") else None})
        selections.append({"utterance_id": segment.id, "take_id": take.id, "audio_sha256": take.audio_sha256})
        paths.append(path)
        cursor = end
    return paths, mapping, selections


def _decoded_duration(path: Path) -> float:
    if path.suffix.casefold() == ".wav":
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / source.getframerate()
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
                             "-vn", "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1"],
                            capture_output=True, timeout=60, check=True)
    return len(result.stdout) / (24000 * 2)


def _assemble_pcm(paths: Sequence[Path], output: Path) -> None:
    """Lossless concatenation without silence, trimming, speed or gain changes."""
    if output.exists():
        raise ListeningRenderError("Assembly path already exists without a completed immutable export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        with wave.open(target, "wb") as destination:
            expected = None
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    params = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if expected is None:
                        expected = params
                        destination.setnchannels(params[0])
                        destination.setsampwidth(params[1])
                        destination.setframerate(params[2])
                    elif params != expected:
                        raise ListeningRenderError("Source PCM formats differ; explicit lossless conversion is required")
                    for block in iter(lambda: source.readframes(65536), b""):
                        destination.writeframesraw(block)


def _mp3(master: Path, destination: Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(master),
                    "-vn", "-codec:a", "libmp3lame", "-b:a", "192k", str(destination)],
                   check=True, capture_output=True, timeout=60)


def _loudness(path: Path) -> float | None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
                             "-af", "ebur128", "-f", "null", "-"],
                            check=True, capture_output=True, text=True, timeout=60)
    matches = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", result.stderr)
    return float(matches[-1]) if matches and math.isfinite(float(matches[-1])) else None


def _transitions(mapping: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Advisory measured utterance-level changes, distinct from voice identity."""
    result = []
    for left, right in zip(mapping, mapping[1:]):
        levels = [left.get("rms_dbfs"), right.get("rms_dbfs")]
        measured = all(isinstance(value, (int, float)) and math.isfinite(value) for value in levels)
        delta = levels[1] - levels[0] if measured else None
        result.append({"left_utterance_id": left["utterance_id"], "right_utterance_id": right["utterance_id"],
                       "seconds": right["start_seconds"], "method": "adjacent_utterance_rms_dbfs",
                       "level_change_db": delta, "status": "unmeasured" if delta is None else "review" if abs(delta) > 12 else "within_advisory_range",
                       "blocking": False})
    return result


def _publish(project_id: str, manifest: dict, result: dict, selections: list[dict]) -> dict:
    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            current = project.voice_lab.excerpt or {}
            if current.get("manifest_id") != manifest["manifest_id"]:
                raise ListeningRenderError("A newer excerpt selection superseded this render")
            segments = _canonical(project_id, project, manifest)
            _paths, _mapping, current_selections = _take_audio(safe_project_root(project_id), segments, manifest)
            if current_selections != selections:
                raise ListeningRenderError("Newer selected Takes superseded this export")
            for segment in segments:
                role = next((r for r in project.roles if r.id == segment.role_id), None)
                take = takes.selected_take(segment)
                if not role or not role.voice_identity or role.voice_identity.id != take.generation_metadata["character_voice_identity"]["id"]:
                    raise ListeningRenderError("Role identity changed while rendering; older audio cannot replace it")
                requirement = pipeline._requirement_for_take(project, segment, take)
                if requirement is None or take.input_fingerprint != requirement.fingerprint():
                    raise ListeningRenderError("Synthesis requirement changed while assembling")
            from .voice_lab import digest
            comparisons = {}
            for candidate in [*current.get("comparisons", []), current, result]:
                loudness = candidate.get("loudness_lufs", candidate.get("integrated_loudness_lufs"))
                measurement = candidate.get("loudness_measurement") or {}
                if (candidate.get("manifest_id", candidate.get("passage_id")) != manifest["manifest_id"]
                        or candidate.get("utterance_ids") != manifest["utterance_ids"]
                        or not isinstance(loudness, (int, float)) or not math.isfinite(loudness)
                        or measurement.get("audio_sha256") != candidate.get("audio_sha256")
                        or not candidate.get("audio_file")):
                    continue
                artifact = _path(safe_project_root(project_id), candidate["audio_file"])
                if not artifact.is_file() or file_sha256(artifact) != candidate.get("audio_sha256"):
                    continue
                key = "v_" + digest(candidate["audio_sha256"])[:28]
                comparisons[key] = {"key": key, "label": f"Listening edition {len(comparisons) + 1}",
                                    "audio_file": candidate["audio_file"], "audio_sha256": candidate["audio_sha256"],
                                    "passage_id": manifest["manifest_id"], "utterance_ids": manifest["utterance_ids"],
                                    "loudness_lufs": loudness, "loudness_measurement": measurement,
                                    "map": candidate["map"], "duration_seconds": candidate["duration_seconds"]}
            project.voice_lab.excerpt = {**current, **result, "comparisons": list(comparisons.values())}
            project.voice_lab.revision += 1
            save_project(project)
            return dict(project.voice_lab.excerpt)


def _render_listening_excerpt_once(project_id: str, *, expected_manifest_id: str,
                                  renderer: Any = None) -> dict:
    """Explicit bounded render/resume; call inside the running Studio owner.

    No global backend is changed. Production jobs get priority; the operation
    returns busy instead of starting another queue or stealing the GPU lane.
    ``renderer`` is a test seam, not a runtime fallback or user preference.
    """
    from .voice_lab import production_busy
    manifest, _source_text = load_excerpt_manifest(project_id, expected_manifest_id)
    project = load_project(project_id)
    _canonical(project_id, project, manifest)
    needed = {row["role_id"] for row in manifest["utterances"]}
    for role_id in needed:
        role = next((r for r in project.roles if r.id == role_id), None)
        if not role or not role.voice_identity or role.voice_identity.status != "LOCKED" or role.voice_identity.mechanism != "qwen_reference":
            raise ListeningRenderError("Every role requires a stable prepared Qwen Base voice before excerpt rendering; no audition batches are started")
    current = project.voice_lab.excerpt or {}
    if current.get("status") == "ready_for_listening":
        try:
            _paths, _mapping, selections = _take_audio(safe_project_root(project_id), _canonical(project_id, project, manifest), manifest)
            audio = _path(safe_project_root(project_id), current["audio_file"])
            master = _path(safe_project_root(project_id), current["master_file"])
            if (selections == current.get("selections") and audio.is_file() and master.is_file()
                    and file_sha256(audio) == current["audio_sha256"] and file_sha256(master) == current["master_sha256"]):
                return {**_publish(project_id, manifest, current, selections), "_run_generated_seconds": 0.0}
        except (ListeningRenderError, KeyError):
            pass  # Explicit render/resume may repair affected Takes, never a review action.
    if production_busy():
        raise ListeningRenderError("Existing production jobs retain GPU priority; resume this excerpt when the production lane is idle")
    if renderer is None:
        from .voice_lab_runtime import ensure_backend_ready
        ensure_backend_ready("qwen3tts", project_id=project_id)
    lease = try_register_running_batch(project_id)
    if lease is None:
        raise ListeningRenderError("Production TTS lane is busy; excerpt is safe to resume")
    root = safe_project_root(project_id)
    identity_version = hashlib.sha256(_json([r.voice_identity.model_dump(mode="json") for r in project.roles if r.id in needed])).hexdigest()[:12]
    job_id = "voice_lab_" + manifest["manifest_id"] + "_" + identity_version
    try:
        job = get_job(project_id, job_id)
        if not job:
            upsert_job(project_id, SynthesisJob(id=job_id, project_id=project_id, job_type="voice_lab_listening",
                                               segment_ids=manifest["utterance_ids"], request_id=job_id, idempotency_key=job_id))
        if job and job.status == SynthesisJobStatus.SUCCEEDED:
            # An explicit repair found changed or missing bytes after a completed
            # export. Keep the same durable request and repair only invalid Takes.
            mutate_job(project_id, job_id, lambda j: setattr(j, "status", SynthesisJobStatus.INTERRUPTED))
        mark_job_active(project_id, job_id, True)
        start_job(project_id, job_id)
        actual_renderer = renderer or renderer_module.BackendRenderer(backend="qwen3tts", engine_id="qwen3tts",
                                                                       renderer_id="qwen3tts", project_root=root)
        records = pipeline.render_utterances(project_id, manifest["utterance_ids"], renderer=actual_renderer,
                                              resume=True, run_qa=True, promote=True, job_id=job_id)
        if len(records) != len(manifest["utterance_ids"]) or any(not r.success or not r.selected or r.stale for r in records):
            raise ListeningRenderError("Some selected-source utterances failed render/QA or became stale; resume repairs only affected utterances")
        project = load_project(project_id)
        if (project.voice_lab.excerpt or {}).get("manifest_id") != expected_manifest_id:
            raise ListeningRenderError("Excerpt selection changed while synthesis ran")
        segments = _canonical(project_id, project, manifest)
        paths, mapping, selections = _take_audio(root, segments, manifest)
        if not 285 <= mapping[-1]["end_seconds"] <= 315:
            raise _TimingAdjustmentNeeded(manifest, mapping, records)
        export_id = hashlib.sha256(_json(selections)).hexdigest()[:24]
        relative = f"voice_lab/excerpts/{expected_manifest_id}/exports/{export_id}"
        master, mp3, receipt = [_path(root, relative + name) for name in ("/clean-master.wav", "/listening.mp3", "/export.json")]
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding="utf-8"))
            if (result.get("selections") != selections or not master.is_file() or not mp3.is_file()
                    or file_sha256(master) != result.get("master_sha256") or file_sha256(mp3) != result.get("audio_sha256")):
                raise ListeningRenderError("Previously completed export no longer matches its immutable receipt")
        else:
            # A failed partial assembly is never overwritten. A fresh directory
            # records the next export attempt, reusing the already valid Takes.
            if master.exists() or mp3.exists():
                from uuid import uuid4
                relative += "/resume_" + uuid4().hex[:12]
                master, mp3, receipt = [_path(root, relative + name) for name in ("/clean-master.wav", "/listening.mp3", "/export.json")]
            _assemble_pcm(paths, master)
            duration = _decoded_duration(master)
            master_hash = file_sha256(master)
            evidence = validate_rendered_excerpt(manifest, audio_map=mapping,
                                                  decoded_duration_seconds=duration, audio_sha256=master_hash)
            texts = [takes.selected_take(s).synthesis_text or s.text for s in segments]
            master_qa = qa.qa_take_audio(take_id="assembly_" + export_id, audio_path=master,
                                         text="\n".join(texts), expected_sha256=master_hash, language="English")
            if not master_qa.passed or master_qa.blocking_failures() or not master_qa.executed_check_ids():
                raise ListeningRenderError("Assembled master fails existing blocking technical QA")
            _mp3(master, mp3)
            delivered_seconds = _decoded_duration(mp3)
            if not 285 <= delivered_seconds <= 315 or abs(delivered_seconds - duration) > 0.1:
                raise ListeningRenderError("Decoded listening export duration differs from the valid natural-speed master")
            result = {**evidence, "status": "ready_for_listening", "chapter_title": manifest["chapter"]["chapter_title"],
                      "master_file": str(master.relative_to(root)).replace("\\", "/"), "master_sha256": master_hash,
                      "audio_file": str(mp3.relative_to(root)).replace("\\", "/"), "audio_sha256": file_sha256(mp3),
                      "duration_seconds": delivered_seconds, "master_duration_seconds": duration,
                      "map": mapping, "selections": selections, "technical_qa": "passed", "human_review": "pending",
                      "qa_report": master_qa.model_dump(mode="json"), "master_integrated_loudness_lufs": _loudness(master),
                      "voice_consistency": "final_conditioning_identity_verified_per_take",
                      "acoustic_voice_consistency": "human_review_pending", "comparisons": [],
                      "transition_diagnostics": _transitions(mapping),
                      "generation_totals": {"conditioning_reference_seconds": 0, "candidate_seconds": 0,
                                            "production_seconds_generated_this_run": sum(r.duration_ms / 1000 for r in records if not r.reused),
                                            "final_listening_seconds": delivered_seconds}}
            result["utterance_ids"] = manifest["utterance_ids"]
            result["integrated_loudness_lufs"] = _loudness(mp3)
            result["loudness_measurement"] = {"audio_sha256": result["audio_sha256"],
                                             "integrated_lufs": result["integrated_loudness_lufs"], "method": "ffmpeg_ebur128"}
            _immutable(receipt, _json(result))
            _immutable(receipt.parent / "source-to-audio-map.json", _json({"manifest_id": expected_manifest_id, "master_sha256": master_hash,
                                                                       "audio_sha256": result["audio_sha256"], "map": mapping}))
        published = _publish(project_id, manifest, result, selections)
        mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.SUCCEEDED),
                                                  setattr(j, "completed_segment_ids", manifest["utterance_ids"]),
                                                  setattr(j, "finished_at", datetime.now(timezone.utc))))
        return {**published, "_run_generated_seconds": sum(r.duration_ms / 1000 for r in records if not r.reused)}
    except Exception as exc:
        if get_job(project_id, job_id):
            mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED),
                                                      setattr(j, "message", str(exc)[:400])))
        raise
    finally:
        mark_job_active(project_id, job_id, False)
        unregister_running_batch(lease)


def _conditioning_totals(project: Any, role_ids: set[str]) -> dict:
    """Count saved design receipts separately from production utterance audio."""
    root = safe_project_root(project.id)
    seen = set()
    seconds = 0.0
    missing_measurements = []
    for request_id, design in project.voice_lab.designs.items():
        if (not request_id.startswith("excerpt_") or design.get("status") != "ready"
                or design.get("role_id") not in role_ids or not design.get("reference_generation_count")):
            continue
        identity_id = design.get("identity_id")
        if identity_id in seen:
            continue
        seen.add(identity_id)
        record = project.voice_lab.voices.get(design.get("key"), {})
        identity = record.get("identity") or {}
        relative = identity.get("reference_audio")
        try:
            path = _path(root, relative) if relative else None
            if not path or not path.is_file() or file_sha256(path) != identity.get("reference_audio_sha256"):
                raise ListeningRenderError("Conditioning reference receipt has no verified audio")
            seconds += _decoded_duration(path)
        except (OSError, ValueError, wave.Error, subprocess.SubprocessError):
            missing_measurements.append(identity_id)
    return {"conditioning_reference_scope": "saved_excerpt_design_receipts_in_this_isolated_project",
            "conditioning_reference_count": len(seen),
            "conditioning_reference_seconds": seconds if not missing_measurements else None,
            "conditioning_reference_unmeasured_identity_ids": missing_measurements}


def render_listening_excerpt(project_id: str, *, expected_manifest_id: str,
                             renderer: Any = None) -> dict:
    """Render one listening test with at most two bounded timing adjustments.

    Reuse previously valid Takes. Adjust only adjacent complete source utterances
    when decoded timing requires it, retaining all originally required speakers.
    Each adjustment admits at most eight new utterances / 60 estimated seconds;
    actual decoded duration is checked again. Every revised manifest is persisted
    before any newly required conditioning or utterance generation. There is one
    published listening edition; unsuccessful scopes remain private Take history.
    """
    from .voice_lab import prepare_listening_cast
    attempted = []
    repair_seconds = 0.0
    for attempt in range(3):
        try:
            result = _render_listening_excerpt_once(project_id, expected_manifest_id=expected_manifest_id, renderer=renderer)
        except _TimingAdjustmentNeeded as exc:
            generated = sum(record.duration_ms / 1000 for record in exc.records if not record.reused)
            if attempt:
                repair_seconds += generated
            attempted.append({"manifest_id": exc.manifest["manifest_id"],
                              "decoded_duration_seconds": exc.mapping[-1]["end_seconds"],
                              "new_production_seconds": generated})
            if attempt == 2:
                raise ListeningRenderError("Decoded timing still conflicts with 285–315 seconds after two bounded adjacent-source adjustments; no further synthesis was dispatched") from exc
            _manifest, source_text = load_excerpt_manifest(project_id, expected_manifest_id)
            measured = {row["utterance_id"]: row["end_seconds"] - row["start_seconds"] for row in exc.mapping}
            adjusted = adjust_excerpt_to_measured_duration(exc.manifest, source_text=source_text,
                                                          measured_utterance_seconds=measured)
            persist_excerpt_manifest(project_id, adjusted, source_text=source_text,
                                     expected_manifest_id=expected_manifest_id)
            expected_manifest_id = adjusted["manifest_id"]
            project = load_project(project_id)
            required = set(adjusted["character_ids"]) | {adjusted["narrator_role_id"]}
            if any(r.id in required and r.voice_identity is None for r in project.roles):
                prepare_listening_cast(project_id, expected_manifest_id)
            continue
        totals = dict(result.get("generation_totals", {}))
        current_generated = float(result.pop("_run_generated_seconds", totals.get("production_seconds_generated_this_run", 0)))
        if attempt:
            repair_seconds += current_generated
        totals["production_seconds_generated_this_run"] = current_generated + sum(row["new_production_seconds"] for row in attempted)
        totals["bounded_timing_repair_seconds_this_run"] = repair_seconds
        totals["production_generation_scope"] = "this_explicit_render_operation_including_bounded_timing_adjustments"
        totals.update(_conditioning_totals(load_project(project_id), {row["role_id"] for row in result["map"]}))
        with project_meta_lock(project_id):
            project = load_project(project_id)
            current = project.voice_lab.excerpt or {}
            if current.get("manifest_id") != expected_manifest_id or current.get("audio_sha256") != result.get("audio_sha256"):
                raise ListeningRenderError("Newer listening selection superseded generation accounting")
            current.update(generation_totals=totals, timing_adjustments=attempted)
            project.voice_lab.excerpt = current
            project.voice_lab.revision += 1
            save_project(project)
            return dict(current)
    raise AssertionError("bounded render loop exhausted")  # pragma: no cover
