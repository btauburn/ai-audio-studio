"""Explicit alternate chapter renditions through the single Studio owner.

Forking never changes the source project. Rendering uses the existing GPU
lease, jobs, hash-bound QA and optimistic Take merge; it has no audition path.
These operations are deliberately separate from the Chapter 1 listening test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Callable

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.core.tts_operations import (
    release_project_mutation, try_acquire_project_mutation,
    try_register_running_batch, unregister_running_batch,
)
from dubbing_service.app.models.character import CharacterCastState
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.enums import SegmentReviewStatus, SegmentStatus
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.storage.jobs import (
    get_job, load_jobs, mark_job_active, mutate_job, record_job_segment,
    start_job, try_mark_job_active, upsert_job,
)
from dubbing_service.app.storage.projects import create_project, load_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments, project_segments_lock
from . import pipeline, release
from .persistence import commit_project_state
from .renderer import BackendRenderer
from .voice_lab_library import _publish_blob, _read_verified, _safe_path

MAX_CHAPTER_UTTERANCES = 2000
MAX_SOURCE_BYTES = 512 * 1024 * 1024

class ChapterRenditionError(ValueError):
    pass


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_row(segment) -> dict:
    return {"utterance_id": segment.id, "index": segment.index,
            "chapter_id": segment.chapter_id, "source_text_sha256": _text_hash(segment.text),
            "source_anchor": segment.source_anchor.model_dump(mode="json") if segment.source_anchor else None,
            "spoken_text_revision_id": segment.spoken_text_revision_id}


def _scene_break(text: str) -> bool:
    return bool(re.fullmatch(r"(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}", text.strip()))


def _source_proof(project, rows) -> dict:
    return {"chapter": project.chapters[0].model_dump(mode="json"), "utterances": rows,
            "source_revisions": [r.model_dump(mode="json") for r in project.source_revisions],
            "spoken_text_revisions": [r.model_dump(mode="json") for r in project.spoken_text_revisions]}


def _apply_reviewed_attribution(source, target, segments, review: dict | None = None) -> list[dict]:
    """Normalize already-known identities; apply only the exact reviewed source.

    Original attribution remains in derivation metadata. No text is changed,
    no caller-supplied instructions are executed, and no human approval is
    invented: the review is an evidence-backed analysis recorded as inline.
    """
    changes, by_id = [], {s.id: s for s in segments}
    corrections = {}
    if review is not None:
        if len(json.dumps(review, ensure_ascii=False).encode("utf-8")) > 128 * 1024:
            raise ChapterRenditionError("Attribution review exceeds its bounded metadata limit")
        if source.id != review.get("source_project_id") or target.chapters[0].id != review.get("chapter_id"):
            raise ChapterRenditionError("Attribution review must bind to this exact source project and chapter")
        if not any(r.id == review["source_revision_id"] and r.sha256 == review["source_sha256"]
                   for r in source.source_revisions):
            raise ChapterRenditionError("Reviewed chapter attribution refers to another source revision")
        for relative, expected in review.get("source_project_file_sha256", {}).items():
            if relative not in {"project.json", "segments.json"}:
                raise ChapterRenditionError("Attribution audit may bind only canonical project metadata files")
            _read_verified(_safe_path(safe_project_root(source.id), relative), expected)
        reviewed_rows = list(review.get("corrections", []))
        if review.get("narration_fragment"):
            reviewed_rows.append(review["narration_fragment"])
        if len(reviewed_rows) > MAX_CHAPTER_UTTERANCES:
            raise ChapterRenditionError("Attribution review exceeds this chapter's bounded scope")
        roles = {r.id: r for r in target.roles}
        for row in reviewed_rows:
            segment = by_id.get(row["utterance_id"])
            if segment is None or _text_hash(segment.text) != row["source_text_sha256"]:
                raise ChapterRenditionError("Reviewed speaker correction no longer matches exact source text")
            if (row["role_id"] not in roles or not row.get("reason", "").strip()
                    or row["utterance_id"] in corrections
                    or (row.get("from_role_id") and row["from_role_id"] != segment.role_id)):
                raise ChapterRenditionError("Reviewed speaker correction has unsupported or duplicate role evidence")
            if any(sid not in by_id for sid in row.get("evidence_utterance_ids", [])):
                raise ChapterRenditionError("Reviewed speaker correction lacks its surrounding source evidence")
            corrections[segment.id] = row
        aliases = review.get("alias_corrections", {})
        if len(aliases) > 2000 or any(a not in roles or b not in roles or a == b for a, b in aliases.items()):
            raise ChapterRenditionError("Reviewed role aliases must resolve existing distinct role identities")
        merged = {**target.character_cast.role_aliases, **aliases}
        for alias in aliases:
            seen, current = set(), alias
            while current in merged:
                if current in seen:
                    raise ChapterRenditionError("Reviewed role aliases contain a cycle")
                seen.add(current)
                current = merged[current]
        target.character_cast.role_aliases.update(aliases)
    roles = {r.id: r for r in target.roles}
    for segment in segments:
        row = corrections.get(segment.id)
        original_role, original_attribution = segment.role_id, segment.attribution
        canonical = target.character_cast.role_aliases.get(segment.role_id, segment.role_id)
        role = roles.get(canonical)
        narrator = bool(role and role.role_type == "narrator" and role.name.lower() in {"narrator", "narration"})
        if row:
            canonical = row["role_id"]
            reason, evidence = row["reason"], row.get("evidence_utterance_ids", [segment.id])
        elif canonical != segment.role_id:
            reason, evidence = "Existing canonical role alias; the same speaker retains one identity.", [segment.id]
        elif narrator and (not original_attribution or original_attribution.speaker_id != canonical or original_attribution.source == "unknown"):
            reason, evidence = "Existing canonical narration role; normalize legacy narrator label to its actual role ID.", [segment.id]
        elif (role and original_attribution and original_attribution.source != "unknown"
              and original_attribution.speaker_id != canonical
              and (original_attribution.speaker_id or "").casefold() == role.name.casefold()):
            reason, evidence = "Existing attributed speaker name resolves exactly to the same canonical role ID.", [segment.id]
        else:
            continue
        if canonical not in roles:
            raise ChapterRenditionError("Reviewed speaker is missing its source character profile")
        changes.append({"utterance_id": segment.id, "source_text_sha256": _text_hash(segment.text),
                        "original_role_id": original_role,
                        "original_attribution": original_attribution.model_dump(mode="json") if original_attribution else None,
                        "role_id": canonical, "evidence_utterance_ids": evidence, "reason": reason})
        segment.role_id = canonical
        segment.attribution = Attribution(speaker_id=canonical,
            source=original_attribution.source if not row and original_attribution and original_attribution.source != "unknown" else "inline",
            confidence=1.0 if narrator else 0.95,
            reviewed=False, note=reason + " Evidence utterance IDs: " + ", ".join(evidence))
    return changes


def _fork_result(project, *, reused: bool) -> dict:
    meta = project.metadata["chapter_rerun"]
    return {"project_id": project.id, "chapter_id": meta["chapter_id"], "reused": reused,
            "source_project_id": meta["source_project_id"], "source_hash": meta["expected_source_hash"],
            "speech_utterance_count": len(meta["speech_utterance_ids"]),
            "scene_break_count": len(meta["scene_breaks"]), "scope": project.voice_lab.scope}


def fork_chapter(project_id: str, *, chapter_id: str, name: str,
                 request_id: str | None = None, attribution_review: dict | None = None) -> dict:
    """Copy one exact chapter with fresh cast/review state, no audio or selection.

    The request ID (or chapter/name pair) is retry-stable. Source locks serialize
    retries without adding an authority, daemon or index. Incomplete new folders
    remain harmless evidence if persistence fails; originals are never cleaned.
    """
    name = name.strip()
    if not name or len(name) > 200 or (request_id is not None and not 1 <= len(request_id) <= 100):
        raise ChapterRenditionError("A bounded project name and request identity are required")
    lease = try_acquire_project_mutation(project_id, kind="voice_lab_chapter_fork")
    if lease is None:
        raise ChapterRenditionError("The source project is busy; retry after its current production operation")
    try:
        with project_meta_lock(project_id), project_segments_lock(project_id):
            source = load_project(project_id)
            chapter = next((c for c in source.chapters if c.id == chapter_id), None)
            if chapter is None:
                raise ChapterRenditionError("Exact canonical chapter ID is required; imported position is not chapter identity")
            segments = sorted((s for s in load_segments(project_id)
                               if s.chapter_id == chapter_id and release.is_live(s)), key=lambda s: (s.index, s.id))
            if not segments or len(segments) > MAX_CHAPTER_UTTERANCES:
                raise ChapterRenditionError("Chapter must contain 1â€“2000 canonical utterances")
            if any(s.source_anchor is None for s in segments):
                raise ChapterRenditionError("Every chapter utterance requires a canonical source anchor")
            source_ids = {s.source_anchor.source_revision_id for s in segments}
            revisions = [r.model_copy(deep=True) for r in source.source_revisions if r.id in source_ids]
            if {r.id for r in revisions} != source_ids:
                raise ChapterRenditionError("A chapter source revision is missing")
            original_root = safe_project_root(project_id)
            snapshots = []
            for revision in revisions:
                content = _read_verified(_safe_path(original_root, revision.snapshot_file or ""), revision.sha256)
                if revision.byte_length and len(content) != revision.byte_length:
                    raise ChapterRenditionError("Source snapshot length differs from its recorded revision")
                snapshots.append((revision, content))
            if sum(len(content) for _, content in snapshots) > MAX_SOURCE_BYTES:
                raise ChapterRenditionError("Source snapshots exceed the bounded fork limit")
            spoken_by_id = {r.id: r for r in source.spoken_text_revisions}
            spoken_ids = {s.spoken_text_revision_id for s in segments if s.spoken_text_revision_id}
            pending = list(spoken_ids)
            while pending:
                revision = spoken_by_id.get(pending.pop())
                if revision is None:
                    raise ChapterRenditionError("Spoken-text history is incomplete")
                previous = revision.previous_revision_id
                if previous and previous not in spoken_ids:
                    spoken_ids.add(previous)
                    pending.append(previous)
            spoken = [r.model_copy(deep=True) for r in source.spoken_text_revisions if r.id in spoken_ids]
            fork_key = _hash({"source_project_id": project_id, "chapter_id": chapter_id,
                              "request": request_id or name})
            source_rows = [_source_row(s) for s in segments]
            # Compare retries with exactly the same source proof before returning.
            proof_project = source.model_copy(deep=True)
            proof_project.chapters, proof_project.source_revisions = [chapter], revisions
            proof_project.spoken_text_revisions = spoken
            source_hash = _hash(_source_proof(proof_project, source_rows))
            reviewed_segments = [s.model_copy(deep=True) for s in segments]
            derivation = _apply_reviewed_attribution(source, proof_project, reviewed_segments, attribution_review)
            review_hash = _hash(attribution_review)
            for path in get_config().projects_root_abs.glob("*/project.json"):
                if path.parent.name == project_id:
                    continue
                existing = load_project(path.parent.name)
                meta = existing.metadata.get("chapter_rerun", {})
                if meta.get("fork_key") == fork_key:
                    if (meta.get("expected_source_hash") != source_hash
                            or meta.get("attribution_review_hash") != review_hash):
                        raise ChapterRenditionError("Fork request now refers to changed source; use a new explicit request")
                    return _fork_result(existing, reused=True)
            breaks = []
            supplied_breaks = {}
            manifest_path = source.metadata.get("scene_break_manifest")
            if manifest_path:
                raw = json.loads(_safe_path(original_root, manifest_path).read_text(encoding="utf-8"))
                for row in (raw if isinstance(raw, list) else raw.get("scene_breaks", [])):
                    supplied_breaks[row.get("segment_id", row.get("utterance_id"))] = row
            for segment in segments:
                if _scene_break(segment.text):
                    old = supplied_breaks.get(segment.id, {})
                    duration = float(old.get("seconds", old.get("duration_seconds", 1.25)))
                    if not 0 <= duration <= 10:
                        raise ChapterRenditionError("Scene-break timing is outside its bounded source convention")
                    breaks.append({"utterance_id": segment.id, "index": segment.index,
                                   "duration_seconds": duration, "source_text_sha256": _text_hash(segment.text)})
                elif segment.id in supplied_breaks:
                    raise ChapterRenditionError("Scene-break metadata refers to spoken source text")
            break_ids = {row["utterance_id"] for row in breaks}
            if len(break_ids) == len(segments):
                raise ChapterRenditionError("Chapter contains scene breaks but no spoken utterances")
            fresh = create_project(name, description="Separate chapter rendition; source project and approved Takes preserved")
            target_lease = try_acquire_project_mutation(fresh.id, kind="voice_lab_chapter_fork_commit")
            if target_lease is None:
                raise ChapterRenditionError("New project became busy before source commit")
            try:
                target_root = safe_project_root(fresh.id)
                for revision, content in snapshots:
                    _publish_blob(target_root, revision.snapshot_file, content, revision.sha256)
                fresh.chapters = [chapter.model_copy(deep=True)]
                fresh.source_revisions, fresh.spoken_text_revisions = revisions, spoken
                # Keep source character context for explicit attribution repairs,
                # including a dossier for a presently unresolved speaking role.
                fresh.roles = [r.model_copy(deep=True) for r in source.roles]
                for role in fresh.roles:
                    role.voice_identity = None
                    role.speaker_audio = role.speaker_voice_id = role.qwen3_speaker = None
                    role.tts_backend = "qwen3tts"
                    role.voice_description = role.style_instruction = None
                    role.voice_source = role.voice_license_name = role.voice_commercial_use = role.voice_rights_note = None
                fresh.character_cast = CharacterCastState(
                    dossiers=copy.deepcopy(source.character_cast.dossiers),
                    profiles=copy.deepcopy(source.character_cast.profiles),
                    role_aliases=dict(proof_project.character_cast.role_aliases))
                fresh.character_cast.policy = source.character_cast.policy.model_copy(deep=True)
                fresh.character_cast.policy.enabled = False
                fresh.character_cast.policy.audition_mode = "off"
                fresh.character_cast.policy.max_audition_characters = 0
                fresh.voice_lab.scope = "production"
                fresh.metadata = {key: copy.deepcopy(value) for key, value in source.metadata.items()
                                  if key in {"book_title", "title", "author", "language", "source_language", "pronunciation_dictionary"}}
                fresh.metadata["chapter_rerun"] = {
                    "schema": "voice-lab-chapter-rerun-v1", "authorized": True, "fork_key": fork_key,
                    "source_project_id": project_id, "chapter_id": chapter_id, "chapter_title": chapter.title,
                    "source_utterance_ids": [s.id for s in segments], "source_utterances": source_rows,
                    "speech_utterance_ids": [s.id for s in segments if s.id not in break_ids],
                    "scene_breaks": breaks, "expected_source_hash": source_hash,
                    "attribution_review": copy.deepcopy(attribution_review), "attribution_review_hash": review_hash,
                    "attribution_derivation": derivation,
                    "source_revisions": [{"id": r.id, "sha256": r.sha256, "snapshot_file": r.snapshot_file} for r in revisions],
                    "source_revision_id": revisions[0].id if len(revisions) == 1 else None,
                    "source_sha256": revisions[0].sha256 if len(revisions) == 1 else None,
                    "created_at": datetime.now(timezone.utc).isoformat(), "human_review": "pending"}
                fresh_segments = []
                for original in reviewed_segments:
                    segment = original.model_copy(deep=True)
                    segment.takes, segment.take_slots, segment.review_issues = [], {}, []
                    segment.selected_take_id = segment.audio_file = segment.alt_audio_file = None
                    segment.edit_state = segment.reviewed_at = segment.review_note = segment.engine_text = None
                    segment.status, segment.review_status = SegmentStatus.PENDING, SegmentReviewStatus.DRAFT
                    segment.active_version, segment.playback_speed, segment.volume_gain = "a", 1.0, 1.0
                    segment.tts_backend, segment.alt_tts_backend, segment.qwen3_speaker = "qwen3tts", None, None
                    if segment.emotion_audio:
                        raise ChapterRenditionError("External emotion references require explicit migration before this fork")
                    fresh_segments.append(segment)
                speech_segments = [s for s in fresh_segments if s.id not in break_ids]
                fresh.metadata["chapter_rerun"]["speech_diagnostics"] = {
                    "max_source_characters": max(len(s.text) for s in speech_segments),
                    "max_source_words": max(len(s.text.split()) for s in speech_segments),
                    "renderer_chunking": "existing sentence-greedy renderer; token limits remain provider-owned"}
                commit_project_state(fresh.id, project=fresh, segments=fresh_segments)
                return _fork_result(fresh, reused=False)
            finally:
                release_project_mutation(target_lease)
    finally:
        release_project_mutation(lease)


def _bound_state(project_id: str, chapter_id: str, *, verify_files: bool = True):
    with project_meta_lock(project_id), project_segments_lock(project_id):
        project = load_project(project_id)
        meta = project.metadata.get("chapter_rerun", {})
        if (not meta.get("authorized") or meta.get("chapter_id") != chapter_id
                or meta.get("source_project_id") == project_id or project.voice_lab.scope != "production"):
            raise ChapterRenditionError("Render only an explicit isolated alternate chapter rendition")
        if len(project.chapters) != 1 or project.chapters[0].id != chapter_id:
            raise ChapterRenditionError("Canonical chapter identity changed")
        segments = sorted((s for s in load_segments(project_id) if release.is_live(s)), key=lambda s: (s.index, s.id))
        rows = [_source_row(s) for s in segments]
        if (rows != meta.get("source_utterances") or [s.id for s in segments] != meta.get("source_utterance_ids")
                or _hash(_source_proof(project, rows)) != meta.get("expected_source_hash")):
            raise ChapterRenditionError("Canonical source text, snapshot revision or spoken-text history changed")
        root = safe_project_root(project_id)
        if verify_files:
            for revision in project.source_revisions:
                _read_verified(_safe_path(root, revision.snapshot_file or ""), revision.sha256)
        break_ids = {row["utterance_id"] for row in meta.get("scene_breaks", [])}
        if break_ids != {s.id for s in segments if _scene_break(s.text)}:
            raise ChapterRenditionError("Scene-break coverage changed")
        speech = [s for s in segments if s.id not in break_ids]
        if not speech or len(speech) > MAX_CHAPTER_UTTERANCES or [s.id for s in speech] != meta.get("speech_utterance_ids"):
            raise ChapterRenditionError("Chapter speech coverage is empty, changed or exceeds 2000 utterances")
        if project.character_cast.policy.enabled or project.character_cast.policy.audition_mode != "off":
            raise ChapterRenditionError("Alternate rendition requires explicit cast; automatic legacy auditions must be off")
        roles = {r.id: r for r in project.roles}
        identities, requirements = {}, {}
        for segment in speech:
            role = roles.get(segment.role_id)
            identity = role.voice_identity if role else None
            if (not identity or identity.status != "LOCKED" or identity.mechanism != "qwen_reference"
                    or identity.backend != "qwen3tts" or (segment.tts_backend or role.tts_backend) != "qwen3tts"):
                raise ChapterRenditionError(f"Role {segment.role_id!r} needs a locked, ready Qwen Base reference identity")
            if identity.role_id != role.id:
                raise ChapterRenditionError("Voice identity belongs to another role")
            if not segment.attribution or segment.attribution.source == "unknown" or segment.attribution.speaker_id != segment.role_id:
                raise ChapterRenditionError(f"Utterance {segment.id} needs resolved source-backed speaker attribution")
            if segment.playback_speed != 1 or segment.edit_state is not None:
                raise ChapterRenditionError("Chapter source must use complete unedited utterances at natural speed")
            if role.id not in identities:
                if not all((identity.reference_audio, identity.reference_audio_sha256,
                            identity.prompt_artifact, identity.prompt_sha256)):
                    raise ChapterRenditionError("Voice conditioning artifacts are incomplete")
                if verify_files:
                    _read_verified(_safe_path(root, identity.reference_audio), identity.reference_audio_sha256)
                    _read_verified(_safe_path(root, identity.prompt_artifact), identity.prompt_sha256)
                identities[role.id] = identity.model_dump(mode="json")
            requirements[segment.id] = pipeline.requirement_for(project, segment, renderer_id="qwen3tts").fingerprint()
        binding = {"source_hash": meta["expected_source_hash"], "identities": identities, "requirements": requirements,
                   "aliases": project.character_cast.role_aliases, "scene_breaks": meta.get("scene_breaks", [])}
        return project, speech, binding


def _production_waiting(project_id: str, job_id: str) -> bool:
    for path in get_config().projects_root_abs.glob("*/project.json"):
        for job in load_jobs(path.parent.name):
            if ((job.project_id, job.id) != (project_id, job_id)
                    and job.job_type in {"batch_synthesize", "single_synthesize"}
                    and job.status in {SynthesisJobStatus.QUEUED, SynthesisJobStatus.RUNNING}):
                return True
    return False


def _run_chapter(project_id, chapter_id, ids, job_id, intent, lease, renderer):
    records = []
    try:
        renderer = renderer or BackendRenderer(backend="qwen3tts", engine_id="qwen3tts",
                                               renderer_id="qwen3tts", project_root=safe_project_root(project_id))
        if hasattr(renderer, "runtime_contract"):
            from .runtime_binding import bind_job
            project, speech, _ = _bound_state(project_id, chapter_id, verify_files=False)
            bind_job(project_id, job_id, renderer, [s for s in speech if s.id in ids], project.roles)
        def reset(job):
            job.status, job.finished_at = SynthesisJobStatus.INTERRUPTED, None
            job.completed_segment_ids = []
            job.failed_segments = {}
            job.message = "Verifying and resuming the same source and cast through production QA"
        mutate_job(project_id, job_id, reset)
        start_job(project_id, job_id)
        renderer = renderer or BackendRenderer(backend="qwen3tts", engine_id="qwen3tts",
                                               renderer_id="qwen3tts", project_root=safe_project_root(project_id))
        for segment_id in ids:
            current = get_job(project_id, job_id)
            if current.stop_requested or current.status == SynthesisJobStatus.CANCELLED:
                return {"job": current.model_dump(mode="json"), "records": records}
            _, _, bound = _bound_state(project_id, chapter_id, verify_files=False)
            if _hash({"chapter_id": chapter_id, "segment_ids": ids, "binding": bound}) != intent:
                raise ChapterRenditionError("Source or cast changed after dispatch; remaining chapter render stopped")
            batch = pipeline.render_utterances(project_id, [segment_id], renderer=renderer,
                                               resume=True, run_qa=True, promote=True, job_id=job_id)
            if len(batch) != 1 or batch[0].segment_id != segment_id:
                raise ChapterRenditionError("Production pipeline returned different utterance coverage")
            record = batch[0]
            records.append(record.as_dict())
            success = record.success and record.selected and record.qa_passed and not record.stale
            record_job_segment(project_id, job_id, segment_id, success=success,
                               error=record.error or "Take did not pass current source/cast/QA selection gates")
            if not success:
                raise ChapterRenditionError(f"Utterance {segment_id} needs bounded repair: {record.error or 'blocking QA or stale selection'}")
        _, _, bound = _bound_state(project_id, chapter_id)
        if _hash({"chapter_id": chapter_id, "segment_ids": ids, "binding": bound}) != intent:
            raise ChapterRenditionError("Source or cast changed before job completion")
        def finished(job):
            job.status, job.finished_at = SynthesisJobStatus.SUCCEEDED, datetime.now(timezone.utc)
            job.message = "Chapter utterances selected with blocking technical QA passed; human listening review pending"
        job = mutate_job(project_id, job_id, finished)
        return {"job": job.model_dump(mode="json"), "records": records}
    except Exception as exc:
        def interrupted(job):
            if job.status != SynthesisJobStatus.CANCELLED:
                job.status, job.finished_at = SynthesisJobStatus.INTERRUPTED, None
                job.message = str(exc)[:2000]
        mutate_job(project_id, job_id, interrupted)
        # A background job reports through the established jobs owner.
        return {"job": get_job(project_id, job_id).model_dump(mode="json"), "records": records}
    finally:
        mark_job_active(project_id, job_id, False)
        unregister_running_batch(lease)


def render_chapter(project_id: str, *, chapter_id: str, request_id: str,
                   segment_ids: list[str] | None = None, schedule: Callable | None = None,
                   renderer=None) -> dict:
    """Claim bounded production work; same request resumes the same bound intent.

    HTTP uses BackgroundTasks in the Studio process and returns the durable job
    immediately. Tests may execute synchronously with an injected renderer.
    """
    if not request_id or len(request_id) > 100:
        raise ChapterRenditionError("A bounded durable render request ID is required")
    project, speech, binding = _bound_state(project_id, chapter_id)
    speech_ids = [s.id for s in speech]
    wanted = set(segment_ids) if segment_ids is not None else set(speech_ids)
    if (not wanted or len(wanted) > MAX_CHAPTER_UTTERANCES or not wanted.issubset(speech_ids)
            or (segment_ids is not None and len(segment_ids) != len(wanted))):
        raise ChapterRenditionError("Render selection must contain 1â€“2000 distinct speech utterances from this chapter")
    ids = [sid for sid in speech_ids if sid in wanted]
    intent = _hash({"chapter_id": chapter_id, "segment_ids": ids, "binding": binding})
    job_id = "chapter_" + _hash({"project_id": project_id, "request_id": request_id})[:24]
    previous = get_job(project_id, job_id)
    if previous and previous.intent_hash != intent:
        raise ChapterRenditionError("Request ID is already bound to another source, cast or utterance selection")
    if not try_mark_job_active(project_id, job_id):
        return {"job": get_job(project_id, job_id).model_dump(mode="json") if get_job(project_id, job_id) else None,
                "job_id": job_id, "already_running": True}
    lease = None
    dispatched = False
    try:
        if _production_waiting(project_id, job_id):
            raise ChapterRenditionError("Existing production jobs retain GPU priority; retry after they finish")
        if renderer is None:
            from .voice_lab_runtime import ensure_backend_ready
            ensure_backend_ready("qwen3tts", project_id=project_id, job_id=job_id)
        lease = try_register_running_batch(project_id)
        if lease is None:
            raise ChapterRenditionError("Existing production owns the GPU or project; retry when idle")
        _, _, current = _bound_state(project_id, chapter_id)
        if current != binding:
            raise ChapterRenditionError("Source or cast changed before the production lease was acquired")
        if previous is None:
            upsert_job(project_id, SynthesisJob(id=job_id, project_id=project_id, project_name=project.name,
                job_type="batch_synthesize", request_id=request_id, idempotency_key=job_id, intent_hash=intent,
                segment_ids=ids, segment_backends={sid: "qwen3tts" for sid in ids}))
        if schedule is not None:
            schedule(_run_chapter, project_id, chapter_id, ids, job_id, intent, lease, renderer)
            dispatched = True
            return {"job": get_job(project_id, job_id).model_dump(mode="json"), "job_id": job_id,
                    "status_url": f"/api/v1/projects/{project_id}/synthesis-requests/{request_id}"}
        dispatched = True
        return _run_chapter(project_id, chapter_id, ids, job_id, intent, lease, renderer)
    finally:
        if not dispatched:
            mark_job_active(project_id, job_id, False)
            if lease is not None:
                unregister_running_batch(lease)
