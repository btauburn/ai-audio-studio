"""Explicit short previews through Studio's ordinary design and Take pipeline.

A preview is an isolated canonical project with an original neutral source,
never a temporary edit to a book's cast. The parent stores only request links
and an immutable, hash-verified review copy. Existing jobs and GPU leases own
execution; read/poll/play/favorite actions cannot dispatch synthesis.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import PurePosixPath
import wave
from datetime import datetime, timezone
from typing import Callable

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.models.project import Chapter, Segment
from dubbing_service.app.storage.jobs import (
    get_job, is_job_active, load_jobs, mark_job_active, mutate_job, record_job_segment,
    start_job, try_mark_job_active, upsert_job,
)
from dubbing_service.app.storage.projects import create_project, load_project, project_meta_lock, save_project
from dubbing_service.app.storage.segments import load_segments
from . import audition_scripts, pipeline, voice_lab as lab
from .fingerprint import voice_revision_for_role
from .persistence import commit_project_state
from .renderer import BackendRenderer
from .voice_lab_library import _publish_blob, _read_verified, _safe_path
from .voice_lab_runtime import ensure_backend_ready

PREVIEW_TEXT = ("The room grew quiet as I opened the letter. There was still time to change the plan, "
                "but first, I needed to hear what you thought. "
                "Outside, a light appeared in the window, and someone began walking toward us.")
TARGET_SECONDS = 10
MAX_AUDITIONS = 12
MAX_VERSIONED_AUDITIONS_PER_ROLE = 6
#: Baseline allowance for the preview-request history, and the per-character
#: allowance added to it. The history keeps every request so a retry is idempotent,
#: which means a legitimate re-cast adds requests without removing any: a fixed
#: ceiling would block the third casting generation of a ten-character book, which
#: is exactly what it did. The bound therefore scales with the cast, while still
#: refusing unbounded growth from a single character.
MAX_REQUESTS = 100
REQUESTS_PER_CAST_CHARACTER = 12
MAX_ATTEMPTS = 3
_META = "voice_lab_auditions"


class AuditionError(lab.VoiceLabError):
    pass


def _state(project) -> dict:
    return project.metadata.setdefault(_META, {"schema": "voice-lab-auditions-v1", "requests": {}, "intents": {}})


def _current_identity(role):
    return role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)


def _request_budget(project) -> int:
    """How much preview-request history this project's cast may accumulate.

    The history is append-only by design — a retained request is what makes a retry
    idempotent — so a re-cast adds entries without removing any. A fixed ceiling
    therefore cannot distinguish "legitimate third casting generation of a large
    book" from "runaway growth", and blocks the former. The budget scales with the
    number of characters actually in the cast, and a re-cast needs no more than the
    per-character allowance per generation.
    """
    characters = {existing.get("role_id") for existing in _state(project)["intents"].values()}
    characters = {role_id for role_id in characters if role_id}
    return MAX_REQUESTS + REQUESTS_PER_CAST_CHARACTER * max(1, len(characters))


def _section_control(role, section):
    """Use per-section instruct only where this renderer supports it."""
    cue = section.get("delivery_instruction")
    if not cue:
        return None, "not_applicable"
    if role.tts_backend != "qwen3tts_cv":
        return None, "unsupported"
    instruction = " ".join(part for part in (role.style_instruction, cue) if part)
    return instruction, "sent"


def _candidate(project, role_id, candidate_id):
    # A general shortlist is eligible for an explicitly chosen role. A match
    # belonging only to a different character cannot silently bleed across.
    for slot in (role_id, "general"):
        for candidate in project.voice_lab.shortlists.get(slot, {}).get("matches", []):
            if candidate.get("candidate_id", candidate.get("id")) == candidate_id:
                if candidate.get("original_casting") is True:
                    return candidate
                if not candidate.get("performer") or not candidate.get("performance"):
                    raise AuditionError("This candidate lacks its performer/performance evidence")
                return candidate
    raise AuditionError("Choose a current Auto Match suggestion for this character")


def _reuse_record(project, reuse_key: str) -> dict:
    try:
        record = lab._record(project, reuse_key)
        raw_identity = record.get("identity")
        identity = CharacterVoiceIdentity.model_validate(raw_identity) if raw_identity else None
        root = safe_project_root(project.id)
        builtin = (record.get("builtin_speaker") and record.get("builtin_snapshot")
                   and (not identity or (identity.mechanism == "builtin" and identity.backend == "qwen3tts_cv")))
        reference = (identity is not None and identity.mechanism == "qwen_reference" and identity.status == "LOCKED"
                     and identity.backend == "qwen3tts")
        if not (builtin or reference):
            raise AuditionError("Reuse requires a heard, reusable Qwen Base or CustomVoice identity")
        if reference:
            _read_verified(_safe_path(root, identity.reference_audio), identity.reference_audio_sha256)
            _read_verified(_safe_path(root, identity.prompt_artifact), identity.prompt_sha256)
        if not record.get("audio_file") or not record.get("audio_sha256"):
            if reference and record.get("kind") == "reference_clone" and identity.provenance.get("origin") == "eligible_reference_clone":
                from service.qwen_character_voice import _eligible_acquisition
                receipt, _, _ = _eligible_acquisition(root, identity.provenance["acquisition_id"], identity.provenance["acquisition_sha256"], allow_draft=True)
                if receipt["reference_audio_sha256"] != identity.reference_audio_sha256 or receipt["reference_text"] != identity.reference_text:
                    raise AuditionError("Prepared clone differs from its acquired recording")
                return record  # First audition of an exact eligible recording, never an unheard VoiceDesign reuse.
            raise AuditionError("Reuse requires a heard production voice with a verified sample")
        _read_verified(_safe_path(root, record["audio_file"]), record["audio_sha256"])
        return record
    except AuditionError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise AuditionError(f"Reusable voice identity or assets are invalid: {exc}") from exc


def _candidate_for_intent(project, role_id, candidate_id, reuse_key):
    if reuse_key:
        record = _reuse_record(project, reuse_key)
        for slot in (role_id, "general"):
            for candidate in project.voice_lab.shortlists.get(slot, {}).get("matches", []):
                if candidate.get("candidate_id", candidate.get("id")) == candidate_id:
                    return copy.deepcopy(candidate)
        return {"candidate_id": candidate_id, "id": candidate_id,
                "original_casting": True, "reuse_key": reuse_key,
                "creative_direction": "Reuse this exact heard, reusable voice identity.",
                "name": record.get("name"), "tags": copy.deepcopy(record.get("tags", []))}
    return _candidate(project, role_id, candidate_id)


def _bound_intent(project, role_id, candidate_id, *, script_version=None, reuse_key=None) -> tuple[str, dict]:
    role = next((r for r in project.roles if r.id == role_id), None)
    if role is None:
        raise AuditionError("Choose a real character before previewing")
    candidate = _candidate_for_intent(project, role_id, candidate_id, reuse_key)
    from .cloned_voice_policy import policy_mode, require_record, ClonePolicyError
    clone_mode = policy_mode(project)
    if clone_mode == "cloned_only":
        if not reuse_key:
            raise AuditionError("Cloned-only auditions require an exact saved recorded reference; synthetic design is disabled")
        try:
            require_record(_reuse_record(project, reuse_key), mode=clone_mode)
        except ClonePolicyError as exc:
            raise AuditionError(str(exc)) from exc
    profile = project.character_cast.profiles.get(role_id)
    dossier = project.character_cast.dossiers.get(role_id)
    # Ranking scores/preferences do not alter a voice and therefore must not
    # invalidate cached sound every time the owner presses a favorite heart.
    direction = {key: copy.deepcopy(candidate.get(key)) for key in (
        "candidate_id", "id", "performer", "performance", "design_instruction", "creative_direction",
        "source_urls", "tags", "original_casting", "reuse_key", "name", "round_id")}
    direction["delivery_instruction"] = copy.deepcopy(candidate.get("delivery_instruction"))
    from .original_inspired_design import authorized_candidate
    if authorized_candidate(candidate):
        direction["original_design"] = copy.deepcopy(candidate["reference"])
    script_data = audition_scripts.script(script_version)
    sections = []
    for index, section in enumerate(script_data["sections"]):
        text = section.get("spoken_text") if script_version else PREVIEW_TEXT
        section_id = section["id"]
        segment_id = "preview-utterance-v1" if script_version is None else f"preview-{script_version}-{section_id}"
        section_row = {"id": section_id, "label": section["label"], "spoken_text": text,
                       "segment_id": segment_id}
        if script_version:
            section_row["delivery_instruction"] = section.get("delivery_instruction")
        sections.append(section_row)
    preview_text = "\n\n".join(row["spoken_text"] for row in sections)
    bound = {"schema": "voice-lab-short-preview-v2", "role_id": role_id,
             "candidate_id": candidate_id, "candidate": direction,
             "profile": profile.model_dump(mode="json") if profile else None,
             "dossier_revision": dossier.revision if dossier else None,
             "preview_text": preview_text, "target_seconds": script_data["target_seconds"],
             "duration_window_seconds": script_data["duration_window_seconds"],
             "script_version": script_version, "sections": sections, "reuse_key": reuse_key}
    if clone_mode == "cloned_only":
        bound["director_voice_policy"] = {"mode": clone_mode}
    # Freeze new Base requests before hashing; stored intents are never rehashed.
    # A reused built-in CustomVoice keeps its independent execution contract.
    backend = "qwen3tts"
    if reuse_key:
        reusable = _reuse_record(project, reuse_key)
        backend = ((reusable.get("builtin_snapshot") or {}).get("settings", {}).get("tts_backend")
                   or (reusable.get("identity") or {}).get("backend") or backend)
    if backend in {"qwen3tts", "qwen3tts_voicedesign"}:
        from .project_qwen_runtime import project_profile
        bound["requested_runtime"] = project_profile(project.id)
    return lab.digest(bound), bound


# Previews whose audio was already checked, by file identity: (path, size, mtime) -> hash.
# Every page load reports every preview, and re-reading and hashing each audio file to say
# "this one is ready" made opening the page cost seconds. A file is checked again the moment
# its size or modification time changes, so a replaced or damaged file is still caught.
_CHECKED_AUDIO: dict[tuple, str] = {}


def _audio_still_verified(path, expected):
    try:
        stat = path.stat()
    except OSError:
        stat = None
    identity = (str(path), stat.st_size, stat.st_mtime_ns) if stat else None
    if identity is not None and _CHECKED_AUDIO.get(identity) == expected:
        return
    _read_verified(path, expected)
    if identity is not None:
        _CHECKED_AUDIO[identity] = expected


def _result(project, request_id, *, reused=True, jobs=None) -> dict:
    state = project.metadata.get(_META, {})
    request = state.get("requests", {}).get(request_id)
    if not request:
        raise AuditionError("Unknown preview request")
    intent = state["intents"][request["intent"]]
    job = jobs.get(intent["job_id"]) if jobs is not None else get_job(project.id, intent["job_id"])
    status = job.status.value if job else "queued"
    reason = job.message if job else "Preview request is saved; press Resume to dispatch it"
    if status == "running" and not is_job_active(project.id, intent["job_id"]):
        status, reason = "interrupted", "Studio restarted; resume the same preview without duplicating completed work"
    media_key = intent.get("media_key") or intent.get("key")
    record = project.voice_lab.voices.get(media_key, {})
    key = media_key if status == "succeeded" and record else None
    if key:
        _audio_still_verified(_safe_path(safe_project_root(project.id), record["audio_file"]), record["audio_sha256"])
    return {"request_id": request_id, "job_id": intent["job_id"], "status": status,
            "role_id": intent["role_id"], "candidate_id": intent["candidate_id"],
            "expected_identity_id": request["expected_identity_id"], "key": key,
            "reason": reason, "reused": reused,
            "status_url": f"/api/v1/projects/{project.id}/voice-lab/auditions/{request_id}",
            "job": job.model_dump(mode="json") if job else None,
            "preview_text": intent["binding"]["preview_text"],
            "target_seconds": intent["binding"].get("target_seconds", TARGET_SECONDS),
            "script_version": intent["binding"].get("script_version"),
            "duration_seconds": record.get("duration_seconds") if key else None,
            "audio_sha256": record.get("audio_sha256") if key else None,
            "sections": copy.deepcopy(record.get("sections")) if key else None,
            "duration_window_seconds": record.get("duration_window_seconds") if key else None,
            "duration_outlier": record.get("duration_outlier") if key else None,
            "duration_status": record.get("duration_status") if key else None}


def audition_status(project_id: str, request_id: str) -> dict:
    """Read only: polling and reopening a page cannot trigger model work."""
    return _result(load_project(project_id), request_id)


def audition_summaries(project) -> list[dict]:
    """One newest request per intent for ordinary Voice Lab refresh/resume."""
    state = project.metadata.get(_META, {})
    latest = {}
    for request_id, request in state.get("requests", {}).items():
        latest[request["intent"]] = request_id
    if not latest:
        return []
    # The jobs file is read once for all of them, not once per preview.
    jobs = {job.id: job for job in load_jobs(project.id)}
    return [_result(project, request_id, jobs=jobs) for request_id in latest.values()]


def _job_status(project_id, job_id, status, message):
    def change(job):
        job.status, job.message = status, str(message)[:2000]
        job.finished_at = datetime.now(timezone.utc) if status == SynthesisJobStatus.SUCCEEDED else None
    return mutate_job(project_id, job_id, change)


def _verified_preview_source(child, binding):
    rows = load_segments(child.id)
    expected_text = binding["preview_text"]
    sections = binding.get("sections") or [{"id": "preview", "label": "PREVIEW",
                                             "spoken_text": expected_text,
                                             "segment_id": "preview-utterance-v1"}]
    expected = child.metadata.get("voice_lab_audition", {})
    saved_sections = expected.get("sections") or sections
    if (expected.get("preview_text") != expected_text
            or expected.get("script_version") != binding.get("script_version")
            or saved_sections != sections
            or len(rows) != len(sections)
            or len(child.source_revisions) != 1 or len(child.roles) != 1):
        raise AuditionError("Original preview source changed; use a new explicit preview version")
    source = child.source_revisions[0]
    content = _read_verified(_safe_path(safe_project_root(child.id), source.snapshot_file), source.sha256)
    anchor = rows[0].source_anchor
    if content != expected_text.encode("utf-8"):
        raise AuditionError("Original preview source changed")
    cursor = 0
    for row, section in zip(rows, sections):
        anchor = row.source_anchor
        if (row.text != section["spoken_text"] or row.id != section["segment_id"]
                or not anchor or anchor.source_revision_id != source.id or anchor.start_offset != cursor
                or anchor.end_offset != cursor + len(row.text) or row.role_id != child.roles[0].id
                or row.playback_speed != 1.0 or row.edit_state):
            raise AuditionError("Original preview source changed or natural playback settings changed")
        expected_control, _ = _section_control(child.roles[0], section)
        if row.emotion_text != expected_control:
            raise AuditionError("Original preview delivery control changed")
        cursor = anchor.end_offset + 2
    return rows


def _clone_reusable_voice(parent_id, child, intent):
    reuse_key = intent.get("reuse_key")
    if not reuse_key:
        return None
    parent = load_project(parent_id)
    source_record = _reuse_record(parent, reuse_key)
    identity = (CharacterVoiceIdentity.model_validate(source_record["identity"])
                if source_record.get("identity") else None)
    child_root = safe_project_root(child.id)
    parent_root = safe_project_root(parent_id)
    cloned = copy.deepcopy(source_record)
    cloned_identity = identity.model_copy(deep=True) if identity else None
    if identity and identity.mechanism == "qwen_reference":
        server_receipt = parent_root / f"casting/voices/{identity.id}/reference.json"
        if server_receipt.is_file():
            blob = server_receipt.read_bytes()
            _publish_blob(child_root, f"casting/voices/{identity.id}/reference.json", blob, hashlib.sha256(blob).hexdigest())
        elif identity.provenance.get("audition_draft_policy"):
            raise AuditionError("Recorded draft lacks server-owned reference provenance")
        for field, hash_field, name in (("reference_audio", "reference_audio_sha256", "reference.wav"),
                                        ("prompt_artifact", "prompt_sha256", "conditioning.npz")):
            blob = _read_verified(_safe_path(parent_root, getattr(identity, field)), getattr(identity, hash_field))
            cloned_path = f"casting/voices/{identity.id}/{name}"
            cloned_identity = cloned_identity.model_copy(update={field: _publish_blob(
                child_root, cloned_path, blob, getattr(identity, hash_field))})
    if identity and identity.provenance.get("origin") == "eligible_reference_clone":
        from service.qwen_character_voice import _eligible_acquisition
        provenance = identity.provenance
        acquired, original, reference = _eligible_acquisition(parent_root, provenance["acquisition_id"], provenance["acquisition_sha256"], allow_draft=True)
        receipt_path = parent_root / f"voice_lab/discovery/acquisitions/{provenance['acquisition_id']}/acquisition.json"
        for source, expected_hash in [(receipt_path, provenance["acquisition_sha256"]),
                                      (original, acquired["original_sha256"]), (reference, acquired["reference_audio_sha256"])]:
            relative = source.relative_to(parent_root).as_posix()
            _publish_blob(child_root, relative, _read_verified(source, expected_hash), expected_hash)
    if source_record.get("audio_file") and source_record.get("audio_sha256"):
        sample = _read_verified(_safe_path(parent_root, source_record["audio_file"]), source_record["audio_sha256"])
        suffix = PurePosixPath(source_record["audio_file"]).suffix or ".wav"
        cloned["audio_file"] = _publish_blob(child_root,
            f"voice_lab/library/reuse/{reuse_key}/sample-{source_record['audio_sha256']}{suffix}",
            sample, source_record["audio_sha256"])
    cloned["identity"] = cloned_identity.model_dump(mode="json") if cloned_identity else None
    cloned["role_id"] = intent["role_id"]
    cloned["role_name"] = next(r.name for r in child.roles if r.id == intent["role_id"])
    child.voice_lab.voices[reuse_key] = cloned
    role = next(r for r in child.roles if r.id == intent["role_id"])
    if ((cloned_identity and (not role.voice_identity or role.voice_identity.id != cloned_identity.id))
            or not cloned_identity):
        lab.apply_assignment(child, reuse_key, role.id, _current_identity(role))
    delivery = intent["binding"].get("candidate", {}).get("delivery_instruction")
    if delivery:
        if role.tts_backend != "qwen3tts_cv" or not role.qwen3_speaker:
            raise AuditionError("This voice backend cannot apply the requested performance direction")
        if delivery not in (role.style_instruction or ""):
            role.style_instruction = " ".join(part for part in (role.style_instruction, delivery) if part)
    save_project(child)
    return reuse_key


def _prepare_project(parent_id, intent_hash):
    with project_meta_lock(parent_id):
        parent = load_project(parent_id)
        intent = _state(parent)["intents"][intent_hash]
        binding = intent["binding"]
        preview_text = binding["preview_text"]
        sections = binding.get("sections") or [{"id": "preview", "label": "PREVIEW",
            "spoken_text": preview_text, "segment_id": "preview-utterance-v1"}]
        child = None
        if intent.get("audition_project_id"):
            child = load_project(intent["audition_project_id"])
            if (child.metadata.get("voice_lab_audition", {}).get("intent") != intent_hash
                    or child.metadata.get("voice_lab_audition", {}).get("parent_project_id") != parent_id):
                raise AuditionError("Preview project is not bound to this request")
        else:
            # Recover a fully committed child if a crash occurred immediately
            # before its parent pointer was saved. No new generation is needed.
            for path in get_config().projects_root_abs.glob("*/project.json"):
                candidate = load_project(path.parent.name)
                meta = candidate.metadata.get("voice_lab_audition", {})
                if meta.get("parent_project_id") == parent_id and meta.get("intent") == intent_hash:
                    child = candidate
                    break
        if child is None:
            original = next(r for r in parent.roles if r.id == intent["role_id"])
            child = create_project(f"Voice preview · {original.name} · {intent_hash[:8]}",
                                   description="Original neutral audition; never a book excerpt")
            role = original.model_copy(deep=True)
            role.voice_identity = None
            role.speaker_audio = role.speaker_voice_id = role.qwen3_speaker = None
            role.tts_backend, role.qwen3_language = "qwen3tts", "English"
            role.voice_description = role.style_instruction = role.default_emotion_text = None
            role.default_emotion_vector = None
            child.roles = [role]
            child.chapters = [Chapter(id="preview", title="Original voice preview")]
            child.character_cast.policy.enabled = False
            child.character_cast.policy.audition_mode = "off"
            child.character_cast.policy.max_audition_characters = 0
            if intent["binding"]["profile"]:
                from dubbing_service.app.models.character import CastingProfile
                child.character_cast.profiles[role.id] = CastingProfile.model_validate(intent["binding"]["profile"])
            child.voice_lab.shortlists[role.id] = {"matches": [copy.deepcopy(intent["candidate_snapshot"])]}
            if binding.get("director_voice_policy"):
                child.metadata["director_voice_policy"] = copy.deepcopy(binding["director_voice_policy"])
            child.metadata["voice_lab_audition"] = {"parent_project_id": parent_id, "intent": intent_hash,
                "role_id": role.id, "preview_text": preview_text, "script_version": binding.get("script_version"),
                "sections": copy.deepcopy(sections), "source_kind": "original_neutral_preview"}
            content = preview_text.encode("utf-8")
            checksum = hashlib.sha256(content).hexdigest()
            relative = "source_revisions/voice-preview.txt"
            _publish_blob(safe_project_root(child.id), relative, content, checksum)
            child.source_revisions = [SourceRevision(id="preview-source-v1", uri="inline:voice-lab-original-neutral-preview-v1",
                kind="inline", media_type="text/plain", sha256=checksum, byte_length=len(content), snapshot_file=relative,
                title="Original neutral voice preview")]
            if intent.get("reuse_key"):
                _clone_reusable_voice(parent_id, child, intent)
                child = load_project(child.id)
            role = child.roles[0]
            segments = []
            cursor = 0
            for index, section in enumerate(sections):
                section_control, _ = _section_control(role, section)
                segment = Segment(id=section["segment_id"], chapter_id="preview", role_id=role.id,
                    index=index, text=section["spoken_text"], tts_backend=role.tts_backend,
                    emotion_text=section_control,
                    attribution=Attribution(speaker_id=role.id, source="inline", confidence=1.0,
                                            note="Original audition section assigned to one explicitly requested role"),
                    source_anchor=SourceAnchor(source_revision_id="preview-source-v1", chapter_id="preview",
                                               start_offset=cursor, end_offset=cursor + len(section["spoken_text"]),
                                               occurrence_index=index))
                segments.append(segment)
                cursor += len(section["spoken_text"]) + 2
            commit_project_state(child.id, project=child, segments=segments)
        if intent.get("reuse_key"):
            _clone_reusable_voice(parent_id, child, intent)
            child = load_project(child.id)
            backend = child.roles[0].tts_backend
            stored_rows = load_segments(child.id)
            changed = False
            for row, section in zip(stored_rows, sections):
                expected_control, _ = _section_control(child.roles[0], section)
                if row.tts_backend != backend or row.emotion_text != expected_control:
                    row.tts_backend = backend
                    row.emotion_text = expected_control
                    changed = True
            if changed:
                from dubbing_service.app.storage.segments import save_segments
                save_segments(child.id, stored_rows)
        intent["audition_project_id"] = child.id
        lab._save(parent)
        if binding.get("requested_runtime"):
            selected = binding["requested_runtime"]
            prior = child.metadata.get("requested_runtime")
            if prior is not None and prior != selected:
                raise AuditionError("Preview runtime changed; start a new whole audition intent")
            child.metadata["requested_runtime"] = selected
            save_project(child)
        return child.id


def _assemble_audition_wav(parts: list[bytes]) -> bytes:
    """Join verified selected PCM Takes without inserting, trimming, or stretching audio."""
    output = io.BytesIO()
    expected = None
    with wave.open(output, "wb") as destination:
        for content in parts:
            try:
                with wave.open(io.BytesIO(content), "rb") as source:
                    params = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if expected is None:
                        expected = params
                        destination.setnchannels(params[0])
                        destination.setsampwidth(params[1])
                        destination.setframerate(params[2])
                    elif params != expected:
                        raise AuditionError("Audition sections have incompatible PCM formats")
                    destination.writeframes(source.readframes(source.getnframes()))
            except wave.Error as exc:
                raise AuditionError("Selected audition Take is not a decodable PCM WAV") from exc
    return output.getvalue()


def _publish(parent_id, intent_hash, child_id, key):
    intent = _state(load_project(parent_id))["intents"][intent_hash]
    binding = intent["binding"]
    preview_text = binding["preview_text"]
    child = load_project(child_id)
    rows = _verified_preview_source(child, binding)
    role = child.roles[0]
    expected_voice_id = role.voice_identity.id if role.voice_identity else None
    expected_version = voice_revision_for_role(child.roles[0])
    selected = []
    audio_paths = []
    audio_bytes = []
    section_manifest = []
    cursor = 0.0
    for row, section in zip(rows, binding["sections"]):
        take = next((t for t in row.takes if t.id == row.selected_take_id), None)
        identity_stamp = (take.generation_metadata.get("character_voice_identity") or {}) if take else {}
        is_builtin = role.tts_backend == "qwen3tts_cv"
        identity_ok = (identity_stamp.get("id") == expected_voice_id
                       and identity_stamp.get("role_id") == role.id) if expected_voice_id else (
                           is_builtin and take is not None
                           and take.fingerprint_components.get("builtin_speaker") == role.qwen3_speaker)
        if (not take or take.stale or not take.qa_report or not take.qa_report.passed
                or not take.qa_report.matches(take.id, take.audio_sha256)
                or take.role_id != role.id or not identity_ok
                or take.fingerprint_components.get("voice_revision") != expected_version
                or pipeline._requirement_for_take(child, row, take).fingerprint() != take.input_fingerprint):
            raise AuditionError("Preview lacks a current, selected, identity-matched, hash-bound QA-passing Take")
        audio_path = _safe_path(safe_project_root(child_id), take.audio_file)
        content = _read_verified(audio_path, take.audio_sha256)
        try:
            with wave.open(io.BytesIO(content), "rb") as source:
                duration = source.getnframes() / source.getframerate()
        except (wave.Error, ZeroDivisionError) as exc:
            raise AuditionError("Selected audition Take is not a decodable PCM WAV") from exc
        selected.append((row, section, take, duration, content))
        audio_paths.append(audio_path)
        audio_bytes.append(content)
        _, control_status = _section_control(role, section)
        section_manifest.append({"id": section["id"], "label": section["label"],
            "spoken_text": section["spoken_text"], "utterance_id": row.id,
            "delivery_instruction": section.get("delivery_instruction"),
            "control_status": control_status,
            "take_id": take.id, "audio_sha256": take.audio_sha256,
            "input_fingerprint": take.input_fingerprint,
            "start_seconds": round(cursor, 6), "end_seconds": round(cursor + duration, 6)})
        cursor += duration
    assembled = _assemble_audition_wav(audio_bytes)
    audio_hash = hashlib.sha256(assembled).hexdigest()
    duration = sum(row[3] for row in selected)
    if not selected or duration <= 0:
        raise AuditionError("Audition assembly has no verified production Takes")
    from . import qa
    master_path = _publish_blob(safe_project_root(parent_id),
        f"voice_lab/auditions/{intent_hash}/sample-{audio_hash}.wav", assembled, audio_hash)
    master_qa = qa.qa_take_audio(take_id="audition_" + intent_hash[:24], audio_path=_safe_path(
        safe_project_root(parent_id), master_path), text=preview_text, expected_sha256=audio_hash,
        notes="Audition assembly from ordered selected, blocking-QA-passing identity-bound Takes")
    if not master_qa.passed or not master_qa.matches("audition_" + intent_hash[:24], audio_hash):
        raise AuditionError("Assembled audition failed output-bound technical QA")
    base_record = lab.voice_records(child, rows).get(key) or child.voice_lab.voices.get(key)
    if not base_record or (expected_voice_id and base_record.get("identity", {}).get("id") != expected_voice_id):
        raise AuditionError("Production preview voice version differs from the assigned identity")
    identity = copy.deepcopy(base_record.get("identity"))
    if identity and identity.get("mechanism") == "qwen_reference":
        if identity.get("status") != "LOCKED":
            raise AuditionError("Preview must use locked reusable Base conditioning")
    elif not (role.tts_backend == "qwen3tts_cv" and base_record.get("builtin_speaker") == role.qwen3_speaker):
        raise AuditionError("Preview voice version differs from the assigned reusable identity")
    child_root, parent_root = safe_project_root(child_id), safe_project_root(parent_id)
    prefix = f"voice_lab/auditions/{intent_hash}"
    if identity and identity.get("mechanism") == "qwen_reference":
        for field, hash_field, name in (("reference_audio", "reference_audio_sha256", "reference.wav"),
                                        ("prompt_artifact", "prompt_sha256", "conditioning.npz")):
            content = _read_verified(_safe_path(child_root, identity[field]), identity[hash_field])
            identity[field] = _publish_blob(parent_root, f"{prefix}/{name}", content, identity[hash_field])
    first_row, first_section, first_take, _, _ = selected[0]
    media_key = "v_" + lab.digest([expected_voice_id or role.qwen3_speaker,
                                    base_record.get("version"), intent_hash])[:28]
    record = copy.deepcopy(base_record)
    record.update(key=media_key, identity=identity, role_id=intent["role_id"],
        role_name=next(r.name for r in load_project(parent_id).roles if r.id == intent["role_id"]),
        audio_file=master_path, audio_sha256=audio_hash, duration_seconds=duration,
        generation_language=role.qwen3_language,
        take_id=first_take.id, take_ids=[item[2].id for item in selected],
        utterance_id=first_row.id, technical_qa="passed", human_review="pending",
        preview_text=preview_text, audition_project_id=child_id, script_version=binding.get("script_version"),
        target_seconds=binding["target_seconds"], duration_window_seconds=binding.get("duration_window_seconds"),
        duration_outlier=bool(binding.get("duration_window_seconds") and not (
            binding["duration_window_seconds"][0] <= duration <= binding["duration_window_seconds"][1])),
        duration_status=("measured" if not binding.get("duration_window_seconds") else
            "measured_outlier" if not (binding["duration_window_seconds"][0] <= duration <= binding["duration_window_seconds"][1])
            else "measured_within_target"),
        sections=section_manifest, lines=[{"utterance_id": item[0].id,
            "start_seconds": section_manifest[index]["start_seconds"],
            "end_seconds": section_manifest[index]["end_seconds"]}
            for index, item in enumerate(selected)])
    proof_takes = [item[2].model_dump(mode="json") for item in selected]
    proof = (json.dumps({"schema": "voice-lab-audition-takes-v2", "project_id": child_id,
        "source": child.source_revisions[0].model_dump(mode="json"), "source_text": preview_text,
        "script_version": binding.get("script_version"), "sections": section_manifest,
        "take": proof_takes[0] if len(proof_takes) == 1 else None, "takes": proof_takes,
        "assembled_audio_sha256": audio_hash, "assembled_qa": master_qa.model_dump(mode="json")},
        ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    proof_hash = hashlib.sha256(proof).hexdigest()
    proof_file = _publish_blob(parent_root, f"{prefix}/take.json", proof, proof_hash)
    record["explanation"] = "One reusable synthetic voice identity, rendered through the production path for each audition section."
    record["details"] = {**record.get("details", {}), "audition_project_id": child_id,
        "take_proof_file": proof_file, "take_proof_sha256": proof_hash,
        "source_kind": "original_neutral_preview", "script_version": binding.get("script_version"),
        "target_seconds": binding["target_seconds"], "duration_window_seconds": binding.get("duration_window_seconds"),
        "duration_outlier": record["duration_outlier"], "duration_status": record["duration_status"],
        "sections": section_manifest,
        "assembled_qa": master_qa.model_dump(mode="json")}
    with project_meta_lock(parent_id):
        parent = load_project(parent_id)
        intent = _state(parent)["intents"][intent_hash]
        existing = parent.voice_lab.voices.get(media_key)
        if existing and existing.get("audio_sha256") and existing["audio_sha256"] != audio_hash:
            raise AuditionError("A heard voice version is immutable; refusing to replace its preview")
        parent.voice_lab.voices.setdefault(media_key, record)
        intent["key"] = key
        intent["media_key"] = media_key
        intent["completed_at"] = lab.now()
        lab._save(parent)


def _run(parent_id, intent_hash, *, design_provider=None, renderer=None):
    parent = load_project(parent_id)
    intent = _state(parent)["intents"][intent_hash]
    job_id = intent["job_id"]
    lease = None
    try:
        if lab.production_busy():
            _job_status(parent_id, job_id, SynthesisJobStatus.QUEUED,
                        "Existing production has GPU priority. Press Resume when it finishes.")
            return
        start_job(parent_id, job_id)
        current = get_job(parent_id, job_id)
        if current.stop_requested or current.status == SynthesisJobStatus.CANCELLED:
            return
        child_id = _prepare_project(parent_id, intent_hash)
        child = load_project(child_id)
        binding = intent["binding"]
        rows = _verified_preview_source(child, binding)
        role = child.roles[0]
        ensure_backend_ready(role.tts_backend or "qwen3tts", job_id=job_id, project_id=child_id)
        if intent.get("reuse_key"):
            voice_key = _clone_reusable_voice(parent_id, child, intent)
            child = load_project(child_id)
            role = child.roles[0]
        else:
            designed = lab.design(child_id, role.id, intent["candidate_id"],
                # Keep the design intent stable on retries after assigning it below.
                child.metadata.get("preview_design_expected", _current_identity(role)),
                "short-preview-design-v1", provider=design_provider)
            with project_meta_lock(child_id):
                child = load_project(child_id)
                role = child.roles[0]
                if "preview_design_expected" not in child.metadata:
                    child.metadata["preview_design_expected"] = _current_identity(role)
                identity = CharacterVoiceIdentity.model_validate(child.voice_lab.voices[designed["key"]]["identity"])
                if role.voice_identity and role.voice_identity.id != identity.id:
                    raise AuditionError("Isolated preview cast changed; refusing stale synthesis")
                role.voice_identity, role.tts_backend = identity, "qwen3tts"
                save_project(child)
            voice_key = designed["key"]
        current = get_job(parent_id, job_id)
        if current.stop_requested or current.status == SynthesisJobStatus.CANCELLED:
            return
        if lab.production_busy():
            _job_status(parent_id, job_id, SynthesisJobStatus.QUEUED,
                        "Conditioning is saved. Production has GPU priority; press Resume for the short preview.")
            return
        lease = try_register_running_batch(child_id)
        if lease is None:
            _job_status(parent_id, job_id, SynthesisJobStatus.QUEUED,
                        "GPU lane is busy. Saved conditioning will be reused when you press Resume.")
            return
        backend = role.tts_backend or "qwen3tts"
        renderer = renderer or BackendRenderer(backend=backend, engine_id=backend, renderer_id=backend,
                                               project_root=safe_project_root(child_id))
        if hasattr(renderer, "runtime_contract"):
            from .runtime_binding import bind_job
            bind_job(child_id, job_id, renderer, rows, child.roles,
                     child.character_cast.role_aliases, storage_project_id=parent_id)
        result = pipeline.render_utterances(child_id, [row.id for row in rows], renderer=renderer,
                                           promote=True, run_qa=True, resume=True)
        if len(result) != len(rows) or not all((item.success and item.selected and item.qa_passed and not item.stale for item in result)):
            details = "; ".join(f"{item.segment_id}: {item.error or ','.join(item.stale_components) or 'blocking QA failed'}"
                                 for item in result if not (item.success and item.selected and item.qa_passed and not item.stale))
            raise AuditionError("Preview needs bounded repair: " + (details or "blocking production QA failed"))
        current = get_job(parent_id, job_id)
        if current.stop_requested or current.status == SynthesisJobStatus.CANCELLED:
            return  # Preserve the Take; do not publish cancelled work.
        _publish(parent_id, intent_hash, child_id, voice_key)
        for row in rows:
            record_job_segment(parent_id, job_id, row.id, success=True)
        _job_status(parent_id, job_id, SynthesisJobStatus.SUCCEEDED,
                    "Production-path audition ready; human review pending. Use for character is a separate explicit action.")
    except Exception as exc:
        current = get_job(parent_id, job_id)
        if current and current.status != SynthesisJobStatus.CANCELLED:
            _job_status(parent_id, job_id, SynthesisJobStatus.INTERRUPTED, str(exc))
    finally:
        if lease:
            unregister_running_batch(lease)
        mark_job_active(parent_id, job_id, False)


def audition(project_id: str, *, role_id: str, candidate_id: str,
             expected_identity_id: str | None, request_id: str,
             schedule: Callable | None = None, design_provider=None, renderer=None,
             script_version: str | None = None, reuse_key: str | None = None) -> dict:
    """An explicit bounded request; duplicate clicks reuse one durable intent."""
    if not request_id or len(request_id) > 100 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in request_id):
        raise AuditionError("Preview request ID must use 1–100 letters, digits, underscores or hyphens")
    try:
        audition_scripts.script(script_version)
    except ValueError as exc:
        raise AuditionError(str(exc)) from exc
    payload_hash = lab.digest([role_id, candidate_id, expected_identity_id, script_version, reuse_key])
    with project_meta_lock(project_id):
        project = load_project(project_id)
        state = _state(project)
        prior = state["requests"].get(request_id)
        if prior:
            if prior["payload_hash"] != payload_hash:
                raise AuditionError("Request ID is already bound to a different preview")
            intent_hash = prior["intent"]
            reused = True
        else:
            role = next((r for r in project.roles if r.id == role_id), None)
            if role is None or _current_identity(role) != expected_identity_id:
                raise AuditionError("Casting changed; refresh before requesting a preview")
            intent_hash, binding = _bound_intent(project, role_id, candidate_id,
                                                 script_version=script_version, reuse_key=reuse_key)
            reused = intent_hash in state["intents"]
            if len(state["requests"]) >= _request_budget(project):
                raise AuditionError("Preview request history budget reached")
            if not reused:
                legacy_count = sum(1 for existing in state["intents"].values()
                    if existing.get("binding", {}).get("script_version") is None)
                if script_version is None and legacy_count >= MAX_AUDITIONS:
                    raise AuditionError("This project's 12 explicit legacy preview budget is reached")
                if script_version is not None:
                    round_id = binding.get("candidate", {}).get("round_id") or "direct"
                    role_auditions = sum(1 for existing in state["intents"].values()
                        if existing.get("role_id") == role_id
                        and existing.get("binding", {}).get("script_version") is not None
                        and (existing.get("binding", {}).get("candidate", {}).get("round_id") or "direct") == round_id)
                    if role_auditions >= MAX_VERSIONED_AUDITIONS_PER_ROLE:
                        raise AuditionError("This character's casting round has reached the six-proposal audition budget")
                state["intents"][intent_hash] = {"role_id": role_id, "candidate_id": candidate_id,
                    "binding": binding, "candidate_snapshot": copy.deepcopy(
                        _candidate_for_intent(project, role_id, candidate_id, reuse_key)),
                    "reuse_key": reuse_key, "media_key": None,
                    "job_id": "audition_" + intent_hash[:24], "attempts": 0, "created_at": lab.now()}
            state["requests"][request_id] = {"intent": intent_hash, "payload_hash": payload_hash,
                                           "expected_identity_id": expected_identity_id}
            lab._save(project)
        intent = state["intents"][intent_hash]
        job_id = intent["job_id"]
        job = get_job(project_id, job_id)
        if job is None:
            section_ids = [section["segment_id"] for section in intent["binding"].get("sections", [])]
            backend = "qwen3tts"
            if intent.get("reuse_key"):
                reusable = _reuse_record(project, intent["reuse_key"])
                backend = ((reusable.get("builtin_snapshot") or {}).get("settings", {}).get("tts_backend")
                           or (reusable.get("identity") or {}).get("backend") or backend)
            job = upsert_job(project_id, SynthesisJob(id=job_id, project_id=project_id,
                project_name=project.name, job_type="voice_lab_listening", request_id=request_id,
                idempotency_key=job_id, intent_hash=intent_hash, segment_ids=section_ids,
                segment_backends={section_id: backend for section_id in section_ids},
                message="Explicit Voice Lab audition requested"))
        if job.status in {SynthesisJobStatus.SUCCEEDED, SynthesisJobStatus.CANCELLED}:
            return _result(project, request_id, reused=True)
        if not try_mark_job_active(project_id, job_id):
            return _result(project, request_id, reused=True)
        if lab.production_busy():
            mark_job_active(project_id, job_id, False)
            _job_status(project_id, job_id, SynthesisJobStatus.QUEUED,
                        "Existing production has GPU priority. Press Resume when it finishes.")
            return _result(project, request_id, reused=reused)
        if intent["attempts"] >= MAX_ATTEMPTS:
            mark_job_active(project_id, job_id, False)
            _job_status(project_id, job_id, SynthesisJobStatus.INTERRUPTED,
                        "Bounded preview repair attempts exhausted; inspect the saved job before further generation")
            return _result(project, request_id, reused=True)
        intent["attempts"] += 1
        lab._save(project)
    try:
        if schedule:
            schedule(_run, project_id, intent_hash, design_provider=design_provider, renderer=renderer)
        else:
            _run(project_id, intent_hash, design_provider=design_provider, renderer=renderer)
    except Exception:
        mark_job_active(project_id, job_id, False)
        raise
    return _result(load_project(project_id), request_id, reused=reused)
