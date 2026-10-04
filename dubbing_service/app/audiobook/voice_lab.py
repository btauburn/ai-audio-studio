"""Studio-owned Voice Lab. Reading/reviewing never synthesizes or recasts.

Production-path Takes supply the heard version; design references are conditioning
artifacts only. The private delivery process proxies this owner instead of becoming
a second writer. All long discovery work publishes with a generation token.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing_service.app.core.config import safe_project_root, get_config
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage.projects import _project_meta_path, load_project, save_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments, order_segments
from .fingerprint import file_sha256, voice_revision_for_role


class VoiceLabError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _save(project) -> None:
    project.voice_lab.revision += 1
    save_project(project)


def _path(project_id: str, relative: str) -> Path:
    root = safe_project_root(project_id).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".npz", ".json"}:
        raise VoiceLabError("Artifact path is outside this project")
    return path


def _duration(take) -> float:
    if take.chunk_map:
        return max(c.audio_end_ms or 0 for c in take.chunk_map) / 1000
    report = take.qa_report
    if report:
        for check in report.results:
            if check.check_id == "duration_positive":
                return float(check.details.get("measured", 0)) / 1000
    return 0


def candidate_conditioning(candidate) -> dict | None:
    """The acoustic conditioning a generated candidate must be built from.

    Either the conditioning recorded on the candidate when it was proposed, or —
    for a candidate proposed before that field existed — the same numbers derived
    again from its own anchor and casting concept. Derivation is pure and
    deterministic, so a resumed design repeats exactly the conditioning it started
    with. A candidate with no anchor has no conditioning at all, which is what
    stops production from quietly emitting an unanchored voice.
    """
    from . import original_inspired_design, performer_reference
    if original_inspired_design.authorized_candidate(candidate):
        return None
    stored = candidate.get("conditioning")
    if stored:
        return copy.deepcopy(stored)
    anchor = candidate.get("anchor") or {}
    if not anchor.get("anchored") or not anchor.get("performer") or not anchor.get("performance"):
        return None
    concept = candidate.get("concept") or {
        "key": "anchored-performance", "label": "its recorded reference performance",
        "emphasis": "the delivery the reference performance is known for",
        "bias": {}, "supported": {}, "inferred": {}}
    return performer_reference.conditioning_payload(performer_reference.resolve(anchor, concept))


def _inspiration(identity: dict) -> dict | None:
    inspiration = identity.get("inspiration")
    if not inspiration:
        return None
    return {"performer": inspiration.get("performer"),
            "performance": inspiration.get("performance") or inspiration.get("character"),
            "source_urls": inspiration.get("source_urls") or [inspiration.get("source", "")],
            "evidence_kind": "metadata_only_exploratory"}


def identity_for_assignment(identity: CharacterVoiceIdentity) -> CharacterVoiceIdentity:
    """Return the exact identity shape used by new provider assignments.

    Qwen Base requires conditioning assets to be identity-scoped. Library
    records remain immutable at their original paths; assignment changes only
    these two path fields while preserving all other identity evidence.
    """
    if identity.mechanism == "qwen_reference":
        return identity.model_copy(update={
            "reference_audio": f"casting/voices/{identity.id}/reference.wav",
            "prompt_artifact": f"casting/voices/{identity.id}/conditioning.npz",
        })
    return identity.model_copy(deep=True)


def voice_records(project, segments=None) -> dict[str, dict]:
    """Read exact Take stamps, never attribute old audio to today's role settings."""
    records = {k: dict(v) for k, v in project.voice_lab.voices.items()}
    roles = {r.id: r for r in project.roles}
    segments = order_segments(load_segments(project.id) if segments is None else segments, project.chapters)
    for segment in segments:
        for take in segment.takes:
            if not take.audio_sha256 or not take.audio_file or take.missing:
                continue
            revision = take.fingerprint_components.get("voice_revision")
            if not revision:
                continue  # No invented voice identity for unbound legacy sound.
            role = roles.get(take.role_id)
            identity = take.generation_metadata.get("character_voice_identity") or {}
            builtin = take.fingerprint_components.get("builtin_speaker")
            voice_id = identity.get("id") or (f"builtin_{take.engine_id}_{builtin}" if builtin else f"take_voice_{revision}")
            key = "v_" + digest([voice_id, revision])[:28]
            if key in records and records[key].get("audio_sha256"):
                continue
            conditioning = records.get(key, {})
            builtin_snapshot = None
            if builtin and (not identity or identity.get("mechanism") == "builtin"):
                from dubbing_service.app.models.project import Role
                fields = ("speaker_audio", "speaker_voice_id", "tts_backend", "qwen3_speaker",
                          "qwen3_language", "voice_description", "style_instruction", "default_emotion_mode",
                          "default_emotion_text", "default_emotion_vector", "emotion_alpha")
                stamped = take.generation_metadata.get("builtin_voice_snapshot")
                proof = "immutable_take_role_snapshot"
                # Legacy scalar revisions hash every field plus source role ID.
                # A matching current configuration can recover those values; a
                # changed role must never be used to relabel historical audio.
                if not stamped and not identity and role and voice_revision_for_role(role) == revision:
                    stamped = {field: getattr(role, field) for field in fields}
                    proof = "configuration_matches_immutable_take_revision"
                if isinstance(stamped, dict) and all(field in stamped for field in fields):
                    try:
                        candidate = Role(id=take.role_id, name="Historical voice", **{field: stamped[field] for field in fields},
                                         voice_identity=CharacterVoiceIdentity.model_validate(identity) if identity else None)
                        components = take.fingerprint_components
                        complete_effective_stamp = all(field in components for field in ("builtin_speaker", "language", "style_instruction"))
                        if (complete_effective_stamp and voice_revision_for_role(candidate) == revision
                                and candidate.tts_backend == "qwen3tts_cv" and candidate.qwen3_language
                                and candidate.qwen3_speaker == builtin
                                and (not candidate.voice_identity or (candidate.voice_identity.status == "LOCKED"
                                     and candidate.voice_identity.backend == "qwen3tts_cv"
                                     and candidate.voice_identity.builtin_speaker == builtin))
                                and str(candidate.qwen3_language).casefold() == str(components["language"]).casefold()
                                and (candidate.style_instruction or "") == components["style_instruction"]):
                            builtin_snapshot = {"schema": "voice-lab-builtin-role-v1", "source_role_id": take.role_id,
                                "source_voice_revision": revision, "source_take_id": take.id, "proof": proof,
                                "settings": {field: candidate.model_dump(mode="json")[field] for field in fields}}
                    except (ValueError, TypeError):
                        pass  # Incomplete historical configuration stays audible, not assignable.
            profile = project.character_cast.profiles.get(take.role_id)
            tags = [d.value for d in profile.decisions[:3]] if profile and profile.revision == identity.get("casting_profile_revision") else []
            kind = "original_inspired" if identity.get("provenance", {}).get("origin") == "generated_original_voice" else "provider_builtin" if builtin else "reference_clone"
            names = ["Cedar", "Willow", "Harbor", "Linden", "Rowan", "Juniper", "Ash", "Wren"]
            records[key] = {
                "key": key, "voice_id": voice_id, "version": revision,
                "name": f"{names[int(digest(voice_id)[:4],16) % len(names)]} {revision[:6]}",
                "kind": kind, "synthetic": True, "role_id": take.role_id,
                "role_name": role.name if role else "Historical role", "tags": tags,
                "inspiration": _inspiration(identity),
                "explanation": "Saved production Take from this exact conditioning identity." + (" Performance traits are creative interpretation, not measured celebrity acoustics." if identity.get("inspiration") else ""),
                "identity": identity, "builtin_speaker": builtin, "backend": identity.get("backend") or take.tts_backend,
                "builtin_snapshot": builtin_snapshot,
                "assignment_hold": "Historical built-in settings cannot be recovered from this Take" if builtin and not builtin_snapshot else None,
                "audio_file": take.audio_file, "audio_sha256": take.audio_sha256,
                "duration_seconds": _duration(take), "take_id": take.id,
                "utterance_id": segment.id, "technical_qa": "passed" if take.qa_report and take.qa_report.passed and take.qa_report.output_audio_hash == take.audio_sha256 else "unverified",
                "human_review": "pending", "lines": [{"utterance_id": segment.id, "start_seconds": 0, "end_seconds": _duration(take)}],
                "details": {"model": identity.get("model") or take.model_id, "runtime": identity.get("runtime_version"),
                            "reference_audio_sha256": identity.get("reference_audio_sha256"), "prompt_sha256": identity.get("prompt_sha256"),
                            "generation_settings": identity.get("generation_settings", {}),
                            "provenance": identity.get("provenance", {}), "production_take_id": take.id,
                            "voice_version": revision, "audio_sha256": take.audio_sha256},
            }
            if conditioning:
                # Identity/design data is immutable. Append the first actual
                # production-path binding; never swap an already heard sample.
                records[key].update({k: conditioning[k] for k in (
                    "name", "identity", "inspiration", "tags", "kind"
                ) if k in conditioning})
    return records


def view(project_id: str, base_url: str | None = None) -> dict:
    from .voice_lab_audition import audition_summaries
    project = load_project(project_id)
    base = base_url or f"/api/v1/projects/{project_id}/voice-lab"
    voices = []
    for key, record in voice_records(project).items():
        preference = project.voice_lab.preferences.get(key, {})
        # Readiness derives from verified bytes, never a saved URL/hash alone.
        available, audio_error = False, None
        if record.get("audio_file") and record.get("audio_sha256"):
            try:
                from .voice_lab_audition import _audio_still_verified
                _audio_still_verified(_path(project_id, record["audio_file"]), record["audio_sha256"])
                available = True
            except (ValueError, OSError) as exc:
                audio_error = str(exc)
        voices.append({k: v for k, v in {**record, "favorite": preference.get("favorite", False),
            "rating": preference.get("rating"), "suitability": preference.get("suitability", {}),
            "audio_available": available, "audio_error": audio_error, "audio_url": f"{base}/audio/{key}" if available else None}.items() if k not in {"audio_file", "identity"}})
    excerpt = dict(project.voice_lab.excerpt) if project.voice_lab.excerpt else None
    if excerpt and excerpt.get("audio_file"):
        try:
            from .voice_lab_audition import _audio_still_verified
            _audio_still_verified(_path(project_id, excerpt["audio_file"]), excerpt["audio_sha256"])
            excerpt["audio_available"] = True
            excerpt["audio_url"] = f"{base}/audio/excerpt"
        except (ValueError, OSError, KeyError) as exc:
            excerpt["audio_available"] = False
            excerpt["audio_url"] = None
            excerpt["audio_error"] = str(exc)
        excerpt.pop("audio_file", None)
        excerpt["comparisons"] = [{**{k: v for k, v in row.items() if k != "audio_file"},
            "audio_url": f"{base}/audio/{key}"} for key, row in comparison_records(project).items()]
    return {"project_id": project_id, "project_name": project.name, "revision": project.voice_lab.revision,
            "scope": project.voice_lab.scope, "voices": voices, "auditions": audition_summaries(project),
            "capabilities": {"casting_review": True, "casting_enabled": bool(project.voice_lab.casting.get("enabled") or project.metadata.get("autocast_import", {}).get("mode") == "autocast")},
            "roles": [{"id": r.id, "name": r.name, "current_identity_id": r.voice_identity.id if r.voice_identity else voice_revision_for_role(r),
                       "dossier": project.character_cast.dossiers[r.id].model_dump(mode="json") if r.id in project.character_cast.dossiers else None}
                      for r in project.roles],
            "shortlists": project.voice_lab.shortlists, "review_position": project.voice_lab.review_position,
            "automation": project.voice_lab.automation, "excerpt": excerpt, "issues": project.voice_lab.issues}


def _record(project, key: str) -> dict:
    # A voice that is already recorded with its audio is returned as it is: ``voice_records``
    # never rewrites such a record, it only adds voices found in Takes. Rebuilding the whole
    # index means reading the entire segment store, and a rating (which looks up the voice it
    # is about) paid that on every save.
    known = project.voice_lab.voices.get(key)
    if known and known.get("audio_sha256"):
        return dict(known)
    record = voice_records(project).get(key)
    if not record:
        raise VoiceLabError("Voice version no longer available; refresh Voice Lab")
    return record


def comparison_records(project) -> dict[str, dict]:
    excerpt = project.voice_lab.excerpt or {}
    expected = [row["utterance_id"] for row in excerpt.get("map", [])]
    result = {}
    if not expected:
        return result
    for row in excerpt.get("comparisons", []):
        measurement = row.get("loudness_measurement", {})
        lufs = row.get("loudness_lufs")
        if (row.get("utterance_ids") == expected and row.get("passage_id") == excerpt.get("manifest_id")
                and isinstance(lufs, (int, float)) and math.isfinite(lufs)
                and measurement.get("audio_sha256") == row.get("audio_sha256")
                and measurement.get("integrated_lufs") == lufs
                and row.get("audio_file") and row.get("key")):
            result[row["key"]] = row
    return result


# A player issues several ranged requests per audition. Resolving each one parses and
# validates the whole project file (megabytes for a many-character book), which made
# the first byte of audio wait behind work that never changes between requests. The
# resolved lookup is therefore remembered against the project file's own identity:
# any save replaces that file, changes its stamp, and so invalidates the entry.
_AUDIO_LOOKUP: dict[tuple[str, str], tuple[tuple[int, int], tuple[Path, dict]]] = {}
_AUDIO_LOOKUP_LIMIT = 1024


def _project_stamp(project_id: str) -> tuple[int, int] | None:
    try:
        stat = _project_meta_path(project_id).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def audio_record(project_id: str, key: str) -> tuple[Path, dict]:
    # The stamp is read before the project so a save that lands mid-lookup can only
    # make the entry look stale, never make stale data look current.
    stamp = _project_stamp(project_id)
    hit = _AUDIO_LOOKUP.get((project_id, key)) if stamp else None
    if hit and hit[0] == stamp:
        return hit[1]
    project = load_project(project_id)
    record = project.voice_lab.excerpt if key == "excerpt" else comparison_records(project).get(key) or _record(project, key)
    if not record or not record.get("audio_file") or not record.get("audio_sha256"):
        raise VoiceLabError("No production audio is available for this version")
    path = _path(project_id, record["audio_file"])
    if stamp:
        if len(_AUDIO_LOOKUP) >= _AUDIO_LOOKUP_LIMIT:
            _AUDIO_LOOKUP.clear()
        _AUDIO_LOOKUP[(project_id, key)] = (stamp, (path, record))
    return path, record


def _verified_record(project, key: str, audio_sha256: str) -> dict:
    record = _record(project, key)
    if not record.get("audio_sha256") or record["audio_sha256"] != audio_sha256:
        raise VoiceLabError("Audio version changed; refresh before reviewing")
    path = _path(project.id, record["audio_file"])
    if not path.is_file() or file_sha256(path) != audio_sha256:
        raise VoiceLabError("Audio bytes no longer match this version")
    return record


def save_preference(project_id: str, payload: dict) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        record = _verified_record(project, payload["key"], payload["audio_sha256"])
        key = record["key"]
        pref = dict(project.voice_lab.preferences.get(key, {}))
        if "favorite" in payload:
            pref["favorite"] = payload["favorite"]
        if "rating" in payload:
            pref["rating"] = payload["rating"]
        if "suitability" in payload:
            role_id = payload.get("role_id")
            if role_id not in {r.id for r in project.roles}:
                raise VoiceLabError("Choose a real character for role suitability")
            pref["suitability"] = {**pref.get("suitability", {}), role_id: payload["suitability"]}
        pref.update(key=key, voice_id=record["voice_id"], version=record["version"],
                    audio_sha256=record["audio_sha256"], take_id=record.get("take_id"),
                    inspiration=record.get("inspiration"), tags=record.get("tags", []), updated_at=now())
        if not project.voice_lab.voices.get(key, {}).get("audio_sha256"):
            project.voice_lab.voices[key] = record
        project.voice_lab.preferences[key] = pref
        project.voice_lab.preference_history.append(dict(pref))
        from .performer_discovery import rank_candidates
        for shortlist_role, shortlist in project.voice_lab.shortlists.items():
            shortlist["matches"] = rank_candidates(shortlist.get("description", ""),
                traits=shortlist.get("traits", {}), preferences=list(project.voice_lab.preferences.values()), limit=5,
                role_id=None if shortlist_role == "general" else shortlist_role)
        _save(project)
    return {"saved": True, "preference": pref, "revision": project.voice_lab.revision,
            "shortlists": project.voice_lab.shortlists}


def save_position(project_id: str, payload: dict) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        seconds = payload.get("seconds", 0)
        if not math.isfinite(seconds) or seconds < 0:
            raise VoiceLabError("Invalid review position")
        if payload.get("key"):
            record = project.voice_lab.excerpt if payload["key"] == "excerpt" else comparison_records(project).get(payload["key"]) or _record(project, payload["key"])
            if not record or payload.get("audio_sha256") != record.get("audio_sha256") or seconds > record.get("duration_seconds", 0) + .1:
                raise VoiceLabError("Position does not belong to this audio version")
        elif seconds:
            raise VoiceLabError("Position requires an audio version")
        project.voice_lab.review_position = {**payload, "updated_at": now()}
        _save(project)
    return {"saved": True}


def add_issue(project_id: str, payload: dict) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        records = list(voice_records(project).values()) + list(comparison_records(project).values()) + ([project.voice_lab.excerpt] if project.voice_lab.excerpt else [])
        record = next((v for v in records if v.get("audio_sha256") == payload["audio_sha256"]), None)
        seconds = payload["seconds"]
        if not record or not math.isfinite(seconds) or not 0 <= seconds <= record.get("duration_seconds", 0):
            raise VoiceLabError("Issue timestamp is outside the exact audio version")
        if not payload["text"].strip():
            raise VoiceLabError("Enter feedback")
        path = _path(project_id, record["audio_file"])
        if file_sha256(path) != payload["audio_sha256"]:
            raise VoiceLabError("Audio bytes no longer match this version")
        issue = {**payload, "id": uuid.uuid4().hex, "created_at": now(), "voice_key": record.get("key"),
                 "take_id": record.get("take_id"), "human_review": "pending"}
        project.voice_lab.issues.append(issue)
        _save(project)
    return {"saved": True, "issue": issue}


def match(project_id: str, description: str = "", role_id: str | None = None,
          limit: int = 5, *, online: bool = True) -> dict:
    from .performer_discovery import discover
    with project_meta_lock(project_id):
        project = load_project(project_id)
        role = next((r for r in project.roles if r.id == role_id), None)
        if role_id and role is None:
            raise VoiceLabError("Unknown character")
        # Only abstract profile dimensions leave this boundary; discovery itself
        # sends performer names only, never source text or a character dossier.
        profile = project.character_cast.profiles.get(role_id)
        traits = {d.dimension: d.value for d in profile.decisions} if profile else {}
        slot = role_id or "general"
        token = uuid.uuid4().hex
        project.voice_lab.shortlists[slot] = {"status": "discovering", "token": token,
            "description": description, "traits": traits, "matches": []}
        preferences = list(project.voice_lab.preferences.values())
        _save(project)
    result = discover(safe_project_root(project_id) / "voice_lab" / "discovery", description,
                      traits=traits, preferences=preferences, limit=min(5, max(1, limit)), online=online, role_id=role_id)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        if project.voice_lab.shortlists.get(slot, {}).get("token") != token:
            return {"stale": True, "matches": [], "reason": "A newer match request is active"}
        # Feedback arriving during discovery wins; re-rank against fresh state.
        from .performer_discovery import rank_candidates
        ranked = rank_candidates(description, traits=traits,
                                 preferences=list(project.voice_lab.preferences.values()), limit=min(5, max(1, limit)), role_id=role_id)
        if isinstance(ranked, list):
            by_id = {(r.get("id") or r.get("candidate_id")): r for r in result.get("matches", [])}
            result["matches"] = [{**by_id.get(r.get("id") or r.get("candidate_id"), {}), **r} for r in ranked]
        stored = {**result, "status": "complete", "token": token, "description": description, "traits": traits}
        project.voice_lab.shortlists[slot] = stored
        _save(project)
    return stored


def more_like(project_id: str, key: str, request_id: str) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        if request_id in project.voice_lab.variants:
            prior = project.voice_lab.variants[request_id]
            if prior["parent_key"] != key:
                raise VoiceLabError("Request ID already belongs to a different voice")
            return prior
        if len(project.voice_lab.variants) >= 20:
            raise VoiceLabError("Related candidate budget reached")
        record = _record(project, key)
        directions = ["Keep the identity's warmth; explore slightly more direct articulation.",
                      "Keep the identity's clarity; explore more relaxed conversational phrasing."]
        result = {"request_id": request_id, "parent_key": key, "status": "directions_ready_audio_idle",
                  "reason": "Speculative audio replenishment is idle while the first listening test is pending.",
                  "candidates": [{"id": "variant_" + digest([key, request_id, i])[:20],
                      "design_instruction": record.get("identity", {}).get("design_instruction", "Original clear natural voice.") + " " + direction,
                      "inspiration": record.get("inspiration"), "rating": None, "favorite": False}
                      for i, direction in enumerate(directions)]}
        project.voice_lab.variants[request_id] = result
        project.voice_lab.voices.setdefault(key, record)
        _save(project)
    return result


def apply_assignment(project, key: str, role_id: str, expected_identity_id: str | None):
    """Apply validated reusable conditioning to a locked project; caller commits.

    Shared by explicit legacy assignment and atomic human casting approval.
    """
    project_id = project.id
    record = _record(project, key)
    role = next((r for r in project.roles if r.id == role_id), None)
    if role is None:
        raise VoiceLabError("Unknown character")
    current = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
    if current != expected_identity_id:
        raise VoiceLabError("Casting changed since this page loaded; refresh before assigning")
    builtin_identity = (record.get("identity") or {}).get("mechanism") == "builtin"
    if builtin_identity or (not record.get("identity") and record.get("builtin_speaker")):
        from dubbing_service.app.models.project import Role
        snapshot = record.get("builtin_snapshot") or {}
        fields = ("speaker_audio", "speaker_voice_id", "tts_backend", "qwen3_speaker",
                  "qwen3_language", "voice_description", "style_instruction", "default_emotion_mode",
                  "default_emotion_text", "default_emotion_vector", "emotion_alpha")
        settings = snapshot.get("settings", {})
        if (snapshot.get("schema") != "voice-lab-builtin-role-v1"
                or snapshot.get("source_voice_revision") != record["version"]
                or not snapshot.get("source_role_id") or not all(field in settings for field in fields)):
            raise VoiceLabError("Historical built-in settings are incomplete; exact voice reuse is unavailable")
        identity = CharacterVoiceIdentity.model_validate(record["identity"]) if builtin_identity else None
        recovered = Role(id=snapshot["source_role_id"], name="Historical voice", voice_identity=identity,
                         **{field: settings[field] for field in fields})
        if (voice_revision_for_role(recovered) != record["version"]
                or recovered.tts_backend != "qwen3tts_cv" or not recovered.qwen3_language
                or recovered.qwen3_speaker != record.get("builtin_speaker")
                or (identity and (identity.status != "LOCKED" or identity.backend != "qwen3tts_cv"
                                 or identity.builtin_speaker != recovered.qwen3_speaker))):
            raise VoiceLabError("Saved built-in settings do not match the heard voice revision")
        if role.voice_identity:
            project.character_cast.identity_history.append(role.voice_identity.model_copy(deep=True))
        for field in fields:
            setattr(role, field, getattr(recovered, field))
        role.voice_identity = identity.model_copy(update={"role_id": role_id}) if identity else None
    elif record.get("identity"):
        identity = CharacterVoiceIdentity.model_validate(record["identity"])
        artifacts = [("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")] if identity.mechanism == "qwen_reference" else [("reference_audio", "reference_audio_sha256")] if identity.mechanism == "existing_reference" else []
        if identity.mechanism == "fixture":
            raise VoiceLabError("Signal fixture identities cannot be assigned as voices")
        for name, hash_name in artifacts:
            relative, expected = getattr(identity, name), getattr(identity, hash_name)
            if not relative or not expected or file_sha256(_path(project_id, relative)) != expected:
                raise VoiceLabError("Reusable voice conditioning is missing or changed")
        if identity.mechanism == "qwen_reference":
            # Review/library records keep their immutable source paths. The
            # assigned role needs the same verified bytes in the provider's
            # identity-scoped namespace, whether approval came from the inbox
            # or the legacy assignment endpoint.
            from .voice_lab_library import _publish_blob, _read_verified, _safe_path
            root = safe_project_root(project_id)
            promoted = identity_for_assignment(identity)
            try:
                for field, hash_field in (("reference_audio", "reference_audio_sha256"),
                                          ("prompt_artifact", "prompt_sha256")):
                    expected = getattr(identity, hash_field)
                    content = _read_verified(_safe_path(root, getattr(identity, field)), expected)
                    relative = getattr(promoted, field)
                    _publish_blob(root, relative, content, expected)
                identity = promoted
            except Exception as exc:
                raise VoiceLabError(f"Reusable voice conditioning could not be safely published: {exc}") from exc
        if role.voice_identity:
            project.character_cast.identity_history.append(role.voice_identity.model_copy(deep=True))
        role.voice_identity = identity.model_copy(update={"role_id": role_id})
        role.tts_backend = identity.backend
    else:
        raise VoiceLabError("Historical audio has no reusable conditioning; it cannot be assigned")
    project.voice_lab.voices.setdefault(key, record)
    return role, record


def assign(project_id: str, key: str, role_id: str, expected_identity_id: str | None) -> dict:
    from dubbing_service.app.core.tts_operations import try_acquire_project_mutation, release_project_mutation
    lease = try_acquire_project_mutation(project_id, kind="voice_lab_assign")
    if lease is None:
        raise VoiceLabError("This project is rendering; assignment must wait")
    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            if project.voice_lab.casting.get("enabled") or project.metadata.get("autocast_import", {}).get("mode") == "autocast":
                raise VoiceLabError("Use the casting inbox to explicitly select the exact audition or give a 4- or 5-star character-fit rating")
            role, record = apply_assignment(project, key, role_id, expected_identity_id)
            _save(project)
        assigned_revision = voice_revision_for_role(role)
        return {"saved": True, "role_id": role_id, "key": key, "existing_takes_unchanged": True,
                "source_voice_revision": record["version"], "assigned_voice_revision": assigned_revision,
                "same_voice_revision": assigned_revision == record["version"]}
    finally:
        release_project_mutation(lease)


def production_busy() -> bool:
    from dubbing_service.app.core.tts_operations import has_active_tts_operation
    from dubbing_service.app.storage.jobs import load_jobs
    if has_active_tts_operation():
        return True
    for path in get_config().projects_root_abs.glob("*/project.json"):
        for job in load_jobs(path.parent.name):
            if job.job_type in {"batch_synthesize", "single_synthesize"} and job.status.value in {"queued", "running"}:
                return True
    return False


def set_automation(project_id: str, paused: bool) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        auto = project.voice_lab.automation
        auto.update(paused=paused, status="paused" if paused else "ready",
                    audio_replenishment="idle_pending_listening_test")
        _save(project)
        return dict(auto)


def resume_discoveries() -> None:
    """Recover at most four opted-in discovery cycles on Studio owner startup."""
    import logging
    count = 0
    for path in sorted(get_config().projects_root_abs.glob("*/project.json")):
        project = load_project(path.parent.name)
        auto = project.voice_lab.automation
        if (not auto.get("paused", True) and auto.get("cursor", 0) < auto.get("max_cycles", 4)
                and auto.get("status") in {"ready", "discovering", "waiting_for_production", "interrupted_resume_available"}):
            try:
                discovery_tick(project.id)
            except Exception:
                logging.getLogger(__name__).exception("Voice Lab discovery remains resumable: %s", project.id)
            count += 1
            if count >= 4:
                break


def discovery_tick(project_id: str, *, online: bool = True) -> dict:
    """One bounded cycle in the existing durable job owner; no extra daemon/GPU."""
    from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
    from dubbing_service.app.storage.jobs import upsert_job, mutate_job, get_job, start_job, mark_job_active, try_mark_job_active
    with project_meta_lock(project_id):
        project = load_project(project_id)
        auto = project.voice_lab.automation
        if auto["paused"] or auto["cursor"] >= auto["max_cycles"]:
            return dict(auto)
        if production_busy():
            auto["status"] = "waiting_for_production"
            _save(project)
            return dict(auto)
        cursor = auto["cursor"]
        job_id = f"voice_lab_discovery_{cursor}"
        auto.update(status="discovering", job_id=job_id)
        _save(project)
    job = get_job(project_id, job_id)
    if not try_mark_job_active(project_id, job_id):
        return {"status": "already_running", "job_id": job_id}
    try:
        if not job:
            upsert_job(project_id, SynthesisJob(id=job_id, project_id=project_id,
                job_type="voice_lab_discovery", request_id=job_id, idempotency_key=job_id))
        if not job or job.status != SynthesisJobStatus.SUCCEEDED:
            start_job(project_id, job_id)
            match(project_id, "natural expressive audiobook narration", online=online)
            mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.SUCCEEDED), setattr(j, "finished_at", datetime.now(timezone.utc))))
        with project_meta_lock(project_id):
            project = load_project(project_id)
            auto = project.voice_lab.automation
            auto["cursor"] = max(auto["cursor"], cursor + 1)
            auto["status"] = "paused" if auto["paused"] else "bounded_cycle_complete"
            _save(project)
            return dict(auto)
    except Exception as exc:
        failure_message = str(exc)[:200]
        mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED), setattr(j, "message", failure_message)))
        with project_meta_lock(project_id):
            project = load_project(project_id)
            project.voice_lab.automation["status"] = "interrupted_resume_available"
            _save(project)
        raise
    finally:
        mark_job_active(project_id, job_id, False)


def design(project_id: str, role_id: str, candidate_id: str,
           expected_identity_id: str | None, request_id: str, *, provider=None) -> dict:
    """One original reference and reusable Base identity, with no demo batch.

    A ready record is not yet audible/favoritable. Only subsequent immutable
    production Takes add review audio. Generation never silently assigns a role.
    """
    from dubbing_service.app.core.tts_operations import try_acquire_tts_operation, release_tts_operation
    from dubbing_service.app.models.character import InspirationMatch
    from dubbing_service.app.providers.tts import Qwen3TTSProvider
    with project_meta_lock(project_id):
        project = load_project(project_id)
        if project.voice_lab.scope == "isolated_listening_test":
            excerpt = project.voice_lab.excerpt or {}
            if not excerpt.get("manifest_id") or excerpt.get("status") in {"blocked", "blocked_source", "missing_chapter_1"}:
                raise VoiceLabError(excerpt.get("reason") or "Persist the verified Chapter 1 excerpt manifest before GPU work")
            from .voice_lab_render import load_excerpt_manifest
            load_excerpt_manifest(project_id, excerpt["manifest_id"])
        prior = project.voice_lab.designs.get(request_id)
        intent = digest([role_id, candidate_id, expected_identity_id])
        if prior:
            if prior["intent"] != intent:
                raise VoiceLabError("Design request ID already has a different intent")
            if prior["status"] == "ready":
                return prior
        elif len(project.voice_lab.designs) >= 12:
            raise VoiceLabError("Explicit design budget reached")
        role = next((r for r in project.roles if r.id == role_id), None)
        if role is None:
            raise VoiceLabError("Unknown character")
        current = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
        if current != expected_identity_id:
            raise VoiceLabError("Casting changed; refresh before designing")
        candidates = [m for s in project.voice_lab.shortlists.values() for m in s.get("matches", [])]
        candidate = next((m for m in candidates if m.get("candidate_id", m.get("id")) == candidate_id), None)
        if candidate is None:
            raise VoiceLabError("Auto Match this character before designing")
        profile = project.character_cast.profiles.get(role_id)
        from .original_inspired_design import authorized_candidate
        original_design = authorized_candidate(candidate)
        instruction = "Create an original synthetic voice. " + (profile.design_instruction + " " if profile and not original_design else "") + candidate.get("design_instruction", candidate.get("creative_direction", "Clear conversational storytelling."))
        anchor = candidate.get("anchor") or {}
        reference = candidate.get("inspiration") or {}
        if not reference and anchor.get("performer"):
            reference = {"performer": anchor["performer"], "performance": anchor["performance"],
                         "source_urls": anchor.get("source_urls") or []}
        conditioning = candidate_conditioning(candidate)
        if conditioning is None and not original_design:
            # Production never falls back to an unanchored generic voice: a
            # candidate without a reference performance has no acoustic target,
            # so its design is refused and the reason is stated.
            raise VoiceLabError(
                (anchor.get("unanchored_reason")
                 or "This candidate has no performer reference, so it has no acoustic conditioning; "
                    "re-cast the character before designing."))
        conditioning = dict(conditioning) if conditioning else None
        if conditioning and conditioning.get("instruction"):
            # The acoustic targets join the design instruction as numbers, so the
            # model is asked for the measured profile and not only the reference.
            instruction = instruction + " " + str(conditioning["instruction"])
        # A generated candidate is original by construction, but it is still cast
        # against a specific reference performance. Recording that reference is
        # what makes the voice reviewable as a deliberate choice rather than a
        # generic synthetic narrator.
        inspiration = (InspirationMatch(palette="original-inspired-design-v1" if original_design else "catalogue-performer-anchors-v1",
            character=reference["performance"], performer=reference["performer"],
            mode="archetype",
            reasons=[candidate.get("match_rationale") or anchor.get("rationale")
                     or "Casting intent from a catalogue performance reference"],
            score=anchor.get("score") or 0,
            source=("Owner-authored descriptive inspiration; no performer recording" if original_design else
                    "; ".join(reference.get("source_urls") or []) or "catalogue metadata"))
            if reference.get("performer") and reference.get("performance") else None)
        identity_id = "voice_" + digest([project_id, request_id, intent])[:24]
        pending = {"status": "pending", "intent": intent, "identity_id": identity_id,
                   "role_id": role_id, "candidate_id": candidate_id, "request_id": request_id}
        project.voice_lab.designs[request_id] = pending
        _save(project)
    if production_busy():
        raise VoiceLabError("Existing production work has GPU priority; resume this design after it completes")
    lease = try_acquire_tts_operation(kind="voice_lab_design", project_id=project_id)
    if lease is None:
        raise VoiceLabError("GPU lane is busy; resume the same design request later")
    try:
        if project.voice_lab.scope == "isolated_listening_test":
            load_excerpt_manifest(project_id, excerpt["manifest_id"])
        if provider is None:
            from .project_qwen_runtime import project_profile
            provider = Qwen3TTSProvider()
            provider._project_runtime_profile = project_profile(project_id)
            provider.runtime_identity()  # Reject wrong runtime before conditioning.
        transcript = "There is still time to find the right answer. Let us begin again, and listen carefully."
        metadata = provider.design_character_voice(project_id, identity_id, instruction, transcript,
                                                   seed=int(digest(identity_id)[:7], 16),
                                                   conditioning=conditioning)
        allowed = set(CharacterVoiceIdentity.model_fields) - {"id", "role_id", "dossier_revision", "casting_profile_revision", "inspiration", "design_instruction", "status"}
        details = {k: v for k, v in metadata.items() if k in allowed}
        details.setdefault("mechanism", "qwen_reference")
        details.setdefault("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        identity = CharacterVoiceIdentity(id=identity_id, role_id=role_id,
            dossier_revision=profile.dossier_revision if profile else "unknown",
            casting_profile_revision=profile.revision if profile else digest(instruction)[:24],
            design_instruction=instruction,
            inspiration=inspiration, **details)
        for attr, hash_attr in [("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")]:
            relative, expected = getattr(identity, attr), getattr(identity, hash_attr)
            if not relative or not expected or file_sha256(_path(project_id, relative)) != expected:
                raise VoiceLabError("Worker returned incomplete or modified conditioning artifacts")
        probe_role = role.model_copy(update={"voice_identity": identity})
        version = voice_revision_for_role(probe_role)
        key = "v_" + digest([identity_id, version])[:28]
        record = {"key": key, "voice_id": identity_id, "version": version,
            "name": f"Rowan {version[:6]}", "kind": "original_inspired", "synthetic": True,
            "role_id": role_id, "role_name": role.name, "tags": candidate.get("tags", [])[:3],
            # The reference travels with the heard voice, so the library, the
            # review surface and a later cross-project proposal can all say why
            # this voice was designed the way it was.
            "inspiration": ({"performer": reference["performer"], "performance": reference["performance"],
                             "source_urls": list(reference.get("source_urls") or []),
                             "rationale": candidate.get("match_rationale") or anchor.get("rationale")}
                            if reference.get("performer") and reference.get("performance") else None),
            "anchor": copy.deepcopy(anchor) if anchor else None,
            # What the acoustic layer did to this voice, and what it achieved, so
            # the review surface reports the conditioning rather than a claim.
            "acoustics": copy.deepcopy(identity.acoustics),
            "conditioning": copy.deepcopy(conditioning),
            "reference": copy.deepcopy(candidate.get("reference")),
            "explanation": ("Original synthetic design, acoustically conditioned toward its performer "
                            "reference. Awaiting production-path audio before review."
                            if identity.acoustics else
                            "Original synthetic design. Awaiting production-path audio before review."),
            "identity": identity.model_dump(mode="json"), "backend": "qwen3tts", "details": identity.model_dump(mode="json"),
            "audio_file": None, "audio_sha256": None, "duration_seconds": 0,
            "technical_qa": "conditioning_only", "human_review": "pending", "lines": []}
        with project_meta_lock(project_id):
            latest = load_project(project_id)
            latest_role = next(r for r in latest.roles if r.id == role_id)
            latest_id = latest_role.voice_identity.id if latest_role.voice_identity else voice_revision_for_role(latest_role)
            result = {**pending, "status": "ready", "key": key, "cast_changed_during_generation": latest_id != expected_identity_id,
                      "reference_generation_count": 1, "candidate_audio_generation_count": 0, "production_audio_generation_count": 0}
            latest.voice_lab.voices.setdefault(key, record)
            latest.voice_lab.designs[request_id] = result
            _save(latest)
        return result
    finally:
        release_tts_operation(lease)


def source_preflight(project_id: str) -> dict:
    """Inspect fixed local audiobook roots, never crawl the book's website."""
    from scripts.voice_lab_source_preflight import inspect_roots
    repo = Path(__file__).resolve().parents[3]
    project = load_project(project_id)
    if project.roles or load_segments(project_id):
        if project.voice_lab.scope != "isolated_listening_test":
            raise VoiceLabError("Create an empty isolated listening-test project first")
    report = inspect_roots([repo.parent / "source-books", get_config().projects_root_abs])
    with project_meta_lock(project_id):
        project = load_project(project_id)
        project.voice_lab.scope = "isolated_listening_test"
        project.voice_lab.excerpt = report.get("excerpt") or {
            "status": "source_requires_inspection", "reason": "Chapter 1 candidates found. Inspect the full chapter and persist a contiguous excerpt manifest before synthesis.",
            "candidate_paths": report.get("chapter_one_candidates", []), "human_review": "pending"}
        project.voice_lab.automation.update(paused=True, status="paused", audio_replenishment="idle_pending_listening_test")
        _save(project)
    folder = safe_project_root(project_id) / "voice_lab"
    folder.mkdir(exist_ok=True)
    path = folder / ("source-preflight-" + digest(report)[:16] + ".json")
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    return {"excerpt": project.voice_lab.excerpt, "report_file": str(path), "available_chapters": report.get("available_chapters", [])}


def prepare_listening_cast(project_id: str, expected_manifest_id: str) -> dict:
    """Auto-select the best original design only for the isolated approved scope."""
    from .voice_lab_render import load_excerpt_manifest
    manifest, _ = load_excerpt_manifest(project_id, expected_manifest_id)
    project = load_project(project_id)
    if project.voice_lab.scope != "isolated_listening_test":
        raise VoiceLabError("Automatic assignment is restricted to the isolated listening test")
    roles = list(dict.fromkeys([manifest["narrator_role_id"], *manifest["character_ids"]]))
    for role in project.roles:
        if role.id not in roles or not role.voice_identity:
            continue
        identity = role.voice_identity
        if identity.status != "LOCKED" or identity.mechanism != "qwen_reference":
            raise VoiceLabError("Existing cast needs a locked prepared Qwen Base identity before any design work")
        for attr, hash_attr in [("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")]:
            relative, expected = getattr(identity, attr), getattr(identity, hash_attr)
            if not relative or not expected or file_sha256(_path(project_id, relative)) != expected:
                raise VoiceLabError("Existing cast conditioning is missing or modified")
    results = []
    for role_id in roles:
        # The source gate is repeated between each bounded expensive operation.
        load_excerpt_manifest(project_id, expected_manifest_id)
        project = load_project(project_id)
        role = next(r for r in project.roles if r.id == role_id)
        if role.voice_identity:
            if role.voice_identity.status != "LOCKED" or role.voice_identity.mechanism != "qwen_reference":
                raise VoiceLabError("This role needs a prepared, locked Qwen Base identity; no voice fallback is permitted")
            results.append({"role_id": role_id, "status": "reused", "identity_id": role.voice_identity.id})
            continue
        direction = "clear warm natural audiobook narrator, measured and expressive" if role_id == manifest["narrator_role_id"] else ""
        candidates = match(project_id, direction, role_id=role_id, limit=3)
        if candidates.get("stale") or not candidates.get("matches"):
            raise VoiceLabError("Auto Match changed while preparing this cast")
        best = candidates["matches"][0]
        expected = voice_revision_for_role(role)
        request_id = "excerpt_" + digest([expected_manifest_id, role_id, best["candidate_id"]])[:24]
        result = design(project_id, role_id, best["candidate_id"], expected, request_id)
        load_excerpt_manifest(project_id, expected_manifest_id)
        assign(project_id, result["key"], role_id, expected)
        results.append(result)
    return {"roles": results, "manifest_id": expected_manifest_id, "audio_auditions_generated": 0}


def import_reference(project_id: str, role_id: str, acquisition_id: str, acquisition_sha256: str,
                     expected_identity_id: str | None, request_id: str, *, provider=None) -> dict:
    """Eligible acquired references enter the same versioned review/casting flow."""
    from dubbing_service.app.core.tts_operations import try_acquire_tts_operation, release_tts_operation
    from dubbing_service.app.providers.tts import Qwen3TTSProvider
    project = load_project(project_id)
    if project.voice_lab.scope == "isolated_listening_test":
        from .voice_lab_render import load_excerpt_manifest
        load_excerpt_manifest(project_id, (project.voice_lab.excerpt or {}).get("manifest_id"))
    role = next((r for r in project.roles if r.id == role_id), None)
    if not role:
        raise VoiceLabError("Unknown character")
    current = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
    if current != expected_identity_id:
        raise VoiceLabError("Casting changed; refresh before importing")
    intent = digest([role_id, acquisition_id, acquisition_sha256])
    with project_meta_lock(project_id):
        project = load_project(project_id)
        prior = project.voice_lab.designs.get(request_id)
        if prior:
            if prior["intent"] != intent:
                raise VoiceLabError("Reference request ID has a different intent")
            if prior["status"] == "ready":
                return prior
        elif len(project.voice_lab.designs) >= 12:
            raise VoiceLabError("Reference preparation budget reached")
    if production_busy():
        raise VoiceLabError("Production work has GPU priority")
    lease = try_acquire_tts_operation(kind="voice_lab_import_reference", project_id=project_id)
    if not lease:
        raise VoiceLabError("GPU lane busy; resume the same reference request")
    try:
        if project.voice_lab.scope == "isolated_listening_test":
            load_excerpt_manifest(project_id, (project.voice_lab.excerpt or {}).get("manifest_id"))
        identity_id = "voice_" + digest([project_id, request_id, intent])[:24]
        if provider is None:
            from .project_qwen_runtime import project_profile
            provider = Qwen3TTSProvider()
            provider._project_runtime_profile = project_profile(project_id)
            provider.runtime_identity()  # Reject wrong runtime before conditioning.
        metadata = provider.import_character_reference(project_id, identity_id, acquisition_id, acquisition_sha256)
        fields = set(CharacterVoiceIdentity.model_fields) - {"id", "role_id", "dossier_revision", "casting_profile_revision", "design_instruction", "inspiration", "status"}
        identity = CharacterVoiceIdentity(id=identity_id, role_id=role_id, dossier_revision="eligible-reference",
            casting_profile_revision=acquisition_sha256[:24], design_instruction="Eligible reference clone; synthetic output.",
            **{k: v for k, v in metadata.items() if k in fields})
        version = voice_revision_for_role(role.model_copy(update={"voice_identity": identity}))
        key = "v_" + digest([identity_id, version])[:28]
        record = {"key": key, "voice_id": identity_id, "version": version,
            "name": f"Reference {version[:6]}", "kind": "reference_clone", "synthetic": True,
            "role_id": role_id, "role_name": role.name, "tags": [], "inspiration": None,
            "explanation": ("Actual recorded reference; unapproved audition draft. Source delivery and character fit remain unknown; human selection required." if identity.provenance.get("audition_draft_policy") else "Eligible reference clone. Awaiting production-path audio before review."),
            "identity": identity.model_dump(mode="json"), "details": identity.model_dump(mode="json"),
            "audio_file": None, "audio_sha256": None, "duration_seconds": 0,
            "technical_qa": "conditioning_only", "human_review": "pending", "lines": []}
        result = {"intent": intent, "status": "ready", "key": key, "identity_id": identity_id,
                  "request_id": request_id, "reference_generation_count": 0, "candidate_audio_generation_count": 0}
        with project_meta_lock(project_id):
            latest = load_project(project_id)
            latest.voice_lab.voices.setdefault(key, record)
            from service.qwen_character_voice import _eligible_acquisition
            acquired, _, _ = _eligible_acquisition(safe_project_root(project_id), acquisition_id, acquisition_sha256, allow_draft=True)
            performer = acquired["source"].get("performer")
            if performer:
                source_page = acquired["source"]["source_page"]
                row = {"key": key, "performer": performer,
                    "performance": acquired["source"].get("title") or "Verified recorded speaking voice",
                    "acquisition_id": acquisition_id, "acquisition_sha256": acquisition_sha256}
                rows = latest.metadata.setdefault("voice_lab_named_references", {}).setdefault(role_id, [])
                if not any(r.get("acquisition_id") == acquisition_id for r in rows):
                    rows.append(row)
                latest.voice_lab.voices[key]["inspiration"] = {"performer": performer,
                    "performance": row["performance"], "source_urls": [source_page]}
                latest.voice_lab.voices[key]["name"] = f"{performer} reference clone - review pending"
            latest.voice_lab.designs[request_id] = result
            _save(latest)
        return result
    finally:
        release_tts_operation(lease)
