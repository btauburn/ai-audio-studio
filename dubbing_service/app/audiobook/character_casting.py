"""Automatic research, ensemble casting and resumable original voice auditions.

Studio's Project/Role remain the only authority. Audio is prepared outside the
metadata lock and merged optimistically; a locked identity is never replaced by
new research or a later inspiration request.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.character import (
    AuditionCandidate,
    CharacterVoiceIdentity,
    ScenePerformanceDirection,
)
from dubbing_service.app.models.project import Project, Role, Segment
from dubbing_service.app.storage.projects import (
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.segments import load_segments

REFERENCE_TEXT = "There is still time to find the right answer. Let us begin again, and listen carefully."
AUDITION_REVISION = "audition-v2-sampled-base-scene-controls"
AUDITION_TEXTS = {
    "neutral": "The door is open. We can leave when you are ready.",
    "elevated": "Wait! We must get everyone out of here, right now!",
    "quiet": "I remember that day. I wish we had more time.",
}


def _key(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]


def research_and_profile(
    project: Project, segments: Sequence[Segment], *, external_provider=None
) -> None:
    """Cheap ingest step: cached research and profiles; never loads a GPU model."""
    if not project.character_cast.policy.enabled:
        return
    from .casting_profiles import build_profiles
    from .character_identity import canonical_role_id
    from .character_research import _active_aliases, _refresh, _retire_provider_facts, research_project

    state = project.character_cast
    if external_provider is None and (
        state.policy.external_research
        or any(fact.extractor.startswith("mediawiki-book-validated-v1")
               for dossier in state.dossiers.values() for fact in dossier.facts)
    ):
        from .external_character_research import provider_for_project
        # Ingest remains local/cheap: reuse valid cache, defer network to refresh.
        # Owned prior-edition evidence still expires locally if queries are off.
        external_provider = provider_for_project(project, segments, online=False)
    if external_provider is not None:
        for dossier in state.dossiers.values():
            _retire_provider_facts(project, dossier, external_provider)
            _refresh(dossier)
    roles_by_id = {role.id: role for role in project.roles}
    # Classify provenance BEFORE new claims are added. A newly found alias
    # cannot convert an established identity merge into an inferred link.
    evidence_aliases = {
        (canonical_role_id(project, dossier.role_id), str(fact.value).casefold().strip())
        for dossier in state.dossiers.values()
        for fact in dossier.facts
        if fact.field == "aliases" and fact.value
    }
    research_project(project, segments, external_provider=external_provider)
    # Explicit aliases bind existing role IDs without changing source text,
    # utterance IDs, or human attribution. Never merge ambiguous identities.
    # Rebuild evidence-derived aliases under the current chapter/source policy.
    # Stored identity reconciliations without alias facts have a separate
    # authority and must not disappear merely because research is refreshed.
    state.role_aliases = {
        alias: canonical
        for alias, canonical in state.role_aliases.items()
        if alias in roles_by_id and canonical in roles_by_id
        and (canonical_role_id(project, canonical), roles_by_id[alias].name.casefold().strip())
        not in evidence_aliases
    }
    for role in project.roles:
        owners = {
            d.role_id
            for d in state.dossiers.values()
            if d.role_id in roles_by_id and d.role_id != role.id
            and role.name.casefold().strip()
            in {a.casefold().strip() for a in _active_aliases(d, state.policy)}
        }
        if len(owners) == 1:
            owner = next(iter(owners))
            established = state.role_aliases.get(role.id)
            if established is not None and established != owner:
                reason = (
                    f"Research alias for {role.name} conflicts with an established identity merge; "
                    "explicit identity review is required"
                )
                if reason not in state.review_reasons:
                    state.review_reasons.append(reason)
            else:
                state.role_aliases[role.id] = owner
    for alias in list(state.role_aliases):
        seen, canonical = {alias}, state.role_aliases[alias]
        while canonical in state.role_aliases and canonical not in seen:
            seen.add(canonical)
            canonical = state.role_aliases[canonical]
        if canonical in seen:
            state.role_aliases.pop(alias, None)
        else:
            state.role_aliases[alias] = canonical
    profiles = build_profiles(project, segments)
    for role_id, profile in profiles.items():
        previous = state.profiles.get(role_id)
        if (
            previous
            and previous.revision != profile.revision
            and not any(p.revision == previous.revision for p in state.profile_history)
        ):
            state.profile_history.append(previous)
        state.profiles[role_id] = profile
        role = next((r for r in project.roles if r.id == role_id), None)
        if role and role.voice_identity:
            # A new source revision alone is not a reason to recast.
            if role.voice_identity.design_instruction != profile.design_instruction:
                state.proposed_revisions[role_id] = profile.revision
            else:
                state.proposed_revisions.pop(role_id, None)
    for role_id in set(state.profiles) - set(profiles):
        old = state.profiles.pop(role_id)
        if not any(p.revision == old.revision for p in state.profile_history):
            state.profile_history.append(old)


def _research_stamp(project: Project, segments: Sequence[Segment]) -> str:
    return _key([
        [role.model_dump(mode="json") for role in project.roles],
        [chapter.model_dump(mode="json") for chapter in project.chapters],
        project.metadata,
        [revision.model_dump(mode="json") for revision in project.source_revisions],
        project.character_cast.model_dump(mode="json"),
        [segment.model_dump(mode="json") for segment in segments],
    ])


def refresh_cast(project_id: str, *, external_provider=None) -> Project:
    # Public HTTP research must not hold the metadata/rating lock. Commit only
    # derived research after validating the captured book/casting inputs.
    with project_meta_lock(project_id):
        project = load_project(project_id)
        segments = load_segments(project_id)
        stamp = _research_stamp(project, segments)
        researched = project.model_copy(deep=True)
    if researched.character_cast.policy.external_research and external_provider is None:
        from .external_character_research import provider_for_project
        external_provider = provider_for_project(researched, segments, online=True)
    research_and_profile(researched, segments, external_provider=external_provider)
    with project_meta_lock(project_id):
        latest = load_project(project_id)
        if _research_stamp(latest, load_segments(project_id)) != stamp:
            raise ValueError("Research inputs changed during refresh; retry against the current book")
        # Preserve live Voice Lab ratings/candidates and all unrelated metadata.
        latest.character_cast = researched.character_cast
        save_project(latest)
    return latest


def _bind_aliases(project: Project) -> None:
    by_id = {r.id: r for r in project.roles}
    for alias, canonical in project.character_cast.role_aliases.items():
        target, owner = by_id.get(alias), by_id.get(canonical)
        if not target or not owner or not owner.voice_identity:
            continue
        if (
            target.voice_identity
            and target.voice_identity.id != owner.voice_identity.id
        ):
            reason = f"Alias {target.name} has a different locked voice; explicit identity revision requires review"
            if reason not in project.character_cast.review_reasons:
                project.character_cast.review_reasons.append(reason)
            continue
        target.voice_identity = owner.voice_identity.model_copy(deep=True)
        target.tts_backend = owner.tts_backend
        target.speaker_audio = owner.speaker_audio
        target.voice_description = owner.voice_description


def performance_for(
    segment: Segment, role: Role | None = None
) -> ScenePerformanceDirection:
    return ScenePerformanceDirection(
        emotion=segment.emotion_text or (role.default_emotion_text if role else None),
        intensity=segment.emotion_alpha,
        volume=segment.volume_gain,
        pace=segment.playback_speed,
        delivery=role.style_instruction if role else None,
    )


def _artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("casting artifact escapes project root")
    return path


def _sha(path: Path) -> str:
    from .fingerprint import file_sha256

    return file_sha256(path)


def _identity(
    project: Project, role: Role, profile, candidate_id: str, metadata: dict
) -> CharacterVoiceIdentity:
    allowed = set(CharacterVoiceIdentity.model_fields) - {
        "id",
        "role_id",
        "dossier_revision",
        "casting_profile_revision",
        "inspiration",
        "design_instruction",
        "status",
    }
    details = {key: value for key, value in metadata.items() if key in allowed}
    details.setdefault("mechanism", "qwen_reference")
    details.setdefault("model", "Qwen3-TTS-12Hz-1.7B-Base")
    return CharacterVoiceIdentity(
        id=candidate_id,
        role_id=role.id,
        dossier_revision=profile.dossier_revision,
        casting_profile_revision=profile.revision,
        inspiration=profile.inspiration,
        design_instruction=profile.design_instruction,
        **details,
    )


def _save_candidate(project_id: str, candidate: AuditionCandidate) -> None:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        rows = project.character_cast.auditions.setdefault(candidate.role_id, [])
        old = next((i for i, row in enumerate(rows) if row.id == candidate.id), None)
        if old is None:
            rows.append(candidate)
        else:
            rows[old] = candidate
        save_project(project)


def _existing_identity(
    project_id: str, role: Role, profile
) -> CharacterVoiceIdentity | None:
    """Preserve explicit pre-existing user voice assignments."""
    if role.qwen3_speaker and role.tts_backend == "qwen3tts_cv":
        return CharacterVoiceIdentity(
            id="voice_" + _key([role.id, role.qwen3_speaker]),
            role_id=role.id,
            mechanism="builtin",
            builtin_speaker=role.qwen3_speaker,
            backend="qwen3tts_cv",
            model="Qwen3-TTS-CustomVoice",
            dossier_revision=profile.dossier_revision,
            casting_profile_revision=profile.revision,
            design_instruction=role.voice_description or "",
            provenance={
                "source": "preserved explicit project voice assignment",
                "audition": "not re-auditioned",
            },
        )
    if (role.speaker_audio or role.speaker_voice_id) and role.tts_backend in (
        "indextts",
        "cosyvoice3",
    ):
        import os
        import uuid

        from dubbing_service.app.core.helpers import resolve_role_reference_audio

        source = resolve_role_reference_audio(
            project_id, role.speaker_audio, role.speaker_voice_id
        )
        provenance = {
            "source": "preserved user reference; no new recording acquired",
            "original_reference": role.speaker_audio,
            "original_voice_id": role.speaker_voice_id,
        }
        identity_id = "voice_" + _key(
            [role.id, role.tts_backend, role.speaker_audio, role.speaker_voice_id]
        )
        relative, checksum, status = None, None, "FAILED"
        try:
            if source is None:
                raise ValueError(
                    "configured user reference is missing; explicit repair is required"
                )
            checksum = _sha(source)
            identity_id = "voice_" + _key([role.id, role.tts_backend, checksum])
            root = safe_project_root(project_id).resolve()
            relative = f"casting/voices/{identity_id}/reference{source.suffix.lower() or '.wav'}"
            target = root / relative
            target.resolve().relative_to(root)
            cursor = target
            while cursor != root:
                if cursor.is_symlink() or (
                    cursor.exists()
                    and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x400
                ):
                    raise ValueError("linked reference snapshot is forbidden")
                cursor = cursor.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                staged = target.with_name(target.name + ".stage-" + uuid.uuid4().hex)
                shutil.copy2(source, staged)
                if _sha(staged) != checksum:
                    raise ValueError(
                        "user reference changed while creating its identity snapshot"
                    )
                if not target.exists():
                    os.replace(staged, target)
            if _sha(target) != checksum:
                raise ValueError("immutable reference snapshot checksum mismatch")
            status = "LOCKED"
        except (OSError, ValueError) as exc:
            provenance["error"] = str(exc)
        return CharacterVoiceIdentity(
            id=identity_id,
            role_id=role.id,
            status=status,
            mechanism="existing_reference",
            backend=role.tts_backend,
            model=role.tts_backend,
            reference_audio=relative,
            reference_audio_sha256=checksum,
            dossier_revision=profile.dossier_revision,
            casting_profile_revision=profile.revision,
            design_instruction=role.voice_description or "",
            provenance=provenance,
        )
    return None


def _candidate_design(
    project: Project, role: Role, profile, index: int
) -> tuple[str, dict | None]:
    """Vary an original candidate, preserving research and accepted identities.

    Only a later candidate may respond to a recorded collision. Pitch readings
    support ensemble contrast, not conclusions about age, gender or identity.
    """
    instruction = profile.design_instruction + (
        " Use a natural, distinct original timbre; avoid exaggerated delivery."
        if index == 0
        else " Use a subtly different original resonance and articulation, preserving all specified vocal traits."
    )
    if index < 2 or index > 3 or role.voice_identity is not None:
        return instruction, None
    neighbors = []
    roles = {item.id: item for item in project.roles}
    for collision in project.character_cast.collisions:
        if role.id not in (collision.left, collision.right):
            continue
        other_id = collision.right if collision.left == role.id else collision.left
        other = roles.get(other_id)
        if (
            other is None
            or other.voice_identity is None
            or other.voice_identity.status != "LOCKED"
        ):
            continue
        selected = next(
            (
                candidate
                for candidate in project.character_cast.auditions.get(other_id, [])
                if candidate.selected
                and candidate.id == other.voice_identity.id
                and not candidate.error
            ),
            None,
        )
        if selected is None:
            continue
        pitch = selected.measurements.get("neutral", {}).get("pitch_hz", 0)
        if not math.isfinite(pitch) or not 55 <= pitch <= 550:
            continue
        evidence = collision.model_dump(mode="json")
        neighbors.append(
            {
                "collision_id": "collision_" + _key(evidence),
                "collision": evidence,
                "role_id": other_id,
                "candidate_id": selected.id,
                "neutral_pitch_hz": pitch,
            }
        )
    if not neighbors:
        return instruction, None
    neighbors.sort(
        key=lambda item: (
            -item["collision"]["shared_scene_weight"],
            -item["collision"]["similarity"],
            item["role_id"],
        )
    )
    neighbor = neighbors[0]
    protected = {
        decision.dimension
        for decision in profile.decisions
        if decision.level in {"EXPLICIT", "SUPPORTED"}
    }
    ages = {
        decision.value.casefold()
        for decision in profile.decisions
        if decision.dimension == "perceived_vocal_age"
    }
    youthful = bool(ages & {"child", "adolescent", "youthful", "young adult"})
    low_neighbor = neighbor["neutral_pitch_hz"] < 150
    decisions, clauses = [], []

    def suggest(dimension: str, value: str, clause: str) -> None:
        decisions.append(
            {
                "dimension": dimension,
                "value": value,
                "level": "INFERRED",
                "reason": "Candidate-only ensemble contrast after a measured audition collision",
                "evidence_ids": [neighbor["collision_id"]],
            }
        )
        clauses.append(clause)

    def neutral_default(dimension: str, values: set[str]) -> bool:
        requirements = [d for d in profile.decisions if d.dimension == dimension]
        if any(
            d.level != "INFERRED" or d.value.casefold() not in values
            for d in requirements
        ):
            return False
        inspired = (
            profile.inspiration.traits.get(dimension) if profile.inspiration else None
        )
        return inspired is None or inspired.casefold() in values

    if index == 2 and "pitch" not in protected and (low_neighbor or not youthful):
        register = "higher-mid" if low_neighbor else "lower-mid"
        suggest(
            "pitch",
            register,
            f"Use a {register} register within the already established vocal age and presentation",
        )
    if index == 3 and neutral_default(
        "texture", {"clear", "neutral", "natural", "unspecified"}
    ):
        suggest(
            "texture",
            "lightly grainy and dry",
            "For this candidate only, replace the neutral inferred texture default with a lightly grainy, dry timbre; keep it natural and understated",
        )
    if (
        index == 3
        and not protected & {"texture", "resonance"}
        and neutral_default(
            "resonance", {"neutral", "natural", "balanced", "unspecified"}
        )
    ):
        suggest(
            "resonance",
            "gently forward with light nasality",
            "Use gently forward resonance with light nasality, without caricature or exaggerated delivery",
        )
    elif index == 2 and not protected & {"texture", "resonance"}:
        resonance = "brighter" if low_neighbor else "rounded"
        suggest(
            "resonance",
            resonance,
            f"Use a {resonance} resonance while retaining the specified texture",
        )
    if not protected & {"articulation", "cadence", "tempo"} and (
        index == 2
        or neutral_default(
            "articulation", {"neutral", "natural", "clear", "unspecified"}
        )
    ):
        suggest(
            "articulation",
            "forward and precise",
            "Use forward, precise articulation without changing the established cadence or accent",
        )
    if clauses:
        instruction += (
            " For this additional original audition only, apply an inferred ensemble contrast: "
            + ". ".join(clauses)
            + ". Preserve every explicit or supported vocal requirement, and preserve the established age, presentation and accent."
        )
    return instruction, {
        "schema": "ensemble-collision-retry-v2"
        if index == 3
        else "ensemble-collision-retry-v1",
        "level": "INFERRED",
        "candidate_index": index,
        "reason": "Recorded cast collision; candidate variation only, not new character facts",
        "neighbors": neighbors,
        "decisions": decisions,
        "protected_dimensions": sorted(protected),
        "profile_revision": profile.revision,
    }


def _candidate_budget(
    project: Project, role: Role, profile, compatible, root: Path
) -> int:
    """A fourth attempt requires an explicit cap and three complete failures.

    Incomplete or broken auditions need repair, not additional voice designs.
    The normal uncertainty-based budget remains unchanged for other cases.
    """
    from .auditions import candidate_count

    policy = project.character_cast.policy
    normal = candidate_count(profile, policy)
    if (
        policy.max_candidates != 4
        or policy.audition_mode == "off"
        or role.voice_identity is not None
    ):
        return normal
    if not any(role.id in (c.left, c.right) for c in project.character_cast.collisions):
        return normal
    if len(compatible) < 3:
        return normal
    for candidate in compatible[:3]:
        if (
            candidate.error
            or candidate.selected
            or candidate.identity is None
            or candidate.identity.status != "LOCKED"
            or candidate.audition_revision != AUDITION_REVISION
        ):
            return normal
        try:
            for relative, checksum in (
                (
                    candidate.identity.reference_audio,
                    candidate.identity.reference_audio_sha256,
                ),
                (candidate.identity.prompt_artifact, candidate.identity.prompt_sha256),
            ):
                if (
                    not relative
                    or not checksum
                    or not _artifact(root, relative).is_file()
                ):
                    return normal
                if _sha(_artifact(root, relative)) != checksum:
                    return normal
            for delivery in AUDITION_TEXTS:
                relative = candidate.samples.get(delivery)
                values = candidate.measurements.get(delivery, {})
                if (
                    not relative
                    or not _artifact(root, relative).is_file()
                    or values.get("rms", 0) < 0.003
                    or values.get("silence_fraction", 1) > 0.85
                    or values.get("duration_seconds", 0) < 0.15
                    or values.get("clipping_fraction", 1) > 0.005
                ):
                    return normal
        except (ValueError, OSError):
            return normal
    # Require an actionable selected-neighbor contrast; stale collisions alone
    # must not spend the final candidate budget.
    _, retry = _candidate_design(project, role, profile, 3)
    return 4 if retry and retry["decisions"] else normal


def prepare_cast(
    project_id: str,
    *,
    provider=None,
    role_ids: Sequence[str] | None = None,
    max_characters: int | None = None,
) -> dict:
    """Design at most policy.max_candidates per role and checkpoint every stage.

    All design calls precede Base prompt extraction so only one GPU model needs
    to be resident. Re-entry reuses completed artifacts and locked identities.
    """
    from dubbing_service.app.core.tts_operations import (
        release_tts_operation,
        try_acquire_tts_operation,
    )
    from dubbing_service.app.providers.tts import Qwen3TTSProvider

    from .auditions import evaluate_candidates, measure_audio

    project = refresh_cast(project_id)
    state = project.character_cast
    if not state.policy.enabled:
        return cast_report(project_id)
    root = safe_project_root(project_id)
    wanted = (
        {state.role_aliases.get(r, r) for r in role_ids}
        if role_ids is not None
        else None
    )
    roles = [
        r
        for r in project.roles
        if r.id in state.profiles and (wanted is None or r.id in wanted)
    ]
    roles.sort(key=lambda r: (-state.profiles[r.id].dialogue_count, r.id))
    limit = (
        state.policy.max_audition_characters
        if max_characters is None
        else max(0, min(max_characters, 50))
    )
    pending: list[tuple[Role, Any, list[AuditionCandidate]]] = []
    for role in roles:
        if role.voice_identity:
            continue
        profile = state.profiles[role.id]
        existing = _existing_identity(project_id, role, profile)
        if existing:
            with project_meta_lock(project_id):
                latest = load_project(project_id)
                target = next(r for r in latest.roles if r.id == role.id)
                if target.voice_identity is None:
                    target.voice_identity = existing
                    if existing.status != "LOCKED":
                        latest.character_cast.review_reasons.append(
                            f"{role.name}: existing voice reference needs repair: {existing.provenance.get('error', 'snapshot unavailable')}"
                        )
                    save_project(latest)
            continue
        if len(pending) >= limit or state.policy.audition_mode == "off":
            continue
        rows = []
        compatible = [
            c
            for c in state.auditions.get(role.id, [])
            if c.identity
            and c.identity.design_instruction == profile.design_instruction
            and not c.selected
        ]
        for i in range(_candidate_budget(project, role, profile, compatible, root)):
            cid = "voice_" + _key([project.id, role.id, profile.revision, i])
            previous = next(
                (c for c in state.auditions.get(role.id, []) if c.id == cid), None
            )
            if previous is None and i < len(compatible):
                # New algorithm/research metadata with the same design does not
                # justify spending another VoiceDesign call.
                previous = compatible[i]
            rows.append(
                previous.model_copy(deep=True)
                if previous
                else AuditionCandidate(
                    id=cid,
                    role_id=role.id,
                    profile_revision=profile.revision,
                )
            )
            candidate = rows[-1]
            candidate.profile_revision = profile.revision
            if candidate.audition_revision != AUDITION_REVISION:
                if candidate.samples:
                    candidate.sample_history.append(
                        {
                            "revision": candidate.audition_revision,
                            "samples": candidate.samples,
                            "measurements": candidate.measurements,
                        }
                    )
                candidate.samples = {}
                candidate.measurements = {}
                candidate.error = None
                candidate.review_reasons = []
                candidate.audition_revision = AUDITION_REVISION
        pending.append((role, profile, rows))
    if not pending:
        with project_meta_lock(project_id):
            latest = load_project(project_id)
            _bind_aliases(latest)
            save_project(latest)
        return cast_report(project_id)
    lease = try_acquire_tts_operation(kind="character_casting", project_id=project_id)
    if lease is None:
        raise RuntimeError("TTS lane is busy; character casting can be resumed")
    provider = provider or Qwen3TTSProvider()
    try:
        # Phase one: every candidate is an original designed reference.
        for role, profile, rows in pending:
            for i, candidate in enumerate(rows):
                if candidate.identity is not None:
                    continue
                try:
                    seed = int(_key([candidate.id])[:8], 16) % (2**31)
                    instruction, retry = _candidate_design(project, role, profile, i)
                    meta = provider.design_character_reference(
                        project_id, candidate.id, instruction, REFERENCE_TEXT, seed
                    )
                    candidate.identity = _identity(
                        project, role, profile, candidate.id, meta
                    )
                    candidate.identity.generation_settings.setdefault("seed", seed)
                    candidate.identity.generation_settings["candidate_instruction"] = (
                        instruction
                    )
                    if retry:
                        candidate.identity.provenance["ensemble_collision_retry"] = (
                            retry
                        )
                    candidate.error = None
                except Exception as exc:  # noqa: BLE001 - checkpoint provider failures for resumable review
                    candidate.error = f"voice design failed: {exc}"
                _save_candidate(project_id, candidate)
        # Phase two: prompt extraction and multi-delivery Base auditions.
        selected = [
            c
            for rows in load_project(project_id).character_cast.auditions.values()
            for c in rows
            if c.selected
        ]
        for role, profile, rows in pending:
            for candidate in rows:
                if candidate.identity is None:
                    continue
                try:
                    if not candidate.identity.prompt_artifact:
                        meta = provider.prepare_character_voice(
                            project_id,
                            candidate.id,
                            candidate.identity.model_dump(mode="json"),
                        )
                        candidate.identity = _identity(
                            project,
                            role,
                            profile,
                            candidate.id,
                            {
                                **candidate.identity.model_dump(mode="json"),
                                **meta,
                                "generation_settings": {
                                    **candidate.identity.generation_settings,
                                    **meta.get("generation_settings", {}),
                                },
                                "provenance": {
                                    **candidate.identity.provenance,
                                    **meta.get("provenance", {}),
                                },
                            },
                        )
                        _save_candidate(project_id, candidate)
                    audition_role = role.model_copy(deep=True)
                    audition_role.voice_identity = candidate.identity
                    audition_role.tts_backend = "qwen3tts"
                    for delivery, text in AUDITION_TEXTS.items():
                        relative = candidate.samples.get(delivery)
                        if relative and _artifact(root, relative).is_file():
                            continue
                        segment = Segment(
                            id=f"audition_{candidate.id}_{delivery}",
                            role_id=role.id,
                            text=text,
                            qwen3_language=role.qwen3_language or "English",
                            emotion_text=delivery,
                            playback_speed=0.93 if delivery == "quiet" else 1.0,
                            volume_gain=0.8 if delivery == "quiet" else 1.0,
                        )
                        result = provider.synthesize_segment(
                            project_id, audition_role, segment
                        )
                        if not result.success or not result.audio_path:
                            raise RuntimeError(str(result.raw))
                        target = _artifact(
                            root,
                            f"audio/cast/{candidate.id}/{AUDITION_REVISION}/{delivery}.wav",
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if Path(result.audio_path).resolve() != target:
                            shutil.copy2(result.audio_path, target)
                        from .performance import apply_scene_audio

                        apply_scene_audio(
                            target,
                            pace=segment.playback_speed,
                            volume=segment.volume_gain,
                        )
                        candidate.samples[delivery] = target.relative_to(
                            root
                        ).as_posix()
                        candidate.measurements[delivery] = measure_audio(target)
                        candidate.identity.provenance.setdefault(
                            "audition_generations", {}
                        )[delivery] = {
                            "inference_policy_revision": result.raw.get(
                                "inference_policy_revision"
                            ),
                            "generation_settings": result.raw.get(
                                "generation_settings", {}
                            ),
                        }
                        candidate.identity.generation_settings["render_policy"] = {
                            k: v
                            for k, v in result.raw.get(
                                "generation_settings", {}
                            ).items()
                            if k != "seed"
                        }
                        candidate.identity.provenance["performance_support"] = {
                            "emotion": False,
                            "pace": "ffmpeg atempo",
                            "volume": "ffmpeg gain",
                            "note": "Elevated/quiet text differs; Base emotion instructions are unsupported",
                        }
                        _save_candidate(project_id, candidate)
                    candidate.error = None
                except Exception as exc:  # noqa: BLE001 - checkpoint provider/audio failures for resumable review
                    candidate.error = f"voice audition failed: {exc}"
                _save_candidate(project_id, candidate)
            best, collisions = evaluate_candidates(
                rows, profile, selected, profile.contrast_with
            )
            with project_meta_lock(project_id):
                latest = load_project(project_id)
                target = next(r for r in latest.roles if r.id == role.id)
                live_profile = latest.character_cast.profiles.get(role.id)
                if (
                    target.voice_identity is None
                    and live_profile
                    and live_profile.revision == profile.revision
                    and best
                    and best.identity
                ):
                    target.voice_identity = best.identity
                    target.tts_backend = best.identity.backend
                    target.speaker_audio = best.identity.reference_audio
                    target.voice_description = best.identity.design_instruction
                    best.selected = True
                    selected.append(best)
                    resolved_reason = (
                        f"{role.id}: no satisfactory audition; review required"
                    )
                    latest.character_cast.review_reasons = [
                        r
                        for r in latest.character_cast.review_reasons
                        if r != resolved_reason
                    ]
                elif best is None:
                    reason = f"{role.id}: no satisfactory audition; review required"
                    if reason not in latest.character_cast.review_reasons:
                        latest.character_cast.review_reasons.append(reason)
                old = latest.character_cast.auditions.get(role.id, [])
                ids = {c.id for c in rows}
                latest.character_cast.auditions[role.id] = [
                    c for c in old if c.id not in ids
                ] + rows
                latest.character_cast.collisions = [
                    c
                    for c in latest.character_cast.collisions
                    if role.id not in (c.left, c.right)
                ] + collisions
                _bind_aliases(latest)
                save_project(latest)
    finally:
        release_tts_operation(lease)
    return cast_report(project_id)


def cast_report(project_id: str) -> dict:
    project = load_project(project_id)
    state = project.character_cast
    characters = []
    for role in project.roles:
        if role.role_type != "character":
            continue
        dossier = state.dossiers.get(role.id)
        profile = state.profiles.get(role.id)
        if not dossier:
            continue
        characters.append(
            {
                "role_id": role.id,
                "name": role.name,
                "alias_of": state.role_aliases.get(role.id),
                "research": dossier.model_dump(mode="json"),
                "casting": profile.model_dump(mode="json") if profile else None,
                "voice": role.voice_identity.model_dump(mode="json")
                if role.voice_identity
                else None,
                "status": role.voice_identity.status
                if role.voice_identity
                else "PENDING_VOICE",
                "proposed_revision": state.proposed_revisions.get(role.id),
                "auditions": [
                    c.model_dump(mode="json") for c in state.auditions.get(role.id, [])
                ],
            }
        )
    return {
        "project_id": project.id,
        "schema_version": state.schema_version,
        "policy": state.policy.model_dump(mode="json"),
        "characters": characters,
        "locked": sum(
            r.voice_identity is not None and r.voice_identity.status == "LOCKED"
            for r in project.roles
        ),
        "collisions": [c.model_dump(mode="json") for c in state.collisions],
        "review_reasons": state.review_reasons,
        "measurement_limits": "Acoustic proxies detect obvious similarity; they do not prove speaker identity or age. Qwen Base has no instruction-based emotion control.",
    }


def markdown_report(report: dict) -> str:
    lines = [
        f"# Character / Cast: {report['project_id']}",
        "",
        report["measurement_limits"],
        "",
    ]
    for row in report["characters"]:
        lines.extend([f"## {row['name']} — {row['status']}", "", "Research:"])
        for fact in row["research"]["facts"]:
            if fact["level"] != "UNKNOWN":
                lines.append(
                    f"- {fact['field']}: {fact['value']} [{fact['level']}, {fact['confidence']:.2f}]; {fact['source']} / {fact.get('location') or fact.get('chapter_id')}"
                )
        profile = row["casting"]
        if profile:
            lines.extend(["", "Casting:"])
            for decision in profile["decisions"]:
                lines.append(
                    f"- {decision['dimension']}: {decision['value']} — {decision['reason']} [{decision['level']}]"
                )
            inspiration = profile.get("inspiration")
            if inspiration:
                lines.extend(
                    [
                        "",
                        f"Inspiration: {inspiration['character']} / {inspiration.get('performer') or 'archetype'}",
                        "; ".join(inspiration["reasons"]),
                    ]
                )
        if row["voice"]:
            lines.extend(["", f"Voice: {row['voice']['id']} ({row['voice']['model']})"])
        if row["proposed_revision"]:
            lines.append(
                f"Proposed revision: {row['proposed_revision']} (locked identity preserved)"
            )
        review = row["research"]["review_reasons"] + (
            profile["review_reasons"] if profile else []
        )
        if review:
            lines.extend(["", "Review: " + "; ".join(review)])
        lines.append("")
    lines.extend(
        [
            "## Collisions / review",
            "",
            json.dumps(report["collisions"], indent=2),
            "",
            "\n".join(report["review_reasons"]),
        ]
    )
    return "\n".join(lines) + "\n"
