"""Canonical Take/A-B helpers.

The legacy WebUI still reads flat ``Segment`` fields.  This module keeps those
fields as a projection while ``take_slots`` and ``takes`` remain the canonical
source of truth.  Centralising the projection prevents A/B switching from
swapping or destroying generated audio and edit metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from dubbing_service.app.models.project import (
    Segment,
    SegmentEditState,
    SegmentTake,
    SegmentTakeSlot,
)
from dubbing_service.app.models.enums import SegmentReviewStatus

TakeSlotName = Literal["a", "b"]
SLOT_NAMES: tuple[TakeSlotName, TakeSlotName] = ("a", "b")


def normalize_slot(value: object) -> TakeSlotName:
    return "b" if str(value or "").lower() == "b" else "a"


def other_slot(slot: TakeSlotName) -> TakeSlotName:
    return "a" if slot == "b" else "b"


def _copy_edit_state(value: SegmentEditState | None) -> SegmentEditState | None:
    return value.model_copy(deep=True) if value is not None else None


def _base_settings(segment: Segment) -> SegmentTakeSlot:
    return SegmentTakeSlot(
        tts_backend=segment.tts_backend,
        qwen3_speaker=segment.qwen3_speaker,
        qwen3_language=segment.qwen3_language,
        emotion_mode=segment.emotion_mode,
        emotion_text=segment.emotion_text,
        emotion_vector=list(segment.emotion_vector) if segment.emotion_vector else None,
        emotion_audio=segment.emotion_audio,
        emotion_alpha=segment.emotion_alpha,
    )


def _alternate_settings(segment: Segment) -> SegmentTakeSlot:
    """Best-effort settings recovery for the old partial ``alt_*`` model."""

    base = _base_settings(segment)
    base.tts_backend = segment.alt_tts_backend or segment.tts_backend
    base.emotion_text = (
        segment.alt_emotion_text
        if segment.alt_emotion_text is not None
        else segment.emotion_text
    )
    base.emotion_alpha = (
        segment.alt_emotion_alpha
        if segment.alt_emotion_alpha is not None
        else segment.emotion_alpha
    )
    return base


def _legacy_take_id(segment: Segment, slot: TakeSlotName, audio_file: str) -> str:
    value = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"dubbing-take:{segment.id}:{slot}:{audio_file}",
    )
    return value.hex[:16]


def _find_take_unchecked(segment: Segment, take_id: str | None) -> SegmentTake | None:
    if not take_id:
        return None
    return next((item for item in segment.takes if item.id == take_id), None)


def adopt_legacy_audio_references(segment: Segment) -> Segment:
    """Import flat audio fields changed by an older client into Take slots.

    Undo/redo snapshots and older API clients can submit canonical fields from
    before the Take migration.  When a submitted path differs from the slot's
    current Take, preserve it as a Take instead of dropping it during
    projection.
    """

    if not segment.take_slots:
        return ensure_take_state(segment)
    active = normalize_slot(segment.active_version)
    paths = {
        active: segment.audio_file,
        other_slot(active): segment.alt_audio_file,
    }
    adopted = False
    for slot, rel_path in paths.items():
        if not rel_path:
            continue
        state = segment.take_slots.setdefault(slot, _base_settings(segment))
        current = _find_take_unchecked(segment, state.take_id)
        if current is not None and current.audio_file == str(rel_path):
            continue
        matching = next(
            (take for take in segment.takes if take.audio_file == str(rel_path)),
            None,
        )
        settings = _base_settings(segment) if slot == active else _alternate_settings(segment)
        if matching is None:
            matching = SegmentTake(
                id=_legacy_take_id(segment, slot, str(rel_path)),
                audio_file=str(rel_path),
                text=segment.text,
                role_id=segment.role_id,
                tts_backend=settings.tts_backend,
                qwen3_speaker=settings.qwen3_speaker,
                qwen3_language=settings.qwen3_language,
                emotion_mode=settings.emotion_mode,
                emotion_text=settings.emotion_text,
                emotion_vector=(
                    list(settings.emotion_vector) if settings.emotion_vector else None
                ),
                emotion_audio=settings.emotion_audio,
                emotion_alpha=settings.emotion_alpha,
                playback_speed=segment.playback_speed,
                volume_gain=segment.volume_gain,
                baked_speed=segment.playback_speed,
                baked_volume=segment.volume_gain,
                edit_state=(
                    _copy_edit_state(segment.edit_state) if slot == active else None
                ),
                created_at=segment.updated_at,
            )
            segment.takes.append(matching)
        state.tts_backend = settings.tts_backend
        state.qwen3_speaker = settings.qwen3_speaker
        state.qwen3_language = settings.qwen3_language
        state.emotion_mode = settings.emotion_mode
        state.emotion_text = settings.emotion_text
        state.emotion_vector = (
            list(settings.emotion_vector) if settings.emotion_vector else None
        )
        state.emotion_audio = settings.emotion_audio
        state.emotion_alpha = settings.emotion_alpha
        state.take_id = matching.id
        state.needs_synthesis = False
        adopted = True
    if adopted and segment.selected_take_id is None:
        segment.selected_take_id = segment.take_slots[active].take_id
    if adopted:
        sync_legacy_projection(segment)
    return segment


def ensure_take_state(segment: Segment) -> Segment:
    """Populate canonical Take state from a legacy Segment when necessary."""

    active = normalize_slot(segment.active_version)
    inactive = other_slot(active)

    if not segment.take_slots:
        # The old swap endpoint kept the active version in the primary fields.
        # Therefore primary=B and alternate=A when active_version is ``b``.
        segment.take_slots = {
            active: _base_settings(segment),
            inactive: _alternate_settings(segment),
        }

        audio_by_slot = {
            active: segment.audio_file,
            inactive: segment.alt_audio_file,
        }
        for slot in SLOT_NAMES:
            audio_file = audio_by_slot.get(slot)
            if not audio_file:
                continue
            slot_state = segment.take_slots[slot]
            take = SegmentTake(
                id=_legacy_take_id(segment, slot, str(audio_file)),
                audio_file=str(audio_file),
                text=segment.text,
                role_id=segment.role_id,
                tts_backend=slot_state.tts_backend,
                qwen3_speaker=slot_state.qwen3_speaker,
                qwen3_language=slot_state.qwen3_language,
                emotion_mode=slot_state.emotion_mode,
                emotion_text=slot_state.emotion_text,
                emotion_vector=(
                    list(slot_state.emotion_vector)
                    if slot_state.emotion_vector
                    else None
                ),
                emotion_audio=slot_state.emotion_audio,
                emotion_alpha=slot_state.emotion_alpha,
                playback_speed=segment.playback_speed,
                volume_gain=segment.volume_gain,
                baked_speed=segment.playback_speed,
                baked_volume=segment.volume_gain,
                edit_state=(
                    _copy_edit_state(segment.edit_state) if slot == active else None
                ),
                created_at=segment.updated_at,
            )
            segment.takes.append(take)
            slot_state.take_id = take.id
            slot_state.needs_synthesis = False

    for slot in SLOT_NAMES:
        if slot not in segment.take_slots:
            segment.take_slots[slot] = _base_settings(segment)

    # Broken references are detached instead of manufacturing a Take without
    # an audio file.  The consistency audit can then report the empty slot.
    for slot in SLOT_NAMES:
        state = segment.take_slots[slot]
        if state.take_id and _find_take_unchecked(segment, state.take_id) is None:
            state.take_id = None

    if segment.selected_take_id and _find_take_unchecked(
        segment, segment.selected_take_id
    ) is None:
        segment.selected_take_id = None
    if segment.selected_take_id is None:
        segment.selected_take_id = segment.take_slots[active].take_id

    sync_legacy_projection(segment)
    return segment


def get_slot(segment: Segment, slot: object | None = None) -> SegmentTakeSlot:
    ensure_take_state(segment)
    name = normalize_slot(segment.active_version if slot is None else slot)
    return segment.take_slots[name]


def get_take(segment: Segment, take_id: str | None) -> SegmentTake | None:
    ensure_take_state(segment)
    return _find_take_unchecked(segment, take_id)


def get_slot_take(segment: Segment, slot: object | None = None) -> SegmentTake | None:
    state = get_slot(segment, slot)
    take = _find_take_unchecked(segment, state.take_id)
    return None if take is not None and take.missing else take


def get_selected_take(segment: Segment) -> SegmentTake | None:
    ensure_take_state(segment)
    selected = _find_take_unchecked(segment, segment.selected_take_id)
    if selected is not None and selected.missing:
        selected = None
    return selected or get_slot_take(segment, segment.active_version)


def capture_active_legacy_settings(segment: Segment) -> None:
    """Persist edits made through the compatibility flat fields."""

    if not segment.take_slots:
        ensure_take_state(segment)
        return
    for name in SLOT_NAMES:
        if name not in segment.take_slots:
            segment.take_slots[name] = _base_settings(segment)
    state = segment.take_slots[normalize_slot(segment.active_version)]
    state.tts_backend = segment.tts_backend
    state.qwen3_speaker = segment.qwen3_speaker
    state.qwen3_language = segment.qwen3_language
    state.emotion_mode = segment.emotion_mode
    state.emotion_text = segment.emotion_text
    state.emotion_vector = (
        list(segment.emotion_vector) if segment.emotion_vector else None
    )
    state.emotion_audio = segment.emotion_audio
    state.emotion_alpha = segment.emotion_alpha

    take = _find_take_unchecked(segment, state.take_id)
    if take is not None:
        take.playback_speed = segment.playback_speed
        take.volume_gain = segment.volume_gain
        take.edit_state = _copy_edit_state(segment.edit_state)


def sync_legacy_projection(segment: Segment) -> None:
    """Project the canonical active/other slots onto legacy response fields."""

    active = normalize_slot(segment.active_version)
    inactive = other_slot(active)
    active_state = segment.take_slots[active]
    inactive_state = segment.take_slots[inactive]
    active_take = _find_take_unchecked(segment, active_state.take_id)
    inactive_take = _find_take_unchecked(segment, inactive_state.take_id)
    if active_take is not None and active_take.missing:
        active_take = None
    if inactive_take is not None and inactive_take.missing:
        inactive_take = None

    segment.active_version = active
    segment.audio_file = active_take.audio_file if active_take is not None else None
    segment.alt_audio_file = inactive_take.audio_file if inactive_take is not None else None
    segment.tts_backend = active_state.tts_backend
    segment.qwen3_speaker = active_state.qwen3_speaker
    segment.qwen3_language = active_state.qwen3_language
    segment.emotion_mode = active_state.emotion_mode
    segment.emotion_text = active_state.emotion_text
    segment.emotion_vector = (
        list(active_state.emotion_vector) if active_state.emotion_vector else None
    )
    segment.emotion_audio = active_state.emotion_audio
    segment.emotion_alpha = active_state.emotion_alpha

    segment.alt_tts_backend = inactive_state.tts_backend
    segment.alt_emotion_text = inactive_state.emotion_text
    segment.alt_emotion_alpha = inactive_state.emotion_alpha

    if active_take is not None:
        segment.playback_speed = active_take.playback_speed
        segment.volume_gain = active_take.volume_gain
        segment.edit_state = _copy_edit_state(active_take.edit_state)
    else:
        segment.playback_speed = 1.0
        segment.volume_gain = 1.0
        segment.edit_state = None


def set_active_slot(segment: Segment, slot: object) -> None:
    ensure_take_state(segment)
    capture_active_legacy_settings(segment)
    segment.active_version = normalize_slot(slot)
    sync_legacy_projection(segment)


def create_take(
    segment: Segment,
    *,
    slot: object,
    take_id: str,
    audio_file: str,
    audio_sha256: str | None = None,
    source_take_id: str | None = None,
    synthesis_text: str | None = None,
    generation_metadata: dict | None = None,
) -> SegmentTake:
    """Attach a new immutable audio result to an A/B slot."""

    ensure_take_state(segment)
    slot_name = normalize_slot(slot)
    state = segment.take_slots[slot_name]
    previous_id = state.take_id
    if _find_take_unchecked(segment, take_id) is not None:
        raise ValueError(f"Take ID 已存在: {take_id}")

    take = SegmentTake(
        id=take_id,
        audio_file=audio_file,
        audio_sha256=audio_sha256,
        text=segment.text,
        synthesis_text=synthesis_text or segment.text,
        generation_metadata=dict(generation_metadata or {}),
        role_id=segment.role_id,
        tts_backend=state.tts_backend,
        qwen3_speaker=state.qwen3_speaker,
        qwen3_language=state.qwen3_language,
        emotion_mode=state.emotion_mode,
        emotion_text=state.emotion_text,
        emotion_vector=list(state.emotion_vector) if state.emotion_vector else None,
        emotion_audio=state.emotion_audio,
        emotion_alpha=state.emotion_alpha,
        source_take_id=source_take_id,
    )
    segment.takes.append(take)
    state.take_id = take.id
    state.needs_synthesis = False

    previous = _find_take_unchecked(segment, previous_id)
    if previous is not None and not any(
        item.take_id == previous.id for item in segment.take_slots.values()
    ):
        previous.archived_at = datetime.now(timezone.utc)

    if segment.selected_take_id in {None, previous_id}:
        segment.selected_take_id = take.id
    segment.active_version = slot_name
    segment.review_status = SegmentReviewStatus.TO_REVIEW
    segment.reviewed_at = None
    sync_legacy_projection(segment)
    return take


def clear_slot(segment: Segment, slot: object) -> SegmentTake | None:
    ensure_take_state(segment)
    capture_active_legacy_settings(segment)
    slot_name = normalize_slot(slot)
    state = segment.take_slots[slot_name]
    old = _find_take_unchecked(segment, state.take_id)
    old_id = state.take_id
    state.take_id = None
    state.needs_synthesis = True
    if old is not None and not any(
        item.take_id == old.id for item in segment.take_slots.values()
    ):
        old.archived_at = datetime.now(timezone.utc)
    if segment.selected_take_id == old_id:
        segment.selected_take_id = segment.take_slots[other_slot(slot_name)].take_id
    segment.review_status = (
        SegmentReviewStatus.NEEDS_REVISION
        if any(item.take_id for item in segment.take_slots.values())
        else SegmentReviewStatus.DRAFT
    )
    segment.reviewed_at = None
    sync_legacy_projection(segment)
    return old


def select_take(segment: Segment, take_id: str) -> SegmentTake:
    take = get_take(segment, take_id)
    if take is None:
        raise ValueError(f"Take 不存在: {take_id}")
    if segment.selected_take_id != take.id:
        segment.selected_take_id = take.id
        segment.review_status = SegmentReviewStatus.TO_REVIEW
        segment.reviewed_at = None
    return take


def restore_take_to_slot(segment: Segment, take_id: str, slot: object) -> SegmentTake:
    ensure_take_state(segment)
    take = _find_take_unchecked(segment, take_id)
    if take is None:
        raise ValueError(f"Take 不存在: {take_id}")
    slot_name = normalize_slot(slot)
    current_id = segment.take_slots[slot_name].take_id
    current = _find_take_unchecked(segment, current_id)
    segment.take_slots[slot_name].take_id = take.id
    segment.take_slots[slot_name].needs_synthesis = False
    take.archived_at = None
    if current is not None and not any(
        item.take_id == current.id for item in segment.take_slots.values()
    ):
        current.archived_at = datetime.now(timezone.utc)
    if segment.selected_take_id in {None, current_id}:
        segment.selected_take_id = take.id
    segment.active_version = slot_name
    segment.review_status = SegmentReviewStatus.TO_REVIEW
    segment.reviewed_at = None
    sync_legacy_projection(segment)
    return take


def update_active_edit_state(
    segment: Segment,
    edit_state: SegmentEditState | None,
    *,
    speed: float | None = None,
    volume: float | None = None,
) -> SegmentTake:
    take = get_slot_take(segment, segment.active_version)
    if take is None:
        raise ValueError("当前槽位没有可编辑的 Take")
    previous_edit = _copy_edit_state(take.edit_state)
    next_edit = _copy_edit_state(edit_state)
    if previous_edit is not None and previous_edit != next_edit:
        take.edit_history.append(previous_edit)
        take.edit_history = take.edit_history[-20:]
    take.edit_state = next_edit
    if speed is not None:
        take.playback_speed = speed
    if volume is not None:
        take.volume_gain = volume
    segment.review_status = SegmentReviewStatus.TO_REVIEW
    segment.reviewed_at = None
    sync_legacy_projection(segment)
    return take


def restore_active_edit_revision(segment: Segment, revision_index: int) -> SegmentTake:
    take = get_slot_take(segment, segment.active_version)
    if take is None:
        raise ValueError("当前槽位没有可编辑的 Take")
    if revision_index < 0 or revision_index >= len(take.edit_history):
        raise ValueError("编辑修订不存在")
    current = _copy_edit_state(take.edit_state)
    restored = _copy_edit_state(take.edit_history[revision_index])
    take.edit_history = take.edit_history[:revision_index] + take.edit_history[revision_index + 1:]
    if current is not None:
        take.edit_history.append(current)
    take.edit_history = take.edit_history[-20:]
    take.edit_state = restored
    if restored is not None:
        take.playback_speed = restored.speed or take.playback_speed
        take.volume_gain = restored.volume if restored.volume is not None else take.volume_gain
    segment.review_status = SegmentReviewStatus.TO_REVIEW
    segment.reviewed_at = None
    sync_legacy_projection(segment)
    return take


def mark_slots_dirty(segment: Segment, *slots: object) -> None:
    """Mark generation settings stale without discarding playable history."""

    capture_active_legacy_settings(segment)
    names = {normalize_slot(item) for item in slots} or {
        normalize_slot(segment.active_version)
    }
    for name in names:
        segment.take_slots[name].needs_synthesis = True
    segment.review_status = (
        SegmentReviewStatus.NEEDS_REVISION
        if segment.takes
        else SegmentReviewStatus.DRAFT
    )
    segment.reviewed_at = None
    sync_legacy_projection(segment)


def prune_archived_takes(
    segment: Segment,
    *,
    now: datetime | None = None,
    max_archived: int = 20,
    max_age_days: int = 90,
    protected_take_ids: set[str] | None = None,
) -> list[SegmentTake]:
    """Remove unprotected archived Take metadata according to retention policy."""

    ensure_take_state(segment)
    reference_ids = {
        state.take_id for state in segment.take_slots.values() if state.take_id
    }
    if segment.selected_take_id:
        reference_ids.add(segment.selected_take_id)
    reference_ids.update(protected_take_ids or set())
    candidates = [
        take
        for take in segment.takes
        if take.id not in reference_ids and not take.pinned and take.archived_at is not None
    ]
    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(days=max(0, max_age_days))
    remove_ids = {
        take.id
        for take in candidates
        if max_age_days >= 0 and take.archived_at is not None and take.archived_at < cutoff
    }
    remaining = sorted(
        (take for take in candidates if take.id not in remove_ids),
        key=lambda take: take.archived_at or take.created_at,
        reverse=True,
    )
    if max_archived >= 0:
        remove_ids.update(take.id for take in remaining[max_archived:])
    removed = [take for take in segment.takes if take.id in remove_ids]
    if removed:
        segment.takes = [take for take in segment.takes if take.id not in remove_ids]
        sync_legacy_projection(segment)
    return removed
