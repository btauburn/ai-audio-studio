"""Take commit, staleness and safe promotion (Phases 6, 10, 11).

A Take is the immutable artifact of one synthesis requirement.  It is never
mutated in place, and it is never auto-promoted into the selected slot: a Take
that is stale, mismatched with its sentence, or failing a blocking QA gate can
be kept as history but must not silently replace a good recording.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable, List, Optional, Sequence

from dubbing_service.app.models.audiobook import ChunkRecord, TakeQaReport
from dubbing_service.app.models.project import Segment, SegmentTake
from dubbing_service.app.services.takes import (
    ensure_take_state,
    normalize_slot,
    sync_legacy_projection,
)
from dubbing_service.app.audiobook.fingerprint import (
    SynthesisRequirement,
    changed_components,
)


class PromotionError(ValueError):
    """Raised when a Take must not become the selected result."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_take(segment: Segment, take_id: str) -> SegmentTake:
    for take in segment.takes:
        if take.id == take_id:
            return take
    raise PromotionError(f"Take 不存在: {take_id}")


def selected_take(segment: Segment) -> Optional[SegmentTake]:
    if segment.selected_take_id:
        return next(
            (item for item in segment.takes if item.id == segment.selected_take_id),
            None,
        )
    return None


def build_take(
    segment: Segment,
    *,
    slot: object,
    take_id: str,
    audio_file: str,
    audio_sha256: Optional[str],
    requirement: SynthesisRequirement,
    renderer_id: Optional[str] = None,
    engine_id: Optional[str] = None,
    model_id: Optional[str] = None,
    spoken_text_revision_id: Optional[str] = None,
    source_text_hash: Optional[str] = None,
    synthesis_text: Optional[str] = None,
    chunk_map: Optional[Sequence[ChunkRecord]] = None,
    generation_metadata: Optional[dict] = None,
) -> SegmentTake:
    """Build (but do not attach) a Take bound to one synthesis requirement."""

    components = requirement.components()
    return SegmentTake(
        id=take_id,
        audio_file=audio_file,
        audio_sha256=audio_sha256,
        text=segment.text,
        synthesis_text=synthesis_text if synthesis_text is not None else segment.text,
        generation_metadata=dict(generation_metadata or {}),
        role_id=segment.role_id,
        tts_backend=segment.tts_backend,
        qwen3_speaker=segment.qwen3_speaker,
        qwen3_language=segment.qwen3_language,
        emotion_mode=segment.emotion_mode,
        emotion_text=segment.emotion_text,
        emotion_vector=list(segment.emotion_vector) if segment.emotion_vector else None,
        emotion_audio=segment.emotion_audio,
        emotion_alpha=segment.emotion_alpha,
        input_fingerprint=requirement.fingerprint(),
        fingerprint_components=components,
        renderer_id=renderer_id,
        engine_id=engine_id,
        model_id=model_id,
        spoken_text_revision_id=spoken_text_revision_id,
        source_text_hash=source_text_hash,
        chunk_map=list(chunk_map or []),
    )


def attach_take(
    segment: Segment,
    take: SegmentTake,
    *,
    slot: object = "a",
    promote: bool = False,
) -> SegmentTake:
    """Attach an immutable Take.  Never deletes or replaces prior audio."""

    ensure_take_state(segment)
    if any(item.id == take.id for item in segment.takes):
        raise PromotionError(f"Take ID 已存在: {take.id}")

    slot_name = normalize_slot(slot)
    segment.takes.append(take)
    segment.take_slots[slot_name].take_id = take.id
    segment.take_slots[slot_name].needs_synthesis = False

    if promote and segment.selected_take_id is None:
        segment.selected_take_id = take.id
        segment.active_version = slot_name
    sync_legacy_projection(segment)
    return take


def take_is_stale(
    take: SegmentTake, requirement: SynthesisRequirement
) -> tuple[bool, List[str]]:
    """Staleness plus the exact dependency names that changed."""

    if take.input_fingerprint and take.input_fingerprint == requirement.fingerprint():
        return False, []
    components = take.fingerprint_components
    if components:
        changed = changed_components(components, requirement.components())
        if not changed and take.input_fingerprint:
            changed = ["fingerprint"]
        return bool(changed), changed
    if take.input_fingerprint:
        return True, ["fingerprint"]
    # Legacy Take with no requirement stamp at all: nothing is provable, so we
    # do not invent staleness.
    return False, []


def mark_stale_takes(
    segment: Segment, requirement: SynthesisRequirement
) -> List[str]:
    """Flag Takes that no longer satisfy ``requirement``; return their IDs."""

    stale: List[str] = []
    for take in segment.takes:
        is_stale, changed = take_is_stale(take, requirement)
        if is_stale:
            take.stale = True
            take.stale_components = changed
            stale.append(take.id)
    return stale


def mark_stale_for_requirement(
    segments: Iterable[Segment],
    requirement_for: Callable[[Segment, SegmentTake], Optional[SynthesisRequirement]],
) -> dict[str, List[str]]:
    """Mark stale Takes using a per-Take requirement factory.

    The factory receives the Take as well, because the live requirement of an
    utterance that declares no engine of its own is resolved from that Take's
    recorded renderer identity (see ``pipeline.refresh_staleness``).
    """

    result: dict[str, List[str]] = {}
    for segment in segments:
        stale: List[str] = []
        for take in segment.takes:
            requirement = requirement_for(segment, take)
            if requirement is None:
                continue
            is_stale, changed = take_is_stale(take, requirement)
            if is_stale:
                take.stale = True
                take.stale_components = changed
                stale.append(take.id)
        if stale:
            result[segment.id] = stale
    return result


def attach_qa_report(
    segment: Segment, take_id: str, report: TakeQaReport
) -> SegmentTake:
    take = find_take(segment, take_id)
    take.qa_report = report
    return take


def promote_take(
    segment: Segment,
    take_id: str,
    *,
    requirement: Optional[SynthesisRequirement] = None,
    allow_stale: bool = False,
    allow_blocking_qa: bool = False,
) -> SegmentTake:
    """Select a Take only when it is safe to do so.

    Refusals are explicit errors rather than silent fallbacks; the previously
    selected Take is never removed.
    """

    take = find_take(segment, take_id)
    if not take.audio_file:
        raise PromotionError(f"Take 没有音频文件，不能选用: {take_id}")

    if take.stale and not allow_stale:
        raise PromotionError(
            f"Take 已过期（依赖变化: {', '.join(take.stale_components) or 'unknown'}），不能选用"
        )

    if requirement is not None:
        is_stale, changed = take_is_stale(take, requirement)
        if is_stale and not allow_stale:
            raise PromotionError(
                f"Take 与当前需求不一致（{', '.join(changed) or 'fingerprint'}），不能选用"
            )

    report = take.qa_report
    if report is not None:
        if report.output_audio_hash and take.audio_sha256:
            if report.output_audio_hash != take.audio_sha256:
                raise PromotionError("QA 报告与 Take 音频哈希不匹配，不能选用")
        blocking = report.blocking_failures()
        if blocking and not allow_blocking_qa:
            checks = ", ".join(item.check_id for item in blocking)
            raise PromotionError(f"QA 阻断项未通过（{checks}），不能选用")

    segment.selected_take_id = take.id
    segment.active_version = normalize_slot(segment.active_version)
    for slot_name, state in segment.take_slots.items():
        if state.take_id == take.id:
            segment.active_version = slot_name
            state.needs_synthesis = False
    # The previously selected Take stays in the project untouched: promotion is
    # a pointer change, not destruction.
    sync_legacy_projection(segment)
    return take


def slot_take_id(segment: Segment, slot: object) -> Optional[str]:
    ensure_take_state(segment)
    return segment.take_slots[normalize_slot(slot)].take_id
