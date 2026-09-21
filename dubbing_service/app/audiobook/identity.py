"""Utterance identity and split/merge lineage (Phase 2).

Rules implemented here:

* an ordinary text edit keeps ``segment.id``;
* chapter reorder keeps IDs and only changes ordering;
* a split retires the original ID and mints new IDs recording ``split_from``;
* a merge retires the source IDs and mints one new ID recording ``merged_from``.

A semantic split or merge never pretends the original utterance survived.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Sequence

from dubbing_service.app.models.audiobook import SegmentLineage, SourceAnchor
from dubbing_service.app.models.project import Segment


class LineageError(ValueError):
    """Raised when an identity operation cannot preserve the invariants."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_segment_id(prefix: str = "seg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def is_live(segment: Segment) -> bool:
    return segment.lineage is None or segment.lineage.retired_at is None


def live_segments(segments: Iterable[Segment]) -> List[Segment]:
    return [segment for segment in segments if is_live(segment)]


def retired_segments(segments: Iterable[Segment]) -> List[Segment]:
    return [segment for segment in segments if not is_live(segment)]


def find_segment(segments: Iterable[Segment], segment_id: str) -> Segment:
    for segment in segments:
        if segment.id == segment_id:
            return segment
    raise LineageError(f"utterance 不存在: {segment_id}")


def lineage_of(segment: Segment) -> SegmentLineage:
    if segment.lineage is None:
        segment.lineage = SegmentLineage()
    return segment.lineage


def note_content_edit(segment: Segment) -> SegmentLineage:
    """An ordinary text edit preserves the ID and bumps the content revision."""

    lineage = lineage_of(segment)
    lineage.content_revision += 1
    return lineage


def retire_segment(
    segment: Segment,
    *,
    reason: str,
    when: Optional[datetime] = None,
    split_into: Optional[Sequence[str]] = None,
    merged_into: Optional[str] = None,
) -> SegmentLineage:
    lineage = lineage_of(segment)
    if lineage.retired_at is None:
        lineage.retired_at = when or utcnow()
        lineage.retired_reason = reason
    if split_into:
        lineage.split_into = list(split_into)
    if merged_into:
        lineage.merged_into = merged_into
    return lineage


# Performance settings are inherited by split/merge children; identity, audio
# and Take state never are.
_CLONED_FIELDS = (
    "emotion_mode",
    "emotion_text",
    "emotion_vector",
    "emotion_audio",
    "emotion_alpha",
    "tts_backend",
    "qwen3_speaker",
    "qwen3_language",
    "playback_speed",
    "volume_gain",
)


def _clone_utterance_settings(source: Segment, target: Segment) -> None:
    for name in _CLONED_FIELDS:
        setattr(target, name, copy.deepcopy(getattr(source, name)))


def _anchor_slices(
    anchor: Optional[SourceAnchor], lengths: Sequence[int]
) -> List[Optional[SourceAnchor]]:
    """Split a source anchor across new pieces when offsets are known."""

    if anchor is None or anchor.start_offset is None or anchor.end_offset is None:
        return [None for _ in lengths]
    cursor = anchor.start_offset
    base_occurrence = anchor.occurrence_index or 0
    anchors: List[Optional[SourceAnchor]] = []
    for position, length in enumerate(lengths):
        end = min(cursor + length, anchor.end_offset)
        anchors.append(
            SourceAnchor(
                source_revision_id=anchor.source_revision_id,
                chapter_id=anchor.chapter_id,
                start_offset=cursor,
                end_offset=end,
                occurrence_index=base_occurrence + position,
            )
        )
        cursor = end
    return anchors


def _merge_anchor(
    anchors: Sequence[Optional[SourceAnchor]], chapter_id: Optional[str]
) -> Optional[SourceAnchor]:
    usable = [
        item
        for item in anchors
        if item is not None
        and item.start_offset is not None
        and item.end_offset is not None
    ]
    if not usable or len(usable) != len([item for item in anchors if item is not None]):
        return None
    revision_ids = {item.source_revision_id for item in usable}
    if len(revision_ids) != 1:
        return None
    return SourceAnchor(
        source_revision_id=usable[0].source_revision_id,
        chapter_id=chapter_id or usable[0].chapter_id,
        start_offset=min(
            item.start_offset for item in usable if item.start_offset is not None
        ),
        end_offset=max(item.end_offset for item in usable if item.end_offset is not None),
        occurrence_index=min((item.occurrence_index or 0) for item in usable),
    )


@dataclass
class SplitResult:
    original: Segment
    new_segments: List[Segment] = field(default_factory=list)
    retired_id: str = ""


def split_segment(
    segments: List[Segment],
    segment_id: str,
    *,
    parts: Sequence[str],
    id_factory: Callable[[], str] = new_segment_id,
) -> SplitResult:
    """Retire ``segment_id`` and mint one new utterance per part."""

    if len(parts) < 2:
        raise LineageError("split 至少需要两个片段")
    if any(not str(part).strip() for part in parts):
        raise LineageError("split 片段不能为空")

    original = find_segment(segments, segment_id)
    if not is_live(original):
        raise LineageError(f"utterance 已退役，不能再次 split: {segment_id}")

    anchors = _anchor_slices(original.source_anchor, [len(str(part)) for part in parts])
    created: List[Segment] = []
    for offset, (part, anchor) in enumerate(zip(parts, anchors)):
        child = Segment(
            id=id_factory(),
            chapter_id=original.chapter_id,
            role_id=original.role_id,
            index=original.index + offset,
            text=str(part),
            status=original.status,
        )
        _clone_utterance_settings(original, child)
        child.index = original.index + offset
        child.source_anchor = anchor
        child.attribution = copy.deepcopy(original.attribution)
        child.lineage = SegmentLineage(split_from=original.id)
        created.append(child)

    shift = len(created) - 1
    for sibling in segments:
        if sibling is original or sibling.chapter_id != original.chapter_id:
            continue
        if sibling.index > original.index and is_live(sibling):
            sibling.index += shift

    position = segments.index(original)
    updated = list(segments)
    # The retired original stays in the list so its Take history and lineage
    # survive; it is simply no longer live.
    updated[position : position + 1] = [*created, original]
    segments[:] = updated

    retire_segment(original, reason="split", split_into=[item.id for item in created])
    return SplitResult(original=original, new_segments=created, retired_id=original.id)


def merge_segments(
    segments: List[Segment],
    segment_ids: Sequence[str],
    *,
    text: Optional[str] = None,
    id_factory: Callable[[], str] = new_segment_id,
    separator: str = " ",
) -> Segment:
    """Retire every source utterance and mint exactly one new utterance."""

    ids = list(segment_ids)
    if len(ids) < 2:
        raise LineageError("merge 至少需要两个 utterance")
    sources = [find_segment(segments, item) for item in ids]
    for source in sources:
        if not is_live(source):
            raise LineageError(f"utterance 已退役，不能参与 merge: {source.id}")

    merged_text = (
        text if text is not None else separator.join(item.text for item in sources)
    )
    primary = sources[0]
    created = Segment(
        id=id_factory(),
        chapter_id=primary.chapter_id,
        role_id=primary.role_id,
        index=min(item.index for item in sources),
        text=merged_text,
        status=primary.status,
    )
    _clone_utterance_settings(primary, created)
    created.index = min(item.index for item in sources)
    created.source_anchor = _merge_anchor(
        [item.source_anchor for item in sources], primary.chapter_id
    )
    created.attribution = copy.deepcopy(primary.attribution)
    created.lineage = SegmentLineage(merged_from=[item.id for item in sources])

    source_ids = set(ids)
    updated: List[Segment] = []
    placed = False
    for item in segments:
        if item.id in source_ids:
            if not placed:
                # Keep the retired sources alongside the merged utterance so
                # their Takes and lineage remain recoverable.
                updated.extend([created, *sources])
                placed = True
            continue
        updated.append(item)
    if not placed:  # pragma: no cover - defensive
        updated.append(created)
    segments[:] = updated

    for source in sources:
        retire_segment(source, reason="merge", merged_into=created.id)
    return created


def reorder_utterances(
    segments: List[Segment], ordered_ids: Sequence[str]
) -> dict[str, int]:
    """Chapter/utterance reorder: IDs are preserved, only ordering changes."""

    live = live_segments(segments)
    if len(ordered_ids) != len(live):
        raise LineageError(
            f"reorder 需要全部 {len(live)} 个活跃 utterance，收到 {len(ordered_ids)} 个"
        )
    if set(ordered_ids) != {item.id for item in live}:
        raise LineageError("reorder 必须使用相同的一组 utterance ID（ID 保持不变）")

    positions: dict[str, int] = {}
    for position, segment_id in enumerate(ordered_ids):
        segment = find_segment(segments, segment_id)
        segment.index = position
        positions[segment_id] = position

    # The persisted file order must match the new order; retired utterances keep
    # their relative position at the end so their history stays readable.
    ordered = [find_segment(segments, segment_id) for segment_id in ordered_ids]
    segments[:] = [*ordered, *retired_segments(segments)]
    return positions
