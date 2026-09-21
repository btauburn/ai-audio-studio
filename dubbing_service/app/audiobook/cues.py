"""Utterance-anchored audio cues (Phase 15 schema readiness).

Cues are never anchored to absolute final-timeline time.  A re-rendered Take
changes an utterance's duration; resolving a cue against the new duration
realigns it (and reports when it had to be clamped) instead of silently
drifting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from dubbing_service.app.models.audiobook import AudioCue, CueType

CUE_LANES: tuple[str, ...] = ("dialogue", "bgm", "ambience", "sfx")


def new_cue_id() -> str:
    return f"cue_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class CueWindow:
    """Resolved placement of one cue for one Take duration."""

    cue_id: str
    utterance_id: str
    cue_type: str
    start_ms: int
    duration_ms: int
    gain_db: float
    fade_in_ms: int
    fade_out_ms: int
    asset_id: Optional[str] = None
    clamped: bool = False

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


def make_cue(
    *,
    utterance_id: str,
    cue_type: CueType = "sfx",
    relative_offset_ms: int = 0,
    relative_end_ms: Optional[int] = None,
    source_revision_id: Optional[str] = None,
    occurrence_index: int = 0,
    intent: Optional[str] = None,
    asset_id: Optional[str] = None,
    gain_db: float = 0.0,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
    ducking: Optional[Dict] = None,
    loop: bool = False,
    provenance: Optional[Dict] = None,
) -> AudioCue:
    return AudioCue(
        id=new_cue_id(),
        utterance_id=utterance_id,
        source_revision_id=source_revision_id,
        occurrence_index=occurrence_index,
        relative_offset_ms=max(0, int(relative_offset_ms)),
        relative_end_ms=None if relative_end_ms is None else int(relative_end_ms),
        cue_type=cue_type,
        intent=intent,
        asset_id=asset_id,
        gain_db=gain_db,
        fade_in_ms=max(0, int(fade_in_ms)),
        fade_out_ms=max(0, int(fade_out_ms)),
        ducking=ducking,
        loop=loop,
        provenance=dict(provenance or {}),
    )


def resolve_cue_window(
    cue: AudioCue, *, utterance_start_ms: int, utterance_duration_ms: int
) -> CueWindow:
    """Resolve one cue against the *current* Take duration."""

    duration = max(0, int(utterance_duration_ms))
    clamped = False

    start_offset = max(0, int(cue.relative_offset_ms))
    if start_offset > duration:
        start_offset = duration
        clamped = True

    if cue.relative_end_ms is None:
        end_offset = duration
    else:
        end_offset = int(cue.relative_end_ms)
        if end_offset > duration:
            end_offset = duration
            clamped = True
        if end_offset <= start_offset:
            end_offset = max(start_offset, min(duration, end_offset))
            clamped = True

    window_length = max(0, end_offset - start_offset)
    fade_in = min(max(0, int(cue.fade_in_ms)), window_length)
    fade_out = min(max(0, int(cue.fade_out_ms)), window_length)

    return CueWindow(
        cue_id=cue.id,
        utterance_id=cue.utterance_id,
        cue_type=cue.cue_type,
        start_ms=max(0, int(utterance_start_ms)) + start_offset,
        duration_ms=window_length,
        gain_db=cue.gain_db,
        fade_in_ms=fade_in,
        fade_out_ms=fade_out,
        asset_id=cue.asset_id,
        clamped=clamped,
    )


def resolve_cues(
    cues: Iterable[AudioCue],
    placements: Mapping[str, tuple[int, int]],
) -> List[CueWindow]:
    """Resolve cues for a whole project.

    ``placements`` maps ``utterance_id -> (start_ms, duration_ms)`` of the
    currently selected Take.
    """

    windows: List[CueWindow] = []
    for cue in cues:
        placement = placements.get(cue.utterance_id)
        if placement is None:
            continue
        start_ms, duration_ms = placement
        windows.append(
            resolve_cue_window(
                cue,
                utterance_start_ms=start_ms,
                utterance_duration_ms=duration_ms,
            )
        )
    return windows


def cues_for_utterance(cues: Iterable[AudioCue], utterance_id: str) -> List[AudioCue]:
    return [cue for cue in cues if cue.utterance_id == utterance_id]


def reassign_cues(
    cues: Iterable[AudioCue], *, from_utterance_id: str, to_utterance_id: str
) -> int:
    """Move cues onto a new utterance ID (used after split/merge)."""

    moved = 0
    for cue in cues:
        if cue.utterance_id == from_utterance_id:
            cue.utterance_id = to_utterance_id
            moved += 1
    return moved


def lanes(cues: Iterable[AudioCue]) -> Dict[str, List[AudioCue]]:
    """Dialogue / BGM / ambience / SFX stay separate (clean narration preserved)."""

    grouped: Dict[str, List[AudioCue]] = {name: [] for name in CUE_LANES}
    for cue in cues:
        grouped.setdefault(cue.cue_type, []).append(cue)
    return grouped
