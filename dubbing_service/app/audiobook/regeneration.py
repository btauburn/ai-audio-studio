"""Dependency-aware regeneration planning (Phase 14).

A change should regenerate the smallest set of things that actually depends on
it.  This module answers three questions without doing any rendering:

* which utterances are affected by a change;
* whether the change needs no render at all (cover art, title, review notes);
* whether the change is a remix-only change (music level, cue timing) that
  reuses every existing Take.

The planner never mutates audio and never selects a Take; it produces a plan
that :mod:`dubbing_service.app.audiobook.pipeline` can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from dubbing_service.app.models.audiobook import AudioCue
from dubbing_service.app.models.project import Project, Segment

NO_RENDER = "no_render"
REMIX_ONLY = "remix_only"
UTTERANCE_RENDER = "utterance_render"

#: Documented classification of every change kind the studio can trigger.
CHANGE_KINDS: Mapping[str, str] = {
    "book_title": NO_RENDER,
    "author": NO_RENDER,
    "cover_art": NO_RENDER,
    "description": NO_RENDER,
    "review_note": NO_RENDER,
    "ui_state": NO_RENDER,
    "project_name": NO_RENDER,
    "music_level": REMIX_ONLY,
    "cue_timing": REMIX_ONLY,
    "cue_lane": REMIX_ONLY,
    "spoken_text": UTTERANCE_RENDER,
    "pronunciation": UTTERANCE_RENDER,
    "voice": UTTERANCE_RENDER,
    "reference_audio": UTTERANCE_RENDER,
    "renderer": UTTERANCE_RENDER,
    "model": UTTERANCE_RENDER,
    "emotion": UTTERANCE_RENDER,
    "speed": UTTERANCE_RENDER,
    "attribution": UTTERANCE_RENDER,
}

#: Change kinds that are scoped to explicit utterance ids.
UTTERANCE_SCOPED_CHANGE_KINDS = frozenset(
    {
        "spoken_text",
        "pronunciation",
        "voice",
        "reference_audio",
        "emotion",
        "speed",
        "attribution",
    }
)

#: Change kinds that affect every live utterance in the project.
PROJECT_WIDE_CHANGE_KINDS = frozenset({"renderer", "model"})


@dataclass
class RegenerationPlan:
    """What must happen for a change, with no work performed yet."""

    kind: str
    classification: str
    utterance_ids: List[str] = field(default_factory=list)
    remix_only: bool = False
    requires_render: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "classification": self.classification,
            "utterance_ids": list(self.utterance_ids),
            "remix_only": self.remix_only,
            "requires_render": self.requires_render,
            "notes": list(self.notes),
        }


def classify_change(kind: str) -> str:
    """Return ``no_render``, ``remix_only`` or ``utterance_render``."""

    try:
        return CHANGE_KINDS[kind]
    except KeyError as exc:  # pragma: no cover - guarded by callers
        raise KeyError(f"unknown change kind: {kind}") from exc


def live_utterance_ids(segments: Iterable[Segment]) -> List[str]:
    """Every live, in-scope utterance id (retired utterances are excluded)."""

    return [segment.id for segment in segments if _is_live(segment)]


def _is_live(segment: Segment) -> bool:
    lineage = segment.lineage
    return lineage is None or lineage.retired_at is None


def plan_regeneration(
    kind: str,
    *,
    project: Optional[Project] = None,
    segments: Sequence[Segment] = (),
    utterance_ids: Optional[Iterable[str]] = None,
    cues: Sequence[AudioCue] = (),
) -> RegenerationPlan:
    """Build the plan for one change kind.

    ``utterance_ids`` is honoured only for utterance-scoped kinds; project-wide
    kinds (renderer/model change) always expand to every live utterance, which
    is the Phase 5 promise that a model change invalidates existing Takes.
    """

    classification = classify_change(kind)
    live = live_utterance_ids(segments)

    if classification == NO_RENDER:
        return RegenerationPlan(
            kind=kind,
            classification=classification,
            notes=["metadata-only change: no audio is regenerated"],
        )

    if classification == REMIX_ONLY:
        touched = [cue.id for cue in cues if _cue_matches(kind, cue)]
        return RegenerationPlan(
            kind=kind,
            classification=classification,
            remix_only=True,
            notes=[
                "mix-only change: every existing Take is reused",
                f"cues touched: {len(touched)}",
            ],
        )

    if kind in PROJECT_WIDE_CHANGE_KINDS:
        return RegenerationPlan(
            kind=kind,
            classification=classification,
            utterance_ids=list(live),
            requires_render=True,
            notes=[
                f"{kind} change is project-wide: all {len(live)} live utterances "
                "must be re-rendered"
            ],
        )

    requested = list(dict.fromkeys(utterance_ids or []))
    unknown = [item for item in requested if item not in set(live)]
    selected = [item for item in requested if item in set(live)]
    notes: List[str] = []
    if unknown:
        notes.append(
            "unknown or retired utterance ids ignored: " + ", ".join(sorted(unknown))
        )
    return RegenerationPlan(
        kind=kind,
        classification=classification,
        utterance_ids=selected,
        requires_render=bool(selected),
        notes=notes,
    )


def _cue_matches(kind: str, cue: AudioCue) -> bool:
    if kind in {"cue_timing", "cue_lane"}:
        return True
    if kind == "music_level":
        return cue.cue_type in {"bgm", "ambience"}
    return False


def utterance_ids_needing_render(
    segments: Sequence[Segment],
    takes_by_segment: Mapping[str, Sequence[object]],
) -> List[str]:
    """Live utterances with no Take at all, ignoring staleness.

    Companion to the fingerprint check: staleness answers "is this Take still
    correct", this answers "is there anything to render at all".
    """

    missing: List[str] = []
    for segment in segments:
        if not _is_live(segment):
            continue
        if not takes_by_segment.get(segment.id):
            missing.append(segment.id)
    return missing


def stale_utterance_ids(
    segments: Sequence[Segment],
    *,
    requirement_for: Callable[[Segment], Optional[object]],
) -> List[str]:
    """Live utterances whose Takes no longer match the live requirement.

    Delegates the comparison to :func:`takes.take_is_stale` so there is exactly
    one staleness rule in the codebase.
    """

    from dubbing_service.app.audiobook import takes as takes_module

    stale: List[str] = []
    for segment in segments:
        if not _is_live(segment):
            continue
        requirement = requirement_for(segment)
        if requirement is None:
            continue
        if any(
            takes_module.take_is_stale(take, requirement)[0]  # type: ignore[arg-type]
            for take in segment.takes
        ):
            stale.append(segment.id)
    return stale
