"""Release readiness gates (Phase 11 coverage, Phase 15 cue readiness).

The studio refuses to *claim* a release is ready when the evidence is missing.
Two rules matter here:

* every live, in-scope utterance must have a selected Take whose audio file
  exists, decodes, and matches its recorded hash;
* a Take whose QA never ran is not a passed Take — see
  :meth:`TakeQaReport.executed_check_ids`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from dubbing_service.app.models.audiobook import AudioCue, TakeQaReport
from dubbing_service.app.models.project import Segment

from dubbing_service.app.audiobook import renderer as renderer_module
from dubbing_service.app.audiobook import takes as takes_module

RESOLUTION_MISSING_AUDIO = "missing_audio"
RESOLUTION_HASH_MISMATCH = "hash_mismatch"
RESOLUTION_NO_TAKE = "no_selected_take"
RESOLUTION_BLOCKING_QA = "blocking_qa"
RESOLUTION_STALE = "stale_take"
RESOLUTION_ZERO_LENGTH = "zero_length_text"


@dataclass
class CoverageIssue:
    segment_id: str
    code: str
    detail: str
    take_id: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "code": self.code,
            "detail": self.detail,
            "take_id": self.take_id,
        }


@dataclass
class CoverageReport:
    total_live: int
    covered: int
    issues: List[CoverageIssue] = field(default_factory=list)
    cue_issues: List[CoverageIssue] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.issues and not self.cue_issues

    @property
    def missing(self) -> int:
        return len(self.issues)

    def as_dict(self) -> Dict[str, object]:
        return {
            "ready": self.ready,
            "total_live": self.total_live,
            "covered": self.covered,
            "missing": self.missing,
            "issues": [issue.as_dict() for issue in self.issues],
            "cue_issues": [issue.as_dict() for issue in self.cue_issues],
        }


def is_live(segment: Segment) -> bool:
    lineage = segment.lineage
    return lineage is None or lineage.retired_at is None


def audio_path_for(segment: Segment, take: object, project_root: Optional[Path]) -> Optional[Path]:
    audio_file = getattr(take, "audio_file", None)
    if not audio_file:
        return None
    path = Path(audio_file)
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def check_coverage(
    segments: Sequence[Segment],
    *,
    project_root: Optional[Path] = None,
    verify_bytes: bool = True,
    require_qa: bool = True,
) -> CoverageReport:
    """Verify every live utterance has a usable selected Take."""

    issues: List[CoverageIssue] = []
    live_segments = [segment for segment in segments if is_live(segment)]
    covered = 0

    for segment in live_segments:
        if not segment.text.strip():
            issues.append(
                CoverageIssue(
                    segment_id=segment.id,
                    code=RESOLUTION_ZERO_LENGTH,
                    detail="live utterance has no text to speak",
                )
            )
            continue

        take = takes_module.selected_take(segment)
        if take is None:
            issues.append(
                CoverageIssue(
                    segment_id=segment.id,
                    code=RESOLUTION_NO_TAKE,
                    detail="no selected Take for a live utterance",
                )
            )
            continue

        path = audio_path_for(segment, take, project_root)
        if path is None or not path.is_file():
            issues.append(
                CoverageIssue(
                    segment_id=segment.id,
                    code=RESOLUTION_MISSING_AUDIO,
                    detail=f"selected Take audio file is missing: {take.audio_file}",
                    take_id=take.id,
                )
            )
            continue

        if verify_bytes and take.audio_sha256:
            digest = renderer_module.sha256_file(path)
            if digest != take.audio_sha256:
                issues.append(
                    CoverageIssue(
                        segment_id=segment.id,
                        code=RESOLUTION_HASH_MISMATCH,
                        detail=(
                            "selected Take audio hash does not match the recorded "
                            f"hash ({take.audio_sha256[:12]}… != {digest[:12]}…)"
                        ),
                        take_id=take.id,
                    )
                )
                continue

        if take.stale:
            issues.append(
                CoverageIssue(
                    segment_id=segment.id,
                    code=RESOLUTION_STALE,
                    detail=(
                        "selected Take is stale ("
                        + (", ".join(take.stale_components) or "unknown")
                        + ")"
                    ),
                    take_id=take.id,
                )
            )
            continue

        if require_qa:
            report = take.qa_report
            if report is None:
                detail = "no QA report bound to this Take"
            elif not report.matches(take.id, take.audio_sha256):
                detail = "QA report is bound to different bytes than this Take"
            elif not _qa_passed(report):
                failures = [failure.check_id for failure in report.blocking_failures()]
                detail = (
                    "QA blocking failures: " + ", ".join(failures)
                    if failures
                    else "QA produced no executed checks"
                )
            else:
                detail = ""
            if detail:
                issues.append(
                    CoverageIssue(
                        segment_id=segment.id,
                        code=RESOLUTION_BLOCKING_QA,
                        detail=detail,
                        take_id=take.id,
                    )
                )
                continue

        covered += 1

    return CoverageReport(total_live=len(live_segments), covered=covered, issues=issues)


def _qa_passed(report: TakeQaReport) -> bool:
    """A QA report counts as a pass only when it ran something and failed nothing."""

    if not report.passed:
        return False
    if report.blocking_failures():
        return False
    return bool(report.executed_check_ids())


def check_cues(
    cues: Iterable[AudioCue], segments: Sequence[Segment]
) -> List[CoverageIssue]:
    """Every cue must point at a live utterance; no timeline-only anchoring."""

    live = {segment.id: segment for segment in segments if is_live(segment)}
    issues: List[CoverageIssue] = []
    for cue in cues:
        if cue.utterance_id not in live:
            issues.append(
                CoverageIssue(
                    segment_id=cue.utterance_id,
                    code="orphan_cue",
                    detail=f"cue {cue.id} is anchored to a missing or retired utterance",
                )
            )
            continue
        if cue.relative_end_ms is not None and cue.relative_end_ms < cue.relative_offset_ms:
            issues.append(
                CoverageIssue(
                    segment_id=cue.utterance_id,
                    code="invalid_cue_window",
                    detail=f"cue {cue.id} ends before it starts",
                )
            )
    return issues


def release_report(
    segments: Sequence[Segment],
    *,
    cues: Sequence[AudioCue] = (),
    project_root: Optional[Path] = None,
    verify_bytes: bool = True,
    require_qa: bool = True,
) -> CoverageReport:
    """Coverage plus cue readiness, for the release gate."""

    report = check_coverage(
        segments,
        project_root=project_root,
        verify_bytes=verify_bytes,
        require_qa=require_qa,
    )
    report.cue_issues = check_cues(cues, segments)
    return report


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
