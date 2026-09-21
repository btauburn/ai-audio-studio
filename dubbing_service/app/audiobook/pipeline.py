"""End-to-end audiobook pipeline (Phases 5-11 on top of the studio models).

This is the single code path the CLI and the tests share:

    analyze -> reconcile -> plan -> render -> QA -> Take -> promote -> release

Design rules enforced here, not by convention:

* one committed Take is exactly one authoritative utterance (chunking happens
  inside the renderer and is reassembled before the Take is created);
* a failed synthesis never removes the previously selected Take;
* a stale Take can never be selected;
* every Take carries the input fingerprint of the requirement it satisfied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dubbing_service.app.models.project import Project, Role, Segment, SegmentTake
from dubbing_service.app.services.takes import take_is_selectable
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments

from dubbing_service.app.audiobook import fingerprint as fingerprint_module
from dubbing_service.app.audiobook import language as language_module
from dubbing_service.app.audiobook import persistence as persistence_module
from dubbing_service.app.audiobook import qa as qa_module
from dubbing_service.app.audiobook import release as release_module
from dubbing_service.app.audiobook import renderer as renderer_module
from dubbing_service.app.audiobook import takes as takes_module


class PipelineError(RuntimeError):
    """Raised when the pipeline refuses to proceed."""


@dataclass
class RenderRecord:
    segment_id: str
    take_id: str
    success: bool
    audio_sha256: Optional[str] = None
    duration_ms: int = 0
    chunked: bool = False
    qa_passed: bool = False
    selected: bool = False
    error: Optional[str] = None
    stale_components: List[str] = field(default_factory=list)
    # A candidate that lost an optimistic race is not a failed render: the
    # engine produced audio, and the state refused to promote it.  Callers need
    # both facts, so they are separate fields rather than one error string.
    stale: bool = False
    # A Take that was written but is refused by blocking QA stays visible as a
    # candidate: the refusal is a property of the state, not of this return
    # value, so callers get the exact checks that refused it.
    candidate_created: bool = False
    blocking_qa: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def project_root(project_id: str) -> Path:
    from dubbing_service.app.core.config import safe_project_root

    return safe_project_root(project_id)


def requirement_for(
    project: Project,
    segment: Segment,
    *,
    role: Optional[Role] = None,
    renderer_id: Optional[str] = None,
    engine_id: Optional[str] = None,
    reference_engine_id: Optional[str] = None,
    reference_renderer_id: Optional[str] = None,
    **overrides: object,
) -> fingerprint_module.SynthesisRequirement:
    """Build the live synthesis requirement for one utterance.

    Engine identity is taken from declared intent: an explicit argument wins,
    then ``segment.tts_backend``, then the configured default.  ``reference_*``
    is the last resort and is only used to carry over the renderer identity
    already recorded on an existing Take, so that an utterance which declares no
    engine of its own is not reported as stale purely because some other engine
    is the process-wide default.  Declaring ``segment.tts_backend`` is what makes
    an engine switch detectable.
    """

    if role is None and segment.role_id:
        role = next(
            (item for item in project.roles if item.id == segment.role_id), None
        )
    from dubbing_service.app.audiobook import engines as engines_module

    declared = segment.tts_backend or None
    resolved_engine = (
        engine_id
        or declared
        or reference_engine_id
        or engines_module.default_engine_id()
    )
    resolved_renderer = renderer_id or declared or reference_renderer_id

    return fingerprint_module.requirement_for_segment(
        segment,
        project_id=project.id,
        engine_id=resolved_engine,
        renderer_id=resolved_renderer,
        role=role,
        **overrides,  # type: ignore[arg-type]
    )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        return str(Path(path).resolve())


def render_segment(
    project: Project,
    segment: Segment,
    *,
    renderer: renderer_module.UtteranceRenderer,
    root: Path,
    slot: str = "a",
    promote: bool = False,
    run_qa: bool = True,
    role: Optional[Role] = None,
    allow_missing_role: bool = False,
    requirement: Optional[fingerprint_module.SynthesisRequirement] = None,
    extra_requirement: Optional[Dict[str, object]] = None,
) -> RenderRecord:
    """Render one utterance into a new immutable Take."""

    if role is None:
        role = next(
            (item for item in project.roles if item.id == segment.role_id), None
        )
    if role is None and not allow_missing_role:
        raise PipelineError(f"utterance {segment.id} has no role to render with")

    # Resolved once, here: the requirement, the renderer, QA and the Take all
    # receive this exact value.  Resolving again in any of them is how the
    # renderer and QA came to disagree about what language a Take was made in.
    language = language_module.resolve_synthesis_language(segment, role)

    requirement = requirement or requirement_for(
        project,
        segment,
        role=role,
        renderer_id=getattr(renderer, "renderer_id", None),
        engine_id=getattr(renderer, "engine_id", None),
        language=language,
        **(extra_requirement or {}),
    )
    take_id = renderer_module.new_take_id()
    request = renderer_module.RenderRequest(
        project_id=project.id,
        segment=segment,
        role=role,  # type: ignore[arg-type]
        text=segment.text,
        engine_id=getattr(renderer, "engine_id", "") or "",
        backend=segment.tts_backend or "",
        requirement_fingerprint=requirement.fingerprint(),
        requirement_components=requirement.components(),
        take_id=take_id,
        language=language,
    )

    outcome = renderer_module.reassemble_with_retry(renderer, request)
    if not outcome.success or not outcome.audio_path:
        return RenderRecord(
            segment_id=segment.id,
            take_id=take_id,
            success=False,
            error=outcome.error or "渲染失败",
        )

    audio_path = Path(outcome.audio_path)
    audio_sha = outcome.audio_sha256 or renderer_module.sha256_file(audio_path)
    audio_file = _relative_to_root(audio_path, root)

    take = takes_module.build_take(
        segment,
        slot=slot,
        take_id=take_id,
        audio_file=audio_file,
        audio_sha256=audio_sha,
        requirement=requirement,
        renderer_id=getattr(renderer, "renderer_id", None),
        engine_id=getattr(renderer, "engine_id", None),
        model_id=requirement.model_id,
        spoken_text_revision_id=segment.spoken_text_revision_id,
        source_text_hash=fingerprint_module.text_sha256(segment.text),
        synthesis_text=request.synthesis_text or segment.text,
        chunk_map=outcome.chunk_map(),
        language=language,
    )
    takes_module.attach_take(segment, take, slot=slot, promote=False)

    record = RenderRecord(
        segment_id=segment.id,
        take_id=take_id,
        success=True,
        candidate_created=True,
        audio_sha256=audio_sha,
        duration_ms=outcome.duration_ms,
        chunked=outcome.chunked,
    )

    if run_qa:
        report = qa_module.qa_take_audio(
            take_id=take_id,
            audio_path=audio_path,
            text=segment.text,
            expected_sha256=audio_sha,
            language=language,
        )
        takes_module.attach_qa_report(segment, take_id, report)
        record.qa_passed = report.passed
        record.blocking_qa = [
            item.check_id for item in report.blocking_failures()
        ]

    if promote:
        try:
            takes_module.promote_take(
                segment,
                take_id,
                requirement=requirement,
                allow_blocking_qa=bool(not run_qa),
            )
            record.selected = True
        except takes_module.PromotionError as exc:
            record.error = str(exc)
            record.stale_components = list(take.stale_components)

    return record


def _requirement_for_take(
    project: Project,
    segment: Segment,
    take: SegmentTake,
    *,
    renderer_id: Optional[str] = None,
) -> Optional[fingerprint_module.SynthesisRequirement]:
    """The live requirement of one Take — the single staleness rule.

    An utterance with no declared backend keeps the engine identity recorded on
    each Take, so re-running this cannot invent staleness on its own; only a real
    content or declared-configuration change does.  ``None`` means the engine is
    unknown, which is not a staleness claim.
    """

    try:
        return requirement_for(
            project,
            segment,
            renderer_id=renderer_id,
            reference_engine_id=getattr(take, "engine_id", None),
            reference_renderer_id=getattr(take, "renderer_id", None),
        )
    except Exception:  # noqa: BLE001 - unknown engine is not a staleness claim
        return None


def _canonical_requirement_factory(
    project_id: str,
) -> persistence_module.RequirementFactory:
    """A per-Take staleness rule that reads the project from disk on first use.

    The merge calls it inside its write lock, so a worker's Take is judged against
    the state being committed — including role voice configuration, which is
    project-level and therefore not visible in the segment the worker rendered.
    """

    loaded: List[Project] = []

    def requirement_for_take(
        segment: Segment, take: SegmentTake
    ) -> Optional[fingerprint_module.SynthesisRequirement]:
        if not loaded:
            loaded.append(load_project(project_id))
        return _requirement_for_take(loaded[0], segment, take)

    return requirement_for_take


def refresh_staleness(
    project: Project, segments: Sequence[Segment], *, renderer_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Recompute and record staleness for every Take, without touching audio."""

    return takes_module.mark_stale_for_requirement(
        segments,
        lambda segment, take: _requirement_for_take(
            project, segment, take, renderer_id=renderer_id
        ),
    )


def render_utterances(
    project_id: str,
    utterance_ids: Sequence[str],
    *,
    renderer: renderer_module.UtteranceRenderer,
    slot: str = "a",
    promote: bool = True,
    run_qa: bool = True,
) -> List[RenderRecord]:
    """Render a bounded set of utterances and merge each result into canonical state.

    Each result is merged under a short write lock against state re-read at that
    moment, so an edit that landed while the engine was running survives the
    render's commit instead of being overwritten by it.  Synthesis never holds the
    project lock: only the merge does, and only for the length of one commit.
    """

    project = load_project(project_id)
    root = project_root(project_id)
    requirement_for = _canonical_requirement_factory(project_id)
    records: List[RenderRecord] = []

    for utterance_id in utterance_ids:
        segments = load_segments(project_id)
        segment = next((item for item in segments if item.id == utterance_id), None)
        if segment is None:
            raise PipelineError(f"utterance not found: {utterance_id}")

        # What canonical state held when this utterance was read.  The merge
        # refuses to move a pointer that no longer holds these values, which is
        # how an explicit newer choice survives a render that started earlier.
        observed_selected = segment.selected_take_id
        observed_slot = takes_module.slot_take_id(segment, slot)

        # Promotion is decided at the merge, not here: the render pass only
        # produces the artifact, and the state decides whether it may be claimed.
        record = render_segment(
            project,
            segment,
            renderer=renderer,
            root=root,
            slot=slot,
            promote=False,
            run_qa=run_qa,
        )
        if not record.success:
            records.append(record)
            continue

        merge = persistence_module.commit_take_result(
            project_id,
            segment_id=utterance_id,
            take=takes_module.find_take(segment, record.take_id),
            requirement_for=requirement_for,
            slot=slot,
            promote=promote,
            allow_blocking_qa=bool(not run_qa),
            observed_selected_take_id=observed_selected,
            observed_slot_take_id=observed_slot,
        )
        record.selected = merge.selected
        record.stale = merge.stale
        record.stale_components = list(merge.stale_components)
        if not merge.selected and merge.reason:
            record.error = record.error or merge.reason
        records.append(record)

    return records


def release_report(project_id: str) -> release_module.CoverageReport:
    project = load_project(project_id)
    segments = load_segments(project_id)
    return release_module.release_report(
        segments,
        cues=project.audio_cues,
        project_root=project_root(project_id),
    )


def blocked_utterances(segments: Sequence[Segment]) -> List[Dict[str, object]]:
    """Live utterances that blocking QA left without a usable selection.

    An utterance that already has a selectable Take is *not* blocked: the
    refused candidate stays in its slot as history.  Derived from the committed
    Takes rather than from one command's return value, because the refusal is
    durable state that has to survive a restart.
    """

    blocked: List[Dict[str, object]] = []
    for segment in segments:
        if not release_module.is_live(segment):
            continue
        selection = next(
            (take for take in segment.takes if take.id == segment.selected_take_id),
            None,
        )
        if selection is not None and take_is_selectable(selection) is None:
            continue
        refused: Dict[str, List[str]] = {}
        for take in segment.takes:
            report = take.qa_report
            if report is None:
                continue
            failures = [item.check_id for item in report.blocking_failures()]
            if failures:
                refused[take.id] = sorted(failures)
        if not refused:
            continue
        checks = sorted({check for ids in refused.values() for check in ids})
        blocked.append(
            {
                "segment_id": segment.id,
                "state": "qa_blocked",
                "selected_take_id": segment.selected_take_id,
                "candidate_take_ids": sorted(refused),
                "blocking_qa": {
                    "checks": checks,
                    "takes": {take_id: ids for take_id, ids in sorted(refused.items())},
                    "qa_check_version": next(
                        (
                            take.qa_report.qa_check_version
                            for take in segment.takes
                            if take.id in refused and take.qa_report is not None
                        ),
                        None,
                    ),
                },
            }
        )
    return blocked


def reaudit_take(
    project_id: str, segment_id: str, take_id: str
) -> qa_module.TakeQaReport:
    """Re-run Tier-1 QA on a stored Take from that Take's own evidence.

    The language is read from the Take, never from the role or segment: either may
    have been edited after the audio was generated, and letting current state
    decide how history is interpreted is exactly the defect this closes.  A Take
    with no recorded language re-audits under the same fallback policy it was
    created with, and the report says so rather than inventing a language.
    """

    segments = load_segments(project_id)
    segment = next((item for item in segments if item.id == segment_id), None)
    if segment is None:
        raise PipelineError(f"utterance not found: {segment_id}")
    take = takes_module.find_take(segment, take_id)

    notes = None
    if take.language is None:
        notes = (
            "这次 Take 没有记录语言来源（旧 Take 或当时没有任何声明），"
            "按既有回退策略复核，未推断语言"
        )
    return qa_module.reaudit_take(
        take_id=take.id,
        audio_path=Path(project_root(project_id)) / take.audio_file,
        text=take.text or segment.text,
        language=take.language,
        expected_sha256=take.audio_sha256,
        notes=notes,
    )


def language_evidence_report(project_id: str) -> Dict[str, object]:
    """What every stored Take can prove about its own language, from itself.

    Replaces the earlier corpus re-audit, which had to assume a language from the
    role or the default.  Nothing here reads a role: ``persisted_language`` is the
    Take's own snapshot, and a Take that has none says so explicitly instead of
    inheriting today's setting.
    """

    segments = load_segments(project_id)
    root = Path(project_root(project_id))
    rows: List[Dict[str, object]] = []
    for segment in segments:
        for take in segment.takes:
            profile = take.qa_report.coverage_profile if take.qa_report else None
            facts = qa_module.read_audio_facts(root / take.audio_file)
            if take.language:
                recorded = take.language
            elif take.input_fingerprint:
                recorded = "undeclared"
            else:
                recorded = "legacy_unknown"
            rows.append(
                {
                    "take_id": take.id,
                    "utterance_id": segment.id,
                    "persisted_language": recorded,
                    "coverage_profile": (
                        None
                        if profile is None
                        else {
                            "language": profile.language,
                            "profile": profile.profile,
                            "profile_version": profile.profile_version,
                        }
                    ),
                    "chars": len((take.text or "").strip()),
                    "duration_ms": facts.duration_ms if facts else 0,
                    "qa_passed": None if take.qa_report is None else take.qa_report.passed,
                    "selected": segment.selected_take_id == take.id,
                    "stale": take.stale,
                }
            )

    def _count(value: str) -> int:
        return sum(1 for row in rows if row["persisted_language"] == value)

    return {
        "project_id": project_id,
        "takes": rows,
        "counts": {
            "takes": len(rows),
            "with_language": len(rows) - _count("undeclared") - _count("legacy_unknown"),
            "undeclared": _count("undeclared"),
            "legacy_unknown": _count("legacy_unknown"),
        },
    }


def status_report(project_id: str) -> Dict[str, object]:
    """One object with everything a caller needs to see where a book stands."""

    project = load_project(project_id)
    segments = load_segments(project_id)
    stale = refresh_staleness(project, segments)
    live = [segment for segment in segments if release_module.is_live(segment)]
    report = release_module.release_report(
        segments,
        cues=project.audio_cues,
        project_root=project_root(project_id),
    )
    blocked = blocked_utterances(segments)
    return {
        "project_id": project.id,
        "name": project.name,
        "chapters": len(project.chapters),
        "roles": len(project.roles),
        "utterances": len(segments),
        "live_utterances": len(live),
        "takes": sum(len(segment.takes) for segment in segments),
        "stale_utterances": sorted(stale),
        "coverage": report.as_dict(),
        "release_ready": report.ready,
        "qa_blocked": len(blocked),
        "blocked_utterances": blocked,
    }


def write_report(path: Path | str, payload: Dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
