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

from dubbing_service.app.models.project import Project, Role, Segment
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments

from dubbing_service.app.audiobook import fingerprint as fingerprint_module
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

    requirement = requirement or requirement_for(
        project,
        segment,
        role=role,
        renderer_id=getattr(renderer, "renderer_id", None),
        engine_id=getattr(renderer, "engine_id", None),
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
    )
    takes_module.attach_take(segment, take, slot=slot, promote=False)

    record = RenderRecord(
        segment_id=segment.id,
        take_id=take_id,
        success=True,
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
        )
        takes_module.attach_qa_report(segment, take_id, report)
        record.qa_passed = report.passed

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


def refresh_staleness(
    project: Project, segments: Sequence[Segment], *, renderer_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Recompute and record staleness for every Take, without touching audio.

    An utterance with no declared backend keeps the engine identity recorded on
    each Take, so re-running this pass cannot invent staleness on its own; only a
    real content or declared-configuration change does.
    """

    def requirement_for_take(segment: Segment, take: object):
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

    return takes_module.mark_stale_for_requirement(segments, requirement_for_take)


def render_utterances(
    project_id: str,
    utterance_ids: Sequence[str],
    *,
    renderer: renderer_module.UtteranceRenderer,
    slot: str = "a",
    promote: bool = True,
    run_qa: bool = True,
) -> List[RenderRecord]:
    """Render a bounded set of utterances and persist the result."""

    project = load_project(project_id)
    segments = load_segments(project_id)
    root = project_root(project_id)
    records: List[RenderRecord] = []

    for utterance_id in utterance_ids:
        segment = next((item for item in segments if item.id == utterance_id), None)
        if segment is None:
            raise PipelineError(f"utterance not found: {utterance_id}")
        records.append(
            render_segment(
                project,
                segment,
                renderer=renderer,
                root=root,
                slot=slot,
                promote=promote,
                run_qa=run_qa,
            )
        )

    refresh_staleness(project, segments)
    persistence_module.commit_project_state(
        project_id, project=project, segments=segments
    )
    return records


def release_report(project_id: str) -> release_module.CoverageReport:
    project = load_project(project_id)
    segments = load_segments(project_id)
    return release_module.release_report(
        segments,
        cues=project.audio_cues,
        project_root=project_root(project_id),
    )


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
    }


def write_report(path: Path | str, payload: Dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
