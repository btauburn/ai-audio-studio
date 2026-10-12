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
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dubbing_service.app.models.project import Project, Role, Segment, SegmentTake
from dubbing_service.app.services.takes import take_is_selectable
from dubbing_service.app.storage.projects import load_project, project_meta_lock
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
    stage_timings: Dict[str, object] = field(default_factory=dict)
    reused: bool = False
    resume_reason: Optional[str] = None
    reaudited: bool = False
    dispatched: bool = False

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

    This is the single funnel through which the Performance Director's delivery
    direction reaches a fingerprint (M4).  ``Project.metadata`` is otherwise
    documented as never participating in a requirement, and reading the active
    direction pointer here is the one deliberate exception to that note; see
    ``performance_director`` and ``docs/performance_director.md``.  With no
    index — which is every project today — the call below reads no performance
    storage at all and the requirement is byte-identical to before M4.
    """

    from .performance_director import PerformanceDirectorError, requirement_direction

    if "performance_direction" in overrides:
        raise PerformanceDirectorError("direction_conflict", "performance_direction is owned by the active direction loader")
    overrides["performance_direction"] = requirement_direction(project, segment)

    if role is None and segment.role_id:
        canonical_role = segment.role_id
        if project.voice_lab.casting.get("productions"):
            from .casting_review import _canonical
            canonical_role = _canonical(project, canonical_role)
        role = next(
            (item for item in project.roles if item.id == canonical_role), None
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

    # Strict casting and its auditions pin effective local model/runtime inputs.
    # Legacy Takes keep their original fingerprint contract; enabling this new
    # workflow never rewrites historical provenance or eagerly regenerates a book.
    if ((project.voice_lab.casting.get("productions") or project.metadata.get("voice_lab_audition", {}).get("script_version"))
            and resolved_engine in {"qwen3tts", "qwen3tts_cv", "qwen3tts_customvoice", "qwen3tts_voicedesign"}
            and "engine_components" not in overrides):
        from .provider_capabilities import synthesis_components
        overrides["engine_components"] = synthesis_components(resolved_engine, role.voice_identity.model_dump(mode="json") if role and role.voice_identity else None)

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

    render_segment_started = time.perf_counter()
    preparation_started = render_segment_started

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

    runtime_binding = renderer.runtime_contract() if hasattr(renderer, "runtime_contract") else None
    if runtime_binding is not None:
        from .runtime_binding import engine_components
        extra_requirement = {**(extra_requirement or {}), "engine_components": engine_components(runtime_binding)}
        if requirement is not None and requirement.runtime_version != runtime_binding["sha256"]:
            raise PipelineError("Supplied synthesis requirement is not bound to the actual worker runtime")
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
    preparation_seconds = time.perf_counter() - preparation_started

    renderer_started = time.perf_counter()
    outcome = renderer_module.reassemble_with_retry(renderer, request)
    renderer_seconds = time.perf_counter() - renderer_started
    if not outcome.success or not outcome.audio_path:
        return RenderRecord(
            segment_id=segment.id,
            take_id=take_id,
            success=False,
            error=outcome.error or "渲染失败",
            stage_timings={
                "schema": "audiobook-pipeline-stage-timings-v1",
                "clock": "time.perf_counter",
                "inclusive": {
                    "render_segment_seconds": round(
                        time.perf_counter() - render_segment_started, 6
                    ),
                    "renderer_seconds": round(renderer_seconds, 6),
                },
                "exclusive": {
                    "preparation_seconds": round(preparation_seconds, 6),
                    "qa_seconds": 0.0,
                },
            },
        )

    audio_path = Path(outcome.audio_path)
    audio_sha = outcome.audio_sha256 or renderer_module.sha256_file(audio_path)
    audio_file = _relative_to_root(audio_path, root)

    take_build_started = time.perf_counter()
    generation_metadata = dict(outcome.generation_metadata)
    if runtime_binding is not None:
        generation_metadata["runtime_binding"] = runtime_binding
    if role and role.voice_identity:
        from .character_casting import performance_for
        generation_metadata["character_voice_identity"] = role.voice_identity.model_dump(mode="json")
        generation_metadata["scene_performance"] = performance_for(segment, role).model_dump(mode="json")
    generation_metadata["renderer_stage_timings"] = dict(outcome.stage_timings)
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
        generation_metadata=generation_metadata,
        language=language,
    )
    if outcome.generation_metadata.get("scene_audio_applied"):
        take.baked_speed = take.playback_speed = segment.playback_speed
        take.baked_volume = take.volume_gain = segment.volume_gain
    takes_module.attach_take(segment, take, slot=slot, promote=False)
    take_build_seconds = time.perf_counter() - take_build_started

    record = RenderRecord(
        segment_id=segment.id,
        take_id=take_id,
        success=True,
        candidate_created=True,
        audio_sha256=audio_sha,
        duration_ms=outcome.duration_ms,
        chunked=outcome.chunked,
    )

    qa_started = time.perf_counter()
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
    qa_seconds = time.perf_counter() - qa_started

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

    record.stage_timings = {
        "schema": "audiobook-pipeline-stage-timings-v1",
        "clock": "time.perf_counter",
        "inclusive": {
            "render_segment_seconds": round(
                time.perf_counter() - render_segment_started, 6
            ),
            "renderer_seconds": round(renderer_seconds, 6),
        },
        "exclusive": {
            "preparation_seconds": round(preparation_seconds, 6),
            "take_build_seconds": round(take_build_seconds, 6),
            "qa_seconds": round(qa_seconds, 6),
        },
    }
    take.generation_metadata["pipeline_precommit_stage_timings"] = dict(
        record.stage_timings
    )

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
        bound = (take.generation_metadata or {}).get("runtime_binding")
        options = {}
        if bound is not None:
            from .runtime_binding import engine_components
            options["engine_components"] = engine_components(bound)
        return requirement_for(
            project,
            segment,
            **options,
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
    resume: bool = False,
    casting_manifest: Optional[dict] = None,
    job_id: Optional[str] = None,
) -> List[RenderRecord]:
    """Render a bounded set of utterances and merge each result into canonical state.

    Each result is merged under a short write lock against state re-read at that
    moment, so an edit that landed while the engine was running survives the
    render's commit instead of being overwritten by it.  Synthesis never holds the
    project lock: only the merge does, and only for the length of one commit.
    """

    project = load_project(project_id)
    metadata = project.metadata or {}
    # An explicit cloned-only book cannot trigger synthetic character design or
    # reuse a preset. This guard runs before any provider/runtime/GPU work.
    from .cloned_voice_policy import policy_mode, role_holds
    clone_mode = policy_mode(project)
    if clone_mode == "cloned_only":
        if getattr(renderer, "backend", None) != "qwen3tts":
            raise PipelineError("Cloned-only synthesis requires the Qwen Base renderer")
        targets = set(utterance_ids)
        requested = {s.role_id for s in load_segments(project_id) if s.id in targets}
        for role_id in requested:
            role = next((r for r in project.roles if r.id == role_id), None)
            holds = role_holds(role, mode=clone_mode)
            if role is None or role.tts_backend != "qwen3tts" or holds:
                raise PipelineError("Cloned-only synthesis requires a recorded Qwen Base clone for every requested role: "
                                    + str(role_id) + "; " + str(holds))
    strict_casting = bool(project.voice_lab.casting.get("enabled")) or (
        (metadata.get("autocast_import") or {}).get("mode") == "autocast"
    )
    audition_child = bool(metadata.get("voice_lab_audition"))
    if casting_manifest is None and strict_casting and not audition_child:
        raise PipelineError(
            "Strict casting requires a current approved cast manifest; use the casting render workflow"
        )
    if (casting_manifest is None and project.character_cast.policy.enabled and project.character_cast.dossiers
            and getattr(renderer, "backend", None) == "qwen3tts"):
        from .character_casting import prepare_cast
        targets = set(utterance_ids)
        role_ids = {project.character_cast.role_aliases.get(s.role_id, s.role_id) for s in load_segments(project_id)
                    if s.id in targets and (s.role_id in project.character_cast.profiles or s.role_id in project.character_cast.role_aliases)}
        ordered_roles = sorted(role_ids)
        batch_size = max(1, project.character_cast.policy.max_audition_characters)
        for start in range(0, len(ordered_roles), batch_size):
            prepare_cast(project_id, provider=renderer._get_provider(), role_ids=ordered_roles[start:start + batch_size])
        project = load_project(project_id)
        missing = [r.name for r in project.roles if r.id in role_ids and r.voice_identity is None]
        if missing:
            raise PipelineError("Character voice preparation needs review: " + ", ".join(missing))
    root = project_root(project_id)
    runtime_binding = renderer.runtime_contract() if hasattr(renderer, "runtime_contract") else None
    if runtime_binding is not None:
        from .runtime_binding import check_scope, bind_job
        scope_ids = casting_manifest["utterance_ids"] if casting_manifest else list(utterance_ids)
        scope = [s for s in load_segments(project_id) if s.id in scope_ids]
        if job_id is not None:
            from dubbing_service.app.storage.jobs import get_job
            job = get_job(project_id, job_id)
            scope = [s for s in load_segments(project_id) if s.id in job.segment_ids]
            bind_job(project_id, job_id, renderer, scope, project.roles, project.character_cast.role_aliases)
        elif resume:
            check_scope(scope, runtime_binding)
    requirement_for = _canonical_requirement_factory(project_id)
    records: List[RenderRecord] = []

    for utterance_id in utterance_ids:
        if casting_manifest is not None:
            project = load_project(project_id)
        utterance_started = time.perf_counter()
        state_read_started = time.perf_counter()
        segments = load_segments(project_id)
        state_read_seconds = time.perf_counter() - state_read_started
        if casting_manifest is not None:
            # A per-line guard over the state just read; re-deriving the whole book here
            # (as this once did, before every line) made a full render quadratic.
            from .casting_review import manifest_still_current
            if not manifest_still_current(project, segments, casting_manifest, utterance_id):
                raise PipelineError("Current human cast no longer matches the frozen production manifest")
        segment = next((item for item in segments if item.id == utterance_id), None)
        if segment is None:
            raise PipelineError(f"utterance not found: {utterance_id}")

        resume_reason: Optional[str] = None
        if resume:
            reuse_record, resume_reason = _resume_selected_take(
                project,
                segment,
                renderer=renderer,
                root=root,
                project_id=project_id,
            )
            if reuse_record is not None:
                records.append(reuse_record)
                continue
            if resume_reason == "unknown_provenance":
                records.append(
                    RenderRecord(
                        segment_id=segment.id,
                        take_id=segment.selected_take_id or "",
                        success=False,
                        error=(
                            "selected Take lacks a provable current synthesis requirement; "
                            "resume will not guess or rewrite legacy provenance; use new-take explicitly"
                        ),
                        resume_reason=resume_reason,
                    )
                )
                continue
            if resume_reason == "reaudit_failed":
                records.append(
                    RenderRecord(
                        segment_id=segment.id,
                        take_id=segment.selected_take_id or "",
                        success=False,
                        error="selected Take could not be re-audited safely; no synthesis was dispatched",
                        resume_reason=resume_reason,
                    )
                )
                continue
            if resume_reason == "selection_changed":
                records.append(
                    RenderRecord(
                        segment_id=segment.id,
                        take_id=segment.selected_take_id or "",
                        success=False,
                        error="canonical selection changed during resume validation; rerun status before rendering",
                        resume_reason=resume_reason,
                    )
                )
                continue
            latest_segments = load_segments(project_id)
            latest = next(
                (item for item in latest_segments if item.id == utterance_id),
                None,
            )
            if latest is None:
                records.append(
                    RenderRecord(
                        segment_id=utterance_id,
                        take_id="",
                        success=False,
                        error="utterance was removed during resume validation",
                        resume_reason="selection_changed",
                    )
                )
                continue
            selection_expected_to_clear = (
                resume_reason == "qa_blocked"
                and segment.selected_take_id is not None
                and latest.selected_take_id is None
            )
            if (
                latest.selected_take_id != segment.selected_take_id
                and not selection_expected_to_clear
            ):
                records.append(
                    RenderRecord(
                        segment_id=utterance_id,
                        take_id=latest.selected_take_id or "",
                        success=False,
                        error="canonical selection changed during resume validation; rerun status before rendering",
                        resume_reason="selection_changed",
                    )
                )
                continue
            segment = latest

        # What canonical state held when this utterance was read.  The merge
        # refuses to move a pointer that no longer holds these values, which is
        # how an explicit newer choice survives a render that started earlier.
        observed_selected = segment.selected_take_id
        observed_slot = takes_module.slot_take_id(segment, slot)

        render_role = None
        if casting_manifest is not None and segment.role_id:
            from .casting_review import _canonical
            canonical_role_id = _canonical(project, segment.role_id)
            render_role = next((role for role in project.roles if role.id == canonical_role_id), None)

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
            role=render_role,
        )
        record.dispatched = True
        if resume:
            record.resume_reason = resume_reason
        if not record.success:
            record.stage_timings["production_path"] = {
                "schema": "audiobook-production-path-stage-timings-v1",
                "clock": "time.perf_counter",
                "inclusive": {
                    "utterance_seconds": round(
                        time.perf_counter() - utterance_started, 6
                    )
                },
                "exclusive": {
                    "canonical_state_read_seconds": round(state_read_seconds, 6),
                    "canonical_merge_lock_seconds": 0.0,
                },
                "merge_skipped": True,
            }
            records.append(record)
            continue

        merge_started = time.perf_counter()
        # Rejection and the pointer promotion share this short project boundary.
        # A finished obsolete waveform remains an immutable unselected Take.
        from .casting_review import cast_current
        with project_meta_lock(project_id) if casting_manifest is not None else nullcontext():
            # Under the project lock only the approval is re-read: it is what can be revoked
            # while a line renders, and the merge judges the Take itself against current state.
            approval_current = casting_manifest is None or cast_current(load_project(project_id), casting_manifest)
            merge = persistence_module.commit_take_result(
                project_id,
                segment_id=utterance_id,
                take=takes_module.find_take(segment, record.take_id),
                requirement_for=requirement_for,
                slot=slot,
                promote=promote and approval_current,
                allow_blocking_qa=bool(not run_qa),
                observed_selected_take_id=observed_selected,
                observed_slot_take_id=observed_slot,
            )
            if not approval_current:
                record.error = "Human casting approval changed; completed audio retained without promotion"
        merge_seconds = time.perf_counter() - merge_started
        record.selected = merge.selected
        record.stale = merge.stale
        record.stale_components = list(merge.stale_components)
        if not merge.selected and merge.reason:
            record.error = record.error or merge.reason
        record.stage_timings["production_path"] = {
            "schema": "audiobook-production-path-stage-timings-v1",
            "clock": "time.perf_counter",
            "inclusive": {
                "utterance_seconds": round(
                    time.perf_counter() - utterance_started, 6
                )
            },
            "exclusive": {
                "canonical_state_read_seconds": round(state_read_seconds, 6),
                "canonical_merge_lock_seconds": round(merge_seconds, 6),
            },
        }
        records.append(record)

    return records


def _resume_selected_take(
    project: Project,
    segment: Segment,
    *,
    renderer: Optional[renderer_module.UtteranceRenderer],
    root: Path,
    project_id: str,
    audit_missing_qa: bool = True,
) -> tuple[Optional[RenderRecord], str]:
    """Reuse only the selected Take after checking current requirements and bytes.

    A status-only caller passes ``renderer=None``; that compares against the
    Take's own recorded engine/renderer policy and never today's global default.
    A render-resume caller supplies its explicit backend, so a different
    requested renderer cannot silently reuse another profile's audio.
    """

    selected_id = segment.selected_take_id
    take = next((item for item in segment.takes if item.id == selected_id), None)
    if take is None:
        return None, "missing_selection"
    if not take.input_fingerprint:
        return None, "unknown_provenance"
    if take.missing or not take.audio_file:
        return None, "missing_audio"
    if take.archived_at is not None:
        return None, "archived_selection"

    role = next((item for item in project.roles if item.id == segment.role_id), None)
    if role is None:
        return None, "missing_role"
    try:
        if renderer is None:
            requirement = _requirement_for_take(project, segment, take)
        else:
            language = language_module.resolve_synthesis_language(segment, role)
            options = {}
            bound = renderer.runtime_contract() if hasattr(renderer, "runtime_contract") else None
            if bound is not None:
                from .runtime_binding import require_take, engine_components
                require_take(take, bound)
                options["engine_components"] = engine_components(bound)
            requirement = requirement_for(
                project,
                segment,
                role=role,
                renderer_id=getattr(renderer, "renderer_id", None),
                engine_id=getattr(renderer, "engine_id", None),
                language=language,
                **options,
            )
    except Exception:
        requirement = None
    if requirement is None:
        return None, "unknown_provenance"
    stale, _changed = takes_module.take_is_stale(take, requirement)
    if stale or take.input_fingerprint != requirement.fingerprint():
        return None, "stale_requirement"

    try:
        audio_path = (root / take.audio_file).resolve()
        audio_path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None, "invalid_audio_path"
    if not audio_path.is_file():
        return None, "missing_audio"
    try:
        actual_sha256 = fingerprint_module.file_sha256(audio_path)
    except OSError:
        return None, "unreadable_audio"
    if not take.audio_sha256:
        return None, "unknown_provenance"
    if actual_sha256 != take.audio_sha256:
        return None, "audio_hash_mismatch"
    facts = qa_module.read_audio_facts(audio_path)
    if facts is None or not facts.decodable or facts.duration_ms <= 0:
        return None, "undecodable_audio"

    report = take.qa_report
    report_identity_current = bool(
        report is not None
        and report.matches(take.id, actual_sha256)
        and report.qa_check_version == qa_module.QA_CHECK_VERSION
        and report.coverage_profile == qa_module.coverage_profile_identity(take.language)
    )
    if report_identity_current and report is not None and report.blocking_failures():
        return None, "qa_blocked"
    qa_current = bool(report_identity_current and report is not None and report.passed)
    if not qa_current:
        if not audit_missing_qa:
            return None, "qa_required"
        try:
            audit = qa_module.reaudit_take(
                take_id=take.id,
                audio_path=audio_path,
                text=take.text or segment.text,
                language=take.language,
                expected_sha256=actual_sha256,
                notes="Resume validation re-audit using the selected Take's stored text and language.",
            )
        except Exception:
            return None, "reaudit_failed"
        committed = persistence_module.commit_selected_take_qa_report(
            project_id,
            segment_id=segment.id,
            take_id=take.id,
            expected_audio_sha256=actual_sha256,
            report=audit,
        )
        if not committed:
            return None, "selection_changed"
        if not audit.passed or audit.blocking_failures():
            return None, "qa_blocked"

        # The selected pointer, synthesis requirement and audio bytes may have
        # changed while QA ran. Re-read before deciding to reuse the Take.
        refreshed = load_segments(project_id)
        latest = next((item for item in refreshed if item.id == segment.id), None)
        if latest is None or latest.selected_take_id != take.id:
            return None, "selection_changed"
        record, reason = _resume_selected_take(
            project,
            latest,
            renderer=renderer,
            root=root,
            project_id=project_id,
        )
        if record is not None:
            record.reaudited = True
        return record, reason

    refusal = takes_module.take_is_selectable(take)
    if refusal is not None:
        return None, refusal[0]
    return (
        RenderRecord(
            segment_id=segment.id,
            take_id=take.id,
            success=True,
            audio_sha256=actual_sha256,
            duration_ms=facts.duration_ms,
            qa_passed=True,
            selected=True,
            reused=True,
            resume_reason="selected_take_valid",
        ),
        "selected_take_valid",
    )


def resume_status(project_id: str, utterance_ids: Sequence[str]) -> Dict[str, object]:
    """Read-only resume eligibility summary for a bounded explicit scope."""

    project = load_project(project_id)
    segments = load_segments(project_id)
    root = project_root(project_id)
    by_id = {segment.id: segment for segment in segments}
    rows: List[Dict[str, object]] = []
    for utterance_id in utterance_ids:
        segment = by_id.get(utterance_id)
        if segment is None:
            rows.append({"segment_id": utterance_id, "state": "missing_utterance"})
            continue
        record, reason = _resume_selected_take(
            project,
            segment,
            renderer=None,
            root=root,
            project_id=project_id,
            audit_missing_qa=False,
        )
        state = "reusable" if record is not None else (
            "audit_required" if reason == "qa_required" else "actionable"
        )
        rows.append(
            {
                "segment_id": segment.id,
                "selected_take_id": segment.selected_take_id,
                "state": state,
                "reason": reason,
                "audio_sha256": record.audio_sha256 if record else None,
            }
        )
    counts = {
        key: sum(row.get("state") == key for row in rows)
        for key in ("reusable", "audit_required", "actionable", "missing_utterance")
    }
    return {"project_id": project_id, "scope_count": len(rows), "counts": counts, "utterances": rows}


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
