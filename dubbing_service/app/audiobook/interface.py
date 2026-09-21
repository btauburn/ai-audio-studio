"""Narrow local capability boundary over the audiobook authority (Phase 16).

This is deliberately a thin facade, not a second project manager.  Every
operation delegates to the module that already owns the behaviour:

``create_request``          -> ``storage.jobs.register_synthesis_request``
``get_status``              -> ``storage.jobs`` + ``audiobook.pipeline.status_report``
``cancel``                  -> ``storage.jobs.cancel_job``
``inspect_failure``         -> job records + per-utterance Take/QA facts
``request_targeted_repair`` -> ``audiobook.regeneration.plan_regeneration``
``select_take`` / ``reject_take`` -> ``audiobook.takes`` + Take slots
``export``                  -> ``audiobook.release.release_report``
``list_artifacts``          -> the project root as it exists on disk

There is no Mission Control coupling and no queue of its own: the persistence
authority remains AI Audio Studio's files plus its API.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from dubbing_service.app.audiobook import pipeline as pipeline_module
from dubbing_service.app.audiobook import regeneration as regeneration_module
from dubbing_service.app.audiobook import release as release_module
from dubbing_service.app.audiobook import takes as takes_module
from dubbing_service.app.models.jobs import SynthesisJob
from dubbing_service.app.models.project import Segment
from dubbing_service.app.storage.jobs import (
    JobStorageError,
    cancel_job,
    get_job,
    get_job_by_request_id,
    load_jobs,
    register_synthesis_request,
    start_job,
    synthesis_intent_hash,
)
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments


class InterfaceError(RuntimeError):
    """Raised when an agent-facing operation cannot be satisfied."""


def _find_take(segment: Segment, take_id: str):
    """Resolve a Take, reporting absence as a domain error rather than a crash."""

    try:
        return takes_module.find_take(segment, take_id)
    except takes_module.PromotionError as exc:
        raise InterfaceError(str(exc)) from exc


@dataclass
class OperationResult:
    ok: bool
    operation: str
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "operation": self.operation, **self.detail}


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def create_request(
    project_id: str,
    *,
    segment_ids: Sequence[str],
    job_type: str = "batch_synthesize",
    job_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    request_id: Optional[str] = None,
    segment_backends: Optional[Dict[str, str]] = None,
    project_name: Optional[str] = None,
) -> OperationResult:
    """Create (or reuse) a synthesis request for a bounded utterance set."""

    ordered = list(dict.fromkeys(segment_ids))
    if not ordered:
        raise InterfaceError("至少需要一个分段 ID")
    existing_ids = {segment.id for segment in load_segments(project_id)}
    missing = [item for item in ordered if item not in existing_ids]
    if missing:
        raise InterfaceError(f"分段不存在: {missing[:5]}")
    backends = {
        key: value
        for key, value in (segment_backends or {}).items()
        if key in set(ordered)
    }
    resolved_job_id = job_id or f"job_{uuid.uuid4().hex[:16]}"
    resolved_request_id = request_id or new_request_id()
    intent_hash = synthesis_intent_hash(
        project_id,
        job_type=job_type,
        segment_ids=ordered,
        segment_backends=backends,
    )
    try:
        job, created = register_synthesis_request(
            project_id,
            job_id=resolved_job_id,
            job_type=job_type,
            segment_ids=ordered,
            segment_backends=backends,
            project_name=project_name,
            request_id=resolved_request_id,
            idempotency_key=idempotency_key,
            intent_hash=intent_hash,
        )
    except (JobStorageError, ValueError) as exc:
        raise InterfaceError(str(exc)) from exc
    return OperationResult(
        ok=True,
        operation="create_request",
        detail={
            "created": created,
            "job": job.model_dump(mode="json"),
            "intent_hash": intent_hash,
            "request_id": job.request_id,
        },
    )


def start(project_id: str, job_id: str) -> OperationResult:
    try:
        job = start_job(project_id, job_id)
    except (JobStorageError, ValueError) as exc:
        raise InterfaceError(str(exc)) from exc
    return OperationResult(
        ok=True, operation="start", detail={"job": job.model_dump(mode="json")}
    )


def get_status(
    project_id: str, *, request_id: Optional[str] = None, job_id: Optional[str] = None
) -> OperationResult:
    """Durable status after a timeout or disconnect."""

    job: Optional[SynthesisJob] = None
    if request_id:
        try:
            job = get_job_by_request_id(project_id, request_id)
        except JobStorageError as exc:
            raise InterfaceError(str(exc)) from exc
        if job is None:
            raise InterfaceError(f"请求不存在: {request_id}")
    elif job_id:
        job = get_job(project_id, job_id)
        if job is None:
            raise InterfaceError(f"任务不存在: {job_id}")
    return OperationResult(
        ok=True,
        operation="get_status",
        detail={
            "job": job.model_dump(mode="json") if job is not None else None,
            "project": pipeline_module.status_report(project_id),
        },
    )


def cancel(project_id: str, job_id: str) -> OperationResult:
    """Cancellation is operational only; fingerprints decide correctness."""

    try:
        job = cancel_job(project_id, job_id)
    except (JobStorageError, ValueError) as exc:
        raise InterfaceError(str(exc)) from exc
    return OperationResult(
        ok=True, operation="cancel", detail={"job": job.model_dump(mode="json")}
    )


def inspect_failure(
    project_id: str, *, request_id: Optional[str] = None, job_id: Optional[str] = None
) -> OperationResult:
    """Report the exact failure surface: job errors plus blocking QA results."""

    if request_id:
        job = get_job_by_request_id(project_id, request_id)
        if job is None:
            raise InterfaceError(f"请求不存在: {request_id}")
    elif job_id:
        job = get_job(project_id, job_id)
        if job is None:
            raise InterfaceError(f"任务不存在: {job_id}")
    else:
        jobs = load_jobs(project_id)
        job = jobs[-1] if jobs else None

    segments = load_segments(project_id)
    failures: List[Dict[str, object]] = []
    for segment in segments:
        for take in segment.takes:
            report = take.qa_report
            if report is None or report.passed:
                continue
            failures.append(
                {
                    "segment_id": segment.id,
                    "take_id": take.id,
                    "blocking": report.blocking_failures(),
                    "tier": report.tier,
                    "qa_check_version": report.qa_check_version,
                }
            )
    return OperationResult(
        ok=True,
        operation="inspect_failure",
        detail={
            "job": job.model_dump(mode="json") if job is not None else None,
            "job_message": (job.message if job is not None else None),
            "failed_segments": dict(job.failed_segments) if job is not None else {},
            "qa_failures": failures,
        },
    )


def request_targeted_repair(
    project_id: str,
    *,
    kind: str,
    segment_ids: Sequence[str] = (),
    reason: str = "",
) -> OperationResult:
    """Plan (never perform) the render scope for one change kind."""

    project = load_project(project_id)
    segments = load_segments(project_id)
    try:
        plan = regeneration_module.plan_regeneration(
            kind,
            project=project,
            segments=segments,
            utterance_ids=list(segment_ids),
            cues=project.audio_cues,
        )
    except KeyError as exc:
        raise InterfaceError(str(exc)) from exc
    detail = plan.as_dict()
    if reason:
        detail["reason"] = reason
    return OperationResult(ok=True, operation="request_targeted_repair", detail=detail)


def _segment(project_id: str, segment_id: str) -> Segment:
    """Look up one utterance in the project's authoritative segment list."""

    for segment in load_segments(project_id):
        if segment.id == segment_id:
            return segment
    raise InterfaceError(f"分段不存在: {segment_id}")


def select_take(
    project_id: str, *, segment_id: str, take_id: str, slot: str = "a"
) -> OperationResult:
    """Select an existing Take; stale Takes are refused."""

    from dubbing_service.app.storage.segments import save_segments

    segments = load_segments(project_id)
    segment = next((item for item in segments if item.id == segment_id), None)
    if segment is None:
        raise InterfaceError(f"分段不存在: {segment_id}")
    take = _find_take(segment, take_id)
    if getattr(take, "stale", False):
        raise InterfaceError(
            f"Take {take_id} 已过期（{', '.join(take.stale_components) or 'unknown'}），"
            "必须先重新生成"
        )
    report = take.qa_report
    if report is not None and report.blocking_failures():
        raise InterfaceError(f"Take {take_id} 未通过阻塞性 QA 检查")
    segment.selected_take_id = take_id
    segment.active_version = takes_module.normalize_slot(slot)
    takes_module.sync_legacy_projection(segment)
    save_segments(project_id, segments)
    return OperationResult(
        ok=True,
        operation="select_take",
        detail={"segment_id": segment_id, "take_id": take_id, "slot": slot},
    )


def reject_take(
    project_id: str, *, segment_id: str, take_id: str, reason: str = ""
) -> OperationResult:
    """Archive a Take as unusable and detach it from every slot.

    ``SegmentTake`` already carries ``archived_at``; rejection reuses it instead
    of adding a parallel status field.
    """

    from datetime import datetime, timezone

    from dubbing_service.app.storage.segments import save_segments

    segments = load_segments(project_id)
    segment = next((item for item in segments if item.id == segment_id), None)
    if segment is None:
        raise InterfaceError(f"分段不存在: {segment_id}")
    take = _find_take(segment, take_id)
    take.archived_at = datetime.now(timezone.utc)
    metadata = dict(take.generation_metadata or {})
    metadata["rejected_reason"] = reason or "rejected_by_reviewer"
    take.generation_metadata = metadata
    if segment.selected_take_id == take_id:
        segment.selected_take_id = None
    for state in segment.take_slots.values():
        if state.take_id == take_id:
            state.take_id = None
            state.needs_synthesis = True
    takes_module.ensure_take_state(segment)
    takes_module.sync_legacy_projection(segment)
    save_segments(project_id, segments)
    return OperationResult(
        ok=True,
        operation="reject_take",
        detail={"segment_id": segment_id, "take_id": take_id, "reason": reason},
    )


def export(project_id: str) -> OperationResult:
    """Return the release coverage manifest (never writes audio)."""

    project = load_project(project_id)
    segments = load_segments(project_id)
    report = release_module.release_report(
        segments, cues=project.audio_cues, project_root=pipeline_module.project_root(project_id)
    )
    return OperationResult(
        ok=True,
        operation="export",
        detail={
            "coverage": report.as_dict(),
            "ready": report.ready,
        },
    )


def list_artifacts(project_id: str) -> OperationResult:
    """List project artifacts on disk with sizes, restricted to the project root."""

    root = pipeline_module.project_root(project_id)
    artifacts: List[Dict[str, object]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            artifacts.append(
                {"path": relative, "bytes": path.stat().st_size}
            )
    return OperationResult(
        ok=True, operation="list_artifacts", detail={"artifacts": artifacts}
    )
