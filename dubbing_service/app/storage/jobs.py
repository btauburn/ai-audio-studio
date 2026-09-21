"""Atomic project-local storage for durable synthesis jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Generator, Iterable, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.jobs import (
    SynthesisJob,
    SynthesisJobStatus,
    TERMINAL_JOB_STATUSES,
)


class JobStorageError(Exception):
    pass


_PROJECT_JOB_LOCKS: Dict[str, Lock] = {}
_LOCKS_LOCK = Lock()
_ACTIVE_JOB_IDS: set[tuple[str, str]] = set()
_ACTIVE_LOCK = Lock()


def _get_lock(project_id: str) -> Lock:
    with _LOCKS_LOCK:
        return _PROJECT_JOB_LOCKS.setdefault(project_id, Lock())


@contextmanager
def project_jobs_lock(project_id: str) -> Generator[None, None, None]:
    lock = _get_lock(project_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def mark_job_active(project_id: str, job_id: str, active: bool) -> None:
    key = (project_id, job_id)
    with _ACTIVE_LOCK:
        if active:
            _ACTIVE_JOB_IDS.add(key)
        else:
            _ACTIVE_JOB_IDS.discard(key)


def is_job_active(project_id: str, job_id: str) -> bool:
    with _ACTIVE_LOCK:
        return (project_id, job_id) in _ACTIVE_JOB_IDS


def _jobs_path(project_id: str) -> str:
    try:
        root = safe_project_root(project_id)
    except ValueError as exc:
        raise JobStorageError("非法项目 ID") from exc
    if not (root / "project.json").is_file():
        raise JobStorageError("项目不存在")
    return str(root / "jobs.json")


def load_jobs(project_id: str) -> List[SynthesisJob]:
    path = _jobs_path(project_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise JobStorageError(f"jobs.json 无法读取: {exc}") from exc
    if not isinstance(raw, list):
        raise JobStorageError("jobs.json 应为数组")
    jobs = []
    for item in raw:
        try:
            job = SynthesisJob(**item)
            validate_simple_component_id(job.id, label="job_id")
        except (ValidationError, ValueError) as exc:
            raise JobStorageError("jobs.json 包含非法任务") from exc
        jobs.append(job)
    jobs.sort(key=lambda item: item.created_at, reverse=True)
    return jobs


def save_jobs(project_id: str, jobs: Iterable[SynthesisJob]) -> None:
    path = _jobs_path(project_id)
    parent = os.path.dirname(path)
    data = [
        item.model_dump(
            mode="json",
            exclude={"total", "done", "failed", "remaining_segment_ids"},
        )
        for item in jobs
    ]
    fd, temp_path = tempfile.mkstemp(dir=parent, prefix="jobs_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def upsert_job(project_id: str, job: SynthesisJob) -> SynthesisJob:
    validate_simple_component_id(job.id, label="job_id")
    if job.project_id != project_id:
        raise JobStorageError("任务 project_id 不匹配")
    with project_jobs_lock(project_id):
        jobs = load_jobs(project_id)
        index = next((idx for idx, item in enumerate(jobs) if item.id == job.id), None)
        if index is None:
            jobs.insert(0, job)
        else:
            existing = jobs[index]
            if existing.status in TERMINAL_JOB_STATUSES:
                return existing
            job.created_at = existing.created_at
            jobs[index] = job
        save_jobs(project_id, jobs[:100])
    return job


def get_job(project_id: str, job_id: str) -> SynthesisJob | None:
    return next((item for item in load_jobs(project_id) if item.id == job_id), None)


def get_job_by_request_id(project_id: str, request_id: str) -> SynthesisJob | None:
    """Durable ``status(request_id)`` lookup for uncertain agent submissions."""

    return next(
        (item for item in load_jobs(project_id) if item.request_id == request_id),
        None,
    )


def get_job_by_idempotency_key(
    project_id: str, idempotency_key: str
) -> SynthesisJob | None:
    return next(
        (
            item
            for item in load_jobs(project_id)
            if item.idempotency_key == idempotency_key
        ),
        None,
    )


def synthesis_intent_hash(
    project_id: str,
    *,
    job_type: str,
    segment_ids: Sequence[str],
    segment_backends: Optional[Mapping[str, str]] = None,
) -> str:
    """Canonical hash of a synthesis intent (what would be synthesized).

    Segment order is not part of the intent: the same set of utterances, in the
    same order *by user intent*, describes the same GPU work.
    """

    payload = {
        "project_id": project_id,
        "job_type": job_type,
        "segment_ids": list(segment_ids),
        "segment_backends": {
            key: value
            for key, value in sorted((segment_backends or {}).items())
            if key in set(segment_ids)
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def register_synthesis_request(
    project_id: str,
    *,
    job_id: str,
    job_type: str,
    segment_ids: Iterable[str],
    segment_backends: Dict[str, str] | None = None,
    project_name: str | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    intent_hash: str | None = None,
) -> tuple[SynthesisJob, bool]:
    """Create a job, or return the existing one for a repeated request.

    Returns ``(job, created)``.  A repeated ``idempotency_key`` with the same
    ``intent_hash`` never creates duplicate GPU work; reusing a key for a
    different intent is rejected instead of silently coalescing two requests.
    Cancellation is deliberately not consulted here: it is operational, while
    request identity is correctness.
    """

    with project_jobs_lock(project_id):
        if idempotency_key:
            existing = get_job_by_idempotency_key(project_id, idempotency_key)
            if existing is not None:
                if intent_hash and existing.intent_hash and existing.intent_hash != intent_hash:
                    raise JobStorageError(
                        "idempotency_key 已用于不同的合成意图；请使用新的 key"
                    )
                return existing, False

        job = SynthesisJob(
            id=job_id,
            project_id=project_id,
            job_type=job_type,
            project_name=project_name,
            segment_ids=list(segment_ids),
            segment_backends=dict(segment_backends or {}),
            request_id=request_id or uuid.uuid4().hex,
            idempotency_key=idempotency_key,
            intent_hash=intent_hash,
        )
        jobs = load_jobs(project_id)
        jobs.insert(0, job)
        save_jobs(project_id, jobs[:100])
    return job, True


def mutate_job(project_id: str, job_id: str, mutator) -> SynthesisJob:
    with project_jobs_lock(project_id):
        jobs = load_jobs(project_id)
        job = next((item for item in jobs if item.id == job_id), None)
        if job is None:
            raise JobStorageError("任务不存在")
        mutator(job)
        job.updated_at = datetime.now(timezone.utc)
        save_jobs(project_id, jobs)
        return job


def start_job(project_id: str, job_id: str) -> SynthesisJob:
    def apply(job: SynthesisJob) -> None:
        if job.status in TERMINAL_JOB_STATUSES:
            raise JobStorageError("任务已经结束")
        job.status = SynthesisJobStatus.RUNNING
        job.stop_requested = False
        job.started_at = job.started_at or datetime.now(timezone.utc)
        job.finished_at = None

    return mutate_job(project_id, job_id, apply)


def record_job_segment(
    project_id: str,
    job_id: str,
    segment_id: str,
    *,
    success: bool,
    error: str | None = None,
) -> SynthesisJob:
    def apply(job: SynthesisJob) -> None:
        if segment_id not in job.segment_ids:
            raise JobStorageError("分段不属于该任务")
        if success:
            if segment_id not in job.completed_segment_ids:
                job.completed_segment_ids.append(segment_id)
            job.failed_segments.pop(segment_id, None)
        else:
            job.completed_segment_ids = [
                item for item in job.completed_segment_ids if item != segment_id
            ]
            job.failed_segments[segment_id] = error or "合成失败"
        if job.done + job.failed >= job.total:
            job.status = (
                SynthesisJobStatus.SUCCEEDED
                if job.failed == 0
                else SynthesisJobStatus.SUCCEEDED_WITH_ERRORS
            )
            job.finished_at = datetime.now(timezone.utc)

    return mutate_job(project_id, job_id, apply)


def interrupt_job(project_id: str, job_id: str, message: str | None = None) -> SynthesisJob:
    def apply(job: SynthesisJob) -> None:
        if job.status not in TERMINAL_JOB_STATUSES:
            job.status = SynthesisJobStatus.INTERRUPTED
            job.message = message or job.message
            job.finished_at = None

    return mutate_job(project_id, job_id, apply)


def cancel_job(project_id: str, job_id: str) -> SynthesisJob:
    def apply(job: SynthesisJob) -> None:
        if job.status not in TERMINAL_JOB_STATUSES:
            job.status = SynthesisJobStatus.CANCELLED
            job.stop_requested = True
            job.finished_at = datetime.now(timezone.utc)

    return mutate_job(project_id, job_id, apply)


def resume_job(project_id: str, job_id: str) -> SynthesisJob:
    def apply(job: SynthesisJob) -> None:
        if job.status != SynthesisJobStatus.INTERRUPTED:
            raise JobStorageError("只有中断任务可以继续")
        job.status = SynthesisJobStatus.QUEUED
        job.stop_requested = False
        job.message = None

    return mutate_job(project_id, job_id, apply)


def recover_interrupted_jobs(project_id: str) -> int:
    with project_jobs_lock(project_id):
        jobs = load_jobs(project_id)
        changed = 0
        for job in jobs:
            if job.status == SynthesisJobStatus.RUNNING and not is_job_active(project_id, job.id):
                job.status = SynthesisJobStatus.INTERRUPTED
                job.message = "平台或浏览器在任务完成前退出"
                job.updated_at = datetime.now(timezone.utc)
                changed += 1
        if changed:
            save_jobs(project_id, jobs)
        return changed


def delete_job(project_id: str, job_id: str) -> bool:
    with project_jobs_lock(project_id):
        jobs = load_jobs(project_id)
        job = next((item for item in jobs if item.id == job_id), None)
        if job is None:
            return False
        if job.status in {SynthesisJobStatus.RUNNING, SynthesisJobStatus.INTERRUPTED}:
            raise JobStorageError("运行中或中断任务不能直接删除")
        jobs = [item for item in jobs if item.id != job_id]
        save_jobs(project_id, jobs)
        return True
