"""Atomic project-local storage for sound generation jobs."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Iterator

from pydantic import ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.sound_generation import SoundCandidateStatus
from dubbing_service.app.models.sound_jobs import SoundGenerationJob, SoundGenerationJobStatus, utc_now
from dubbing_service.app.storage.production import _atomic_json
from dubbing_service.app.storage.sound_generation import (
    candidate_audio_path,
    load_sound_plan,
    save_sound_plan,
    sound_generation_root,
    sound_plan_lock,
)


class SoundJobStorageError(RuntimeError):
    pass


_JOB_LOCKS: dict[tuple[str, str], RLock] = {}
_JOB_LOCKS_GUARD = Lock()


@contextmanager
def sound_job_lock(project_id: str, job_id: str) -> Iterator[None]:
    """Serialize job state transitions, including browser cancellation."""

    key = (project_id, job_id)
    with _JOB_LOCKS_GUARD:
        lock = _JOB_LOCKS.setdefault(key, RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def generate_sound_job_id() -> str:
    return f"sfxjob-{uuid.uuid4().hex[:12]}"


def _jobs_dir(project_id: str, *, ensure: bool = False) -> Path:
    project_scope = safe_project_root(project_id).resolve()
    generation_root = sound_generation_root(project_id, ensure=ensure).resolve()
    expected_generation_root = project_scope / "production" / "sound_generation"
    if generation_root != expected_generation_root:
        raise SoundJobStorageError("音效生成目录逃逸项目范围")
    path = (generation_root / "jobs").resolve()
    if path != generation_root / "jobs":
        raise SoundJobStorageError("音效任务目录逃逸项目范围")
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve() != generation_root / "jobs":
            raise SoundJobStorageError("音效任务目录逃逸项目范围")
    return path


def _job_path(project_id: str, job_id: str, *, ensure: bool = False) -> Path:
    try:
        safe = validate_simple_component_id(job_id, label="sound generation job id")
    except ValueError as exc:
        raise SoundJobStorageError("非法音效生成任务 ID") from exc
    jobs_root = _jobs_dir(project_id, ensure=ensure)
    expected_target = jobs_root / f"{safe}.json"
    target = expected_target.resolve()
    if target != expected_target:
        raise SoundJobStorageError("音效任务路径逃逸任务目录")
    return target


def save_sound_job(job: SoundGenerationJob) -> None:
    with sound_job_lock(job.project_id, job.id):
        try:
            path = _job_path(job.project_id, job.id, ensure=True)
            # Cancellation is monotonic. A worker may have loaded an earlier
            # snapshot before the browser requested cancellation; never let
            # that stale save silently clear the request.
            if path.is_file():
                existing = SoundGenerationJob.model_validate_json(path.read_text(encoding="utf-8"))
                if existing.cancel_requested:
                    job.cancel_requested = True
                    if job.status in {
                        SoundGenerationJobStatus.QUEUED,
                        SoundGenerationJobStatus.RUNNING,
                    }:
                        job.status = SoundGenerationJobStatus.CANCELLING
            _atomic_json(path, job.model_dump(mode="json"))
        except Exception as exc:
            if isinstance(exc, SoundJobStorageError):
                raise
            raise SoundJobStorageError(f"保存音效生成任务失败: {exc}") from exc


def delete_sound_job(project_id: str, job_id: str) -> bool:
    """Delete one job record, primarily for a failed multi-file prepare transaction."""

    with sound_job_lock(project_id, job_id):
        path = _job_path(project_id, job_id)
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError as exc:
            raise SoundJobStorageError(f"删除音效生成任务失败: {exc}") from exc
        return True


def sound_job_exists(project_id: str, job_id: str) -> bool:
    return _job_path(project_id, job_id).is_file()


def load_sound_job(project_id: str, job_id: str) -> SoundGenerationJob:
    path = _job_path(project_id, job_id)
    if not path.is_file():
        raise SoundJobStorageError("音效生成任务不存在")
    try:
        job = SoundGenerationJob.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise SoundJobStorageError(f"音效生成任务损坏: {exc}") from exc
    if job.project_id != project_id or job.id != job_id:
        raise SoundJobStorageError("音效生成任务归属或文件名不一致")
    return job


def list_sound_jobs(project_id: str) -> list[SoundGenerationJob]:
    path = _jobs_dir(project_id)
    if not path.is_dir():
        return []
    jobs: list[SoundGenerationJob] = []
    for item in path.glob("*.json"):
        try:
            job = SoundGenerationJob.model_validate_json(item.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise SoundJobStorageError(f"音效生成任务损坏 ({item.name}): {exc}") from exc
        if job.project_id != project_id or job.id != item.stem:
            raise SoundJobStorageError(f"音效生成任务归属或文件名不一致 ({item.name})")
        jobs.append(job)
    return sorted(jobs, key=lambda job: job.created_at, reverse=True)


def recover_interrupted_sound_jobs(project_id: str) -> list[SoundGenerationJob]:
    """Mark jobs left in a volatile state after a platform restart."""

    recovered: list[SoundGenerationJob] = []
    for job in list_sound_jobs(project_id):
        if job.status not in {
            SoundGenerationJobStatus.QUEUED,
            SoundGenerationJobStatus.RUNNING,
            SoundGenerationJobStatus.CANCELLING,
        }:
            continue
        job.status = SoundGenerationJobStatus.INTERRUPTED
        job.error = "平台进程在任务完成前退出；已保留成功候选，可重新提交失败项"
        job.finished_at = utc_now()
        job.updated_at = job.finished_at
        for task in job.tasks:
            if task.status in {"queued", "running"}:
                task.status = "failed"
                task.error = "interrupted"
        try:
            orphan_paths: list[Path] = []
            with sound_plan_lock(project_id, job.plan_id):
                plan = load_sound_plan(project_id, job.plan_id)
                candidates = {
                    candidate.id: candidate
                    for cue in plan.cues
                    for candidate in cue.candidates
                }
                for task in job.tasks:
                    candidate = candidates.get(task.candidate_id)
                    if candidate is None:
                        if task.status == "succeeded":
                            task.status = "failed"
                            task.error = "interrupted: candidate metadata missing"
                        continue
                    if (
                        task.status == "succeeded"
                        and candidate.status
                        not in {SoundCandidateStatus.READY, SoundCandidateStatus.ADOPTED}
                    ):
                        task.status = "failed"
                        task.error = "interrupted: candidate was not committed"
                    if (
                        task.status in {"failed", "cancelled"}
                        and candidate.status
                        in {SoundCandidateStatus.QUEUED, SoundCandidateStatus.GENERATING}
                    ):
                        candidate.status = SoundCandidateStatus.FAILED
                        candidate.error = task.error or task.status
                        candidate.audio_file = None
                        candidate.audio_sha256 = None
                        candidate.duration_seconds = None
                        candidate.updated_at = utc_now()
                        orphan_paths.append(
                            candidate_audio_path(project_id, candidate.id, ensure=True)
                        )
                save_sound_plan(project_id, plan)
            for orphan_path in orphan_paths:
                try:
                    orphan_path.unlink(missing_ok=True)
                except OSError as exc:
                    job.error = f"{job.error}；删除未提交候选失败: {exc}"[:4000]
        except Exception as exc:
            job.error = f"{job.error}；恢复候选状态失败: {exc}"[:4000]
        job.completed = sum(task.status == "succeeded" for task in job.tasks)
        job.failed = sum(task.status == "failed" for task in job.tasks)
        save_sound_job(job)
        recovered.append(job)
    return recovered
