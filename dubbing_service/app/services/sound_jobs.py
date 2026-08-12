"""Durable Stable Audio candidate generation orchestration."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import secrets
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing_service.app.core.helpers import project_root
from dubbing_service.app.core.sfx_worker import (
    SFX_BASE_URL,
    load_model,
    start_worker,
    stop_worker,
    unload_model,
    worker_auth_token,
)
from dubbing_service.app.core.tts_operations import GPUOperationLease, release_gpu_operation
from dubbing_service.app.models.sound_generation import (
    GenerationProvenance,
    SoundCandidate,
    SoundCandidateStatus,
)
from dubbing_service.app.models.sound_jobs import (
    SoundGenerationJob,
    SoundGenerationJobStatus,
    SoundGenerationTask,
    utc_now,
)
from dubbing_service.app.services.sound_generation import (
    attach_sound_candidate,
    fingerprint_sound_cue,
    refresh_sound_plan_staleness,
)
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.sound_generation import (
    candidate_audio_path,
    candidate_audio_root,
    load_sound_plan,
    save_sound_plan,
    sound_plan_lock,
)
from dubbing_service.app.storage.sound_jobs import (
    delete_sound_job,
    load_sound_job,
    save_sound_job,
    sound_job_lock,
    sound_job_exists,
)


logger = logging.getLogger(__name__)

MODEL_ID = "stable-audio-3-small-sfx"
MODEL_REVISION = "ae12755283df9d62ca39a9b050a39a0b607b8c20"
CODE_REVISION = "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"
MODEL_SOURCE = "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"
MODEL_LICENSE_COMMUNITY = "Stability AI Community License + Google Gemma Terms"
MODEL_LICENSE_ENTERPRISE = "Stability AI Enterprise License + Google Gemma Terms"


class SoundJobError(RuntimeError):
    pass


_RUNNING_THREADS: dict[str, threading.Thread] = {}
_RUNNING_THREADS_GUARD = threading.Lock()


def _license_snapshot() -> tuple[str | None, dict[str, Any]]:
    manifest_path = Path(__file__).resolve().parents[3] / "models" / "manifests" / "stable_audio_3_small_sfx.json"
    try:
        from scripts.model_preflight import (
            has_license_acceptance,
            license_receipt_path,
            load_manifest,
        )

        manifest = load_manifest(manifest_path)
        if not has_license_acceptance(manifest, project_root=Path(__file__).resolve().parents[3]):
            return None, {}
        receipt_path = license_receipt_path(manifest, project_root=Path(__file__).resolve().parents[3])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path and receipt_path.is_file() else {}
        confirmation_id = receipt.get("operator_confirmation_id")
        snapshot = {
            "manifest": manifest_path.name,
            "confirmation_sha256": receipt.get("confirmation_sha256"),
            "license_tier": receipt.get("license_tier"),
            "stability_registered": receipt.get("stability_registered"),
            "stability_terms_accepted": receipt.get("stability_terms_accepted"),
            "gemma_terms_accepted": receipt.get("gemma_terms_accepted"),
            "accepted_at_utc": receipt.get("accepted_at_utc"),
        }
        return str(confirmation_id) if confirmation_id else None, snapshot
    except Exception:
        return None, {}


def _provenance(
    prompt: str,
    duration: float,
    seed: int,
    *,
    negative_prompt: str | None = None,
) -> GenerationProvenance:
    confirmation_id, license_snapshot = _license_snapshot()
    model_license = (
        MODEL_LICENSE_ENTERPRISE
        if license_snapshot.get("license_tier") == "enterprise"
        else MODEL_LICENSE_COMMUNITY
    )
    return GenerationProvenance(
        provider="stableaudio-local-cuda",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_source=MODEL_SOURCE,
        model_license=model_license,
        commercial_use=True if confirmation_id else None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=8,
        guidance_scale=1.0,
        requested_duration_seconds=duration,
        sample_rate=44100,
        parameters={"batch_size": 1, "dtype": "float16", "device": "cuda:0"},
        code_revision=CODE_REVISION,
        license_confirmation_id=confirmation_id,
        license_snapshot=license_snapshot,
    )


def prepare_sound_job(
    project_id: str,
    plan_id: str,
    cue_ids: list[str],
    *,
    candidate_count: int,
    job_id: str,
) -> SoundGenerationJob:
    """Persist queued immutable candidate identities before GPU work starts."""

    if not 1 <= candidate_count <= 4:
        raise SoundJobError("每个 cue 每次只能生成 1–4 个候选")
    if not cue_ids or len(cue_ids) != len(set(cue_ids)):
        raise SoundJobError("必须提供不重复的 cue ID")
    if sound_job_exists(project_id, job_id):
        raise SoundJobError("音效生成任务 ID 已存在，拒绝覆盖")
    original_plan: Any | None = None
    job: SoundGenerationJob | None = None
    with sound_plan_lock(project_id, plan_id):
        plan = refresh_sound_plan_staleness(load_sound_plan(project_id, plan_id), load_segments(project_id))
        if plan.stale:
            save_sound_plan(project_id, plan)
            raise SoundJobError("SoundPlan 已因文本变化而过期，请重新起草或重新批准")
        original_plan = plan.model_copy(deep=True)
        cues = {cue.id: cue for cue in plan.cues}
        missing = [cue_id for cue_id in cue_ids if cue_id not in cues]
        if missing:
            raise SoundJobError(f"cue 不存在: {', '.join(missing[:5])}")
        tasks: list[SoundGenerationTask] = []
        for cue_id in cue_ids:
            cue = cues[cue_id]
            review = str(getattr(cue, "review_status", "draft")).lower()
            if "." in review:
                review = review.rsplit(".", 1)[-1]
            if review != "approved":
                raise SoundJobError(f"cue {cue_id} 尚未人工批准")
            if not cue.enabled or cue.stale:
                raise SoundJobError(f"cue {cue_id} 已禁用或过期")
            if cue.track.value == "sfx" and not 1 <= cue.requested_duration_seconds <= 15:
                raise SoundJobError("SFX 生成源时长必须为 1–15 秒")
            if cue.track.value == "ambience" and not 20 <= cue.requested_duration_seconds <= 60:
                raise SoundJobError("Ambience 生成源时长必须为 20–60 秒")
            live_candidates = [item for item in cue.candidates if item.status != SoundCandidateStatus.FAILED]
            if len(live_candidates) + candidate_count > 4:
                raise SoundJobError(f"cue {cue_id} 最多保留 4 个候选")
            for _ in range(candidate_count):
                candidate_id = f"candidate-{secrets.token_hex(8)}"
                seed = secrets.randbelow(2_147_483_648)
                candidate = SoundCandidate(
                    id=candidate_id,
                    status=SoundCandidateStatus.QUEUED,
                    provenance=_provenance(
                        cue.prompt,
                        cue.requested_duration_seconds,
                        seed,
                        negative_prompt=cue.negative_prompt,
                    ),
                    cue_fingerprint=fingerprint_sound_cue(cue),
                )
                plan = attach_sound_candidate(plan, cue_id, candidate)
                tasks.append(SoundGenerationTask(cue_id=cue_id, candidate_id=candidate_id, seed=seed))
        job = SoundGenerationJob(id=job_id, project_id=project_id, plan_id=plan_id, tasks=tasks)
        save_sound_plan(project_id, plan)
        try:
            save_sound_job(job)
        except Exception as exc:
            rollback_errors: list[str] = []
            plan_restored = False
            try:
                save_sound_plan(project_id, original_plan)
                plan_restored = True
            except Exception as rollback_exc:
                rollback_errors.append(f"恢复 SoundPlan 失败: {rollback_exc}")
            if plan_restored:
                try:
                    delete_sound_job(project_id, job_id)
                except Exception as rollback_exc:
                    rollback_errors.append(f"删除不完整任务失败: {rollback_exc}")
            detail = f"保存音效生成任务失败，已回滚候选: {exc}"
            if rollback_errors:
                detail = f"{detail}；{'; '.join(rollback_errors)}"
            raise SoundJobError(detail) from exc
    assert job is not None
    return job


def _candidate(plan: Any, cue_id: str, candidate_id: str) -> Any:
    cue = next((item for item in plan.cues if item.id == cue_id), None)
    if cue is None:
        raise SoundJobError("生成期间 cue 被删除")
    candidate = next((item for item in cue.candidates if item.id == candidate_id), None)
    if candidate is None:
        raise SoundJobError("生成期间候选记录被删除")
    return cue, candidate


def _validate_wav(payload: bytes, expected_duration: float) -> tuple[int, float]:
    if len(payload) > 256 * 1024 * 1024:
        raise SoundJobError("worker 返回的 WAV 超过 256 MiB 上限")
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            if stream.getcomptype() != "NONE":
                raise SoundJobError("worker 返回的 WAV 不是未压缩 PCM")
    except (wave.Error, EOFError) as exc:
        raise SoundJobError(f"worker 返回了无效 WAV: {exc}") from exc
    duration = frames / sample_rate if sample_rate else 0.0
    if channels != 2 or sample_rate != 44100:
        raise SoundJobError(f"worker 输出必须是 44.1 kHz 立体声，实际 {sample_rate} Hz/{channels}ch")
    if duration <= 0 or abs(duration - expected_duration) > 0.1:
        raise SoundJobError(f"worker 输出时长误差超过 100 ms: 请求 {expected_duration:.3f}s，实际 {duration:.3f}s")
    return sample_rate, duration


def _gpu_memory_used_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
    return None


def _handoff_gpu_before_sfx() -> int | None:
    try:
        from dubbing_service.app.routers.settings import unload_managed_tts_model_for_gpu_handoff

        unload_managed_tts_model_for_gpu_handoff()
    except Exception as exc:
        raise SoundJobError(f"卸载 TTS 模型失败，拒绝与 SFX 共享显存: {exc}") from exc
    try:
        from dubbing_service.app.providers.subtitles import clear_whisper_model_cache

        clear_whisper_model_cache()
    except Exception as exc:
        raise SoundJobError(f"卸载 Whisper 模型失败，拒绝与 SFX 共享显存: {exc}") from exc
    return _gpu_memory_used_mib()


def _enforce_gpu_release(baseline_mib: int | None) -> None:
    try:
        unload_model()
    except Exception:
        logger.warning("Stable Audio worker 卸载模型失败，将结束 worker", exc_info=True)
        stop_worker()
        return
    if baseline_mib is None:
        logger.warning("无法读取生成前显存基线；为避免未验证的显存驻留，将结束 Stable Audio worker")
        stop_worker()
        return
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        current = _gpu_memory_used_mib()
        if current is not None and current <= baseline_mib + 512:
            return
        time.sleep(0.25)
    logger.warning("Stable Audio worker 未在 10 秒内释放显存，结束进程")
    stop_worker()


def _default_provider() -> Any:
    from dubbing_service.app.providers.audio_generation import StableAudioSFXProvider

    return StableAudioSFXProvider(SFX_BASE_URL, api_key=worker_auth_token(), timeout=900.0)


def _mark_plan_candidates_failed(
    project_id: str,
    plan_id: str,
    candidate_errors: dict[str, str],
) -> None:
    """Terminalize queued/generating candidates in one atomic plan update."""

    if not candidate_errors:
        return
    with sound_plan_lock(project_id, plan_id):
        plan = load_sound_plan(project_id, plan_id)
        candidates = {
            candidate.id: candidate
            for cue in plan.cues
            for candidate in cue.candidates
        }
        missing = sorted(set(candidate_errors) - set(candidates))
        if missing:
            raise SoundJobError(
                f"生成期间候选记录被删除: {', '.join(missing[:5])}"
            )
        changed = False
        for candidate_id, error in candidate_errors.items():
            candidate = candidates[candidate_id]
            if candidate.status not in {
                SoundCandidateStatus.QUEUED,
                SoundCandidateStatus.GENERATING,
            }:
                continue
            candidate.status = SoundCandidateStatus.FAILED
            candidate.error = error[:2000]
            candidate.updated_at = utc_now()
            changed = True
        if changed:
            save_sound_plan(project_id, plan)


def _terminalize_failed_batch(job: SoundGenerationJob, error: str) -> None:
    """Make every unfinished task/candidate terminal after a batch-level fault."""

    message = error[:2000] or "sound generation batch failed"
    for task in job.tasks:
        if task.status in {"queued", "running"}:
            task.status = "failed"
            task.error = message
    candidate_errors = {
        task.candidate_id: (
            "cancelled"
            if task.status == "cancelled"
            else (task.error or message)
        )
        for task in job.tasks
    }
    try:
        _mark_plan_candidates_failed(
            job.project_id,
            job.plan_id,
            candidate_errors,
        )
    except Exception as exc:
        logger.exception("批次失败后无法终结所有候选")
        error = f"{error}；终结候选失败: {exc}"
    job.completed = sum(task.status == "succeeded" for task in job.tasks)
    job.failed = sum(task.status == "failed" for task in job.tasks)
    job.status = SoundGenerationJobStatus.FAILED
    job.error = error[:4000]


def _remove_uncommitted_candidate_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("删除未提交的候选音频失败: %s", path, exc_info=True)


def run_sound_job(
    project_id: str,
    job_id: str,
    lease: GPUOperationLease,
    *,
    provider_factory: Callable[[], Any] | None = None,
    manage_worker: bool = True,
) -> None:
    """Run one durable batch; safe to execute in a daemon thread or a test."""

    job: SoundGenerationJob | None = None
    baseline: int | None = None
    try:
        job = load_sound_job(project_id, job_id)
        job.status = SoundGenerationJobStatus.RUNNING
        job.started_at = utc_now()
        job.updated_at = job.started_at
        save_sound_job(job)
        if manage_worker:
            baseline = _handoff_gpu_before_sfx()
            start_worker(require_quality_gate=True)
            load_model()
        provider = (provider_factory or _default_provider)()
        from dubbing_service.app.providers.audio_generation import AudioGenerationRequest

        task_ids = [task.candidate_id for task in job.tasks]
        for candidate_id in task_ids:
            job = load_sound_job(project_id, job_id)
            task = next(
                (item for item in job.tasks if item.candidate_id == candidate_id),
                None,
            )
            if task is None:
                raise SoundJobError(f"生成期间任务记录被删除: {candidate_id}")
            if job.cancel_requested:
                task.status = "cancelled"
                task.error = "cancelled"
                _mark_plan_candidates_failed(
                    project_id,
                    job.plan_id,
                    {task.candidate_id: "cancelled"},
                )
                job.updated_at = utc_now()
                save_sound_job(job)
                continue
            task.status = "running"
            job.updated_at = utc_now()
            save_sound_job(job)
            promoted_target: Path | None = None
            metadata_committed = False
            try:
                with sound_plan_lock(project_id, job.plan_id):
                    plan = refresh_sound_plan_staleness(
                        load_sound_plan(project_id, job.plan_id), load_segments(project_id)
                    )
                    cue, candidate = _candidate(plan, task.cue_id, task.candidate_id)
                    if plan.stale or cue.stale or candidate.stale:
                        raise SoundJobError("生成前 cue 已因文本或 prompt 变化而过期")
                    candidate.status = SoundCandidateStatus.GENERATING
                    candidate.updated_at = utc_now()
                    save_sound_plan(project_id, plan)
                result = provider.generate(
                    AudioGenerationRequest(
                        prompt=cue.prompt,
                        duration_seconds=cue.requested_duration_seconds,
                        negative_prompt=cue.negative_prompt,
                        seed=task.seed,
                        steps=8,
                        cfg_scale=1.0,
                        chunked_decode=True,
                    )
                )
                if not result.success or not result.audio_bytes:
                    raise SoundJobError(result.error or "Stable Audio worker 生成失败")
                if result.seed is not None and result.seed != task.seed:
                    raise SoundJobError(f"worker 返回的 seed 不匹配: {result.seed} != {task.seed}")
                if result.model_id is not None and result.model_id != MODEL_ID:
                    raise SoundJobError(f"worker 返回的模型不匹配: {result.model_id}")
                if result.sample_rate is not None and result.sample_rate != 44100:
                    raise SoundJobError(f"worker 响应头采样率不匹配: {result.sample_rate}")
                sample_rate, actual_duration = _validate_wav(result.audio_bytes, cue.requested_duration_seconds)
                digest = hashlib.sha256(result.audio_bytes).hexdigest()
                target_dir = candidate_audio_root(project_id, ensure=True)
                target = candidate_audio_path(project_id, task.candidate_id, ensure=True)
                if target.exists():
                    raise SoundJobError("候选文件已存在，拒绝覆盖不可变结果")
                fd, staged_name = tempfile.mkstemp(dir=target_dir, prefix="candidate_", suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(result.audio_bytes)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(staged_name, target)
                    promoted_target = target
                except Exception:
                    Path(staged_name).unlink(missing_ok=True)
                    raise
                with sound_plan_lock(project_id, job.plan_id):
                    plan = load_sound_plan(project_id, job.plan_id)
                    cue, candidate = _candidate(plan, task.cue_id, task.candidate_id)
                    candidate.audio_file = str(target.relative_to(project_root(project_id, ensure_dir=False)))
                    candidate.audio_sha256 = digest
                    candidate.duration_seconds = actual_duration
                    candidate.status = SoundCandidateStatus.READY
                    candidate.provenance.sample_rate = sample_rate
                    candidate.provenance.audio_sha256 = digest
                    candidate.provenance.generated_at = utc_now()
                    candidate.updated_at = utc_now()
                    save_sound_plan(project_id, plan)
                metadata_committed = True
                task.status = "succeeded"
                job.completed += 1
            except Exception as exc:
                if not metadata_committed:
                    _remove_uncommitted_candidate_file(promoted_target)
                task.status = "failed"
                task.error = str(exc)[:2000]
                job.failed += 1
                try:
                    _mark_plan_candidates_failed(
                        project_id,
                        job.plan_id,
                        {task.candidate_id: str(exc)},
                    )
                except Exception as state_exc:
                    raise SoundJobError(
                        f"生成失败且无法终结候选 {task.candidate_id}: {state_exc}"
                    ) from state_exc
            job.updated_at = utc_now()
            save_sound_job(job)
        job = load_sound_job(project_id, job_id)
        if job.cancel_requested:
            job.status = SoundGenerationJobStatus.CANCELLED
        elif job.completed and job.failed:
            job.status = SoundGenerationJobStatus.PARTIAL
        elif job.completed:
            job.status = SoundGenerationJobStatus.SUCCEEDED
        else:
            job.status = SoundGenerationJobStatus.FAILED
            job.error = job.error or "所有候选均生成失败"
    except Exception as exc:
        try:
            # The durable copy is authoritative here.  In particular, a
            # failed save may leave a task queued/running on disk even if the
            # in-memory object had already advanced it.
            job = load_sound_job(project_id, job_id)
        except Exception:
            if job is None:
                logger.exception("音效任务载入失败，无法写入终态")
        if job is not None:
            _terminalize_failed_batch(job, str(exc))
    finally:
        try:
            if manage_worker:
                try:
                    _enforce_gpu_release(baseline)
                except Exception:
                    logger.exception("Stable Audio worker 结束清理失败")
            if job is not None:
                job.finished_at = utc_now()
                job.updated_at = job.finished_at
                try:
                    save_sound_job(job)
                except Exception:
                    logger.exception("保存音效任务最终状态失败")
        finally:
            try:
                release_gpu_operation(lease)
            finally:
                with _RUNNING_THREADS_GUARD:
                    _RUNNING_THREADS.pop(job_id, None)


def launch_sound_job(job: SoundGenerationJob, lease: GPUOperationLease) -> None:
    thread = threading.Thread(
        target=run_sound_job,
        args=(job.project_id, job.id, lease),
        name=f"sound-generation-{job.id}",
        daemon=True,
    )
    with _RUNNING_THREADS_GUARD:
        _RUNNING_THREADS[job.id] = thread
    try:
        thread.start()
    except Exception as exc:
        with _RUNNING_THREADS_GUARD:
            _RUNNING_THREADS.pop(job.id, None)
        try:
            failed = load_sound_job(job.project_id, job.id)
            _terminalize_failed_batch(failed, f"无法启动音效生成线程: {exc}")
            failed.finished_at = utc_now()
            failed.updated_at = failed.finished_at
            save_sound_job(failed)
        except Exception:
            logger.exception("音效生成线程启动失败后无法写入终态")
        raise


def request_sound_job_cancel(project_id: str, job_id: str) -> SoundGenerationJob:
    with sound_job_lock(project_id, job_id):
        job = load_sound_job(project_id, job_id)
        if job.status in {
            SoundGenerationJobStatus.SUCCEEDED,
            SoundGenerationJobStatus.PARTIAL,
            SoundGenerationJobStatus.FAILED,
            SoundGenerationJobStatus.CANCELLED,
            SoundGenerationJobStatus.INTERRUPTED,
        }:
            return job
        job.cancel_requested = True
        job.status = SoundGenerationJobStatus.CANCELLING
        job.updated_at = utc_now()
        save_sound_job(job)
        return job


def shutdown_sound_jobs(timeout_seconds: float = 5.0) -> None:
    with _RUNNING_THREADS_GUARD:
        threads = list(_RUNNING_THREADS.values())
    for thread in threads:
        thread.join(timeout=max(0.0, timeout_seconds / max(len(threads), 1)))
    stop_worker()
