from __future__ import annotations

import io
import json
import os
import subprocess
import struct
import threading
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.core.tts_operations import GPUOperationLease
from dubbing_service.app.main import create_app
from dubbing_service.app.models.sound_generation import (
    GenerationProvenance,
    SemanticAnchor,
    SemanticPlacement,
    SoundCandidate,
    SoundCandidateStatus,
    SoundCue,
    SoundCueReviewStatus,
    SoundCueTrack,
    SoundPlan,
)
from dubbing_service.app.models.sound_jobs import (
    SoundGenerationJob,
    SoundGenerationJobStatus,
    SoundGenerationTask,
)
from dubbing_service.app.providers.audio_generation import AudioGenerationResult
from dubbing_service.app.services.sound_generation import bind_sound_plan
from dubbing_service.app.services import sound_jobs
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.sound_generation import (
    SoundGenerationStorageError,
    candidate_audio_path,
    candidate_audio_root,
    load_sound_plan,
    save_sound_plan,
)
from dubbing_service.app.storage.sound_jobs import (
    SoundJobStorageError,
    load_sound_job,
    recover_interrupted_sound_jobs,
    save_sound_job,
)


def _stereo_wav(duration_seconds: float = 1.0) -> bytes:
    sample_rate = 44100
    payload = io.BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(
            struct.pack("<hh", 400, -400) * round(duration_seconds * sample_rate)
        )
    return payload.getvalue()


@pytest.fixture
def sound_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id = client.post(
                "/api/v1/projects",
                json={"name": "sound job transaction", "description": ""},
            ).json()["id"]
            imported = client.post(
                f"/api/v1/projects/{project_id}/segments/import-json",
                json={
                    "json_text": json.dumps(
                        [{"speaker_name": "旁白", "text": "木门在远处关上。"}],
                        ensure_ascii=False,
                    )
                },
            )
            assert imported.status_code == 200, imported.text
            yield projects, project_id
    finally:
        clear_config_caches()


def _save_approved_plan(project_id: str, *, negative_prompt: str | None = None) -> SoundPlan:
    segment = load_segments(project_id)[0]
    plan = SoundPlan(
        id="soundplan-transaction",
        project_id=project_id,
        cues=[
            SoundCue(
                id="cue-1",
                segment_id=segment.id,
                track=SoundCueTrack.SFX,
                description_cn="木门关闭",
                prompt="A heavy wooden door closes, no speech, no music",
                negative_prompt=negative_prompt,
                placement=SemanticPlacement(start=SemanticAnchor(segment_id=segment.id)),
                requested_duration_seconds=1.0,
                review_status=SoundCueReviewStatus.APPROVED,
            )
        ],
    )
    plan = bind_sound_plan(plan, load_segments(project_id))
    save_sound_plan(project_id, plan)
    return plan


class _SuccessfulProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return AudioGenerationResult(
            success=True,
            audio_bytes=_stereo_wav(request.duration_seconds),
            sample_rate=44100,
            duration_seconds=request.duration_seconds,
            seed=request.seed,
            model_id=sound_jobs.MODEL_ID,
        )


def _lease(project_id: str, token: str = "test-lease") -> GPUOperationLease:
    return GPUOperationLease(token, "sound_generation", project_id)


def _create_directory_redirect(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"无法创建 Windows junction: {completed.stderr or completed.stdout}")
        return
    link.symlink_to(target, target_is_directory=True)


def _remove_directory_redirect(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def test_prepare_rolls_back_plan_when_job_persistence_fails(sound_project) -> None:
    projects, project_id = sound_project
    plan = _save_approved_plan(project_id)

    with patch.object(sound_jobs, "save_sound_job", side_effect=OSError("disk full")):
        with pytest.raises(sound_jobs.SoundJobError, match="已回滚候选"):
            sound_jobs.prepare_sound_job(
                project_id,
                plan.id,
                ["cue-1"],
                candidate_count=2,
                job_id="sfxjob-prepare-rollback",
            )

    assert load_sound_plan(project_id, plan.id).cues[0].candidates == []
    assert not (
        projects
        / project_id
        / "production"
        / "sound_generation"
        / "jobs"
        / "sfxjob-prepare-rollback.json"
    ).exists()


def test_negative_prompt_is_frozen_and_forwarded_to_provider(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id, negative_prompt="voices, dialogue")
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=1,
        job_id="sfxjob-negative-prompt",
    )
    queued = load_sound_plan(project_id, plan.id).cues[0].candidates[0]
    assert queued.provenance.negative_prompt == "voices, dialogue"

    provider = _SuccessfulProvider()
    sound_jobs.run_sound_job(
        project_id,
        job.id,
        _lease(project_id),
        provider_factory=lambda: provider,
        manage_worker=False,
    )

    assert provider.requests[0].negative_prompt == "voices, dialogue"
    assert load_sound_job(project_id, job.id).status == SoundGenerationJobStatus.SUCCEEDED


def test_batch_level_failure_terminalizes_every_pending_task_and_candidate(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=2,
        job_id="sfxjob-batch-failure",
    )

    def fail_provider():
        raise RuntimeError("provider construction failed")

    sound_jobs.run_sound_job(
        project_id,
        job.id,
        _lease(project_id),
        provider_factory=fail_provider,
        manage_worker=False,
    )

    stored_job = load_sound_job(project_id, job.id)
    assert stored_job.status == SoundGenerationJobStatus.FAILED
    assert stored_job.failed == 2
    assert {task.status for task in stored_job.tasks} == {"failed"}
    assert {
        candidate.status
        for candidate in load_sound_plan(project_id, plan.id).cues[0].candidates
    } == {SoundCandidateStatus.FAILED}


def test_promoted_wav_is_removed_when_candidate_metadata_commit_fails(sound_project) -> None:
    projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=1,
        job_id="sfxjob-plan-commit-failure",
    )
    candidate_id = job.tasks[0].candidate_id
    real_save_sound_plan = sound_jobs.save_sound_plan

    def fail_ready_commit(pid, candidate_plan):
        statuses = {
            candidate.status
            for cue in candidate_plan.cues
            for candidate in cue.candidates
        }
        if SoundCandidateStatus.READY in statuses:
            raise OSError("plan metadata commit failed")
        return real_save_sound_plan(pid, candidate_plan)

    with patch.object(sound_jobs, "save_sound_plan", side_effect=fail_ready_commit):
        sound_jobs.run_sound_job(
            project_id,
            job.id,
            _lease(project_id),
            provider_factory=_SuccessfulProvider,
            manage_worker=False,
        )

    assert not (
        candidate_audio_root(project_id) / f"{candidate_id}.wav"
    ).exists()
    candidate = load_sound_plan(project_id, plan.id).cues[0].candidates[0]
    assert candidate.status == SoundCandidateStatus.FAILED
    assert "metadata commit failed" in str(candidate.error)
    assert load_sound_job(project_id, job.id).status == SoundGenerationJobStatus.FAILED


def test_final_save_failure_still_releases_lease_and_thread_registry(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=1,
        job_id="sfxjob-final-save-failure",
    )
    real_save_sound_job = sound_jobs.save_sound_job
    released: list[GPUOperationLease] = []

    def fail_final_save(candidate_job):
        if candidate_job.finished_at is not None:
            raise OSError("final save failed")
        return real_save_sound_job(candidate_job)

    with sound_jobs._RUNNING_THREADS_GUARD:
        sound_jobs._RUNNING_THREADS[job.id] = threading.current_thread()
    with (
        patch.object(sound_jobs, "save_sound_job", side_effect=fail_final_save),
        patch.object(sound_jobs, "release_gpu_operation", side_effect=released.append),
    ):
        sound_jobs.run_sound_job(
            project_id,
            job.id,
            _lease(project_id, "release-me"),
            provider_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            manage_worker=False,
        )

    assert [lease.token for lease in released] == ["release-me"]
    with sound_jobs._RUNNING_THREADS_GUARD:
        assert job.id not in sound_jobs._RUNNING_THREADS


def test_recovery_fails_uncommitted_candidate_and_removes_promoted_orphan(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=1,
        job_id="sfxjob-recovery-orphan",
    )
    job.status = SoundGenerationJobStatus.RUNNING
    job.tasks[0].status = "running"
    save_sound_job(job)
    candidate_id = job.tasks[0].candidate_id
    persisted_plan = load_sound_plan(project_id, plan.id)
    persisted_plan.cues[0].candidates[0].status = SoundCandidateStatus.GENERATING
    save_sound_plan(project_id, persisted_plan)
    orphan = candidate_audio_root(project_id, ensure=True) / f"{candidate_id}.wav"
    orphan.write_bytes(_stereo_wav())

    recovered = recover_interrupted_sound_jobs(project_id)

    assert [item.id for item in recovered] == [job.id]
    assert not orphan.exists()
    recovered_job = load_sound_job(project_id, job.id)
    assert recovered_job.tasks[0].status == "failed"
    assert recovered_job.failed == 1
    candidate = load_sound_plan(project_id, plan.id).cues[0].candidates[0]
    assert candidate.status == SoundCandidateStatus.FAILED
    assert candidate.error == "interrupted"


def test_sound_task_seed_keeps_signed_31_bit_boundary() -> None:
    SoundGenerationTask(cue_id="cue", candidate_id="candidate", seed=2_147_483_647)
    with pytest.raises(ValidationError):
        SoundGenerationTask(cue_id="cue", candidate_id="candidate", seed=2_147_483_648)


def test_candidate_ids_cannot_escape_project_staging(sound_project) -> None:
    _projects, project_id = sound_project
    provenance = GenerationProvenance(
        provider="stable_audio_3",
        model_id="stable-audio-3-small-sfx",
        prompt="door closes",
    )

    with pytest.raises(ValidationError):
        SoundCandidate(id="../../../../../victim", provenance=provenance)
    with pytest.raises(ValidationError):
        SoundGenerationTask(
            cue_id="cue",
            candidate_id="../../../../../victim",
            seed=1,
        )
    with pytest.raises(SoundGenerationStorageError, match="非法音效候选 ID"):
        candidate_audio_path(project_id, "../../../../../victim", ensure=True)


def test_sound_generation_root_redirect_is_rejected_before_any_persisted_write(
    sound_project,
) -> None:
    projects, project_id = sound_project
    project = projects / project_id
    production = project / "production"
    redirect_target = production / "assets" / "sfx"
    redirect_target.mkdir(parents=True, exist_ok=True)
    generation_link = production / "sound_generation"
    _create_directory_redirect(generation_link, redirect_target)
    try:
        plan = SoundPlan(id="soundplan-root-redirect", project_id=project_id)
        with pytest.raises(SoundGenerationStorageError, match="音效生成目录逃逸"):
            save_sound_plan(project_id, plan)

        job = SoundGenerationJob(
            id="sfxjob-root-redirect",
            project_id=project_id,
            plan_id=plan.id,
        )
        with pytest.raises(SoundJobStorageError, match="音效生成目录逃逸"):
            save_sound_job(job)

        with pytest.raises(SoundGenerationStorageError, match="音效生成目录逃逸"):
            candidate_audio_path(project_id, "candidate-root-redirect", ensure=True)

        assert list(redirect_target.iterdir()) == []
    finally:
        _remove_directory_redirect(generation_link)


def test_sound_job_storage_rejects_cross_project_record(sound_project) -> None:
    projects, project_id = sound_project
    other_id = "other-project"
    other_root = projects / other_id
    other_root.mkdir(parents=True)
    (other_root / "project.json").write_text(
        json.dumps({"id": other_id, "name": "other", "chapters": []}),
        encoding="utf-8",
    )
    jobs_dir = (
        projects / project_id / "production" / "sound_generation" / "jobs"
    )
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "sfxjob-cross.json").write_text(
        json.dumps(
            {
                "id": "sfxjob-cross",
                "project_id": other_id,
                "plan_id": "soundplan-transaction",
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )

    from dubbing_service.app.storage.sound_jobs import SoundJobStorageError

    with pytest.raises(SoundJobStorageError, match="归属或文件名不一致"):
        load_sound_job(project_id, "sfxjob-cross")


def test_stale_runner_save_cannot_clear_persisted_cancel_request(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=1,
        job_id="sfxjob-monotonic-cancel",
    )
    stale_runner_copy = load_sound_job(project_id, job.id)

    cancelled = sound_jobs.request_sound_job_cancel(project_id, job.id)
    assert cancelled.cancel_requested is True
    assert cancelled.status == SoundGenerationJobStatus.CANCELLING

    stale_runner_copy.status = SoundGenerationJobStatus.RUNNING
    stale_runner_copy.cancel_requested = False
    save_sound_job(stale_runner_copy)

    persisted = load_sound_job(project_id, job.id)
    assert persisted.cancel_requested is True
    assert persisted.status == SoundGenerationJobStatus.CANCELLING


def test_thread_start_failure_terminalizes_job_candidates_and_registry(sound_project) -> None:
    _projects, project_id = sound_project
    plan = _save_approved_plan(project_id)
    job = sound_jobs.prepare_sound_job(
        project_id,
        plan.id,
        ["cue-1"],
        candidate_count=2,
        job_id="sfxjob-thread-start-failure",
    )

    with patch.object(threading.Thread, "start", side_effect=RuntimeError("thread unavailable")):
        with pytest.raises(RuntimeError, match="thread unavailable"):
            sound_jobs.launch_sound_job(job, _lease(project_id, "never-started"))

    persisted_job = load_sound_job(project_id, job.id)
    assert persisted_job.status == SoundGenerationJobStatus.FAILED
    assert persisted_job.finished_at is not None
    assert persisted_job.failed == 2
    assert {task.status for task in persisted_job.tasks} == {"failed"}
    assert all("thread unavailable" in str(task.error) for task in persisted_job.tasks)
    persisted_plan = load_sound_plan(project_id, plan.id)
    assert {
        candidate.status for candidate in persisted_plan.cues[0].candidates
    } == {SoundCandidateStatus.FAILED}
    with sound_jobs._RUNNING_THREADS_GUARD:
        assert job.id not in sound_jobs._RUNNING_THREADS
