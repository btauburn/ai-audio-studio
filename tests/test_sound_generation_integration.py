from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.core.helpers import sha256_file
from dubbing_service.app.core.tts_operations import GPUOperationLease
from dubbing_service.app.main import create_app
from dubbing_service.app.models.master import AudioMaster
from dubbing_service.app.models.production import (
    AssetLicense,
    AudioAsset,
    AudioAssetOrigin,
    ProductionSettings,
    SoundDesign,
    SoundTrack,
    TimelineClip,
)
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
from dubbing_service.app.models.sound_jobs import SoundGenerationJobStatus
from dubbing_service.app.providers.audio_generation import AudioGenerationResult
from dubbing_service.app.services.sound_generation import attach_sound_candidate, bind_sound_plan
from dubbing_service.app.services.sound_jobs import (
    MODEL_ID,
    prepare_sound_job,
    request_sound_job_cancel,
    run_sound_job,
)
from dubbing_service.app.storage.production import (
    load_assets,
    load_settings,
    save_assets,
    save_settings,
)
from dubbing_service.app.storage.masters import masters_dir, save_master
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.sound_generation import (
    SoundGenerationStorageError,
    candidate_audio_root,
    list_sound_plans,
    load_sound_plan,
    save_sound_plan,
)
from dubbing_service.app.storage.sound_jobs import (
    load_sound_job,
    recover_interrupted_sound_jobs,
    save_sound_job,
)
from tests.audio_test_utils import make_wav_bytes


def _stereo_wav(*, duration_seconds: float = 1.0, sample_rate: int = 44100) -> bytes:
    frames = round(duration_seconds * sample_rate)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack("<hh", 400, -400) * frames)
    return payload.getvalue()


def _create_project_with_segments(
    client: TestClient,
    projects_root: Path,
    *,
    count: int = 2,
    with_audio: bool = False,
    approve: bool = False,
    confirm_voice_rights: bool = False,
) -> tuple[str, list[dict]]:
    project_id = client.post(
        "/api/v1/projects", json={"name": "sound integration", "description": ""},
    ).json()["id"]
    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={"json_text": json.dumps([
            {
                "speaker_name": "旁白",
                "text": f"第 {index + 1} 句里发生了声音事件。",
                "emotion_description": "平静",
            }
            for index in range(count)
        ], ensure_ascii=False)},
    )
    assert imported.status_code == 200, imported.text
    segments = imported.json()["segments"]
    if with_audio:
        for index, segment in enumerate(segments):
            path = projects_root / project_id / "audio" / "segments" / f"{segment['id']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(make_wav_bytes(sample_value=300 + index, frames=8000))
            segment.update({
                "status": "done",
                "audio_file": f"audio/segments/{segment['id']}.wav",
            })
        restored = client.put(
            f"/api/v1/projects/{project_id}/segments/restore",
            json={"segments": segments},
        )
        assert restored.status_code == 200, restored.text
        segments = restored.json()
    if approve:
        for segment in segments:
            response = client.patch(
                f"/api/v1/projects/{project_id}/segments/{segment['id']}/review",
                json={"review_status": "approved"},
            )
            assert response.status_code == 200, response.text
    if confirm_voice_rights:
        role_id = segments[0]["role_id"]
        response = client.patch(
            f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
            json={
                "voice_source": "self",
                "voice_license_name": "owned",
                "voice_commercial_use": True,
                "voice_rights_note": "recorded by owner",
            },
        )
        assert response.status_code == 200, response.text
    return project_id, segments


class _PlannerProvider:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        return SimpleNamespace(success=True, raw={}, text=json.dumps(payload, ensure_ascii=False))


def _planner_cue(segment_id: str, **updates: object) -> dict:
    cue = {
        "segment_id": segment_id,
        "track": "sfx",
        "description_cn": "木门关闭",
        "prompt_en": "A heavy wooden door closes, no speech, no music",
        "start_edge": "start",
        "start_offset_seconds": 0.0,
        "end_segment_id": None,
        "end_edge": None,
        "end_offset_seconds": 0.0,
        "source_duration_seconds": 2.0,
        "gain_db": 0.0,
        "fade_in": 0.0,
        "fade_out": 0.1,
    }
    cue.update(updates)
    return cue


def test_sound_plan_router_validates_unknown_anchors_and_refreshes_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, segments = _create_project_with_segments(client, projects)
            provider = _PlannerProvider([{"cues": [_planner_cue(segments[0]["id"])]}])
            monkeypatch.setattr(
                "dubbing_service.app.routers.sound_generation._provider_from_config",
                lambda _config_id: (provider, "fake-provider", "fake-model"),
            )
            drafted = client.post(
                f"/api/v1/projects/{project_id}/sound-plans",
                json={"density": "balanced"},
            )
            assert drafted.status_code == 200, drafted.text
            plan = drafted.json()
            cue = plan["cues"][0]
            assert cue["review_status"] == "draft"
            assert cue["placement"]["start"]["segment_id"] == segments[0]["id"]
            assert provider.calls and "audio_file" not in provider.calls[0][1]["content"]

            unknown = client.patch(
                f"/api/v1/projects/{project_id}/sound-plans/{plan['id']}/cues/{cue['id']}",
                json={
                    "placement": {
                        "start": {"segment_id": "missing-segment", "edge": "start"},
                    },
                    "review_status": "approved",
                },
            )
            assert unknown.status_code == 422, unknown.text
            unchanged = client.get(
                f"/api/v1/projects/{project_id}/sound-plans/{plan['id']}",
            ).json()
            assert unchanged["cues"][0]["placement"] == cue["placement"]

            approved = client.patch(
                f"/api/v1/projects/{project_id}/sound-plans/{plan['id']}/cues/{cue['id']}",
                json={"review_status": "approved"},
            )
            assert approved.status_code == 200, approved.text
            assert approved.json()["cues"][0]["review_status"] == "approved"

            changed = client.patch(
                f"/api/v1/projects/{project_id}/segments/{segments[0]['id']}",
                json={"text": "源台词已经发生变化。"},
            )
            assert changed.status_code == 200, changed.text
            stale = client.get(
                f"/api/v1/projects/{project_id}/sound-plans/{plan['id']}",
            )
            assert stale.status_code == 200, stale.text
            assert stale.json()["stale"] is True
            assert stale.json()["cues"][0]["stale"] is True
    finally:
        clear_config_caches()


def test_planner_retries_invalid_json_and_never_persists_unknown_or_backward_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, segments = _create_project_with_segments(client, projects)
            unknown_provider = _PlannerProvider([
                {"cues": [_planner_cue("unknown-segment")]},
                {"cues": [_planner_cue("unknown-segment")]},
            ])
            monkeypatch.setattr(
                "dubbing_service.app.routers.sound_generation._provider_from_config",
                lambda _config_id: (unknown_provider, "fake", "fake"),
            )
            rejected = client.post(f"/api/v1/projects/{project_id}/sound-plans", json={})
            assert rejected.status_code == 502, rejected.text
            assert len(unknown_provider.calls) == 2
            assert list_sound_plans(project_id) == []

            backward = _planner_cue(
                segments[1]["id"],
                track="ambience",
                description_cn="雨夜环境",
                prompt_en="Steady rain outside at night, no speech, no music",
                source_duration_seconds=20.0,
                end_segment_id=segments[0]["id"],
                end_edge="end",
                fade_in=1.0,
                fade_out=1.0,
            )
            backward_provider = _PlannerProvider([
                {"cues": [backward]},
                {"cues": [backward]},
            ])
            monkeypatch.setattr(
                "dubbing_service.app.routers.sound_generation._provider_from_config",
                lambda _config_id: (backward_provider, "fake", "fake"),
            )
            rejected = client.post(f"/api/v1/projects/{project_id}/sound-plans", json={})
            assert rejected.status_code == 502, rejected.text
            # This is a cross-cue semantic validation after the provider JSON
            # itself has parsed, so no repair round trip is needed.
            assert len(backward_provider.calls) == 1
            assert list_sound_plans(project_id) == []
    finally:
        clear_config_caches()


def _approved_plan(project_id: str, *, cue_count: int = 1) -> SoundPlan:
    segments = load_segments(project_id)
    cues = [
        SoundCue(
            id=f"cue-{index + 1}",
            segment_id=segments[min(index, len(segments) - 1)].id,
            track=SoundCueTrack.SFX,
            description_cn=f"音效 {index + 1}",
            prompt=f"A short wooden impact number {index + 1}, no speech, no music",
            placement=SemanticPlacement(
                start=SemanticAnchor(segment_id=segments[min(index, len(segments) - 1)].id),
            ),
            requested_duration_seconds=1.0,
            review_status=SoundCueReviewStatus.APPROVED,
        )
        for index in range(cue_count)
    ]
    return bind_sound_plan(
        SoundPlan(id="soundplan-jobs", project_id=project_id, cues=cues),
        segments,
    )


class _SequenceAudioProvider:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        success = self.outcomes.pop(0)
        if not success:
            return AudioGenerationResult(success=False, error="simulated provider failure")
        return AudioGenerationResult(
            success=True,
            audio_bytes=_stereo_wav(duration_seconds=request.duration_seconds),
            content_type="audio/wav",
            sample_rate=44100,
            duration_seconds=request.duration_seconds,
            seed=request.seed,
            model_id=MODEL_ID,
        )


@pytest.mark.parametrize(
    ("outcomes", "expected_status", "completed", "failed", "candidate_statuses"),
    [
        ([True, True], SoundGenerationJobStatus.SUCCEEDED, 2, 0, ["ready", "ready"]),
        ([True, False], SoundGenerationJobStatus.PARTIAL, 1, 1, ["ready", "failed"]),
    ],
)
def test_persistent_sound_job_with_fake_provider_success_and_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[bool],
    expected_status: SoundGenerationJobStatus,
    completed: int,
    failed: int,
    candidate_statuses: list[str],
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            save_sound_plan(project_id, _approved_plan(project_id))
            job = prepare_sound_job(
                project_id,
                "soundplan-jobs",
                ["cue-1"],
                candidate_count=2,
                job_id="sfxjob-integration",
            )
            provider = _SequenceAudioProvider(outcomes)
            run_sound_job(
                project_id,
                job.id,
                GPUOperationLease("test-lease", "sound_generation", project_id),
                provider_factory=lambda: provider,
                manage_worker=False,
            )

            stored_job = load_sound_job(project_id, job.id)
            assert stored_job.status == expected_status
            assert stored_job.completed == completed
            assert stored_job.failed == failed
            stored_plan = load_sound_plan(project_id, "soundplan-jobs")
            candidates = stored_plan.cues[0].candidates
            assert [item.status.value for item in candidates] == candidate_statuses
            assert len(provider.requests) == 2
            for candidate in candidates:
                if candidate.status == SoundCandidateStatus.READY:
                    path = projects / project_id / str(candidate.audio_file)
                    assert path.is_file()
                    assert sha256_file(path) == candidate.audio_sha256
                else:
                    assert candidate.error == "simulated provider failure"
    finally:
        clear_config_caches()


def test_cancelled_and_recovered_jobs_release_persisted_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            cancelled_project, _ = _create_project_with_segments(client, projects, count=1)
            cancelled_plan = _approved_plan(cancelled_project)
            cancelled_plan.id = "soundplan-cancel"
            save_sound_plan(cancelled_project, cancelled_plan)
            cancelled_job = prepare_sound_job(
                cancelled_project,
                cancelled_plan.id,
                ["cue-1"],
                candidate_count=2,
                job_id="sfxjob-cancel",
            )
            request_sound_job_cancel(cancelled_project, cancelled_job.id)
            never_called = _SequenceAudioProvider([True, True])
            run_sound_job(
                cancelled_project,
                cancelled_job.id,
                GPUOperationLease("cancel-lease", "sound_generation", cancelled_project),
                provider_factory=lambda: never_called,
                manage_worker=False,
            )
            stored_cancelled = load_sound_job(cancelled_project, cancelled_job.id)
            assert stored_cancelled.status == SoundGenerationJobStatus.CANCELLED
            assert {item.status for item in stored_cancelled.tasks} == {"cancelled"}
            assert never_called.requests == []
            assert {
                item.status for item in load_sound_plan(cancelled_project, cancelled_plan.id).cues[0].candidates
            } == {SoundCandidateStatus.FAILED}

            recovered_project, _ = _create_project_with_segments(client, projects, count=1)
            recovered_plan = _approved_plan(recovered_project)
            recovered_plan.id = "soundplan-recover"
            save_sound_plan(recovered_project, recovered_plan)
            recovery_job = prepare_sound_job(
                recovered_project,
                recovered_plan.id,
                ["cue-1"],
                candidate_count=2,
                job_id="sfxjob-recover",
            )
            recovery_job.status = SoundGenerationJobStatus.RUNNING
            recovery_job.tasks[0].status = "succeeded"
            recovery_job.tasks[1].status = "running"
            recovery_job.completed = 1
            save_sound_job(recovery_job)
            persisted = load_sound_plan(recovered_project, recovered_plan.id)
            ready_file = candidate_audio_root(recovered_project, ensure=True) / (
                persisted.cues[0].candidates[0].id + ".wav"
            )
            ready_file.write_bytes(_stereo_wav())
            first = persisted.cues[0].candidates[0]
            first.audio_file = str(ready_file.relative_to(projects / recovered_project))
            first.audio_sha256 = sha256_file(ready_file)
            first.duration_seconds = 1.0
            first.status = SoundCandidateStatus.READY
            persisted.cues[0].candidates[1].status = SoundCandidateStatus.GENERATING
            save_sound_plan(recovered_project, persisted)

            recovered = recover_interrupted_sound_jobs(recovered_project)
            assert [item.id for item in recovered] == [recovery_job.id]
            stored_recovered = load_sound_job(recovered_project, recovery_job.id)
            assert stored_recovered.status == SoundGenerationJobStatus.INTERRUPTED
            assert stored_recovered.tasks[0].status == "succeeded"
            assert stored_recovered.tasks[1].status == "failed"
            candidates = load_sound_plan(recovered_project, recovered_plan.id).cues[0].candidates
            assert candidates[0].status == SoundCandidateStatus.READY
            assert candidates[1].status == SoundCandidateStatus.FAILED
            assert candidates[1].error == "interrupted"
    finally:
        clear_config_caches()


def _ready_candidate_plan(project_id: str, candidate_file: Path) -> SoundPlan:
    plan = _approved_plan(project_id)
    plan.id = "soundplan-adopt"
    cue = plan.cues[0]
    provenance = GenerationProvenance(
        provider="test",
        model_id=MODEL_ID,
        model_revision="revision",
        model_license="test license",
        commercial_use=True,
        license_confirmation_id="00000000-0000-4000-8000-000000000001",
        license_snapshot={
            "confirmation_sha256": "c" * 64,
            "license_tier": "community",
        },
        prompt=cue.prompt,
        seed=7,
    )
    candidate = SoundCandidate(
        id="candidate-adopt",
        status=SoundCandidateStatus.READY,
        audio_file=str(candidate_file.relative_to(candidate_file.parents[3])),
        audio_sha256=sha256_file(candidate_file),
        duration_seconds=1.0,
        provenance=provenance,
    )
    return attach_sound_candidate(plan, cue.id, candidate)


def test_candidate_adoption_is_idempotent_and_rolls_back_metadata_and_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            receipt = tmp_path / "license-receipt.json"
            receipt.write_text(json.dumps({
                # Restoring a backup requires a fresh receipt ID.  Equivalent
                # terms and tier must not invalidate immutable provenance.
                "operator_confirmation_id": "00000000-0000-4000-8000-000000000002",
                "confirmation_sha256": "c" * 64,
                "license_tier": "community",
            }), encoding="utf-8")
            monkeypatch.setattr("scripts.model_preflight.has_license_acceptance", lambda *_a, **_k: True)
            monkeypatch.setattr("scripts.model_preflight.license_receipt_path", lambda *_a, **_k: receipt)
            endpoint = (
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1/"
                "candidates/candidate-adopt/adopt"
            )
            first = client.post(endpoint)
            assert first.status_code == 200, first.text
            second = client.post(endpoint)
            assert second.status_code == 200, second.text
            assert second.json()["cues"][0]["adopted_asset_id"] == first.json()["cues"][0]["adopted_asset_id"]
            assert len(load_assets(project_id)) == 1
            assert len(load_settings(project_id).sound_design.clips) == 1
            assert len(list((projects / project_id / "production" / "assets" / "sfx").glob("*.wav"))) == 1
            assert load_assets(project_id)[0].generation is not None
            assert (
                load_assets(project_id)[0].generation.license_confirmation_id
                == "00000000-0000-4000-8000-000000000001"
            )

            rollback_project, _ = _create_project_with_segments(client, projects, count=1)
            rollback_file = candidate_audio_root(rollback_project, ensure=True) / "candidate-adopt.wav"
            rollback_file.write_bytes(_stereo_wav())
            rollback_plan = _ready_candidate_plan(rollback_project, rollback_file)
            save_sound_plan(rollback_project, rollback_plan)
            rollback_endpoint = (
                f"/api/v1/projects/{rollback_project}/sound-plans/{rollback_plan.id}/cues/cue-1/"
                "candidates/candidate-adopt/adopt"
            )
            with patch(
                "dubbing_service.app.routers.sound_generation.save_sound_plan",
                side_effect=RuntimeError("simulated plan write failure"),
            ):
                with pytest.raises(RuntimeError, match="simulated plan write failure"):
                    client.post(rollback_endpoint)
            assert load_assets(rollback_project) == []
            assert load_settings(rollback_project).sound_design.clips == []
            promoted_dir = projects / rollback_project / "production" / "assets" / "sfx"
            assert not promoted_dir.exists() or list(promoted_dir.glob("*.wav")) == []
            assert load_sound_plan(rollback_project, rollback_plan.id).cues[0].adopted_asset_id is None
    finally:
        clear_config_caches()


@pytest.mark.parametrize(
    ("confirmation_sha256", "license_tier"),
    [
        ("d" * 64, "community"),
        ("c" * 64, "enterprise"),
    ],
)
def test_candidate_adoption_rejects_restored_receipt_with_changed_terms_or_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirmation_sha256: str,
    license_tier: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            receipt = tmp_path / "restored-license-receipt.json"
            receipt.write_text(json.dumps({
                "operator_confirmation_id": "00000000-0000-4000-8000-000000000002",
                "confirmation_sha256": confirmation_sha256,
                "license_tier": license_tier,
            }), encoding="utf-8")
            monkeypatch.setattr(
                "scripts.model_preflight.has_license_acceptance",
                lambda *_a, **_k: True,
            )
            monkeypatch.setattr(
                "scripts.model_preflight.license_receipt_path",
                lambda *_a, **_k: receipt,
            )

            blocked = client.post(_adopt_endpoint(project_id, plan.id))

            assert blocked.status_code == 409, blocked.text
            assert load_assets(project_id) == []
            assert load_sound_plan(project_id, plan.id).cues[0].adopted_asset_id is None
    finally:
        clear_config_caches()


@pytest.mark.parametrize("corruption", ["clip_asset_mismatch", "missing_asset_file"])
def test_candidate_adoption_idempotency_rejects_corrupted_production_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            endpoint = _adopt_endpoint(project_id, plan.id)
            adopted = client.post(endpoint)
            assert adopted.status_code == 200, adopted.text
            cue = adopted.json()["cues"][0]
            asset = next(item for item in load_assets(project_id) if item.id == cue["adopted_asset_id"])
            promoted_file = projects / project_id / asset.audio_file

            if corruption == "clip_asset_mismatch":
                settings = load_settings(project_id)
                clip = next(
                    item for item in settings.sound_design.clips
                    if item.id == cue["adopted_clip_id"]
                )
                clip.asset_id = "different-asset"
                save_settings(project_id, settings)
            else:
                promoted_file.unlink()

            repeated = client.post(endpoint)
            assert repeated.status_code == 409, repeated.text
            stored = load_sound_plan(project_id, plan.id).cues[0]
            assert stored.adopted_asset_id == cue["adopted_asset_id"]
            assert stored.adopted_clip_id == cue["adopted_clip_id"]
            assert len(load_assets(project_id)) == 1
            assert len(load_settings(project_id).sound_design.clips) == 1
    finally:
        clear_config_caches()


def test_preview_warns_but_master_blocks_unconfirmed_generated_asset_license(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(
                client,
                projects,
                count=1,
                with_audio=True,
                approve=True,
                confirm_voice_rights=True,
            )
            root = projects / project_id
            asset_path = root / "production" / "assets" / "sfx" / "unlicensed.wav"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(_stereo_wav())
            asset = AudioAsset(
                id="generated-unlicensed",
                track=SoundTrack.SFX,
                name="Unlicensed generated sound",
                audio_file="production/assets/sfx/unlicensed.wav",
                audio_sha256=sha256_file(asset_path),
                duration_seconds=1.0,
                license=AssetLicense(commercial_use=None),
                origin=AudioAssetOrigin.GENERATED,
                generation=GenerationProvenance(
                    provider="test",
                    model_id=MODEL_ID,
                    prompt="an impact",
                    commercial_use=None,
                ),
            )
            save_assets(project_id, [asset])
            save_settings(project_id, ProductionSettings(sound_design=SoundDesign(clips=[
                TimelineClip(
                    id="generated-clip",
                    track=SoundTrack.SFX,
                    asset_id=asset.id,
                    start=0.0,
                    duration=0.5,
                ),
            ])))

            preview_qc = client.get(f"/api/v1/projects/{project_id}/quality-check?purpose=preview")
            assert preview_qc.status_code == 200, preview_qc.text
            preview_issues = {
                item["code"]: item["severity"] for item in preview_qc.json()["issues"]
            }
            assert preview_issues["license_unconfirmed"] == "warning"
            assert preview_issues["generated_license_stale"] == "warning"
            preview = client.post(f"/api/v1/projects/{project_id}/sound-preview")
            assert preview.status_code == 200, preview.text
            assert (root / "production" / "previews" / "sound_mix_latest.wav").is_file()

            master_qc = client.get(f"/api/v1/projects/{project_id}/quality-check?purpose=master")
            assert master_qc.status_code == 200, master_qc.text
            master_issues = {
                item["code"]: item["severity"] for item in master_qc.json()["issues"]
            }
            assert master_issues["license_unconfirmed"] == "error"
            assert master_issues["generated_license_stale"] == "error"
            master = client.post(
                f"/api/v1/projects/{project_id}/masters", json={"name": "must be blocked"},
            )
            assert master.status_code == 400, master.text
            assert "许可" in json.dumps(master.json(), ensure_ascii=False) or "商用" in json.dumps(
                master.json(), ensure_ascii=False,
            )
    finally:
        clear_config_caches()


def test_qc_accepts_restored_receipt_id_but_rejects_changed_terms_or_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(
                client,
                projects,
                count=1,
                with_audio=True,
                approve=True,
                confirm_voice_rights=True,
            )
            historical_receipt_id = "00000000-0000-4000-8000-000000000001"
            restored_receipt_id = "00000000-0000-4000-8000-000000000002"
            confirmation_hash = "c" * 64
            receipt = tmp_path / "restored-license-receipt.json"

            def write_receipt(*, terms_hash: str, tier: str) -> None:
                receipt.write_text(json.dumps({
                    "operator_confirmation_id": restored_receipt_id,
                    "confirmation_sha256": terms_hash,
                    "license_tier": tier,
                }), encoding="utf-8")

            write_receipt(terms_hash=confirmation_hash, tier="community")
            monkeypatch.setattr(
                "scripts.model_preflight.has_license_acceptance",
                lambda *_a, **_k: True,
            )
            monkeypatch.setattr(
                "scripts.model_preflight.license_receipt_path",
                lambda *_a, **_k: receipt,
            )

            root = projects / project_id
            asset_path = root / "production" / "assets" / "sfx" / "old-receipt.wav"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(_stereo_wav())
            asset_hash = sha256_file(asset_path)
            asset = AudioAsset(
                id="generated-old-receipt",
                track=SoundTrack.SFX,
                name="generated under earlier confirmation",
                audio_file=str(asset_path.relative_to(root)),
                audio_sha256=asset_hash,
                duration_seconds=1.0,
                license=AssetLicense(commercial_use=True),
                origin=AudioAssetOrigin.GENERATED,
                generation=GenerationProvenance(
                    provider="test",
                    model_id=MODEL_ID,
                    prompt="a wooden impact",
                    commercial_use=True,
                    license_confirmation_id=historical_receipt_id,
                    license_snapshot={
                        "confirmation_sha256": confirmation_hash,
                        "license_tier": "community",
                    },
                    audio_sha256=asset_hash,
                ),
            )
            save_assets(project_id, [asset])
            save_settings(project_id, ProductionSettings(sound_design=SoundDesign(clips=[
                TimelineClip(
                    id="old-receipt-clip",
                    track=SoundTrack.SFX,
                    asset_id=asset.id,
                    start=0.0,
                    duration=0.5,
                ),
            ])))

            restored = client.get(
                f"/api/v1/projects/{project_id}/quality-check?purpose=master",
            )
            assert restored.status_code == 200, restored.text
            assert "generated_license_stale" not in {
                item["code"] for item in restored.json()["issues"]
            }
            assert load_assets(project_id)[0].generation is not None
            assert (
                load_assets(project_id)[0].generation.license_confirmation_id
                == historical_receipt_id
            )

            for terms_hash, tier in (
                ("d" * 64, "community"),
                (confirmation_hash, "enterprise"),
            ):
                write_receipt(terms_hash=terms_hash, tier=tier)
                formal = client.get(
                    f"/api/v1/projects/{project_id}/quality-check?purpose=master",
                )
                assert formal.status_code == 200, formal.text
                issues = {
                    item["code"]: item["severity"] for item in formal.json()["issues"]
                }
                assert issues["generated_license_stale"] == "error"
    finally:
        clear_config_caches()


def _failed_candidate_plan(project_id: str, candidate_file: Path, *, plan_id: str) -> SoundPlan:
    plan = _approved_plan(project_id)
    plan.id = plan_id
    cue = plan.cues[0]
    candidate = SoundCandidate(
        id=f"candidate-{plan_id}",
        status=SoundCandidateStatus.FAILED,
        audio_file=str(candidate_file.relative_to(candidate_file.parents[3])),
        audio_sha256=sha256_file(candidate_file),
        duration_seconds=1.0,
        provenance=GenerationProvenance(
            provider="test", model_id=MODEL_ID, prompt=cue.prompt, seed=9,
        ),
        error="kept for retry",
    )
    return attach_sound_candidate(plan, cue.id, candidate)


def test_delete_candidate_rolls_file_back_when_plan_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-delete.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _failed_candidate_plan(project_id, candidate_file, plan_id="soundplan-delete-candidate")
            save_sound_plan(project_id, plan)
            candidate_id = plan.cues[0].candidates[0].id
            endpoint = (
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1/"
                f"candidates/{candidate_id}"
            )
            with patch(
                "dubbing_service.app.routers.sound_generation.save_sound_plan",
                side_effect=SoundGenerationStorageError("simulated write failure"),
            ):
                with pytest.raises(SoundGenerationStorageError, match="simulated write failure"):
                    client.delete(endpoint)
            assert candidate_file.is_file()
            stored = load_sound_plan(project_id, plan.id)
            assert stored.cues[0].candidates[0].id == candidate_id

            deleted = client.delete(endpoint)
            assert deleted.status_code == 200, deleted.text
            assert candidate_file.is_file() is False
            assert load_sound_plan(project_id, plan.id).cues[0].candidates == []
    finally:
        clear_config_caches()


def test_delete_plan_cleans_all_candidate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidates_dir = candidate_audio_root(project_id, ensure=True)
            first_file = candidates_dir / "candidate-plan-first.wav"
            second_file = candidates_dir / "candidate-plan-second.wav"
            first_file.write_bytes(_stereo_wav())
            second_file.write_bytes(_stereo_wav())
            plan = _failed_candidate_plan(project_id, first_file, plan_id="soundplan-delete-all")
            cue = plan.cues[0]
            second = SoundCandidate(
                id="candidate-second",
                status=SoundCandidateStatus.FAILED,
                audio_file=str(second_file.relative_to(projects / project_id)),
                audio_sha256=sha256_file(second_file),
                duration_seconds=1.0,
                provenance=GenerationProvenance(
                    provider="test", model_id=MODEL_ID, prompt=cue.prompt, seed=10,
                ),
                cue_fingerprint=cue.candidates[0].cue_fingerprint,
                error="failed",
            )
            cue.candidates.append(second)
            save_sound_plan(project_id, plan)

            deleted = client.delete(
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}",
            )
            assert deleted.status_code == 200, deleted.text
            assert deleted.json() == {"deleted": True}
            assert first_file.is_file() is False
            assert second_file.is_file() is False
            assert list_sound_plans(project_id) == []
    finally:
        clear_config_caches()


def test_get_sound_plan_refreshes_stale_without_writing_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, segments = _create_project_with_segments(client, projects, count=1)
            plan = _approved_plan(project_id)
            plan.id = "soundplan-read-only"
            save_sound_plan(project_id, plan)
            plan_file = (
                projects / project_id / "production" / "sound_generation" / "plans" /
                f"{plan.id}.json"
            )
            before_bytes = plan_file.read_bytes()
            before_mtime = plan_file.stat().st_mtime_ns
            changed = client.patch(
                f"/api/v1/projects/{project_id}/segments/{segments[0]['id']}",
                json={"text": "GET 需要返回 stale，但不能落盘。"},
            )
            assert changed.status_code == 200, changed.text

            with patch(
                "dubbing_service.app.routers.sound_generation.save_sound_plan",
                side_effect=AssertionError("GET must not persist"),
            ):
                response = client.get(
                    f"/api/v1/projects/{project_id}/sound-plans/{plan.id}",
                )
            assert response.status_code == 200, response.text
            assert response.json()["stale"] is True
            assert response.json()["cues"][0]["stale"] is True
            assert plan_file.read_bytes() == before_bytes
            assert plan_file.stat().st_mtime_ns == before_mtime
            assert load_sound_plan(project_id, plan.id).stale is False
    finally:
        clear_config_caches()


def test_sound_prompt_contract_rejects_more_than_one_thousand_characters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            plan = _approved_plan(project_id)
            plan.id = "soundplan-prompt-limit"
            save_sound_plan(project_id, plan)
            too_long = "a" * 1001
            response = client.patch(
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1",
                json={"prompt": too_long},
            )
            assert response.status_code == 422, response.text
            assert load_sound_plan(project_id, plan.id).cues[0].prompt != too_long

            provider = _PlannerProvider([{"cues": [_planner_cue(
                load_segments(project_id)[0].id,
                prompt_en=too_long,
            )]}])
            monkeypatch.setattr(
                "dubbing_service.app.routers.sound_generation._provider_from_config",
                lambda _config_id: (provider, "fake", "fake"),
            )
            drafted = client.post(
                f"/api/v1/projects/{project_id}/sound-plans", json={},
            )
            assert drafted.status_code == 502, drafted.text
            assert len(provider.calls) == 2
    finally:
        clear_config_caches()


def test_prepare_job_rolls_back_queued_candidates_when_job_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            plan = _approved_plan(project_id)
            plan.id = "soundplan-prepare-rollback"
            save_sound_plan(project_id, plan)
            with patch(
                "dubbing_service.app.services.sound_jobs.save_sound_job",
                side_effect=RuntimeError("simulated job write failure"),
            ):
                with pytest.raises(RuntimeError, match="simulated job write failure"):
                    prepare_sound_job(
                        project_id,
                        plan.id,
                        ["cue-1"],
                        candidate_count=2,
                        job_id="sfxjob-prepare-rollback",
                    )
            stored = load_sound_plan(project_id, plan.id)
            assert stored.cues[0].candidates == []
            assert stored.cues[0].selected_candidate_id is None
    finally:
        clear_config_caches()


def test_runner_outer_failure_finalizes_candidates_and_releases_gpu_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            plan = _approved_plan(project_id)
            plan.id = "soundplan-outer-failure"
            save_sound_plan(project_id, plan)
            job = prepare_sound_job(
                project_id,
                plan.id,
                ["cue-1"],
                candidate_count=2,
                job_id="sfxjob-outer-failure",
            )
            released: list[GPUOperationLease] = []
            monkeypatch.setattr(
                "dubbing_service.app.services.sound_jobs.release_gpu_operation",
                lambda lease: released.append(lease),
            )
            lease = GPUOperationLease("outer-failure-lease", "sound_generation", project_id)
            run_sound_job(
                project_id,
                job.id,
                lease,
                provider_factory=lambda: (_ for _ in ()).throw(RuntimeError("provider init failed")),
                manage_worker=False,
            )
            stored_job = load_sound_job(project_id, job.id)
            assert stored_job.status == SoundGenerationJobStatus.FAILED
            assert "provider init failed" in (stored_job.error or "")
            assert {task.status for task in stored_job.tasks} == {"failed"}
            stored_plan = load_sound_plan(project_id, plan.id)
            assert {
                candidate.status for candidate in stored_plan.cues[0].candidates
            } == {SoundCandidateStatus.FAILED}
            assert released == [lease]
    finally:
        clear_config_caches()


def _patch_license_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    receipt = tmp_path / "license-receipt.json"
    receipt.write_text(json.dumps({
        "operator_confirmation_id": "00000000-0000-4000-8000-000000000001",
        "confirmation_sha256": "c" * 64,
        "license_tier": "community",
    }), encoding="utf-8")
    monkeypatch.setattr("scripts.model_preflight.has_license_acceptance", lambda *_a, **_k: True)
    monkeypatch.setattr("scripts.model_preflight.license_receipt_path", lambda *_a, **_k: receipt)


def _adopt_endpoint(project_id: str, plan_id: str) -> str:
    return (
        f"/api/v1/projects/{project_id}/sound-plans/{plan_id}/cues/cue-1/"
        "candidates/candidate-adopt/adopt"
    )


def test_adopted_cue_semantic_and_mix_edits_synchronize_timeline_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, segments = _create_project_with_segments(client, projects, count=2)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            clip_id = adopted.json()["cues"][0]["adopted_clip_id"]

            updated = client.patch(
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1",
                json={
                    "description_cn": "移到第二句话的门声",
                    "placement": {
                        "start": {
                            "segment_id": segments[1]["id"],
                            "edge": "start",
                            "offset_seconds": 0.25,
                        },
                    },
                    "playback_duration_seconds": 0.75,
                    "gain_db": -4.0,
                    "fade_in": 0.1,
                    "fade_out": 0.2,
                },
            )
            assert updated.status_code == 200, updated.text
            cue = updated.json()["cues"][0]
            assert cue["segment_id"] == segments[1]["id"]
            assert cue["selected_candidate_id"] == "candidate-adopt"
            assert cue["review_status"] == "approved"
            clip = next(
                item for item in load_settings(project_id).sound_design.clips if item.id == clip_id
            )
            assert clip.semantic_placement == SemanticPlacement(
                start=SemanticAnchor(
                    segment_id=segments[1]["id"], offset_seconds=0.25,
                ),
            )
            assert clip.duration == pytest.approx(0.75)
            assert clip.gain_db == pytest.approx(-4.0)
            assert clip.fade_in == pytest.approx(0.1)
            assert clip.fade_out == pytest.approx(0.2)
    finally:
        clear_config_caches()


def test_unadopt_atomically_removes_production_objects_and_allows_readoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            adopted_cue = adopted.json()["cues"][0]
            asset = next(item for item in load_assets(project_id) if item.id == adopted_cue["adopted_asset_id"])
            promoted_file = projects / project_id / asset.audio_file
            assert promoted_file.is_file()

            endpoint = (
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1/adoption"
            )
            master = AudioMaster(
                id="master-block-unadopt",
                project_id=project_id,
                name="frozen generated sound",
                audio_file="audio/masters/master-block-unadopt.wav",
                audio_sha256="0" * 64,
                input_fingerprint="f" * 64,
                sound_design_snapshot={
                    "clips": [{"id": adopted_cue["adopted_clip_id"], "asset_id": asset.id}],
                },
            )
            save_master(master)
            blocked = client.delete(endpoint)
            assert blocked.status_code == 409, blocked.text
            assert "Master" in json.dumps(blocked.json(), ensure_ascii=False)
            assert load_sound_plan(project_id, plan.id).cues[0].adopted_asset_id == asset.id
            assert [item.id for item in load_assets(project_id)] == [asset.id]
            assert [
                item.id for item in load_settings(project_id).sound_design.clips
            ] == [adopted_cue["adopted_clip_id"]]
            assert promoted_file.is_file()

            (masters_dir(project_id) / f"{master.id}.json").unlink()
            substring_master = master.model_copy(
                update={
                    "id": "master-substring-only",
                    "sound_design_snapshot": {
                        "clips": [{"asset_id": f"{asset.id}-different"}],
                    },
                },
            )
            save_master(substring_master)
            removed = client.delete(endpoint)
            assert removed.status_code == 200, removed.text
            (masters_dir(project_id) / f"{substring_master.id}.json").unlink()
            cue = removed.json()["cues"][0]
            assert cue["selected_candidate_id"] is None
            assert cue["adopted_asset_id"] is None
            assert cue["adopted_clip_id"] is None
            assert cue["candidates"][0]["status"] == "ready"
            assert load_assets(project_id) == []
            assert load_settings(project_id).sound_design.clips == []
            assert promoted_file.is_file() is False
            assert candidate_file.is_file() is True

            adopted_again = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted_again.status_code == 200, adopted_again.text
            assert adopted_again.json()["cues"][0]["adopted_asset_id"] is not None

            with patch(
                "dubbing_service.app.routers.sound_generation.save_sound_plan",
                side_effect=SoundGenerationStorageError("unadopt plan failure"),
            ):
                with pytest.raises(SoundGenerationStorageError, match="unadopt plan failure"):
                    client.delete(endpoint)
            # Compensating rollback restores metadata and promoted file.
            assert len(load_assets(project_id)) == 1
            assert len(load_settings(project_id).sound_design.clips) == 1
            current_asset = load_assets(project_id)[0]
            assert (projects / project_id / current_asset.audio_file).is_file()
            assert load_sound_plan(project_id, plan.id).cues[0].adopted_asset_id == current_asset.id
    finally:
        clear_config_caches()


@pytest.mark.parametrize("staging_state", ["missing", "hash_mismatch"])
def test_unadopt_rebuilds_an_invalid_staging_candidate_before_removing_formal_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staging_state: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            expected_candidate_sha = sha256_file(candidate_file)
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            adopted_cue = adopted.json()["cues"][0]
            formal_asset = next(
                item for item in load_assets(project_id)
                if item.id == adopted_cue["adopted_asset_id"]
            )
            formal_file = projects / project_id / formal_asset.audio_file
            assert formal_file.is_file()
            assert sha256_file(formal_file) == expected_candidate_sha

            if staging_state == "missing":
                candidate_file.unlink()
            else:
                candidate_file.write_bytes(_stereo_wav(duration_seconds=1.1))
                assert sha256_file(candidate_file) != expected_candidate_sha

            endpoint = (
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}/cues/cue-1/adoption"
            )
            removed = client.delete(endpoint)
            assert removed.status_code == 200, removed.text
            cue = removed.json()["cues"][0]
            candidate = cue["candidates"][0]
            assert candidate["status"] == "ready"
            assert candidate["audio_sha256"] == expected_candidate_sha
            assert candidate_file.is_file()
            assert sha256_file(candidate_file) == expected_candidate_sha
            assert formal_file.is_file() is False
            assert load_assets(project_id) == []
            assert load_settings(project_id).sound_design.clips == []

            # The stated unadopt contract promises a reusable staging result,
            # not merely a READY metadata label pointing at missing/corrupt bytes.
            adopted_again = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted_again.status_code == 200, adopted_again.text
    finally:
        clear_config_caches()


@pytest.mark.parametrize(
    "inconsistency",
    [
        "missing_selected_candidate",
        "candidate_not_adopted",
        "orphan_adopted_candidate",
        "clip_mix_drift",
    ],
)
def test_qc_reports_adoption_inconsistency_with_purpose_specific_severity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inconsistency: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(
                client,
                projects,
                count=1,
                with_audio=True,
                approve=True,
                confirm_voice_rights=True,
            )
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text

            if inconsistency in {
                "missing_selected_candidate",
                "candidate_not_adopted",
                "orphan_adopted_candidate",
            }:
                damaged = load_sound_plan(project_id, plan.id)
                cue = damaged.cues[0]
                if inconsistency == "missing_selected_candidate":
                    cue.selected_candidate_id = None
                elif inconsistency == "candidate_not_adopted":
                    cue.candidates[0].status = SoundCandidateStatus.READY
                else:
                    cue.selected_candidate_id = None
                    cue.adopted_asset_id = None
                    cue.adopted_clip_id = None
                save_sound_plan(project_id, damaged)
            else:
                settings = load_settings(project_id)
                settings.sound_design.clips[0].gain_db = -4.0
                save_settings(project_id, settings)

            expected = {
                "preview": "warning",
                "master": "error",
                "release": "error",
            }
            for purpose, severity in expected.items():
                response = client.get(
                    f"/api/v1/projects/{project_id}/quality-check?purpose={purpose}",
                )
                assert response.status_code == 200, response.text
                matching = [
                    item for item in response.json()["issues"]
                    if item["code"] == "sound_adoption_inconsistent"
                ]
                assert matching, response.text
                assert {item["severity"] for item in matching} == {severity}
    finally:
        clear_config_caches()


@pytest.mark.parametrize(
    "mutation",
    [
        "remove",
        "replace_asset",
        "semantic_placement",
        "start",
        "duration",
        "gain",
        "fades",
        "loop",
        "source_offset",
    ],
)
def test_sound_design_put_cannot_mutate_an_adopted_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            cue = adopted.json()["cues"][0]
            adopted_asset_id = cue["adopted_asset_id"]
            adopted_clip_id = cue["adopted_clip_id"]

            current_sound_design = load_settings(project_id).sound_design
            current_clip = next(
                item.model_copy(deep=True)
                for item in current_sound_design.clips
                if item.id == adopted_clip_id
            )
            sound_design = current_sound_design.model_dump(mode="json")
            if mutation == "remove":
                sound_design["clips"] = []
            elif mutation == "replace_asset":
                replacement_file = (
                    projects / project_id / "production" / "assets" / "sfx" / "replacement.wav"
                )
                replacement_file.parent.mkdir(parents=True, exist_ok=True)
                replacement_file.write_bytes(_stereo_wav())
                replacement = AudioAsset(
                    id="replacement-asset",
                    track=SoundTrack.SFX,
                    name="replacement",
                    audio_file=str(replacement_file.relative_to(projects / project_id)),
                    audio_sha256=sha256_file(replacement_file),
                    duration_seconds=1.0,
                    license=AssetLicense(commercial_use=True),
                )
                save_assets(project_id, [*load_assets(project_id), replacement])
                sound_design["clips"][0]["asset_id"] = replacement.id
            elif mutation == "semantic_placement":
                sound_design["clips"][0]["semantic_placement"]["start"][
                    "offset_seconds"
                ] = 0.25
            elif mutation == "start":
                sound_design["clips"][0]["start"] = 0.25
            elif mutation == "duration":
                sound_design["clips"][0]["duration"] = 0.75
            elif mutation == "gain":
                sound_design["clips"][0]["gain_db"] = -4.0
            elif mutation == "fades":
                sound_design["clips"][0]["fade_in"] = 0.1
                sound_design["clips"][0]["fade_out"] = 0.2
            elif mutation == "loop":
                sound_design["clips"][0]["loop"] = True
            elif mutation == "source_offset":
                sound_design["clips"][0]["source_offset"] = 0.1

            response = client.put(
                f"/api/v1/projects/{project_id}/production/sound-design",
                json=sound_design,
            )
            assert response.status_code == 409, response.text
            persisted = load_settings(project_id).sound_design.clips
            adopted_clip = next(item for item in persisted if item.id == adopted_clip_id)
            assert adopted_clip == current_clip
            stored_cue = load_sound_plan(project_id, plan.id).cues[0]
            assert stored_cue.adopted_asset_id == adopted_asset_id
            assert stored_cue.adopted_clip_id == adopted_clip_id
    finally:
        clear_config_caches()


def test_asset_delete_still_blocks_a_soundplan_adoption_when_timeline_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            adopted_asset_id = adopted.json()["cues"][0]["adopted_asset_id"]
            asset = next(item for item in load_assets(project_id) if item.id == adopted_asset_id)
            asset_file = projects / project_id / asset.audio_file

            # Simulate legacy or partially corrupted state: SoundPlan remains the
            # authoritative adoption record even when its TimelineClip vanished.
            settings = load_settings(project_id)
            settings.sound_design.clips = []
            save_settings(project_id, settings)
            response = client.delete(
                f"/api/v1/projects/{project_id}/production/assets/{adopted_asset_id}",
            )
            assert response.status_code == 409, response.text
            assert any(item.id == adopted_asset_id for item in load_assets(project_id))
            assert asset_file.is_file()
            assert load_sound_plan(project_id, plan.id).cues[0].adopted_asset_id == adopted_asset_id
    finally:
        clear_config_caches()


def test_asset_delete_master_reference_matching_is_not_a_string_substring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            asset_file = projects / project_id / "production" / "assets" / "sfx" / "short.wav"
            asset_file.parent.mkdir(parents=True, exist_ok=True)
            asset_file.write_bytes(_stereo_wav())
            asset = AudioAsset(
                id="asset-short",
                track=SoundTrack.SFX,
                name="short id",
                audio_file=str(asset_file.relative_to(projects / project_id)),
                audio_sha256=sha256_file(asset_file),
                duration_seconds=1.0,
                license=AssetLicense(commercial_use=True),
            )
            save_assets(project_id, [asset])
            save_master(AudioMaster(
                id="master-substring-reference",
                project_id=project_id,
                name="different asset",
                audio_file="audio/masters/master-substring-reference.wav",
                audio_sha256="0" * 64,
                input_fingerprint="f" * 64,
                sound_design_snapshot={
                    "clips": [{"asset_id": "asset-short-but-different"}],
                },
            ))

            response = client.delete(
                f"/api/v1/projects/{project_id}/production/assets/{asset.id}",
            )
            assert response.status_code == 200, response.text
            assert load_assets(project_id) == []
            assert asset_file.is_file() is False
    finally:
        clear_config_caches()


def _save_manual_sound_asset(
    projects: Path,
    project_id: str,
    *,
    asset_id: str,
) -> tuple[AudioAsset, Path]:
    asset_file = projects / project_id / "production" / "assets" / "sfx" / f"{asset_id}.wav"
    asset_file.parent.mkdir(parents=True, exist_ok=True)
    asset_file.write_bytes(_stereo_wav())
    asset = AudioAsset(
        id=asset_id,
        track=SoundTrack.SFX,
        name=asset_id,
        audio_file=str(asset_file.relative_to(projects / project_id)),
        audio_sha256=sha256_file(asset_file),
        duration_seconds=1.0,
        license=AssetLicense(commercial_use=True),
    )
    save_assets(project_id, [asset])
    return asset, asset_file


def test_asset_delete_keeps_metadata_when_the_registered_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            asset, asset_file = _save_manual_sound_asset(
                projects,
                project_id,
                asset_id="asset-missing-file",
            )
            asset_file.unlink()

            response = client.delete(
                f"/api/v1/projects/{project_id}/production/assets/{asset.id}",
            )
            assert response.status_code in {404, 409}, response.text
            assert [item.id for item in load_assets(project_id)] == [asset.id]
    finally:
        clear_config_caches()


def test_asset_delete_restores_file_when_metadata_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            asset, asset_file = _save_manual_sound_asset(
                projects,
                project_id,
                asset_id="asset-delete-rollback",
            )
            with patch(
                "dubbing_service.app.routers.production.save_assets",
                side_effect=RuntimeError("simulated asset metadata failure"),
            ):
                with pytest.raises(RuntimeError, match="simulated asset metadata failure"):
                    client.delete(
                        f"/api/v1/projects/{project_id}/production/assets/{asset.id}",
                    )
            assert [item.id for item in load_assets(project_id)] == [asset.id]
            assert asset_file.is_file()
            assert sha256_file(asset_file) == asset.audio_sha256
    finally:
        clear_config_caches()


def test_partial_adoption_pointer_blocks_sound_design_and_asset_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text
            cue_payload = adopted.json()["cues"][0]
            adopted_asset_id = cue_payload["adopted_asset_id"]
            adopted_clip_id = cue_payload["adopted_clip_id"]
            asset = next(item for item in load_assets(project_id) if item.id == adopted_asset_id)
            asset_file = projects / project_id / asset.audio_file

            damaged = load_sound_plan(project_id, plan.id)
            damaged.cues[0].adopted_clip_id = None
            save_sound_plan(project_id, damaged)
            sound_design = load_settings(project_id).sound_design.model_dump(mode="json")
            sound_design["clips"] = []
            design_response = client.put(
                f"/api/v1/projects/{project_id}/production/sound-design",
                json=sound_design,
            )
            assert design_response.status_code == 409, design_response.text
            assert [
                item.id for item in load_settings(project_id).sound_design.clips
            ] == [adopted_clip_id]

            # Even without the clip pointer, the remaining adopted asset pointer
            # is authoritative and must protect metadata plus the promoted WAV.
            settings = load_settings(project_id)
            settings.sound_design.clips = []
            save_settings(project_id, settings)
            delete_response = client.delete(
                f"/api/v1/projects/{project_id}/production/assets/{adopted_asset_id}",
            )
            assert delete_response.status_code == 409, delete_response.text
            assert any(item.id == adopted_asset_id for item in load_assets(project_id))
            assert asset_file.is_file()
    finally:
        clear_config_caches()


@pytest.mark.parametrize("damage", ["half_pointer", "orphan_adopted_candidate"])
def test_adoption_damage_blocks_plan_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id, _ = _create_project_with_segments(client, projects, count=1)
            candidate_file = candidate_audio_root(project_id, ensure=True) / "candidate-adopt.wav"
            candidate_file.write_bytes(_stereo_wav())
            plan = _ready_candidate_plan(project_id, candidate_file)
            save_sound_plan(project_id, plan)
            _patch_license_receipt(monkeypatch, tmp_path)
            adopted = client.post(_adopt_endpoint(project_id, plan.id))
            assert adopted.status_code == 200, adopted.text

            damaged = load_sound_plan(project_id, plan.id)
            cue = damaged.cues[0]
            if damage == "half_pointer":
                cue.adopted_asset_id = None
            else:
                cue.selected_candidate_id = None
                cue.adopted_asset_id = None
                cue.adopted_clip_id = None
                assert cue.candidates[0].status == SoundCandidateStatus.ADOPTED
            save_sound_plan(project_id, damaged)
            response = client.delete(
                f"/api/v1/projects/{project_id}/sound-plans/{plan.id}",
            )
            assert response.status_code == 409, response.text
            persisted_cue = load_sound_plan(project_id, plan.id).cues[0]
            if damage == "half_pointer":
                assert persisted_cue.adopted_clip_id is not None
            else:
                assert persisted_cue.candidates[0].status == SoundCandidateStatus.ADOPTED
            assert candidate_file.is_file()
            assert len(load_assets(project_id)) == 1
            assert len(load_settings(project_id).sound_design.clips) == 1
    finally:
        clear_config_caches()
