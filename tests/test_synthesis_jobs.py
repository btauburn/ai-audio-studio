from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.storage.jobs import get_job
from tests.audio_test_utils import make_wav_bytes


def _create_project_segment(client: TestClient) -> tuple[str, str]:
    project = client.post(
        "/api/v1/projects",
        json={"name": "persistent-job", "description": ""},
    )
    assert project.status_code == 200
    project_id = project.json()["id"]
    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={
            "json_text": json.dumps(
                [{"speaker_name": "旁白", "text": "任务测试", "emotion_description": "平静"}],
                ensure_ascii=False,
            )
        },
    )
    assert imported.status_code == 200
    return project_id, imported.json()["segments"][0]["id"]


def _create_job(client: TestClient, project_id: str, segment_id: str, job_id: str = "job00001"):
    return client.put(
        f"/api/v1/projects/{project_id}/jobs/{job_id}",
        json={
            "id": job_id,
            "job_type": "single_synthesize",
            "project_name": "persistent-job",
            "segment_ids": [segment_id],
            "segment_backends": {segment_id: "qwen3tts"},
        },
    )


def test_job_survives_app_recreation_and_running_becomes_interrupted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id = _create_project_segment(client)
        created = _create_job(client, project_id, segment_id)
        assert created.status_code == 200
        assert created.json()["status"] == "queued"
        started = client.post(f"/api/v1/projects/{project_id}/jobs/job00001/start")
        assert started.status_code == 200
        assert started.json()["status"] == "running"

    with TestClient(create_app()) as client:
        listed = client.get(f"/api/v1/projects/{project_id}/jobs")
        assert listed.status_code == 200
        assert listed.json()[0]["status"] == "interrupted"
        resumed = client.post(f"/api/v1/projects/{project_id}/jobs/job00001/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "queued"

    clear_config_caches()


def test_job_progress_is_durable_and_terminal_jobs_are_not_reopened(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id = _create_project_segment(client)
        assert _create_job(client, project_id, segment_id).status_code == 200
        result = client.post(
            f"/api/v1/projects/{project_id}/jobs/job00001/segments/{segment_id}/result",
            json={"success": True},
        )
        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "succeeded"
        assert data["done"] == 1
        assert data["remaining_segment_ids"] == []
        reopened = client.post(f"/api/v1/projects/{project_id}/jobs/job00001/start")
        assert reopened.status_code == 409

    persisted = get_job(project_id, "job00001")
    assert persisted is not None
    assert persisted.status.value == "succeeded"
    clear_config_caches()


def test_rest_synthesis_records_persistent_job_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id = _create_project_segment(client)
        updated = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}",
            json={"tts_backend": "qwen3tts"},
        )
        assert updated.status_code == 200
        assert _create_job(client, project_id, segment_id).status_code == 200

        audio_rel = f"audio/segments/{segment_id}_jobtake01.wav"
        audio_path = projects_root / project_id / audio_rel
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(make_wav_bytes())
        fake_result = {
            "success": True,
            "audio_file": audio_rel,
            "audio_sha256": "abc",
            "take_id": "jobtake01",
            "slot": "a",
            "warning": None,
        }
        with patch(
            "dubbing_service.app.routers.synthesis._synthesize_one",
            return_value=fake_result,
        ):
            synthesized = client.post(
                f"/api/v1/projects/{project_id}/segments/synthesize",
                json={
                    "segment_ids": [segment_id],
                    "retry_failed": True,
                    "job_id": "job00001",
                },
            )
        assert synthesized.status_code == 200, synthesized.text
        assert synthesized.json()["results"][0]["success"]
        job = client.get(f"/api/v1/projects/{project_id}/jobs").json()[0]
        assert job["status"] == "succeeded"
        assert job["done"] == 1

    clear_config_caches()
