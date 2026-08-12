from __future__ import annotations

import io
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from tests.audio_test_utils import make_wav_bytes


def test_master_requires_approval_and_freezes_reviewed_take(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={"name": "master", "description": ""},
        ).json()["id"]
        imported = client.post(
            f"/api/v1/projects/{project_id}/segments/import-json",
            json={
                "json_text": json.dumps(
                    [{"speaker_name": "旁白", "text": "母带测试", "emotion_description": "平静"}],
                    ensure_ascii=False,
                )
            },
        )
        segment_id = imported.json()["segments"][0]["id"]
        rejected = client.post(
            f"/api/v1/projects/{project_id}/masters",
            json={"name": "正式版"},
        )
        assert rejected.status_code == 400

        audio_path = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(make_wav_bytes(sample_value=500, frames=3200))
        segment = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        segment["audio_file"] = f"audio/segments/{segment_id}.wav"
        segment["status"] = "done"
        restored = client.put(
            f"/api/v1/projects/{project_id}/segments/restore",
            json={"segments": [segment]},
        )
        assert restored.status_code == 200
        approved = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review",
            json={"review_status": "approved"},
        )
        assert approved.status_code == 200
        role_id = approved.json()["role_id"]
        rights = client.patch(
            f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
            json={
                "voice_source": "self",
                "voice_license_name": "owned",
                "voice_commercial_use": True,
                "voice_rights_note": "recorded by owner",
            },
        )
        assert rights.status_code == 200
        selected_take_id = approved.json()["selected_take_id"]

        created = client.post(
            f"/api/v1/projects/{project_id}/masters",
            json={"name": "正式版"},
        )
        assert created.status_code == 200, created.text
        master = created.json()
        assert master["name"] == "正式版"
        assert master["segments"][0]["take_id"] == selected_take_id
        master_path = projects_root / project_id / master["audio_file"]
        assert master_path.is_file()
        with wave.open(io.BytesIO(master_path.read_bytes()), "rb") as audio:
            assert audio.getframerate() == 48000
            assert audio.getsampwidth() == 3

        edited = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio",
            json={"speed": 1.1, "volume": 1.0},
        )
        assert edited.status_code == 200
        blocked = client.post(
            f"/api/v1/projects/{project_id}/masters",
            json={"name": "未复听版"},
        )
        assert blocked.status_code == 400
        download = client.get(
            f"/api/v1/projects/{project_id}/masters/{master['id']}/audio"
        )
        assert download.status_code == 200
        assert download.content == master_path.read_bytes()

    clear_config_caches()
