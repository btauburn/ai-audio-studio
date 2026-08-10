from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from tests.audio_test_utils import make_wav_bytes


def test_review_requires_take_and_unresolved_issues_block_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={"name": "review", "description": ""},
        ).json()["id"]
        imported = client.post(
            f"/api/v1/projects/{project_id}/segments/import-json",
            json={
                "json_text": json.dumps(
                    [{"speaker_name": "旁白", "text": "审听文本", "emotion_description": "平静"}],
                    ensure_ascii=False,
                )
            },
        )
        segment_id = imported.json()["segments"][0]["id"]
        rejected = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review",
            json={"review_status": "approved"},
        )
        assert rejected.status_code == 400

        audio_path = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(make_wav_bytes())
        segment = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        segment["audio_file"] = f"audio/segments/{segment_id}.wav"
        segment["status"] = "done"
        restored = client.put(
            f"/api/v1/projects/{project_id}/segments/restore",
            json={"segments": [segment]},
        )
        assert restored.status_code == 200

        issue = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review-issues",
            json={"category": "pronunciation", "note": "人名读音错误"},
        )
        assert issue.status_code == 200
        issue_data = issue.json()
        assert issue_data["review_status"] == "needs_revision"
        issue_id = issue_data["review_issues"][0]["id"]
        blocked = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review",
            json={"review_status": "approved"},
        )
        assert blocked.status_code == 400

        resolved = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review-issues/{issue_id}",
            json={"resolved": True},
        )
        assert resolved.status_code == 200
        assert resolved.json()["review_status"] == "to_review"
        approved = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/review",
            json={"review_status": "approved", "note": "复听通过"},
        )
        assert approved.status_code == 200
        assert approved.json()["review_status"] == "approved"

        edited = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio",
            json={"speed": 1.05, "volume": 1.0},
        )
        assert edited.status_code == 200
        assert edited.json()["review_status"] == "to_review"

    clear_config_caches()
