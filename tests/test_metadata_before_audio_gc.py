from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.enums import SegmentStatus
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import (
    SegmentStorageError,
    load_segments,
    save_segments,
)
from tests.audio_test_utils import make_wav_bytes


def _create_project_with_segment(client: TestClient, text: str = "旧台词") -> tuple[str, str, str]:
    created = client.post("/api/v1/projects", json={"name": "transaction-test"})
    assert created.status_code == 200
    project_id = created.json()["id"]
    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={
            "json_text": json.dumps(
                [{"speaker_name": "角色A", "text": text, "emotion_description": "平静"}],
                ensure_ascii=False,
            )
        },
    )
    assert imported.status_code == 200
    segment = imported.json()["segments"][0]
    return project_id, segment["id"], segment["role_id"]


def _attach_audio(projects_root: Path, project_id: str, segment_id: str) -> Path:
    audio = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"original-audio")
    segments = load_segments(project_id)
    segments[0].status = SegmentStatus.DONE
    segments[0].audio_file = f"audio/segments/{segment_id}.wav"
    save_segments(project_id, segments)
    return audio


def test_segment_update_keeps_audio_when_metadata_save_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id, _ = _create_project_with_segment(client)
        audio = _attach_audio(tmp_path, project_id, segment_id)

        with patch(
            "dubbing_service.app.routers.segments.save_segments",
            side_effect=SegmentStorageError("simulated disk failure"),
        ):
            response = client.patch(
                f"/api/v1/projects/{project_id}/segments/{segment_id}",
                json={"text": "新台词"},
            )

    assert response.status_code == 500
    assert audio.is_file()
    stored = load_segments(project_id)[0]
    assert stored.text == "旧台词"
    assert stored.audio_file == f"audio/segments/{segment_id}.wav"


def test_segment_delete_keeps_audio_when_metadata_save_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id, _ = _create_project_with_segment(client)
        audio = _attach_audio(tmp_path, project_id, segment_id)

        with patch(
            "dubbing_service.app.routers.segments.save_segments",
            side_effect=SegmentStorageError("simulated disk failure"),
        ):
            response = client.delete(
                f"/api/v1/projects/{project_id}/segments/{segment_id}"
            )

    assert response.status_code == 500
    assert audio.is_file()
    assert load_segments(project_id)[0].id == segment_id


def test_emotion_upload_restores_previous_file_when_metadata_save_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id, _ = _create_project_with_segment(client)
        main_audio = _attach_audio(tmp_path, project_id, segment_id)
        emotion_audio = tmp_path / project_id / "audio" / "emotion" / f"{segment_id}.wav"
        emotion_audio.parent.mkdir(parents=True, exist_ok=True)
        emotion_audio.write_bytes(b"old-emotion")
        segments = load_segments(project_id)
        segments[0].emotion_audio = f"audio/emotion/{segment_id}.wav"
        save_segments(project_id, segments)

        with patch(
            "dubbing_service.app.routers.segments.save_segments",
            side_effect=SegmentStorageError("simulated disk failure"),
        ):
            response = client.post(
                f"/api/v1/projects/{project_id}/segments/{segment_id}/emotion-audio",
                files={"file": ("replacement.wav", make_wav_bytes(sample_value=800), "audio/wav")},
            )

    assert response.status_code == 500
    assert main_audio.is_file()
    assert emotion_audio.read_bytes() == b"old-emotion"
    stored = load_segments(project_id)[0]
    assert stored.audio_file == f"audio/segments/{segment_id}.wav"
    assert stored.emotion_audio == f"audio/emotion/{segment_id}.wav"


def test_json_import_rolls_back_metadata_before_audio_gc(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id, _ = _create_project_with_segment(client)
        audio = _attach_audio(tmp_path, project_id, segment_id)

        real_replace = os.replace
        calls = 0

        def fail_second_promote(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated second promote failure")
            return real_replace(source, destination)

        with patch(
            "dubbing_service.app.storage.segmentation_commit.os.replace",
            side_effect=fail_second_promote,
        ):
            response = client.post(
                f"/api/v1/projects/{project_id}/segments/import-json",
                json={
                    "json_text": json.dumps(
                        [{"speaker_name": "角色B", "text": "替换台词", "emotion_description": "激动"}],
                        ensure_ascii=False,
                    )
                },
            )

    assert response.status_code == 500
    assert audio.is_file()
    assert load_segments(project_id)[0].text == "旧台词"
    assert all(role.name != "角色B" for role in load_project(project_id).roles)


def test_role_delete_rolls_back_project_segments_and_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        uploaded = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
            files={"file": ("prompt.wav", make_wav_bytes(sample_value=900), "audio/wav")},
        )
        assert uploaded.status_code == 200
        prompt = tmp_path / project_id / uploaded.json()["speaker_audio"]
        assert prompt.is_file()

        real_replace = os.replace
        calls = 0

        def fail_second_promote(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated second promote failure")
            return real_replace(source, destination)

        with patch(
            "dubbing_service.app.storage.segmentation_commit.os.replace",
            side_effect=fail_second_promote,
        ):
            response = client.delete(f"/api/v1/projects/{project_id}/roles/{role_id}")

    assert response.status_code == 500
    assert prompt.is_file()
    assert any(role.id == role_id for role in load_project(project_id).roles)
    assert load_segments(project_id)[0].id == segment_id
    assert load_segments(project_id)[0].role_id == role_id


def test_role_prompt_upload_keeps_previous_file_when_metadata_save_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, _, role_id = _create_project_with_segment(client)
        first = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
            files={"file": ("first.wav", make_wav_bytes(sample_value=1000), "audio/wav")},
        )
        assert first.status_code == 200
        old_prompt = tmp_path / project_id / first.json()["speaker_audio"]
        assert old_prompt.read_bytes() == make_wav_bytes(sample_value=1000)

        with patch(
            "dubbing_service.app.storage.segmentation_commit._replace_files_transactionally",
            side_effect=OSError("simulated project metadata failure"),
        ):
            second = client.post(
                f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
                files={"file": ("second.wav", make_wav_bytes(sample_value=1100), "audio/wav")},
            )

    assert second.status_code == 500
    assert old_prompt.read_bytes() == make_wav_bytes(sample_value=1000)
    prompt_files = list((tmp_path / project_id / "roles" / "prompts").glob(f"{role_id}_*"))
    assert prompt_files == [old_prompt]
    stored_role = next(role for role in load_project(project_id).roles if role.id == role_id)
    assert stored_role.speaker_audio == first.json()["speaker_audio"]
