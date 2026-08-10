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
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.storage.voices import VoiceStorageError, load_voices
from tests.audio_test_utils import make_wav_bytes


def _configure_isolated_storage(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    projects_root = tmp_path / "projects"
    voices_root = tmp_path / "library" / "voices"
    voices_json = tmp_path / "library" / "voices.json"
    projects_root.mkdir(parents=True)

    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()

    def audio_dir() -> Path:
        voices_root.mkdir(parents=True, exist_ok=True)
        return voices_root

    monkeypatch.setattr(
        "dubbing_service.app.storage.voices._voices_file_path",
        lambda: voices_json,
    )
    monkeypatch.setattr(
        "dubbing_service.app.storage.voices.voices_audio_dir",
        audio_dir,
    )
    monkeypatch.setattr("dubbing_service.app.routers.voices.voices_audio_dir", audio_dir)
    monkeypatch.setattr("dubbing_service.app.routers.roles.voices_audio_dir", audio_dir)
    monkeypatch.setattr("dubbing_service.app.routers.segments.voices_audio_dir", audio_dir)
    return projects_root, voices_root


def _create_project_with_segment(client: TestClient) -> tuple[str, str, str]:
    created = client.post("/api/v1/projects", json={"name": "voice-consistency"})
    assert created.status_code == 200
    project_id = created.json()["id"]
    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={
            "json_text": json.dumps(
                [{"speaker_name": "角色A", "text": "测试台词", "emotion_description": "平静"}],
                ensure_ascii=False,
            )
        },
    )
    assert imported.status_code == 200
    segment = imported.json()["segments"][0]
    return project_id, segment["id"], segment["role_id"]


def _attach_both_outputs(
    projects_root: Path,
    project_id: str,
    segment_id: str,
) -> tuple[Path, Path]:
    output_dir = projects_root / project_id / "audio" / "segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = output_dir / f"{segment_id}.wav"
    alternate = output_dir / f"{segment_id}_alt.wav"
    primary.write_bytes(b"primary-output")
    alternate.write_bytes(b"alternate-output")

    segments = load_segments(project_id)
    segment = next(item for item in segments if item.id == segment_id)
    segment.status = SegmentStatus.DONE
    segment.audio_file = f"audio/segments/{primary.name}"
    segment.alt_audio_file = f"audio/segments/{alternate.name}"
    save_segments(project_id, segments)
    return primary, alternate


def _create_voice_with_audio(client: TestClient, *, sample_value: int = 100) -> str:
    created = client.post(
        "/api/v1/voices",
        json={"name": "测试音色", "tags": [], "description": ""},
    )
    assert created.status_code == 200
    voice_id = created.json()["id"]
    uploaded = client.post(
        f"/api/v1/voices/{voice_id}/audio",
        files={"file": ("voice.wav", make_wav_bytes(sample_value=sample_value), "audio/wav")},
    )
    assert uploaded.status_code == 200
    return voice_id


def test_role_voice_profile_marks_slots_stale_and_preserves_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)

        response = client.patch(
            f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
            json={"tts_backend": "qwen3tts", "qwen3_speaker": "Vivian"},
        )

    assert response.status_code == 200
    stored = load_segments(project_id)[0]
    assert stored.status == SegmentStatus.PENDING
    assert stored.audio_file is not None
    assert stored.alt_audio_file is not None
    assert stored.take_slots["a"].needs_synthesis
    assert stored.take_slots["b"].needs_synthesis
    assert primary.is_file()
    assert alternate.is_file()


def test_role_voice_profile_noop_preserves_existing_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)

        response = client.patch(
            f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
            json={"style_instruction": None},
        )

    assert response.status_code == 200
    stored = load_segments(project_id)[0]
    assert stored.status == SegmentStatus.DONE
    assert stored.audio_file is not None
    assert stored.alt_audio_file is not None
    assert primary.is_file()
    assert alternate.is_file()


def test_role_profile_second_promotion_failure_rolls_back_both_metadata_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)
        project_path = projects_root / project_id / "project.json"
        segments_path = projects_root / project_id / "segments.json"
        old_project = project_path.read_bytes()
        old_segments = segments_path.read_bytes()

        real_replace = os.replace
        calls = 0

        def fail_second_promotion(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated second metadata promotion failure")
            return real_replace(source, destination)

        with patch(
            "dubbing_service.app.storage.segmentation_commit.os.replace",
            side_effect=fail_second_promotion,
        ):
            response = client.patch(
                f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
                json={"tts_backend": "qwen3tts"},
            )

    assert response.status_code == 500
    assert project_path.read_bytes() == old_project
    assert segments_path.read_bytes() == old_segments
    assert primary.is_file()
    assert alternate.is_file()


def test_role_prompt_upload_and_voice_binding_preserve_take_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)
        uploaded = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
            files={"file": ("prompt.wav", make_wav_bytes(sample_value=200), "audio/wav")},
        )
        assert uploaded.status_code == 200
        assert primary.is_file()
        assert alternate.is_file()

        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)
        voice_id = _create_voice_with_audio(client, sample_value=300)
        bound = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/bind-voice",
            json={"voice_id": voice_id},
        )

    assert bound.status_code == 200
    assert bound.json()["speaker_voice_id"] == voice_id
    stored = load_segments(project_id)[0]
    assert stored.audio_file is not None
    assert stored.alt_audio_file is not None
    assert stored.take_slots["a"].needs_synthesis
    assert stored.take_slots["b"].needs_synthesis
    assert primary.is_file()
    assert alternate.is_file()


def test_delete_voice_reports_role_and_segment_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, voices_root = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        voice_id = _create_voice_with_audio(client)
        bound = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/bind-voice",
            json={"voice_id": voice_id},
        )
        assert bound.status_code == 200
        updated = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}",
            json={"emotion_mode": "audio", "emotion_audio": f"voices/{voice_id}.wav"},
        )
        assert updated.status_code == 200

        deleted = client.delete(f"/api/v1/voices/{voice_id}")

    assert deleted.status_code == 409
    detail = deleted.json()["detail"]
    assert detail["role_dependency_count"] == 1
    assert detail["emotion_dependency_count"] == 1
    assert detail["projects"][0]["id"] == project_id
    assert any(voice.id == voice_id for voice in load_voices())
    assert (voices_root / f"{voice_id}.wav").is_file()


def test_voice_replacement_metadata_failure_restores_old_audio_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, voices_root = _configure_isolated_storage(tmp_path, monkeypatch)
    old_audio = make_wav_bytes(sample_value=400)
    with TestClient(create_app()) as client:
        voice_id = _create_voice_with_audio(client, sample_value=400)
        before = next(voice for voice in load_voices() if voice.id == voice_id)

        with patch(
            "dubbing_service.app.routers.voices.save_voices",
            side_effect=VoiceStorageError("simulated metadata failure"),
        ):
            replaced = client.post(
                f"/api/v1/voices/{voice_id}/audio",
                files={
                    "file": ("voice.wav", make_wav_bytes(sample_value=500), "audio/wav")
                },
            )

    assert replaced.status_code == 500
    after = next(voice for voice in load_voices() if voice.id == voice_id)
    assert after.audio_path == before.audio_path
    assert after.updated_at == before.updated_at
    assert (voices_root / f"{voice_id}.wav").read_bytes() == old_audio
    assert not list(voices_root.glob(".*.backup"))


def test_fake_audio_uploads_are_rejected_without_changing_old_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, voices_root = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        project_id, segment_id, role_id = _create_project_with_segment(client)
        voice_id = _create_voice_with_audio(client, sample_value=600)
        old_voice_bytes = (voices_root / f"{voice_id}.wav").read_bytes()
        old_voice = next(voice for voice in load_voices() if voice.id == voice_id)

        invalid_voice = client.post(
            f"/api/v1/voices/{voice_id}/audio",
            files={"file": ("fake.wav", b"not-audio", "audio/wav")},
        )
        invalid_role = client.post(
            f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
            files={"file": ("fake.wav", b"not-audio", "audio/wav")},
        )
        primary, alternate = _attach_both_outputs(projects_root, project_id, segment_id)
        invalid_emotion = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/emotion-audio",
            files={"file": ("fake.wav", b"not-audio", "audio/wav")},
        )

    assert invalid_voice.status_code == 400
    assert invalid_role.status_code == 400
    assert invalid_emotion.status_code == 400
    current_voice = next(voice for voice in load_voices() if voice.id == voice_id)
    assert current_voice.audio_path == old_voice.audio_path
    assert current_voice.updated_at == old_voice.updated_at
    assert (voices_root / f"{voice_id}.wav").read_bytes() == old_voice_bytes
    role = next(item for item in load_project(project_id).roles if item.id == role_id)
    assert role.speaker_audio is None
    stored = load_segments(project_id)[0]
    assert stored.audio_file is not None
    assert stored.alt_audio_file is not None
    assert primary.is_file()
    assert alternate.is_file()


def test_emotion_recording_rejects_fake_replacement_and_preserves_old_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    valid_recording = make_wav_bytes(sample_value=700)
    with TestClient(create_app()) as client:
        project_id, segment_id, _ = _create_project_with_segment(client)
        first = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/emotion-recording",
            files={"file": ("recording.webm", valid_recording, "audio/webm")},
        )
        assert first.status_code == 200

        invalid = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/emotion-recording",
            files={"file": ("recording.webm", b"fake-webm", "audio/webm")},
        )

    assert invalid.status_code == 400
    recording = projects_root / project_id / "audio" / "emotion" / f"{segment_id}.webm"
    assert recording.read_bytes() == valid_recording


def test_unreferenced_voice_can_be_deleted_after_dependency_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, voices_root = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        voice_id = _create_voice_with_audio(client, sample_value=800)
        deleted = client.delete(f"/api/v1/voices/{voice_id}")

    assert deleted.status_code == 200
    assert all(voice.id != voice_id for voice in load_voices())
    assert not (voices_root / f"{voice_id}.wav").exists()
