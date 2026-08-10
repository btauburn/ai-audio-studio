from __future__ import annotations

import io
import json
import wave
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.services.production import apply_pronunciation_dictionary
from tests.audio_test_utils import make_wav_bytes


def _approved_project(client: TestClient, projects_root: Path) -> tuple[str, str]:
    project_id = client.post("/api/v1/projects", json={"name": "studio", "description": ""}).json()["id"]
    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={"json_text": json.dumps([
            {"speaker_name": "旁白", "text": "重庆银行", "emotion_description": "平静"}
        ], ensure_ascii=False)},
    )
    segment_id = imported.json()["segments"][0]["id"]
    path = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_wav_bytes(sample_value=700, frames=16000))
    segment = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
    segment.update({"audio_file": f"audio/segments/{segment_id}.wav", "status": "done"})
    assert client.put(
        f"/api/v1/projects/{project_id}/segments/restore", json={"segments": [segment]}
    ).status_code == 200
    assert client.patch(
        f"/api/v1/projects/{project_id}/segments/{segment_id}/review",
        json={"review_status": "approved"},
    ).status_code == 200
    return project_id, segment_id


def test_pronunciation_dictionary_is_generation_only(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id = client.post("/api/v1/projects", json={"name": "terms", "description": ""}).json()["id"]
        response = client.put(
            f"/api/v1/projects/{project_id}/production/pronunciations",
            json={"entries": [{"id": "term1", "source": "重庆", "pronunciation": "重(zhong4)庆"}]},
        )
        assert response.status_code == 200, response.text
        assert apply_pronunciation_dictionary(project_id, "重庆银行") == "重(zhong4)庆银行"
        assert client.get(f"/api/v1/projects/{project_id}/production").json()[
            "pronunciation_dictionary"
        ][0]["source"] == "重庆"
    clear_config_caches()


def test_chapter_assignment_and_voice_rights_do_not_invalidate_take(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id = _approved_project(client, projects_root)
        before = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        chapter = client.post(
            f"/api/v1/projects/{project_id}/chapters", json={"title": "第一章"}
        )
        assert chapter.status_code == 200, chapter.text
        assigned = client.patch(
            f"/api/v1/projects/{project_id}/segments/{segment_id}",
            json={"chapter_id": chapter.json()["id"]},
        )
        assert assigned.status_code == 200, assigned.text
        assert assigned.json()["selected_take_id"] == before["selected_take_id"]
        assert assigned.json()["status"] == "done"

        role_id = assigned.json()["role_id"]
        rights = client.patch(
            f"/api/v1/projects/{project_id}/roles/{role_id}/voice-profile",
            json={
                "preferred_speed_min": 0.85, "preferred_speed_max": 1.05,
                "voice_source": "self", "voice_license_name": "owned",
                "voice_commercial_use": True, "voice_rights_note": "recorded by owner",
            },
        )
        assert rights.status_code == 200, rights.text
        assert rights.json()["voice_commercial_use"] is True
        after = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        assert after["selected_take_id"] == before["selected_take_id"]
        assert after["status"] == "done"
        blocked_delete = client.delete(
            f"/api/v1/projects/{project_id}/chapters/{chapter.json()['id']}"
        )
        assert blocked_delete.status_code == 409
    clear_config_caches()


def test_four_track_master_qc_and_release_package(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()

    with TestClient(create_app()) as client:
        project_id, _ = _approved_project(client, projects_root)
        asset_upload = client.post(
            f"/api/v1/projects/{project_id}/production/assets",
            data={
                "track": "bgm", "name": "测试音乐", "tags": "温暖,片头",
                "source": "local", "author": "tester", "license_name": "owned",
                "commercial_use": "true", "proof_note": "self-made",
            },
            files={"file": ("music.wav", make_wav_bytes(sample_value=120, frames=16000), "audio/wav")},
        )
        assert asset_upload.status_code == 200, asset_upload.text
        asset = asset_upload.json()
        sound = client.put(
            f"/api/v1/projects/{project_id}/production/sound-design",
            json={
                "clips": [{
                    "id": "clip1", "track": "bgm", "asset_id": asset["id"],
                    "start": 0, "duration": 1, "gain_db": 0, "fade_in": 0.05,
                    "fade_out": 0.05, "loop": False,
                }],
                "mix": {"preset": "audio_story", "bgm_ducking": True},
            },
        )
        assert sound.status_code == 200, sound.text
        qc = client.get(f"/api/v1/projects/{project_id}/quality-check")
        assert qc.status_code == 200
        assert qc.json()["passed"] is True

        master_response = client.post(
            f"/api/v1/projects/{project_id}/masters", json={"name": "四轨成品"}
        )
        assert master_response.status_code == 200, master_response.text
        master = master_response.json()
        assert master["sound_design_snapshot"]["clips"][0]["asset_sha256"] == asset["audio_sha256"]
        master_path = projects_root / project_id / master["audio_file"]
        with wave.open(io.BytesIO(master_path.read_bytes()), "rb") as audio:
            assert audio.getframerate() == 48000
            assert audio.getsampwidth() == 3
            assert audio.getnchannels() == 2

        cover = client.post(
            f"/api/v1/projects/{project_id}/production/cover",
            files={"file": ("cover.png", b"\x89PNG\r\n\x1a\n" + b"x" * 64, "image/png")},
        )
        assert cover.status_code == 200, cover.text
        metadata = cover.json()["release"]
        metadata.update({"title": "测试节目", "author": "作者"})
        assert client.put(
            f"/api/v1/projects/{project_id}/production/release-metadata", json=metadata
        ).status_code == 200
        package_response = client.post(
            f"/api/v1/projects/{project_id}/release-packages/{master['id']}"
        )
        assert package_response.status_code == 200, package_response.text
        package = package_response.json()
        archive = projects_root / project_id / package["archive_file"]
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            assert {"master.wav", "publish.mp3", "subtitles.srt", "subtitles.vtt"}.issubset(names)
            assert {"metadata.json", "credits_and_licenses.json", "checksums.sha256", "cover.png"}.issubset(names)
        download = client.get(
            f"/api/v1/projects/{project_id}/release-packages/{package['id']}/download"
        )
        assert download.status_code == 200
        assert download.content == archive.read_bytes()
    clear_config_caches()


def test_voice_finishing_recipe_is_non_destructive(tmp_path: Path, monkeypatch) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()
    with TestClient(create_app()) as client:
        project_id, segment_id = _approved_project(client, projects_root)
        before = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        source_take = next(item for item in before["takes"] if item["id"] == before["selected_take_id"])
        source_path = projects_root / project_id / source_take["audio_file"]
        source_bytes = source_path.read_bytes()
        edited = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio",
            json={
                "fade_in": 0.05, "fade_out": 0.05, "leading_silence": 0.1,
                "trailing_silence": 0.1, "processing_preset": "clear_podcast",
            },
        )
        assert edited.status_code == 200, edited.text
        assert edited.json()["edit_state"]["processing_preset"] == "clear_podcast"
        assert source_path.read_bytes() == source_bytes
        revised = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio",
            json={"processing_preset": "audio_story", "fade_in": 0.1},
        )
        assert revised.status_code == 200, revised.text
        revised_take = next(
            item for item in revised.json()["takes"]
            if item["id"] == revised.json()["selected_take_id"]
        )
        assert len(revised_take["edit_history"]) == 1
        restored = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-history/restore",
            json={"revision_index": 0},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["edit_state"]["processing_preset"] == "clear_podcast"
        rendered = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/render-as-new-take"
        )
        assert rendered.status_code == 200, rendered.text
        assert rendered.json()["selected_take_id"] != before["selected_take_id"]
        assert source_path.read_bytes() == source_bytes
    clear_config_caches()
