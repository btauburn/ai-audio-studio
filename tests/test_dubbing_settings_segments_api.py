import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.core.helpers import get_max_upload_bytes as _get_max_upload_bytes
from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.project import Segment, SegmentEditRegion, SegmentEditSilence, SegmentEditState
from dubbing_service.app.routers.synthesis import (
    _build_segment_audio_edit_request_from_state,
    _copy_audio_to_project,
    _resolve_audio_edit_targets,
)
from dubbing_service.app.routers import settings as settings_router
from dubbing_service.app.storage.projects import load_project, save_project
from tests.audio_test_utils import make_wav_bytes


class TestDubbingSettingsAndSegmentsApi(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            "DUBBING_PROJECTS_ROOT": os.environ.get("DUBBING_PROJECTS_ROOT"),
            "DUBBING_KEEP_SEG_PROMPT_ARCHIVES": os.environ.get("DUBBING_KEEP_SEG_PROMPT_ARCHIVES"),
            "DUBBING_KEEP_SEG_LLM_LOGS": os.environ.get("DUBBING_KEEP_SEG_LLM_LOGS"),
            "DUBBING_MAX_UPLOAD_BYTES": os.environ.get("DUBBING_MAX_UPLOAD_BYTES"),
        }

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_caches()

    def test_get_tts_backend_normalizes_invalid_config_value(self) -> None:
        clear_config_caches()
        app = create_app()

        with patch(
            "dubbing_service.app.routers.settings._load_raw_file_config",
            return_value={"tts_backend": "invalid-backend"},
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/settings/tts-backend")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("backend"), "indextts")

    def test_resolve_audio_edit_targets_uses_current_segment_state_as_base(self) -> None:
        desired_speed, desired_volume, speed_ratio, volume_ratio = _resolve_audio_edit_targets(
            target_speed=1.5,
            target_volume=1.8,
            base_speed=1.2,
            base_volume=0.9,
        )

        self.assertAlmostEqual(desired_speed, 1.5)
        self.assertAlmostEqual(desired_volume, 1.8)
        self.assertAlmostEqual(speed_ratio, 1.25)
        self.assertAlmostEqual(volume_ratio, 2.0)

        desired_speed2, desired_volume2, speed_ratio2, volume_ratio2 = _resolve_audio_edit_targets(
            target_speed=None,
            target_volume=None,
            base_speed=1.2,
            base_volume=0.9,
        )
        self.assertAlmostEqual(desired_speed2, 1.2)
        self.assertAlmostEqual(desired_volume2, 0.9)
        self.assertAlmostEqual(speed_ratio2, 1.0)
        self.assertAlmostEqual(volume_ratio2, 1.0)

    def test_build_segment_audio_edit_request_from_state_supports_legacy_single_region(self) -> None:
        state = SegmentEditState(
            speed=1.3,
            volume=0.8,
            region=SegmentEditRegion(start=0.2, end=0.5),
            silences=[SegmentEditSilence(at=1.0, duration=0.4)],
        )

        payload = _build_segment_audio_edit_request_from_state(state)

        self.assertAlmostEqual(payload.speed, 1.3)
        self.assertAlmostEqual(payload.volume, 0.8)
        self.assertIsNone(payload.delete_regions)
        self.assertIsNotNone(payload.delete_region)
        self.assertAlmostEqual(payload.delete_region.start, 0.2)
        self.assertAlmostEqual(payload.delete_region.end, 0.5)
        self.assertIsNotNone(payload.silence_intervals)
        self.assertEqual(len(payload.silence_intervals), 1)
        self.assertAlmostEqual(payload.silence_intervals[0].at, 1.0)
        self.assertAlmostEqual(payload.silence_intervals[0].duration, 0.4)

    def test_start_tts_service_rejects_invalid_backend(self) -> None:
        clear_config_caches()
        app = create_app()

        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/settings/tts-service/start",
                json={"backend": "not-supported"},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("无效的后端类型", resp.text)

    def test_tts_service_status_reports_runtime_backend_separately_from_config_backend(self) -> None:
        clear_config_caches()
        app = create_app()

        class _FakeProcess:
            pid = 4321

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

            def terminate(self):
                return None

        old_proc = settings_router._tts_process
        old_runtime_backend = settings_router._tts_runtime_backend
        settings_router._tts_process = _FakeProcess()
        settings_router._tts_runtime_backend = "indextts"

        try:
            with patch(
                "dubbing_service.app.routers.settings._load_raw_file_config",
                return_value={"tts_backend": "cosyvoice3"},
            ):
                with TestClient(app) as client:
                    resp = client.get("/api/v1/settings/tts-service/status")
        finally:
            settings_router._tts_process = old_proc
            settings_router._tts_runtime_backend = old_runtime_backend

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("running"))
        self.assertEqual(data.get("backend"), "indextts")
        self.assertEqual(data.get("runtime_backend"), "indextts")
        self.assertEqual(data.get("config_backend"), "cosyvoice3")
        self.assertEqual(data.get("pid"), 4321)

    def test_tts_service_load_model_prefers_runtime_backend_over_requested_backend(self) -> None:
        clear_config_caches()
        app = create_app()

        class _FakeProcess:
            pid = 2468

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

            def terminate(self):
                return None

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"success": true, "model_loaded": true}'

        class _Cfg:
            tts_base_url = "http://127.0.0.1:8001"

        captured: dict[str, object] = {}

        def _fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            return _FakeResponse()

        old_proc = settings_router._tts_process
        old_runtime_backend = settings_router._tts_runtime_backend
        settings_router._tts_process = _FakeProcess()
        settings_router._tts_runtime_backend = "indextts"

        try:
            with patch(
                "dubbing_service.app.routers.settings._load_raw_file_config",
                return_value={"tts_backend": "qwen3tts"},
            ), patch(
                "dubbing_service.app.routers.settings.get_config",
                return_value=_Cfg(),
            ), patch(
                "dubbing_service.app.routers.settings.get_qwen3tts_base_url",
                return_value="http://127.0.0.1:8003",
            ), patch(
                "dubbing_service.app.routers.settings.urllib.request.urlopen",
                side_effect=_fake_urlopen,
            ):
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/v1/settings/tts-service/load-model",
                        json={"backend": "qwen3tts"},
                    )
        finally:
            settings_router._tts_process = old_proc
            settings_router._tts_runtime_backend = old_runtime_backend

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(captured.get("url"), "http://127.0.0.1:8001/api/v1/tts/load-model")
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("backend"), "indextts")

    def test_health_prefers_runtime_backend_probe_and_reports_runtime_fields(self) -> None:
        clear_config_caches()
        app = create_app()

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"backend": "indextts", "model_loaded": true, "device": "cpu"}'

        captured: dict[str, object] = {}

        def _fake_base_url(backend):
            captured["backend"] = backend
            return {
                "indextts": "http://127.0.0.1:8001",
                "qwen3tts": "http://127.0.0.1:8003",
            }[backend]

        def _fake_urlopen(url, timeout=0):
            captured["url"] = url
            captured["timeout"] = timeout
            return _FakeResponse()

        with patch(
            "dubbing_service.app.main._load_file_config",
            return_value={"tts_backend": "qwen3tts"},
        ), patch(
            "dubbing_service.app.main.settings_router._get_tts_runtime_state",
            return_value=(True, 4321, "indextts"),
        ), patch(
            "dubbing_service.app.main.settings_router._get_tts_backend_base_url",
            side_effect=_fake_base_url,
        ), patch(
            "dubbing_service.app.main.urllib.request.urlopen",
            side_effect=_fake_urlopen,
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(captured.get("backend"), "indextts")
        self.assertEqual(captured.get("url"), "http://127.0.0.1:8001/health")
        data = resp.json()
        self.assertTrue(data.get("tts_health_ok"))
        self.assertEqual(data.get("tts_runtime_backend"), "indextts")
        self.assertEqual(data.get("tts_config_backend"), "qwen3tts")
        self.assertTrue(data.get("tts_service_running"))

    def test_restore_rejects_invalid_tts_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "restore-invalid-backend", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg = imported.json()["segments"][0]
                seg["tts_backend"] = "bad-backend"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [seg]},
                )

            self.assertEqual(restored.status_code, 400)
            self.assertIn("tts_backend 值不合法", restored.text)

    def test_synthesize_alt_rejects_invalid_tts_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "alt-invalid-backend", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg_id = imported.json()["segments"][0]["id"]

                resp = client.post(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}/synthesize-alt",
                    json={"tts_backend": "bad-backend"},
                )

            self.assertEqual(resp.status_code, 400)
            self.assertIn("tts_backend 必须是", resp.text)

    def test_cleanup_seg_debug_files_prunes_prompt_and_log_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            os.environ["DUBBING_KEEP_SEG_PROMPT_ARCHIVES"] = "1"
            os.environ["DUBBING_KEEP_SEG_LLM_LOGS"] = "1"
            clear_config_caches()

            project_id = "p1"
            project_root = Path(tmp_root) / project_id
            text_dir = project_root / "text"
            logs_dir = project_root / "logs"
            text_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            old_prompt = text_dir / "seg_prompt_20250101_000000.txt"
            new_prompt = text_dir / "seg_prompt_20250102_000000.txt"
            old_prompt.write_text("old", encoding="utf-8")
            new_prompt.write_text("new", encoding="utf-8")

            old_raw = logs_dir / "seg_llm_raw_20250101_000000.txt"
            old_parsed = logs_dir / "seg_llm_parsed_20250101_000000.json"
            new_raw = logs_dir / "seg_llm_raw_20250102_000000.txt"
            new_parsed = logs_dir / "seg_llm_parsed_20250102_000000.json"
            old_raw.write_text("old raw", encoding="utf-8")
            old_parsed.write_text("{}", encoding="utf-8")
            new_raw.write_text("new raw", encoding="utf-8")
            new_parsed.write_text("{}", encoding="utf-8")

            app = create_app()
            with TestClient(app) as client:
                resp = client.post(
                    f"/api/v1/projects/{project_id}/segments/cleanup-debug-files"
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data.get("deleted_text_files"), 1)
            self.assertEqual(data.get("deleted_log_files"), 2)

            self.assertFalse(old_prompt.exists())
            self.assertTrue(new_prompt.exists())
            self.assertFalse(old_raw.exists())
            self.assertFalse(old_parsed.exists())
            self.assertTrue(new_raw.exists())
            self.assertTrue(new_parsed.exists())

    def test_list_apis_for_missing_project_do_not_create_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            project_id = "missing123"
            project_dir = Path(tmp_root) / project_id

            with TestClient(app) as client:
                seg_resp = client.get(f"/api/v1/projects/{project_id}/segments")
                sub_resp = client.get(f"/api/v1/projects/{project_id}/subtitles")

            self.assertEqual(seg_resp.status_code, 404)
            self.assertEqual(sub_resp.status_code, 404)
            self.assertFalse(project_dir.exists())

    def test_reupload_voice_audio_cleans_old_extension_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            voices_json = Path(tmp_root) / "voices.json"
            voices_dir = Path(tmp_root) / "voices"

            def _voices_audio_dir() -> Path:
                voices_dir.mkdir(parents=True, exist_ok=True)
                return voices_dir

            clear_config_caches()
            app = create_app()

            with patch(
                "dubbing_service.app.storage.voices._voices_file_path",
                return_value=voices_json,
            ), patch(
                "dubbing_service.app.storage.voices.voices_audio_dir",
                side_effect=_voices_audio_dir,
            ), patch(
                "dubbing_service.app.routers.voices.voices_audio_dir",
                side_effect=_voices_audio_dir,
            ), patch(
                "dubbing_service.app.routers.roles.voices_audio_dir",
                side_effect=_voices_audio_dir,
            ):
                with TestClient(app) as client:
                    created = client.post(
                        "/api/v1/voices",
                        json={"name": "voice-clean", "tags": [], "description": ""},
                    )
                    self.assertEqual(created.status_code, 200)
                    voice_id = created.json()["id"]

                    up_wav = client.post(
                        f"/api/v1/voices/{voice_id}/audio",
                        files={"file": ("sample.wav", make_wav_bytes(sample_value=100), "audio/wav")},
                    )
                    self.assertEqual(up_wav.status_code, 200)

                    up_mp3 = client.post(
                        f"/api/v1/voices/{voice_id}/audio",
                        files={"file": ("sample.mp3", make_wav_bytes(sample_value=200), "audio/mpeg")},
                    )
                    self.assertEqual(up_mp3.status_code, 200)

                files = sorted([p.name for p in voices_dir.glob("*") if p.is_file()])
                self.assertEqual(files, [f"{voice_id}.mp3"])

    def test_synthesize_plan_rejects_stale_speaker_voice_id_without_local_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "stale-speaker-voice-id", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "角色A", "role_type": "character", "description": ""},
                )
                self.assertEqual(role_resp.status_code, 200)

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "角色A", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                segment_id = imported.json()["segments"][0]["id"]

                project = load_project(project_id)
                role = next(r for r in project.roles if r.name == "角色A")
                role.speaker_audio = None
                role.speaker_voice_id = "missingvoice"
                save_project(project)

                with patch(
                    "dubbing_service.app.routers.settings._load_raw_file_config",
                    return_value={"tts_backend": "indextts"},
                ):
                    plan = client.post(
                        f"/api/v1/projects/{project_id}/segments/synthesize/plan",
                        json={"segment_ids": [segment_id], "retry_failed": True},
                    )

            self.assertEqual(plan.status_code, 400)
            self.assertIn("缺少参考音频", plan.text)

    def test_delete_role_removes_prompt_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "role-delete-clean", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "角色A", "role_type": "character", "description": ""},
                )
                self.assertEqual(role_resp.status_code, 200)
                role_id = role_resp.json()["id"]

                uploaded = client.post(
                    f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
                    files={"file": ("prompt.wav", make_wav_bytes(sample_value=300), "audio/wav")},
                )
                self.assertEqual(uploaded.status_code, 200)
                rel_path = uploaded.json().get("speaker_audio")
                self.assertIsNotNone(rel_path)

                prompt_path = Path(tmp_root) / project_id / str(rel_path)
                self.assertTrue(prompt_path.is_file())

                deleted = client.delete(f"/api/v1/projects/{project_id}/roles/{role_id}")
                self.assertEqual(deleted.status_code, 200)

                self.assertFalse(prompt_path.exists())

    def test_upload_segment_emotion_audio_preserves_existing_takes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "emotion-upload-clean", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg_id = imported.json()["segments"][0]["id"]

                segments_dir = Path(tmp_root) / project_id / "audio" / "segments"
                segments_dir.mkdir(parents=True, exist_ok=True)
                main_audio = segments_dir / f"{seg_id}.wav"
                alt_audio = segments_dir / f"{seg_id}_alt.wav"
                main_audio.write_bytes(b"main")
                alt_audio.write_bytes(b"alt")

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}.wav"
                restored_seg["alt_audio_file"] = f"audio/segments/{seg_id}_alt.wav"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                uploaded = client.post(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}/emotion-audio",
                    files={"file": ("emo.wav", make_wav_bytes(sample_value=400), "audio/wav")},
                )
                self.assertEqual(uploaded.status_code, 200)
                uploaded_data = uploaded.json()
                self.assertEqual(uploaded_data.get("status"), "pending")
                self.assertEqual(uploaded_data.get("audio_file"), f"audio/segments/{seg_id}.wav")
                self.assertEqual(uploaded_data.get("alt_audio_file"), f"audio/segments/{seg_id}_alt.wav")
                self.assertEqual(uploaded_data.get("emotion_mode"), "audio")
                self.assertTrue(uploaded_data["take_slots"]["a"]["needs_synthesis"])

                emotion_rel = uploaded_data.get("emotion_audio")
                self.assertTrue(emotion_rel)
                self.assertTrue(main_audio.exists())
                self.assertTrue(alt_audio.exists())
                self.assertTrue((Path(tmp_root) / project_id / str(emotion_rel)).is_file())

    def test_update_segment_emotion_alpha_marks_current_take_stale_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "segment-alpha-invalidate", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg_id = imported.json()["segments"][0]["id"]

                segments_dir = Path(tmp_root) / project_id / "audio" / "segments"
                segments_dir.mkdir(parents=True, exist_ok=True)
                main_audio = segments_dir / f"{seg_id}.wav"
                alt_audio = segments_dir / f"{seg_id}_alt.wav"
                main_audio.write_bytes(b"main")
                alt_audio.write_bytes(b"alt")

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}.wav"
                restored_seg["alt_audio_file"] = f"audio/segments/{seg_id}_alt.wav"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                updated = client.patch(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}",
                    json={"emotion_alpha": 0.9},
                )
                self.assertEqual(updated.status_code, 200)
                updated_data = updated.json()
                self.assertEqual(updated_data.get("status"), "pending")
                self.assertEqual(updated_data.get("audio_file"), f"audio/segments/{seg_id}.wav")
                self.assertEqual(updated_data.get("alt_audio_file"), f"audio/segments/{seg_id}_alt.wav")
                self.assertAlmostEqual(updated_data.get("emotion_alpha"), 0.9)
                self.assertTrue(updated_data["take_slots"]["a"]["needs_synthesis"])
                self.assertTrue(main_audio.exists())
                self.assertTrue(alt_audio.exists())

    def test_update_segment_tts_backend_marks_current_take_stale_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "segment-backend-invalidate", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg_id = imported.json()["segments"][0]["id"]

                segments_dir = Path(tmp_root) / project_id / "audio" / "segments"
                segments_dir.mkdir(parents=True, exist_ok=True)
                main_audio = segments_dir / f"{seg_id}.wav"
                alt_audio = segments_dir / f"{seg_id}_alt.wav"
                main_audio.write_bytes(b"main")
                alt_audio.write_bytes(b"alt")

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}.wav"
                restored_seg["alt_audio_file"] = f"audio/segments/{seg_id}_alt.wav"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                updated = client.patch(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}",
                    json={"tts_backend": "cosyvoice3"},
                )
                self.assertEqual(updated.status_code, 200)
                updated_data = updated.json()
                self.assertEqual(updated_data.get("status"), "pending")
                self.assertEqual(updated_data.get("audio_file"), f"audio/segments/{seg_id}.wav")
                self.assertEqual(updated_data.get("alt_audio_file"), f"audio/segments/{seg_id}_alt.wav")
                self.assertEqual(updated_data.get("tts_backend"), "cosyvoice3")
                self.assertTrue(updated_data["take_slots"]["a"]["needs_synthesis"])
                self.assertTrue(main_audio.exists())
                self.assertTrue(alt_audio.exists())

    def test_delete_segment_removes_emotion_audio_and_recording_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "segment-delete-emotion-clean", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)
                seg_id = imported.json()["segments"][0]["id"]

                project_root = Path(tmp_root) / project_id
                segments_dir = project_root / "audio" / "segments"
                emotion_dir = project_root / "audio" / "emotion"
                segments_dir.mkdir(parents=True, exist_ok=True)
                emotion_dir.mkdir(parents=True, exist_ok=True)

                main_audio = segments_dir / f"{seg_id}.wav"
                alt_audio = segments_dir / f"{seg_id}_alt.wav"
                emotion_audio = emotion_dir / f"{seg_id}.wav"
                emotion_recording = emotion_dir / f"{seg_id}.webm"
                main_audio.write_bytes(b"main")
                alt_audio.write_bytes(b"alt")
                emotion_audio.write_bytes(b"emotion")
                emotion_recording.write_bytes(b"recording")

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}.wav"
                restored_seg["alt_audio_file"] = f"audio/segments/{seg_id}_alt.wav"
                restored_seg["emotion_audio"] = f"audio/emotion/{seg_id}.wav"
                restored_seg["emotion_mode"] = "audio"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                deleted = client.delete(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}"
                )
                self.assertEqual(deleted.status_code, 200)

                self.assertFalse(main_audio.exists())
                self.assertFalse(alt_audio.exists())
                self.assertFalse(emotion_audio.exists())
                self.assertFalse(emotion_recording.exists())

    def test_cleanup_all_removes_orphan_segment_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "cleanup-orphan-segments", "description": ""},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "hello", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported.status_code, 200)

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                seg = listed.json()[0]
                seg["audio_file"] = "audio/segments/keep.wav"
                seg["alt_audio_file"] = None

                # Restore now removes references whose files are already
                # missing, so materialize the referenced file first.
                segments_dir = Path(tmp_root) / project_id / "audio" / "segments"
                segments_dir.mkdir(parents=True, exist_ok=True)
                keep_path = segments_dir / "keep.wav"
                orphan_path = segments_dir / "orphan.wav"
                keep_path.write_bytes(b"keep")
                orphan_path.write_bytes(b"orphan")

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [seg]},
                )
                self.assertEqual(restored.status_code, 200)

                cleaned = client.post("/api/v1/projects/cleanup-all")
                self.assertEqual(cleaned.status_code, 200)

                self.assertTrue(keep_path.is_file())
                self.assertFalse(orphan_path.exists())

    def test_update_platform_settings_partial_does_not_clobber_other_keys(self) -> None:
        clear_config_caches()
        app = create_app()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "config.json"
            initial = {
                "tts_service_name": "old-tts",
                "tts_base_url": "http://127.0.0.1:8010",
                "llm_base_url": "http://127.0.0.1:9010/v1",
                "llm_api_key": "sk-old",
            }
            cfg_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")

            with patch(
                "dubbing_service.app.routers.settings._config_file_path",
                return_value=cfg_path,
            ):
                with TestClient(app) as client:
                    resp = client.put(
                        "/api/v1/settings/platform",
                        json={"llm_service_name": "updated-llm"},
                    )

            self.assertEqual(resp.status_code, 200)
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(saved.get("tts_service_name"), "old-tts")
            self.assertEqual(saved.get("tts_base_url"), "http://127.0.0.1:8010")
            self.assertEqual(saved.get("llm_base_url"), "http://127.0.0.1:9010/v1")
            self.assertEqual(saved.get("llm_api_key"), "sk-old")
            self.assertEqual(saved.get("llm_service_name"), "updated-llm")

    def test_update_tts_params_merges_existing_values(self) -> None:
        clear_config_caches()
        app = create_app()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "config.json"
            initial = {
                "tts_params": {
                    "temperature": 0.7,
                    "top_p": 0.85,
                }
            }
            cfg_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")

            with patch(
                "dubbing_service.app.routers.settings._config_file_path",
                return_value=cfg_path,
            ):
                with TestClient(app) as client:
                    resp = client.put(
                        "/api/v1/settings/tts-params",
                        json={"top_k": 44},
                    )

            self.assertEqual(resp.status_code, 200)
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            tts_params = saved.get("tts_params") or {}
            self.assertEqual(tts_params.get("temperature"), 0.7)
            self.assertEqual(tts_params.get("top_p"), 0.85)
            self.assertEqual(tts_params.get("top_k"), 44)

    def test_update_raw_file_config_is_thread_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "config.json"
            cfg_path.write_text("{}", encoding="utf-8")

            barrier = threading.Barrier(2)

            def _worker(key: str) -> None:
                def _mut(data: dict) -> None:
                    data[key] = True
                    # 放大临界区，便于暴露并发覆盖问题
                    time.sleep(0.05)

                barrier.wait(timeout=2)
                settings_router._update_raw_file_config(_mut)

            with patch(
                "dubbing_service.app.routers.settings._config_file_path",
                return_value=cfg_path,
            ):
                t1 = threading.Thread(target=_worker, args=("k1",))
                t2 = threading.Thread(target=_worker, args=("k2",))
                t1.start()
                t2.start()
                t1.join(timeout=3)
                t2.join(timeout=3)

            self.assertFalse(t1.is_alive())
            self.assertFalse(t2.is_alive())

            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertTrue(saved.get("k1"))
            self.assertTrue(saved.get("k2"))

    def test_swap_alt_without_b_audio_resets_status_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "ab-swap-status", "description": "ab"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [
                                {
                                    "speaker_name": "旁白",
                                    "text": "A/B 状态回归测试",
                                    "emotion_description": "平静",
                                }
                            ],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported_segments.status_code, 200)
                seg = imported_segments.json()["segments"][0]
                seg_id = seg["id"]

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}.wav"
                restored_seg["alt_audio_file"] = None
                restored_seg["active_version"] = "a"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                swapped = client.post(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}/swap-alt"
                )
                self.assertEqual(swapped.status_code, 200)
                swapped_data = swapped.json()
                self.assertEqual(swapped_data.get("active_version"), "b")
                self.assertIsNone(swapped_data.get("audio_file"))
                self.assertEqual(swapped_data.get("status"), "pending")

    def test_clear_alt_without_a_audio_resets_status_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "ab-clear-status", "description": "ab"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [
                                {
                                    "speaker_name": "旁白",
                                    "text": "清除 B 版本状态回归测试",
                                    "emotion_description": "平静",
                                }
                            ],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported_segments.status_code, 200)
                seg = imported_segments.json()["segments"][0]
                seg_id = seg["id"]

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["status"] = "done"
                restored_seg["active_version"] = "b"
                restored_seg["audio_file"] = f"audio/segments/{seg_id}_alt.wav"
                restored_seg["alt_audio_file"] = None

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 200)

                cleared = client.delete(
                    f"/api/v1/projects/{project_id}/segments/{seg_id}/alt"
                )
                self.assertEqual(cleared.status_code, 200)
                cleared_data = cleared.json()
                self.assertEqual(cleared_data.get("active_version"), "a")
                self.assertIsNone(cleared_data.get("audio_file"))
                self.assertIsNone(cleared_data.get("alt_audio_file"))
                self.assertEqual(cleared_data.get("status"), "pending")

    def test_restore_rejects_escape_path_in_alt_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "restore-path-guard", "description": "guard"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [
                                {
                                    "speaker_name": "旁白",
                                    "text": "路径校验测试",
                                    "emotion_description": "平静",
                                }
                            ],
                            ensure_ascii=False,
                        )
                    },
                )
                self.assertEqual(imported_segments.status_code, 200)

                listed = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed.status_code, 200)
                restored_seg = listed.json()[0]
                restored_seg["alt_audio_file"] = "../outside.wav"

                restored = client.put(
                    f"/api/v1/projects/{project_id}/segments/restore",
                    json={"segments": [restored_seg]},
                )
                self.assertEqual(restored.status_code, 400)
                self.assertIn("路径越界", restored.text)

    def test_copy_audio_to_project_rejects_absolute_path_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            app = create_app()
            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "copy-audio-guard", "description": "guard"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

            source = Path(tmp_root) / "outside.wav"
            source.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

            seg = Segment(id="seg-test", index=0, text="hello")

            with self.assertRaises(FileNotFoundError):
                _copy_audio_to_project(project_id, seg, str(source))

    def test_get_max_upload_bytes_has_safe_default(self) -> None:
        os.environ.pop("DUBBING_MAX_UPLOAD_BYTES", None)
        self.assertEqual(_get_max_upload_bytes(), 20 * 1024 * 1024)

        os.environ["DUBBING_MAX_UPLOAD_BYTES"] = "0"
        self.assertEqual(_get_max_upload_bytes(), 20 * 1024 * 1024)

        os.environ["DUBBING_MAX_UPLOAD_BYTES"] = "2048"
        self.assertEqual(_get_max_upload_bytes(), 2048)


if __name__ == "__main__":
    unittest.main()
