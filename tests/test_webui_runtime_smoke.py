import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.storage.segments import load_segments
from tests.audio_test_utils import make_wav_bytes


class TestWebUIRuntimeSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            "DUBBING_PROJECTS_ROOT": os.environ.get("DUBBING_PROJECTS_ROOT"),
        }

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_caches()

    def test_homepage_and_health_smoke(self) -> None:
        clear_config_caches()
        app = create_app()

        with TestClient(app) as client:
            resp = client.get("/")
            self.assertEqual(resp.status_code, 200)
            html = resp.text
            for marker in ("项目工作区", "音色库", "平台配置", "新建项目"):
                self.assertIn(marker, html)

            ready = client.get("/api/v1/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json().get("status"), "ok")

            health = client.get("/api/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json().get("status"), "ok")

    def test_project_workspace_runtime_flow_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "smoke-project", "description": "runtime flow"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                imported = client.post(
                    f"/api/v1/projects/{project_id}/text/import",
                    json={"text": "第一段文本。\n第二段文本。"},
                )
                self.assertEqual(imported.status_code, 200)

                loaded_text = client.get(f"/api/v1/projects/{project_id}/text")
                self.assertEqual(loaded_text.status_code, 200)
                self.assertIn("第一段文本", loaded_text.json().get("text", ""))

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "旁白", "role_type": "narrator"},
                )
                self.assertEqual(role_resp.status_code, 200)

                seg_payload = [
                    {
                        "speaker_name": "旁白",
                        "text": "这是用于按钮链路冒烟检查的分段。",
                        "emotion_description": "平静",
                    }
                ]
                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={"json_text": json.dumps(seg_payload, ensure_ascii=False)},
                )
                self.assertEqual(imported_segments.status_code, 200)
                segments = imported_segments.json().get("segments") or []
                self.assertEqual(len(segments), 1)
                segment_id = segments[0]["id"]

                listed_segments = client.get(f"/api/v1/projects/{project_id}/segments")
                self.assertEqual(listed_segments.status_code, 200)
                self.assertEqual(len(listed_segments.json()), 1)

                refreshed = client.post(f"/api/v1/projects/{project_id}/segments/refresh-status")
                self.assertEqual(refreshed.status_code, 200)
                self.assertEqual(len(refreshed.json()), 1)

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

                cleanup_tmp = client.post(f"/api/v1/projects/{project_id}/cleanup-tmp-files")
                self.assertEqual(cleanup_tmp.status_code, 200)

                cleanup_debug = client.post(
                    f"/api/v1/projects/{project_id}/segments/cleanup-debug-files"
                )
                self.assertEqual(cleanup_debug.status_code, 200)

                cleanup_prompts = client.post(
                    f"/api/v1/projects/{project_id}/roles/cleanup-prompts"
                )
                self.assertEqual(cleanup_prompts.status_code, 200)

                deleted = client.delete(f"/api/v1/projects/{project_id}")
                self.assertEqual(deleted.status_code, 200)

    def test_voice_library_bind_flow_smoke(self) -> None:
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
                        "/api/v1/projects",
                        json={"name": "voice-bind-project", "description": "voice flow"},
                    )
                    self.assertEqual(created.status_code, 200)
                    project_id = created.json()["id"]

                    role_resp = client.post(
                        f"/api/v1/projects/{project_id}/roles",
                        json={"name": "角色A", "role_type": "character"},
                    )
                    self.assertEqual(role_resp.status_code, 200)
                    role_id = role_resp.json()["id"]

                    voice_resp = client.post(
                        "/api/v1/voices",
                        json={"name": "测试音色", "tags": ["女"], "description": "runtime smoke"},
                    )
                    self.assertEqual(voice_resp.status_code, 200)
                    voice_id = voice_resp.json()["id"]

                    uploaded = client.post(
                        f"/api/v1/voices/{voice_id}/audio",
                        files={"file": ("sample.wav", make_wav_bytes(sample_value=500), "audio/wav")},
                    )
                    self.assertEqual(uploaded.status_code, 200)

                    fetched = client.get(f"/api/v1/voices/{voice_id}/audio")
                    self.assertEqual(fetched.status_code, 200)

                    bound = client.post(
                        f"/api/v1/projects/{project_id}/roles/{role_id}/bind-voice",
                        json={"voice_id": voice_id},
                    )
                    self.assertEqual(bound.status_code, 200)
                    self.assertEqual(bound.json().get("speaker_voice_id"), voice_id)

                    listed_roles = client.get(f"/api/v1/projects/{project_id}/roles")
                    self.assertEqual(listed_roles.status_code, 200)
                    self.assertTrue(
                        any(r.get("id") == role_id and r.get("speaker_voice_id") == voice_id for r in listed_roles.json())
                    )

                    deleted = client.delete(f"/api/v1/voices/{voice_id}")
                    self.assertEqual(deleted.status_code, 409)
                    detail = deleted.json().get("detail") or {}
                    self.assertEqual(detail.get("role_dependency_count"), 1)

                    listed_roles_after_delete = client.get(f"/api/v1/projects/{project_id}/roles")
                    self.assertEqual(listed_roles_after_delete.status_code, 200)
                    role_after_delete = next(
                        r for r in listed_roles_after_delete.json() if r.get("id") == role_id
                    )
                    self.assertEqual(role_after_delete.get("speaker_voice_id"), voice_id)
                    self.assertTrue(role_after_delete.get("speaker_audio"))

                    role_audio_after_delete = client.get(
                        f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio"
                    )
                    self.assertEqual(role_audio_after_delete.status_code, 200)

    def test_websocket_batch_synthesis_guardrails(self) -> None:
        clear_config_caches()
        app = create_app()

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/synthesis/any-project") as ws:
                ws.send_json({"action": "ping"})
                msg = ws.receive_json()
            self.assertEqual(msg.get("type"), "error")
            self.assertIn("action=start", msg.get("message", ""))

        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "ws-guardrail-project", "description": "ws flow"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "旁白", "role_type": "narrator"},
                )
                self.assertEqual(role_resp.status_code, 200)

                seg_payload = [
                    {
                        "speaker_name": "旁白",
                        "text": "websocket 合成守卫测试",
                        "emotion_description": "平静",
                    }
                ]
                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={"json_text": json.dumps(seg_payload, ensure_ascii=False)},
                )
                self.assertEqual(imported_segments.status_code, 200)
                segment_id = imported_segments.json()["segments"][0]["id"]

                with patch(
                    "dubbing_service.app.routers.settings._load_raw_file_config",
                    return_value={"tts_backend": "indextts"},
                ):
                    with client.websocket_connect(f"/api/v1/ws/synthesis/{project_id}") as ws:
                        ws.send_json({"action": "start", "segment_ids": [segment_id]})
                        msg = ws.receive_json()

                self.assertEqual(msg.get("type"), "error")
                self.assertIn("缺少参考音频", msg.get("message", ""))

    def test_websocket_batch_synthesis_persists_generating_and_stop_keeps_unstarted_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "ws-stop-project", "description": "ws stop"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "旁白", "role_type": "narrator"},
                )
                self.assertEqual(role_resp.status_code, 200)
                role_id = role_resp.json()["id"]

                uploaded = client.post(
                    f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
                    files={"file": ("prompt.wav", make_wav_bytes(sample_value=600), "audio/wav")},
                )
                self.assertEqual(uploaded.status_code, 200)

                seg_payload = [
                    {
                        "speaker_name": "旁白",
                        "text": "第一段 websocket stop 测试",
                        "emotion_description": "平静",
                    },
                    {
                        "speaker_name": "旁白",
                        "text": "第二段 websocket stop 测试",
                        "emotion_description": "平静",
                    },
                ]
                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={"json_text": json.dumps(seg_payload, ensure_ascii=False)},
                )
                self.assertEqual(imported_segments.status_code, 200)
                segment_ids = [seg["id"] for seg in imported_segments.json()["segments"]]

                block_event = threading.Event()

                def _fake_synthesize(*args, **kwargs):
                    self.assertTrue(block_event.wait(timeout=2))
                    return {"success": False, "error": "stopped for test"}

                with patch(
                    "dubbing_service.app.routers.settings._load_raw_file_config",
                    return_value={"tts_backend": "indextts"},
                ), patch(
                    "dubbing_service.app.routers.ws_synthesis._synthesize_one",
                    side_effect=_fake_synthesize,
                ):
                    with client.websocket_connect(f"/api/v1/ws/synthesis/{project_id}") as ws:
                        ws.send_json({"action": "start", "segment_ids": segment_ids})
                        batch_start = ws.receive_json()
                        self.assertEqual(batch_start.get("type"), "batch_start")

                        deadline = time.time() + 1.5
                        while time.time() < deadline:
                            current_segments = {s.id: s for s in load_segments(project_id)}
                            if current_segments[segment_ids[0]].status == "generating":
                                break
                            time.sleep(0.02)

                        current_segments = {s.id: s for s in load_segments(project_id)}
                        self.assertEqual(current_segments[segment_ids[0]].status, "generating")
                        self.assertEqual(current_segments[segment_ids[1]].status, "pending")

                        ws.send_json({"action": "stop"})
                        block_event.set()

                        first_result = ws.receive_json()
                        final_msg = ws.receive_json()

                self.assertEqual(first_result.get("type"), "segment_result")
                self.assertEqual(first_result.get("segment_id"), segment_ids[0])
                self.assertFalse(first_result.get("success"))
                self.assertEqual(final_msg.get("type"), "batch_complete")
                self.assertTrue(final_msg.get("stopped"))

                final_segments = {s.id: s for s in load_segments(project_id)}
                self.assertEqual(final_segments[segment_ids[0]].status, "failed")
                self.assertEqual(final_segments[segment_ids[1]].status, "pending")

    def test_reset_stale_preserves_actively_running_segment_and_blocks_duplicate_batch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()
            app = create_app()

            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/projects",
                    json={"name": "ws-refresh-project", "description": "ws refresh"},
                )
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["id"]

                role_resp = client.post(
                    f"/api/v1/projects/{project_id}/roles",
                    json={"name": "旁白", "role_type": "narrator"},
                )
                self.assertEqual(role_resp.status_code, 200)
                role_id = role_resp.json()["id"]

                uploaded = client.post(
                    f"/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio",
                    files={"file": ("prompt.wav", make_wav_bytes(sample_value=700), "audio/wav")},
                )
                self.assertEqual(uploaded.status_code, 200)

                seg_payload = [
                    {
                        "speaker_name": "旁白",
                        "text": "第一段 websocket 刷新恢复测试",
                        "emotion_description": "平静",
                    },
                    {
                        "speaker_name": "旁白",
                        "text": "第二段 websocket 刷新恢复测试",
                        "emotion_description": "平静",
                    },
                ]
                imported_segments = client.post(
                    f"/api/v1/projects/{project_id}/segments/import-json",
                    json={"json_text": json.dumps(seg_payload, ensure_ascii=False)},
                )
                self.assertEqual(imported_segments.status_code, 200)
                segment_ids = [seg["id"] for seg in imported_segments.json()["segments"]]

                block_event = threading.Event()

                def _fake_synthesize(*args, **kwargs):
                    self.assertTrue(block_event.wait(timeout=2))
                    return {"success": False, "error": "stopped for refresh test"}

                with patch(
                    "dubbing_service.app.routers.settings._load_raw_file_config",
                    return_value={"tts_backend": "indextts"},
                ), patch(
                    "dubbing_service.app.routers.ws_synthesis._synthesize_one",
                    side_effect=_fake_synthesize,
                ):
                    with client.websocket_connect(f"/api/v1/ws/synthesis/{project_id}") as ws:
                        ws.send_json({"action": "start", "segment_ids": segment_ids})
                        batch_start = ws.receive_json()
                        self.assertEqual(batch_start.get("type"), "batch_start")

                        deadline = time.time() + 1.5
                        while time.time() < deadline:
                            current_segments = {s.id: s for s in load_segments(project_id)}
                            if current_segments[segment_ids[0]].status == "generating":
                                break
                            time.sleep(0.02)

                        current_segments = {s.id: s for s in load_segments(project_id)}
                        self.assertEqual(current_segments[segment_ids[0]].status, "generating")
                        self.assertEqual(current_segments[segment_ids[1]].status, "pending")

                        reset_resp = client.post(
                            f"/api/v1/projects/{project_id}/segments/reset-stale",
                        )
                        self.assertEqual(reset_resp.status_code, 200)
                        reset_data = reset_resp.json()
                        self.assertEqual(reset_data.get("active_generating_count"), 1)
                        self.assertEqual(reset_data.get("reset_count"), 0)

                        after_reset = {s.id: s for s in load_segments(project_id)}
                        self.assertEqual(after_reset[segment_ids[0]].status, "generating")
                        self.assertEqual(after_reset[segment_ids[1]].status, "pending")

                        plan_resp = client.post(
                            f"/api/v1/projects/{project_id}/segments/synthesize/plan",
                            json={"retry_failed": True},
                        )
                        self.assertEqual(plan_resp.status_code, 409)
                        self.assertIn("后台批量合成正在执行", plan_resp.text)

                        ws.send_json({"action": "stop"})
                        block_event.set()

                        first_result = ws.receive_json()
                        final_msg = ws.receive_json()

                self.assertEqual(first_result.get("type"), "segment_result")
                self.assertEqual(first_result.get("segment_id"), segment_ids[0])
                self.assertFalse(first_result.get("success"))
                self.assertEqual(final_msg.get("type"), "batch_complete")
                self.assertTrue(final_msg.get("stopped"))

                final_segments = {s.id: s for s in load_segments(project_id)}
                self.assertEqual(final_segments[segment_ids[0]].status, "failed")
                self.assertEqual(final_segments[segment_ids[1]].status, "pending")


if __name__ == "__main__":
    unittest.main()
