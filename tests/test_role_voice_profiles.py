from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.project import Project, Role, Segment
from dubbing_service.app.providers.tts import Qwen3TTSCustomVoiceProvider, Qwen3TTSProvider
from dubbing_service.app.routers.synthesis import _collect_missing_speaker_roles


class RoleVoiceProfileTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_config_caches()

    def test_legacy_role_remains_backward_compatible(self) -> None:
        role = Role(id="role1", name="旁白")
        self.assertIsNone(role.tts_backend)
        self.assertIsNone(role.qwen3_speaker)
        self.assertIsNone(role.voice_description)

    def test_custom_voice_uses_segment_then_role_then_safe_default(self) -> None:
        provider = Qwen3TTSCustomVoiceProvider()
        role = Role(
            id="role1",
            name="女主",
            qwen3_speaker="serena",
            qwen3_language="Chinese",
            style_instruction="沉稳克制",
        )
        segment = Segment(id="seg1", role_id=role.id, text="你好", index=0)

        payload = provider._build_payload("project", role, segment)
        self.assertEqual(payload["speaker"], "serena")
        self.assertEqual(payload["language"], "Chinese")
        self.assertEqual(payload["instruct"], "沉稳克制")

        segment.qwen3_speaker = "vivian"
        self.assertEqual(provider._build_payload("project", role, segment)["speaker"], "vivian")

    def test_voice_design_preserves_identity_and_appends_performance(self) -> None:
        provider = Qwen3TTSProvider()
        role = Role(
            id="role1",
            name="长者",
            voice_description="低沉、有岁月感的男性声音",
            style_instruction="语速偏慢",
        )
        segment = Segment(id="seg1", role_id=role.id, text="往事如烟。", index=0, emotion_text="略带怀念")

        instruct = provider._build_payload("project", role, segment)["instruct"]
        self.assertIn("低沉、有岁月感", instruct)
        self.assertIn("语速偏慢", instruct)
        self.assertIn("略带怀念", instruct)

    def test_role_backend_prevents_false_missing_reference_error(self) -> None:
        role = Role(id="role1", name="设计音色", tts_backend="qwen3tts")
        project = Project(id="project1", name="test", roles=[role])
        segment = Segment(id="seg1", role_id=role.id, text="测试", index=0)
        self.assertEqual(_collect_missing_speaker_roles(project, [segment], "indextts"), [])

    def test_voice_profile_api_partially_updates_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"DUBBING_PROJECTS_ROOT": temporary},
            clear=False,
        ):
            clear_config_caches()
            app = create_app()
            with TestClient(app) as client:
                project = client.post("/api/v1/projects", json={"name": "profile-test"}).json()
                role = client.post(
                    f"/api/v1/projects/{project['id']}/roles",
                    json={"name": "角色", "role_type": "character"},
                ).json()
                response = client.patch(
                    f"/api/v1/projects/{project['id']}/roles/{role['id']}/voice-profile",
                    json={
                        "tts_backend": "qwen3tts_cv",
                        "qwen3_speaker": "serena",
                        "qwen3_language": "Chinese",
                        "style_instruction": "温柔",
                    },
                )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["tts_backend"], "qwen3tts_cv")
            self.assertEqual(data["qwen3_speaker"], "serena")
            self.assertEqual(data["style_instruction"], "温柔")
            self.assertEqual(data["name"], "角色")


if __name__ == "__main__":
    unittest.main()
