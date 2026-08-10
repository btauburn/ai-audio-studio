import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.models.project import Segment
from dubbing_service.app.providers.llm import (
    LLMResult,
    OpenAIStyleLLMProvider,
    is_deepseek_base_url,
)
from dubbing_service.app.routers import llm_configs as llm_configs_router
from dubbing_service.app.routers import segments as segments_router
from dubbing_service.app.storage import llm_configs as llm_config_storage
from dubbing_service.app.storage.llm_configs import LLMConfig
from dubbing_service.app.storage.projects import create_project, load_project
from dubbing_service.app.storage.segmentation_commit import commit_auto_segmentation
from dubbing_service.app.storage.segments import load_segments, save_segments


class TestDeepSeekProvider(unittest.TestCase):
    def test_deepseek_detection_uses_parsed_hostname(self) -> None:
        for base_url in (
            "https://deepseek.com",
            "https://api.deepseek.com",
            "HTTPS://API.DEEPSEEK.COM.:443/v1",
        ):
            with self.subTest(base_url=base_url):
                self.assertTrue(is_deepseek_base_url(base_url))

        for bypass_url in (
            "https://deepseek.com.evil.example/v1",
            "https://evil-deepseek.com/v1",
            "https://evil.example/deepseek.com/v1",
            "https://deepseek.com@evil.example/v1",
            "https://evil.example/?next=https://api.deepseek.com",
            "https://evil.example/#deepseek.com",
            "not-a-url/deepseek.com",
        ):
            with self.subTest(base_url=bypass_url):
                self.assertFalse(is_deepseek_base_url(bypass_url))

    def test_deepseek_like_custom_host_does_not_change_model(self) -> None:
        provider = OpenAIStyleLLMProvider(
            base_url="https://deepseek.com.evil.example/v1",
            api_key="test-only",
        )

        self.assertEqual(provider.model, "gpt-4o-mini")

    def test_local_custom_provider_remains_supported(self) -> None:
        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "pong"}}]},
            )

        provider = OpenAIStyleLLMProvider(
            base_url="http://127.0.0.1:9010/v1/",
            api_key="test-only",
            transport=httpx.MockTransport(handler),
        )
        result = provider.chat([{"role": "user", "content": "ping"}])

        self.assertTrue(result.success)
        self.assertEqual(provider.model, "gpt-4o-mini")
        self.assertEqual(captured_urls, ["http://127.0.0.1:9010/v1/chat/completions"])

    def test_invalid_or_credential_bearing_base_url_is_rejected(self) -> None:
        for base_url in (
            "ftp://api.deepseek.com/v1",
            "https://user:password@api.deepseek.com/v1",
            "https://api.deepseek.com:99999/v1",
            "https://api.deepseek.com/\n.evil",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(RuntimeError):
                    OpenAIStyleLLMProvider(base_url=base_url, api_key="test-only")

    def test_deepseek_defaults_to_v4_flash_and_sends_v4_options(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                headers={"x-request-id": "req-1"},
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"segments": []}'}}
                    ],
                    "usage": {"total_tokens": 9},
                },
            )

        provider = OpenAIStyleLLMProvider(
            base_url="https://api.deepseek.com",
            api_key="test-only",
            transport=httpx.MockTransport(handler),
            sleep=lambda _: None,
        )
        result = provider.chat(
            [{"role": "user", "content": "请返回 JSON"}],
            thinking={"type": "disabled"},
            reasoning_effort="high",
            response_format={"type": "json_object"},
        )

        self.assertTrue(result.success)
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertEqual(captured[0]["reasoning_effort"], "high")
        self.assertEqual(captured[0]["response_format"], {"type": "json_object"})
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.request_id, "req-1")
        self.assertEqual(result.usage, {"total_tokens": 9})

    def test_retries_429_but_does_not_retry_auth_errors(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        def retry_handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "busy"})
            return httpx.Response(
                200,
                json={"choices": [{"finish_reason": "stop", "message": {"content": "pong"}}]},
            )

        provider = OpenAIStyleLLMProvider(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-only",
            transport=httpx.MockTransport(retry_handler),
            sleep=sleeps.append,
        )
        result = provider.chat([{"role": "user", "content": "ping"}])
        self.assertTrue(result.success)
        self.assertEqual(attempts, 2)
        self.assertEqual(sleeps, [0.0])

        auth_attempts = 0

        def auth_handler(_: httpx.Request) -> httpx.Response:
            nonlocal auth_attempts
            auth_attempts += 1
            return httpx.Response(401, json={"error": "unauthorized"})

        provider = OpenAIStyleLLMProvider(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-only",
            transport=httpx.MockTransport(auth_handler),
            sleep=lambda _: None,
        )
        result = provider.chat([{"role": "user", "content": "ping"}])
        self.assertFalse(result.success)
        self.assertFalse(result.retryable)
        self.assertEqual(result.status_code, 401)
        self.assertEqual(auth_attempts, 1)


class TestSegmentationValidation(unittest.TestCase):
    def test_content_coverage_ignores_punctuation_but_not_missing_text(self) -> None:
        self.assertIsNone(
            segments_router._validate_source_coverage(
                "甲说：你好！乙回答，好。",
                [{"text": "甲说你好"}, {"text": "乙回答：好"}],
            )
        )
        error = segments_router._validate_source_coverage(
            "甲说：你好！乙回答，好。",
            [{"text": "甲说你好"}],
        )
        self.assertIsNotNone(error)
        self.assertIn("遗漏", error)
        self.assertIsNotNone(
            segments_router._validate_source_coverage("ABC", [{"text": "abc"}])
        )


class TestDeepSeekConfigMigration(unittest.TestCase):
    def test_seeded_defaults_do_not_trust_deepseek_text_outside_hostname(self) -> None:
        cases = (
            ("https://api.deepseek.com/v1", "deepseek-v4-flash", True),
            ("https://provider.example/v1/deepseek.com", "gpt-4o-mini", False),
            ("https://deepseek.com.attacker.example/v1", "gpt-4o-mini", False),
        )

        for base_url, expected_model, expected_json_mode in cases:
            with (
                self.subTest(base_url=base_url),
                patch.object(
                    llm_configs_router,
                    "get_config",
                    return_value=SimpleNamespace(llm_base_url=base_url),
                ),
                patch.object(llm_configs_router, "save_llm_configs") as save_configs,
            ):
                seeded = llm_configs_router._seed_default_if_empty([])

                self.assertEqual(seeded[0].model, expected_model)
                self.assertEqual(seeded[0].response_format_json, expected_json_mode)
                save_configs.assert_called_once_with(seeded)

    def test_legacy_presets_are_migrated_once_and_keep_ids_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "llm_models.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "chat-id",
                            "name": "deepseek-chat",
                            "provider_id": "provider-id",
                            "model": "deepseek-chat",
                            "max_tokens": 8000,
                            "is_default": False,
                        },
                        {
                            "id": "reasoner-id",
                            "name": "deepseek-reasoner",
                            "provider_id": "provider-id",
                            "model": "deepseek-reasoner",
                            "max_tokens": 64000,
                            "is_default": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(llm_config_storage, "_llm_models_path", return_value=path):
                first = llm_config_storage.load_llm_configs()
                first_bytes = path.read_bytes()
                second = llm_config_storage.load_llm_configs()
                second_bytes = path.read_bytes()

        self.assertEqual(path.name, "llm_models.json")
        self.assertEqual([cfg.id for cfg in first], ["chat-id", "reasoner-id"])
        self.assertEqual([cfg.provider_id for cfg in first], ["provider-id", "provider-id"])
        self.assertEqual([cfg.model for cfg in first], ["deepseek-v4-flash", "deepseek-v4-pro"])
        self.assertTrue(first[0].is_default)
        self.assertFalse(first[1].is_default)
        self.assertEqual([cfg.model for cfg in second], ["deepseek-v4-flash", "deepseek-v4-pro"])
        self.assertEqual(first_bytes, second_bytes)

    def test_old_ui_update_does_not_disable_flash_safety_fields(self) -> None:
        config = LLMConfig(
            id="flash",
            name="DeepSeek V4 Flash",
            model="deepseek-v4-flash",
            max_tokens=64000,
            response_format_json=True,
            thinking="disabled",
            fallback_model="deepseek-v4-pro",
            fallback_enabled=True,
            is_default=True,
        )
        # This payload mirrors the pre-V4 UI, which knows none of the new fields.
        payload = llm_configs_router.LLMConfigPayload(
            name="DeepSeek V4 Flash",
            model="deepseek-v4-flash",
            max_tokens=64000,
            response_format_json=True,
        )
        with (
            patch.object(llm_configs_router, "load_llm_configs", return_value=[config]),
            patch.object(llm_configs_router, "save_llm_configs"),
        ):
            response = asyncio.run(llm_configs_router.api_update_llm_config("flash", payload))

        self.assertEqual(response.thinking, "disabled")
        self.assertTrue(response.fallback_enabled)
        self.assertEqual(response.fallback_model, "deepseek-v4-pro")


class TestAutoSegmentationSafety(unittest.TestCase):
    def setUp(self) -> None:
        self._old_root = os.environ.get("DUBBING_PROJECTS_ROOT")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DUBBING_PROJECTS_ROOT"] = self._tmp.name
        clear_config_caches()

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("DUBBING_PROJECTS_ROOT", None)
        else:
            os.environ["DUBBING_PROJECTS_ROOT"] = self._old_root
        clear_config_caches()
        self._tmp.cleanup()

    def _seed_project(self):
        project = create_project("安全分段测试")
        root = Path(self._tmp.name) / project.id
        old_audio = root / "audio" / "segments" / "old.wav"
        old_audio.write_bytes(b"old-audio")
        (root / "text" / "raw.txt").write_text("旧原文", encoding="utf-8")
        save_segments(
            project.id,
            [Segment(id="old-seg", index=0, text="旧原文", audio_file="audio/segments/old.wav")],
        )
        return project, root, old_audio

    @staticmethod
    def _flash_config() -> LLMConfig:
        return LLMConfig(
            id="flash",
            name="DeepSeek V4 Flash",
            model="deepseek-v4-flash",
            max_tokens=64000,
            response_format_json=True,
            thinking="disabled",
            fallback_model="deepseek-v4-pro",
            fallback_enabled=True,
            is_default=True,
        )

    def test_invalid_flash_output_falls_back_once_and_then_commits(self) -> None:
        project, root, old_audio = self._seed_project()

        class FakeProvider:
            def __init__(self, result: LLMResult):
                self.result = result
                self.calls = 0

            def chat(self, *_args, **_kwargs):
                self.calls += 1
                return self.result

        flash = FakeProvider(
            LLMResult(
                success=True,
                text='{"segments":[{"speaker_name":"旁白","text":"甲","emotion_description":"平静"}]}',
                raw={},
                finish_reason="stop",
            )
        )
        pro = FakeProvider(
            LLMResult(
                success=True,
                text=(
                    '{"segments":['
                    '{"speaker_name":"旁白","text":"甲说你好",'
                    '"emotion_description":"平静"},'
                    '{"speaker_name":"乙","text":"乙回答好",'
                    '"emotion_description":"自然"}]}'
                ),
                raw={},
                finish_reason="stop",
            )
        )

        with (
            patch.object(segments_router, "load_llm_configs", return_value=[self._flash_config()]),
            patch.object(segments_router, "load_llm_providers", return_value=[]),
            patch.object(segments_router, "_ensure_llm_provider", side_effect=[flash, pro]) as make_provider,
        ):
            response = segments_router.api_auto_segment(
                project.id,
                segments_router.AutoSegmentRequest(
                    text="甲说：你好！乙回答，好。",
                    llm_config_id="flash",
                ),
            )

        self.assertEqual(len(response.segments), 2)
        self.assertEqual(flash.calls, 1)
        self.assertEqual(pro.calls, 1)
        self.assertEqual(make_provider.call_count, 2)
        self.assertEqual((root / "text" / "raw.txt").read_text(encoding="utf-8"), "甲说：你好！乙回答，好。")
        self.assertFalse(old_audio.exists())
        self.assertEqual([s.text for s in load_segments(project.id)], ["甲说你好", "乙回答好"])
        self.assertIn("乙", {r.name for r in load_project(project.id).roles})

    def test_failed_primary_and_fallback_preserve_everything(self) -> None:
        project, root, old_audio = self._seed_project()
        old_segments = (root / "segments.json").read_bytes()
        old_project = (root / "project.json").read_bytes()

        class FakeProvider:
            def chat(self, *_args, **_kwargs):
                return LLMResult(success=False, raw={"status": 503}, status_code=503, retryable=True)

        with (
            patch.object(segments_router, "load_llm_configs", return_value=[self._flash_config()]),
            patch.object(segments_router, "load_llm_providers", return_value=[]),
            patch.object(segments_router, "_ensure_llm_provider", side_effect=[FakeProvider(), FakeProvider()]),
        ):
            with self.assertRaises(HTTPException) as raised:
                segments_router.api_auto_segment(
                    project.id,
                    segments_router.AutoSegmentRequest(text="新原文", llm_config_id="flash"),
                )

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual((root / "text" / "raw.txt").read_text(encoding="utf-8"), "旧原文")
        self.assertEqual((root / "segments.json").read_bytes(), old_segments)
        self.assertEqual((root / "project.json").read_bytes(), old_project)
        self.assertEqual(old_audio.read_bytes(), b"old-audio")

    def test_same_project_auto_segmentation_is_mutually_exclusive(self) -> None:
        project, _, _ = self._seed_project()
        self.assertTrue(segments_router._begin_auto_segment(project.id))
        try:
            self.assertFalse(segments_router._begin_auto_segment(project.id))
        finally:
            segments_router._end_auto_segment(project.id)
        self.assertTrue(segments_router._begin_auto_segment(project.id))
        segments_router._end_auto_segment(project.id)


class TestSegmentationCommitRollback(unittest.TestCase):
    def setUp(self) -> None:
        self._old_root = os.environ.get("DUBBING_PROJECTS_ROOT")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DUBBING_PROJECTS_ROOT"] = self._tmp.name
        clear_config_caches()

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("DUBBING_PROJECTS_ROOT", None)
        else:
            os.environ["DUBBING_PROJECTS_ROOT"] = self._old_root
        clear_config_caches()
        self._tmp.cleanup()

    def test_multi_file_commit_rolls_back_if_a_replace_fails(self) -> None:
        project = create_project("事务测试")
        root = Path(self._tmp.name) / project.id
        (root / "text" / "raw.txt").write_text("旧原文", encoding="utf-8")
        save_segments(project.id, [Segment(id="old", index=0, text="旧原文")])
        before = {
            "raw": (root / "text" / "raw.txt").read_bytes(),
            "segments": (root / "segments.json").read_bytes(),
            "project": (root / "project.json").read_bytes(),
        }

        real_replace = os.replace
        promoted = 0

        def flaky_replace(src, dst):
            nonlocal promoted
            if str(src).endswith(".stage"):
                promoted += 1
                if promoted == 2:
                    raise OSError("simulated replace failure")
            return real_replace(src, dst)

        with patch("dubbing_service.app.storage.segmentation_commit.os.replace", side_effect=flaky_replace):
            with self.assertRaises(OSError):
                commit_auto_segmentation(
                    project.id,
                    raw_text="新原文",
                    segments=[Segment(id="new", index=0, text="新原文")],
                    new_roles=[],
                )

        self.assertEqual((root / "text" / "raw.txt").read_bytes(), before["raw"])
        self.assertEqual((root / "segments.json").read_bytes(), before["segments"])
        self.assertEqual((root / "project.json").read_bytes(), before["project"])


if __name__ == "__main__":
    unittest.main()
