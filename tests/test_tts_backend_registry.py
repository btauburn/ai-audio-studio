from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dubbing_service.app.core.tts_backends import (
    BACKEND_SPECS,
    build_backend_command,
    public_backend_definitions,
)


class TTSBackendRegistryTests(unittest.TestCase):
    def test_registry_exposes_all_restored_backends_on_distinct_ports(self) -> None:
        self.assertEqual(
            set(BACKEND_SPECS),
            {"indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"},
        )
        self.assertEqual(
            {spec.port for spec in BACKEND_SPECS.values()},
            {8001, 8002, 8003, 8004},
        )
        public = {item["id"]: item for item in public_backend_definitions()}
        self.assertIn("builtin_speaker", public["qwen3tts_cv"]["capabilities"])
        self.assertIn("voice_design", public["qwen3tts"]["capabilities"])

    def test_qwen_command_uses_isolated_environment_and_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "environments" / "qwen3tts" / ".venv" / "Scripts" / "python.exe"
            server = root / "service" / "qwen3tts_customvoice_server.py"
            model = root / "models" / "qwen3_tts" / "Qwen3-TTS-CustomVoice"
            for path in (python, server):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            model.mkdir(parents=True)

            command = build_backend_command(
                root,
                "qwen3tts_cv",
                root / ".venv" / "Scripts" / "python.exe",
            )

            self.assertEqual(Path(command[0]), python.resolve())
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--port") + 1], "8004")
            self.assertEqual(command[command.index("--model_dir") + 1], str(model.resolve()))
            self.assertEqual(command[-4:], ["--device", "cuda:0", "--dtype", "bfloat16"])

    def test_missing_isolated_environment_fails_before_old_worker_is_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "service" / "qwen3tts_server.py"
            model = root / "models" / "qwen3_tts" / "Qwen3-TTS-VoiceDesign"
            server.parent.mkdir(parents=True)
            server.touch()
            model.mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "environment is missing"):
                build_backend_command(
                    root,
                    "qwen3tts",
                    root / "missing-python.exe",
                )

    def test_legacy_qwen_environment_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / ".venv_qwen3tts" / "Scripts" / "python.exe"
            legacy.parent.mkdir(parents=True)
            legacy.touch()

            with self.assertRaisesRegex(FileNotFoundError, "environment is missing"):
                build_backend_command(root, "qwen3tts", root / "missing-platform.exe")

    def test_cosyvoice_does_not_fall_back_to_platform_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            platform_python = root / ".venv" / "Scripts" / "python.exe"
            platform_python.parent.mkdir(parents=True)
            platform_python.touch()

            with self.assertRaisesRegex(FileNotFoundError, "environment is missing"):
                build_backend_command(root, "cosyvoice3", platform_python)


if __name__ == "__main__":
    unittest.main()
