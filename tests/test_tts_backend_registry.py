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
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
            self.assertEqual(command[command.index("--attention") + 1], "auto")

    def test_qwen_attention_is_explicit_and_validated(self) -> None:
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
                qwen_attention="sdpa",
            )
            self.assertEqual(command[command.index("--attention") + 1], "sdpa")

            with self.assertRaisesRegex(ValueError, "unsupported Qwen attention"):
                build_backend_command(
                    root,
                    "qwen3tts_cv",
                    root / ".venv" / "Scripts" / "python.exe",
                    qwen_attention="made-up",
                )

    def test_qwen_execution_profiles_route_to_matching_isolated_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_python = root / "environments" / "qwen3tts" / ".venv" / "Scripts" / "python.exe"
            graph_python = root / "environments" / "faster_qwen3tts" / ".venv" / "Scripts" / "python.exe"
            native_server = root / "service" / "qwen3tts_customvoice_server.py"
            graph_server = root / "service" / "faster_qwen3tts_customvoice_server.py"
            model = root / "models" / "qwen3_tts" / "Qwen3-TTS-CustomVoice"
            for path in (native_python, graph_python, native_server, graph_server):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            model.mkdir(parents=True)

            platform_python = root / ".venv" / "Scripts" / "python.exe"
            current = build_backend_command(
                root,
                "qwen3tts_cv",
                platform_python,
                qwen_execution_profile="native-current",
                qwen_attention="auto",
            )
            sdpa = build_backend_command(
                root,
                "qwen3tts_cv",
                platform_python,
                qwen_execution_profile="native-sdpa",
                qwen_attention="auto",
            )
            graph = build_backend_command(
                root,
                "qwen3tts_cv",
                platform_python,
                qwen_execution_profile="qwen-cudagraph",
            )

            self.assertEqual(Path(current[0]), native_python.resolve())
            self.assertEqual(current[current.index("--attention") + 1], "auto")
            self.assertEqual(current[current.index("--execution-profile") + 1], "native-current")
            self.assertEqual(Path(sdpa[0]), native_python.resolve())
            self.assertEqual(sdpa[sdpa.index("--attention") + 1], "sdpa")
            self.assertEqual(sdpa[sdpa.index("--execution-profile") + 1], "native-sdpa")
            self.assertEqual(Path(graph[0]), graph_python.resolve())
            self.assertIn(str(graph_server.resolve()), graph)
            self.assertEqual(graph[graph.index("--attention") + 1], "sdpa")
            self.assertEqual(graph[graph.index("--execution-profile") + 1], "qwen-cudagraph")

            with self.assertRaisesRegex(ValueError, "unsupported Qwen execution profile"):
                build_backend_command(
                    root,
                    "qwen3tts_cv",
                    platform_python,
                    qwen_execution_profile="unknown",
                )

    def test_customvoice_attention_flag_is_not_misreported_for_voicedesign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "environments" / "qwen3tts" / ".venv" / "Scripts" / "python.exe"
            server = root / "service" / "qwen3tts_server.py"
            model = root / "models" / "qwen3_tts" / "Qwen3-TTS-VoiceDesign"
            for path in (python, server):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            model.mkdir(parents=True)

            command = build_backend_command(
                root,
                "qwen3tts",
                root / ".venv" / "Scripts" / "python.exe",
                qwen_execution_profile="native-current",
                qwen_attention="sdpa",
            )

            self.assertNotIn("--attention", command)
            protected = build_backend_command(root, "qwen3tts", python, qwen_execution_profile="native-current", preserve_outputs=True)
            self.assertEqual(protected[protected.index("--outputs-keep-days") + 1], "0")
            self.assertEqual(protected[protected.index("--outputs-max-files") + 1], "0")
            self.assertNotIn("--outputs-keep-days", command)

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
