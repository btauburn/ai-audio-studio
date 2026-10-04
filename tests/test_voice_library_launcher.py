"""Launcher decision tests; real Windows launch is separately smoke-tested."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts import voice_library_launcher as launcher


def test_existing_correct_studio_is_reused_without_a_worker_or_second_process(tmp_path, monkeypatch):
    runtime = tmp_path / ".venv" / "Scripts"
    runtime.mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"fixture")
    monkeypatch.setattr(launcher, "studio_ready", lambda root: True)
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path / "no-config")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("No duplicate Studio or GPU worker"))
    result = launcher.ensure_services(tmp_path, tmp_path)
    assert result["studio"] == "already_running" and result["gpu_started"] is False


def test_wrong_or_older_server_is_never_stopped_or_replaced(tmp_path, monkeypatch):
    runtime = tmp_path / ".venv" / "Scripts"
    runtime.mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"fixture")
    monkeypatch.setattr(launcher, "studio_ready", lambda root: False)
    monkeypatch.setattr(launcher, "port_in_use", lambda: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("Port owner must be preserved"))
    with pytest.raises(RuntimeError, match="occupied"):
        launcher.ensure_services(tmp_path, tmp_path)


def test_cold_start_uses_exact_hidden_process_argv_and_waits_for_same_workspace(tmp_path, monkeypatch):
    root = tmp_path / "Studio with spaces"
    runtime = root / ".venv" / "Scripts"
    runtime.mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"fixture")
    checks = iter([False, False, True])
    monkeypatch.setattr(launcher, "studio_ready", lambda candidate: next(checks))
    monkeypatch.setattr(launcher, "port_in_use", lambda: False)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path / "no-config")
    calls = []
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=12, poll=lambda: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    result = launcher.ensure_services(root, tmp_path)
    assert result["studio"] == "started"
    argv, kwargs = calls[0]
    assert argv == [str(runtime / "python.exe"), "-m", "uvicorn", "dubbing_service.app.main:app",
                    "--host", "127.0.0.1", "--port", "9000", "--no-proxy-headers"]
    assert kwargs["cwd"] == root and kwargs["creationflags"] & launcher.subprocess.CREATE_NO_WINDOW


def test_service_identity_requires_matching_workspace(tmp_path, monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"service": "ai-audio-studio-voice-library", "projects_root": str(tmp_path / "different")}).encode()
    monkeypatch.setattr(launcher, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **kw: Response()))
    assert launcher.studio_ready(tmp_path) is False
