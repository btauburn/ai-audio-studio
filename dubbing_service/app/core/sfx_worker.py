"""Lifecycle management for the isolated Stable Audio SFX worker.

The platform process intentionally never imports Stable Audio, Torch 2.7 or
Transformers 5.  Those dependencies live in ``environments/stableaudio`` and
are reached only through this loopback worker.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SFX_PORT = 8005
SFX_BASE_URL = "http://127.0.0.1:8005"
_TOKEN_HEADER = "X-Dubbing-Worker-Token"


class SFXWorkerError(RuntimeError):
    """The managed Stable Audio worker could not satisfy an operation."""


@dataclass(frozen=True, slots=True)
class SFXWorkerStatus:
    running: bool
    pid: int | None = None
    model_loaded: bool = False
    model_id: str = "stable-audio-3-small-sfx"
    device: str | None = None
    dtype: str | None = None
    quality_gate_passed: bool = False
    model_ready: bool = False
    license_accepted: bool = False
    runtime_revision_verified: bool = False
    runtime_code_revision: str | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None


_LOCK = threading.RLock()
_PROCESS: subprocess.Popen[str] | None = None
_INSTANCE_TOKEN: str | None = None
_LOGS: "queue.Queue[str]" = queue.Queue(maxsize=1000)


def _port_listening(port: int = SFX_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _runtime_python() -> Path:
    if sys.platform.startswith("win"):
        return PROJECT_ROOT / "environments" / "stableaudio" / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / "environments" / "stableaudio" / ".venv" / "bin" / "python"


def _model_dir() -> Path:
    return PROJECT_ROOT / "models" / "stable_audio_3" / "small-sfx"


def _manifest_path() -> Path:
    return PROJECT_ROOT / "models" / "manifests" / "stable_audio_3_small_sfx.json"


def _read_quality_gate() -> tuple[bool, str | None]:
    """Validate the immutable real-model gate marker, never a mock marker."""

    try:
        from scripts.model_preflight import load_manifest, quality_gate_status

        manifest = load_manifest(_manifest_path())
        passed, detail = quality_gate_status(manifest, project_root=PROJECT_ROOT)
        return passed, None if passed else f"Stable Audio 3 真实 CUDA 质量门: {detail}"
    except Exception as exc:
        return False, f"质量门报告不可用: {exc}"


def _runtime_revision_state() -> tuple[bool, str | None, str | None]:
    """Verify that the isolated distribution came from the pinned Git commit."""

    try:
        from scripts.model_preflight import (
            load_manifest,
            runtime_code_revision_status,
        )

        manifest = load_manifest(_manifest_path())
        passed, detail = runtime_code_revision_status(manifest, _runtime_python())
        revision = str(manifest.get("code_revision")) if passed else None
        return (
            passed,
            revision,
            None if passed else f"Stable Audio 上游代码 revision 校验失败: {detail}",
        )
    except Exception as exc:
        return False, None, f"Stable Audio 上游代码 revision 校验不可用: {exc}"


def _supply_state(*, verify_hash: bool = False) -> tuple[bool, bool, str | None]:
    """Return model integrity and legal acknowledgement state."""

    manifest = _manifest_path()
    if not manifest.is_file():
        return False, False, "Stable Audio 模型 manifest 不存在"
    try:
        from scripts.model_preflight import has_license_acceptance, verify_manifest

        manifest_data, statuses = verify_manifest(
            manifest,
            project_root=PROJECT_ROOT,
            verify_hash=verify_hash,
            groups=None,
        )
        model_ready = all(item.ok for item in statuses)
        accepted = bool(has_license_acceptance(manifest_data, project_root=PROJECT_ROOT))
        return model_ready, accepted, None if model_ready else "Stable Audio 本地文件预检未通过"
    except Exception as exc:
        return False, False, f"Stable Audio 本地文件预检失败: {exc}"


def _reader(proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        try:
            _LOGS.put_nowait(line.rstrip())
        except queue.Full:
            try:
                _LOGS.get_nowait()
                _LOGS.put_nowait(line.rstrip())
            except queue.Empty:
                pass


def _drain_logs(limit: int = 100) -> list[str]:
    items: list[str] = []
    while len(items) < limit:
        try:
            items.append(_LOGS.get_nowait())
        except queue.Empty:
            break
    return items


def _request_json(path: str, *, method: str = "GET", timeout: float = 5.0) -> dict[str, Any]:
    token = _INSTANCE_TOKEN
    headers = {_TOKEN_HEADER: token} if token else {}
    request = urllib.request.Request(
        f"{SFX_BASE_URL}{path}",
        data=None if method.upper() in {"GET", "HEAD"} else b"",
        headers=headers,
        method=method.upper(),
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        raise SFXWorkerError("SFX worker 返回了非对象 JSON")
    return data


def _health(timeout: float = 2.0) -> dict[str, Any]:
    payload = _request_json("/health", timeout=timeout)
    if payload.get("backend") != "stableaudio_sfx":
        raise SFXWorkerError("8005 端口不是 Stable Audio SFX worker")
    if payload.get("instance_token") != _INSTANCE_TOKEN:
        raise SFXWorkerError("SFX worker 实例令牌不匹配")
    return payload


def worker_status(*, include_logs: bool = True) -> SFXWorkerStatus:
    global _PROCESS, _INSTANCE_TOKEN
    gate, gate_error = _read_quality_gate()
    model_ready, license_accepted, supply_error = _supply_state(verify_hash=False)
    runtime_ok, runtime_revision, runtime_error = _runtime_revision_state()
    model_ready = model_ready and runtime_ok
    with _LOCK:
        running = _PROCESS is not None and _PROCESS.poll() is None
        if not running:
            _PROCESS = None
            _INSTANCE_TOKEN = None
            return SFXWorkerStatus(
                running=False,
                quality_gate_passed=gate,
                model_ready=model_ready,
                license_accepted=license_accepted,
                runtime_revision_verified=runtime_ok,
                runtime_code_revision=runtime_revision,
                logs=_drain_logs() if include_logs else [],
                error=supply_error or runtime_error or gate_error,
            )
        try:
            health = _health()
            return SFXWorkerStatus(
                running=True,
                pid=_PROCESS.pid,
                model_loaded=bool(health.get("model_loaded")),
                model_id=str(health.get("model_id") or "stable-audio-3-small-sfx"),
                device=str(health.get("device") or "cuda:0"),
                dtype=str(health.get("dtype") or "float16"),
                quality_gate_passed=gate,
                model_ready=model_ready,
                license_accepted=license_accepted,
                runtime_revision_verified=runtime_ok,
                runtime_code_revision=runtime_revision,
                logs=_drain_logs() if include_logs else [],
                error=supply_error or runtime_error or gate_error,
            )
        except Exception as exc:
            return SFXWorkerStatus(
                running=True,
                pid=_PROCESS.pid,
                quality_gate_passed=gate,
                model_ready=model_ready,
                license_accepted=license_accepted,
                runtime_revision_verified=runtime_ok,
                runtime_code_revision=runtime_revision,
                logs=_drain_logs() if include_logs else [],
                error=str(exc),
            )


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=3)
            return
        except (OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()


def start_worker(*, require_quality_gate: bool = True, timeout_seconds: float = 45.0) -> SFXWorkerStatus:
    """Start exactly one pinned, offline, loopback worker."""

    global _PROCESS, _INSTANCE_TOKEN
    gate, gate_error = _read_quality_gate()
    model_ready, license_accepted, supply_error = _supply_state(verify_hash=True)
    runtime_ok, _, runtime_error = _runtime_revision_state()
    failures = []
    if not model_ready:
        failures.append(supply_error or "模型预检未通过")
    if not license_accepted:
        failures.append("尚未确认 Stability AI Community/Enterprise 与 Gemma 条款")
    if not runtime_ok:
        failures.append(runtime_error or "Stable Audio 上游代码 revision 校验未通过")
    if require_quality_gate and not gate:
        failures.append(gate_error or "真实 CUDA 质量门未通过")
    if failures:
        raise SFXWorkerError("；".join(failures))
    python = _runtime_python()
    server = PROJECT_ROOT / "service" / "stableaudio_sfx_server.py"
    if not python.is_file() or not server.is_file():
        raise SFXWorkerError("Stable Audio 隔离环境或 worker 尚未安装")

    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:
            return worker_status()
        if _port_listening():
            raise SFXWorkerError("端口 8005 已被非当前平台管理的进程占用")
        token = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "DUBBING_WORKER_INSTANCE_TOKEN": token,
            }
        )
        command = [
            str(python), "-u", str(server),
            "--host", "127.0.0.1", "--port", str(SFX_PORT),
            "--model-dir", str(_model_dir()), "--device", "cuda:0", "--dtype", "float16",
            "--model-id", "stable-audio-3-small-sfx",
        ]
        popen_kwargs: dict[str, Any] = {}
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        _PROCESS = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
        _INSTANCE_TOKEN = token
        threading.Thread(target=_reader, args=(_PROCESS,), daemon=True).start()

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if _PROCESS is None or _PROCESS.poll() is not None:
            stop_worker()
            raise SFXWorkerError("Stable Audio SFX worker 启动后异常退出")
        try:
            _health(timeout=0.8)
            return worker_status()
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    stop_worker()
    raise SFXWorkerError(f"等待 Stable Audio SFX worker 就绪超时: {last_error}")


def load_model() -> dict[str, Any]:
    return _request_json("/api/v1/sfx/load-model", method="POST", timeout=900)


def unload_model() -> dict[str, Any] | None:
    status = worker_status(include_logs=False)
    if not status.running:
        return None
    return _request_json("/api/v1/sfx/unload-model", method="POST", timeout=120)


def worker_auth_token() -> str | None:
    """Return the transient token for the in-process HTTP provider only."""

    return _INSTANCE_TOKEN


def stop_worker() -> None:
    global _PROCESS, _INSTANCE_TOKEN
    with _LOCK:
        proc = _PROCESS
        _PROCESS = None
        _INSTANCE_TOKEN = None
    if proc is not None:
        _terminate(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
