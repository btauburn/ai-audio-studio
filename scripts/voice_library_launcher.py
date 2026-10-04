"""Start the existing Studio and private review services, then open Voice Lab.

Used by the user's desktop/sign-in shortcuts. This launcher creates no new
generation request or provider configuration. The Studio owner can resume a
previously authorized book request when its automatic recovery is enabled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:9000/web/voice-lab.html"


def studio_ready(root: Path) -> bool:
    try:
        with build_opener(ProxyHandler({})).open(
                "http://127.0.0.1:9000/api/v1/voice-library/status", timeout=2) as response:
            data = json.load(response)
        return (data.get("service") == "ai-audio-studio-voice-library"
                and Path(data["projects_root"]).resolve() == (root / "dubbing_projects").resolve())
    except (OSError, ValueError, KeyError):
        return False


def port_in_use() -> bool:
    with socket.socket() as connection:
        connection.settimeout(.4)
        return connection.connect_ex(("127.0.0.1", 9000)) == 0


@contextmanager
def launch_lock(path: Path):
    """Serialize double-click/sign-in launches without another service daemon."""
    import msvcrt
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 90
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Another Voice Library launch is still starting. Try again shortly.")
                time.sleep(.25)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def ensure_services(root: Path, logs: Path) -> dict:
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError("Studio's Python environment is missing; no software was installed.")
    result = {"studio": "already_running", "gpu_started": False, "url": URL}
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    if not studio_ready(root):
        if port_in_use():
            raise RuntimeError("Port 9000 is occupied by another or older Studio instance. No process was stopped.")
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with (logs / "studio.stdout.log").open("ab") as output, (logs / "studio.stderr.log").open("ab") as error:
            process = subprocess.Popen([str(python), "-m", "uvicorn", "dubbing_service.app.main:app",
                "--host", "127.0.0.1", "--port", "9000", "--no-proxy-headers"], cwd=root,
                stdin=subprocess.DEVNULL, stdout=output, stderr=error, env=environment, creationflags=flags)
        result.update(studio="started", launcher_pid=process.pid)
        deadline = time.monotonic() + 75
        while not studio_ready(root):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"Studio did not become ready. See {logs / 'studio.stderr.log'}")
            time.sleep(.5)
    config = Path.home() / ".config" / "audiobook-delivery" / "config.json"
    if config.is_file():
        completed = subprocess.run([str(python), "-m", "dubbing_service.app.audiobook.delivery",
            "--config", str(config), "start"], cwd=root, capture_output=True, text=True,
            encoding="utf-8", timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if completed.returncode:
            result["private_review"] = "unavailable; desktop library is ready"
        else:
            result["private_review"] = json.loads(completed.stdout).get("service", "ready")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Start services quietly at Windows sign-in")
    args = parser.parse_args(argv)
    logs = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AI Audio Studio" / "voice-library"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with launch_lock(logs / "launch.lock"):
            result = ensure_services(ROOT, logs)
            (logs / "last-launch.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.no_browser:
            os.startfile(URL)
        if sys.stdout:
            print(json.dumps(result))
        return 0
    except Exception as error:
        message = str(error)
        (logs / "last-error.txt").write_text(message, encoding="utf-8")
        if not args.no_browser:
            ctypes.windll.user32.MessageBoxW(None, message, "Voice Library could not start", 0x10)
        if sys.stderr:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
