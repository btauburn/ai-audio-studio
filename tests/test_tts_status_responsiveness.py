"""An in-progress renderer switch must not pause unrelated Studio reads."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.routers import settings


@pytest.mark.parametrize("read_path", ["/ready", "/voice-lab", "/voice-lab/casting"])
def test_renderer_status_wait_does_not_block_unrelated_reads(monkeypatch, read_path):
    held_lock = threading.Lock()
    status_waiting = threading.Event()
    status_finished = threading.Event()
    read_finished = threading.Event()
    responses = {}
    failures = []

    class ObservedLock:
        def __enter__(self):
            status_waiting.set()
            held_lock.acquire()

        def __exit__(self, *args):
            held_lock.release()

    monkeypatch.setattr(settings, "_tts_process_lock", ObservedLock())
    monkeypatch.setattr(settings, "_tts_process", SimpleNamespace(poll=lambda: None, pid=123))
    monkeypatch.setattr(settings, "_tts_runtime_backend", "qwen3tts")
    monkeypatch.setattr(settings, "_tts_instance_token", "test-instance")
    monkeypatch.setattr(settings, "_tts_log_queue", queue.Queue())
    monkeypatch.setattr(settings, "_load_raw_file_config", lambda: {"tts_backend": "indextts"})

    # No production lifespan, project data, model loading, or worker launch.
    app = FastAPI()
    app.include_router(settings.router, prefix="/api/v1")

    @app.get("/ready")
    async def ready():
        return {"ready": True}

    @app.get("/voice-lab")
    @app.get("/voice-lab/casting")
    def independent_read():
        return {"roles": [{"id": "preserved-role"}]}

    def get(client, name, path, finished):
        try:
            responses[name] = client.get(path)
        except Exception as exc:  # noqa: BLE001 - Report failures from the request thread.
            failures.append(exc)
        finally:
            finished.set()

    held_lock.acquire()
    status_thread = None
    read_thread = None
    read_completed_before_release = False
    try:
        with TestClient(app) as client:
            status_thread = threading.Thread(
                target=get,
                args=(client, "status", "/api/v1/settings/tts-service/status", status_finished),
                daemon=True,
            )
            status_thread.start()
            assert status_waiting.wait(timeout=3), "Status did not reach the renderer lock"
            read_thread = threading.Thread(
                target=get, args=(client, "read", read_path, read_finished), daemon=True,
            )
            read_thread.start()
            try:
                read_completed_before_release = read_finished.wait(timeout=1)
                assert not status_finished.is_set(), "Fixture released the renderer lock early"
            finally:
                held_lock.release()
                status_thread.join(timeout=3)
                read_thread.join(timeout=3)
    finally:
        if held_lock.locked():
            held_lock.release()
        for thread in (status_thread, read_thread):
            if thread is not None:
                thread.join(timeout=3)

    assert not failures
    assert read_completed_before_release, "Renderer status blocked unrelated reads until its lock released"
    assert status_thread is not None and not status_thread.is_alive()
    assert read_thread is not None and not read_thread.is_alive()
    assert responses["read"].status_code == 200
    assert responses["status"].status_code == 200
    assert responses["status"].json() == {
        "running": True, "backend": "qwen3tts", "runtime_backend": "qwen3tts",
        "config_backend": "indextts", "pid": 123, "logs": [],
    }
