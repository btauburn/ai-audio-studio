"""On-demand readiness retains existing worker ownership and production priority."""
from types import SimpleNamespace
import pytest
from dubbing_service.app.audiobook import voice_lab_runtime as runtime
from dubbing_service.app.routers import settings
from dubbing_service.app.core import tts_operations, config
from dubbing_service.app.storage import jobs


def test_healthy_external_worker_is_reused_without_start_or_settings_change(monkeypatch):
    monkeypatch.setattr(runtime, "_healthy", lambda _: True)
    monkeypatch.setattr(settings, "api_start_tts_service", lambda *a: pytest.fail("Do not restart healthy external worker"))
    runtime.ensure_backend_ready("qwen3tts")


def test_other_managed_worker_is_preserved(monkeypatch):
    monkeypatch.setattr(runtime, "_healthy", lambda _: False)
    monkeypatch.setattr(settings, "_get_tts_runtime_state", lambda: (True, 12, "qwen3tts_cv"))
    monkeypatch.setattr(settings, "api_start_tts_service", lambda *a: pytest.fail("Never switch another worker automatically"))
    with pytest.raises(ValueError, match="Another voice worker"):
        runtime.ensure_backend_ready("qwen3tts")


def test_queued_production_prevents_worker_start(tmp_path, monkeypatch):
    (tmp_path / "existing").mkdir(); (tmp_path / "existing" / "project.json").write_text('{}')
    monkeypatch.setattr(config, "get_config", lambda: SimpleNamespace(projects_root_abs=tmp_path))
    monkeypatch.setattr(runtime, "_healthy", lambda _: False)
    monkeypatch.setattr(settings, "_get_tts_runtime_state", lambda: (False, None, None))
    monkeypatch.setattr(tts_operations, "has_active_tts_operation", lambda: False)
    monkeypatch.setattr(jobs, "load_jobs", lambda _: [SimpleNamespace(id="production", job_type="batch_synthesize", status=SimpleNamespace(value="queued"))])
    monkeypatch.setattr(settings, "api_start_tts_service", lambda *a: pytest.fail("Production must retain priority"))
    with pytest.raises(ValueError, match="priority"):
        runtime.ensure_backend_ready("qwen3tts", job_id="preview")


def test_own_job_can_start_absent_worker_without_changing_default(tmp_path, monkeypatch):
    (tmp_path / "preview").mkdir(); (tmp_path / "preview" / "project.json").write_text('{}')
    monkeypatch.setattr(config, "get_config", lambda: SimpleNamespace(projects_root_abs=tmp_path))
    ready = iter([False, False, True])
    monkeypatch.setattr(runtime, "_healthy", lambda _: next(ready))
    monkeypatch.setattr(settings, "_get_tts_runtime_state", lambda: (False, None, None))
    monkeypatch.setattr(tts_operations, "has_active_tts_operation", lambda: False)
    monkeypatch.setattr(jobs, "load_jobs", lambda _: [SimpleNamespace(id="preview", job_type="batch_synthesize", status=SimpleNamespace(value="queued"))])
    calls = []
    monkeypatch.setattr(settings, "api_start_tts_service", lambda payload: calls.append(payload))
    runtime.ensure_backend_ready("qwen3tts", job_id="preview")
    assert len(calls) == 1 and calls[0].backend == "qwen3tts" and calls[0].persist_default is False
    assert calls[0].only_if_absent and calls[0].preserve_outputs


def test_worker_that_started_after_readiness_probe_cannot_be_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "_get_project_root", lambda: tmp_path)
    monkeypatch.setattr(settings, "_get_configured_tts_backend", lambda: "qwen3tts_cv")
    monkeypatch.setattr(settings, "_get_tts_params_from_config", lambda: {})
    monkeypatch.setattr(settings, "build_backend_command", lambda *a, **kw: ["unused"])
    monkeypatch.setattr(settings, "_tts_process", SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(settings, "_kill_process_tree", lambda *a: pytest.fail("Racing worker must be preserved"))
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        settings._start_tts_service_exclusive(settings.TTSServiceStartPayload(
            backend="qwen3tts", persist_default=False, only_if_absent=True, preserve_outputs=True))
    assert caught.value.status_code == 409
