from __future__ import annotations

import inspect
import json
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from dubbing_service.app.main import create_app
from dubbing_service.app.routers import qwen3tts, settings
from dubbing_service.app.routers.settings import TTSServiceStartPayload


class _HealthResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_blocking_platform_routes_run_in_fastapi_threadpool() -> None:
    for endpoint in (
        settings.api_test_llm,
        settings.api_unload_tts_model,
        settings.api_start_tts_service,
        settings.api_load_tts_model,
        settings.api_stop_tts_service,
        qwen3tts.api_generate_voice,
        qwen3tts.api_get_status,
    ):
        assert not inspect.iscoroutinefunction(endpoint), endpoint.__name__

    app = create_app()
    health_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/v1/health"
    )
    assert not inspect.iscoroutinefunction(health_route.endpoint)


def test_worker_ready_requires_backend_and_instance_token() -> None:
    proc = Mock()
    proc.poll.return_value = None
    opener = Mock()
    opener.open.return_value = _HealthResponse(
        {
            "status": "healthy",
            "backend": "indextts",
            "instance_token": "expected-token",
        }
    )

    with patch(
        "dubbing_service.app.routers.settings.urllib.request.build_opener",
        return_value=opener,
    ):
        payload = settings._wait_for_tts_ready(
            proc,
            "indextts",
            expected_instance_token="expected-token",
            timeout_seconds=0.1,
        )

    assert payload["backend"] == "indextts"


def test_qwen_worker_ready_requires_requested_profile_and_attention() -> None:
    proc = Mock()
    proc.poll.return_value = None
    opener = Mock()
    opener.open.return_value = _HealthResponse(
        {
            "status": "model_not_loaded",
            "backend": "qwen3tts_cv",
            "instance_token": "expected-token",
            "configured_execution_profile": "native-sdpa",
            "configured_attention": "sdpa",
        }
    )
    with patch(
        "dubbing_service.app.routers.settings.urllib.request.build_opener",
        return_value=opener,
    ):
        payload = settings._wait_for_tts_ready(
            proc,
            "qwen3tts_cv",
            expected_instance_token="expected-token",
            expected_execution_profile="native-sdpa",
            expected_attention="sdpa",
            timeout_seconds=0.1,
        )
    assert payload["configured_execution_profile"] == "native-sdpa"


def test_qwen_worker_rejects_health_with_wrong_execution_profile() -> None:
    proc = Mock()
    proc.poll.return_value = None
    opener = Mock()
    opener.open.return_value = _HealthResponse(
        {
            "status": "model_not_loaded",
            "backend": "qwen3tts_cv",
            "instance_token": "expected-token",
            "configured_execution_profile": "native-current",
            "configured_attention": "auto",
        }
    )
    with patch(
        "dubbing_service.app.routers.settings.urllib.request.build_opener",
        return_value=opener,
    ), patch("dubbing_service.app.routers.settings.time.sleep", return_value=None):
        with pytest.raises(TimeoutError, match="执行配置不匹配"):
            settings._wait_for_tts_ready(
                proc,
                "qwen3tts_cv",
                expected_instance_token="expected-token",
                expected_execution_profile="qwen-cudagraph",
                expected_attention="sdpa",
                timeout_seconds=0.01,
            )


def test_stale_worker_token_is_rejected() -> None:
    proc = Mock()
    proc.poll.return_value = None
    opener = Mock()
    opener.open.return_value = _HealthResponse(
        {
            "status": "healthy",
            "backend": "indextts",
            "instance_token": "stale-token",
        }
    )

    with patch(
        "dubbing_service.app.routers.settings.urllib.request.build_opener",
        return_value=opener,
    ), patch("dubbing_service.app.routers.settings.time.sleep", return_value=None):
        with pytest.raises(TimeoutError, match="实例令牌不匹配"):
            settings._wait_for_tts_ready(
                proc,
                "indextts",
                expected_instance_token="new-token",
                timeout_seconds=0.01,
            )


def test_start_refuses_unmanaged_process_on_target_port() -> None:
    old_process = settings._tts_process
    old_backend = settings._tts_runtime_backend
    old_token = settings._tts_instance_token
    settings._tts_process = None
    settings._tts_runtime_backend = None
    settings._tts_instance_token = None
    try:
        with patch(
            "dubbing_service.app.routers.settings.build_backend_command",
            return_value=["python", "worker.py"],
        ), patch(
            "dubbing_service.app.routers.settings._is_loopback_port_listening",
            return_value=True,
        ), patch("dubbing_service.app.routers.settings.subprocess.Popen") as popen:
            with pytest.raises(HTTPException) as captured:
                settings._start_tts_service_exclusive(
                    TTSServiceStartPayload(backend="indextts")
                )

        assert captured.value.status_code == 409
        assert "未受当前平台管理" in str(captured.value.detail)
        popen.assert_not_called()
    finally:
        settings._tts_process = old_process
        settings._tts_runtime_backend = old_backend
        settings._tts_instance_token = old_token


def test_internal_worker_switch_does_not_persist_default_backend() -> None:
    old_process = settings._tts_process
    old_backend = settings._tts_runtime_backend
    old_token = settings._tts_instance_token
    settings._tts_process = None
    settings._tts_runtime_backend = None
    settings._tts_instance_token = None
    proc = Mock()
    proc.poll.return_value = None
    proc.pid = 24680
    try:
        with patch(
            "dubbing_service.app.routers.settings._get_configured_tts_backend",
            return_value="indextts",
        ), patch(
            "dubbing_service.app.routers.settings.build_backend_command",
            return_value=["python", "worker.py"],
        ), patch(
            "dubbing_service.app.routers.settings._is_loopback_port_listening",
            return_value=False,
        ), patch(
            "dubbing_service.app.routers.settings._update_raw_file_config"
        ) as update_config, patch(
            "dubbing_service.app.routers.settings.clear_config_caches"
        ) as clear_caches, patch(
            "dubbing_service.app.routers.settings.subprocess.Popen",
            return_value=proc,
        ), patch(
            "dubbing_service.app.routers.settings.threading.Thread"
        ), patch(
            "dubbing_service.app.routers.settings._wait_for_tts_ready",
            return_value={"status": "healthy"},
        ):
            response = settings._start_tts_service_exclusive(
                TTSServiceStartPayload(
                    backend="qwen3tts",
                    persist_default=False,
                )
            )

        assert response.runtime_backend == "qwen3tts"
        assert response.config_backend == "indextts"
        update_config.assert_not_called()
        clear_caches.assert_not_called()
    finally:
        settings._tts_process = old_process
        settings._tts_runtime_backend = old_backend
        settings._tts_instance_token = old_token
