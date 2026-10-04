from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing_service.app.core.qwen_graph_policy import (
    GraphGenerationTruncated,
    classify_graph_termination,
)


def test_graph_termination_accepts_only_early_eos_exit() -> None:
    assert classify_graph_termination(
        generated_steps=50,
        max_new_tokens=8192,
        prefill_tokens=120,
        graph_max_seq_len=2048,
    ) == "eos"


@pytest.mark.parametrize(
    ("steps", "max_new", "prefill", "capacity", "reason"),
    [
        (20, 20, 10, 2048, "max_new_tokens"),
        (1998, 8192, 50, 2048, "fixed CUDA graph buffer capacity"),
        (1, 8192, 2048, 2048, "prefill"),
    ],
)
def test_graph_termination_rejects_each_unconfirmed_limit(
    steps: int, max_new: int, prefill: int, capacity: int, reason: str
) -> None:
    with pytest.raises(GraphGenerationTruncated, match=reason):
        classify_graph_termination(
            generated_steps=steps,
            max_new_tokens=max_new,
            prefill_tokens=prefill,
            graph_max_seq_len=capacity,
        )


def test_graph_termination_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="invalid token counts"):
        classify_graph_termination(
            generated_steps=-1,
            max_new_tokens=8192,
            prefill_tokens=1,
            graph_max_seq_len=2048,
        )


def test_graph_worker_cancel_sets_only_the_matching_active_request() -> None:
    import threading

    from service import faster_qwen3tts_customvoice_server as worker

    original = worker._request_events
    try:
        event = threading.Event()
        worker._request_events = {"request-1": event}
        result = worker.cancel_tts(worker.CancelRequest(request_id="request-1"))
        assert result["state"] == "cancellation_requested"
        assert event.is_set()
        missing = worker.cancel_tts(worker.CancelRequest(request_id="request-2"))
        assert missing["state"] == "not_found"
    finally:
        worker._request_events = original


def test_graph_worker_reports_completed_request_for_uncertain_client_retry() -> None:
    from service import faster_qwen3tts_customvoice_server as worker

    original = worker._request_results
    try:
        worker._request_results = {
            "request-1": {"state": "completed", "response": {"audio_file": "take.wav"}}
        }
        result = worker.tts_request_status(worker.CancelRequest(request_id="request-1"))
        assert result["state"] == "completed"
        assert result["response"]["audio_file"] == "take.wav"
        assert worker.tts_request_status(
            worker.CancelRequest(request_id="missing")
        )["state"] == "not_found"
    finally:
        worker._request_results = original


def test_graph_worker_exposes_one_compatible_synthesize_route() -> None:
    from service import faster_qwen3tts_customvoice_server as worker

    routes = [
        route
        for route in worker.app.routes
        if getattr(route, "path", None) == "/api/v1/tts/synthesize"
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    assert len(routes) == 1
    assert routes[0].endpoint is worker.synthesize_tts


def test_graph_worker_entrypoint_imports_from_outside_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    server = root / "service" / "faster_qwen3tts_customvoice_server.py"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(server), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Qwen3-TTS CustomVoice API Server" in result.stdout
