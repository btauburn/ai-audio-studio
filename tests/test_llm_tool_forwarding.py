"""Typed, bounded, non-streaming LLM protocol tests with offline HTTP transport."""

import json
import threading
import time

import httpx
import pytest

from dubbing_service.app.audiobook.public_discovery_tools import tool_schemas
from dubbing_service.app.providers import llm


def provider(handler):
    return llm.OpenAIStyleLLMProvider(
        model="deepseek/offline-fixture-not-a-live-model",
        base_url="https://openrouter.ai/api/v1",
        api_key="OFFLINE-NOT-A-CREDENTIAL",
        max_attempts=1,
        transport=httpx.MockTransport(handler),
    )


def response(request, calls=None, **kwargs):
    data = {
        "choices": [
            {
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"content": None, "tool_calls": calls or []},
            }
        ],
        "usage": {"total_tokens": 123},
    }
    return httpx.Response(
        200,
        stream=httpx.ByteStream(json.dumps(data).encode()),
        request=request,
        headers={"x-request-id": "offline-request"},
        **kwargs,
    )


def rawcall(
    name="search_public_metadata", args='{"query":"Douglas Adams"}', id="call1"
):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def test_forwarding_preserves_tool_roles_and_exact_model(monkeypatch):
    monkeypatch.setattr(
        llm, "get_config", lambda: pytest.fail("Unrelated credentials/config read")
    )
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return response(request, [rawcall()])

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [rawcall(id="prior")]},
        {"role": "tool", "tool_call_id": "prior", "content": "public metadata"},
    ]
    result = provider(handle).chat(
        messages,
        tools=tool_schemas(),
        parallel_tool_calls=True,
        max_response_bytes=65536,
        request_timeout_seconds=15,
    )
    assert result.success and result.tool_calls == [
        llm.LLMToolCall("call1", "search_public_metadata", {"query": "Douglas Adams"})
    ]
    assert result.request_id == "offline-request" and result.usage == {
        "total_tokens": 123
    }
    assert seen[0]["messages"] == messages and seen[0]["stream"] is False
    assert seen[0]["model"] == "deepseek/offline-fixture-not-a-live-model"
    assert seen[0]["parallel_tool_calls"] is False


@pytest.mark.parametrize(
    "calls",
    [
        [rawcall(name="write_file")],
        [rawcall(args="[]")],
        [rawcall(args="{bad")],
        [rawcall(id="bad id")],
        [rawcall(), rawcall()],
        [rawcall(args='{"q":"' + "x" * 8100 + '"}')],
        {"wrong": "shape"},
        [{"type": "function", "function": []}],
    ],
)
def test_invalid_response_rejected_without_retry(calls):
    requests = []

    def handle(request):
        requests.append(request)
        return response(request, calls)

    result = provider(handle).chat([], tools=tool_schemas(), max_response_bytes=65536)
    assert not result.success and not result.retryable and len(requests) == 1


def test_unrequested_tool_and_body_cap_fail_closed():
    result = provider(lambda req: response(req, [rawcall()])).chat([])
    assert not result.success
    result = provider(
        lambda req: httpx.Response(
            200, request=req, stream=httpx.ByteStream(b"x" * 1025)
        )
    ).chat([], max_response_bytes=1024)
    assert not result.success and not result.retryable


def test_compressed_response_cannot_bypass_raw_cap():
    p = provider(
        lambda req: httpx.Response(
            200,
            request=req,
            stream=httpx.ByteStream(b"compressed"),
            headers={"content-encoding": "gzip"},
        )
    )
    assert not p.chat([], max_response_bytes=1024).success


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_response_bytes": True},
        {"max_response_bytes": 262145},
        {"max_response_bytes": 1024, "request_timeout_seconds": 16},
        {"max_response_bytes": 1024, "request_timeout_seconds": float("nan")},
        {"tools": [{"type": "bad"}]},
        {"tools": tool_schemas(), "tool_choice": "invented"},
    ],
)
def test_invalid_settings_reject_before_http(kwargs):
    with pytest.raises(ValueError):
        provider(lambda req: pytest.fail("No HTTP request allowed")).chat([], **kwargs)


def test_timeout_single_attempt_and_header_deadline(monkeypatch):
    seen = []

    def timeout(request):
        seen.append(request)
        raise httpx.ReadTimeout("offline", request=request)

    result = provider(timeout).chat([], max_response_bytes=1024, max_attempts=1)
    assert not result.success and result.retryable and len(seen) == 1
    ticks = iter([0, 16])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(ticks))
    result = provider(lambda req: response(req)).chat(
        [], max_response_bytes=1024, request_timeout_seconds=15
    )
    assert not result.success


def test_total_deadline_returns_while_transport_is_still_waiting():
    release = threading.Event()
    ended = threading.Event()
    seen = []

    def handler(request):
        seen.append(request)
        release.wait(2)
        ended.set()
        return response(request)

    try:
        start = time.perf_counter()
        result = provider(handler).chat(
            [], max_response_bytes=1024, request_timeout_seconds=0.08, max_attempts=1
        )
        assert time.perf_counter() - start < 0.8
        assert (
            not result.success
            and result.retryable
            and len(seen) == 1
            and not ended.is_set()
        )
        assert seen[0].headers["accept-encoding"] == "identity"
    finally:
        release.set()
        assert ended.wait(2)
