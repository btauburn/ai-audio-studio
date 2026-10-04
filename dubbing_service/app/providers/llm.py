"""LLM provider abstraction and hardened OpenAI-compatible implementation."""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import SplitResult, urlsplit

import httpx

from dubbing_service.app.core.config import get_config

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_RETRY_DELAY_SECONDS = 30.0
_HTTP_SCHEMES = frozenset({"http", "https"})
_DEEPSEEK_DOMAIN = "deepseek.com"


def _parse_llm_base_url(value: str) -> tuple[str, SplitResult]:
    """Validate an HTTP(S) LLM base URL without restricting its network.

    Loopback and private hosts are intentionally valid: this desktop-first
    application supports locally hosted OpenAI-compatible providers. Embedded
    credentials are rejected because API keys have a dedicated configuration
    field and userinfo makes origin checks unnecessarily ambiguous.
    """

    text = str(value).strip()
    if not text or any(ord(char) < 0x20 or ord(char) == 0x7F for char in text):
        raise ValueError("LLM Base URL 不能为空或包含控制字符")

    try:
        parsed = urlsplit(text)
        # Accessing ``port`` validates malformed and out-of-range values.
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM Base URL 格式无效") from exc

    if parsed.scheme.lower() not in _HTTP_SCHEMES or not parsed.netloc or not parsed.hostname:
        raise ValueError("LLM Base URL 必须是包含主机名的 http:// 或 https:// 地址")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM Base URL 不应包含用户名或密码，请使用独立的 API Key 配置")

    return text.rstrip("/"), parsed


def is_deepseek_base_url(value: str | None) -> bool:
    """Return whether *value* targets DeepSeek's domain or a true subdomain."""

    if value is None:
        return False
    try:
        _, parsed = _parse_llm_base_url(value)
    except (TypeError, ValueError):
        return False

    hostname = (parsed.hostname or "").lower().rstrip(".")
    return hostname == _DEEPSEEK_DOMAIN or hostname.endswith(f".{_DEEPSEEK_DOMAIN}")


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self):
        if (not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.id)
                or not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.name)
                or not isinstance(self.arguments, dict)
                or len(json.dumps(self.arguments, allow_nan=False).encode()) > 8000):
            raise ValueError("Invalid typed model tool call")


def _typed_tool_calls(message, tools):
    raw = message.get("tool_calls")
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > 4:
        raise ValueError("Invalid model tool response")
    names = {tool["function"]["name"] for tool in tools or []}
    calls = []
    for row in raw:
        if not isinstance(row, dict) or row.get("type") != "function" or not isinstance(row.get("function"), dict):
            raise ValueError("Invalid model tool response")
        function = row["function"]
        arguments = function.get("arguments")
        if function.get("name") not in names or not isinstance(arguments, str) or len(arguments.encode()) > 8000:
            raise ValueError("Unrequested or malformed model tool response")
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            raise ValueError("Invalid model tool JSON") from None
        call = LLMToolCall(row.get("id"), function["name"], arguments)
        if any(previous.id == call.id for previous in calls):
            raise ValueError("Duplicate model tool call identity")
        calls.append(call)
    return calls


def _bounded_post_read(client, url, body, headers, *, timeout, cap):
    deadline = time.monotonic() + timeout
    data = bytearray()
    with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as response:
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout("Bounded model response deadline", request=response.request)
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("Compressed bounded model response rejected")
        for chunk in response.iter_raw():
            if time.monotonic() >= deadline:
                raise httpx.ReadTimeout("Bounded model response deadline", request=response.request)
            if len(data) + len(chunk) > cap:
                raise ValueError("Model response exceeds the byte limit")
            data.extend(chunk)
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout("Bounded model response deadline", request=response.request)
        return httpx.Response(response.status_code, headers=response.headers,
                              content=bytes(data), request=response.request)


# A stalled transport must not hold its caller past the declared deadline.
# Daemon requests are bounded even when OS/provider cancellation is uncertain.
_BOUNDED_MODEL_SLOTS = threading.BoundedSemaphore(2)


def _bounded_post(client, url, body, headers, *, timeout, cap):
    if not _BOUNDED_MODEL_SLOTS.acquire(blocking=False):
        raise httpx.ReadTimeout("Bounded model requests are still outstanding")
    results = queue.Queue(maxsize=1)
    def request():
        try:
            value = _bounded_post_read(client, url, body,
                {**headers, "Accept-Encoding": "identity"}, timeout=timeout, cap=cap)
            results.put((True, value))
        except BaseException as exc:  # noqa: BLE001 -- Forward worker failure and release the bounded request slot.
            results.put((False, exc))
        finally:
            _BOUNDED_MODEL_SLOTS.release()
    threading.Thread(target=request, daemon=True, name="bounded-public-model-request").start()
    try:
        ok, value = results.get(timeout=timeout)
    except queue.Empty:
        # Closing the client releases owned sockets; the remote completion may
        # still have incurred usage. The owner must never retry it automatically.
        client.close()
        raise httpx.ReadTimeout("Bounded model request total deadline") from None
    if not ok:
        raise value
    return value


@dataclass
class LLMResult:
    """Normalized result of an LLM request without exposing credentials."""

    success: bool
    raw: Dict[str, Any]
    text: Optional[str] = None
    status_code: Optional[int] = None
    retryable: bool = False
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class LLMProvider:
    """LLM Provider abstract base class."""

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> LLMResult:  # pragma: no cover - interface
        raise NotImplementedError


def _retry_after_seconds(value: str | None, attempt: int) -> float:
    """Parse Retry-After and return a bounded delay with exponential fallback."""

    fallback = min(float(2 ** max(attempt - 1, 0)), _MAX_RETRY_DELAY_SECONDS)
    if not value:
        return fallback
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return min(max(delay, 0.0), _MAX_RETRY_DELAY_SECONDS)


def _safe_response_json(response: httpx.Response) -> Dict[str, Any]:
    try:
        value = response.json()
    except (json.JSONDecodeError, ValueError):
        text = response.text[:2000]
        return {"raw": text} if text else {}
    return value if isinstance(value, dict) else {"data": value}


class OpenAIStyleLLMProvider(LLMProvider):
    """OpenAI-compatible provider with bounded timeout and retry behavior.

    Retries are limited to transport failures, HTTP 429, and HTTP 5xx. Client,
    authentication, and validation errors are returned immediately.
    """

    def __init__(
        self,
        model: str | None = None,
        default_temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        timeout: httpx.Timeout | float | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Avoid loading the global configuration (and therefore unrelated local
        # credentials) when a caller supplied a complete provider explicitly.
        cfg = get_config() if base_url is None or api_key is None else None

        effective_base_url = base_url or (cfg.llm_base_url if cfg is not None else None)
        if not effective_base_url:
            raise RuntimeError("LLM 未配置：请设置 DUBBING_LLM_BASE_URL 或在 Provider 中提供 base_url。")

        try:
            self.base_url, _ = _parse_llm_base_url(str(effective_base_url))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self.api_key = api_key if api_key is not None else (cfg.llm_api_key if cfg is not None else None)

        effective_model = model or "gpt-4o-mini"
        # Keep backward compatibility for callers that relied on provider-based
        # model inference, while selecting the current DeepSeek production model.
        if effective_model == "gpt-4o-mini" and is_deepseek_base_url(self.base_url):
            effective_model = "deepseek-v4-flash"

        self.model = effective_model
        self.default_temperature = default_temperature
        self.timeout = timeout or _DEFAULT_TIMEOUT
        self.max_attempts = max(1, min(int(max_attempts), 5))
        self.transport = transport
        self._sleep = sleep

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        temp = kwargs.get("temperature", self.default_temperature)
        if temp is not None:
            body["temperature"] = temp

        for key in (
            "max_tokens",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "thinking",
            "reasoning_effort",
        ):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]

        tools = kwargs.get("tools")
        if tools is not None:
            if (not isinstance(tools, (list, tuple)) or not 1 <= len(tools) <= 4
                    or len(json.dumps(tools).encode()) > 20000
                    or any(not isinstance(t, dict) or t.get("type") != "function"
                           or not isinstance(t.get("function"), dict)
                           or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(t["function"].get("name", "")))
                           or not isinstance(t["function"].get("parameters"), dict) for t in tools)):
                raise ValueError("Invalid bounded model tool schema")
            if kwargs.get("tool_choice", "auto") not in {"auto", "none", "required"}:
                raise ValueError("Unsupported model tool choice")
            body.update(tools=list(tools), tool_choice=kwargs.get("tool_choice", "auto"), parallel_tool_calls=False)
        response_cap = kwargs.get("max_response_bytes")
        request_seconds = kwargs.get("request_timeout_seconds", 15.0)
        if response_cap is not None and (type(response_cap) is not int or not 1 <= response_cap <= 262144
                or isinstance(request_seconds, bool) or not isinstance(request_seconds, (int, float))
                or not math.isfinite(request_seconds) or not 0 < request_seconds <= 15):
            raise ValueError("Invalid bounded model response settings")

        requested_attempts = kwargs.get("max_attempts")
        max_attempts = self.max_attempts
        if requested_attempts is not None:
            try:
                max_attempts = max(1, min(int(requested_attempts), self.max_attempts))
            except (TypeError, ValueError):
                max_attempts = self.max_attempts

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_result: LLMResult | None = None
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = (_bounded_post(client, url, body, headers, timeout=request_seconds, cap=response_cap)
                                if response_cap is not None else client.post(url, json=body, headers=headers))
                except ValueError:
                    return LLMResult(success=False, raw={"error": "Bounded model response rejected"}, retryable=False)
                except httpx.RequestError as exc:
                    last_result = LLMResult(
                        success=False,
                        raw={"error": type(exc).__name__, "message": str(exc)},
                        retryable=True,
                    )
                    if attempt < max_attempts:
                        self._sleep(_retry_after_seconds(None, attempt))
                        continue
                    return last_result

                request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
                raw = _safe_response_json(response)
                if not 200 <= response.status_code < 300:
                    retryable = response.status_code == 429 or 500 <= response.status_code <= 599
                    raw.setdefault("status", response.status_code)
                    last_result = LLMResult(
                        success=False,
                        raw=raw,
                        status_code=response.status_code,
                        retryable=retryable,
                        request_id=request_id,
                    )
                    if retryable and attempt < max_attempts:
                        self._sleep(_retry_after_seconds(response.headers.get("retry-after"), attempt))
                        continue
                    return last_result

                choices = raw.get("choices") or []
                if not isinstance(choices, list):
                    return LLMResult(success=False, raw={"error": "Invalid model choices schema"},
                                     status_code=response.status_code, retryable=False, request_id=request_id)
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                message = choice.get("message") if isinstance(choice, dict) else {}
                message = message if isinstance(message, dict) else {}
                text = message.get("content")
                if text is not None and not isinstance(text, str):
                    text = str(text)
                try:
                    calls = _typed_tool_calls(message, tools)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    return LLMResult(success=False, raw={"error": "Invalid typed model tool response"},
                                     status_code=response.status_code, retryable=False, request_id=request_id)
                usage = raw.get("usage")
                return LLMResult(
                    success=True,
                    raw=raw,
                    text=text,
                    status_code=response.status_code,
                    retryable=False,
                    request_id=request_id,
                    finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
                    usage=usage if isinstance(usage, dict) else {},
                    tool_calls=calls,
                )

        return last_result or LLMResult(success=False, raw={"error": "LLM request did not run"})
