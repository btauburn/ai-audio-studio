"""LLM provider abstraction and hardened OpenAI-compatible implementation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional

import httpx

from dubbing_service.app.core.config import get_config


_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_RETRY_DELAY_SECONDS = 30.0


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


class LLMProvider:
    """LLM Provider abstract base class."""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResult:  # pragma: no cover - interface
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

        self.base_url = str(effective_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else (cfg.llm_api_key if cfg is not None else None)

        effective_model = model or "gpt-4o-mini"
        # Keep backward compatibility for callers that relied on provider-based
        # model inference, while selecting the current DeepSeek production model.
        if effective_model == "gpt-4o-mini" and "deepseek.com" in self.base_url.lower():
            effective_model = "deepseek-v4-flash"

        self.model = effective_model
        self.default_temperature = default_temperature
        self.timeout = timeout or _DEFAULT_TIMEOUT
        self.max_attempts = max(1, min(int(max_attempts), 5))
        self.transport = transport
        self._sleep = sleep

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResult:
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
                    response = client.post(url, json=body, headers=headers)
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
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                message = choice.get("message") if isinstance(choice, dict) else {}
                message = message if isinstance(message, dict) else {}
                text = message.get("content")
                if text is not None and not isinstance(text, str):
                    text = str(text)
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
                )

        return last_result or LLMResult(success=False, raw={"error": "LLM request did not run"})
