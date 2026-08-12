"""Audio-generation provider contracts and Stable Audio SFX HTTP client."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
_WAV_CONTENT_TYPES = frozenset(
    {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
)


@dataclass(frozen=True)
class AudioGenerationRequest:
    """Normalized request shared by local and remote SFX providers."""

    prompt: str
    duration_seconds: float
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int = 8
    cfg_scale: float = 1.0
    chunked_decode: bool = True

    def __post_init__(self) -> None:
        prompt = str(self.prompt).strip()
        if not prompt or len(prompt) > 1000:
            raise ValueError("prompt must contain between 1 and 1000 characters")
        if not 0.25 <= float(self.duration_seconds) <= 120.0:
            raise ValueError("duration_seconds must be between 0.25 and 120.0")
        if self.negative_prompt is not None and len(str(self.negative_prompt)) > 1000:
            raise ValueError("negative_prompt must not exceed 1000 characters")
        if self.seed is not None and not 0 <= int(self.seed) <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2147483647")
        if not 1 <= int(self.steps) <= 100:
            raise ValueError("steps must be between 1 and 100")
        if not 0.0 <= float(self.cfg_scale) <= 20.0:
            raise ValueError("cfg_scale must be between 0.0 and 20.0")
        object.__setattr__(self, "prompt", prompt)
        if self.negative_prompt is not None:
            negative_prompt = str(self.negative_prompt).strip()
            object.__setattr__(self, "negative_prompt", negative_prompt or None)


@dataclass(frozen=True)
class AudioGenerationResult:
    """Provider result that carries audio bytes, never a worker filesystem path."""

    success: bool
    audio_bytes: bytes | None = None
    content_type: str | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None
    seed: int | None = None
    model_id: str | None = None
    status_code: int | None = None
    error: str | None = None


class AudioGenerationProvider(ABC):
    """Provider interface for text-conditioned sound-effect generation."""

    @abstractmethod
    def generate(self, request: AudioGenerationRequest) -> AudioGenerationResult:
        """Generate one sound effect and return an in-memory encoded audio file."""


def _validated_base_url(value: str) -> str:
    text = str(value).strip()
    if not text or any(ord(char) < 0x20 or ord(char) == 0x7F for char in text):
        raise ValueError(
            "audio-generation base URL is empty or contains control characters"
        )
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("audio-generation base URL is invalid") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("audio-generation base URL must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("audio-generation base URL must not contain credentials")
    return text.rstrip("/")


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _header_float(headers: httpx.Headers, name: str) -> float | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_error(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:2000] or f"HTTP {response.status_code}"
    if isinstance(payload, dict) and "detail" in payload:
        detail = payload["detail"]
        return (
            detail
            if isinstance(detail, str)
            else json.dumps(detail, ensure_ascii=False)
        )
    return json.dumps(payload, ensure_ascii=False)[:2000]


class StableAudioSFXProvider(AudioGenerationProvider):
    """Client for the Stable Audio SFX worker's byte-oriented REST contract."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: httpx.Timeout | float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = _validated_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout or _DEFAULT_TIMEOUT
        self.transport = transport

    def generate(self, request: AudioGenerationRequest) -> AudioGenerationResult:
        payload = {
            "prompt": request.prompt,
            "duration_seconds": request.duration_seconds,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "chunked_decode": request.chunked_decode,
        }
        headers = {"Accept": "audio/wav"}
        if self.api_key:
            # This is a per-process instance token, not a reusable web API
            # credential. Keep it out of URLs and conventional proxy auth logs.
            headers["X-Dubbing-Worker-Token"] = self.api_key

        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/sfx/generate",
                    json=payload,
                    headers=headers,
                )
        except httpx.RequestError as exc:
            return AudioGenerationResult(success=False, error=str(exc))

        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if not 200 <= response.status_code < 300:
            return AudioGenerationResult(
                success=False,
                status_code=response.status_code,
                error=_safe_error(response),
            )
        if (
            content_type not in _WAV_CONTENT_TYPES
            or len(response.content) < 12
            or response.content[:4] != b"RIFF"
            or response.content[8:12] != b"WAVE"
        ):
            return AudioGenerationResult(
                success=False,
                status_code=response.status_code,
                error="audio-generation worker returned a non-WAV response",
            )

        return AudioGenerationResult(
            success=True,
            audio_bytes=response.content,
            content_type=content_type,
            sample_rate=_header_int(response.headers, "x-audio-sample-rate"),
            duration_seconds=_header_float(
                response.headers, "x-audio-duration-seconds"
            ),
            seed=_header_int(response.headers, "x-audio-seed"),
            model_id=response.headers.get("x-audio-model-id"),
            status_code=response.status_code,
        )
