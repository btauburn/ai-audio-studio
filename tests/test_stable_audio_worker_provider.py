from __future__ import annotations

import json

import httpx
import pytest

from dubbing_service.app.providers.audio_generation import (
    AudioGenerationRequest,
    StableAudioSFXProvider,
)


def test_provider_posts_generation_request_and_returns_wav_bytes() -> None:
    wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/sfx/generate"
        assert request.headers["x-dubbing-worker-token"] == "secret"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload == {
            "prompt": "A sharp metal impact",
            "duration_seconds": 1.5,
            "negative_prompt": None,
            "seed": 77,
            "steps": 8,
            "cfg_scale": 1.0,
            "chunked_decode": True,
        }
        return httpx.Response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "X-Audio-Sample-Rate": "44100",
                "X-Audio-Duration-Seconds": "1.5",
                "X-Audio-Seed": "77",
                "X-Audio-Model-Id": "stable-audio-3-small-sfx",
            },
        )

    provider = StableAudioSFXProvider(
        "http://127.0.0.1:8005",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    result = provider.generate(
        AudioGenerationRequest(
            prompt="A sharp metal impact",
            duration_seconds=1.5,
            seed=77,
        )
    )

    assert result.success is True
    assert result.audio_bytes == wav
    assert result.sample_rate == 44100
    assert result.duration_seconds == 1.5
    assert result.seed == 77
    assert result.model_id == "stable-audio-3-small-sfx"


def test_provider_returns_bounded_error_and_rejects_non_wav_success() -> None:
    def failing_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "model is not provisioned"})

    provider = StableAudioSFXProvider(
        "http://127.0.0.1:8005",
        transport=httpx.MockTransport(failing_handler),
    )
    request = AudioGenerationRequest(prompt="rain", duration_seconds=1)
    failed = provider.generate(request)
    assert failed.success is False
    assert failed.status_code == 503
    assert failed.error == "model is not provisioned"
    assert failed.audio_bytes is None

    def json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"path": "C:/worker/output.wav"})

    provider = StableAudioSFXProvider(
        "http://127.0.0.1:8005",
        transport=httpx.MockTransport(json_handler),
    )
    invalid = provider.generate(request)
    assert invalid.success is False
    assert invalid.audio_bytes is None
    assert "non-WAV" in (invalid.error or "")


def test_provider_accepts_ambience_duration_and_enforces_worker_bounds() -> None:
    request = AudioGenerationRequest(
        prompt="A quiet forest at night",
        duration_seconds=60,
        seed=2_147_483_647,
    )
    assert request.duration_seconds == 60
    assert request.seed == 2_147_483_647

    with pytest.raises(ValueError, match="duration_seconds"):
        AudioGenerationRequest(prompt="rain", duration_seconds=120.01)
    with pytest.raises(ValueError, match="seed"):
        AudioGenerationRequest(
            prompt="rain",
            duration_seconds=1,
            seed=2_147_483_648,
        )
