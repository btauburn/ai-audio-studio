from __future__ import annotations

import builtins

import pytest

from service import qwen3tts_customvoice_server as worker


def _without_flash_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def importing(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "flash_attn":
            raise ImportError("synthetic missing flash_attn")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)


def test_auto_preserves_historical_eager_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_flash_attention(monkeypatch)

    assert worker._resolve_attention("auto") == "eager"


def test_explicit_sdpa_is_selected_without_flash_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_flash_attention(monkeypatch)

    assert worker._resolve_attention("sdpa") == "sdpa"


def test_explicit_flash_attention_fails_instead_of_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_flash_attention(monkeypatch)

    with pytest.raises(RuntimeError, match="no CPU or eager fallback"):
        worker._resolve_attention("flash_attention_2")


def test_execution_profile_id_changes_with_sound_affecting_runtime() -> None:
    eager = {"model_id": "same", "attention_resolved": "eager"}
    sdpa = {"model_id": "same", "attention_resolved": "sdpa"}

    assert worker._profile_identity(eager) != worker._profile_identity(sdpa)


def test_request_contract_carries_request_and_utterance_identity() -> None:
    request = worker.TTSRequest(
        text="A private fixture sentence.",
        speaker="ryan",
        request_id="req-1",
        utterance_id="utt-1",
    )

    assert request.request_id == "req-1"
    assert request.utterance_id == "utt-1"
