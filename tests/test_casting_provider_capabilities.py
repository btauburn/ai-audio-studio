"""CPU-only provider identity/capability contracts for casting."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException

from dubbing_service.app.audiobook.provider_capabilities import (
    capability_record,
    effective_identity,
    resolve_customvoice_speaker,
)


def test_customvoice_speaker_aliases_resolve_and_unknowns_reject() -> None:
    assert resolve_customvoice_speaker("Vivian") == "vivian"
    assert resolve_customvoice_speaker("UNCLE FU") == "uncle_fu"
    assert resolve_customvoice_speaker("Ono_Anna") == "ono_anna"
    with pytest.raises(ValueError, match="unsupported CustomVoice speaker"):
        resolve_customvoice_speaker("unknown voice")


def test_capability_record_reports_exact_checkpoint_and_truthful_control_status() -> None:
    record = capability_record("qwen3tts_cv")
    assert record["schema"] == "provider-capabilities-v1"
    assert record["checkpoint"]["source_model_id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert record["checkpoint"]["model_content_sha256"]
    assert len(record["adapter"]["files"]) == 2
    assert record["runtime"]["qwen_tts_version"] == "0.1.1"
    assert len(record["preset_inventory"]) == 9
    assert record["performance"]["customvoice_0_6b_instruct"]["status"] == "unsupported"
    assert record["performance"]["style_instruction"]["status"] == "supported"
    assert record["performance"]["style_instruction"]["locally_tested"] == "not acoustically tested"

    base = capability_record("qwen3tts", {
        "mechanism": "qwen_reference", "reference_audio_sha256": "a" * 64,
    })
    assert base["mechanisms"]["reference_clone"]["checkpoint"] == "qwen3_tts_base"
    assert base["performance"]["intensity"]["status"] == "unsupported"


def test_effective_identity_ignores_display_and_random_identity_labels() -> None:
    preset = capability_record("qwen3tts_cv")
    first = {**preset, "identity": {"mechanism": "builtin", "builtin_speaker": "Vivian", "id": "random-a"}}
    second = {**preset, "identity": {"mechanism": "builtin", "builtin_speaker": "vivian", "id": "random-b", "label": "Narrator"}}
    assert effective_identity(first) == effective_identity(second)

    clone = capability_record("qwen3tts")
    one = {**clone, "identity": {"mechanism": "qwen_reference", "reference_audio_sha256": "A" * 64, "id": "a"}}
    two = {**clone, "identity": {"mechanism": "qwen_reference", "reference_audio_sha256": "a" * 64, "id": "b"}}
    assert effective_identity(one) == effective_identity(two)
    two["identity"]["reference_audio_sha256"] = "b" * 64
    assert effective_identity(one) != effective_identity(two)


@pytest.mark.parametrize("speaker", ["not-a-preset", None, ""])
def test_provider_rejects_unknown_speaker_before_http(monkeypatch: pytest.MonkeyPatch, speaker) -> None:
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers import tts
    from dubbing_service.app.providers.tts import Qwen3TTSCustomVoiceProvider

    monkeypatch.setattr(tts.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not make request"))
    role = Role(id="r", name="R", qwen3_speaker=speaker)
    segment = Segment(id="s", role_id="r", index=0, text="hello")
    result = Qwen3TTSCustomVoiceProvider().synthesize_segment("p", role, segment)
    assert not result.success
    assert result.raw["outcome"] == "invalid_voice_selection"


def test_recovered_transport_result_must_match_the_requested_voice(monkeypatch):
    from urllib.error import URLError
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers import tts
    provider = tts.Qwen3TTSCustomVoiceProvider()
    def lost(*args, **kwargs):
        raise URLError("SYNTHETIC response loss")
    monkeypatch.setattr(tts.urllib.request, "urlopen", lost)
    monkeypatch.setattr(provider, "_request_cancel", lambda *args: None)
    monkeypatch.setattr(provider, "_request_status", lambda *args: {"state": "completed", "response": {
        "success": True, "effective_speaker": "vivian", "audio_file": "SYNTHETIC.wav"}})
    result = provider.synthesize_segment("fixture", Role(id="r", name="R", qwen3_speaker="ryan"),
        Segment(id="s", role_id="r", index=0, text="Fixture."))
    assert not result.success
    assert result.raw["outcome"] == "effective_voice_mismatch"


def test_provider_records_and_checks_worker_effective_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers import tts
    from dubbing_service.app.providers.tts import Qwen3TTSCustomVoiceProvider

    seen = []

    def request(req, **kwargs):
        seen.append(json.loads(req.data))
        return io.BytesIO(b'{"success":true,"audio_file":"outputs/test.wav","effective_speaker":"Uncle Fu"}')

    monkeypatch.setattr(tts.urllib.request, "urlopen", request)
    role = Role(id="r", name="R", qwen3_speaker="Uncle Fu")
    segment = Segment(id="s", role_id="r", index=0, text="hello")
    result = Qwen3TTSCustomVoiceProvider().synthesize_segment("p", role, segment)
    assert result.success
    assert seen[0]["speaker"] == "uncle_fu"
    assert result.raw["effective_speaker"] == "uncle_fu"
    assert result.raw["provider_voice_resolution"] == {
        "requested_speaker": "uncle_fu", "effective_speaker": "uncle_fu", "verified": True,
    }

    def mismatched(req, **kwargs):
        return io.BytesIO(b'{"success":true,"audio_file":"outputs/test.wav","effective_speaker":"Vivian"}')

    monkeypatch.setattr(tts.urllib.request, "urlopen", mismatched)
    rejected = Qwen3TTSCustomVoiceProvider().synthesize_segment("p", role, segment)
    assert not rejected.success
    assert rejected.raw["outcome"] == "effective_voice_mismatch"


def _fake_torch() -> SimpleNamespace:
    cuda = SimpleNamespace(
        is_available=lambda: False,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
    )
    return SimpleNamespace(cuda=cuda)


def _fake_soundfile() -> SimpleNamespace:
    return SimpleNamespace(write=lambda *a, **k: None)


def test_server_rejects_unknown_speaker_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import service.qwen3tts_customvoice_server as server

    monkeypatch.setattr(server, "_load_qwen3tts_model_if_needed", lambda: pytest.fail("model must not load"))
    request = server.TTSRequest(text="hello", speaker="not-a-preset")
    with pytest.raises(HTTPException) as error:
        server.synthesize_tts(request, BackgroundTasks())
    assert error.value.status_code == 422


@pytest.mark.parametrize("waveform_count", [0, 2])
def test_server_rejects_wrong_waveform_cardinality(monkeypatch: pytest.MonkeyPatch, waveform_count: int) -> None:
    import service.qwen3tts_customvoice_server as server

    monkeypatch.setattr(server, "_load_qwen3tts_model_if_needed", lambda: None)
    monkeypatch.setattr(server, "qwen3tts_model", object())
    monkeypatch.setattr(server, "_generate_custom_voice_with_optional_profile",
                        lambda **kwargs: (([np.zeros(16, dtype=np.float32) for _ in range(waveform_count)], 24000), None))
    monkeypatch.setattr(server, "_execution_profile", {"execution_profile_id": "cpu-fixture"})
    monkeypatch.setattr(server, "_model_load_count", 0)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())
    request = server.TTSRequest(text="hello", speaker="Vivian")
    with pytest.raises(HTTPException) as error:
        server.synthesize_tts(request, BackgroundTasks())
    assert error.value.status_code == 500
    assert "expected exactly one waveform" in str(error.value.detail)


def test_server_response_carries_canonical_effective_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    import service.qwen3tts_customvoice_server as server

    monkeypatch.setattr(server, "_load_qwen3tts_model_if_needed", lambda: None)
    monkeypatch.setattr(server, "qwen3tts_model", object())
    monkeypatch.setattr(server, "_generate_custom_voice_with_optional_profile",
                        lambda **kwargs: (([np.zeros(16, dtype=np.float32)], 24000), None))
    monkeypatch.setattr(server, "_execution_profile", {"execution_profile_id": "cpu-fixture"})
    monkeypatch.setattr(server, "_model_load_count", 0)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "soundfile", _fake_soundfile())
    response = server.synthesize_tts(server.TTSRequest(text="hello", speaker="Uncle Fu", save_file=False), BackgroundTasks())
    assert response.effective_speaker == "uncle_fu"
    assert response.generation["effective_speaker"] == "uncle_fu"
