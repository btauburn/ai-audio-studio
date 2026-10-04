"""An adopted user reference is an immutable project asset, never a live alias."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook.character_casting import _existing_identity
from dubbing_service.app.core.config import clear_config_caches, safe_project_root
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.providers import tts
from dubbing_service.app.providers.tts import CosyVoice3TTSProvider, IndexTTSTTSProvider


@pytest.fixture
def setup_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    root = safe_project_root("p")
    root.mkdir(parents=True)
    (root / "original.wav").write_bytes(b"synthetic-reference-content")
    profile = SimpleNamespace(dossier_revision="d1", revision="c1")
    yield root, profile
    clear_config_caches()


@pytest.mark.parametrize("backend,provider_type", [("indextts", IndexTTSTTSProvider), ("cosyvoice3", CosyVoice3TTSProvider)])
def test_adopted_reference_survives_live_role_swap_and_source_edit(setup_reference, monkeypatch, backend, provider_type):
    root, profile = setup_reference
    role = Role(id="r", name="R", tts_backend=backend, speaker_audio="original.wav")
    role.voice_identity = _existing_identity("p", role, profile)
    identity = role.voice_identity
    assert identity.status == "LOCKED"
    assert identity.reference_audio.startswith(f"casting/voices/{identity.id}/")
    snapshot = root / identity.reference_audio
    assert snapshot.read_bytes() == b"synthetic-reference-content"
    # Both mutable sources change after adoption; neither may replace the voice.
    (root / "original.wav").write_bytes(b"different-source-content")
    (root / "replacement.wav").write_bytes(b"another-speaker")
    role.speaker_audio = "replacement.wav"
    role.speaker_voice_id = "different-library-entry"
    payloads = []
    def request(req, **kwargs):
        payloads.append(json.loads(req.data))
        return io.BytesIO(b'{"success":true,"audio_file":"outputs/example.wav"}')
    monkeypatch.setattr(tts.urllib.request, "urlopen", request)
    result = provider_type().synthesize_segment("p", role, Segment(id="s", role_id="r", index=0, text="Hello."))
    assert result.success
    assert Path(payloads[0]["speaker_audio"]) == snapshot
    assert snapshot.read_bytes() == b"synthetic-reference-content"


@pytest.mark.parametrize("backend,provider_type", [("indextts", IndexTTSTTSProvider), ("cosyvoice3", CosyVoice3TTSProvider)])
@pytest.mark.parametrize("damage", ["missing_hash", "missing_file", "corrupt", "backend", "mechanism", "path", "status"])
def test_invalid_locked_reference_fails_before_network(setup_reference, monkeypatch, backend, provider_type, damage):
    root, profile = setup_reference
    role = Role(id="r", name="R", tts_backend=backend, speaker_audio="original.wav")
    role.voice_identity = _existing_identity("p", role, profile)
    identity = role.voice_identity
    if damage == "missing_hash":
        identity.reference_audio_sha256 = None
    elif damage == "missing_file":
        identity.reference_audio = f"casting/voices/{identity.id}/missing.wav"
    elif damage == "corrupt":
        (root / identity.reference_audio).write_bytes(b"modified-snapshot")
    elif damage == "backend":
        identity.backend = "qwen3tts"
    elif damage == "mechanism":
        identity.mechanism = "qwen_reference"
    elif damage == "path":
        identity.reference_audio = "../escape.wav"
    elif damage == "status":
        identity.status = "REVIEW"
    calls = []
    monkeypatch.setattr(tts.urllib.request, "urlopen", lambda *a, **kw: calls.append(a))
    result = provider_type().synthesize_segment("p", role, Segment(id="s", role_id="r", index=0, text="Hello."))
    assert not result.success
    assert calls == []


@pytest.mark.parametrize("backend,provider_type", [("indextts", IndexTTSTTSProvider), ("cosyvoice3", CosyVoice3TTSProvider)])
def test_legacy_unrecorded_reference_keeps_original_payload_behavior(setup_reference, backend, provider_type):
    root, _ = setup_reference
    role = Role(id="r", name="R", tts_backend=backend, speaker_audio="original.wav")
    payload = provider_type()._build_payload("p", role, Segment(id="s", role_id="r", index=0, text="Hello."))
    assert Path(payload["speaker_audio"]) == root / "original.wav"


def test_missing_legacy_reference_becomes_failed_identity_without_new_design(setup_reference):
    _, profile = setup_reference
    identity = _existing_identity("p", Role(id="r", name="R", tts_backend="indextts", speaker_audio="missing.wav"), profile)
    assert identity.status == "FAILED"
    assert identity.mechanism == "existing_reference"
    assert "missing" in identity.provenance["error"]


def test_voice_library_reference_is_snapshotted_without_acquiring_new_audio(setup_reference, monkeypatch):
    root, profile = setup_reference
    from dubbing_service.app.core import helpers
    monkeypatch.setattr(helpers, "resolve_role_reference_audio", lambda *args: root / "original.wav")
    role = Role(id="r", name="R", tts_backend="cosyvoice3", speaker_voice_id="owned-library-voice")
    first = _existing_identity("p", role, profile)
    stat = (root / first.reference_audio).stat().st_mtime_ns
    second = _existing_identity("p", role, profile)
    assert first.id == second.id
    assert first.reference_audio_sha256 == second.reference_audio_sha256
    assert (root / first.reference_audio).stat().st_mtime_ns == stat
    assert first.provenance["original_voice_id"] == "owned-library-voice"
