"""Contract tests for generated references and reusable, non-pickle Qwen prompts."""
from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from service import qwen_character_voice as voice


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value


@dataclass
class Prompt:
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(voice, "safe_project_root", lambda _: project)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=Tensor, cuda=SimpleNamespace(is_available=lambda: False), manual_seed=lambda _: None,
        random=SimpleNamespace(fork_rng=lambda **_: contextlib.nullcontext()),
    ))
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model", SimpleNamespace(VoiceClonePromptItem=Prompt))
    calls = []

    class Model:
        def generate_voice_design(self, **kwargs):
            calls.append(("design", kwargs))
            return [np.sin(np.arange(2400) / 20).astype(np.float32) * .1], 24000

        def create_voice_clone_prompt(self, **kwargs):
            calls.append(("prepare", kwargs))
            return [Prompt(Tensor(np.ones((12, 16), dtype=np.int64)), Tensor(np.arange(128, dtype=np.float32)), False, True, kwargs["ref_text"])]

        def generate_voice_clone(self, **kwargs):
            calls.append(("clone", kwargs))
            return [np.ones(2400, dtype=np.float32) * .01], 24000

    rt = voice.CharacterVoiceRuntime(tmp_path, lambda kind: Model())
    monkeypatch.setattr(rt, "_model_metadata", lambda kind: {"model": "fixture-" + kind, "model_revision": "revision-1", "runtime_version": "0.1.1"})
    return rt, project, calls


def create(rt):
    rt.design("p", "voice1", "Clear mid-register voice; careful articulation.", "The gate is open.", 4)
    return rt.prepare("p", "voice1")


def render(rt, meta, **overrides):
    args = {key: meta[key] for key in ("prompt_artifact", "prompt_sha256", "reference_audio", "reference_audio_sha256")}
    args.update(overrides)
    return rt.synthesize("p", "voice1", text="We should go.", language="English", **args)


def test_prompt_roundtrip_preserves_original_identity_after_reload(runtime):
    rt, project, calls = runtime
    meta = create(rt)
    item, persisted = voice.load_prompt(project / meta["prompt_artifact"], meta["prompt_sha256"], identity_id="voice1", reference_hash=meta["reference_audio_sha256"])
    assert item.ref_text == "The gate is open."
    assert item.ref_code.value.shape == (12, 16)
    assert persisted["model_revision"] == "revision-1"
    result = render(rt, meta)
    assert Path(result["audio_file"]).is_file()
    assert result["identity_id"] == "voice1"
    assert calls[-1][0] == "clone"
    assert calls[-1][1]["voice_clone_prompt"][0].ref_text == item.ref_text


def test_idempotent_design_and_prepare_do_not_regenerate(runtime):
    rt, _, calls = runtime
    first = create(rt)
    second = create(rt)
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert [name for name, _ in calls] == ["design", "prepare"]
    with pytest.raises(ValueError, match="different inputs"):
        rt.design("p", "voice1", "Different voice.", "The gate is open.", 4)


def test_recover_completed_reference_when_receipt_publication_was_interrupted(runtime, monkeypatch):
    rt, project, calls = runtime
    publish = voice._publish_json
    def crash_receipt(path, payload):
        if path.name == "reference.json":
            raise OSError("synthetic process interruption")
        publish(path, payload)
    monkeypatch.setattr(voice, "_publish_json", crash_receipt)
    with pytest.raises(OSError):
        rt.design("p", "voice1", "precise", "The gate is open.", 4)
    assert (project / "casting/voices/voice1/reference.wav").is_file()
    monkeypatch.setattr(voice, "_publish_json", publish)
    recovered = rt.design("p", "voice1", "precise", "The gate is open.", 4)
    assert recovered["reference_audio_sha256"]
    assert [name for name, _ in calls] == ["design"]


def test_interrupted_prompt_write_does_not_publish_partial_prompt(runtime, monkeypatch):
    rt, project, _ = runtime
    rt.design("p", "voice1", "precise", "The gate is open.", 4)
    writer = np.savez_compressed
    def crash(stream, **kwargs):
        stream.write(b"partial")
        raise OSError("synthetic interruption")
    monkeypatch.setattr(np, "savez_compressed", crash)
    with pytest.raises(OSError):
        rt.prepare("p", "voice1")
    assert not (project / "casting/voices/voice1/prompt.npz").exists()
    monkeypatch.setattr(np, "savez_compressed", writer)
    assert rt.prepare("p", "voice1")["prompt_sha256"]


def test_scene_emotion_does_not_replace_prompt_or_call_voice_design(runtime):
    rt, _, calls = runtime
    meta = create(rt)
    result = render(rt, meta, performance={"emotion": "angry", "pace": 1.1})
    assert result["prompt_sha256"] == meta["prompt_sha256"]
    assert result["unsupported_performance_fields"] == ["emotion"]
    assert [name for name, _ in calls] == ["design", "prepare", "clone"]
    assert "instruct" not in calls[-1][1]
    assert calls[-1][1]["do_sample"] is True
    assert calls[-1][1]["non_streaming_mode"] is True
    assert result["inference_policy_revision"] == "qwen-base-sampled-v1"
    repeated = render(rt, meta, performance={"emotion": "quiet"})
    assert repeated["generation_settings"]["seed"] == result["generation_settings"]["seed"]


def test_overlong_generated_audio_is_not_accepted_as_an_audition(runtime):
    rt, project, _ = runtime
    meta = create(rt)
    class RepeatingModel:
        def generate_voice_clone(self, **kwargs):
            return [np.ones(24000 * 20, dtype=np.float32)], 24000
    rt.load_model = lambda kind: RepeatingModel()
    with pytest.raises(ValueError, match="overlong"):
        render(rt, meta)
    assert voice.sha256(project / meta["prompt_artifact"]) == meta["prompt_sha256"]


def test_changed_model_or_runtime_requires_explicit_migration(runtime, monkeypatch):
    rt, _, _ = runtime
    meta = create(rt)
    monkeypatch.setattr(rt, "_model_metadata", lambda kind: {"model": "new", "model_revision": "revision-2", "runtime_version": "0.1.1"})
    with pytest.raises(ValueError, match="migration"):
        render(rt, meta)


def test_missing_or_corrupt_prompt_fails_without_new_voice(runtime):
    rt, project, calls = runtime
    meta = create(rt)
    (project / meta["prompt_artifact"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        render(rt, meta)
    assert [name for name, _ in calls] == ["design", "prepare"]


def test_reference_integrity_and_identity_ownership_are_required(runtime):
    rt, project, _ = runtime
    meta = create(rt)
    with pytest.raises(ValueError, match="does not belong"):
        render(rt, meta, prompt_artifact="casting/voices/other/prompt.npz")
    (project / meta["reference_audio"]).write_bytes(b"different audio")
    with pytest.raises(ValueError, match="checksum"):
        render(rt, meta)


@pytest.mark.parametrize("relative", ["../outside.wav", "C:/outside.wav", "/outside.wav"])
def test_reference_path_escape_rejected(runtime, relative):
    _, project, _ = runtime
    with pytest.raises(ValueError):
        voice.scoped_file(project, relative)


def test_no_external_actor_audio_accepted_by_prepare(runtime):
    rt, project, calls = runtime
    folder = project / "casting/voices/voice1"
    folder.mkdir(parents=True)
    (folder / "reference.json").write_text(json.dumps({"identity_id": "voice1", "provenance": {"origin": "actor_recording"}}))
    with pytest.raises(ValueError, match="only generated original"):
        rt.prepare("p", "voice1")
    assert calls == []


def test_provider_uses_exact_locked_identity_and_reports_failure_without_fallback(monkeypatch):
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers.tts import Qwen3TTSProvider
    identity = CharacterVoiceIdentity(id="voice1", role_id="r", mechanism="qwen_reference", dossier_revision="d",
                                      casting_profile_revision="c", design_instruction="precise", model="base",
                                      reference_audio="casting/voices/voice1/reference.wav", reference_audio_sha256="a" * 64,
                                      prompt_artifact="casting/voices/voice1/prompt.npz", prompt_sha256="b" * 64)
    role = Role(id="r", name="R", voice_identity=identity)
    segment = Segment(id="s", role_id="r", text="No!", index=0, emotion_text="angry")
    provider = Qwen3TTSProvider()
    seen = []
    def call(operation, payload):
        seen.append((operation, payload))
        raise RuntimeError("missing prompt")
    monkeypatch.setattr(provider, "_character_request", call)
    result = provider.synthesize_segment("p", role, segment)
    assert not result.success
    assert seen[0][0] == "synthesize"
    assert seen[0][1]["prompt_sha256"] == "b" * 64
    assert seen[0][1]["performance"]["emotion"] == "angry"
    assert len(seen) == 1


@pytest.mark.parametrize("mechanism,backend", [("builtin", "qwen3tts_cv"), ("existing_reference", "indextts"), ("qwen_reference", "qwen3tts_cv")])
def test_voicedesign_provider_rejects_incompatible_locked_identity(mechanism, backend):
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers.tts import Qwen3TTSProvider
    identity = CharacterVoiceIdentity(id="v", role_id="r", mechanism=mechanism, backend=backend,
                                     dossier_revision="d", casting_profile_revision="c", design_instruction="", model="m")
    result = Qwen3TTSProvider().synthesize_segment("p", Role(id="r", name="R", voice_identity=identity),
                                                 Segment(id="s", role_id="r", index=0, text="Hello."))
    assert not result.success
    assert "incompatible" in result.raw["error"]


def test_customvoice_locked_builtin_controls_speaker_and_rejects_override(monkeypatch):
    import io

    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers import tts
    from dubbing_service.app.providers.tts import Qwen3TTSCustomVoiceProvider
    identity = CharacterVoiceIdentity(id="v", role_id="r", mechanism="builtin", backend="qwen3tts_cv", builtin_speaker="ryan",
                                     dossier_revision="d", casting_profile_revision="c", design_instruction="", model="m")
    role = Role(id="r", name="R", qwen3_speaker="vivian", voice_identity=identity)
    segment = Segment(id="s", role_id="r", index=0, text="Hello.")
    payloads = []
    def request(req, **kwargs):
        payloads.append(json.loads(req.data))
        return io.BytesIO(b'{"success":true,"audio_file":"outputs/test.wav"}')
    monkeypatch.setattr(tts.urllib.request, "urlopen", request)
    provider = Qwen3TTSCustomVoiceProvider()
    assert provider.synthesize_segment("p", role, segment).success
    assert payloads[0]["speaker"] == "ryan"
    assert role.qwen3_speaker == "vivian"
    segment.qwen3_speaker = "aiden"
    result = provider.synthesize_segment("p", role, segment)
    assert not result.success
    assert "conflicts" in result.raw["error"]
    assert len(payloads) == 1


def test_cached_prompt_rehashes_same_size_tampering_on_hit(runtime):
    rt, project, calls = runtime
    meta = create(rt)  # prepare warms the CPU cache
    render(rt, meta)
    prompt = project / meta["prompt_artifact"]
    data = bytearray(prompt.read_bytes())
    data[len(data) // 2] ^= 1
    prompt.write_bytes(data)
    clones = sum(name == "clone" for name, _ in calls)
    with pytest.raises(ValueError, match="checksum"):
        render(rt, meta)
    assert sum(name == "clone" for name, _ in calls) == clones


def test_cached_prompt_keeps_private_tensor_and_metadata_copies(runtime):
    rt, project, _ = runtime
    meta = create(rt)
    path = project / meta["prompt_artifact"]
    kwargs = dict(identity_id="voice1", reference_hash=meta["reference_audio_sha256"])
    first, metadata = rt._cached_prompt(path, meta["prompt_sha256"], **kwargs)
    original = first.ref_code.value.copy()
    first.ref_code.value[:] = 99
    first.ref_spk_embedding.value[:] = 99
    metadata["identity_id"] = "other"
    second, next_metadata = rt._cached_prompt(path, meta["prompt_sha256"], **kwargs)
    assert np.array_equal(second.ref_code.value, original)
    assert not np.all(second.ref_spk_embedding.value == 99)
    assert next_metadata["identity_id"] == "voice1"
    with pytest.raises(ValueError, match="provenance"):
        rt._cached_prompt(path, meta["prompt_sha256"], identity_id="other",
                          reference_hash=meta["reference_audio_sha256"])


def test_cache_avoids_decoding_each_chunk_and_reports_stage_timings(runtime, monkeypatch):
    rt, _, _ = runtime
    loads = []
    original = voice.load_prompt
    def counted(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(voice, "load_prompt", counted)
    meta = create(rt)
    first, second = render(rt, meta), render(rt, meta)
    assert len(loads) == 1
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert first["generation_settings"] == second["generation_settings"]
    stages = second["stage_timings"]
    phases = [stages[key] for key in ("conditioning_seconds", "model_load_seconds",
                                     "model_generate_seconds", "audio_write_seconds")]
    assert all(phase >= 0 for phase in phases)
    assert sum(phases) <= stages["synthesis_seconds"] + .00001


def test_prompt_cache_lru_bound_and_missing_file_fail_closed(runtime):
    rt, project, _ = runtime
    for index in range(17):
        rt.design("p", f"v{index}", "clear", "The gate is open.", index)
        rt.prepare("p", f"v{index}")
    assert len(rt._prompt_cache) == 16
    assert not any(key[2] == "v0" for key in rt._prompt_cache)
    assert 0 < rt._prompt_cache_bytes <= 32 * 1024 * 1024
    meta = create(rt)
    path = project / meta["prompt_artifact"]
    # A cached item never authorizes a missing artifact. Renaming only this
    # disposable fixture is needed to reproduce deletion after cache warmup.
    path.rename(path.with_suffix(".missing"))
    with pytest.raises(ValueError, match="missing"):
        render(rt, meta)


def test_warm_cache_rejects_runtime_only_migration(runtime, monkeypatch):
    rt, _, calls = runtime
    meta = create(rt)
    render(rt, meta)
    count = len(calls)
    monkeypatch.setattr(rt, "_model_metadata", lambda kind: {
        "model": "fixture-" + kind, "model_revision": "revision-1", "runtime_version": "0.2.0"})
    with pytest.raises(ValueError, match="migration"):
        render(rt, meta)
    assert len(calls) == count


def test_prompt_cache_bounds_decompressed_bytes_not_compressed_file_size(runtime, monkeypatch):
    rt, project, _ = runtime
    meta = create(rt)
    prompt = project / meta["prompt_artifact"]
    uncached = prompt.with_name("large.npz")
    uncached.write_bytes(prompt.read_bytes())
    original = voice.load_prompt
    loads = []
    def large_codes(*args, **kwargs):
        item, metadata = original(*args, **kwargs)
        # Simulate a heavily compressed numeric archive with real resident CPU
        # array bytes, just above the bound. No model or GPU allocation occurs.
        item.ref_code = Tensor(np.zeros(33 * 1024 * 1024 // 8, dtype=np.int64))
        loads.append(1)
        return item, metadata
    monkeypatch.setattr(voice, "load_prompt", large_codes)
    for _ in range(2):
        item, _ = rt._cached_prompt(uncached, meta["prompt_sha256"], identity_id="voice1",
                                    reference_hash=meta["reference_audio_sha256"])
        assert item.ref_code.value.nbytes > 32 * 1024 * 1024
    assert len(loads) == 2  # Over-budget entries are usable but never retained.
    assert not any(key[0] == uncached.resolve() for key in rt._prompt_cache)
    assert rt._prompt_cache_bytes <= 32 * 1024 * 1024
