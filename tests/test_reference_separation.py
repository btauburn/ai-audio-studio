"""Processed references remain hash bound; disposable fixtures never reach TTS."""
import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf
from dubbing_service.app.audiobook import performer_discovery as discovery
from dubbing_service.app.audiobook import reference_separation as separation
from service import qwen_character_voice as voice


@pytest.fixture
def acquired(tmp_path, monkeypatch):
    root = tmp_path / "project"
    directory = root / "voice_lab/discovery"
    wave = (np.sin(np.arange(16000 * 5) / 31) * .1).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, wave, 16000, format="WAV", subtype="FLOAT")
    raw = output.getvalue()
    source = {"source_id": "synthetic-processing-fixture",
              "source_page": "https://commons.wikimedia.org/wiki/File:Fixture.wav",
              "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.wav",
              "attribution": "Owner-created synthetic fixture",
              "license_url": "https://creativecommons.org/licenses/by/3.0/",
              "terms_url": discovery.TERMS_URL, "language": "en", "duration_seconds": 5}
    permission = {"source_id": source["source_id"], "status": "approved",
                  "use": "private_synthetic_voice", "authority": "owner_original_voice",
                  "evidence_ref": "disposable-test:ownership"}
    transcript = "Disposable test fixture."
    review = {"source_id": source["source_id"], "original_sha256": hashlib.sha256(raw).hexdigest(),
              "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
              "start_seconds": 0, "end_seconds": 4, "method": "local_tools",
              "evidence_ref": "disposable-test:processing",
              "single_speaker": True, "transcript_aligned": True,
              "overlapping_speech": False, "heavy_music": False, "unsuitable_noise": False}
    # This fixture isolates the acquisition/worker gate, not the separator math.
    recipe = {"schema": "synthetic-test-only", "output_sample_rate": 16000,
              "possible_timbral_artifacts": True}
    def process(samples, rate):
        return samples * np.float32(.75), rate, dict(recipe)
    monkeypatch.setattr(discovery, "_download_media", lambda *args, **kwargs: raw)
    monkeypatch.setattr(separation, "isolate_speech", process)
    def acquire(*, processing=True, reviewed=True):
        return discovery.acquire_reference(directory, source, intended_use_evidence=permission,
            transcript=transcript, start_seconds=0, end_seconds=4,
            audio_review=review | ({"source_processing": recipe} if reviewed else {}),
            isolate_speech=processing)
    return root, directory, raw, recipe, acquire


def test_processed_reference_replays_and_preserves_original(acquired):
    import time

    root, _directory, raw, recipe, acquire = acquired
    receipt = acquire()
    assert receipt["clone_ready"] and receipt["source_processing"] == recipe
    verified, original, reference = voice._eligible_acquisition(
        root, receipt["acquisition_id"], receipt["acquisition_sha256"])
    assert original.read_bytes() == raw and verified["reference_path"].endswith("/reference.wav")
    assert sf.info(reference).samplerate == 16000
    time.sleep(1.05)  # libsndfile FLOAT WAV PEAK timestamps change across seconds.
    assert acquire()["acquisition_sha256"] == receipt["acquisition_sha256"]


def test_processing_requires_exact_processed_review(acquired):
    root, _directory, _raw, _recipe, acquire = acquired
    receipt = acquire(reviewed=False)
    assert not receipt["clone_ready"]
    assert any("recipe/output" in h for h in receipt["technical_qa"]["holds"])
    with pytest.raises(ValueError, match="passing technical QA"):
        voice._eligible_acquisition(root, receipt["acquisition_id"], receipt["acquisition_sha256"])


@pytest.mark.parametrize("tamper", ["recipe", "review", "pcm"])
def test_owner_modified_processed_artifacts_fail_replay(acquired, tamper):
    root, directory, _raw, _recipe, acquire = acquired
    receipt = acquire()
    path = directory / "acquisitions" / receipt["acquisition_id"] / "acquisition.json"
    data = json.loads(path.read_text("utf-8"))
    if tamper == "recipe":
        data["source_processing"]["schema"] = "changed"
    elif tamper == "review":
        data["audio_review"]["source_processing"]["schema"] = "changed"
    else:
        audio = directory / data["reference_path"]
        samples, rate = sf.read(audio, dtype="float32")
        samples[100] += np.float32(.001)
        sf.write(audio, samples, rate, subtype="FLOAT")
        data["reference_audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ValueError, match="replay|claimed contiguous"):
        voice._eligible_acquisition(root, receipt["acquisition_id"],
                                     hashlib.sha256(path.read_bytes()).hexdigest())


def test_opt_out_preserves_legacy_receipt_and_pcm(acquired):
    root, _directory, _raw, _recipe, acquire = acquired
    receipt = acquire(processing=False, reviewed=False)
    assert receipt["clone_ready"] and "source_processing" not in receipt
    _, original, reference = voice._eligible_acquisition(
        root, receipt["acquisition_id"], receipt["acquisition_sha256"])
    original_samples, _ = sf.read(original, dtype="float32")
    reference_samples, _ = sf.read(reference, dtype="float32")
    assert np.array_equal(reference_samples, original_samples[:16000 * 4])
    assert acquire(processing=False, reviewed=False)["acquisition_sha256"] == receipt["acquisition_sha256"]


def test_source_edit_requires_restart_before_optional_imports(monkeypatch):
    monkeypatch.setattr(separation, "_PIPELINE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="restart"):
        separation.isolate_speech(np.ones(64000, dtype=np.float32), 16000)


def test_processing_option_requires_strict_boolean(tmp_path):
    with pytest.raises(discovery.DiscoveryHold, match="boolean"):
        discovery.acquire_reference(tmp_path, {}, intended_use_evidence={},
            start_seconds=0, end_seconds=4, isolate_speech="yes")


@pytest.mark.parametrize("malformed", [None, "pcm", "schema", "oversize"])
def test_lightweight_owner_replay_is_bounded_and_checks_provenance(tmp_path, monkeypatch, malformed):
    from pathlib import Path

    from dubbing_service.app.audiobook import reference_codec


    pipeline = tmp_path / "repo/dubbing_service/app/audiobook/reference_separation.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("Synthetic fixed test module", "utf-8")
    python = pipeline.parents[3] / "environments/faster_qwen_base032/.venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"Synthetic installed-environment fixture")
    monkeypatch.setattr(separation, "_PIPELINE_PATH", pipeline)
    wave = (np.sin(np.arange(16000 * 4) / 31) * .1).astype(np.float32)
    result = np.repeat(wave, 3)[:44100 * 4].astype(np.float32)
    def run(argv, **kwargs):
        assert argv[:3] == [str(python), str(pipeline), "--cpu-replay"]
        assert kwargs == {"timeout_seconds": 90, "max_stdout_bytes": 4096}
        source, target = Path(argv[3]), Path(argv[4])
        with np.load(source, allow_pickle=False) as inp:
            assert np.array_equal(inp["wave"], wave) and int(inp["sample_rate"]) == 16000
        if malformed == "oversize":
            target.write_bytes(b"x" * 6_000_001)
            return b""
        recipe = {"pipeline_sha256": separation._PIPELINE_SHA256,
                  "input_pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                  "output_pcm_sha256": hashlib.sha256(result.tobytes()).hexdigest()}
        if malformed == "pcm":
            recipe["output_pcm_sha256"] = "0" * 64
        contents = {"wave": result, "sample_rate": np.asarray(44100),
                    "recipe": np.frombuffer(json.dumps(recipe).encode(), dtype=np.uint8)}
        if malformed == "schema":
            contents["unexpected"] = np.ones(1)
        np.savez(target, **contents)
        return b""
    monkeypatch.setattr(reference_codec, "_run", run)
    if malformed is None:
        processed, rate, _ = separation._isolated_replay(wave, 16000)
        assert rate == 44100 and np.array_equal(processed, result)
    else:
        with pytest.raises(ValueError, match="mismatch|invalid|oversized"):
            separation._isolated_replay(wave, 16000)
    assert not list(tmp_path.glob("voice-reference-replay-*"))


def test_missing_installed_owner_environment_never_installs(tmp_path, monkeypatch):
    pipeline = tmp_path / "repo/dubbing_service/app/audiobook/reference_separation.py"
    monkeypatch.setattr(separation, "_PIPELINE_PATH", pipeline)
    with pytest.raises(ValueError, match="no installation attempted"):
        separation._isolated_replay(np.ones(64000, dtype=np.float32), 16000)


def test_lightweight_owner_serializes_requests_and_releases_after_failure(monkeypatch):
    import builtins
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("Synthetic absent Torch", name="torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    guard = threading.Lock()
    barrier = threading.Barrier(3)
    active = maximum = 0
    def replay(wave, rate):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(.05)
            return wave, rate, {}
        finally:
            with guard:
                active -= 1
    monkeypatch.setattr(separation, "_isolated_replay", replay)
    wave = np.ones(64000, dtype=np.float32)
    def call():
        barrier.wait(timeout=5)
        return separation.isolate_speech(wave, 16000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(call), pool.submit(call)]
        barrier.wait(timeout=5)
        for job in jobs:
            assert job.result(timeout=5)[1] == 16000
    assert maximum == 1 and active == 0
    def fail(*args):
        raise ValueError("Synthetic CPU child failure")
    monkeypatch.setattr(separation, "_isolated_replay", fail)
    with pytest.raises(ValueError, match="child failure"):
        separation.isolate_speech(wave, 16000)
    monkeypatch.setattr(separation, "_isolated_replay", replay)
    assert separation.isolate_speech(wave, 16000)[1] == 16000
