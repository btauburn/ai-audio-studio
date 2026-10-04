"""Real installed-codec integration, entirely disposable synthetic fixtures."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import performer_discovery as discovery
from dubbing_service.app.audiobook import reference_codec as codec
from service import qwen_character_voice as voice


@pytest.fixture
def webm(tmp_path):
    try:
        binary = codec._binary("ffmpeg")
        codec._binary("ffprobe")
    except ValueError:
        pytest.skip("Existing FFmpeg installation unavailable; no installation attempted")
    source = tmp_path / "fixture.wav"
    t = np.arange(48000 * 6, dtype=np.float32) / 48000
    samples = np.column_stack([np.sin(t * 900) * .12, np.sin(t * 1300) * .08]).astype(np.float32)
    sf.write(source, samples, 48000, subtype="FLOAT")
    target = tmp_path / "fixture.webm"
    subprocess.run([str(binary), "-v", "error", "-nostdin", "-i", str(source),
                    "-c:a", "libopus", str(target)], check=True, capture_output=True, timeout=15)
    return target


def test_webm_exact_crop_replay_and_decode_provenance(webm):
    first = codec.decode_reference(webm, .75, 4.75)
    assert first.sample_rate == 48000 and first.channels == 2
    assert first.wave.shape == (48000 * 4, 2)
    assert np.isfinite(first.wave).all()
    assert first.decoder["operation"] == "webm_opus_decode_float32"
    assert len(first.decoder["ffmpeg_sha256"]) == 64
    again = codec.decode_reference(webm, .75, 4.75, expected_decoder=first.decoder)
    assert np.array_equal(first.wave, again.wave)
    with pytest.raises(ValueError, match="provenance changed"):
        codec.decode_reference(webm, .75, 4.75,
                               expected_decoder=first.decoder | {"ffmpeg_sha256": "0" * 64})


def test_existing_float_wav_uses_identical_soundfile_samples(tmp_path):
    path = tmp_path / "original.wav"
    wave = np.column_stack([np.sin(np.arange(80000) / 37) * .1,
                            np.cos(np.arange(80000) / 47) * .1]).astype(np.float32)
    sf.write(path, wave, 16000, subtype="FLOAT")
    decoded = codec.decode_reference(path, 1, 4)
    assert decoded.decoder is None
    assert np.array_equal(decoded.wave, wave[16000:64000])
    with pytest.raises(ValueError, match="cannot be replaced"):
        codec.decode_reference(path, 1, 4, expected_decoder={"decoder": "ffmpeg"})


@pytest.mark.parametrize("start,end", [(float("nan"), 4), (0, float("inf")),
                                      (-1, 4), (True, 4), (0, 2), (0, 31)])
def test_invalid_crops_rejected_before_process(webm, start, end):
    with pytest.raises(ValueError, match="finite"):
        codec.decode_reference(webm, start, end)


def test_crop_outside_original_rejected(webm):
    with pytest.raises(ValueError, match="exceeds"):
        codec.decode_reference(webm, 4, 8)


def test_unsupported_media_and_missing_decoder_fail_clearly(tmp_path, webm, monkeypatch):
    bad = tmp_path / "not-audio"
    bad.write_bytes(b"not audio and not WebM")
    with pytest.raises(ValueError, match="Unsupported"):
        codec.decode_reference(bad, 0, 4)
    monkeypatch.setattr(codec, "_binary", lambda _: (_ for _ in ()).throw(ValueError("decoder missing")))
    with pytest.raises(ValueError, match="decoder missing"):
        codec.decode_reference(webm, 0, 4)


def test_decoder_timeout_and_error_fail_clearly():
    with pytest.raises(ValueError, match="timed out"):
        codec._run([sys.executable, "-c", "import time; time.sleep(5)"],
                   timeout_seconds=.1)
    with pytest.raises(ValueError, match="could not decode"):
        codec._run([sys.executable, "-c", "import sys; sys.exit(1)"])


@pytest.mark.parametrize("pipe", ["stdout", "stderr"])
def test_decoder_output_limits_kill_process_and_leave_no_hang(pipe):
    code = f"import sys; sys.{pipe}.buffer.write(b'x' * 2000000); sys.{pipe}.flush()"
    with pytest.raises(ValueError, match="output byte bound"):
        codec._run([sys.executable, "-c", code], max_stdout_bytes=1024)
    assert codec._run([sys.executable, "-c", "print('still usable')"]).strip() == b"still usable"


def test_acquisition_and_worker_reconstruct_original_without_model(webm, tmp_path, monkeypatch):
    root = tmp_path / "project"
    directory = root / "voice_lab/discovery"
    raw = webm.read_bytes()
    transcript = "Disposable synthetic codec fixture."
    source = {"source_id": "synthetic-codec-fixture",
              "source_page": "https://commons.wikimedia.org/wiki/File:Fixture.webm",
              "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.webm",
              "attribution": "Synthetic test fixture",
              "license_url": "https://creativecommons.org/licenses/by/3.0/",
              "terms_url": discovery.TERMS_URL, "language": "en", "duration_seconds": 6,
              "upstream_media_sha1": hashlib.sha1(raw).hexdigest()}
    permission = {"source_id": source["source_id"], "status": "approved",
                  "use": "private_synthetic_voice", "authority": "owner_original_voice",
                  "evidence_ref": "disposable-test-fixture:ownership"}
    review = {"source_id": source["source_id"], "original_sha256": hashlib.sha256(raw).hexdigest(),
              "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
              "start_seconds": .75, "end_seconds": 4.75, "method": "local_tools",
              "evidence_ref": "disposable-test-fixture:generated-signal",
              "single_speaker": True, "transcript_aligned": True,
              "overlapping_speech": False, "heavy_music": False, "unsuitable_noise": False}
    monkeypatch.setattr(discovery, "_download_media", lambda *args, **kwargs: raw)
    receipt = discovery.acquire_reference(directory, source, intended_use_evidence=permission,
                 transcript=transcript, start_seconds=.75, end_seconds=4.75, audio_review=review)
    assert receipt["clone_ready"] and receipt["source_decode"]["decoder"] == "ffmpeg"
    assert (directory / receipt["original_path"]).read_bytes() == raw
    verified, original, reference = voice._eligible_acquisition(
        root, receipt["acquisition_id"], receipt["acquisition_sha256"])
    assert verified["reference_audio_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    decoded = codec.decode_reference(original, .75, 4.75)
    samples, _ = sf.read(reference, dtype="float32")
    assert np.array_equal(samples, decoded.wave.mean(axis=1))

    path = directory / "acquisitions" / receipt["acquisition_id"] / "acquisition.json"
    changed = json.loads(path.read_text("utf-8"))
    changed["source_decode"]["ffmpeg_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), "utf-8")
    with pytest.raises(ValueError, match="provenance changed"):
        voice._eligible_acquisition(root, receipt["acquisition_id"],
                                    hashlib.sha256(path.read_bytes()).hexdigest())


def test_review_only_webm_still_cannot_reach_model(webm, tmp_path, monkeypatch):
    raw = webm.read_bytes()
    source = {"source_id": "synthetic-held", "source_page": "https://commons.wikimedia.org/fixture",
              "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/fixture.webm",
              "attribution": "Synthetic fixture", "license_url": "https://creativecommons.org/licenses/by/3.0",
              "terms_url": discovery.TERMS_URL, "language": "en", "duration_seconds": 6}
    monkeypatch.setattr(discovery, "_download_media", lambda *args, **kwargs: raw)
    root = tmp_path / "held"
    receipt = discovery.acquire_reference(root / "voice_lab/discovery", source,
              intended_use_evidence={}, transcript="Synthetic fixture", start_seconds=0,
              end_seconds=4, review_only=True)
    assert not receipt["clone_ready"]
    with pytest.raises(ValueError, match="eligible use"):
        voice._eligible_acquisition(root, receipt["acquisition_id"], receipt["acquisition_sha256"])


def test_webm_bound_does_not_relax_default_download_budget():
    assert discovery.DiscoveryLimits().max_download_bytes == 8_000_000
    assert discovery.DiscoveryLimits(max_download_bytes=32_000_000).max_download_bytes == 32_000_000
    with pytest.raises(ValueError):
        discovery.DiscoveryLimits(max_download_bytes=32_000_001)



def test_larger_original_budget_is_webm_only(tmp_path):
    path = tmp_path / "original.wav"
    sf.write(path, np.sin(np.arange(16000 * 5) / 31).astype(np.float32) * .1,
             16000, subtype="FLOAT")
    with path.open("r+b") as stream:
        stream.truncate(16_000_001)
    with pytest.raises(ValueError, match="legacy byte limit"):
        codec.decode_reference(path, 0, 4)
