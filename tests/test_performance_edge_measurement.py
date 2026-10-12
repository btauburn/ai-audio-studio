"""CPU edge-measurement contracts: exact frames, conservative silence, honest holds."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries

RATE = 24000
HOP = 240  # one 10 ms frame at 24 kHz


def tone(blocks: int, dbfs: float) -> np.ndarray:
    """A tone with an exact integer number of cycles per frame, at a known level."""

    amplitude = np.sqrt(2.0) * (10.0 ** (dbfs / 20.0))
    return amplitude * np.sin(2.0 * np.pi * np.arange(blocks * HOP) / HOP)


def silent(blocks: int) -> np.ndarray:
    return np.zeros(blocks * HOP, dtype=np.float64)


def alternating(blocks: int, level: float = 0.5) -> np.ndarray:
    """Half-frames that begin negative: a direct probe that the sign survives decoding."""

    index = np.arange(blocks * HOP) // (HOP // 2)
    return np.where(index % 2 == 0, -level, level)


def write_wav(path: Path, samples: np.ndarray, *, rate: int = RATE, width: int = 2) -> Path:
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    if width == 1:
        pcm = np.round(data * 127.0 + 128.0).astype(np.uint8).tobytes()
    elif width == 3:
        whole = np.round(data * 8388607.0).astype("<i4")
        pcm = whole.view(np.uint8).reshape(-1, 4)[:, :3].copy().tobytes()
    elif width == 4:
        pcm = np.round(data * 2147483647.0).astype("<i4").tobytes()
    else:
        pcm = np.round(data * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return path


def measure(path: Path, **kwargs):
    kwargs.setdefault("text", "The room was quiet.")
    return boundaries.measure_take(path, hashlib.sha256(path.read_bytes()).hexdigest(), **kwargs)


@pytest.fixture()
def layered(tmp_path: Path) -> Path:
    """10 frames of digital silence, 10 of a -60 dBFS fricative, 50 of speech,
    10 of a -53 dBFS breath tail, 10 of digital silence."""

    signal = np.concatenate([silent(10), tone(10, -60.0), tone(50, -9.0),
                             tone(10, -53.0), silent(10)])
    return write_wav(tmp_path / "layered.wav", signal)


def test_frame_count_is_exact_and_edges_are_frame_aligned(layered: Path) -> None:
    record = measure(layered)
    assert record["schema_version"] == "edge-measurement-v1"
    assert record["status"] == "measured"
    assert record["frames"] == 90 * HOP
    assert record["sample_rate"] == RATE
    assert record["channels"] == 1
    assert record["detector"]["frame_ms"] == 10
    assert record["detector"]["frame_samples"] == HOP
    for key in ("content_frames", "silent_frames"):
        assert record["leading"][key] % HOP == 0, key
        assert record["trailing"][key] % HOP == 0, key


def test_detector_finds_the_known_runs_and_never_claims_a_longer_one(layered: Path) -> None:
    record = measure(layered)
    # True digital silence is 10 frames at each end.  The detector is quantised
    # from sample 0 and needs every sample in a frame below the ceiling, so it
    # can never report more than the fixture contains.
    assert record["leading"]["silent_frames"] == 10 * HOP
    assert record["trailing"]["silent_frames"] == 10 * HOP
    # The leading edge region is the span BEFORE the first content window, so the
    # first speech frame (index 20) is not counted as preceding silence.  It was
    # 21 * HOP before the repair; fixture truth is 20.
    assert record["leading"]["content_frames"] == 20 * HOP
    # The trailing edge region starts after the last content frame (index 69).
    assert record["trailing"]["content_frames"] == (90 - 70) * HOP
    assert record["leading"]["floor_dbfs"] is not None
    assert record["leading"]["floor_dbfs"] < boundaries.CONTENT_FLOOR_DBFS
    assert record["trailing"]["floor_dbfs"] is not None
    assert record["trailing"]["floor_dbfs"] > boundaries.SILENCE_CEILING_DBFS


def test_a_quiet_fricative_is_not_swallowed_by_the_silent_run(layered: Path) -> None:
    record = measure(layered)
    # The -60 dBFS fricative sits between the two thresholds: it is not content,
    # but it is also not strictly silent, so the destructive-shortening measure
    # stops at the true silence instead of running into it.
    assert record["leading"]["silent_frames"] < record["leading"]["content_frames"]
    assert record["leading"]["silent_frames"] == 10 * HOP


def test_detector_is_conservative_on_every_fixture(tmp_path: Path) -> None:
    cases = [("plain", np.concatenate([silent(7), tone(20, -12.0), silent(3)]), 7, 3),
             ("breath", np.concatenate([silent(4), tone(30, -20.0), tone(5, -58.0), silent(6)]), 4, 6),
             ("reverb", np.concatenate([tone(2, -30.0), tone(40, -14.0), tone(12, -64.0)]), 0, 0),
             ("all_silent", silent(12), 12, 12)]
    for name, signal, true_lead, true_tail in cases:
        record = measure(write_wav(tmp_path / f"{name}.wav", signal))
        assert record["leading"]["silent_frames"] <= true_lead * HOP, name
        assert record["trailing"]["silent_frames"] <= true_tail * HOP, name
        assert record["leading"]["content_frames"] <= record["frames"], name


def test_thresholds_have_cases_on_both_sides(tmp_path: Path) -> None:
    above = measure(write_wav(tmp_path / "above.wav", tone(20, -45.0)))
    assert above["status"] == "measured"
    # Content starts in the very first window, so there is no pre-content lead-in
    # at all.  It was HOP before the repair; fixture truth is 0.
    assert above["leading"]["content_frames"] == 0
    assert above["leading"]["silent_frames"] == 0
    assert above["trailing"]["content_frames"] == 0

    below = measure(write_wav(tmp_path / "below.wav", np.concatenate(
        [silent(3), tone(6, -63.0), silent(3)])))
    # Nothing reached the -50 dBFS content floor, so this material is held.
    assert below["status"] == "all_silent"
    assert below["leading"]["silent_frames"] == 3 * HOP
    assert below["trailing"]["silent_frames"] == 3 * HOP
    assert boundaries.shortening_eligible(below) is False


def test_an_all_silent_take_is_held_and_never_shortenable(tmp_path: Path) -> None:
    record = measure(write_wav(tmp_path / "silence.wav", silent(40)))
    assert record["status"] == "all_silent"
    assert boundaries.ALL_SILENT_FLAG in record["flags"]
    assert boundaries.shortening_eligible(record) is False
    assert boundaries.edge_uncertain(record) is True
    assert record["flags"] and record["machine_generated"] is True
    assert record["human_verified"] is False


def test_dc_offset_is_removed_before_the_edges_are_measured(tmp_path: Path) -> None:
    signal = np.concatenate([silent(10), tone(50, -9.0), silent(10)]) + 0.4
    record = measure(write_wav(tmp_path / "dc.wav", signal))
    assert record["detector"]["dc_removed"] is True
    assert record["detector"]["dc_offset"] > 0.3
    # Without DC removal the lead-in would look like content at the floor level.
    assert record["leading"]["silent_frames"] == 10 * HOP
    assert record["status"] == "measured"


def test_onset_reports_the_first_sample_and_the_first_two_milliseconds(tmp_path: Path) -> None:
    record = measure(write_wav(tmp_path / "onset.wav", tone(20, -6.0)))
    assert record["onset"]["dc_removed"] is True
    assert record["onset"]["first_sample"] is not None
    assert record["onset"]["max_step_2ms"] is not None
    assert record["onset"]["max_step_2ms"] > 0

    quiet = measure(write_wav(tmp_path / "quiet.wav", tone(20, -60.0)))
    assert quiet["onset"]["first_sample"] is not None
    assert abs(quiet["onset"]["first_sample"]) < 0.01


def test_lexical_evidence_is_never_guessed(layered: Path) -> None:
    default = measure(layered)
    assert default["lexical"]["status"] == "not_requested"
    assert default["lexical"]["first_word_start_s"] is None
    assert default["lexical"]["last_word_end_s"] is None
    assert default["lexical"]["wer"] is None
    assert "lexical_unavailable" not in default["flags"]

    asked = measure(layered, lexical=True)
    assert asked["lexical"]["status"] == "unavailable"
    assert asked["lexical"]["reason"]
    assert asked["lexical"]["model_sha256"] is None
    assert asked["lexical"]["settings"] is None
    assert asked["lexical"]["first_word_start_s"] is None
    assert "lexical_unavailable" in asked["flags"]


def test_the_record_carries_its_own_digest_and_no_writes(layered: Path) -> None:
    before = sorted(item.name for item in layered.parent.iterdir())
    record = measure(layered)
    assert record["content_sha256"] == boundaries.content_sha256(record)
    assert len(record["content_sha256"]) == 64
    assert record["text_sha256"] == boundaries.sha256_bytes("The room was quiet.".encode("utf-8"))
    assert sorted(item.name for item in layered.parent.iterdir()) == before


def test_expected_digest_shape_is_validated_before_comparison(layered: Path) -> None:
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(layered, "not-a-digest", text=None)
    assert refused.value.code == "input_ineligible"
    with pytest.raises(boundaries.PerformanceDirectorError) as upper:
        boundaries.measure_take(layered, "A" * 64, text=None)
    assert upper.value.code == "input_ineligible"


def test_hash_mismatch_missing_and_malformed_inputs_refuse(tmp_path: Path) -> None:
    good = write_wav(tmp_path / "good.wav", tone(20, -12.0))
    with pytest.raises(boundaries.PerformanceDirectorError) as mismatch:
        boundaries.measure_take(good, "0" * 64, text=None)
    assert mismatch.value.code == "input_ineligible"

    with pytest.raises(boundaries.PerformanceDirectorError) as missing:
        boundaries.measure_take(tmp_path / "absent.wav", "0" * 64, text=None)
    assert missing.value.code == "input_ineligible"

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is not audio at all")
    with pytest.raises(boundaries.PerformanceDirectorError) as malformed:
        boundaries.measure_take(broken, hashlib.sha256(broken.read_bytes()).hexdigest(), text=None)
    assert malformed.value.code == "input_ineligible"

    empty = tmp_path / "empty.wav"
    write_wav(empty, np.zeros(0, dtype=np.float64))
    with pytest.raises(boundaries.PerformanceDirectorError) as absent_frames:
        boundaries.measure_take(empty, hashlib.sha256(empty.read_bytes()).hexdigest(), text=None)
    assert absent_frames.value.code == "input_ineligible"


def test_non_pcm_wav_is_refused(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    float_wav = tmp_path / "float.wav"
    soundfile.write(str(float_wav), tone(20, -12.0), RATE, subtype="FLOAT")
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(float_wav, hashlib.sha256(float_wav.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_supported_integer_sample_width_decodes(tmp_path: Path, width: int) -> None:
    path = write_wav(tmp_path / f"w{width}.wav", np.concatenate([silent(5), tone(15, -9.0), silent(5)]),
                     width=width)
    record = measure(path)
    assert record["frames"] == 25 * HOP
    assert record["status"] == "measured"
    assert record["leading"]["silent_frames"] == 5 * HOP
    assert record["trailing"]["silent_frames"] == 5 * HOP
    # A symmetric tone has no DC drift, so nothing leaks into the silent runs.
    assert abs(record["detector"]["dc_offset"]) < 0.01


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_negative_samples_keep_their_sign_at_every_width(tmp_path: Path, width: int) -> None:
    # A 24-bit two's-complement value whose top bit is set must not decode as a
    # large positive number.  This fixture begins negative and has zero mean, so
    # the first sample survives DC removal unchanged.
    path = write_wav(tmp_path / f"neg{width}.wav", alternating(20), width=width)
    record = measure(path)
    assert record["status"] == "measured"
    assert record["onset"]["first_sample"] == pytest.approx(-0.5, abs=0.01)
    assert abs(record["detector"]["dc_offset"]) < 0.01


def test_multi_channel_audio_is_measured_not_rejected(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    signal = np.concatenate([silent(5), tone(15, -9.0), silent(5)])
    interleaved = np.repeat(np.round(np.clip(signal, -1, 1) * 32767).astype("<i2"), 2)
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(interleaved.tobytes())
    record = measure(stereo)
    assert record["channels"] == 2
    assert record["frames"] == 25 * HOP
    assert record["leading"]["silent_frames"] == 5 * HOP


def test_shortening_eligibility_requires_measured_audio_and_silence(layered: Path) -> None:
    eligible = measure(layered)
    assert boundaries.shortening_eligible(eligible) is True
    assert boundaries.edge_uncertain(eligible) is False
    assert boundaries.shortening_eligible({}) is False
    assert boundaries.edge_uncertain({}) is True


# ------------------------------------------------------------------ repair regressions
# Each of these fails against the original M2 module bytes.


def test_content_from_the_first_sample_has_no_leading_edge(tmp_path: Path) -> None:
    record = measure(write_wav(tmp_path / "immediate.wav", alternating(20)))
    # The old detector reported (first_content + 1) * frame_samples == 240 here.
    assert record["leading"]["content_frames"] == 0
    assert record["leading"]["silent_frames"] == 0
    assert record["trailing"]["content_frames"] == 0
    assert record["status"] == "measured"


def test_an_active_final_partial_window_is_not_trailing_silence(tmp_path: Path) -> None:
    # 5 frames of content, 50 frames of digital silence, then 120 loud samples:
    # the last window is shorter than 10 ms and must still be measured.  The old
    # detector dropped it and reported 12000 frames of trailing content edge.
    signal = np.concatenate([alternating(5), silent(50), alternating(0.5)[:120]])
    record = measure(write_wav(tmp_path / "partial-tail.wav", signal))
    assert record["frames"] == 5 * HOP + 50 * HOP + 120
    assert record["trailing"]["content_frames"] == 0
    assert record["trailing"]["silent_frames"] == 0
    assert record["detector"]["partial_final_window_samples"] == 120
    assert boundaries.shortening_eligible(record) is False


def test_a_partial_terminal_window_of_silence_is_still_the_trailing_edge(tmp_path: Path) -> None:
    signal = np.concatenate([alternating(5), silent(50), np.zeros(120)])
    record = measure(write_wav(tmp_path / "partial-silent-tail.wav", signal))
    assert record["leading"]["content_frames"] == 0
    assert record["trailing"]["content_frames"] == 50 * HOP + 120
    assert record["trailing"]["silent_frames"] == 50 * HOP + 120


def test_opposite_phase_channels_never_cancel_into_safe_silence(tmp_path: Path) -> None:
    # Channel 0 is a tone; channel 1 is its inverse for the first 10 frames.
    # Averaging the channels yields exact silence there, which would let a
    # downstream offset discard audible samples.
    tone = alternating(30)
    stereo = np.column_stack([tone, np.concatenate([-tone[: 10 * HOP], tone[10 * HOP:]])])
    path = tmp_path / "opposite-phase.wav"
    interleaved = np.round(np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(interleaved.tobytes())
    record = measure(path)
    assert record["channels"] == 2
    assert record["status"] == "measured"
    assert record["leading"]["silent_frames"] == 0
    assert record["leading"]["content_frames"] == 0
    assert boundaries.shortening_eligible(record) is False
    # The channel mean of the prefix really is zero, so this fixture would pass
    # a mean-based detector.
    assert float(np.mean(np.abs(stereo[: 10 * HOP].mean(axis=1)))) == pytest.approx(0.0, abs=1e-9)


def test_a_channel_silent_while_another_is_audible_is_not_safe_silence(tmp_path: Path) -> None:
    loud = alternating(20, level=0.5)
    stereo = np.column_stack([loud, np.zeros_like(loud)])
    path = tmp_path / "one-channel-loud.wav"
    interleaved = np.round(np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(interleaved.tobytes())
    record = measure(path)
    assert record["leading"]["silent_frames"] == 0
    assert record["leading"]["content_frames"] == 0
    assert boundaries.shortening_eligible(record) is False


def test_per_channel_dc_is_removed_separately(tmp_path: Path) -> None:
    stereo = np.column_stack([alternating(20) + 0.4, alternating(20) - 0.4])
    path = tmp_path / "per-channel-dc.wav"
    interleaved = np.round(np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(interleaved.tobytes())
    record = measure(path)
    assert record["detector"]["dc_removed"] is True
    assert len(record["detector"]["dc_offsets"]) == 2
    assert record["detector"]["dc_offset"] > 0.3
    assert record["leading"]["silent_frames"] == 0
    assert record["leading"]["content_frames"] == 0


def test_truncated_pcm_payload_is_refused(tmp_path: Path) -> None:
    good = write_wav(tmp_path / "aligned.wav", alternating(20))
    honest = measure(good)
    assert honest["frames"] == 20 * HOP
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(good.read_bytes()[:-480])
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(truncated, hashlib.sha256(truncated.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"
    assert "truncated" in str(refused.value)


def test_an_unaligned_payload_is_refused(tmp_path: Path) -> None:
    good = write_wav(tmp_path / "odd.wav", alternating(20))
    odd = tmp_path / "unaligned.wav"
    odd.write_bytes(good.read_bytes()[:-1])
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(odd, hashlib.sha256(odd.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"


def test_bytes_after_the_data_chunk_do_not_change_the_measurement(tmp_path: Path) -> None:
    """Frames come from the declared data chunk; the file digest still binds."""

    good = write_wav(tmp_path / "over.wav", alternating(20))
    baseline = measure(good)
    padded = tmp_path / "padded.wav"
    padded.write_bytes(good.read_bytes() + b"\x00\x00" * 240)
    after = boundaries.measure_take(padded, hashlib.sha256(padded.read_bytes()).hexdigest(), text=None)
    assert after["frames"] == baseline["frames"] == 20 * HOP
    assert after["audio_sha256"] != baseline["audio_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.measure_take(padded, baseline["audio_sha256"], text=None)


def _data_chunk_offset(raw: bytes) -> int:
    return raw.index(b"data") + 4


def test_a_declared_data_chunk_unaligned_to_frames_is_refused(tmp_path: Path) -> None:
    """A declared odd-sized 16-bit chunk leaves an incomplete sample."""

    good = write_wav(tmp_path / "aligned-16.wav", alternating(20))
    raw = bytearray(good.read_bytes())
    declared = int.from_bytes(raw[_data_chunk_offset(raw):_data_chunk_offset(raw) + 4], "little")
    assert declared % 2 == 0
    # Declare one more byte than the payload holds, plus a legal RIFF pad byte,
    # so the file stays a well-formed RIFF while the declared chunk is odd.
    raw.extend(b"\x01\x00")
    raw[_data_chunk_offset(raw):_data_chunk_offset(raw) + 4] = (declared + 1).to_bytes(4, "little")
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    path = tmp_path / "declared-unaligned.wav"
    path.write_bytes(raw)
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(path, hashlib.sha256(path.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"
    assert "aligned" in str(refused.value)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_valid_aligned_chunks_still_measure(tmp_path: Path, width: int) -> None:
    path = write_wav(tmp_path / f"aligned-{width}.wav",
                     np.concatenate([silent(5), tone(15, -9.0), silent(5)]), width=width)
    record = measure(path)
    assert record["status"] == "measured"
    assert record["frames"] == 25 * HOP
    assert record["leading"]["silent_frames"] == 5 * HOP


def test_the_declared_chunk_is_validated_before_frame_rounding(tmp_path: Path) -> None:
    """Physical truncation is still refused after the alignment check."""

    good = write_wav(tmp_path / "rounding.wav", alternating(20))
    short = tmp_path / "rounding-short.wav"
    short.write_bytes(good.read_bytes()[:-240])
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(short, hashlib.sha256(short.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"
    assert "truncated" in str(refused.value)
    # The aligned original still measures.
    assert measure(good)["frames"] == 20 * HOP


def test_a_multi_channel_declared_chunk_must_align_to_whole_frames(tmp_path: Path) -> None:
    stereo = np.column_stack([alternating(10), alternating(10)])
    interleaved = np.round(np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes()
    aligned = tmp_path / "stereo-aligned.wav"
    with wave.open(str(aligned), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(interleaved)
    assert measure(aligned)["frames"] == 10 * HOP

    raw = bytearray(aligned.read_bytes())
    declared = int.from_bytes(raw[_data_chunk_offset(raw):_data_chunk_offset(raw) + 4], "little")
    raw.extend(b"\x00\x00")
    raw[_data_chunk_offset(raw):_data_chunk_offset(raw) + 4] = (declared + 2).to_bytes(4, "little")
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    odd = tmp_path / "stereo-unaligned.wav"
    odd.write_bytes(raw)
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.measure_take(odd, hashlib.sha256(odd.read_bytes()).hexdigest(), text=None)
    assert refused.value.code == "input_ineligible"
    # Two spare bytes in a 4-byte frame size is unaligned even though it is even.
    assert "aligned" in str(refused.value)


def test_edge_regions_are_disjoint_halves_of_the_content_span(layered: Path) -> None:
    record = measure(layered)
    leading = record["leading"]["content_frames"]
    content_end = record["frames"] - record["trailing"]["content_frames"]
    # The content span is bounded by the two edge regions and is non-empty.
    assert 0 < leading < content_end <= record["frames"]
    # The first content window is entirely inside the content span.
    assert leading + HOP <= content_end
