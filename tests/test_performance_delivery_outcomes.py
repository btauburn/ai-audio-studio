"""Advisory delivery outcome measurement, same-line comparison and read-only review (M6, P12).

Everything here is CPU and synthetic.  The fixture Takes are generated PCM WAVs
at a known tempo, the "listening" that never happened is never claimed, and the
lexical receipts are **fake** records shaped like the existing ``reference-asr-v1``
owner's output: no ASR model runs, no model is downloaded, and an ASR confidence
is never treated as reviewer authentication.

Fake fidelity: the fake receipts reproduce the accepted owner's field names
(``audio_sha256``, ``status``, ``text``/``text_sha256``, ``segments`` with
``start``/``end``/``text``/``words``, ``model``/``model_sha256``, ``settings``,
``language``, ``pipeline_sha256``, ``dependency_versions``, ``duration_seconds``,
``machine_generated``, ``human_verified``) and its draft/unavailable/failed
states.  They do **not** reproduce recognition accuracy, timing accuracy, real
word timestamps or a real model identity, so nothing here is evidence about a
real transcript or about how a Take sounds.

A real Stage B listening review does not exist, so every review status asserted
below is ``review_pending``.  The M3 owner gate is exercised both through its
real code path and through a stated test double; no double is a verdict.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.audiobook.fingerprint import sha256_bytes
from dubbing_service.app.audiobook.performance_director import PerformanceDirectorError
from tests.test_performance_pilot_assembly import (  # noqa: F401
    studio,
)

RATE = 24000
HOP = 240  # one 10 ms detection window at 24 kHz

TEMPO_TEXT = "You cannot be serious about this line."
#: A 100-token text, so one substitution is exactly 0.01 of a word error rate.
TOKENS = [f"w{index}" for index in range(100)]
LONG_TEXT = " ".join(TOKENS)


# --------------------------------------------------------------------------- fixtures and builders


def write_wav(path: Path, samples: np.ndarray, *, channels: int = 1, rate: int = RATE) -> Path:
    """Exact 16-bit PCM of a chosen channel count."""

    flat = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0).reshape(-1)
    if channels > 1:
        flat = np.repeat(flat.reshape(-1, 1), channels, axis=1).reshape(-1)
    raw = np.round(flat * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(raw)
    return path


def tone_bursts(count: int, on_frames: int, off_frames: int, *, level: float = 0.4) -> np.ndarray:
    """Whole-cycle tone bursts separated by digital silence, at a known tempo."""

    amplitude = np.sqrt(2.0) * level
    parts: list[np.ndarray] = []
    for _ in range(count):
        parts.append(amplitude * np.sin(2.0 * np.pi * np.arange(on_frames) / 160.0))
        parts.append(np.zeros(off_frames))
    return np.concatenate(parts)


def decoded_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def measure(path: Path, text: str | None, scratch_root: Path, **kwargs) -> dict:
    """Measure one Take through the public M6 entry point, binding its exact bytes."""

    return boundaries.measure_delivery_take(
        path, sha256_bytes(path.read_bytes()), text=text, scratch_root=scratch_root, **kwargs)


def word_rows(tokens: list[str], duration: float, *, probability: float = 0.9) -> list[dict]:
    if not tokens:
        return []
    step = duration / len(tokens)
    return [{"word": token, "start": round(index * step, 3),
             "end": round((index + 1) * step - 0.01, 3), "probability": probability}
            for index, token in enumerate(tokens)]


def fake_receipt(audio: Path, text: str, tokens: list[str], *, duration: float | None = None,
                 probability: float = 0.9, **overrides) -> dict:
    """A synthetic ``reference-asr-v1``-shaped receipt.  No ASR ran (see the module docstring)."""

    if duration is None:
        duration = decoded_duration(audio)
    receipt = {
        "schema": "reference-asr-v1",
        "status": "draft",
        "audio_sha256": sha256_bytes(audio.read_bytes()),
        "text": text,
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "segments": [{"start": 0.0, "end": duration, "text": text,
                      "words": word_rows(tokens, duration, probability=probability)}],
        "model": "faster-whisper-base.en",
        "model_sha256": "e" * 64,
        "settings": {"beam_size": 5, "word_timestamps": True, "temperature": 0.0},
        "language": "en",
        "pipeline_sha256": "f" * 64,
        "dependency_versions": {"numpy": "2.0.2"},
        "duration_seconds": duration,
        "machine_generated": True,
        "human_verified": False,
    }
    receipt.update(overrides)
    return receipt


def substitution_receipt(audio: Path, substitutions: int) -> dict:
    """A receipt whose text differs from ``LONG_TEXT`` in exactly N tokens."""

    tokens = list(TOKENS)
    for index in range(substitutions):
        tokens[index] = f"x{index}"
    return fake_receipt(audio, " ".join(tokens), tokens)


def metric_blocks(value, path: str = ""):
    """Every metric block in a record, as (path, block)."""

    if isinstance(value, dict):
        if {"value", "units", "method", "sample_count", "scope"} <= set(value):
            yield path, value
            return
        for key, item in value.items():
            yield from metric_blocks(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from metric_blocks(item, f"{path}[{index}]")


def all_numbers(value) -> list[float]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in all_numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in all_numbers(item)]
    return []


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    """A fresh, explicit scratch root.  Snapshots are only ever written here."""

    root = tmp_path / "scratch"
    root.mkdir()
    return root


@pytest.fixture()
def tempo(tmp_path: Path, scratch: Path) -> dict:
    """The same expected text at two tempos: 3.2 s and 2.1333 s, exactly 1.5x."""

    control = write_wav(tmp_path / "control.wav", tone_bursts(8, 5760, 3840))
    directed = write_wav(tmp_path / "directed.wav", tone_bursts(8, 3840, 2560))
    return {"control": control, "directed": directed,
            "control_record": measure(control, TEMPO_TEXT, scratch),
            "directed_record": measure(directed, TEMPO_TEXT, scratch)}


# --------------------------------------------------------------------------- A. the extended record


def test_the_delivery_record_extends_the_edge_record_without_rewriting_it(tempo) -> None:
    record = tempo["control_record"]
    base = boundaries.measure_take(tempo["control"], sha256_bytes(tempo["control"].read_bytes()),
                                   text=TEMPO_TEXT)
    assert record["schema_version"] == "edge-measurement-v1"
    assert record["delivery"]["schema_version"] == "pd-delivery-outcome-1"
    assert "delivery" not in base
    for key, value in base.items():
        if key == "content_sha256":
            continue
        assert record[key] == value, key
    # The M2 identity survives verbatim, and the enrichment has its own.
    assert record["delivery"]["edge_identity"]["content_sha256"] == base["content_sha256"]
    assert record["content_sha256"] != base["content_sha256"]
    assert boundaries.content_sha256(record) == record["content_sha256"]


def test_old_edge_callers_are_unaffected(tempo) -> None:
    base = boundaries.measure_take(tempo["control"], sha256_bytes(tempo["control"].read_bytes()),
                                   text=TEMPO_TEXT)
    assert set(base) == {"schema_version", "status", "audio_sha256", "sample_rate", "sample_domain",
                         "frames", "channels", "text_sha256", "detector", "leading", "trailing",
                         "onset", "lexical", "flags", "limitations", "machine_generated",
                         "human_verified", "content_sha256"}
    assert base["lexical"]["status"] == "not_requested"


def test_two_tempos_of_the_same_text_give_the_expected_rate_sign_and_size(tempo) -> None:
    control = tempo["control_record"]["delivery"]
    directed = tempo["directed_record"]["delivery"]
    assert control["audio"]["duration_seconds"]["value"] == pytest.approx(3.2)
    assert directed["audio"]["duration_seconds"]["value"] == pytest.approx(8 * 6400 / RATE)
    control_rate = control["elapsed_text_rate"]["words_per_minute"]["value"]
    directed_rate = directed["elapsed_text_rate"]["words_per_minute"]["value"]
    assert control_rate == pytest.approx(len(TEMPO_TEXT.split()) / 3.2 * 60.0)
    assert directed_rate > control_rate
    assert directed_rate / control_rate == pytest.approx(1.5, rel=1e-9)
    # The explicitly named articulation proxy moves the same way at this tempo.
    assert (directed["articulation_rate_proxy"]["syllables_per_minute"]["value"]
            > control["articulation_rate_proxy"]["syllables_per_minute"]["value"])


def test_the_counted_word_rate_is_not_the_acoustics_signal_proxy(tempo) -> None:
    delivery = tempo["control_record"]["delivery"]
    counted = delivery["elapsed_text_rate"]["words_per_minute"]
    proxy = delivery["articulation_rate_proxy"]["syllables_per_minute"]
    assert "counted expected-text tokens" in counted["method"]
    assert counted["sample_count"] == len(TEMPO_TEXT.split())
    assert "signal proxy" in proxy["derivation"]
    assert "not the acoustics" in counted["derivation"] or "not a lexical" in proxy["derivation"]
    # A pure tone bursts at syllable-like intervals, so the two numbers are not
    # forced to agree; the record states which one is which.
    assert proxy["units"] == "syllables/minute"
    assert counted["units"] == "words/minute"


def test_duration_rate_frame_and_channel_accounting_are_exact(tempo) -> None:
    audio = tempo["control_record"]["delivery"]["audio"]
    assert audio["frames"]["value"] == 76800
    assert audio["sample_rate"]["value"] == RATE
    assert audio["channels"]["value"] == 1
    assert audio["duration_seconds"]["value"] == pytest.approx(76800 / RATE)
    assert audio["sample_domain"] == f"pcm-mono-{RATE}"
    assert audio["frames"]["sample_count"] == 76800


def test_a_known_pause_envelope_is_reported(tmp_path, scratch) -> None:
    """Six 150 ms bursts separated by 300 ms of silence: six detected pauses."""

    path = write_wav(tmp_path / "pauses.wav", tone_bursts(6, 3600, 7200))
    delivery = measure(path, "A line with pauses here.", scratch)["delivery"]
    assert delivery["audio"]["duration_seconds"]["value"] == pytest.approx(2.7)
    assert delivery["pauses"]["count"]["value"] == 6
    assert delivery["pauses"]["count_per_minute"]["value"] > 0
    assert delivery["pauses"]["mean_pause_ms"]["value"] == pytest.approx(250.0, abs=1.0)
    assert delivery["pauses"]["count"]["derivation"].startswith("derived from")


def test_median_pitch_range_onset_and_peak_are_reported(tempo) -> None:
    delivery = tempo["control_record"]["delivery"]
    assert delivery["pitch"]["median_hz"]["value"] > 0
    assert delivery["pitch"]["range_semitones"]["value"] is not None
    assert delivery["pitch"]["median_hz"]["units"] == "Hz"
    assert delivery["onset"]["first_sample"]["value"] is not None
    assert delivery["onset"]["max_step_2ms"]["value"] is not None
    assert delivery["onset"]["max_step_2ms"]["sample_count"] == 48  # 2 ms at 24 kHz
    assert delivery["loudness"]["peak_dbfs"]["value"] < 0


def test_every_reported_number_carries_units_method_count_and_scope(tempo) -> None:
    record = tempo["control_record"]
    blocks = list(metric_blocks(record))
    assert len(blocks) >= 15
    for path, block in blocks:
        assert block["units"], path
        assert block["method"], path
        assert block["scope"], path
        assert isinstance(block["sample_count"], int) and not isinstance(block["sample_count"], bool)
        if block["value"] is None:
            assert block["reason"], path
        else:
            assert math.isfinite(block["value"]), path
    numbers = all_numbers(record)
    assert numbers and all(math.isfinite(number) for number in numbers)


def test_a_stereo_take_is_measured_and_its_downmix_is_disclosed(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "stereo.wav", tone_bursts(4, 4800, 4800), channels=2)
    record = measure(path, "A stereo line.", scratch)
    assert record["channels"] == 2
    assert record["delivery"]["audio"]["channels"]["value"] == 2
    notes = record["delivery"]["acoustics"]["notes"]
    assert any("downmixed" in note for note in notes)
    assert "downmixes multi-channel input by mean" in record["delivery"]["acoustics"]["limitations"]


def test_the_thirty_second_analysis_limit_and_scope_are_retained(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "long.wav", tone_bursts(16, 24000, 24000))
    delivery = measure(path, LONG_TEXT, scratch)["delivery"]
    assert delivery["audio"]["duration_seconds"]["value"] == pytest.approx(32.0)
    assert delivery["acoustics"]["analysis_limit_seconds"] == 30.0
    assert delivery["acoustics"]["analysed_seconds"] == pytest.approx(30.0)
    assert "first 30.000 s of 32.000 s" in delivery["pitch"]["median_hz"]["scope"]
    assert delivery["pitch"]["median_hz"]["sample_count"] == int(round(30.0 * 16000))


# --------------------------------------------------------------------------- A. the scratch snapshot


def test_a_snapshot_is_content_addressed_reused_and_retained(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    first = measure(path, "A line.", scratch)
    folder = scratch / boundaries.DELIVERY_SNAPSHOT_DIRECTORY
    snapshot = folder / f"{sha256_bytes(path.read_bytes())}.wav"
    captured = snapshot.read_bytes()
    second = measure(path, "A line.", scratch)
    files = sorted(folder.iterdir())
    # Exactly one content-addressed capture, and verified reuse never rewrote it.
    assert len(files) == 1
    assert files[0] == snapshot
    assert snapshot.read_bytes() == captured == path.read_bytes()
    assert first["delivery"]["snapshot"]["sha256"] == sha256_bytes(path.read_bytes())
    assert first["delivery"]["snapshot"]["retained"] is True
    assert "never overwritten" in first["delivery"]["snapshot"]["reuse_policy"]
    # Two measurements of unchanged bytes are the same measurement.
    assert first["content_sha256"] == second["content_sha256"]


def test_a_missing_scratch_root_is_refused_and_nothing_is_written(tmp_path) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    absent = tmp_path / "absent"
    with pytest.raises(PerformanceDirectorError, match="scratch_root"):
        measure(path, "A line.", absent)
    assert not absent.exists()


def test_planted_snapshot_bytes_are_never_overwritten(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    folder = scratch / boundaries.DELIVERY_SNAPSHOT_DIRECTORY
    folder.mkdir()
    planted = folder / f"{sha256_bytes(path.read_bytes())}.wav"
    planted.write_bytes(b"planted bytes")
    with pytest.raises(PerformanceDirectorError, match="never overwritten"):
        measure(path, "A line.", scratch)
    assert planted.read_bytes() == b"planted bytes"


def test_a_snapshot_that_changes_during_measurement_is_refused(tmp_path, scratch, monkeypatch) -> None:
    from service import voice_acoustics

    path = write_wav(tmp_path / "take.wav", tone_bursts(6, 4800, 4800))
    real_measure = voice_acoustics.measure

    def tampering(measured, **kwargs):
        result = real_measure(measured, **kwargs)
        with Path(measured).open("ab") as stream:
            stream.write(b"\x00\x00")
        return result

    monkeypatch.setattr(voice_acoustics, "measure", tampering)
    with pytest.raises(PerformanceDirectorError, match="changed during measurement"):
        measure(path, "A line.", scratch)


def test_a_hash_mismatch_a_missing_file_and_a_non_wav_are_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    with pytest.raises(PerformanceDirectorError, match="expected digest"):
        boundaries.measure_delivery_take(path, "a" * 64, text="A line.", scratch_root=scratch)
    with pytest.raises(PerformanceDirectorError, match="missing"):
        boundaries.measure_delivery_take(tmp_path / "absent.wav",
                                         sha256_bytes(path.read_bytes()),
                                         text="A line.", scratch_root=scratch)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wave file at all")
    with pytest.raises(PerformanceDirectorError):
        boundaries.measure_delivery_take(broken, sha256_bytes(broken.read_bytes()),
                                         text="A line.", scratch_root=scratch)


# --------------------------------------------------------------------------- A. lexical receipts


def test_no_receipt_leaves_the_alignment_fields_null(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    delivery = measure(path, "A line.", scratch)["delivery"]
    assert delivery["lexical"] is None
    assert "no lexical receipt was supplied" in delivery["lexical_reason"]
    assert not [row for row in delivery["unknowns"] if row["field"].startswith("lexical")]
    assert delivery["lexical_reason"] is not None


def test_an_unavailable_or_failed_receipt_stays_null_with_its_reason(tmp_path, scratch) -> None:
    from dubbing_service.app.audiobook.reference_transcripts import SETTINGS

    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    receipt = {"schema": "reference-asr-v1", "status": "unavailable", "reason": "no model is installed",
               "audio_sha256": sha256_bytes(path.read_bytes()), "text": "", "segments": [],
               "settings": SETTINGS, "language": "en", "uncertainties": [], "correction_available": True,
               "machine_generated": True, "human_verified": False}
    delivery = measure(path, "A line.", scratch, lexical_receipt=receipt)["delivery"]
    lexical = delivery["lexical"]
    assert lexical["status"] == "unavailable"
    assert lexical["owner_reason"] == "no model is installed"
    assert "no alignment is available" in lexical["reason"]
    assert lexical["wer"] is None and lexical["final_word"] is None
    assert lexical["alignment"] is None
    assert lexical["settings_sha256"] is not None
    failed = dict(receipt, status="failed", reason="dependencies changed")
    assert measure(path, "A line.", scratch,
                   lexical_receipt=failed)["delivery"]["lexical"]["status"] == "failed"


def test_a_usable_receipt_gives_a_deterministic_rate_and_final_word(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 5760, 3840))
    text = "one two three four five six seven eight nine ten"
    tokens = text.split()
    receipt = fake_receipt(path, text, tokens, probability=0.8)
    lexical = measure(path, text, scratch, lexical_receipt=receipt)["delivery"]["lexical"]
    assert lexical["wer"]["value"] == 0.0
    assert lexical["wer"]["edit_count"] == 0
    assert lexical["wer"]["sample_count"] == 10
    assert lexical["minimum_word_probability"]["value"] == pytest.approx(0.8)
    assert lexical["final_word"]["present"] is True
    assert lexical["final_word"]["is_last_recognised_token"] is True
    assert lexical["final_word"]["probability"]["value"] == pytest.approx(0.8)
    assert lexical["alignment"]["first_word_start_s"] == 0.0
    assert lexical["alignment"]["last_word_end_s"] <= decoded_duration(path)
    assert lexical["machine_generated"] is True and lexical["human_verified"] is False


def test_a_receipt_bound_to_other_audio_is_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    other = write_wav(tmp_path / "other.wav", tone_bursts(5, 4800, 4800))
    receipt = fake_receipt(other, "A line.", ["A", "line."])
    with pytest.raises(PerformanceDirectorError, match="does not bind"):
        measure(path, "A line.", scratch, lexical_receipt=receipt)


@pytest.mark.parametrize("overrides, match", [
    ({"text_sha256": "b" * 64}, "does not match its own text"),
    ({"schema": "reference-asr-v1-draft"}, "schema"),
    ({"machine_generated": False}, "machine_generated"),
    ({"human_verified": True}, "human_verified"),
    ({"status": "verified"}, "status must be one of"),
    ({"duration_seconds": 99.0}, "duration_seconds"),
    ({"text_sha256": None}, "text_sha256"),
])
def test_a_malformed_receipt_is_refused(tmp_path, scratch, overrides, match) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    receipt = fake_receipt(path, "A line.", ["A", "line."])
    receipt.update(overrides)
    with pytest.raises(PerformanceDirectorError, match=match):
        measure(path, "A line.", scratch, lexical_receipt=receipt)


def test_non_finite_and_out_of_bounds_word_evidence_is_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    base = fake_receipt(path, "A line.", ["A", "line."])
    cases = [
        (lambda row: row.update(probability=float("nan")), "non-finite"),
        (lambda row: row.update(probability=float("inf")), "non-finite"),
        (lambda row: row.update(probability=1.5), "between 0 and 1"),
        (lambda row: row.update(start=-1.0), "must be at least 0.0"),
        (lambda row: row.update(end=999.0), "outside the exact Take audio"),
        (lambda row: row.update(start=1.5, end=1.4), "outside the exact Take audio"),
        (lambda row: row.update(start=float("nan")), "non-finite"),
    ]
    for mutate, match in cases:
        broken = json.loads(json.dumps(base))
        mutate(broken["segments"][0]["words"][0])
        with pytest.raises(PerformanceDirectorError, match=match):
            measure(path, "A line.", scratch, lexical_receipt=broken)


def test_unordered_or_non_finite_segment_evidence_is_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    base = fake_receipt(path, "A line.", ["A", "line."])
    first = dict(base["segments"][0], start=0.5, end=0.9, words=[])
    later = {"start": 0.1, "end": 0.4, "text": "again", "words": []}
    out_of_order = dict(base, segments=[first, later])
    with pytest.raises(PerformanceDirectorError, match="not ordered"):
        measure(path, "A line.", scratch, lexical_receipt=out_of_order)
    tail = json.loads(json.dumps(base))
    tail["segments"][0]["words"][1]["start"] = 0.0
    tail["segments"][0]["words"][1]["end"] = 0.05
    tail["segments"][0]["words"][0]["start"] = 0.2
    tail["segments"][0]["words"][0]["end"] = 0.3
    with pytest.raises(PerformanceDirectorError, match="not ordered"):
        measure(path, "A line.", scratch, lexical_receipt=tail)


def test_an_oversized_receipt_is_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    base = fake_receipt(path, "A line.", ["A", "line."])
    many = [dict(base["segments"][0]) for _ in range(boundaries.LEXICAL_MAX_SEGMENTS + 1)]
    with pytest.raises(PerformanceDirectorError, match="segments"):
        measure(path, "A line.", scratch, lexical_receipt=dict(base, segments=many))


def test_a_receipt_change_invalidates_the_enriched_identity(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(8, 4800, 4800))
    plain = measure(path, "A line.", scratch)
    receipt = fake_receipt(path, "A line.", ["A", "line."])
    supplied = measure(path, "A line.", scratch, lexical_receipt=receipt)
    resettings = measure(path, "A line.", scratch,
                         lexical_receipt=dict(receipt, settings={"beam_size": 3}))
    assert plain["content_sha256"] != supplied["content_sha256"]
    assert supplied["content_sha256"] != resettings["content_sha256"]
    # The M2 edge identity is untouched by any of it.
    assert (plain["delivery"]["edge_identity"]["content_sha256"]
            == supplied["delivery"]["edge_identity"]["content_sha256"]
            == resettings["delivery"]["edge_identity"]["content_sha256"])


def test_a_detector_change_invalidates_the_enrichment_only(tempo, scratch, monkeypatch) -> None:
    plain = tempo["control_record"]
    monkeypatch.setattr(boundaries, "DETECTOR_VERSION", "pd-edge-2")
    changed = measure(tempo["control"], TEMPO_TEXT, scratch)
    assert changed["delivery"]["edge_identity"]["detector_version"] == "pd-edge-2"
    assert changed["delivery"]["edge_identity"]["content_sha256"] != plain["delivery"]["edge_identity"]["content_sha256"]
    assert changed["content_sha256"] != plain["content_sha256"]


# --------------------------------------------------------------------------- A. generation evidence


def test_generation_evidence_is_reported_only_when_it_was_recorded(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    assert measure(path, "A line.", scratch)["delivery"]["generation"] is None
    recorded = measure(path, "A line.", scratch,
                       generation={"token_budget": 200, "instruction_tokens": 40,
                                   "seed_salt": 1, "identity_id": "voice-narrator"})["delivery"]
    generation = recorded["generation"]
    assert generation["token_budget"] == pytest.approx(200.0)
    assert generation["instruction_tokens"] == pytest.approx(40.0)
    assert generation["guard_proximity"]["value"] == pytest.approx(0.2)
    assert generation["guard_proximity"]["units"] == "ratio"
    assert generation["approached"] is None
    assert "no guard threshold was recorded" in generation["approached_reason"]
    assert len(generation["identity"]["sha256"]) == 64


def test_an_unknown_budget_or_threshold_is_null_not_a_guess(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    unknown = measure(path, "A line.", scratch,
                      generation={"seed_salt": 2})["delivery"]["generation"]
    assert unknown["token_budget"] is None
    assert unknown["guard_proximity"]["value"] is None
    assert "no token budget" in unknown["guard_proximity"]["reason"]
    zero = measure(path, "A line.", scratch,
                   generation={"token_budget": 0, "instruction_tokens": 5})["delivery"]["generation"]
    assert zero["guard_proximity"]["value"] is None
    assert "zero" in zero["guard_proximity"]["reason"]
    approached = measure(path, "A line.", scratch,
                         generation={"token_budget": 100, "instruction_tokens": 90,
                                     "guard_threshold": 0.8})["delivery"]["generation"]
    assert approached["approached"] is True
    assert approached["guard_proximity"]["value"] == pytest.approx(0.9)


def test_non_finite_generation_evidence_is_refused(tmp_path, scratch) -> None:
    path = write_wav(tmp_path / "take.wav", tone_bursts(4, 4800, 4800))
    for bad in ({"token_budget": float("nan")}, {"instruction_tokens": float("inf")},
                {"token_budget": float("-inf")}):
        with pytest.raises(PerformanceDirectorError, match="finite"):
            measure(path, "A line.", scratch, generation=bad)
    with pytest.raises(PerformanceDirectorError, match="mapping"):
        measure(path, "A line.", scratch, generation=[1, 2])


# --------------------------------------------------------------------------- B. the same-line comparison


def test_the_comparison_reports_hashes_deltas_methods_and_sample_counts(tempo) -> None:
    comparison = boundaries.compare_delivery_outcomes(tempo["control_record"],
                                                     tempo["directed_record"])
    assert comparison["schema_version"] == "pd-delivery-comparison-1"
    assert comparison["wer_threshold"] == 0.1
    assert "not Brian's listening criterion" in comparison["flags"]["threshold_basis"]
    assert comparison["control"]["content_sha256"] == tempo["control_record"]["content_sha256"]
    assert comparison["directed"]["audio_sha256"] == sha256_bytes(tempo["directed"].read_bytes())
    assert comparison["control"]["edge_content_sha256"] == (
        tempo["control_record"]["delivery"]["edge_identity"]["content_sha256"])
    assert comparison["text_sha256"] == tempo["control_record"]["delivery"]["text"]["sha256"]
    duration = comparison["deltas"]["audio.duration_seconds"]
    assert duration["control"] == pytest.approx(3.2)
    assert duration["directed"] == pytest.approx(8 * 6400 / RATE)
    assert duration["delta"] == pytest.approx(duration["directed"] - duration["control"])
    assert comparison["speed_ratio"]["value"] == pytest.approx(1.5, rel=1e-9)
    assert comparison["speed_ratio"]["sample_count"] == len(TEMPO_TEXT.split())
    assert comparison["machine_generated"] is True and comparison["human_verified"] is False


@pytest.mark.parametrize("substitutions, flagged", [(10, False), (11, True)])
def test_the_word_error_rate_boundary_sits_exactly_on_the_threshold(
        tmp_path, scratch, substitutions, flagged) -> None:
    control_path = write_wav(tmp_path / "control.wav", tone_bursts(8, 5760, 3840))
    directed_path = write_wav(tmp_path / "directed.wav", tone_bursts(12, 3840, 2560))
    control = measure(control_path, LONG_TEXT, scratch,
                      lexical_receipt=fake_receipt(control_path, LONG_TEXT, TOKENS))
    directed = measure(directed_path, LONG_TEXT, scratch,
                       lexical_receipt=substitution_receipt(directed_path, substitutions))
    assert directed["delivery"]["lexical"]["wer"]["value"] == pytest.approx(substitutions / 100)
    comparison = boundaries.compare_delivery_outcomes(control, directed)
    assert comparison["flags"]["wer_above_threshold"] is flagged
    assert comparison["flags"]["directed_missing_final_word"] is False
    assert comparison["flags"]["insufficient_evidence"] is False
    assert comparison["flags"]["for_listening"] is flagged


def test_a_directed_missing_final_word_flags_for_listening(tmp_path, scratch) -> None:
    control_path = write_wav(tmp_path / "control.wav", tone_bursts(8, 5760, 3840))
    directed_path = write_wav(tmp_path / "directed.wav", tone_bursts(12, 3840, 2560))
    text = "one two three four five six seven eight nine ten"
    tokens = text.split()
    control = measure(control_path, text, scratch,
                      lexical_receipt=fake_receipt(control_path, text, tokens))
    # The alignment stops one word early: the last expected word never appears.
    clipped = fake_receipt(directed_path, " ".join(tokens[:-1]), tokens[:-1])
    directed = measure(directed_path, text, scratch, lexical_receipt=clipped)
    assert directed["delivery"]["lexical"]["final_word"]["present"] is False
    assert directed["delivery"]["lexical"]["wer"]["value"] > 0.0
    comparison = boundaries.compare_delivery_outcomes(control, directed)
    assert comparison["flags"]["directed_missing_final_word"] is True
    assert comparison["flags"]["for_listening"] is True
    # It flags; it never rejects or retries.
    assert comparison["flags"]["rejects_take"] is False
    assert comparison["flags"]["triggers_retry"] is False
    assert comparison["flags"]["preferred_arm"] is None
    assert comparison["flags"]["grants_scope"] is False


def test_missing_measurements_stay_unknown_and_never_a_zero_delta(tempo) -> None:
    comparison = boundaries.compare_delivery_outcomes(tempo["control_record"],
                                                     tempo["directed_record"])
    unknown = comparison["flags"]["unknown_metrics"]
    assert "lexical.wer" in unknown
    assert "lexical.final_word.probability" in unknown
    for name in unknown:
        assert comparison["deltas"][name]["delta"] is None
        assert "unknown rather than zero" in comparison["deltas"][name]["reason"]
    assert comparison["flags"]["insufficient_evidence"] is True
    assert comparison["flags"]["for_listening"] is True
    assert comparison["flags"]["wer_above_threshold"] is None
    # A number that both arms measured is still reported honestly.
    assert comparison["deltas"]["pauses.count"]["delta"] is not None


def test_a_json_round_trip_still_verifies_and_compares(tempo) -> None:
    control = json.loads(json.dumps(tempo["control_record"]))
    directed = json.loads(json.dumps(tempo["directed_record"]))
    comparison = boundaries.compare_delivery_outcomes(control, directed)
    assert comparison["control"]["content_sha256"] == tempo["control_record"]["content_sha256"]
    assert comparison["speed_ratio"]["value"] == pytest.approx(1.5, rel=1e-9)


def test_a_same_text_mismatch_is_refused(tempo, scratch) -> None:
    other = measure(tempo["control"], "A different line entirely.", scratch)
    with pytest.raises(PerformanceDirectorError, match="same expected text"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], other)


def test_a_take_is_never_its_own_control(tempo, scratch) -> None:
    again = measure(tempo["control"], TEMPO_TEXT, scratch)
    with pytest.raises(PerformanceDirectorError, match="not its own control"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], again)


def test_tampered_or_foreign_records_are_refused(tempo) -> None:
    tampered = json.loads(json.dumps(tempo["directed_record"]))
    tampered["delivery"]["audio"]["duration_seconds"]["value"] = 1.0
    with pytest.raises(PerformanceDirectorError, match="does not verify"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], tampered)
    with pytest.raises(PerformanceDirectorError, match="delivery record"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], {"status": "measured"})
    stripped = json.loads(json.dumps(tempo["directed_record"]))
    stripped["delivery"].pop("edge_identity")
    stripped["content_sha256"] = boundaries.content_sha256(stripped)
    with pytest.raises(PerformanceDirectorError, match="edge identity"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], stripped)


def test_an_incompatible_measurement_method_is_refused(tempo) -> None:
    """A record produced by another measurement build is refused, not blended.

    The double is a record whose ``source`` names a different acoustic method with
    its own valid content hash, standing in for a record from an older build.
    """

    foreign = json.loads(json.dumps(tempo["directed_record"]))
    foreign["delivery"]["source"]["acoustics_method"] = "some-other-method-v9"
    foreign["content_sha256"] = boundaries.content_sha256(foreign)
    with pytest.raises(PerformanceDirectorError, match="compatible methods"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], foreign)
    narrower = json.loads(json.dumps(tempo["directed_record"]))
    narrower["delivery"]["source"]["acoustics_parameters"]["analysis_limit_seconds"] = 10.0
    narrower["content_sha256"] = boundaries.content_sha256(narrower)
    with pytest.raises(PerformanceDirectorError, match="compatible methods"):
        boundaries.compare_delivery_outcomes(tempo["control_record"], narrower)


# --------------------------------------------------------------------------- C. the read-only summary


def _fake_gate(table: dict):
    def gate(pilot_project_id, *, pair_id):
        return dict(table[pair_id], pair_id=pair_id)
    return gate


def test_every_pair_is_resolved_by_the_m3_gate(monkeypatch) -> None:
    calls: list[tuple] = []

    def gate(pilot_project_id, *, pair_id):
        calls.append((pilot_project_id, pair_id))
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": "no review has been recorded for this pair"}

    monkeypatch.setattr(pilot, "improvement_gate", gate)
    summary = boundaries.delivery_review_summary("pilot-x", ["pair-a", "pair-b"])
    assert calls == [("pilot-x", "pair-a"), ("pilot-x", "pair-b")]
    assert [row["reason"] for row in summary["pairs"]] == ["no review has been recorded for this pair"] * 2
    assert summary["aggregate"]["status"] == "review_pending"
    assert summary["requires_separate_stage_c_approval"] is True


@pytest.mark.parametrize("statuses, expected", [
    (["supported"], "supported"),
    (["supported", "review_pending"], "review_pending"),
    (["supported", "blocked"], "review_pending"),
    (["inconclusive"], "inconclusive"),
    (["not_improved"], "inconclusive"),
    (["supported", "inconclusive"], "inconclusive"),
])
def test_the_aggregate_is_never_vacuously_supported(monkeypatch, statuses, expected) -> None:
    table = {f"p{index}": {"status": status, "supports_improvement": status == "supported",
                           "reason": "synthetic gate result"}
             for index, status in enumerate(statuses)}
    monkeypatch.setattr(pilot, "improvement_gate", _fake_gate(table))
    summary = boundaries.delivery_review_summary("pilot-x", list(table))
    assert summary["aggregate"]["status"] == expected
    assert summary["aggregate"]["supports_improvement"] is (expected == "supported")
    assert summary["aggregate"]["pair_count"] == len(statuses)
    assert summary["human_verified"] is False


def test_a_supported_flag_without_the_owner_verdict_stays_pending(monkeypatch) -> None:
    """A gate that claims a status but not the owner's verdict is not support."""

    monkeypatch.setattr(pilot, "improvement_gate",
                        _fake_gate({"pair-a": {"status": "supported", "supports_improvement": False,
                                               "reason": "unverified"}}))
    summary = boundaries.delivery_review_summary("pilot-x", ["pair-a"])
    assert summary["aggregate"]["status"] == "review_pending"
    assert summary["aggregate"]["supports_improvement"] is False


def test_no_pair_is_never_supported() -> None:
    summary = boundaries.delivery_review_summary("pilot-x", [])
    assert summary["pairs"] == []
    assert summary["aggregate"]["status"] == "review_pending"
    assert summary["aggregate"]["supports_improvement"] is False
    assert "nothing can be supported" in summary["aggregate"]["reason"]
    assert summary["requires_separate_stage_c_approval"] is True


def test_malformed_requests_are_refused() -> None:
    with pytest.raises(PerformanceDirectorError, match="pair_ids"):
        boundaries.delivery_review_summary("pilot-x", "pair-a")
    with pytest.raises(PerformanceDirectorError, match="unique"):
        boundaries.delivery_review_summary("pilot-x", ["pair-a", "pair-a"])
    with pytest.raises(PerformanceDirectorError, match="pilot_project_id"):
        boundaries.delivery_review_summary("", ["pair-a"])
    with pytest.raises(PerformanceDirectorError, match="pilot_project_id"):
        boundaries.delivery_review_summary("../escape", ["pair-a"])


def test_a_real_pilot_with_unresolved_audio_stays_pending(studio) -> None:
    """The real M3 gate is read, not reproduced: no editions, so no support."""

    pilot.blind_pairs(studio["pilot_id"],
                      [{"pair_id": "pair-a", "baseline_audio_sha256": "b" * 64,
                        "variant_audio_sha256": "c" * 64}], seed=7)
    summary = boundaries.delivery_review_summary(studio["pilot_id"], ["pair-a"])
    assert summary["pairs"][0]["status"] == "review_pending"
    assert summary["pairs"][0]["supports_improvement"] is False
    assert summary["aggregate"]["status"] == "review_pending"
    assert summary["aggregate"]["supports_improvement"] is False


def test_an_unknown_pair_is_pending_and_the_owner_refusal_is_reported(studio) -> None:
    summary = boundaries.delivery_review_summary(studio["pilot_id"], ["pair-missing"])
    assert summary["pairs"][0]["status"] == "review_pending"
    assert "refused this pair" in summary["pairs"][0]["reason"]
    assert summary["aggregate"]["status"] == "review_pending"


def test_the_summary_reads_no_blind_key_and_writes_nothing(studio) -> None:
    pilot.blind_pairs(studio["pilot_id"],
                      [{"pair_id": "pair-a", "baseline_audio_sha256": "b" * 64,
                        "variant_audio_sha256": "c" * 64}], seed=7)
    root = pilot.safe_project_root(studio["pilot_id"])
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    summary = boundaries.delivery_review_summary(studio["pilot_id"], ["pair-a"])
    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert before == after
    assert summary["blind_key_read"] is False
    assert summary["records_written"] is False
    assert summary["grants_scope"] is False
    blob = json.dumps(summary)
    assert '"x_arm"' not in blob and '"y_arm"' not in blob and '"key"' not in blob
    assert "baseline" not in blob and "variant" not in blob
