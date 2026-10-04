"""Offline regression tests for :mod:`service.voice_acoustics`.

Every input is synthesized locally with numpy + soundfile into ``tmp_path``:
no network access, no model weights and no GPU are involved.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import soundfile as sf

from service import voice_acoustics

SR = 16000


def _tone(freq: float, seconds: float = 2.0, sr: int = SR, amplitude: float = 0.35):
    times = np.arange(int(round(sr * seconds))) / sr
    return amplitude * np.sin(2.0 * np.pi * freq * times)


def _write(path, samples, sr: int = SR):
    sf.write(str(path), samples, sr)
    return path


def _burst_train(
    sr: int, period_ms: float, on_ms: float, seconds: float, freq: float = 200.0
):
    """Amplitude-gated tone: one burst per ``period_ms`` stands in for a syllable."""
    count = int(round(sr * seconds))
    carrier = 0.35 * np.sin(2.0 * np.pi * freq * np.arange(count) / sr)
    envelope = np.zeros(count)
    period = int(round(sr * period_ms / 1000.0))
    on = int(round(sr * on_ms / 1000.0))
    for start in range(0, count, period):
        envelope[start : start + on] = 1.0
    return carrier * envelope


# 1 --------------------------------------------------------------------------
def test_tone_pitch_estimates(tmp_path):
    measured = voice_acoustics.measure(_write(tmp_path / "t200.wav", _tone(200.0)))
    assert measured["f0_median_hz"] == pytest.approx(200.0, rel=0.08)
    assert measured["schema"] == voice_acoustics.SCHEMA
    assert measured["method"] == voice_acoustics.METHOD

    low = voice_acoustics.measure(_write(tmp_path / "t120.wav", _tone(120.0)))
    high = voice_acoustics.measure(_write(tmp_path / "t220.wav", _tone(220.0)))
    assert low["f0_median_hz"] < high["f0_median_hz"] * 0.8

    # the 22050 Hz path exercises the resample-to-16 kHz branch
    resampled = voice_acoustics.measure(
        _write(tmp_path / "t200_22k.wav", _tone(200.0, sr=22050), sr=22050)
    )
    assert resampled["sample_rate"] == 22050
    assert resampled["f0_median_hz"] == pytest.approx(200.0, rel=0.08)


# 2 --------------------------------------------------------------------------
def test_longer_pauses_raise_pause_metrics(tmp_path):
    tone = _tone(180.0, seconds=2.5)
    gapped = tone.copy()
    start = int(SR * 0.8)
    gapped[start : start + int(SR * 0.7)] = 0.0  # 700 ms of silence

    plain = voice_acoustics.measure(_write(tmp_path / "plain.wav", tone))
    paused = voice_acoustics.measure(_write(tmp_path / "paused.wav", gapped))

    assert paused["pause_fraction"] > plain["pause_fraction"]
    assert plain["pause_fraction"] == 0.0
    assert paused["mean_pause_ms"] is not None
    assert paused["mean_pause_ms"] >= 200.0
    assert (plain["mean_pause_ms"] or 0.0) < paused["mean_pause_ms"]


# 3 --------------------------------------------------------------------------
def test_faster_syllable_train_has_higher_word_rate(tmp_path):
    fast = _burst_train(SR, period_ms=200.0, on_ms=120.0, seconds=4.0)
    slow = _burst_train(SR, period_ms=500.0, on_ms=220.0, seconds=4.0)

    fast_profile = voice_acoustics.measure(_write(tmp_path / "fast.wav", fast))
    slow_profile = voice_acoustics.measure(_write(tmp_path / "slow.wav", slow))

    assert fast_profile["words_per_minute"] > slow_profile["words_per_minute"]
    assert fast_profile["syllables_per_minute"] == pytest.approx(
        fast_profile["words_per_minute"] * 1.5
    )


# 4 --------------------------------------------------------------------------
def test_white_noise_has_no_pitch_and_notes_why(tmp_path):
    noise = 0.2 * np.random.default_rng(20240928).standard_normal(SR * 2)
    profile = voice_acoustics.measure(_write(tmp_path / "noise.wav", noise))

    assert profile["f0_median_hz"] is None
    assert profile["notes"]
    assert "f0_median_hz" not in profile["measured"]


# 5 --------------------------------------------------------------------------
def test_degenerate_inputs_do_not_raise(tmp_path):
    silent = voice_acoustics.measure(_write(tmp_path / "silence.wav", np.zeros(SR)))
    assert silent["f0_median_hz"] is None
    assert silent["brightness_hz"] is None
    assert silent["rasp"] is None
    assert silent["notes"]

    single = voice_acoustics.measure(_write(tmp_path / "one.wav", np.zeros(1)))
    assert single["f0_median_hz"] is None
    assert single["voiced_fraction"] is None
    assert any("frame" in note for note in single["notes"])

    broken = tmp_path / "not_audio.txt"
    broken.write_bytes(b"this is definitely not a wave file")
    assert issubclass(voice_acoustics.VoiceAcousticsError, ValueError)
    with pytest.raises(voice_acoustics.VoiceAcousticsError):
        voice_acoustics.measure(broken)


# 6 --------------------------------------------------------------------------
def test_measure_is_reproducible(tmp_path):
    path = _write(tmp_path / "repeat.wav", _tone(150.0))
    assert voice_acoustics.measure(path) == voice_acoustics.measure(path)


# 7 --------------------------------------------------------------------------
def test_shape_rate_changes_duration(tmp_path):
    source = _write(tmp_path / "src.wav", _tone(200.0, seconds=3.0))

    faster = voice_acoustics.shape(source, tmp_path / "fast.wav", rate=1.15)
    slower = voice_acoustics.shape(source, tmp_path / "slow.wav", rate=0.9)

    # rate is a speed factor, so duration == source / rate; 1.15 is therefore
    # ~13% shorter (comfortably more than the 3% sanity bound) and 0.9 is
    # ~11% longer.
    assert faster["output_seconds"] <= faster["source_seconds"] * 0.97
    assert slower["output_seconds"] >= slower["source_seconds"] * 1.03
    for result, rate in ((faster, 1.15), (slower, 0.9)):
        ideal = result["source_seconds"] / rate
        assert abs(result["output_seconds"] - ideal) / ideal <= 0.02
        assert result["changed"] is True


# 8 --------------------------------------------------------------------------
def test_shape_pitch_moves_pitch_not_duration(tmp_path):
    source = _write(tmp_path / "pitch_src.wav", _tone(200.0, seconds=3.0))
    before = voice_acoustics.measure(source)["f0_median_hz"]

    result = voice_acoustics.shape(source, tmp_path / "pitch_up.wav", semitones=2.0)
    after = voice_acoustics.measure(tmp_path / "pitch_up.wav")["f0_median_hz"]

    ideal = 2.0 ** (2.0 / 12.0)
    assert after > before
    assert after / before == pytest.approx(ideal, rel=0.06)
    assert abs(result["output_seconds"] - result["source_seconds"]) / result["source_seconds"] <= 0.03


# 9 --------------------------------------------------------------------------
def test_plan_shaping_clamps_and_reports(tmp_path):
    plan = voice_acoustics.plan_shaping(
        {"f0_median_hz": 100.0, "words_per_minute": 150.0},
        {"f0_median_hz": 200.0, "words_per_minute": 150.0},
        max_semitones=3.0,
        rate_range=(0.85, 1.18),
    )
    assert plan["requested"]["semitones"] == pytest.approx(-12.0)
    assert plan["semitones"] == -3.0
    assert plan["clamped"] is True
    assert plan["applied_to"] == ["pitch"]
    assert plan["reasons"]

    rate_plan = voice_acoustics.plan_shaping(
        {"f0_median_hz": 150.0, "words_per_minute": 600.0},
        {"f0_median_hz": 150.0, "words_per_minute": 50.0},
    )
    assert rate_plan["rate"] == 1.18
    assert rate_plan["clamped"] is True

    equal = voice_acoustics.plan_shaping(
        {"f0_median_hz": 150.0, "words_per_minute": 140.0},
        {"f0_median_hz": 150.0, "words_per_minute": 140.0},
    )
    assert equal["semitones"] == 0.0
    assert equal["rate"] == 1.0
    assert equal["clamped"] is False
    assert equal["applied_to"] == []

    missing = voice_acoustics.plan_shaping({}, {})
    assert missing["semitones"] == 0.0
    assert missing["rate"] == 1.0
    assert len(missing["reasons"]) == 2


# 10 -------------------------------------------------------------------------
def test_parameters_and_instruction_fragment_are_numeric_and_anonymous(tmp_path):
    profile = voice_acoustics.measure(_write(tmp_path / "named.wav", _tone(200.0)))
    params = voice_acoustics.parameters(profile)
    fragment = voice_acoustics.instruction_fragment(profile)

    assert params["pitch_target_hz"] == pytest.approx(200.0, rel=0.08)
    assert str(int(round(profile["f0_median_hz"]))) in fragment
    assert fragment.startswith("Measured acoustic targets:")
    assert "Narrator Jenkins" not in fragment
    assert voice_acoustics.instruction_fragment({}) == ""
    assert (
        voice_acoustics.parameters(voice_acoustics.parameters(profile))["pitch_target_hz"]
        == params["pitch_target_hz"]
    )


# 11 -------------------------------------------------------------------------
def test_merge_does_not_mutate_and_differentiates(tmp_path):
    profile = voice_acoustics.measure(
        _write(tmp_path / "merge_src.wav", _burst_train(SR, 500.0, 220.0, 4.0))
    )
    snapshot = copy.deepcopy(profile)

    lower = voice_acoustics.merge(profile, {"pitch_semitones": -2.0, "rate": 0.9})
    higher = voice_acoustics.merge(profile, {"pitch_semitones": 2.0, "rate": 1.1})

    assert profile == snapshot
    assert lower["bias"] == {"pitch_semitones": -2.0, "rate": 0.9}

    low_params = voice_acoustics.parameters(lower)
    high_params = voice_acoustics.parameters(higher)
    assert low_params["pitch_target_hz"] < high_params["pitch_target_hz"]
    assert low_params["rate_words_per_minute"] < high_params["rate_words_per_minute"]

    # Anything the bias names but the profile lacks is reported, not invented.
    thin = voice_acoustics.merge({"schema": voice_acoustics.SCHEMA, "notes": []},
                                 {"breathiness": 0.2})
    assert any("breathiness" in note for note in thin["notes"])


def test_a_pause_bias_scales_pause_behaviour_without_destroying_it(tmp_path):
    """A pause bias is a multiplier on pauses, not a ratio to be clamped to 0-1."""
    profile = voice_acoustics.measure(
        _write(tmp_path / "pause_bias.wav", _burst_train(SR, 500.0, 200.0, 4.0))
    )
    assert profile["mean_pause_ms"] >= 200.0
    longer = voice_acoustics.merge(profile, {"pause_ms": 1.15})
    shorter = voice_acoustics.merge(profile, {"pause_ms": 0.85})
    assert longer["mean_pause_ms"] > profile["mean_pause_ms"] > shorter["mean_pause_ms"]
    assert longer["mean_pause_ms"] > 200.0, "a millisecond value must not be clamped to a ratio"
    assert longer["pause_count_per_minute"] < profile["pause_count_per_minute"], \
        "longer pauses mean fewer pauses per minute"
    # The derived generation parameters carry the scaled pause through.
    assert voice_acoustics.parameters(longer)["pause_ms"] == pytest.approx(longer["mean_pause_ms"])
    assert voice_acoustics.parameters(longer)["pause_ms"] != 1.0


def test_a_stretched_take_reports_the_rate_it_was_stretched_to(tmp_path):
    """A time-stretch is the signal syllable detection reads, so rate is derived.

    Envelope-peak detection counts the stretch's own segment duplication as new
    syllables, so a take that was slowed by the planned factor measures *faster*
    than it started. The rate metrics are therefore taken from the pre-stretch
    measurement and the applied factor, and the substitution is recorded.
    """
    source = _write(tmp_path / "stretch_src.wav", _burst_train(SR, 400.0, 210.0, 4.0))
    before = voice_acoustics.measure(source)
    shaped = tmp_path / "stretch_out.wav"
    shaping = voice_acoustics.shape(source, shaped, rate=0.90)
    raw = voice_acoustics.measure(shaped)
    profile, reconciliation = voice_acoustics.reconcile_after(before, raw, shaping)

    assert shaping["output_seconds"] > shaping["source_seconds"], "rate < 1 must lengthen the take"
    assert reconciliation["rate_agrees"] is True
    assert reconciliation["derived_keys"] == ["syllables_per_minute", "words_per_minute"]
    expected = before["words_per_minute"] * 0.90
    assert profile["words_per_minute"] == pytest.approx(expected)
    assert reconciliation["replaced"]["words_per_minute"] == {
        "measured": raw["words_per_minute"], "derived": pytest.approx(expected)}
    # The raw measurement is always retained beside the derived value, even where
    # this particular signal happens to agree with it; on real speech it does not
    # (recorded in the acceptance report).
    assert reconciliation["replaced"]["words_per_minute"]["measured"] == raw["words_per_minute"]
    assert any("pre-stretch measurement" in note for note in profile["notes"])
    # Per-frame measurements the stretch does not invalidate stay as measured.
    assert profile["f0_median_hz"] == pytest.approx(raw["f0_median_hz"])


def test_reachability_separates_shortfall_from_out_of_scope_and_from_no_measurement():
    """A target key no actuator can move is neither a reach nor a shortfall.

    A documented acoustic prior carries keys (peak level, voiced fraction) that no
    pitch/rate transform touches. Counting them as unreached would report the
    actuator's scope as this run's failure; counting them as reached would report
    an unmeasured target as achieved. Both are avoided by keeping three outcomes.
    """
    mixed = voice_acoustics.achievement({
        "exceeded": ["f0_median_hz", "words_per_minute", "peak_dbfs", "voiced_fraction"],
        "compared": ["f0_median_hz", "f0_mean_hz", "words_per_minute", "peak_dbfs", "voiced_fraction"],
    })
    assert mixed["verdict"] == "shortfall"
    assert mixed["shortfall"] == ["f0_median_hz", "words_per_minute"]
    assert mixed["out_of_scope"] == ["peak_dbfs", "voiced_fraction"]
    assert mixed["reached"] == ["f0_mean_hz"]
    assert mixed["reached_target"] is False

    reached = voice_acoustics.achievement({"exceeded": [], "compared": ["f0_median_hz", "words_per_minute"]})
    assert reached["verdict"] == "reached" and reached["reached_target"] is True

    for empty in ({"exceeded": ["peak_dbfs"], "compared": ["peak_dbfs"]},
                  {"exceeded": [], "compared": []}):
        unmeasured = voice_acoustics.achievement(empty)
        assert unmeasured["verdict"] == "could_not_measure"
        assert unmeasured["measurable"] is False
        assert unmeasured["reached_target"] is False, "an empty population is never a pass"

    # Every key the shaping pass can move is in the controllable set.
    assert set(["f0_median_hz", "words_per_minute", "pause_count_per_minute"]) <= set(
        voice_acoustics.CONTROLLABLE_KEYS)


# 12 -------------------------------------------------------------------------
def test_compare_flags_differences(tmp_path):
    profile = voice_acoustics.measure(_write(tmp_path / "cmp200.wav", _tone(200.0)))
    other = voice_acoustics.measure(_write(tmp_path / "cmp120.wav", _tone(120.0)))

    different = voice_acoustics.compare(profile, other)
    assert different["exceeded"]
    assert "f0_median_hz" in different["exceeded"]
    assert different["within_tolerance"] is False
    assert different["compared"] == sorted(different["deltas"])
    assert "f0_median_hz" in different["relative"]

    same = voice_acoustics.compare(profile, dict(profile))
    assert same["exceeded"] == []
    assert same["within_tolerance"] is True
    assert all(delta == 0.0 for delta in same["deltas"].values())
