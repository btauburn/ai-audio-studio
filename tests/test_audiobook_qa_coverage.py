"""Calibrated duration QA: real TTS audio passes, broken audio still fails.

The durations below are the measured lengths of the real Qwen3-TTS English
corpus rendered on an RTX 4070 (project 7fd2da08, 28 Takes, 12 passed and 16
blocked under the previous calibration).  Each case is rebuilt here as a WAV of
exactly that length, so the calibration stays testable without shipping audio.
"""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import qa
from tests.audiobook_helpers import make_silence_wav, make_tone_wav

# (label, characters, measured duration in ms) for 27 of the 28 real Takes.
REAL_SAMPLES = [
    ("four_speakers_01", 34, 1680),
    ("quiet_whisper_00", 40, 2080),
    ("emotion_range_00", 26, 1440),
    ("plain_narration_00", 93, 5360),
    ("difficult_names_01", 65, 3760),
    ("four_speakers_03", 46, 2720),
    ("chapter_boundary_00", 39, 2320),
    ("long_paragraph_00", 616, 37280),
    ("plain_narration_01", 104, 6320),
    ("chapter_boundary_01", 21, 1280),
    ("four_speakers_02", 34, 2160),
    ("two_person_dialogue_00", 21, 1360),
    ("two_person_dialogue_03", 35, 2560),
    ("quiet_whisper_02", 15, 1120),
    ("emotion_range_03", 46, 3520),
    ("difficult_names_00", 92, 7040),
    ("four_speakers_00", 32, 2720),
    ("emotion_range_01", 14, 1200),
    ("emotion_range_02", 42, 3680),
    ("dates_numbers_currency_01", 85, 7920),
    ("two_person_dialogue_02", 15, 1440),
    ("quiet_whisper_01", 12, 1200),
    ("dates_numbers_currency_00", 106, 11680),
    ("repeated_short_dialogue_02", 6, 720),
    ("two_person_dialogue_01", 20, 2560),
    ("repeated_short_dialogue_03", 6, 880),
    ("repeated_short_dialogue_01", 6, 1360),
]

# The 28th real Take: six characters of text that took 4.8 seconds of audio.
PATHOLOGICAL_SAMPLE = ("repeated_short_dialogue_00", 6, 4800)

FILLER = "the narrator reads this line aloud; "


def _text(characters: int) -> str:
    """Text of exactly ``characters`` length (QA only reads the length)."""

    text = (FILLER * (characters // len(FILLER) + 1))[:characters]
    return text if not text.endswith(" ") else text[:-1] + "x"


def _coverage(checks):
    return next(item for item in checks if item.check_id == "coverage_plausible")


def _wav_for(tmp_path, label: str, duration_ms: float):
    return make_tone_wav(tmp_path / f"{label}.wav", seconds=duration_ms / 1000.0)


def test_text_helper_is_exact():
    sizes = (1, 6, 15, 21, 33, 34, 35, 67, 93, 616)
    assert [len(_text(size)) for size in sizes] == list(sizes)
    # QA measures the stripped text, so the fixture must not end in whitespace.
    assert [len(_text(size).strip()) for size in sizes] == list(sizes)


@pytest.mark.parametrize("label, characters, duration_ms", REAL_SAMPLES)
def test_real_qwen_sample_durations_are_plausible(
    tmp_path, label, characters, duration_ms
):
    audio = _wav_for(tmp_path, label, duration_ms)
    checks = qa.run_tier1_checks(
        audio_path=audio, text=_text(characters), language="English"
    )
    coverage = _coverage(checks)
    assert coverage.status == "passed", f"{label}: {coverage.message}"


def test_a_pathological_short_render_is_still_rejected(tmp_path):
    """A six character line stretched to 4.8 s is a defect, not a reading speed."""

    label, characters, duration_ms = PATHOLOGICAL_SAMPLE
    audio = _wav_for(tmp_path, label, duration_ms)
    checks = qa.run_tier1_checks(
        audio_path=audio, text=_text(characters), language="English"
    )
    assert _coverage(checks).status == "failed"

    report = qa.build_report("take_pathological", qa.sha256_file(audio), checks)
    assert report.passed is False
    assert [item.check_id for item in report.blocking_failures()] == [
        "coverage_plausible"
    ]


def test_extremely_short_output_for_long_text_is_rejected(tmp_path):
    # 1.5 s of audio for 616 characters, i.e. 2.4 ms per character.
    audio = _wav_for(tmp_path, "stub", 1500)
    checks = qa.run_tier1_checks(
        audio_path=audio, text=_text(616), language="English"
    )
    assert _coverage(checks).status == "failed"


def test_severe_truncation_of_a_long_render_is_rejected(tmp_path):
    # The real long paragraph needs 37280 ms; keeping the first quarter of it
    # leaves 9320 ms, which is below 35% of the character budget.
    audio = _wav_for(tmp_path, "truncated", 9320)
    checks = qa.run_tier1_checks(
        audio_path=audio, text=_text(616), language="English"
    )
    assert _coverage(checks).status == "failed"


def test_silence_heavy_audio_at_a_plausible_duration_stays_caught(tmp_path):
    # A silent file of exactly the real plain_narration_00 length keeps a
    # plausible duration, so the silence check alone has to reject it.
    audio = make_silence_wav(tmp_path / "silent.wav", seconds=5.36)
    checks = qa.run_tier1_checks(
        audio_path=audio, text=_text(93), language="English"
    )
    assert _coverage(checks).status == "passed"
    assert next(
        item for item in checks if item.check_id == "not_silent"
    ).status == "failed"

    report = qa.build_report("take_silent", qa.sha256_file(audio), checks)
    assert report.passed is False
    assert [item.check_id for item in report.blocking_failures()] == ["not_silent"]


def test_language_resolves_to_a_duration_profile():
    assert qa.coverage_profile_for(None) is qa.ENGLISH_COVERAGE_PROFILE
    assert qa.coverage_profile_for("Auto") is qa.ENGLISH_COVERAGE_PROFILE
    assert qa.coverage_profile_for(" English ") is qa.ENGLISH_COVERAGE_PROFILE
    assert qa.coverage_profile_for("Chinese") is qa.CHINESE_COVERAGE_PROFILE
    assert qa.coverage_profile_for("zh-CN") is qa.CHINESE_COVERAGE_PROFILE


def test_a_line_too_fast_for_its_language_is_rejected(tmp_path):
    # 20 syllables in 0.9 s is too fast for Chinese but acceptable as 20 English
    # characters, so the profile -- not the audio -- decides the verdict.
    audio = _wav_for(tmp_path, "fast_chinese", 900)
    text = "汉" * 20

    chinese = qa.run_tier1_checks(audio_path=audio, text=text, language="Chinese")
    assert _coverage(chinese).status == "failed"

    english = qa.run_tier1_checks(audio_path=audio, text=text, language="English")
    assert _coverage(english).status == "passed"
