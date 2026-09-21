"""Quality checks for Takes, in two explicit tiers (Phase 11 / Phase 12).

Tier 1 is deterministic and always available (no models, no GPU): file
existence, decodability, non-zero duration, hash integrity, duration anomaly,
silence anomaly, clipping/peak and text-to-audio coverage.

Tier 2 is optional and model-backed (ASR completeness).  When Tier 2 cannot
run, the report says so explicitly (``status="skipped"`` with a reason) instead
of pretending the audio passed.
"""

from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from dubbing_service.app.models.audiobook import (
    QaCheckResult,
    QaCoverageProfile,
    TakeQaReport,
)

QA_CHECK_VERSION = "tier1-v1"

# Version of the duration-window implementation itself.  A report records it so
# that a re-audit can tell "this Take was judged English" from "this Take was
# judged by the English window as it existed then".
COVERAGE_PROFILE_VERSION = "coverage-v1"

DEFAULT_ANOMALY_RATIO = 2.5
DEFAULT_SILENCE_RATIO = 0.98
DEFAULT_PEAK_CEILING = 0.999
DEFAULT_MIN_RMS = 1e-4

# Duration plausibility, measured on a real Qwen3-TTS English corpus rendered on
# an RTX 4070 (28 Takes, project 7fd2da08: 12 passed / 16 blocked under the
# previous values).  Cost per character there was 49.4 ms at the p10, 75.6 ms at
# the median and 155 ms at the p90, with long-form narration clustered at
# 57-61 ms/char.  The former 12.0 ms/char placeholder sat ~5x below real English
# speech, which is what made ordinary Takes look "implausible".  The window
# answers one question only -- is this duration implausible for this text -- so
# it stays far wider than the natural spread of reading speeds.
DEFAULT_COVERAGE_PER_CHAR_MS = 60.0
DEFAULT_COVERAGE_RATIO = 0.35
DEFAULT_MAX_COVERAGE_RATIO = 6.0
# Fixed lead-in/trail-out padding dominates a short utterance, so the long side
# carries an additive allowance; the short side carries an absolute floor for
# text too brief for a per-character ratio to mean anything.
DEFAULT_COVERAGE_OVERHEAD_MS = 1200.0
DEFAULT_MIN_COVERAGE_MS = 120.0

BLOCKING_CHECK_IDS = frozenset(
    {
        "file_exists",
        "decodable",
        "duration_positive",
        "hash_match",
        "not_silent",
        "peak_ok",
        "coverage_plausible",
    }
)


@dataclass(frozen=True)
class CoverageProfile:
    """Accepted duration window for one spoken language.

    ``per_char_ms`` is the measured cost of one character of text; the window is
    ``min_ratio``-``max_ratio`` of it, floored by an absolute minimum duration
    and extended by a fixed overhead allowance on the long side.
    """

    name: str
    per_char_ms: float
    min_ratio: float = DEFAULT_COVERAGE_RATIO
    max_ratio: float = DEFAULT_MAX_COVERAGE_RATIO
    overhead_ms: float = DEFAULT_COVERAGE_OVERHEAD_MS
    min_duration_ms: float = DEFAULT_MIN_COVERAGE_MS

    def bounds(self, characters: int) -> tuple[float, float]:
        """Lowest and highest plausible duration (ms) for ``characters`` glyphs."""

        expected_ms = characters * self.per_char_ms
        low = max(self.min_duration_ms, expected_ms * self.min_ratio)
        high = expected_ms * self.max_ratio + self.overhead_ms
        return low, high


# English is the only language with measured evidence.  A Chinese character is a
# syllable, and a syllable is worth roughly three Latin characters, so the
# Chinese cost is derived from the same measurement (~200 ms) rather than
# invented separately.
ENGLISH_COVERAGE_PROFILE = CoverageProfile("english", DEFAULT_COVERAGE_PER_CHAR_MS)
CHINESE_COVERAGE_PROFILE = CoverageProfile("chinese", 200.0)
DEFAULT_COVERAGE_PROFILE = ENGLISH_COVERAGE_PROFILE

_CHINESE_LANGUAGE_TOKENS = frozenset(
    {"chinese", "cmn", "mandarin", "zh", "zh-cn", "zh-hans", "zh-hant"}
)


def coverage_profile_for(language: Optional[str]) -> CoverageProfile:
    """Resolve the duration profile for a declared language.

    Text that declares no language, or declares ``Auto`` (which the engine only
    resolves at synthesis time), falls back to the measured English profile.
    That direction is deliberate: a slower, syllable-per-character language lands
    inside the English window anyway, whereas resolving English text as Chinese
    would recreate the false "too short" verdicts this window exists to prevent.
    """

    token = (language or "").strip().lower()
    if token in _CHINESE_LANGUAGE_TOKENS or token.startswith("zh-"):
        return CHINESE_COVERAGE_PROFILE
    return DEFAULT_COVERAGE_PROFILE


@dataclass
class AudioFacts:
    duration_ms: int
    peak: float
    rms: float
    silent_ratio: float
    sample_rate: int
    channels: int
    decodable: bool


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffprobe_duration_ms(path: Path) -> Optional[int]:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    try:
        return int(float(probe.stdout.strip()) * 1000)
    except (TypeError, ValueError):
        return None


def _pcm_facts(path: Path) -> Optional[AudioFacts]:
    """Exact peak/RMS/silence for uncompressed PCM WAV; ``None`` otherwise."""

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, OSError, EOFError):
        return None
    if width not in (1, 2, 3, 4) or not raw:
        return None

    if width == 1:
        values = [(byte - 128) / 128.0 for byte in raw]
    else:
        count = len(raw) // width
        try:
            if width == 2:
                values = [
                    item / 32768.0 for item in struct.unpack(f"<{count}h", raw[: count * 2])
                ]
            elif width == 4:
                values = [
                    item / 2147483648.0
                    for item in struct.unpack(f"<{count}i", raw[: count * 4])
                ]
            else:
                values = []
                for index in range(count):
                    chunk = raw[index * 3 : index * 3 + 3]
                    value = int.from_bytes(chunk + b"\x00", "little", signed=True)
                    values.append(value / 8388608.0)
        except struct.error:
            return None

    if not values:
        return None
    peak = max(abs(value) for value in values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    silent = sum(1 for value in values if abs(value) < 1e-4) / len(values)
    duration_ms = int((frames / rate) * 1000) if rate else 0
    return AudioFacts(
        duration_ms=duration_ms,
        peak=peak,
        rms=rms,
        silent_ratio=silent,
        sample_rate=rate,
        channels=channels,
        decodable=True,
    )


def read_audio_facts(path: Path | str) -> Optional[AudioFacts]:
    target = Path(path)
    if not target.is_file():
        return None
    facts = _pcm_facts(target)
    if facts is not None:
        return facts
    duration = _ffprobe_duration_ms(target)
    if duration is None:
        return None
    return AudioFacts(
        duration_ms=duration,
        peak=0.0,
        rms=0.0,
        silent_ratio=0.0,
        sample_rate=0,
        channels=0,
        decodable=True,
    )


_STATUS_ALIASES = {
    "pass": "passed",
    "fail": "failed",
    "warn": "passed",
    "not_run": "skipped",
}


def _check(
    check_id: str,
    status: str,
    *,
    detail: str = "",
    measured: Optional[float] = None,
    limit: Optional[float] = None,
    tier: int = 1,
) -> QaCheckResult:
    details: dict = {}
    if measured is not None:
        details["measured"] = measured
    if limit is not None:
        details["limit"] = limit
    blocking = check_id in BLOCKING_CHECK_IDS and tier == 1
    if status == "warn":
        # Advisory by construction: a warning never blocks a release.
        blocking = False
    return QaCheckResult(
        check_id=check_id,
        check_version=QA_CHECK_VERSION,
        tier=tier,
        status=_STATUS_ALIASES.get(status, status),  # type: ignore[arg-type]
        message=detail or None,
        details=details,
        blocking=blocking,
    )


def run_tier1_checks(
    *,
    audio_path: Path | str,
    expected_sha256: Optional[str] = None,
    text: str = "",
    expected_duration_ms: Optional[float] = None,
    anomaly_ratio: float = DEFAULT_ANOMALY_RATIO,
    silence_ratio: float = DEFAULT_SILENCE_RATIO,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
    min_rms: float = DEFAULT_MIN_RMS,
    language: Optional[str] = None,
) -> List[QaCheckResult]:
    results: List[QaCheckResult] = []
    target = Path(audio_path)

    if not target.is_file():
        results.append(
            _check("file_exists", "fail", detail=f"音频文件不存在: {target}")
        )
        return results
    results.append(_check("file_exists", "pass"))

    facts = read_audio_facts(target)
    if facts is None or not facts.decodable:
        results.append(
            _check("decodable", "fail", detail="文件无法解码为音频（可能是 0 字节或损坏）")
        )
        return results
    results.append(_check("decodable", "pass"))

    if facts.duration_ms <= 0:
        results.append(
            _check(
                "duration_positive",
                "fail",
                detail="音频时长不是正数",
                measured=0.0,
                limit=1.0,
            )
        )
    else:
        results.append(_check("duration_positive", "pass", measured=float(facts.duration_ms)))

    if expected_sha256:
        actual = sha256_file(target)
        if actual == expected_sha256:
            results.append(_check("hash_match", "pass"))
        else:
            results.append(
                _check(
                    "hash_match",
                    "fail",
                    detail="音频内容与登记的 sha256 不一致",
                )
            )

    if expected_duration_ms and facts.duration_ms > 0:
        low = expected_duration_ms / anomaly_ratio
        high = expected_duration_ms * anomaly_ratio
        status = "pass" if low <= facts.duration_ms <= high else "fail"
        results.append(
            _check(
                "duration_anomaly",
                status,
                detail=f"本次 {facts.duration_ms} ms，预期约 {int(expected_duration_ms)} ms",
                measured=float(facts.duration_ms),
                limit=float(int(high)),
            )
        )

    if facts.sample_rate:
        if facts.silent_ratio >= silence_ratio:
            results.append(
                _check(
                    "not_silent",
                    "fail",
                    detail=f"{facts.silent_ratio:.2%} 采样接近静音",
                    measured=facts.silent_ratio,
                    limit=silence_ratio,
                )
            )
        else:
            results.append(_check("not_silent", "pass", measured=facts.silent_ratio))

        if facts.peak > peak_ceiling:
            results.append(
                _check(
                    "peak_ok",
                    "fail",
                    detail=f"峰值 {facts.peak:.4f} 存在削波风险",
                    measured=facts.peak,
                    limit=peak_ceiling,
                )
            )
        else:
            results.append(_check("peak_ok", "pass", measured=facts.peak))

        if facts.rms < min_rms:
            results.append(
                _check(
                    "rms_ok",
                    "warn",
                    detail=f"整体音量极低（RMS {facts.rms:.6f}）",
                    measured=facts.rms,
                    limit=min_rms,
                )
            )
        else:
            results.append(_check("rms_ok", "pass", measured=facts.rms))

    if text.strip():
        characters = len(text.strip())
        profile = coverage_profile_for(language)
        low_ms, high_ms = profile.bounds(characters)
        duration_ms = float(facts.duration_ms)
        ratio = duration_ms / max(1.0, characters * profile.per_char_ms)
        if low_ms <= duration_ms <= high_ms:
            results.append(_check("coverage_plausible", "pass", measured=ratio))
        else:
            results.append(
                _check(
                    "coverage_plausible",
                    "fail",
                    detail=(
                        f"音频长度与文本规模不匹配（{facts.duration_ms} ms / "
                        f"{characters} 字，允许 {int(low_ms)}-{int(high_ms)} ms，"
                        f"{profile.name}）"
                    ),
                    measured=ratio,
                )
            )

    return results


def run_tier1_checks_for_chunks(
    audio_path: Path | str,
    *,
    chunk_texts: Sequence[str],
    expected_sha256: Optional[str] = None,
    expected_duration_ms: Optional[float] = None,
    language: Optional[str] = None,
) -> List[QaCheckResult]:
    """Chunked renders are QA'd as one Take, using the concatenated text."""

    return run_tier1_checks(
        audio_path=audio_path,
        expected_sha256=expected_sha256,
        text="".join(chunk_texts),
        expected_duration_ms=expected_duration_ms,
        language=language,
    )


def run_asr_completeness(
    audio_path: Path | str,
    expected_text: str,
    *,
    model_size: str = "tiny",
    language: Optional[str] = None,
    device: str = "cpu",
) -> List[QaCheckResult]:
    """Tier-2 ASR completeness.  Reports ``skipped`` when unavailable."""

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return [
            _check(
                "asr_completeness",
                "not_run",
                detail=f"faster-whisper 不可用: {exc}",
                tier=2,
            )
        ]

    try:
        model = WhisperModel(model_size, device=device, compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), language=language)
        transcript = "".join(segment.text for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001 - model/runtime failure
        return [
            _check(
                "asr_completeness",
                "not_run",
                detail=f"ASR 运行失败: {exc}",
                tier=2,
            )
        ]

    ratio = similarity_ratio(expected_text, transcript)
    return [
        _check(
            "asr_completeness",
            "pass" if ratio >= 0.6 else "warn",
            detail=f"ASR 覆盖率 {ratio:.2%}: {transcript[:80]}",
            measured=ratio,
            limit=0.6,
            tier=2,
        )
    ]


def _normalized_characters(text: str) -> str:
    return "".join(character for character in text if not character.isspace()).lower()


def similarity_ratio(expected: str, actual: str) -> float:
    """Character-level similarity (1.0 = identical)."""

    left = _normalized_characters(expected)
    right = _normalized_characters(actual)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return 1.0 - (previous[-1] / max(len(left), len(right)))


def coverage_profile_identity(language: Optional[str]) -> QaCoverageProfile:
    """The persisted identity of the window ``coverage_plausible`` will apply."""

    profile = coverage_profile_for(language)
    return QaCoverageProfile(
        language=language,
        profile=profile.name,
        profile_version=COVERAGE_PROFILE_VERSION,
    )


def build_report(
    take_id: str,
    output_audio_hash: Optional[str],
    checks: Iterable[QaCheckResult],
    *,
    tier: int = 1,
    notes: Optional[str] = None,
    coverage_profile: Optional[QaCoverageProfile] = None,
) -> TakeQaReport:
    """Bind the check results to the exact bytes that were evaluated."""

    results = list(checks)
    report = TakeQaReport(
        take_id=take_id,
        output_audio_hash=output_audio_hash or "",
        qa_check_version=QA_CHECK_VERSION,
        tier=tier,
        results=results,
        notes=notes,
        coverage_profile=coverage_profile,
    )
    executed = [item for item in results if item.status != "skipped"]
    report.passed = bool(executed) and not report.blocking_failures()
    return report


def qa_take_audio(
    *,
    take_id: str,
    audio_path: Path | str,
    text: str,
    expected_sha256: Optional[str] = None,
    expected_duration_ms: Optional[float] = None,
    notes: Optional[str] = None,
    tier2_text: Optional[str] = None,
    tier2_model: str = "tiny",
    tier2_language: Optional[str] = None,
    language: Optional[str] = None,
) -> TakeQaReport:
    """Run Tier 1 (and Tier 2 when requested) and bind the report to the audio.

    ``language`` is the language of the spoken text and selects the duration
    window; ``tier2_language`` is the recognizer's language and is unrelated.
    The window that governed ``coverage_plausible`` is recorded on the report as
    an identity, so a later re-audit can prove which profile was applied.
    """

    checks = run_tier1_checks(
        audio_path=audio_path,
        expected_sha256=expected_sha256,
        text=text,
        expected_duration_ms=expected_duration_ms,
        language=language,
    )
    sha = expected_sha256
    target = Path(audio_path)
    if target.is_file():
        sha = sha or sha256_file(target)

    report_tier = 1
    if tier2_text is not None:
        checks.extend(
            run_asr_completeness(
                audio_path,
                tier2_text,
                model_size=tier2_model,
                language=tier2_language,
            )
        )
        report_tier = 2

    return build_report(
        take_id,
        sha,
        checks,
        tier=report_tier,
        notes=notes,
        coverage_profile=coverage_profile_identity(language),
    )


def reaudit_take(
    *,
    take_id: str,
    audio_path: Path | str,
    text: str,
    language: Optional[str],
    expected_sha256: Optional[str] = None,
    notes: Optional[str] = None,
    tier2_text: Optional[str] = None,
    tier2_model: str = "tiny",
    tier2_language: Optional[str] = None,
) -> TakeQaReport:
    """Re-run Tier 1 for an existing Take from *its own* evidence.

    ``language`` must be the value persisted on the Take, never the current role
    or segment setting: a role edited after the audio was generated would
    otherwise silently re-interpret history under a different duration window.

    ``expected_sha256`` should likewise be the Take's recorded ``audio_sha256``,
    which is what makes the new report comparable with the original one.
    """

    return qa_take_audio(
        take_id=take_id,
        audio_path=audio_path,
        text=text,
        expected_sha256=expected_sha256,
        notes=notes,
        tier2_text=tier2_text,
        tier2_model=tier2_model,
        tier2_language=tier2_language,
        language=language,
    )
