"""CPU edge measurement, boundary classification and pure timeline compilation (M2).

M2 is additive, deterministic and side-effect free.  Nothing here reads a
project, writes a file, calls a service, loads a model or uses a GPU.  The
module measures the edges of one already-rendered PCM Take, classifies the
boundary between two adjacent source lines, and compiles an integer-frame
timing plan.  It never assembles audio, never changes a Take and never spends
a synthesis attempt: a timing plan is arithmetic, not synthesis.

Every refusal is a :class:`PerformanceDirectorError` carrying one of the M1
hold codes.  Unknown schema versions, corrupt records and unusable inputs
never silently become valid state.
"""

from __future__ import annotations

import hashlib
import math
import re
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dubbing_service.app.audiobook import qa
from dubbing_service.app.audiobook.fingerprint import canonical_json_bytes, sha256_bytes
from dubbing_service.app.audiobook.performance_director import (
    BOUNDARY_HINTS,
    PerformanceDirectorError,
    content_sha256,
)

EDGE_MEASUREMENT_SCHEMA = "edge-measurement-v1"
BOUNDARY_CLASSIFICATION_SCHEMA = "boundary-classification-v1"
BOUNDARY_TIMELINE_SCHEMA = "boundary-timeline-v1"
BOUNDARY_POLICY_SCHEMA = "boundary-policy-v1"

DETECTOR_VERSION = "pd-edge-1"
CLASSIFIER_VERSION = "pd-boundary-1"
ASSEMBLER_VERSION = "pd-assemble-1"

#: Explicit detector parameters.  A new value makes a new detector version.
FRAME_MS = 10
CONTENT_FLOOR_DBFS = -50.0
SILENCE_CEILING_DBFS = -70.0
_SILENCE_FLOOR_LINEAR = 1e-12

#: Descriptor guard: the frames nearest the content onset are never removed.
LEAD_OFFSET_GUARD_MS = 40
#: Declared scene-separator pause, unchanged and never padded twice.
SCENE_SEPARATOR_MS = 1250

#: Hint arithmetic from section 9.2: floor, cap and additive adjustments.
HINT_FLOOR_MS = 80.0
HINT_CAP_MS = 1500.0
HINT_ADD_MS = {"quick": None, "hesitant": 300.0, "withheld": 600.0, "beat_after": 350.0}

#: Measurement record states.  Only ``measured`` can ever be shortened.
MEASUREMENT_STATES = ("measured", "all_silent", "uncertain")
EDGE_UNCERTAIN = "edge_uncertain"
ALL_SILENT_FLAG = "all_silent"
SHORTENING_BLOCKING_FLAGS = (EDGE_UNCERTAIN, ALL_SILENT_FLAG)

SCENE_SEPARATOR = "SCENE_SEPARATOR"
UNKNOWN_CLASS = "UNKNOWN"
BOUNDARY_CLASSES = (
    "D_TAG", "TAG_D_CONTINUATION", "N_D_SAME_PARA", "N_D_NEW_PARA", "D_N_BEAT_SAME_PARA",
    "D_N_NEW_PARA", "D_D_REPLY", "D_D_INTERRUPTION", "D_D_AFTER_TRAILOFF", "D_D_SAME_SPEAKER",
    "N_N_SAME_PARA", "N_N_NEW_PARA", SCENE_SEPARATOR, UNKNOWN_CLASS,
)
POLICY_TARGET_CLASSES = tuple(name for name in BOUNDARY_CLASSES[:12])
BOUNDARY_STATES = ("met", "limited_by_existing_audio", "protected_uncertain_edge", "baseline")

#: Classes whose "needs lead-in offset" column in section 9.2 is Yes or
#: "only for outliers".  The outlier bound is the arithmetic itself: an offset
#: never removes more than the excess over the target.
OFFSET_ELIGIBLE_CLASSES = ("TAG_D_CONTINUATION", "N_D_SAME_PARA", "N_D_NEW_PARA",
                           "D_D_REPLY", "D_D_INTERRUPTION")

NARRATOR = "narrator"
CHARACTER = "character"

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,200}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_TAG_RE = re.compile(
    r"(?i)^[^a-z0-9]*"
    r"(?:(?:he|she|they|it|the\s+\w+|[A-Z][\w'’-]*)\s+)?"
    r"(?:said|says|asked|asks|replied|replies|answered|whispered|murmured|muttered|shouted|"
    r"snapped|growled|sighed|laughed|breathed|added|continued|began|repeated|called|told)\b"
)
_DASH_ENDINGS = ("-", "\u2014", "\u2013")
_ELLIPSIS_ENDINGS = ("\u2026", "...")

EDGE_MEASUREMENT_LIMITATION = (
    "energy edges are not lexical word boundaries; no aligner or ASR runs in this detector"
)
TIMELINE_ELIGIBILITY_LIMITATION = (
    "this compiler reads no project, no audio bytes and no QA report, so it does not and cannot "
    "establish project eligibility; a caller's file presence or qa_passed flag is not proof and is "
    "ignored here.  Live eligibility belongs to the assembler."
)
TIMELINE_FINGERPRINT_LIMITATION = (
    "a timing plan never changes a Take fingerprint and never consumes a synthesis attempt"
)


# --------------------------------------------------------------------------- helpers


def _require_sha256(value: Any, what: str, code: str = "input_ineligible") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PerformanceDirectorError(code, f"{what} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value) or ".." in value:
        raise PerformanceDirectorError(
            "input_ineligible", f"{what} must be 1-200 characters without control characters or path escapes")
    return value


def _require_finite(value: Any, what: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PerformanceDirectorError("input_ineligible", f"{what} must be a finite number")
    number = float(value)
    if number < minimum:
        raise PerformanceDirectorError("input_ineligible", f"{what} must be at least {minimum}")
    return number


def _require_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerformanceDirectorError("input_ineligible", f"{what} must be a non-negative integer")
    return value


def sample_domain(sample_rate: int) -> str:
    """One sample domain: mono-analysis PCM at a single rate."""

    return f"pcm-mono-{int(sample_rate)}"


def frames_for_ms(milliseconds: float, sample_rate: int) -> int:
    """Convert a duration to integer frames in one declared sample domain."""

    if not math.isfinite(milliseconds) or milliseconds < 0 or sample_rate <= 0:
        raise PerformanceDirectorError("direction_conflict", "a target must be a finite non-negative duration")
    return int(round(milliseconds * sample_rate / 1000.0))


# --------------------------------------------------------------------------- measurement


def _declared_data_bytes(path: Path) -> int | None:
    """The RIFF ``data`` chunk's declared byte count, or ``None`` when unreadable.

    ``wave.getnframes()`` floors the declared byte length to whole frames, so a
    chunk that is not a whole number of frames is silently rounded down and an
    incomplete sample is treated as valid audio.  The declared size is read here
    directly so the caller can validate alignment before trusting it.
    """

    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            while True:
                chunk = stream.read(8)
                if len(chunk) < 8:
                    return None
                identifier = chunk[0:4]
                size = int.from_bytes(chunk[4:8], "little")
                if identifier == b"data":
                    return size
                if size == 0xFFFFFFFF:
                    return None
                # Chunks are word-aligned: an odd size carries one pad byte.
                stream.seek(size + (size & 1), 1)
    except OSError:
        return None


def _open_pcm_wav(path: Path) -> tuple[int, int, int, bytes]:
    """Read one uncompressed, frame-aligned PCM WAV.  Malformed files refuse.

    The declared frame count is never trusted.  The declared ``data`` chunk must
    be a whole number of sample frames and the payload must contain exactly that
    many bytes, so a chunk that is short, unaligned or over-long is refused
    rather than silently rounded down to whole frames.
    """

    try:
        with wave.open(str(path), "rb") as handle:
            comptype = handle.getcomptype()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            declared = handle.getnframes()
            raw = handle.readframes(declared)
    except (wave.Error, OSError, EOFError) as exc:
        raise PerformanceDirectorError("input_ineligible", f"not a readable PCM WAV: {exc}") from exc
    if comptype != "NONE":
        raise PerformanceDirectorError("input_ineligible", f"not uncompressed PCM WAV: {comptype}")
    if channels < 1 or rate <= 0 or width not in (1, 2, 3, 4):
        raise PerformanceDirectorError(
            "input_ineligible", "unsupported PCM WAV channel count, sample rate or sample width")
    if declared <= 0:
        raise PerformanceDirectorError("input_ineligible", "empty PCM WAV: no decoded frames")
    frame_size = channels * width
    declared_data = _declared_data_bytes(path)
    if declared_data is None:
        # No readable RIFF data header: fall back to the whole-frame expectation.
        expected_bytes = declared * frame_size
    else:
        if declared_data % frame_size:
            raise PerformanceDirectorError(
                "input_ineligible",
                f"declared PCM data chunk of {declared_data} bytes is not aligned to whole "
                f"{frame_size}-byte sample frames; an incomplete sample is malformed audio")
        expected_bytes = declared_data
    if len(raw) != expected_bytes:
        raise PerformanceDirectorError(
            "input_ineligible",
            f"truncated PCM WAV: {len(raw)} payload bytes for a declared {expected_bytes}-byte "
            "data chunk")
    return channels, width, rate, raw


def _samples(raw: bytes, width: int, channels: int) -> np.ndarray:
    usable = len(raw) // width
    if usable < channels:
        raise PerformanceDirectorError("input_ineligible", "truncated PCM WAV: fewer samples than channels")
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8, count=usable).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2", count=usable).astype(np.float64) / 32768.0
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8, count=usable * 3).reshape(-1, 3)
        padded = np.zeros((octets.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = octets
        # The zero pad leaves each 24-bit two's-complement value with a clear
        # sign byte, so restore the sign from bit 23 before scaling.
        widened = padded.view(np.dtype("<i4")).reshape(-1).astype(np.int64)
        widened = np.where(widened & 0x800000, widened - 0x1000000, widened)
        values = widened.astype(np.float64) / 8388608.0
    else:
        values = np.frombuffer(raw, dtype="<i4", count=usable).astype(np.float64) / 2147483648.0
    if not np.isfinite(values).all():
        raise PerformanceDirectorError("input_ineligible", "non-finite PCM samples")
    return values.reshape(-1, channels)


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), _SILENCE_FLOOR_LINEAR))


def _region_dbfs(centered: np.ndarray, start: int, end: int) -> float | None:
    """RMS dBFS of a sample region, taken as the loudest channel (never a mean)."""

    if end <= start:
        return None
    region = centered[start:end]
    per_channel = np.sqrt(np.mean(np.square(region, dtype=np.float64), axis=0))
    return round(_dbfs(float(np.max(per_channel))), 3)


def _window_evidence(centered: np.ndarray, frame_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-window content level and peak, covering the final partial window.

    Both statistics are taken per original channel and reduced with a
    conservative maximum, never a channel mean: averaging channels can cancel
    audible samples (a +0.5/-0.5 prefix sums to zero), and a cancelled window
    must not be able to masquerade as safe silence.

    ``level`` is the loudest channel's RMS in dBFS, ``peak`` the loudest sample
    magnitude in any channel, both over one 10 ms window.  ``partial_samples``
    is the length of the trailing window that is shorter than ``frame_samples``
    (0 when the audio divides exactly).
    """

    total, _ = centered.shape
    full = total // frame_samples
    levels: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    if full:
        block = centered[: full * frame_samples].reshape(full, frame_samples, centered.shape[1])
        per_channel = np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=1))
        levels.append(20.0 * np.log10(np.maximum(per_channel.max(axis=1), _SILENCE_FLOOR_LINEAR)))
        peaks.append(np.max(np.abs(block), axis=(1, 2)))
    partial = total - full * frame_samples
    if partial:
        tail = centered[full * frame_samples:]
        tail_rms = np.sqrt(np.mean(np.square(tail, dtype=np.float64), axis=0))
        levels.append(np.array([20.0 * math.log10(max(float(tail_rms.max()), _SILENCE_FLOOR_LINEAR))]))
        peaks.append(np.array([float(np.max(np.abs(tail)))]))
    if not levels:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), 0)
    return (np.concatenate(levels), np.concatenate(peaks), int(partial))


def _run_from_start(mask: np.ndarray) -> int:
    for index, flag in enumerate(mask):
        if not flag:
            return index
    return int(mask.shape[0])


def _run_to_end(mask: np.ndarray) -> int:
    for offset, flag in enumerate(reversed(mask)):
        if not flag:
            return offset
    return int(mask.shape[0])


def _lexical_block(lexical: bool) -> dict[str, Any]:
    """Lexical evidence is never guessed: M2 runs no aligner and no ASR."""

    if lexical:
        return {
            "status": "unavailable",
            "reason": "no aligner or ASR runs in M2; CPU word timestamps are not measured here",
            "first_word_start_s": None, "last_word_end_s": None, "min_word_probability": None,
            "wer": None, "model_sha256": None, "settings": None,
        }
    return {
        "status": "not_requested",
        "reason": "lexical evidence was not requested; energy edges are not word boundaries",
        "first_word_start_s": None, "last_word_end_s": None, "min_word_probability": None,
        "wer": None, "model_sha256": None, "settings": None,
    }


def measure_take(audio_path: Path, expected_sha256: str, *, text: str | None,
                 lexical: bool = False) -> dict[str, Any]:
    """Measure the content edges and silence of one Take.  Reads bytes; changes nothing."""

    expected = _require_sha256(expected_sha256, "expected_sha256")
    path = Path(audio_path)
    if not path.is_file():
        raise PerformanceDirectorError("input_ineligible", "Take audio file is missing")
    actual = qa.sha256_file(path)
    if actual != expected:
        raise PerformanceDirectorError("input_ineligible", "Take bytes do not match the expected digest")
    facts = qa.read_audio_facts(path)
    if facts is None or not facts.decodable:
        raise PerformanceDirectorError("input_ineligible", "Take is not a decodable PCM WAV")
    channels, width, sample_rate, raw = _open_pcm_wav(path)
    decoded = _samples(raw, width, channels)
    frames = int(decoded.shape[0])
    if frames == 0:
        raise PerformanceDirectorError("input_ineligible", "empty PCM WAV: no decoded frames")
    if facts.sample_rate > 0 and (facts.sample_rate != sample_rate or facts.channels != channels):
        raise PerformanceDirectorError("input_ineligible", "PCM facts disagree with the decoded stream")

    # DC is removed per channel before any measurement.  A multi-channel Take is
    # measured, not rejected, because M2 must not invent a format policy it does
    # not own; safety is derived from every original channel rather than a mean.
    dc_offsets = [float(value) for value in np.mean(decoded, axis=0)]
    centered = decoded - np.asarray(dc_offsets, dtype=np.float64)
    if channels > 1 and not np.isfinite(centered).all():
        raise PerformanceDirectorError("input_ineligible", "non-finite PCM samples")
    frame_samples = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
    levels, peaks, partial_samples = _window_evidence(centered, frame_samples)
    windows = int(levels.shape[0])

    content = levels >= CONTENT_FLOOR_DBFS
    strictly_silent = peaks <= (10.0 ** (SILENCE_CEILING_DBFS / 20.0))
    indexes = np.flatnonzero(content)
    first_content = int(indexes[0]) if indexes.size else None
    last_content = int(indexes[-1]) if indexes.size else None

    # A window's span is its real sample range, so a final partial window is
    # measured rather than dropped, and an edge is the pre-content region
    # itself: it never counts a content window as preceding silence.
    if first_content is None:
        leading_content = frames
        leading_floor = _region_dbfs(centered, 0, frames)
    else:
        leading_content = first_content * frame_samples
        leading_floor = _region_dbfs(centered, 0, leading_content)
    head = strictly_silent if first_content is None else strictly_silent[:first_content]
    leading_silent = min(frames, (_run_from_start(head) if head.size else 0) * frame_samples)

    if last_content is None:
        trailing_content = frames
        trailing_floor = _region_dbfs(centered, 0, frames)
    else:
        content_end = min(frames, (last_content + 1) * frame_samples)
        trailing_content = frames - content_end
        trailing_floor = _region_dbfs(centered, content_end, frames)
    tail = strictly_silent if last_content is None else strictly_silent[last_content + 1:]
    silent_tail_windows = _run_to_end(tail) if tail.size else 0
    # The silent run is reported as its real sample span, so a shorter final
    # window contributes only the samples it actually covers.
    trailing_silent = frames - (windows - silent_tail_windows) * frame_samples if silent_tail_windows else 0

    onset_samples = max(2, int(round(sample_rate * 0.002)))
    onset_window = centered[: min(onset_samples, frames)]
    first_index = int(np.argmax(np.abs(centered[0])))
    onset = {
        "first_sample": round(float(centered[0, first_index]), 8),
        "max_step_2ms": round(float(np.max(np.abs(np.diff(onset_window, axis=0)))), 8)
        if onset_window.shape[0] > 1 else None,
        "dc_removed": True,
    }

    flags: list[str] = []
    if windows == 0:
        status = "uncertain"
        flags.append("insufficient_frames")
    elif first_content is None:
        status = "all_silent"
        flags.append(ALL_SILENT_FLAG)
    else:
        status = "measured"
    if lexical:
        flags.append("lexical_unavailable")

    record = {
        "schema_version": EDGE_MEASUREMENT_SCHEMA,
        "status": status,
        "audio_sha256": actual,
        "sample_rate": int(sample_rate),
        "sample_domain": sample_domain(sample_rate),
        "frames": frames,
        "channels": int(channels),
        "text_sha256": sha256_bytes(text.encode("utf-8")) if isinstance(text, str) else None,
        "detector": {
            "version": DETECTOR_VERSION,
            "frame_ms": FRAME_MS,
            "frame_samples": frame_samples,
            "windows": windows,
            "partial_final_window_samples": partial_samples,
            "content_floor_dbfs": CONTENT_FLOOR_DBFS,
            "silence_ceiling_dbfs": SILENCE_CEILING_DBFS,
            "dc_removed": True,
            "dc_offset": round(max(dc_offsets, key=abs), 8),
            "dc_offsets": [round(value, 8) for value in dc_offsets],
            "strict_silence_basis": "every sample in every channel of the window is at or below the "
                                    "silence ceiling",
            "content_basis": "the loudest channel's window RMS is at or above the content floor",
            "channel_reduction": "maximum across channels, never a mean: averaging channels could "
                                 "cancel audible opposite-phase samples and pass them off as silence",
            "edge_basis": "edge regions are the real sample spans before the first and after the last "
                          "content window, including a final partial window",
            "conservatism": "windows are quantised from sample 0, so a reported run is never longer than the true one",
        },
        "leading": {
            "content_frames": int(leading_content),
            "silent_frames": int(leading_silent),
            "floor_dbfs": leading_floor,
        },
        "trailing": {
            "content_frames": int(trailing_content),
            "silent_frames": int(trailing_silent),
            "floor_dbfs": trailing_floor,
        },
        "onset": onset,
        "lexical": _lexical_block(bool(lexical)),
        "flags": flags,
        "limitations": [EDGE_MEASUREMENT_LIMITATION],
        "machine_generated": True,
        "human_verified": False,
    }
    record["content_sha256"] = content_sha256(record)
    return record


def shortening_eligible(measurement: Mapping[str, Any]) -> bool:
    """True only when a leading-edge shortening decision is defensible for this record."""

    if not isinstance(measurement, Mapping) or measurement.get("status") != "measured":
        return False
    if set(measurement.get("flags") or ()) & set(SHORTENING_BLOCKING_FLAGS):
        return False
    leading = measurement.get("leading")
    if not isinstance(leading, Mapping):
        return False
    silent = leading.get("silent_frames")
    return isinstance(silent, int) and not isinstance(silent, bool) and silent > 0


def edge_uncertain(measurement: Mapping[str, Any]) -> bool:
    """True when this record's edges must not be shortened."""

    if not isinstance(measurement, Mapping):
        return True
    if measurement.get("status") != "measured":
        return True
    return EDGE_UNCERTAIN in set(measurement.get("flags") or ())


# --------------------------------------------------------------------------- classification


def _resolve_alias(role_id: str, aliases: Mapping[str, str]) -> str:
    seen: set[str] = set()
    current = role_id
    while current in aliases:
        if current in seen:
            raise PerformanceDirectorError("input_ineligible", f"alias cycle at {role_id}")
        seen.add(current)
        target = aliases[current]
        if not isinstance(target, str) or not target:
            raise PerformanceDirectorError("input_ineligible", f"alias for {current} is not a role id")
        current = target
    return current


def _final_punctuation(text: str) -> str:
    stripped = text.rstrip()
    return stripped[-1] if stripped else ""


def _starts_lower(text: str) -> bool:
    for character in text.lstrip():
        if character.isalpha():
            return character.islower()
        if character.isdigit():
            return False
    return False


def _classification(klass: str, features: dict[str, Any], reason: str | None,
                    a_role: str | None, b_role: str | None) -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_CLASSIFICATION_SCHEMA,
        "classifier_version": CLASSIFIER_VERSION,
        "class": klass,
        "features": features,
        "reason": reason,
        "a_role_id": a_role,
        "b_role_id": b_role,
        "machine_generated": True,
        "human_verified": False,
    }


def classify_boundary(a: Mapping[str, Any], b: Mapping[str, Any], *,
                      role_types: Mapping[str, str],
                      aliases: Mapping[str, str]) -> dict[str, Any]:
    """Classify the boundary between two adjacent source lines into section 9.2's classes.

    Input row contract, minimal and explicit::

        {"utterance_id": str, "role_id": str, "text": str,
         "source_revision_id": str | None, "start_offset": int | None,
         "end_offset": int | None, "is_tag": bool,
         "separator": bool,                       # explicit separator between a and b
         "previous_tag": {"role_id": ..., "class": ...} | None}

    ``text`` is the line AFTER any speech tag, never including quotation marks.
    ``previous_tag`` is the only source of continuation context: M2 never
    invents a third turn from a two-row signature.
    """

    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise PerformanceDirectorError("input_ineligible", "both boundary rows must be mappings")
    for name, row in (("a", a), ("b", b)):
        _require_identifier(row.get("utterance_id"), f"{name}.utterance_id")
    text_a = a.get("text")
    text_b = b.get("text")
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise PerformanceDirectorError("input_ineligible", "both boundary rows need a text string")

    features: dict[str, Any] = {
        "a_text_is_empty": not text_a.strip(),
        "b_text_is_empty": not text_b.strip(),
        "a_end": _final_punctuation(text_a),
        "b_lower": _starts_lower(text_b),
        "b_words": len(text_b.split()),
        "a_is_tag": bool(a.get("is_tag")),
        "explicit_separator": bool(b.get("separator")),
        "source_gap": None,
        "para_break": None,
        "same_role": None,
        "a_role_type": None,
        "b_role_type": None,
        "previous_tag": None,
    }

    # An explicit separator takes precedence and is represented exactly once.
    if b.get("separator"):
        features["source_gap"] = None
        return _classification(SCENE_SEPARATOR, features, "explicit source separator between the lines",
                               None, None)

    revision_a = a.get("source_revision_id")
    revision_b = b.get("source_revision_id")
    if not isinstance(revision_a, str) or not isinstance(revision_b, str) or not revision_a or not revision_b:
        return _classification(UNKNOWN_CLASS, features, "source revision is missing", None, None)
    if revision_a != revision_b:
        return _classification(UNKNOWN_CLASS, features,
                               "the two lines come from different source revisions", None, None)

    start = b.get("start_offset")
    end = a.get("end_offset")
    if not isinstance(start, int) or isinstance(start, bool) or \
            not isinstance(end, int) or isinstance(end, bool) or start < 0 or end < 0:
        return _classification(UNKNOWN_CLASS, features, "source anchor offsets are missing", None, None)
    gap = start - end
    features["source_gap"] = gap
    if gap < 0:
        return _classification(UNKNOWN_CLASS, features, "the anchors overlap or are out of order", None, None)
    # A gap of 9 is a larger structural break; it is recorded, never promoted to
    # a scene separator, because that meaning awaits the baseline audit.
    features["para_break"] = gap > 0

    role_a_raw = a.get("role_id")
    role_b_raw = b.get("role_id")
    if not isinstance(role_a_raw, str) or not isinstance(role_b_raw, str):
        return _classification(UNKNOWN_CLASS, features, "a role id is missing", None, None)
    try:
        role_a = _resolve_alias(role_a_raw, aliases)
        role_b = _resolve_alias(role_b_raw, aliases)
    except PerformanceDirectorError as exc:
        return _classification(UNKNOWN_CLASS, features, str(exc), None, None)

    type_a = role_types.get(role_a, role_types.get(role_a_raw))
    type_b = role_types.get(role_b, role_types.get(role_b_raw))
    features["a_role_type"] = type_a
    features["b_role_type"] = type_b
    features["same_role"] = role_a == role_b
    if type_a not in (NARRATOR, "character") or type_b not in (NARRATOR, "character"):
        return _classification(UNKNOWN_CLASS, features,
                               "a role type is unknown or not speech (roles: "
                               f"{type_a!r}, {type_b!r})", role_a, role_b)

    if type_a == NARRATOR and type_b == NARRATOR:
        klass = "N_N_NEW_PARA" if features["para_break"] else "N_N_SAME_PARA"
        return _classification(klass, features, None, role_a, role_b)

    if type_a == NARRATOR and type_b == "character":
        if features["para_break"]:
            return _classification("N_D_NEW_PARA", features, None, role_a, role_b)
        if not features["a_is_tag"]:
            return _classification("N_D_SAME_PARA", features, None, role_a, role_b)
        previous = b.get("previous_tag")
        if previous is None:
            return _classification(UNKNOWN_CLASS, features,
                                   "insufficient continuation context: a narrator tag precedes this "
                                   "quotation and no previous_tag was supplied", role_a, role_b)
        if not isinstance(previous, Mapping):
            return _classification(UNKNOWN_CLASS, features, "ambiguous tag: previous_tag is not a mapping",
                                   role_a, role_b)
        previous_class = previous.get("class")
        features["previous_tag"] = {"role_id": previous.get("role_id"), "class": previous_class}
        if previous_class not in BOUNDARY_CLASSES:
            return _classification(UNKNOWN_CLASS, features, "ambiguous tag: previous_tag class is unknown",
                                   role_a, role_b)
        previous_role = previous.get("role_id")
        if previous_class != "D_TAG" or not isinstance(previous_role, str):
            return _classification(UNKNOWN_CLASS, features,
                                   "insufficient continuation context: the previous turn is not a D_TAG",
                                   role_a, role_b)
        try:
            previous_canonical = _resolve_alias(previous_role, aliases)
        except PerformanceDirectorError as exc:
            return _classification(UNKNOWN_CLASS, features, str(exc), role_a, role_b)
        if previous_canonical != role_b:
            return _classification(UNKNOWN_CLASS, features,
                                   "insufficient continuation context: the tagged speaker is not this speaker",
                                   role_a, role_b)
        features["same_role"] = True
        return _classification("TAG_D_CONTINUATION", features, None, role_a, role_b)

    if type_a == "character" and type_b == NARRATOR:
        if not features["para_break"]:
            if features["a_end"] == "," or features["b_lower"] or _TAG_RE.match(text_b):
                return _classification("D_TAG", features, None, role_a, role_b)
        klass = "D_N_NEW_PARA" if features["para_break"] else "D_N_BEAT_SAME_PARA"
        return _classification(klass, features, None, role_a, role_b)

    if role_a == role_b:
        return _classification("D_D_SAME_SPEAKER", features, None, role_a, role_b)
    if text_a.rstrip().endswith(_DASH_ENDINGS):
        return _classification("D_D_INTERRUPTION", features, None, role_a, role_b)
    if text_a.rstrip().endswith(_ELLIPSIS_ENDINGS):
        return _classification("D_D_AFTER_TRAILOFF", features, None, role_a, role_b)
    return _classification("D_D_REPLY", features, None, role_a, role_b)


# --------------------------------------------------------------------------- timeline


def default_policy() -> dict[str, Any]:
    """Boundary policy version 0, the experimental hypotheses of section 9.2."""

    return {
        "schema_version": BOUNDARY_POLICY_SCHEMA,
        "targets_ms": {
            "D_TAG": 0.0, "TAG_D_CONTINUATION": 180.0, "N_D_SAME_PARA": 250.0, "N_D_NEW_PARA": 450.0,
            "D_N_BEAT_SAME_PARA": 250.0, "D_N_NEW_PARA": 450.0, "D_D_REPLY": 400.0,
            "D_D_INTERRUPTION": 120.0, "D_D_AFTER_TRAILOFF": 650.0, "D_D_SAME_SPEAKER": 300.0,
            "N_N_SAME_PARA": 220.0, "N_N_NEW_PARA": 500.0,
        },
        "separator_ms": float(SCENE_SEPARATOR_MS),
        "offset_guard_ms": float(LEAD_OFFSET_GUARD_MS),
        "hint_floor_ms": HINT_FLOOR_MS,
        "hint_cap_ms": HINT_CAP_MS,
    }


def _checked_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a boundary policy, enforcing the fixed 9.2 invariants.

    Target *values* are experimental.  Three properties are not, and a policy
    cannot relax them: the leading-offset guard is at least 40 ms, the quick
    hint floor is at least 80 ms, and no target exceeds the 1500 ms cap outside
    a scene separator.  A target class omitted by design routes to the row's
    explicit baseline spacing; a misspelled class name is still a refusal.
    """

    if not isinstance(policy, Mapping):
        raise PerformanceDirectorError("schema_unknown", "policy must be a mapping")
    if policy.get("schema_version") != BOUNDARY_POLICY_SCHEMA:
        raise PerformanceDirectorError("schema_unknown", "policy schema version is unknown")
    targets = policy.get("targets_ms")
    if not isinstance(targets, Mapping):
        raise PerformanceDirectorError("schema_unknown", "policy targets_ms must be a mapping")
    unknown = sorted(set(targets) - set(POLICY_TARGET_CLASSES))
    if unknown:
        raise PerformanceDirectorError("schema_unknown", f"policy names unknown boundary classes: {unknown}")
    resolved = {name: _require_finite(targets[name], f"policy target for {name}")
                for name in POLICY_TARGET_CLASSES if name in targets}
    separator_ms = _require_finite(policy.get("separator_ms"), "policy separator_ms")
    guard_ms = _require_finite(policy.get("offset_guard_ms"), "policy offset_guard_ms")
    hint_floor_ms = _require_finite(policy.get("hint_floor_ms"), "policy hint_floor_ms")
    hint_cap_ms = _require_finite(policy.get("hint_cap_ms"), "policy hint_cap_ms")
    if guard_ms < LEAD_OFFSET_GUARD_MS:
        raise PerformanceDirectorError(
            "direction_conflict",
            f"policy offset_guard_ms must be at least {LEAD_OFFSET_GUARD_MS:.0f} ms; "
            f"the leading-offset guard is an invariant, not a target")
    if hint_floor_ms < HINT_FLOOR_MS:
        raise PerformanceDirectorError(
            "direction_conflict",
            f"policy hint_floor_ms must be at least {HINT_FLOOR_MS:.0f} ms")
    if hint_cap_ms > HINT_CAP_MS:
        raise PerformanceDirectorError(
            "direction_conflict",
            f"policy hint_cap_ms must be at most {HINT_CAP_MS:.0f} ms outside a scene separator")
    if hint_floor_ms > hint_cap_ms:
        raise PerformanceDirectorError("direction_conflict",
                                       "policy hint_floor_ms must not exceed hint_cap_ms")
    return {
        "schema_version": BOUNDARY_POLICY_SCHEMA,
        "targets_ms": resolved,
        "separator_ms": separator_ms,
        "offset_guard_ms": guard_ms,
        "hint_floor_ms": hint_floor_ms,
        "hint_cap_ms": hint_cap_ms,
    }


def _checked_row(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PerformanceDirectorError("input_ineligible", f"row {index} must be a mapping")
    if raw.get("derived") or raw.get("derived_from") or raw.get("source_take_id") is not None:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} is a derived-input record; derived audio is not an input")
    if raw.get("kind") not in (None, "speech"):
        raise PerformanceDirectorError("input_ineligible", f"row {index} has an unsupported kind")
    utterance_id = _require_identifier(raw.get("utterance_id"), f"row {index} utterance_id")
    project_id = _require_identifier(raw.get("project_id"), f"row {index} project_id")
    take_id = _require_identifier(raw.get("take_id"), f"row {index} take_id")
    audio_sha256 = _require_sha256(raw.get("audio_sha256"), f"row {index} audio_sha256")
    measurement_sha256 = _require_sha256(raw.get("measurement_sha256"), f"row {index} measurement_sha256")
    sample_rate = raw.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise PerformanceDirectorError("input_ineligible", f"row {index} sample_rate must be a positive integer")
    start = _require_count(raw.get("source_start_frame"), f"row {index} source_start_frame")
    end = _require_count(raw.get("source_end_frame"), f"row {index} source_end_frame")
    if end <= start:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} has an impossible source frame range [{start}, {end})")
    measurement = raw.get("measurement")
    if not isinstance(measurement, Mapping):
        raise PerformanceDirectorError("input_ineligible", f"row {index} measurement must be a mapping")
    if measurement.get("schema_version") != EDGE_MEASUREMENT_SCHEMA:
        raise PerformanceDirectorError("schema_unknown", f"row {index} measurement schema is unknown")
    if content_sha256(measurement) != measurement_sha256:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} measurement digest does not match its own record")
    if measurement.get("audio_sha256") != audio_sha256:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} measurement belongs to different audio")
    if measurement.get("sample_rate") != sample_rate:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} measurement was taken in a different sample domain")
    measured_frames = _require_count(measurement.get("frames"), f"row {index} measurement frames")
    if measured_frames <= 0:
        raise PerformanceDirectorError("input_ineligible",
                                       f"row {index} measurement reports an empty Take")

    # M2's contract is whole original Takes.  Only the explicitly gated
    # leading-offset descriptor below may shorten anything, so a caller-supplied
    # pre-cropped source range is refused rather than silently honoured, and the
    # trailing end of every Take is always protected in full.
    if start != 0 or end != measured_frames:
        raise PerformanceDirectorError(
            "input_ineligible",
            f"row {index} selection [{start}, {end}) is not the whole measured Take "
            f"(0, {measured_frames}); a pre-cropped source range is not a timeline input")

    edges: dict[str, tuple[int, int]] = {}
    for edge in ("leading", "trailing"):
        block = measurement.get(edge)
        if not isinstance(block, Mapping):
            raise PerformanceDirectorError("input_ineligible",
                                           f"row {index} measurement has no {edge} block")
        content_frames = _require_count(block.get("content_frames"),
                                        f"row {index} measurement {edge}.content_frames")
        silent_frames = _require_count(block.get("silent_frames"),
                                       f"row {index} measurement {edge}.silent_frames")
        if content_frames > measured_frames:
            raise PerformanceDirectorError(
                "input_ineligible",
                f"row {index} measurement {edge}.content_frames exceeds the Take length")
        if silent_frames > content_frames:
            raise PerformanceDirectorError(
                "input_ineligible",
                f"row {index} measurement {edge}.silent_frames exceeds its own {edge} edge region")
        edges[edge] = (content_frames, silent_frames)
    if measurement.get("status") == "measured" and \
            edges["leading"][0] + edges["trailing"][0] > measured_frames:
        raise PerformanceDirectorError(
            "input_ineligible",
            f"row {index} measurement edge regions are inconsistent with a {measured_frames}-frame Take")

    boundary = raw.get("boundary")
    baseline_spacer_frames = 0
    if index == 0:
        if boundary is not None:
            raise PerformanceDirectorError("direction_conflict", "the first row cannot have a preceding boundary")
        if raw.get("baseline_spacer_frames") is not None:
            raise PerformanceDirectorError(
                "direction_conflict",
                "baseline_spacer_frames belongs to a boundary and the first row has none")
    else:
        if not isinstance(boundary, Mapping) or boundary.get("class") not in BOUNDARY_CLASSES:
            raise PerformanceDirectorError("direction_conflict",
                                           f"row {index} needs a classification of the boundary before it")
        # Explicit baseline spacing is the only representation of an existing
        # gap for a boundary whose class has no experimental target: UNKNOWN,
        # or a class the policy omits by design.  It is preserved, not inferred.
        supplied = raw.get("baseline_spacer_frames")
        if supplied is not None:
            baseline_spacer_frames = _require_count(supplied, f"row {index} baseline_spacer_frames")
    return {
        "project_id": project_id, "utterance_id": utterance_id, "take_id": take_id,
        "audio_sha256": audio_sha256, "measurement_sha256": measurement_sha256,
        "sample_rate": sample_rate, "source_start_frame": start, "source_end_frame": end,
        "measurement": measurement, "boundary": boundary,
        "baseline_spacer_frames": baseline_spacer_frames,
    }


def _checked_hints(hints: Mapping[str, str] | None, rows: Sequence[dict[str, Any]],
                   policy: Mapping[str, Any]) -> dict[str, str]:
    """A hint belongs to the line whose FOLLOWING boundary it adjusts."""

    if hints is None:
        return {}
    if not isinstance(hints, Mapping):
        raise PerformanceDirectorError("direction_conflict", "hints must be a mapping of utterance id to hint")
    positions = {row["utterance_id"]: index for index, row in enumerate(rows)}
    resolved: dict[str, str] = {}
    for key, value in hints.items():
        if key not in positions:
            raise PerformanceDirectorError("direction_conflict", f"hint names an unknown utterance: {key!r}")
        if value not in BOUNDARY_HINTS:
            raise PerformanceDirectorError("direction_conflict", f"hint for {key!r} is not a known hint: {value!r}")
        index = positions[key]
        if index == len(rows) - 1:
            raise PerformanceDirectorError("direction_conflict",
                                           f"hint for {key!r} has no following boundary to adjust")
        following = rows[index + 1]["boundary"].get("class")
        if following == SCENE_SEPARATOR:
            raise PerformanceDirectorError("direction_conflict",
                                           f"hint for {key!r} would modify an explicit scene separator")
        if following not in policy["targets_ms"]:
            raise PerformanceDirectorError(
                "direction_conflict",
                f"hint for {key!r} has no experimental target to adjust on a {following} boundary; "
                "a baseline boundary keeps its explicit baseline spacing")
        resolved[key] = value
    return resolved


def _hinted_target_ms(base_ms: float, hint: str | None, policy: Mapping[str, Any]) -> float:
    if hint is None:
        target = base_ms
    elif hint == "quick":
        target = max(policy["hint_floor_ms"], base_ms / 2.0)
    else:
        target = base_ms + HINT_ADD_MS[hint]
    return min(target, policy["hint_cap_ms"])


def _boundary_arithmetic(klass: str, previous: dict[str, Any], row: dict[str, Any], sample_rate: int, *,
                         policy: dict[str, Any], hint: str | None,
                         allow_lead_offset: bool, guard_frames: int) -> dict[str, Any]:
    """Integer-frame arithmetic for one boundary.  Never negative, never destructive."""

    result: dict[str, Any] = {
        "class": klass,
        "features": row["boundary"].get("features"),
        "hint": hint,
        "existing_frames": None,
        "target_frames": None,
        "spacer_frames": 0,
        "offset_frames": 0,
        "effective_frames": None,
        "eligible_offset_frames": 0,
        "baseline_spacer_frames": 0,
        "uncertain_edges": [],
        "status": "met",
    }
    if klass == SCENE_SEPARATOR:
        separator = frames_for_ms(policy["separator_ms"], sample_rate)
        result.update({"target_frames": separator, "effective_frames": separator, "status": "met"})
        return result

    existing = int(previous["measurement"]["trailing"]["content_frames"]) + \
        int(row["measurement"]["leading"]["content_frames"])
    # A class with no experimental target is a baseline boundary: UNKNOWN, or a
    # class the policy omits by design.  Its spacing is the caller's explicit
    # baseline value, preserved verbatim, never an inferred assembly timing.
    baseline = klass not in policy["targets_ms"]
    if baseline:
        spacer = int(row["baseline_spacer_frames"])
        result.update({
            "existing_frames": existing, "target_frames": None, "spacer_frames": spacer,
            "offset_frames": 0, "eligible_offset_frames": 0,
            "baseline_spacer_frames": spacer, "effective_frames": existing + spacer, "status": "baseline",
        })
        return result

    target = frames_for_ms(_hinted_target_ms(float(policy["targets_ms"][klass]), hint, policy), sample_rate)
    spacer = max(0, target - existing)
    uncertain: list[str] = []
    if edge_uncertain(previous["measurement"]):
        uncertain.append("trailing")
    if edge_uncertain(row["measurement"]):
        uncertain.append("leading")
    offset = 0
    offset_permitted = bool(allow_lead_offset) and klass in OFFSET_ELIGIBLE_CLASSES
    needs_shortening = existing > target
    eligible = max(0, int(row["measurement"]["leading"]["silent_frames"]) - guard_frames)

    if offset_permitted and needs_shortening and not uncertain:
        offset = min(existing - target, eligible)
    if offset_permitted and needs_shortening and uncertain:
        # Uncertainty prevents a trustworthy comparison, so the protected status
        # stands rather than reporting an achieved target.
        status = "protected_uncertain_edge"
    else:
        # Status is derived from the arithmetic actually achieved, never from the
        # fact that some adjustment was attempted: a partial shortening that
        # leaves the gap above its target is still limited by the existing audio.
        effective = existing - offset + spacer
        status = "met" if effective == target else "limited_by_existing_audio"

    result.update({
        "existing_frames": existing, "target_frames": target, "spacer_frames": spacer,
        "offset_frames": offset, "eligible_offset_frames": eligible,
        "effective_frames": existing - offset + spacer, "uncertain_edges": uncertain, "status": status,
    })
    return result


def compile_timeline(rows: Sequence[dict[str, Any]], *, policy: dict[str, Any],
                     hints: Mapping[str, str], allow_lead_offset: bool) -> dict[str, Any]:
    """Compile an integer-frame timing plan.  Pure arithmetic; no file is written."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) == 0:
        raise PerformanceDirectorError("input_ineligible", "the timeline needs at least one row")
    if not isinstance(allow_lead_offset, bool):
        raise PerformanceDirectorError("direction_conflict", "allow_lead_offset must be a boolean")
    checked_policy = _checked_policy(policy)
    checked_rows = [_checked_row(raw, index) for index, raw in enumerate(rows)]
    checked_hints = _checked_hints(hints, checked_rows, checked_policy)

    sample_rate = checked_rows[0]["sample_rate"]
    for index, row in enumerate(checked_rows):
        if row["sample_rate"] != sample_rate:
            raise PerformanceDirectorError("input_ineligible",
                                           f"row {index} is in a different sample domain than row 0")

    guard_frames = frames_for_ms(checked_policy["offset_guard_ms"], sample_rate)
    placements: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    cursor = 0
    for index, row in enumerate(checked_rows):
        offset = 0
        if index > 0:
            previous = checked_rows[index - 1]
            klass = row["boundary"]["class"]
            arithmetic = _boundary_arithmetic(
                klass, previous, row, sample_rate, policy=checked_policy,
                hint=checked_hints.get(previous["utterance_id"]), allow_lead_offset=allow_lead_offset,
                guard_frames=guard_frames)
            arithmetic["boundary_index"] = len(boundaries)
            arithmetic["after_utterance_id"] = previous["utterance_id"]
            arithmetic["before_utterance_id"] = row["utterance_id"]
            boundaries.append(arithmetic)
            offset = int(arithmetic["offset_frames"])
            if klass == SCENE_SEPARATOR:
                placements.append({
                    "kind": "scene_separator", "frames": int(arithmetic["target_frames"]),
                    "boundary_index": arithmetic["boundary_index"],
                    "boundary": {"class": klass, "features": arithmetic["features"], "status": "met"},
                })
                cursor += int(arithmetic["target_frames"])
            elif arithmetic["spacer_frames"]:
                placements.append({
                    "kind": "spacer", "frames": int(arithmetic["spacer_frames"]),
                    "boundary_index": arithmetic["boundary_index"],
                    "boundary": {"class": klass, "features": arithmetic["features"],
                                 "hint": arithmetic["hint"],
                                 "existing_frames": arithmetic["existing_frames"],
                                 "target_frames": arithmetic["target_frames"],
                                 "status": arithmetic["status"]},
                })
                cursor += int(arithmetic["spacer_frames"])
        source_frames = row["source_end_frame"] - row["source_start_frame"]
        derived_frames = source_frames - offset
        if derived_frames < 0 or cursor < 0:
            raise PerformanceDirectorError(
                "input_ineligible",
                f"row {index} would derive a negative placement length "
                f"({derived_frames} frames from {source_frames} source frames and a {offset}-frame offset)")
        placements.append({
            "kind": "speech", "project_id": row["project_id"], "utterance_id": row["utterance_id"],
            "take_id": row["take_id"], "audio_sha256": row["audio_sha256"],
            "measurement_sha256": row["measurement_sha256"],
            "source_start_frame": row["source_start_frame"], "source_end_frame": row["source_end_frame"],
            "offset_frames": offset, "derived_start_frame": cursor, "derived_frames": derived_frames,
        })
        cursor += derived_frames

    policy_sha256 = content_sha256(checked_policy)
    dependency = {
        "schema_version": BOUNDARY_TIMELINE_SCHEMA,
        "classifier_version": CLASSIFIER_VERSION,
        "assembler_version": ASSEMBLER_VERSION,
        "sample_rate": sample_rate,
        "sample_domain": sample_domain(sample_rate),
        "policy_sha256": policy_sha256,
        "allow_lead_offset": allow_lead_offset,
        "hints": [{"utterance_id": row["utterance_id"], "hint": checked_hints[row["utterance_id"]]}
                  for row in checked_rows if row["utterance_id"] in checked_hints],
        "rows": [{
            "project_id": row["project_id"], "utterance_id": row["utterance_id"], "take_id": row["take_id"],
            "audio_sha256": row["audio_sha256"], "measurement_sha256": row["measurement_sha256"],
            "source_start_frame": row["source_start_frame"], "source_end_frame": row["source_end_frame"],
            "baseline_spacer_frames": row["baseline_spacer_frames"],
            "boundary": None if row["boundary"] is None else {
                "class": row["boundary"]["class"],
                "features": row["boundary"].get("features"),
                "classifier_version": row["boundary"].get("classifier_version"),
            },
        } for row in checked_rows],
    }
    return {
        "schema_version": BOUNDARY_TIMELINE_SCHEMA,
        "sample_rate": sample_rate,
        "sample_domain": sample_domain(sample_rate),
        "policy_sha256": policy_sha256,
        "classifier_version": CLASSIFIER_VERSION,
        "assembler_version": ASSEMBLER_VERSION,
        "allow_lead_offset": allow_lead_offset,
        "placements": placements,
        "boundaries": boundaries,
        "total_frames": cursor,
        "limitations": [
            TIMELINE_ELIGIBILITY_LIMITATION,
            TIMELINE_FINGERPRINT_LIMITATION,
            "the compiler does not verify that a Take is eligible for a pilot; that check belongs to the assembler",
        ],
        "timeline_sha256": sha256_bytes(canonical_json_bytes(dependency)),
        "machine_generated": True,
        "human_verified": False,
    }


# --------------------------------------------------------------------------- delivery outcomes (M6)


# M6 appendix.  The M2 block above is deterministic and side-effect free by
# design; these functions differ in exactly two named ways, and neither changes
# an existing owner.  One Take is measured once from an explicitly supplied
# scratch snapshot, and the review summary reads one pilot through the accepted
# M3 owner.  Nothing here synthesizes audio, loads a model or an ASR, uses a
# GPU, calls a network or a service, deletes or truncates a file, writes a
# project, records a review or produces a listening verdict.

DELIVERY_OUTCOME_SCHEMA = "pd-delivery-outcome-1"
DELIVERY_COMPARISON_SCHEMA = "pd-delivery-comparison-1"
DELIVERY_REVIEW_SUMMARY_SCHEMA = "pd-delivery-review-summary-1"
DELIVERY_AGGREGATOR_VERSION = "pd-delivery-aggregate-1"

#: The accepted acoustic owner.  Imported lazily: the M2 path never pays for it.
DELIVERY_ACOUSTICS_MODULE = "service.voice_acoustics"
DELIVERY_ACOUSTICS_LIMIT_SECONDS = 30.0
#: The only directory this milestone ever creates, and only under a caller's root.
DELIVERY_SNAPSHOT_DIRECTORY = "pd-delivery-snapshots"

#: The existing reference-ASR owner's record.  M6 reads a receipt; it never runs
#: that owner, never installs it and never reaches the network for it.
LEXICAL_RECEIPT_SCHEMA = "reference-asr-v1"
LEXICAL_RECEIPT_STATUSES = ("draft", "unavailable", "failed")
LEXICAL_MAX_SEGMENTS = 200
LEXICAL_MAX_WORDS = 1000
LEXICAL_MAX_CHARACTERS = 10000
LEXICAL_DURATION_TOLERANCE_SECONDS = 0.05

#: The local initial advisory threshold convention.  P12 names a threshold
#: without fixing its value, so this is chosen here, reported in every
#: comparison, and is not Brian's listening criterion.
DELIVERY_WER_THRESHOLD = 0.1

DELIVERY_LIMITATION = (
    "delivery outcomes are advisory machine measurements of one Take: they are not a listening "
    "verdict, they never reject or retry a Take, they never choose an arm and they grant no scope"
)
DELIVERY_LEXICAL_LIMITATION = (
    "a lexical receipt is machine evidence from the existing reference ASR owner; ASR confidence is "
    "not reviewer authentication, and a missing, unavailable or failed alignment stays null"
)
DELIVERY_COMPARISON_LIMITATION = (
    "a comparison is arithmetic over two machine measurements; it can only flag a pair for listening "
    "and can neither accept nor reject a Take"
)
DELIVERY_SUMMARY_LIMITATION = (
    "this summary reuses the accepted M3 improvement gate verbatim and adds no verdict of its own; it "
    "reads no blind key and writes nothing"
)

_COMPARABLE_RE = re.compile(r"[\W_]+", re.UNICODE)
_NUMBER_TYPES = (int, float)
_DELIVERY_OWNER_CACHE: dict[str, Any] = {}

#: Every delta a comparison reports, as a path into a delivery block.
_DELIVERY_METRIC_PATHS = (
    ("audio", "duration_seconds"),
    ("elapsed_text_rate", "words_per_minute"),
    ("articulation_rate_proxy", "syllables_per_minute"),
    ("pauses", "count"),
    ("pitch", "median_hz"),
    ("pitch", "range_semitones"),
    ("onset", "max_step_2ms"),
    ("loudness", "peak_dbfs"),
    ("lexical", "wer"),
    ("lexical", "minimum_word_probability"),
    ("lexical", "final_word", "probability"),
)


def _bounded_text(value: Any, what: str, *, limit: int = 400) -> str:
    """A present, bounded, single-record string.  Longer or odd input refuses."""

    if not isinstance(value, str) or not value.strip():
        raise PerformanceDirectorError("input_ineligible", f"{what} must be a non-empty string")
    if len(value) > limit or _CONTROL_RE.search(value):
        raise PerformanceDirectorError(
            "input_ineligible", f"{what} must be at most {limit} characters without control characters")
    return value


def _require_finite_numbers(value: Any, what: str) -> None:
    """Every numeric leaf must be finite, at any nesting depth."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, _NUMBER_TYPES):
        if not math.isfinite(value):
            raise PerformanceDirectorError("input_ineligible", f"{what} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_numbers(item, f"{what}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{what}[{index}]")


def _canonical_digest(value: Any, what: str) -> str:
    """The existing canonical owner's digest, with an honest refusal on odd input."""

    try:
        return sha256_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise PerformanceDirectorError(
            "input_ineligible", f"{what} cannot be canonicalised for its identity: {exc}") from exc


def _metric(value: Any, *, units: str, method: str, sample_count: int, scope: str,
            derivation: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """One numeric measurement with its units, method, sample count and scope.

    An unmeasurable or non-finite value is ``None`` with a reason; it is never a
    guessed number and never a silent zero.
    """

    if isinstance(value, bool) or not isinstance(value, _NUMBER_TYPES) or not math.isfinite(value):
        return {"value": None, "units": units, "method": method, "sample_count": int(sample_count),
                "scope": scope, "derivation": derivation,
                "reason": reason or "not measurable from this Take's evidence"}
    return {"value": float(value), "units": units, "method": method, "sample_count": int(sample_count),
            "scope": scope, "derivation": derivation, "reason": None}


def _metric_value(node: Any) -> float | None:
    """A finite value from a metric block or a bare number, else ``None``."""

    if isinstance(node, Mapping):
        node = node.get("value")
    if isinstance(node, bool) or not isinstance(node, _NUMBER_TYPES) or not math.isfinite(node):
        return None
    return float(node)


def _dig(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = source
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _collect_unknowns(block: Any, prefix: str = "") -> list[dict[str, str]]:
    """Every metric that stayed ``None``, with the reason the record gave."""

    found: list[dict[str, str]] = []
    if isinstance(block, Mapping):
        if {"value", "units", "method", "sample_count", "scope"} <= set(block):
            if block.get("value") is None:
                found.append({"field": prefix, "reason": str(block.get("reason") or "not measured")})
            return found
        for key, item in block.items():
            found.extend(_collect_unknowns(item, f"{prefix}.{key}" if prefix else str(key)))
        return found
    if isinstance(block, Sequence) and not isinstance(block, (str, bytes)):
        for index, item in enumerate(block):
            found.extend(_collect_unknowns(item, f"{prefix}[{index}]"))
    return found


def _acoustics_owner() -> Any:
    """The accepted acoustic owner, imported lazily and never reimplemented."""

    from service import voice_acoustics  # noqa: PLC0415 - deliberately lazy

    return voice_acoustics


def _delivery_owner_identity() -> dict[str, Any]:
    """Source, method, parameter and package identity of the owners M6 reads.

    Re-read on every measurement so source, parameters and installed packages
    cannot change while a stale process-wide identity continues to be cited.
    """

    import importlib.metadata  # noqa: PLC0415 - only needed for identity

    acoustics = _acoustics_owner()

    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    _DELIVERY_OWNER_CACHE.update({  # noqa: RUF012 - module-level memo of file identities
        "module": __name__,
        "module_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "detector_version": DETECTOR_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "acoustics_module": DELIVERY_ACOUSTICS_MODULE,
        "acoustics_source_sha256": sha256_bytes(Path(acoustics.__file__).read_bytes()),
        "acoustics_schema": acoustics.SCHEMA,
        "acoustics_method": acoustics.METHOD,
        "acoustics_parameters": {
            "analysis_rate": int(acoustics.ANALYSIS_RATE),
            "frame_ms": float(acoustics.FRAME_MS),
            "hop_ms": float(acoustics.HOP_MS),
            "f0_min_hz": float(acoustics.F0_MIN_HZ),
            "f0_max_hz": float(acoustics.F0_MAX_HZ),
            "pause_min_ms": float(acoustics.PAUSE_MIN_MS),
            "analysis_limit_seconds": float(DELIVERY_ACOUSTICS_LIMIT_SECONDS),
        },
        "packages": {name: version(name) for name in ("numpy", "soundfile")},
    })
    return _DELIVERY_OWNER_CACHE


def _delivery_source() -> dict[str, Any]:
    """A defensive copy of the cached identity, so a record can be mutated safely."""

    identity = _delivery_owner_identity()
    return {**identity,
            "acoustics_parameters": dict(identity["acoustics_parameters"]),
            "packages": dict(identity["packages"])}


def _take_snapshot(scratch_root: Any, audio_sha256: str, payload: bytes) -> Path:
    """Exclusively capture the exact Take bytes under the caller's scratch root.

    The digest names the file, so an existing snapshot is verified reuse rather
    than a byte conflict.  Nothing is ever overwritten, moved, truncated or
    deleted, and no hidden temporary or default projects root is ever used.
    """

    root = Path(scratch_root)
    if not root.is_dir():
        raise PerformanceDirectorError(
            "input_ineligible",
            "scratch_root must be an existing directory the caller owns; a snapshot is never written "
            "to a hidden temporary or default projects root")
    folder = root / DELIVERY_SNAPSHOT_DIRECTORY
    if folder.resolve() != root.resolve() / DELIVERY_SNAPSHOT_DIRECTORY:
        raise PerformanceDirectorError("input_ineligible", "snapshot directory escapes the explicit scratch root")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{audio_sha256}.wav"
    if target.resolve() != folder.resolve() / target.name:
        raise PerformanceDirectorError("input_ineligible", "snapshot file escapes the explicit scratch root")
    if target.is_file():
        if target.read_bytes() != payload:
            raise PerformanceDirectorError(
                "input_ineligible",
                "the existing snapshot for this digest holds different bytes; it is never overwritten")
        return target
    try:
        with target.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if target.read_bytes() != payload:
            raise PerformanceDirectorError(
                "input_ineligible",
                "the snapshot appeared with different bytes during capture; the measurement is refused")
    return target


def _owner_number(source: Mapping[str, Any], key: str, what: str,
                  *, minimum: float = 0.0) -> float | None:
    """A finite number from the accepted owner, or ``None`` when it reported none."""

    value = source.get(key)
    if value is None:
        return None
    return _require_finite(value, f"{what}.{key}", minimum=minimum)


def _analysis_scope(analysed_seconds: float | None, duration: float, sample_rate: int) -> tuple[str, int]:
    """The acoustic owner's analysed scope and the PCM sample count it covers."""

    if analysed_seconds is None:
        return ("no analysed span: the acoustic owner reported no analysis length", 0)
    samples = int(round(analysed_seconds * _acoustics_owner().ANALYSIS_RATE))
    scope = (f"first {analysed_seconds:.3f} s of {duration:.3f} s at the owner's analysis rate, "
             "downmixed to mono by mean when the Take has more than one channel")
    return (scope, samples)


def _checked_lexical_receipt(receipt: Any, *, audio_sha256: str,
                             decoded_duration: float) -> dict[str, Any]:
    """Validate a caller-supplied reference-asr-v1 receipt.

    Malformed, mismatched or unbound evidence is refused rather than applied.
    A receipt is machine evidence: it records its own identity and never claims
    human verification.
    """

    if not isinstance(receipt, Mapping):
        raise PerformanceDirectorError("input_ineligible", "lexical_receipt must be a mapping")
    status = receipt.get("status")
    if status not in LEXICAL_RECEIPT_STATUSES:
        raise PerformanceDirectorError(
            "input_ineligible", f"lexical_receipt.status must be one of {LEXICAL_RECEIPT_STATUSES}")
    if receipt.get("schema") != LEXICAL_RECEIPT_SCHEMA:
        raise PerformanceDirectorError(
            "input_ineligible", f"lexical_receipt.schema must be {LEXICAL_RECEIPT_SCHEMA}")
    if receipt.get("machine_generated") is not True:
        raise PerformanceDirectorError(
            "input_ineligible", "a lexical receipt must be machine_generated")
    if receipt.get("human_verified") is not False:
        raise PerformanceDirectorError(
            "input_ineligible",
            "a lexical receipt is machine evidence; human_verified must be false and never stands in "
            "for reviewer authentication")
    _require_finite_numbers(receipt, "lexical_receipt")
    if _require_sha256(receipt.get("audio_sha256"),
                       "lexical_receipt.audio_sha256") != audio_sha256:
        raise PerformanceDirectorError(
            "input_ineligible",
            "lexical_receipt.audio_sha256 does not bind to this Take's exact approved bytes")

    declared = receipt.get("duration_seconds")
    declared_seconds: float | None = None
    if declared is not None:
        declared_seconds = _require_finite(declared, "lexical_receipt.duration_seconds")
        if abs(declared_seconds - decoded_duration) > LEXICAL_DURATION_TOLERANCE_SECONDS:
            raise PerformanceDirectorError(
                "input_ineligible",
                "lexical_receipt.duration_seconds disagrees with the decoded Take duration")

    identity: dict[str, Any] = {
        "schema": LEXICAL_RECEIPT_SCHEMA,
        "status": status,
        "reason": None,
        "model": None,
        "model_sha256": None,
        "settings_sha256": None,
        "pipeline_sha256": None,
        "language": None,
        "duration_seconds": declared_seconds,
        "segment_count": 0,
        "word_count": 0,
        "receipt_sha256": _canonical_digest(dict(receipt), "lexical_receipt"),
        "machine_generated": True,
        "human_verified": False,
    }
    if receipt.get("model") is not None:
        identity["model"] = _require_identifier(receipt["model"], "lexical_receipt.model")
    if receipt.get("model_sha256") is not None:
        identity["model_sha256"] = _require_sha256(receipt["model_sha256"], "lexical_receipt.model_sha256")
    if receipt.get("pipeline_sha256") is not None:
        identity["pipeline_sha256"] = _require_sha256(receipt["pipeline_sha256"],
                                                     "lexical_receipt.pipeline_sha256")
    if receipt.get("language") is not None:
        identity["language"] = _require_identifier(receipt["language"], "lexical_receipt.language")
    if receipt.get("settings") is not None:
        settings = receipt["settings"]
        if not isinstance(settings, Mapping):
            raise PerformanceDirectorError(
                "input_ineligible", "lexical_receipt.settings must be a mapping")
        identity["settings_sha256"] = _canonical_digest(dict(settings), "lexical_receipt.settings")

    if status != "draft":
        if receipt.get("reason") is not None:
            identity["reason"] = _bounded_text(receipt["reason"], "lexical_receipt.reason")
        return {"identity": identity, "text": None, "tokens": None, "words": None}

    for key in ("model", "model_sha256", "pipeline_sha256", "settings", "dependency_versions", "duration_seconds"):
        if receipt.get(key) is None:
            raise PerformanceDirectorError("input_ineligible", f"a draft receipt requires {key} provenance")
    dependencies = receipt["dependency_versions"]
    if not isinstance(dependencies, Mapping) or not dependencies or any(
            not isinstance(key, str) or not isinstance(value, str) or not value.strip()
            for key, value in dependencies.items()):
        raise PerformanceDirectorError("input_ineligible", "draft dependency_versions must name recorded packages")
    if not receipt["settings"]:
        raise PerformanceDirectorError("input_ineligible", "draft settings must record the ASR configuration")
    if not 3 <= decoded_duration <= 30.01:
        raise PerformanceDirectorError("input_ineligible", "a draft requires the existing ASR owner's 3–30 second crop")

    text = receipt.get("text")
    if not isinstance(text, str) or not text.strip():
        raise PerformanceDirectorError(
            "input_ineligible", "a draft receipt must carry the text it recognised")
    if len(text) > LEXICAL_MAX_CHARACTERS:
        raise PerformanceDirectorError(
            "input_ineligible", f"lexical_receipt.text is limited to {LEXICAL_MAX_CHARACTERS} characters")
    if _require_sha256(receipt.get("text_sha256"),
                       "lexical_receipt.text_sha256") != sha256_bytes(text.encode("utf-8")):
        raise PerformanceDirectorError(
            "input_ineligible", "lexical_receipt.text_sha256 does not match its own text")

    segments = receipt.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        raise PerformanceDirectorError(
            "input_ineligible", "a draft receipt must carry at least one segment")
    if len(segments) > LEXICAL_MAX_SEGMENTS:
        raise PerformanceDirectorError(
            "input_ineligible", f"lexical_receipt is limited to {LEXICAL_MAX_SEGMENTS} segments")

    bound = decoded_duration + LEXICAL_DURATION_TOLERANCE_SECONDS
    previous_start = 0.0
    words: list[dict[str, Any]] = []
    for number, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise PerformanceDirectorError(
                "input_ineligible", f"lexical_receipt.segments[{number}] must be a mapping")
        start = _require_finite(segment.get("start"), f"lexical_receipt.segments[{number}].start")
        end = _require_finite(segment.get("end"), f"lexical_receipt.segments[{number}].end")
        if start > end or end > bound:
            raise PerformanceDirectorError(
                "input_ineligible",
                f"lexical_receipt.segments[{number}] timestamps fall outside the exact Take audio")
        if start < previous_start:
            raise PerformanceDirectorError(
                "input_ineligible", "lexical_receipt segments are not ordered by their start time")
        previous_start = start
        rows = segment.get("words") or []
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise PerformanceDirectorError(
                "input_ineligible", f"lexical_receipt.segments[{number}].words must be a list")
        for position, row in enumerate(rows):
            where = f"lexical_receipt.segments[{number}].words[{position}]"
            if not isinstance(row, Mapping):
                raise PerformanceDirectorError("input_ineligible", f"{where} must be a mapping")
            word = row.get("word")
            if not isinstance(word, str) or not word.strip() or len(word) > 100:
                raise PerformanceDirectorError(
                    "input_ineligible", f"{where}.word must be a non-empty word of at most 100 characters")
            word_start = _require_finite(row.get("start"), f"{where}.start")
            word_end = _require_finite(row.get("end"), f"{where}.end")
            probability = _require_finite(row.get("probability"), f"{where}.probability")
            if word_start > word_end or word_end > bound:
                raise PerformanceDirectorError(
                    "input_ineligible", f"{where} timestamps fall outside the exact Take audio")
            if word_start < start or word_end > end:
                raise PerformanceDirectorError("input_ineligible", f"{where} timestamps fall outside their segment")
            if probability > 1.0:
                raise PerformanceDirectorError(
                    "input_ineligible", f"{where}.probability must be between 0 and 1")
            if words and word_start < words[-1]["start"]:
                raise PerformanceDirectorError(
                    "input_ineligible", "lexical_receipt words are not ordered by their start time")
            words.append({"word": word, "start": word_start, "end": word_end, "probability": probability})
            if len(words) > LEXICAL_MAX_WORDS:
                raise PerformanceDirectorError(
                    "input_ineligible", f"lexical_receipt is limited to {LEXICAL_MAX_WORDS} words")

    normalize = lambda value: [_comparable_token(token) for token in _token_list(value) if _comparable_token(token)]
    if normalize(text) != normalize(" ".join(str(s.get("text", "")) for s in segments)):
        raise PerformanceDirectorError("input_ineligible", "receipt text disagrees with segment text")
    if words and normalize(text) != normalize(" ".join(row["word"] for row in words)):
        raise PerformanceDirectorError("input_ineligible", "receipt text disagrees with aligned word rows")
    identity["segment_count"] = len(segments)
    identity["word_count"] = len(words)
    return {"identity": identity, "text": text, "tokens": _token_list(text), "words": words}


def _token_list(text: str) -> list[str]:
    """Whitespace-delimited tokens, used for counted words and for the WER."""

    return re.findall(r"\S+", text)


def _comparable_token(token: str) -> str:
    """One token reduced to case-folded alphanumerics for a fair edit distance."""

    return _COMPARABLE_RE.sub("", token.casefold())


def _token_edit_distance(expected: Sequence[str], actual: Sequence[str]) -> int:
    """Deterministic Levenshtein distance over token sequences; two rolling rows."""

    previous = list(range(len(actual) + 1))
    for index, left in enumerate(expected, start=1):
        current = [index]
        for position, right in enumerate(actual, start=1):
            current.append(min(previous[position] + 1,
                               current[position - 1] + 1,
                               previous[position - 1] + (left != right)))
        previous = current
    return previous[-1]


def _lexical_outcome(checked: Mapping[str, Any], *, expected_text: str | None) -> dict[str, Any]:
    """The receipt's identity plus the advisory numbers it can honestly support."""

    identity = dict(checked["identity"])
    outcome: dict[str, Any] = {**identity, "alignment": None, "wer": None, "final_word": None,
                               "minimum_word_probability": None,
                               "limitations": [DELIVERY_LEXICAL_LIMITATION]}
    words = checked["words"] or []
    if identity["status"] != "draft":
        outcome["owner_reason"] = identity.get("reason")
        outcome["reason"] = (f"the reference ASR owner reported {identity['status']}; no alignment is "
                             "available for this Take")
        return outcome
    if not words:
        outcome["reason"] = "the draft has no word alignment; WER and final-word outcomes remain unknown"
        return outcome
    if words:
        outcome["alignment"] = {"first_word_start_s": words[0]["start"],
                                "last_word_end_s": words[-1]["end"],
                                "word_count": len(words)}
        outcome["minimum_word_probability"] = _metric(
            min(row["probability"] for row in words), units="probability",
            method="the lowest word probability in the reference ASR alignment",
            sample_count=len(words), scope="the whole recognised alignment")
    expected = [_comparable_token(token) for token in _token_list(expected_text or "")]
    expected = [token for token in expected if token]
    actual = [_comparable_token(token) for token in (checked["tokens"] or [])]
    actual = [token for token in actual if token]
    if not expected:
        outcome["reason"] = ("no expected text was supplied, so no word error rate or final word can "
                             "be compared; the alignment stays advisory and unverified")
        return outcome
    distance = _token_edit_distance(expected, actual)
    outcome["wer"] = _metric(
        distance / len(expected), units="ratio",
        method="deterministic token-normalised edit distance over case-folded, punctuation-stripped "
               "tokens divided by the expected token count",
        sample_count=len(expected), scope="the whole recognised text of this Take",
        derivation="counted against the expected text supplied by the caller")
    outcome["wer"]["edit_count"] = distance
    outcome["wer"]["expected_token_count"] = len(expected)
    outcome["wer"]["actual_token_count"] = len(actual)
    final_token = expected[-1]
    matches = [index for index, row in enumerate(words)
               if _comparable_token(row["word"]) == final_token]
    # A repeated earlier token cannot stand in for a missing terminal occurrence.
    normalized_words = [_comparable_token(row["word"]) for row in words]
    if (not normalized_words or normalized_words[-1] != final_token
            or normalized_words.count(final_token) < expected.count(final_token)):
        matches = []
    outcome["final_word"] = {
        "token": final_token,
        "present": bool(matches),
        "position": matches[-1] if matches else None,
        "is_last_recognised_token": (matches[-1] == len(words) - 1) if matches else None,
        "probability": _metric(
            words[matches[-1]]["probability"] if matches else None, units="probability",
            method="the reference ASR probability of the word row matching the expected final token",
            sample_count=len(words), scope="the whole recognised alignment",
            reason="the expected final word is not present anywhere in the alignment"),
        "end_s": words[matches[-1]]["end"] if matches else None,
        "method": "conservative terminal-token presence: last aligned token matches and the "
                  "alignment contains every expected occurrence of that token",
    }
    return outcome


def _generation_outcome(generation: Any) -> dict[str, Any] | None:
    """Recorded generation and budget evidence, or ``None`` when none was kept.

    An unknown token budget or an unknown alignment yields null, never a guess,
    and no threshold is invented: ``approached`` stays null unless the caller's
    own evidence names the guard threshold it was measured against.
    """

    if generation is None:
        return None
    if not isinstance(generation, Mapping):
        raise PerformanceDirectorError("input_ineligible", "generation evidence must be a mapping")
    _require_finite_numbers(generation, "generation")
    budget = _owner_number(generation, "token_budget", "generation")
    used = _owner_number(generation, "instruction_tokens", "generation")
    if used is None and generation.get("tokens_used") is not None:
        used = _require_finite(generation["tokens_used"], "generation.tokens_used")
    threshold = _owner_number(generation, "guard_threshold", "generation")
    proximity: float | None = None
    reason = "the recorded generation evidence names no token budget"
    if budget is not None and budget > 0 and used is not None:
        proximity = used / budget
        reason = None
    elif budget == 0:
        reason = "the recorded token budget is zero, so no proximity ratio exists"
    approached: bool | None = None
    if proximity is not None and threshold is not None:
        approached = bool(proximity >= threshold)
    return {
        "recorded": True,
        "identity": {"sha256": _canonical_digest(dict(generation), "generation"),
                     "inputs": dict(generation)},
        "token_budget": budget,
        "instruction_tokens": used,
        "guard_threshold": threshold,
        "guard_proximity": _metric(
            proximity, units="ratio",
            method="recorded instruction tokens divided by the recorded token budget",
            sample_count=1, scope="the recorded generation evidence for this Take",
            derivation="reported only when both recorded numbers exist",
            reason=reason or "not measurable from the recorded generation evidence"),
        "approached": approached,
        "approached_reason": ("no guard threshold was recorded, so no proximity to a guard is asserted"
                              if approached is None else None),
    }


def measure_delivery_take(audio_path: Path, expected_sha256: str, *, text: str | None,
                          scratch_root: Any, lexical_receipt: Mapping[str, Any] | None = None,
                          generation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure one Take's advisory delivery outcome from its exact approved bytes.

    The record is a copy of the M2 ``edge-measurement-v1`` record extended with a
    ``pd-delivery-outcome-1`` delivery block.  The M2 owner's own identity is
    preserved verbatim in ``delivery.edge_identity``, so M2 evidence is never
    rewritten or confused with this enrichment, and the enriched
    ``content_sha256`` is recomputed through the same canonical owner.

    The exact bytes are captured once, exclusively, into a content-addressed
    snapshot under the caller's explicit ``scratch_root``.  Snapshots are
    retained, reused only when verified identical, and never deleted.
    """

    expected = _require_sha256(expected_sha256, "expected_sha256")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise PerformanceDirectorError("input_ineligible", "text must be a non-empty string or None")
    if isinstance(text, str) and len(text) > LEXICAL_MAX_CHARACTERS:
        raise PerformanceDirectorError(
            "input_ineligible", f"text is limited to {LEXICAL_MAX_CHARACTERS} characters")
    path = Path(audio_path)
    if not path.is_file():
        raise PerformanceDirectorError("input_ineligible", "Take audio file is missing")
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected:
        raise PerformanceDirectorError(
            "input_ineligible", "Take bytes do not match the expected digest")

    snapshot = _take_snapshot(scratch_root, actual, payload)
    base = measure_take(snapshot, expected, text=text)
    if base["audio_sha256"] != actual:
        raise PerformanceDirectorError(
            "input_ineligible", "the scratch snapshot does not reproduce the Take's own digest")

    owner = _delivery_source()
    acoustics_module = _acoustics_owner()
    acoustics = acoustics_module.measure(snapshot, max_seconds=DELIVERY_ACOUSTICS_LIMIT_SECONDS)
    if snapshot.read_bytes() != payload:
        raise PerformanceDirectorError(
            "input_ineligible",
            "the scratch snapshot changed during measurement; both measurements are rejected rather "
            "than attributed to different bytes")
    if (acoustics.get("schema") != acoustics_module.SCHEMA
            or acoustics.get("method") != acoustics_module.METHOD):
        raise PerformanceDirectorError(
            "input_ineligible",
            "the acoustic owner returned a record whose schema or method is not the one it advertises")

    rate = int(base["sample_rate"])
    frames = int(base["frames"])
    channels = int(base["channels"])
    duration = frames / rate
    analysed_seconds = _owner_number(acoustics, "analysed_seconds", "acoustics")
    scope, analysis_samples = _analysis_scope(analysed_seconds, duration, rate)

    tokens = _token_list(text) if isinstance(text, str) else []
    onset_frames = min(frames, max(2, int(round(rate * 0.002))))
    if lexical_receipt is not None and not isinstance(lexical_receipt, Mapping):
        raise PerformanceDirectorError("input_ineligible", "lexical_receipt must be a mapping")
    if lexical_receipt is not None and lexical_receipt.get("status") == "draft" and channels != 1:
        raise PerformanceDirectorError("input_ineligible", "a draft requires the existing ASR owner's mono crop")
    lexical_outcome = (None if lexical_receipt is None else _lexical_outcome(
        _checked_lexical_receipt(lexical_receipt, audio_sha256=actual,
                                 decoded_duration=duration),
        expected_text=text))
    text_block = {
        "provided": isinstance(text, str),
        "sha256": sha256_bytes(text.encode("utf-8")) if isinstance(text, str) else None,
        "token_count": len(tokens),
        "token_method": "whitespace-delimited tokens of the expected text; the same tokenisation "
                        "serves the elapsed text-word rate and the word error rate",
    }
    elapsed_rate = _metric(
        (len(tokens) / duration * 60.0) if tokens and duration > 0 else None, units="words/minute",
        method="counted expected-text tokens divided by the Take's total decoded duration",
        sample_count=len(tokens), scope=f"the whole {duration:.3f} s Take",
        derivation="counted words over elapsed duration, deliberately not the acoustics owner's "
                   "signal-derived words per minute",
        reason="no expected text was supplied, so no text-word rate exists")

    pitch = {
        "median_hz": _metric(
            _owner_number(acoustics, "f0_median_hz", "acoustics"), units="Hz",
            method="voice-acoustics-v1 autocorrelation f0 median (40 ms frame / 10 ms hop)",
            sample_count=analysis_samples, scope=scope,
            reason="the acoustic owner measured no median pitch for this Take"),
        "range_semitones": _metric(
            _owner_number(acoustics, "f0_range_semitones", "acoustics"), units="semitones",
            method="voice-acoustics-v1 f0 range between the p10 and p90 statistics",
            sample_count=analysis_samples, scope=scope,
            reason="the acoustic owner measured no pitch range for this Take"),
        "p10_hz": _metric(
            _owner_number(acoustics, "f0_p10_hz", "acoustics"), units="Hz",
            method="voice-acoustics-v1 autocorrelation f0 tenth percentile",
            sample_count=analysis_samples, scope=scope,
            reason="the acoustic owner measured no f0 tenth percentile for this Take"),
        "p90_hz": _metric(
            _owner_number(acoustics, "f0_p90_hz", "acoustics"), units="Hz",
            method="voice-acoustics-v1 autocorrelation f0 ninetieth percentile",
            sample_count=analysis_samples, scope=scope,
            reason="the acoustic owner measured no f0 ninetieth percentile for this Take"),
    }

    pause_rate = _owner_number(acoustics, "pause_count_per_minute", "acoustics")
    pause_count = (round(pause_rate * analysed_seconds / 60.0)
                   if pause_rate is not None and analysed_seconds is not None else None)
    delivery: dict[str, Any] = {
        "schema_version": DELIVERY_OUTCOME_SCHEMA,
        "aggregator_version": DELIVERY_AGGREGATOR_VERSION,
        "source": owner,
        "edge_identity": {
            "schema_version": base["schema_version"],
            "content_sha256": base["content_sha256"],
            "audio_sha256": base["audio_sha256"],
            "status": base["status"],
            "flags": list(base["flags"]),
            "sample_rate": rate,
            "frames": frames,
            "channels": channels,
            "detector_version": base["detector"]["version"],
            "note": "the M2 detector's own identity for this Take; the enrichment never rewrites it",
        },
        "snapshot": {
            "directory": DELIVERY_SNAPSHOT_DIRECTORY,
            "file": snapshot.name,
            "capture": "exclusive content-addressed capture (mode 'xb')",
            "reuse_policy": ("an existing snapshot named by this digest is reused only after its bytes "
                             "are verified identical; it is never overwritten, moved or deleted"),
            "bytes": len(payload),
            "sha256": actual,
            "retained": True,
            "verified_after_measurement": True,
        },
        "audio": {
            "duration_seconds": _metric(
                duration, units="seconds", method="decoded frame count divided by the declared rate",
                sample_count=frames, scope="the whole Take"),
            "sample_rate": _metric(rate, units="Hz", method="the decoded PCM WAV header",
                                   sample_count=frames, scope="the whole Take"),
            "frames": _metric(frames, units="frames", method="the decoded PCM WAV payload",
                              sample_count=frames, scope="the whole Take"),
            "channels": _metric(channels, units="channels", method="the decoded PCM WAV header",
                                sample_count=frames, scope="the whole Take"),
            "sample_domain": base["sample_domain"],
        },
        "text": text_block,
        "elapsed_text_rate": {"words_per_minute": elapsed_rate},
        "articulation_rate_proxy": {
            "syllables_per_minute": _metric(
                _owner_number(acoustics, "syllables_per_minute", "acoustics"), units="syllables/minute",
                method="voice-acoustics-v1 syllable-envelope rate from the accepted acoustic owner",
                sample_count=analysis_samples, scope=scope,
                derivation="an explicit signal proxy: it is never the counted expected words over "
                           "elapsed duration",
                reason="the acoustic owner could not measure a syllable rate for this Take"),
            "articulation_index": _metric(
                _owner_number(acoustics, "articulation_index", "acoustics"), units="ratio",
                method="voice-acoustics-v1 high-band articulation index",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner could not measure an articulation index for this Take"),
        },
        "pauses": {
            "count": _metric(
                pause_count, units="pauses",
                method="round(acoustics pause_count_per_minute x analysed_seconds / 60)",
                sample_count=analysis_samples, scope=scope,
                derivation="derived from the owner's per-minute rate over its own analysed span",
                reason="the acoustic owner reported no pause rate for this Take"),
            "count_per_minute": _metric(
                pause_rate, units="pauses/minute", method="voice-acoustics-v1 pause detection",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner reported no pause rate for this Take"),
            "mean_pause_ms": _metric(
                _owner_number(acoustics, "mean_pause_ms", "acoustics"), units="milliseconds",
                method="voice-acoustics-v1 mean detected pause length",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner reported no mean pause for this Take"),
            "pause_fraction": _metric(
                _owner_number(acoustics, "pause_fraction", "acoustics"), units="ratio",
                method="voice-acoustics-v1 detected pause time over the analysed span",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner reported no pause fraction for this Take"),
        },
        "pitch": pitch,
        "onset": {
            "first_sample": _metric(
                base["onset"]["first_sample"], units="linear PCM amplitude",
                method="the M2 detector's DC-removed first decoded sample; no aligner is involved",
                sample_count=1, scope="the first sample of the Take"),
            "max_step_2ms": _metric(
                base["onset"]["max_step_2ms"], units="linear PCM amplitude",
                method="the M2 detector's largest absolute sample-to-sample step in the first 2 ms",
                sample_count=onset_frames, scope="the first 2 ms of the Take",
                reason="the Take is too short for a sample-to-sample step"),
        },
        "loudness": {
            "peak_dbfs": _metric(
                _owner_number(acoustics, "peak_dbfs", "acoustics", minimum=float("-inf")), units="dBFS",
                method="voice-acoustics-v1 peak magnitude after mean downmix, DC removal and resampling",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner measured no peak level for this Take"),
            "rms_dbfs": _metric(
                _owner_number(acoustics, "rms_dbfs", "acoustics", minimum=float("-inf")), units="dBFS",
                method="voice-acoustics-v1 RMS after mean downmix, DC removal and resampling",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner measured no RMS level for this Take"),
            "dynamic_range_db": _metric(
                _owner_number(acoustics, "dynamic_range_db", "acoustics"), units="dB",
                method="voice-acoustics-v1 p95 minus p10 frame level",
                sample_count=analysis_samples, scope=scope,
                reason="the acoustic owner measured no dynamic range for this Take"),
        },
        "acoustics": {
            "schema": acoustics.get("schema"),
            "method": acoustics.get("method"),
            "analysis_limit_seconds": float(DELIVERY_ACOUSTICS_LIMIT_SECONDS),
            "analysed_seconds": analysed_seconds,
            "duration_seconds": _owner_number(acoustics, "duration_seconds", "acoustics"),
            "sample_rate": _owner_number(acoustics, "sample_rate", "acoustics"),
            "owner_source_sha256": owner["acoustics_source_sha256"],
            "parameters": dict(owner["acoustics_parameters"]),
            "packages": dict(owner["packages"]),
            "measured": list(acoustics.get("measured") or []),
            "inferred": list(acoustics.get("inferred") or []),
            "notes": list(acoustics.get("notes") or []),
            "limitations": "the owner analyses at most "
                           f"{DELIVERY_ACOUSTICS_LIMIT_SECONDS:.1f} s, resamples to its own analysis "
                           "rate and downmixes multi-channel input by mean",
        },
        "lexical": lexical_outcome,
        "generation": _generation_outcome(generation),
    }
    delivery["lexical_reason"] = ("no lexical receipt was supplied; word error rate, final word and "
                                 "ASR timing proxies stay null" if lexical_outcome is None else None)
    delivery["unknowns"] = _collect_unknowns({key: value for key, value in delivery.items()})
    delivery["limitations"] = [DELIVERY_LIMITATION, DELIVERY_LEXICAL_LIMITATION]
    delivery["machine_generated"] = True
    delivery["human_verified"] = False

    record = dict(base)
    record.pop("content_sha256", None)
    record["delivery"] = delivery
    record["content_sha256"] = content_sha256(record)
    return record


def _checked_delivery_record(record: Any, what: str) -> dict[str, Any]:
    """A delivery record whose content identity verifies, or a refusal."""

    if not isinstance(record, Mapping):
        raise PerformanceDirectorError("input_ineligible", f"{what} must be a delivery outcome record")
    delivery = record.get("delivery")
    if not isinstance(delivery, Mapping) or delivery.get("schema_version") != DELIVERY_OUTCOME_SCHEMA:
        raise PerformanceDirectorError(
            "input_ineligible", f"{what} is not a {DELIVERY_OUTCOME_SCHEMA} delivery record")
    recorded = _require_sha256(record.get("content_sha256"), f"{what}.content_sha256")
    if content_sha256(dict(record)) != recorded:
        raise PerformanceDirectorError(
            "input_ineligible", f"{what}'s content hash does not verify; its identity is invalid")
    edge = delivery.get("edge_identity")
    if not isinstance(edge, Mapping):
        raise PerformanceDirectorError(
            "input_ineligible", f"{what} carries no base edge identity")
    _require_sha256(edge.get("content_sha256"), f"{what}.delivery.edge_identity.content_sha256")
    _require_sha256(edge.get("audio_sha256"), f"{what}.delivery.edge_identity.audio_sha256")
    _require_finite_numbers(record, what)
    base = {key: value for key, value in record.items() if key != "delivery"}
    if content_sha256(base) != edge["content_sha256"]:
        raise PerformanceDirectorError("input_ineligible", f"{what} base edge identity does not verify")
    snapshot = delivery.get("snapshot") or {}
    if not (record.get("audio_sha256") == edge["audio_sha256"] == snapshot.get("sha256")):
        raise PerformanceDirectorError("input_ineligible", f"{what} carries conflicting audio identities")
    for key in ("sample_rate", "frames", "channels"):
        if record.get(key) != edge.get(key) or _metric_value(_dig(delivery, ("audio", key))) != record.get(key):
            raise PerformanceDirectorError("input_ineligible", f"{what} carries conflicting {key} evidence")
    _require_sha256(_dig(delivery, ("text", "sha256")), f"{what}.delivery.text.sha256")
    if record.get("text_sha256") != _dig(delivery, ("text", "sha256")):
        raise PerformanceDirectorError("input_ineligible", f"{what} carries conflicting expected-text identities")
    return dict(record)


def compare_delivery_outcomes(control: Mapping[str, Any], directed: Mapping[str, Any],
                              *, wer_threshold: float = DELIVERY_WER_THRESHOLD) -> dict[str, Any]:
    """Compare two measured Takes of the same line on the same text.

    Both records must be valid, enriched, same-text measurements of different
    audio produced by compatible methods.  The result is advisory: it reports
    deltas with the methods and sample counts behind them and flags a pair for
    listening.  It never rejects a Take, never retries synthesis, never picks a
    preferred arm and never grants scope.
    """

    threshold = _require_finite(wer_threshold, "wer_threshold")
    left = _checked_delivery_record(control, "control")
    right = _checked_delivery_record(directed, "directed")
    left_delivery = left["delivery"]
    right_delivery = right["delivery"]

    left_text = _dig(left_delivery, ("text", "sha256"))
    right_text = _dig(right_delivery, ("text", "sha256"))
    if left_text is None or left_text != right_text:
        raise PerformanceDirectorError(
            "direction_conflict",
            "the two Takes were not measured against the same expected text, so their delivery "
            "outcomes are not comparable")
    left_audio = left_delivery["edge_identity"]["audio_sha256"]
    right_audio = right_delivery["edge_identity"]["audio_sha256"]
    if left_audio == right_audio:
        raise PerformanceDirectorError(
            "direction_conflict",
            "both records describe the same audio bytes; a Take is not its own control")

    left_source = left_delivery.get("source") or {}
    right_source = right_delivery.get("source") or {}
    for key in ("module_sha256", "detector_version", "acoustics_schema", "acoustics_method",
                "acoustics_parameters", "acoustics_source_sha256", "packages"):
        if left_source.get(key) != right_source.get(key):
            raise PerformanceDirectorError(
                "direction_conflict",
                f"the two Takes were not measured by compatible methods and scopes ({key}); the "
                "comparison is refused rather than mixing measurement bases")

    deltas: dict[str, Any] = {}
    unknown: list[str] = []
    def acoustic_coverage(delivery):
        analysed = _dig(delivery, ("acoustics", "analysed_seconds"))
        duration = _metric_value(_dig(delivery, ("audio", "duration_seconds")))
        if analysed is None or duration is None:
            return "unknown"
        return "whole" if analysed >= duration else "prefix"
    coverage_mismatch = acoustic_coverage(left_delivery) != acoustic_coverage(right_delivery)
    for path in _DELIVERY_METRIC_PATHS:
        name = ".".join(path)
        left_metric = _dig(left_delivery, path)
        right_metric = _dig(right_delivery, path)
        if isinstance(left_metric, Mapping) and isinstance(right_metric, Mapping):
            if any(left_metric.get(key) != right_metric.get(key) for key in ("units", "method")):
                raise PerformanceDirectorError("direction_conflict", f"incompatible metric basis for {name}")
        left_value = _metric_value(left_metric)
        right_value = _metric_value(right_metric)
        if coverage_mismatch and path[0] in ("pitch", "pauses", "loudness", "articulation_rate_proxy"):
            unknown.append(name)
            deltas[name] = {"delta": None, "control": left_value, "directed": right_value,
                            "reason": "incompatible analysed scopes: whole Take versus truncated prefix"}
        elif left_value is None or right_value is None:
            unknown.append(name)
            deltas[name] = {"delta": None, "control": left_value, "directed": right_value,
                            "reason": "one or both Takes have no finite value for this metric, so the "
                                      "difference is unknown rather than zero"}
        else:
            deltas[name] = {"delta": round(right_value - left_value, 6), "control": left_value,
                            "directed": right_value, "reason": None}
        deltas[name].update({"units": (left_metric or {}).get("units") if isinstance(left_metric, Mapping) else None,
                            "method": (left_metric or {}).get("method") if isinstance(left_metric, Mapping) else None,
                            "control_sample_count": left_metric.get("sample_count") if isinstance(left_metric, Mapping) else None,
                            "directed_sample_count": right_metric.get("sample_count") if isinstance(right_metric, Mapping) else None,
                            "control_scope": left_metric.get("scope") if isinstance(left_metric, Mapping) else None,
                            "directed_scope": right_metric.get("scope") if isinstance(right_metric, Mapping) else None})

    control_rate = _metric_value(_dig(left_delivery, ("elapsed_text_rate", "words_per_minute")))
    directed_rate = _metric_value(_dig(right_delivery, ("elapsed_text_rate", "words_per_minute")))
    text_block = left_delivery.get("text") or {}
    speed_ratio = _metric(
        (directed_rate / control_rate) if control_rate and directed_rate else None, units="ratio",
        method="the directed elapsed text-word rate divided by the control's",
        sample_count=int(text_block.get("token_count") or 0),
        scope="the whole Take for both arms",
        derivation="counted text words over elapsed duration, not the acoustics signal proxy",
        reason="one or both Takes have no counted text-word rate")

    wer_value = _metric_value(_dig(right_delivery, ("lexical", "wer")))
    final_word = _dig(right_delivery, ("lexical", "final_word"))
    present = final_word.get("present") if isinstance(final_word, Mapping) else None
    missing_final = None if present is None else (not bool(present))
    above = None if wer_value is None else bool(wer_value > threshold)
    insufficient = missing_final is None or above is None

    flags = {
        "directed_missing_final_word": missing_final,
        "wer_above_threshold": above,
        "wer_threshold": threshold,
        "threshold_basis": ("a local advisory convention chosen here because P12 names a threshold "
                            "without fixing its value; it is not Brian's listening criterion"),
        "unknown_metrics": unknown,
        "insufficient_evidence": insufficient,
        "for_listening": bool(missing_final) or bool(above) or insufficient,
        "rejects_take": False,
        "triggers_retry": False,
        "preferred_arm": None,
        "grants_scope": False,
    }
    return {
        "schema_version": DELIVERY_COMPARISON_SCHEMA,
        "wer_threshold": threshold,
        "control": {"content_sha256": left["content_sha256"], "audio_sha256": left_audio,
                    "edge_content_sha256": left_delivery["edge_identity"]["content_sha256"],
                    "source": left_source},
        "directed": {"content_sha256": right["content_sha256"], "audio_sha256": right_audio,
                     "edge_content_sha256": right_delivery["edge_identity"]["content_sha256"],
                     "source": right_source},
        "text_sha256": left_text,
        "deltas": deltas,
        "speed_ratio": speed_ratio,
        "flags": flags,
        "limitations": [DELIVERY_COMPARISON_LIMITATION, DELIVERY_LIMITATION],
        "machine_generated": True,
        "human_verified": False,
    }


def delivery_review_summary(pilot_project_id: str, pair_ids: Sequence[str]) -> dict[str, Any]:
    """Read-only review status for a pilot's blind pairs.

    Every pair is resolved by the accepted M3 ``improvement_gate`` and returned
    verbatim; this function reproduces no part of that acceptance predicate.  A
    pair with no review, no resolved edition or a refused pair stays
    ``review_pending``, and an empty request is never vacuously supported.  No
    blind key is read, no review, rating, favourite, approval or ledger entry is
    written, and nothing here combines measured records into a human verdict.
    """

    from dubbing_service.app.audiobook import performance_pilot  # noqa: PLC0415 - avoids a cycle

    identifier = _require_identifier(pilot_project_id, "pilot_project_id")
    if not isinstance(pair_ids, Sequence) or isinstance(pair_ids, (str, bytes)):
        raise PerformanceDirectorError("input_ineligible", "pair_ids must be a sequence of pair ids")
    checked = [_require_identifier(pair_id, f"pair_ids[{index}]")
               for index, pair_id in enumerate(pair_ids)]
    if len(set(checked)) != len(checked):
        raise PerformanceDirectorError("input_ineligible", "pair_ids must be unique")

    gates: list[dict[str, Any]] = []
    for pair_id in checked:
        try:
            gate = performance_pilot.improvement_gate(identifier, pair_id=pair_id)
        except (PerformanceDirectorError, OSError) as exc:
            gates.append({"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                          "reason": f"the pilot owner refused this pair: {exc}"})
            continue
        if not isinstance(gate, Mapping):
            raise PerformanceDirectorError(
                "input_ineligible", "the pilot owner returned a non-mapping improvement gate")
        gates.append(dict(gate))

    classified: dict[str, list[str]] = {"supported": [], "review_pending": [], "inconclusive": [],
                                        "blocked": [], "not_improved": []}
    for gate in gates:
        status = gate.get("status")
        if status == "supported" and gate.get("supports_improvement") is True:
            classified["supported"].append(gate["pair_id"])
        elif status in ("review_pending", "inconclusive", "blocked", "not_improved"):
            classified[status].append(gate["pair_id"])
        else:
            # A status M6 does not recognise, or a claimed "supported" the owner
            # did not actually assert, is never support: it stays pending.
            classified["review_pending"].append(gate["pair_id"])

    negatives = classified["inconclusive"] + classified["not_improved"]
    if classified["supported"] and not (classified["review_pending"] or classified["blocked"]
                                        or negatives):
        aggregate_status = "supported"
        reason = ("every requested pair is supported by an owner-authenticated review of the exact "
                  "current edition")
    elif not classified["review_pending"] and not classified["blocked"] and negatives:
        aggregate_status = "inconclusive"
        reason = ("at least one recorded review is negative or inconclusive; the requested set "
                  "does not unanimously support improvement")
    elif not gates:
        aggregate_status = "review_pending"
        reason = "no pair was supplied, so nothing can be supported"
    else:
        aggregate_status = "review_pending"
        reason = ("at least one pair has no owner-authenticated review of its current edition, so the "
                  "pilot stays pending rather than partially supported")
    return {
        "schema_version": DELIVERY_REVIEW_SUMMARY_SCHEMA,
        "pilot_project_id": identifier,
        "pairs": gates,
        "aggregate": {**classified, "pair_count": len(gates), "status": aggregate_status,
                      "supports_improvement": aggregate_status == "supported", "reason": reason},
        "requires_separate_stage_c_approval": True,
        "blind_key_read": False,
        "records_written": False,
        "grants_scope": False,
        "limitations": [DELIVERY_SUMMARY_LIMITATION],
        "machine_generated": True,
        "human_verified": False,
    }
