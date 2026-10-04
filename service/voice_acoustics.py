"""Deterministic acoustic measurement and shaping for voice casting profiles.

Everything in this module is pure DSP: no model weights, no network access, no
clock reads and no randomness, so identical input bytes always produce
identical output mappings.  The only file write in the module happens inside
:func:`shape`.

Typical use::

    from service import voice_acoustics as va

    profile = va.measure("performer_reference.wav")
    plan = va.plan_shaping(target_profile, profile)
    va.shape("performer_reference.wav", "candidate.wav", semitones=plan["semitones"],
             rate=plan["rate"])
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SCHEMA = "voice-acoustics-v1"
METHOD = "autocorrelation-f0-v1"
ANALYSIS_RATE = 16000
FRAME_MS = 40
HOP_MS = 10

# --- analysis thresholds (documented so every cut-off is auditable) --------
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
VOICED_MIN_NCC = 0.3
PEAK_SELECT_RATIO = 0.85
NOISE_MARGIN_DB = 6.0
DBFS_FLOOR = -120.0
MIN_VOICED_FRAMES = 5
PAUSE_MIN_MS = 200.0
SYLLABLE_MIN_GAP_MS = 120.0
ADAPTIVE_WINDOW_MS = 250.0
ADAPTIVE_FRACTION = 0.35
# The local dynamic range must be this wide before envelope peaks are trusted,
# otherwise the tiny frame-to-frame ripple of a steady tone counts as speech.
ADAPTIVE_MIN_RANGE_DB = 3.0
ENVELOPE_SMOOTH_FRAMES = 3
SENTENCE_TAIL_MS = 250.0
SENTENCE_END_MS = 100.0
SENTENCE_MIN_FRAMES = 6
HIGH_BAND_HZ = 4000.0
ARTICULATION_REFERENCE_DB = 20.0
# English proxy: 1.5 syllables per word converts the measured syllable rate
# into the word rate that speech engines and casting briefs usually quote.
SYLLABLES_PER_WORD = 1.5

# --- compare() tolerances --------------------------------------------------
TOL_PITCH_RELATIVE = 0.12
TOL_BRIGHTNESS_RELATIVE = 0.12
TOL_RATE_RELATIVE = 0.15
TOL_PAUSE_RELATIVE = 0.15
TOL_ENERGY_DB = 3.0
TOL_PROXY_ABSOLUTE = 0.15
TOL_SENTENCE_SLOPE = 1.5

COMPARE_RELATIVE_TOLERANCES = {
    "f0_median_hz": TOL_PITCH_RELATIVE,
    "f0_mean_hz": TOL_PITCH_RELATIVE,
    "f0_p10_hz": TOL_PITCH_RELATIVE,
    "f0_p90_hz": TOL_PITCH_RELATIVE,
    "f0_range_semitones": TOL_PITCH_RELATIVE,
    "brightness_hz": TOL_BRIGHTNESS_RELATIVE,
    "syllables_per_minute": TOL_RATE_RELATIVE,
    "words_per_minute": TOL_RATE_RELATIVE,
    "pause_count_per_minute": TOL_PAUSE_RELATIVE,
    "mean_pause_ms": TOL_PAUSE_RELATIVE,
    "pause_fraction": TOL_PAUSE_RELATIVE,
}
COMPARE_ABSOLUTE_TOLERANCES = {
    "rms_dbfs": TOL_ENERGY_DB,
    "peak_dbfs": TOL_ENERGY_DB,
    "dynamic_range_db": TOL_ENERGY_DB,
    "sentence_end_energy_drop_db": TOL_ENERGY_DB,
    "sentence_end_f0_slope": TOL_SENTENCE_SLOPE,
    "breathiness": TOL_PROXY_ABSOLUTE,
    "rasp": TOL_PROXY_ABSOLUTE,
    "articulation_index": TOL_PROXY_ABSOLUTE,
    "voiced_fraction": TOL_PROXY_ABSOLUTE,
}

# --- parameters() thresholds ----------------------------------------------
RESONANCE_DEEP_F0_HZ = 115.0
RESONANCE_LIGHT_F0_HZ = 185.0
RESONANCE_BRIGHT_HZ = 2300.0
RESONANCE_DARK_HZ = 1300.0
VOCAL_WEIGHT_HEAVY_F0_HZ = 120.0
VOCAL_WEIGHT_LIGHT_F0_HZ = 180.0
VOCAL_WEIGHT_HEAVY_RANGE_DB = 14.0
VOCAL_WEIGHT_LIGHT_RANGE_DB = 8.0
AGE_OLD_F0_HZ = 100.0
AGE_MATURE_F0_HZ = 130.0
AGE_ADULT_F0_HZ = 190.0
AGE_RASP_BUMP = 0.45
ENGINE_PACE_REFERENCE_WPM = 150.0
ENGINE_PACE_RANGE = (0.85, 1.18)
ENGINE_VOLUME_REFERENCE_DBFS = -20.0
ENGINE_VOLUME_SWING_DB = 40.0
ENGINE_VOLUME_RANGE = (0.5, 1.5)

# --- shaper constants ------------------------------------------------------
SOLA_WINDOW_MS = 30.0
SOLA_SYNTH_HOP_MS = 7.5
SOLA_SEARCH_MS = 6.0
NORMALISE_CEILING = 0.999
_PLAN_PITCH_FLOOR = 0.15
_PLAN_RATE_FLOOR = 0.02
_TRIVIAL = 1e-6

MEASUREMENT_KEYS = (
    "sample_rate",
    "duration_seconds",
    "analysed_seconds",
    "f0_median_hz",
    "f0_mean_hz",
    "f0_p10_hz",
    "f0_p90_hz",
    "f0_range_semitones",
    "syllables_per_minute",
    "words_per_minute",
    "pause_count_per_minute",
    "mean_pause_ms",
    "pause_fraction",
    "rms_dbfs",
    "peak_dbfs",
    "dynamic_range_db",
    "brightness_hz",
    "breathiness",
    "rasp",
    "articulation_index",
    "sentence_end_f0_slope",
    "sentence_end_energy_drop_db",
    "voiced_fraction",
)
_NON_MEASURED_KEYS = frozenset({"schema", "method", "measured", "inferred", "notes", "bias"})

# bias key -> the profile keys it rescales, and how.
_BIAS_PITCH_KEYS = ("f0_median_hz", "f0_mean_hz", "f0_p10_hz", "f0_p90_hz")
_BIAS_RATE_KEYS = ("syllables_per_minute", "words_per_minute")
#: A pause bias scales pause behaviour; the resulting mean pause stays inside a
#: range a speaking voice can actually produce.
_BIAS_PAUSE_MS_RANGE = (120.0, 900.0)


class VoiceAcousticsError(ValueError):
    """Raised only when a file cannot be read as audio at all."""


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_db(value: float) -> float:
    """dBFS with a hard floor so every level field is always finite."""
    return max(DBFS_FLOOR, 20.0 * math.log10(max(float(value), 1e-6)))


def _to_db_array(values: np.ndarray) -> np.ndarray:
    return np.maximum(DBFS_FLOOR, 20.0 * np.log10(np.maximum(values, 1e-6)))


def _measured_list(source: dict) -> list[str]:
    return sorted(k for k in MEASUREMENT_KEYS if source.get(k) is not None)


def _first(source: dict, *keys: str):
    """First non-None value among ``keys`` (keeps parameters() idempotent)."""
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _frames(samples: np.ndarray, length: int, hop: int) -> np.ndarray | None:
    """Stack fixed-length, hop-spaced analysis frames (None when too short)."""
    if samples.size < length:
        return None
    count = 1 + (samples.size - length) // hop
    if count <= 0:
        return None
    return np.stack([samples[i * hop : i * hop + length] for i in range(count)], axis=0)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of True values."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded) != 0)
    return [(int(s), int(e - s)) for s, e in zip(edges[0::2], edges[1::2])]


def _split_index_runs(indices: np.ndarray, max_gap: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    bounds = np.flatnonzero(np.diff(indices) > max_gap) + 1
    return [part for part in np.split(indices, bounds) if part.size]


def _rolling(values: np.ndarray, width: int, reducer) -> np.ndarray:
    """Edge-padded rolling reduction; output length equals input length."""
    width = max(1, min(width, values.size))
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return reducer(windows, axis=1)


def _resample_to_analysis(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == ANALYSIS_RATE or samples.size < 2:
        return samples
    divisor = math.gcd(int(sample_rate), ANALYSIS_RATE)
    up = ANALYSIS_RATE // divisor
    down = int(sample_rate) // divisor
    return signal.resample_poly(samples, up, down)


def _linear_resample(samples: np.ndarray, out_len: int) -> np.ndarray:
    """Plain linear resample, used only for inputs too short for SOLA."""
    if out_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if samples.size == 0:
        return np.zeros(out_len, dtype=np.float64)
    if samples.size == 1:
        return np.full(out_len, float(samples[0]), dtype=np.float64)
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.linspace(0.0, samples.size - 1.0, out_len)
    return np.interp(target_positions, source_positions, samples)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def measure(path, *, max_seconds: float = 30.0) -> dict:
    """Characterise a reference recording.

    Reads at most ``max_seconds`` of audio, downmixes multi-channel input by
    mean and resamples to 16000 Hz for analysis.  Every documented key is
    always present; genuinely unmeasurable values are ``None`` and the reason
    is appended to ``"notes"``.  Raises :class:`VoiceAcousticsError` only when
    the file cannot be decoded as audio at all.
    """
    source = Path(path)
    notes: list[str] = []

    try:
        info = sf.info(str(source))
        sample_rate = int(info.samplerate or 0)
        total_frames = int(info.frames or 0)
        if sample_rate <= 0:
            raise VoiceAcousticsError(f"unsupported sample rate in {source}")
        if max_seconds and max_seconds > 0:
            requested = int(round(max_seconds * sample_rate))
        else:
            requested = total_frames
        limit = max(0, min(requested, total_frames))
        if limit > 0:
            data, read_rate = sf.read(
                str(source), start=0, stop=limit, always_2d=True, dtype="float64"
            )
            sample_rate = int(read_rate)
        else:
            data = np.zeros((0, 1), dtype=np.float64)
    except VoiceAcousticsError:
        raise
    except Exception as exc:  # soundfile raises RuntimeError/OSError for non-audio
        raise VoiceAcousticsError(f"cannot read audio file: {source}") from exc

    duration_seconds = total_frames / sample_rate if total_frames > 0 else 0.0
    if limit < total_frames:
        notes.append(
            f"analysed the first {limit / sample_rate:.1f} s of {duration_seconds:.1f} s"
        )

    if data.ndim == 1:
        data = data[:, None]
    channels = int(data.shape[1])
    if channels > 1:
        notes.append(f"downmixed {channels} channels by mean")
        mono = data.mean(axis=1)
    elif channels == 1:
        mono = data[:, 0]
    else:
        mono = np.zeros(0, dtype=np.float64)

    analysed_seconds = mono.size / sample_rate
    if mono.size:
        # A DC offset would otherwise masquerade as a very low "pitch" signal.
        offset = float(mono.mean())
        if abs(offset) > 1e-4:
            mono = mono - offset
            notes.append(f"removed {offset:+.4f} DC offset")

    analysis = _resample_to_analysis(mono, sample_rate)

    result: dict = {
        "schema": SCHEMA,
        "method": METHOD,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "analysed_seconds": analysed_seconds,
    }
    for key in MEASUREMENT_KEYS[3:]:
        result[key] = None
    result["measured"] = []
    result["inferred"] = []  # reserved: callers may attach inferred values later
    result["notes"] = notes

    frame_length = int(round(ANALYSIS_RATE * FRAME_MS / 1000.0))
    hop_length = int(round(ANALYSIS_RATE * HOP_MS / 1000.0))
    try:
        _analyse(result, analysis, frame_length, hop_length, notes)
    except Exception as exc:  # never raise on unusual-but-readable audio
        notes.append(f"analysis degraded: {type(exc).__name__}")

    result["measured"] = _measured_list(result)
    return result


def _analyse(
    result: dict, analysis: np.ndarray, frame_length: int, hop_length: int, notes: list[str]
) -> None:
    """Fill ``result`` with frame-domain measurements; never raises outward."""
    if analysis.size:
        result["rms_dbfs"] = _to_db(math.sqrt(float(np.mean(analysis.astype(np.float64) ** 2))))
        result["peak_dbfs"] = _to_db(float(np.max(np.abs(analysis))))

    frames = _frames(analysis, frame_length, hop_length)
    if frames is None:
        notes.append("audio shorter than one analysis frame")
        return

    frame_count = frames.shape[0]
    centred = frames - frames.mean(axis=1, keepdims=True)
    energy = np.einsum("ij,ij->i", centred, centred)
    rms = np.sqrt(energy / frame_length)
    rms_db = _to_db_array(rms)

    # The file's own noise level: the 10th percentile of frame levels, minus a
    # small margin; clamped so digital silence still counts as "below floor".
    noise_db = float(np.percentile(rms_db, 10.0))
    floor_db = max(noise_db - NOISE_MARGIN_DB, DBFS_FLOOR + 1.0)

    result["dynamic_range_db"] = float(
        np.percentile(rms_db, 95.0) - np.percentile(rms_db, 10.0)
    )

    # --- spectra (Hann windowed rFFT of every frame) ----------------------
    window = np.hanning(frame_length)
    spectra = np.fft.rfft(centred * window, axis=1)
    magnitude = np.abs(spectra)
    power = magnitude**2
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / ANALYSIS_RATE)

    total_power = power.sum(axis=1)
    high_band = power[:, freqs >= HIGH_BAND_HZ].sum(axis=1)
    # high-band / total energy ratio: the classic "air vs body" balance.
    high_ratio = np.where(total_power > 0, high_band / np.maximum(total_power, 1e-30), 0.0)

    mag_sum = magnitude.sum(axis=1)
    centroid = np.where(
        mag_sum > 0, (magnitude * freqs).sum(axis=1) / np.maximum(mag_sum, 1e-30), 0.0
    )

    eps = 1e-12
    flatness = np.exp(np.mean(np.log(magnitude + eps), axis=1)) / (
        np.mean(magnitude, axis=1) + eps
    )
    flatness = np.clip(flatness, 0.0, 1.0)

    # --- autocorrelation F0 (batched, so the whole file is one numpy call) --
    n_fft = 2 * frame_length
    padded_spectra = np.fft.rfft(centred, n=n_fft, axis=1)
    autocorr = np.fft.irfft(np.abs(padded_spectra) ** 2, n=n_fft, axis=1)[:, : frame_length + 1]
    r0 = autocorr[:, 0:1]
    ncc = np.where(r0 > 0, autocorr / np.maximum(r0, 1e-30), 0.0)

    min_lag = max(1, int(math.floor(ANALYSIS_RATE / F0_MAX_HZ)))
    max_lag = min(frame_length, int(math.ceil(ANALYSIS_RATE / F0_MIN_HZ)))
    segment = ncc[:, min_lag : max_lag + 1]
    seg_len = segment.shape[1]

    # Pick the *earliest* strong local maximum so a period never wins over its
    # own multiples (that would report an octave-below F0 for pure tones).
    is_local_max = np.zeros(segment.shape, dtype=bool)
    if seg_len >= 3:
        is_local_max[:, 1:-1] = (segment[:, 1:-1] >= segment[:, :-2]) & (
            segment[:, 1:-1] >= segment[:, 2:]
        )
    best = segment.max(axis=1)
    threshold = PEAK_SELECT_RATIO * best
    candidates = np.where(
        is_local_max & (segment >= threshold[:, None]),
        np.arange(seg_len)[None, :],
        seg_len,
    )
    earliest = candidates.min(axis=1)
    chosen = np.where(earliest < seg_len, earliest, segment.argmax(axis=1))
    lag = chosen + min_lag

    voiced = (rms_db > floor_db) & (best >= VOICED_MIN_NCC)
    f0_all = np.full(frame_count, np.nan, dtype=np.float64)
    f0_all[voiced] = ANALYSIS_RATE / lag[voiced]

    voiced_values = f0_all[voiced]
    result["voiced_fraction"] = float(voiced.sum()) / frame_count
    if voiced_values.size:
        result["brightness_hz"] = float(np.median(centroid[voiced]))
    if voiced_values.size >= MIN_VOICED_FRAMES:
        result["f0_median_hz"] = float(np.median(voiced_values))
        result["f0_mean_hz"] = float(np.mean(voiced_values))
        p10 = float(np.percentile(voiced_values, 10.0))
        p90 = float(np.percentile(voiced_values, 90.0))
        result["f0_p10_hz"] = p10
        result["f0_p90_hz"] = p90
        if p10 > 0.0 and p90 > 0.0:
            result["f0_range_semitones"] = float(12.0 * math.log2(p90 / p10))
    else:
        notes.append("fewer than 5 voiced frames: no pitch statistics")

    # --- envelope peaks: syllable rate and onset sharpness ------------------
    envelope = _rolling(rms, ENVELOPE_SMOOTH_FRAMES, np.mean)
    window_frames = max(1, int(round(ADAPTIVE_WINDOW_MS / HOP_MS)))
    local_max = _rolling(envelope, window_frames, np.max)
    local_min = _rolling(envelope, window_frames, np.min)
    adaptive_threshold = local_min + ADAPTIVE_FRACTION * (local_max - local_min)
    # Only regions with a real local dynamic range can carry syllable onsets.
    wide_enough = (_to_db_array(local_max) - _to_db_array(local_min)) >= ADAPTIVE_MIN_RANGE_DB
    adaptive_threshold = np.where(wide_enough, adaptive_threshold, np.inf)
    min_distance = max(1, int(math.ceil(SYLLABLE_MIN_GAP_MS / HOP_MS)))
    peaks, _ = signal.find_peaks(
        envelope, height=adaptive_threshold, distance=min_distance
    )

    minutes = result["analysed_seconds"] / 60.0
    if minutes > 0.0:
        syllables = float(peaks.size) / minutes
        result["syllables_per_minute"] = syllables
        result["words_per_minute"] = syllables / SYLLABLES_PER_WORD

    if peaks.size:
        minima, _ = signal.find_peaks(-envelope)
        rises = []
        for peak in peaks:
            earlier = minima[minima < peak]
            if earlier.size:
                rises.append(_to_db(envelope[peak]) - _to_db(envelope[earlier[-1]]))
        if rises:
            result["articulation_index"] = _clamp(
                float(np.mean(rises)) / ARTICULATION_REFERENCE_DB, 0.0, 1.0
            )

    # --- pauses ------------------------------------------------------------
    pause_frames = int(math.ceil(PAUSE_MIN_MS / HOP_MS))
    paused = rms_db <= floor_db
    long_pauses = [run for run in _true_runs(paused) if run[1] * HOP_MS >= PAUSE_MIN_MS]
    if minutes > 0.0:
        result["pause_count_per_minute"] = float(len(long_pauses)) / minutes
    result["pause_fraction"] = float(paused.sum()) / frame_count
    if long_pauses:
        result["mean_pause_ms"] = float(
            np.mean([length * HOP_MS for _start, length in long_pauses])
        )

    # --- breathiness / rasp -------------------------------------------------
    sounding = rms_db > floor_db
    unvoiced = (~voiced) & sounding
    if voiced.any() and unvoiced.any():
        voiced_ratio = float(np.mean(high_ratio[voiced]))
        unvoiced_ratio = float(np.mean(high_ratio[unvoiced]))
        ratio = unvoiced_ratio / (voiced_ratio + 1e-6)
        result["breathiness"] = _clamp(ratio / (1.0 + ratio), 0.0, 1.0)

    if voiced_values.size:
        if voiced_values.size >= 2:
            jitter = float(np.median(np.abs(np.diff(voiced_values)))) / max(
                float(np.median(voiced_values)), 1e-9
            )
        else:
            jitter = 0.0
        result["rasp"] = _clamp(
            0.5 * min(jitter, 1.0) + 0.5 * float(np.mean(flatness[voiced])), 0.0, 1.0
        )

    # --- sentence endings ---------------------------------------------------
    slopes: list[float] = []
    drops: list[float] = []
    tail_frames = max(1, int(round(SENTENCE_TAIL_MS / HOP_MS)))
    end_frames = max(1, int(round(SENTENCE_END_MS / HOP_MS)))
    for run in _split_index_runs(np.flatnonzero(voiced), pause_frames):
        if run.size < SENTENCE_MIN_FRAMES:
            continue
        span = min(tail_frames, run.size)
        tail = f0_all[run[-span:]]
        times = np.arange(span, dtype=np.float64) * (HOP_MS / 1000.0)
        reference = float(np.median(tail))
        semitones = 12.0 * np.log2(tail / reference)
        if span >= 2:
            slopes.append(float(np.polyfit(times, semitones, 1)[0]))
        run_db = rms_db[run]
        drops.append(float(np.mean(run_db[-min(end_frames, run.size) :]) - np.max(run_db)))
    if slopes:
        result["sentence_end_f0_slope"] = float(np.mean(slopes))
        result["sentence_end_energy_drop_db"] = float(np.mean(drops))


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def compare(target: dict, actual: dict) -> dict:
    """Compare two profiles key by key using the documented tolerances.

    Keys that are missing or ``None`` on either side are skipped entirely: they
    are neither compared nor reported in ``"exceeded"``.
    """
    deltas: dict[str, float] = {}
    relative: dict[str, float] = {}
    exceeded: list[str] = []

    for key, tolerance in COMPARE_RELATIVE_TOLERANCES.items():
        left, right = target.get(key), actual.get(key)
        if left is None or right is None:
            continue
        delta = float(right) - float(left)
        deltas[key] = delta
        if float(left) != 0.0:
            ratio = delta / abs(float(left))
            relative[key] = ratio
            if abs(ratio) > tolerance:
                exceeded.append(key)
        elif abs(delta) > tolerance:
            # No relative reference: fall back to the same number as an
            # absolute tolerance so the key is still compared.
            exceeded.append(key)

    for key, tolerance in COMPARE_ABSOLUTE_TOLERANCES.items():
        left, right = target.get(key), actual.get(key)
        if left is None or right is None:
            continue
        delta = float(right) - float(left)
        deltas[key] = delta
        if float(left) != 0.0:
            relative[key] = delta / abs(float(left))
        if abs(delta) > tolerance:
            exceeded.append(key)

    compared = sorted(deltas)
    return {
        "schema": SCHEMA,
        "deltas": deltas,
        "relative": relative,
        "within_tolerance": not exceeded,
        "exceeded": sorted(set(exceeded)),
        "compared": compared,
    }


# --------------------------------------------------------------------------- #
# shaping plan
# --------------------------------------------------------------------------- #
def plan_shaping(
    target: dict,
    actual: dict,
    *,
    max_semitones: float = 3.0,
    rate_range: tuple[float, float] = (0.85, 1.18),
) -> dict:
    """Derive the pitch/rate nudges that move ``actual`` toward ``target``."""
    reasons: list[str] = []

    target_f0, actual_f0 = target.get("f0_median_hz"), actual.get("f0_median_hz")
    if target_f0 and actual_f0:
        requested_semitones = float(12.0 * math.log2(float(target_f0) / float(actual_f0)))
        if abs(requested_semitones) > 0.0:
            reasons.append(
                f"median pitch {float(actual_f0):.1f} Hz -> {float(target_f0):.1f} Hz "
                f"needs {requested_semitones:+.2f} semitones"
            )
    else:
        requested_semitones = 0.0
        reasons.append("median pitch could not be derived: f0_median_hz missing")

    target_wpm, actual_wpm = target.get("words_per_minute"), actual.get("words_per_minute")
    if target_wpm and actual_wpm:
        requested_rate = float(target_wpm) / float(actual_wpm)
        if abs(requested_rate - 1.0) > 0.0:
            reasons.append(
                f"speaking rate {float(actual_wpm):.0f} wpm -> {float(target_wpm):.0f} wpm "
                f"needs x{requested_rate:.3f}"
            )
    else:
        requested_rate = 1.0
        reasons.append("speaking rate could not be derived: words_per_minute missing")

    low, high = float(rate_range[0]), float(rate_range[1])
    semitones = _clamp(requested_semitones, -abs(max_semitones), abs(max_semitones))
    rate = _clamp(requested_rate, low, high)
    clamped = semitones != requested_semitones or rate != requested_rate
    if semitones != requested_semitones:
        reasons.append(
            f"pitch clamped to {semitones:+.2f} semitones (requested {requested_semitones:+.2f})"
        )
    if rate != requested_rate:
        reasons.append(f"rate clamped to x{rate:.3f} (requested x{requested_rate:.3f})")

    applied_to = []
    if abs(semitones) >= _PLAN_PITCH_FLOOR:
        applied_to.append("pitch")
    if abs(rate - 1.0) >= _PLAN_RATE_FLOOR:
        applied_to.append("rate")

    return {
        "schema": SCHEMA,
        "semitones": float(semitones),
        "rate": float(rate),
        "clamped": bool(clamped),
        "requested": {"semitones": float(requested_semitones), "rate": float(requested_rate)},
        "reasons": reasons,
        "applied_to": applied_to,
    }


# --------------------------------------------------------------------------- #
# shaping
# --------------------------------------------------------------------------- #
#: Rate metrics a time-stretch invalidates: a SOLA stretch duplicates waveform
#: segments, and envelope-peak syllable detection counts those duplicates as new
#: syllables, so a *slowed* take measures faster. These keys are therefore taken
#: from the applied factor rather than re-detected on the stretched audio.
_STRETCH_DERIVED_KEYS = ("syllables_per_minute", "words_per_minute")

#: The keys the bounded shaping pass can actually move. A comparison over the full
#: target profile is not a reachability verdict: a documented prior carries keys
#: (peak level, voiced fraction, sentence-ending energy drop) that no pitch/rate
#: transform touches, so counting them as "not reached" would report the actuator's
#: reach as the run's failure. Reachability is therefore evaluated over this set,
#: and everything else is reported as out of the actuator's scope.
CONTROLLABLE_KEYS = ("f0_median_hz", "f0_mean_hz", "f0_p10_hz", "f0_p90_hz",
                     "words_per_minute", "syllables_per_minute", "pause_count_per_minute")


def achievement(comparison: dict) -> dict:
    """Split a target comparison into reached, shortfall and out-of-scope.

    Three outcomes, kept three-valued: a key the actuator cannot move is neither
    a success nor a failure, and collapsing it to either one makes the verdict a
    claim the measurement never made.
    """
    exceeded = set(comparison.get("exceeded") or [])
    compared = set(comparison.get("compared") or [])
    controllable = set(CONTROLLABLE_KEYS)
    shortfall = sorted(exceeded & controllable)
    out_of_scope = sorted(exceeded - controllable)
    reached = sorted((compared & controllable) - exceeded)
    measurable = bool(reached or shortfall)
    # A comparison that touched no controllable key has not established reach, so
    # its verdict is its own value rather than the flattering default: an
    # empty population is never a pass.
    verdict = "shortfall" if shortfall else ("reached" if measurable else "could_not_measure")
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "reached": reached,
        "shortfall": shortfall,
        "out_of_scope": out_of_scope,
        "reached_count": len(reached),
        "shortfall_count": len(shortfall),
        "out_of_scope_count": len(out_of_scope),
        # The verdict a caller may act on: did the actuator reach the targets it
        # was given, where it had any reach?
        "reached_target": verdict == "reached",
        "measurable": measurable,
    }


def reconcile_after(before: dict, after: dict, shaping: dict) -> tuple[dict, dict]:
    """Keep what the stretch can measure and derive what it cannot.

    Returns ``(profile, reconciliation)``. Per-frame measurements (pitch, level,
    brightness, texture) survive time-stretching and are kept as measured. The
    syllable-rate metrics do not: the stretch is itself the signal they detect,
    so they are recomputed from the pre-stretch rate and the applied factor —
    which the output duration independently confirms.
    """
    rate = float((shaping.get("applied") or {}).get("rate", 1.0)) or 1.0
    source_seconds = float(shaping.get("source_seconds") or 0.0)
    output_seconds = float(shaping.get("output_seconds") or 0.0)
    observed = (source_seconds / output_seconds) if output_seconds > 0.0 else rate
    profile = dict(after)
    notes = list(profile.get("notes") or [])
    replaced = {}
    for key in _STRETCH_DERIVED_KEYS:
        base = before.get(key)
        if base is None:
            continue
        replaced[key] = {"measured": after.get(key), "derived": float(base) * rate}
        profile[key] = float(base) * rate
    # A pause count is a per-minute figure and the minute got longer, so the
    # applied factor is the authority there too.
    if before.get("pause_count_per_minute") is not None:
        profile["pause_count_per_minute"] = float(before["pause_count_per_minute"]) / rate
    if replaced:
        notes.append(
            "syllable-rate metrics are derived from the pre-stretch measurement and the applied "
            "factor rather than re-detected: envelope-peak detection counts the time-stretch's own "
            "segment duplication as new syllables")
    profile["notes"] = notes
    profile["measured"] = _measured_list(profile)
    return profile, {"schema": SCHEMA, "rate_applied": rate, "rate_observed": observed,
                     "rate_agrees": abs(observed - rate) <= 0.02 * rate,
                     "replaced": replaced, "derived_keys": sorted(replaced)}


def _rational_approx(value: float, max_denominator: int = 512) -> tuple[int, int]:
    """Continued-fraction rational approximation (stdlib only, no Fraction)."""
    if value <= 0.0 or not math.isfinite(value):
        return 1, 1
    x = float(value)
    h_prev, h_prev2 = 1, 0
    k_prev, k_prev2 = 0, 1
    while True:
        whole = int(math.floor(x))
        h = whole * h_prev + h_prev2
        k = whole * k_prev + k_prev2
        if k > max_denominator or k <= 0:
            break
        h_prev2, k_prev2 = h_prev, k_prev
        h_prev, k_prev = h, k
        remainder = x - whole
        if remainder <= 1e-12 or not math.isfinite(remainder):
            break
        x = 1.0 / remainder
    return max(1, h_prev), max(1, k_prev)


def _best_alignment(
    samples: np.ndarray, reference: np.ndarray, low: int, high: int, length: int
) -> int:
    """Largest normalised cross-correlation of ``reference`` over [low, high]."""
    if high <= low or length <= 0:
        return low
    starts = np.arange(low, high + 1)
    blocks = samples[starts[:, None] + np.arange(length)[None, :]]
    ref = reference[:length]
    if ref.size < length:
        ref = np.pad(ref, (0, length - ref.size))
    numerator = blocks @ ref
    denominator = np.sqrt((blocks * blocks).sum(axis=1) * float(ref @ ref)) + 1e-12
    return int(starts[int(np.argmax(numerator / denominator))])


def _sola_channel(samples: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """Synchronous overlap-add time stretch.

    ``rate`` is a speed factor: the result lasts ``len(samples) / rate`` samples.
    The synthesis hop is fixed at 7.5 ms while the analysis hop scales with
    ``rate``; the read cursor is re-derived from the ideal (drift-free) position
    every frame and only then corrected by a bounded +/-6 ms normalised
    cross-correlation search.  A Hann window plus an overlap weight accumulator
    keeps the level constant.
    """
    count = samples.size
    window_length = max(4, int(round(sample_rate * SOLA_WINDOW_MS / 1000.0)))
    synth_hop = max(1, int(round(sample_rate * SOLA_SYNTH_HOP_MS / 1000.0)))
    search = max(1, int(round(sample_rate * SOLA_SEARCH_MS / 1000.0)))
    out_len = int(round(count / rate))
    if out_len <= 0 or count < window_length:
        return _linear_resample(samples, out_len)

    window = np.hanning(window_length)
    output = np.zeros(out_len + window_length, dtype=np.float64)
    weight = np.zeros_like(output)
    overlap = max(1, window_length - synth_hop)
    step = max(1, int(round(synth_hop * rate)))
    last_start = max(0, count - window_length)

    read_from = 0
    index = 0
    while True:
        out_at = index * synth_hop
        if out_at >= out_len:
            break
        ideal = int(round(out_at * rate))
        if ideal >= count:
            break
        if index == 0:
            position = 0
        else:
            predicted = read_from + step
            low = max(0, predicted - search, ideal - 2 * search)
            high = min(last_start, predicted + search, ideal + 2 * search)
            if high < low:
                position = max(0, min(ideal, last_start))
            else:
                position = _best_alignment(
                    samples, output[out_at : out_at + overlap], low, high, overlap
                )
        block = samples[position : position + window_length]
        if block.size < window_length:
            block = np.pad(block, (0, window_length - block.size))
        output[out_at : out_at + window_length] += block * window
        weight[out_at : out_at + window_length] += window
        read_from = position
        index += 1

    output /= np.maximum(weight, 1e-6)
    return output[:out_len]


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if samples.size == length:
        return samples
    if samples.size > length:
        return samples[:length]
    return np.pad(samples, (0, length - samples.size))


def _shape_channel(samples: np.ndarray, sample_rate: int, semitones: float, rate: float,
                   out_len: int) -> np.ndarray:
    shaped = samples
    if abs(semitones) >= _TRIVIAL:
        factor = 2.0 ** (semitones / 12.0)
        up, down = _rational_approx(factor)
        # Playing back a signal with fewer samples per waveform cycle raises the
        # pitch: resample to len/factor (scipy wants up/down of 1/factor), then
        # stretch back to the original duration with a matching SOLA pass.
        shaped = signal.resample_poly(shaped, down, up)
        shaped = _sola_channel(shaped, sample_rate, 1.0 / factor)
    if abs(rate - 1.0) >= _TRIVIAL:
        shaped = _sola_channel(shaped, sample_rate, rate)
    return _fit_length(shaped, out_len)


def _source_subtype(path: Path) -> str:
    try:
        subtype = str(sf.info(str(path)).subtype or "")
    except Exception:
        return "FLOAT"
    if subtype in {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        return subtype
    return "FLOAT"


def shape(src, dst, *, semitones: float = 0.0, rate: float = 1.0) -> dict:
    """Write a pitch/rate shaped WAV copy of ``src`` to ``dst``.

    Invariant: after the pitch step restores its own duration, the final file
    lasts ``source_seconds / rate`` (within one sample, i.e. far inside 2%).
    Sample rate and channel count are preserved; the source subtype is kept
    when it is a valid WAV subtype.  When both ``semitones`` and ``rate`` are
    trivial the decoded samples are written unchanged and ``changed`` is False.
    """
    if not math.isfinite(rate) or rate <= 0.0:
        raise VoiceAcousticsError(f"rate must be a positive finite number, got {rate}")
    if not math.isfinite(semitones):
        raise VoiceAcousticsError(f"semitones must be finite, got {semitones}")

    source = Path(src)
    target = Path(dst)
    try:
        data, sample_rate = sf.read(str(source), always_2d=True, dtype="float64")
    except Exception as exc:
        raise VoiceAcousticsError(f"cannot read audio file: {source}") from exc
    if data.size == 0:
        raise VoiceAcousticsError(f"cannot read audio file: {source}")
    sample_rate = int(sample_rate)
    frame_count = data.shape[0]
    channels = data.shape[1]

    out_len = int(round(frame_count / rate))
    shaped = np.empty((out_len, channels), dtype=np.float64)
    for index in range(channels):
        shaped[:, index] = _shape_channel(
            np.ascontiguousarray(data[:, index]), sample_rate, float(semitones), float(rate),
            out_len
        )

    peak = float(np.max(np.abs(shaped))) if shaped.size else 0.0
    normalised = peak > NORMALISE_CEILING
    if normalised:
        shaped = shaped / peak

    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), shaped, sample_rate, subtype=_source_subtype(source))

    changed = abs(semitones) >= _TRIVIAL or abs(rate - 1.0) >= _TRIVIAL
    return {
        "schema": SCHEMA,
        "applied": {"semitones": float(semitones), "rate": float(rate)},
        "source_seconds": frame_count / sample_rate,
        "output_seconds": out_len / sample_rate,
        "sample_rate": sample_rate,
        "changed": bool(changed),
        "normalised": bool(normalised),
    }


# --------------------------------------------------------------------------- #
# derived generation parameters
# --------------------------------------------------------------------------- #
def _resonance(f0, brightness) -> str:
    if f0 is not None:
        if f0 < RESONANCE_DEEP_F0_HZ:
            if brightness is not None and brightness >= RESONANCE_BRIGHT_HZ:
                return "warm-rounded"
            return "deep-chested"
        if f0 < RESONANCE_LIGHT_F0_HZ:
            if brightness is not None and brightness >= RESONANCE_BRIGHT_HZ:
                return "bright-forward"
            return "warm-rounded"
        if brightness is not None and brightness < RESONANCE_DARK_HZ:
            return "warm-rounded"
        return "light"
    if brightness is not None and brightness >= RESONANCE_BRIGHT_HZ:
        return "bright-forward"
    return "warm-rounded"


def _vocal_weight(f0, dynamic_range) -> str:
    score = 0
    if f0 is not None:
        if f0 < VOCAL_WEIGHT_HEAVY_F0_HZ:
            score += 1
        elif f0 >= VOCAL_WEIGHT_LIGHT_F0_HZ:
            score -= 1
    if dynamic_range is not None:
        if dynamic_range >= VOCAL_WEIGHT_HEAVY_RANGE_DB:
            score += 1
        elif dynamic_range < VOCAL_WEIGHT_LIGHT_RANGE_DB:
            score -= 1
    if score >= 1:
        return "heavy"
    if score <= -1:
        return "light"
    return "medium"


def _age_impression(f0, rasp):
    if f0 is None:
        return None
    if f0 < AGE_OLD_F0_HZ:
        level = 3
    elif f0 < AGE_MATURE_F0_HZ:
        level = 2
    elif f0 < AGE_ADULT_F0_HZ:
        level = 1
    else:
        level = 0
    if rasp is not None and rasp >= AGE_RASP_BUMP:
        level += 1
    level = min(level, 3)
    return ["young adult", "adult", "mature adult", "older adult"][level]


def parameters(profile: dict) -> dict:
    """Map a measured (or previously derived) profile onto generation values.

    Both the raw measurement keys and the derived keys are accepted so the
    function is idempotent: ``parameters(parameters(p)) == parameters(p)``.
    """
    f0 = _first(profile, "f0_median_hz", "pitch_target_hz")
    f0_range = _first(profile, "f0_range_semitones", "pitch_range_semitones")
    words_per_minute = _first(profile, "words_per_minute", "rate_words_per_minute")
    pause_ms = _first(profile, "mean_pause_ms", "pause_ms")
    pause_rate = _first(profile, "pause_count_per_minute", "pause_count_per_minute")
    energy = _first(profile, "rms_dbfs", "energy_dbfs")
    dynamic_range = _first(profile, "dynamic_range_db", "dynamic_range_db")
    brightness = _first(profile, "brightness_hz", "brightness_hz")
    breathiness = _first(profile, "breathiness", "breathiness")
    rasp = _first(profile, "rasp", "rasp")
    articulation = _first(profile, "articulation_index", "articulation")
    slope = _first(profile, "sentence_end_f0_slope", "sentence_end_f0_slope")

    if words_per_minute is not None:
        pace = _clamp(
            float(words_per_minute) / ENGINE_PACE_REFERENCE_WPM,
            ENGINE_PACE_RANGE[0],
            ENGINE_PACE_RANGE[1],
        )
    else:
        pace = 1.0
    if energy is not None:
        volume = _clamp(
            1.0 + (float(energy) - ENGINE_VOLUME_REFERENCE_DBFS) / ENGINE_VOLUME_SWING_DB,
            ENGINE_VOLUME_RANGE[0],
            ENGINE_VOLUME_RANGE[1],
        )
    else:
        volume = 1.0

    return {
        "schema": SCHEMA,
        "pitch_target_hz": None if f0 is None else float(f0),
        "pitch_range_semitones": None if f0_range is None else float(f0_range),
        "rate_words_per_minute": None if words_per_minute is None else float(words_per_minute),
        "pause_ms": None if pause_ms is None else float(pause_ms),
        "pause_count_per_minute": None if pause_rate is None else float(pause_rate),
        "energy_dbfs": None if energy is None else float(energy),
        "dynamic_range_db": None if dynamic_range is None else float(dynamic_range),
        "brightness_hz": None if brightness is None else float(brightness),
        "breathiness": None if breathiness is None else float(breathiness),
        "rasp": None if rasp is None else float(rasp),
        "articulation": None if articulation is None else float(articulation),
        "sentence_end_f0_slope": None if slope is None else float(slope),
        "resonance": _resonance(f0, brightness),
        "vocal_weight": _vocal_weight(f0, dynamic_range),
        "age_impression": _age_impression(f0, rasp),
        # The acoustic shaping pass already applied any pitch shift, so the
        # engine hint stays at zero by construction.
        "engine_hints": {
            "pace": float(pace),
            "volume": float(volume),
            "pitch_shift_semitones": 0.0,
        },
    }


def instruction_fragment(profile: dict) -> str:
    """Numeric, performer-free description of the acoustic targets."""
    params = parameters(profile)
    clauses: list[str] = []

    if params["pitch_target_hz"] is not None:
        if params["pitch_range_semitones"] is not None:
            clauses.append(
                f"median pitch {params['pitch_target_hz']:.0f} Hz with a "
                f"{params['pitch_range_semitones']:.1f} semitone range"
            )
        else:
            clauses.append(f"median pitch {params['pitch_target_hz']:.0f} Hz")
    if params["rate_words_per_minute"] is not None:
        clauses.append(f"speaking rate {params['rate_words_per_minute']:.0f} words per minute")
    if params["pause_ms"] is not None and params["pause_count_per_minute"] is not None:
        clauses.append(
            f"mean pause {params['pause_ms']:.0f} ms at "
            f"{params['pause_count_per_minute']:.0f} pauses per minute"
        )
    elif params["pause_ms"] is not None:
        clauses.append(f"mean pause {params['pause_ms']:.0f} ms")
    if params["energy_dbfs"] is not None:
        if params["dynamic_range_db"] is not None:
            clauses.append(
                f"level {params['energy_dbfs']:.1f} dBFS with "
                f"{params['dynamic_range_db']:.1f} dB dynamic range"
            )
        else:
            clauses.append(f"level {params['energy_dbfs']:.1f} dBFS")
    if params["brightness_hz"] is not None:
        clauses.append(f"spectral brightness {params['brightness_hz']:.0f} Hz")
    if params["breathiness"] is not None:
        clauses.append(f"breathiness {params['breathiness']:.2f}")
    if params["rasp"] is not None:
        clauses.append(f"rasp {params['rasp']:.2f}")
    if params["articulation"] is not None:
        clauses.append(f"articulation {params['articulation']:.2f}")
    if params["sentence_end_f0_slope"] is not None:
        slope = params["sentence_end_f0_slope"]
        verb = "fall" if slope <= 0.0 else "rise"
        clauses.append(f"sentence endings {verb} {abs(slope):.1f} semitones per second")

    if not clauses:
        return ""

    clauses.append(f"{params['resonance']} resonance")
    clauses.append(f"{params['vocal_weight']} vocal weight")
    if params["age_impression"] is not None:
        clauses.append(f"{params['age_impression']} impression")
    return "Measured acoustic targets: " + "; ".join(clauses) + "."


# --------------------------------------------------------------------------- #
# casting-concept bias
# --------------------------------------------------------------------------- #
def _round_bias(bias: dict) -> dict:
    rounded = {}
    for key, value in bias.items():
        if isinstance(value, bool) or value is None:
            rounded[key] = value
        elif isinstance(value, (int, float)):
            rounded[key] = round(float(value), 4)
        else:
            rounded[key] = value
    return rounded


def _bias_summary(bias: dict) -> str:
    if not bias:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in _round_bias(bias).items())


def merge(profile: dict, bias: dict) -> dict:
    """Apply a casting-concept bias to a measured profile.

    Returns a NEW mapping; the profile is never mutated.  Anything the bias
    names that the profile does not actually carry is reported in ``"notes"``
    instead of being invented.
    """
    bias = dict(bias or {})
    merged = dict(profile)
    notes = list(merged.get("notes") or [])
    unavailable: list[str] = []

    def scale_keys(keys, factor, low=None, high=None):
        touched = False
        for key in keys:
            value = merged.get(key)
            if value is None:
                continue
            scaled = float(value) * factor
            if low is not None or high is not None:
                scaled = _clamp(scaled, low if low is not None else scaled,
                                high if high is not None else scaled)
            merged[key] = scaled
            touched = True
        return touched

    if "pitch_semitones" in bias:
        factor = 2.0 ** (float(bias["pitch_semitones"]) / 12.0)
        touched = scale_keys(_BIAS_PITCH_KEYS, factor)
        if merged.get("f0_range_semitones") is not None:
            merged["f0_range_semitones"] = float(merged["f0_range_semitones"]) * abs(factor)
            touched = True
        if not touched:
            unavailable.append("pitch_semitones")

    if "rate" in bias:
        if not scale_keys(_BIAS_RATE_KEYS, float(bias["rate"])):
            unavailable.append("rate")

    if "energy_db" in bias:
        if merged.get("rms_dbfs") is None:
            unavailable.append("energy_db")
        else:
            merged["rms_dbfs"] = float(merged["rms_dbfs"]) + float(bias["energy_db"])

    if "brightness" in bias:
        if merged.get("brightness_hz") is None:
            unavailable.append("brightness")
        else:
            merged["brightness_hz"] = float(merged["brightness_hz"]) * float(bias["brightness"])

    if "breathiness" in bias:
        if merged.get("breathiness") is None:
            unavailable.append("breathiness")
        else:
            merged["breathiness"] = _clamp(
                float(merged["breathiness"]) * float(bias["breathiness"]), 0.0, 1.0
            )

    if "rasp" in bias:
        if merged.get("rasp") is None:
            unavailable.append("rasp")
        else:
            merged["rasp"] = _clamp(float(merged["rasp"]) * float(bias["rasp"]), 0.0, 1.0)

    if "articulation" in bias:
        if merged.get("articulation_index") is None:
            unavailable.append("articulation")
        else:
            merged["articulation_index"] = _clamp(
                float(merged["articulation_index"]) * float(bias["articulation"]), 0.0, 1.0
            )

    if "pause_ms" in bias:
        # Bounds are on the resulting pause behaviour, not on a 0-1 ratio: these
        # keys carry milliseconds and a per-minute count, so clamping them to the
        # bias factor's own scale would destroy the measurement.
        factor = float(bias["pause_ms"])
        touched = False
        if merged.get("mean_pause_ms") is not None:
            merged["mean_pause_ms"] = _clamp(float(merged["mean_pause_ms"]) * factor,
                                             _BIAS_PAUSE_MS_RANGE[0], _BIAS_PAUSE_MS_RANGE[1])
            touched = True
        if merged.get("pause_count_per_minute") is not None:
            merged["pause_count_per_minute"] = max(0.1, float(merged["pause_count_per_minute"]) / factor)
            touched = True
        if merged.get("pause_fraction") is not None:
            merged["pause_fraction"] = _clamp(float(merged["pause_fraction"]) * factor, 0.01, 0.6)
            touched = True
        if not touched:
            unavailable.append("pause_ms")

    if "dynamic_range_db" in bias:
        if merged.get("dynamic_range_db") is None:
            unavailable.append("dynamic_range_db")
        else:
            merged["dynamic_range_db"] = max(
                float(merged["dynamic_range_db"]) * float(bias["dynamic_range_db"]), 0.1
            )

    known = {
        "pitch_semitones", "rate", "energy_db", "brightness", "breathiness",
        "rasp", "articulation", "pause_ms", "dynamic_range_db",
    }
    for key in bias:
        if key not in known:
            unavailable.append(str(key))

    merged["bias"] = _round_bias(bias)
    notes.append(f"casting-inference bias applied: {_bias_summary(bias)}")
    for key in unavailable:
        notes.append(f"bias '{key}' had no effect: the profile does not carry that measurement")
    merged["notes"] = notes
    merged["measured"] = _measured_list(merged)
    return merged
