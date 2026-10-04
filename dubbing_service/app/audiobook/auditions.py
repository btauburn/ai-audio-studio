"""Bounded automatic auditions and transparent acoustic collision proxies.

These measurements are signal-quality and timbre proxies, not a learned speaker
embedding, a biometric identity test, or a reliable estimate of gender/age.
Generated original references are the only recordings required.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np

from ..models.character import (
    AuditionCandidate, CastCollision, CastingProfile, CharacterCastingPolicy,
)

MEASUREMENT_VERSION = "acoustic-proxy-v1"
AUDITION_TEXTS = {
    "neutral": "The path is clear. We can leave when everyone is ready.",
    "elevated": "Wait! We have to go back. I will not leave anyone behind!",
    "quiet": "I remember that morning. For a moment, everything felt possible.",
}


def candidate_count(profile: CastingProfile, policy: CharacterCastingPolicy) -> int:
    """A hard per-character bound; the orchestrator owns the total budget."""
    if policy.audition_mode == "off" or profile.importance == "minor":
        return 1
    uncertain = bool(profile.review_reasons) or not any(
        d.level == "EXPLICIT" and d.dimension in {"pitch", "texture", "cadence"}
        for d in profile.decisions)
    desired = 3 if uncertain else 2
    return min(policy.max_candidates, desired)


def measure_audio(path: str | Path) -> dict[str, float]:
    """Measure at most 30 seconds; decode via the existing audio dependency."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as source:
        sample_rate = source.samplerate
        if sample_rate <= 0:
            raise ValueError("Invalid audio sample rate")
        total_duration = len(source) / sample_rate
        audio = source.read(frames=min(len(source), sample_rate * 30), dtype="float64", always_2d=True).mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audition audio")
    peak = float(np.max(np.abs(audio)))
    clipping = float(np.mean(np.abs(audio) >= 0.995))
    rms = float(np.sqrt(np.mean(audio * audio)))
    # Rational anti-aliased resampling makes thresholds sample-rate invariant.
    if sample_rate > 16000:
        from scipy.signal import resample_poly
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
        sample_rate = 16000
    frame_length = max(256, int(sample_rate * 0.064))
    hop = max(128, int(sample_rate * 0.032))
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    starts = range(0, len(audio) - frame_length + 1, hop)
    frame_rms, pitches, harmonicities, centroids, flatness, zcrs, bands = [], [], [], [], [], [], []
    window = np.hanning(frame_length)
    frequencies = np.fft.rfftfreq(frame_length, 1 / sample_rate)
    silence_threshold = max(0.003, rms * 0.08)
    for start in starts:
        frame = audio[start:start + frame_length]
        energy = float(np.sqrt(np.mean(frame * frame)))
        frame_rms.append(energy)
        if energy < silence_threshold:
            continue
        centered = frame - frame.mean()
        spectrum = np.abs(np.fft.rfft(centered * window)) ** 2
        total = float(spectrum.sum()) + 1e-16
        centroids.append(float(np.sum(frequencies * spectrum) / total))
        flatness.append(float(np.exp(np.mean(np.log(spectrum + 1e-16))) / (spectrum.mean() + 1e-16)))
        zcrs.append(float(np.mean(centered[:-1] * centered[1:] < 0)))
        bands.append([float(spectrum[(frequencies >= lo) & (frequencies < hi)].sum() / total)
                      for lo, hi in ((0, 500), (500, 1500), (1500, 4000), (4000, 8001))])
        correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        if correlation[0] <= 1e-12:
            continue
        correlation /= correlation[0]
        low, high = max(1, int(sample_rate / 550)), min(len(correlation) - 2, int(sample_rate / 55))
        peaks = [lag for lag in range(low + 1, high)
                 if correlation[lag] > correlation[lag - 1] and correlation[lag] >= correlation[lag + 1]]
        if peaks:
            maximum = max(correlation[lag] for lag in peaks)
            # Prefer the earliest strong periodic peak over octave multiples.
            strong = [lag for lag in peaks if correlation[lag] >= max(0.35, maximum * 0.9)]
            if strong:
                lag = strong[0]
                pitches.append(sample_rate / lag)
                harmonicities.append(float(correlation[lag]))
    median = lambda values: float(np.median(values)) if len(values) else 0.0
    result = {
        "duration_seconds": float(total_duration), "measured_seconds": min(30.0, float(total_duration)),
        "peak": peak, "rms": rms, "clipping_fraction": clipping,
        "silence_fraction": float(np.mean(np.array(frame_rms) < silence_threshold)),
        "pitch_hz": median(pitches), "pitch_spread_semitones": float(np.std(12 * np.log2(pitches))) if pitches else 0.0,
        "voiced_fraction": len(pitches) / max(1, len(frame_rms)),
        "harmonicity": median(harmonicities), "spectral_centroid_hz": median(centroids),
        "spectral_flatness": median(flatness), "zero_crossing_rate": median(zcrs),
    }
    for index in range(4):
        result[f"band_{index}"] = median([band[index] for band in bands])
    return result


def acoustic_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Comparable [0,1] proximity; returns zero when there is no useful signal."""
    if min(left.get("rms", 0), right.get("rms", 0)) < 0.003:
        return 0.0
    distances, weights = [], []
    if left.get("pitch_hz", 0) > 0 and right.get("pitch_hz", 0) > 0:
        semitones = abs(12 * math.log2(left["pitch_hz"] / right["pitch_hz"]))
        distances.append(semitones / 6)
        weights.append(0.5)
    if left.get("spectral_centroid_hz", 0) > 0 and right.get("spectral_centroid_hz", 0) > 0:
        distances.append(abs(math.log(left["spectral_centroid_hz"] / right["spectral_centroid_hz"])) / 0.7)
        weights.append(0.2)
    if all(f"band_{i}" in left and f"band_{i}" in right for i in range(4)):
        distances.append(sum(abs(left[f"band_{i}"] - right[f"band_{i}"]) for i in range(4)) / 0.7)
        weights.append(0.3)
    if not weights:
        return 0.0
    return round(math.exp(-sum(w * d * d for w, d in zip(weights, distances)) / sum(weights)), 6)


def _representative(candidate: AuditionCandidate) -> dict[str, float]:
    if "neutral" in candidate.measurements:
        return candidate.measurements["neutral"]
    if not candidate.measurements:
        return {}
    keys = set.intersection(*(set(item) for item in candidate.measurements.values()))
    return {key: float(np.median([item[key] for item in candidate.measurements.values()])) for key in keys}


def _existing_list(existing_candidates) -> list[AuditionCandidate]:
    if isinstance(existing_candidates, Mapping):
        existing_candidates = existing_candidates.values()
    result = []
    for item in existing_candidates:
        result.extend(item if isinstance(item, (list, tuple)) else [item])
    return result


def _shared_weight(weights: Mapping, left: str, right: str) -> float:
    value = weights.get((left, right), weights.get((right, left), weights.get(right, 0)))
    if isinstance(value, Mapping):
        value = value.get(left, 0)
    return max(0.0, min(1.0, float(value)))


def evaluate_candidates(
    candidates: Sequence[AuditionCandidate], profile: CastingProfile,
    existing_candidates: Sequence[AuditionCandidate] | Mapping,
    shared_scene_weights: Mapping | None = None,
) -> tuple[AuditionCandidate | None, list[CastCollision]]:
    """Measure, rank, and choose acceptable audio; flag uncertain decisions.

    Mutates candidate measurement/score/review/selected fields for persistence.
    Returned collisions concern the selected candidate, or the best rejected
    candidate when no satisfactory voice exists. Identity itself is untouched.
    """
    weights = shared_scene_weights if shared_scene_weights is not None else profile.contrast_with
    prior = _existing_list(existing_candidates)
    evaluated = []
    for candidate in candidates:
        candidate.selected = False
        if candidate.error:
            candidate.score = 0
            continue
        reasons = list(candidate.review_reasons)
        try:
            for kind, path in candidate.samples.items():
                if kind not in candidate.measurements:
                    candidate.measurements[kind] = measure_audio(path)
        except (OSError, ValueError, RuntimeError) as exc:
            candidate.error = f"Audition measurement failed: {exc}"
            candidate.score = 0
            continue
        if not candidate.measurements:
            candidate.error = "No audition audio or measurements available"
            candidate.score = 0
            continue
        score, failed = 1.0, False
        for kind, values in candidate.measurements.items():
            if values.get("rms", 0) < 0.003 or values.get("silence_fraction", 1) > 0.85 or values.get("duration_seconds", 0) < 0.15:
                reasons.append(f"{kind}: silent or unusably short audition")
                failed = True
            if values.get("clipping_fraction", 0) > 0.005:
                reasons.append(f"{kind}: severe clipping")
                failed = True
            score -= min(0.2, values.get("clipping_fraction", 0) * 5)
        reference = _representative(candidate)
        for decision in profile.decisions:
            if decision.dimension != "pitch" or decision.level not in {"EXPLICIT", "SUPPORTED"}:
                continue
            pitch = reference.get("pitch_hz", 0)
            if pitch and ((decision.value == "high" and pitch < 120) or (decision.value == "low" and pitch > 280)):
                reasons.append("Severe register mismatch with explicit voice evidence (pitch proxy)")
                failed = True
        # Large shifts in pitch alone can be acting. Require timbre movement too.
        for kind, values in candidate.measurements.items():
            left_pitch, right_pitch = reference.get("pitch_hz", 0), values.get("pitch_hz", 0)
            if left_pitch and right_pitch and max(left_pitch, right_pitch) / min(left_pitch, right_pitch) > 2.5:
                if acoustic_similarity(reference, values) < 0.25:
                    reasons.append(f"{kind}: possible identity instability across performances (acoustic proxy)")
                    failed = True
        collisions = []
        for existing in prior:
            if existing.role_id == candidate.role_id or existing.error:
                continue
            for kind, path in existing.samples.items():
                if kind not in existing.measurements:
                    try:
                        existing.measurements[kind] = measure_audio(path)
                    except (OSError, ValueError, RuntimeError):
                        pass
            similarity = acoustic_similarity(reference, _representative(existing))
            shared = _shared_weight(weights, candidate.role_id, existing.role_id)
            threshold = round(0.97 - 0.12 * shared, 4)
            # Penalize approaching collisions even below review threshold.
            score -= max(0, similarity - 0.65) * (0.25 + shared * 0.5)
            if similarity >= threshold:
                collisions.append(CastCollision(left=candidate.role_id, right=existing.role_id,
                    similarity=similarity, threshold=threshold, shared_scene_weight=shared,
                    reason="Acoustic proxies are unusually similar; listening review required, especially for shared scenes"))
                reasons.append(f"Cast collision with {existing.role_id} (acoustic proxy)")
                if profile.importance in {"major", "recurring"}:
                    failed = True
        candidate.score = round(max(0, score), 6)
        candidate.review_reasons = list(dict.fromkeys(reasons))
        evaluated.append((not failed, candidate.score, candidate, collisions))
    if not evaluated:
        return None, []
    accepted, _, best, collisions = max(evaluated, key=lambda item: (item[0], item[1]))
    if not accepted:
        return None, collisions
    best.selected = True
    return best, collisions
