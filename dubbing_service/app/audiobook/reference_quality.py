"""Bounded offline CPU reference checks. Technical evidence never approves a voice."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import re
import threading
import time
import unicodedata
from pathlib import Path

from . import reference_transcripts as asr

_LOADED_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VERSION = "reference-quality-v1"
SETTINGS = {"vad_rate": 16000, "vad_threshold": .5, "min_speech_ms": 250,
            "min_silence_ms": 200, "speech_pad_ms": 30, "max_bytes": 24_000_000,
            "duration_min": 3, "duration_max": 30.01, "clip_fraction_max": .001,
            "speech_seconds_min": 1.5, "speech_fraction_min": .25,
            "background_ratio_review_below_db": 8, "word_confidence_mean_min": .85, "low_word_probability": .5,
            "low_word_fraction_max": .1, "wer_pass_max": .1, "wer_fail_min": .5,
            "uncovered_speech_max_seconds": .65, "word_vad_coverage_min": .6}
_lock = threading.Lock()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return asr._digest(value)


def _pipeline():
    if _sha(Path(__file__).read_bytes()) != _LOADED_SOURCE_SHA or _sha(Path(asr.__file__).read_bytes()) != asr._LOADED_SOURCE_SHA:
        raise ValueError("Reference checker source changed; restart Studio before verification")
    from faster_whisper.utils import get_assets_path
    versions = {p: importlib.metadata.version(p) for p in
                ("faster-whisper", "ctranslate2", "onnxruntime", "numpy", "scipy", "soundfile", "av")}
    vad = Path(get_assets_path()) / "silero_vad_v6.onnx"
    _, model_sha = asr._model()
    return {"version": VERSION, "source_sha256": _sha(Path(__file__).read_bytes()),
            "asr_source_sha256": _sha(Path(asr.__file__).read_bytes()),
            "versions": versions, "vad_model_sha256": _sha(vad.read_bytes()),
            "asr_model_sha256": model_sha, "settings": SETTINGS, "asr_settings": asr.SETTINGS}


def _tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"\w+(?:'\w+)?", text)


def word_error_rate(expected, actual):
    a, b = _tokens(expected), _tokens(actual)
    if not a:
        return None
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    return previous[-1] / len(a)


def _speech(wave, rate):
    import numpy as np
    from scipy.signal import resample_poly
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    factor = math.gcd(rate, SETTINGS["vad_rate"])
    samples = resample_poly(wave, SETTINGS["vad_rate"] // factor, rate // factor).astype(np.float32)
    spans = get_speech_timestamps(samples, VadOptions(
        threshold=SETTINGS["vad_threshold"], min_speech_duration_ms=SETTINGS["min_speech_ms"],
        min_silence_duration_ms=SETTINGS["min_silence_ms"], speech_pad_ms=SETTINGS["speech_pad_ms"]))
    return samples, [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in spans]


def _evaluate(wave, rate, spans, recognition, expected, expected_origin="provided"):
    """Pure metrics/decision stage, separately testable with controlled evidence."""
    import numpy as np
    checks, metrics = {}, {}
    duration = len(wave) / rate
    peak = float(np.max(np.abs(wave)))
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    clipping = float(np.mean(np.abs(wave) >= .999))
    silent = float(np.mean(np.abs(wave) < .001))
    metrics.update(duration_seconds=duration, sample_rate=rate, peak=peak, rms=rms,
                   clipping_fraction=clipping, near_silence_fraction=silent)
    checks["decode"] = {"status": "pass", "reason": "Finite mono audio, bounded 3-30 second crop."}
    checks["clipping"] = {"status": "fail" if peak > 1 or clipping > SETTINGS["clip_fraction_max"] else "pass",
                           "reason": f"Peak {peak:.4f}; clipped sample fraction {clipping:.6f}."}
    checks["signal"] = {"status": "fail" if rms < .005 or silent > .65 else "pass",
                        "reason": f"RMS {rms:.4f}; near-silent sample fraction {silent:.3f}."}
    mask = np.zeros(len(wave), dtype=bool)
    for span in spans or []:
        if not 0 <= span["start"] < span["end"] <= duration + .04:
            raise ValueError("VAD timestamps outside exact crop")
        mask[max(0, int(span["start"] * rate)):min(len(wave), int(span["end"] * rate))] = True
    seconds = float(mask.sum() / rate)
    metrics.update(speech_seconds=seconds, speech_fraction=seconds / duration, speech_spans=spans)
    checks["speech"] = {"status": "unavailable" if spans is None else
                        ("fail" if seconds < SETTINGS["speech_seconds_min"] or seconds / duration < SETTINGS["speech_fraction_min"] else "pass"),
                        "reason": "CPU Silero VAD unavailable." if spans is None else f"VAD found {seconds:.2f}s of speech; this does not count speakers."}
    if spans and (~mask).sum() / rate >= .3:
        speech_rms = float(np.sqrt(np.mean(wave[mask].astype(np.float64) ** 2)))
        background_rms = float(np.sqrt(np.mean(wave[~mask].astype(np.float64) ** 2)))
        proxy = 20 * math.log10(max(speech_rms, 1e-12) / max(background_rms, 1e-12))
        metrics["speech_background_ratio_db"] = round(proxy, 3)
        checks["background_energy"] = {"status": "review" if proxy < SETTINGS["background_ratio_review_below_db"] else "pass",
                                       "reason": f"Speech/non-speech RMS ratio {proxy:.1f}dB; a heuristic, not measured SNR or music detection."}
    else:
        checks["background_energy"] = {"status": "unavailable", "reason": "Less than 0.3s of non-speech; background estimate unavailable."}
    # Installed VAD/Whisper cannot prove one speaker or absence of music.
    checks["music"] = {"status": "unavailable", "reason": "No installed validated music classifier; not assessed."}
    checks["overlap_speakers"] = {"status": "unavailable", "reason": "No installed overlap detector or diarizer; speaker count and overlap not assessed."}
    checks["identity_rights"] = {"status": "unavailable", "reason": "Audio metrics do not prove performer identity, consent, licensing or endorsement."}
    if recognition.get("status") != "draft":
        checks["asr"] = {"status": "unavailable", "reason": recognition.get("reason", "Local ASR unavailable.")}
        checks["transcript_alignment"] = {"status": "unavailable", "reason": "ASR required for transcript comparison."}
        return checks, metrics
    rows = recognition.get("segments", [])
    words = [w for row in rows for w in row.get("words", [])]
    for row in rows:
        numbers = [row["start"], row["end"], row["avg_logprob"], row["no_speech_prob"]]
        if not all(math.isfinite(float(v)) for v in numbers) or not 0 <= row["start"] <= row["end"] <= duration + .1:
            raise ValueError("ASR segment timestamps/confidence invalid")
    for word in words:
        if not all(math.isfinite(float(word[v])) for v in ("start", "end", "probability")) or not 0 <= word["start"] <= word["end"] <= duration + .1 or not 0 <= word["probability"] <= 1:
            raise ValueError("ASR word timestamps/confidence invalid")
    probabilities = [w["probability"] for w in words]
    mean = float(np.mean(probabilities)) if probabilities else 0
    low = sum(p < SETTINGS["low_word_probability"] for p in probabilities) / max(1, len(probabilities))
    confidence_ok = bool(words) and mean >= SETTINGS["word_confidence_mean_min"] and low <= SETTINGS["low_word_fraction_max"] and all(r["avg_logprob"] >= -1 and r["no_speech_prob"] <= .4 for r in rows)
    metrics.update(mean_word_probability=round(mean, 4), low_confidence_word_fraction=low,
                   recognized_text=recognition.get("text", ""), asr_word_count=len(words))
    checks["asr"] = {"status": "pass" if confidence_ok else "review",
                     "reason": f"Mean word confidence {mean:.3f}; low-confidence fraction {low:.3f}. Confidence is not transcript truth."}
    wer = word_error_rate(expected, recognition.get("text", "")) if expected else None
    metrics["word_error_rate"] = wer
    checks["transcript_alignment"] = {"status": "review" if wer is None else
                                       ("fail" if wer >= SETTINGS["wer_fail_min"] else "review" if wer > SETTINGS["wer_pass_max"] else "pass"),
                                       "reason": "No independent expected transcript supplied." if wer is None else f"Normalized word error rate {wer:.3f} against the supplied transcript; word timing is ASR-estimated, not forced alignment."}
    if expected_origin in {"machine_draft", "unverified_entry"} and checks["transcript_alignment"]["status"] == "pass":
        checks["transcript_alignment"] = {"status": "review", "reason": "Expected words are an unchanged ASR draft or unsaved entry without independent provenance; WER is self-consistency, not independent transcript alignment."}
    if words and spans:
        first, last = min(w["start"] for w in words), max(w["end"] for w in words)
        outside = sum(max(0, min(s["end"], first) - s["start"]) + max(0, s["end"] - max(s["start"], last)) for s in spans)
        total, covered = 0., 0.
        for word in words:
            total += word["end"] - word["start"]
            covered += sum(max(0, min(word["end"], s["end"]) - max(word["start"], s["start"])) for s in spans)
        coverage = covered / max(total, 1e-9)
        metrics.update(uncovered_edge_speech_seconds=outside, word_vad_coverage=coverage,
                       first_word_seconds=first, last_word_seconds=last)
        checks["crop_coverage"] = {"status": "review" if outside > SETTINGS["uncovered_speech_max_seconds"] or coverage < SETTINGS["word_vad_coverage_min"] else "pass",
                                   "reason": f"Uncovered edge speech {outside:.2f}s; word/VAD overlap {coverage:.3f}. Clipped phonemes still require listening when uncertain."}
    else:
        checks["crop_coverage"] = {"status": "unavailable", "reason": "Word times and VAD speech spans required."}
    return checks, metrics


def _finish(base, checks, metrics, *, cache_hit=False):
    required = {"decode", "clipping", "signal", "speech", "asr", "transcript_alignment", "crop_coverage"}
    status = "fail" if any(c["status"] == "fail" for c in checks.values()) else (
        "review" if any(c["status"] == "review" or (k in required and c["status"] == "unavailable") for k, c in checks.items()) else "pass")
    return base | {"status": status, "scope": "supported technical checks only", "checks": checks,
                   "metrics": metrics, "unavailable_checks": [k for k, v in checks.items() if v["status"] == "unavailable"],
                   "source_isolation_status": "unverified", "automatic_source_approval": False,
                   "clone_eligibility_changed": False, "human_verified": False, "cache_hit": cache_hit}


def verify_bytes(directory, audio, expected_text=None, *, language="en", expected_origin="provided"):
    """Cache immutable byte/transcript/pipeline-bound evidence; no downloads, GPU or mutations."""
    import numpy as np
    import soundfile as sf
    if expected_text is not None and (not isinstance(expected_text, str) or not 1 <= len(expected_text.strip()) <= 1000):
        raise ValueError("Expected transcript must contain 1-1000 characters")
    expected_text = expected_text.strip() if expected_text else None
    base = {"schema": VERSION, "audio_sha256": _sha(audio), "expected_text": expected_text,
            "expected_text_sha256": _sha(expected_text.encode()) if expected_text else None,
            "language": language, "expected_text_origin": expected_origin, "device": "cpu", "created_at_unix": time.time()}
    checks = {}
    with _lock:
        try:
            if not audio or len(audio) > SETTINGS["max_bytes"]:
                raise ValueError("Empty or oversized reference audio")
            info = sf.info(io.BytesIO(audio))
            if info.channels != 1 or not SETTINGS["duration_min"] <= info.duration <= SETTINGS["duration_max"] or not 8000 <= info.samplerate <= 192000:
                raise ValueError("Requires mono 3-30s crop, sample rate 8-192kHz")
            wave, rate = sf.read(io.BytesIO(audio), dtype="float32")
            if not np.isfinite(wave).all():
                raise ValueError("Nonfinite audio samples")
        except Exception as exc:
            return _finish(base, {"decode": {"status": "fail", "reason": str(exc)}}, {})
        try:
            pipeline = _pipeline()
        except Exception as exc:
            checks, metrics = _evaluate(wave, rate, None, {"status": "unavailable", "reason": str(exc)}, expected_text, expected_origin)
            return _finish(base | {"pipeline_unavailable_reason": str(exc)}, checks, metrics)
        key = _digest({"audio": base["audio_sha256"], "text": base["expected_text_sha256"], "language": language, "origin": expected_origin, "pipeline": pipeline})
        base.update(pipeline=pipeline, cache_key=key)
        folder = Path(directory) / "reference-quality"
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / (key + ".json")
        if cache.exists():
            try:
                envelope = json.loads(cache.read_text("utf-8"))
                result = envelope["result"]
                if envelope["sha256"] != _digest(result) or any(result.get(k) != base.get(k) for k in ("audio_sha256", "expected_text_sha256", "language", "expected_text_origin", "pipeline", "cache_key")):
                    raise ValueError("Quality cache checksum or binding mismatch")
                return result | {"cache_hit": True}
            except (ValueError, KeyError, TypeError) as exc:
                return _finish(base, {"cache_integrity": {"status": "fail", "reason": str(exc)}}, {})
        try:
            _, spans = _speech(wave, rate)
            recognition = asr.transcribe_bytes(directory, audio, language=language)
            if recognition.get("audio_sha256") != base["audio_sha256"]:
                raise ValueError("ASR evidence differs from exact reference bytes")
            checks, metrics = _evaluate(wave, rate, spans, recognition, expected_text, expected_origin)
            result = _finish(base, checks, metrics)
            result["elapsed_seconds"] = round(time.time() - base["created_at_unix"], 4)
            asr._save(cache, {"sha256": _digest(result), "result": result})
            return result
        except Exception as exc:
            return _finish(base, {"pipeline_execution": {"status": "review", "reason": str(exc)}}, {})


def acquisition_quality(project_id, acquisition_id, acquisition_sha256, expected_text=None):
    from dubbing_service.app.core.config import safe_project_root
    from .voice_lab_audition import _read_verified, _safe_path
    asr.acquisition_audio(project_id, acquisition_id, acquisition_sha256)  # fixed ID/path/hash validation
    directory = safe_project_root(project_id) / "voice_lab/discovery"
    receipt = json.loads(_read_verified(_safe_path(directory, f"acquisitions/{acquisition_id}/acquisition.json"), acquisition_sha256))
    audio = _read_verified(_safe_path(directory, receipt["reference_path"]), receipt["reference_audio_sha256"])
    expected = expected_text.strip() if expected_text is not None else (receipt.get("reference_text") or "").strip() or None
    embedded = receipt.get("transcript_evidence") or {}
    provided = (receipt.get("reference_text") or "").strip()
    origin = "provided" if provided and expected == provided and not embedded.get("machine_generated") else "unverified_entry"
    if embedded.get("machine_generated") and expected == (embedded.get("text") or "").strip():
        origin = "machine_draft"
    correction = asr._correction(directory, acquisition_id, acquisition_sha256, receipt["reference_audio_sha256"], None, require_asr_match=False)
    if correction and expected == correction["text"].strip():
        origin = "user_correction"
    result = verify_bytes(directory, audio, expected, language=receipt["source"].get("language", "en"), expected_origin=origin)
    # Quality is a separate artifact, never a rewritten receipt/role/preference.
    if result.get("cache_key") and "pipeline_execution" not in result["checks"] and "cache_integrity" not in result["checks"]:
        links = directory / "reference-quality/links"
        links.mkdir(parents=True, exist_ok=True)
        asr._save(links / f"{acquisition_id}-{acquisition_sha256}.json", {"cache_key": result["cache_key"], "expected_text_sha256": result["expected_text_sha256"]})
    return result | {"acquisition_id": acquisition_id, "acquisition_sha256": acquisition_sha256}


def current_quality(directory, acquisition_id, acquisition_sha256, audio_sha256, text):
    """Cheap GET read: never runs inference; only displays current hash-bound evidence."""
    link = Path(directory) / "reference-quality/links" / f"{acquisition_id}-{acquisition_sha256}.json"
    if not link.exists():
        return None
    try:
        from .voice_lab_audition import _safe_path
        audio_path = _safe_path(Path(directory), f"acquisitions/{acquisition_id}/reference.wav")
        if audio_path.stat().st_size > SETTINGS["max_bytes"] or _sha(audio_path.read_bytes()) != audio_sha256:
            return {"status": "fail", "reason": "Current reference bytes differ from the checked audio.", "automatic_source_approval": False}
        pointer = json.loads(link.read_text("utf-8"))
        key = pointer["cache_key"]
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid quality cache key")
        envelope = json.loads((link.parent.parent / (key + ".json")).read_text("utf-8"))
        result = envelope["result"]
        expected_sha = _sha(text.strip().encode()) if text and text.strip() else None
        if envelope["sha256"] != _digest(result):
            raise ValueError("Quality artifact checksum mismatch")
        if result.get("audio_sha256") != audio_sha256 or result.get("expected_text_sha256") != expected_sha or result.get("pipeline") != _pipeline():
            return {"status": "stale", "reason": "Audio, transcript or checker changed; run the local check again."}
        return result | {"cache_hit": True}
    except Exception as exc:
        return {"status": "review", "reason": f"Quality evidence unavailable: {exc}", "automatic_source_approval": False}
