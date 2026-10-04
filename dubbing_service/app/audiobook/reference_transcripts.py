"""Offline CPU ASR for exact reference bytes; machine evidence is never human review."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import tempfile
import threading
from pathlib import Path

from dubbing_service.app.storage.atomic import replace_metadata_file
from service.model_path_policy import require_project_model_dir

_LOADED_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROOT = Path(__file__).resolve().parents[3]
MODEL_RELATIVE = "models/whisper/faster-whisper-base.en"
REQUIRED = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
SETTINGS = {"device": "cpu", "compute_type": "int8", "cpu_threads": 2,
            "beam_size": 5, "word_timestamps": True, "vad_filter": False,
            "condition_on_previous_text": False, "temperature": 0.0}
_lock = threading.Lock()
_correction_lock = threading.Lock()


def _installed_dependencies():
    versions = {}
    for package in ("faster-whisper", "ctranslate2", "onnxruntime", "numpy", "scipy", "soundfile", "av"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


_LOADED_DEPENDENCIES = _installed_dependencies()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save(path, value):
    import os
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stage:
        stage.write(json.dumps(value, indent=2).encode())
        stage.flush()
        os.fsync(stage.fileno())
        staged = Path(stage.name)
    try:
        replace_metadata_file(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _model():
    if _installed_dependencies() != _LOADED_DEPENDENCIES:
        raise ValueError("Reference ASR dependencies changed; restart Studio before transcription")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_SOURCE_SHA:
        raise ValueError("Reference ASR source changed; restart Studio before transcription")
    folder = require_project_model_dir(ROOT, MODEL_RELATIVE, label="Reference ASR", required_files=REQUIRED)
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in REQUIRED}
    return folder, _digest(hashes)


def transcribe_bytes(directory, audio, *, language="en"):
    """Return timestamped draft or an explicit unavailable/failed state; never download."""
    import io
    import numpy as np
    import soundfile as sf

    audio_sha = hashlib.sha256(audio).hexdigest()
    base = {"schema": "reference-asr-v1", "audio_sha256": audio_sha,
            "machine_generated": True, "human_verified": False, "alignment_method": "ASR word timestamps",
            "speaker_music_review": "not assessed", "settings": SETTINGS, "language": language, "pipeline_sha256": _LOADED_SOURCE_SHA, "dependency_versions": _LOADED_DEPENDENCIES,
            "text": "", "segments": [], "uncertainties": [], "correction_available": True}
    if language not in {"en", "fr", "de", "es", "it", "pt", "nl", "zh", "ja", "ko", None}:
        raise ValueError("Unsupported reference ASR language")
    if language not in {"en", None}:
        return base | {"status": "unavailable", "reason": "Installed reference model is English only; supply/correct the transcript."}
    with _lock:
        try:
            model_path, model_sha = _model()
        except (ValueError, OSError) as exc:
            return base | {"status": "unavailable", "reason": str(exc)}
        base.update(model="faster-whisper-base.en", model_sha256=model_sha)
        cache_key = _digest({"audio": audio_sha, "model": model_sha, "settings": SETTINGS, "language": language, "pipeline": base["pipeline_sha256"], "dependencies": base["dependency_versions"]})
        folder = Path(directory) / "transcripts"
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / (cache_key + ".json")
        if cache.exists():
            try:
                envelope = json.loads(cache.read_text("utf-8"))
                result = envelope["result"]
                if envelope["sha256"] != _digest(result) or any(result.get(k) != base[k] for k in ("audio_sha256", "model_sha256", "settings", "language", "pipeline_sha256", "dependency_versions")):
                    raise ValueError("Cached reference transcript checksum mismatch")
                return result | {"cache_hit": True}
            except (ValueError, KeyError, TypeError) as exc:
                return base | {"status": "failed", "reason": str(exc)}
        try:
            wave, rate = sf.read(io.BytesIO(audio), dtype="float32")
            duration = len(wave) / rate
            if wave.ndim != 1 or not 3 <= duration <= 30.01 or not np.isfinite(wave).all():
                raise ValueError("ASR requires a finite mono 3-30 second reference crop")
            from faster_whisper import WhisperModel
            model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=2, local_files_only=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=folder, delete=False) as temporary:
                temporary.write(audio)
                temporary_path = Path(temporary.name)
            try:
                segments, _ = model.transcribe(str(temporary_path), language=language,
                    **{k: v for k, v in SETTINGS.items() if k not in {"device", "compute_type", "cpu_threads"}})
                rows = []
                flags = []
                for s in segments:
                    words = [{"start": float(w.start), "end": float(w.end), "word": w.word,
                              "probability": float(w.probability)} for w in (s.words or [])]
                    numbers = [s.start, s.end, s.avg_logprob, s.no_speech_prob] + [v for w in words for v in (w["start"], w["end"], w["probability"])]
                    if not all(math.isfinite(float(v)) for v in numbers) or not 0 <= s.start <= s.end <= duration + .1 or any(not 0 <= w["start"] <= w["end"] <= duration + .1 for w in words):
                        raise ValueError("ASR timestamps are outside the exact cropped audio")
                    if s.avg_logprob < -1 or s.no_speech_prob > .6 or any(w["probability"] < .8 for w in words):
                        flags.append({"start": float(s.start), "end": float(s.end), "reason": "Low-confidence recognition; listen and correct."})
                    rows.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                                 "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob), "words": words})
            finally:
                temporary_path.unlink(missing_ok=True)
                del model
            text = " ".join(row["text"] for row in rows).strip()
            if not text or len(text) > 1000:
                raise ValueError("ASR produced empty or oversized text; supply/correct the reference transcript.")
            result = base | {"status": "draft", "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "segments": rows, "duration_seconds": duration, "uncertainties": flags,
                             "review_required": True, "cache_hit": False}
            _save(cache, {"sha256": _digest(result), "result": result})
            return result
        except Exception as exc:
            return base | {"status": "failed", "reason": str(exc)}


def acquisition_transcript(project_id, acquisition_id, acquisition_sha256):
    """Held acquisitions can be transcribed; fixed IDs and hashes constrain local reads."""
    from .voice_lab_audition import _read_verified, _safe_path
    from dubbing_service.app.core.config import safe_project_root
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", acquisition_id) or not re.fullmatch(r"[a-f0-9]{64}", acquisition_sha256):
        raise ValueError("Invalid acquisition identity")
    directory = safe_project_root(project_id) / "voice_lab/discovery"
    raw = _read_verified(_safe_path(directory, f"acquisitions/{acquisition_id}/acquisition.json"), acquisition_sha256)
    receipt = json.loads(raw)
    if receipt["acquisition_id"] != acquisition_id or receipt["reference_path"] != f"acquisitions/{acquisition_id}/reference.wav":
        raise ValueError("Reference path differs from acquisition identity")
    audio = _read_verified(_safe_path(directory, receipt["reference_path"]), receipt["reference_audio_sha256"])
    result = transcribe_bytes(directory, audio, language=receipt["source"].get("language", "en"))
    correction = _correction(directory, acquisition_id, acquisition_sha256,
                             receipt["reference_audio_sha256"], result.get("text_sha256"))
    return result | ({"correction": correction} if correction else {})


def correct_acquisition_transcript(project_id, acquisition_id, acquisition_sha256, text, expected_text_sha256,
                                   expected_correction_sha256=None):
    with _correction_lock:
        return _correct_acquisition_transcript(project_id, acquisition_id, acquisition_sha256, text,
                                               expected_text_sha256, expected_correction_sha256)


def _correct_acquisition_transcript(project_id, acquisition_id, acquisition_sha256, text, expected_text_sha256,
                                    expected_correction_sha256=None):
    """Save a separate correction without overwriting ASR, source, or QA acceptance."""
    from dubbing_service.app.core.config import safe_project_root
    result = acquisition_transcript(project_id, acquisition_id, acquisition_sha256)
    if result.get("text_sha256") != expected_text_sha256:
        raise ValueError("Transcript changed; reload before correcting")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
        raise ValueError("Corrected transcript must contain 1-1000 characters")
    if result.get("correction", {}).get("text_sha256") != expected_correction_sha256:
        raise ValueError("Transcript correction changed; reload before saving")
    corrected = {"schema": "reference-transcript-correction-v1", "status": "corrected_unreviewed",
                 "acquisition_id": acquisition_id, "acquisition_sha256": acquisition_sha256,
                 "audio_sha256": result["audio_sha256"], "original_asr_text_sha256": expected_text_sha256,
                 "text": text.strip(), "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest(),
                 "human_verified": False, "alignment_method": "not aligned after correction", "review_required": True}
    folder = safe_project_root(project_id) / "voice_lab/discovery/transcripts/corrections"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (_digest(corrected) + ".json")
    _save(path, corrected)
    links = folder.parent / "correction-links"
    links.mkdir(parents=True, exist_ok=True)
    _save(links / f"{acquisition_id}-{acquisition_sha256}.json",
          {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return corrected


def _correction(directory, acquisition_id, acquisition_sha256, audio_sha256, asr_text_sha256, *, require_asr_match=True):
    import re
    from .voice_lab_audition import _read_verified, _safe_path
    link = Path(directory) / "transcripts/correction-links" / f"{acquisition_id}-{acquisition_sha256}.json"
    if not link.exists():
        return None
    pointer = json.loads(link.read_text("utf-8"))
    if not re.fullmatch(r"[a-f0-9]{64}\.json", pointer.get("file", "")):
        raise ValueError("Invalid correction artifact")
    raw = _read_verified(_safe_path(Path(directory), "transcripts/corrections/" + pointer["file"]), pointer["sha256"])
    corrected = json.loads(raw)
    if (corrected.get("acquisition_id") != acquisition_id or corrected.get("acquisition_sha256") != acquisition_sha256
            or corrected.get("audio_sha256") != audio_sha256
            or (require_asr_match and corrected.get("original_asr_text_sha256") != asr_text_sha256)
            or corrected.get("text_sha256") != hashlib.sha256(corrected.get("text", "").encode()).hexdigest()):
        raise ValueError("Reference correction is stale or differs from these exact bytes")
    return corrected


def acquisition_audio(project_id, acquisition_id, acquisition_sha256):
    import re
    from dubbing_service.app.core.config import safe_project_root
    from .voice_lab_audition import _read_verified, _safe_path
    if not re.fullmatch(r"[a-f0-9]{32}", acquisition_id) or not re.fullmatch(r"[a-f0-9]{64}", acquisition_sha256):
        raise ValueError("Invalid acquisition identity")
    directory = safe_project_root(project_id) / "voice_lab/discovery"
    receipt = json.loads(_read_verified(_safe_path(directory, f"acquisitions/{acquisition_id}/acquisition.json"), acquisition_sha256))
    relative = f"acquisitions/{acquisition_id}/reference.wav"
    if receipt.get("acquisition_id") != acquisition_id or receipt.get("reference_path") != relative:
        raise ValueError("Reference path differs from acquisition identity")
    return _safe_path(directory, relative), receipt["reference_audio_sha256"]


def list_acquisitions(project_id):
    from dubbing_service.app.core.config import safe_project_root
    directory = safe_project_root(project_id) / "voice_lab/discovery/acquisitions"
    rows = []
    for path in sorted(directory.glob("*/acquisition.json"))[:30]:
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if receipt.get("acquisition_id") != path.parent.name:
            continue
        receipt_sha = hashlib.sha256(raw).hexdigest()
        evidence = receipt.get("transcript_evidence")
        correction = _correction(directory.parent, path.parent.name, receipt_sha,
                                 receipt.get("reference_audio_sha256"), None, require_asr_match=False)
        from .reference_quality import current_quality
        current_text = correction["text"] if correction else receipt.get("reference_text")
        quality = current_quality(directory.parent, path.parent.name, receipt_sha,
                                  receipt.get("reference_audio_sha256"), current_text)
        rows.append({"acquisition_id": path.parent.name, "acquisition_sha256": receipt_sha,
                     "source": receipt.get("source"), "duration_seconds": receipt.get("duration_seconds"),
                     "clone_ready": receipt.get("clone_ready", False), "holds": receipt.get("readiness_holds", receipt.get("technical_qa", {}).get("holds", [])),
                     "narration_delivery_fit": receipt.get("narration_delivery_fit"),
                     "transcript_evidence": evidence, "correction": correction, "quality_evidence": quality,
                     "original_sha256": receipt.get("original_sha256"),
                     "start_seconds": receipt.get("start_seconds"), "end_seconds": receipt.get("end_seconds"),
                     "reference_text": receipt.get("reference_text"), "source_review_method": receipt.get("audio_review", {}).get("method"),
                     "reference_audio_sha256": receipt.get("reference_audio_sha256")})
    return {"references": rows}
