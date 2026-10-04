"""Opt-in, pinned CPU reference cleanup; never an automatic TTS transform."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import tempfile
from pathlib import Path
from threading import Lock

import numpy as np

MODEL_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
_LOCK = Lock()
_PIPELINE_PATH = Path(__file__).resolve()
_PIPELINE_SHA256 = hashlib.sha256(_PIPELINE_PATH.read_bytes()).hexdigest()


def _weights() -> Path:
    return Path(__file__).resolve().parents[3] / "models/reference_preparation/hdemucs_high_trained.pt"


def _isolated_replay(wave: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict]:
    """The lightweight Studio owner uses the existing fixed worker environment."""
    from .reference_codec import _run

    root = _PIPELINE_PATH.parents[3]
    python = root / "environments/faster_qwen_base032/.venv/Scripts/python.exe"
    if not python.is_file() or python.is_symlink():
        raise ValueError("existing CPU replay environment is unavailable; no installation attempted")
    with tempfile.TemporaryDirectory(prefix="voice-reference-replay-") as folder:
        source, output = Path(folder) / "input.npz", Path(folder) / "output.npz"
        np.savez(source, wave=wave, sample_rate=np.asarray(sample_rate, dtype=np.int64))
        _run([str(python), str(_PIPELINE_PATH), "--cpu-replay", str(source), str(output)],
             timeout_seconds=90, max_stdout_bytes=4096)
        if not output.is_file() or output.stat().st_size > 6_000_000:
            raise ValueError("isolated CPU replay output missing or oversized")
        with np.load(output, allow_pickle=False) as data:
            if set(data.files) != {"wave", "sample_rate", "recipe"}:
                raise ValueError("invalid isolated CPU replay output")
            result = data["wave"].copy()
            rate = int(data["sample_rate"])
            recipe = json.loads(data["recipe"].tobytes().decode("utf-8"))
    if (rate != 44100 or result.dtype != np.float32 or result.ndim != 1
            or not 3 <= len(result) / rate <= 30 or not np.isfinite(result).all()
            or recipe.get("pipeline_sha256") != _PIPELINE_SHA256
            or recipe.get("input_pcm_sha256") != hashlib.sha256(wave.tobytes()).hexdigest()
            or recipe.get("output_pcm_sha256") != hashlib.sha256(result.tobytes()).hexdigest()):
        raise ValueError("isolated CPU replay provenance/output mismatch")
    return result, rate, recipe


def isolate_speech(wave: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict]:
    """Replay a fixed vocals-stem recipe; no download, CUDA, or caller model path."""
    if hashlib.sha256(_PIPELINE_PATH.read_bytes()).hexdigest() != _PIPELINE_SHA256:
        raise ValueError("speech-isolation source changed; restart the worker")
    if (sample_rate not in range(8000, 96001) or wave.ndim not in (1, 2)
            or (wave.ndim == 2 and wave.shape[1] != 2)
            or not 3 <= len(wave) / sample_rate <= 30
            or not np.isfinite(wave).all()):
        raise ValueError("unsuitable speech-isolation input")
    try:
        import soundfile
        import torch
        import torchaudio
        from torchaudio.models import hdemucs_high
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "torchaudio"}:
            raise
        with _LOCK:
            return _isolated_replay(wave, sample_rate)
    weights = _weights()
    if (not weights.is_file() or weights.is_symlink()
            or weights.stat().st_size != 334_697_255):
        raise ValueError("pinned speech-isolation weights missing or changed")
    weight_bytes = weights.read_bytes()
    if hashlib.sha256(weight_bytes).hexdigest() != MODEL_SHA256:
        raise ValueError("pinned speech-isolation weights missing or changed")
    versions = {"torch": str(torch.__version__), "torchaudio": torchaudio.__version__,
                "numpy": np.__version__, "soundfile": soundfile.__version__}
    if any(importlib.metadata.version(p) != value for p, value in versions.items()):
        raise ValueError("speech-isolation dependency changed; restart the worker")
    with _LOCK:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with torch.random.fork_rng(devices=[]):
                model = hdemucs_high(sources=["drums", "bass", "other", "vocals"]).cpu().eval()
            model.load_state_dict(torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True), strict=True)
            del weight_bytes
            stereo = wave.T.copy() if wave.ndim == 2 else np.stack((wave, wave))
            mixture = torchaudio.functional.resample(torch.from_numpy(stereo), sample_rate, 44100)
            mono = mixture.mean(0)
            mean, std = mono.mean(), mono.std()
            if not torch.isfinite(std) or std <= .005:
                raise ValueError("speech-isolation input has insufficient signal")
            with torch.inference_mode():
                separated = model(((mixture - mean) / std)[None])[0] * std + mean
            if separated.shape != (4, 2, mixture.shape[-1]) or not torch.isfinite(separated).all():
                raise ValueError("speech-isolation output is invalid")
            result = separated[3].mean(0).numpy().astype(np.float32)
        finally:
            torch.set_num_threads(previous_threads)
    recipe = {
        "schema": "reference-vocals-isolation-v1", "model": "torchaudio.models.hdemucs_high",
        "model_sha256": MODEL_SHA256, "weights_only_load": True, "device": "cpu",
        "torch_threads": 2, "versions": versions, "input_sample_rate": sample_rate,
        "input_channels": 2 if wave.ndim == 2 else 1,
        "input_pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        "output_sample_rate": 44100, "output_samples": len(result),
        "output_pcm_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "pipeline_sha256": _PIPELINE_SHA256,
        "operations": ["sinc_resample_44100", "normalize_mixture_mono_mean_std",
                       "hdemucs_high_vocals_stem", "undo_normalization", "stereo_to_mono_mean"],
        "possible_timbral_artifacts": True, "production_approved": False,
    }
    if hashlib.sha256(_PIPELINE_PATH.read_bytes()).hexdigest() != _PIPELINE_SHA256:
        raise ValueError("speech-isolation source changed; restart the worker")
    return result, 44100, recipe


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "--cpu-replay":
        raise SystemExit("Only fixed local CPU replay is supported")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source, output = Path(sys.argv[2]), Path(sys.argv[3])
    if source.stat().st_size > 24_000_000 or output.exists():
        raise SystemExit("Invalid bounded replay input/output")
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != {"wave", "sample_rate"}:
            raise SystemExit("Invalid bounded replay input schema")
        wave, sample_rate = data["wave"].copy(), int(data["sample_rate"])
    result, rate, recipe = isolate_speech(wave, sample_rate)
    np.savez(output, wave=result, sample_rate=np.asarray(rate, dtype=np.int64),
             recipe=np.frombuffer(json.dumps(recipe, sort_keys=True).encode("utf-8"), dtype=np.uint8))
