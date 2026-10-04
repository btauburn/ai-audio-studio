"""Local, offline source classifier for bounded reference crops; no identity claims."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODEL_FILES = {
    "ast_audioset/model.safetensors": "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",
    "ast_audioset/config.json": "a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff",
    "ast_audioset/preprocessor_config.json": "8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231",
    "pyannote_segmentation/model.onnx": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
}
SETTINGS = {"schema": "source-classifier-v1", "window_seconds": 10,
            "music_max": .2, "noise_max": .2, "speech_min": .5, "overlap_peak_max": .1,
            "device": "cpu", "threads": 2}


def manifest():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("source classifier changed; restart before preparation")
    for name, expected in MODEL_FILES.items():
        path = ROOT / "models/reference_preparation" / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("existing pinned source-QC model missing or changed: " + name)
    return {"source_sha256": SOURCE_SHA, "models": MODEL_FILES, "settings": SETTINGS,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "onnxruntime", "numpy", "scipy", "soundfile")}}


def evaluate_windows(rows):
    """Conservative source gates: local classes do not establish biometric identity."""
    if not rows:
        return {"status": "unavailable", "contaminated": False, "reason": "No classifier windows"}
    if len(rows) > 3:
        raise ValueError("Source classifier window count exceeds crop bound")
    previous_end = 0.
    for row in rows:
        for key in ("music", "noise", "speech", "overlap_peak"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Source classifier probability invalid")
        for key, maximum in (("singleton_count", 3), ("overlap_frames", 4000)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError("Source classifier count invalid")
        start, duration = row.get("start_seconds"), row.get("duration_seconds")
        if (isinstance(start, bool) or isinstance(duration, bool)
                or not isinstance(start, (int, float)) or not isinstance(duration, (int, float))
                or not math.isfinite(start) or not math.isfinite(duration)
                or start < 0 or not 0 < duration <= 10 or start + duration > 30.001
                or start > previous_end + .001):
            raise ValueError("Source classifier window coverage invalid")
        previous_end = max(previous_end, start + duration)
    # Independent windows have permutation-invariant local speaker labels.
    # Without an installed embedding/diarization continuity check, >10s needs review.
    multiple = len(rows) > 1 or any(r["singleton_count"] != 1 or r["overlap_frames"] != 0
                   or r["overlap_peak"] >= SETTINGS["overlap_peak_max"] for r in rows)
    contamination = any(r["music"] >= SETTINGS["music_max"] or r["noise"] >= SETTINGS["noise_max"] for r in rows)
    speech = all(r["speech"] > SETTINGS["speech_min"] for r in rows)
    return {"status": "review" if multiple or not speech else ("contaminated" if contamination else "pass"),
            "contaminated": contamination, "single_local_speaker": not multiple,
            "speech_detected": speech,
            "reason": "Window-local classifier evidence; no performer identity or human quality acceptance."}


def classify(wave, rate):
    import onnxruntime as ort
    import torch
    from scipy.signal import resample_poly
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    pipeline = manifest()
    input_sha = hashlib.sha256(wave.tobytes()).hexdigest()
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if (wave.ndim != 1 or not 8000 <= rate <= 96000 or not 3 <= len(wave) / rate <= 30
            or not np.isfinite(wave).all()):
        raise ValueError("invalid source-QC crop")
    factor = math.gcd(rate, 16000)
    samples = resample_poly(wave, 16000 // factor, rate // factor).astype(np.float32)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
    model_root = ROOT / "models/reference_preparation"
    seg = ort.InferenceSession(str(model_root / "pyannote_segmentation/model.onnx"),
                               providers=["CPUExecutionProvider"], sess_options=opts)
    meta = seg.get_modelmeta().custom_metadata_map
    if any(int(meta[k]) != v for k, v in
           {"sample_rate": 16000, "num_speakers": 3, "num_classes": 7, "powerset_max_classes": 2}.items()):
        raise ValueError("source segmentation metadata changed")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            ast = ASTForAudioClassification.from_pretrained(str(model_root / "ast_audioset"),
                    local_files_only=True, trust_remote_code=False, use_safetensors=True).cpu().eval()
        features = ASTFeatureExtractor.from_pretrained(str(model_root / "ast_audioset"), local_files_only=True)
        starts = sorted(set(list(range(0, max(1, len(samples) - 160000 + 1), 160000))
                            + [max(0, len(samples) - 160000)]))
        rows = []
        for start in starts:
            chunk = samples[start:start + 160000]
            valid = len(chunk)
            chunk = np.pad(chunk, (0, 160000 - valid))
            probs = np.exp(seg.run(None, {seg.get_inputs()[0].name: chunk.reshape(1, 1, -1)})[0][0])
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(-1), 1, atol=1e-4):
                raise ValueError("invalid source segmentation scores")
            probs = probs[:max(1, int(len(probs) * valid / 160000))]
            classes = probs.argmax(-1)
            with torch.inference_mode():
                tags = torch.sigmoid(ast(**features(chunk[:valid], sampling_rate=16000, return_tensors="pt")).logits[0]).numpy()
            labels = {ast.config.id2label[i]: float(tags[i]) for i in range(len(tags))}
            if not {"Music", "Speech", "Noise"} <= labels.keys() or not np.isfinite(tags).all():
                raise ValueError("required source classifier labels missing")
            rows.append({"start_seconds": start / 16000, "duration_seconds": valid / 16000,
                         "music": labels["Music"], "speech": labels["Speech"], "noise": labels["Noise"],
                         "singleton_count": len(set(classes.tolist()) & {1, 2, 3}),
                         "overlap_frames": int(np.sum(classes >= 4)),
                         "overlap_peak": float(probs[:, 4:].sum(-1).max())})
    finally:
        torch.set_num_threads(previous_threads)
    if manifest() != pipeline:
        raise ValueError("source classifier changed during evaluation")
    return {"pipeline": pipeline, "input_pcm_sha256": input_sha, "sample_rate": rate,
            "windows": rows, **evaluate_windows(rows)}


if __name__ == "__main__":
    import sys
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    if sys.argv[1] == "--manifest":
        print(json.dumps(manifest(), sort_keys=True))
    else:
        path = Path(sys.argv[1])
        if path.stat().st_size > 24_000_000:
            raise ValueError("source-QC input oversized")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"wave", "sample_rate"}:
                raise ValueError("invalid source-QC input")
            result = classify(data["wave"], int(data["sample_rate"]))
        print(json.dumps(result, sort_keys=True))
