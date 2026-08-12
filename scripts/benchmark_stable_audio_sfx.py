#!/usr/bin/env python3
"""Run the real Stable Audio 3 Small-SFX acceptance benchmark.

The benchmark has two deliberate phases:

* ``--generate`` calls the loopback worker for 12 categories x 2 seeds and for
  five technical rounds x six runs. It writes WAVs, measurements, and a blank
  human-scoring CSV.
* ``--evaluate`` validates every generated file and the completed CSV. Only a
  real run with 30/30 technical passes and at least 10/12 quality categories
  passing is allowed to atomically create ``quality_gate.json``.

No mock or synthetic-audio mode exists in this command. Missing or mismatched
model weights fail before any worker request is made.
"""

from __future__ import annotations

import argparse
import array
import csv
import ipaddress
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .model_preflight import (
        PROJECT_ROOT,
        ManifestError,
        has_license_acceptance,
        load_manifest,
        quality_gate_path,
        runtime_code_revision_status,
        sha256_file,
        verify_manifest,
    )
except ImportError:  # Direct execution: python scripts/benchmark_stable_audio_sfx.py
    from model_preflight import (
        PROJECT_ROOT,
        ManifestError,
        has_license_acceptance,
        load_manifest,
        quality_gate_path,
        runtime_code_revision_status,
        sha256_file,
        verify_manifest,
    )


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "models" / "manifests" / "stable_audio_3_small_sfx.json"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:8005/api/v1/sfx/generate"
NEGATIVE_PROMPT = "music, melody, speech, voice, vocals, narration"
QUALITY_SEEDS = (104_729, 130_363)
SCORE_FIELDS = (
    "category",
    "case_id",
    "seed",
    "prompt",
    "file",
    "semantic_match_1_5",
    "event_timing_1_5",
    "artifact_control_1_5",
    "background_cleanliness_1_5",
    "usable_yes_no",
    "reviewer_notes",
)

# Twelve production-relevant sound classes, each rendered with two fixed seeds.
QUALITY_CATEGORIES: tuple[dict[str, Any], ...] = (
    {
        "id": "footsteps",
        "duration_seconds": 6.0,
        "prompt": "Close-up heavy leather boots walking steadily across wet concrete, distinct individual footsteps, realistic Foley, no voices, no music",
    },
    {
        "id": "impact",
        "duration_seconds": 4.0,
        "prompt": "A heavy wooden shipping crate drops once onto a warehouse floor, sharp initial impact followed by wood rattling and a short room decay, no music",
    },
    {
        "id": "door_mechanism",
        "duration_seconds": 5.0,
        "prompt": "A large rusty metal door creaks open slowly and ends with a solid latch clunk, isolated sound effect, no voices",
    },
    {
        "id": "cloth_foley",
        "duration_seconds": 5.0,
        "prompt": "Close-miked thick canvas jacket and leather gear rustling as a person turns quickly, detailed dry Foley, no footsteps, no speech",
    },
    {
        "id": "household_object",
        "duration_seconds": 4.0,
        "prompt": "A ceramic coffee mug is placed firmly on a wooden table, one clean contact with a small ceramic ring, quiet room, no voices",
    },
    {
        "id": "vehicle_passby",
        "duration_seconds": 8.0,
        "prompt": "A passenger car approaches on a wet road, passes left to right close to the listener, then recedes into the distance, realistic exterior recording, no music",
    },
    {
        "id": "weather",
        "duration_seconds": 8.0,
        "prompt": "Heavy rain striking a metal roof with one distant thunder roll, natural storm ambience, no people, no music",
    },
    {
        "id": "water",
        "duration_seconds": 6.0,
        "prompt": "A large stone splashes into a calm pond, strong single splash followed by ripples and dripping water, isolated outdoors, no voices",
    },
    {
        "id": "fire",
        "duration_seconds": 7.0,
        "prompt": "Close crackling campfire with small pops and shifting embers at night, clean natural ambience, no insects, no speech, no music",
    },
    {
        "id": "animal",
        "duration_seconds": 5.0,
        "prompt": "A medium-sized dog gives two alert barks in a quiet backyard, realistic distance and short outdoor reflections, no human voice",
    },
    {
        "id": "electronic_interface",
        "duration_seconds": 4.0,
        "prompt": "A futuristic access panel emits three short clean confirmation beeps followed by a mechanical unlock click, isolated user-interface sound, no music",
    },
    {
        "id": "fantasy_weapon",
        "duration_seconds": 5.0,
        "prompt": "A magical sword is drawn quickly from a metal scabbard, bright steel scrape followed by a brief low energy shimmer, cinematic sound effect, no music, no voice",
    },
)

TECHNICAL_CASES: tuple[dict[str, Any], ...] = (
    {"id": "dry_click", "prompt": "One dry wooden switch click in a silent room"},
    {"id": "glass", "prompt": "A small glass bottle rolls briefly and stops on stone"},
    {"id": "paper", "prompt": "A single sheet of paper is crumpled close to the microphone"},
    {"id": "bell", "prompt": "One small brass service bell ring with natural decay"},
    {"id": "whoosh", "prompt": "A fast clean left-to-right cinematic air whoosh, no music"},
    {"id": "gravel", "prompt": "A handful of gravel pours onto concrete and settles"},
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark evidence cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_pins(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": item["path"],
            "size": item["size"],
            "sha256": item["sha256"].lower(),
        }
        for item in manifest["files"]
        if item.get("group") == "runtime"
    ]


def _code_pins(manifest: dict[str, Any]) -> dict[str, str]:
    revision = manifest.get("code_revision")
    source = manifest.get("code_source")
    if revision is None and source is None:
        return {}
    if not isinstance(revision, str) or not isinstance(source, dict):
        raise BenchmarkError("manifest code provenance is incomplete")
    distribution = source.get("distribution")
    source_url = source.get("url")
    if not isinstance(distribution, str) or not isinstance(source_url, str):
        raise BenchmarkError("manifest code provenance is incomplete")
    return {
        "code_revision": revision,
        "code_distribution": distribution,
        "code_source_url": source_url,
    }


def _runtime_python() -> Path:
    if sys.platform.startswith("win"):
        return PROJECT_ROOT / "environments" / "stableaudio" / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / "environments" / "stableaudio" / ".venv" / "bin" / "python"


def _preflight_real_model(manifest_path: Path) -> dict[str, Any]:
    manifest, statuses = verify_manifest(
        manifest_path,
        project_root=PROJECT_ROOT,
        verify_hash=True,
        groups=("runtime",),
    )
    failures = [item for item in statuses if not item.ok]
    if failures:
        detail = "; ".join(f"{item.path}: {item.state}" for item in failures[:4])
        extra = f" (+{len(failures) - 4} more)" if len(failures) > 4 else ""
        raise BenchmarkError(
            "Stable Audio 3 Small-SFX weights are missing or unverified: "
            f"{detail}{extra}. Provision the pinned gated manifest first; the benchmark "
            "does not download models."
        )
    if not has_license_acceptance(manifest, PROJECT_ROOT):
        raise BenchmarkError(
            "The current Stable Audio/Gemma license acceptance receipt is missing or stale. "
            "Record the exact manifest confirmation before benchmarking."
        )
    runtime_ok, runtime_state = runtime_code_revision_status(manifest, _runtime_python())
    if not runtime_ok:
        raise BenchmarkError(
            "Stable Audio isolated runtime does not match the pinned upstream code: "
            f"{runtime_state}"
        )
    return manifest


def _build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    quality_cases = []
    for category in QUALITY_CATEGORIES:
        for seed in QUALITY_SEEDS:
            case_id = f"{category['id']}-seed-{seed}"
            quality_cases.append(
                {
                    "category": category["id"],
                    "case_id": case_id,
                    "seed": seed,
                    "prompt": category["prompt"],
                    "negative_prompt": NEGATIVE_PROMPT,
                    "duration_seconds": category["duration_seconds"],
                    "steps": 8,
                    "cfg_scale": 1.0,
                    "file": f"quality/{case_id}.wav",
                }
            )
    technical_runs = []
    for round_number in range(1, 6):
        for slot, case in enumerate(TECHNICAL_CASES, start=1):
            seed = 200_000 + round_number * 100 + slot
            run_id = f"round-{round_number:02d}-{case['id']}"
            technical_runs.append(
                {
                    "round": round_number,
                    "slot": slot,
                    "run_id": run_id,
                    "seed": seed,
                    "prompt": case["prompt"],
                    "negative_prompt": NEGATIVE_PROMPT,
                    "duration_seconds": 3.0,
                    "steps": 8,
                    "cfg_scale": 1.0,
                    "file": f"technical/round-{round_number:02d}/{run_id}.wav",
                }
            )
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "model_id": manifest["model_id"],
        "model_revision": manifest["revision"],
        "source_revisions": {
            key: value.get("revision")
            for key, value in manifest["sources"].items()
            if isinstance(value, dict)
        },
        "model_files": _model_pins(manifest),
        **_code_pins(manifest),
        "quality_cases": quality_cases,
        "technical_runs": technical_runs,
    }


def _write_score_template(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            for case in plan["quality_cases"]:
                writer.writerow(
                    {
                        "category": case["category"],
                        "case_id": case["case_id"],
                        "seed": case["seed"],
                        "prompt": case["prompt"],
                        "file": case["file"],
                        "semantic_match_1_5": "",
                        "event_timing_1_5": "",
                        "artifact_control_1_5": "",
                        "background_cleanliness_1_5": "",
                        "usable_yes_no": "",
                        "reviewer_notes": "",
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def analyze_wav(path: Path, expected_duration: float) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise BenchmarkError(f"invalid WAV {path}: {exc}") from exc
    if sample_width != 2:
        raise BenchmarkError(f"{path}: expected PCM16, got {sample_width * 8}-bit samples")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        raise BenchmarkError(f"{path}: WAV contains no samples")
    peak = max(abs(value) for value in samples) / 32768.0
    rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768.0
    clipping_fraction = sum(abs(value) >= 32760 for value in samples) / len(samples)
    duration = frame_count / sample_rate if sample_rate else 0.0
    failures = []
    if compression != "NONE":
        failures.append("not_uncompressed_pcm")
    if channels != 2:
        failures.append("not_stereo")
    if sample_rate != 44_100:
        failures.append("sample_rate_not_44100")
    if abs(duration - expected_duration) > max(0.05, expected_duration * 0.01):
        failures.append("duration_out_of_tolerance")
    if rms < 0.001:
        failures.append("near_silent")
    if clipping_fraction > 0.01:
        failures.append("excessive_clipping")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frame_count,
        "duration_seconds": duration,
        "peak": peak,
        "rms": rms,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "clipping_fraction": clipping_fraction,
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "passed": not failures,
        "failures": failures,
    }


def _request_audio(
    case: dict[str, Any],
    *,
    endpoint: str,
    worker_token: str,
    timeout_seconds: float,
    output_path: Path,
    expected_worker_model_id: str,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "prompt": case["prompt"],
            "negative_prompt": case["negative_prompt"],
            "duration_seconds": case["duration_seconds"],
            "seed": case["seed"],
            "steps": case["steps"],
            "cfg_scale": case["cfg_scale"],
            "chunked_decode": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Dubbing-Worker-Token": worker_token,
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
            headers = response.headers
            status = response.status
    except urllib.error.HTTPError as exc:
        message = exc.read(1024).decode("utf-8", errors="replace")
        raise BenchmarkError(f"worker HTTP {exc.code}: {message}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(f"worker request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    if status != 200:
        raise BenchmarkError(f"worker returned unexpected HTTP status {status}")
    if not str(headers.get("Content-Type", "")).lower().startswith("audio/wav"):
        raise BenchmarkError("worker response is not audio/wav")
    if headers.get("X-Audio-Model-Id") != expected_worker_model_id:
        raise BenchmarkError("worker returned an unexpected model id")
    if headers.get("X-Audio-Seed") != str(case["seed"]):
        raise BenchmarkError("worker returned an unexpected seed")
    if headers.get("X-Audio-Sample-Rate") != "44100":
        raise BenchmarkError("worker returned an unexpected sample-rate header")
    _atomic_bytes(output_path, payload)
    metrics = analyze_wav(output_path, float(case["duration_seconds"]))
    metrics.update(
        {
            "elapsed_seconds": elapsed,
            "real_time_factor": elapsed / float(case["duration_seconds"]),
            "header_duration_seconds": float(
                headers.get("X-Audio-Duration-Seconds", "nan")
            ),
        }
    )
    if not math.isfinite(metrics["header_duration_seconds"]):
        metrics["failures"].append("invalid_duration_header")
    elif abs(
        metrics["header_duration_seconds"] - metrics["duration_seconds"]
    ) > 0.05:
        metrics["failures"].append("duration_header_mismatch")
    metrics["passed"] = not metrics["failures"]
    return metrics


def _require_loopback_endpoint(endpoint: str, label: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BenchmarkError(f"{label} must be an absolute HTTP endpoint")
    if parsed.hostname.lower() == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise BenchmarkError(f"{label} must resolve explicitly to a loopback address")


def _post_control(
    endpoint: str,
    *,
    token: str | None,
    timeout_seconds: float,
    expected_loaded: bool | None,
    label: str,
) -> dict[str, Any]:
    _require_loopback_endpoint(endpoint, label)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Dubbing-Worker-Token"] = token
    request = urllib.request.Request(
        endpoint,
        data=b"{}",
        method="POST",
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        message = exc.read(1024).decode("utf-8", errors="replace")
        raise BenchmarkError(f"{label} HTTP {exc.code}: {message}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(f"{label} failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    if status != 200:
        raise BenchmarkError(f"{label} returned HTTP {status}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{label} did not return JSON") from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise BenchmarkError(f"{label} did not confirm success")
    model_loaded = payload.get("model_loaded")
    if expected_loaded is not None and model_loaded is not None:
        if model_loaded is not expected_loaded:
            raise BenchmarkError(f"{label} returned model_loaded={model_loaded!r}")
    return {
        "success": True,
        "model_loaded": model_loaded,
        "elapsed_seconds": elapsed,
    }


def _gpu_memory_used_mib(gpu_index: int) -> int:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkError(f"nvidia-smi memory probe failed: {exc}") from exc
    if completed.returncode != 0:
        raise BenchmarkError(
            f"nvidia-smi memory probe failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise BenchmarkError("nvidia-smi did not return exactly one GPU memory value")
    try:
        return int(values[0])
    except ValueError as exc:
        raise BenchmarkError(f"nvidia-smi returned invalid memory value: {values[0]}") from exc


def _handoff_to_sfx(
    *,
    tts_unload_endpoint: str,
    sfx_load_endpoint: str,
    tts_token: str | None,
    sfx_token: str,
    timeout_seconds: float,
    gpu_index: int,
) -> dict[str, Any]:
    before = _gpu_memory_used_mib(gpu_index)
    tts_unload = _post_control(
        tts_unload_endpoint,
        token=tts_token,
        timeout_seconds=timeout_seconds,
        expected_loaded=False,
        label="TTS unload",
    )
    baseline = _gpu_memory_used_mib(gpu_index)
    sfx_load = _post_control(
        sfx_load_endpoint,
        token=sfx_token,
        timeout_seconds=timeout_seconds,
        expected_loaded=True,
        label="SFX load",
    )
    after_load = _gpu_memory_used_mib(gpu_index)
    return {
        "tts_unload": tts_unload,
        "sfx_load": sfx_load,
        "memory_before_tts_unload_mib": before,
        "memory_baseline_after_tts_unload_mib": baseline,
        "memory_after_sfx_load_mib": after_load,
        "memory_peak_sfx_mib": after_load,
    }


def _handoff_back_to_tts(
    handoff: dict[str, Any],
    *,
    sfx_unload_endpoint: str,
    tts_load_endpoint: str,
    tts_token: str | None,
    sfx_token: str,
    timeout_seconds: float,
    gpu_index: int,
    release_tolerance_mib: int,
) -> dict[str, Any]:
    sfx_unload = _post_control(
        sfx_unload_endpoint,
        token=sfx_token,
        timeout_seconds=timeout_seconds,
        expected_loaded=False,
        label="SFX unload",
    )
    released = _gpu_memory_used_mib(gpu_index)
    baseline = int(handoff["memory_baseline_after_tts_unload_mib"])
    release_passed = released <= baseline + release_tolerance_mib
    tts_load = _post_control(
        tts_load_endpoint,
        token=tts_token,
        timeout_seconds=timeout_seconds,
        expected_loaded=True,
        label="TTS load",
    )
    after_tts_load = _gpu_memory_used_mib(gpu_index)
    handoff.update(
        {
            "sfx_unload": sfx_unload,
            "tts_load": tts_load,
            "memory_after_sfx_unload_mib": released,
            "memory_release_delta_mib": released - baseline,
            "memory_release_tolerance_mib": release_tolerance_mib,
            "memory_release_passed": release_passed,
            "memory_after_tts_load_mib": after_tts_load,
            "passed": release_passed,
        }
    )
    if not release_passed:
        raise BenchmarkError(
            "GPU memory did not return to the post-TTS-unload baseline: "
            f"{released} MiB > {baseline} + {release_tolerance_mib} MiB"
        )
    return handoff


def _plan_matches_manifest(plan: dict[str, Any], manifest: dict[str, Any]) -> bool:
    code_pins = _code_pins(manifest)
    return (
        plan.get("model_id") == manifest.get("model_id")
        and plan.get("model_revision") == manifest.get("revision")
        and plan.get("model_files") == _model_pins(manifest)
        and all(plan.get(key) == value for key, value in code_pins.items())
        and len(plan.get("quality_cases", [])) == 24
        and len(plan.get("technical_runs", [])) == 30
    )


def generate(
    manifest: dict[str, Any],
    *,
    results_dir: Path,
    endpoint: str,
    worker_token: str,
    sfx_load_endpoint: str,
    sfx_unload_endpoint: str,
    tts_load_endpoint: str,
    tts_unload_endpoint: str,
    tts_token: str | None,
    timeout_seconds: float,
    gpu_index: int,
    release_tolerance_mib: int,
    resume: bool,
) -> None:
    plan_path = results_dir / "benchmark_plan.json"
    plan = _build_plan(manifest)
    if resume and plan_path.is_file():
        prior_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not _plan_matches_manifest(prior_plan, manifest):
            raise BenchmarkError("existing benchmark plan does not match the pinned model")
        plan = prior_plan
    else:
        if results_dir.exists() and any(results_dir.iterdir()):
            raise BenchmarkError("results directory is not empty; use --resume or a new directory")
        _atomic_json(plan_path, plan)
        _write_score_template(results_dir / "manual_scores.csv", plan)

    results_path = results_dir / "generation_results.json"
    if resume and results_path.is_file():
        evidence = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        evidence = {
            "schema_version": 1,
            "model_id": manifest["model_id"],
            "model_revision": manifest["revision"],
            "quality": [],
            "technical": [],
            "technical_handoffs": [],
        }
    expected_worker_model_id = manifest.get("runtime_policy", {}).get(
        "worker_model_id", "stable-audio-3-small-sfx"
    )

    def prior_is_valid(record: object, case: dict[str, Any]) -> bool:
        if not isinstance(record, dict) or record.get("passed") is not True:
            return False
        output_path = results_dir / case["file"]
        return (
            output_path.is_file()
            and sha256_file(output_path) == record.get("metrics", {}).get("sha256")
        )

    def generate_case(case: dict[str, Any], key: str) -> dict[str, Any]:
        case_id = case.get("case_id") or case.get("run_id")
        output_path = results_dir / case["file"]
        metrics = _request_audio(
            case,
            endpoint=endpoint,
            worker_token=worker_token,
            timeout_seconds=timeout_seconds,
            output_path=output_path,
            expected_worker_model_id=expected_worker_model_id,
        )
        record = {
            "id": case_id,
            "round": case.get("round"),
            "slot": case.get("slot"),
            "duration_seconds": case["duration_seconds"],
            "latency_budget_seconds": max(5.0, float(case["duration_seconds"])),
            "file": case["file"],
            "passed": metrics["passed"],
            "metrics": metrics,
        }
        evidence[key] = [item for item in evidence[key] if item.get("id") != case_id]
        evidence[key].append(record)
        evidence["updated_at_utc"] = _utc_now()
        _atomic_json(results_path, evidence)
        if not record["passed"]:
            raise BenchmarkError(f"{case_id} failed WAV checks: {metrics['failures']}")
        return record

    def restore_tts_after_failed_start() -> None:
        try:
            _post_control(
                sfx_unload_endpoint,
                token=worker_token,
                timeout_seconds=timeout_seconds,
                expected_loaded=False,
                label="SFX recovery unload",
            )
        except BenchmarkError:
            # Loading TTS while SFX residency is uncertain could overcommit the
            # GPU. Leave recovery to the operator in that case.
            return
        try:
            _post_control(
                tts_load_endpoint,
                token=tts_token,
                timeout_seconds=timeout_seconds,
                expected_loaded=True,
                label="TTS recovery load",
            )
        except BenchmarkError:
            pass

    quality_existing = {item.get("id"): item for item in evidence["quality"]}
    quality_missing = [
        case
        for case in plan["quality_cases"]
        if not (resume and prior_is_valid(quality_existing.get(case["case_id"]), case))
    ]
    if quality_missing:
        try:
            quality_handoff = _handoff_to_sfx(
                tts_unload_endpoint=tts_unload_endpoint,
                sfx_load_endpoint=sfx_load_endpoint,
                tts_token=tts_token,
                sfx_token=worker_token,
                timeout_seconds=timeout_seconds,
                gpu_index=gpu_index,
            )
        except BenchmarkError:
            restore_tts_after_failed_start()
            raise
        quality_error: BenchmarkError | None = None
        try:
            for index, case in enumerate(quality_missing, start=1):
                print(f"quality {index}/{len(quality_missing)}: {case['case_id']}")
                generate_case(case, "quality")
                quality_handoff["memory_peak_sfx_mib"] = max(
                    quality_handoff["memory_peak_sfx_mib"],
                    _gpu_memory_used_mib(gpu_index),
                )
        except BenchmarkError as exc:
            quality_error = exc
        try:
            _handoff_back_to_tts(
                quality_handoff,
                sfx_unload_endpoint=sfx_unload_endpoint,
                tts_load_endpoint=tts_load_endpoint,
                tts_token=tts_token,
                sfx_token=worker_token,
                timeout_seconds=timeout_seconds,
                gpu_index=gpu_index,
                release_tolerance_mib=release_tolerance_mib,
            )
        except BenchmarkError as exc:
            quality_error = quality_error or exc
        evidence["quality_handoff"] = quality_handoff
        _atomic_json(results_path, evidence)
        if quality_error:
            raise quality_error

    for round_number in range(1, 6):
        round_cases = [
            case for case in plan["technical_runs"] if case["round"] == round_number
        ]
        technical_existing = {item.get("id"): item for item in evidence["technical"]}
        handoff_existing = next(
            (
                item
                for item in evidence.get("technical_handoffs", [])
                if item.get("round") == round_number
            ),
            None,
        )
        if (
            resume
            and isinstance(handoff_existing, dict)
            and handoff_existing.get("passed") is True
            and all(
                prior_is_valid(technical_existing.get(case["run_id"]), case)
                for case in round_cases
            )
        ):
            print(f"resume: technical handoff round {round_number}")
            continue

        evidence["technical"] = [
            item for item in evidence["technical"] if item.get("round") != round_number
        ]
        evidence["technical_handoffs"] = [
            item
            for item in evidence.get("technical_handoffs", [])
            if item.get("round") != round_number
        ]
        try:
            handoff = _handoff_to_sfx(
                tts_unload_endpoint=tts_unload_endpoint,
                sfx_load_endpoint=sfx_load_endpoint,
                tts_token=tts_token,
                sfx_token=worker_token,
                timeout_seconds=timeout_seconds,
                gpu_index=gpu_index,
            )
        except BenchmarkError:
            restore_tts_after_failed_start()
            raise
        handoff["round"] = round_number
        round_error: BenchmarkError | None = None
        try:
            for case in round_cases:
                print(f"technical round {round_number}/5 slot {case['slot']}/6: {case['run_id']}")
                generate_case(case, "technical")
                handoff["memory_peak_sfx_mib"] = max(
                    handoff["memory_peak_sfx_mib"],
                    _gpu_memory_used_mib(gpu_index),
                )
        except BenchmarkError as exc:
            round_error = exc
        try:
            _handoff_back_to_tts(
                handoff,
                sfx_unload_endpoint=sfx_unload_endpoint,
                tts_load_endpoint=tts_load_endpoint,
                tts_token=tts_token,
                sfx_token=worker_token,
                timeout_seconds=timeout_seconds,
                gpu_index=gpu_index,
                release_tolerance_mib=release_tolerance_mib,
            )
        except BenchmarkError as exc:
            round_error = round_error or exc
        evidence["technical_handoffs"].append(handoff)
        evidence["updated_at_utc"] = _utc_now()
        _atomic_json(results_path, evidence)
        if round_error:
            raise round_error

    warm_latency = _warm_latency_gate(evidence["technical"])
    evidence["warm_latency"] = warm_latency
    _atomic_json(results_path, evidence)
    if warm_latency["p95_passed"] is not True:
        raise BenchmarkError(
            "warm-latency gate failed: p95 latency exceeded max(5 seconds, request duration); "
            f"budget ratio={warm_latency['p95_budget_ratio']:.3f}"
        )

    print(
        "Generation complete: 24 quality WAVs, 30/30 technical runs, and five "
        "TTS->SFX->TTS GPU handoffs passed. "
        "A human reviewer must now complete manual_scores.csv, then run --evaluate."
    )


def _read_scores(path: Path, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        raise BenchmarkError(f"manual score template is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {item["case_id"]: item for item in plan["quality_cases"]}
    if len(rows) != 24 or {row.get("case_id") for row in rows} != set(expected):
        raise BenchmarkError("manual_scores.csv must contain exactly the 24 planned cases")
    scored: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        values = {}
        for field in SCORE_FIELDS[5:9]:
            try:
                value = int(str(row.get(field, "")).strip())
            except ValueError as exc:
                raise BenchmarkError(f"{case_id}: {field} must be an integer 1..5") from exc
            if value not in range(1, 6):
                raise BenchmarkError(f"{case_id}: {field} must be an integer 1..5")
            values[field] = value
        usable = str(row.get("usable_yes_no", "")).strip().lower()
        if usable not in {"yes", "no"}:
            raise BenchmarkError(f"{case_id}: usable_yes_no must be yes or no")
        passed = (
            values["semantic_match_1_5"] >= 4
            and values["event_timing_1_5"] >= 3
            and values["artifact_control_1_5"] >= 3
            and values["background_cleanliness_1_5"] >= 3
            and usable == "yes"
        )
        scored.append(
            {
                "case_id": case_id,
                "category": expected[case_id]["category"],
                "seed": expected[case_id]["seed"],
                **values,
                "usable": usable == "yes",
                "passed": passed,
            }
        )
    categories = []
    for category in QUALITY_CATEGORIES:
        cases = [item for item in scored if item["category"] == category["id"]]
        passed_count = sum(item["passed"] for item in cases)
        categories.append(
            {
                "category": category["id"],
                "cases_total": 2,
                "cases_passed": passed_count,
                # Two fixed seeds are candidate variants. A category passes when
                # at least one candidate meets all four rubric thresholds and is
                # marked usable by the reviewer.
                "passed": len(cases) == 2 and passed_count >= 1,
            }
        )
    return scored, categories


def _validate_evidence(
    evidence: dict[str, Any], plan: dict[str, Any], results_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality = evidence.get("quality", [])
    technical = evidence.get("technical", [])
    if len(quality) != 24 or len(technical) != 30:
        raise BenchmarkError("generation_results.json must contain 24 quality and 30 technical records")
    for record in [*quality, *technical]:
        output = results_dir / str(record.get("file", ""))
        if record.get("passed") is not True or not output.is_file():
            raise BenchmarkError(f"generation evidence failed or is missing: {record.get('id')}")
        expected_hash = record.get("metrics", {}).get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(output) != expected_hash:
            raise BenchmarkError(f"generated WAV changed after measurement: {record.get('id')}")
    return quality, technical


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _warm_latency_gate(technical: list[dict[str, Any]]) -> dict[str, Any]:
    warm = [item for item in technical if int(item.get("slot") or 0) >= 2]
    if len(warm) != 25:
        raise BenchmarkError("warm-latency gate requires five warm runs in each of five rounds")
    ratios = []
    for item in warm:
        elapsed = float(item.get("metrics", {}).get("elapsed_seconds", float("nan")))
        budget = float(item.get("latency_budget_seconds", 0.0))
        if not math.isfinite(elapsed) or budget <= 0:
            raise BenchmarkError(f"invalid warm-latency evidence: {item.get('id')}")
        ratios.append(elapsed / budget)
    p95_ratio = _percentile(ratios, 0.95)
    passed = p95_ratio <= 1.0
    return {
        "runs_total": len(warm),
        "budget_rule": "elapsed_seconds <= max(5, requested_duration_seconds) at p95",
        "p95_budget_ratio": p95_ratio,
        "p95_passed": passed,
        "violations": sum(value > 1.0 for value in ratios),
    }


def evaluate(manifest: dict[str, Any], *, results_dir: Path) -> Path:
    plan_path = results_dir / "benchmark_plan.json"
    results_path = results_dir / "generation_results.json"
    if not plan_path.is_file() or not results_path.is_file():
        raise BenchmarkError("run --generate successfully before --evaluate")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    evidence = json.loads(results_path.read_text(encoding="utf-8"))
    if not _plan_matches_manifest(plan, manifest):
        raise BenchmarkError("benchmark plan is stale for the current pinned model")
    _, technical = _validate_evidence(evidence, plan, results_dir)
    _, categories = _read_scores(results_dir / "manual_scores.csv", plan)
    categories_passed = sum(item["passed"] for item in categories)

    rounds = []
    for round_number in range(1, 6):
        runs = [item for item in technical if item.get("round") == round_number]
        rounds.append(
            {
                "round": round_number,
                "runs_total": len(runs),
                "runs_passed": sum(item.get("passed") is True for item in runs),
            }
        )
    technical_passed = sum(item.get("passed") is True for item in technical)
    handoffs = evidence.get("technical_handoffs", [])
    if (
        len(handoffs) != 5
        or {item.get("round") for item in handoffs if isinstance(item, dict)}
        != {1, 2, 3, 4, 5}
        or any(
            not isinstance(item, dict)
            or item.get("passed") is not True
            or item.get("memory_release_passed") is not True
            or item.get("tts_unload", {}).get("success") is not True
            or item.get("sfx_load", {}).get("success") is not True
            or item.get("sfx_unload", {}).get("success") is not True
            or item.get("tts_load", {}).get("success") is not True
            for item in handoffs
        )
    ):
        raise BenchmarkError(
            "technical gate requires five real TTS unload -> SFX load/6 runs -> "
            "SFX unload -> TTS load handoffs with GPU-memory release evidence"
        )
    warm_latency = _warm_latency_gate(technical)
    if categories_passed < 10:
        raise BenchmarkError(
            f"quality gate failed: {categories_passed}/12 categories passed; 10 are required"
        )
    if len(rounds) != 5 or any(
        item["runs_total"] != 6 or item["runs_passed"] != 6 for item in rounds
    ):
        raise BenchmarkError(
            f"technical gate failed: {technical_passed}/30 runs passed across five rounds"
        )
    if warm_latency["p95_passed"] is not True:
        raise BenchmarkError(
            "warm-latency gate failed: p95 latency exceeded max(5 seconds, request duration); "
            f"budget ratio={warm_latency['p95_budget_ratio']:.3f}"
        )

    latencies = [float(item["metrics"]["elapsed_seconds"]) for item in technical]
    real_time_factors = [float(item["metrics"]["real_time_factor"]) for item in technical]
    gate = {
        "schema_version": 1,
        "passed": True,
        "evaluated_at_utc": _utc_now(),
        "model_id": manifest["model_id"],
        "model_revision": manifest["revision"],
        "source_revisions": plan["source_revisions"],
        "model_files": _model_pins(manifest),
        **_code_pins(manifest),
        "quality": {
            "categories_total": 12,
            "cases_total": 24,
            "minimum_categories_passed": 10,
            "categories_passed": categories_passed,
            "conclusion": f"passed {categories_passed}/12; minimum is 10/12",
            "categories": categories,
        },
        "technical": {
            "rounds_total": 5,
            "runs_total": 30,
            "runs_passed": technical_passed,
            "handoffs_total": 5,
            "handoffs_passed": 5,
            "conclusion": "passed 30/30 across 5 real TTS-SFX-TTS handoff rounds",
            "rounds": rounds,
            "handoffs": handoffs,
            "warm_latency": warm_latency,
            "latency_seconds": {
                "mean": statistics.fmean(latencies),
                "p50": statistics.median(latencies),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
            "real_time_factor": {
                "mean": statistics.fmean(real_time_factors),
                "p50": statistics.median(real_time_factors),
                "p95": _percentile(real_time_factors, 0.95),
                "max": max(real_time_factors),
            },
        },
        "evidence": {
            "benchmark_plan_sha256": sha256_file(plan_path),
            "generation_results_sha256": sha256_file(results_path),
            "manual_scores_sha256": sha256_file(results_dir / "manual_scores.csv"),
        },
    }
    target = quality_gate_path(manifest, PROJECT_ROOT)
    if target is None:
        raise BenchmarkError("manifest does not define a quality gate path")
    _atomic_json(target, gate)
    return target


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--worker-token-env", default="DUBBING_WORKER_INSTANCE_TOKEN")
    parser.add_argument(
        "--sfx-load-endpoint",
        default="http://127.0.0.1:8005/api/v1/sfx/load-model",
    )
    parser.add_argument(
        "--sfx-unload-endpoint",
        default="http://127.0.0.1:8005/api/v1/sfx/unload-model",
    )
    parser.add_argument(
        "--tts-load-endpoint",
        help="required real TTS worker or platform-coordinator load endpoint",
    )
    parser.add_argument(
        "--tts-unload-endpoint",
        help="required real TTS worker or platform-coordinator unload endpoint",
    )
    parser.add_argument(
        "--tts-control-token-env",
        help="optional environment variable holding the TTS/coordinator token",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--memory-release-tolerance-mib", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        # load_manifest provides a clearer schema error before the expensive hashes.
        load_manifest(manifest_path)
        manifest = _preflight_real_model(manifest_path)
        results_dir = args.results_dir.resolve()
        if args.generate:
            worker_token = os.environ.get(args.worker_token_env)
            if not worker_token:
                raise BenchmarkError(
                    f"{args.worker_token_env} is not set; the benchmark accepts worker "
                    "credentials through the environment only"
                )
            if not args.tts_load_endpoint or not args.tts_unload_endpoint:
                raise BenchmarkError(
                    "--tts-unload-endpoint and --tts-load-endpoint are required. "
                    "Five real TTS->SFX->TTS handoff rounds cannot be replaced by "
                    "consecutive SFX requests."
                )
            if args.memory_release_tolerance_mib < 0:
                raise BenchmarkError("--memory-release-tolerance-mib must be non-negative")
            tts_token = (
                os.environ.get(args.tts_control_token_env)
                if args.tts_control_token_env
                else None
            )
            if args.tts_control_token_env and not tts_token:
                raise BenchmarkError(
                    f"{args.tts_control_token_env} is not set; control tokens are read "
                    "from the environment only"
                )
            generate(
                manifest,
                results_dir=results_dir,
                endpoint=args.endpoint,
                worker_token=worker_token,
                sfx_load_endpoint=args.sfx_load_endpoint,
                sfx_unload_endpoint=args.sfx_unload_endpoint,
                tts_load_endpoint=args.tts_load_endpoint,
                tts_unload_endpoint=args.tts_unload_endpoint,
                tts_token=tts_token,
                timeout_seconds=args.timeout_seconds,
                gpu_index=args.gpu_index,
                release_tolerance_mib=args.memory_release_tolerance_mib,
                resume=args.resume,
            )
        else:
            target = evaluate(manifest, results_dir=results_dir)
            print(f"QUALITY GATE PASSED: {target}")
    except (BenchmarkError, ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"Stable Audio SFX benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
