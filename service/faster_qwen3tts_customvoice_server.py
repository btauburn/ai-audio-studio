#!/usr/bin/env python3
"""Opt-in faster-qwen3-tts 0.5.0 worker behind the existing CustomVoice API.

This worker shares the native server's routes, worker token, output policy and
request schema. Its model and CUDA graphs live only in the isolated
``faster_qwen3tts`` environment. Upstream 0.5.0 does not report why its decode
loop stopped, so this adapter accepts only an early EOS exit and refuses to
write audio when a token or static-cache limit was reached.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional
import uuid

# Never allow a local model path to trigger a Hub lookup or download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from dubbing_service.app.core.qwen_graph_policy import classify_graph_termination
PROFILE = "qwen-cudagraph"
import service.qwen3tts_customvoice_server as native

# The shared CLI schema remains native-only when launching the native worker.
# This worker adds its profile to the same parser before FastAPI lifespan parses.
native.parser._option_string_actions["--execution-profile"].choices = [
    "native-current",
    "native-sdpa",
    PROFILE,
]


FASTER_VERSION = "0.5.0"
GRAPH_MAX_SEQ_LEN = 2048
_generation_local = threading.local()
_request_events: dict[str, threading.Event] = {}
_request_results: dict[str, dict[str, Any]] = {}
_request_events_lock = threading.Lock()
_raw_fast_generate = None
_generation_config: dict[str, Any] = {}
_capture_count = 0
_warmup_seconds: Optional[float] = None


class CancelRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=200)


def _version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _model_config(model_path: Path) -> dict[str, Any]:
    path = model_path / "generation_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"missing or invalid local Qwen generation config: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Qwen generation config must be a JSON object")
    required = (
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "max_new_tokens",
        "subtalker_dosample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(f"Qwen generation config is missing controls: {missing}")
    return {key: value[key] for key in required}


def _verify_local_weight_hashes(model_path: Path) -> dict[str, str]:
    """Validate the local talker and codec files against the checked-in manifest."""

    manifest_path = Path(native.project_root) / "models" / "manifests" / "qwen3_tts_customvoice.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"missing or invalid local Qwen model manifest: {manifest_path}") from exc
    listed = {
        str(item.get("path", "")): str(item.get("sha256", ""))
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    verified: dict[str, str] = {}
    for relative in ("model.safetensors", "speech_tokenizer/model.safetensors"):
        target = (model_path / relative).resolve()
        expected = next(
            (digest for name, digest in listed.items() if name.replace("\\", "/").endswith(relative)),
            "",
        )
        if target is None or len(expected) != 64:
            raise RuntimeError(f"manifest or local model file is missing for {relative}")
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"local model checksum mismatch for {relative}")
        verified[relative] = actual
    return verified


def _profile_id(profile: dict[str, Any]) -> str:
    encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _install_generation_guard() -> None:
    """Guard the exact pinned upstream loop without editing site-packages."""

    global _raw_fast_generate
    if _raw_fast_generate is not None:
        return
    if _version("faster-qwen3-tts") != FASTER_VERSION:
        raise RuntimeError(
            f"qwen-cudagraph requires faster-qwen3-tts=={FASTER_VERSION}; "
            f"found {_version('faster-qwen3-tts')!r}"
        )

    import inspect
    from faster_qwen3_tts import generate as generation_module

    raw = generation_module.fast_generate
    source = inspect.getsource(raw)
    sentinel = "current_pos >= talker_graph.max_seq_len - 1"
    if sentinel not in source or "timing =" not in source or "'steps': n_steps" not in source:
        raise RuntimeError(
            "faster-qwen3-tts decode source no longer matches the reviewed 0.5.0 "
            "static-buffer guard contract"
        )
    _raw_fast_generate = raw

    def guarded_fast_generate(**kwargs: Any):
        event: Optional[threading.Event] = getattr(_generation_local, "cancel_event", None)
        if event is not None and event.is_set():
            raise RuntimeError("generation cancelled before graph decode")

        graph = kwargs["talker_graph"]
        predictor = kwargs["predictor_graph"]
        original_prefill = graph.prefill_kv
        original_talker_run = graph.run
        original_predictor_run = predictor.run
        observed: dict[str, int] = {}

        def record_prefill(cache: Any) -> int:
            length = int(original_prefill(cache))
            observed["prefill_tokens"] = length
            return length

        def checked_talker_run(input_embeds: Any, position: int):
            if event is not None and event.is_set():
                raise RuntimeError("generation cancelled during talker graph decode")
            return original_talker_run(input_embeds, position)

        def checked_predictor_run(pred_input: Any):
            if event is not None and event.is_set():
                raise RuntimeError("generation cancelled during predictor graph decode")
            return original_predictor_run(pred_input)

        graph.prefill_kv = record_prefill
        graph.run = checked_talker_run
        predictor.run = checked_predictor_run
        started = time.perf_counter()
        try:
            kwargs["subtalker_dosample"] = _generation_config["subtalker_dosample"]
            kwargs["subtalker_top_k"] = _generation_config["subtalker_top_k"]
            kwargs["subtalker_top_p"] = _generation_config["subtalker_top_p"]
            kwargs["subtalker_temperature"] = _generation_config["subtalker_temperature"]
            codec_ids, timing = _raw_fast_generate(**kwargs)
        finally:
            graph.prefill_kv = original_prefill
            graph.run = original_talker_run
            predictor.run = original_predictor_run

        elapsed = time.perf_counter() - started
        if event is not None and event.is_set():
            raise RuntimeError("generation cancelled before waveform decode")
        if codec_ids is None:
            raise RuntimeError("graph generation returned no codec frames")

        steps = int(timing.get("steps", -1))
        prefill_tokens = observed.get("prefill_tokens")
        if prefill_tokens is None:
            raise RuntimeError("graph generation did not expose the prefill length")
        termination = classify_graph_termination(
            generated_steps=steps,
            max_new_tokens=int(kwargs["max_new_tokens"]),
            prefill_tokens=prefill_tokens,
            graph_max_seq_len=int(graph.max_seq_len),
        )
        _generation_local.metadata = {
            "generated_codec_steps": steps,
            "termination_reason": termination,
            "prefill_tokens": prefill_tokens,
            "graph_max_seq_len": int(graph.max_seq_len),
            "max_new_tokens": int(kwargs["max_new_tokens"]),
            "fast_generate_seconds": elapsed,
            "fast_generate_timing": dict(timing),
        }
        return codec_ids, timing

    generation_module.fast_generate = guarded_fast_generate


def _graph_objects(model: Any) -> tuple[Any, Any]:
    predictor = getattr(model, "predictor_graph", None)
    talker = getattr(model, "talker_graph", None)
    if predictor is None or talker is None:
        raise RuntimeError("faster-qwen3-tts did not construct both CUDA graphs")
    return predictor, talker


def _record_request_result(request_id: str, result: dict[str, Any]) -> None:
    with _request_events_lock:
        _request_results[request_id] = result
        while len(_request_results) > 512:
            _request_results.pop(next(iter(_request_results)))


def _load_model_if_needed() -> None:
    global _generation_config, _capture_count, _warmup_seconds
    import torch

    if native.qwen3tts_model is not None:
        return
    lock = native._model_lock
    if lock is not None:
        lock.acquire()
    try:
        if native.qwen3tts_model is not None:
            return
        args = native.config or native._parse_server_args()
        native.config = args
        if args.execution_profile != PROFILE:
            raise RuntimeError(
                f"graph worker received execution profile {args.execution_profile!r}; expected {PROFILE!r}"
            )
        model_path = native._validated_model_path(args)
        if str(args.device) != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("qwen-cudagraph requires the configured CUDA device; no CPU fallback was used")
        if str(args.dtype) != "bfloat16":
            raise RuntimeError("qwen-cudagraph requires the existing BF16 precision")
        load_started = time.perf_counter()
        _generation_config = _model_config(model_path)
        verify_started = time.perf_counter()
        verified_weight_hashes = _verify_local_weight_hashes(model_path)
        weight_verify_seconds = time.perf_counter() - verify_started

        _install_generation_guard()
        from faster_qwen3_tts import FasterQwen3TTS

        model_init_started = time.perf_counter()
        model = FasterQwen3TTS.from_pretrained(
            str(model_path),
            device=args.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            max_seq_len=GRAPH_MAX_SEQ_LEN,
            backend="torch",
        )
        initialized = time.perf_counter()

        components = native._attention_components(model.model)
        for key in ("root", "talker", "predictor"):
            actual = components.get(key)
            if actual != "sdpa":
                raise RuntimeError(
                    f"qwen-cudagraph requested SDPA, but {key} resolved to {actual!r}"
                )

        predictor, talker = _graph_objects(model)
        predictor_controls = {
            "do_sample": bool(_generation_config["subtalker_dosample"]),
            "top_k": int(_generation_config["subtalker_top_k"]),
            "top_p": float(_generation_config["subtalker_top_p"]),
            "temperature": float(_generation_config["subtalker_temperature"]),
        }
        actual_predictor_controls = {
            "do_sample": bool(predictor.do_sample),
            "top_k": int(predictor.top_k),
            "top_p": float(predictor.top_p),
            "temperature": float(predictor.temperature),
        }
        if actual_predictor_controls != predictor_controls:
            raise RuntimeError(
                "CUDA predictor graph sampling controls differ from the local model config: "
                f"expected {predictor_controls}, got {actual_predictor_controls}"
            )

        warmup_started = time.perf_counter()
        model.warmup(prefill_len=min(100, GRAPH_MAX_SEQ_LEN - 2))
        predictor, talker = _graph_objects(model)
        if not bool(getattr(predictor, "captured", False)) or not bool(getattr(talker, "captured", False)):
            raise RuntimeError("CUDA graph warmup returned without capturing both graphs")
        _warmup_seconds = time.perf_counter() - warmup_started
        _capture_count = 2

        load_seconds = time.perf_counter() - load_started
        native.qwen3tts_model = model
        native._model_load_count += 1
        native._last_model_load_seconds = load_seconds
        profile: dict[str, Any] = {
            **native._model_manifest_identity(),
            "backend": "qwen3tts_cv",
            "requested_execution_profile": PROFILE,
            "resolved_execution_profile": PROFILE,
            "backend_api_version": "1.1.0",
            "faster_qwen3_tts_version": _version("faster-qwen3-tts"),
            "qwen_tts_hf_version": _version("qwen-tts-hf"),
            "transformers_version": _version("transformers"),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(args.device),
            "device": args.device,
            "dtype": str(model.dtype).replace("torch.", ""),
            "attention_requested": "sdpa",
            "attention_resolved": "sdpa",
            "attention_components": components,
            "decoding_path": "faster_qwen3_tts_0.5.0.torch_cuda_graph",
            "non_streaming_mode": True,
            "graph_max_seq_len": GRAPH_MAX_SEQ_LEN,
            "graph_capture_count": _capture_count,
            "verified_weight_sha256": verified_weight_hashes,
            "graphs_captured": True,
            "graph_warmup_seconds": round(_warmup_seconds or 0.0, 6),
            "model_initialize_seconds": round(initialized - model_init_started, 6),
            "weight_hash_verification_seconds": round(weight_verify_seconds, 6),
            "generation_controls": {**_generation_config, "non_streaming_mode": True},
            "seed_policy": "existing unseeded global RNG policy; no per-request seed",
            "batching_policy": "single-request-serialized",
        }
        profile["execution_profile_id"] = _profile_id(profile)
        native._execution_profile = profile
        print(
            "Qwen CUDA graphs captured: predictor=1 talker=1; profile="
            + profile["execution_profile_id"],
            flush=True,
        )
    finally:
        if lock is not None and lock.locked():
            lock.release()


native._load_qwen3tts_model_if_needed = _load_model_if_needed


def _remove_output_file(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


def _controls() -> dict[str, Any]:
    return {**_generation_config, "min_new_tokens": 2, "non_streaming_mode": True}


def _remove_native_synth_route() -> None:
    native.app.router.routes = [
        route
        for route in native.app.router.routes
        if not (
            getattr(route, "path", None) == "/api/v1/tts/synthesize"
            and "POST" in (getattr(route, "methods", set()) or set())
        )
    ]


_remove_native_synth_route()


@native.app.post("/api/v1/tts/synthesize", response_model=native.TTSResponse)
def synthesize_tts(request: native.TTSRequest, background_tasks: BackgroundTasks):
    try:
        speaker = native.resolve_customvoice_speaker(request.speaker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    import torch
    import soundfile as sf

    request_started = time.perf_counter()
    request_id = request.request_id or uuid.uuid4().hex
    cancel_event = threading.Event()
    with _request_events_lock:
        if request_id in _request_events:
            raise HTTPException(status_code=409, detail="request_id is already active")
        _request_events[request_id] = cancel_event
        _request_results[request_id] = {"state": "active", "request_id": request_id}

    acquired = False
    output_path: Optional[Path] = None
    try:
        before_load = native._model_load_count
        load_started = time.perf_counter()
        _load_model_if_needed()
        load_call_seconds = time.perf_counter() - load_started
        load_performed = native._model_load_count > before_load
        if native.qwen3tts_model is None:
            raise HTTPException(status_code=500, detail="CUDA graph model failed to load")

        preparation_started = time.perf_counter()
        instruct = request.instruct or ""
        if request.emotion_text and request.emotion_text.strip():
            instruct = f"{instruct}，{request.emotion_text}" if instruct else request.emotion_text
        language_map = {
            "auto": "Auto", "zh": "Chinese", "en": "English", "ja": "Japanese",
            "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian",
            "pt": "Portuguese", "es": "Spanish", "it": "Italian",
        }
        language = language_map.get(request.language.casefold(), request.language)
        model_speaker = native._SPEAKER_ID_TO_MODEL_NAME.get(speaker, speaker)
        preparation_seconds = time.perf_counter() - preparation_started

        queue_started = time.perf_counter()
        lock = native._infer_lock
        if lock is not None:
            lock.acquire()
            acquired = True
        queue_wait_seconds = time.perf_counter() - queue_started
        if cancel_event.is_set():
            raise HTTPException(status_code=409, detail="generation cancelled before dispatch")

        torch.cuda.reset_peak_memory_stats()
        generation_started = time.perf_counter()
        _generation_local.cancel_event = cancel_event
        _generation_local.metadata = None
        try:
            wavs, sample_rate = native.qwen3tts_model.generate_custom_voice(
                text=request.text,
                speaker=model_speaker,
                language=language,
                instruct=instruct or None,
                non_streaming_mode=True,
                max_new_tokens=int(_generation_config["max_new_tokens"]),
                min_new_tokens=2,
                temperature=float(_generation_config["temperature"]),
                top_k=int(_generation_config["top_k"]),
                top_p=float(_generation_config["top_p"]),
                do_sample=bool(_generation_config["do_sample"]),
                repetition_penalty=float(_generation_config["repetition_penalty"]),
            )
        finally:
            _generation_local.cancel_event = None
        generation_seconds = time.perf_counter() - generation_started
        generation_metadata = getattr(_generation_local, "metadata", None)
        if not isinstance(generation_metadata, dict):
            raise RuntimeError("guarded graph generation did not report EOS and token timing")
        if cancel_event.is_set():
            raise HTTPException(status_code=409, detail="generation cancelled before audio save")
        if wavs is None or len(wavs) != 1 or sample_rate != 24000:
            raise RuntimeError(f"unexpected Qwen graph waveform output: sample_rate={sample_rate}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("outputs") / f"qwen3tts_graph_{timestamp}_{uuid.uuid4().hex}.wav"
        file_started = time.perf_counter()
        sf.write(str(output_path), wavs[0], sample_rate)
        file_write_seconds = time.perf_counter() - file_started
        if cancel_event.is_set():
            output_path.unlink(missing_ok=True)
            output_path = None
            raise HTTPException(status_code=409, detail="generation cancelled during audio write")
        duration = len(wavs[0]) / sample_rate
        audio_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        total_seconds = time.perf_counter() - request_started
        fast_seconds = float(generation_metadata["fast_generate_seconds"])
        generation_timings = generation_metadata.get("fast_generate_timing", {})
        gpu_peak_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        profile = dict(native._execution_profile)
        stage_timings = {
            "schema": "qwen3tts-stage-timings-v1",
            "clock": "time.perf_counter",
            "inclusive": {
                "request_through_file_seconds": round(total_seconds, 6),
                "model_generate_seconds": round(generation_seconds, 6),
            },
            "exclusive": {
                "model_load_call_seconds": round(load_call_seconds, 6),
                "preparation_seconds": round(preparation_seconds, 6),
                "infer_queue_wait_seconds": round(queue_wait_seconds, 6),
                "fast_generate_seconds": round(fast_seconds, 6),
                "waveform_decode_seconds": round(max(0.0, generation_seconds - fast_seconds), 6),
                "file_write_seconds": round(file_write_seconds, 6),
                "graph_warmup_capture_seconds": (
                    round(float(_warmup_seconds or 0.0), 6) if load_performed else 0.0
                ),
            },
            "model_load_performed": load_performed,
            "worker_pid": os.getpid(),
            "model_load_count": native._model_load_count,
            "model_load_seconds": (
                round(native._last_model_load_seconds, 6)
                if load_performed and native._last_model_load_seconds is not None
                else 0.0
            ),
            "graph_capture_count": _capture_count,
            "gpu_peak_allocated_mib": gpu_peak_mib,
            "fast_generate_timing": generation_timings,
        }
        response = native.TTSResponse(
            success=True,
            message="合成成功",
            audio_file=str(output_path.absolute()) if request.save_file else None,
            duration_seconds=duration,
            sample_rate=sample_rate,
            audio_base64=(
                base64.b64encode(output_path.read_bytes()).decode("ascii")
                if request.return_base64
                else None
            ),
            request_id=request_id,
            utterance_id=request.utterance_id,
            effective_speaker=speaker,
            execution_profile=profile,
            stage_timings=stage_timings,
            generation={
                "effective_speaker": speaker,
                "controls": _controls(),
                "input_text_characters": len(request.text),
                "generated_codec_steps": generation_metadata["generated_codec_steps"],
                "termination_reason": generation_metadata["termination_reason"],
                "prefill_tokens": generation_metadata["prefill_tokens"],
                "graph_max_seq_len": generation_metadata["graph_max_seq_len"],
                "audio_sha256": audio_sha256,
            },
        )
        _record_request_result(
            request_id,
            {"state": "completed", "request_id": request_id, "response": response.model_dump()},
        )
        return response
    except HTTPException as exc:
        _record_request_result(
            request_id,
            {"state": "failed", "request_id": request_id, "error": str(exc.detail)},
        )
        raise
    except Exception as exc:  # noqa: BLE001 - surface graph/device failures as API detail
        _record_request_result(
            request_id,
            {"state": "failed", "request_id": request_id, "error": str(exc)},
        )
        native.logger.error("CUDA graph 合成失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CUDA graph 合成失败: {exc}") from exc
    finally:
        _generation_local.cancel_event = None
        if acquired and native._infer_lock is not None:
            native._infer_lock.release()
        with _request_events_lock:
            _request_events.pop(request_id, None)
        if output_path is not None and not request.save_file:
            background_tasks.add_task(_remove_output_file, output_path)
        elif output_path is not None and request.save_file and native.config:
            background_tasks.add_task(
                native.cleanup_outputs_dir,
                Path("outputs"),
                native.config.outputs_keep_days,
                native.config.outputs_max_files,
            )


@native.app.post("/api/v1/tts/cancel")
def cancel_tts(request: CancelRequest):
    with _request_events_lock:
        event = _request_events.get(request.request_id)
        if event is None:
            state = _request_results.get(request.request_id, {}).get("state", "not_found")
            return {"success": False, "request_id": request.request_id, "state": state}
        event.set()
    return {"success": True, "request_id": request.request_id, "state": "cancellation_requested"}


@native.app.post("/api/v1/tts/status")
def tts_request_status(request: CancelRequest):
    with _request_events_lock:
        result = _request_results.get(request.request_id)
        if result is None:
            return {"state": "not_found", "request_id": request.request_id}
        return dict(result)


app = native.app


if __name__ == "__main__":
    args = native._parse_server_args()
    if args.execution_profile != PROFILE:
        raise SystemExit(f"expected --execution-profile {PROFILE}")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")
