#!/usr/bin/env python3
"""Loopback-only, headless Stable Audio 3 Small-SFX worker."""

from __future__ import annotations

import argparse
import copy
import gc
import io
import ipaddress
import json
import logging
import os
import secrets
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.model_path_policy import (  # noqa: E402
    ModelPathPolicyError,
    require_project_model_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_ID = "stableaudio_sfx"
DEFAULT_MODEL_ID = "stable-audio-3-small-sfx"
SAMPLE_RATE = 44_100
_REQUIRED_MODEL_FILES = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/generation_config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/special_tokens_map.json",
    "t5gemma-b-b-ul2/tokenizer.json",
    "t5gemma-b-b-ul2/tokenizer.model",
    "t5gemma-b-b-ul2/tokenizer_config.json",
)


@dataclass(frozen=True)
class WorkerConfig:
    project_root: Path = PROJECT_ROOT
    model_dir: str = "models/stable_audio_3/small-sfx"
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda:0"
    dtype: str = "float16"
    instance_token: str | None = None


@dataclass(frozen=True)
class LocalModelAssets:
    model_dir: Path
    checkpoint_path: Path
    text_encoder_dir: Path
    model_config: dict[str, Any]


@dataclass(frozen=True)
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int
    seed: int


class AudioModel(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        cfg_scale: float,
        chunked_decode: bool,
    ) -> GeneratedAudio: ...

    def close(self) -> None: ...


ModelFactory = Callable[[LocalModelAssets, WorkerConfig], AudioModel]


def _prepare_local_model_config(
    source_config: dict[str, Any], text_encoder_dir: Path
) -> dict[str, Any]:
    """Replace every gated T5Gemma Hub reference with a verified local directory."""

    config = copy.deepcopy(source_config)
    try:
        conditioners = config["model"]["conditioning"]["configs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model_config.json has no conditioning configuration") from exc

    replaced = 0
    for conditioner in conditioners:
        if not isinstance(conditioner, dict) or conditioner.get("type") != "t5gemma":
            continue
        conditioner_config = conditioner.get("config")
        if not isinstance(conditioner_config, dict):
            raise ValueError("T5Gemma conditioner configuration is invalid")
        conditioner_config.pop("repo_id", None)
        conditioner_config.pop("subfolder", None)
        conditioner_config["model_path"] = str(text_encoder_dir.resolve())
        replaced += 1
    if replaced == 0:
        raise ValueError("model_config.json has no T5Gemma conditioner")
    if int(config.get("sample_rate", 0)) != SAMPLE_RATE:
        raise ValueError("Stable Audio SFX model must use a 44100 Hz sample rate")
    if int(config.get("audio_channels", 0)) != 2:
        raise ValueError("Stable Audio SFX model must produce stereo audio")
    return config


def load_local_model_assets(worker_config: WorkerConfig) -> LocalModelAssets:
    """Validate and resolve the complete checkpoint/config/T5Gemma local asset set."""

    model_dir = require_project_model_dir(
        worker_config.project_root,
        worker_config.model_dir,
        label="Stable Audio 3 Small-SFX model directory",
        required_files=_REQUIRED_MODEL_FILES,
    )
    config_path = model_dir / "model_config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Stable Audio model_config.json is unreadable or invalid"
        ) from exc
    if not isinstance(raw_config, dict):
        raise ValueError("Stable Audio model_config.json must contain an object")

    text_encoder_dir = model_dir / "t5gemma-b-b-ul2"
    local_config = _prepare_local_model_config(raw_config, text_encoder_dir)
    return LocalModelAssets(
        model_dir=model_dir,
        checkpoint_path=model_dir / "model.safetensors",
        text_encoder_dir=text_encoder_dir,
        model_config=local_config,
    )


class LocalStableAudioModel:
    """Thin adapter around upstream SA3 that never resolves a Hub model ID."""

    def __init__(self, assets: LocalModelAssets, worker_config: WorkerConfig) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        model_half = worker_config.dtype == "float16"
        diffusion_model = load_diffusion_cond(
            assets.model_config,
            str(assets.checkpoint_path),
            device=worker_config.device,
            model_half=model_half,
        )
        if worker_config.dtype == "bfloat16":
            diffusion_model.to(torch.bfloat16)
        self._model: Any = StableAudioModel(
            diffusion_model,
            assets.model_config,
            worker_config.device,
            model_half,
        )

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        cfg_scale: float,
        chunked_decode: bool,
    ) -> GeneratedAudio:
        tensor = self._model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration_seconds,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=1,
            truncate_output_to_duration=True,
            seed=seed,
            chunked_decode=chunked_decode,
        )
        import torch

        samples = tensor[0].detach().to("cpu", dtype=torch.float32).numpy()
        return GeneratedAudio(samples=samples, sample_rate=SAMPLE_RATE, seed=seed)

    def close(self) -> None:
        self._model = None


def _default_model_factory(
    assets: LocalModelAssets, worker_config: WorkerConfig
) -> AudioModel:
    return LocalStableAudioModel(assets, worker_config)


class StableAudioRuntime:
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        self.config = worker_config
        self._model_factory = model_factory
        self._model: AudioModel | None = None
        self._model_lock = Lock()
        self._inference_lock = Lock()

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            assets = load_local_model_assets(self.config)
            self._model = self._model_factory(assets, self.config)

    def unload_model(self) -> None:
        with self._inference_lock, self._model_lock:
            model = self._model
            self._model = None
            if model is not None:
                model.close()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("CUDA cache cleanup failed", exc_info=True)
            pass

    def generate(self, request: SFXGenerateRequest) -> GeneratedAudio:
        with self._inference_lock:
            self.load_model()
            if self._model is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Stable Audio model did not load")
            seed = request.seed
            if seed is None:
                seed = secrets.randbelow(2_147_483_648)
            return self._model.generate(
                prompt=request.prompt.strip(),
                duration_seconds=request.duration_seconds,
                negative_prompt=(request.negative_prompt or "").strip() or None,
                seed=seed,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                chunked_decode=request.chunked_decode,
            )


class SFXGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1000)
    duration_seconds: float = Field(ge=0.25, le=120.0)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    steps: int = Field(default=8, ge=1, le=100)
    cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    chunked_decode: bool = True

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("prompt must not be blank")
        return prompt


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    backend: str = BACKEND_ID
    instance_token: str | None = None
    model_id: str
    model_loaded: bool
    device: str
    dtype: str


def _wav_bytes(generated: GeneratedAudio) -> tuple[bytes, float]:
    samples = np.asarray(generated.samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=0)
    if samples.ndim != 2:
        raise ValueError("generated audio must have one batch-free channel dimension")
    if samples.shape[0] in (1, 2):
        if samples.shape[0] == 1:
            samples = np.repeat(samples, 2, axis=0)
        samples = samples.T
    if samples.shape[1] != 2 or samples.shape[0] == 0:
        raise ValueError("generated audio must be non-empty stereo audio")
    if not np.isfinite(samples).all():
        raise ValueError("generated audio contains non-finite samples")
    if generated.sample_rate != SAMPLE_RATE:
        raise ValueError("generated audio has an unexpected sample rate")

    samples = np.clip(samples, -1.0, 1.0)
    target = io.BytesIO()
    sf.write(target, samples, generated.sample_rate, format="WAV", subtype="PCM_16")
    payload = target.getvalue()
    duration = samples.shape[0] / generated.sample_rate
    return payload, duration


def create_app(runtime: StableAudioRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        runtime.unload_model()

    app = FastAPI(
        title="Stable Audio 3 Small-SFX worker",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        client_host = request.client.host if request.client is not None else ""
        try:
            is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            return JSONResponse(
                status_code=403,
                content={"detail": "Stable Audio worker accepts loopback clients only"},
            )
        return await call_next(request)

    def require_worker_token(request: Request) -> None:
        expected = runtime.config.instance_token
        supplied = request.headers.get("X-Dubbing-Worker-Token")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="worker instance token is not configured",
            )
        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=403, detail="worker instance token mismatch"
            )

    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request) -> HealthResponse:
        # Health also carries the instance identity used by the parent process;
        # never expose that capability token to arbitrary local processes.
        require_worker_token(request)
        loaded = runtime.is_model_loaded()
        return HealthResponse(
            status="healthy" if loaded else "model_not_loaded",
            instance_token=runtime.config.instance_token,
            model_id=runtime.config.model_id,
            model_loaded=loaded,
            device=runtime.config.device,
            dtype=runtime.config.dtype,
        )

    @app.post("/api/v1/sfx/load-model")
    def load_model(request: Request) -> dict[str, Any]:
        require_worker_token(request)
        try:
            runtime.load_model()
        except (ModelPathPolicyError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Stable Audio model load failed")
            raise HTTPException(
                status_code=500, detail="Stable Audio model load failed"
            ) from exc
        return {"success": True, "model_loaded": runtime.is_model_loaded()}

    @app.post("/api/v1/sfx/unload-model")
    def unload_model(request: Request) -> dict[str, Any]:
        require_worker_token(request)
        runtime.unload_model()
        return {"success": True, "model_loaded": runtime.is_model_loaded()}

    @app.post("/api/v1/sfx/generate")
    def generate(request: SFXGenerateRequest, http_request: Request) -> Response:
        require_worker_token(http_request)
        try:
            generated = runtime.generate(request)
            payload, duration = _wav_bytes(generated)
        except (ModelPathPolicyError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Stable Audio generation failed")
            raise HTTPException(
                status_code=500, detail="Stable Audio generation failed"
            ) from exc
        return Response(
            content=payload,
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": str(generated.sample_rate),
                "X-Audio-Duration-Seconds": f"{duration:.6f}",
                "X-Audio-Seed": str(generated.seed),
                "X-Audio-Model-Id": runtime.config.model_id,
            },
        )

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stable Audio 3 Small-SFX worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8005, type=int)
    parser.add_argument("--model-dir", default="models/stable_audio_3/small-sfx")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    return parser


def _parse_server_args() -> argparse.Namespace:
    args, unknown = _build_parser().parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown worker arguments: %s", unknown)
    return args


def require_loopback_host(host: str) -> str:
    value = str(host).strip()
    if value.lower() == "localhost":
        return value
    try:
        if ipaddress.ip_address(value).is_loopback:
            return value
    except ValueError:
        pass
    raise ValueError("Stable Audio worker host must be a loopback address")


_args = _parse_server_args()
_default_runtime = StableAudioRuntime(
    WorkerConfig(
        model_dir=_args.model_dir,
        model_id=_args.model_id,
        device=_args.device,
        dtype=_args.dtype,
        instance_token=os.environ.get("DUBBING_WORKER_INSTANCE_TOKEN"),
    )
)
app = create_app(_default_runtime)


def is_model_loaded() -> bool:
    """Return the default worker model state for lifecycle managers and tests."""

    return _default_runtime.is_model_loaded()


if __name__ == "__main__":
    host = require_loopback_host(_args.host)
    uvicorn.run(app, host=host, port=_args.port, reload=False, log_level="info")
