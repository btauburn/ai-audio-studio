from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from service.stableaudio_sfx_server import (
    GeneratedAudio,
    SFXGenerateRequest,
    StableAudioRuntime,
    WorkerConfig,
    create_app,
    load_local_model_assets,
    require_loopback_host,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = PROJECT_ROOT / "service" / "stableaudio_sfx_server.py"


def _write_fake_assets(root: Path) -> Path:
    model_dir = root / "models" / "stable_audio_3" / "small-sfx"
    text_encoder = model_dir / "t5gemma-b-b-ul2"
    text_encoder.mkdir(parents=True)
    config = {
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "t5gemma",
                        "config": {
                            "repo_id": "stabilityai/stable-audio-3-small-sfx",
                            "subfolder": "t5gemma-b-b-ul2",
                        },
                    }
                ]
            }
        },
    }
    (model_dir / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake")
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        (text_encoder / name).write_bytes(b"fake")
    return model_dir


class _FakeModel:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> GeneratedAudio:
        self.calls.append(kwargs)
        duration = float(kwargs["duration_seconds"])
        frames = int(44100 * duration)
        time = np.arange(frames, dtype=np.float32) / 44100
        mono = 0.1 * np.sin(2 * np.pi * 440 * time)
        return GeneratedAudio(
            samples=np.stack([mono, mono]),
            sample_rate=44100,
            seed=int(kwargs["seed"]),
        )

    def close(self) -> None:
        self.closed = True


def test_local_asset_adapter_removes_all_hub_references(tmp_path: Path) -> None:
    model_dir = _write_fake_assets(tmp_path)
    assets = load_local_model_assets(
        WorkerConfig(project_root=tmp_path, model_dir=str(model_dir))
    )

    conditioner = assets.model_config["model"]["conditioning"]["configs"][0]["config"]
    assert conditioner["model_path"] == str((model_dir / "t5gemma-b-b-ul2").resolve())
    assert "repo_id" not in conditioner
    assert "subfolder" not in conditioner
    assert assets.checkpoint_path == (model_dir / "model.safetensors").resolve()


def test_worker_health_load_generate_unload_with_injected_model(
    tmp_path: Path,
) -> None:
    model_dir = _write_fake_assets(tmp_path)
    fake_model = _FakeModel()
    captured = {}

    def factory(assets, config):
        captured["assets"] = assets
        captured["config"] = config
        return fake_model

    runtime = StableAudioRuntime(
        WorkerConfig(
            project_root=tmp_path,
            model_dir=str(model_dir),
            instance_token="worker-token",
        ),
        model_factory=factory,
    )
    with TestClient(create_app(runtime), client=("127.0.0.1", 50000)) as client:
        worker_headers = {"X-Dubbing-Worker-Token": "worker-token"}
        unauthenticated_health = client.get("/health")
        assert unauthenticated_health.status_code == 403
        health = client.get("/health", headers=worker_headers)
        assert health.status_code == 200
        assert health.json() == {
            "status": "model_not_loaded",
            "backend": "stableaudio_sfx",
            "instance_token": "worker-token",
            "model_id": "stable-audio-3-small-sfx",
            "model_loaded": False,
            "device": "cuda:0",
            "dtype": "float16",
        }

        unauthorized = client.post("/api/v1/sfx/load-model")
        assert unauthorized.status_code == 403

        loaded = client.post("/api/v1/sfx/load-model", headers=worker_headers)
        assert loaded.status_code == 200
        assert loaded.json()["model_loaded"] is True

        generated = client.post(
            "/api/v1/sfx/generate",
            headers=worker_headers,
            json={
                "prompt": "A close wooden door slam",
                "duration_seconds": 0.5,
                "seed": 123,
            },
        )
        assert generated.status_code == 200
        assert generated.headers["content-type"] == "audio/wav"
        assert generated.headers["x-audio-seed"] == "123"
        assert generated.headers["x-audio-sample-rate"] == "44100"
        assert generated.content.startswith(b"RIFF")
        audio, sample_rate = sf.read(io.BytesIO(generated.content), always_2d=True)
        assert sample_rate == 44100
        assert audio.shape == (22050, 2)
        assert fake_model.calls == [
            {
                "prompt": "A close wooden door slam",
                "duration_seconds": 0.5,
                "negative_prompt": None,
                "seed": 123,
                "steps": 8,
                "cfg_scale": 1.0,
                "chunked_decode": True,
            }
        ]

        rejected = client.post(
            "/api/v1/sfx/generate",
            headers=worker_headers,
            json={
                "prompt": "door slam",
                "duration_seconds": 1,
                "output_path": "C:/arbitrary/file.wav",
            },
        )
        assert rejected.status_code == 422

        unloaded = client.post("/api/v1/sfx/unload-model", headers=worker_headers)
        assert unloaded.status_code == 200
        assert unloaded.json()["model_loaded"] is False
        assert fake_model.closed is True

    assert captured["assets"].model_dir == model_dir.resolve()


def test_worker_rejects_incomplete_or_outside_model_assets(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-stable-audio-model"
    runtime = StableAudioRuntime(
        WorkerConfig(
            project_root=tmp_path,
            model_dir=str(outside),
            instance_token="worker-token",
        ),
        model_factory=lambda _assets, _config: _FakeModel(),
    )
    with TestClient(create_app(runtime), client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/v1/sfx/load-model",
            headers={"X-Dubbing-Worker-Token": "worker-token"},
        )
    assert response.status_code == 503
    assert "project models directory" in response.json()["detail"]

    model_dir = _write_fake_assets(tmp_path)
    (model_dir / "t5gemma-b-b-ul2" / "model.safetensors").unlink()
    runtime = StableAudioRuntime(
        WorkerConfig(
            project_root=tmp_path,
            model_dir=str(model_dir),
            instance_token="worker-token",
        ),
        model_factory=lambda _assets, _config: _FakeModel(),
    )
    with TestClient(create_app(runtime), client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/v1/sfx/load-model",
            headers={"X-Dubbing-Worker-Token": "worker-token"},
        )
    assert response.status_code == 503
    assert "incomplete" in response.json()["detail"]


def test_worker_routes_and_loopback_contract() -> None:
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"), filename=str(WORKER_PATH))
    create_app_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    nested = {
        node.name: node
        for node in create_app_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert isinstance(nested["health_check"], ast.AsyncFunctionDef)
    for heavy_route in ("load_model", "unload_model", "generate"):
        assert isinstance(nested[heavy_route], ast.FunctionDef)

    assert require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert require_loopback_host("::1") == "::1"
    try:
        require_loopback_host("0.0.0.0")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("non-loopback worker host was accepted")

    assert SFXGenerateRequest(prompt="rain", duration_seconds=60).duration_seconds == 60
    with pytest.raises(ValidationError):
        SFXGenerateRequest(prompt="rain", duration_seconds=120.01)
    with pytest.raises(ValidationError):
        SFXGenerateRequest(prompt="   ", duration_seconds=1)


def test_worker_rejects_non_loopback_clients_and_has_no_hub_loader(
    tmp_path: Path,
) -> None:
    model_dir = _write_fake_assets(tmp_path)
    runtime = StableAudioRuntime(
        WorkerConfig(
            project_root=tmp_path,
            model_dir=str(model_dir),
            instance_token="worker-token",
        ),
        model_factory=lambda _assets, _config: _FakeModel(),
    )
    with TestClient(create_app(runtime), client=("192.0.2.10", 50000)) as client:
        response = client.get("/health")
    assert response.status_code == 403

    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "hf_hub_download" not in source
    assert "StableAudioModel.from_pretrained" not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
