from __future__ import annotations

import ast
from pathlib import Path

import pytest

from service.model_path_policy import ModelPathPolicyError, require_project_model_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS = {
    "index": (
        PROJECT_ROOT / "service" / "server.py",
        {"api_load_tts_model", "api_unload_tts_model", "synthesize_tts"},
    ),
    "cosyvoice3": (
        PROJECT_ROOT / "service" / "cosyvoice_server.py",
        {"load_model", "unload_model", "synthesize_tts"},
    ),
    "qwen_voicedesign": (
        PROJECT_ROOT / "service" / "qwen3tts_server.py",
        {"load_model", "unload_model", "synthesize_tts"},
    ),
    "qwen_customvoice": (
        PROJECT_ROOT / "service" / "qwen3tts_customvoice_server.py",
        {"load_model", "unload_model", "synthesize_tts"},
    ),
}


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize(("worker", "spec"), WORKERS.items())
def test_heavy_worker_routes_are_sync(worker: str, spec: tuple[Path, set[str]]) -> None:
    path, required_names = spec
    functions = _top_level_functions(path)

    assert required_names <= functions.keys(), worker
    for name in required_names:
        assert isinstance(functions[name], ast.FunctionDef), f"{worker}.{name} must use threadpool"
    assert isinstance(functions["health_check"], ast.AsyncFunctionDef)


@pytest.mark.parametrize(("worker", "spec"), WORKERS.items())
def test_health_exposes_instance_token_contract(worker: str, spec: tuple[Path, set[str]]) -> None:
    path, _ = spec
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    health_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HealthResponse"
    )
    fields = {
        node.target.id
        for node in health_model.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "instance_token" in fields, worker
    assert "DUBBING_WORKER_INSTANCE_TOKEN" in source, worker


def test_model_path_policy_rejects_repo_ids_and_outside_paths(tmp_path: Path) -> None:
    models = tmp_path / "models"
    valid = models / "engine"
    valid.mkdir(parents=True)
    (valid / "model.bin").write_bytes(b"model")

    assert require_project_model_dir(
        tmp_path,
        "models/engine",
        label="test model",
        required_files=("model.bin",),
    ) == valid.resolve()

    with pytest.raises(ModelPathPolicyError, match="project models directory"):
        require_project_model_dir(
            tmp_path,
            "Qwen/Qwen3-TTS-floating",
            label="test model",
            required_files=("model.bin",),
        )
    with pytest.raises(ModelPathPolicyError, match="project models directory"):
        require_project_model_dir(
            tmp_path,
            tmp_path.parent,
            label="test model",
            required_files=("model.bin",),
        )
    with pytest.raises(ModelPathPolicyError, match="incomplete"):
        require_project_model_dir(
            tmp_path,
            "models/engine",
            label="test model",
            required_files=("missing.bin",),
        )


@pytest.mark.parametrize(
    "filename",
    ["qwen3tts_server.py", "qwen3tts_customvoice_server.py"],
)
def test_qwen_workers_have_no_remote_model_fallback(filename: str) -> None:
    source = (PROJECT_ROOT / "service" / filename).read_text(encoding="utf-8")

    assert "Qwen/Qwen3-TTS-12Hz" not in source
    assert "local_files_only=True" in source
    assert "require_project_model_dir" in source
