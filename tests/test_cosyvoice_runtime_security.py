from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COSY_PROJECT = ROOT / "environments" / "cosyvoice" / "pyproject.toml"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def test_cosyvoice_runtime_uses_security_baseline_versions() -> None:
    project = tomllib.loads(COSY_PROJECT.read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])

    assert "torch==2.6.0" in dependencies
    assert "torchaudio==2.6.0" in dependencies
    assert "modelscope==1.27.0" in dependencies
    assert "protobuf==5.29.6" in dependencies

    training_only = {
        "diffusers",
        "gdown",
        "hydra-core",
        "lightning",
        "matplotlib",
        "onnx",
        "pyarrow",
        "pyworld",
        "setuptools",
    }
    declared_names = {dependency.split("=", 1)[0].split("<", 1)[0] for dependency in dependencies}
    assert declared_names.isdisjoint(training_only)


def test_cosyvoice_runtime_defers_optional_training_imports() -> None:
    processor = ROOT / "cosyvoice" / "cosyvoice" / "dataset" / "processor.py"
    hifigan_utils = (
        ROOT
        / "cosyvoice"
        / "third_party"
        / "Matcha-TTS"
        / "matcha"
        / "hifigan"
        / "xutils.py"
    )
    flow_matching = (
        ROOT
        / "cosyvoice"
        / "third_party"
        / "Matcha-TTS"
        / "matcha"
        / "models"
        / "components"
        / "flow_matching.py"
    )

    assert _top_level_imports(processor).isdisjoint({"pyarrow", "pyworld"})
    assert "matplotlib" not in _top_level_imports(hifigan_utils)
    assert "diffusers" not in _top_level_imports(flow_matching)


def test_cosyvoice_loader_is_offline_only() -> None:
    loader = ROOT / "cosyvoice" / "cosyvoice" / "cli" / "cosyvoice.py"
    source = loader.read_text(encoding="utf-8")

    assert "modelscope" not in _top_level_imports(loader)
    assert "snapshot_download" not in source
    assert "scripts/install_models.ps1" in source
    assert "_require_local_model_dir" in source
