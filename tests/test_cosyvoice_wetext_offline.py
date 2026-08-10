from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WETEXT_UTILS = PROJECT_ROOT / "cosyvoice" / "cosyvoice" / "utils" / "wetext_utils.py"
_SPEC = importlib.util.spec_from_file_location("cosyvoice_wetext_utils_test", WETEXT_UTILS)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_project_wetext_normalizers = _MODULE.load_project_wetext_normalizers
load_wetext_from_environment = _MODULE.load_wetext_from_environment


def _write_test_fsts(root: Path) -> None:
    for language in ("en", "zh"):
        directory = root / language / "tn"
        directory.mkdir(parents=True)
        (directory / "tagger.fst").write_bytes(b"tagger")
        (directory / "verbalizer.fst").write_bytes(b"verbalizer")


def test_wetext_uses_explicit_local_graphs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_test_fsts(tmp_path)
    calls: list[dict] = []

    class FakeNormalizer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "wetext",
        types.SimpleNamespace(Normalizer=FakeNormalizer),
    )
    monkeypatch.setenv("COSYVOICE_WETEXT_DIR", str(tmp_path))

    zh_normalizer, en_normalizer = load_wetext_from_environment()

    assert isinstance(zh_normalizer, FakeNormalizer)
    assert isinstance(en_normalizer, FakeNormalizer)
    assert [call["lang"] for call in calls] == ["zh", "en"]
    assert all(call["operator"] == "tn" for call in calls)
    assert all(call["tagger_path"].endswith("tagger.fst") for call in calls)
    assert all(call["verbalizer_path"].endswith("verbalizer.fst") for call in calls)
    assert all(Path(call["tagger_path"]).is_file() for call in calls)
    assert all(Path(call["verbalizer_path"]).is_file() for call in calls)


def test_wetext_fails_before_import_when_graph_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delitem(sys.modules, "wetext", raising=False)

    with pytest.raises(FileNotFoundError, match="missing project-local WeText FST"):
        load_project_wetext_normalizers(str(tmp_path))


def test_wetext_environment_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COSYVOICE_WETEXT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="COSYVOICE_WETEXT_DIR is required"):
        load_wetext_from_environment()
