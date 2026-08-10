from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COSY_FRONTEND = ROOT / "cosyvoice" / "cosyvoice" / "cli" / "frontend.py"
INDEX_FRONTEND = ROOT / "indextts" / "utils" / "front.py"


def _load_function(path: Path, name: str) -> tuple[Callable[[str], str], ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - execute only the extracted, repository-owned helper
        compile(ast.fix_missing_locations(isolated), str(path), "exec"),
        namespace,
    )
    return namespace[name], function  # type: ignore[return-value]


def _uses_regex(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
            return True
    return False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一句话，，,、", "一句话。"),
        ("一句话。", "一句话。"),
        ("，,、", "。"),
        ("", ""),
    ],
)
def test_cosy_trailing_comma_replacement_preserves_behavior(text: str, expected: str) -> None:
    replace_trailing, _ = _load_function(COSY_FRONTEND, "_replace_trailing_comma_run")

    assert replace_trailing(text) == expected


def test_cosy_trailing_comma_replacement_is_linear_string_processing() -> None:
    replace_trailing, function = _load_function(COSY_FRONTEND, "_replace_trailing_comma_run")
    adversarial = "," * 100_000 + "x"

    assert not _uses_regex(function)
    assert replace_trailing(adversarial) == adversarial
    assert "_replace_trailing_comma_run(text)" in COSY_FRONTEND.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GPT <H> 五 <H> nano", "GPT-五-nano"),
        ("A<H><H>B", "A--B"),
        (" <H> ", "-"),
        ("没有占位符", "没有占位符"),
    ],
)
def test_index_hyphen_placeholder_restoration_preserves_behavior(text: str, expected: str) -> None:
    restore_hyphens, _ = _load_function(INDEX_FRONTEND, "_restore_hyphen_placeholders")

    assert restore_hyphens(text) == expected


def test_index_hyphen_restoration_is_linear_string_processing() -> None:
    restore_hyphens, function = _load_function(INDEX_FRONTEND, "_restore_hyphen_placeholders")
    adversarial = " " * 100_000 + "x"

    assert not _uses_regex(function)
    assert restore_hyphens(adversarial) == adversarial
    assert "return _restore_hyphen_placeholders(normalized_text)" in INDEX_FRONTEND.read_text(encoding="utf-8")
