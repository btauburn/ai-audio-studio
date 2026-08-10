"""Fail-closed path policy shared by local TTS workers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_REPARSE_POINT = 0x400


class ModelPathPolicyError(RuntimeError):
    """Raised when a worker model path is not a materialized project asset."""


def _is_link_or_reparse(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(
        getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT
    )


def require_project_model_dir(
    project_root: str | Path,
    configured_path: str | Path,
    *,
    label: str,
    required_files: Iterable[str],
) -> Path:
    """Return a verified directory below ``<project>/models``.

    Repository IDs, missing paths, paths outside the project model root, and
    cache links are rejected.  Workers must never reinterpret those values as
    a remote model identifier.
    """
    root = Path(project_root).resolve()
    models_root = (root / "models").resolve()
    raw_value = str(configured_path).strip()
    if not raw_value:
        raise ModelPathPolicyError(f"{label} is empty")

    raw_path = Path(raw_value).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(models_root)
    except ValueError as exc:
        raise ModelPathPolicyError(
            f"{label} must be inside the project models directory: {models_root}"
        ) from exc
    if _is_link_or_reparse(candidate):
        raise ModelPathPolicyError(f"{label} must be a materialized directory, not a link")
    if not resolved.is_dir():
        raise ModelPathPolicyError(
            f"{label} does not exist as a local directory: {resolved}"
        )

    invalid: list[str] = []
    for relative_value in required_files:
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelPathPolicyError(
                f"invalid required-file declaration for {label}: {relative_value}"
            )
        target = resolved / relative
        target_resolved = target.resolve(strict=False)
        try:
            target_resolved.relative_to(resolved)
        except ValueError:
            invalid.append(relative.as_posix())
            continue
        if (
            not target.is_file()
            or _is_link_or_reparse(target)
            or not target_resolved.is_file()
        ):
            invalid.append(relative.as_posix())
    if invalid:
        raise ModelPathPolicyError(
            f"{label} is incomplete; missing or linked required file(s): "
            + ", ".join(invalid)
        )
    return resolved


def require_project_model_dir_from_env(
    project_root: str | Path,
    variable: str,
    *,
    label: str,
    required_files: Iterable[str],
) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ModelPathPolicyError(f"{variable} is required for {label}")
    return require_project_model_dir(
        project_root,
        value,
        label=label,
        required_files=required_files,
    )
