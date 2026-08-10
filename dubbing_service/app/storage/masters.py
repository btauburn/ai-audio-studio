"""Project-local immutable Master metadata storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

from pydantic import ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.master import AudioMaster


class MasterStorageError(Exception):
    pass


def masters_dir(project_id: str, *, ensure: bool = False) -> Path:
    try:
        root = safe_project_root(project_id)
    except ValueError as exc:
        raise MasterStorageError("非法项目 ID") from exc
    if not (root / "project.json").is_file():
        raise MasterStorageError("项目不存在")
    path = root / "masters"
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_master(master: AudioMaster) -> None:
    validate_simple_component_id(master.id, label="master_id")
    directory = masters_dir(master.project_id, ensure=True)
    path = directory / f"{master.id}.json"
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix="master_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(master.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_master(project_id: str, master_id: str) -> AudioMaster:
    validate_simple_component_id(master_id, label="master_id")
    path = masters_dir(project_id) / f"{master_id}.json"
    if not path.is_file():
        raise MasterStorageError("Master 不存在")
    try:
        return AudioMaster(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise MasterStorageError(f"Master 元数据损坏: {exc}") from exc


def list_masters(project_id: str) -> List[AudioMaster]:
    directory = masters_dir(project_id)
    if not directory.is_dir():
        return []
    result = []
    for path in directory.glob("*.json"):
        try:
            result.append(AudioMaster(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    result.sort(key=lambda item: item.created_at, reverse=True)
    return result
