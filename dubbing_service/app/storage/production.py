"""Atomic project-local production settings, assets and release records."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, Iterator, List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.production import AudioAsset, ProductionSettings, ReleasePackage


class ProductionStorageError(Exception):
    pass


_LOCKS: Dict[str, Lock] = {}
_LOCKS_GUARD = Lock()
T = TypeVar("T", bound=BaseModel)


def production_root(project_id: str, *, ensure: bool = False) -> Path:
    try:
        root = safe_project_root(project_id)
    except ValueError as exc:
        raise ProductionStorageError("非法项目 ID") from exc
    if not (root / "project.json").is_file():
        raise ProductionStorageError("项目不存在")
    path = root / "production"
    if ensure:
        for sub in ("assets/bgm", "assets/ambience", "assets/sfx", "cover", "releases"):
            (path / sub).mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def production_lock(project_id: str) -> Iterator[None]:
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(project_id, Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_settings(project_id: str) -> ProductionSettings:
    path = production_root(project_id) / "settings.json"
    if not path.is_file():
        return ProductionSettings()
    try:
        return ProductionSettings(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ProductionStorageError(f"制作设置损坏: {exc}") from exc


def save_settings(project_id: str, settings: ProductionSettings) -> None:
    _atomic_json(
        production_root(project_id, ensure=True) / "settings.json",
        settings.model_dump(mode="json"),
    )


def _load_items(project_id: str, name: str, model: Type[T]) -> List[T]:
    path = production_root(project_id) / name
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [model(**item) for item in raw]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ProductionStorageError(f"{name} 损坏: {exc}") from exc


def load_assets(project_id: str) -> List[AudioAsset]:
    return _load_items(project_id, "assets.json", AudioAsset)


def save_assets(project_id: str, assets: List[AudioAsset]) -> None:
    _atomic_json(
        production_root(project_id, ensure=True) / "assets.json",
        [item.model_dump(mode="json") for item in assets],
    )


def load_releases(project_id: str) -> List[ReleasePackage]:
    items = _load_items(project_id, "releases.json", ReleasePackage)
    return sorted(items, key=lambda item: item.created_at, reverse=True)


def save_releases(project_id: str, releases: List[ReleasePackage]) -> None:
    _atomic_json(
        production_root(project_id, ensure=True) / "releases.json",
        [item.model_dump(mode="json") for item in releases],
    )
