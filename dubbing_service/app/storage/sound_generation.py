"""Project-local storage for semantic sound plans and generated candidates."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Iterator

from pydantic import ValidationError

from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.sound_generation import SoundPlan
from dubbing_service.app.storage.production import _atomic_json, production_root


class SoundGenerationStorageError(Exception):
    pass


_PLAN_LOCKS: dict[tuple[str, str], RLock] = {}
_PLAN_LOCKS_GUARD = Lock()


@contextmanager
def sound_plan_lock(project_id: str, plan_id: str) -> Iterator[None]:
    """Serialize every read-modify-write transaction for one SoundPlan."""

    key = (project_id, plan_id)
    with _PLAN_LOCKS_GUARD:
        lock = _PLAN_LOCKS.setdefault(key, RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def generate_sound_plan_id() -> str:
    return f"soundplan-{uuid.uuid4().hex[:12]}"


def sound_generation_root(project_id: str, *, ensure: bool = False) -> Path:
    # Validate the complete canonical root before creating any child.  In
    # particular, ``production/sound_generation`` may be restored from an
    # untrusted project archive and must not be allowed to redirect writes via
    # a directory junction/reparse point.
    production_path = production_root(project_id, ensure=False)
    project_scope = production_path.parent.resolve()
    expected_production = project_scope / "production"
    resolved_production = production_path.resolve()
    if resolved_production != expected_production:
        raise SoundGenerationStorageError("production 目录逃逸项目范围")

    root = production_path / "sound_generation"
    expected_root = expected_production / "sound_generation"
    resolved_root = root.resolve()
    if resolved_root != expected_root:
        raise SoundGenerationStorageError("音效生成目录逃逸 production 范围")
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != expected_root:
            raise SoundGenerationStorageError("音效生成目录逃逸 production 范围")
    return expected_root


def candidate_audio_root(project_id: str, *, ensure: bool = False) -> Path:
    """Return the staging area; candidates become production assets only on adoption."""

    generation_root = sound_generation_root(project_id, ensure=ensure)
    root = generation_root / "candidates"
    resolved_root = root.resolve()
    if resolved_root != generation_root / "candidates":
        raise SoundGenerationStorageError("音效候选目录逃逸项目范围")
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != generation_root / "candidates":
            raise SoundGenerationStorageError("音效候选目录逃逸项目范围")
    return resolved_root


def candidate_audio_path(
    project_id: str,
    candidate_id: str,
    *,
    ensure: bool = False,
) -> Path:
    """Resolve one immutable candidate WAV below the project staging root."""

    try:
        safe_id = validate_simple_component_id(candidate_id, label="sound candidate id")
    except ValueError as exc:
        raise SoundGenerationStorageError("非法音效候选 ID") from exc
    root = candidate_audio_root(project_id, ensure=ensure)
    expected_target = root / f"{safe_id}.wav"
    target = expected_target.resolve()
    if target != expected_target:
        raise SoundGenerationStorageError("音效候选路径逃逸候选目录")
    return target


def _plan_path(project_id: str, plan_id: str, *, ensure: bool = False) -> Path:
    try:
        safe_id = validate_simple_component_id(plan_id, label="sound plan id")
    except ValueError as exc:
        raise SoundGenerationStorageError("非法音效计划 ID") from exc
    generation_root = sound_generation_root(project_id, ensure=ensure)
    plans_path = generation_root / "plans"
    plans_root = plans_path.resolve()
    if plans_root != generation_root / "plans":
        raise SoundGenerationStorageError("音效计划目录逃逸项目范围")
    if ensure:
        plans_path.mkdir(parents=True, exist_ok=True)
        if plans_path.resolve() != generation_root / "plans":
            raise SoundGenerationStorageError("音效计划目录逃逸项目范围")
    expected_target = plans_root / f"{safe_id}.json"
    target = expected_target.resolve()
    if target != expected_target:
        raise SoundGenerationStorageError("音效计划路径逃逸计划目录")
    return target


def save_sound_plan(project_id: str, plan: SoundPlan) -> None:
    if plan.project_id != project_id:
        raise SoundGenerationStorageError("音效计划不属于当前项目")
    try:
        _atomic_json(
            _plan_path(project_id, plan.id, ensure=True),
            plan.model_dump(mode="json"),
        )
    except SoundGenerationStorageError:
        raise
    except Exception as exc:
        raise SoundGenerationStorageError(f"保存音效计划失败: {exc}") from exc


def load_sound_plan(project_id: str, plan_id: str) -> SoundPlan:
    path = _plan_path(project_id, plan_id)
    if not path.is_file():
        raise SoundGenerationStorageError("音效计划不存在")
    try:
        plan = SoundPlan(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise SoundGenerationStorageError(f"音效计划损坏: {exc}") from exc
    if plan.project_id != project_id:
        raise SoundGenerationStorageError("音效计划项目归属不一致")
    return plan


def list_sound_plans(project_id: str) -> list[SoundPlan]:
    # Resolve the directory through the same scoped helper used by individual
    # records, so a project-local junction cannot redirect enumeration.
    plans_dir = _plan_path(project_id, "scope-probe").parent
    if not plans_dir.is_dir():
        return []
    plans: list[SoundPlan] = []
    for path in plans_dir.glob("*.json"):
        if not path.is_file():
            continue
        try:
            plan = SoundPlan(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise SoundGenerationStorageError(f"音效计划损坏 ({path.name}): {exc}") from exc
        if plan.project_id != project_id:
            raise SoundGenerationStorageError(f"音效计划项目归属不一致 ({path.name})")
        plans.append(plan)
    return sorted(plans, key=lambda item: item.created_at, reverse=True)


def delete_sound_plan(project_id: str, plan_id: str) -> bool:
    path = _plan_path(project_id, plan_id)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise SoundGenerationStorageError(f"删除音效计划失败: {exc}") from exc
    return True
