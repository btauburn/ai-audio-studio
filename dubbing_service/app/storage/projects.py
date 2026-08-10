"""File-based storage helpers for projects.

负责在磁盘上创建/读取/更新配音项目：
- 每个项目 = projects_root/<project_id>/
- 项目元数据保存在 project.json 中
- 预先创建若干子目录：text/、roles/prompts/、audio/segments/ 等
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, Generator, List

from pydantic import ValidationError

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.models.project import Project


class ProjectStorageError(Exception):
    """与项目存储相关的异常。"""


# 项目级别的锁，防止同一项目 project.json 的读-改-写竞态条件
_PROJECT_META_LOCKS: Dict[str, Lock] = {}
_META_LOCKS_LOCK = Lock()


def _get_project_meta_lock(project_id: str) -> Lock:
    with _META_LOCKS_LOCK:
        if project_id not in _PROJECT_META_LOCKS:
            _PROJECT_META_LOCKS[project_id] = Lock()
        return _PROJECT_META_LOCKS[project_id]


@contextmanager
def project_meta_lock(project_id: str) -> Generator[None, None, None]:
    """项目元数据读写锁的上下文管理器。

    在需要 "读取-修改-写入" project.json 的场景中使用（如角色增删改）：

        with project_meta_lock(project_id):
            project = load_project(project_id)
            # 修改 project...
            save_project(project)
    """
    lock = _get_project_meta_lock(project_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _ensure_projects_root() -> Path:
    cfg = get_config()
    root = cfg.projects_root_abs
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project_dir(project_id: str) -> Path:
    _ = _ensure_projects_root()
    try:
        return safe_project_root(project_id)
    except ValueError as exc:
        raise ProjectStorageError("非法项目 ID") from exc


def _project_meta_path(project_id: str) -> Path:
    return _project_dir(project_id) / "project.json"


def create_project(name: str, description: str | None = None) -> Project:
    """创建一个新的项目目录及其 project.json。

    返回创建好的 Project 对象。
    """

    root = _ensure_projects_root()

    # 使用短 UUID 作为项目 ID 和目录名，避免冲突
    project_id = uuid.uuid4().hex[:8]
    project_dir = root / project_id

    if project_dir.exists():  # 极小概率冲突，直接报错提示重试
        raise ProjectStorageError(f"项目目录已存在: {project_dir}")

    # 创建项目目录及子目录
    project_dir.mkdir(parents=True, exist_ok=False)

    # 子目录结构：
    # - text/: 原始文本与分章节文本
    # - roles/prompts/: 角色参考音频
    # - audio/segments/: 每段音频
    # - audio/chapters/: 章节级合并音频
    # - audio/final/: 整书导出
    # - logs/: 调试日志
    for sub in [
        "text",
        "roles/prompts",
        "audio/segments",
        "audio/chapters",
        "audio/final",
        "logs",
    ]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    project = Project(id=project_id, name=name, description=description)
    save_project(project)

    return project


def save_project(project: Project) -> None:
    """将 Project 元数据写入对应目录下的 project.json。

    使用原子写入：先写入临时文件，再重命名，防止并发写入导致数据损坏。
    """

    project_dir = _project_dir(project.id)
    if not project_dir.exists():
        raise ProjectStorageError(f"项目目录不存在: {project_dir}")

    meta_path = project_dir / "project.json"
    data = project.model_dump(mode="json")

    # 原子写入：先写临时文件，再重命名
    fd, tmp_path = tempfile.mkstemp(dir=project_dir, suffix=".tmp", prefix="project_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_project(project_id: str) -> Project:
    """读取指定项目 ID 的 project.json 并解析为 Project 对象。"""

    meta_path = _project_meta_path(project_id)
    if not meta_path.is_file():
        raise ProjectStorageError(f"项目元数据不存在: {meta_path}")

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProjectStorageError(
            f"project.json 内容损坏，无法解析 JSON: {meta_path}"
        ) from exc

    try:
        return Project(**raw)
    except ValidationError as exc:
        raise ProjectStorageError(f"解析 Project 元数据失败: {meta_path}") from exc


def list_projects() -> List[Project]:
    """列出所有有效的项目。

    遍历 projects_root 下的子目录，读取其中的 project.json。
    如果某个目录缺少或解析失败，会被跳过。
    """

    root = _ensure_projects_root()
    results: List[Project] = []

    for item in root.iterdir():
        if not item.is_dir():
            continue
        meta_path = item / "project.json"
        if not meta_path.is_file():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            project = Project(**raw)
        except Exception:
            # 单个项目损坏不影响其他项目的读取
            continue
        else:
            results.append(project)

    # Legacy projects may contain offset-naive timestamps while current
    # projects are stored in UTC. Comparing both forms directly raises a
    # TypeError and used to make the whole project-list API fail after the
    # first new-format project was created. Treat legacy values as UTC for a
    # stable, backwards-compatible order.
    def _created_at_utc(project: Project) -> datetime:
        created_at = project.created_at
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=timezone.utc)
        return created_at.astimezone(timezone.utc)

    # 按创建时间排序，旧项目在前
    results.sort(key=_created_at_utc)
    return results


def touch_project_updated_at(project_id: str, when: datetime | None = None) -> None:
    """Safely update project.updated_at under project_meta_lock."""

    ts = when or datetime.now(timezone.utc)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        project.updated_at = ts
        save_project(project)


def delete_project(project_id: str) -> None:
    """物理删除整个项目目录及其下的所有文件。

    - 删除 projects_root/<project_id>/ 目录；
    - 其中包含的 project.json、segments.json、audio、text、logs 等都会一并移除；
    - 如果目录不存在或删除失败，则抛出 ProjectStorageError。
    """

    project_dir = _project_dir(project_id)
    if not project_dir.exists():
        raise ProjectStorageError(f"项目目录不存在: {project_dir}")

    try:
        shutil.rmtree(project_dir)
    except Exception as exc:  # pragma: no cover - 简单封装
        raise ProjectStorageError(f"删除项目失败: {exc}") from exc
