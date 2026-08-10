"""File-based helpers for managing segments within a project."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, Generator, List

from pydantic import ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.project import Segment
from dubbing_service.app.services.takes import (
    adopt_legacy_audio_references,
    capture_active_legacy_settings,
    ensure_take_state,
)


class SegmentStorageError(Exception):
    """与段落存储相关的异常。"""


# 项目级别的锁，防止同一项目的读-改-写竞态条件
_PROJECT_LOCKS: Dict[str, Lock] = {}
_LOCKS_LOCK = Lock()  # 用于保护 _PROJECT_LOCKS 字典本身


def _get_project_lock(project_id: str) -> Lock:
    """获取指定项目的锁，如果不存在则创建。"""
    with _LOCKS_LOCK:
        if project_id not in _PROJECT_LOCKS:
            _PROJECT_LOCKS[project_id] = Lock()
        return _PROJECT_LOCKS[project_id]


@contextmanager
def project_segments_lock(project_id: str) -> Generator[None, None, None]:
    """项目段落读写锁的上下文管理器。

    在需要 "读取-修改-写入" segments.json 的场景中使用，确保操作的原子性：

        with project_segments_lock(project_id):
            segments = load_segments(project_id)
            # 修改 segments...
            save_segments(project_id, segments)
    """
    lock = _get_project_lock(project_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _segments_path(project_id: str, *, ensure_dir: bool = False) -> Path:
    try:
        project_root = safe_project_root(project_id)
    except ValueError as exc:
        raise SegmentStorageError("非法项目 ID") from exc
    if ensure_dir:
        project_root.mkdir(parents=True, exist_ok=True)
    return project_root / "segments.json"


def load_segments(project_id: str) -> List[Segment]:
    """加载项目下的所有段落。

    如果不存在 segments.json，则返回空列表。
    """

    path = _segments_path(project_id)
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_list = json.load(f)
    except json.JSONDecodeError as exc:
        raise SegmentStorageError(
            f"segments.json 内容损坏，无法解析 JSON: {exc}"
        ) from exc

    segments: List[Segment] = []
    if not isinstance(raw_list, list):
        raise SegmentStorageError("segments.json 内容格式错误：应为列表")

    for item in raw_list:
        try:
            seg = Segment(**item)
        except ValidationError as exc:
            raise SegmentStorageError("解析 Segment 数据失败") from exc
        try:
            validate_simple_component_id(seg.id, label="segment_id")
        except ValueError as exc:
            raise SegmentStorageError(f"分段 ID 非法: {seg.id!r}") from exc
        else:
            segments.append(ensure_take_state(seg))

    segment_ids = [segment.id for segment in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise SegmentStorageError("segments.json 包含重复的分段 ID")

    # 按 index 排序，保证顺序稳定
    segments.sort(key=lambda s: s.index)
    return segments


def save_segments(project_id: str, segments: List[Segment]) -> None:
    """将段落列表保存到项目的 segments.json。

    使用原子写入：先写入临时文件，再重命名，防止并发写入导致数据损坏。
    """

    seen_ids: set[str] = set()
    for segment in segments:
        if segment.take_slots:
            adopt_legacy_audio_references(segment)
            capture_active_legacy_settings(segment)
        else:
            ensure_take_state(segment)
        try:
            segment_id = validate_simple_component_id(segment.id, label="segment_id")
        except ValueError as exc:
            raise SegmentStorageError(f"分段 ID 非法: {segment.id!r}") from exc
        if segment_id in seen_ids:
            raise SegmentStorageError(f"分段 ID 重复: {segment_id}")
        seen_ids.add(segment_id)

    path = _segments_path(project_id, ensure_dir=True)
    data = [s.model_dump(mode="json") for s in segments]

    # 原子写入：先写临时文件，再重命名
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="segments_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # os.replace 是原子操作，即使并发也能保证文件完整性
        os.replace(tmp_path, path)
    except Exception:
        # 写入失败时清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def generate_segment_id() -> str:
    """生成短 UUID 作为段落 ID。"""

    return uuid.uuid4().hex[:8]
