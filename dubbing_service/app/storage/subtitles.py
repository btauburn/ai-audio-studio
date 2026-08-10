"""File-based helpers for managing subtitles within a project."""

from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from typing import List

from pydantic import ValidationError

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.subtitles import SubtitleFile


class SubtitleStorageError(Exception):
    """与字幕存储相关的异常。"""


def _write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _subtitles_dir(project_id: str, *, ensure_dir: bool = False) -> Path:
    """返回项目下字幕目录的绝对路径。"""

    try:
        project_root = safe_project_root(project_id)
    except ValueError as exc:
        raise SubtitleStorageError("非法项目 ID") from exc
    subs_dir = project_root / "subtitles"
    if ensure_dir:
        subs_dir.mkdir(parents=True, exist_ok=True)
    return subs_dir


def _subtitle_path(project_id: str, subtitle_id: str, *, ensure_dir: bool = False) -> Path:
    if Path(subtitle_id).name != subtitle_id:
        raise SubtitleStorageError("非法字幕 ID")
    return _subtitles_dir(project_id, ensure_dir=ensure_dir) / f"{subtitle_id}.json"


def generate_subtitle_id() -> str:
    """生成短 UUID 作为字幕 ID。"""

    return uuid.uuid4().hex[:8]


def save_subtitle(project_id: str, subtitle: SubtitleFile) -> None:
    """将字幕对象保存为 JSON 文件。
    
    文件路径为 projects_root/<project_id>/subtitles/<subtitle_id>.json。
    """
    
    path = _subtitle_path(project_id, subtitle.id, ensure_dir=True)
    data = subtitle.model_dump(mode="json")

    try:
        _write_json_atomic(path, data)
    except Exception as exc:  # pragma: no cover - 简单封装
        raise SubtitleStorageError(f"保存字幕失败: {exc}") from exc


def load_subtitle(project_id: str, subtitle_id: str) -> SubtitleFile:
    """加载指定字幕 ID 对应的字幕文件。"""

    path = _subtitle_path(project_id, subtitle_id)
    if not path.is_file():
        raise SubtitleStorageError("字幕文件不存在")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # pragma: no cover - 简单封装
        raise SubtitleStorageError(f"读取字幕文件失败: {exc}") from exc

    try:
        return SubtitleFile(**raw)
    except ValidationError as exc:
        raise SubtitleStorageError("解析字幕文件失败") from exc


def list_subtitles(project_id: str) -> List[SubtitleFile]:
    """列出项目下的所有字幕文件。"""

    subs_dir = _subtitles_dir(project_id)
    if not subs_dir.is_dir():
        return []
    items: List[SubtitleFile] = []

    for path in subs_dir.glob("*.json"):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            sub = SubtitleFile(**raw)
        except Exception:
            # 单个字幕文件损坏不影响整体列表
            continue
        else:
            items.append(sub)

    # 按创建时间排序，旧的在前
    try:
        items.sort(key=lambda s: s.created_at)
    except Exception:
        pass

    return items


def delete_subtitle(project_id: str, subtitle_id: str) -> None:
    """删除指定字幕 JSON 文件，并尽量删除对应的 SRT 文件。
    
    - 若字幕文件不存在，则静默返回；
    - SRT 文件删除失败不会影响整体操作。
    """
    
    json_path = _subtitle_path(project_id, subtitle_id)

    if not json_path.is_file():
        return

    srt_path = json_path.with_suffix(".srt")

    try:
        json_path.unlink()
    except Exception as exc:  # pragma: no cover - 简单封装
        raise SubtitleStorageError(f"删除字幕文件失败: {exc}") from exc

    # 删除对应的 SRT 文件
    if srt_path.is_file():
        try:
            srt_path.unlink()
        except Exception:
            pass
