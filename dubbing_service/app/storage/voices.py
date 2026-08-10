"""File-based helpers for managing reusable voices (tone library)."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Generator, List

from pydantic import BaseModel, Field, ValidationError


class VoiceStorageError(Exception):
    """与音色库存储相关的异常。"""


# 全局音色库读写锁，防止并发 "读取-修改-写入" 竞态条件
_VOICES_LOCK = Lock()


@contextmanager
def voices_lock() -> Generator[None, None, None]:
    """音色库读写锁的上下文管理器。

    在需要 "读取-修改-写入" voices.json 的场景中使用：

        with voices_lock():
            voices = load_voices()
            # 修改 voices...
            save_voices(voices)
    """
    _VOICES_LOCK.acquire()
    try:
        yield
    finally:
        _VOICES_LOCK.release()


class Voice(BaseModel):
    """全局可复用的音色条目。"""

    id: str
    name: str
    tags: list[str] = Field(default_factory=list)
    # 保存后端可访问的音频文件路径（绝对路径）
    audio_path: str | None = None
    description: str | None = None
    created_at: datetime
    updated_at: datetime


def _voices_file_path() -> Path:
    """返回 voices.json 的路径。"""

    return Path(__file__).resolve().parents[2] / "voices.json"


def voices_audio_dir() -> Path:
    """返回音色库音频文件保存目录，并确保其存在。"""

    base = Path(__file__).resolve().parents[2] / "voices"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_voices() -> List[Voice]:
    path = _voices_file_path()
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # pragma: no cover - 简单封装
        raise VoiceStorageError(f"读取音色库失败: {exc}") from exc

    if not isinstance(raw, list):
        raise VoiceStorageError("voices.json 格式错误：应为列表")

    voices: List[Voice] = []
    for item in raw:
        if isinstance(item, dict):
            ap = item.get("audio_path")
            if isinstance(ap, str) and ap:
                p = Path(ap)
                if p.is_absolute():
                    item = dict(item)
                    item["audio_path"] = p.name
        try:
            v = Voice(**item)
        except ValidationError as exc:  # pragma: no cover - 配置错误简单抛出
            raise VoiceStorageError("解析音色库配置失败") from exc
        voices.append(v)

    return voices


def save_voices(voices: List[Voice]) -> None:
    path = _voices_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [v.model_dump(mode="json") for v in voices]
    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except Exception as exc:  # pragma: no cover - 简单封装
        raise VoiceStorageError(f"保存音色库失败: {exc}") from exc
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def generate_voice_id() -> str:
    return uuid.uuid4().hex[:8]
