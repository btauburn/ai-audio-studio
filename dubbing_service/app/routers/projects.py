"""Project management APIs for the dubbing service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List
import shutil
import subprocess
import sys
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.core.helpers import (
    check_open_location_enabled as _check_open_location_enabled,
    get_keep_count as _get_keep_count,
    get_keep_hours as _get_keep_hours,
)
from dubbing_service.app.core.tts_operations import (
    release_project_mutation,
    try_acquire_project_mutation,
)
from dubbing_service.app.models.project import Chapter, Project
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.routers.operation_guards import project_mutation_guard
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    create_project,
    project_meta_lock,
    list_projects,
    load_project,
    save_project,
    delete_project,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ProjectCreateRequest(BaseModel):
    """创建项目的请求体。"""

    name: str = Field(..., min_length=1, description="项目名称")
    description: str | None = Field(None, description="项目简介或备注")


class ChapterRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    index: int | None = Field(default=None, ge=0)


@router.post(
    "/{project_id}/chapters",
    response_model=Chapter,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_create_chapter(project_id: str, payload: ChapterRequest) -> Chapter:
    import uuid
    with project_meta_lock(project_id):
        project = load_project(project_id)
        chapter = Chapter(
            id=uuid.uuid4().hex[:8], title=payload.title.strip(),
            description=payload.description,
            index=payload.index if payload.index is not None else len(project.chapters),
        )
        project.chapters.append(chapter)
        project.chapters.sort(key=lambda item: item.index)
        for index, item in enumerate(project.chapters):
            item.index = index
        project.updated_at = datetime.now(timezone.utc)
        save_project(project)
    return chapter


@router.put(
    "/{project_id}/chapters/{chapter_id}",
    response_model=Chapter,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_chapter(project_id: str, chapter_id: str, payload: ChapterRequest) -> Chapter:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        chapter = next((item for item in project.chapters if item.id == chapter_id), None)
        if chapter is None:
            raise HTTPException(status_code=404, detail="章节/场景不存在")
        chapter.title = payload.title.strip()
        chapter.description = payload.description
        if payload.index is not None:
            chapter.index = payload.index
        project.chapters.sort(key=lambda item: item.index)
        for index, item in enumerate(project.chapters):
            item.index = index
        project.updated_at = datetime.now(timezone.utc)
        save_project(project)
    return chapter


@router.delete(
    "/{project_id}/chapters/{chapter_id}",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_chapter(project_id: str, chapter_id: str) -> dict[str, bool]:
    if any(item.chapter_id == chapter_id for item in load_segments(project_id)):
        raise HTTPException(status_code=409, detail="章节/场景仍包含分段，请先移动这些分段")
    with project_meta_lock(project_id):
        project = load_project(project_id)
        if not any(item.id == chapter_id for item in project.chapters):
            raise HTTPException(status_code=404, detail="章节/场景不存在")
        project.chapters = [item for item in project.chapters if item.id != chapter_id]
        for index, item in enumerate(sorted(project.chapters, key=lambda row: row.index)):
            item.index = index
        project.updated_at = datetime.now(timezone.utc)
        save_project(project)
    return {"success": True}


class CleanupTmpFilesResponse(BaseModel):
    deleted_files: int


def _should_delete_tmp_file(p: Path) -> bool:
    name = p.name
    if name.endswith(".tmp"):
        return True
    if ".tmp-" in name:
        return True
    return False


@router.post("", response_model=Project)
async def api_create_project(payload: ProjectCreateRequest) -> Project:
    """创建一个新的配音项目。

    将在 projects_root 目录下生成对应的项目目录和 project.json。
    """

    try:
        project = create_project(name=payload.name, description=payload.description)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return project


@router.get("", response_model=List[Project])
async def api_list_projects() -> List[Project]:
    """列出所有项目。"""

    return list_projects()


@router.put(
    "/{project_id}",
    response_model=Project,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_project(project_id: str, payload: ProjectCreateRequest) -> Project:
    """更新项目的名称和描述。"""

    with project_meta_lock(project_id):
        try:
            project = load_project(project_id)
        except ProjectStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        project.name = payload.name.strip()
        project.description = payload.description
        project.updated_at = datetime.now(timezone.utc)

        try:
            save_project(project)
        except ProjectStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return project


@router.delete("/{project_id}")
async def api_delete_project(project_id: str) -> dict[str, bool]:
    """删除指定项目及其所有关联文件。"""

    # Lazy import keeps the project router independent from synthesis routing
    # while making the active-state check atomic with batch registration.
    from dubbing_service.app.core.tts_operations import (
        try_register_project_deletion,
        unregister_project_deletion,
    )

    deletion_lease = try_register_project_deletion(project_id)
    if deletion_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "当前项目正在批量合成、编辑或导出，不能删除。"
                "请先等待当前任务完成或停止合成。"
            ),
        )
    try:
        try:
            with project_meta_lock(project_id):
                delete_project(project_id)
        except ProjectStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
    finally:
        unregister_project_deletion(deletion_lease)

    return {"success": True}


@router.post(
    "/{project_id}/cleanup-tmp-files",
    response_model=CleanupTmpFilesResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_cleanup_project_tmp_files(project_id: str) -> CleanupTmpFilesResponse:
    try:
        project_root = safe_project_root(project_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法项目 ID",
        ) from exc

    if not project_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定项目目录不存在",
        )

    keep_hours = _get_keep_hours("DUBBING_KEEP_PROJECT_TMP_HOURS", 24)
    now = time.time()
    ttl_seconds = float(keep_hours) * 3600.0

    deleted = 0
    delete_errors = 0
    for root, _, files in os.walk(str(project_root)):
        base = Path(root)
        for name in files:
            fp = base / name
            try:
                if not _should_delete_tmp_file(fp):
                    continue
                age = now - float(fp.stat().st_mtime)
                if age < ttl_seconds:
                    continue
                fp.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                delete_errors += 1
                continue

    if delete_errors:
        logger.debug(
            "清理项目 tmp 文件时有删除失败: project_id=%s failed=%d",
            project_id,
            delete_errors,
        )

    return CleanupTmpFilesResponse(deleted_files=deleted)


@router.post("/{project_id}/open-location")
async def api_open_project_location(project_id: str) -> dict[str, bool]:
    """在服务器操作系统中打开指定项目的根目录。"""

    _check_open_location_enabled()

    try:
        project_root = safe_project_root(project_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法项目 ID",
        ) from exc

    # 确保目录存在
    if not project_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定项目目录不存在",
        )

    try:
        if sys.platform.startswith("win"):
            # Windows 使用资源管理器打开目录
            subprocess.Popen(["explorer", str(project_root)])
        elif sys.platform == "darwin":
            # macOS 使用 Finder 打开目录
            subprocess.Popen(["open", str(project_root)])
        else:
            # Linux 下使用 xdg-open 打开目录
            subprocess.Popen(["xdg-open", str(project_root)])
    except Exception as exc:  # pragma: no cover - 与本地环境相关
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"打开项目目录失败: {exc}",
        ) from exc

    return {"success": True}


# ==================== 全局一键清理 ====================


class CleanupAllResponse(BaseModel):
    """全局清理响应。"""
    projects_cleaned: int
    total_deleted_files: int
    total_deleted_dirs: int
    errors: List[str] = Field(default_factory=list)


def _cleanup_single_project(project_root: Path) -> tuple[int, int]:
    """清理单个项目中的各类中间文件，返回 (deleted_files, deleted_dirs)。"""
    import re

    deleted_files = 0
    deleted_dirs = 0
    now = time.time()
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    # 1. 清理 audio/final/ 旧导出文件（保留最新 N 个）
    keep_finals = _get_keep_count("DUBBING_KEEP_FINAL_EXPORTS", 5)
    final_dir = project_root / "audio" / "final"
    if final_dir.is_dir() and keep_finals > 0:
        old_final_delete_errors = 0
        try:
            files = [p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            if len(files) > keep_finals:
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for old in files[keep_finals:]:
                    try:
                        old.unlink(missing_ok=True)
                        deleted_files += 1
                    except Exception:
                        old_final_delete_errors += 1
            if old_final_delete_errors:
                logger.debug(
                    "清理旧导出音频时有删除失败: dir=%s failed=%d",
                    final_dir,
                    old_final_delete_errors,
                )
        except Exception:
            logger.debug("清理旧导出音频失败: %s", final_dir, exc_info=True)

    # 2. 清理 audio/tmp_full_with_edits_* 遗留目录
    audio_dir = project_root / "audio"
    if audio_dir.is_dir():
        tmp_dir_delete_errors = 0
        try:
            for p in audio_dir.iterdir():
                if p.is_dir() and p.name.startswith("tmp_full_with_edits"):
                    try:
                        shutil.rmtree(p, ignore_errors=True)
                        deleted_dirs += 1
                    except Exception:
                        tmp_dir_delete_errors += 1
            if tmp_dir_delete_errors:
                logger.debug(
                    "清理临时导出目录时有删除失败: dir=%s failed=%d",
                    audio_dir,
                    tmp_dir_delete_errors,
                )
        except Exception:
            logger.debug("清理临时导出目录失败: %s", audio_dir, exc_info=True)

    # 3. 清理孤儿 dialogue_timeline
    timeline_dir = project_root / "dialogue_timeline"
    if timeline_dir.is_dir():
        orphan_timeline_delete_errors = 0
        try:
            existing_stems = set()
            if final_dir.is_dir():
                existing_stems = {p.stem for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
            for p in timeline_dir.iterdir():
                if not p.is_file() or p.suffix.lower() != ".json":
                    continue
                name = p.stem
                if name.startswith("dialogue_timeline_"):
                    audio_stem = name[len("dialogue_timeline_"):]
                else:
                    continue
                if audio_stem not in existing_stems:
                    try:
                        p.unlink(missing_ok=True)
                        deleted_files += 1
                    except Exception:
                        orphan_timeline_delete_errors += 1
            if orphan_timeline_delete_errors:
                logger.debug(
                    "清理孤儿 timeline 时有删除失败: dir=%s failed=%d",
                    timeline_dir,
                    orphan_timeline_delete_errors,
                )
        except Exception:
            logger.debug("清理孤儿 timeline 失败: %s", timeline_dir, exc_info=True)

    # 4. 清理 .preview. / .tmp. 临时音频文件（超过 1 小时）
    if audio_dir.is_dir():
        try:
            ttl = 3600.0
            for root, _, files in os.walk(str(audio_dir)):
                base = Path(root)
                for name in files:
                    if ".preview." not in name and ".tmp." not in name:
                        continue
                    fp = base / name
                    try:
                        if now - fp.stat().st_mtime < ttl:
                            continue
                        fp.unlink(missing_ok=True)
                        deleted_files += 1
                    except Exception:
                        continue
        except Exception:
            logger.debug("清理预览/临时音频失败: %s", audio_dir, exc_info=True)

    # 5. 清理全局 .tmp / .tmp-* 文件（超过 24 小时）
    try:
        ttl_tmp = 86400.0
        for root, _, files in os.walk(str(project_root)):
            base = Path(root)
            for name in files:
                fp = base / name
                if not (name.endswith(".tmp") or ".tmp-" in name):
                    continue
                try:
                    if now - fp.stat().st_mtime < ttl_tmp:
                        continue
                    fp.unlink(missing_ok=True)
                    deleted_files += 1
                except Exception:
                    continue
    except Exception:
        logger.debug("清理 .tmp 临时文件失败: %s", project_root, exc_info=True)

    # 6. 清理 logs/ 旧的 LLM 日志（保留最新 N 组）
    logs_dir = project_root / "logs"
    if logs_dir.is_dir():
        keep_logs = _get_keep_count("DUBBING_KEEP_SEG_LLM_LOGS", 3)
        try:
            groups: dict[str, list[Path]] = {}
            for p in logs_dir.iterdir():
                if not p.is_file():
                    continue
                m = re.match(r"^seg_llm_(raw|parsed)_(\d{8}_\d{6})\.(txt|json)$", p.name)
                if not m:
                    continue
                ts = m.group(2)
                groups.setdefault(ts, []).append(p)
            if groups:
                ts_list = sorted(groups.keys(), reverse=True)
                for ts in ts_list[keep_logs:]:
                    for p in groups.get(ts, []):
                        try:
                            p.unlink()
                            deleted_files += 1
                        except Exception:
                            continue
        except Exception:
            logger.debug("清理 LLM 日志归档失败: %s", logs_dir, exc_info=True)

    # 7. 清理 text/ 旧的 seg_prompt 归档（保留最新 N 个）
    text_dir = project_root / "text"
    if text_dir.is_dir():
        keep_text = _get_keep_count("DUBBING_KEEP_SEG_PROMPT_ARCHIVES", 3)
        try:
            prompt_files: list[tuple[str, Path]] = []
            for p in text_dir.iterdir():
                if (
                    p.is_file()
                    and p.suffix == ".txt"
                    and p.name != "seg_prompt_template.txt"
                    and p.name.startswith("seg_prompt_")
                ):
                    m = re.search(r"\d{8}_\d{6}", p.name)
                    key = m.group(0) if m else p.name
                    prompt_files.append((key, p))
            prompt_files.sort(key=lambda x: x[0], reverse=True)
            for _, path in prompt_files[keep_text:]:
                try:
                    path.unlink()
                    deleted_files += 1
                except Exception:
                    continue
        except Exception:
            logger.debug("清理分段提示词归档失败: %s", text_dir, exc_info=True)

    # 8. 清理旧字幕（保留最新 N 组）
    subs_dir = project_root / "subtitles"
    if subs_dir.is_dir():
        keep_subs = _get_keep_count("DUBBING_KEEP_SUBTITLES", 5)
        try:
            json_files = sorted(
                [p for p in subs_dir.glob("*.json") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_json in json_files[keep_subs:]:
                try:
                    old_srt = old_json.with_suffix(".srt")
                    old_json.unlink(missing_ok=True)
                    deleted_files += 1
                    if old_srt.is_file():
                        old_srt.unlink(missing_ok=True)
                        deleted_files += 1
                except Exception:
                    continue
        except Exception:
            logger.debug("清理旧字幕文件失败: %s", subs_dir, exc_info=True)

    # 9. 清理 audio/emotion/ 下的孤儿情感音频（分段已删除但音频残留）
    emotion_dir = project_root / "audio" / "emotion"
    segments_json = project_root / "segments.json"
    if emotion_dir.is_dir() and segments_json.is_file():
        orphan_emotion_delete_errors = 0
        try:
            import json as _json
            with segments_json.open("r", encoding="utf-8") as _f:
                seg_data = _json.load(_f)
            active_ids = set()
            if isinstance(seg_data, list):
                for item in seg_data:
                    if isinstance(item, dict) and item.get("id"):
                        active_ids.add(str(item["id"]))
            if active_ids:
                for p in emotion_dir.iterdir():
                    if not p.is_file():
                        continue
                    # emotion 文件命名格式: {segment_id}.wav / {segment_id}.webm
                    stem = p.stem
                    if stem not in active_ids:
                        try:
                            p.unlink(missing_ok=True)
                            deleted_files += 1
                        except Exception:
                            orphan_emotion_delete_errors += 1
            if orphan_emotion_delete_errors:
                logger.debug(
                    "清理孤儿情感音频时有删除失败: dir=%s failed=%d",
                    emotion_dir,
                    orphan_emotion_delete_errors,
                )
        except Exception:
            logger.debug("清理孤儿情感音频失败: %s", emotion_dir, exc_info=True)

    # 10. 清理 audio/segments/ 下的孤儿分段音频（segments.json 未引用）
    segments_audio_dir = project_root / "audio" / "segments"
    if segments_audio_dir.is_dir() and segments_json.is_file():
        orphan_segments_delete_errors = 0
        try:
            import json as _json
            with segments_json.open("r", encoding="utf-8") as _f:
                seg_data = _json.load(_f)

            active_audio_paths: set[str] = set()
            if isinstance(seg_data, list):
                for item in seg_data:
                    if not isinstance(item, dict):
                        continue
                    for key in ("audio_file", "alt_audio_file"):
                        rel = item.get(key)
                        if not isinstance(rel, str) or not rel:
                            continue
                        rel_norm = rel.replace("\\", "/")
                        if rel_norm.startswith("audio/segments/"):
                            active_audio_paths.add(rel_norm)
                    takes = item.get("takes")
                    if isinstance(takes, list):
                        for take in takes:
                            if not isinstance(take, dict):
                                continue
                            rel = take.get("audio_file")
                            if not isinstance(rel, str) or not rel:
                                continue
                            rel_norm = rel.replace("\\", "/")
                            if rel_norm.startswith("audio/segments/"):
                                active_audio_paths.add(rel_norm)

            for p in segments_audio_dir.iterdir():
                if not p.is_file():
                    continue
                try:
                    rel_norm = str(p.relative_to(project_root)).replace("\\", "/")
                except Exception:
                    continue
                if rel_norm in active_audio_paths:
                    continue
                try:
                    p.unlink(missing_ok=True)
                    deleted_files += 1
                except Exception:
                    orphan_segments_delete_errors += 1

            if orphan_segments_delete_errors:
                logger.debug(
                    "清理孤儿分段音频时有删除失败: dir=%s failed=%d",
                    segments_audio_dir,
                    orphan_segments_delete_errors,
                )
        except Exception:
            logger.debug("清理孤儿分段音频失败: %s", segments_audio_dir, exc_info=True)

    return deleted_files, deleted_dirs


@router.post("/cleanup-all", response_model=CleanupAllResponse)
async def api_cleanup_all_projects() -> CleanupAllResponse:
    """一键清理所有项目的中间文件和临时文件。

    清理范围包括：
    - audio/final/ 旧导出文件（保留最新 5 个，可通过 DUBBING_KEEP_FINAL_EXPORTS 配置）
    - audio/tmp_full_with_edits_* 遗留临时目录
    - dialogue_timeline/ 孤儿 timeline 文件
    - .preview. / .tmp. / .tmp-* 临时文件
    - logs/ 旧 LLM 调试日志（保留最新 3 组，可通过 DUBBING_KEEP_SEG_LLM_LOGS 配置）
    - text/ 旧 seg_prompt 归档（保留最新 3 个，可通过 DUBBING_KEEP_SEG_PROMPT_ARCHIVES 配置）
    - subtitles/ 旧字幕文件（保留最新 5 组，可通过 DUBBING_KEEP_SUBTITLES 配置）
    - audio/emotion/ 孤儿情感音频（对应分段已删除）
    - audio/segments/ 孤儿分段音频（segments.json 未引用）
    """
    cfg = get_config()
    projects_root = cfg.projects_root_abs

    if not projects_root.is_dir():
        return CleanupAllResponse(projects_cleaned=0, total_deleted_files=0, total_deleted_dirs=0)

    projects_cleaned = 0
    total_files = 0
    total_dirs = 0
    errors: list[str] = []

    try:
        entries = [p for p in projects_root.iterdir() if p.is_dir()]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取项目目录失败: {exc}",
        ) from exc

    project_dirs = [
        project_dir
        for project_dir in entries
        if (project_dir / "project.json").is_file()
    ]
    cleanup_leases = []
    try:
        # Reserve every target before deleting anything. If one project is
        # active, fail without partially cleaning the remaining projects.
        for project_dir in project_dirs:
            lease = try_acquire_project_mutation(
                project_dir.name,
                kind="cleanup_all_projects",
            )
            if lease is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"项目 {project_dir.name} 正在合成、导出或删除，"
                        "本次全局清理未执行。"
                    ),
                )
            cleanup_leases.append(lease)

        for project_dir in project_dirs:
            try:
                df, dd = _cleanup_single_project(project_dir)
                total_files += df
                total_dirs += dd
                if df > 0 or dd > 0:
                    projects_cleaned += 1
            except Exception as exc:
                errors.append(f"{project_dir.name}: {exc}")
    finally:
        for lease in reversed(cleanup_leases):
            release_project_mutation(lease)

    return CleanupAllResponse(
        projects_cleaned=projects_cleaned,
        total_deleted_files=total_files,
        total_deleted_dirs=total_dirs,
        errors=errors[:10],
    )
