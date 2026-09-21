"""Subtitle generation and management APIs for the dubbing service."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from dubbing_service.app.core.helpers import (
    env_truthy as _env_truthy,
    get_keep_count as _get_keep_count,
    project_root,
    sha256_file as _sha256_file,
)
from dubbing_service.app.core.tts_operations import (
    release_tts_operation,
    try_acquire_tts_operation,
)
from dubbing_service.app.models.subtitles import (
    GenerateSubtitleRequest,
    GenerateSubtitleResponse,
    SubtitleFile,
    SubtitleSummary,
)
from dubbing_service.app.providers.subtitles import (
    DEFAULT_WHISPER_MODEL_ID,
    FastWhisperSubtitleProvider,
    SubtitleModelUnavailableError,
    apply_timeline_items_to_subtitle,
    apply_source_text_to_subtitle_by_alignment,
    subtitle_file_to_srt,
)
from dubbing_service.app.storage.segments import SegmentStorageError, load_segments
from dubbing_service.app.storage.subtitles import (
    SubtitleStorageError,
    generate_subtitle_id,
    list_subtitles,
    load_subtitle,
    save_subtitle,
    delete_subtitle,
    write_text_atomic,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _prune_old_subtitles(project_id: str, keep: int = 5) -> int:
    """保留最新的 keep 组字幕文件，删除更早的 JSON+SRT。"""
    try:
        subs = list_subtitles(project_id)
    except Exception:
        return 0

    if len(subs) <= keep:
        return 0

    # list_subtitles 已按 created_at 升序排列，最旧的在前
    to_delete = subs[:len(subs) - keep]
    deleted = 0
    for sub in to_delete:
        try:
            delete_subtitle(project_id, sub.id)
            deleted += 1
        except Exception:
            continue
    return deleted

def _project_root(project_id: str) -> Path:
    root = project_root(project_id, ensure_dir=False)
    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定项目不存在",
        )
    return root

def _final_audio_dir(project_id: str) -> Path:
    return _project_root(project_id) / "audio" / "final"

def _validate_final_audio_filename(audio_filename: str | None) -> None:
    """Reject path-like input before any model or GPU work is attempted."""

    if audio_filename and Path(audio_filename).name != audio_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

def _validate_subtitle_model_language(
    model_size: str,
    language: str | None,
) -> None:
    """Keep the English-only distilled model out of multilingual flows."""

    if model_size == "distil-large-v3" and (language or "").lower() != "en":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="distil-large-v3 仅支持明确指定为英文 (en) 的音频",
        )

def _pick_final_audio_path(project_id: str, audio_filename: str | None) -> Path:
    """根据给定文件名或最新导出记录选出一个整书音频文件。"""

    final_dir = _final_audio_dir(project_id)

    if not final_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前项目还没有任何整书导出音频，请先完成导出。",
        )

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    if audio_filename:
        _validate_final_audio_filename(audio_filename)
        path = final_dir / audio_filename
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的整书导出音频不存在",
            )
        if path.suffix.lower() not in exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定文件不是受支持的音频格式",
            )
        return path

    files = [p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前项目还没有任何整书导出音频，请先完成导出。",
        )

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _ensure_subtitle_paths(project_id: str, subtitle: SubtitleFile) -> SubtitleFile:
    """确保 json_rel_path / srt_rel_path 字段有合理的默认值。"""
    subtitle.json_rel_path = (Path("subtitles") / f"{subtitle.id}.json").as_posix()
    subtitle.srt_rel_path = (Path("subtitles") / f"{subtitle.id}.srt").as_posix()

    # 在此处并不强制写回磁盘，只在保存/更新时同步写回
    return subtitle

@router.post(
    "/projects/{project_id}/subtitles/generate-from-final",
    response_model=GenerateSubtitleResponse,
)
def api_generate_subtitle_from_final(
    project_id: str,
    payload: GenerateSubtitleRequest,
) -> GenerateSubtitleResponse:
    """Run blocking ASR in FastAPI's worker pool under the global GPU lease."""

    # Input validation must take precedence over local-model readiness so that
    # a missing model never masks a traversal attempt.
    _validate_final_audio_filename(payload.audio_filename)

    model_size = payload.model_size or DEFAULT_WHISPER_MODEL_ID
    _validate_subtitle_model_language(model_size, payload.language)
    try:
        FastWhisperSubtitleProvider.resolve_model_path(model_size)
    except SubtitleModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    operation_lease = try_acquire_tts_operation(
        kind="subtitle_generation",
        project_id=project_id,
    )
    if operation_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="TTS/字幕 GPU 任务正在执行，请等待当前任务完成后重试。",
        )

    try:
        try:
            from dubbing_service.app.routers.settings import (
                unload_managed_tts_model_for_gpu_handoff,
            )

            unload_managed_tts_model_for_gpu_handoff()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"字幕生成前释放 TTS 显存失败: {exc}",
            ) from exc
        return _generate_subtitle_from_final_exclusive(project_id, payload)
    finally:
        release_tts_operation(operation_lease)


def _generate_subtitle_from_final_exclusive(
    project_id: str,
    payload: GenerateSubtitleRequest,
) -> GenerateSubtitleResponse:
    """Generate and persist subtitles while the caller owns the GPU lease."""

    try:
        audio_path = _pick_final_audio_path(project_id, payload.audio_filename)
    except HTTPException:
        # 直接向上抛出 HTTP 错误
        raise

    subtitle_id = generate_subtitle_id()
    audio_filename = audio_path.name
    audio_rel_path = str(Path("audio") / "final" / audio_filename)
    project_root = _project_root(project_id)

    model_size = payload.model_size or DEFAULT_WHISPER_MODEL_ID
    try:
        provider = FastWhisperSubtitleProvider(model_size=model_size)
    except SubtitleModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        subtitle = provider.generate_subtitles(
            subtitle_id=subtitle_id,
            project_id=project_id,
            audio_path=audio_path,
            audio_rel_path=audio_rel_path,
            audio_filename=audio_filename,
            language=payload.language,
            task=payload.task,
        )
    except Exception as exc:  # pragma: no cover - 依赖运行环境
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成字幕失败: {exc}",
        ) from exc

    branch_log_enabled = _env_truthy("DUBBING_SUBTITLE_GENERATE_FROM_FINAL_LOG_BRANCH")
    branch = "asr_only"
    branch_detail = ""

    if payload.text_source == "segments" and payload.task != "translate":
        timeline_items: list[object] | None = None
        timeline_source = ""
        timeline_rel = (
            Path("dialogue_timeline") / f"dialogue_timeline_{Path(audio_filename).stem}.json"
        )
        timeline_path = project_root / timeline_rel

        if timeline_path.is_file():
            try:
                with timeline_path.open("r", encoding="utf-8") as f:
                    timeline_raw = json.load(f)
                if isinstance(timeline_raw, dict):
                    items = timeline_raw.get("items")
                    if isinstance(items, list):
                        timeline_items = items
                        timeline_source = timeline_rel.as_posix()
            except Exception:
                timeline_items = None

        if not timeline_items:
            latest_timeline_path = project_root / "dialogue_timeline.json"
            if latest_timeline_path.is_file():
                try:
                    with latest_timeline_path.open("r", encoding="utf-8") as f:
                        timeline_raw2 = json.load(f)
                    if isinstance(timeline_raw2, dict) and timeline_raw2.get(
                        "audio_filename"
                    ) == audio_filename:
                        items2 = timeline_raw2.get("items")
                        if isinstance(items2, list):
                            timeline_items = items2
                            timeline_source = "dialogue_timeline.json"
                except Exception:
                    timeline_items = None

        if timeline_items:
            branch = "timeline"
            branch_detail = timeline_source
            subtitle = apply_timeline_items_to_subtitle(subtitle, timeline_items)
        else:
            try:
                segments = load_segments(project_id)
            except SegmentStorageError:
                segments = []

            if segments:
                playable_texts: list[str] = []
                for seg in segments:
                    if not getattr(seg, "audio_file", None):
                        continue
                    try:
                        project_root_resolved = project_root.resolve()
                        audio_file_path = (project_root / str(seg.audio_file)).resolve()
                    except Exception:
                        continue
                    if not audio_file_path.is_relative_to(project_root_resolved):
                        continue
                    if not audio_file_path.is_file():
                        continue
                    text = (getattr(seg, "text", "") or "").strip()
                    if text:
                        playable_texts.append(text)

                source_text = "\n".join(playable_texts)
                if source_text.strip():
                    branch = "segments_alignment"
                    branch_detail = f"segments={len(segments)}"
                    subtitle, _ = apply_source_text_to_subtitle_by_alignment(
                        subtitle,
                        source_text,
                    )
                else:
                    branch = "segments_no_source_text"
                    branch_detail = f"segments={len(segments)}"
            else:
                branch = "segments_no_segments"
                branch_detail = ""
    else:
        branch = "asr_only"
        branch_detail = ""

    if branch_log_enabled:
        logger.info(
            "generate-from-final branch=%s detail=%s project_id=%s subtitle_id=%s audio_filename=%s text_source=%s task=%s",
            branch,
            branch_detail,
            project_id,
            subtitle.id,
            audio_filename,
            payload.text_source,
            payload.task,
        )

    subtitle = _ensure_subtitle_paths(project_id, subtitle)

    audio_sha256: str | None
    try:
        audio_sha256 = _sha256_file(audio_path)
    except Exception:
        audio_sha256 = None
    subtitle.audio_sha256 = audio_sha256

    try:
        save_subtitle(project_id, subtitle)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # 同步生成并保存 SRT 文件，便于后续下载
    srt_rel = subtitle.srt_rel_path or f"subtitles/{subtitle.id}.srt"
    try:
        project_root_resolved = project_root.resolve()
        srt_path = (project_root / srt_rel).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )
    if not srt_path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )

    srt_text = subtitle_file_to_srt(subtitle)
    try:
        write_text_atomic(srt_path, srt_text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - 简单封装
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存 SRT 文件失败: {exc}",
        ) from exc

    # 自动清理旧字幕文件，防止无限积累
    try:
        keep = _get_keep_count("DUBBING_KEEP_SUBTITLES", 5)
        _prune_old_subtitles(project_id, keep=keep)
    except Exception:
        pass

    return GenerateSubtitleResponse(
        success=True,
        subtitle_id=subtitle.id,
        source_audio_filename=audio_filename,
        source_audio_rel_path=audio_rel_path,
        audio_sha256=audio_sha256,
        json_rel_path=subtitle.json_rel_path,
        srt_rel_path=subtitle.srt_rel_path,
        message="字幕生成完成",
    )


@router.get(
    "/projects/{project_id}/subtitles",
    response_model=List[SubtitleSummary],
)
async def api_list_subtitles(project_id: str) -> List[SubtitleSummary]:
    """列出项目下已生成的字幕文件。"""

    _project_root(project_id)

    try:
        items = list_subtitles(project_id)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    summaries: List[SubtitleSummary] = []
    for sub in items:
        sub = _ensure_subtitle_paths(project_id, sub)
        summaries.append(
            SubtitleSummary(
                id=sub.id,
                audio_rel_path=sub.audio_rel_path,
                audio_filename=sub.audio_filename,
                created_at=sub.created_at,
                updated_at=sub.updated_at,
                language=sub.language,
                duration=sub.duration,
                model_name=sub.model_name,
                json_rel_path=sub.json_rel_path or "",
                srt_rel_path=sub.srt_rel_path or "",
            )
        )

    # 按创建时间倒序返回，最新的在前面
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


@router.get(
    "/projects/{project_id}/subtitles/{subtitle_id}",
    response_model=SubtitleFile,
)
async def api_get_subtitle(project_id: str, subtitle_id: str) -> SubtitleFile:
    """获取指定字幕的完整 JSON 结构。"""

    try:
        subtitle = load_subtitle(project_id, subtitle_id)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return _ensure_subtitle_paths(project_id, subtitle)


@router.put(
    "/projects/{project_id}/subtitles/{subtitle_id}",
    response_model=SubtitleFile,
)
async def api_update_subtitle(
    project_id: str,
    subtitle_id: str,
    payload: SubtitleFile,
) -> SubtitleFile:
    """更新指定字幕 JSON 内容，并重新生成对应 SRT 文件。

    该接口为后续前端字幕编辑预留，目前前端尚未直接调用。
    """

    try:
        existing = load_subtitle(project_id, subtitle_id)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    subtitle = payload
    subtitle.id = subtitle_id
    subtitle.project_id = project_id

    # 锁定音频与落盘路径相关字段，避免前端 JSON 编辑误改导致不可用或路径穿越
    subtitle.audio_rel_path = existing.audio_rel_path
    subtitle.audio_filename = existing.audio_filename
    subtitle.audio_sha256 = existing.audio_sha256
    subtitle.created_at = existing.created_at

    subtitle = _ensure_subtitle_paths(project_id, subtitle)
    subtitle.updated_at = datetime.now(timezone.utc)

    try:
        save_subtitle(project_id, subtitle)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # 重新生成 SRT
    project_root = _project_root(project_id)
    srt_rel = subtitle.srt_rel_path or f"subtitles/{subtitle.id}.srt"
    try:
        project_root_resolved = project_root.resolve()
        srt_path = (project_root / srt_rel).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )
    if not srt_path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )

    srt_text = subtitle_file_to_srt(subtitle)
    try:
        write_text_atomic(srt_path, srt_text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存 SRT 文件失败: {exc}",
        ) from exc

    return subtitle


@router.delete(
    "/projects/{project_id}/subtitles/{subtitle_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def api_delete_subtitle(project_id: str, subtitle_id: str) -> None:
    """删除指定字幕 JSON 及其对应的 SRT 文件。"""

    try:
        delete_subtitle(project_id, subtitle_id)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/projects/{project_id}/subtitles/{subtitle_id}/download")
async def api_download_subtitle(
    project_id: str,
    subtitle_id: str,
    format: str = Query("srt", pattern="^(srt|json)$"),
) -> FileResponse:
    """下载指定字幕文件，支持 SRT 或 JSON 格式。"""

    try:
        subtitle = load_subtitle(project_id, subtitle_id)
    except SubtitleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    subtitle = _ensure_subtitle_paths(project_id, subtitle)

    project_root = _project_root(project_id)

    if format == "json":
        rel = subtitle.json_rel_path or f"subtitles/{subtitle.id}.json"
        try:
            project_root_resolved = project_root.resolve()
            path = (project_root / rel).resolve()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="字幕文件路径非法",
            )
        if not path.is_relative_to(project_root_resolved):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="字幕文件路径非法",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # 如果磁盘上缺失，则重新保存一次 JSON
        if not path.is_file():
            try:
                save_subtitle(project_id, subtitle)
            except SubtitleStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="字幕 JSON 文件不存在或无法创建",
            )
        return FileResponse(path, media_type="application/json", filename=path.name)

    # 默认返回 SRT
    rel = subtitle.srt_rel_path or f"subtitles/{subtitle.id}.srt"
    try:
        project_root_resolved = project_root.resolve()
        path = (project_root / rel).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )
    if not path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="字幕文件路径非法",
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.is_file():
        srt_text = subtitle_file_to_srt(subtitle)
        try:
            write_text_atomic(path, srt_text, encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"保存 SRT 文件失败: {exc}",
            ) from exc

    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="字幕 SRT 文件不存在或无法创建",
        )

    # 常见的 SRT MIME 类型为 application/x-subrip
    return FileResponse(path, media_type="application/x-subrip", filename=path.name)
