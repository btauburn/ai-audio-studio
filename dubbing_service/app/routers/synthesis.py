"""Batch TTS synthesis and export APIs for the dubbing service."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import ffmpeg  # type: ignore
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from dubbing_service.app.core.helpers import (
    check_open_location_enabled as _check_open_location_enabled,
    get_keep_count as _get_keep_count,
    get_keep_hours as _get_keep_hours,
    project_root,
    resolve_role_reference_audio,
    sha256_file as _sha256_file,
)
from dubbing_service.app.core.path_safety import (
    SEGMENT_AUDIO_EXTENSIONS,
    resolve_scoped_project_asset,
    validate_simple_component_id,
)
from dubbing_service.app.core.tts_operations import (
    count_actively_synthesizing_segments,
    has_running_batch_for_project,
    set_segment_actively_synthesizing,
    try_register_running_batch,
    unregister_running_batch,
)
from dubbing_service.app.models.project import (
    Project,
    Role,
    Segment,
    SegmentEditRegion,
    SegmentEditSilence,
    SegmentEditState,
    SegmentTake,
)
from dubbing_service.app.models.enums import SegmentStatus
from dubbing_service.app.providers.tts import IndexTTSTTSProvider, CosyVoice3TTSProvider, Qwen3TTSProvider, Qwen3TTSCustomVoiceProvider, TTSResult
from dubbing_service.app.routers.operation_guards import (
    project_export_guard,
    project_mutation_guard,
    standalone_segment_synthesis_guard,
)
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.segments import (
    SegmentStorageError,
    load_segments,
    order_segments,
    save_segments,
    project_segments_lock,
)
from dubbing_service.app.storage.subtitles import delete_subtitle, list_subtitles
from dubbing_service.app.storage.jobs import (
    JobStorageError,
    get_job,
    interrupt_job,
    mark_job_active,
    record_job_segment,
    start_job,
)
from dubbing_service.app.storage.masters import list_masters
from dubbing_service.app.services.takes import (
    capture_active_legacy_settings,
    clear_slot as clear_take_slot,
    create_take,
    ensure_take_state,
    get_selected_take,
    get_slot_take,
    get_take,
    normalize_slot,
    prune_archived_takes,
    restore_take_to_slot,
    restore_active_edit_revision,
    select_take,
    set_active_slot,
    sync_legacy_projection,
    update_active_edit_state,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_SUPPORTED_TTS_BACKENDS = {
    "indextts",
    "cosyvoice3",
    "qwen3tts",
    "qwen3tts_cv",
}

_KEEP_AUDIO = object()


class CleanupAudioTempResponse(BaseModel):
    deleted_dirs: int
    deleted_files: int


class RestoreEditRevisionRequest(BaseModel):
    revision_index: int = Field(..., ge=0)


def _project_root(project_id: str) -> Path:
    root = project_root(project_id, ensure_dir=False)
    if not (root / "project.json").is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定项目不存在",
        )
    return root


def _cleanup_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug("清理临时文件失败: %s", path, exc_info=True)


def _unlink_project_relative_file(project_root: Path, rel_path: str | None) -> None:
    """Safely unlink a generated file under ``audio/segments``.

    Persisted paths are not trusted merely because they resolve somewhere in
    the project.  In particular, a malformed snapshot must not turn this
    helper into a project metadata or role-prompt deletion primitive.
    """

    if not rel_path:
        return

    try:
        target = resolve_scoped_project_asset(
            project_root,
            rel_path,
            required_subdir="audio/segments",
            allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
        )
    except Exception:
        return

    try:
        if target.is_file():
            target.unlink()
    except Exception:
        logger.debug("清理项目内文件失败: %s", target, exc_info=True)


def _audio_mime_from_suffix(suffix: str) -> str:
    ext = (suffix or "").lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


def _selected_take_projection(segment: Segment) -> Segment:
    """Return an export-only Segment view backed by the selected Take."""

    take = get_selected_take(segment)
    if take is None:
        return segment
    projected = segment.model_copy(deep=True)
    projected.audio_file = take.audio_file
    projected.playback_speed = take.baked_speed
    projected.volume_gain = take.baked_volume
    projected.edit_state = (
        take.edit_state.model_copy(deep=True) if take.edit_state is not None else None
    )
    return projected


def _strict_casting_binding(project: Project, segments: list[Segment]) -> dict | None:
    from dubbing_service.app.audiobook.casting_assembly import (
        CastingAssemblyError,
        current_frozen_binding,
    )
    try:
        return current_frozen_binding(project, segments)
    except CastingAssemblyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _revalidate_strict_casting(project: Project, segments: list[Segment], binding: dict | None) -> None:
    if binding is None:
        return
    from dubbing_service.app.audiobook.casting_assembly import (
        CastingAssemblyError,
        revalidate_binding,
    )
    try:
        revalidate_binding(project, segments, binding)
    except CastingAssemblyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _prune_audio_temp(project_root: Path, *, keep_hours: int) -> CleanupAudioTempResponse:
    audio_dir = project_root / "audio"
    if not audio_dir.is_dir():
        return CleanupAudioTempResponse(deleted_dirs=0, deleted_files=0)

    try:
        now = datetime.now(timezone.utc).timestamp()
    except Exception:
        logger.debug("获取当前时间戳失败，跳过 TTL 判断", exc_info=True)
        now = None

    ttl_seconds = None
    if now is not None:
        ttl_seconds = float(keep_hours) * 3600.0

    deleted_dirs = 0
    deleted_files = 0

    try:
        for p in audio_dir.iterdir():
            if p.is_dir() and p.name.startswith("tmp_full_with_edits"):
                try:
                    if ttl_seconds is not None:
                        age = now - float(p.stat().st_mtime)
                        if age < ttl_seconds:
                            continue
                    shutil.rmtree(p, ignore_errors=True)
                    deleted_dirs += 1
                except Exception:
                    continue
    except Exception:
        logger.debug("遍历临时目录失败: %s", audio_dir, exc_info=True)

    try:
        for root, _, files in os.walk(str(audio_dir)):
            base = Path(root)
            for name in files:
                if ".preview." not in name and ".tmp." not in name:
                    continue
                fp = base / name
                try:
                    if ttl_seconds is not None:
                        age = now - float(fp.stat().st_mtime)
                        if age < ttl_seconds:
                            continue
                    fp.unlink(missing_ok=True)
                    deleted_files += 1
                except Exception:
                    continue
    except Exception:
        logger.debug("遍历临时文件失败: %s", audio_dir, exc_info=True)

    return CleanupAudioTempResponse(deleted_dirs=deleted_dirs, deleted_files=deleted_files)


def _auto_prune_after_export(project_root: Path) -> None:
    """导出完成后自动清理项目中积累的中间文件。

    清理范围：
    1. audio/final/ 下的旧导出文件（保留最新 N 个，默认 5）
    2. audio/tmp_full_with_edits_* 历史遗留目录（立即清理）
    3. dialogue_timeline/ 下与已删除导出音频不再匹配的孤儿 timeline
    4. audio/ 下的 .preview. / .tmp. 临时文件（超过 1 小时的）
    """
    keep_finals = _get_keep_count("DUBBING_KEEP_FINAL_EXPORTS", 5)

    # --- 1. 清理旧的导出音频 ---
    final_dir = project_root / "audio" / "final"
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if final_dir.is_dir() and keep_finals > 0:
        old_final_delete_errors = 0
        try:
            files = [p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            if len(files) > keep_finals:
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for old in files[keep_finals:]:
                    try:
                        old.unlink(missing_ok=True)
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

    # --- 2. 清理 tmp_full_with_edits_* 遗留目录 ---
    audio_dir = project_root / "audio"
    if audio_dir.is_dir():
        tmp_dir_delete_errors = 0
        try:
            for p in audio_dir.iterdir():
                if p.is_dir() and p.name.startswith("tmp_full_with_edits"):
                    try:
                        shutil.rmtree(p, ignore_errors=True)
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

    # --- 3. 清理孤儿 dialogue_timeline ---
    timeline_dir = project_root / "dialogue_timeline"
    if timeline_dir.is_dir() and final_dir.is_dir():
        orphan_timeline_delete_errors = 0
        try:
            existing_stems = set()
            if final_dir.is_dir():
                existing_stems = {p.stem for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
            for p in timeline_dir.iterdir():
                if not p.is_file() or p.suffix.lower() != ".json":
                    continue
                # dialogue_timeline_full_20250101_120000.json -> full_20250101_120000
                name = p.stem
                if name.startswith("dialogue_timeline_"):
                    audio_stem = name[len("dialogue_timeline_"):]
                else:
                    continue
                if audio_stem not in existing_stems:
                    try:
                        p.unlink(missing_ok=True)
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

    # --- 4. 清理 .preview. / .tmp. 临时音频文件（超过 1 小时） ---
    if audio_dir.is_dir():
        try:
            now = datetime.now(timezone.utc).timestamp()
            ttl = 3600.0  # 1 小时
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
                    except Exception:
                        continue
        except Exception:
            logger.debug("清理预览/临时音频失败: %s", audio_dir, exc_info=True)


@router.post(
    "/projects/{project_id}/audio/cleanup-temp",
    response_model=CleanupAudioTempResponse,
    dependencies=[Depends(project_mutation_guard)],
)
def api_cleanup_audio_temp(project_id: str) -> CleanupAudioTempResponse:
    project_root = _project_root(project_id)
    keep_hours = _get_keep_hours("DUBBING_KEEP_AUDIO_TEMP_HOURS", 24)
    return _prune_audio_temp(project_root, keep_hours=keep_hours)


def _probe_audio_duration_seconds(path: Path) -> float:
    try:
        probe = ffmpeg.probe(str(path))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio" and "duration" in stream:
                try:
                    dur = float(stream["duration"])
                except (TypeError, ValueError):
                    dur = 0.0
                if dur > 0:
                    return dur
            if stream.get("codec_type") != "audio":
                continue
            duration_ts = stream.get("duration_ts")
            time_base = stream.get("time_base")
            if duration_ts is not None and time_base:
                try:
                    ts = float(duration_ts)
                    if "/" in str(time_base):
                        num, den = str(time_base).split("/", 1)
                        base = float(num) / float(den)
                    else:
                        base = float(time_base)
                    dur = ts * base
                except (TypeError, ValueError, ZeroDivisionError):
                    dur = 0.0
                if dur > 0:
                    return dur
            nb_samples = stream.get("nb_samples")
            sample_rate = stream.get("sample_rate")
            if nb_samples is not None and sample_rate is not None:
                try:
                    dur = float(nb_samples) / float(sample_rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    dur = 0.0
                if dur > 0:
                    return dur
        fmt = probe.get("format") or {}
        if "duration" in fmt:
            try:
                dur = float(fmt["duration"])
            except (TypeError, ValueError):
                dur = 0.0
            if dur > 0:
                return dur
    except Exception:
        logger.debug("探测音频时长失败: %s", path, exc_info=True)
    return 0.0


def _probe_audio_format(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        probe = ffmpeg.probe(str(path))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "audio":
                continue
            sample_rate: Optional[int] = None
            channels: Optional[int] = None
            if "sample_rate" in stream:
                try:
                    sample_rate = int(stream["sample_rate"])
                except (TypeError, ValueError):  # pragma: no cover - 容错
                    sample_rate = None
            if "channels" in stream:
                try:
                    channels = int(stream["channels"])
                except (TypeError, ValueError):  # pragma: no cover - 容错
                    channels = None
            return sample_rate, channels
    except Exception:
        logger.debug("探测音频格式失败: %s", path, exc_info=True)
    return None, None


def _normalize_audio_stream(audio_stream, sample_rate: Optional[int], channels: Optional[int]):
    if sample_rate is None and channels is None:
        return audio_stream
    params: dict[str, object] = {}
    if sample_rate is not None and sample_rate > 0:
        params["sample_rates"] = sample_rate
    if channels == 1:
        params["channel_layouts"] = "mono"
    elif channels == 2:
        params["channel_layouts"] = "stereo"
    if not params:
        return audio_stream
    return audio_stream.filter_("aformat", **params)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f"{path.stem}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class SegmentSynthesisResult(BaseModel):
    """单个段落合成结果。"""

    segment_id: str
    success: bool
    audio_file: Optional[str] = None
    take_id: Optional[str] = None
    slot: Optional[str] = None
    error: Optional[str] = None
    # 非致命 warning，例如情感文本解析存在异常但已降级处理
    warning: Optional[str] = None


class BatchSynthesisRequest(BaseModel):
    """批量合成请求体。"""

    segment_ids: Optional[List[str]] = Field(
        None,
        description="要合成的段落 ID 列表；为空则默认合成所有 pending/failed 的段落。",
    )
    retry_failed: bool = Field(
        True,
        description="当 segment_ids 为空时，是否同时重试 status=failed 的段落。",
    )
    job_id: Optional[str] = Field(
        default=None,
        description="可选的后端持久任务 ID",
    )


class BatchSynthesisResponse(BaseModel):
    results: List[SegmentSynthesisResult]


class BatchSynthesisPlanResponse(BaseModel):
    segment_ids: List[str]


class ExportFullResponse(BaseModel):
    success: bool
    output_file: str
    dialogue_timeline_file: Optional[str] = None
    audio_sha256: Optional[str] = None


class FinalAudioInfo(BaseModel):
    """项目下已导出的整书音频信息。"""

    filename: str
    rel_path: str
    created_at: datetime
    duration_seconds: Optional[float] = None


class SegmentAudioRegion(BaseModel):
    """单个分段音频中要删除的时间区间（秒）。"""

    start: float = Field(..., ge=0.0, description="区间起始时间（秒）")
    end: float = Field(..., ge=0.0, description="区间结束时间（秒），必须大于起始时间")


class SegmentSilenceInterval(BaseModel):
    """单个静音插入区间（秒）。"""

    at: float = Field(..., ge=0.0, description="插入静音的起始时间（秒）")
    duration: float = Field(..., ge=0.0, description="静音时长（秒）")


class SegmentAudioEditRequest(BaseModel):
    """单个分段音频编辑请求体：变速 / 调整音量 / 删除区间。"""

    speed: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="目标播放速度倍数，0.5-2.0，1.0 为原速；未提供则使用当前值",
    )
    volume: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="目标音量倍数，0-2.0，1.0 为原始音量；未提供则使用当前值",
    )
    delete_region: Optional[SegmentAudioRegion] = Field(
        default=None,
        description="要删除的时间区间（秒）；未提供则不裁剪，仅应用变速和音量",
    )
    delete_regions: Optional[List[SegmentAudioRegion]] = Field(
        default=None,
        description=(
            "要删除的时间区间列表（秒）；与 delete_region 同时提供时，优先使用该列表。"
        ),
    )
    insert_silence_at: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="在该时间点（秒）插入静音；未提供则不插入静音",
    )
    silence_duration: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="要插入的静音时长（秒）；未提供或为 0 时不生效",
    )
    silence_intervals: Optional[List[SegmentSilenceInterval]] = Field(
        default=None,
        description=(
            "要插入的静音区间列表；与 insert_silence_at/silence_duration 同时提供时，"
            "将优先使用该列表。"
        ),
    )
    fade_in: float = Field(default=0.0, ge=0.0, le=10.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=10.0)
    leading_silence: float = Field(default=0.0, ge=0.0, le=30.0)
    trailing_silence: float = Field(default=0.0, ge=0.0, le=30.0)
    processing_preset: str = Field(
        default="none", pattern="^(none|natural_dialogue|audio_story|clear_podcast)$",
    )


def _normalize_silence_intervals(
    payload: SegmentAudioEditRequest,
) -> list[tuple[float, float]]:
    """从请求体中抽取规范化后的静音插入区间列表。

    - 优先使用 silence_intervals 字段；
    - 当未提供 silence_intervals 时，退回到单一 insert_silence_at/silence_duration；
    - 自动过滤非法或时长为 0 的区间，并按时间从小到大排序。
    """

    intervals: list[tuple[float, float]] = []

    if payload.silence_intervals:
        for item in payload.silence_intervals:
            try:
                at = float(item.at)
                dur = float(item.duration)
            except (TypeError, ValueError):  # pragma: no cover - 容错
                continue
            if at < 0.0 or dur <= 0.0:
                continue
            intervals.append((at, dur))

    # 向后兼容：当未提供 silence_intervals 时，退回到单一 insert_silence_at/silence_duration
    if (
        not intervals
        and payload.insert_silence_at is not None
        and payload.silence_duration is not None
    ):
        try:
            at = float(payload.insert_silence_at)
            dur = float(payload.silence_duration)
        except (TypeError, ValueError):  # pragma: no cover - 容错
            at = -1.0
            dur = 0.0
        if at >= 0.0 and dur > 0.0:
            intervals.append((at, dur))

    intervals.sort(key=lambda x: x[0])
    return intervals


def _normalize_delete_regions(
    payload: SegmentAudioEditRequest,
    duration: Optional[float],
) -> list[tuple[float, float]]:
    """从请求体中抽取规范化后的删除区间列表。

    - 优先使用 delete_regions 字段；
    - 当未提供 delete_regions 时，退回到单一 delete_region 字段；
    - 自动过滤 / 修正非法时间，并按时间从小到大排序；
    - 如果整体删除覆盖了整段音频，则抛出 400 错误。
    """

    regions: list[tuple[float, float]] = []

    if payload.delete_regions:
        for item in payload.delete_regions:
            try:
                start = float(item.start)
                end = float(item.end)
            except (TypeError, ValueError):  # pragma: no cover - 容错
                continue
            start = max(0.0, start)
            end = max(0.0, end)
            if end <= start:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="删除区间的结束时间必须大于起始时间。",
                )
            if duration is not None and start >= duration:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="删除区间起始时间超出音频长度。",
                )
            if duration is not None and end > duration:
                end = duration
            regions.append((start, end))

    if not regions and payload.delete_region is not None:
        # 向后兼容：退回到单一区间字段
        try:
            start = float(payload.delete_region.start)
            end = float(payload.delete_region.end)
        except (TypeError, ValueError):  # pragma: no cover - 容错
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="删除区间参数非法。",
            )
        start = max(0.0, start)
        end = max(0.0, end)
        if end <= start:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="删除区间的结束时间必须大于起始时间。",
            )
        if duration is not None and start >= duration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="删除区间起始时间超出音频长度。",
            )
        if duration is not None and end > duration:
            end = duration
        regions.append((start, end))

    if not regions:
        return []

    # 按起始时间排序并合并重叠区间
    regions.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for start, end in regions:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

    if duration is not None:
        total_deleted = sum(e - s for s, e in merged)
        if total_deleted >= duration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不允许删除整段音频，请保留至少一小段内容。",
            )

    return merged


def _clamp_audio_edit_speed(value: object, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.5, min(2.0, numeric))


def _clamp_audio_edit_volume(value: object, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(2.0, numeric))


def _resolve_audio_edit_targets(
    *,
    target_speed: float | None,
    target_volume: float | None,
    base_speed: float | None,
    base_volume: float | None,
) -> tuple[float, float, float, float]:
    resolved_base_speed = _clamp_audio_edit_speed(base_speed, default=1.0)
    resolved_base_volume = _clamp_audio_edit_volume(base_volume, default=1.0)
    desired_speed = _clamp_audio_edit_speed(
        resolved_base_speed if target_speed is None else target_speed,
        default=resolved_base_speed,
    )
    desired_volume = _clamp_audio_edit_volume(
        resolved_base_volume if target_volume is None else target_volume,
        default=resolved_base_volume,
    )
    speed_ratio = desired_speed / resolved_base_speed if resolved_base_speed > 1e-6 else desired_speed
    volume_ratio = desired_volume / resolved_base_volume if resolved_base_volume > 1e-6 else desired_volume
    return desired_speed, desired_volume, speed_ratio, volume_ratio


def _build_segment_audio_edit_request_from_state(
    state: SegmentEditState | None,
) -> SegmentAudioEditRequest:
    delete_region_payload: SegmentAudioRegion | None = None
    delete_regions_payload: list[SegmentAudioRegion] | None = None
    if state and state.regions:
        items: list[SegmentAudioRegion] = []
        for r in state.regions:
            try:
                start = float(r.start)
                end = float(r.end)
            except (TypeError, ValueError):
                continue
            items.append(SegmentAudioRegion(start=start, end=end))
        if items:
            delete_regions_payload = items
    elif state and state.region:
        try:
            delete_region_payload = SegmentAudioRegion(
                start=float(state.region.start),
                end=float(state.region.end),
            )
        except (TypeError, ValueError):
            delete_region_payload = None

    silence_intervals_payload: list[SegmentSilenceInterval] | None = None
    if state and state.silences:
        items2: list[SegmentSilenceInterval] = []
        for s in state.silences:
            try:
                at = float(s.at)
                dur = float(s.duration)
            except (TypeError, ValueError):
                continue
            items2.append(SegmentSilenceInterval(at=at, duration=dur))
        if items2:
            silence_intervals_payload = items2

    return SegmentAudioEditRequest(
        speed=state.speed if state and state.speed is not None else None,
        volume=state.volume if state and state.volume is not None else None,
        delete_region=delete_region_payload,
        delete_regions=delete_regions_payload,
        silence_intervals=silence_intervals_payload,
        fade_in=state.fade_in if state else 0.0,
        fade_out=state.fade_out if state else 0.0,
        leading_silence=state.leading_silence if state else 0.0,
        trailing_silence=state.trailing_silence if state else 0.0,
        processing_preset=state.processing_preset if state else "none",
    )


def _apply_delete_regions(
    audio_stream,
    regions: list[tuple[float, float]],
):
    """在音频流中按顺序删除多个区间，并将前后拼接。

    所有区间的 start/end 基于原始时间轴；函数内部会自动根据之前删除的
    时长对后续区间进行偏移校正。
    """

    if not regions:
        return audio_stream

    removed = 0.0
    for start, end in regions:
        # 将区间映射到当前已裁剪后的时间轴
        length = float(end) - float(start)
        if length <= 0.0:
            continue
        real_start = max(0.0, float(start) - removed)
        real_end = max(real_start, float(end) - removed)
        if real_end <= real_start:
            continue

        split = audio_stream.filter_multi_output("asplit", 2)
        pre_src = split[0]
        post_src = split[1]

        pre = pre_src.filter_("atrim", start=0, end=real_start).filter_(
            "asetpts",
            "PTS-STARTPTS",
        )
        post = post_src.filter_("atrim", start=real_end).filter_(
            "asetpts",
            "PTS-STARTPTS",
        )

        audio_stream = ffmpeg.concat(pre, post, v=0, a=1)
        removed += length

    return audio_stream


def _apply_silence_intervals(
    audio_stream,
    intervals: list[tuple[float, float]],
    sample_rate: Optional[int],
    channels: Optional[int],
):
    """在音频流中按顺序插入多段静音。

    insert 坐标基于原始时间轴：对于第 i 段静音，实际插入位置为

        at_i + sum_{j < i}(duration_j)

    这样可以保证多段静音的插入位置与前端看到的原始时间轴一致。
    """

    if not intervals:
        return audio_stream

    anullsrc_expr = "anullsrc"
    params: list[str] = []
    if channels == 1:
        params.append("channel_layout=mono")
    elif channels == 2:
        params.append("channel_layout=stereo")
    if sample_rate is not None and sample_rate > 0:
        params.append(f"sample_rate={sample_rate}")
    if params:
        anullsrc_expr = "anullsrc=" + ":".join(params)

    inserted = 0.0
    for at, dur in intervals:
        if dur <= 0.0:
            continue

        insert_pos = max(0.0, float(at) + inserted)

        # 将当前流在插入点前后切成两段。这里先 asplit 一次，避免同一上游节点
        # 被多个下游滤镜同时消费而触发 ffmpeg-python 的 multiple outgoing edges 错误。
        split = audio_stream.filter_multi_output("asplit", 2)
        pre_src = split[0]
        post_src = split[1]

        pre = pre_src.filter_("atrim", start=0, end=insert_pos).filter_(
            "asetpts",
            "PTS-STARTPTS",
        )
        post = post_src.filter_("atrim", start=insert_pos).filter_(
            "asetpts",
            "PTS-STARTPTS",
        )

        silence = ffmpeg.input(anullsrc_expr, f="lavfi", t=float(dur)).audio
        audio_stream = ffmpeg.concat(pre, silence, post, v=0, a=1)
        inserted += float(dur)

    return audio_stream


def _get_role(project: Project, role_id: Optional[str]) -> Optional[Role]:
    if role_id is None:
        return None
    for r in project.roles:
        if r.id == role_id:
            return r
    return None


def _resolve_tts_backend(backend: object, default_backend: str) -> str:
    """将分段后端规范化为受支持的后端 ID。"""

    value = str(backend or "").strip()
    if value in _SUPPORTED_TTS_BACKENDS:
        return value

    default_value = str(default_backend or "").strip()
    if default_value in _SUPPORTED_TTS_BACKENDS:
        return default_value

    return "indextts"


def _role_has_usable_speaker_reference(project_id: str, role: Role | None) -> bool:
    if role is None:
        return False

    return (
        resolve_role_reference_audio(
            project_id,
            getattr(role, "speaker_audio", None),
            getattr(role, "speaker_voice_id", None),
        )
        is not None
    )


def _collect_missing_speaker_roles(
    project: Project,
    targets: List[Segment],
    default_backend: str,
) -> list[str]:
    missing: set[str] = set()
    for seg in targets:
        role = _get_role(project, seg.role_id)
        backend = _resolve_tts_backend(
            getattr(seg, "tts_backend", None) or getattr(role, "tts_backend", None),
            default_backend,
        )
        if backend in ("qwen3tts", "qwen3tts_cv"):
            continue
        if role and not _role_has_usable_speaker_reference(project.id, role):
            missing.add(role.name or role.id)
    return sorted(missing)


def _render_audio_edit_to_path(
    source_path: Path,
    output_path: Path,
    payload: SegmentAudioEditRequest,
    *,
    base_speed: float = 1.0,
    base_volume: float = 1.0,
) -> None:
    """Render one EditRecipe to a new file without touching its source."""

    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    try:
        probe = ffmpeg.probe(str(source_path))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "audio":
                continue
            if "duration" in stream:
                duration = float(stream["duration"])
            if "sample_rate" in stream:
                sample_rate = int(stream["sample_rate"])
            if "channels" in stream:
                channels = int(stream["channels"])
            break
    except Exception:
        duration = None

    _desired_speed, _desired_volume, speed_ratio, volume_ratio = _resolve_audio_edit_targets(
        target_speed=payload.speed,
        target_volume=payload.volume,
        base_speed=base_speed,
        base_volume=base_volume,
    )
    audio_stream = ffmpeg.input(str(source_path)).audio
    delete_regions = _normalize_delete_regions(payload, duration)
    if delete_regions:
        audio_stream = _apply_delete_regions(audio_stream, delete_regions)
    intervals = _normalize_silence_intervals(payload)
    if intervals:
        audio_stream = _apply_silence_intervals(
            audio_stream,
            intervals,
            sample_rate=sample_rate,
            channels=channels,
        )
    if abs(speed_ratio - 1.0) > 1e-3:
        audio_stream = audio_stream.filter_("atempo", speed_ratio)
    if abs(volume_ratio - 1.0) > 1e-3:
        audio_stream = audio_stream.filter_("volume", volume_ratio)

    if payload.processing_preset == "natural_dialogue":
        audio_stream = audio_stream.filter("highpass", f=65).filter(
            "acompressor", threshold=0.16, ratio=1.8, attack=20, release=220,
        )
    elif payload.processing_preset == "audio_story":
        audio_stream = audio_stream.filter("highpass", f=55).filter(
            "acompressor", threshold=0.14, ratio=2.2, attack=18, release=260,
        )
    elif payload.processing_preset == "clear_podcast":
        audio_stream = audio_stream.filter("highpass", f=80).filter("lowpass", f=15500).filter(
            "acompressor", threshold=0.12, ratio=3.0, attack=12, release=180,
        )
    if payload.leading_silence > 0:
        delay_ms = round(payload.leading_silence * 1000)
        channel_count = channels if channels in {1, 2} else 2
        audio_stream = audio_stream.filter("adelay", "|".join([str(delay_ms)] * channel_count))
    if payload.trailing_silence > 0:
        audio_stream = audio_stream.filter("apad", pad_dur=payload.trailing_silence)
    if duration is not None:
        removed = sum(end - start for start, end in delete_regions)
        inserted = sum(seconds for _, seconds in intervals)
        rendered_duration = max(0.0, duration - removed + inserted) / max(speed_ratio, 0.01)
        if payload.fade_in > 0:
            audio_stream = audio_stream.filter(
                "afade", type="in", start_time=payload.leading_silence,
                duration=min(payload.fade_in, rendered_duration),
            )
        if payload.fade_out > 0:
            audio_stream = audio_stream.filter(
                "afade", type="out",
                start_time=payload.leading_silence + max(rendered_duration - payload.fade_out, 0.0),
                duration=min(payload.fade_out, rendered_duration),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.output(audio_stream, str(output_path))
    ffmpeg.run(stream, overwrite_output=True)


def _copy_audio_to_project(
    project_id: str,
    segment: Segment,
    audio_path: str,
    name_suffix: str = "",
    *,
    take_id: str | None = None,
) -> str:
    """将 IndexTTS 生成的音频复制到项目的 audio/segments 目录下。

    新合成必须传 ``take_id``，从而为每次结果创建不可变文件。未传时仅
    保留给旧测试和兼容调用，沿用历史 ``name_suffix`` 命名。
    返回相对路径（相对于项目根目录）。
    """

    segment_id = validate_simple_component_id(segment.id, label="segment_id")
    if name_suffix not in {"", "_alt"}:
        raise ValueError("invalid generated-audio name suffix")
    if take_id is not None:
        take_id = validate_simple_component_id(take_id, label="take_id")

    project_root = _project_root(project_id)
    segments_dir = project_root / "audio" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    src = Path(audio_path)
    repo_root = Path(__file__).resolve().parents[3]
    allowed_roots = [
        (repo_root / "outputs").resolve(),
        (repo_root / "service" / "outputs").resolve(),
    ]

    if not src.is_absolute():
        # IndexTTS2 REST 服务默认将文件保存在 outputs 或 service/outputs 下。
        candidate = (repo_root / src).resolve()
        candidate_service = (repo_root / "service" / src).resolve()

        resolved_src: Path | None = None
        for cand in (candidate, candidate_service):
            if not cand.is_file():
                continue
            if any(cand.is_relative_to(root) for root in allowed_roots):
                resolved_src = cand
                break

        if resolved_src is None:
            raise FileNotFoundError(
                f"找不到或不允许访问 TTS 生成的音频文件: {candidate} 或 {candidate_service}"
            )
        src = resolved_src
    else:
        src = src.resolve()
        if not src.is_file():
            raise FileNotFoundError(f"找不到 TTS 生成的音频文件: {src}")
        if not any(src.is_relative_to(root) for root in allowed_roots):
            raise FileNotFoundError(f"TTS 生成的音频路径不在允许目录中: {src}")

    ext = (src.suffix or ".wav").lower()
    if ext not in SEGMENT_AUDIO_EXTENSIONS:
        raise ValueError(f"unsupported generated audio extension: {ext}")
    target_name = (
        f"{segment_id}_{take_id}{ext}"
        if take_id is not None
        else f"{segment_id}{name_suffix}{ext}"
    )
    rel_path = Path("audio") / "segments" / target_name
    dst = resolve_scoped_project_asset(
        project_root,
        str(rel_path),
        required_subdir="audio/segments",
        allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
    )

    # Take 文件不可变。只有旧兼容调用仍清理同一 legacy stem 的其他扩展名。
    if take_id is None:
        glob_pattern = f"{segment_id}{name_suffix}.*"
        try:
            for old in segments_dir.glob(glob_pattern):
                try:
                    if old.is_file() and old.name != target_name:
                        old.unlink()
                except Exception:
                    continue
        except Exception:
            logger.debug("清理旧扩展名分段音频失败: pattern=%s", glob_pattern, exc_info=True)

    shutil.copy2(src, dst)

    try:
        if any(src.is_relative_to(root) for root in allowed_roots):
            try:
                src.unlink(missing_ok=True)
            except Exception:
                logger.debug("清理来源输出音频失败: %s", src, exc_info=True)
    except Exception:
        logger.debug("尝试清理来源输出音频时发生异常", exc_info=True)

    return str(rel_path)


def _synthesize_one(
    project: Project,
    project_id: str,
    seg: Segment,
    providers: dict,
    default_backend: str,
    previous_text: Optional[str],
) -> dict:
    """合成单个段落，返回结果 dict（不修改 seg 对象的状态字段）。

    返回格式：
      成功: {"success": True, "audio_file": "...", "warning": "..."|None}
      失败: {"success": False, "error": "..."}
    """

    ensure_take_state(seg)
    target_slot = normalize_slot(seg.active_version)
    role = _get_role(project, seg.role_id)
    if role is None:
        return {"success": False, "error": "段落未绑定有效角色，无法合成。"}

    if not getattr(seg, "text", None) or not str(seg.text).strip():
        return {"success": False, "error": "段落文本为空，无法合成。"}

    seg_backend = _resolve_tts_backend(
        getattr(seg, "tts_backend", None) or getattr(role, "tts_backend", None),
        default_backend,
    )

    if seg_backend not in ("qwen3tts", "qwen3tts_cv"):
        if not _role_has_usable_speaker_reference(project_id, role):
            return {"success": False, "error": "角色未配置可用的参考音频，无法合成。"}

    provider = providers.get(seg_backend, providers.get("indextts"))

    from dubbing_service.app.services.production import apply_pronunciation_dictionary
    effective_segment = seg.model_copy(deep=True)
    effective_segment.text = apply_pronunciation_dictionary(project_id, seg.text)
    effective_previous_text = apply_pronunciation_dictionary(project_id, previous_text)
    try:
        tts_result: TTSResult = provider.synthesize_segment(
            project_id, role, effective_segment, previous_text=effective_previous_text,
        )
    except Exception as exc:
        return {"success": False, "error": f"TTS 调用异常: {exc}"}

    if not tts_result.success or not tts_result.audio_path:
        error_msg: Optional[str] = None
        if isinstance(tts_result.raw, dict):
            error_msg = tts_result.raw.get("error") or tts_result.raw.get("detail")
        if not error_msg:
            error_msg = "TTS 未返回音频路径" if not tts_result.audio_path else "TTS 调用失败"
        return {"success": False, "error": error_msg}

    take_id = uuid.uuid4().hex[:16]
    try:
        rel_path = _copy_audio_to_project(
            project_id,
            seg,
            tts_result.audio_path,
            take_id=take_id,
        )
    except Exception as exc:
        return {"success": False, "error": f"复制音频失败: {exc}"}

    try:
        audio_sha256 = _sha256_file(_project_root(project_id) / rel_path)
    except Exception as exc:
        _unlink_project_relative_file(_project_root(project_id), rel_path)
        return {"success": False, "error": f"校验生成音频失败: {exc}"}

    warning_msg: Optional[str] = None
    if isinstance(tts_result.raw, dict):
        raw_warning = tts_result.raw.get("emotion_warning")
        if isinstance(raw_warning, str) and raw_warning:
            warning_msg = raw_warning

    raw_metadata: dict[str, object] = {}
    if isinstance(tts_result.raw, dict):
        for key, value in tts_result.raw.items():
            normalized_key = str(key).lower()
            if any(secret in normalized_key for secret in ("api_key", "token", "authorization", "base64")):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                text_value = str(value) if isinstance(value, str) else value
                raw_metadata[str(key)] = text_value[:2000] if isinstance(text_value, str) else text_value
    return {
        "success": True,
        "audio_file": rel_path,
        "audio_sha256": audio_sha256,
        "take_id": take_id,
        "slot": target_slot,
        "synthesis_text": effective_segment.text,
        "generation_metadata": raw_metadata,
        "warning": warning_msg,
    }


def _save_batch_segment_state(
    project_id: str,
    segment_id: str,
    segment_status: SegmentStatus,
    *,
    audio_file: object = _KEEP_AUDIO,
    take_id: str | None = None,
    slot: str | None = None,
    audio_sha256: str | None = None,
    synthesis_text: str | None = None,
    generation_metadata: dict | None = None,
    when: datetime | None = None,
) -> None:
    """Commit one batch segment state without overwriting concurrent edits."""

    ts = when or datetime.now(timezone.utc)
    with project_segments_lock(project_id):
        current_segments = load_segments(project_id)
        target = next((item for item in current_segments if item.id == segment_id), None)
        if target is None:
            raise SegmentStorageError(f"待持久化的段落不存在: {segment_id}")
        target.status = segment_status
        if audio_file is not _KEEP_AUDIO:
            if audio_file is None:
                clear_take_slot(target, slot or target.active_version)
            elif take_id:
                create_take(
                    target,
                    slot=slot or target.active_version,
                    take_id=take_id,
                    audio_file=str(audio_file),
                    audio_sha256=audio_sha256,
                    synthesis_text=synthesis_text,
                    generation_metadata=generation_metadata,
                )
            else:
                # Compatibility for external callers not yet supplying Take data.
                target.audio_file = str(audio_file)
        target.updated_at = ts
        save_segments(project_id, current_segments)


def _batch_durability_failure(stage: str, exc: Exception) -> dict[str, object]:
    return {
        "success": False,
        "audio_file": None,
        "error": f"合成状态持久化失败（{stage}）: {type(exc).__name__}",
        "warning": None,
    }


def _select_segments_for_synthesis(
    project_id: str,
    payload: BatchSynthesisRequest,
) -> tuple[Project, List[Segment], List[Segment]]:
    """根据请求体选择需要执行合成的段落列表。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    segments = load_segments(project_id)
    if not segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前项目还没有任何段落，请先进行 LLM 分段。",
        )

    # 选择需要合成的段落
    if payload.segment_ids:
        target_ids = set(payload.segment_ids)
        targets = [s for s in segments if s.id in target_ids]
    else:
        project_root = _project_root(project_id)

        def _needs_synthesis(seg: Segment) -> bool:
            # 待处理段落
            if seg.status == SegmentStatus.PENDING:
                return True

            # 生成中段落：若超过超时时间则允许重试，避免一直卡住
            if seg.status == SegmentStatus.GENERATING:
                timeout_hours = _get_keep_hours(
                    "DUBBING_GENERATING_TIMEOUT_HOURS",
                    default=2,
                )
                if timeout_hours <= 0:
                    return False
                try:
                    age = (datetime.now(timezone.utc) - seg.updated_at).total_seconds()
                except Exception:
                    return True
                return age >= timeout_hours * 3600

            # 失败段落，按需重试
            if seg.status == SegmentStatus.FAILED and payload.retry_failed:
                return True

            # 已完成但音频文件丢失的段落，允许重新合成
            if seg.status == SegmentStatus.DONE and seg.audio_file:
                try:
                    project_root_resolved = project_root.resolve()
                    audio_path = (project_root / str(seg.audio_file)).resolve()
                except Exception:
                    return True
                if not audio_path.is_relative_to(project_root_resolved):
                    return True
                if not audio_path.is_file():
                    return True

            return False

        targets = [s for s in segments if _needs_synthesis(s)]

    return project, segments, targets


@router.post(
    "/projects/{project_id}/segments/synthesize/plan",
    response_model=BatchSynthesisPlanResponse,
)
async def api_plan_batch_synthesize_segments(
    project_id: str,
    payload: BatchSynthesisRequest,
) -> BatchSynthesisPlanResponse:
    """仅计算本次批量合成会处理哪些段落，不实际调用 TTS。"""

    if has_running_batch_for_project(project_id) or count_actively_synthesizing_segments(project_id) > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="当前项目仍有后台批量合成正在执行，请稍候刷新状态。",
        )

    project, _, targets = _select_segments_for_synthesis(project_id, payload)
    from dubbing_service.app.routers.settings import _load_raw_file_config
    default_backend = _resolve_tts_backend(None, _load_raw_file_config().get("tts_backend", "indextts"))
    missing_roles = _collect_missing_speaker_roles(project, targets, default_backend)
    if missing_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"以下角色缺少参考音频: {', '.join(missing_roles)}",
        )
    return BatchSynthesisPlanResponse(segment_ids=[s.id for s in targets])


@router.post("/projects/{project_id}/segments/synthesize", response_model=BatchSynthesisResponse)
def api_batch_synthesize_segments(
    project_id: str,
    payload: BatchSynthesisRequest,
) -> BatchSynthesisResponse:
    """批量为项目下的段落执行语音合成。

    - 可以指定 segment_ids 精确控制；
    - 如果不指定，则默认对所有 PENDING（以及可选 FAILED）以及 DONE 但音频文件缺失的段落进行合成；
    - 成功后会更新 segments.json 中对应段落的 status 和 audio_file。
    """

    batch_lease = try_register_running_batch(project_id)
    if batch_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="当前已有后台批量合成或 TTS 管理操作正在执行，请稍候重试。",
        )

    persistent_job_active = False
    completed_normally = False
    try:
        project, segments, targets = _select_segments_for_synthesis(project_id, payload)

        if payload.job_id:
            job = get_job(project_id, payload.job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="持久合成任务不存在")
            if any(segment.id not in job.segment_ids for segment in targets):
                raise HTTPException(status_code=400, detail="合成分段不属于持久任务")
            try:
                mark_job_active(project_id, payload.job_id, True)
                persistent_job_active = True
                start_job(project_id, payload.job_id)
            except JobStorageError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        if not targets:
            completed_normally = True
            return BatchSynthesisResponse(results=[])

        from dubbing_service.app.routers.settings import _load_raw_file_config

        default_backend = _resolve_tts_backend(
            None,
            _load_raw_file_config().get("tts_backend", "indextts"),
        )
        missing_roles = _collect_missing_speaker_roles(project, targets, default_backend)
        if missing_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"以下角色缺少参考音频: {', '.join(missing_roles)}",
            )

        providers = {
            "indextts": IndexTTSTTSProvider(),
            "cosyvoice3": CosyVoice3TTSProvider(),
            "qwen3tts": Qwen3TTSProvider(),
            "qwen3tts_cv": Qwen3TTSCustomVoiceProvider(),
        }
        results: List[SegmentSynthesisResult] = []

        # 上下文注入：构建分段索引映射，用于查找前一分段的文本。load_segments 已按章节顺序 + 章节内行号返回。
        sorted_segments = segments
        seg_index_map = {s.id: i for i, s in enumerate(sorted_segments)}

        for seg in targets:
            try:
                _save_batch_segment_state(
                    project_id,
                    seg.id,
                    SegmentStatus.GENERATING,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to mark REST batch segment generating: project=%s seg=%s",
                    project_id,
                    seg.id,
                    exc_info=True,
                )
                result = _batch_durability_failure("GENERATING", exc)
                results.append(
                    SegmentSynthesisResult(
                        segment_id=seg.id,
                        success=False,
                        error=str(result["error"]),
                    )
                )
                if payload.job_id:
                    record_job_segment(
                        project_id,
                        payload.job_id,
                        seg.id,
                        success=False,
                        error=str(result["error"]),
                    )
                continue

            previous_text: Optional[str] = None
            current_idx = seg_index_map.get(seg.id)
            if current_idx is not None and current_idx > 0:
                previous_text = sorted_segments[current_idx - 1].text

            set_segment_actively_synthesizing(project_id, seg.id, True)
            try:
                result = _synthesize_one(
                    project,
                    project_id,
                    seg,
                    providers,
                    default_backend,
                    previous_text,
                )
            finally:
                set_segment_actively_synthesizing(project_id, seg.id, False)

            result = dict(result)
            now = datetime.now(timezone.utc)
            persisted_status = (
                SegmentStatus.DONE if result.get("success") else SegmentStatus.FAILED
            )
            try:
                if persisted_status == SegmentStatus.DONE:
                    _save_batch_segment_state(
                        project_id,
                        seg.id,
                        persisted_status,
                        audio_file=result.get("audio_file"),
                        take_id=result.get("take_id"),
                        slot=result.get("slot"),
                        audio_sha256=result.get("audio_sha256"),
                        synthesis_text=result.get("synthesis_text"),
                        generation_metadata=result.get("generation_metadata"),
                        when=now,
                    )
                else:
                    _save_batch_segment_state(
                        project_id,
                        seg.id,
                        persisted_status,
                        when=now,
                    )
            except Exception as exc:
                generated_path = result.get("audio_file")
                logger.warning(
                    "Failed to commit REST batch segment result: project=%s seg=%s status=%s",
                    project_id,
                    seg.id,
                    persisted_status,
                    exc_info=True,
                )
                result = _batch_durability_failure(
                    persisted_status.value.upper(),
                    exc,
                )
                if generated_path:
                    _unlink_project_relative_file(_project_root(project_id), str(generated_path))

            results.append(
                SegmentSynthesisResult(
                    segment_id=seg.id,
                    success=bool(result.get("success")),
                    audio_file=result.get("audio_file"),
                    take_id=result.get("take_id"),
                    slot=result.get("slot"),
                    error=result.get("error"),
                    warning=result.get("warning"),
                )
            )
            if payload.job_id:
                try:
                    record_job_segment(
                        project_id,
                        payload.job_id,
                        seg.id,
                        success=bool(result.get("success")),
                        error=str(result.get("error") or "") or None,
                    )
                except JobStorageError:
                    logger.warning(
                        "Failed to persist REST synthesis job progress: project=%s job=%s seg=%s",
                        project_id,
                        payload.job_id,
                        seg.id,
                        exc_info=True,
                    )

        try:
            with project_meta_lock(project_id):
                latest_project = load_project(project_id)
                latest_project.updated_at = datetime.now(timezone.utc)
                save_project(latest_project)
        except Exception:
            logger.warning(
                "Failed to save project timestamp after REST batch: project=%s",
                project_id,
                exc_info=True,
            )

        completed_normally = True
        return BatchSynthesisResponse(results=results)
    finally:
        if payload.job_id and persistent_job_active:
            if not completed_normally:
                try:
                    interrupt_job(project_id, payload.job_id, "REST 合成任务异常中断")
                except JobStorageError:
                    logger.debug("标记 REST 持久任务中断失败", exc_info=True)
            mark_job_active(project_id, payload.job_id, False)
        unregister_running_batch(batch_lease)


def _ordered_segments(project_id: str, segments: list[Segment]) -> list[Segment]:
    """Return ``segments`` in canonical reading order for export.

    ``Segment.index`` counts lines inside its own chapter, so the order must be
    driven by the chapter order first; sorting by ``index`` alone interleaves
    the chapters of a multi-chapter project.
    """

    try:
        chapters = load_project(project_id).chapters
    except ProjectStorageError:
        return segments
    return order_segments(segments, chapters)


@router.post(
    "/projects/{project_id}/export/full",
    response_model=ExportFullResponse,
    dependencies=[Depends(project_export_guard)],
)
def api_export_full_project(project_id: str) -> ExportFullResponse:
    """将项目中所有已生成的段落音频按顺序拼接为整书音频。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    segments = load_segments(project_id)
    if not segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前项目没有任何段落，无法导出。",
        )

    ordered_segments = _ordered_segments(project_id, segments)
    casting_binding = _strict_casting_binding(project, ordered_segments)

    project_root = _project_root(project_id)
    playable: list[tuple[Segment, Path]] = []
    missing_segments: list[Segment] = []
    project_root_resolved = project_root.resolve()
    for seg in ordered_segments:
        export_seg = _selected_take_projection(seg)
        if not export_seg.audio_file:
            missing_segments.append(seg)
            continue
        try:
            path = (project_root / str(export_seg.audio_file)).resolve()
        except Exception:
            missing_segments.append(seg)
            continue
        if not path.is_relative_to(project_root_resolved):
            missing_segments.append(seg)
            continue
        if path.is_file():
            playable.append((export_seg, path))
        else:
            missing_segments.append(seg)

    if missing_segments:
        sample = ", ".join(str(seg.index) for seg in missing_segments[:5])
        hint = f"（索引示例: {sample}）" if sample else ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"存在 {len(missing_segments)} 个段落缺少音频，无法导出{hint}。"
                "请先完成合成或刷新状态。"
            ),
        )
    if not playable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有可导出的音频，请先完成批量合成。",
        )

    audio_files = [p for _, p in playable]

    final_dir = project_root / "audio" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # 使用本地时区时间生成文件名时间戳，便于与系统时间直观对应
    now_local = datetime.now(timezone.utc).astimezone()
    output_ext = ".wav"
    if len(audio_files) == 1:
        output_ext = audio_files[0].suffix.lower() or ".wav"
    output_name = f"full_{now_local.strftime('%Y%m%d_%H%M%S')}{output_ext}"
    output_path = final_dir / output_name

    # 使用 ffmpeg-python 拼接音频（所有段落采样率一致，适合直接 concat）。
    # 当仅有一个音频文件时，若扩展名一致直接复制；否则使用 ffmpeg 转码到目标格式。
    try:
        if len(audio_files) == 1:
            if audio_files[0].suffix.lower() == output_ext:
                shutil.copyfile(audio_files[0], output_path)
            else:
                stream = ffmpeg.input(str(audio_files[0])).output(str(output_path))
                ffmpeg.run(stream, overwrite_output=True)
        else:
            target_sample_rate, target_channels = _probe_audio_format(audio_files[0])
            audio_streams = [
                _normalize_audio_stream(
                    ffmpeg.input(str(p)).audio,
                    target_sample_rate,
                    target_channels,
                )
                for p in audio_files
            ]
            stream = ffmpeg.concat(*audio_streams, v=0, a=1).output(str(output_path))
            ffmpeg.run(stream, overwrite_output=True)
    except FileNotFoundError as exc:
        # Windows 下如果系统未安装 ffmpeg 或不在 PATH 中，会触发此错误
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="音频合并失败：系统未找到 ffmpeg 可执行文件，请安装 ffmpeg 并确保其在系统 PATH 中。",
        ) from exc
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"音频合并失败: {exc}",
        ) from exc

    rel_output = Path("audio") / "final" / output_name

    timeline_items: list[dict[str, object]] = []
    cursor = 0.0
    invalid_duration_segments: list[Segment] = []
    try:
        from dubbing_service.app.storage.production import load_settings as _load_production_settings
        dialogue_crossfade = _load_production_settings(project_id).sound_design.mix.dialogue_crossfade_seconds
    except Exception:
        dialogue_crossfade = 0.0
    for seg, path in playable:
        dur = _probe_audio_duration_seconds(path)
        if not (dur > 0):
            invalid_duration_segments.append(seg)
            continue
        start = cursor
        end = cursor + dur
        timeline_items.append(
            {
                "segment_id": seg.id,
                "index": seg.index,
                "role_id": getattr(seg, "role_id", None),
                "text": (getattr(seg, "text", "") or "").strip(),
                "start": start,
                "end": end,
            }
        )
        cursor = end

    if invalid_duration_segments:
        sample = ", ".join(str(seg.index) for seg in invalid_duration_segments[:5])
        hint = f"（索引示例: {sample}）" if sample else ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"存在 {len(invalid_duration_segments)} 个段落无法获取音频时长，"
                f"无法生成时间轴{hint}。请尝试重新合成或检查音频文件。"
            ),
        )

    audio_sha256: Optional[str]
    try:
        audio_sha256 = _sha256_file(output_path)
    except Exception:
        audio_sha256 = None

    timeline_name = f"dialogue_timeline_{output_path.stem}.json"
    timeline_rel = Path("dialogue_timeline") / timeline_name
    timeline_path = project_root / timeline_rel
    latest_path = project_root / "dialogue_timeline.json"

    timeline_data = {
        "project_id": project_id,
        "audio_filename": output_name,
        "audio_rel_path": str(rel_output),
        "audio_sha256": audio_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": timeline_items,
    }

    try:
        with project_meta_lock(project_id), project_segments_lock(project_id):
            current_project = load_project(project_id)
            current_segments = _ordered_segments(project_id, load_segments(project_id))
            _revalidate_strict_casting(current_project, current_segments, casting_binding)
            _write_json_atomic(timeline_path, timeline_data)
            _write_json_atomic(latest_path, timeline_data)
    except HTTPException:
        if casting_binding is not None:
            output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存 dialogue_timeline 失败: {exc}",
        ) from exc

    # 导出成功后自动清理旧的导出文件、临时目录等
    try:
        _auto_prune_after_export(project_root)
    except Exception:
        logger.debug("导出后自动清理失败（full）: %s", project_root, exc_info=True)

    return ExportFullResponse(
        success=True,
        output_file=str(rel_output),
        dialogue_timeline_file=str(timeline_rel),
        audio_sha256=audio_sha256,
    )


@router.post(
    "/projects/{project_id}/export/full-with-edits",
    response_model=ExportFullResponse,
    dependencies=[Depends(project_export_guard)],
)
def api_export_full_project_with_edits(project_id: str) -> ExportFullResponse:
    """按当前编辑草稿非破坏性导出整书音频。

    - 与 /export/full 不同，本接口不会修改各分段原始音频文件；
    - 会根据每个 Segment.edit_state 中的 speed/volume/regions/silences
      在导出时临时对各分段音频进行处理，并将结果拼接为一条完整音频；
    - 适合作为“按当前编辑导出（非破坏性）”的导出方案。
    """

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    segments = load_segments(project_id)
    if not segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前项目没有任何段落，无法导出。",
        )

    ordered_segments = _ordered_segments(project_id, segments)
    casting_binding = _strict_casting_binding(project, ordered_segments)

    project_root = _project_root(project_id)

    # 仅导出音频文件实际存在的段落（无论 status 是否为 DONE），避免因状态异常漏掉可播放音频
    playable: list[tuple[Segment, Path]] = []
    missing_segments: list[Segment] = []
    project_root_resolved = project_root.resolve()
    for seg in ordered_segments:
        export_seg = _selected_take_projection(seg)
        if not export_seg.audio_file:
            missing_segments.append(seg)
            continue
        try:
            path = (project_root / str(export_seg.audio_file)).resolve()
        except Exception:
            missing_segments.append(seg)
            continue
        if not path.is_relative_to(project_root_resolved):
            missing_segments.append(seg)
            continue
        if path.is_file():
            playable.append((export_seg, path))
        else:
            missing_segments.append(seg)

    if missing_segments:
        sample = ", ".join(str(seg.index) for seg in missing_segments[:5])
        hint = f"（索引示例: {sample}）" if sample else ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"存在 {len(missing_segments)} 个段落缺少音频，无法导出{hint}。"
                "请先完成合成或刷新状态。"
            ),
        )
    if not playable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有可导出的音频，请先完成批量合成。",
        )

    final_dir = project_root / "audio" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = project_root / "audio" / f"tmp_full_with_edits_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    temp_files: list[Path] = []
    timeline_items: list[dict[str, object]] = []
    cursor = 0.0
    invalid_duration_segments: list[Segment] = []

    try:
        for idx, (seg, audio_path) in enumerate(playable):
            payload = _build_segment_audio_edit_request_from_state(seg.edit_state)
            # 将处理后的单段音频导出为临时 wav 文件，供后续 concat 使用
            seg_tmp = tmp_dir / f"full_with_edits_{idx:04d}_{seg.id}.wav"
            temp_files.append(seg_tmp)

            try:
                _render_audio_edit_to_path(
                    audio_path,
                    seg_tmp,
                    payload,
                    base_speed=getattr(seg, "playback_speed", 1.0),
                    base_volume=getattr(seg, "volume_gain", 1.0),
                )
            except FileNotFoundError as exc:
                # ffmpeg 可执行文件缺失
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "音频处理失败：系统未找到 ffmpeg 可执行文件，请安装 ffmpeg 并确保其在系统 PATH 中。"
                    ),
                ) from exc
            except ffmpeg.Error as exc:  # type: ignore[attr-defined]
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"处理分段 {seg.id} 的音频失败: {exc}",
                ) from exc

            dur = _probe_audio_duration_seconds(seg_tmp)
            if not (dur > 0):
                invalid_duration_segments.append(seg)
                continue
            start = max(0.0, cursor - (dialogue_crossfade if idx > 0 else 0.0))
            end = cursor + dur
            if idx > 0:
                end -= dialogue_crossfade
            timeline_items.append(
                {
                    "segment_id": seg.id,
                    "index": seg.index,
                    "role_id": getattr(seg, "role_id", None),
                    "text": (getattr(seg, "text", "") or "").strip(),
                    "start": start,
                    "end": end,
                }
            )
            cursor = end

        if not temp_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有可导出的音频，请先完成批量合成。",
            )

        if invalid_duration_segments:
            sample = ", ".join(str(seg.index) for seg in invalid_duration_segments[:5])
            hint = f"（索引示例: {sample}）" if sample else ""
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"存在 {len(invalid_duration_segments)} 个段落无法获取音频时长，"
                    f"无法生成时间轴{hint}。请尝试重新合成或检查音频文件。"
                ),
            )

        # 使用本地时区时间生成文件名时间戳，便于与系统时间直观对应
        now_local = datetime.now(timezone.utc).astimezone()
        output_name = f"full_edit_{now_local.strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = final_dir / output_name

        try:
            if len(temp_files) == 1:
                shutil.copyfile(temp_files[0], output_path)
            else:
                target_sample_rate, target_channels = _probe_audio_format(temp_files[0])
                audio_streams = [
                    _normalize_audio_stream(
                        ffmpeg.input(str(p)).audio,
                        target_sample_rate,
                        target_channels,
                    )
                    for p in temp_files
                ]
                if dialogue_crossfade > 0:
                    merged = audio_streams[0]
                    for next_stream in audio_streams[1:]:
                        merged = ffmpeg.filter(
                            [merged, next_stream], "acrossfade",
                            d=dialogue_crossfade, c1="tri", c2="tri",
                        )
                    stream = merged.output(str(output_path))
                else:
                    stream = ffmpeg.concat(*audio_streams, v=0, a=1).output(str(output_path))
                ffmpeg.run(stream, overwrite_output=True)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="音频合并失败：系统未找到 ffmpeg 可执行文件，请安装 ffmpeg 并确保其在系统 PATH 中。",
            ) from exc
        except ffmpeg.Error as exc:  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"音频合并失败: {exc}",
            ) from exc

        rel_output = Path("audio") / "final" / output_name

        audio_sha256: Optional[str]
        try:
            audio_sha256 = _sha256_file(output_path)
        except Exception:
            audio_sha256 = None

        timeline_name = f"dialogue_timeline_{output_path.stem}.json"
        timeline_rel = Path("dialogue_timeline") / timeline_name
        timeline_path = project_root / timeline_rel
        latest_path = project_root / "dialogue_timeline.json"

        timeline_data = {
            "project_id": project_id,
            "audio_filename": output_name,
            "audio_rel_path": str(rel_output),
            "audio_sha256": audio_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "items": timeline_items,
        }

        try:
            with project_meta_lock(project_id), project_segments_lock(project_id):
                current_project = load_project(project_id)
                current_segments = _ordered_segments(project_id, load_segments(project_id))
                _revalidate_strict_casting(current_project, current_segments, casting_binding)
                _write_json_atomic(timeline_path, timeline_data)
                _write_json_atomic(latest_path, timeline_data)
        except HTTPException:
            if casting_binding is not None and "output_path" in locals():
                output_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"保存 dialogue_timeline 失败: {exc}",
            ) from exc

        # 导出成功后自动清理旧的导出文件、临时目录等
        try:
            _auto_prune_after_export(project_root)
        except Exception:
            logger.debug("导出后自动清理失败（full-with-edits）: %s", project_root, exc_info=True)

        return ExportFullResponse(
            success=True,
            output_file=str(rel_output),
            dialogue_timeline_file=str(timeline_rel),
            audio_sha256=audio_sha256,
        )

    finally:
        # 清理临时分段文件
        for p in temp_files:
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                logger.debug("删除临时分段文件失败: %s", p, exc_info=True)

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            logger.debug("删除临时导出目录失败: %s", tmp_dir, exc_info=True)



@router.get(
    "/projects/{project_id}/exports/final",
    response_model=List[FinalAudioInfo],
)
async def api_list_final_audios(project_id: str) -> List[FinalAudioInfo]:
    """列出项目下已导出的整书音频列表。"""

    project_root = _project_root(project_id)
    final_dir = project_root / "audio" / "final"

    if not final_dir.is_dir():
        return []

    items: List[FinalAudioInfo] = []
    # 支持常见音频后缀，按修改时间倒序排列
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for path in files:
        stat = path.stat()
        # 使用 UTC 时区感知时间，前端再按浏览器本地时区展示，避免多次时区偏移
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        duration: Optional[float] = None
        try:
            probe = ffmpeg.probe(str(path))
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio" and "duration" in stream:
                    duration = float(stream["duration"])
                    break
        except Exception:
            duration = None

        rel_path = Path("audio") / "final" / path.name
        items.append(
            FinalAudioInfo(
                filename=path.name,
                rel_path=str(rel_path),
                created_at=created_at,
                duration_seconds=duration,
            )
        )

    return items


@router.get("/projects/{project_id}/exports/final/{filename}")
async def api_get_final_audio(project_id: str, filename: str) -> FileResponse:
    """返回指定整书导出音频文件，用于前端试听。"""

    # 仅允许简单文件名，防止路径穿越
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    project_root = _project_root(project_id)
    final_dir = project_root / "audio" / "final"
    try:
        final_dir_resolved = final_dir.resolve()
        audio_path = (final_dir / filename).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )
    if not audio_path.is_relative_to(final_dir_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的导出音频不存在",
        )

    return FileResponse(audio_path, media_type=_audio_mime_from_suffix(audio_path.suffix))


@router.post("/projects/{project_id}/exports/final/{filename}/open-location")
async def api_open_final_audio_location(project_id: str, filename: str) -> dict[str, bool]:
    """在服务器操作系统中打开指定导出音频所在目录。"""

    _check_open_location_enabled()

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    project_root = _project_root(project_id)
    final_dir = project_root / "audio" / "final"
    try:
        final_dir_resolved = final_dir.resolve()
        audio_path = (final_dir / filename).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )
    if not audio_path.is_relative_to(final_dir_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的导出音频不存在",
        )

    try:
        if sys.platform.startswith("win"):
            # Windows 使用资源管理器选中该文件
            subprocess.Popen(["explorer", f"/select,{audio_path}"])
        elif sys.platform == "darwin":
            # macOS 使用 Finder 显示该文件
            subprocess.Popen(["open", "-R", str(audio_path)])
        else:
            # Linux 下退化为打开目录
            subprocess.Popen(["xdg-open", str(final_dir)])
    except Exception as exc:  # pragma: no cover - 与本地环境相关
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"打开文件所在目录失败: {exc}",
        ) from exc

    return {"success": True}


@router.get("/projects/{project_id}/exports/final/{filename}/download")
async def api_download_final_audio(project_id: str, filename: str) -> FileResponse:
    """下载指定整书导出音频文件。"""

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    project_root = _project_root(project_id)
    final_dir = project_root / "audio" / "final"
    try:
        final_dir_resolved = final_dir.resolve()
        audio_path = (final_dir / filename).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )
    if not audio_path.is_relative_to(final_dir_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的导出音频不存在",
        )

    return FileResponse(
        audio_path,
        media_type=_audio_mime_from_suffix(audio_path.suffix),
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete(
    "/projects/{project_id}/exports/final/{filename}",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_final_audio(project_id: str, filename: str) -> dict[str, bool]:
    """删除指定整书导出音频及其相关文件（dialogue_timeline 等）。"""

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    project_root = _project_root(project_id)
    final_dir = project_root / "audio" / "final"
    try:
        final_dir_resolved = final_dir.resolve()
        audio_path = (final_dir / filename).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )
    if not audio_path.is_relative_to(final_dir_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的导出音频不存在",
        )

    # 删除音频文件
    try:
        audio_path.unlink()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除音频文件失败: {exc}",
        ) from exc

    # 删除对应的 dialogue_timeline 文件（如果存在）
    stem = Path(filename).stem
    timeline_path = project_root / "dialogue_timeline" / f"dialogue_timeline_{stem}.json"
    if timeline_path.is_file():
        try:
            timeline_path.unlink()
        except Exception:
            logger.debug("删除 timeline 文件失败: %s", timeline_path, exc_info=True)

    # 如果根目录的 dialogue_timeline.json 指向被删除的音频，也清理
    latest_timeline = project_root / "dialogue_timeline.json"
    if latest_timeline.is_file():
        try:
            import json
            with latest_timeline.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("audio_filename") == filename:
                latest_timeline.unlink()
        except Exception:
            logger.debug("更新 latest timeline 失败: %s", latest_timeline, exc_info=True)

    audio_rel_path = (Path("audio") / "final" / filename).as_posix()
    try:
        subtitles = list_subtitles(project_id)
    except Exception:
        subtitles = []
    for sub in subtitles:
        sub_audio_rel_path = Path(sub.audio_rel_path or "").as_posix()
        if sub.audio_filename != filename and sub_audio_rel_path != audio_rel_path:
            continue
        try:
            delete_subtitle(project_id, sub.id)
        except Exception:
            logger.debug("删除字幕文件失败: %s/%s", project_id, sub.id, exc_info=True)

    return {"success": True}


@router.get("/projects/{project_id}/segments/{segment_id}/audio")
async def api_get_segment_audio(project_id: str, segment_id: str) -> FileResponse:
    """返回单个分段对应的音频文件，用于前端试听。"""

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    segments = load_segments(project_id)
    target: Segment | None = None
    for seg in segments:
        if seg.id == segment_id:
            target = seg
            break

    if target is None or not target.audio_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频不存在",
        )

    project_root = _project_root(project_id)
    try:
        project_root_resolved = project_root.resolve()
        audio_path = (project_root / str(target.audio_file)).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )
    if not audio_path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    return FileResponse(audio_path)


@router.get("/projects/{project_id}/segments/{segment_id}/alt-audio")
async def api_get_segment_alt_audio(project_id: str, segment_id: str) -> FileResponse:
    """返回单个分段的 B 版本（alt）音频文件，用于 A/B 对比试听。"""

    if Path(segment_id).name != segment_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非法分段 ID")

    segments = load_segments(project_id)
    target = next((s for s in segments if s.id == segment_id), None)

    if target is None or not target.alt_audio_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="该段落没有 B 版本音频")

    project_root_path = _project_root(project_id)
    audio_path = (project_root_path / str(target.alt_audio_file)).resolve()
    if not audio_path.is_relative_to(project_root_path.resolve()):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="B 版本音频文件不存在")
    if not audio_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="B 版本音频文件不存在")

    return FileResponse(audio_path)


@router.post(
    "/projects/{project_id}/segments/{segment_id}/edit-audio",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
def api_edit_segment_audio(
    project_id: str,
    segment_id: str,
    payload: SegmentAudioEditRequest,
) -> Segment:
    """保存单个分段的非破坏编辑方案。

    - 支持调整播放速度（speed）和音量倍数（volume）；
    - 可选删除 delete_region 指定的时间区间，并将前后拼接；
    - 只更新当前 Take 的 EditRecipe，不覆盖原始 audio_file；
    - 预览和最终导出使用同一组编辑参数。
    """

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    try:
        segments = load_segments(project_id)
    except SegmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Segment | None = None
    for seg in segments:
        if seg.id == segment_id:
            target = seg
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"指定的分段不存在: {segment_id}",
        )

    if not target.audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前分段没有可编辑的音频，请先完成合成。",
        )

    project_root = _project_root(project_id)
    try:
        project_root_resolved = project_root.resolve()
        audio_path = (project_root / str(target.audio_file)).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )
    if not audio_path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    # Validate against the immutable source, then persist only the recipe.
    duration = _probe_audio_duration_seconds(audio_path)
    normalized_duration = duration if duration > 0 else None
    delete_regions = _normalize_delete_regions(payload, normalized_duration)
    silence_intervals = _normalize_silence_intervals(payload)
    speed = payload.speed if payload.speed is not None else target.playback_speed
    volume = payload.volume if payload.volume is not None else target.volume_gain
    edit_state = SegmentEditState(
        speed=speed,
        volume=volume,
        regions=[
            SegmentEditRegion(id=f"region-{idx + 1}", start=start, end=end)
            for idx, (start, end) in enumerate(delete_regions)
        ],
        silences=[
            SegmentEditSilence(id=f"silence-{idx + 1}", at=at, duration=seconds)
            for idx, (at, seconds) in enumerate(silence_intervals)
        ],
        fade_in=payload.fade_in,
        fade_out=payload.fade_out,
        leading_silence=payload.leading_silence,
        trailing_silence=payload.trailing_silence,
        processing_preset=payload.processing_preset,
    )

    updated_at = datetime.now(timezone.utc)
    try:
        with project_segments_lock(project_id):
            current_segments = load_segments(project_id)
            current_target = next(
                (item for item in current_segments if item.id == segment_id),
                None,
            )
            if current_target is None:
                raise HTTPException(status_code=404, detail="段落不存在")
            try:
                update_active_edit_state(
                    current_target,
                    edit_state,
                    speed=speed,
                    volume=volume,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            current_target.updated_at = updated_at
            save_segments(project_id, current_segments)
            target = current_target
    except SegmentStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            project.updated_at = updated_at
            save_project(project)
    except ProjectStorageError:
        pass

    return target


@router.post(
    "/projects/{project_id}/segments/{segment_id}/edit-history/restore",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
def api_restore_segment_edit_revision(
    project_id: str,
    segment_id: str,
    payload: RestoreEditRevisionRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        target = next((item for item in segments if item.id == segment_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        try:
            restore_active_edit_revision(target, payload.revision_index)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)
    return target



@router.post(
    "/projects/{project_id}/segments/{segment_id}/render-as-new-take",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
def api_render_segment_as_new_take(project_id: str, segment_id: str) -> Segment:
    """固化当前 EditRecipe 为新的派生 Take，原始 Take 保持不变。"""

    segments = load_segments(project_id)
    segment = next((item for item in segments if item.id == segment_id), None)
    if segment is None:
        raise HTTPException(status_code=404, detail="段落不存在")
    source_take = get_slot_take(segment, segment.active_version)
    if source_take is None:
        raise HTTPException(status_code=400, detail="当前版本没有可渲染的 Take")
    if source_take.edit_state is None:
        raise HTTPException(status_code=400, detail="当前 Take 没有已保存的编辑方案")

    project_root_path = _project_root(project_id)
    try:
        source_path = resolve_scoped_project_asset(
            project_root_path,
            source_take.audio_file,
            required_subdir="audio/segments",
            allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="原始 Take 文件不存在或路径非法") from exc
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="原始 Take 文件不存在")

    take_id = uuid.uuid4().hex[:16]
    rel_path = Path("audio") / "segments" / f"{segment.id}_{take_id}.wav"
    output_path = resolve_scoped_project_asset(
        project_root_path,
        str(rel_path),
        required_subdir="audio/segments",
        allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
    )
    stage_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.stage.wav")
    payload = _build_segment_audio_edit_request_from_state(source_take.edit_state)

    try:
        _render_audio_edit_to_path(
            source_path,
            stage_path,
            payload,
            base_speed=source_take.baked_speed,
            base_volume=source_take.baked_volume,
        )
        if not stage_path.is_file() or _probe_audio_duration_seconds(stage_path) <= 0:
            raise RuntimeError("渲染结果为空或不可解码")
        os.replace(stage_path, output_path)
        audio_sha256 = _sha256_file(output_path)
    except FileNotFoundError as exc:
        _cleanup_file(stage_path)
        raise HTTPException(status_code=500, detail="未找到 FFmpeg，无法渲染新 Take") from exc
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        _cleanup_file(stage_path)
        raise HTTPException(status_code=500, detail=f"渲染新 Take 失败: {exc}") from exc
    except Exception as exc:
        _cleanup_file(stage_path)
        _cleanup_file(output_path)
        raise HTTPException(status_code=500, detail=f"渲染新 Take 失败: {exc}") from exc

    try:
        with project_segments_lock(project_id):
            current_segments = load_segments(project_id)
            current = next((item for item in current_segments if item.id == segment_id), None)
            if current is None:
                raise HTTPException(status_code=404, detail="段落不存在")
            current_source = get_slot_take(current, current.active_version)
            if current_source is None or current_source.id != source_take.id:
                raise HTTPException(status_code=409, detail="渲染期间当前 Take 已发生变化，请重新操作")
            new_take = create_take(
                current,
                slot=current.active_version,
                take_id=take_id,
                audio_file=str(rel_path),
                audio_sha256=audio_sha256,
                source_take_id=source_take.id,
            )
            new_take.edit_state = None
            new_take.playback_speed = 1.0
            new_take.volume_gain = 1.0
            new_take.baked_speed = 1.0
            new_take.baked_volume = 1.0
            sync_legacy_projection(current)
            current.status = SegmentStatus.DONE
            current.updated_at = datetime.now(timezone.utc)
            save_segments(project_id, current_segments)
            segment = current
    except Exception:
        _cleanup_file(output_path)
        raise

    return segment


@router.post(
    "/projects/{project_id}/segments/{segment_id}/preview-audio",
    dependencies=[Depends(project_export_guard)],
)
def api_preview_segment_audio(
    project_id: str,
    segment_id: str,
    payload: SegmentAudioEditRequest,
) -> Response:
    """生成单个分段按当前编辑参数处理后的临时音频，用于前端试听。

    - 支持与 /edit-audio 相同的变速、音量、删除区间和插入静音；
    - 不会覆盖原 audio_file，也不会更新 Segment 或项目，仅返回一次性音频数据；
    - 适合在前端点击“试听编辑结果”时调用。
    """

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    try:
        segments = load_segments(project_id)
    except SegmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Segment | None = None
    for seg in segments:
        if seg.id == segment_id:
            target = seg
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"指定的分段不存在: {segment_id}",
        )

    if not target.audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前分段没有可编辑的音频，请先完成合成。",
        )

    project_root = _project_root(project_id)
    try:
        project_root_resolved = project_root.resolve()
        audio_path = (project_root / str(target.audio_file)).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )
    if not audio_path.is_relative_to(project_root_resolved):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定段落的音频文件不存在",
        )

    active_take = get_slot_take(target, target.active_version)
    base_speed = active_take.baked_speed if active_take is not None else target.playback_speed
    base_volume = active_take.baked_volume if active_take is not None else target.volume_gain
    _desired_speed, _desired_volume, speed_ratio, volume_ratio = _resolve_audio_edit_targets(
        target_speed=payload.speed,
        target_volume=payload.volume,
        base_speed=base_speed,
        base_volume=base_volume,
    )

    # 解析删除区间，并尽量根据真实音频长度做校验，同时尝试获取采样率和声道数，供插入静音时复用
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    try:
        probe = ffmpeg.probe(str(audio_path))
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio":
                if "duration" in stream:
                    duration = float(stream["duration"])
                if "sample_rate" in stream:
                    try:
                        sample_rate = int(stream["sample_rate"])
                    except (TypeError, ValueError):  # pragma: no cover - 容错
                        sample_rate = None
                if "channels" in stream:
                    try:
                        channels = int(stream["channels"])
                    except (TypeError, ValueError):  # pragma: no cover - 容错
                        channels = None
                break
    except Exception:
        duration = None

    # 规范化并应用（可能为多段）删除区间
    delete_regions = _normalize_delete_regions(payload, duration)

    base = ffmpeg.input(str(audio_path))
    audio_stream = base.audio

    if delete_regions:
        audio_stream = _apply_delete_regions(audio_stream, delete_regions)

    # 根据请求在指定时间插入一段或多段静音
    intervals = _normalize_silence_intervals(payload)
    if intervals:
        audio_stream = _apply_silence_intervals(
            audio_stream,
            intervals,
            sample_rate=sample_rate,
            channels=channels,
        )

    if abs(speed_ratio - 1.0) > 1e-3:
        audio_stream = audio_stream.filter_("atempo", speed_ratio)

    if abs(volume_ratio - 1.0) > 1e-3:
        audio_stream = audio_stream.filter_("volume", volume_ratio)

    # 使用独立的预览临时文件，避免覆盖原音频
    tmp_path = audio_path.with_name(
        f"{audio_path.stem}.preview.{uuid.uuid4().hex}{audio_path.suffix}"
    )
    stream = ffmpeg.output(audio_stream, str(tmp_path))

    try:
        ffmpeg.run(stream, overwrite_output=True)
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        _cleanup_file(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"音频处理失败: {exc}",
        ) from exc
    except Exception:
        _cleanup_file(tmp_path)
        raise

    if not tmp_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="读取预览音频失败: 预览文件不存在",
        )

    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    elif suffix in {".m4a", ".aac"}:
        media_type = "audio/aac"
    elif suffix == ".ogg":
        media_type = "audio/ogg"
    elif suffix == ".flac":
        media_type = "audio/flac"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        tmp_path,
        media_type=media_type,
        background=BackgroundTask(_cleanup_file, tmp_path),
    )


# ---------------------------------------------------------------------------
# 段落 A/B 对比：每个段落可有一个 B 版本（alt），与 A 共享角色和文本，
# 但可使用不同的情感描述、情感权重、TTS 后端。
# ---------------------------------------------------------------------------

class SynthesizeAltRequest(BaseModel):
    """合成 B 版本的请求体。"""
    emotion_text: Optional[str] = Field(None, description="B 版本情感描述")
    emotion_alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="B 版本情感权重")
    tts_backend: Optional[str] = Field(None, description="B 版本 TTS 后端")


@router.post(
    "/projects/{project_id}/segments/{segment_id}/synthesize-alt",
    response_model=Segment,
    dependencies=[Depends(standalone_segment_synthesis_guard)],
)
async def api_synthesize_alt(
    project_id: str,
    segment_id: str,
    payload: SynthesizeAltRequest,
) -> Segment:
    """为指定段落合成 B 版本音频。角色和文本与 A 相同，可覆盖情感和后端。"""
    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    segments = load_segments(project_id)
    seg = next((s for s in segments if s.id == segment_id), None)
    if seg is None:
        raise HTTPException(status_code=404, detail="段落不存在")

    # 构建临时段落副本用于合成，应用 B 槽位参数；不会改变当前查看版本。
    seg_copy = seg.model_copy(deep=True)
    set_active_slot(seg_copy, "b")
    if payload.emotion_text is not None:
        seg_copy.emotion_text = payload.emotion_text
    if payload.emotion_alpha is not None:
        seg_copy.emotion_alpha = payload.emotion_alpha
    if payload.tts_backend is not None:
        if payload.tts_backend not in _SUPPORTED_TTS_BACKENDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tts_backend 必须是 indextts、cosyvoice3、qwen3tts 或 qwen3tts_cv。",
            )
        seg_copy.tts_backend = payload.tts_backend
    capture_active_legacy_settings(seg_copy)

    from dubbing_service.app.routers.ws_synthesis import _create_providers, _get_default_backend
    providers = _create_providers()
    default_backend = _get_default_backend()

    # 获取前一分段文本（上下文）；load_segments 已按章节顺序 + 章节内行号返回
    sorted_segs = segments
    seg_index_map = {s.id: i for i, s in enumerate(sorted_segs)}
    previous_text: Optional[str] = None
    current_idx = seg_index_map.get(seg.id)
    if current_idx is not None and current_idx > 0:
        previous_text = sorted_segs[current_idx - 1].text

    import asyncio
    result = await asyncio.to_thread(
        _synthesize_one, project, project_id, seg_copy, providers, default_backend, previous_text,
    )

    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error", "B 版本合成失败"))

    alt_rel = str(result.get("audio_file", ""))

    # 原子登记新 Take；旧 B Take 只归档，不删除原始音频。
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        target = next((s for s in segments if s.id == segment_id), None)
        if target is None:
            _unlink_project_relative_file(_project_root(project_id), alt_rel)
            raise HTTPException(status_code=404, detail="段落不存在")
        b_state = target.take_slots["b"]
        b_state.emotion_text = seg_copy.emotion_text
        b_state.emotion_alpha = seg_copy.emotion_alpha
        b_state.tts_backend = _resolve_tts_backend(seg_copy.tts_backend, default_backend)
        create_take(
            target,
            slot="b",
            take_id=str(result["take_id"]),
            audio_file=alt_rel,
            audio_sha256=result.get("audio_sha256"),
        )
        # Compatibility endpoint must not unexpectedly move the user's view.
        set_active_slot(target, seg.active_version)
        target.status = SegmentStatus.DONE
        target.updated_at = datetime.now(timezone.utc)
        try:
            save_segments(project_id, segments)
        except Exception:
            _unlink_project_relative_file(_project_root(project_id), alt_rel)
            raise

    return target


@router.post(
    "/projects/{project_id}/segments/{segment_id}/swap-alt",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_swap_alt(project_id: str, segment_id: str) -> Segment:
    """兼容端点：切换 A/B 视图，不交换或覆盖任何 Take 数据。"""
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        seg = next((s for s in segments if s.id == segment_id), None)
        if seg is None:
            raise HTTPException(status_code=404, detail="段落不存在")

        destination = "b" if normalize_slot(seg.active_version) == "a" else "a"
        set_active_slot(seg, destination)

        active_state = seg.take_slots[normalize_slot(seg.active_version)]
        seg.status = (
            SegmentStatus.DONE
            if seg.audio_file and not active_state.needs_synthesis
            else SegmentStatus.PENDING
        )

        seg.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)

    return seg


@router.delete(
    "/projects/{project_id}/segments/{segment_id}/alt",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_clear_alt(project_id: str, segment_id: str) -> Segment:
    """从 B 槽解除当前 Take；音频进入历史归档而非立即删除。"""
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        seg = next((s for s in segments if s.id == segment_id), None)
        if seg is None:
            raise HTTPException(status_code=404, detail="段落不存在")

        clear_take_slot(seg, "b")
        set_active_slot(seg, "a")
        if seg.status == SegmentStatus.DONE and not seg.audio_file:
            seg.status = SegmentStatus.PENDING
        seg.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)

    return seg


class SelectTakeRequest(BaseModel):
    take_id: str


class RestoreTakeRequest(BaseModel):
    slot: str = Field(..., pattern="^[ab]$")


class PinTakeRequest(BaseModel):
    pinned: bool = True


class PruneTakesResponse(BaseModel):
    removed_takes: int
    removed_files: int


@router.get(
    "/projects/{project_id}/segments/{segment_id}/takes",
    response_model=List[SegmentTake],
)
async def api_list_segment_takes(project_id: str, segment_id: str) -> List[SegmentTake]:
    segment = next((s for s in load_segments(project_id) if s.id == segment_id), None)
    if segment is None:
        raise HTTPException(status_code=404, detail="段落不存在")
    return sorted(segment.takes, key=lambda item: item.created_at, reverse=True)


@router.get("/projects/{project_id}/segments/{segment_id}/takes/{take_id}/audio")
async def api_get_take_audio(
    project_id: str,
    segment_id: str,
    take_id: str,
) -> FileResponse:
    segment = next((s for s in load_segments(project_id) if s.id == segment_id), None)
    if segment is None:
        raise HTTPException(status_code=404, detail="段落不存在")
    take = get_take(segment, take_id)
    if take is None or take.missing:
        raise HTTPException(status_code=404, detail="Take 音频不存在")
    root = _project_root(project_id)
    try:
        path = resolve_scoped_project_asset(
            root,
            take.audio_file,
            required_subdir="audio/segments",
            allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Take 音频路径非法") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Take 音频不存在")
    return FileResponse(path, media_type=_audio_mime_from_suffix(path.suffix))


@router.put(
    "/projects/{project_id}/segments/{segment_id}/active-version/{slot}",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_set_active_version(project_id: str, segment_id: str, slot: str) -> Segment:
    if slot not in {"a", "b"}:
        raise HTTPException(status_code=400, detail="slot 必须是 a 或 b")
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((s for s in segments if s.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        set_active_slot(segment, slot)
        active_state = segment.take_slots[normalize_slot(segment.active_version)]
        segment.status = (
            SegmentStatus.DONE
            if segment.audio_file and not active_state.needs_synthesis
            else SegmentStatus.PENDING
        )
        segment.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)
    return segment


@router.put(
    "/projects/{project_id}/segments/{segment_id}/selected-take",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_select_segment_take(
    project_id: str,
    segment_id: str,
    payload: SelectTakeRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((s for s in segments if s.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        try:
            select_take(segment, payload.take_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        segment.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)
    return segment


@router.post(
    "/projects/{project_id}/segments/{segment_id}/takes/{take_id}/restore",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_restore_segment_take(
    project_id: str,
    segment_id: str,
    take_id: str,
    payload: RestoreTakeRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((s for s in segments if s.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        try:
            restore_take_to_slot(segment, take_id, payload.slot)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        segment.status = SegmentStatus.DONE
        segment.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)
    return segment


@router.put(
    "/projects/{project_id}/segments/{segment_id}/takes/{take_id}/pin",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_pin_segment_take(
    project_id: str,
    segment_id: str,
    take_id: str,
    payload: PinTakeRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((s for s in segments if s.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        take = get_take(segment, take_id)
        if take is None:
            raise HTTPException(status_code=404, detail="Take 不存在")
        take.pinned = payload.pinned
        segment.updated_at = datetime.now(timezone.utc)
        save_segments(project_id, segments)
    return segment


@router.post(
    "/projects/{project_id}/takes/prune",
    response_model=PruneTakesResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_prune_project_takes(project_id: str) -> PruneTakesResponse:
    """Apply the default 90-day / 20-history retention policy safely."""

    removed = []
    master_take_ids = {
        snapshot.take_id
        for master in list_masters(project_id)
        for snapshot in master.segments
    }
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        for segment in segments:
            removed.extend(
                prune_archived_takes(
                    segment,
                    max_archived=20,
                    max_age_days=90,
                    protected_take_ids=master_take_ids,
                )
            )
        if removed:
            save_segments(project_id, segments)
        remaining_paths = {
            take.audio_file for segment in segments for take in segment.takes
        }

    removed_files = 0
    for take in removed:
        if take.audio_file in remaining_paths:
            continue
        root = _project_root(project_id)
        try:
            path = resolve_scoped_project_asset(
                root,
                take.audio_file,
                required_subdir="audio/segments",
                allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
            )
        except Exception:
            continue
        try:
            if path.is_file():
                path.unlink()
                removed_files += 1
        except OSError:
            logger.warning("清理过期 Take 文件失败: %s", path, exc_info=True)
    return PruneTakesResponse(
        removed_takes=len(removed),
        removed_files=removed_files,
    )
