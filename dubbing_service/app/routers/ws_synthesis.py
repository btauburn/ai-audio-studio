"""WebSocket endpoint for real-time batch TTS synthesis progress.

客户端通过 WebSocket 发送待合成的段落 ID 列表，服务端逐段合成并实时推送进度。
替代原有的前端逐段 HTTP 轮询模式，减少请求开销并提供更精准的进度反馈。

协议：
  Client → Server:
    {"action": "start", "segment_ids": [...], "segment_backends": {"id": "backend"}}
    {"action": "stop"}

  Server → Client:
    {"type": "batch_start", "total": N, "segment_ids": [...]}
    {"type": "segment_result", "segment_id": "...", "success": bool, "audio_file": "...",
     "error": "...", "warning": "...", "done": N, "total": N}
    {"type": "batch_complete", "total": N, "succeeded": N, "failed": N, "stopped": bool}
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from dubbing_service.app.models.enums import SegmentStatus
from dubbing_service.app.models.project import Segment
from dubbing_service.app.core.tts_operations import (
    count_actively_synthesizing_segments,
    has_running_batch_for_project,
    is_segment_actively_synthesizing,
    set_segment_actively_synthesizing,
    try_register_running_batch,
    unregister_running_batch,
)
from dubbing_service.app.providers.tts import (
    CosyVoice3TTSProvider,
    IndexTTSTTSProvider,
    Qwen3TTSCustomVoiceProvider,
    Qwen3TTSProvider,
)
from dubbing_service.app.routers.synthesis import _collect_missing_speaker_roles, _synthesize_one
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.segments import (
    load_segments,
    project_segments_lock,
    save_segments,
)
from dubbing_service.app.services.takes import clear_slot, create_take
from dubbing_service.app.routers.synthesis import _unlink_project_relative_file, _project_root
from dubbing_service.app.storage.jobs import (
    JobStorageError,
    cancel_job,
    get_job,
    interrupt_job,
    mark_job_active,
    record_job_segment,
    start_job,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Kept as compatibility exports for older router code that imported the
# process-local coordination helpers from this module.
__all__ = [
    "count_actively_synthesizing_segments",
    "has_running_batch_for_project",
    "is_segment_actively_synthesizing",
    "router",
    "ws_batch_synthesis",
]

_SUPPORTED_TTS_BACKENDS = {
    "indextts",
    "cosyvoice3",
    "qwen3tts",
    "qwen3tts_cv",
}

_KEEP_AUDIO = object()


# ---------------------------------------------------------------------------
# Helpers (reused from synthesis.py logic)
# ---------------------------------------------------------------------------

def _get_default_backend() -> str:
    from dubbing_service.app.routers.settings import _load_raw_file_config, _normalize_tts_backend
    file_cfg = _load_raw_file_config()
    return _normalize_tts_backend(file_cfg.get("tts_backend"))


def _resolve_tts_backend(backend: object, default_backend: str) -> str:
    value = str(backend or "").strip()
    if value in _SUPPORTED_TTS_BACKENDS:
        return value
    return default_backend if default_backend in _SUPPORTED_TTS_BACKENDS else "indextts"


def _create_providers() -> Dict[str, Any]:
    return {
        "indextts": IndexTTSTTSProvider(),
        "cosyvoice3": CosyVoice3TTSProvider(),
        "qwen3tts": Qwen3TTSProvider(),
        "qwen3tts_cv": Qwen3TTSCustomVoiceProvider(),
    }


def _save_segment_state(
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
    ts = when or datetime.now(timezone.utc)
    with project_segments_lock(project_id):
        current_segments = load_segments(project_id)
        current_target = next((s for s in current_segments if s.id == segment_id), None)
        if current_target is None:
            raise RuntimeError(f"待持久化的段落不存在: {segment_id}")
        current_target.status = segment_status
        if audio_file is not _KEEP_AUDIO:
            if audio_file is None:
                clear_slot(current_target, slot or current_target.active_version)
            elif take_id:
                create_take(
                    current_target,
                    slot=slot or current_target.active_version,
                    take_id=take_id,
                    audio_file=str(audio_file),
                    audio_sha256=audio_sha256,
                    synthesis_text=synthesis_text,
                    generation_metadata=generation_metadata,
                )
            else:
                current_target.audio_file = str(audio_file)
        current_target.updated_at = ts
        save_segments(project_id, current_segments)


def _durability_failure(stage: str, exc: Exception) -> dict[str, object]:
    """Build a client-visible failure for a state that was not committed."""

    return {
        "success": False,
        "audio_file": None,
        "error": f"合成状态持久化失败（{stage}）: {type(exc).__name__}",
        "warning": None,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/synthesis/{project_id}")
async def ws_batch_synthesis(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    logger.info("WebSocket synthesis connected: project=%s", project_id)

    stop_event = asyncio.Event()
    listen_task: asyncio.Task | None = None
    batch_lease = None
    job_id: str | None = None
    job_registered_active = False
    transport_interrupted = False
    explicit_stop_requested = False

    async def _listen_for_stop() -> None:
        """后台任务：监听客户端发来的 stop 指令。"""
        nonlocal explicit_stop_requested
        try:
            while not stop_event.is_set():
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("action") == "stop":
                    explicit_stop_requested = True
                    stop_event.set()
                    break
        except WebSocketDisconnect:
            stop_event.set()
        except Exception:
            logger.debug("监听 stop 指令时发生异常", exc_info=True)
            stop_event.set()

    try:
        # 等待客户端发送 start 指令
        raw = await websocket.receive_text()
        msg = json.loads(raw)

        if msg.get("action") != "start":
            await websocket.send_json({"type": "error", "message": "期望 action=start"})
            await websocket.close()
            return

        segment_ids: List[str] = msg.get("segment_ids") or []
        segment_backends: Dict[str, str] = msg.get("segment_backends") or {}
        raw_job_id = msg.get("job_id")
        job_id = str(raw_job_id).strip() if raw_job_id else None

        if not segment_ids:
            await websocket.send_json({"type": "error", "message": "segment_ids 为空"})
            await websocket.close()
            return

        if job_id:
            try:
                job = get_job(project_id, job_id)
                if job is None:
                    raise JobStorageError("持久任务不存在")
                invalid = [item for item in segment_ids if item not in job.segment_ids]
                if invalid:
                    raise JobStorageError("WebSocket 分段不属于持久任务")
                mark_job_active(project_id, job_id, True)
                job_registered_active = True
                start_job(project_id, job_id)
            except JobStorageError as exc:
                if job_registered_active:
                    mark_job_active(project_id, job_id, False)
                    job_registered_active = False
                await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.close()
                return

        # Acquire the process-wide TTS/GPU lane before touching project state.
        # REST batches and worker lifecycle operations use the same lease.
        batch_lease = try_register_running_batch(project_id)
        if batch_lease is None:
            await websocket.send_json({
                "type": "error",
                "message": "当前已有后台批量合成或 TTS 管理操作正在执行，请稍候重试。",
            })
            await websocket.close()
            return

        # 加载项目和段落
        try:
            project = load_project(project_id)
        except ProjectStorageError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close()
            return

        segments = load_segments(project_id)
        seg_map = {s.id: s for s in segments}

        targets: List[Segment] = []
        for sid in segment_ids:
            seg = seg_map.get(sid)
            if seg is None:
                continue
            targets.append(seg)

        if not targets:
            await websocket.send_json({"type": "error", "message": "没有找到匹配的段落"})
            await websocket.close()
            return

        # 如果客户端指定了后端，覆盖分段的 tts_backend
        for seg in targets:
            if seg.id in segment_backends:
                seg.tts_backend = _resolve_tts_backend(segment_backends[seg.id], _get_default_backend())

        default_backend = _get_default_backend()
        missing_roles = _collect_missing_speaker_roles(project, targets, default_backend)

        if missing_roles:
            await websocket.send_json({
                "type": "error",
                "message": f"以下角色缺少参考音频: {', '.join(missing_roles)}",
            })
            await websocket.close()
            return

        total = len(targets)
        await websocket.send_json({
            "type": "batch_start",
            "total": total,
            "segment_ids": [s.id for s in targets],
        })

        # 启动 stop 监听
        listen_task = asyncio.create_task(_listen_for_stop())

        # 准备合成
        providers = _create_providers()
        default_backend = _get_default_backend()

        # load_segments 已按章节顺序 + 章节内行号返回，直接用于上下文映射
        sorted_segments = segments
        seg_index_map = {s.id: i for i, s in enumerate(sorted_segments)}

        succeeded = 0
        failed = 0

        for seg in targets:
            if stop_event.is_set():
                # 客户端请求停止
                await websocket.send_json({
                    "type": "batch_complete",
                    "total": total,
                    "succeeded": succeeded,
                    "failed": failed,
                    "stopped": True,
                })
                break

            try:
                _save_segment_state(
                    project_id,
                    seg.id,
                    SegmentStatus.GENERATING,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to mark segment generating: project=%s seg=%s",
                    project_id,
                    seg.id,
                    exc_info=True,
                )
                result = _durability_failure("GENERATING", exc)
                failed += 1
                try:
                    await websocket.send_json({
                        "type": "segment_result",
                        "segment_id": seg.id,
                        **result,
                        "done": succeeded + failed,
                        "total": total,
                    })
                except Exception:
                    logger.debug(
                        "推送分段持久化错误失败，提前结束批量推送: project=%s seg=%s",
                        project_id,
                        seg.id,
                    )
                    break
                continue

            # 获取前一分段文本
            previous_text: Optional[str] = None
            current_idx = seg_index_map.get(seg.id)
            if current_idx is not None and current_idx > 0:
                prev_seg = sorted_segments[current_idx - 1]
                previous_text = prev_seg.text

            # 在线程池中执行同步 TTS 调用，避免阻塞事件循环
            set_segment_actively_synthesizing(project_id, seg.id, True)
            try:
                result = await asyncio.to_thread(
                    _synthesize_one,
                    project, project_id, seg, providers,
                    default_backend, previous_text,
                )
            finally:
                set_segment_actively_synthesizing(project_id, seg.id, False)

            result = dict(result)
            now = datetime.now(timezone.utc)
            if result["success"]:
                persisted_status = SegmentStatus.DONE
                persisted_audio_file = result["audio_file"]
            else:
                persisted_status = SegmentStatus.FAILED
                persisted_audio_file = None

            # 立即持久化该段落结果
            try:
                if persisted_status == SegmentStatus.DONE:
                    _save_segment_state(
                        project_id,
                        seg.id,
                        persisted_status,
                        audio_file=persisted_audio_file,
                        take_id=result.get("take_id"),
                        slot=result.get("slot"),
                        audio_sha256=result.get("audio_sha256"),
                        synthesis_text=result.get("synthesis_text"),
                        generation_metadata=result.get("generation_metadata"),
                        when=now,
                    )
                else:
                    _save_segment_state(
                        project_id,
                        seg.id,
                        persisted_status,
                        when=now,
                    )
            except Exception as exc:
                generated_path = result.get("audio_file")
                logger.warning(
                    "Failed to save segment state: project=%s seg=%s status=%s",
                    project_id,
                    seg.id,
                    persisted_status,
                    exc_info=True,
                )
                result = _durability_failure(persisted_status.value.upper(), exc)
                if generated_path:
                    _unlink_project_relative_file(_project_root(project_id), str(generated_path))

            if job_id:
                try:
                    record_job_segment(
                        project_id,
                        job_id,
                        seg.id,
                        success=bool(result.get("success")),
                        error=str(result.get("error") or "") or None,
                    )
                except JobStorageError:
                    logger.warning(
                        "Failed to persist synthesis job progress: project=%s job=%s seg=%s",
                        project_id,
                        job_id,
                        seg.id,
                        exc_info=True,
                    )

            if result["success"]:
                succeeded += 1
            else:
                failed += 1

            # 推送进度
            progress_msg: Dict[str, Any] = {
                "type": "segment_result",
                "segment_id": seg.id,
                "success": result["success"],
                "audio_file": result.get("audio_file"),
                "take_id": result.get("take_id"),
                "slot": result.get("slot"),
                "error": result.get("error"),
                "warning": result.get("warning"),
                "done": succeeded + failed,
                "total": total,
            }
            try:
                await websocket.send_json(progress_msg)
            except Exception:
                logger.debug("推送分段进度失败，提前结束批量推送: project=%s seg=%s", project_id, seg.id)
                break

            # 让事件循环有机会处理 stop 消息
            await asyncio.sleep(0)
        else:
            # 正常完成（没有被 break）
            try:
                with project_meta_lock(project_id):
                    latest_project = load_project(project_id)
                    latest_project.updated_at = datetime.now(timezone.utc)
                    save_project(latest_project)
            except Exception:
                logger.warning("保存项目更新时间失败: project=%s", project_id, exc_info=True)

            await websocket.send_json({
                "type": "batch_complete",
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
                "stopped": False,
            })

    except WebSocketDisconnect:
        transport_interrupted = True
        logger.info("WebSocket synthesis disconnected: project=%s", project_id)
    except Exception as exc:
        transport_interrupted = True
        logger.exception("WebSocket synthesis error: project=%s", project_id)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            logger.debug("向客户端发送错误消息失败: project=%s", project_id, exc_info=True)
    finally:
        if job_id:
            try:
                if explicit_stop_requested:
                    cancel_job(project_id, job_id)
                elif transport_interrupted or (stop_event.is_set() and not explicit_stop_requested):
                    interrupt_job(project_id, job_id, "WebSocket 合成连接中断")
            except JobStorageError:
                logger.debug("更新持久任务终态失败: job=%s", job_id, exc_info=True)
            if job_registered_active:
                mark_job_active(project_id, job_id, False)
        if batch_lease is not None:
            unregister_running_batch(batch_lease)
        # 确保 stop 监听任务被取消
        if listen_task is not None:
            listen_task.cancel()
        try:
            await websocket.close()
        except Exception:
            logger.debug("关闭 WebSocket 连接失败: project=%s", project_id, exc_info=True)
