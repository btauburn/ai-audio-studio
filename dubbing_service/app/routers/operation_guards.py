"""FastAPI dependencies for project-scoped consistency reservations."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException, status

from dubbing_service.app.core.tts_operations import (
    TTSOperationLease,
    release_project_export,
    release_project_mutation,
    release_tts_operation,
    try_acquire_project_export,
    try_acquire_project_mutation,
    try_acquire_tts_operation,
    try_register_active_segment,
    unregister_active_segment,
)


PROJECT_MUTATION_CONFLICT = (
    "当前项目正在合成、导出或删除，不能修改项目内容。请等待当前任务完成后重试。"
)
PROJECT_EXPORT_CONFLICT = (
    "当前项目正在合成、修改、导出或删除，暂时无法导出。请等待当前任务完成后重试。"
)


def project_mutation_guard(project_id: str) -> Generator[TTSOperationLease, None, None]:
    """Hold a mutation reservation for the endpoint's complete execution."""

    lease = try_acquire_project_mutation(project_id)
    if lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=PROJECT_MUTATION_CONFLICT,
        )
    try:
        yield lease
    finally:
        release_project_mutation(lease)


def project_export_guard(project_id: str) -> Generator[TTSOperationLease, None, None]:
    """Hold a stable read/export reservation through response construction."""

    lease = try_acquire_project_export(project_id)
    if lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=PROJECT_EXPORT_CONFLICT,
        )
    try:
        yield lease
    finally:
        release_project_export(lease)


def standalone_segment_synthesis_guard(
    project_id: str,
    segment_id: str,
) -> Generator[TTSOperationLease, None, None]:
    """Own the GPU lane and one segment for a standalone synthesis request."""

    gpu_lease = try_acquire_tts_operation(
        kind="standalone_segment_synthesis",
        project_id=project_id,
    )
    if gpu_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="当前已有合成、字幕或 TTS 管理任务正在执行，请稍候重试。",
        )

    segment_lease = None
    try:
        segment_lease = try_register_active_segment(
            project_id,
            segment_id,
            kind="standalone_segment_synthesis",
        )
        if segment_lease is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=PROJECT_MUTATION_CONFLICT,
            )
        yield segment_lease
    finally:
        if segment_lease is not None:
            unregister_active_segment(segment_lease, segment_id)
        release_tts_operation(gpu_lease)
