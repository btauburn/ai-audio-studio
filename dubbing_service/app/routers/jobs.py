"""Persistent synthesis-job API."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dubbing_service.app.models.jobs import SynthesisJob
from dubbing_service.app.routers.operation_guards import project_mutation_guard
from dubbing_service.app.storage.jobs import (
    JobStorageError,
    cancel_job,
    delete_job,
    get_job,
    load_jobs,
    record_job_segment,
    recover_interrupted_jobs,
    resume_job,
    start_job,
    upsert_job,
)
from dubbing_service.app.storage.segments import load_segments

router = APIRouter()


class CreateSynthesisJobRequest(BaseModel):
    id: str
    job_type: str = Field(pattern="^(batch_synthesize|single_synthesize)$")
    project_name: Optional[str] = None
    segment_ids: List[str] = Field(min_length=1)
    segment_backends: Dict[str, str] = Field(default_factory=dict)


class JobSegmentResultRequest(BaseModel):
    success: bool
    error: Optional[str] = None


@router.get("/projects/{project_id}/jobs", response_model=List[SynthesisJob])
async def api_list_synthesis_jobs(project_id: str) -> List[SynthesisJob]:
    try:
        recover_interrupted_jobs(project_id)
        return load_jobs(project_id)
    except JobStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put(
    "/projects/{project_id}/jobs/{job_id}",
    response_model=SynthesisJob,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_create_synthesis_job(
    project_id: str,
    job_id: str,
    payload: CreateSynthesisJobRequest,
) -> SynthesisJob:
    if payload.id != job_id:
        raise HTTPException(status_code=400, detail="任务 ID 不一致")
    segment_ids = list(dict.fromkeys(payload.segment_ids))
    existing_ids = {segment.id for segment in load_segments(project_id)}
    missing = [item for item in segment_ids if item not in existing_ids]
    if missing:
        raise HTTPException(status_code=400, detail=f"任务包含不存在的分段: {missing[:5]}")
    existing = get_job(project_id, job_id)
    if existing is not None:
        return existing
    job = SynthesisJob(
        id=job_id,
        project_id=project_id,
        job_type=payload.job_type,
        project_name=payload.project_name,
        segment_ids=segment_ids,
        segment_backends={
            key: value
            for key, value in payload.segment_backends.items()
            if key in set(segment_ids)
        },
    )
    try:
        return upsert_job(project_id, job)
    except (JobStorageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/jobs/{job_id}/start",
    response_model=SynthesisJob,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_start_synthesis_job(project_id: str, job_id: str) -> SynthesisJob:
    try:
        return start_job(project_id, job_id)
    except JobStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/jobs/{job_id}/segments/{segment_id}/result",
    response_model=SynthesisJob,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_record_synthesis_job_result(
    project_id: str,
    job_id: str,
    segment_id: str,
    payload: JobSegmentResultRequest,
) -> SynthesisJob:
    try:
        job = get_job(project_id, job_id)
        if job is None:
            raise JobStorageError("任务不存在")
        if job.status.value == "queued":
            start_job(project_id, job_id)
        return record_job_segment(
            project_id,
            job_id,
            segment_id,
            success=payload.success,
            error=payload.error,
        )
    except JobStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/jobs/{job_id}/resume",
    response_model=SynthesisJob,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_resume_synthesis_job(project_id: str, job_id: str) -> SynthesisJob:
    try:
        return resume_job(project_id, job_id)
    except JobStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/jobs/{job_id}/cancel",
    response_model=SynthesisJob,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_cancel_synthesis_job(project_id: str, job_id: str) -> SynthesisJob:
    try:
        return cancel_job(project_id, job_id)
    except JobStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete(
    "/projects/{project_id}/jobs/{job_id}",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_synthesis_job(project_id: str, job_id: str) -> dict[str, bool]:
    try:
        deleted = delete_job(project_id, job_id)
    except JobStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": deleted}
