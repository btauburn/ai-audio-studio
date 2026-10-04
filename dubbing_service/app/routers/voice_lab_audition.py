"""Explicit short-preview routes; normal Voice Lab owns audio and feedback."""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field

from dubbing_service.app.audiobook import voice_lab_audition as previews
from dubbing_service.app.routers.voice_lab import call

router = APIRouter(prefix="/projects/{project_id}/voice-lab")


class AuditionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    role_id: str = Field(min_length=1, max_length=100)
    candidate_id: str = Field(min_length=1, max_length=100)
    expected_identity_id: str | None = Field(max_length=100)
    request_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


@router.post("/audition")
def audition(project_id: str, payload: AuditionInput, tasks: BackgroundTasks):
    return call(previews.audition, project_id, **payload.model_dump(), schedule=tasks.add_task)


@router.get("/auditions/{request_id}")
def audition_status(project_id: str, request_id: str):
    return call(previews.audition_status, project_id, request_id)
