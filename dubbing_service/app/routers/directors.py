"""Explicit analysis and audition controls within the existing Voice Lab API."""
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from dubbing_service.app.audiobook import casting_director
from dubbing_service.app.routers.voice_lab import call

router = APIRouter(prefix="/projects/{project_id}/voice-lab/directors")


class Auditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    role_id: str = Field(min_length=1, max_length=100)
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _review(header):
    if header != "1":
        raise HTTPException(403, "Explicit review action required")


@router.get("")
def directors_view(project_id: str):
    return call(casting_director.view, project_id)


@router.post("/analyze")
def directors_analyze(project_id: str, x_review_request: str | None = Header(default=None)):
    _review(x_review_request)
    return call(casting_director.analyze, project_id)


@router.post("/auditions")
def directors_auditions(project_id: str, payload: Auditions, tasks: BackgroundTasks,
                       x_review_request: str | None = Header(default=None)):
    _review(x_review_request)
    result = call(casting_director.prepare_auditions, project_id, **payload.model_dump())
    from dubbing_service.app.audiobook import casting_review
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result
