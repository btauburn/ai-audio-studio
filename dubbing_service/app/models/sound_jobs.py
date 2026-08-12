"""Persistent background jobs for Stable Audio sound candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from dubbing_service.app.core.path_safety import validate_simple_component_id


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SoundGenerationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class SoundGenerationTask(BaseModel):
    cue_id: str
    candidate_id: str
    seed: int = Field(..., ge=0, le=2_147_483_647)
    status: str = Field(default="queued", pattern="^(queued|running|succeeded|failed|cancelled)$")
    error: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_is_one_safe_component(cls, value: str) -> str:
        return validate_simple_component_id(value, label="sound candidate id")


class SoundGenerationJob(BaseModel):
    id: str
    project_id: str
    plan_id: str
    status: SoundGenerationJobStatus = SoundGenerationJobStatus.QUEUED
    tasks: list[SoundGenerationTask] = Field(default_factory=list)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    error: Optional[str] = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=utc_now)
