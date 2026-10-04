"""Persistent production-job models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class SynthesisJobStatus(str, Enum):
    QUEUED = "queued"
    AWAITING_REVIEW = "awaiting_review"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_ERRORS = "succeeded_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_JOB_STATUSES = {
    SynthesisJobStatus.SUCCEEDED,
    SynthesisJobStatus.SUCCEEDED_WITH_ERRORS,
    SynthesisJobStatus.FAILED,
    SynthesisJobStatus.CANCELLED,
}


class SynthesisJob(BaseModel):
    """A durable synthesis request independent of the browser lifecycle."""

    id: str
    project_id: str
    job_type: str = Field(pattern="^(batch_synthesize|single_synthesize|voice_lab_discovery|voice_lab_listening)$")
    project_name: Optional[str] = None

    # --- audiobook architecture v1.2: durable agent request identity ---
    request_id: Optional[str] = Field(
        default=None,
        description="调用方可轮询的持久请求 ID（超时/断线后仍可 status(request_id)）",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="相同 request + 相同 key 必须复用既有 Job，避免重复 GPU 工作",
    )
    intent_hash: Optional[str] = Field(
        default=None,
        description="请求意图的规范化哈希（项目/段落集合/后端），用于识别同一意图",
    )
    segment_ids: List[str] = Field(default_factory=list)
    segment_backends: Dict[str, str] = Field(default_factory=dict)
    runtime_bindings: Dict[str, dict] = Field(default_factory=dict)
    status: SynthesisJobStatus = SynthesisJobStatus.QUEUED
    completed_segment_ids: List[str] = Field(default_factory=list)
    failed_segments: Dict[str, str] = Field(default_factory=dict)
    message: Optional[str] = None
    stop_requested: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @model_validator(mode="after")
    def normalize_progress(self) -> "SynthesisJob":
        ordered = []
        seen = set()
        for segment_id in self.segment_ids:
            if segment_id and segment_id not in seen:
                ordered.append(segment_id)
                seen.add(segment_id)
        self.segment_ids = ordered
        allowed = set(ordered)
        self.completed_segment_ids = list(
            dict.fromkeys(
                item for item in self.completed_segment_ids if item in allowed
            )
        )
        self.failed_segments = {
            key: str(value)
            for key, value in self.failed_segments.items()
            if key in allowed
        }
        completed = set(self.completed_segment_ids)
        self.failed_segments = {
            key: value
            for key, value in self.failed_segments.items()
            if key not in completed
        }
        return self

    @computed_field
    @property
    def total(self) -> int:
        return len(self.segment_ids)

    @computed_field
    @property
    def done(self) -> int:
        return len(self.completed_segment_ids)

    @computed_field
    @property
    def failed(self) -> int:
        return len(self.failed_segments)

    @computed_field
    @property
    def remaining_segment_ids(self) -> List[str]:
        finished = set(self.completed_segment_ids) | set(self.failed_segments)
        return [item for item in self.segment_ids if item not in finished]
