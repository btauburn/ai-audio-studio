"""Domain models for semantic sound-effect and ambience generation.

The generation layer deliberately stores semantic anchors (segment plus edge)
instead of baking dialogue timestamps into a plan.  Dialogue timing can then be
regenerated without silently moving a sound cue to the wrong sentence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator

from dubbing_service.app.core.path_safety import validate_simple_component_id


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SoundCueTrack(str, Enum):
    SFX = "sfx"
    AMBIENCE = "ambience"


class SoundPlanDensity(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CINEMATIC = "cinematic"


class SoundCueReviewStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class SemanticAnchorEdge(str, Enum):
    START = "start"
    END = "end"


class SemanticAnchor(BaseModel):
    segment_id: str = Field(..., min_length=1, max_length=160)
    edge: SemanticAnchorEdge = SemanticAnchorEdge.START
    offset_seconds: float = Field(default=0.0, ge=-600.0, le=600.0)


class SemanticPlacement(BaseModel):
    start: SemanticAnchor
    end: Optional[SemanticAnchor] = None


class GenerationProvenance(BaseModel):
    """Reproducibility and commercial-use evidence for one generated file."""

    provider: str = Field(..., min_length=1, max_length=120)
    model_id: str = Field(..., min_length=1, max_length=300)
    model_revision: Optional[str] = Field(default=None, max_length=160)
    code_revision: Optional[str] = Field(default=None, max_length=160)
    model_source: Optional[str] = Field(default=None, max_length=500)
    model_license: Optional[str] = Field(default=None, max_length=160)
    commercial_use: Optional[bool] = None
    license_confirmation_id: Optional[str] = Field(default=None, max_length=160)
    license_snapshot: dict[str, JsonValue] = Field(default_factory=dict)
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default=None, max_length=1000)
    seed: Optional[int] = Field(default=None, ge=0)
    steps: Optional[int] = Field(default=None, ge=1, le=1000)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=1000.0)
    requested_duration_seconds: Optional[float] = Field(default=None, gt=0.0, le=3600.0)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=384000)
    audio_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utc_now)


class SoundCandidateStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    ADOPTED = "adopted"


class SoundCandidate(BaseModel):
    id: str = Field(..., min_length=1, max_length=160)
    status: SoundCandidateStatus = SoundCandidateStatus.QUEUED
    audio_file: Optional[str] = Field(default=None, max_length=1000)
    audio_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)
    duration_seconds: Optional[float] = Field(default=None, gt=0.0)
    provenance: GenerationProvenance
    cue_fingerprint: Optional[str] = Field(default=None, min_length=64, max_length=64)
    stale: bool = False
    error: Optional[str] = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("id")
    @classmethod
    def candidate_id_is_one_safe_component(cls, value: str) -> str:
        return validate_simple_component_id(value, label="sound candidate id")

    @model_validator(mode="after")
    def ready_candidate_has_artifact(self) -> "SoundCandidate":
        if self.status in (SoundCandidateStatus.READY, SoundCandidateStatus.ADOPTED):
            if not self.audio_file or not self.audio_sha256 or self.duration_seconds is None:
                raise ValueError("可采用候选必须包含音频文件、SHA-256 和有效时长")
        return self


class SoundCue(BaseModel):
    id: str = Field(..., min_length=1, max_length=160)
    segment_id: str = Field(..., min_length=1, max_length=160)
    track: SoundCueTrack
    description_cn: Optional[str] = Field(default=None, max_length=1000)
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default=None, max_length=1000)
    placement: SemanticPlacement
    requested_duration_seconds: float = Field(..., gt=0.0, le=3600.0)
    playback_duration_seconds: Optional[float] = Field(default=None, gt=0.0, le=86400.0)
    gain_db: float = Field(default=0.0, ge=-60.0, le=18.0)
    fade_in: float = Field(default=0.0, ge=0.0, le=30.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=30.0)
    loop: bool = False
    candidates: list[SoundCandidate] = Field(default_factory=list)
    selected_candidate_id: Optional[str] = None
    adopted_asset_id: Optional[str] = None
    adopted_clip_id: Optional[str] = None
    source_fingerprint: Optional[str] = Field(default=None, min_length=64, max_length=64)
    stale: bool = False
    enabled: bool = True
    review_status: SoundCueReviewStatus = SoundCueReviewStatus.DRAFT

    @model_validator(mode="after")
    def candidate_references_are_valid(self) -> "SoundCue":
        ids = [item.id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("同一音效提示中的候选 ID 不能重复")
        if self.selected_candidate_id and self.selected_candidate_id not in set(ids):
            raise ValueError("选中的候选不属于当前音效提示")
        if self.track == SoundCueTrack.SFX:
            if not 1.0 <= self.requested_duration_seconds <= 15.0:
                raise ValueError("SFX 生成源时长必须为 1–15 秒")
            if self.placement.end is not None:
                raise ValueError("事件 SFX 只使用开始锚点")
            if self.loop:
                raise ValueError("事件 SFX 不循环播放")
        else:
            if not 20.0 <= self.requested_duration_seconds <= 60.0:
                raise ValueError("Ambience 生成源时长必须为 20–60 秒")
            if self.placement.end is None:
                raise ValueError("Ambience 必须包含语义结束锚点")
            if not self.loop:
                raise ValueError("Ambience 必须按语义场景跨度循环播放")
        return self


class SoundPlan(BaseModel):
    version: int = Field(default=1, ge=1)
    id: str = Field(..., min_length=1, max_length=160)
    project_id: str = Field(..., min_length=1, max_length=160)
    source_fingerprint: Optional[str] = Field(default=None, min_length=64, max_length=64)
    density: SoundPlanDensity = SoundPlanDensity.BALANCED
    cues: list[SoundCue] = Field(default_factory=list)
    planner_provider: Optional[str] = Field(default=None, max_length=120)
    planner_model: Optional[str] = Field(default=None, max_length=300)
    stale: bool = False
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def unique_cue_ids(self) -> "SoundPlan":
        ids = [item.id for item in self.cues]
        if len(ids) != len(set(ids)):
            raise ValueError("音效计划中的提示 ID 不能重复")
        return self
