"""Project-local professional audio production models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from dubbing_service.app.models.sound_generation import GenerationProvenance, SemanticPlacement


class SoundTrack(str, Enum):
    BGM = "bgm"
    AMBIENCE = "ambience"
    SFX = "sfx"


class AudioAssetOrigin(str, Enum):
    UPLOADED = "uploaded"
    GENERATED = "generated"


class PronunciationEntry(BaseModel):
    id: str
    source: str = Field(..., min_length=1, max_length=120)
    pronunciation: str = Field(..., min_length=1, max_length=240)
    note: Optional[str] = Field(default=None, max_length=300)
    enabled: bool = True


class AssetLicense(BaseModel):
    source: Optional[str] = Field(default=None, max_length=500)
    author: Optional[str] = Field(default=None, max_length=160)
    license_name: Optional[str] = Field(default=None, max_length=160)
    commercial_use: Optional[bool] = None
    proof_note: Optional[str] = Field(default=None, max_length=1000)


class AudioAsset(BaseModel):
    id: str
    track: SoundTrack
    name: str
    audio_file: str
    audio_sha256: str
    duration_seconds: float = Field(..., gt=0)
    tags: List[str] = Field(default_factory=list)
    favorite: bool = False
    license: AssetLicense = Field(default_factory=AssetLicense)
    origin: AudioAssetOrigin = AudioAssetOrigin.UPLOADED
    generation: Optional[GenerationProvenance] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TimelineClip(BaseModel):
    id: str
    track: SoundTrack
    asset_id: str
    start: float = Field(default=0.0, ge=0.0)
    source_offset: float = Field(default=0.0, ge=0.0)
    duration: Optional[float] = Field(default=None, gt=0.0)
    gain_db: float = Field(default=0.0, ge=-60.0, le=18.0)
    fade_in: float = Field(default=0.0, ge=0.0, le=30.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=30.0)
    loop: bool = False
    semantic_placement: Optional[SemanticPlacement] = None


class MixSettings(BaseModel):
    preset: str = Field(default="spoken_word", pattern="^(spoken_word|audio_story|clear_podcast)$")
    dialogue_gain_db: float = Field(default=0.0, ge=-24.0, le=12.0)
    bgm_gain_db: float = Field(default=-16.0, ge=-60.0, le=6.0)
    ambience_gain_db: float = Field(default=-20.0, ge=-60.0, le=6.0)
    sfx_gain_db: float = Field(default=-8.0, ge=-60.0, le=12.0)
    bgm_ducking: bool = True
    ducking_amount_db: float = Field(default=10.0, ge=0.0, le=30.0)
    loudness_lufs: float = Field(default=-16.0, ge=-24.0, le=-12.0)
    true_peak_db: float = Field(default=-1.5, ge=-3.0, le=-0.5)
    fade_in: float = Field(default=0.15, ge=0.0, le=10.0)
    fade_out: float = Field(default=0.5, ge=0.0, le=10.0)
    dialogue_crossfade_seconds: float = Field(default=0.0, ge=0.0, le=2.0)
    output_channels: int = Field(default=2, ge=1, le=2)


class SoundDesign(BaseModel):
    version: int = 1
    clips: List[TimelineClip] = Field(default_factory=list)
    mix: MixSettings = Field(default_factory=MixSettings)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def unique_clip_ids(self) -> "SoundDesign":
        ids = [item.id for item in self.clips]
        if len(ids) != len(set(ids)):
            raise ValueError("时间线片段 ID 不能重复")
        return self


class ReleaseMetadata(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=5000)
    author: Optional[str] = Field(default=None, max_length=160)
    series: Optional[str] = Field(default=None, max_length=200)
    episode: Optional[str] = Field(default=None, max_length=120)
    copyright_notice: Optional[str] = Field(default=None, max_length=500)
    cover_file: Optional[str] = None


class ProductionSettings(BaseModel):
    content_mode: str = Field(default="story", pattern="^(story|information|knowledge|podcast)$")
    pronunciation_dictionary: List[PronunciationEntry] = Field(default_factory=list)
    sound_design: SoundDesign = Field(default_factory=SoundDesign)
    release: ReleaseMetadata = Field(default_factory=ReleaseMetadata)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityIssue(BaseModel):
    code: str
    severity: str = Field(pattern="^(error|warning|info)$")
    message: str
    segment_id: Optional[str] = None
    asset_id: Optional[str] = None
    master_id: Optional[str] = None


class QualityReport(BaseModel):
    project_id: str
    purpose: str = Field(default="preview", pattern="^(preview|master|release)$")
    passed: bool
    issues: List[QualityIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReleasePackage(BaseModel):
    id: str
    project_id: str
    master_id: str
    archive_file: str
    archive_sha256: str
    files: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
