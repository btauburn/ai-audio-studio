"""Immutable reviewed Master metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from dubbing_service.app.models.project import SegmentEditState


class MasterSegmentSnapshot(BaseModel):
    segment_id: str
    index: int
    text: str
    role_id: Optional[str] = None
    take_id: str
    take_audio_sha256: Optional[str] = None
    tts_backend: Optional[str] = None
    qwen3_speaker: Optional[str] = None
    qwen3_language: Optional[str] = None
    voice_license_snapshot: Optional[Dict[str, Any]] = None
    edit_state: Optional[SegmentEditState] = None


class AudioMaster(BaseModel):
    id: str
    project_id: str
    name: str
    audio_file: str
    audio_sha256: str
    dialogue_timeline_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 48000
    bit_depth: int = 24
    input_fingerprint: str
    segments: List[MasterSegmentSnapshot] = Field(default_factory=list)
    sound_design_snapshot: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
