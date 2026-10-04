"""Versioned character research and casting state owned by Studio project.json.

Evidence is append-only. Interpretations and voice identities are independent
revisions; changing research never changes a locked voice implicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(UTC)


EvidenceLevel = Literal["EXPLICIT", "SUPPORTED", "INFERRED", "UNKNOWN"]
SourceKind = Literal["book", "project", "official", "reference", "adaptation"]


class StoryScope(BaseModel):
    book: str = "current"
    chapter_start: int | None = Field(None, ge=0)
    chapter_end: int | None = Field(None, ge=0)
    story_time: str | None = None


class CharacterFact(BaseModel):
    id: str
    field: str
    value: Any = None
    level: EvidenceLevel = "UNKNOWN"
    confidence: float = Field(0.0, ge=0, le=1)
    source_kind: SourceKind = "book"
    source: str
    source_revision_id: str | None = None
    chapter_id: str | None = None
    location: str | None = None
    excerpt: str | None = None
    scope: StoryScope = Field(default_factory=StoryScope)
    available_after_chapter: int | None = Field(None, ge=0)
    spoiler_safe: bool = False
    researched_at: datetime = Field(default_factory=now)
    extractor: str = "anchored-text-v1"


class CharacterDossier(BaseModel):
    role_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    revision: str = ""
    facts: list[CharacterFact] = Field(default_factory=list)
    conflicts: list[list[str]] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    processed_chunks: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=now)


class CastingDecision(BaseModel):
    dimension: str
    value: str
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    level: EvidenceLevel = "INFERRED"
    weight: float = 1.0


class InspirationMatch(BaseModel):
    palette: str
    character: str
    performer: str | None = None
    mode: Literal["archetype", "direct_adaptation"] = "archetype"
    traits: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    score: float = 0
    source: str = "editorial performance palette"
    original_voice_only: Literal[True] = True


class CastingProfile(BaseModel):
    role_id: str
    revision: str
    dossier_revision: str
    decisions: list[CastingDecision] = Field(default_factory=list)
    design_instruction: str
    inspiration: InspirationMatch | None = None
    importance: Literal["major", "recurring", "minor"] = "minor"
    dialogue_count: int = 0
    contrast_with: dict[str, float] = Field(default_factory=dict)
    review_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


class ScenePerformanceDirection(BaseModel):
    emotion: str | None = None
    intensity: float = Field(0.65, ge=0, le=1)
    volume: float = Field(1, ge=0, le=2)
    pace: float = Field(1, ge=0.5, le=2)
    emphasis: str | None = None
    delivery: str | None = None


class CharacterVoiceIdentity(BaseModel):
    id: str
    role_id: str
    status: Literal["LOCKED", "REVIEW", "FAILED"] = "LOCKED"
    mechanism: Literal["qwen_reference", "builtin", "existing_reference", "fixture"]
    dossier_revision: str
    casting_profile_revision: str
    inspiration: InspirationMatch | None = None
    design_instruction: str
    reference_audio: str | None = None
    reference_audio_sha256: str | None = None
    #: The acoustic conditioning applied to this identity's reference, and what
    #: the conditioning pass actually achieved. Absent on identities whose
    #: reference carries no performer target (imported acquisitions, fixtures).
    acoustics: dict[str, Any] | None = None
    reference_text: str | None = None
    prompt_artifact: str | None = None
    prompt_sha256: str | None = None
    builtin_speaker: str | None = None
    backend: str = "qwen3tts"
    model: str
    model_revision: str = "unknown"
    runtime_version: str = "unknown"
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class AuditionCandidate(BaseModel):
    id: str
    role_id: str
    profile_revision: str
    identity: CharacterVoiceIdentity | None = None
    samples: dict[str, str] = Field(default_factory=dict)
    measurements: dict[str, dict[str, float]] = Field(default_factory=dict)
    score: float = 0
    review_reasons: list[str] = Field(default_factory=list)
    error: str | None = None
    selected: bool = False
    audition_revision: str = ""
    sample_history: list[dict[str, Any]] = Field(default_factory=list)


class CastCollision(BaseModel):
    left: str
    right: str
    similarity: float
    threshold: float
    shared_scene_weight: float = 0
    reason: str
    measurement: str = "acoustic-proxy-v1"


class CharacterCastingPolicy(BaseModel):
    enabled: bool = True
    max_chapter: int | None = Field(None, ge=0)
    allow_spoilers: bool = False
    external_research: bool = False
    inspiration_palette: str | None = None
    adaptation_requested: bool = False
    max_candidates: int = Field(3, ge=1, le=4)
    max_audition_characters: int = Field(8, ge=0, le=50)
    # No GPU work during pure ingest; the normal render preparation casts
    # automatically through its configured provider before dialogue generation.
    audition_mode: Literal["auto", "off"] = "auto"


# Bump when the automatic casting pipeline's observable behaviour changes, so a
# previously completed run is not mistaken for a current one. A bump also makes
# rounds produced by the earlier pipeline stale for characters the human has not
# approved, so a re-submitted run re-casts them instead of re-surfacing candidates
# the current pipeline would no longer propose.
#
# v3: the performer pool grew from 10 seeds to 100 validated references and a
#     book's characters are now spread across it, so every cast a v2 round proposed
#     would be chosen differently today.
AUTOCAST_PIPELINE_VERSION = "autocast-pipeline-v3"


class AutoCastRun(BaseModel):
    """One durable automatic casting run, bound to a source revision.

    The run is the idempotency unit: the same ``key`` can be requested any
    number of times and still produce exactly one effective run.  A new source
    revision or a new pipeline version mints a new key, so genuinely new or
    changed characters are processed while existing human ratings and accepted
    voices are preserved.
    """

    schema_version: Literal["autocast-run-v1"] = "autocast-run-v1"
    key: str
    project_id: str
    source_revision_id: str | None = None
    source_sha256: str | None = None
    pipeline_version: str = AUTOCAST_PIPELINE_VERSION
    request_id: str
    status: Literal[
        "pending", "running", "complete", "partial", "needs_attention", "failed"
    ] = "pending"
    attempts: int = Field(0, ge=0)
    reason: str | None = None
    trigger: str = "source_import"
    #: An explicit whole-pipeline re-run that also re-casts characters holding an
    #: approval. Off for every routine run, so a version bump never overturns a
    #: human decision; the superseded rounds and their ratings stay in history.
    recast_approved: bool = False
    requested_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    required_role_ids: list[str] = Field(default_factory=list)
    queued_role_ids: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


class CharacterCastState(BaseModel):
    schema_version: Literal[1] = 1
    policy: CharacterCastingPolicy = Field(default_factory=CharacterCastingPolicy)
    # The current automatic casting run. Null until a source import requests one.
    autocast: AutoCastRun | None = None
    dossiers: dict[str, CharacterDossier] = Field(default_factory=dict)
    role_aliases: dict[str, str] = Field(default_factory=dict)
    profiles: dict[str, CastingProfile] = Field(default_factory=dict)
    profile_history: list[CastingProfile] = Field(default_factory=list)
    identity_history: list[CharacterVoiceIdentity] = Field(default_factory=list)
    auditions: dict[str, list[AuditionCandidate]] = Field(default_factory=dict)
    collisions: list[CastCollision] = Field(default_factory=list)
    proposed_revisions: dict[str, str] = Field(default_factory=dict)
    external_cache: dict[str, Any] = Field(default_factory=dict)
    review_reasons: list[str] = Field(default_factory=list)
