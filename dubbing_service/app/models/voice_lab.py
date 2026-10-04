"""Versioned review state owned by the canonical Studio project."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class VoiceLabState(BaseModel):
    schema_version: Literal[1] = 1
    revision: int = 0
    scope: Literal["production", "isolated_listening_test", "regression_fixture"] = "production"
    # Immutable snapshots are appended when reviewed or explicitly designed.
    voices: dict[str, dict[str, Any]] = Field(default_factory=dict)
    preferences: dict[str, dict[str, Any]] = Field(default_factory=dict)
    preference_history: list[dict[str, Any]] = Field(default_factory=list)
    shortlists: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review_position: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    designs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Human casting decisions are separate from legacy library preferences.
    # Additive default: never reinterpret old stars as a scoped approval.
    casting: dict[str, Any] = Field(default_factory=lambda: {
        "schema": "casting-review-v1", "revision": 0, "roles": {},
        "events": {}, "intents": {}, "requests": {}, "productions": {},
        # Durable delivery records for the project-level review events only;
        # per-character review records live on their own round.
        "notifications": {},
    })
    excerpt: dict[str, Any] | None = None
    automation: dict[str, Any] = Field(default_factory=lambda: {
        "paused": True, "status": "paused", "cursor": 0,
        "audio_replenishment": "idle_pending_listening_test",
        "max_candidates": 5, "max_related": 2, "max_cycles": 4,
        "max_gpu_utterances": 0, "job_id": None,
    })
