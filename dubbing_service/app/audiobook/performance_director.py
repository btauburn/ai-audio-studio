"""Performance Director records and pure deterministic helpers (M1).

M1 is CPU-only and side-effect free.  These functions validate and hash
performance-direction records; they read no audio, write no project data,
load no model, use no GPU and call no network or model API.  Nothing here
changes an existing voice identity, so no existing Take is invalidated.

The module deliberately depends only on the standard library, ``pydantic``,
``fingerprint`` and the casting models.  Importing it must never pull in a
TTS library, the provider package or the renderer.
"""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from dubbing_service.app.audiobook.fingerprint import (
    canonical_json_bytes,
    sha256_bytes,
    text_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.models.character import CastingProfile, CharacterDossier, EvidenceLevel

#: Key under which the Performance Director index is stored in project metadata.
INDEX_KEY = "performance_director"
CONTROL_POLICY = "native-instruct-icl-x1"
MAX_INSTRUCTION_CHARS = 240
MAX_HABIT_CHARS = 160
DELIVERY_DIMENSIONS = ("articulation", "cadence", "confidence", "emotional_range", "formality",
                       "intensity", "speech_patterns", "tempo", "theatricality", "warmth")
INTENTIONS = ("explain", "reassure", "conceal", "challenge", "deflect", "recognize", "concede", "unknown")
BOUNDARY_HINTS = ("quick", "hesitant", "withheld", "beat_after")
DISPOSITION_STATES = ("forward_native", "assembly_timing", "advisory", "unsupported", "hold")
HOLD_CODES = ("attribution_unresolved", "capability_unsupported", "direction_conflict", "plan_stale",
              "measurement_uncertain", "input_ineligible", "runtime_mismatch", "gpu_busy",
              "budget_exhausted", "unknown_outcome", "review_pending", "schema_unknown")

#: Dossier facts that carry a delivery habit, and the horizon field name.
_DOSSIER_DIMENSIONS = ("cadence", "speech_patterns")
_ORIGIN_RANK = {"manual": 3, "dossier": 2, "casting_profile": 1}
_LEVEL_RANK = {"EXPLICIT": 3, "SUPPORTED": 2, "INFERRED": 1, "UNKNOWN": 0}

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_NUMERIC_RATE_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent\b|wpm\b|words per minute\b|x\b|times\b)")
_VOICE_PATTERNS = (
    re.compile(r"(?i)\b(?:sound|sounds|sounding) like\b"),
    re.compile(r"(?i)\bin the (?:voice|style) of\b"),
    re.compile(r"(?i)\bimpersonat"),
    re.compile(r"(?i)\bimitat"),
    re.compile(r"(?i)\bclone\b"),
)

_SHA256_TEXT = r"^[a-f0-9]{64}$"
_ID_TEXT = r"^[^\x00-\x1f\x7f]{1,200}$"
_SHA256 = Annotated[str, StringConstraints(pattern=_SHA256_TEXT)]
_ID = Annotated[str, StringConstraints(pattern=_ID_TEXT)]
_BOUNDED_60 = Annotated[str, StringConstraints(min_length=1, max_length=60)]
_BOUNDED_100 = Annotated[str, StringConstraints(max_length=100)]
_BOUNDED_120 = Annotated[str, StringConstraints(max_length=120)]
_BOUNDED_240 = Annotated[str, StringConstraints(max_length=240)]

Dimension = Literal["articulation", "cadence", "confidence", "emotional_range", "formality",
                    "intensity", "speech_patterns", "tempo", "theatricality", "warmth"]
Intention = Literal["explain", "reassure", "conceal", "challenge", "deflect", "recognize", "concede", "unknown"]
BoundaryHint = Literal["quick", "hesitant", "withheld", "beat_after"]
DispositionState = Literal["forward_native", "assembly_timing", "advisory", "unsupported", "hold"]
DirectionOrigin = Literal["dossier", "casting_profile", "manual"]


class PerformanceDirectorError(ValueError):
    """A refusal with a stable hold code, never a partial record."""

    def __init__(self, code: str, message: str) -> None:
        if code not in HOLD_CODES:
            raise ValueError(f"unknown Performance Director code: {code}")
        self.code = code
        super().__init__(f"{code}: {message}")


class RecordAuthor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: Literal["manual", "deterministic", "model"]
    actor: _BOUNDED_120 | None = None
    event_id: _BOUNDED_120 | None = None


class DeliveryHabit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    value: str
    level: EvidenceLevel
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    origin: DirectionOrigin

    @field_validator("value")
    @classmethod
    def _one_line_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("habit value must not be empty")
        if len(stripped) > MAX_HABIT_CHARS:
            raise ValueError(f"habit value must be at most {MAX_HABIT_CHARS} characters")
        if _CONTROL_RE.search(stripped):
            raise ValueError("habit value must be one line without control characters")
        return stripped


class DeliveryConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_performer_reference: Literal[True] = True
    no_voice_change: Literal[True] = True
    source_text_unchanged: Literal[True] = True


class DeliveryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["delivery-profile-v1"] = "delivery-profile-v1"
    role_id: _ID
    dossier_revision: str | None = None
    casting_profile_revision: str | None = None
    parent_sha256: _SHA256 | None = None
    narrator_preservation: bool = False
    habits: list[DeliveryHabit] = Field(default_factory=list)
    constraints: DeliveryConstraints = Field(default_factory=DeliveryConstraints)
    author: RecordAuthor
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _one_habit_per_dimension(self) -> DeliveryProfile:
        seen: set[str] = set()
        for habit in self.habits:
            if habit.dimension in seen:
                raise ValueError(f"duplicate delivery habit dimension: {habit.dimension}")
            seen.add(habit.dimension)
        return self


class SceneUtterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: _ID
    text_sha256: _SHA256
    role_id: _ID | None = None


class SceneTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance_id: _ID
    intention: Intention = "unknown"
    transition: _BOUNDED_120 | None = None
    boundary_hint_before: BoundaryHint | None = None
    boundary_hint_after: BoundaryHint | None = None
    confidence: Literal["manual", "deterministic", "model"]


class SceneInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["manual", "deterministic", "model"]
    planner_version: str
    model: str | None = None
    prompt_sha256: _SHA256 | None = None
    response_sha256: _SHA256 | None = None
    budget_reservation: str | None = None

    @model_validator(mode="after")
    def _model_mode_requires_provenance(self) -> SceneInterpretation:
        provenance = (self.model, self.prompt_sha256, self.response_sha256, self.budget_reservation)
        if self.mode == "model":
            if any(item is None for item in provenance):
                raise ValueError("model interpretation requires model, prompt, response and budget provenance")
        elif any(item is not None for item in provenance):
            raise ValueError(f"{self.mode} interpretation must not carry model provenance")
        return self


class SceneIntention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scene-intention-v1"] = "scene-intention-v1"
    scene_id: _ID
    chapter_id: _ID
    source_revision_sha256: _SHA256
    utterances: list[SceneUtterance] = Field(min_length=1, max_length=2000)
    attribution_digest: _SHA256
    alias_digest: _SHA256
    turns: list[SceneTurn] = Field(default_factory=list)
    interpretation: SceneInterpretation
    parent_sha256: _SHA256 | None = None
    cache_key: _SHA256
    author: RecordAuthor
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _turns_bind_to_utterances(self) -> SceneIntention:
        ids = [utterance.id for utterance in self.utterances]
        if len(set(ids)) != len(ids):
            raise ValueError("scene utterance ids must be unique")
        known = set(ids)
        seen: set[str] = set()
        for turn in self.turns:
            if turn.utterance_id not in known:
                raise ValueError(f"turn references an unknown utterance: {turn.utterance_id}")
            if turn.utterance_id in seen:
                raise ValueError(f"more than one turn for utterance: {turn.utterance_id}")
            seen.add(turn.utterance_id)
        expected = scene_cache_key(
            source_revision_sha256=self.source_revision_sha256,
            utterances=self.utterances,
            attribution_digest=self.attribution_digest,
            alias_digest=self.alias_digest,
            interpretation=self.interpretation,
        )
        if self.cache_key != expected:
            raise ValueError("scene cache_key does not match the record's own inputs")
        return self


class ControlDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control: _BOUNDED_60
    state: DispositionState
    reason: _BOUNDED_240 = ""


class ResolvedControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control_policy: Literal["native-instruct-icl-x1"] = CONTROL_POLICY
    native_instruction: str | None = None
    seed_salt: int = Field(0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _instruction_is_forwardable(self) -> ResolvedControls:
        if self.native_instruction is not None:
            validate_native_instruction(self.native_instruction)
        return self


class DirectionSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_sha256: _SHA256 | None = None
    scene_plan_sha256: _SHA256 | None = None
    capability_sha256: _SHA256
    resolver_version: str


class UtteranceDirection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["utterance-direction-v1"] = "utterance-direction-v1"
    utterance_id: _ID
    text_sha256: _SHA256
    role_id: _ID
    voice_revision: str | None = None
    requested: dict[str, Any] = Field(default_factory=dict, max_length=20)
    resolved: ResolvedControls = Field(default_factory=ResolvedControls)
    dispositions: list[ControlDisposition] = Field(default_factory=list)
    sources: DirectionSources
    precedence_trace: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _native_instruction_matches_disposition(self) -> UtteranceDirection:
        controls = [disposition.control for disposition in self.dispositions]
        if len(set(controls)) != len(controls):
            raise ValueError("dispositions must name each control at most once")
        forwarded = any(disposition.control == "native_instruction" and disposition.state == "forward_native"
                        for disposition in self.dispositions)
        if (self.resolved.native_instruction is not None) != forwarded:
            raise ValueError("native_instruction must be set exactly when a forward_native disposition exists")
        if any(disposition.state == "hold" for disposition in self.dispositions) and \
                self.resolved.native_instruction is not None:
            raise ValueError("a held direction must not carry a native instruction")
        return self


class DirectionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["direction-set-v1"] = "direction-set-v1"
    project_id: _ID
    chapter_id: _ID | None = None
    directions: list[UtteranceDirection] = Field(default_factory=list, max_length=5000)
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _unique_utterances(self) -> DirectionSet:
        ids = [direction.utterance_id for direction in self.directions]
        if len(set(ids)) != len(ids):
            raise ValueError("direction set utterance ids must be unique")
        return self


class ActivePointers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, _SHA256] = Field(default_factory=dict)
    scene_plans: dict[str, _SHA256] = Field(default_factory=dict)
    direction_set: str | None = None
    boundary_policy: str | None = None


class PerformanceIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["performance-director-index-v1"] = "performance-director-index-v1"
    enabled: bool = False
    revision: int = Field(0, ge=0)
    pilot: _BOUNDED_100 | None = None
    active: ActivePointers = Field(default_factory=ActivePointers)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


def content_sha256(record: BaseModel | Mapping[str, Any]) -> str:
    """Hash a record's content, ignoring its own hash and its creation time."""

    payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
    payload.pop("content_sha256", None)
    payload.pop("created_at", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _check_direction_text(text: str, *, max_chars: int, forbidden_terms: Sequence[str], what: str) -> str:
    """Shared direction-text gate: one line, no rate promise, no borrowed voice."""

    if not isinstance(text, str):
        raise PerformanceDirectorError("direction_conflict", f"{what} must be a string")
    stripped = text.strip()
    if not stripped:
        raise PerformanceDirectorError("direction_conflict", f"{what} must not be empty")
    if len(stripped) > max_chars:
        raise PerformanceDirectorError("direction_conflict",
                                       f"{what} must be at most {max_chars} characters")
    if _CONTROL_RE.search(stripped):
        raise PerformanceDirectorError("direction_conflict",
                                       f"{what} must be one line without control characters")
    if _NUMERIC_RATE_RE.search(stripped):
        raise PerformanceDirectorError("direction_conflict", f"{what} must not promise a numeric rate")
    for pattern in _VOICE_PATTERNS:
        if pattern.search(stripped):
            raise PerformanceDirectorError("direction_conflict",
                                           f"{what} must not reference another performer's voice")
    folded = stripped.casefold()
    for term in forbidden_terms:
        if str(term) and str(term).casefold() in folded:
            raise PerformanceDirectorError("direction_conflict", f"{what} contains a forbidden term")
    return stripped


def validate_native_instruction(text: str, *, forbidden_terms: Sequence[str] = ()) -> str:
    """Return the stripped instruction, or refuse with ``direction_conflict``."""

    return _check_direction_text(text, max_chars=MAX_INSTRUCTION_CHARS,
                                 forbidden_terms=forbidden_terms, what="native instruction")


def direction_component(controls: ResolvedControls | UtteranceDirection) -> str | None:
    """Hash the delivery component of a direction, or ``None`` when it has none."""

    if isinstance(controls, UtteranceDirection):
        resolved = controls.resolved
    elif isinstance(controls, ResolvedControls):
        resolved = controls
    else:
        raise PerformanceDirectorError("direction_conflict",
                                       "direction_component needs ResolvedControls or UtteranceDirection")
    if resolved.native_instruction is None and resolved.seed_salt == 0:
        return None
    return sha256_bytes(canonical_json_bytes({
        "control_policy": resolved.control_policy,
        "native_instruction": resolved.native_instruction,
        "seed_salt": resolved.seed_salt,
    }))


def scene_cache_key(*, source_revision_sha256: str, utterances: Sequence[SceneUtterance],
                    attribution_digest: str, alias_digest: str,
                    interpretation: SceneInterpretation) -> str:
    """Cache identity of a scene plan; deliberately independent of any voice."""

    return sha256_bytes(canonical_json_bytes({
        "source_revision_sha256": source_revision_sha256,
        "utterances": [[utterance.id, utterance.text_sha256, utterance.role_id]
                       for utterance in utterances],
        "attribution_digest": attribution_digest,
        "alias_digest": alias_digest,
        "planner_version": interpretation.planner_version,
        "mode": interpretation.mode,
        "model": interpretation.model,
        "prompt_sha256": interpretation.prompt_sha256,
    }))


def _first_evidence_id(evidence_ids: Sequence[str]) -> str:
    return evidence_ids[0] if evidence_ids else ""


def _candidate_key(candidate: tuple[str, str, str, list[str], str]) -> tuple[int, int, str]:
    _, _, level, evidence_ids, origin = candidate
    return (-_ORIGIN_RANK[origin], -_LEVEL_RANK[level], _first_evidence_id(evidence_ids))


def compile_profile(role_id: str, *, narrator: bool,
                    dossier: CharacterDossier | None = None,
                    casting_profile: CastingProfile | None = None,
                    edits: Sequence[DeliveryHabit] = (),
                    max_chapter: int | None = None,
                    forbidden_terms: Sequence[str] = (),
                    actor: str | None = None, event_id: str | None = None,
                    parent_sha256: str | None = None) -> DeliveryProfile:
    """Compile one delivery profile from casting, dossier and manual edits.

    Inputs are read, never mutated.  Exactly one habit survives per delivery
    dimension: a manual edit outranks a dossier fact, which outranks a casting
    decision, and a stronger evidence level decides within one origin.
    """

    candidates: list[tuple[str, str, str, list[str], str]] = []
    if casting_profile is not None:
        for decision in casting_profile.decisions:
            if decision.dimension not in DELIVERY_DIMENSIONS:
                continue
            if decision.level == "UNKNOWN":
                continue
            value = str(decision.value).strip()
            if not value or value.casefold() == "unspecified":
                continue
            candidates.append((decision.dimension, value, decision.level,
                               list(decision.evidence_ids)[:20], "casting_profile"))
    if dossier is not None:
        for fact in dossier.facts:
            if fact.field not in _DOSSIER_DIMENSIONS:
                continue
            if fact.level == "UNKNOWN":
                continue
            available_after = fact.available_after_chapter
            if max_chapter is not None and available_after is not None and available_after > max_chapter:
                continue
            candidates.append((fact.field, str(fact.value).strip(), fact.level, [fact.id], "dossier"))
    for edit in edits:
        if edit.origin != "manual":
            raise PerformanceDirectorError("direction_conflict",
                                           f"edit for {edit.dimension} must have origin 'manual'")
        candidates.append((edit.dimension, edit.value, edit.level, list(edit.evidence_ids), "manual"))

    best: dict[str, tuple[str, str, str, list[str], str]] = {}
    for candidate in candidates:
        current = best.get(candidate[0])
        if current is None or _candidate_key(candidate) < _candidate_key(current):
            best[candidate[0]] = candidate

    habits: list[DeliveryHabit] = []
    for dimension in sorted(best):
        _, value, level, evidence_ids, origin = best[dimension]
        checked = _check_direction_text(value, max_chars=MAX_HABIT_CHARS,
                                        forbidden_terms=forbidden_terms, what=f"{dimension} habit")
        habits.append(DeliveryHabit(dimension=dimension, value=checked, level=level,
                                    evidence_ids=evidence_ids, origin=origin))

    return DeliveryProfile(
        role_id=role_id,
        dossier_revision=(dossier.revision or None) if dossier is not None else None,
        casting_profile_revision=(casting_profile.revision or None) if casting_profile is not None else None,
        parent_sha256=parent_sha256,
        narrator_preservation=narrator,
        habits=habits,
        author=RecordAuthor(origin="manual" if edits else "deterministic", actor=actor, event_id=event_id),
        created_at=None,
    )


def default_index() -> PerformanceIndex:
    """A disabled, empty index; enabling it is a separate reviewed decision."""

    return PerformanceIndex()


def read_index(metadata: Mapping[str, Any] | None) -> PerformanceIndex | None:
    """Read the index from project metadata; never writes and never guesses."""

    if not metadata or INDEX_KEY not in metadata:
        return None
    raw = metadata[INDEX_KEY]
    if not isinstance(raw, Mapping):
        raise PerformanceDirectorError("schema_unknown",
                                       f"{INDEX_KEY} must be a mapping, found {type(raw).__name__}")
    try:
        return PerformanceIndex.model_validate(dict(raw))
    except ValidationError as exc:
        raise PerformanceDirectorError("schema_unknown",
                                       f"{INDEX_KEY} does not match this schema: {exc.error_count()} error(s)") from exc

# --------------------------------------------------------------------------- M4
#
# M4 adds project-owned immutable direction storage, an expected-revision index
# writer, a pure direction resolver and the one canonical requirement hook.
#
# Nothing below synthesizes, renders, loads a model, uses a GPU, starts a
# service, calls a network or model API, or writes outside the project it was
# given.  Every storage and pipeline owner is imported lazily so that importing
# this module stays pure (see tests/test_performance_contracts.py).
#
# The default runtime profile forwards nothing.  On `faster-base-032` the
# capability record reports the native instruction as
# `available_not_forwarded`, so every native control resolves to `advisory`,
# no instruction reaches a fingerprint, and every existing Take stays current.
# Only the experimental profile with full supported reference conditioning can
# resolve `forward_native`, and M4 neither registers nor launches it.

#: Project-owned immutable stores, all relative to the canonical project root.
PROFILE_DIRECTORY = "performance/profiles"
SCENE_DIRECTORY = "performance/scenes"
DIRECTION_DIRECTORY = "performance/directions"
POLICY_DIRECTORY = "performance/policies"

#: Resolver identity, recorded in every row's `sources`.
RESOLVER_VERSION = "pd-resolver-1"
DIRECTION_POLICY_SCHEMA = "direction-policy-v1"
#: Mirrors the accepted M1 value in ``provider_capabilities``; provider_capabilities
#: itself is not imported here so that a resolution never depends on the provider
#: package being importable.
PERFORMANCE_CAPABILITY_SCHEMA = "performance-capability-v1"
#: The M1 index schema bounds history at 50 entries; the writer keeps the newest.
INDEX_HISTORY_LIMIT = 50
MAX_DELIVERY_PRIORITIES = 3
#: At most this many parsed direction sets live in one process.
DIRECTION_CACHE_LIMIT = 16

_FORBIDDEN_PATH_CHARS = frozenset('/\\:*?"<>|')

#: Bounded antonym pairs.  Two of these in one line is a contradiction, and a
#: contradiction is a hold, never a blended instruction.
_CONTRADICTIONS = (("faster", "slower"), ("quicker", "slower"), ("louder", "quieter"),
                   ("loud", "quiet"), ("shouted", "whispered"), ("shout", "whisper"))

_WordPattern = re.compile(r"[a-z]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refuse(code: str, message: str) -> "PerformanceDirectorError":
    return PerformanceDirectorError(code, message)


def _bounded_text(value: Any, what: str, *, max_length: int = 200) -> str:
    if not isinstance(value, str):
        raise _refuse("direction_conflict", f"{what} must be a string")
    if not value.strip():
        raise _refuse("direction_conflict", f"{what} must not be empty")
    if len(value) > max_length:
        raise _refuse("direction_conflict", f"{what} must be at most {max_length} characters")
    if _CONTROL_RE.search(value):
        raise _refuse("direction_conflict", f"{what} must be one line without control characters")
    return value


def _require_component(value: Any, what: str) -> str:
    """One identifier used as a path component; traversal and separators refuse."""

    text = _bounded_text(value, what)
    if any(char in _FORBIDDEN_PATH_CHARS for char in text):
        raise _refuse("direction_conflict", f"{what} must be a single path component")
    if text.rstrip(". ") != text or text.strip() != text or not text.strip(". "):
        raise _refuse("direction_conflict",
                      f"{what} must not be a dot path, a traversal or a padded name")
    return text


def _require_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise _refuse("direction_conflict", f"{what} must be a lowercase SHA-256")
    return value


def _project_root(project_id: str) -> Path:
    from dubbing_service.app.core.config import safe_project_root

    return Path(safe_project_root(project_id))


def _require_within(root: Path, candidate: Path, what: str) -> Path:
    """Resolve a path inside the canonical project root, refusing symlink escapes.

    The nearest existing ancestor is resolved, so a symlinked ``performance/``
    directory pointing outside the project is refused even before the leaf file
    exists.
    """

    root_resolved = Path(root).resolve()
    probe = Path(candidate)
    missing: list[str] = []
    while not probe.exists() and probe != probe.parent:
        missing.append(probe.name)
        probe = probe.parent
    try:
        anchor = probe.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _refuse("plan_stale", f"{what} is not resolvable") from exc
    resolved = Path(anchor, *reversed(missing)) if missing else anchor
    if not resolved.is_relative_to(root_resolved):
        raise _refuse("plan_stale", f"{what} escapes the canonical project root")
    return resolved


def _record_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2,
                        allow_nan=False) + "\n").encode("utf-8")


def _store_immutable(path: Path, data: bytes) -> bool:
    """Publish complete bytes once.  Never overwrites; keeps failed staging."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise _refuse("direction_conflict", "immutable artifact exists with different bytes")
        return False
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.stage")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise _refuse("direction_conflict",
                              "immutable artifact exists with different bytes")
            staged.unlink(missing_ok=True)
            return False
    except BaseException:
        # The staging file is deliberately kept: an interrupted publish is
        # evidence, and its name can never be mistaken for the artifact.
        raise
    staged.unlink(missing_ok=True)
    return True


def _store_record(project_id: str, parts: Sequence[str], payload: Mapping[str, Any]) -> dict:
    """Publish one immutable record under its own content hash.

    Reuse is verified by content identity, not raw bytes: timestamps are excluded
    from identity, so a second store of the same content is a verified no-op
    rather than a byte conflict.
    """

    if not isinstance(payload, Mapping):
        raise _refuse("input_ineligible", "a stored performance record must be a mapping")
    digest = content_sha256(payload)
    root = _project_root(project_id)
    path = _require_within(root, root.joinpath(*parts, f"{digest}.json"),
                           "performance artifact")
    created = False
    if path.exists() and not path.is_file():
        raise _refuse("direction_conflict", f"performance artifact {path.name} is not a file")
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _refuse("schema_unknown",
                          f"stored artifact {path.name} is unreadable") from exc
        if not isinstance(existing, Mapping) or content_sha256(existing) != digest:
            raise _refuse("direction_conflict",
                          f"stored artifact {path.name} does not match its own content hash")
    else:
        created = _store_immutable(path, _record_bytes(payload))
    return {"sha256": digest, "path": str(path),
            "relative": path.relative_to(root).as_posix(), "created": created}


def _read_record(project_id: str, parts: Sequence[str], expected_sha: Any,
                 model: type[BaseModel] | None, what: str) -> Any:
    """Read one immutable record, verifying its identity before trusting it."""

    sha = _require_sha256(expected_sha, f"{what} pointer")
    root = _project_root(project_id)
    path = _require_within(root, root.joinpath(*parts, f"{sha}.json"), f"{what} artifact")
    if not path.is_file():
        raise _refuse("plan_stale", f"the active {what} artifact is missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse("schema_unknown", f"the active {what} artifact is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise _refuse("schema_unknown", f"the active {what} artifact is not a JSON object")
    if content_sha256(raw) != sha:
        raise _refuse("plan_stale",
                      f"the active {what} artifact does not match its active hash")
    if model is None:
        return dict(raw)
    try:
        return model.model_validate(dict(raw))
    except ValidationError as exc:
        raise _refuse("schema_unknown",
                      f"the active {what} artifact does not match its schema "
                      f"({exc.error_count()} error(s))") from exc


def write_delivery_profile(project_id: str, profile: DeliveryProfile) -> dict:
    """Store one immutable delivery profile under its role directory."""

    if not isinstance(profile, DeliveryProfile):
        raise _refuse("input_ineligible", "a delivery profile record is required")
    _require_component(profile.role_id, "profile role_id")
    return _store_record(project_id, (PROFILE_DIRECTORY, profile.role_id),
                         profile.model_dump(mode="json"))


def write_scene_intention(project_id: str, scene: SceneIntention) -> dict:
    """Store one immutable scene intention plan under its scene directory."""

    if not isinstance(scene, SceneIntention):
        raise _refuse("input_ineligible", "a scene intention record is required")
    _require_component(scene.scene_id, "scene_id")
    return _store_record(project_id, (SCENE_DIRECTORY, scene.scene_id),
                         scene.model_dump(mode="json"))


def write_direction_set(project_id: str, direction_set: DirectionSet) -> dict:
    """Store one immutable direction set.  This is the only file the hot path reads."""

    if not isinstance(direction_set, DirectionSet):
        raise _refuse("input_ineligible", "a direction set record is required")
    return _store_record(project_id, (DIRECTION_DIRECTORY,),
                         direction_set.model_dump(mode="json"))


def write_boundary_policy(project_id: str, policy: Mapping[str, Any]) -> dict:
    """Store one immutable boundary policy so its index pointer can be verified."""

    if not isinstance(policy, Mapping) or not policy:
        raise _refuse("input_ineligible", "a boundary policy needs a non-empty mapping")
    return _store_record(project_id, (POLICY_DIRECTORY,), dict(policy))


def read_delivery_profile(project_id: str, role_id: str, sha256: str) -> DeliveryProfile:
    _require_component(role_id, "profile role_id")
    return _read_record(project_id, (PROFILE_DIRECTORY, role_id), sha256, DeliveryProfile,
                        "delivery profile")


def read_scene_intention(project_id: str, scene_id: str, sha256: str) -> SceneIntention:
    _require_component(scene_id, "scene_id")
    return _read_record(project_id, (SCENE_DIRECTORY, scene_id), sha256, SceneIntention,
                        "scene intention")


def read_direction_set(project_id: str, sha256: str) -> DirectionSet:
    return _read_record(project_id, (DIRECTION_DIRECTORY,), sha256, DirectionSet,
                        "direction set")


def read_boundary_policy(project_id: str, sha256: str) -> dict:
    return _read_record(project_id, (POLICY_DIRECTORY,), sha256, None, "boundary policy")


def _require_present(root: Path, parts: Sequence[str], sha: Any, model: type[BaseModel] | None,
                     what: str) -> None:
    """An index pointer may only name an artifact that exists, parses and matches."""

    digest = _require_sha256(sha, f"{what} pointer")
    path = _require_within(root, root.joinpath(*parts, f"{digest}.json"), f"{what} artifact")
    if not path.is_file():
        raise _refuse("plan_stale", f"the index names a {what} artifact that does not exist")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse("schema_unknown", f"the {what} artifact is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise _refuse("schema_unknown", f"the {what} artifact is not a JSON object")
    if content_sha256(raw) != digest:
        raise _refuse("plan_stale",
                      f"the {what} artifact does not match its own content hash")
    if model is None:
        return
    try:
        model.model_validate(dict(raw))
    except ValidationError as exc:
        raise _refuse("schema_unknown",
                      f"the {what} artifact does not match its schema "
                      f"({exc.error_count()} error(s))") from exc


def _validate_active_pointers(root: Path, active: Mapping[str, Any]) -> None:
    """Every pointer must resolve to a present, well-formed, matching artifact."""

    if not isinstance(active, Mapping):
        raise _refuse("direction_conflict", "active must be a mapping of pointers")
    for role_id, sha in (active.get("profiles") or {}).items():
        _require_component(role_id, "profile role_id")
        _require_present(root, (PROFILE_DIRECTORY, str(role_id)), sha, DeliveryProfile,
                         "delivery profile")
    for scene_id, sha in (active.get("scene_plans") or {}).items():
        _require_component(scene_id, "scene_id")
        _require_present(root, (SCENE_DIRECTORY, str(scene_id)), sha, SceneIntention,
                         "scene intention")
    if active.get("direction_set") is not None:
        _require_present(root, (DIRECTION_DIRECTORY,), active["direction_set"], DirectionSet,
                         "direction set")
    if active.get("boundary_policy") is not None:
        _require_present(root, (POLICY_DIRECTORY,), active["boundary_policy"], None,
                         "boundary policy")


def commit_index(project_id: str, expected_revision: int, change: Callable[[dict], None], *,
                 actor: str, event_id: str) -> dict:
    """Commit exactly one index change under an expected revision.

    The projects-root writer lock and the project metadata fence are both held,
    so two writers cannot both win.  The caller's revision is compared against a
    freshly loaded index, never trusted.  The change callback may not rewrite the
    index schema, its revision or its history, and every pointer it sets is
    verified before anything is saved.  A repeated ``event_id`` whose result
    payload is identical returns the verified prior result without a second
    increment; the same ``event_id`` with a different payload conflicts.  A
    failure leaves the previous pointer and the immutable artifacts intact.
    """

    project_id = _bounded_text(project_id, "project_id")
    actor = _bounded_text(actor, "actor", max_length=120)
    event_id = _bounded_text(event_id, "event_id", max_length=120)
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) \
            or expected_revision < 0:
        raise _refuse("direction_conflict",
                      "expected_revision must be a non-negative integer")
    if not callable(change):
        raise _refuse("direction_conflict", "change must be callable")

    from dubbing_service.app.core.config import get_config
    from dubbing_service.app.storage.process_lock import acquire_projects_root
    from dubbing_service.app.storage.projects import (
        load_project,
        project_meta_lock,
        save_project,
    )

    root = _project_root(project_id)
    lock = acquire_projects_root(Path(get_config().projects_root_abs),
                                 label="performance-director-index")
    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            before = read_index(project.metadata) or default_index()
            before_dict = before.model_dump(mode="json")
            fresh: Any = copy.deepcopy(before_dict)
            change(fresh)
            if isinstance(fresh, Mapping):
                fresh = dict(fresh)
            if not isinstance(fresh, dict):
                raise _refuse("direction_conflict",
                              "the change must leave the index a mapping")
            for field in ("schema_version", "revision", "history"):
                if fresh.get(field) != before_dict.get(field):
                    raise _refuse("direction_conflict",
                                  f"the change may not rewrite the index {field} itself")
            payload = {"enabled": bool(fresh.get("enabled")), "pilot": fresh.get("pilot"),
                       "active": dict(fresh.get("active") or {})}
            payload_sha256 = sha256_bytes(canonical_json_bytes(payload))

            for entry in before.history:
                if not isinstance(entry, Mapping) or entry.get("event_id") != event_id:
                    continue
                if entry.get("payload_sha256") != payload_sha256:
                    raise _refuse("direction_conflict",
                                  f"event {event_id} was already committed with a different "
                                  "payload")
                _validate_active_pointers(root, payload["active"])
                return {**before_dict, "revision": int(entry["revision"]),
                        "enabled": payload["enabled"], "pilot": payload["pilot"],
                        "active": payload["active"]}

            if expected_revision != before.revision:
                raise _refuse("plan_stale",
                              f"the index moved from revision {expected_revision} to "
                              f"{before.revision}")

            fresh["revision"] = before.revision + 1
            fresh["enabled"] = payload["enabled"]
            fresh["pilot"] = payload["pilot"]
            fresh["active"] = payload["active"]
            _validate_active_pointers(root, fresh["active"])
            entry = {"revision": fresh["revision"], "actor": actor, "event_id": event_id,
                     "at": _utc_now(), "previous": dict(before_dict["active"]),
                     "active": dict(payload["active"]), "enabled": payload["enabled"],
                     "pilot": payload["pilot"], "payload_sha256": payload_sha256}
            fresh["history"] = [*before_dict["history"], entry][-INDEX_HISTORY_LIMIT:]
            try:
                committed = PerformanceIndex.model_validate(fresh)
            except ValidationError as exc:
                raise _refuse("direction_conflict",
                              f"the change produced an invalid index "
                              f"({exc.error_count()} error(s))") from exc
            final = committed.model_dump(mode="json")
            project.metadata[INDEX_KEY] = final
            save_project(project)
            return final
    finally:
        lock.release()


# ------------------------------------------------------------------- M4: scenes


def scene_proofs(project: Any, utterance_ids: Sequence[str]) -> dict:
    """The canonical proofs a scene plan must still match to be applicable.

    Recasting is deliberately absent: voice wording is not part of a scene's
    identity, so changing a role's voice keeps every interpretation usable.  A
    corrected attribution changes the digest for the scenes that contain that
    line and for no others.
    """

    from dubbing_service.app.storage.segments import load_segments

    segments = load_segments(project.id)
    by_id = {segment.id: segment for segment in segments}
    ordered = []
    for utterance_id in utterance_ids:
        segment = by_id.get(_bounded_text(utterance_id, "utterance_id"))
        if segment is None:
            raise _refuse("input_ineligible",
                          f"utterance {utterance_id} is not in the source project")
        ordered.append(segment)
    anchors = sorted({(segment.source_anchor.source_revision_id
                       if segment.source_anchor is not None else None)
                      for segment in ordered}, key=lambda item: item or "")
    if len(anchors) != 1:
        raise _refuse("plan_stale",
                      "a scene plan needs exactly one source revision across its utterances")
    if not anchors[0] or not re.fullmatch(r"[a-f0-9]{64}", str(anchors[0])):
        raise _refuse("plan_stale",
                      "a scene plan needs a canonical source revision, and this project has none")
    attribution = [{"id": segment.id,
                    "speaker": (segment.attribution.speaker_id
                                if segment.attribution is not None else None),
                    "source": (segment.attribution.source
                               if segment.attribution is not None else "unknown")}
                   for segment in ordered]
    return {
        "source_revision_sha256": anchors[0] or "",
        "utterances": [SceneUtterance(id=segment.id, text_sha256=text_sha256(segment.text),
                                      role_id=segment.role_id) for segment in ordered],
        "attribution_digest": sha256_bytes(canonical_json_bytes(attribution)),
        "alias_digest": sha256_bytes(canonical_json_bytes(
            _alias_projection(project))),
    }


def _alias_projection(project: Any) -> Any:
    """The reviewed alias data a scene key depends on, read through its owner."""

    casting = getattr(project.voice_lab, "casting", None) or {}
    aliases = casting.get("aliases") if isinstance(casting, Mapping) else None
    aliases = aliases if isinstance(aliases, Mapping) else {}
    canonical = getattr(getattr(project, "character_cast", None), "role_aliases", {}) or {}
    return {"aliases": {str(key): str(value) for key, value in sorted(aliases.items())},
            "canonical_aliases": {str(key): str(value) for key, value in sorted(canonical.items())}}


def _scene_applicable(project: Any, scene: SceneIntention) -> bool:
    """Whether a stored scene plan still matches live canonical state."""

    proofs = scene_proofs(project, [utterance.id for utterance in scene.utterances])
    if proofs["source_revision_sha256"] != scene.source_revision_sha256:
        return False
    if proofs["attribution_digest"] != scene.attribution_digest:
        return False
    if proofs["alias_digest"] != scene.alias_digest:
        return False
    proof_rows = [[utterance.id, utterance.text_sha256, utterance.role_id]
                  for utterance in proofs["utterances"]]
    scene_rows = [[utterance.id, utterance.text_sha256, utterance.role_id]
                  for utterance in scene.utterances]
    return proof_rows == scene_rows


def store_scene_intention(project: Any, scene: SceneIntention) -> dict:
    """Refuse a scene plan whose live source, text, role or attribution moved."""

    if not _scene_applicable(project, scene):
        raise _refuse("plan_stale",
                      "the scene plan does not match live source, text, role or attribution")
    return write_scene_intention(project.id, scene)


def load_scene_intention(project: Any, scene_id: str, sha256: str) -> SceneIntention:
    """Load an active scene plan, refusing stale source, role, alias or text proofs."""

    scene = read_scene_intention(project.id, scene_id, sha256)
    if not _scene_applicable(project, scene):
        raise _refuse("plan_stale",
                      "the stored scene plan no longer matches live source, role, alias or "
                      "text proofs")
    return scene


# ----------------------------------------------------------------- M4: resolver


class DirectionPolicyOverride(BaseModel):
    """One line's human correction, bounded by the same policy limits."""

    model_config = ConfigDict(extra="forbid")

    delivery: list[_BOUNDED_120] = Field(default_factory=list,
                                         max_length=MAX_DELIVERY_PRIORITIES)
    intention: Intention | None = None
    pace_hint: BoundaryHint | None = None
    emotion_label: _BOUNDED_120 | None = None
    intensity: _BOUNDED_120 | None = None
    emphasis: _BOUNDED_120 | None = None
    role_delivery: _BOUNDED_120 | None = None
    numeric_rate: _BOUNDED_120 | None = None
    seed_salt: int = Field(0, ge=0, le=2**31 - 1, strict=True)
    require_forward: bool = False


class DirectionPolicy(BaseModel):
    """The bounded resolution policy.

    ``narrator_forwarding`` is pinned to ``False``: a narrator line is never
    forwarded, so no caller can widen the safeguard through policy.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["direction-policy-v1"] = DIRECTION_POLICY_SCHEMA
    policy_version: str = RESOLVER_VERSION
    max_instruction_chars: int = Field(MAX_INSTRUCTION_CHARS, ge=1, le=MAX_INSTRUCTION_CHARS)
    max_priorities: int = Field(MAX_DELIVERY_PRIORITIES, ge=1, le=MAX_DELIVERY_PRIORITIES)
    forbidden_terms: list[_BOUNDED_120] = Field(default_factory=list, max_length=50)
    narrator_forwarding: Literal[False] = False
    allow_forward_native: bool = True
    line_overrides: dict[str, DirectionPolicyOverride] = Field(default_factory=dict,
                                                               max_length=5000)


def _checked_policy(policy: Any) -> DirectionPolicy:
    if isinstance(policy, DirectionPolicy):
        return policy
    if isinstance(policy, Mapping):
        try:
            return DirectionPolicy.model_validate(dict(policy))
        except ValidationError as exc:
            raise _refuse("direction_conflict",
                          f"the direction policy is invalid ({exc.error_count()} error(s))") from exc
    raise _refuse("direction_conflict", "a direction policy mapping is required")


def _checked_capability(capability: Any) -> dict:
    if not isinstance(capability, Mapping):
        raise _refuse("capability_unsupported", "a capability record mapping is required")
    if capability.get("schema_version") != PERFORMANCE_CAPABILITY_SCHEMA:
        raise _refuse("schema_unknown", "unknown performance capability schema")
    if not isinstance(capability.get("controls"), Mapping):
        raise _refuse("schema_unknown", "the capability record has no controls")
    _require_sha256(capability.get("sha256"), "capability sha256")
    payload = {key: value for key, value in capability.items() if key != "sha256"}
    if sha256_bytes(canonical_json_bytes(payload)) != capability["sha256"]:
        raise _refuse("runtime_mismatch", "capability content does not match its digest")
    return dict(capability)


def _capability_state(capability: Mapping[str, Any], control: str) -> str:
    """The reported state of one control; a missing entry is never optimistic."""

    entry = (capability.get("controls") or {}).get(control)
    if not isinstance(entry, Mapping):
        return "unsupported"
    return str(entry.get("state") or "unsupported")


def _held(reason: str) -> str:
    code = reason.split(":", 1)[0].strip()
    return code if code in HOLD_CODES else "direction_conflict"


def _contradiction(text: str) -> tuple[str, str] | None:
    words = set(_WordPattern.findall(text.casefold()))
    for left, right in _CONTRADICTIONS:
        if left in words and right in words:
            return left, right
    return None


def _compose_priorities(priorities: Sequence[str], policy: DirectionPolicy) -> tuple[str | None, str | None]:
    """Compose the requested priorities into one instruction, or explain a hold.

    A refusal (unsafe wording) raises; a bounded-policy failure holds, and never
    truncates.
    """

    cleaned = []
    for priority in priorities:
        if not isinstance(priority, str) or not priority.strip():
            raise _refuse("direction_conflict", "a delivery priority must be a non-empty string")
        cleaned.append(priority.strip())
    if not cleaned:
        return None, None
    if len(cleaned) > policy.max_priorities:
        return None, (f"direction_conflict: {len(cleaned)} delivery priorities exceed the "
                      f"policy limit of {policy.max_priorities}")
    joined = "; ".join(cleaned)
    if not joined.endswith("."):
        joined += "."
    if len(joined) > policy.max_instruction_chars:
        return None, (f"direction_conflict: the composed instruction is {len(joined)} characters, "
                      f"over the policy limit of {policy.max_instruction_chars}; it is never "
                      "truncated")
    pair = _contradiction(joined)
    if pair is not None:
        return None, (f"direction_conflict: contradictory priorities "
                      f"({pair[0]} and {pair[1]}) are held, never blended")
    # Raises direction_conflict for a performer reference, a voice change, a
    # numeric rate promise or control characters.
    return validate_native_instruction(joined, forbidden_terms=policy.forbidden_terms), None


def _instruction_disposition(capability: Mapping[str, Any], policy: DirectionPolicy, *,
                             narrator_preservation: bool, required: bool,
                             held: str | None) -> tuple[str, str]:
    """One disposition for the requested native instruction.

    A required control that this runtime cannot forward is a hold, before any
    work: "required" must not degrade into a silent advisory.
    """

    if held is not None:
        return "hold", held
    state = _capability_state(capability, "native_instruction")
    can_forward = (state == "forwarded_experimental"
                   and capability.get("runtime_profile") == "faster-base-032-direction-x1"
                   and capability.get("identity_mechanism") == "qwen_reference"
                   and capability.get("conditioning_mode") == "icl" and not narrator_preservation
                   and policy.allow_forward_native)
    if can_forward:
        return "forward_native", "the experimental profile forwards this instruction"
    if required:
        if narrator_preservation:
            return "hold", ("capability_unsupported: narrator preservation forbids forwarding a "
                            "required instruction")
        if not policy.allow_forward_native:
            return "hold", ("capability_unsupported: this policy does not forward native "
                            "instructions")
        return "hold", ("capability_unsupported: a required native instruction is unsupported on "
                        "this runtime")
    if narrator_preservation:
        return "advisory", "narrator preservation forbids forwarding this line"
    if not policy.allow_forward_native:
        return "advisory", "this policy does not forward native instructions"
    if state in ("available_not_forwarded", "available"):
        return "advisory", "this runtime reports the control but does not forward it"
    return "unsupported", "this runtime forwards no native instruction"


def _pace_disposition(capability: Mapping[str, Any], *, required: bool) -> tuple[str, str]:
    state = _capability_state(capability, "pace")
    if state in ("available", "forwarded_experimental"):
        return "assembly_timing", "the project changes pacing after synthesis, in the assembly"
    if state == "unsupported":
        if required:
            return "hold", "capability_unsupported: a required pace control is unsupported here"
        return "unsupported", "no pace control exists on this runtime"
    return "advisory", "pacing is advisory on this runtime"


def _salt_disposition(capability: Mapping[str, Any], *, required: bool) -> tuple[str, str]:
    state = _capability_state(capability, "seed")
    if (state == "forwarded_experimental"
            and capability.get("runtime_profile") == "faster-base-032-direction-x1"
            and capability.get("identity_mechanism") == "qwen_reference"
            and capability.get("conditioning_mode") == "icl"):
        return "forward_native", "this runtime accepts an explicit seed salt"
    if required:
        return "hold", "capability_unsupported: a required seed salt is unsupported on this runtime"
    return "unsupported", "the worker derives the seed from identity and text"


def _legacy_disposition(capability: Mapping[str, Any], control: str) -> tuple[str, str]:
    state = _capability_state(capability, control)
    if state in ("available", "forwarded_experimental"):
        return "advisory", "recorded for the review copy; it is not a native instruction"
    return "unsupported", (f"{control} is not applied by this runtime and never becomes a native "
                           "instruction")


def resolve_directions(profile: DeliveryProfile | None, plan: SceneIntention | None,
                       capability: Mapping[str, Any], *,
                       policy: DirectionPolicy | Mapping[str, Any]) -> list[UtteranceDirection]:
    """Resolve one direction per scene utterance.  Pure; reads no file and calls no model.

    Precedence, highest first: source and voice safeguards, a human line
    correction, a human scene correction, the approved profile, an inferred
    intention, then baseline.  Exactly one disposition is returned per requested
    control, and a contradiction or a required unsupported control is a hold —
    never a blended instruction and never a truncation.
    """

    checked_policy = _checked_policy(policy)
    checked_capability = _checked_capability(capability)
    if plan is None:
        return []
    if profile is not None and not isinstance(profile, DeliveryProfile):
        raise _refuse("direction_conflict", "a delivery profile record is required")

    capability_sha256 = str(checked_capability["sha256"])
    profile_sha256 = content_sha256(profile) if profile is not None else None
    plan_sha256 = content_sha256(plan)
    turns = {turn.utterance_id: turn for turn in plan.turns}

    rows: list[UtteranceDirection] = []
    for utterance in plan.utterances:
        line_profile = profile if profile is not None and profile.role_id == utterance.role_id else None
        narrator = bool(line_profile is not None and line_profile.narrator_preservation)
        override = checked_policy.line_overrides.get(utterance.id)
        turn = turns.get(utterance.id)
        trace: list[str] = ["source_voice_safeguards"]

        intention = "unknown"
        if override is not None and override.intention is not None:
            intention = override.intention
            trace.append("manual_line")
        elif turn is not None and turn.confidence == "manual":
            intention = turn.intention
            trace.append("human_scene")
        elif turn is not None and turn.intention != "unknown":
            intention = turn.intention
            trace.append("inferred_intention")
        else:
            trace.append("baseline")

        priorities: list[str] = []
        if override is not None and override.delivery:
            priorities = list(override.delivery)
            if "manual_line" not in trace:
                trace.append("manual_line")
        elif line_profile is not None and line_profile.habits:
            priorities = [habit.value for habit in line_profile.habits]
            trace.append("approved_profile")
        elif not trace or trace[-1] != "baseline":
            trace.append("baseline")

        pace_hint = None
        if override is not None and override.pace_hint is not None:
            pace_hint = override.pace_hint
            if "manual_line" not in trace:
                trace.append("manual_line")
        elif turn is not None and (turn.boundary_hint_after or turn.boundary_hint_before):
            pace_hint = turn.boundary_hint_after or turn.boundary_hint_before
            if turn.confidence != "manual":
                trace = [label for label in trace if label != "human_scene"]
            if (turn.confidence == "manual" and "human_scene" not in trace):
                trace.append("human_scene")

        required = bool(override is not None and override.require_forward)
        instruction, held = _compose_priorities(priorities, checked_policy)

        dispositions: list[ControlDisposition] = []
        resolved_instruction: str | None = None
        resolved_salt = 0

        if priorities:
            state, reason = _instruction_disposition(
                checked_capability, checked_policy, narrator_preservation=narrator,
                required=required, held=held)
            dispositions.append(ControlDisposition(control="native_instruction", state=state,
                                                   reason=reason[:240]))
            if state == "forward_native":
                resolved_instruction = instruction
            elif state == "hold" and "manual_line" not in trace:
                trace.append("hold")
        requested: dict[str, Any] = {}
        if priorities:
            requested["delivery"] = list(priorities)
        requested["intention"] = intention
        if pace_hint is not None:
            requested["pace_hint"] = pace_hint
            state, reason = _pace_disposition(checked_capability, required=required)
            dispositions.append(ControlDisposition(control="pace_hint", state=state,
                                                   reason=reason[:240]))
        if override is not None:
            for control in ("emotion_label", "intensity", "emphasis", "role_delivery",
                            "numeric_rate"):
                value = getattr(override, control)
                if value is None:
                    continue
                requested[control] = value
                state, reason = _legacy_disposition(checked_capability, control)
                dispositions.append(ControlDisposition(control=control, state=state,
                                                       reason=reason[:240]))
        if override is not None and override.seed_salt:
            requested["seed_salt"] = override.seed_salt
            state, reason = _salt_disposition(checked_capability, required=required)
            if narrator or not checked_policy.allow_forward_native:
                state = "hold" if required else "advisory"
                reason = "capability_unsupported: narrator preservation or policy forbids seed forwarding"
            dispositions.append(ControlDisposition(control="seed_salt", state=state,
                                                   reason=reason[:240]))
            if state == "forward_native":
                resolved_salt = override.seed_salt

        if any(item.state == "hold" for item in dispositions):
            resolved_instruction = None
            resolved_salt = 0
            dispositions = [item.model_copy(update={"state": "hold", "reason": "direction_conflict: another required control is held"})
                            if item.state == "forward_native" else item for item in dispositions]

        rows.append(UtteranceDirection(
            utterance_id=utterance.id,
            text_sha256=utterance.text_sha256,
            role_id=utterance.role_id or "",
            voice_revision=None,
            requested=requested,
            resolved=ResolvedControls(native_instruction=resolved_instruction,
                                      seed_salt=resolved_salt),
            dispositions=dispositions,
            sources=DirectionSources(profile_sha256=profile_sha256 if line_profile is not None else None,
                                     scene_plan_sha256=plan_sha256,
                                     capability_sha256=capability_sha256,
                                     resolver_version=checked_policy.policy_version),
            precedence_trace=trace[:10],
        ))
    return rows


def build_direction_set(project: Any, profile: DeliveryProfile | None, plan: SceneIntention,
                        capability: Mapping[str, Any], *,
                        policy: DirectionPolicy | Mapping[str, Any]) -> DirectionSet:
    """Resolve a scene into one direction set the index can point at."""

    if not _scene_applicable(project, plan):
        raise _refuse("plan_stale", "scene proofs moved before direction resolution")
    rows = resolve_directions(profile, plan, capability, policy=policy)
    for row in rows:
        role = next((item for item in project.roles if item.id == row.role_id), None)
        if role is None:
            raise _refuse("attribution_unresolved", "direction role is absent from canonical state")
        row.voice_revision = voice_revision_for_role(role)
        if role.role_type == "narrator" and direction_component(row) is not None:
            raise _refuse("capability_unsupported", "narrator preservation forbids forwarded controls")
    return DirectionSet(project_id=project.id, chapter_id=plan.chapter_id, directions=rows, created_at=None)


# --------------------------------------------------- M4: active set and hook


#: Parsed direction sets, keyed by (canonical project root, set hash).  The value
#: also carries a byte digest, so a warmed cache still notices a
#: modified or missing file instead of masking it.
_DIRECTION_CACHE: "OrderedDict[tuple[str, str], tuple[str, DirectionSet]]" = OrderedDict()


def clear_direction_cache() -> None:
    """Drop the in-process direction cache.  Clearing the pointer restores the
    previous fingerprints exactly; this is the in-process half of that rollback."""

    _DIRECTION_CACHE.clear()


def _load_direction_set_file(path: Path, sha: str, project_id: str) -> DirectionSet:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse("schema_unknown", "the active direction set is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise _refuse("schema_unknown", "the active direction set is not a JSON object")
    if content_sha256(raw) != sha:
        raise _refuse("plan_stale", "the active direction set does not match its active hash")
    try:
        record = DirectionSet.model_validate(dict(raw))
    except ValidationError as exc:
        raise _refuse("schema_unknown",
                      f"the active direction set does not match its schema "
                      f"({exc.error_count()} error(s))") from exc
    if record.project_id != project_id:
        raise _refuse("plan_stale", "the active direction set belongs to another project")
    return record


def _active_direction_set(project: Any) -> DirectionSet | None:
    """Load the active direction set once per (project, content identity).

    Returns ``None`` — and reads no performance storage at all — when there is no
    index, the index is disabled, or nothing is active.  Anything else that is
    wrong fails closed.
    """

    index = read_index(getattr(project, "metadata", None))
    if index is None or not index.enabled:
        return None
    sha = index.active.direction_set
    if not sha:
        return None
    root = _project_root(project.id)
    path = _require_within(root, root / DIRECTION_DIRECTORY / f"{sha}.json",
                           "active direction set")
    if not path.is_file():
        raise _refuse("plan_stale", "the active direction set file is missing")
    try:
        signature = sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise _refuse("plan_stale", "the active direction set is unreadable") from exc
    key = (str(root), str(sha))
    cached = _DIRECTION_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        _DIRECTION_CACHE.move_to_end(key)
        record = cached[1]
    else:
        record = _load_direction_set_file(path, str(sha), project.id)
        _DIRECTION_CACHE[key] = (signature, record)
        _DIRECTION_CACHE.move_to_end(key)
        while len(_DIRECTION_CACHE) > DIRECTION_CACHE_LIMIT:
            _DIRECTION_CACHE.popitem(last=False)
    return record.model_copy(deep=True)


def _segments_for(project: Any, segments: Sequence[Any] | None) -> Sequence[Any]:
    if segments is not None:
        return segments
    from dubbing_service.app.storage.segments import load_segments

    return load_segments(project.id)


def _live_capability(project: Any, segment: Any) -> dict:
    """The capability record of the runtime this project actually uses."""

    from dubbing_service.app.audiobook.provider_capabilities import performance_capabilities, experimental_direction_capabilities
    from dubbing_service.app.core.tts_backends import DEFAULT_QWEN_BASE_PROFILE

    profile = project.metadata.get("requested_runtime") or DEFAULT_QWEN_BASE_PROFILE
    role = next((item for item in getattr(project, "roles", []) if item.id == segment.role_id),
                None)
    identity = None
    if role is not None and getattr(role, "voice_identity", None) is not None:
        identity = role.voice_identity.model_dump(mode="json")
    owner = experimental_direction_capabilities if str(profile) == "faster-base-032-direction-x1" else performance_capabilities
    return owner(str(profile), identity)


def active_direction_for(project: Any, segment: Any, *,
                         segments: Sequence[Any] | None = None) -> UtteranceDirection | None:
    """The active direction for one utterance, verified against live state.

    A cache hit cannot mask a modified or missing set, and a row that no longer
    matches the live text, role, voice or runtime is refused rather than used.
    """

    direction_set = _active_direction_set(project)
    if direction_set is None:
        return None
    row = next((item for item in direction_set.directions
                if item.utterance_id == segment.id), None)
    if row is None:
        return None

    if row.text_sha256 != text_sha256(segment.text):
        raise _refuse("plan_stale",
                      f"utterance {segment.id} text changed since its direction was resolved")
    if row.role_id != (segment.role_id or ""):
        raise _refuse("plan_stale",
                      f"utterance {segment.id} role changed since its direction was resolved")
    role = next((item for item in getattr(project, "roles", []) if item.id == segment.role_id),
                None)
    if row.voice_revision:
        if role is None or row.voice_revision != voice_revision_for_role(role):
            raise _refuse("plan_stale",
                          f"utterance {segment.id} voice changed since its direction was resolved")

    holds = [item for item in row.dispositions if item.state == "hold"]
    if holds:
        raise _refuse(_held(holds[0].reason),
                      f"utterance {segment.id} is held: {holds[0].reason}")

    if direction_component(row) is not None:
        # Only a row that would really reach the runtime pays for the capability
        # read, and it must still match the runtime that resolved it.
        live = _live_capability(project, segment)
        if row.sources.capability_sha256 != live.get("sha256"):
            raise _refuse("runtime_mismatch",
                          f"utterance {segment.id} was resolved for a different runtime; "
                          "the direction is not applicable here")
        if role is None or role.role_type == "narrator":
            raise _refuse("capability_unsupported", "narrator or unresolved role cannot forward controls")
        if not row.voice_revision:
            raise _refuse("plan_stale", "a forwarded direction needs its resolved voice revision")
        if (live.get("runtime_profile") != "faster-base-032-direction-x1"
                or live.get("conditioning_mode") != "icl"
                or live.get("identity_mechanism") != "qwen_reference"):
            raise _refuse("runtime_mismatch", "only the experimental full-reference runtime forwards controls")
        if row.resolved.native_instruction is not None and _capability_state(live, "native_instruction") != "forwarded_experimental":
            raise _refuse("capability_unsupported", "the live runtime does not forward native instructions")
        if row.resolved.seed_salt and _capability_state(live, "seed") != "forwarded_experimental":
            raise _refuse("capability_unsupported", "the live runtime does not forward seed salt")
    index = read_index(project.metadata)
    if row.sources.scene_plan_sha256 is not None:
        scenes = [scene_id for scene_id, sha in index.active.scene_plans.items()
                  if sha == row.sources.scene_plan_sha256]
        if len(scenes) != 1:
            raise _refuse("plan_stale", "direction scene proof is not active")
        scene = load_scene_intention(project, scenes[0], row.sources.scene_plan_sha256)
        if not any(item.id == segment.id for item in scene.utterances):
            raise _refuse("plan_stale", "direction utterance is absent from its scene proof")
    if row.sources.profile_sha256 is not None:
        if index.active.profiles.get(row.role_id) != row.sources.profile_sha256:
            raise _refuse("plan_stale", "direction delivery profile is not active")
        profile = read_delivery_profile(project.id, row.role_id, row.sources.profile_sha256)
        if profile.role_id != row.role_id:
            raise _refuse("plan_stale", "direction delivery profile belongs to another role")
    return row


def direction_for(project: Any, segment_id: str, *,
                  segments: Sequence[Any] | None = None) -> dict | None:
    """The active direction row for one utterance id, or ``None``.  Fails closed."""

    utterance_id = _bounded_text(segment_id, "segment_id")
    segment = next((item for item in _segments_for(project, segments)
                    if item.id == utterance_id), None)
    if segment is None:
        return None
    row = active_direction_for(project, segment, segments=segments)
    return None if row is None else row.model_dump(mode="json")


def requirement_direction(project: Any, segment: Any) -> str | None:
    """The ``performance_direction`` component for one utterance, or ``None``.

    This is the single hook through which M4 reaches a fingerprint.  It returns
    ``None`` whenever nothing is forwarded, so the requirement is byte-identical
    to the pre-M4 path; it raises when an enabled index is broken or a line is
    held, because an unknown provenance must never be computed silently.
    """

    row = active_direction_for(project, segment, segments=(segment,))
    if row is None:
        return None
    return direction_component(row)
