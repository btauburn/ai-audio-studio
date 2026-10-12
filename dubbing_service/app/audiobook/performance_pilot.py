"""Review-only pilot assembly, durable ledger and blind comparison (M3).

M3 builds the pilot a human will later listen to: it allocates an isolated
review-only project, keeps a durable budget ledger, assembles byte-identical
copies of already-approved Takes into a review master plus a listening
derivative, and prepares blind X/Y pairs with an exact-artifact review gate.

It synthesizes nothing.  No TTS, no model, no GPU, no service, no network.  The
only audio work is CPU copying and encoding of Takes that already exist, and the
only project it may write is a pilot project it allocated itself.  Production
projects, source Takes and the private delivery state are read-only inputs.

Eligibility is established through canonical owners — the segment's resolved
attribution, its role, its live synthesis requirement, its voice identity, the
casting binding and the on-disk bytes — never through a caller Boolean.

The authenticated-commit path is deliberately **fail-closed**.  There is no
public way to supply a verdict: verification happens only at the private
``_owner_verification`` boundary, which ships unconfigured, so a live
owner-authenticated review cannot be committed and the improvement gate cannot
open.  See :data:`OWNER_VERIFICATION_LIMIT`.

Every refusal is a :class:`PerformanceDirectorError` carrying one of the M1 hold
codes.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import uuid
import wave
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing_service.app.audiobook import casting_assembly, pipeline, release, takes
from dubbing_service.app.audiobook.fingerprint import (
    canonical_json_bytes,
    file_sha256,
    sha256_bytes,
    text_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.audiobook.performance_boundaries import (
    BOUNDARY_TIMELINE_SCHEMA,
    CLASSIFIER_VERSION,
    SCENE_SEPARATOR,
    compile_timeline,
    measure_take,
)
from dubbing_service.app.audiobook.performance_director import (
    PerformanceDirectorError,
    content_sha256,
)
from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.storage.process_lock import acquire_projects_root
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    create_project,
    list_project_summaries,
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.segments import load_segments, project_segments_lock

PILOT_INDEX_KEY = "performance_pilot"
PILOT_LEDGER_KEY = "performance_pilot_ledger"
PILOT_INPUTS_KEY = "performance_pilot_inputs"
PILOT_KEYS_KEY = "performance_pilot_pair_keys"
PILOT_EDITIONS_KEY = "performance_pilot_editions"

PILOT_INDEX_SCHEMA = "performance-pilot-index-v1"
LEDGER_SCHEMA = "performance-pilot-ledger-v1"
ASSEMBLY_SCHEMA = "performance-review-assembly-v1"
REVIEW_SCHEMA = "performance-review-v1"
PAIR_INDEX_SCHEMA = "performance-blind-index-v1"
PAIR_KEY_SCHEMA = "performance-blind-key-v1"
MAP_SCHEMA = "performance-source-audio-map-v1"
SCOPE_SCHEMA = "performance-pilot-scope-v1"

PILOT_STATES = ("draft", "approved_scope", "reserved", "running", "awaiting_review",
                "reviewed", "blocked", "cancelled", "unknown_outcome")
COUNTERS = ("tts_calls", "model_calls", "assemblies", "retries")
REVIEW_STATUSES = ("draft", "verified", "stale")
PREFERENCES = ("X", "Y", "tie", "inconclusive")
CONFIDENCES = ("low", "medium", "high")
ARMS = ("baseline", "variant")
DEFAULT_PROJECTS_ROOT_NAME = "dubbing_projects"

#: The only scopes a synthetic fixture root may live inside.  A caller label is
#: not permission: the path must prove it is a disposable pytest root.
DISPOSABLE_ROOT_MARKERS = ("pytest-runs", "pytest-of-")

#: Artifacts every assembly must publish, checked as a complete set on reuse.
REQUIRED_ARTIFACTS = ("review-master.wav", "listening.mp3", "source-to-audio-map.json")

OWNER_VERIFICATION_LIMIT = (
    "no in-process reviewer-verification bridge is configured in this milestone.  The only "
    "authenticated identity owner in this repository is the Tailscale Serve boundary in the "
    "private delivery server, which is not provisioned (decision D8).  Verification happens "
    "only at the private _owner_verification boundary, which ships unconfigured, so a live "
    "owner-authenticated review stays pending and the improvement gate stays closed.  No "
    "caller-supplied verdict is accepted."
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f/\\]{1,200}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


class PilotError(PerformanceDirectorError):
    """An M3 refusal.  Same hold codes as M1/M2, distinct for callers."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- validation


def _refuse(message: str, code: str = "input_ineligible") -> PilotError:
    return PilotError(code, message)


def _require_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise _refuse(f"{what} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value) or ".." in value:
        raise _refuse(f"{what} must be 1-200 characters without control characters or path escapes")
    return value


def _require_mapping(value: Any, what: str, code: str = "input_ineligible") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _refuse(f"{what} must be a mapping", code)
    return value


def _require_str(value: Any, what: str, *, max_length: int = 4000, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _refuse(f"{what} must be a string")
    if not allow_empty and not value.strip():
        raise _refuse(f"{what} must not be empty")
    if len(value) > max_length:
        raise _refuse(f"{what} must be at most {max_length} characters")
    return value


def _require_int(value: Any, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _refuse(f"{what} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise _refuse(f"{what} must be a boolean")
    return value


def _require_finite(value: Any, what: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _refuse(f"{what} must be a finite number")
    number = float(value)
    if number < minimum:
        raise _refuse(f"{what} must be at least {minimum}")
    return number


def _require_iso(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        raise _refuse(f"{what} must be an ISO-8601 timestamp")
    try:
        _utc(value)
    except ValueError as exc:
        raise _refuse(f"{what} is not a valid timestamp") from exc
    return value


# --------------------------------------------------------------------------- bytes


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def _stage_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.stage")


def _immutable(path: Path, data: bytes) -> bool:
    """Publish complete bytes once.  Returns True when it created the artifact.

    An existing artifact with identical bytes is a no-op; different bytes are a
    refusal, never an overwrite.  A failed publish leaves its staging file on
    disk as evidence instead of deleting it.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise _refuse("immutable artifact exists with different bytes")
        return False
    staged = _stage_path(path)
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise _refuse("immutable artifact exists with different bytes")
            staged.unlink(missing_ok=True)
            return False
    except BaseException:
        # The staging file is deliberately kept: an interrupted publish is
        # evidence, and it is noncanonical so it can never be mistaken for the
        # artifact itself.
        raise
    staged.unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- PCM


def _pcm_params(path: Path) -> tuple[int, int, int, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise _refuse("source Take is not uncompressed PCM WAV")
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            if handle._data_chunk.chunksize % (channels * width):
                raise _refuse("source PCM payload is not frame aligned")
            if len(handle.readframes(frames)) != frames * channels * width:
                raise _refuse("source PCM payload is truncated")
    except (wave.Error, OSError, EOFError) as exc:
        raise _refuse(f"source Take is not a readable PCM WAV: {exc}") from exc
    if channels < 1 or rate <= 0 or width not in (1, 2, 3, 4) or frames <= 0:
        raise _refuse("unsupported PCM channel count, sample rate, sample width or length")
    return channels, width, rate, frames


def _silence_bytes(width: int, frames: int, channels: int) -> bytes:
    """PCM silence for ``frames`` sample frames across ``channels`` channels.

    Silence is zero for signed widths and ``0x80`` for unsigned 8-bit.  The
    channel count is part of the byte count: omitting it shortens every spacer
    and separator in multi-channel Takes.
    """

    channels = max(1, int(channels))
    if width == 1:
        return b"\x80" * (frames * channels)
    return b"\x00" * (width * frames * channels)


def _resolve_within(root: Path, candidate: Path, what: str) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _refuse(f"{what} is missing or unresolvable") from exc
    root_resolved = root.resolve()
    if resolved.is_symlink() or not resolved.is_relative_to(root_resolved):
        raise _refuse(f"{what} escapes its canonical root")
    if not resolved.is_file():
        raise _refuse(f"{what} is not a file")
    return resolved


def _assemble_timeline_pcm(timeline: Mapping[str, Any], sources: Mapping[str, Path],
                           output: Path) -> dict[str, Any]:
    """Write the review master for one timeline.  Integer frames, no dithering.

    A placement with no offset and no spacer copies the source frames verbatim,
    so a zero-offset/zero-spacer baseline is byte-identical to the render owner's
    lossless concatenation.  Silence is written for spacers and separators.
    """

    if output.exists():
        raise _refuse("assembly target already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    placements = timeline.get("placements")
    if not isinstance(placements, list) or not placements:
        raise _refuse("timeline has no placements")
    expected = None
    written = 0
    with output.open("xb") as raw:
        with wave.open(raw, "wb") as destination:
            for placement in placements:
                kind = placement.get("kind")
                if kind in ("spacer", "scene_separator"):
                    frames = _require_int(placement.get("frames"), f"{kind} frames")
                    if expected is None:
                        raise _refuse("a spacer cannot precede the first speech placement")
                    destination.writeframesraw(_silence_bytes(expected[1], frames, expected[0]))
                    written += frames
                    continue
                if kind != "speech":
                    raise _refuse(f"unsupported placement kind: {kind!r}")
                utterance_id = placement.get("utterance_id")
                source = sources.get(utterance_id)
                if source is None:
                    raise _refuse(f"no source Take supplied for {utterance_id!r}")
                channels, width, rate, _declared = _pcm_params(source)
                params = (channels, width, rate)
                if expected is None:
                    expected = params
                    destination.setnchannels(channels)
                    destination.setsampwidth(width)
                    destination.setframerate(rate)
                elif params != expected:
                    raise _refuse("mixed PCM formats; all Takes must share one sample domain")
                offset = _require_int(placement.get("offset_frames"), "offset_frames")
                start = _require_int(placement.get("source_start_frame"), "source_start_frame")
                end = _require_int(placement.get("source_end_frame"), "source_end_frame")
                if not 0 <= offset <= end - start:
                    raise _refuse("an offset cannot exceed the source frame range")
                with wave.open(str(source), "rb") as handle:
                    available = handle.getnframes()
                    if end > available:
                        raise _refuse("timeline frames exceed the source Take length")
                    handle.setpos(start + offset)
                    remaining = (end - start) - offset
                    while remaining:
                        block = handle.readframes(min(65536, remaining))
                        if not block:
                            raise _refuse("source Take ended before its declared frames")
                        destination.writeframesraw(block)
                        remaining -= len(block) // (width * channels)
                written += (end - start) - offset
    return {"total_frames": written, "params": expected}


def _verify_pcm(path: Path, expected_frames: int, params: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """Full read of a PCM file: header, payload length, exact frame count, format."""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if frames != expected_frames:
        raise _refuse(f"{path.name} decodes to {frames} frames, not {expected_frames}")
    if len(raw) != frames * channels * width:
        raise _refuse(f"{path.name} payload length does not match its declared frames")
    if params is not None and (channels, width, rate) != params:
        raise _refuse(f"{path.name} sample format differs from the assembled master")
    return channels, width, rate


# --------------------------------------------------------------------------- projects root


def _projects_root() -> Path:
    return Path(get_config().projects_root_abs)


def _is_disposable_root(root: Path) -> bool:
    """True only for a root that proves it is a disposable pytest root.

    A caller's ``synthetic_test`` label is not permission.  An alternate live
    projects root has none of these path components, so a synthetic scope there
    is refused; the authorized M3 development scope is ``m3/pytest-runs/<run>``
    and pytest's own basetemp (``pytest-of-*``).
    """

    try:
        parts = root.resolve().parts
    except OSError:
        return False
    return any(part == "pytest-runs" or part.startswith("pytest-of-") for part in parts)


def _assert_isolated(pilot: Any) -> None:
    if getattr(pilot.voice_lab, "scope", None) != "isolated_listening_test":
        raise _refuse("a pilot project must be scoped isolated_listening_test")


@contextmanager
def _root_write_lock(label: str):
    """The cross-process projects-root writer owner, held only for the mutation."""

    lock = acquire_projects_root(_projects_root(), label=label)
    try:
        yield lock
    finally:
        lock.release()


# --------------------------------------------------------------------------- unit A


def _scope_digest(checked: Mapping[str, Any]) -> str:
    """The complete validated canonical scope, including expected input state.

    Revision, authorization and the authorized destination are part of the
    identity: changing any of them under one operation id is a different scope,
    not a reuse of the old pilot.
    """

    canonical = {
        "name": checked["name"],
        "kind": checked["kind"],
        "operation_id": checked["operation_id"],
        "scope": {"source_projects": list(checked["scope"]["source_projects"]),
                  "utterance_ids": list(checked["scope"]["utterance_ids"]),
                  "arms": list(checked["scope"]["arms"])},
        "ceilings": {counter: checked["ceilings"][counter] for counter in COUNTERS}
        | {"deadline": checked["ceilings"]["deadline"]},
        "expected_revision_sha256": checked["expected_revision_sha256"],
        "authorized_destination_root": checked["authorized_destination_root"],
        "authorization": {"kind": checked["authorization"]["kind"],
                          "event_id": checked["authorization"].get("event_id"),
                          "by": checked["authorization"].get("by"),
                          "at": checked["authorization"].get("at")},
    }
    return sha256_bytes(canonical_json_bytes(canonical))


def _validate_pilot_manifest(name: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the scope, input schema and destination before any write."""

    _require_identifier(name, "name")
    manifest = _require_mapping(manifest, "manifest")
    if manifest.get("schema_version") != SCOPE_SCHEMA:
        raise _refuse("pilot scope schema version is unknown", "schema_unknown")
    if manifest.get("kind") != "review_only":
        raise _refuse("only a review_only pilot may be allocated")
    operation_id = _require_identifier(manifest.get("operation_id"), "operation_id")
    scope = _require_mapping(manifest.get("scope"), "scope")
    sources = scope.get("source_projects")
    if not isinstance(sources, list) or not sources:
        raise _refuse("scope.source_projects must be a non-empty list")
    for item in sources:
        _require_identifier(item, "scope.source_projects entry")
    utterance_ids = scope.get("utterance_ids")
    if not isinstance(utterance_ids, list) or not utterance_ids:
        raise _refuse("scope.utterance_ids must be a non-empty list")
    for item in utterance_ids:
        _require_identifier(item, "scope.utterance_ids entry")
    if len(set(utterance_ids)) != len(utterance_ids):
        raise _refuse("scope.utterance_ids must not repeat an utterance")
    arms = scope.get("arms")
    if not isinstance(arms, list) or not arms or any(arm not in ARMS for arm in arms):
        raise _refuse(f"scope.arms must be a non-empty subset of {list(ARMS)}")
    if len(set(arms)) != len(arms):
        raise _refuse("scope.arms must not repeat an arm")
    ceilings = _require_mapping(manifest.get("ceilings"), "ceilings")
    for counter in COUNTERS:
        _require_int(ceilings.get(counter, 0), f"ceilings.{counter}")
    deadline = _require_iso(ceilings.get("deadline"), "ceilings.deadline")
    # An already-expired deadline never allocates a pilot: nothing may be
    # started, reserved or published after it.
    _require_deadline(deadline, "allocation")
    _require_sha256(manifest.get("expected_revision_sha256"), "expected_revision_sha256")
    destination = manifest.get("authorized_destination_root")
    if not isinstance(destination, str) or not destination:
        raise _refuse("authorized_destination_root is required")
    declared = Path(destination)
    if not declared.is_absolute():
        raise _refuse("authorized_destination_root must be absolute")
    if ".." in declared.parts:
        raise _refuse("authorized_destination_root must not contain '..'")
    root = _projects_root()
    if declared.resolve() != root.resolve():
        raise _refuse("authorized_destination_root is not the configured projects root")

    grant = _require_mapping(manifest.get("authorization"), "authorization")
    kind = grant.get("kind")
    if kind not in ("synthetic_test", "live_pilot"):
        raise _refuse("authorization.kind must be synthetic_test or live_pilot")
    if kind == "synthetic_test":
        if not _is_disposable_root(root):
            raise _refuse(
                "a synthetic_test pilot may only be allocated inside a disposable pytest root; "
                "this root does not prove it is one", "input_ineligible")
    else:
        # A live pilot needs a real, owner-derived verdict.  There is no
        # configured bridge in this milestone, so this stays fail-closed rather
        # than trusting a caller string.
        evidence = _owner_verification(grant.get("authentication"))
        if evidence.get("status") != "verified":
            raise _refuse(
                "a live pilot grant requires verified owner authentication: "
                + str(evidence.get("reason") or "unverified"), "review_pending")
    _require_str(grant.get("event_id", ""), "authorization.event_id", allow_empty=True)
    return {
        "name": name,
        "kind": "review_only",
        "operation_id": operation_id,
        "scope": {"source_projects": list(sources), "utterance_ids": list(utterance_ids),
                  "arms": list(arms)},
        "ceilings": {counter: int(ceilings.get(counter, 0)) for counter in COUNTERS}
        | {"deadline": deadline},
        "expected_revision_sha256": manifest["expected_revision_sha256"],
        "authorized_destination_root": str(declared),
        "authorization": {"kind": kind, "event_id": grant.get("event_id"),
                          "by": grant.get("by"), "at": grant.get("at")},
    }


def _allocation_marker(operation_id: str, digest: str) -> str:
    return f"ai-audio-studio:pilot:{operation_id}:{digest[:16]}"


def _allocation_prefix(operation_id: str) -> str:
    return f"ai-audio-studio:pilot:{operation_id}:"


def _find_allocations(operation_id: str) -> list[tuple[str, str]]:
    """(project_id, scope-digest prefix) for every project claiming this operation id."""

    prefix = _allocation_prefix(operation_id)
    matches: list[tuple[str, str]] = []
    for summary in list_project_summaries():
        project_id = summary.get("id")
        if not isinstance(project_id, str):
            continue
        try:
            project = load_project(project_id)
        except (ProjectStorageError, ValueError):
            continue
        description = project.description or ""
        if prefix in description:
            matches.append((project_id, description.split(prefix, 1)[1][:16]))
    return sorted(matches)


def create_pilot(name: str, manifest: dict) -> dict:
    """Allocate (or re-adopt) one isolated review-only pilot project."""

    checked = _validate_pilot_manifest(name, manifest)
    digest = _scope_digest(checked)
    marker = _allocation_marker(checked["operation_id"], digest)

    with _root_write_lock("performance-pilot"):
        claimed = _find_allocations(checked["operation_id"])
        if len(claimed) > 1:
            raise _refuse(
                f"ambiguous interrupted allocation: {len(claimed)} projects claim operation "
                f"{checked['operation_id']}; resolve by hand rather than retrying into a second pilot")
        if claimed:
            project_id, claimed_digest = claimed[0]
            # The marker prefix is a fast reject; the full binding check below is
            # what actually authorises reuse.
            if claimed_digest != digest[:16]:
                raise _refuse(
                    f"operation {checked['operation_id']} was allocated with a different canonical "
                    "scope; a changed scope conflicts rather than allocating a second pilot",
                    "direction_conflict")
            return _adopt_pilot(project_id, checked, digest, marker)

        project = create_project(checked["name"], description=marker)
        try:
            _write_pilot_state(project, checked, digest, marker, state="approved_scope")
        except Exception:
            # The half-written project stays on disk as recoverable evidence.
            raise
        return _pilot_record(load_project(project.id), checked, digest)


def _adopt_pilot(project_id: str, checked: Mapping[str, Any], digest: str, marker: str) -> dict:
    """Adopt an existing allocation only after its complete binding is verified.

    A project with no readable index is *held*: it may be a partially initialized
    allocation, and M3 does not attempt proof-preserving recovery of one, so it
    is never adopted and never replaced by a second pilot.  That limit is real
    and reported rather than papered over.
    """

    project = load_project(project_id)
    _assert_isolated(project)
    index = project.metadata.get(PILOT_INDEX_KEY)
    if not isinstance(index, Mapping):
        raise _refuse(
            "a matching allocation exists but has no readable pilot binding; it is held rather "
            "than adopted, and a second pilot is never created", "plan_stale")
    expected = {
        "operation_id": checked["operation_id"],
        "scope_sha256": digest,
        "kind": "review_only",
        "expected_revision_sha256": checked["expected_revision_sha256"],
        "authorized_destination_root": checked["authorized_destination_root"],
        "arms": list(checked["scope"]["arms"]),
        "source_projects": list(checked["scope"]["source_projects"]),
        "utterance_ids": list(checked["scope"]["utterance_ids"]),
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise _refuse(
                f"the existing allocation's {key} does not match this scope; a changed scope is a "
                "different pilot", "direction_conflict")
    if marker not in (project.description or ""):
        raise _refuse("the allocation marker is missing from the adopted project", "plan_stale")
    return _pilot_record(project, checked, digest)


def _write_pilot_state(project: Any, checked: Mapping[str, Any], digest: str,
                       marker: str, *, state: str) -> None:
    project.voice_lab.scope = "isolated_listening_test"
    project.description = marker
    project.metadata[PILOT_INDEX_KEY] = {
        "schema_version": PILOT_INDEX_SCHEMA,
        "pilot_id": project.id,
        "operation_id": checked["operation_id"],
        "scope_sha256": digest,
        "kind": "review_only",
        "arms": checked["scope"]["arms"],
        "source_projects": checked["scope"]["source_projects"],
        "utterance_ids": checked["scope"]["utterance_ids"],
        "authorized_destination_root": checked["authorized_destination_root"],
        "authorization": checked["authorization"],
        "expected_revision_sha256": checked["expected_revision_sha256"],
        "state": state,
        "production_enablement": False,
        "human_review": "pending",
        "created_at": _now(),
    }
    project.metadata[PILOT_LEDGER_KEY] = {
        "schema_version": LEDGER_SCHEMA,
        "pilot_id": project.id,
        "ceilings": checked["ceilings"],
        "spend": {counter: 0 for counter in COUNTERS},
        "operations": {},
        "events": [],
        "updated_at": _now(),
    }
    save_project(project)


def _pilot_record(project: Any, checked: Mapping[str, Any], digest: str) -> dict:
    return {
        "pilot_id": project.id,
        "operation_id": checked["operation_id"],
        "scope_sha256": digest,
        "kind": "review_only",
        "scope": project.voice_lab.scope,
        "state": project.metadata[PILOT_INDEX_KEY]["state"],
        "created": False,
    }


def read_pilot(pilot_project_id: str) -> dict:
    """The pilot index plus its ledger.  Read-only."""

    project = load_project(_require_identifier(pilot_project_id, "pilot_project_id"))
    _assert_isolated(project)
    index = project.metadata.get(PILOT_INDEX_KEY)
    if not isinstance(index, Mapping):
        raise _refuse("this project is not a performance pilot")
    ledger = project.metadata.get(PILOT_LEDGER_KEY)
    if not isinstance(ledger, Mapping):
        raise _refuse("the pilot has no ledger", "schema_unknown")
    return {"index": dict(index), "ledger": dict(ledger)}


def _assert_pilot_destination(project: Any, index: Mapping[str, Any]) -> Path:
    directory = safe_project_root(project.id)
    declared = index.get("authorized_destination_root")
    if not isinstance(declared, str) or Path(declared).resolve() != _projects_root().resolve():
        raise _refuse("the pilot's destination root no longer matches the configured root",
                      "plan_stale")
    return directory


# --------------------------------------------------------------------------- unit B


def _require_deadline(deadline: str, what: str) -> str:
    """The original deadline is fixed at allocation and is never extended."""

    if _utc(deadline) <= datetime.now(timezone.utc):
        raise _refuse(f"{what} deadline {deadline} has already passed", "budget_exhausted")
    return deadline


def reserve_operation(pilot_project_id: str, operation_id: str, *,
                      assemblies: int = 1, deadline: str | None = None) -> dict:
    """Durably reserve budget for one operation before any work starts.

    The reservation is persisted under the projects-root writer lock and the
    project's metadata fence.  A repeated identical reservation returns its
    stored receipt; a changed payload conflicts.  A caller-supplied deadline may
    bring work forward but can never extend the pilot's original ceiling.
    """

    operation_id = _require_identifier(operation_id, "operation_id")
    assemblies = _require_int(assemblies, "assemblies")
    if deadline is not None:
        _require_iso(deadline, "deadline")

    with _root_write_lock("performance-pilot-reserve"):
        with project_meta_lock(pilot_project_id):
            project = load_project(pilot_project_id)
            _assert_isolated(project)
            index = project.metadata.get(PILOT_INDEX_KEY)
            ledger = project.metadata.get(PILOT_LEDGER_KEY)
            if not isinstance(index, Mapping) or not isinstance(ledger, Mapping):
                raise _refuse("this project is not a performance pilot")
            _assert_pilot_destination(project, index)
            original_deadline = ledger["ceilings"]["deadline"]
            effective = deadline or original_deadline
            if _utc(effective) > _utc(original_deadline):
                raise _refuse(
                    "a caller deadline cannot extend the pilot's original deadline", "budget_exhausted")
            _require_deadline(effective, "operation")
            operations = dict(ledger.get("operations") or {})
            payload = {"assemblies": assemblies, "deadline": effective}
            previous = operations.get(operation_id)
            if previous is not None:
                if previous.get("payload") != payload:
                    raise _refuse(
                        f"operation {operation_id} was reserved with a different payload; a changed "
                        "payload conflicts rather than re-using the receipt", "direction_conflict")
                return {"reservation": dict(previous), "reused": True}

            ceilings = dict(ledger["ceilings"])
            spend = dict(ledger["spend"])
            if assemblies > ceilings["assemblies"] - spend["assemblies"]:
                raise _refuse("the assembly ceiling is exhausted", "budget_exhausted")
            receipt = {
                "operation_id": operation_id,
                "payload": payload,
                "original_deadline": original_deadline,
                "reserved_at": _now(),
                "state": "reserved",
                "reserved": {"assemblies": assemblies},
                "spent": {counter: 0 for counter in COUNTERS},
                "outputs": [],
                "outcome": None,
            }
            operations[operation_id] = receipt
            # Persist the reservation before any work.
            ledger = dict(ledger)
            ledger["operations"] = operations
            ledger["spend"] = {**spend, "assemblies": spend["assemblies"] + assemblies}
            ledger["updated_at"] = _now()
            ledger["events"] = list(ledger.get("events") or []) + [
                {"kind": "reserved", "operation_id": operation_id, "at": ledger["updated_at"],
                 "assemblies": assemblies}]
            project.metadata[PILOT_LEDGER_KEY] = ledger
            project.metadata[PILOT_INDEX_KEY] = {**dict(index), "state": "reserved"}
            save_project(project)
            return {"reservation": dict(receipt), "reused": False}


def _complete_operation_locked(pilot_project_id: str, operation_id: str, *, outcome: str,
                               spent: Mapping[str, int] | None = None,
                               output: Mapping[str, Any] | None = None) -> dict:
    """Record an outcome and/or claim one produced output.  Caller holds the locks."""

    if outcome not in (None, "completed", "failed", "cancelled", "unknown_outcome"):
        raise _refuse("unsupported operation outcome")
    project = load_project(pilot_project_id)
    ledger = dict(project.metadata.get(PILOT_LEDGER_KEY) or {})
    operations = dict(ledger.get("operations") or {})
    receipt = operations.get(operation_id)
    if receipt is None:
        raise _refuse("unknown operation; reserve before completing")
    receipt = dict(receipt)

    if output is not None:
        identity = {"arm": output["arm"], "timeline_sha256": output["timeline_sha256"]}
        if identity not in (receipt.get("outputs") or []):
            if len(receipt.get("outputs") or []) + 1 > (receipt.get("reserved") or {}).get("assemblies", 0):
                raise _refuse(
                    "this reservation cannot fund another assembly; one reservation covers only the "
                    "assemblies it reserved", "budget_exhausted")
            receipt["outputs"] = list(receipt.get("outputs") or []) + [identity]

    if outcome is not None:
        if receipt.get("outcome") is not None:
            if receipt["outcome"] != outcome:
                raise _refuse("the operation already has a different terminal outcome",
                              "direction_conflict")
        else:
            receipt["outcome"] = outcome
            receipt["state"] = outcome
            receipt["completed_at"] = _now()
            # A cancelled, failed or unknown outcome still consumed its
            # reservation: budget is never released by an operation ending.
            reserved = dict(receipt.get("reserved") or {})
            receipt["spent"] = {
                counter: max(int((spent or {}).get(counter, 0)), int(reserved.get(counter, 0)))
                for counter in COUNTERS}

    operations[operation_id] = receipt
    spend = dict(ledger["spend"])
    if outcome is not None:
        # Spent counters are recorded once, at the terminal outcome, and never
        # fall below what the reservation already consumed.
        for counter, value in (receipt.get("spent") or {}).items():
            spend[counter] = max(int(spend.get(counter, 0)), int(value))
    ledger["operations"] = operations
    ledger["spend"] = spend
    ledger["updated_at"] = _now()
    if outcome is not None:
        ledger["events"] = list(ledger.get("events") or []) + [
            {"kind": outcome, "operation_id": operation_id, "at": ledger["updated_at"]}]
    project.metadata[PILOT_LEDGER_KEY] = ledger
    index = dict(project.metadata.get(PILOT_INDEX_KEY) or {})
    if outcome is not None:
        project.metadata[PILOT_INDEX_KEY] = {
            **index, "state": "awaiting_review" if outcome == "completed" else "blocked"}
    save_project(project)
    return receipt


def complete_operation(pilot_project_id: str, operation_id: str, *, outcome: str,
                       spent: Mapping[str, int] | None = None) -> dict:
    """Public form of the terminal-outcome writer; spent budget is never returned."""

    with _root_write_lock("performance-pilot-complete"):
        with project_meta_lock(pilot_project_id):
            return _complete_operation_locked(pilot_project_id, operation_id, outcome=outcome,
                                              spent=spent)


def assert_operation_open(pilot_project_id: str, operation_id: str) -> dict:
    """A late or cancelled operation may never promote its result."""

    state = read_pilot(pilot_project_id)
    receipt = (state["ledger"].get("operations") or {}).get(operation_id)
    if receipt is None:
        raise _refuse("unknown operation")
    if receipt.get("outcome") not in (None, "completed"):
        raise _refuse(f"operation outcome is {receipt['outcome']!r}; a late result never promotes",
                      "plan_stale")
    return dict(receipt)


# --------------------------------------------------------------------------- unit C


def _timeline_inputs(project: Any) -> dict:
    inputs = project.metadata.get(PILOT_INPUTS_KEY)
    if not isinstance(inputs, Mapping):
        raise _refuse("the pilot has no canonical timeline inputs to recompile from", "plan_stale")
    return dict(inputs)


def _scope_rows(index: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[Any]:
    """The manifest's ordered utterance scope, proved against the stored rows.

    Replacing the stored inputs can never widen, narrow or reorder the approved
    scope: the rows must describe exactly the manifest's utterances, in order,
    from the manifest's source projects.
    """

    rows = inputs.get("rows")
    if not isinstance(rows, list) or not rows:
        raise _refuse("timeline inputs need a non-empty rows list")
    approved = list(index.get("utterance_ids") or [])
    projects = set(index.get("source_projects") or [])
    row_ids = [row.get("utterance_id") if isinstance(row, Mapping) else None for row in rows]
    if row_ids != approved:
        raise _refuse(
            "the timeline rows must be exactly the approved utterances in the approved order; "
            f"extra, missing or reordered rows are refused (approved {approved!r}, rows {row_ids!r})")
    for row in rows:
        if row.get("project_id") not in projects:
            raise _refuse(f"row {row.get('utterance_id')!r} is outside the approved source projects")
    return rows


def store_timeline_inputs(pilot_project_id: str, inputs: Mapping[str, Any]) -> dict:
    """Persist the canonical M2 inputs a later assembly recompiles from."""

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    policy = _require_mapping(inputs.get("policy"), "policy", "schema_unknown")
    hints = _require_mapping(inputs.get("hints"), "hints", "schema_unknown")
    allow = _require_bool(inputs.get("allow_lead_offset"), "allow_lead_offset")
    candidate = {"policy": dict(policy), "hints": dict(hints), "allow_lead_offset": allow,
                 "rows": [dict(row) for row in inputs.get("rows") or []]}
    with _root_write_lock("performance-pilot-inputs"):
        with project_meta_lock(pilot_project_id):
            project = load_project(pilot_project_id)
            _assert_isolated(project)
            index = project.metadata.get(PILOT_INDEX_KEY)
            ledger = project.metadata.get(PILOT_LEDGER_KEY)
            if not isinstance(index, Mapping) or not isinstance(ledger, Mapping):
                raise _refuse("this project is not a performance pilot")
            _assert_pilot_destination(project, index)
            # The rows may describe more than the approved scope here; the
            # manifest's scope is enforced at the acceptance boundary
            # (``assemble_review``), which is what actually publishes audio.
            # What this guard prevents is replacing the inputs of an operation
            # that has already completed, which would turn the reservation into
            # a different operation.
            if any(receipt.get("outcome") == "completed"
                   for receipt in (ledger.get("operations") or {}).values()):
                previous = project.metadata.get(PILOT_INPUTS_KEY)
                if isinstance(previous, Mapping):
                    stored = {key: value for key, value in previous.items()
                              if key != "stored_at"}
                    if content_sha256(stored) != content_sha256(candidate):
                        raise _refuse(
                            "a completed operation's canonical inputs are immutable; replacing them "
                            "would turn the reservation into a different operation", "plan_stale")
            project.metadata[PILOT_INPUTS_KEY] = {**candidate, "stored_at": _now()}
            save_project(project)
    return {"pilot_id": pilot_project_id, "rows": len(candidate["rows"]),
            "allow_lead_offset": allow, "stored": True}


def _recompile(timeline: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict:
    """Recompile with the accepted M2 compiler; the supplied output is not trusted."""

    compiled = compile_timeline(inputs["rows"], policy=inputs["policy"], hints=inputs["hints"],
                                allow_lead_offset=inputs["allow_lead_offset"])
    supplied = {key: timeline.get(key) for key in
                ("schema_version", "sample_rate", "sample_domain", "policy_sha256",
                 "classifier_version", "assembler_version", "allow_lead_offset", "placements",
                 "boundaries", "total_frames")}
    rebuilt = {key: compiled.get(key) for key in supplied}
    if content_sha256(supplied) != content_sha256(rebuilt):
        raise _refuse(
            "the supplied timeline does not match a recompilation from the pilot's own canonical "
            "inputs; mutated placements are refused even when timeline_sha256 is unchanged",
            "plan_stale")
    return compiled


def _eligible_source(project: Any, segments: Sequence[Any], utterance_id: str, *,
                     expected_audio: str) -> dict:
    """Establish eligibility through canonical owners, never a caller DTO.

    Mirrors the chapter-export snapshot rule: resolved attribution, matching
    role, exact source and spoken-text revisions, a current synthesis
    requirement, a stable locked voice identity, natural unedited speed and
    unchanged bytes.
    """

    segment = next((item for item in segments if item.id == utterance_id), None)
    if segment is None:
        raise _refuse(f"utterance {utterance_id} is not in the source project")
    if not release.is_live(segment):
        raise _refuse(f"utterance {utterance_id} is retired")
    if not segment.text.strip():
        raise _refuse(f"utterance {utterance_id} has no spoken text")
    role = next((item for item in project.roles if item.id == segment.role_id), None)
    if role is None:
        raise _refuse(
            f"utterance {utterance_id} has no canonical role; a pilot cannot assemble unattributed "
            "audio", "attribution_unresolved")
    attribution = segment.attribution
    if (attribution is None or attribution.source == "unknown"
            or attribution.speaker_id != role.id):
        raise _refuse(
            f"utterance {utterance_id} has no resolved canonical attribution matching its role",
            "attribution_unresolved")
    anchor = segment.source_anchor
    if anchor is None or not anchor.source_revision_id:
        raise _refuse(f"utterance {utterance_id} has no source anchor", "attribution_unresolved")
    take = takes.selected_take(segment)
    if take is None:
        raise _refuse(f"utterance {utterance_id} has no selected Take")
    if take.source_take_id is not None:
        raise _refuse(f"utterance {utterance_id} selected Take is a derived Take")
    if take.stale:
        raise _refuse(f"utterance {utterance_id} selected Take is stale", "plan_stale")
    report = take.qa_report
    if report is None:
        raise _refuse(f"utterance {utterance_id} has no QA report", "measurement_uncertain")
    if not report.matches(take.id, take.audio_sha256):
        raise _refuse(f"utterance {utterance_id} QA report is bound to different bytes",
                      "measurement_uncertain")
    if report.blocking_failures():
        raise _refuse(f"utterance {utterance_id} has blocking QA failures")
    if not report.passed or not report.executed_check_ids():
        raise _refuse(f"utterance {utterance_id} lacks executed passing QA")
    try:
        casting_binding = casting_assembly.current_frozen_binding(project, [segment])
    except casting_assembly.CastingAssemblyError as exc:
        raise _refuse(str(exc), "plan_stale") from exc
    if not take.audio_sha256 or take.audio_sha256 != expected_audio:
        raise _refuse(f"utterance {utterance_id} Take bytes differ from the expectation", "plan_stale")
    if take.role_id != role.id:
        raise _refuse(f"utterance {utterance_id} Take belongs to a different role", "plan_stale")
    if take.source_text_hash != text_sha256(segment.text):
        raise _refuse(
            f"utterance {utterance_id} Take does not match the exact canonical source text",
            "plan_stale")
    if take.spoken_text_revision_id != segment.spoken_text_revision_id:
        raise _refuse(f"utterance {utterance_id} Take does not match the spoken-text revision",
                      "plan_stale")
    if (segment.playback_speed != 1.0 or segment.edit_state is not None
            or take.playback_speed != 1.0 or take.baked_speed != 1.0
            or take.edit_state is not None):
        raise _refuse(f"utterance {utterance_id} is not natural-speed complete unedited audio")
    requirement = pipeline._requirement_for_take(project, segment, take)
    if requirement is None or take.input_fingerprint != requirement.fingerprint():
        raise _refuse(
            f"utterance {utterance_id} selected Take synthesis requirement is stale or unknown",
            "plan_stale")
    recorded_components = take.fingerprint_components
    if not isinstance(recorded_components, Mapping) or not recorded_components:
        raise _refuse(
            f"utterance {utterance_id} selected Take records no synthesis-requirement identity",
            "plan_stale")
    if content_sha256(dict(recorded_components)) != content_sha256(requirement.components()):
        raise _refuse(
            f"utterance {utterance_id} recorded requirement identity no longer matches the current "
            "requirement", "plan_stale")
    version = voice_revision_for_role(role)
    if role.voice_identity is not None:
        identity = role.voice_identity
        if identity.mechanism == "qwen_reference":
            if identity.status != "LOCKED":
                raise _refuse(f"utterance {utterance_id} cast voice identity is no longer locked",
                              "plan_stale")
            if version is None or take.input_fingerprint is None:
                raise _refuse(f"utterance {utterance_id} has no verifiable voice revision",
                              "plan_stale")
            role_voice = voice_revision_for_role(role)
            if (requirement.voice_revision or "") != role_voice:
                raise _refuse(
                    f"utterance {utterance_id} Take conditioning no longer matches its cast voice "
                    "version", "plan_stale")
            from .voice_lab_render import _identity_for_take, ListeningRenderError
            try:
                identity_id, recorded_version, conditioning = _identity_for_take(take)
            except ListeningRenderError as exc:
                raise _refuse(str(exc), "plan_stale") from exc
            if (identity_id != identity.id or recorded_version != role_voice
                    or conditioning != identity.reference_audio_sha256):
                raise _refuse("Take generation conditioning does not match its current cast", "plan_stale")
        elif identity.status != "LOCKED":
            raise _refuse(f"utterance {utterance_id} cast voice identity is not locked", "plan_stale")
    else:
        recorded = take.fingerprint_components or {}
        if recorded.get("voice_revision") != version or not recorded.get("builtin_speaker"):
            raise _refuse(
                f"utterance {utterance_id} Take lacks a verified stable final provider voice identity",
                "plan_stale")
    audio_file = take.audio_file
    if not audio_file:
        raise _refuse(f"utterance {utterance_id} Take has no audio file")
    root = safe_project_root(project.id)
    revision = next((r for r in project.source_revisions if r.id == anchor.source_revision_id), None)
    if revision is None or not revision.snapshot_file:
        raise _refuse("source anchor has no canonical immutable revision", "plan_stale")
    snapshot = _resolve_within(root, root / revision.snapshot_file, "source revision snapshot")
    if file_sha256(snapshot) != revision.sha256 or snapshot.stat().st_size != revision.byte_length:
        raise _refuse("source revision snapshot does not match canonical evidence", "plan_stale")
    bound = (take.generation_metadata or {}).get("runtime_binding")
    if bound is not None:
        from .runtime_binding import validate_worker, digest as runtime_digest, RuntimeBindingError
        try:
            validate_worker(bound["worker"])
            if bound.get("sha256") != runtime_digest(bound["contract"]):
                raise RuntimeBindingError("Take runtime binding digest does not verify")
        except (KeyError, TypeError, RuntimeBindingError) as exc:
            raise _refuse(str(exc), "runtime_mismatch") from exc
    path = _resolve_within(root, Path(audio_file) if Path(audio_file).is_absolute()
                           else root / audio_file, f"utterance {utterance_id} Take audio")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != take.audio_sha256:
        raise _refuse(f"utterance {utterance_id} Take bytes changed on disk", "plan_stale")
    state_sha256 = content_sha256({
        "source": revision.model_dump(mode="json"),
        "anchor": anchor.model_dump(mode="json"), "text": text_sha256(segment.text),
        "spoken_revision": segment.spoken_text_revision_id,
        "attribution": attribution.model_dump(mode="json"),
        "role": role.model_dump(mode="json"), "casting_binding": casting_binding,
        "take_id": take.id, "qa": report.model_dump(mode="json"),
        "generation": take.generation_metadata,
        "requirement": requirement.components(), "audio_sha256": digest,
    })
    return {"segment": segment, "take": take, "path": path, "data": data, "sha256": digest,
            "role": role,
            "source_revision_id": anchor.source_revision_id,
            "spoken_text_revision_id": segment.spoken_text_revision_id,
            "role_revision": requirement.voice_revision,
            "requirement_fingerprint": requirement.fingerprint(), "source_state_sha256": state_sha256}


def _stage_review(pilot_project_id: str, timeline: Mapping[str, Any], arm: str) -> dict:
    """Prepare one review-only edition in staging, holding no ownership boundary.

    Everything expensive happens here: reading and hashing the eligible Takes,
    assembling the PCM, verifying the decode and encoding the listening
    derivative.  Encoding can take long enough for the caller to cancel the
    durable operation, so no project or root lock may be held while it runs —
    nothing is published from this phase either.
    """

    project = load_project(pilot_project_id)
    _assert_isolated(project)
    index = project.metadata.get(PILOT_INDEX_KEY) or {}
    if arm not in (index.get("arms") or ()):
        raise _refuse(f"arm {arm!r} is not in this pilot's approved scope")
    pilot_root = _assert_pilot_destination(project, index)
    inputs = _timeline_inputs(project)
    _scope_rows(index, inputs)
    compiled = _recompile(timeline, inputs)
    operation_id = str(index.get("operation_id"))
    assert_operation_open(pilot_project_id, operation_id)
    _require_deadline(assert_operation_open(pilot_project_id, operation_id)["payload"]["deadline"], "operation")
    _assert_output_fundable(project, operation_id, arm, compiled["timeline_sha256"])
    deadline = (project.metadata.get(PILOT_LEDGER_KEY) or {})["ceilings"]["deadline"]
    _require_deadline(deadline, "publication")

    by_utterance = {row["utterance_id"]: row for row in inputs["rows"]}
    sources: dict[str, Path] = {}
    source_segments: dict[str, Sequence[Any]] = {}
    proofs: list[dict] = []
    captured: dict[str, bytes] = {}
    for placement in compiled["placements"]:
        if placement.get("kind") != "speech":
            continue
        utterance_id = placement["utterance_id"]
        row = by_utterance.get(utterance_id)
        if row is None:
            raise _refuse(f"timeline placement {utterance_id} is not in the pilot manifest",
                          "plan_stale")
        if row["audio_sha256"] != placement["audio_sha256"]:
            raise _refuse(f"utterance {utterance_id} audio binding changed", "plan_stale")
        if row["project_id"] not in source_segments:
            source_segments[row["project_id"]] = load_segments(row["project_id"])
        source_project = load_project(row["project_id"])
        eligible = _eligible_source(source_project, source_segments[row["project_id"]],
                                   utterance_id, expected_audio=placement["audio_sha256"])
        measured = measure_take(eligible["path"], eligible["sha256"], text=None)
        physical = ("schema_version", "audio_sha256", "sample_rate", "sample_domain", "frames",
                    "channels", "detector", "leading", "trailing", "onset", "flags")
        if any(row["measurement"].get(key) != measured.get(key) for key in physical):
            raise _refuse("Recorded measurement differs from the canonical detector on actual Take bytes",
                          "measurement_uncertain")
        if eligible["take"].id != placement["take_id"]:
            raise _refuse(f"utterance {utterance_id} selected Take changed", "plan_stale")
        sources[utterance_id] = eligible["path"]
        captured[utterance_id] = eligible["data"]
        proofs.append({
            "utterance_id": utterance_id, "project_id": row["project_id"],
            "take_id": eligible["take"].id, "audio_sha256": eligible["sha256"],
            "qa_report_sha256": _qa_digest(eligible["take"]),
            "source_revision_id": eligible["source_revision_id"],
            "spoken_text_revision_id": eligible["spoken_text_revision_id"],
            "role_id": eligible["role"].id,
            "role_revision": eligible["role_revision"],
            "requirement_fingerprint": eligible["requirement_fingerprint"],
            "source_state_sha256": eligible["source_state_sha256"],
            "measurement_sha256": row.get("measurement_sha256"),
        })

    timeline_hash = compiled["timeline_sha256"]
    directory = pilot_root / "performance" / "pilots" / pilot_project_id / arm / timeline_hash
    manifest = {
        "timeline": compiled, "proofs": proofs, "arm": arm,
        "operation_id": operation_id, "deadline": deadline, "pilot_id": pilot_project_id,
        "destinations": {utterance_id: sources[utterance_id] for utterance_id in sources},
    }
    existing = _reuse_if_verified(directory, manifest)
    if existing is not None:
        return {"reused": existing, "pilot_project_id": pilot_project_id, "arm": arm,
                "operation_id": operation_id, "deadline": deadline, "timeline_hash": timeline_hash,
                "directory": directory, "compiled": compiled, "proofs": proofs, "sources": sources}

    if directory.exists() and any(directory.iterdir()):
        raise _refuse("Interrupted edition retained; no automatic retry of an unknown outcome", "unknown_outcome")
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    # Build and publish from the exact bytes whose hashes established eligibility.
    originals = staging / "originals"
    originals.mkdir()
    sources = {}
    for utterance_id, data in captured.items():
        path = originals / f"{utterance_id}.wav"
        with path.open("xb") as out:
            out.write(data)
        sources[utterance_id] = path
    staged_master = staging / "review-master.wav"
    for_arm = {**compiled, "placements": _arm_placements(compiled, inputs, arm)}
    written = _assemble_timeline_pcm(for_arm, sources, staged_master)
    map_rows, total = _build_map(for_arm, sources, written["params"])
    if total != written["total_frames"]:
        raise _refuse("map accounting does not match the assembled frame count",
                      "measurement_uncertain")
    _verify_pcm(staged_master, total, written["params"])
    master_bytes = staged_master.read_bytes()
    mp3_bytes = _encode_mp3(staged_master, staging)
    if not staging.exists():
        raise _refuse("staging disappeared before publication")
    return {
        "pilot_project_id": pilot_project_id, "arm": arm, "operation_id": operation_id,
        "deadline": deadline, "timeline_hash": timeline_hash, "directory": directory,
        "staging": staging, "compiled": compiled, "proofs": proofs, "sources": sources,
        "map_rows": map_rows, "total": total, "params": written["params"],
        "master_bytes": master_bytes, "mp3_bytes": mp3_bytes,
    }


def _assert_output_fundable(project: Any, operation_id: str, arm: str,
                            timeline_hash: str) -> None:
    """Read-only check that this reservation still funds this exact output."""

    ledger = dict(project.metadata.get(PILOT_LEDGER_KEY) or {})
    receipt = dict((ledger.get("operations") or {}).get(operation_id) or {})
    if not receipt:
        raise _refuse("unknown operation; reserve before assembling")
    identity = {"arm": arm, "timeline_sha256": timeline_hash}
    if identity in (receipt.get("outputs") or []):
        return
    reserved = int((receipt.get("reserved") or {}).get("assemblies", 0))
    if len(receipt.get("outputs") or []) + 1 > reserved:
        raise _refuse(
            "this reservation cannot fund another assembly; one reservation covers only the "
            "assemblies it reserved", "budget_exhausted")


def _publish_review_locked(staged: Mapping[str, Any]) -> dict:
    """Re-verify the world and publish a staged edition.  Caller holds the locks.

    Every fact the staging phase relied on is re-read here, immediately before
    the first published byte: the arm scope, the operation and its outcome, the
    deadline, the exact ordered scope rows, the current eligibility proofs of
    every source and the funding of the reservation.
    """

    pilot_project_id = staged["pilot_project_id"]
    arm = staged["arm"]
    operation_id = staged["operation_id"]
    project = load_project(pilot_project_id)
    _assert_isolated(project)
    index = project.metadata.get(PILOT_INDEX_KEY) or {}
    if arm not in (index.get("arms") or ()):
        raise _refuse(f"arm {arm!r} left the pilot's approved scope before publication",
                      "plan_stale")
    _assert_pilot_destination(project, index)
    if str(index.get("operation_id")) != operation_id:
        raise _refuse("the pilot's operation changed before publication", "plan_stale")
    inputs = _timeline_inputs(project)
    _scope_rows(index, inputs)
    _recompile(staged["compiled"], inputs)
    assert_operation_open(pilot_project_id, operation_id)
    _require_deadline(assert_operation_open(pilot_project_id, operation_id)["payload"]["deadline"], "operation")
    ceiling = ((project.metadata.get(PILOT_LEDGER_KEY) or {}).get("ceilings") or {}).get("deadline")
    if ceiling != staged["deadline"]:
        raise _refuse("the pilot deadline changed before publication", "plan_stale")
    _require_deadline(ceiling, "publication")
    _assert_output_fundable(project, operation_id, arm, staged["timeline_hash"])

    by_utterance = {row["utterance_id"]: row for row in inputs["rows"]}
    source_segments: dict[str, Sequence[Any]] = {}
    for proof in staged["proofs"]:
        utterance_id = proof["utterance_id"]
        row = by_utterance.get(utterance_id)
        if row is None:
            raise _refuse(f"utterance {utterance_id} left the pilot manifest", "plan_stale")
        if row["project_id"] not in source_segments:
            source_segments[row["project_id"]] = load_segments(row["project_id"])
        eligible = _eligible_source(load_project(row["project_id"]),
                                    source_segments[row["project_id"]], utterance_id,
                                    expected_audio=proof["audio_sha256"])
        current = {
            "take_id": eligible["take"].id, "audio_sha256": eligible["sha256"],
            "role_id": eligible["role"].id, "role_revision": eligible["role_revision"],
            "requirement_fingerprint": eligible["requirement_fingerprint"],
            "source_revision_id": eligible["source_revision_id"],
            "spoken_text_revision_id": eligible["spoken_text_revision_id"],
            "source_state_sha256": eligible["source_state_sha256"],
        }
        recorded = {field: proof[field] for field in current}
        if current != recorded:
            raise _refuse(f"utterance {utterance_id} eligibility changed before publication",
                          "plan_stale")
        if sha256_bytes(eligible["path"].read_bytes()) != proof["audio_sha256"]:
            raise _refuse(f"utterance {utterance_id} bytes changed before publication",
                          "plan_stale")

    directory = staged["directory"]
    if "reused" in staged:
        existing = _reuse_if_verified(directory, {"timeline": staged["compiled"], "proofs": staged["proofs"],
                                                 "arm": arm, "destinations": staged["sources"]})
        if existing is None:
            raise _refuse("Completed edition disappeared before adoption", "plan_stale")
        _complete_operation_locked(pilot_project_id, operation_id, outcome="completed",
                                   output={"arm": arm, "timeline_sha256": staged["timeline_hash"]})
        _record_edition(pilot_project_id, arm, staged["timeline_hash"], existing)
        return existing
    receipt = _receipt(staged["compiled"], staged["proofs"], staged["map_rows"],
                       staged["total"], staged["master_bytes"], staged["mp3_bytes"],
                       directory, arm)
    _immutable(directory / "review-master.wav", staged["master_bytes"])
    for placement in staged["compiled"]["placements"]:
        if placement.get("kind") != "speech":
            continue
        _immutable(directory / "originals" / f"{placement['utterance_id']}.wav",
                   staged["sources"][placement["utterance_id"]].read_bytes())
    _immutable(directory / "source-to-audio-map.json", _json_bytes(staged["map_rows"]))
    _immutable(directory / "listening.mp3", staged["mp3_bytes"])
    _immutable(directory / "receipt.json", _json_bytes(receipt))
    # Preserve staging, including the original byte capture, as durable evidence.

    # Claim the output against the reservation under the same ownership boundary.
    _complete_operation_locked(pilot_project_id, operation_id, outcome="completed",
                               output={"arm": arm, "timeline_sha256": staged["timeline_hash"]})
    _record_edition(pilot_project_id, arm, staged["timeline_hash"], receipt)
    return receipt


def _arm_placements(compiled: Mapping[str, Any], inputs: Mapping[str, Any],
                    arm: str) -> list[dict]:
    """The placements one arm assembles.

    ``variant`` is the director's compiled plan: measured spacers and, where the
    policy allows it, the leading-silence offsets.  ``baseline`` applies no
    director timing decision at all — every source frame is kept, no lead is
    trimmed, and spacing is exactly the callers' declared
    ``baseline_spacer_frames`` — so a variant is always compared against the
    untouched audio it changes.  A scene separator is mandated by its boundary
    class rather than chosen by the director, so both arms keep it.
    """

    if arm == "variant":
        return [dict(placement) for placement in compiled["placements"]]
    baseline = {row["utterance_id"]: int(row.get("baseline_spacer_frames") or 0)
                for row in inputs["rows"]}
    boundaries = {item["boundary_index"]: item for item in compiled["boundaries"]}
    placements: list[dict] = []
    cursor = 0
    for placement in compiled["placements"]:
        kind = placement.get("kind")
        if kind == "scene_separator":
            placements.append({**placement, "derived_start_frame": cursor})
            cursor += int(placement["frames"])
            continue
        if kind == "spacer":
            boundary = boundaries.get(placement.get("boundary_index")) or {}
            frames = baseline.get(boundary.get("before_utterance_id"), 0)
            if frames:
                placements.append({
                    "kind": "spacer", "frames": frames,
                    "boundary_index": placement.get("boundary_index"),
                    "derived_start_frame": cursor,
                    "boundary": {**(placement.get("boundary") or {}), "status": "baseline"}})
                cursor += frames
            continue
        if kind != "speech":
            raise _refuse(f"unsupported placement kind: {kind!r}")
        start = int(placement["source_start_frame"])
        end = int(placement["source_end_frame"])
        placements.append({**placement, "offset_frames": 0, "derived_start_frame": cursor,
                           "derived_frames": end - start})
        cursor += end - start
    return placements


def _record_edition(pilot_project_id: str, arm: str, timeline_hash: str,
                    receipt: Mapping[str, Any]) -> None:
    """Index the published digests of one arm.  Re-reads: the ledger just changed."""

    project = load_project(pilot_project_id)
    editions = dict(project.metadata.get(PILOT_EDITIONS_KEY) or {})
    for name, digest in (receipt.get("artifacts") or {}).items():
        editions[digest] = {"arm": arm, "timeline_sha256": timeline_hash, "name": name,
                            "receipt_sha256": receipt.get("receipt_sha256")}
    project.metadata[PILOT_EDITIONS_KEY] = editions
    save_project(project)


def assemble_review(pilot_project_id: str, timeline: dict, *, arm: str) -> dict:
    """Assemble one review-only master from eligible Takes and a trusted timeline."""

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    if arm not in ARMS:
        raise _refuse(f"arm must be one of {list(ARMS)}")
    timeline = _require_mapping(timeline, "timeline")
    if timeline.get("schema_version") != BOUNDARY_TIMELINE_SCHEMA:
        raise _refuse("timeline schema version is unknown", "schema_unknown")
    read_pilot(pilot_project_id)
    staged = _stage_review(pilot_project_id, timeline, arm)
    with _root_write_lock("performance-pilot-assemble"):
        with ExitStack() as fences:
            projects = {pilot_project_id, *(proof["project_id"] for proof in staged["proofs"])}
            for project_id in sorted(projects):
                fences.enter_context(project_meta_lock(project_id))
                fences.enter_context(project_segments_lock(project_id))
            return _publish_review_locked(staged)


def _qa_digest(take: Any) -> str | None:
    report = take.qa_report
    if report is None:
        return None
    payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
    return content_sha256(payload)


def _encode_mp3(master: Path, staging: Path) -> bytes:
    target = staging / "listening.mp3"
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-n", "-i",
                        str(master), "-vn", "-codec:a", "libmp3lame", "-b:a", "192k", str(target)],
                       check=True, capture_output=True, timeout=120)
    except FileNotFoundError as exc:
        raise _refuse("ffmpeg is required to produce the listening derivative",
                      "capability_unsupported") from exc
    except subprocess.CalledProcessError as exc:
        raise _refuse(f"the listening derivative could not be encoded: {exc}",
                      "capability_unsupported") from exc
    return target.read_bytes()


def _build_map(timeline: Mapping[str, Any], sources: Mapping[str, Path],
               params: tuple[int, int, int] | None) -> tuple[dict, int]:
    channels, width, rate = params or (1, 2, 24000)
    rows: list[dict] = []
    total = 0
    for placement in timeline["placements"]:
        kind = placement["kind"]
        if kind == "speech":
            start = placement["source_start_frame"] + placement["offset_frames"]
            frames = placement["derived_frames"]
            rows.append({
                "kind": "speech", "utterance_id": placement["utterance_id"],
                "take_id": placement["take_id"], "audio_sha256": placement["audio_sha256"],
                "source_start_frame": placement["source_start_frame"],
                "source_end_frame": placement["source_end_frame"],
                "offset_frames": placement["offset_frames"],
                "effective_source_start_frame": start,
                "derived_start_frame": placement["derived_start_frame"],
                "frames": frames, "seconds": round(frames / rate, 6),
            })
        else:
            frames = placement["frames"]
            rows.append({"kind": kind, "frames": frames, "seconds": round(frames / rate, 6)})
        total += frames
    return {
        "schema_version": MAP_SCHEMA,
        "sample_rate": rate, "channels": channels, "sample_width_bytes": width,
        "timeline_sha256": timeline["timeline_sha256"],
        "total_frames": total, "total_seconds": round(total / rate, 6),
        "rows": rows,
    }, total


def _receipt(timeline: Mapping[str, Any], proofs: Sequence[Mapping[str, Any]],
             mapping: Mapping[str, Any], total: int, master_bytes: bytes, mp3_bytes: bytes,
             directory: Path, arm: str) -> dict:
    receipt = {
        "schema_version": ASSEMBLY_SCHEMA,
        "kind": "review_only",
        "arm": arm,
        "production_enablement": False,
        "human_review": "pending",
        "timeline_sha256": timeline["timeline_sha256"],
        "policy_sha256": timeline["policy_sha256"],
        "classifier_version": timeline["classifier_version"],
        "assembler_version": timeline["assembler_version"],
        "sample_rate": timeline["sample_rate"],
        "sample_domain": timeline["sample_domain"],
        "total_frames": total,
        "inputs": [dict(proof) for proof in proofs],
        "artifacts": {
            "review-master.wav": sha256_bytes(master_bytes),
            "listening.mp3": sha256_bytes(mp3_bytes),
            "source-to-audio-map.json": sha256_bytes(_json_bytes(mapping)),
        },
        "originals": {proof["utterance_id"]: proof["audio_sha256"] for proof in proofs},
        "map_sha256": content_sha256(mapping),
        "created_at": _now(),
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def _reuse_if_verified(directory: Path, manifest: Mapping[str, Any]) -> dict | None:
    """Reuse only when the complete edition still verifies by bytes and decode."""

    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise _refuse("an existing receipt is unreadable; it is retained, never overwritten")
    if receipt.get("schema_version") != ASSEMBLY_SCHEMA:
        raise _refuse("an existing receipt has an unknown schema; it is retained")
    if receipt.get("timeline_sha256") != manifest["timeline"]["timeline_sha256"]:
        raise _refuse("an existing receipt belongs to a different timeline")
    if receipt.get("arm") != manifest["arm"]:
        raise _refuse("an existing receipt belongs to a different arm")
    if receipt.get("inputs") != manifest["proofs"]:
        raise _refuse("Complete source/QA/cast/runtime proof changed; refusing reuse", "plan_stale")
    if receipt.get("receipt_sha256") != content_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise _refuse("an existing receipt's own digest does not verify; refusing reuse")
    # The complete expected artifact set, not whatever the receipt happens to list.
    declared = receipt.get("artifacts") or {}
    if set(declared) != set(REQUIRED_ARTIFACTS):
        raise _refuse("an existing receipt does not declare the complete artifact set")
    for name in REQUIRED_ARTIFACTS:
        path = directory / name
        if not path.is_file() or file_sha256(path) != declared[name]:
            raise _refuse(f"declared artifact {name} does not match its receipt; refusing reuse")
    # The proof identity must still be the current one.
    current = {proof["utterance_id"]: proof["audio_sha256"] for proof in manifest["proofs"]}
    if (receipt.get("originals") or {}) != current:
        raise _refuse("the existing receipt was built from different source Takes")
    for utterance_id, digest in current.items():
        copy = directory / "originals" / f"{utterance_id}.wav"
        if not copy.is_file() or file_sha256(copy) != digest:
            raise _refuse(f"the original copy for {utterance_id} does not match its Take")
    master = directory / "review-master.wav"
    params = None
    for proof in manifest["proofs"]:
        channels, width, rate, _frames = _pcm_params(manifest["destinations"][proof["utterance_id"]])
        params = (channels, width, rate)
        break
    _verify_pcm(master, int(receipt.get("total_frames")), params)
    mapping = json.loads((directory / "source-to-audio-map.json").read_text(encoding="utf-8"))
    if content_sha256(mapping) != receipt.get("map_sha256"):
        raise _refuse("the source-to-audio map does not match its receipt")
    from .voice_lab_render import _decoded_duration
    duration = _decoded_duration(directory / "listening.mp3")
    if not math.isfinite(duration) or abs(duration - int(receipt["total_frames"]) / params[2]) > 0.1:
        raise _refuse("Listening derivative does not decode to the master duration")
    return receipt


def pilot_editions(pilot_project_id: str) -> dict:
    """Every artifact digest this pilot has published, with its arm and timeline."""

    project = load_project(_require_identifier(pilot_project_id, "pilot_project_id"))
    _assert_isolated(project)
    editions = project.metadata.get(PILOT_EDITIONS_KEY) or {}
    if not isinstance(editions, Mapping):
        raise _refuse("the pilot's edition index is malformed", "schema_unknown")
    return {str(key): dict(value) for key, value in editions.items()
            if isinstance(value, Mapping)}


# --------------------------------------------------------------------------- unit D


def _owner_verification(evidence: Any) -> dict:
    """The private owner-verification boundary.  Ships unconfigured, so fail-closed.

    This is not callable through any public mutation: nothing passes a verdict in
    from outside the module.  A real integration would replace this function with
    a call to the authenticated owner (the private delivery identity boundary),
    which is not provisioned in this milestone.
    """

    return {
        "status": "unverified",
        "reason": OWNER_VERIFICATION_LIMIT,
        "fingerprint": content_sha256({"evidence": evidence if isinstance(evidence, Mapping) else None,
                                       "limit": OWNER_VERIFICATION_LIMIT}),
    }


def verify_reviewer(evidence: Any) -> dict:
    """Read-only report of the owner-verification seam's verdict."""

    return _owner_verification(evidence)


def _pair_key(pilot_project_id: str, pair_id: str) -> dict:
    project = load_project(pilot_project_id)
    keys = project.metadata.get(PILOT_KEYS_KEY)
    if not isinstance(keys, Mapping) or pair_id not in keys:
        raise _refuse(f"no blind pair {pair_id!r} is recorded for this pilot")
    entry = keys[pair_id]
    if not isinstance(entry, Mapping):
        raise _refuse("the stored pair key is malformed", "schema_unknown")
    return dict(entry)


def _pair_editions(pilot_project_id: str) -> dict:
    project = load_project(pilot_project_id)
    keys = project.metadata.get(PILOT_KEYS_KEY)
    if not isinstance(keys, Mapping):
        return {}
    return {str(key): dict(value) for key, value in keys.items() if isinstance(value, Mapping)}


def _verified_audio_edition(pilot_project_id: str, digest: str, arm: str) -> dict | None:
    """Resolve an audio digest through an immutable edition, verifying all bytes."""
    entry = pilot_editions(pilot_project_id).get(digest)
    if entry is None:
        return None
    if entry.get("arm") != arm or entry.get("name") not in ("review-master.wav", "listening.mp3"):
        raise _refuse("A blind arm must name audio from its actual published arm")
    timeline_hash = _require_sha256(entry.get("timeline_sha256"), "edition.timeline_sha256")
    directory = safe_project_root(pilot_project_id) / "performance" / "pilots" / pilot_project_id / arm / timeline_hash
    path = _resolve_within(safe_project_root(pilot_project_id), directory / "receipt.json", "edition receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("receipt_sha256") != entry.get("receipt_sha256"):
        raise _refuse("Edition receipt identity changed", "plan_stale")
    # Full source/proof validation happens on assembly/reuse; this read-only lookup
    # verifies the exact immutable audio edition a reviewer will hear.
    manifest = {"timeline": {"timeline_sha256": timeline_hash}, "arm": arm,
                "proofs": receipt.get("inputs") or [],
                "destinations": {p["utterance_id"]: directory / "originals" / f"{p['utterance_id']}.wav"
                                 for p in receipt.get("inputs") or []}}
    _reuse_if_verified(directory, manifest)
    if receipt["artifacts"].get(entry["name"]) != digest:
        raise _refuse("Audio digest differs from the verified edition", "plan_stale")
    return entry


def blind_pairs(pilot_project_id: str, pairs: Sequence[dict], *, seed: int) -> tuple[dict, dict]:
    """Deterministic opaque X/Y labels, with the arm key kept separate.

    Each supplied digest is resolved against the pilot's own published editions.
    A digest that resolves binds the pair to that exact immutable arm/timeline; a
    digest that does not resolve is recorded as unresolved, which is what a
    label-formatting-only fixture produces, and the gate refuses to interpret any
    review of an unresolved pair.
    """

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    _require_int(seed, "seed")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)) or not pairs:
        raise _refuse("pairs must be a non-empty sequence")
    state = read_pilot(pilot_project_id)
    arms = list(state["index"].get("arms") or ())
    if set(arms) != set(ARMS):
        raise _refuse("a blind comparison needs both baseline and variant arms")
    editions = pilot_editions(pilot_project_id)

    entries = []
    for number, pair in enumerate(pairs):
        pair = _require_mapping(pair, f"pairs[{number}]")
        pair_id = _require_identifier(pair.get("pair_id"), f"pairs[{number}].pair_id")
        baseline = _require_sha256(pair.get("baseline_audio_sha256"),
                                   f"pairs[{number}].baseline_audio_sha256")
        variant = _require_sha256(pair.get("variant_audio_sha256"),
                                  f"pairs[{number}].variant_audio_sha256")
        if baseline == variant:
            raise _refuse(f"pairs[{number}] has identical baseline and variant audio")
        resolved = [_verified_audio_edition(pilot_project_id, baseline, "baseline"),
                    _verified_audio_edition(pilot_project_id, variant, "variant")]
        if all(resolved) and resolved[0]["name"] != resolved[1]["name"]:
            raise _refuse("Blind arms must compare matching audio artifact formats")
        arms_seen = {item["arm"] for item in resolved if item}
        digest = sha256_bytes(canonical_json_bytes({"pair_id": pair_id, "seed": seed,
                                                    "baseline": baseline, "variant": variant}))
        order = (int(digest[:8], 16) + number) % 2 == 0
        entries.append({"pair_id": pair_id, "digest": digest, "order": order,
                        "baseline": baseline, "variant": variant,
                        "resolved": all(resolved) and arms_seen == set(ARMS),
                        "baseline_edition": resolved[0], "variant_edition": resolved[1]})

    ordered = sorted(entries, key=lambda item: item["digest"])
    public_rows, key_rows = [], []
    for index, entry in enumerate(ordered, start=1):
        label = f"pair-{index:03d}"
        x_arm, y_arm = ("baseline", "variant") if entry["order"] else ("variant", "baseline")
        public_rows.append({
            "label": label, "pair_id": entry["pair_id"],
            "x_audio_sha256": entry[x_arm], "y_audio_sha256": entry[y_arm],
        })
        key_rows.append({"label": label, "pair_id": entry["pair_id"],
                         "x_arm": x_arm, "y_arm": y_arm,
                         "x_audio_sha256": entry[x_arm], "y_audio_sha256": entry[y_arm],
                         "resolved": entry["resolved"],
                         "x_edition": entry[x_arm + "_edition"],
                         "y_edition": entry[y_arm + "_edition"]})
        key_rows[-1]["edition_sha256"] = content_sha256(key_rows[-1])

    ordered_rows = sorted(key_rows, key=lambda row: row["pair_id"])
    with _root_write_lock("performance-pilot-pairs"):
        with project_meta_lock(pilot_project_id):
            project = load_project(pilot_project_id)
            _assert_isolated(project)
            _assert_pilot_destination(project, project.metadata[PILOT_INDEX_KEY])
            stored = dict(project.metadata.get(PILOT_KEYS_KEY) or {})
            for row in ordered_rows:
                # A pair id keeps every edition it has ever bound: prior keys stay
                # immutable so an old review can still be interpreted as history.
                previous = stored.get(row["pair_id"])
                history = list((previous or {}).get("history") or [])
                if previous is not None and previous.get("edition_sha256") != row["edition_sha256"]:
                    history.append({key: value for key, value in previous.items() if key != "history"})
                stored[row["pair_id"]] = {**row, "history": history}
            project.metadata[PILOT_KEYS_KEY] = stored
            save_project(project)

    public = {"schema_version": PAIR_INDEX_SCHEMA, "pilot_id": pilot_project_id, "seed": seed,
              "pairs": public_rows, "label_seed": seed, "production_enablement": False,
              "editions": {row["pair_id"]: row["edition_sha256"] for row in ordered_rows}}
    private = {"schema_version": PAIR_KEY_SCHEMA, "pilot_id": pilot_project_id, "seed": seed,
               "key": ordered_rows}
    return public, private


_DISALLOWED_REVIEW_EVIDENCE = {
    "notification": "a delivery notification is not evidence of improvement",
    "qa_passed": "a QA pass checks bytes, not whether the listening improved",
    "favorite": "a favourite is a preference for the library, not a blind comparison",
    "rating": "a rating is not a blind pairwise preference",
    "baseline_acceptable": "the baseline being acceptable is not a preference for the variant",
    "human_verified": "a caller Boolean is not reviewer authentication",
}


def _validate_review(review: Mapping[str, Any]) -> dict[str, Any]:
    review = _require_mapping(review, "review")
    if review.get("schema_version") != REVIEW_SCHEMA:
        raise _refuse("review schema version is unknown", "schema_unknown")
    for key, reason in _DISALLOWED_REVIEW_EVIDENCE.items():
        if review.get(key) is not None:
            raise _refuse(f"{key!r} cannot support an improvement gate: {reason}",
                          "direction_conflict")
    rubric = _require_mapping(review.get("rubric"), "rubric")
    if not rubric:
        raise _refuse("a review needs a rubric")
    for field, value in rubric.items():
        _require_identifier(field, "rubric field")
        _require_str(value, f"rubric.{field}", max_length=400)
    preference = review.get("preference")
    if preference not in PREFERENCES:
        raise _refuse(f"preference must be one of {list(PREFERENCES)}")
    confidence = review.get("confidence")
    if confidence not in CONFIDENCES:
        raise _refuse(f"confidence must be one of {list(CONFIDENCES)}")
    heard = _require_mapping(review.get("heard"), "heard")
    result = {
        "schema_version": REVIEW_SCHEMA,
        "pilot_id": _require_identifier(review.get("pilot_id"), "pilot_id"),
        "pair_id": _require_identifier(review.get("pair_id"), "pair_id"),
        "edition_sha256": _require_sha256(review.get("edition_sha256"), "edition_sha256"),
        "heard": {
            "x_audio_sha256": _require_sha256(heard.get("x_audio_sha256"), "heard.x_audio_sha256"),
            "y_audio_sha256": _require_sha256(heard.get("y_audio_sha256"), "heard.y_audio_sha256"),
            "coverage": _require_identifier(heard.get("coverage", "full"), "heard.coverage"),
        },
        "rubric": {str(key): str(value) for key, value in rubric.items()},
        "preference": preference,
        "confidence": confidence,
        "stated_benefit": _require_str(review.get("stated_benefit", ""), "stated_benefit",
                                       allow_empty=True),
        "defects": [],
        "reviewer": _require_str(review.get("reviewer", ""), "reviewer", allow_empty=True),
        "at": _require_iso(review.get("at", _now()), "at"),
    }
    defects = review.get("defects") or []
    if not isinstance(defects, list):
        raise _refuse("defects must be a list")
    for number, defect in enumerate(defects):
        defect = _require_mapping(defect, f"defects[{number}]")
        item = {
            "audio_sha256": _require_sha256(defect.get("audio_sha256"),
                                            f"defects[{number}].audio_sha256"),
            "seconds": _require_finite(defect.get("seconds"), f"defects[{number}].seconds"),
            "note": _require_str(defect.get("note"), f"defects[{number}].note", max_length=1000),
        }
        if item["audio_sha256"] not in (result["heard"]["x_audio_sha256"],
                                        result["heard"]["y_audio_sha256"]):
            raise _refuse(f"defects[{number}] names audio that was not heard")
        result["defects"].append(item)
    return result


def _current_edition(key: Mapping[str, Any]) -> dict | None:
    return {field: key.get(field) for field in
            ("x_arm", "y_arm", "x_audio_sha256", "y_audio_sha256")}


def record_review(pilot_project_id: str, review: dict) -> dict:
    """Validate and store one review against an exact artifact edition.

    There is no parameter for a caller-supplied verdict: verification happens
    only at the private owner boundary, which is unconfigured, so a live
    owner-authenticated review raises ``review_pending``.  A review whose heard
    audio is not the pair's current edition is still written, immutably, as
    stale history — it is evidence about an older edition and is excluded from
    the current gate rather than discarded.
    """

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    checked = _validate_review(review)
    if checked["pilot_id"] != pilot_project_id:
        raise _refuse("review pilot_id does not match the pilot", "direction_conflict")
    claim = review.get("authentication")
    status = "draft"
    verification = None
    if claim is not None:
        claim = _require_mapping(claim, "authentication")
        if claim.get("kind") != "synthetic_test":
            verdict = _owner_verification(claim.get("evidence"))
            if verdict.get("status") != "verified":
                raise _refuse(
                    "an owner-authenticated review cannot be committed: "
                    + str(verdict.get("reason")), "review_pending")
            status = "verified"
            verification = verdict

    with _root_write_lock("performance-pilot-review"):
        with project_meta_lock(pilot_project_id):
            project = load_project(pilot_project_id)
            _assert_isolated(project)
            directory = _assert_pilot_destination(project, project.metadata.get(PILOT_INDEX_KEY) or {})
            key = _pair_key(pilot_project_id, checked["pair_id"])
            current = _current_edition(key)
            heard = checked["heard"]
            # A pair that names no published edition has no current edition: a
            # review of it is history, never a verdict on published audio.
            is_current = bool(key.get("resolved")) and (
                heard["x_audio_sha256"] == current["x_audio_sha256"]
                and heard["y_audio_sha256"] == current["y_audio_sha256"]
                and checked["edition_sha256"] == key.get("edition_sha256"))
            content = {**checked, "status": status if is_current else "stale",
                       "stale_history": not is_current,
                       "production_enablement": False}
            if verification is not None and is_current:
                content["verification"] = {"status": verification["status"],
                                           "fingerprint": verification["fingerprint"]}
            review_sha256 = content_sha256(content)
            record = {**content, "review_sha256": review_sha256}
            path = directory / "performance" / "pilots" / pilot_project_id / "reviews" / \
                f"{checked['pair_id']}-{review_sha256[:16]}.json"
            _immutable(path, _json_bytes(record))
    return {**record, "path": str(path)}


def read_reviews(pilot_project_id: str, *, pair_id: str | None = None) -> list[dict]:
    """Owner-side review history, including stale editions of a pair."""

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    project = load_project(pilot_project_id)
    directory = _assert_pilot_destination(project, project.metadata[PILOT_INDEX_KEY])
    folder = directory / "performance" / "pilots" / pilot_project_id / "reviews"
    if not folder.is_dir():
        return []
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
    return [row for row in rows if pair_id is None or row.get("pair_id") == pair_id]


def improvement_gate(pilot_project_id: str, *, pair_id: str) -> dict:
    """Whether one pair supports an improvement claim, and what would be needed.

    The gate compares the CURRENT X/Y hashes of the pair and resolves the
    preference through the current key only.  Stale-edition reviews stay in
    history and can never accept a newer edition.
    """

    pair_id = _require_identifier(pair_id, "pair_id")
    key = _pair_key(pilot_project_id, pair_id)
    current = _current_edition(key)
    if not key.get("resolved"):
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": "the pair's audio has not been resolved to published editions of this "
                          "pilot, so no authenticated comparison is possible"}
    try:
        for label in ("x", "y"):
            _verified_audio_edition(pilot_project_id, key[f"{label}_audio_sha256"], key[f"{label}_arm"])
    except (PerformanceDirectorError, OSError, ValueError) as exc:
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": f"Current audio edition no longer verifies: {exc}"}
    reviews = read_reviews(pilot_project_id, pair_id=pair_id)
    if not reviews:
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": "no review has been recorded for this pair"}
    latest = max(reviews, key=lambda row: (row.get("at") or "", row.get("review_sha256") or ""))
    if (latest.get("edition_sha256") != key.get("edition_sha256")
            or (latest.get("heard") or {}).get("coverage") != "full"
            or latest.get("review_sha256") != content_sha256({k: v for k, v in latest.items() if k != "review_sha256"})):
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": "Review refers to an older edition, invalid identity or incomplete heard coverage"}
    heard = latest.get("heard") or {}
    if (heard.get("x_audio_sha256") != current["x_audio_sha256"]
            or heard.get("y_audio_sha256") != current["y_audio_sha256"]):
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "reason": "the most recent review heard an older edition of this pair; it remains "
                          "history and cannot accept the current edition"}
    preference = latest.get("preference")
    if preference == "tie":
        return {"pair_id": pair_id, "status": "inconclusive", "supports_improvement": False,
                "reason": "a tie is not a preference for the variant"}
    if preference == "inconclusive":
        return {"pair_id": pair_id, "status": "inconclusive", "supports_improvement": False,
                "reason": "the reviewer was inconclusive"}
    resolved = current["x_arm"] if preference == "X" else current["y_arm"]
    if latest.get("status") != "verified":
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "resolved_arm": resolved,
                "reason": "the review is an unauthenticated draft; the gate stays closed"}
    if latest.get("defects"):
        return {"pair_id": pair_id, "status": "blocked", "supports_improvement": False,
                "resolved_arm": resolved, "reason": "the review records a new defect"}
    if not (latest.get("stated_benefit") or "").strip():
        return {"pair_id": pair_id, "status": "review_pending", "supports_improvement": False,
                "resolved_arm": resolved, "reason": "no stated benefit"}
    if resolved != "variant":
        return {"pair_id": pair_id, "status": "not_improved", "supports_improvement": False,
                "resolved_arm": resolved,
                "reason": "the authenticated preference was for the baseline"}
    return {"pair_id": pair_id, "status": "supported", "supports_improvement": True,
            "resolved_arm": resolved,
            "reason": "an authenticated exact-artifact preference for the variant, a stated "
                      "benefit and no new defect"}


def propose_scene(pilot_project_id: str, *, pairs_a: Sequence[str],
                  pairs_b: Sequence[str]) -> dict:
    """Whether a combined scene (stage C) may even be proposed.

    Both stage A and stage B must support improvement on every listed pair; a
    single unsupported pair blocks the proposal, and this never grants scope.
    """

    pilot_project_id = _require_identifier(pilot_project_id, "pilot_project_id")
    if not pairs_a or not pairs_b:
        raise _refuse("stage A and stage B must each list at least one pair")
    gates_a = [improvement_gate(pilot_project_id, pair_id=_require_identifier(item, "pairs_a entry"))
               for item in pairs_a]
    gates_b = [improvement_gate(pilot_project_id, pair_id=_require_identifier(item, "pairs_b entry"))
               for item in pairs_b]
    supported = all(gate["supports_improvement"] for gate in gates_a) and \
        all(gate["supports_improvement"] for gate in gates_b)
    return {
        "may_propose_scene": bool(supported),
        "stage_a": gates_a,
        "stage_b": gates_b,
        "requires_own_scope_grant": True,
        "reason": ("both stages support improvement; a combined scene still needs its own scope "
                   "grant" if supported else
                   "stage A and stage B must both support improvement before a combined scene "
                   "is proposed"),
    }


# ---------------------------------------------------------------------------
# Additive ELF chapter 6 pilot connection.
#
# Everything above is the reviewed M3 module, preserved byte-for-byte as an
# exact prefix of this file.  The two definitions below are the only additions:
# they connect the private owner boundary to real owner-issued receipts and
# close the one shape that could otherwise smuggle an allocation grant into a
# listening review.
# ---------------------------------------------------------------------------

_owner_verification_without_live_owner = _owner_verification
_record_review_without_live_owner = record_review


def _owner_verification(evidence: Any) -> dict:
    """Route the private owner boundary to a real live-owner receipt when one is presented.

    A live-owner *allocation* grant is a durable, owner-issued receipt that funds
    one bounded operation, issued only through the private delivery server's
    Tailscale identity boundary.  Its reference is the only new evidence shape
    this boundary accepts.  Every other shape — including every shape the
    unconfigured seam used to see — keeps the original fail-closed verdict
    unchanged, so an unconfigured deployment still cannot allocate a live pilot
    and cannot commit an owner-authenticated review.

    Nothing here can manufacture a listening verdict: allocation authorizes work,
    and review authentication remains pending.
    """

    from dubbing_service.app.audiobook.performance_live_owner import verify_receipt
    verdict = verify_receipt(evidence)
    if verdict is not None:
        return verdict
    return _owner_verification_without_live_owner(evidence)


def record_review(pilot_project_id: str, review: dict) -> dict:
    """Refuse an allocation grant offered as a listening review.

    A grant is permission to run a scope.  Recording it as a preference would
    fabricate a human judgement, so any allocation reference inside a review's
    authentication claim is refused before the shared seam sees it.
    """

    from dubbing_service.app.audiobook.performance_live_owner import refuse_allocation_authority
    if isinstance(review, Mapping):
        refuse_allocation_authority(review.get("authentication"))
    return _record_review_without_live_owner(pilot_project_id, review)


_validate_pilot_manifest_without_live_scope_binding = _validate_pilot_manifest


def _validate_pilot_manifest(name: str, manifest: Mapping[str, Any]) -> dict:
    """Consume exact owner scope at every real allocation, including retries."""
    checked = _validate_pilot_manifest_without_live_scope_binding(name, manifest)
    if checked["authorization"]["kind"] == "live_pilot":
        from .performance_live_owner import authorize_manifest
        authorize_manifest(manifest)
    return checked


def _arm_placements(compiled: Mapping[str, Any], inputs: Mapping[str, Any],
                         arm: str) -> list[dict]:
    """Keep declared baseline spacing even when the variant needs no spacer.

    Baseline spacing is an input contract, independent of whether the variant
    compiler emits a spacer placement. Explicit scene separators retain their
    mandated duration in both arms and receive no additional baseline gap.
    """
    if arm == "variant":
        return [dict(item) for item in compiled["placements"]]
    declared = {row["utterance_id"]: int(row.get("baseline_spacer_frames") or 0)
                for row in inputs["rows"]}
    boundaries = {item.get("before_utterance_id"): item
                  for item in compiled["boundaries"]}
    result, cursor, after_separator = [], 0, False
    for item in compiled["placements"]:
        kind = item.get("kind")
        if kind == "spacer":
            continue
        if kind == "scene_separator":
            result.append({**item, "derived_start_frame": cursor})
            cursor += int(item["frames"])
            after_separator = True
            continue
        if kind != "speech":
            raise _refuse(f"unsupported placement kind: {kind!r}")
        boundary = boundaries.get(item["utterance_id"])
        frames = declared.get(item["utterance_id"], 0) if boundary and not after_separator else 0
        if frames:
            result.append({"kind": "spacer", "frames": frames,
                           "boundary_index": boundary["boundary_index"],
                           "derived_start_frame": cursor,
                           "boundary": {**boundary, "status": "baseline"}})
            cursor += frames
        start, end = int(item["source_start_frame"]), int(item["source_end_frame"])
        result.append({**item, "offset_frames": 0, "derived_start_frame": cursor,
                       "derived_frames": end - start})
        cursor += end - start
        after_separator = False
    return result
