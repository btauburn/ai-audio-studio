"""Process-local coordination for exclusive generative-audio/GPU operations.

The platform currently owns one managed TTS worker/GPU execution lane.  A
batch synthesis operation therefore has to exclude every other batch as well
as worker lifecycle changes.  Keeping this state in one module makes the REST
and WebSocket entrypoints share the same lease instead of maintaining parallel
guards that can race each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class GPUOperationLease:
    """Opaque ownership token for one exclusive GPU operation."""

    token: str
    kind: str
    project_id: str | None = None


_STATE_LOCK = Lock()
_ACTIVE_OPERATION: GPUOperationLease | None = None
_ACTIVE_BATCHES: dict[str, str] = {}
_ACTIVE_SYNTH_SEGMENTS: dict[tuple[str, str], str] = {}
_ACTIVE_PROJECT_DELETIONS: dict[str, str] = {}
_ACTIVE_PROJECT_MUTATIONS: dict[str, set[str]] = {}
_ACTIVE_PROJECT_EXPORTS: dict[str, str] = {}


def _new_lease(kind: str, project_id: str | None = None) -> GPUOperationLease:
    return GPUOperationLease(
        token=uuid4().hex,
        kind=str(kind).strip() or "tts_operation",
        project_id=project_id,
    )


def try_acquire_gpu_operation(
    *,
    kind: str,
    project_id: str | None = None,
) -> GPUOperationLease | None:
    """Acquire the single process-local GPU lane without waiting.

    Returning ``None`` is intentional: API entrypoints can fail fast with 409
    rather than leaving a browser request blocked behind a long synthesis job.
    """

    global _ACTIVE_OPERATION

    with _STATE_LOCK:
        if (
            _ACTIVE_OPERATION is not None
            or _ACTIVE_BATCHES
            or _ACTIVE_SYNTH_SEGMENTS
        ):
            return None
        lease = _new_lease(kind, project_id)
        _ACTIVE_OPERATION = lease
        return lease


def release_gpu_operation(lease: GPUOperationLease) -> None:
    """Release ``lease`` if it still owns the TTS/GPU lane.

    Token matching makes release idempotent and prevents a stale ``finally``
    block from releasing a newer owner's operation.
    """

    global _ACTIVE_OPERATION

    with _STATE_LOCK:
        # Batch leases own both _ACTIVE_OPERATION and _ACTIVE_BATCHES and must
        # be released atomically by unregister_running_batch().
        if lease.kind == "batch_synthesis":
            return
        if _ACTIVE_OPERATION is not None and _ACTIVE_OPERATION.token == lease.token:
            _ACTIVE_OPERATION = None


def has_active_gpu_operation() -> bool:
    with _STATE_LOCK:
        return _ACTIVE_OPERATION is not None


def try_register_running_batch(project_id: str) -> GPUOperationLease | None:
    """Atomically register a batch and acquire the global TTS/GPU lane.

    The returned token must be passed to :func:`unregister_running_batch`.
    Requiring the ownership token prevents a stale ``finally`` block from
    unregistering a newer batch for the same project.
    """

    global _ACTIVE_OPERATION

    with _STATE_LOCK:
        if (
            project_id in _ACTIVE_BATCHES
            or project_id in _ACTIVE_PROJECT_DELETIONS
            or project_id in _ACTIVE_PROJECT_MUTATIONS
            or project_id in _ACTIVE_PROJECT_EXPORTS
            or _ACTIVE_OPERATION is not None
            or bool(_ACTIVE_SYNTH_SEGMENTS)
        ):
            return None

        lease = _new_lease("batch_synthesis", project_id)
        _ACTIVE_BATCHES[project_id] = lease.token
        _ACTIVE_OPERATION = lease
        return lease


def unregister_running_batch(lease: GPUOperationLease) -> None:
    """Unregister a batch and release only the lease owned by ``lease``."""

    global _ACTIVE_OPERATION

    with _STATE_LOCK:
        if lease.kind != "batch_synthesis" or lease.project_id is None:
            return
        registered_token = _ACTIVE_BATCHES.get(lease.project_id)
        if registered_token != lease.token:
            return
        _ACTIVE_BATCHES.pop(lease.project_id, None)
        if _ACTIVE_OPERATION is not None and _ACTIVE_OPERATION.token == lease.token:
            _ACTIVE_OPERATION = None


def has_running_batch_for_project(project_id: str) -> bool:
    with _STATE_LOCK:
        return project_id in _ACTIVE_BATCHES


def try_register_project_deletion(project_id: str) -> GPUOperationLease | None:
    """Reserve ``project_id`` for deletion if it has no active synthesis.

    This reservation is separate from the global GPU lane: deleting an idle
    project must not interrupt or unnecessarily block a batch for a different
    project.  Registration and the active-state checks are nevertheless
    atomic with batch registration, closing the check-then-delete race.
    """

    with _STATE_LOCK:
        if (
            project_id in _ACTIVE_PROJECT_DELETIONS
            or project_id in _ACTIVE_BATCHES
            or (
                _ACTIVE_OPERATION is not None
                and _ACTIVE_OPERATION.project_id == project_id
            )
            or project_id in _ACTIVE_PROJECT_MUTATIONS
            or project_id in _ACTIVE_PROJECT_EXPORTS
            or any(pid == project_id for pid, _sid in _ACTIVE_SYNTH_SEGMENTS)
        ):
            return None
        lease = _new_lease("project_delete", project_id)
        _ACTIVE_PROJECT_DELETIONS[project_id] = lease.token
        return lease


def unregister_project_deletion(lease: GPUOperationLease) -> None:
    """Release a project deletion reservation owned by ``lease``."""

    with _STATE_LOCK:
        if lease.kind != "project_delete" or lease.project_id is None:
            return
        if _ACTIVE_PROJECT_DELETIONS.get(lease.project_id) != lease.token:
            return
        _ACTIVE_PROJECT_DELETIONS.pop(lease.project_id, None)


def set_segment_actively_synthesizing(
    project_id: str,
    segment_id: str,
    running: bool,
) -> bool:
    """Compatibility marker used by registered batch synthesis.

    New standalone synthesis entrypoints should use
    :func:`try_register_active_segment` so release is token-owned.
    """

    key = (project_id, segment_id)
    with _STATE_LOCK:
        if running:
            if (
                project_id in _ACTIVE_PROJECT_MUTATIONS
                or project_id in _ACTIVE_PROJECT_EXPORTS
                or project_id in _ACTIVE_PROJECT_DELETIONS
            ):
                return False
            _ACTIVE_SYNTH_SEGMENTS.setdefault(key, uuid4().hex)
        else:
            _ACTIVE_SYNTH_SEGMENTS.pop(key, None)
        return True


def is_segment_actively_synthesizing(project_id: str, segment_id: str) -> bool:
    with _STATE_LOCK:
        return (project_id, segment_id) in _ACTIVE_SYNTH_SEGMENTS


def count_actively_synthesizing_segments(project_id: str) -> int:
    with _STATE_LOCK:
        return sum(1 for pid, _sid in _ACTIVE_SYNTH_SEGMENTS if pid == project_id)


def try_acquire_project_mutation(
    project_id: str,
    *,
    kind: str = "project_mutation",
) -> GPUOperationLease | None:
    """Reserve one project's mutable state without blocking other projects."""

    with _STATE_LOCK:
        if (
            project_id in _ACTIVE_PROJECT_EXPORTS
            or (
                _ACTIVE_OPERATION is not None
                and _ACTIVE_OPERATION.project_id == project_id
            )
            or project_id in _ACTIVE_BATCHES
            or project_id in _ACTIVE_PROJECT_DELETIONS
            or any(pid == project_id for pid, _sid in _ACTIVE_SYNTH_SEGMENTS)
        ):
            return None
        lease = _new_lease(kind, project_id)
        _ACTIVE_PROJECT_MUTATIONS.setdefault(project_id, set()).add(lease.token)
        return lease


def release_project_mutation(lease: GPUOperationLease) -> None:
    """Release only the project mutation reservation owned by ``lease``."""

    with _STATE_LOCK:
        if lease.project_id is None:
            return
        tokens = _ACTIVE_PROJECT_MUTATIONS.get(lease.project_id)
        if tokens is None or lease.token not in tokens:
            return
        tokens.discard(lease.token)
        if not tokens:
            _ACTIVE_PROJECT_MUTATIONS.pop(lease.project_id, None)


def try_acquire_project_export(
    project_id: str,
    *,
    kind: str = "project_export",
) -> GPUOperationLease | None:
    """Reserve a stable project snapshot for a long read/export operation."""

    with _STATE_LOCK:
        if (
            project_id in _ACTIVE_PROJECT_EXPORTS
            or (
                _ACTIVE_OPERATION is not None
                and _ACTIVE_OPERATION.project_id == project_id
            )
            or project_id in _ACTIVE_PROJECT_MUTATIONS
            or project_id in _ACTIVE_BATCHES
            or project_id in _ACTIVE_PROJECT_DELETIONS
            or any(pid == project_id for pid, _sid in _ACTIVE_SYNTH_SEGMENTS)
        ):
            return None
        lease = _new_lease(kind, project_id)
        _ACTIVE_PROJECT_EXPORTS[project_id] = lease.token
        return lease


def release_project_export(lease: GPUOperationLease) -> None:
    """Release only the project export reservation owned by ``lease``."""

    with _STATE_LOCK:
        if lease.project_id is None:
            return
        if _ACTIVE_PROJECT_EXPORTS.get(lease.project_id) != lease.token:
            return
        _ACTIVE_PROJECT_EXPORTS.pop(lease.project_id, None)


def try_register_active_segment(
    project_id: str,
    segment_id: str,
    *,
    kind: str = "segment_synthesis",
) -> GPUOperationLease | None:
    """Token-own one active segment unless the project is being changed/read."""

    key = (project_id, segment_id)
    with _STATE_LOCK:
        if (
            key in _ACTIVE_SYNTH_SEGMENTS
            or project_id in _ACTIVE_PROJECT_MUTATIONS
            or project_id in _ACTIVE_PROJECT_EXPORTS
            or project_id in _ACTIVE_PROJECT_DELETIONS
        ):
            return None
        lease = _new_lease(kind, project_id)
        _ACTIVE_SYNTH_SEGMENTS[key] = lease.token
        return lease


def unregister_active_segment(
    lease: GPUOperationLease,
    segment_id: str,
) -> None:
    """Release an active-segment marker only when the token still owns it."""

    if lease.project_id is None:
        return
    key = (lease.project_id, segment_id)
    with _STATE_LOCK:
        if _ACTIVE_SYNTH_SEGMENTS.get(key) != lease.token:
            return
        _ACTIVE_SYNTH_SEGMENTS.pop(key, None)


# Backwards-compatible names for the existing TTS routes.  New GPU consumers
# (subtitles, sound generation, and future music generation) should use the
# generic names above so the resource being coordinated is explicit.
TTSOperationLease = GPUOperationLease
try_acquire_tts_operation = try_acquire_gpu_operation
release_tts_operation = release_gpu_operation
has_active_tts_operation = has_active_gpu_operation
