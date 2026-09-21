"""Canonical source snapshots, spoken-text revisions and attribution (Phases 1-4).

Canonical source text is immutable and content-addressed; spoken text is a
versioned derivative (pronunciation expansion etc.); whatever the renderer
actually received stays in ``SegmentTake.synthesis_text``.  Engine-specific
markup never enters the canonical or spoken text.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from dubbing_service.app.audiobook.fingerprint import (
    canonical_json_bytes,
    sha256_bytes,
    text_sha256,
)
from dubbing_service.app.core.helpers import project_root as project_root_path
from dubbing_service.app.models.audiobook import (
    Attribution,
    AttributionSource,
    SourceRevision,
    SpokenTextRevision,
)
from dubbing_service.app.models.project import Project, Segment

SNAPSHOT_DIR = "audiobook/source"


class ProvenanceError(RuntimeError):
    """Raised when provenance cannot be trusted (missing or tampered snapshot)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".stage")
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    except BaseException:
        try:
            if staged.is_file():
                staged.unlink()
        except OSError:
            pass
        raise


def snapshot_directory(project_id: str, *, ensure: bool = False) -> Path:
    root = project_root_path(project_id, ensure_dir=ensure) / SNAPSHOT_DIR
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
    return root


def new_revision_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def register_source_revision(
    project_id: str,
    project: Project,
    *,
    uri: str,
    data: str | bytes,
    kind: str = "file",
    media_type: Optional[str] = None,
    title: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[SourceRevision, bool]:
    """Store an immutable, hash-verified source snapshot.

    Returns ``(revision, created)``.  Importing identical bytes twice reuses the
    same revision instead of duplicating project state.
    """

    payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    digest = sha256_bytes(payload)
    for existing in project.source_revisions:
        if existing.sha256 == digest and existing.byte_length == len(payload):
            return existing, False

    relative = f"{SNAPSHOT_DIR}/{digest[:24]}.snapshot"
    target = project_root_path(project_id, ensure_dir=True) / relative
    _atomic_write_bytes(target, payload)
    revision = SourceRevision(
        id=new_revision_id("src"),
        uri=uri,
        kind=kind,  # type: ignore[arg-type]
        media_type=media_type,
        sha256=digest,
        byte_length=len(payload),
        snapshot_file=relative,
        title=title,
        acquired_at=utcnow(),
        notes=notes,
    )
    project.source_revisions.append(revision)
    return revision, True


def read_source_snapshot(project_id: str, revision: SourceRevision) -> str:
    """Read the snapshot back and verify it was not modified on disk."""

    if not revision.snapshot_file:
        raise ProvenanceError(f"来源修订 {revision.id} 没有快照文件")
    path = project_root_path(project_id) / revision.snapshot_file
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProvenanceError(f"来源快照不可读：{path}") from exc
    if sha256_bytes(payload) != revision.sha256:
        raise ProvenanceError(f"来源快照已被修改，拒绝使用：{path}")
    return payload.decode("utf-8")


def get_source_revision(project: Project, revision_id: str) -> Optional[SourceRevision]:
    return next(
        (item for item in project.source_revisions if item.id == revision_id), None
    )


def create_spoken_text_revision(
    project: Project,
    *,
    source_text: str,
    spoken_text: str,
    source_revision_id: Optional[str] = None,
    pronunciation_revision: Optional[str] = None,
    reason: Optional[str] = None,
    previous_revision_id: Optional[str] = None,
) -> tuple[SpokenTextRevision, bool]:
    """Create or reuse a spoken-text revision.

    Canonical source text is never modified: only this derivative changes.
    """

    source_hash = text_sha256(source_text)
    spoken_hash = text_sha256(spoken_text)
    for existing in project.spoken_text_revisions:
        if (
            existing.source_text_hash == source_hash
            and existing.spoken_text_hash == spoken_hash
            and (existing.pronunciation_revision or "") == (pronunciation_revision or "")
        ):
            return existing, False

    if previous_revision_id is None:
        previous_revision_id = latest_spoken_text_revision_id(
            project,
            source_revision_id=source_revision_id,
            source_text_hash=source_hash,
        )
    revision = SpokenTextRevision(
        id=new_revision_id("spk"),
        source_revision_id=source_revision_id,
        source_text_hash=source_hash,
        spoken_text=spoken_text,
        spoken_text_hash=spoken_hash,
        pronunciation_revision=pronunciation_revision,
        previous_revision_id=previous_revision_id,
        created_at=utcnow(),
        reason=reason,
    )
    project.spoken_text_revisions.append(revision)
    return revision, True


def latest_spoken_text_revision_id(
    project: Project,
    *,
    source_revision_id: Optional[str] = None,
    source_text_hash: Optional[str] = None,
) -> Optional[str]:
    """Newest spoken-text revision, optionally narrowed to one canonical text.

    Filtering by ``source_text_hash`` keeps the revision chain per canonical
    text instead of mixing unrelated utterances that happen to be edited in
    sequence.
    """

    candidates: Iterable[SpokenTextRevision] = project.spoken_text_revisions
    if source_revision_id:
        candidates = [
            item
            for item in project.spoken_text_revisions
            if item.source_revision_id == source_revision_id
        ]
    if source_text_hash:
        candidates = [
            item for item in candidates if item.source_text_hash == source_text_hash
        ]
    ordered = sorted(candidates, key=lambda item: (item.created_at, item.id))
    return ordered[-1].id if ordered else None


def get_spoken_text_revision(
    project: Project, revision_id: Optional[str]
) -> Optional[SpokenTextRevision]:
    if not revision_id:
        return None
    return next(
        (item for item in project.spoken_text_revisions if item.id == revision_id), None
    )


def spoken_text_for_segment(project: Project, segment: Segment) -> str:
    """Spoken text for a live utterance, falling back to canonical text."""

    revision = get_spoken_text_revision(project, segment.spoken_text_revision_id)
    if revision is not None:
        return revision.spoken_text
    return segment.text


def set_spoken_text(
    project: Project,
    segment: Segment,
    *,
    spoken_text: str,
    source_text: Optional[str] = None,
    pronunciation_revision: Optional[str] = None,
    reason: Optional[str] = None,
) -> SpokenTextRevision:
    """Re-point a live utterance at a (new or existing) spoken-text revision."""

    canonical = source_text if source_text is not None else segment.text
    revision, _ = create_spoken_text_revision(
        project,
        source_text=canonical,
        spoken_text=spoken_text,
        source_revision_id=(
            segment.source_anchor.source_revision_id if segment.source_anchor else None
        ),
        pronunciation_revision=pronunciation_revision,
        reason=reason,
    )
    segment.spoken_text_revision_id = revision.id
    return revision


def record_attribution(
    segment: Segment,
    *,
    speaker_id: Optional[str],
    source: AttributionSource,
    confidence: Optional[float] = None,
    reviewed: bool = False,
    note: Optional[str] = None,
) -> Attribution:
    attribution = Attribution(
        speaker_id=speaker_id,
        source=source,
        confidence=confidence,
        reviewed=reviewed,
        updated_at=utcnow(),
        note=note,
    )
    segment.attribution = attribution
    return attribution


def apply_attribution_proposal(
    segment: Segment,
    *,
    speaker_id: Optional[str],
    source: AttributionSource,
    confidence: Optional[float] = None,
    note: Optional[str] = None,
) -> tuple[Attribution, bool]:
    """Apply an automatic attribution proposal unless a human decided first.

    Returns ``(attribution, applied)``.  A human correction
    (``source="user"`` + ``reviewed``) is authoritative: later automatic
    re-analysis is recorded as a note and never silently overwrites it.
    """

    current = segment.attribution
    if current is not None and current.human_authoritative:
        if note:
            current.note = note
        return current, False

    attribution = record_attribution(
        segment,
        speaker_id=speaker_id,
        source=source,
        confidence=confidence,
        reviewed=source == "user",
        note=note,
    )
    return attribution, True


def human_correction(
    segment: Segment, *, speaker_id: Optional[str], note: Optional[str] = None
) -> Attribution:
    """Record an authoritative human speaker decision."""

    return record_attribution(
        segment,
        speaker_id=speaker_id,
        source="user",
        confidence=1.0,
        reviewed=True,
        note=note,
    )


def pronunciation_revision_id(project_id: str) -> str:
    """Hash of the enabled pronunciation dictionary (audio-affecting input)."""

    try:
        from dubbing_service.app.storage.production import load_settings

        entries = [
            {
                "source": item.source,
                "pronunciation": item.pronunciation,
                "enabled": bool(item.enabled),
            }
            for item in load_settings(project_id).pronunciation_dictionary
            if item.enabled
        ]
    except Exception:  # pragma: no cover - settings file absent
        entries = []
    entries.sort(key=lambda item: (item["source"], item["pronunciation"]))
    return sha256_bytes(canonical_json_bytes(entries))[:32]


def apply_pronunciations(project_id: str, text: str) -> str:
    """Spoken text for a given pronunciation dictionary revision."""

    from dubbing_service.app.services.production import apply_pronunciation_dictionary

    return apply_pronunciation_dictionary(project_id, text)


def canonical_source_hash(segment: Segment, source_text: Optional[str] = None) -> str:
    return text_sha256(source_text if source_text is not None else segment.text)


def transcript_hash(transcript: Optional[str]) -> Optional[str]:
    return text_sha256(transcript) if transcript else None


def attributions_for(segments: Sequence[Segment]) -> dict[str, Attribution]:
    return {
        segment.id: segment.attribution
        for segment in segments
        if segment.attribution is not None
    }
