"""Transactional commit helper for automatic text segmentation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.enums import SegmentStatus
from dubbing_service.app.models.project import Project, Role, Segment
from dubbing_service.app.storage.atomic import ensure_metadata_writable, replace_metadata_file
from dubbing_service.app.storage.projects import load_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments, project_segments_lock
from dubbing_service.app.services.takes import mark_slots_dirty


@dataclass(frozen=True)
class SegmentationCommitResult:
    project: Project
    new_roles_count: int


@dataclass(frozen=True)
class RoleDeletionCommitResult:
    prompt_to_delete: str | None
    changed_segments: int


@dataclass(frozen=True)
class RoleSynthesisUpdateCommitResult:
    """Result of changing a role setting that affects generated audio."""

    role: Role
    changed: bool
    changed_segments: int
    segment_audio_to_delete: tuple[str, ...]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_stage(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".stage")
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _replace_files_transactionally(files: Sequence[tuple[Path, bytes]]) -> None:
    """Promote fully-written files and restore all originals on any failure.

    A filesystem cannot atomically replace several paths as one operation. This
    helper stages every new file first, keeps same-directory backups, then rolls
    back already-promoted paths if any later promotion fails.
    """

    for directory in {target.parent for target, _ in files}:
        ensure_metadata_writable(directory)
    stages: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    promoted: list[Path] = []
    backups_to_preserve: set[Path] = set()
    transaction_id = uuid.uuid4().hex

    try:
        for target, data in files:
            stages[target] = _write_stage(target, data)

        for target, _ in files:
            if target.is_file():
                backup = target.with_name(f".{target.name}.{transaction_id}.backup")
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None

        for target, _ in files:
            replace_metadata_file(stages[target], target)
            promoted.append(target)

    except Exception:
        for target in reversed(promoted):
            backup = backups.get(target)
            try:
                if backup is not None and backup.is_file():
                    replace_metadata_file(backup, target)
                elif target.exists():
                    target.unlink()
            except OSError:
                if backup is not None:
                    backups_to_preserve.add(backup)
                # Continue restoring the remaining files and retain any backup
                # that could not be promoted back into place.
                continue
        raise
    finally:
        for stage in stages.values():
            try:
                if stage.is_file():
                    stage.unlink()
            except OSError:
                pass
        for backup in backups.values():
            if backup is None:
                continue
            if backup in backups_to_preserve:
                continue
            try:
                if backup.is_file():
                    backup.unlink()
            except OSError:
                pass


def commit_auto_segmentation(
    project_id: str,
    *,
    raw_text: str,
    segments: Sequence[Segment],
    new_roles: Iterable[Role],
) -> SegmentationCommitResult:
    """Commit raw text, segments, and newly discovered roles as one transaction."""

    project_root = safe_project_root(project_id)
    project_path = project_root / "project.json"
    segments_path = project_root / "segments.json"
    raw_path = project_root / "text" / "raw.txt"

    # Keep lock ordering stable anywhere this transaction is used.
    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            existing_ids = {role.id for role in project.roles}
            added = 0
            for role in new_roles:
                if role.id in existing_ids:
                    continue
                project.roles.append(role)
                existing_ids.add(role.id)
                added += 1
            project.updated_at = datetime.now(timezone.utc)

            _replace_files_transactionally(
                [
                    (project_path, _json_bytes(project.model_dump(mode="json"))),
                    (segments_path, _json_bytes([s.model_dump(mode="json") for s in segments])),
                    (raw_path, raw_text.encode("utf-8")),
                ]
            )

    return SegmentationCommitResult(project=project, new_roles_count=added)


def commit_segment_replacement(
    project_id: str,
    *,
    segments: Sequence[Segment],
    new_roles: Iterable[Role],
) -> SegmentationCommitResult:
    """Atomically replace segments and merge roles for manual JSON import.

    Audio garbage collection must run only after this metadata transaction
    succeeds.  That ordering guarantees a failed save never leaves the old
    metadata pointing at files that were already deleted.
    """

    project_root = safe_project_root(project_id)
    project_path = project_root / "project.json"
    segments_path = project_root / "segments.json"

    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            existing_ids = {role.id for role in project.roles}
            added = 0
            for role in new_roles:
                if role.id in existing_ids:
                    continue
                project.roles.append(role)
                existing_ids.add(role.id)
                added += 1
            project.updated_at = datetime.now(timezone.utc)

            _replace_files_transactionally(
                [
                    (project_path, _json_bytes(project.model_dump(mode="json"))),
                    (segments_path, _json_bytes([s.model_dump(mode="json") for s in segments])),
                ]
            )

    return SegmentationCommitResult(project=project, new_roles_count=added)


def commit_role_deletion(project_id: str, role_id: str) -> RoleDeletionCommitResult:
    """Remove a role and every segment reference as one metadata transaction."""

    project_root = safe_project_root(project_id)
    project_path = project_root / "project.json"
    segments_path = project_root / "segments.json"

    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            target = next((role for role in project.roles if role.id == role_id), None)
            if target is None:
                raise LookupError(f"项目 {project_id} 中未找到角色 {role_id}")

            prompt_to_delete: str | None = None
            raw_prompt = str(getattr(target, "speaker_audio", "") or "").strip()
            if raw_prompt.replace("\\", "/").startswith("roles/prompts/"):
                prompt_to_delete = raw_prompt

            project.roles = [role for role in project.roles if role.id != role_id]
            project.updated_at = datetime.now(timezone.utc)

            segments = load_segments(project_id)
            changed_segments = 0
            for segment in segments:
                if segment.role_id != role_id:
                    continue
                segment.role_id = None
                segment.updated_at = datetime.now(timezone.utc)
                changed_segments += 1

            _replace_files_transactionally(
                [
                    (project_path, _json_bytes(project.model_dump(mode="json"))),
                    (segments_path, _json_bytes([s.model_dump(mode="json") for s in segments])),
                ]
            )

    return RoleDeletionCommitResult(
        prompt_to_delete=prompt_to_delete,
        changed_segments=changed_segments,
    )


_ROLE_SYNTHESIS_FIELDS = frozenset(
    {
        "speaker_audio",
        "speaker_voice_id",
        "tts_backend",
        "qwen3_speaker",
        "qwen3_language",
        "voice_description",
        "style_instruction",
        "preferred_speed_min",
        "preferred_speed_max",
        "voice_source",
        "voice_license_name",
        "voice_commercial_use",
        "voice_rights_note",
    }
)

_ROLE_GUIDANCE_FIELDS = frozenset({
    "preferred_speed_min", "preferred_speed_max", "voice_source",
    "voice_license_name", "voice_commercial_use", "voice_rights_note",
})


def commit_role_synthesis_update(
    project_id: str,
    role_id: str,
    updates: dict[str, object],
) -> RoleSynthesisUpdateCommitResult:
    """Atomically update role synthesis settings and mark its slots stale.

    Existing Take audio remains playable and recoverable.  The next synthesis
    creates a replacement Take instead of deleting prior output.
    """

    project_root = safe_project_root(project_id)
    project_path = project_root / "project.json"
    segments_path = project_root / "segments.json"

    normalized_updates: dict[str, object] = {}
    for field_name, value in updates.items():
        if field_name not in _ROLE_SYNTHESIS_FIELDS:
            continue
        if field_name in {"preferred_speed_min", "preferred_speed_max"}:
            normalized_updates[field_name] = float(value) if value is not None else 1.0
        elif field_name == "voice_commercial_use":
            normalized_updates[field_name] = bool(value) if value is not None else None
        else:
            normalized = str(value).strip() if value is not None else None
            normalized_updates[field_name] = normalized or None

    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            project = load_project(project_id)
            target = next((role for role in project.roles if role.id == role_id), None)
            if target is None:
                raise LookupError(f"项目 {project_id} 中未找到角色 {role_id}")

            changed = any(
                getattr(target, field_name, None) != value
                for field_name, value in normalized_updates.items()
            )
            if not changed:
                return RoleSynthesisUpdateCommitResult(
                    role=target,
                    changed=False,
                    changed_segments=0,
                    segment_audio_to_delete=(),
                )

            synthesis_changed = any(
                field_name not in _ROLE_GUIDANCE_FIELDS
                and getattr(target, field_name, None) != value
                for field_name, value in normalized_updates.items()
            )
            for field_name, value in normalized_updates.items():
                setattr(target, field_name, value)

            now = datetime.now(timezone.utc)
            project.updated_at = now
            segments = load_segments(project_id)
            changed_segments = 0
            audio_to_delete: set[str] = set()
            for segment in segments:
                if segment.role_id != role_id:
                    continue
                if synthesis_changed:
                    mark_slots_dirty(segment, "a", "b")
                    segment.status = SegmentStatus.PENDING
                    segment.updated_at = now
                    changed_segments += 1

            _replace_files_transactionally(
                [
                    (project_path, _json_bytes(project.model_dump(mode="json"))),
                    (segments_path, _json_bytes([s.model_dump(mode="json") for s in segments])),
                ]
            )

    return RoleSynthesisUpdateCommitResult(
        role=target,
        changed=True,
        changed_segments=changed_segments,
        segment_audio_to_delete=tuple(sorted(audio_to_delete)),
    )
