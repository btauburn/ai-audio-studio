"""Crash-safe multi-file commits for audiobook project state (Phase 13).

AI Audio Studio's authority stays file-backed: ``project.json`` and
``segments.json`` under the project root.  A render pass has to update both, and
a filesystem cannot replace two files as one operation.  Without help, a process
death between the two promotions leaves a *hybrid* state (new segments against
an old project, or the reverse).

The commit below is the smallest mechanism that makes the candidate invariant
true for hard process death:

1. stage every new file next to its target and ``fsync`` it;
2. copy every existing target to a same-directory backup;
3. write and ``fsync`` one intent marker naming the transaction and its targets;
4. promote every staged file;
5. drop the backups, then drop the marker.

Recovery is monotonic: **an intent marker means the transaction did not finish**,
so every listed target is restored from its backup (or removed when it did not
exist before).  The result is always the complete OLD state; a fully promoted
transaction that was never marked done also rolls back to OLD, which the
acceptance criterion explicitly allows.

A rollback-only design (no marker) was measured first and is insufficient: an
in-process rollback cannot run when the process is killed between promotions, so
the marker is required rather than speculative.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.project import Project, Segment
from dubbing_service.app.storage.projects import project_meta_lock
from dubbing_service.app.storage.segments import project_segments_lock

MARKER_PREFIX = ".commit-"
MARKER_SUFFIX = ".json"
STAGE_SUFFIX = ".stage"
BACKUP_SUFFIX = ".backup"


class PersistenceError(RuntimeError):
    """Raised when a project transaction cannot be completed."""


@dataclass
class CommitReport:
    transaction_id: str
    targets: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"transaction_id": self.transaction_id, "targets": list(self.targets)}


@dataclass
class RecoveryReport:
    """What :func:`recover_project_state` had to repair."""

    recovered: List[str] = field(default_factory=list)
    rolled_back: List[str] = field(default_factory=list)
    stages_removed: int = 0
    markers_removed: int = 0
    leftovers: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when recovery completed: nothing was left in an unknown state."""

        return not self.leftovers

    @property
    def clean(self) -> bool:
        """True when there was nothing to repair at all."""

        return not self.recovered and not self.leftovers

    def as_dict(self) -> Dict[str, object]:
        return {
            "recovered": list(self.recovered),
            "rolled_back": list(self.rolled_back),
            "stages_removed": self.stages_removed,
            "markers_removed": self.markers_removed,
            "leftovers": list(self.leftovers),
            "clean": self.clean,
            "ok": self.ok,
        }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync; not available on every platform."""

    if os.name == "nt":  # pragma: no cover - Windows cannot fsync a directory
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_stage(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=STAGE_SUFFIX
    )
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


def marker_paths(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob(f"{MARKER_PREFIX}*{MARKER_SUFFIX}"))


def commit_files(files: Sequence[Tuple[Path, bytes]], *, root: Path) -> CommitReport:
    """Promote several files under ``root`` as one recoverable transaction."""

    if not files:
        raise PersistenceError("事务没有任何文件")

    transaction_id = uuid.uuid4().hex
    stages: Dict[Path, Path] = {}
    backups: Dict[Path, Optional[Path]] = {}
    promoted: List[Path] = []
    marker = Path(root) / f"{MARKER_PREFIX}{transaction_id}{MARKER_SUFFIX}"

    try:
        for target, data in files:
            stages[target] = _write_stage(target, data)

        for target, _ in files:
            if target.is_file():
                backup = target.with_name(f".{target.name}.{transaction_id}{BACKUP_SUFFIX}")
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None

        payload = {
            "transaction_id": transaction_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "targets": [
                {
                    "name": target.name,
                    "existed": backups[target] is not None,
                }
                for target, _ in files
            ],
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker_fd, marker_tmp_name = tempfile.mkstemp(
            dir=marker.parent, prefix=f"{MARKER_PREFIX}{transaction_id}.", suffix=".tmp"
        )
        marker_tmp = Path(marker_tmp_name)
        with os.fdopen(marker_fd, "wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(marker_tmp, marker)
        _fsync_dir(Path(root))

        for target, _ in files:
            os.replace(stages[target], target)
            promoted.append(target)
    except Exception:
        unrestored: set[Optional[Path]] = set()
        for target in reversed(promoted):
            backup = backups.get(target)
            try:
                if backup is not None and backup.is_file():
                    os.replace(backup, target)
                elif target.exists():
                    target.unlink()
            except OSError:
                unrestored.add(backup)
                continue
        if unrestored:
            # Keep the intent marker and the surviving backups so the next
            # recovery pass can finish the rollback instead of guessing.
            _cleanup(stages, set(), marker, keep_marker=True)
        else:
            _cleanup(stages, {path for path in backups.values() if path}, marker)
        raise

    _finalize(stages, {path for path in backups.values() if path}, marker)
    return CommitReport(
        transaction_id=transaction_id, targets=[target.name for target, _ in files]
    )


def _remove_files(paths) -> None:  # noqa: ANN001 - internal helper
    for path in paths:
        if path is None:
            continue
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def _finalize(
    stages: Dict[Path, Path], backups: set[Optional[Path]], marker: Path
) -> None:
    """Publish the transaction, then sweep.

    The marker is removed *before* the backups: from that moment the promoted
    files are the complete committed state, so an interruption during the sweep
    (which only deletes now-disposable files) can no longer be mistaken for an
    unfinished transaction.
    """

    try:
        if marker.is_file():
            marker.unlink()
    except OSError:
        pass
    _fsync_dir(marker.parent)
    _remove_files(stages.values())
    _remove_files(backups)


def _cleanup(
    stages: Dict[Path, Path],
    backups: set[Optional[Path]],
    marker: Path,
    *,
    keep_marker: bool = False,
) -> None:
    """Sweep after an in-process rollback back to the complete OLD state."""

    _remove_files(stages.values())
    _remove_files(backups)
    if keep_marker:
        return
    try:
        if marker.is_file():
            marker.unlink()
    except OSError:
        pass


def recover_project_state(project_id: str) -> RecoveryReport:
    """Restore complete OLD state after an interrupted commit."""

    root = safe_project_root(project_id)
    return recover_root(root)


def recover_root(root: Path) -> RecoveryReport:
    root = Path(root)
    report = RecoveryReport()
    for marker in marker_paths(root):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            report.leftovers.append(str(marker))
            continue

        transaction_id = str(payload.get("transaction_id") or "")
        targets = payload.get("targets")
        if not isinstance(targets, list) or not transaction_id:
            report.leftovers.append(str(marker))
            continue

        for entry in targets:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            target = root / str(entry["name"])
            backup = target.with_name(
                f".{target.name}.{transaction_id}{BACKUP_SUFFIX}"
            )
            try:
                if backup.is_file():
                    os.replace(backup, target)
                elif not entry.get("existed", False):
                    if target.is_file():
                        target.unlink()
                else:
                    # The marker says this target existed, but its backup is
                    # gone: claiming a rollback here would be a lie, so the
                    # problem is reported instead of guessed away.
                    report.leftovers.append(f"{target.name} (backup missing)")
                    continue
            except OSError:
                report.leftovers.append(str(target))
                continue
            report.recovered.append(target.name)
            report.rolled_back.append(target.name)

        try:
            marker.unlink()
            report.markers_removed += 1
        except OSError:
            report.leftovers.append(str(marker))

    for stale in root.glob(f".*{STAGE_SUFFIX}"):
        try:
            stale.unlink()
            report.stages_removed += 1
        except OSError:
            report.leftovers.append(str(stale))
    for stale in root.glob(f".*{BACKUP_SUFFIX}"):
        try:
            stale.unlink()
        except OSError:
            report.leftovers.append(str(stale))
    for stale in root.glob(f"{MARKER_PREFIX}*.tmp"):
        try:
            stale.unlink()
            report.stages_removed += 1
        except OSError:
            report.leftovers.append(str(stale))
    return report


def commit_project_state(
    project_id: str, *, project: Project, segments: Sequence[Segment]
) -> CommitReport:
    """Persist ``project.json`` and ``segments.json`` as one transaction."""

    root = safe_project_root(project_id)
    files = [
        (root / "project.json", _json_bytes(project.model_dump(mode="json"))),
        (
            root / "segments.json",
            _json_bytes([segment.model_dump(mode="json") for segment in segments]),
        ),
    ]
    with project_meta_lock(project_id):
        with project_segments_lock(project_id):
            return commit_files(files, root=root)


def verify_project_state(project_id: str) -> Tuple[bool, List[str]]:
    """Validate that the on-disk pair is complete and mutually consistent.

    Returns ``(ok, problems)``.  A hybrid state (segments referencing chapters
    that the project does not define, or a missing/malformed file) is reported as
    a problem so tests can assert "never hybrid" directly.
    """

    root = safe_project_root(project_id)
    problems: List[str] = []

    project_path = root / "project.json"
    segments_path = root / "segments.json"
    if not project_path.is_file():
        problems.append("project.json is missing")
    if not segments_path.is_file():
        problems.append("segments.json is missing")
    if problems:
        return False, problems

    try:
        project = Project.model_validate_json(project_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - the problem is the message
        return False, [f"project.json is unreadable: {exc}"]
    try:
        raw_segments = json.loads(segments_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"segments.json is unreadable: {exc}"]
    if not isinstance(raw_segments, list):
        return False, ["segments.json is not a list"]

    try:
        segments = [Segment.model_validate(item) for item in raw_segments]
    except Exception as exc:  # noqa: BLE001
        return False, [f"segments.json contains invalid records: {exc}"]

    chapter_ids = {chapter.id for chapter in project.chapters}
    role_ids = {role.id for role in project.roles}
    for segment in segments:
        if segment.chapter_id and segment.chapter_id not in chapter_ids:
            problems.append(
                f"segment {segment.id} references unknown chapter {segment.chapter_id}"
            )
        if segment.role_id and segment.role_id not in role_ids:
            problems.append(
                f"segment {segment.id} references unknown role {segment.role_id}"
            )
    return not problems, problems


def pending_markers(project_id: str) -> List[str]:
    return [
        path.name for path in marker_paths(safe_project_root(project_id))
    ]
