"""Cross-process ownership for a projects root (Windows-compatible).

AI Audio Studio keeps exactly one canonical, file-backed store per projects
root.  A second *process* writing the same root would break the "atomic write +
rollback" contract, so writers must take this lock first.

Why not PID alone: a reused PID or a crashed process would make the lock look
live forever.  Ownership combines an opaque token, hostname, PID liveness, a
heartbeat, and the lock file's age, and a recovered stale lock is reported so
the operator sees exactly why a lock was stolen.

No administrator rights are required and nothing is left behind after a clean
exit (see :func:`release_all_held`).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Optional

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".ai-audio-studio.lock"
DEFAULT_STALE_SECONDS = 120.0
_ACQUIRE_SETTLE_SECONDS = 0.02


class LockError(RuntimeError):
    """Base class for project-root lock failures."""


class LockBusyError(LockError):
    """Another live owner holds the lock."""

    def __init__(self, message: str, owner: Optional["LockOwner"] = None):
        super().__init__(message)
        self.owner = owner


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check that never kills or signals the target."""

    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class LockOwner:
    token: str
    pid: int
    host: str
    label: str
    root: str
    acquired_at: str
    heartbeat_at: str
    depth: int = 1

    def to_payload(self) -> dict:
        return {
            "token": self.token,
            "pid": self.pid,
            "host": self.host,
            "label": self.label,
            "root": self.root,
            "acquired_at": self.acquired_at,
            "heartbeat_at": self.heartbeat_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "LockOwner":
        return cls(
            token=str(payload.get("token") or ""),
            pid=int(payload.get("pid") or 0),
            host=str(payload.get("host") or ""),
            label=str(payload.get("label") or ""),
            root=str(payload.get("root") or ""),
            acquired_at=str(payload.get("acquired_at") or ""),
            heartbeat_at=str(payload.get("heartbeat_at") or ""),
        )

    def describe(self) -> str:
        return (
            f"pid={self.pid} host={self.host} label={self.label} "
            f"since={self.acquired_at} heartbeat={self.heartbeat_at}"
        )

    def heartbeat_age(self, now: Optional[datetime] = None) -> float:
        try:
            stamp = datetime.fromisoformat(self.heartbeat_at)
        except ValueError:
            return float("inf")
        now = now or _utc_now()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0.0, (now - stamp).total_seconds())

    def is_live(self, *, stale_after_seconds: float) -> bool:
        if self.heartbeat_age() > stale_after_seconds:
            return False
        if self.host == socket.gethostname() and not pid_is_alive(self.pid):
            return False
        return True


_LOCAL_STATE_LOCK = threading.Lock()
_HELD: Dict[str, LockOwner] = {}


def _held_for(path: Path) -> Optional[LockOwner]:
    with _LOCAL_STATE_LOCK:
        return _HELD.get(str(path))


def _remember(path: Path, owner: Optional[LockOwner]) -> None:
    with _LOCAL_STATE_LOCK:
        if owner is None:
            _HELD.pop(str(path), None)
        else:
            _HELD[str(path)] = owner


def release_all_held() -> None:
    """Release every lock this process still holds (registered at exit)."""

    with _LOCAL_STATE_LOCK:
        paths = [Path(item) for item in _HELD]
    for path in paths:
        try:
            lock = ProjectsRootLock(path.parent)
            lock._release_owned(path, missing_ok=True)
        except Exception:  # pragma: no cover - defensive, exit path
            logger.debug("failed to release lock %s", path, exc_info=True)


atexit.register(release_all_held)


class ProjectsRootLock:
    """Exclusive writer lock for one projects root.

    The lock is re-entrant inside a single process so nested service calls do
    not deadlock; cross-process contention is refused fast (or after ``timeout``).
    """

    def __init__(
        self,
        root: Path,
        *,
        stale_after_seconds: float = DEFAULT_STALE_SECONDS,
        label: str = "ai-audio-studio",
    ) -> None:
        self.root = Path(root)
        self.stale_after_seconds = stale_after_seconds
        self.label = label
        self.recovered_stale: Optional[LockOwner] = None

    # ------------------------------------------------------------------ paths
    @property
    def lock_path(self) -> Path:
        return self.root / LOCK_FILENAME

    # ------------------------------------------------------------------ reads
    def read_owner(self) -> Optional[LockOwner]:
        try:
            raw = self.lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        return LockOwner.from_payload(payload)

    def describe_holder(self) -> str:
        local = _held_for(self.lock_path)
        if local is not None:
            return f"本进程已持有（{local.describe()}）"
        owner = self.read_owner()
        if owner is None:
            return "无持有者"
        state = "活跃" if owner.is_live(stale_after_seconds=self.stale_after_seconds) else "已过期"
        return f"{state}：{owner.describe()}"

    @property
    def held_by_this_process(self) -> bool:
        return _held_for(self.lock_path) is not None

    # --------------------------------------------------------------- acquire
    def acquire(self, *, timeout: float = 0.0, poll_interval: float = 0.05) -> LockOwner:
        local = _held_for(self.lock_path)
        if local is not None:
            nested = LockOwner(
                token=local.token,
                pid=local.pid,
                host=local.host,
                label=local.label,
                root=local.root,
                acquired_at=local.acquired_at,
                heartbeat_at=_iso(_utc_now()),
                depth=local.depth + 1,
            )
            _remember(self.lock_path, nested)
            return nested

        deadline = time.monotonic() + max(0.0, timeout)
        last_owner: Optional[LockOwner] = None
        while True:
            owner = self._try_acquire_once()
            if owner is not None:
                return owner
            last_owner = self.read_owner() or last_owner
            if time.monotonic() >= deadline:
                raise LockBusyError(
                    "另一个 AI Audio Studio 进程正在写同一 projects root："
                    f"{self.root}（持有者 {last_owner.describe() if last_owner else '未知'}）。"
                    "请等待其完成或确认其已退出后重试。",
                    owner=last_owner,
                )
            time.sleep(poll_interval)

    def _try_acquire_once(self) -> Optional[LockOwner]:
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.read_owner()
        if self.lock_path.exists():
            if owner is not None and owner.is_live(
                stale_after_seconds=self.stale_after_seconds
            ):
                return None
            if owner is None:
                age = self._lock_age_seconds()
                if age is not None and age <= self.stale_after_seconds:
                    # Unreadable but recent: never steal from an unknown writer.
                    return None
                self.recovered_stale = None
            else:
                self.recovered_stale = owner
            logger.warning(
                "recovering stale projects-root lock at %s (%s)",
                self.lock_path,
                owner.describe() if owner else "unreadable metadata",
            )
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return None

        token = uuid.uuid4().hex
        now = _utc_now()
        candidate = LockOwner(
            token=token,
            pid=os.getpid(),
            host=socket.gethostname(),
            label=self.label,
            root=str(self.root),
            acquired_at=_iso(now),
            heartbeat_at=_iso(now),
        )
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        except OSError:
            return None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(candidate.to_payload(), stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._unlink_quietly()
            return None

        # Two writers can both observe a missing file and race to create it;
        # the winner is decided by who still sees its own token afterwards.
        time.sleep(_ACQUIRE_SETTLE_SECONDS)
        confirmed = self.read_owner()
        if confirmed is None or confirmed.token != token:
            return None
        _remember(self.lock_path, candidate)
        return candidate

    def _lock_age_seconds(self) -> Optional[float]:
        try:
            return max(0.0, time.time() - self.lock_path.stat().st_mtime)
        except OSError:
            return None

    def _unlink_quietly(self) -> None:
        try:
            self.lock_path.unlink()
        except OSError:
            pass

    # --------------------------------------------------------------- release
    def heartbeat(self) -> Optional[LockOwner]:
        local = _held_for(self.lock_path)
        if local is None:
            return None
        refreshed = LockOwner(
            token=local.token,
            pid=local.pid,
            host=local.host,
            label=local.label,
            root=local.root,
            acquired_at=local.acquired_at,
            heartbeat_at=_iso(_utc_now()),
            depth=local.depth,
        )
        tmp = self.lock_path.with_suffix(".hb")
        try:
            tmp.write_text(
                json.dumps(refreshed.to_payload(), ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self.lock_path)
        except OSError:
            if tmp.exists():
                self._unlink_quietly()
        _remember(self.lock_path, refreshed)
        return refreshed

    def release(self, *, missing_ok: bool = True) -> None:
        self._release_owned(self.lock_path, missing_ok=missing_ok)

    def _release_owned(self, path: Path, *, missing_ok: bool) -> None:
        local = _held_for(path)
        if local is None:
            if not missing_ok:
                raise LockError(f"本进程未持有锁：{path}")
            return
        if local.depth > 1:
            _remember(
                path,
                LockOwner(
                    token=local.token,
                    pid=local.pid,
                    host=local.host,
                    label=local.label,
                    root=local.root,
                    acquired_at=local.acquired_at,
                    heartbeat_at=local.heartbeat_at,
                    depth=local.depth - 1,
                ),
            )
            return
        on_disk = self.read_owner()
        if on_disk is not None and on_disk.token != local.token:
            logger.warning("lock %s is owned by another token; not releasing", path)
            _remember(path, None)
            return
        self._unlink_quietly()
        _remember(path, None)

    @contextmanager
    def hold(self, *, timeout: float = 0.0) -> Generator[LockOwner, None, None]:
        owner = self.acquire(timeout=timeout)
        try:
            yield owner
        finally:
            self.release()


def projects_root_lock(
    root: Optional[Path] = None,
    *,
    stale_after_seconds: Optional[float] = None,
    label: str = "ai-audio-studio",
) -> ProjectsRootLock:
    """Build a lock for the configured (or given) projects root."""

    if root is None:
        from dubbing_service.app.core.config import get_config

        root = Path(get_config().projects_root)
    if stale_after_seconds is None:
        try:
            from dubbing_service.app.core.config import get_config

            stale_after_seconds = float(get_config().project_lock_stale_seconds)
        except Exception:
            stale_after_seconds = DEFAULT_STALE_SECONDS
    return ProjectsRootLock(
        Path(root), stale_after_seconds=stale_after_seconds, label=label
    )


def acquire_projects_root(
    root: Optional[Path] = None, *, timeout: float = 0.0, label: str = "ai-audio-studio"
) -> ProjectsRootLock:
    """Acquire the projects-root writer lock or raise :class:`LockBusyError`."""

    lock = projects_root_lock(root, label=label)
    lock.acquire(timeout=timeout)
    return lock
