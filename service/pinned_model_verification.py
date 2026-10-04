"""Reuse model SHA256 verification only while Windows denies file mutation.

Read handles share READ only: existing writers prevent pin acquisition, and
retained pins deny subsequent write/delete/rename. Other platforms and paths
with reparse components always use the existing full verification policy.
No timestamps are used to trust bytes. clear() releases pins for model updates.
"""
from __future__ import annotations

import copy
import ctypes
import logging
import os
from collections import OrderedDict
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from scripts.model_preflight import ArtifactStatus, artifact_target, load_manifest, verify_artifact

logger = logging.getLogger(__name__)


class _FileInformation(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("created", wintypes.FILETIME),
                ("accessed", wintypes.FILETIME), ("written", wintypes.FILETIME),
                ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]


class _ReadPin:
    def __init__(self, path: Path):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        self._close = kernel.CloseHandle
        self._close.argtypes = [wintypes.HANDLE]
        self._close.restype = wintypes.BOOL
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING; do not follow a final
        # reparse point, even if a caller raced the path-policy check.
        self._handle = create(str(path), 0x80000000, 1, None, 3, 0x08200080, None)
        if self._handle == ctypes.c_void_p(-1).value:
            self._handle = None
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            get_info = kernel.GetFileInformationByHandle
            get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileInformation)]
            get_info.restype = wintypes.BOOL
            info = _FileInformation()
            if not get_info(self._handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & (0x400 | 0x10):
                raise OSError("only materialized regular files can be pinned")
            self.identity = (info.volume, info.index_high, info.index_low)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._handle is not None:
            if not self._close(self._handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _windows_pins_available() -> bool:
    return os.name == "nt"


def _cacheable_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    # Unlike full preflight, pinned reuse also excludes in-root directory
    # junctions: retargeting a junction must not redirect a cached identity.
    for component in (path, *path.parents):
        if component.is_symlink() or getattr(component.lstat(), "st_file_attributes", 0) & 0x400:
            return False
        if component == root:
            break
    return True


def _full_verification(manifest: dict[str, Any], root: Path) -> tuple[ArtifactStatus, ...]:
    statuses = tuple(verify_artifact(manifest, artifact, project_root=root, verify_hash=True)
                     for artifact in manifest["files"])
    invalid = [status.path for status in statuses if not status.ok]
    if invalid:
        raise ValueError("character model failed pinned integrity verification: " + ", ".join(invalid))
    return statuses


def _close_pins(pins: dict[Path, _ReadPin]) -> None:
    error = None
    while pins:
        _, pin = pins.popitem()
        try:
            pin.close()
        except OSError as exc:
            # One bad handle must not strand the remaining valid handles.
            if error is None:
                error = exc
    if error is not None:
        raise error


@dataclass(frozen=True)
class ModelVerification:
    manifest: dict[str, Any]
    artifacts: tuple[ArtifactStatus, ...]
    cache_hit: bool
    pinned: bool


@dataclass
class _Entry:
    manifest: dict[str, Any]
    artifacts: tuple[ArtifactStatus, ...]
    pins: dict[Path, _ReadPin]

    def close(self) -> None:
        _close_pins(self.pins)


class PinnedModelVerificationCache:
    """Thread-safe, process-local verification lifetime; never persists trust."""
    def __init__(self):
        self._entries: OrderedDict[tuple[Path, Path], _Entry] = OrderedDict()
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            error = None
            while self._entries:
                _, entry = self._entries.popitem()
                try:
                    entry.close()
                except OSError as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def _matches(self, entry: _Entry, root: Path) -> bool:
        try:
            for path, retained in entry.pins.items():
                if not _cacheable_path(path, root):
                    return False
                # Verify the CURRENT namespace resolves to the retained file,
                # including volume+file ID. Opening this probe denies mutation
                # too; no size/time heuristic authorizes a cache hit.
                with _ReadPin(path) as probe:
                    if probe.identity != retained.identity:
                        return False
            for item in entry.manifest["files"]:
                target = artifact_target(entry.manifest, item, root)
                lexical = Path(os.path.abspath(root / item["path"]))
                # artifact_target resolves parents. Check the original lexical
                # route too, so an in-root junction is never hidden by resolution.
                if target not in entry.pins or not _cacheable_path(lexical, root):
                    return False
                if not verify_artifact(entry.manifest, item, project_root=root, verify_hash=False).ok:
                    return False
            return True
        except OSError:
            return False

    def verify(self, manifest_path: Path, *, project_root: Path) -> ModelVerification:
        root = project_root.resolve()
        path = Path(os.path.abspath(manifest_path))
        key = (root, path)
        with self._lock:
            if not _windows_pins_available():
                manifest = load_manifest(path)
                return ModelVerification(manifest, _full_verification(manifest, root), False, False)
            cached = self._entries.get(key)
            if cached is not None:
                if self._matches(cached, root):
                    self._entries.move_to_end(key)
                    return ModelVerification(copy.deepcopy(cached.manifest), copy.deepcopy(cached.artifacts), True, True)
                self._entries.pop(key).close()
            pins: dict[Path, _ReadPin] = {}
            retained = False
            try:
                if not _cacheable_path(path, root):
                    raise OSError("manifest path is not eligible for read pinning")
                pins[path] = _ReadPin(path)
                manifest = load_manifest(path)
                for artifact in manifest["files"]:
                    target = artifact_target(manifest, artifact, root)
                    lexical = Path(os.path.abspath(root / artifact["path"]))
                    if not _cacheable_path(lexical, root) or not _cacheable_path(target, root):
                        raise OSError("artifact path is not eligible for read pinning")
                    status = verify_artifact(manifest, artifact, project_root=root, verify_hash=False)
                    if not status.ok:
                        raise ValueError("character model failed pinned integrity verification: " + status.path)
                    if target not in pins:
                        pins[target] = _ReadPin(target)
                # All handles are acquired BEFORE any artifact hash. Files are
                # immutable for the entire hash and cached-success lifetime.
                entry = _Entry(manifest, (), pins)
                if not self._matches(entry, root):
                    raise OSError("model namespace changed during read pinning")
                entry.artifacts = _full_verification(manifest, root)
                while len(self._entries) >= 4:
                    _, evicted = self._entries.popitem(last=False)
                    evicted.close()
                self._entries[key] = entry
                retained = True
                return ModelVerification(copy.deepcopy(manifest), copy.deepcopy(entry.artifacts), False, True)
            except OSError as exc:
                # Availability fallback: release EVERY tentative pin first,
                # then run the original full-hash verifier. Never cache it.
                _close_pins(pins)
                logger.info("Model read pin unavailable; performing full SHA256 verification: %s", exc)
                manifest = load_manifest(path)
                return ModelVerification(manifest, _full_verification(manifest, root), False, False)
            finally:
                if not retained:
                    _close_pins(pins)
