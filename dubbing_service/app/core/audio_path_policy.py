"""Shared path policy for audio files handled by local TTS workers.

HTTP values must never be passed directly to filesystem APIs.  This module
normalizes request values as strings, enumerates files from administrator-owned
roots, and only returns paths that were discovered inside those roots.  The
enumeration step also prevents a symlink or Windows junction from escaping an
allowed root.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Iterable, Iterator, Sequence


AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)


class AudioPathPolicyError(ValueError):
    """Base error for rejected or unavailable audio paths."""


class InvalidAudioPathError(AudioPathPolicyError):
    """The supplied path violates the audio path policy."""


class AudioFileNotFoundError(AudioPathPolicyError):
    """The supplied path is valid in shape but no managed file exists."""


def _deduplicate_roots(roots: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except (OSError, RuntimeError):
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen:
            result.append(resolved)
            seen.add(key)
    return tuple(result)


def managed_audio_roots(repo_root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Return the local roots from which TTS workers may read audio.

    ``DUBBING_PROJECTS_ROOT`` preserves support for project data stored outside
    the repository.  Advanced standalone users can explicitly add trusted
    roots with ``DUBBING_TTS_ALLOWED_AUDIO_ROOTS`` (``os.pathsep`` separated).
    Both settings are process-owner configuration, not HTTP input.
    """

    repo = Path(repo_root).resolve()
    configured_projects = os.environ.get("DUBBING_PROJECTS_ROOT", "").strip()
    if not configured_projects:
        config_path = repo / "dubbing_service" / "config.json"
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            config_data = {}
        if isinstance(config_data, dict):
            configured_value = config_data.get("projects_root")
            if isinstance(configured_value, str):
                configured_projects = configured_value.strip()
    if configured_projects:
        projects = Path(configured_projects)
        if not projects.is_absolute():
            projects = repo / projects
    else:
        projects = repo / "dubbing_projects"

    roots: list[Path] = [
        projects,
        repo / "dubbing_service" / "voices",
        repo / "VioceTimbre",
        repo / "examples",
        repo / "cosyvoice" / "asset",
        repo / "prompts",
        repo / "temp",
        repo / "outputs",
        repo / "service" / "prompts",
        repo / "service" / "temp",
        repo / "service" / "outputs",
    ]

    configured_extra = os.environ.get("DUBBING_TTS_ALLOWED_AUDIO_ROOTS", "")
    for raw_root in configured_extra.split(os.pathsep):
        raw_root = raw_root.strip()
        if not raw_root:
            continue
        extra = Path(raw_root)
        if not extra.is_absolute():
            extra = repo / extra
        roots.append(extra)

    return _deduplicate_roots(roots)


def _is_link_or_junction(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True

    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if reparse_flag and attributes & reparse_flag:
            return True
    except OSError:
        return True

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            return bool(is_junction())
        except OSError:
            return True
    return False


def _iter_root_files(root: Path, *, recursive: bool) -> Iterator[Path]:
    if not root.is_dir():
        return

    if not recursive:
        try:
            entries = root.iterdir()
            for entry in entries:
                yield entry
        except OSError:
            return
        return

    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for current_dir, dirnames, filenames in walker:
            current = Path(current_dir)
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_link_or_junction(current / name)
            ]
            for filename in filenames:
                yield current / filename
    except OSError:
        return


def _validate_request_value(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise InvalidAudioPathError("audio path must be a string")

    raw = value.strip()
    if not raw or len(raw) > 4096 or "\x00" in raw:
        raise InvalidAudioPathError("invalid audio path")

    slash_normalized = raw.replace("\\", "/")
    parts = slash_normalized.split("/")
    leading_root_parts = 2 if slash_normalized.startswith("//") else int(
        slash_normalized.startswith("/")
    )
    path_parts = parts[leading_root_parts:]
    if any(part in {"", ".", ".."} for part in path_parts[:-1]):
        raise InvalidAudioPathError("audio path traversal is not allowed")
    if not path_parts or path_parts[-1] in {"", ".", ".."}:
        raise InvalidAudioPathError("invalid audio filename")

    filename = path_parts[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise InvalidAudioPathError("unsupported audio file extension")
    return raw, filename


def _path_key(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _is_key_under(candidate: str, root: str) -> bool:
    try:
        return os.path.commonpath((candidate, root)) == root
    except ValueError:
        return False


def _request_keys(
    raw: str,
    *,
    roots: Sequence[Path],
    relative_bases: Sequence[Path],
    allow_bare_name: bool,
) -> tuple[set[str], bool]:
    raw_is_absolute = os.path.isabs(raw)
    contains_separator = "/" in raw or "\\" in raw
    keys: set[str] = set()

    if raw_is_absolute:
        keys.add(_path_key(raw))
    elif contains_separator:
        for base in relative_bases:
            keys.add(_path_key(os.path.join(str(base), raw)))
    elif allow_bare_name:
        for root in roots:
            keys.add(_path_key(os.path.join(str(root), raw)))
    else:
        raise InvalidAudioPathError("an audio path is required")

    root_keys = tuple(_path_key(root) for root in roots)
    if not any(_is_key_under(key, root) for key in keys for root in root_keys):
        raise InvalidAudioPathError("audio path is outside allowed roots")
    return keys, not contains_separator


def resolve_existing_audio_file(
    value: str,
    *,
    roots: Iterable[str | os.PathLike[str]],
    relative_bases: Iterable[str | os.PathLike[str]] = (),
    recursive: bool = True,
    allow_bare_name: bool = False,
) -> Path:
    """Resolve an HTTP-supplied audio path to a managed existing file.

    The request value is only used for lexical comparison.  Filesystem access
    starts from resolved trusted roots and returns a path discovered there.
    """

    raw, requested_name = _validate_request_value(value)
    trusted_roots = _deduplicate_roots(Path(root) for root in roots)
    if not trusted_roots:
        raise AudioFileNotFoundError("no managed audio roots are available")
    trusted_bases = _deduplicate_roots(Path(base) for base in relative_bases)
    request_keys, bare_name = _request_keys(
        raw,
        roots=trusted_roots,
        relative_bases=trusted_bases,
        allow_bare_name=allow_bare_name,
    )

    requested_name_key = os.path.normcase(requested_name)
    unsafe_match = False
    for root in trusted_roots:
        root_key = _path_key(root)
        for entry in _iter_root_files(root, recursive=recursive):
            if os.path.normcase(entry.name) != requested_name_key:
                continue
            if entry.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            entry_key = _path_key(entry)
            if entry_key not in request_keys and not (
                bare_name and entry.parent == root
            ):
                continue

            try:
                resolved = entry.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            resolved_key = _path_key(resolved)
            if not _is_key_under(resolved_key, root_key):
                unsafe_match = True
                continue
            if not resolved.is_file():
                continue
            return resolved

    if unsafe_match:
        raise InvalidAudioPathError("audio link escapes an allowed root")
    raise AudioFileNotFoundError("managed audio file does not exist")


def resolve_existing_audio_name(
    directory: str | os.PathLike[str],
    filename: str,
) -> Path:
    """Resolve a simple download filename under one managed directory."""

    raw, parsed_name = _validate_request_value(filename)
    if raw != parsed_name:
        raise InvalidAudioPathError("download path must be a simple filename")
    root = Path(directory).resolve()
    return resolve_existing_audio_file(
        parsed_name,
        roots=(root,),
        recursive=False,
        allow_bare_name=True,
    )
