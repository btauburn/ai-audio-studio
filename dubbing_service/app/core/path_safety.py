"""Canonical identifier and project-asset path validation.

Persisted JSON is treated as untrusted input.  These helpers keep identifiers
usable as lock/lease keys and prevent an asset reference from being re-used as
an arbitrary project-relative filesystem path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Collection


_SIMPLE_COMPONENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

SEGMENT_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
)


def validate_simple_component_id(value: object, *, label: str = "id") -> str:
    """Return a canonical filesystem-safe ID or raise ``ValueError``.

    Restricting IDs to one component is also important for synchronization:
    aliases such as ``alias\\..\\real`` must not address one directory through
    a different in-memory lock or lease key.
    """

    if not isinstance(value, str) or not _SIMPLE_COMPONENT_ID_RE.fullmatch(value):
        raise ValueError(f"invalid {label}: expected one simple path component")
    return value


def resolve_scoped_project_asset(
    project_root: Path,
    rel_path: object,
    *,
    required_subdir: str,
    allowed_extensions: Collection[str] | None = None,
) -> Path:
    """Resolve a canonical relative asset path below one required directory.

    Both slash forms are accepted for compatibility with historical Windows
    JSON.  Dot components, empty components and absolute paths are rejected
    before resolving so an alias cannot pass merely because it lands back under
    the allowed directory.
    """

    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("asset path must be a non-empty relative string")

    normalized = rel_path.replace("\\", "/")
    raw_parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("asset path contains a non-canonical component")

    relative = Path(*raw_parts)
    if relative.is_absolute() or relative.drive:
        raise ValueError("asset path must be relative")

    required_parts = required_subdir.replace("\\", "/").split("/")
    if raw_parts[: len(required_parts)] != required_parts:
        raise ValueError(f"asset path must be below {required_subdir}")
    if len(raw_parts) <= len(required_parts):
        raise ValueError("asset path must include a file name")

    root_resolved = project_root.resolve()
    scoped_root = (root_resolved / Path(*required_parts)).resolve()
    if not scoped_root.is_relative_to(root_resolved):
        raise ValueError(f"asset directory escapes project root: {required_subdir}")
    candidate = (root_resolved / relative).resolve()
    if not candidate.is_relative_to(scoped_root):
        raise ValueError(f"asset path escapes {required_subdir}")

    if allowed_extensions is not None:
        allowed = {str(item).lower() for item in allowed_extensions}
        if candidate.suffix.lower() not in allowed:
            raise ValueError("asset path has an unsupported file extension")

    return candidate
