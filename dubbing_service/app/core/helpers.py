"""Shared helper utilities used across multiple routers.

Consolidates duplicate helper functions that were previously copy-pasted
in segments.py, synthesis.py, projects.py, subtitles.py, settings.py,
llm_configs.py, llm_providers.py, and roles.py.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from fastapi import HTTPException, status

from dubbing_service.app.core.config import safe_project_root


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_truthy(name: str) -> bool:
    """Check if an environment variable is set to a truthy value."""
    v = os.getenv(name)
    if not v:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def expose_secrets() -> bool:
    """Never expose credentials through HTTP responses or worker logs.

    Older builds honored ``DUBBING_EXPOSE_SECRETS`` as a debug escape hatch.
    Because the platform may listen on a trusted LAN, an inherited environment
    variable could otherwise disclose every configured API credential. Debug
    secret inspection now belongs in local files/process environment only.
    """
    return False


# ---------------------------------------------------------------------------
# Project root helpers
# ---------------------------------------------------------------------------

def project_root(project_id: str, *, ensure_dir: bool = False) -> Path:
    """Return the project root directory, raising HTTPException on invalid ID.

    Args:
        project_id: The project identifier.
        ensure_dir: If True, create the directory tree if it doesn't exist.
    """
    try:
        root = safe_project_root(project_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法项目 ID",
        ) from exc
    if ensure_dir:
        root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Numeric env config helpers
# ---------------------------------------------------------------------------


def resolve_project_relative_file(base_root: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None

    try:
        base_root_resolved = base_root.resolve()
        target = (base_root / str(rel_path)).resolve()
    except Exception:
        return None

    if not target.is_relative_to(base_root_resolved):
        return None
    if not target.is_file():
        return None
    return target


def resolve_voice_audio_path(audio_path: str | None) -> Path | None:
    if not audio_path:
        return None

    from dubbing_service.app.storage.voices import voices_audio_dir

    base_dir = voices_audio_dir()
    try:
        base_dir_resolved = base_dir.resolve()
        target = Path(audio_path)
        if not target.is_absolute():
            target = (base_dir / str(audio_path)).resolve()
        else:
            target = target.resolve()
    except Exception:
        return None

    if not target.is_relative_to(base_dir_resolved):
        return None
    if not target.is_file():
        return None
    return target


def resolve_role_reference_audio(
    project_id: str,
    speaker_audio: str | None,
    speaker_voice_id: str | None,
) -> Path | None:
    try:
        base_root = safe_project_root(project_id)
    except Exception:
        base_root = None

    if base_root is not None:
        target = resolve_project_relative_file(base_root, speaker_audio)
        if target is not None:
            return target

    voice_id = str(speaker_voice_id or "").strip()
    if not voice_id:
        return None

    try:
        from dubbing_service.app.storage.voices import load_voices

        voices = load_voices()
    except Exception:
        voices = []

    voice = next(
        (item for item in voices if item and getattr(item, "id", None) == voice_id),
        None,
    )
    if voice is None:
        return None

    return resolve_voice_audio_path(getattr(voice, "audio_path", None))

def get_keep_count(env_name: str, default: int) -> int:
    """Read an integer retention count from an environment variable."""
    try:
        v = (os.getenv(env_name) or "").strip()
        if not v:
            return default
        n = int(v)
        return max(n, 0)
    except Exception:
        return default


def get_max_upload_bytes(default: int = 20 * 1024 * 1024) -> int:
    """Read max upload bytes from env with a safe default.

    Returns:
        A positive integer upload limit in bytes.
    """
    try:
        v = (os.getenv("DUBBING_MAX_UPLOAD_BYTES") or "").strip()
        if not v:
            return default
        n = int(v)
        if n <= 0:
            return default
        return n
    except Exception:
        return default


def get_keep_hours(env_name: str, default: int) -> int:
    """Read an integer hour value from an environment variable."""
    try:
        v = (os.getenv(env_name) or "").strip()
        if not v:
            return default
        n = int(v)
        return max(n, 0)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Feature gate helpers
# ---------------------------------------------------------------------------

def check_open_location_enabled() -> None:
    """Raise 403 if the 'open location' feature is disabled via env var."""
    if env_truthy("DUBBING_DISABLE_OPEN_LOCATION"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该功能已被管理员禁用",
        )


# ---------------------------------------------------------------------------
# Secret masking helpers
# ---------------------------------------------------------------------------

def mask_secrets_in_text(s: str) -> str:
    """Mask bearer tokens and API keys in a text string."""
    if expose_secrets():
        return s

    s = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer ********", s)
    s = re.sub(r"\bsk-[A-Za-z0-9]{10,}\b", "sk-********", s)
    return s


def mask_secret_value(value: str | None) -> str | None:
    """Mask a single secret value (e.g. API key) for display.

    Returns the original value unchanged if it is falsy (None / empty string).
    """
    if not value:
        return value
    if expose_secrets():
        return value
    return "********"


def sanitize_raw_for_response(obj: object) -> object:
    """Recursively mask secrets in an arbitrary JSON-like response object."""
    if expose_secrets():
        return obj

    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            key = str(k)
            k_lower = key.lower()
            if k_lower in {
                "api_key",
                "apikey",
                "authorization",
                "token",
                "access_token",
                "secret",
            }:
                out[key] = "********" if v else v
                continue
            out[key] = sanitize_raw_for_response(v)
        return out
    if isinstance(obj, list):
        return [sanitize_raw_for_response(x) for x in obj]
    if isinstance(obj, str):
        return mask_secrets_in_text(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj

    try:
        return str(obj)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
