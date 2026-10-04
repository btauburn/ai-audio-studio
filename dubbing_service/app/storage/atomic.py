"""Bounded atomic metadata replacement for short Windows reader locks."""

from __future__ import annotations

import os
import time
from pathlib import Path

_REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.5)
METADATA_MARKER_PREFIX = ".commit-"
METADATA_MARKER_SUFFIX = ".json"


class MetadataRecoveryRequiredError(RuntimeError):
    """An incomplete transaction must be recovered before accepting new writes."""


def pending_metadata_transactions(root: Path) -> list[Path]:
    return sorted(root.glob(f"{METADATA_MARKER_PREFIX}*{METADATA_MARKER_SUFFIX}"))


def ensure_metadata_writable(root: Path) -> None:
    """Do not accept state that an older transaction's recovery would erase."""

    if pending_metadata_transactions(root):
        raise MetadataRecoveryRequiredError(
            "Project metadata has an unfinished transaction. Use project recovery "
            "(make-audiobook verify --project-id PROJECT_ID --recover) before "
            "writing new state."
        )


def replace_metadata_file(source: str | Path, target: str | Path) -> None:
    """Promote the same complete stage, or propagate the original error.

    Six attempts wait at most 1.25 seconds. The caller retains responsibility
    for staging, fsync, transaction recovery, and cleanup after exhaustion.
    """

    for attempt in range(len(_REPLACE_DELAYS) + 1):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            # WinError 5: access denied; 32/33: sharing/lock violation. Generic
            # PermissionError, missing paths, and other IO errors are not retried.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(
                _REPLACE_DELAYS
            ):
                raise
            time.sleep(_REPLACE_DELAYS[attempt])
