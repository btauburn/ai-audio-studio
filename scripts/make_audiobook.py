"""Repository-local shim for the ``make-audiobook`` CLI.

The implementation lives in ``dubbing_service.app.audiobook.cli`` so that the
installed console script and this checkout path share one code path. See that
module for usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dubbing_service.app.audiobook.cli import build_parser, main  # noqa: E402

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
