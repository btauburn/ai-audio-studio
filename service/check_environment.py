"""Compatibility wrapper for the authoritative platform doctor.

The old implementation checked the removed ``checkpoints/`` layout and
recommended rebuilding one mixed environment.  Keep the filename for users
with old shortcuts, but route every check through the current project doctor.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCTOR = PROJECT_ROOT / "scripts" / "platform_doctor.py"


def main() -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print(
        "[compat] service/check_environment.py now delegates to scripts/platform_doctor.py.",
        file=sys.stderr,
    )
    completed = subprocess.run(
        [sys.executable, str(DOCTOR), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
