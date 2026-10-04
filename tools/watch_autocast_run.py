"""Watch the acceptance project's Auto Cast run until it settles, reporting progress.

Read-only: it polls canonical project state and prints a line whenever a candidate
becomes audible. It never generates audio, sends a notification or writes state.
"""
import os
import sys
import time

ROOT = r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance"
os.environ["DUBBING_PROJECTS_ROOT"] = os.path.join(ROOT, "projects")
sys.path.insert(0, r"C:\Users\brian\omni-audiobook\ai-audio-studio")

from dubbing_service.app.core.config import clear_config_caches  # noqa: E402

clear_config_caches()

from dubbing_service.app.audiobook import casting_review as casting  # noqa: E402
from dubbing_service.app.storage.projects import load_project  # noqa: E402

PROJECT_ID = "ba9786ea"
MAX_MINUTES = float(os.environ.get("WATCH_MINUTES", "80"))
INTERVAL = 30


def snapshot():
    project = load_project(PROJECT_ID)
    run = project.character_cast.autocast
    state = casting._state(project)
    ready = pending = blocked = 0
    rounds = {}
    for item in state["roles"].values():
        round_ = item.get("round") or {}
        rounds[round_.get("status")] = rounds.get(round_.get("status"), 0) + 1
        for candidate in round_.get("candidates") or []:
            status = candidate.get("status")
            ready += status == "ready"
            pending += status == "pending"
            blocked += status == "blocked"
    return run, rounds, ready, pending, blocked


def main():
    started = time.time()
    deadline = started + MAX_MINUTES * 60
    last = None
    while time.time() < deadline:
        run, rounds, ready, pending, blocked = snapshot()
        now = (run.status, ready, pending, blocked, tuple(sorted(rounds.items())))
        if now != last:
            elapsed = int(time.time() - started)
            print(f"[{elapsed:>5}s] run={run.status} ready={ready} pending={pending} "
                  f"blocked={blocked} rounds={dict(sorted(rounds.items()))}", flush=True)
            last = now
        if run.status not in {"pending", "running"}:
            print(f"SETTLED: run {run.status} - {run.reason}", flush=True)
            return 0
        time.sleep(INTERVAL)
    print("WATCH TIMEOUT: the run had not settled; state above is the last observation", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
