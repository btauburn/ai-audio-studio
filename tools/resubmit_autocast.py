"""Re-submit Auto Cast for the acceptance project under the current pipeline.

Uses the production entry point, ``casting_review.request_autocast_run``, exactly as
a source import does. It records a durable run and returns; the Studio process's own
maintenance loop establishes scope, queues rounds, dispatches the workers and sends
the casting notification. Nothing here generates audio or sends a notification.

The new pipeline version makes the run key differ from the stored one, so this mints
a genuinely new run and supersedes the previous one.
"""
import json
import os
import sys

ROOT = r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance"
os.environ["DUBBING_PROJECTS_ROOT"] = os.path.join(ROOT, "projects")
sys.path.insert(0, r"C:\Users\brian\omni-audiobook\ai-audio-studio")

from dubbing_service.app.core.config import clear_config_caches  # noqa: E402

clear_config_caches()

from dubbing_service.app.audiobook import casting_review as casting  # noqa: E402
from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION  # noqa: E402
from dubbing_service.app.storage.projects import load_project  # noqa: E402

PROJECT_ID = "ba9786ea"
#: An explicit whole-pipeline re-run: also re-cast characters holding an approval,
#: so the expanded pool and the cross-character spread are applied to every
#: character rather than only to ones nobody has answered yet.
RECAST_APPROVED = os.environ.get("RECAST_APPROVED") == "1"


def main():
    before = load_project(PROJECT_ID)
    previous = before.character_cast.autocast
    print("pipeline version in code:", AUTOCAST_PIPELINE_VERSION)
    print("previous run:", None if previous is None else {
        "key": previous.key[:16], "status": previous.status,
        "version": previous.pipeline_version, "trigger": previous.trigger})
    print("re-casting approved characters:", RECAST_APPROVED)

    run, created = casting.request_autocast_run(
        PROJECT_ID, trigger="resubmit",
        reason=("Explicit re-cast with the expanded performer pool"
                if RECAST_APPROVED else "Re-cast with the current pipeline"),
        recast_approved=RECAST_APPROVED)
    print("\nrequested run:", {"key": run["key"][:16], "status": run["status"],
                               "version": run["pipeline_version"], "trigger": run["trigger"],
                               "recast_approved": run.get("recast_approved"),
                               "created": created})
    project = load_project(PROJECT_ID)
    print("stored run now:", {"key": project.character_cast.autocast.key[:16],
                              "status": project.character_cast.autocast.status,
                              "version": project.character_cast.autocast.pipeline_version,
                              "recast_approved": project.character_cast.autocast.recast_approved})
    print("\nThe Studio maintenance loop takes over from here.")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
