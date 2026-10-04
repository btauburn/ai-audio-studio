"""Attach achieved acoustic receipts to ready candidates that lack them.

The round worker now copies a finished voice's achieved acoustic receipt onto the
audible candidate, but candidates generated before that fix carry only the target
conditioning. The receipt itself was never lost: it is stored on the immutable
voice record the candidate points at. This copies that stored value across, so an
existing project's review surface reports what conditioning actually achieved.

It writes only the ``acoustics`` field of a candidate that is already ``ready`` and
has a resolvable voice record; everything else is left exactly as it is, and it
reports every candidate it did not touch with the reason.
"""
import json
import os
import sys

PROJECT_ID = os.environ.get("BACKFILL_PROJECT", "ba9786ea")
DRY_RUN = os.environ.get("BACKFILL_DRY_RUN") == "1"


def main():
    from dubbing_service.app.audiobook import casting_review as casting
    from dubbing_service.app.storage.projects import load_project, save_project

    with casting.project_meta_lock(PROJECT_ID):
        project = load_project(PROJECT_ID)
        state = casting._state(project)
        changed, skipped = [], []
        for role_id, item in state["roles"].items():
            for candidate in (item.get("round") or {}).get("candidates") or []:
                if candidate.get("acoustics"):
                    skipped.append((role_id, candidate["candidate_id"], "already carries a receipt"))
                    continue
                if candidate.get("status") != "ready" or not candidate.get("key"):
                    skipped.append((role_id, candidate["candidate_id"],
                                    f"status={candidate.get('status')}, not audible"))
                    continue
                record = project.voice_lab.voices.get(candidate["key"])
                if not record:
                    skipped.append((role_id, candidate["candidate_id"], "voice record is missing"))
                    continue
                receipt = record.get("acoustics") or (record.get("identity") or {}).get("acoustics")
                if not receipt:
                    skipped.append((role_id, candidate["candidate_id"],
                                    "voice record holds no achieved receipt"))
                    continue
                candidate["acoustics"] = json.loads(json.dumps(receipt))
                changed.append((role_id, candidate["candidate_id"],
                                (receipt.get("reach") or {}).get("verdict")))
        if changed and not DRY_RUN:
            save_project(project)
    print(f"project {PROJECT_ID}: {'would attach' if DRY_RUN else 'attached'} "
          f"{len(changed)} achieved acoustic receipt(s)")
    for role_id, candidate_id, verdict in changed:
        print(f"  attached {role_id} {candidate_id} reach={verdict}")
    print(f"left untouched: {len(skipped)}")
    for role_id, candidate_id, reason in skipped[:40]:
        print(f"  untouched {role_id} {candidate_id}: {reason}")
    if len(skipped) > 40:
        print(f"  ... and {len(skipped) - 40} more")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, r"C:\Users\brian\omni-audiobook\ai-audio-studio")
    os.environ.setdefault(
        "DUBBING_PROJECTS_ROOT",
        r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects")
    from dubbing_service.app.core.config import clear_config_caches

    clear_config_caches()
    raise SystemExit(main())
