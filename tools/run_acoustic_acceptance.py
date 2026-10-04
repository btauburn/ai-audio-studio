"""Run the real performer-reference acoustic conditioning chain for the acceptance project.

Walks the production chain for a few characters against the live GPU worker:
dossier -> casting concepts -> anchored performer -> reference evidence ->
conditioning payload -> worker design -> worker acoustics -> receipt.

Writes one JSON receipt per identity plus a combined report file. No test double is
used anywhere in this script; every number it records came from the live worker.
"""
import json
import os
import sys
import urllib.request

ROOT = r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance"
os.environ["DUBBING_PROJECTS_ROOT"] = os.path.join(ROOT, "projects")
sys.path.insert(0, r"C:\Users\brian\omni-audiobook\ai-audio-studio")

from dubbing_service.app.core.config import clear_config_caches  # noqa: E402

clear_config_caches()

from dubbing_service.app.audiobook import casting_anchors, performer_reference  # noqa: E402
from dubbing_service.app.storage.projects import load_project  # noqa: E402

WORKER = "http://127.0.0.1:8003"
PROJECT_ID = "ba9786ea"
WANTED = ["The Sculptor", "The Librarian", "Almon"]
# Identity ids are immutable and their receipts are reused, so a re-run needs a new
# tag rather than new numbers under an old name.
RUN_TAG = os.environ.get("ACOUSTIC_RUN_TAG", "r2")


def post(operation, payload):
    request = urllib.request.Request(
        f"{WORKER}/api/v1/tts/character/{operation}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def main():
    project = load_project(PROJECT_ID)
    by_id = {role.id: role for role in project.roles}
    report = []
    for role_id, profile in project.character_cast.profiles.items():
        role = by_id.get(role_id)
        if role is None or role.name not in WANTED:
            continue
        plan = casting_anchors.anchored_plan(
            profile, project.character_cast.dossiers.get(role_id), count=3, role=role)
        entry = {"character": role.name, "role_id": role_id, "candidates": []}
        for index, anchor in enumerate(plan):
            conditioning = performer_reference.conditioning_payload(
                performer_reference.resolve(anchor, anchor["concept"]))
            identity = f"voice_accept_{role_id[-6:]}_{index}_{RUN_TAG}"
            design = post("design", {
                "project_id": PROJECT_ID, "identity_id": identity,
                "instruction": "Create an original synthetic voice. " + conditioning["instruction"],
                "text": "There is still time to find the right answer. Let us begin again, and listen carefully.",
                "seed": 9000 + index, "conditioning": conditioning,
            })
            acoustics = post("acoustics", {"project_id": PROJECT_ID, "identity_id": identity})
            before, after = acoustics["profile_before"], acoustics["profile_after"]
            target = conditioning["parameters"]
            entry["candidates"].append({
                "concept": anchor["concept"]["key"],
                "performer": anchor["performer"], "performance": anchor["performance"],
                "identity_id": identity,
                "reference_original": design["reference_audio"],
                "reference_original_sha256": design["reference_audio_sha256"],
                "reference_conditioned": acoustics["reference_audio"],
                "reference_conditioned_sha256": acoustics["reference_audio_sha256"],
                "audio_changed": design["reference_audio_sha256"] != acoustics["reference_audio_sha256"],
                "profile_before": {k: before.get(k) for k in (
                    "f0_median_hz", "words_per_minute", "mean_pause_ms", "rms_dbfs",
                    "brightness_hz", "rasp", "breathiness")},
                "profile_after": {k: after.get(k) for k in (
                    "f0_median_hz", "words_per_minute", "mean_pause_ms", "rms_dbfs",
                    "brightness_hz", "rasp", "breathiness")},
                "shaping_plan": acoustics["shaping_plan"],
                "reconciliation": acoustics["reconciliation"],
                "reach": acoustics["reach"],
                "achieved": acoustics["achieved"],
                "achieved_all_keys": acoustics["achieved_all_keys"],
                "target_parameters": target,
            })
            print("{role:<14} {concept:<16} {performer:<20} f0 {b:.1f} -> {a:.1f} "
                  "(target {t:.1f}) reach={v} changed={c}".format(
                      role=role.name, concept=anchor["concept"]["key"], performer=anchor["performer"],
                      b=before["f0_median_hz"], a=after["f0_median_hz"],
                      t=target["pitch_target_hz"], v=acoustics["reach"]["verdict"],
                      c=entry["candidates"][-1]["audio_changed"]))
        report.append(entry)

    out = os.path.join(ROOT, "acoustic-conditioning-real-run.json")
    with open(out, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
