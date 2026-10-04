"""Probe the real performer sources: which anchors have retrievable audio evidence?

Read-only. It asks Wikidata for each pool performer and Wikimedia Commons for that
performer's spoken-voice recordings (P990), then reports per performer what was
found and whether it carries the licence and intended-use evidence a reference
acquisition requires. It never downloads audio, never adopts a performer and never
writes project state — it answers the question "are there more sources than the
names I already have, and is their audio usable?".
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\brian\omni-audiobook\ai-audio-studio")

from dubbing_service.app.audiobook import performer_discovery as discovery  # noqa: E402

OUT = os.environ.get(
    "SOURCING_REPORT",
    r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance"
    r"\performer-sourcing\sourcing-report.json")


def main():
    import tempfile

    directory = tempfile.mkdtemp(prefix="performer-sourcing-")
    pool = discovery.CATALOG
    print(f"pool: {len(pool)} performers "
          f"({discovery.CATALOG_POOL['seeds']} seeds + {discovery.CATALOG_POOL['shipped']} sourced)")

    rows = []
    limits = discovery.DiscoveryLimits(max_requests=12, max_candidates=1,
                                      max_downloads=0)
    for index, row in enumerate(pool, start=1):
        entry = {"candidate_id": row["candidate_id"], "performer": row["performer"],
                 "performance": row["performance"], "tags": row["tags"]}
        try:
            # discover() ranks then retrieves bounded official metadata for the match.
            result = discovery.discover(
                directory, description=row["creative_direction"], traits={}, limit=1,
                online=True, limits=limits)
            match = (result.get("matches") or [{}])[0]
            entry["wikidata_entity"] = match.get("entity_id")
            entry["recording_discovery"] = match.get("recording_discovery", "complete")
            entry["recording_hold"] = match.get("recording_hold")
            sources = [s for s in result.get("sources") or []
                       if s.get("performer") == row["performer"]]
            entry["recordings_found"] = len(sources)
            entry["download_eligible"] = sum(1 for s in sources if s.get("download_eligible"))
            entry["acquisition_eligible"] = sum(1 for s in sources if s.get("acquisition_eligible"))
            entry["example"] = ({"filename": sources[0].get("filename"),
                                 "source_page": sources[0].get("source_page"),
                                 "license": sources[0].get("license"),
                                 "language": sources[0].get("language"),
                                 "hold_reasons": sources[0].get("hold_reasons")}
                                if sources else None)
            entry["status"] = "recordings_available" if sources else "no_recording"
        except Exception as exc:  # noqa: BLE001 - a probe must report, not raise
            entry["status"] = "probe_failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(entry)
        print(f"[{index:>3}/{len(pool)}] {row['performer']:<28} {entry['status']:<20} "
              f"recordings={entry.get('recordings_found', '-')}", flush=True)

    summary = {
        "pool_size": len(pool),
        "probed": len(rows),
        "recordings_available": sum(1 for r in rows if r["status"] == "recordings_available"),
        "no_recording": sum(1 for r in rows if r["status"] == "no_recording"),
        "probe_failed": sum(1 for r in rows if r["status"] == "probe_failed"),
        "download_eligible": sum(r.get("download_eligible", 0) or 0 for r in rows),
        "acquisition_eligible": sum(r.get("acquisition_eligible", 0) or 0 for r in rows),
        "note": ("acquisition_eligible is the count that could become a measured reference; "
                 "it requires an owner-supplied intended-use decision in addition to the "
                 "recording licence, so it is expected to be zero without one"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as stream:
        json.dump({"summary": summary, "performers": rows}, stream, indent=2)
    print()
    print(json.dumps(summary, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
