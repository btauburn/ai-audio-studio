"""Derive the Auto Cast re-submission acceptance evidence from the canonical view.

The report beside this file is a document and is edited like one; this artefact is
the machine-readable half of the same request and is *derived*, so every number in
it can be traced to the canon that produced it.

It reads ``casting_review.view`` rather than reimplementing any of its verdicts.
"That character is approved" is a live check -- the assignment's voice revision must
still be the current one, the rating event must be Brian's, four stars or better,
and human in origin -- and a copy of that rule here would drift silently while still
looking authoritative. The only thing derived outside the view is the attribution
split, which the view reports as one unresolved total.

Reads: the casting view, and ``segments.json`` for that split.
Writes: ``autocast-resubmit/acceptance-evidence.json``.
Never mutates project state.

Usage:
    python tools/build_autocast_resubmit_evidence.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import UTC, datetime

PROJECTS_ROOT = os.environ.get(
    "DUBBING_PROJECTS_ROOT",
    r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects",
)
PROJECT_ID = os.environ.get("AUTOCAST_ACCEPTANCE_PROJECT", "ba9786ea")
EVIDENCE_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "runtime_evidence", "2026-09-28-autocast-elf-acceptance"))
OUT = os.path.join(EVIDENCE_ROOT, "autocast-resubmit", "acceptance-evidence.json")

# The terminal events the delivery layer is allowed to announce. Listed so a missing
# one is reported as missing rather than simply not looked for.
TERMINAL_EVENTS = ("CASTING_COMPLETE", "CASTING_NEEDS_ATTENTION", "PRODUCTION_READY")


def _load(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def _candidates(roles):
    """Every audition slot the view exposes, with its anchor and its verdict."""
    slots = []
    for role in roles:
        for candidate in role.get("candidates") or []:
            anchor = candidate.get("anchor") or {}
            inspiration = candidate.get("inspiration") or {}
            acoustics = candidate.get("acoustics") or {}
            slots.append({
                "role_id": role.get("role_id"),
                "role_name": role.get("name"),
                "candidate_id": candidate.get("candidate_id"),
                "label": candidate.get("label"),
                "status": candidate.get("status"),
                "rating": candidate.get("rating"),
                "needs_replacement": bool(candidate.get("needs_replacement")),
                "concept": (candidate.get("concept") or {}).get("label"),
                "performer": inspiration.get("performer") or anchor.get("performer"),
                "performance": inspiration.get("performance") or anchor.get("performance"),
                "catalog_version": anchor.get("catalog_version"),
                "anchored": bool(anchor.get("anchored")),
                "unanchored_reason": (anchor.get("unanchored_reason")
                                      or anchor.get("unachored_reason")),
                # The reach verdict is three-valued by design; recorded verbatim
                # rather than collapsed, because "could not measure" is not "failed".
                "achievement": acoustics.get("achievement"),
                "achieved": acoustics.get("achieved"),
            })
    return slots


def _attribution_split():
    """Unresolved utterances, split by whether the analyzer proposed a speaker.

    The view reports the unresolved total as one number and cannot say which of them
    a human can clear with one tap, which is the difference between a review and an
    attribution job.
    """
    project_path = os.path.join(PROJECTS_ROOT, PROJECT_ID, "project.json")
    segments_path = os.path.join(PROJECTS_ROOT, PROJECT_ID, "segments.json")
    if not os.path.exists(segments_path):
        return {"available": False, "reason": f"no segments.json at {segments_path}"}
    segments = _load(segments_path)
    held = set((_load(project_path).get("metadata") or {}).get("autocast_import", {})
               .get("unresolved_utterance_ids") or [])
    unresolved = [s for s in segments if not s.get("role_id")]
    with_proposal = [s for s in unresolved if s.get("id") in held
                     and str((s.get("attribution") or {}).get("label") or "").strip().casefold()
                     not in {"", "unknown"}]
    return {
        "available": True,
        "utterances_in_scope": len(segments),
        "unresolved": len(unresolved),
        "with_proposal_awaiting_review": len(with_proposal),
        "without_speaker_evidence": len(unresolved) - len(with_proposal),
        "held_for_review_by_import": len(held),
    }


def _ledger():
    root = os.path.join(PROJECTS_ROOT, PROJECT_ID, "delivery", "notifications")
    outcomes = []
    for path in sorted(glob.glob(os.path.join(root, "outcomes", "*.json"))):
        record = _load(path)
        outcomes.append({
            "recorded_at": record.get("recorded_at"),
            "state": record.get("state"),
            "http_status": record.get("http_status"),
            "attempts": record.get("attempts"),
            "provider_request_id": record.get("provider_request_id"),
            "batch_id": record.get("batch_id"),
            "device_delivery_confirmed": record.get("device_delivery_confirmed"),
        })
    outcomes.sort(key=lambda row: str(row.get("recorded_at") or ""))
    if not outcomes:
        # An empty read is a claim about the instrument until a second probe agrees.
        raise SystemExit(
            f"DELIVERY LEDGER EMPTY at {root}; refusing to report zero accepted pushes")
    return {
        "source": root,
        "outcome_count": len(outcomes),
        "states": sorted({row["state"] for row in outcomes if row.get("state")}),
        "all_first_attempt": all(row.get("attempts") == 1 for row in outcomes),
        "any_device_delivery_confirmed": any(row.get("device_delivery_confirmed")
                                             for row in outcomes),
        "latest": outcomes[-1],
    }


def build():
    sys.path.insert(0, os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
    from dubbing_service.app.audiobook import casting_review

    view = casting_review.view(PROJECT_ID)
    pipeline = view.get("pipeline") or {}
    run = pipeline.get("run") or {}
    roles = view.get("roles") or []
    slots = _candidates(roles)
    performers = [row["performer"] for row in slots if row["performer"]]
    receipt = next((row for row in view.get("productions") or []
                    if row.get("request_id") == casting_review.AUTOMATIC_PRODUCTION_ID), {})
    raw = _load(os.path.join(PROJECTS_ROOT, PROJECT_ID, "project.json"))
    raw_roles = ((raw.get("voice_lab") or {}).get("casting") or {}).get("roles") or {}
    notes = (((raw.get("voice_lab") or {}).get("casting") or {}).get("notifications") or {})
    announced = {(value or {}).get("event") for value in notes.values()}

    approved = [row for row in roles if row.get("status") == "APPROVED"]
    return {
        "schema": "autocast-resubmit-evidence-v1",
        "project_id": PROJECT_ID,
        "source": "casting_review.view + delivery ledger + segments.json",
        "pipeline": {key: pipeline.get(key) for key in
                     ("stage", "label", "characters", "castable", "ready_options",
                      "approved", "needs_attention", "pipeline_version", "scope_error")},
        "run": {key: run.get(key) for key in
                ("key", "status", "trigger", "pipeline_version", "reason",
                 "started_at", "completed_at")},
        "rounds": {
            "count": len(roles),
            # Every round's version, so "did the version bump actually re-materialise
            # the rounds?" is answerable from this artefact. The casting view does not
            # expose it, so it comes from the canonical project state.
            "pipeline_versions": sorted({
                ((item.get("round") or {}).get("pipeline_version") or "unknown")
                for item in (raw_roles or {}).values()}),
            "awaiting_review": sum(1 for row in roles
                                   if row.get("round_status") == "awaiting_review"),
        },
        "characters": [{
            "role_id": row.get("role_id"),
            "name": row.get("name"),
            "status": row.get("status"),
            "castable": row.get("castable"),
            "round_status": row.get("round_status"),
            "round_revision": row.get("round_revision"),
            "candidates": len(row.get("candidates") or []),
            "ready": sum(1 for c in row.get("candidates") or [] if c.get("status") == "ready"),
            "needs_replacement": sum(1 for c in row.get("candidates") or []
                                     if c.get("needs_replacement")),
            "attention_utterances": len(row.get("needs_attention_utterance_ids") or []),
            "notification_state": (row.get("notification") or {}).get("state"),
            "notification_review_url": (row.get("notification") or {}).get("review_url"),
        } for row in roles],
        "characters_summary": {
            "total": len(roles),
            "approved": len(approved),
            "awaiting_review": sum(1 for row in roles if row.get("status") == "NEEDS_REVIEW"),
            "rounds_awaiting_review": sum(1 for row in roles
                                          if row.get("round_status") == "awaiting_review"),
            "review_pushes_accepted": sum(
                1 for row in roles
                if (row.get("notification") or {}).get("state") == "accepted"),
            "approved_names": [row.get("name") for row in approved],
            "review_ready_names": [row.get("name") for row in roles
                                   if row.get("status") == "NEEDS_REVIEW"],
        },
        "candidates": {
            "slots": len(slots),
            "anchored_slots": sum(1 for row in slots if row["anchored"]),
            "unanchored_slots": sum(1 for row in slots if not row["anchored"]),
            "ready_slots": sum(1 for row in slots if row["status"] == "ready"),
            "rated_slots": sum(1 for row in slots if row["rating"]),
            "distinct_performers": len(set(performers)),
            "reused_performers": sorted({name for name in performers
                                         if performers.count(name) > 1}),
            "catalog_versions": sorted({row["catalog_version"] for row in slots
                                        if row["catalog_version"]}),
            "detail": slots,
        },
        "source_attribution": _attribution_split(),
        "production": {
            "request_id": receipt.get("request_id"),
            "status": receipt.get("status"),
            "reason": receipt.get("reason"),
            "reason_version": receipt.get("reason_version"),
            "attempts": receipt.get("attempts"),
            # Retraction is history the receipt keeps, not an absent field: a
            # superseded render is the reason the current one exists.
            "superseded": receipt.get("superseded"),
        },
        "notifications": {
            "terminal_event_records": len(notes),
            "terminal_events_announced": sorted(event for event in announced if event),
            "terminal_events_missing": sorted(set(TERMINAL_EVENTS) - announced),
            "accepted_records": sum(1 for value in notes.values()
                                    if (value or {}).get("state") == "accepted"),
        },
        "delivery": _ledger(),
        "not_established": [
            ("Phone delivery: every ledger entry is provider acceptance and every one "
             "records device_delivery unobserved; only the recipient can confirm reception."),
            ("A finished audiobook: the render is blocked on the unresolved speaker "
             "population under source_attribution."),
            ("Reference recording ingestion: characterization is inferred from descriptor "
             "priors, with the reason stored on every candidate's reference evidence."),
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def main():
    evidence = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary = evidence["characters_summary"]
    candidates = evidence["candidates"]
    print(f"wrote {OUT}")
    print(f"  run: {evidence['run']['status']} @ {evidence['pipeline']['pipeline_version']}")
    print(f"  characters: {summary['approved']} approved, "
          f"{summary['awaiting_review']} awaiting review")
    print(f"  candidates: {candidates['anchored_slots']}/{candidates['slots']} anchored, "
          f"{candidates['ready_slots']} ready, "
          f"{candidates['distinct_performers']} distinct performers")
    print(f"  production: {evidence['production']['status']}")
    print(f"  delivery: {evidence['delivery']['outcome_count']} outcomes, "
          f"states={evidence['delivery']['states']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
