"""Explicit original VoiceDesign intent, distinct from recording-clone requests.

Only an owner-authorized role target chooses this route. Prompts describe vocal
traits; no performer recording, inferred numerical acoustics or shaping is used.
"""
from __future__ import annotations

import copy
import hashlib

SCHEMA = "original-inspired-design-v1"


def target(project, role_id, retained="", explicit=None):
    from .named_reference_feedback import interpret

    # A later explicit request for a recording overrides this design approach.
    if explicit or interpret(retained).get("requested_performer"):
        return None
    targets = project.metadata.get("voice_lab_original_design_targets") or {}
    row = targets.get(role_id) if isinstance(targets, dict) else None
    if not isinstance(row, dict) or row.get("schema") != SCHEMA:
        return None
    if not row.get("requested_by_event") or row.get("actor_audio_used") is not False:
        return None
    if not row.get("performer") or not row.get("performance"):
        return None
    options = row.get("options")
    if not isinstance(options, list) or not 1 <= len(options) <= 3:
        return None
    if any(not isinstance(o, dict) or not isinstance(o.get("instruction"), str)
           or not 20 <= len(o["instruction"]) <= 1600 or not o.get("label") for o in options):
        return None
    if len({o["instruction"] for o in options}) != len(options):
        return None
    return copy.deepcopy(row)


def authorized_candidate(candidate):
    """The frozen owner intent survives into the isolated audition project."""
    ref = candidate.get("reference") or {}
    instruction = candidate.get("design_instruction") or ""
    return (ref.get("schema") == SCHEMA and bool(ref.get("requested_by_event"))
            and ref.get("actor_audio_used") is False
            and ref.get("characterization") == "authored descriptive traits; no recording measurements"
            and ref.get("instruction_sha256") == hashlib.sha256(instruction.encode()).hexdigest()
            and bool((candidate.get("inspiration") or {}).get("performer"))
            and candidate.get("conditioning") is None)


def proposals(row):
    plan = []
    for option in row["options"]:
        instruction = option["instruction"]
        inspiration = {"performer": row["performer"], "performance": row["performance"],
                       "source_urls": []}
        plan.append({"kind": "generated", "reuse_key": None,
            "rationale": "Explicit owner request for an original inspired VoiceDesign; "
                         + option["label"] + ". Human resemblance and British delivery require listening.",
            "design_instruction": instruction, "anchor": None, "inspiration": inspiration,
            "concept": {"key": option["label"], "label": option["label"],
                        "emphasis": instruction, "differing_dimensions": option.get("dimensions") or []},
            "reference": {"schema": SCHEMA, "source_type": "original_inspired_design",
                "performer": row["performer"], "performance": row["performance"],
                "requested_by_event": row["requested_by_event"], "actor_audio_used": False,
                "characterization": "authored descriptive traits; no recording measurements",
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest()},
            "conditioning": None, "supported": [],
            "inferred": [{"dimension": "design_intent",
                          "value": "Authored vocal traits; audible fit remains unverified",
                          "reason": "Explicit owner original-design direction, not book evidence or actor measurements"}],
            "tags": ["original inspired VoiceDesign", "British English", option["label"]],
            "source_urls": [], "blocked": False, "blocked_reason": None})
    return plan
