"""Cloned-only voice eligibility policy (pure, additive, read-only).

A project opts in with ``project.metadata["director_voice_policy"] == {"mode":
"cloned_only"}``.  When the key is absent behaviour is unrestricted (``"any"``);
a present-but-malformed value fails closed rather than silently relaxing.

A *genuine clone* is a voice identity that is all of:

* ``mechanism == "qwen_reference"`` and ``backend == "qwen3tts"``;
* backed by a reference audio and a prompt artifact, each with a real 64-hex
  SHA-256 (``reference_audio_sha256`` / ``prompt_sha256``);
* ``provenance.origin == "eligible_reference_clone"`` with
  ``provenance.actor_audio_used is True``.

``generated_original_voice`` is rejected even when it is reference-conditioned,
as are provider presets, model-design directions and anything whose provenance
is missing or unknown.

This module answers one question -- "is this a clone?" -- and nothing else.  It
never reads approval, rating, selection or consent fields, never grants or
infers rights, and never exempts a non-clone because a human approved it.  Rights
and approval remain with their existing owners (see ``casting_director_discovery``).
It performs no I/O, so file existence and hash *contents* are not re-verified
here; the render path's own verification still applies.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

POLICY_KEY = "director_voice_policy"
MODE_ANY = "any"
MODE_CLONED_ONLY = "cloned_only"
MODES = (MODE_ANY, MODE_CLONED_ONLY)

CLONE_MECHANISM = "qwen_reference"
CLONE_BACKEND = "qwen3tts"
CLONE_ORIGIN = "eligible_reference_clone"
ORIGINAL_ORIGIN = "generated_original_voice"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MIN_DISTINCT_HEX = 8  # a real digest has ~14 distinct nibbles; "00..0" / "abab.." are placeholders


class ClonePolicyError(ValueError):
    """Raised for an invalid policy value or a record that violates ``cloned_only``."""

    def __init__(self, message: str, holds: list[dict] | None = None):
        super().__init__(message)
        self.holds = list(holds or [])


def _hold(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


# --------------------------------------------------------------------------------------
# policy mode
# --------------------------------------------------------------------------------------

def policy_mode(project: Any) -> str:
    """``"any"`` when absent, else the validated mode.  Never mutates the project."""
    metadata = project.get("metadata") if isinstance(project, Mapping) else getattr(project, "metadata", None)
    if metadata is None:
        return MODE_ANY
    if not isinstance(metadata, Mapping):
        raise ClonePolicyError("project.metadata is not a mapping; refusing to guess the voice policy")
    if POLICY_KEY not in metadata:
        return MODE_ANY
    policy = metadata[POLICY_KEY]
    mode = policy.get("mode") if isinstance(policy, Mapping) else None
    if mode not in MODES:  # also rejects unhashable / non-string / missing modes
        raise ClonePolicyError(f"project.metadata.{POLICY_KEY}.mode must be one of {list(MODES)}")
    return mode


def _check_mode(mode: Any) -> str:
    if not isinstance(mode, str) or mode not in MODES:
        raise ClonePolicyError(f"voice policy mode must be one of {list(MODES)}")
    return mode


# --------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------

def _as_mapping(value: Any) -> Mapping | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _real_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value)) and len(set(value)) >= _MIN_DISTINCT_HEX


def _facts_from_identity(identity: Mapping) -> dict:
    provenance = identity.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return {
        "mechanism": identity.get("mechanism"), "backend": identity.get("backend"),
        "reference_audio": bool(_text(identity.get("reference_audio"))),
        "prompt_artifact": bool(_text(identity.get("prompt_artifact"))),
        "reference_audio_sha256": identity.get("reference_audio_sha256"),
        "prompt_sha256": identity.get("prompt_sha256"),
        "origin": provenance.get("origin"), "actor_audio_used": provenance.get("actor_audio_used"),
        "builtin_speaker": _text(identity.get("builtin_speaker")),
    }


def _facts_from_conditioning(candidate: Mapping) -> dict | None:
    """Discovery candidates carry ``conditioning`` (built only when both assets exist) and ``provenance``."""
    conditioning = candidate.get("conditioning")
    if not isinstance(conditioning, Mapping):
        return None
    provenance = candidate.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return {
        "mechanism": conditioning.get("mechanism"), "backend": conditioning.get("backend"),
        "reference_audio": True, "prompt_artifact": True,  # discovery emits conditioning only when both assets exist
        "reference_audio_sha256": conditioning.get("reference_audio_sha256"),
        "prompt_sha256": conditioning.get("prompt_sha256"),
        "origin": provenance.get("stored_origin"), "actor_audio_used": provenance.get("actor_audio_used"),
        "builtin_speaker": "",
    }


def _clone_holds(facts: dict | None) -> list[dict]:
    """Holds for one normalised identity; ``None`` means no identity at all."""
    if facts is None:
        return [_hold("clone_policy_no_voice_identity", "No voice identity: unknown provenance is not a clone")]
    holds: list[dict] = []
    if facts["mechanism"] == "builtin" or facts["builtin_speaker"]:
        holds.append(_hold("clone_policy_provider_preset", "Provider presets are not cloned voices"))
    elif facts["mechanism"] != CLONE_MECHANISM:
        holds.append(_hold("clone_policy_mechanism", f"Identity mechanism must be {CLONE_MECHANISM}"))
    if facts["backend"] != CLONE_BACKEND:
        holds.append(_hold("clone_policy_backend", f"Identity backend must be {CLONE_BACKEND}"))
    if not facts["reference_audio"]:
        holds.append(_hold("clone_policy_reference_audio_missing", "No reference audio is recorded on the identity"))
    if not _real_sha256(facts["reference_audio_sha256"]):
        holds.append(_hold("clone_policy_reference_audio_hash", "reference_audio_sha256 is not a real SHA-256"))
    if not facts["prompt_artifact"]:
        holds.append(_hold("clone_policy_prompt_missing", "No prompt artifact is recorded on the identity"))
    if not _real_sha256(facts["prompt_sha256"]):
        holds.append(_hold("clone_policy_prompt_hash", "prompt_sha256 is not a real SHA-256"))
    elif facts["prompt_sha256"] == facts["reference_audio_sha256"]:
        holds.append(_hold("clone_policy_hash_reuse", "Reference audio and prompt share one digest; one is a placeholder"))
    origin = facts["origin"]
    if origin == CLONE_ORIGIN:
        if facts["actor_audio_used"] is not True:
            holds.append(_hold("clone_policy_actor_audio", "A recorded-reference clone must state actor_audio_used is true"))
    elif origin == ORIGINAL_ORIGIN:
        holds.append(_hold("clone_policy_generated_original_voice",
                           "Generated original voices are not clones, even when reference-conditioned"))
    else:
        holds.append(_hold("clone_policy_origin", f"Provenance origin must be {CLONE_ORIGIN}; unknown or missing is rejected"))
    return holds


def _kind_holds(record: Mapping) -> list[dict]:
    kind = record.get("kind")
    if kind == "provider_builtin" or _text(record.get("builtin_speaker")):
        return [_hold("clone_policy_provider_preset", "Provider presets are not cloned voices")]
    if kind == "original_inspired":
        return [_hold("clone_policy_generated_original_voice", "Original-inspired voices are not clones")]
    return []


def _dedupe(holds: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for hold in holds:
        if hold["code"] not in seen:
            seen.add(hold["code"])
            out.append(hold)
    return out


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

def record_holds(record: Any, *, mode: str) -> list[dict]:
    """Holds for a voice record (``{"identity": {...}, "kind", ...}``) or a bare identity."""
    if _check_mode(mode) == MODE_ANY:
        return []
    data = _as_mapping(record)
    if data is None:
        return [_hold("clone_policy_unreadable", "Voice record is not readable; unknown provenance is rejected")]
    identity = _as_mapping(data.get("identity")) if "identity" in data else (data if "mechanism" in data else None)
    facts = _facts_from_identity(identity) if identity else None
    return _dedupe(_kind_holds(data) + _clone_holds(facts))


def candidate_holds(candidate: Any, *, mode: str) -> list[dict]:
    """Holds for an audition candidate (``identity`` present) or a casting-discovery candidate."""
    if _check_mode(mode) == MODE_ANY:
        return []
    data = _as_mapping(candidate)
    if data is None:
        return [_hold("clone_policy_unreadable", "Candidate is not readable; unknown provenance is rejected")]
    holds = _kind_holds(data)
    if data.get("origin") == "model_design":
        holds.append(_hold("clone_policy_model_design", "Model-design directions are not cloned voices"))
    identity = _as_mapping(data.get("identity"))
    facts = _facts_from_identity(identity) if identity else _facts_from_conditioning(data)
    return _dedupe(holds + _clone_holds(facts))


def role_holds(role: Any, *, mode: str) -> list[dict]:
    """Holds for a role, judged only by ``role.voice_identity.model_dump(mode="json")``."""
    if _check_mode(mode) == MODE_ANY:
        return []
    identity = role.get("voice_identity") if isinstance(role, Mapping) else getattr(role, "voice_identity", None)
    dumped = _as_mapping(identity) if identity is not None else None
    return _dedupe(_clone_holds(_facts_from_identity(dumped) if dumped else None))


def require_record(record: Any, *, mode: str) -> None:
    """Raise :class:`ClonePolicyError` unless the record satisfies ``mode``."""
    holds = record_holds(record, mode=mode)
    if holds:
        codes = ", ".join(hold["code"] for hold in holds)
        raise ClonePolicyError(f"Voice record violates the {mode} policy: {codes}", holds)
