"""Approved-library-first candidate discovery for the casting director.

Read-only and metadata-only: no downloads, inference, synthesis, approvals or writes.
Every candidate is a *recommendation*; a human decision recorded by the existing
casting owners is the only approval. Sources are consulted in a fixed order:

1. approved library -- immutable versions whose 4-5 star human rating is bound to
   the exact heard audio by :mod:`preferred_voices` (the existing preference owner);
2. generic library -- every other immutable version in the global voice library;
3. project source metadata -- voices already in this project and external-source
   metadata, each carrying its own rights and consent holds;
4. original model-design directions -- only for roles that still lack a viable
   reference-conditioned voice. A direction is *not* a generated reference: no
   audio exists until a human requests an audition through the casting owner.

Rating policy: a recorded 4-5 star rating makes a voice ``protected`` (never an
automatic replacement target); 1-3 stars flags the voice and holds it back from
shortlists. Weak generic provider presets are never a fallback. Rights are never
inferred: a recording license is not performer consent, and unresolved historical
provenance on an already human-selected voice is retained *without* asserting any
fresh acquisition right.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from dubbing_service.app.storage import voices as library
from . import preferred_voices
from . import voice_lab_library as shelf

SCHEMA = "casting-director-discovery-v1"
CANDIDATE_SCHEMA = "casting-candidate-v1"

ORIGINS = ("approved_library", "generic_library", "authorized_external", "model_design")
_TIER = {origin: index for index, origin in enumerate(ORIGINS)}

#: A recorded rating at or above this protects a voice; below it the voice is flagged.
PROTECT_MIN_RATING = preferred_voices.REUSE_MIN_RATING
LOW_RATING_MAX = PROTECT_MIN_RATING - 1

DEFAULT_LIMIT = 100
DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_DESIGN_PER_ROLE = 5
MAX_EXTERNAL_SOURCES = 200

#: Identity dimensions that are hard compatibility gates, as in the preference owner.
_GATES = ("presentation", "perceived_vocal_age", "accent", "language")
_URL_FIELDS = ("source_page", "media_url", "license_url", "terms_url")
_PROVENANCE_SOURCE_FIELDS = ("attribution", "license", "license_url", "source_page", "terms_url",
                             "media_url", "performer", "provider", "source_id", "language",
                             "rights_scope", "recording_license_evidence", "checksum_origin")
_UNRESOLVED_CAUTION = ("Historical recording provenance is retained as stored; discovery asserts no "
                       "fresh acquisition, license or performer-consent right.")


def _text(value: Any, limit: int = 300) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _digest(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rating(value: Any) -> int | None:
    return value if type(value) is int and 1 <= value <= 5 else None


def _strings(mapping: Any, limit: int = 300) -> dict:
    if not isinstance(mapping, Mapping):
        return {}
    return {str(key)[:64]: _text(value, limit) for key, value in mapping.items()
            if isinstance(value, str) and _text(value, limit)}


def _https(values: Iterable[Any]) -> list[str]:
    urls: list[str] = []
    for value in values:
        url = _text(value, 500)
        if url.startswith("https://") and len(url) > len("https://") and url not in urls:
            urls.append(url)
    return urls


def _hold(code: str, detail: str, **scope: Any) -> dict:
    return {"code": code, "detail": _text(detail, 400), **{k: v for k, v in scope.items() if v is not None}}


def _candidate(*, candidate_id: str, name: str, origin: str, **fields: Any) -> dict:
    """The shared-contract candidate shape; optional metadata rides in ``fields``."""
    holds = list(fields.pop("holds", []))
    cautions = list(fields.pop("cautions", []))
    rating = fields.pop("rating", None)
    protection = list(fields.pop("protection_reasons", []))
    row = {
        "schema": CANDIDATE_SCHEMA, "candidate_id": candidate_id, "key": fields.pop("key", None),
        "name": _text(name, 200) or candidate_id, "origin": origin,
        "library_voice_id": fields.pop("library_voice_id", None),
        "library_version": fields.pop("library_version", None),
        "reference_conditioned": bool(fields.pop("reference_conditioned", False)),
        "protected": bool(protection), "protection_reasons": protection,
        "rating": rating, "low_rating_flag": rating is not None and rating <= LOW_RATING_MAX,
        "traits": fields.pop("traits", {}), "tags": [t for t in (_text(x, 80) for x in fields.pop("tags", [])) if t],
        "provenance": fields.pop("provenance", {}),
        "holds": holds, "cautions": cautions, "eligible": not holds,
        "recommendation_only": True, "approved": False, "selected_for_roles": [], "duplicates": [],
    }
    row.update(fields)
    return row


# --------------------------------------------------------------------------------------
# project facts
# --------------------------------------------------------------------------------------

def _canonical(project: Any, role_id: str | None) -> str | None:
    aliases = getattr(getattr(project, "character_cast", None), "role_aliases", None) or {}
    seen: set = set()
    while role_id in aliases and role_id not in seen:
        seen.add(role_id)
        role_id = aliases[role_id]
    return role_id


def _human_selections(project: Any) -> dict[str, list[str]]:
    """Voice Lab key -> role ids the human currently has approved with that exact voice."""
    casting = getattr(getattr(project, "voice_lab", None), "casting", None) or {}
    events = casting.get("events") or {}
    selected: dict[str, list[str]] = {}
    for role_id, state in (casting.get("roles") or {}).items():
        assignment = (state or {}).get("assignment") or {}
        event = events.get(assignment.get("event_id")) or {}
        key = assignment.get("key")
        if (key and assignment.get("approved") is True and assignment.get("rejected") is not True
                and event.get("origin") == "human" and event.get("actor") and event.get("key") == key):
            selected.setdefault(key, []).append(role_id)
    return {key: sorted(set(roles)) for key, roles in selected.items()}


def _role_contexts(project: Any, segments: list, holds: list) -> list[dict]:
    """Speaking roles with the book-supported traits that gate and shape candidates."""
    lines: dict[str, int] = {}
    for segment in segments or []:
        lineage = getattr(segment, "lineage", None)
        if (lineage and getattr(lineage, "retired", False)) or not str(getattr(segment, "text", "") or "").strip():
            continue
        role_id = _canonical(project, getattr(segment, "role_id", None))
        if role_id:
            lines[role_id] = lines.get(role_id, 0) + 1
    state = getattr(project, "character_cast", None)
    profiles = dict(getattr(state, "profiles", None) or {})
    built: dict | None = None
    contexts = []
    for role in getattr(project, "roles", []) or []:
        if role.id not in lines:
            continue
        profile, dossier = profiles.get(role.id), (getattr(state, "dossiers", None) or {}).get(role.id)
        if profile is None and dossier is not None:
            if built is None:
                try:
                    from .casting_profiles import build_profiles
                    built = build_profiles(project, segments)
                except Exception as exc:  # evidence-derivation failure must not hide the library
                    built = {}
                    holds.append(_hold("profile_derivation_failed", f"{type(exc).__name__}: {exc}"))
            profile = built.get(role.id)
        supported: dict = {}
        if profile is not None or dossier is not None:
            try:
                from .casting_inference import supported_traits
                supported = supported_traits(profile, dossier, policy=getattr(state, "policy", None))
            except Exception as exc:
                holds.append(_hold("profile_traits_unavailable", f"{type(exc).__name__}: {exc}", role_id=role.id))
        role_type = getattr(role.role_type, "value", role.role_type)
        contexts.append({
            "role_id": role.id, "role_name": role.name, "role_type": str(role_type), "line_count": lines[role.id],
            "profile": profile, "dossier": dossier, "supported": supported,
            "traits": {name: _text(str(row.get("value")), 120) for name, row in supported.items()
                       if isinstance(row, Mapping) and row.get("value")},
        })
    contexts.sort(key=lambda row: (-row["line_count"], row["role_id"]))
    return contexts


# --------------------------------------------------------------------------------------
# provenance and library candidates
# --------------------------------------------------------------------------------------

def _record_provenance(record: dict) -> dict:
    """Rights facts exactly as stored on a voice record; nothing is added or upgraded."""
    identity = record.get("identity") or {}
    stored = identity.get("provenance") or {}
    origin = stored.get("origin")
    source = stored.get("source") or {}
    eligibility = stored.get("eligibility") or {}
    evidence = eligibility.get("intended_use_evidence") or {}
    inspiration = record.get("inspiration") or identity.get("inspiration") or {}
    out: dict = {"stored_origin": origin, "actor_audio_used": stored.get("actor_audio_used"),
                 "source_urls": _https([*(source.get(f) for f in _URL_FIELDS), *(inspiration.get("source_urls") or [])]),
                 "source": {k: _text(source.get(k), 500) for k in _PROVENANCE_SOURCE_FIELDS if source.get(k)},
                 "inspiration": ({"performer": _text(inspiration.get("performer"), 200),
                                  "performance": _text(inspiration.get("performance") or inspiration.get("character"), 200),
                                  "note": "creative direction only; not the voice's source"}
                                 if inspiration.get("performer") else None)}
    if origin == "eligible_reference_clone":
        evidenced = bool(eligibility.get("acquisition_eligible") is True and evidence.get("status") == "approved"
                         and evidence.get("evidence_ref"))
        out["rights_status"] = "recorded_source_evidenced" if evidenced else "historical_provenance_unresolved"
        out["consent_evidence"] = ({"authority": _text(evidence.get("authority"), 80), "scope": _text(evidence.get("scope"), 300),
                                    "evidence_ref": _text(evidence.get("evidence_ref"), 500),
                                    "performer_consent_claimed": evidence.get("performer_consent_claimed")}
                                   if evidence else None)
        out["acquisition_sha256"] = stored.get("acquisition_sha256")
        out["fresh_acquisition_rights_asserted"] = False
    elif origin == "generated_original_voice":
        out["rights_status"] = "original_synthetic_no_actor_audio"
        out["consent_evidence"] = None
    elif record.get("builtin_speaker") or identity.get("builtin_speaker"):
        out["rights_status"] = "provider_preset"
        out["consent_evidence"] = None
    else:
        out["rights_status"] = "historical_provenance_unresolved"
        out["consent_evidence"] = None
        out["fresh_acquisition_rights_asserted"] = False
    return out


def _provenance_holds(provenance: dict) -> list[dict]:
    holds = []
    origin, actor = provenance.get("stored_origin"), provenance.get("actor_audio_used")
    if origin == "generated_original_voice" and actor is True:
        holds.append(_hold("provenance_contradiction", "Original synthetic voice also claims actor audio was used"))
    if origin == "eligible_reference_clone" and actor is False:
        holds.append(_hold("provenance_contradiction", "Recorded-reference clone claims no actor audio was used"))
    return holds


def _is_weak_preset(record: dict) -> bool:
    identity = record.get("identity") or {}
    return record.get("kind") == "provider_builtin" or (
        not identity and bool(record.get("builtin_speaker"))) or identity.get("mechanism") == "builtin"


def _conditioning(record: dict, assets: Mapping) -> dict | None:
    identity = record.get("identity") or {}
    if identity.get("mechanism") != "qwen_reference" or not {"reference_audio", "prompt_artifact"} <= set(assets):
        return None
    return {"mechanism": "qwen_reference", "reference_audio_sha256": identity.get("reference_audio_sha256"),
            "prompt_sha256": identity.get("prompt_sha256"), "backend": identity.get("backend"),
            "model": identity.get("model"), "model_revision": identity.get("model_revision"),
            "runtime_version": identity.get("runtime_version"),
            "design_instruction": _text(identity.get("design_instruction"), 1200) or None}


def _library_candidates(selected: Mapping[str, list[str]], holds: list, excluded: list) -> list[dict]:
    try:
        with library.voices_lock():
            voices = library.load_voices()
    except Exception as exc:  # storage error text may contain a local path
        holds.append(_hold("library_unreadable", type(exc).__name__))
        return []
    try:
        # The preference owner's own verification: only 4-5 star ratings bound to
        # the exact heard audio and key are "approved". Called outside the lock.
        verified = {(row["library_voice_id"], row["library_version"]): row
                    for row in preferred_voices.match_preferred({}, limit=10 ** 6, threshold=-1.0)}
    except Exception as exc:
        verified = {}
        holds.append(_hold("approved_library_verification_failed", type(exc).__name__))
    found = []
    for voice in voices:
        for version, raw in (voice.voice_lab_versions or {}).items():
            preferred = (voice.voice_lab_preferred or {}).get(version) or {}
            stored_ratings = [r for r in (_rating(row.get("rating")) for row in preferred.get("ratings") or []) if r]
            highest = max(stored_ratings, default=None)
            try:
                entry = shelf._validated_entry(version, raw)
            except Exception as exc:  # any malformed version is reported, never silently trusted
                holds.append(_hold("library_version_invalid", f"{type(exc).__name__}: {exc}",
                                   library_voice_id=voice.id, library_version=version,
                                   protected=bool(highest and highest >= PROTECT_MIN_RATING) or None))
                continue
            record = entry["record"]
            proposal = verified.get((voice.id, version))
            rating = proposal["signal"] if proposal else highest
            if _is_weak_preset(record) and not (rating and rating >= PROTECT_MIN_RATING):
                excluded.append(_hold("weak_generic_preset", "Provider presets are not a casting fallback",
                                      library_voice_id=voice.id, library_version=version))
                continue
            provenance = _record_provenance(record)
            key = record.get("key")
            roles_selected = list(selected.get(key, []))
            protection = []
            if highest and highest >= PROTECT_MIN_RATING:
                protection.append("rated_4_5_stars")
            if roles_selected:
                protection.append("human_selected_assignment")
            holds_here, cautions = _provenance_holds(provenance), []
            if preferred.get("retired"):
                holds_here.append(_hold("retired_by_human", _text((preferred["retired"] or {}).get("reason"), 300) or "retired"))
            if rating is not None and rating <= LOW_RATING_MAX:
                holds_here.append(_hold("rated_below_reuse_threshold",
                                        f"Best recorded rating is {rating} of 5; flagged and kept out of shortlists"))
            if highest and highest >= PROTECT_MIN_RATING and not proposal:
                cautions.append("A stored 4-5 star rating is not bound to the exact heard audio/key, so this version is "
                                "treated as a generic library voice, not an approved one; it stays protected.")
            if highest and min(stored_ratings) <= LOW_RATING_MAX and highest >= PROTECT_MIN_RATING:
                cautions.append("Also carries a lower recorded rating; the best rating governs reuse and history is kept.")
            if provenance["rights_status"] == "historical_provenance_unresolved":
                if protection:
                    cautions.append(_UNRESOLVED_CAUTION)
                else:
                    holds_here.append(_hold("provenance_unresolved",
                                            "Recorded-source provenance is unresolved and the voice has no human 4-5 star "
                                            "rating or current selection that would justify retaining it"))
            conditioning = _conditioning(record, entry.get("assets") or {})
            traits = {**_strings(preferred.get("traits")), **(
                {"language": preferred_voices.generation_language(record)} if preferred_voices.generation_language(record) else {})}
            found.append(_candidate(
                candidate_id="cand_" + _digest("library", voice.id, version)[:24],
                name=record.get("name") or voice.name, origin="approved_library" if proposal else "generic_library",
                key=key, library_voice_id=voice.id, library_version=version,
                reference_conditioned=conditioning is not None, rating=rating,
                protection_reasons=protection, traits=traits, tags=record.get("tags") or voice.tags,
                provenance=provenance, holds=holds_here, cautions=cautions,
                kind=record.get("kind"), source_tier="global_library", generated_reference=True,
                sample={"audio_sha256": record.get("audio_sha256"), "duration_seconds": record.get("duration_seconds")},
                conditioning=conditioning, profile_sha256=shelf._profile_fingerprint(record),
                identity_id=(record.get("identity") or {}).get("id"),
                prior_use=({"projects": list(proposal.get("projects") or [])[:5],
                            "characters": [row.get("character") or row.get("role_id")
                                           for row in proposal.get("previously_used") or []]} if proposal else None),
                verification={"manifest_sha256_verified": True, "audio_bytes_rehashed": False,
                              "approval_binding_verified": bool(proposal)},
                selected_for_roles=roles_selected))
    return found


def _project_candidates(project: Any, segments: list, selected: Mapping[str, list[str]],
                        known: list[dict], holds: list, excluded: list) -> list[dict]:
    """Voices that live only in this project (not published to the global library)."""
    try:
        from .voice_lab import voice_records
        records = voice_records(project, segments)
    except Exception as exc:
        holds.append(_hold("project_voices_unreadable", f"{type(exc).__name__}: {exc}"))
        return []
    taken_keys = {c["key"] for c in known if c.get("key")}
    taken_audio = {(c.get("sample") or {}).get("audio_sha256") for c in known} - {None}
    preferences = getattr(project.voice_lab, "preferences", None) or {}
    found = []
    for key, record in sorted(records.items()):
        library_id, library_version = record.get("library_voice_id"), record.get("library_version")
        if key in taken_keys or record.get("audio_sha256") in taken_audio or (library_id and library_version and any(
                c.get("library_voice_id") == library_id and c.get("library_version") == library_version for c in known)):
            continue  # already represented by the (higher-priority) library row
        rating = _rating((preferences.get(key) or {}).get("rating"))
        roles_selected = list(selected.get(key, []))
        if not roles_selected and not (rating and (rating >= PROTECT_MIN_RATING or rating <= LOW_RATING_MAX)):
            continue  # an unrated, unselected audition leftover is not a casting candidate
        if _is_weak_preset(record) and not (rating and rating >= PROTECT_MIN_RATING):
            excluded.append(_hold("weak_generic_preset", "Provider presets are not a casting fallback", key=key))
            continue
        identity = record.get("identity") or {}
        provenance = _record_provenance(record)
        protection = (["rated_4_5_stars"] if rating and rating >= PROTECT_MIN_RATING else []) + (
            ["human_selected_assignment"] if roles_selected else [])
        holds_here, cautions = _provenance_holds(provenance), []
        if rating is not None and rating <= LOW_RATING_MAX and not roles_selected:
            holds_here.append(_hold("rated_below_reuse_threshold",
                                    f"Recorded rating is {rating} of 5; flagged and kept out of shortlists"))
        if provenance["rights_status"] == "historical_provenance_unresolved":
            if protection:
                cautions.append(_UNRESOLVED_CAUTION)
            else:
                holds_here.append(_hold("provenance_unresolved", "Recorded-source provenance is unresolved and nothing "
                                        "human-approved justifies retaining it"))
        clone = record.get("kind") == "reference_clone"
        conditioning = _conditioning(record, {n: 1 for n in ("reference_audio", "prompt_artifact")
                                              if identity.get(n)})
        found.append(_candidate(
            candidate_id="cand_" + _digest("project", project.id, key)[:24], name=record.get("name") or key,
            origin="authorized_external" if clone else "generic_library", key=key,
            reference_conditioned=conditioning is not None, rating=rating, protection_reasons=protection,
            traits=_strings(record.get("traits")), tags=record.get("tags") or [], provenance=provenance,
            holds=holds_here, cautions=cautions, kind=record.get("kind"), source_tier="project_voice_unpublished",
            generated_reference=bool(record.get("audio_sha256")),
            sample={"audio_sha256": record.get("audio_sha256"), "duration_seconds": record.get("duration_seconds")},
            conditioning=conditioning, identity_id=identity.get("id"), selected_for_roles=roles_selected,
            verification={"manifest_sha256_verified": False, "audio_bytes_rehashed": False,
                          "approval_binding_verified": False}))
    return found


# --------------------------------------------------------------------------------------
# external source metadata
# --------------------------------------------------------------------------------------

def _external_candidates(sources: Iterable[Any], holds: list) -> list[dict]:
    rows = list(sources or [])
    if len(rows) > MAX_EXTERNAL_SOURCES:
        holds.append(_hold("external_sources_truncated", f"Only the first {MAX_EXTERNAL_SOURCES} sources were audited"))
        rows = rows[:MAX_EXTERNAL_SOURCES]
    if not rows:
        return []
    try:
        from .performer_discovery import reference_eligibility
    except Exception as exc:
        holds.append(_hold("external_audit_unavailable", type(exc).__name__))
        return []
    found = []
    for index, source in enumerate(rows):
        identity = _text(source.get("source_id") or source.get("media_url"), 500) if isinstance(source, Mapping) else ""
        if not identity:
            holds.append(_hold("external_source_unidentifiable", "A source needs a source_id or media_url", index=index))
            continue
        supplied = source.get("intended_use_evidence")
        # Evidence is passed through for the existing eligibility owner to judge; it is
        # never synthesised here, and its absence is itself the consent hold.
        audit = reference_eligibility(source, intended_use_evidence=supplied if isinstance(supplied, Mapping) else None)
        accepted = bool(audit["acquisition_eligible"])
        evidence = audit.get("intended_use_evidence")
        found.append(_candidate(
            candidate_id="cand_" + _digest("external", identity)[:24],
            name=_text(source.get("performer") or source.get("name"), 200) or f"External recording {identity[:40]}",
            origin="authorized_external", reference_conditioned=False,
            traits={**_strings(source.get("traits")), **({"language": _text(source.get("language"), 16)} if source.get("language") else {})},
            tags=[t for t in (source.get("tags") or []) if isinstance(t, str)],
            provenance={
                "rights_status": "acquisition_authorized_pending_acquisition" if accepted else "held",
                "source_urls": _https(source.get(f) for f in _URL_FIELDS),
                "source": {k: _text(source.get(k), 500) for k in _PROVENANCE_SOURCE_FIELDS if source.get(k)},
                "download_eligible": audit["download_eligible"], "intended_use_eligible": audit["intended_use_eligible"],
                "consent_evidence": dict(evidence) if evidence else None,
                "performer_consent_claimed": (evidence or {}).get("performer_consent_claimed") if evidence else None,
                "note": "A recording license is separate from voice/personality consent; neither is inferred."},
            holds=[_hold("external_rights_hold", reason, source_id=_text(source.get("source_id"), 200) or None)
                   for reason in audit["hold_reasons"]],
            kind="external_recording_metadata", source_tier="external_metadata", generated_reference=False,
            requires=["source_download", "source_quality_check", "clone_preparation", "human_audition"],
            verification={"manifest_sha256_verified": False, "audio_bytes_rehashed": False,
                          "approval_binding_verified": False}))
    return found


# --------------------------------------------------------------------------------------
# original model-design directions (recommendations, never generated references)
# --------------------------------------------------------------------------------------

def _design_instruction(context: dict, concept: Mapping) -> str:
    profile = context["profile"]
    if profile is not None and getattr(profile, "design_instruction", ""):
        base = profile.design_instruction
    else:
        values = {n: v for n, v in context["traits"].items() if n != "language"}
        clauses = [f"{n.replace('_', ' ')}: {v}." for n, v in sorted(values.items())]
        base = " ".join(["Create an original, consistent character voice.", *clauses,
                         "Keep this vocal identity stable across emotion and volume changes."])
    return _text(f"{base} Casting direction: {concept.get('emphasis')}", 1600)


def _design_candidates(contexts: list[dict], per_role: int, holds: list) -> list[dict]:
    if per_role < 1:
        return []
    try:
        from .casting_inference import casting_concepts, query_traits
    except Exception as exc:
        holds.append(_hold("design_directions_unavailable", type(exc).__name__))
        return []
    found = []
    for context in contexts:
        try:
            concepts = casting_concepts(context["profile"], context["dossier"], count=per_role)
        except Exception as exc:
            holds.append(_hold("design_directions_unavailable", f"{type(exc).__name__}: {exc}", role_id=context["role_id"]))
            continue
        for concept in concepts:
            if "accent" not in context["supported"]:
                concept["inferred"].pop("accent", None)
                concept["differing_dimensions"] = [d for d in concept["differing_dimensions"] if d != "accent"]
            instruction = _design_instruction(context, concept)
            traits = {n: _text(str(v), 120) for n, v in query_traits(concept).items() if v}
            if "accent" not in context["supported"]:
                traits.pop("accent", None)
            found.append(_candidate(
                candidate_id="cand_" + _digest("design", context["role_id"], concept["key"], instruction)[:24],
                name=f"Original design: {concept['label']} ({context['role_name']})", origin="model_design",
                reference_conditioned=False, traits=traits, tags=["original synthetic design", concept["key"]],
                provenance={"rights_status": "original_synthetic_no_recording", "actor_audio_used": False,
                            "source_urls": [], "consent_evidence": None,
                            "note": "No performer recording or identity is used; nothing here claims consent."},
                cautions=["A direction only: no audio exists. Resemblance and fit need human listening after an explicit audition."],
                kind="model_design_direction", source_tier="original_design", generated_reference=False,
                designed_for_role_id=context["role_id"],
                design={"concept_key": concept["key"], "label": concept["label"], "emphasis": concept["emphasis"],
                        "design_instruction": instruction, "differing_dimensions": list(concept.get("differing_dimensions") or []),
                        "book_supported_traits": sorted(concept.get("supported") or {}),
                        "inferred_traits": sorted(concept.get("inferred") or {}),
                        "inference_note": "Inferred traits are casting interpretation where the book is silent, not facts."},
                requires=["explicit_audition_request", "human_audition"],
                verification={"manifest_sha256_verified": False, "audio_bytes_rehashed": False,
                              "approval_binding_verified": False}))
    return found


# --------------------------------------------------------------------------------------
# dedup, ordering, shortlists
# --------------------------------------------------------------------------------------

def _dedupe_keys(candidate: dict) -> list[tuple]:
    keys: list[tuple] = []
    if candidate.get("library_voice_id"):
        keys.append(("library", candidate["library_voice_id"], candidate.get("library_version")))
    audio = (candidate.get("sample") or {}).get("audio_sha256")
    if audio:
        keys.append(("audio", audio))
    if candidate.get("key"):
        keys.append(("key", candidate["key"]))
    if candidate.get("identity_id"):
        keys.append(("identity", candidate["identity_id"]))
    provenance = candidate.get("provenance") or {}
    source_id = (provenance.get("source") or {}).get("source_id")
    if source_id:
        keys.append(("source", source_id))
    if provenance.get("acquisition_sha256"):
        keys.append(("acquisition", provenance["acquisition_sha256"]))
    if candidate["origin"] == "model_design":
        keys.append(("design", _digest(candidate["design"]["design_instruction"])))
    return keys


def _order(candidate: dict) -> tuple:
    return (not candidate["eligible"], _TIER[candidate["origin"]], not candidate["reference_conditioned"],
            -(candidate["rating"] or 0), candidate["candidate_id"])


def _dedupe(candidates: list[dict]) -> list[dict]:
    """Collapse the same voice seen through several sources into the best-priority row."""
    kept: list[dict] = []
    index: dict[tuple, dict] = {}
    for candidate in sorted(candidates, key=_order):
        owner = next((index[k] for k in _dedupe_keys(candidate) if k in index), None)
        if owner is None:
            kept.append(candidate)
            owner = candidate
        else:
            owner["duplicates"].append({"candidate_id": candidate["candidate_id"], "origin": candidate["origin"],
                                        "source_tier": candidate.get("source_tier")})
            owner["selected_for_roles"] = sorted({*owner["selected_for_roles"], *candidate["selected_for_roles"]})
            for reason in candidate["protection_reasons"]:
                if reason not in owner["protection_reasons"]:
                    owner["protection_reasons"].append(reason)
            owner["protected"] = bool(owner["protection_reasons"])
            if (candidate["rating"] or 0) > (owner["rating"] or 0):
                owner["rating"] = candidate["rating"]
            owner["low_rating_flag"] = owner["rating"] is not None and owner["rating"] <= LOW_RATING_MAX
        for key in _dedupe_keys(candidate):
            index.setdefault(key, owner)
    return kept


def _compatibility(role_traits: Mapping, candidate: dict) -> tuple[bool, int, list[str]]:
    """Hard identity gates plus a count of agreeing dimensions; unknown never counts as a clash."""
    stored = {k: str(v).strip().casefold() for k, v in (candidate.get("traits") or {}).items()}
    reasons, matched = [], 0
    for name in _GATES:
        wanted = str(role_traits.get(name) or "").strip().casefold()
        actual = stored.get(name, "")
        if name == "language":
            wanted = preferred_voices.normalized_language(wanted) or ""
            actual = preferred_voices.normalized_language(actual) or ""
        if wanted and actual and wanted != actual:
            return False, 0, [f"{name.replace('_', ' ')} differs ({actual} vs {wanted})"]
        if wanted and actual:
            matched += 1
            reasons.append(f"{name.replace('_', ' ')}: {wanted}")
    for name, value in role_traits.items():
        if name not in _GATES and stored.get(name) and stored[name] == str(value).strip().casefold():
            matched += 1
            reasons.append(f"{name.replace('_', ' ')}: {value}")
    return True, matched, reasons


def _viable_for(context: dict, candidate: dict) -> tuple[bool, int, list[str]]:
    if not candidate["eligible"]:
        return False, 0, []
    scoped = candidate.get("designed_for_role_id")
    if scoped is not None and scoped != context["role_id"]:
        return False, 0, []
    return _compatibility(context["traits"], candidate)


def _shortlist(context: dict, candidates: list[dict], size: int) -> list[dict]:
    rows = []
    for candidate in candidates:
        ok, matched, reasons = _viable_for(context, candidate)
        if ok:
            rows.append((candidate, matched, reasons))
    rows.sort(key=lambda row: (_TIER[row[0]["origin"]], not row[0]["reference_conditioned"], -row[1],
                               -(row[0]["rating"] or 0), row[0]["candidate_id"]))
    scored = bool(context["traits"])
    return [{"candidate_id": c["candidate_id"], "origin": c["origin"], "name": c["name"],
             "reference_conditioned": c["reference_conditioned"], "protected": c["protected"], "rating": c["rating"],
             "matched_dimensions": matched, "matched_reasons": reasons,
             "fit": ("gates_passed" if matched else "no_trait_overlap_known") if scored else "unscored_no_book_traits",
             "current_for_role": context["role_id"] in c["selected_for_roles"]}
            for c, matched, reasons in rows[:size]]


def _gaps(context: dict, shortlist: list[dict], all_viable_conditioned: bool) -> list[dict]:
    role = {"role_id": context["role_id"], "role_name": context["role_name"]}
    gaps = []
    if not shortlist:
        gaps.append({**role, "code": "no_viable_candidate", "severity": "blocking",
                     "detail": "No eligible candidate passes this role's identity gates."})
    if not all_viable_conditioned:
        gaps.append({**role, "code": "no_reference_conditioned_candidate", "severity": "attention",
                     "detail": "No reference-conditioned voice is viable; original design directions fill the gap "
                               "only as unheard recommendations."})
    if not any(row["origin"] == "approved_library" for row in shortlist):
        gaps.append({**role, "code": "no_approved_library_candidate", "severity": "info",
                     "detail": "No human-approved library voice fits; none is invented."})
    if not context["traits"]:
        gaps.append({**role, "code": "book_trait_evidence_missing", "severity": "attention",
                     "detail": "The book supports no vocal traits for this role, so candidates cannot be fit-checked."})
    return gaps


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

def discover_candidates(project: Any, segments: list, *, external_sources: Iterable[Any] = (),
                        limit: int = DEFAULT_LIMIT, shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
                        design_per_role: int = DEFAULT_DESIGN_PER_ROLE) -> dict:
    """Ordered, de-duplicated candidates plus holds, shortlists and coverage gaps.

    Reads the global library, the project and caller-supplied source *metadata* only.
    Protected (4-5 star or currently human-selected) candidates are never truncated.
    """
    limit, shortlist_size = max(1, int(limit)), max(1, int(shortlist_size))
    holds: list[dict] = []
    excluded: list[dict] = []
    selected = _human_selections(project)
    contexts = _role_contexts(project, list(segments or []), holds)

    pooled = _library_candidates(selected, holds, excluded)
    pooled += _project_candidates(project, list(segments or []), selected, pooled, holds, excluded)
    pooled += _external_candidates(external_sources, holds)
    unique = _dedupe(pooled)

    # Original directions only where no viable reference-conditioned voice exists.
    needy = [c for c in contexts if not any(
        x["reference_conditioned"] and _viable_for(c, x)[0] for x in unique)]
    unique = _dedupe(unique + _design_candidates(needy, max(0, int(design_per_role)), holds))
    unique.sort(key=_order)

    kept = unique[:limit]
    kept += [c for c in unique[limit:] if c["protected"]]
    truncated = len(unique) - len(kept)

    shortlists, gaps = {}, []
    for context in contexts:
        shortlist = _shortlist(context, kept, shortlist_size)
        shortlists[context["role_id"]] = shortlist
        conditioned = any(row["reference_conditioned"] for row in shortlist)
        gaps.extend(_gaps(context, shortlist, conditioned))
    for candidate in kept:
        if candidate["low_rating_flag"] and "rated_below_reuse_threshold" not in {h["code"] for h in candidate["holds"]}:
            candidate["cautions"].append(f"Rated {candidate['rating']} of 5 by a human: flagged.")

    result = {
        "schema": SCHEMA, "project_id": getattr(project, "id", None), "candidates": kept, "holds": holds,
        "excluded": excluded, "roles": [{k: context[k] for k in ("role_id", "role_name", "role_type", "line_count", "traits")}
                                        for context in contexts],
        "shortlists": shortlists, "coverage_gaps": gaps, "truncated": truncated,
        "counts": {"candidates": len(kept), "protected": sum(c["protected"] for c in kept),
                   "flagged_low_rating": sum(c["low_rating_flag"] for c in kept),
                   "eligible": sum(c["eligible"] for c in kept),
                   **{origin: sum(c["origin"] == origin for c in kept) for origin in ORIGINS}},
        "source_order": list(ORIGINS), "human_review": "pending", "recommendations_are_approvals": False,
        "side_effects": {"downloads": 0, "inference": 0, "writes": 0, "approvals": 0},
    }
    return copy.deepcopy(result)


__all__ = ["CANDIDATE_SCHEMA", "ORIGINS", "PROTECT_MIN_RATING", "SCHEMA", "discover_candidates"]
