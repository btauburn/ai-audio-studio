"""Default famous-performer anchoring for newly proposed character voices.

A generated candidate is only interesting if it sounds *intentionally cast*. The
existing catalogue and its explainable ranking (:mod:`performer_discovery`) already
know which performers were recorded delivering which kind of voice; this module
turns one character's dossier-derived profile into one *specific* performer and
performance per candidate, with the rationale that connects them.

Only abstract vocal traits and catalogue performer names leave this boundary:
never source text, never a character name, never a physical description. The
produced voice remains an original synthetic voice informed by the reference,
never a copy of a recording.
"""
from __future__ import annotations

from typing import Mapping

from .performer_discovery import CATALOG, CATALOG_VERSION, rank_candidates
from .casting_inference import casting_concepts

ANCHOR_SCHEMA = "casting-anchor-v1"
ANCHOR_VERSION = "performer-anchor-v1"

#: An unanchored candidate is a blocked fallback, not a normal proposal: the
#: reason travels with it so the review surface can say why production stopped
#: instead of quietly emitting a generic synthetic voice.
UNANCHORED_REASON = ("No catalogue performance fits this candidate's casting concept, so no reference "
                     "can be justified for it; supply the missing casting dimension or extend the "
                     "reference catalogue rather than generating an unanchored voice.")
NO_TRAIT_REASON = ("This candidate's casting concept has no vocal dimensions at all, so no reference "
                   "performance can be justified; resolve the character dossier before anchoring.")
TAKEN_REASON = ("Every catalogue performance that fits this candidate's casting concept is already used "
                "by another candidate in this round; differentiate the cast or extend the catalogue.")

#: Catalogue tags that name a vocal trait rather than an identity, mapped onto the
#: profile dimensions the existing design instruction already understands.
_TAG_TRAITS = {
    "male": "masculine", "female": "feminine", "young": "youthful", "teen": "adolescent",
    "adult": "adult", "mature": "older adult", "narrator": "narration",
    "warm": "warm", "soft": "soft", "measured": "deliberate", "unhurried": "deliberate",
    "energetic": "high energy", "theatrical": "theatrical", "dry": "dry humour",
    "resonant": "rounded", "clear": "clear", "breathy": "airy", "curious": "curious",
    "confident": "controlled authority", "earnest": "sincere", "expressive": "expressive",
    "reassuring": "reassuring", "thoughtful": "thoughtful", "restrained": "restrained",
}

#: Weight per trait word, so a matched vocal trait counts for more than a label.
_TAG_WEIGHT = {"british": 5.0, "american": 5.0, "narrator": 4.0, "male": 3.0, "female": 3.0,
               "young": 3.0, "teen": 2.0, "mature": 3.0}


def vocal_traits(profile, dossier=None, *, policy=None) -> dict:
    """Traits from the same policy-resolved evidence used by casting concepts."""
    from .casting_inference import supported_traits

    return {name: row["value"] for name, row in
            supported_traits(profile, dossier, policy=policy).items()}


def anchor_query(profile, dossier=None, role=None) -> str:
    """Free-text description the existing ranker also reads, kept source-free.

    Only casting structure is added: a narrator role *is* narration, and the
    profile's own importance band separates a principal from a bit part. No
    character name, appearance or source text reaches this string.
    """
    from ..models.enums import RoleType
    parts = []
    importance = getattr(profile, "importance", "") or ""
    if importance:
        parts.append(f"{importance} character")
    role_type = getattr(getattr(role, "role_type", None), "value", None) or getattr(role, "role_type", None)
    if role_type == RoleType.NARRATOR.value or role_type == RoleType.NARRATOR:
        parts.append("narration narrator")
    return " ".join(parts)


def _rationale(matched) -> str:
    if matched:
        return f"Matched vocal traits: {', '.join(sorted(matched))}."
    return f"Exploratory fit from the {CATALOG_VERSION} catalogue profile; no trait matched directly."


def _trait_values(traits) -> dict:
    """Normalise either a plain mapping or a casting-inference summary to values."""
    values = {}
    for name, row in (traits or {}).items():
        values[name] = row.get("value") if isinstance(row, Mapping) else row
    return {name: value for name, value in values.items() if value}


def concept_traits(concept) -> dict:
    """The vocal query for one casting concept: the book's facts plus that concept's cast."""
    from .casting_inference import query_traits
    return _trait_values(query_traits(concept))


def _concept_affinity(concept, tags) -> float:
    """How well one catalogue performance serves this concept's delivery emphasis."""
    prefer = {str(tag).casefold() for tag in (concept.get("prefer_tags") or [])}
    return float(len(prefer & {str(tag).casefold() for tag in tags or []}))


def anchors_for_role(profile, dossier=None, *, limit: int = 3, role=None,
                     concept=None, exclude_performers=(), avoid_performers=(), policy=None,
                     preferred_candidate_ids=()) -> list[dict]:
    """Distinct performer anchors for one round, best fit for this casting first.

    With a `concept` the query is the concept's cast — the book's supported traits
    plus exactly the dimensions that concept supplies where the book is silent —
    and performances are ranked toward that concept's delivery emphasis. Without
    one, only supported evidence is used.

    Only a reference that shares a trait in the query counts as an anchor: a
    catalogue row returned by a trait-less query would be a name attached to no
    reason, which is the generic result this anchoring exists to prevent.

    ``exclude_performers`` is a hard exclusion and keeps two concepts within one
    round from resolving to the same performer. ``avoid_performers`` is a
    preference: performers this project has already cast elsewhere are ranked last,
    so a book's characters are read by different people. It is a preference rather
    than a rule because a large cast can legitimately exhaust the pool, and an
    unanchored candidate is a worse outcome than a reused performer.

    Pure and deterministic: the same character and context give the same anchors,
    so a refresh or a resumed round cannot reshuffle its own evidence.
    """
    limit = max(1, int(limit))
    traits = concept_traits(concept) if concept is not None else vocal_traits(profile, dossier, policy=policy)
    ranked = rank_candidates(anchor_query(profile, dossier, role), traits=traits, limit=len(CATALOG))
    excluded = {str(name).casefold() for name in exclude_performers if name}
    ranked = [row for row in ranked
              if ((row.get("inspiration") or {}).get("performer") or "").casefold() not in excluded]
    if concept is not None:
        # The ranker's trait score orders the query; the concept's emphasis then
        # decides between performances that fit it about equally well.
        ranked.sort(key=lambda row: (-(float(row.get("score") or 0.0)
                                       + 2.0 * _concept_affinity(concept, row.get("tags"))),
                                     str((row.get("inspiration") or {}).get("performer") or "")))
    # Cited model proposals reorder only the existing catalog; the matched-trait
    # gate and native performer exclusions still apply below.
    preferred = {candidate_id: index for index, candidate_id in enumerate(preferred_candidate_ids)}
    if preferred:
        ranked.sort(key=lambda row: preferred.get(row.get("candidate_id"), len(preferred)))
    avoided = {str(name).casefold() for name in avoid_performers if name}
    if avoided:
        # Stable partition, not a filter: an already-cast performer stays available
        # as a fallback and keeps its relative order within each group.
        ranked.sort(key=lambda row: ((row.get("inspiration") or {}).get("performer") or ""
                                     ).casefold() in avoided)
    anchors = []
    for index, row in enumerate(ranked):
        if len(anchors) >= limit:
            break
        inspiration = row.get("inspiration") or {}
        performer, performance = inspiration.get("performer"), inspiration.get("performance")
        if not performer or not performance:
            continue
        matched = sorted(set(row.get("matched_traits") or []))
        if not matched:
            continue
        anchors.append({
            "schema": ANCHOR_SCHEMA, "version": ANCHOR_VERSION, "catalog_version": CATALOG_VERSION,
            "index": index, "performer": performer, "performance": performance,
            "catalog_candidate_id": row.get("candidate_id"),
            "source_urls": list(inspiration.get("source_urls") or []),
            "verified_facts": list(row.get("verified_facts") or []),
            "tags": list(row.get("tags") or []),
            "concept": concept.get("key") if concept is not None else None,
            "concept_affinity": _concept_affinity(concept, row.get("tags")) if concept is not None else None,
            "matched_traits": matched, "score": row.get("score"),
            "score_factors": dict(row.get("score_factors") or {}),
            "creative_direction": row.get("creative_direction"),
            "generation_direction": row.get("generation_direction"),
            "explanation": row.get("explanation"),
            "rationale": _rationale(matched),
            "anchored": True, "unanchored_reason": None,
        })
    return anchors


def unanchored_candidate(index: int, reason: str) -> dict:
    """A candidate with no reference anchor, carrying the reason it has none."""
    return {"schema": ANCHOR_SCHEMA, "version": ANCHOR_VERSION, "catalog_version": CATALOG_VERSION,
            "index": index, "performer": None, "performance": None, "source_urls": [],
            "verified_facts": [], "tags": [], "matched_traits": [], "score": None,
            "score_factors": {}, "creative_direction": None, "generation_direction": None,
            "explanation": None, "rationale": reason, "anchored": False,
            "unanchored_reason": reason}


def anchored_plan(profile, dossier=None, *, count: int, role=None, avoid_performers=(), policy=None) -> list[dict]:
    """The anchored proposal set for one round: one entry per planned candidate.

    Each candidate is planned from a *different* casting concept, and each concept
    resolves its own performer, so the options differ by casting dimension rather
    than being three near-identical names. A candidate is anchored whenever a
    catalogue performer fits that concept — the normal case, because casting
    inference supplies the dimensions the book omits.

    Anything left over is explicitly unanchored and marked as a blocked fallback
    with the reason it has none, so a generic "AI voice" is never produced by
    omission: either the traits fit no catalogue performance, or no performance
    remains that another candidate has not already taken.
    """
    count = max(1, int(count))
    concepts = casting_concepts(profile, dossier, count=count, role=role, policy=policy)
    plan: list[dict] = []
    taken: list[str] = []
    for index, concept in enumerate(concepts):
        anchors = anchors_for_role(profile, dossier, limit=1, role=role, concept=concept,
                                   exclude_performers=taken, avoid_performers=avoid_performers, policy=policy)
        if anchors:
            entry = dict(anchors[0])
        else:
            # Separate "no performance fits this concept" from "another candidate
            # already took the ones that do" — the two need different remedies.
            if anchors_for_role(profile, dossier, limit=1, role=role, concept=concept, policy=policy):
                reason = TAKEN_REASON
            elif not concept_traits(concept):
                reason = NO_TRAIT_REASON
            else:
                reason = UNANCHORED_REASON
            entry = unanchored_candidate(index, reason)
        entry["concept"] = concept
        entry["index"] = index
        if entry.get("performer"):
            taken.append(entry["performer"])
        plan.append(entry)
    return plan[:count]


def anchor_traits(anchor: Mapping) -> dict:
    """Acoustic-only traits contributed by an anchor to the design instruction."""
    traits = {}
    for tag in anchor.get("tags") or []:
        dimension = _TAG_TRAITS.get(str(tag).casefold())
        if dimension:
            traits.setdefault("delivery", dimension)
    return traits


def design_fragment(anchor: Mapping) -> str:
    """The clause that ties a candidate's design to its specific reference.

    It states the reference and its useful characteristics and then requires an
    original voice: the reference is a casting intention, not a target recording.
    """
    if not anchor.get("anchored"):
        return (f"Unanchored exploration: {anchor.get('unanchored_reason')} "
                "Do not fall back to a generic synthetic narrator.")
    matched = ", ".join(anchor.get("matched_traits") or []) or "exploratory fit"
    return (f"Reference performance for casting intent: {anchor['performer']}, {anchor['performance']}. "
            f"Useful characteristics to inform an original voice: {matched}. "
            "Produce an original synthetic character voice informed by these characteristics; "
            "do not reproduce the reference recording, and do not copy any real person's voice.")
