"""Evidence-based character analysis and explainable audition ranking.

Pure and deterministic: no IO, network, inference, clock or randomness. Character
analysis reads only what the existing casting layer already resolves
(``character_research.effective_facts``, ``casting_profiles.build_profiles`` and
``casting_inference.supported_traits``); ranking compares that analysis with
JSON-compatible candidate dicts.

Honesty rules this module enforces:

* a trait is *supported* only when the dossier evidence behind it is visible under
  the project's story/spoiler/adaptation policy; names, appearance and library
  voice metadata never create a character trait or an accent;
* a quality dimension (fidelity, clarity, naturalness, stability) is scored only
  when the candidate carries a numeric measurement for it; otherwise it is
  ``None``/``unmeasured`` and listed as missing -- never defaulted or estimated;
* unmeasured dimensions contribute ``0`` to the overall score, so unmeasured
  candidates never outrank measured ones by omission;
* ranks are *recommendations*. Nothing here approves, assigns, synthesizes or
  asserts rights/consent; every result says ``human_review='pending'``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..models.enums import RoleType
from .casting_inference import supported_traits
from .casting_profiles import build_profiles, scene_cooccurrence
from .character_research import effective_facts

ALGORITHM = "casting-director-matching-v1"

#: Fewest distinct viable candidates a human can meaningfully choose between.
MIN_DISTINCT = 3
MAX_RANKED = 5

#: Candidate-vs-candidate trait similarity at or above this is the same audition.
NEAR_DUPLICATE = 0.85
#: Mild preference for variety when choosing the next recommended candidate.
DIVERSITY_WEIGHT = 0.15
#: A candidate trait the library does not state counts as this fraction of a miss.
UNKNOWN_FIT_PENALTY = 0.5
#: Scene co-occurrence at or above this marks a role pair that needs contrast.
CONTRAST_THRESHOLD = 0.25

SCORE_WEIGHTS = {
    "fit": 0.35,
    "distinctiveness": 0.20,
    "fidelity": 0.10,
    "clarity": 0.10,
    "naturalness": 0.10,
    "stability": 0.05,
    "human_rating": 0.10,
}
QUALITY_DIMENSIONS = ("fidelity", "clarity", "naturalness", "stability")

_LEVEL_TRUST = {"EXPLICIT": 1.0, "SUPPORTED": 0.8, "INFERRED": 0.4}
_FIT_WEIGHT = {"presentation": 3.0, "accent": 3.0, "perceived_vocal_age": 2.5, "pitch": 2.0,
               "texture": 1.5, "cadence": 1.5}
#: Scalar dossier fields already expressed through presentation/perceived age.
_FIT_SKIPPED = {"gender", "pronouns", "age", "age_range", "language"}
#: Dimensions where a known candidate value that contradicts evidence disqualifies it.
_HARD_DIMENSIONS = {"presentation", "perceived_vocal_age", "accent"}
_KEY_DIMENSIONS = ("presentation", "perceived_vocal_age", "accent", "pitch", "texture", "cadence")

_DIMENSION_ALIASES = {
    "gender": "presentation", "sex": "presentation", "age": "perceived_vocal_age",
    "vocal_age": "perceived_vocal_age", "voice_age": "perceived_vocal_age",
    "age_band": "perceived_vocal_age", "register": "pitch", "timbre": "texture",
}
_SYNONYMS = {
    "presentation": {"male": "masculine", "man": "masculine", "boy": "masculine",
                     "female": "feminine", "woman": "feminine", "girl": "feminine",
                     "nonbinary": "androgynous", "non-binary": "androgynous"},
    "pitch": {"deep": "low", "bass": "low", "baritone": "low", "low-pitched": "low",
              "treble": "high", "soprano": "high", "high-pitched": "high",
              "higher": "mid-high", "light-pitched": "mid-high", "medium": "mid", "neutral": "mid"},
    "texture": {"raspy": "rough", "rasp": "rough", "gravelly": "rough", "gravel": "rough",
                "husky": "rough", "hoarse": "rough", "breathy": "airy",
                "crystalline": "clear", "bell-like": "clear"},
    "warmth": {"gentle": "warm", "kind": "warm", "soft-spoken": "warm"},
    "cadence": {"slow": "deliberate", "slowly": "deliberate", "unhurried": "deliberate",
                "measured": "deliberate", "fast": "brisk", "quick": "brisk", "rapid": "brisk"},
    "tempo": {"slow": "deliberate", "unhurried": "deliberate", "measured": "deliberate",
              "fast": "brisk", "quick": "brisk", "rapid": "brisk"},
}
_PITCH_ORDER = {"low": 0.0, "mid": 1.0, "mid-high": 1.5, "high": 2.0}
_AGE_ORDER = (
    (r"\bchild|\bkid\b", 0.0), (r"\bteen|\badolescent", 1.0), (r"\byoungster|\byouthful|\byouth\b", 1.5),
    (r"\byoung adult|\byoung\b", 2.0), (r"\bolder|\belder|\bsenior|\bold\b|\baged\b", 5.0),
    (r"\bmature adult|\bmature\b|\bmiddle", 4.0), (r"\badult", 3.0),
)
_ACCENT_STOP = {"neutral", "general", "accent", "rp", "standard", "the", "a", "an", "of"}


# --------------------------------------------------------------------------- helpers
def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _unit(value: object) -> float | None:
    number = _number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _text(value: object) -> str:
    return " ".join(str(value).split()) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""


def _dimension(name: object) -> str:
    key = _text(name).casefold().replace(" ", "_")
    return _DIMENSION_ALIASES.get(key, key)


def _canon(dimension: str, value: object) -> str:
    text = _text(value).casefold()
    return _SYNONYMS.get(dimension, {}).get(text, text)


def _values(raw: object) -> list[str]:
    """Flatten a trait value (scalar, ``{"value": ..}`` or list) into strings."""
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    if isinstance(raw, (list, tuple, set, frozenset)):
        return [item for entry in raw for item in _values(entry)]
    text = _text(raw)
    return [text] if text else []


def _candidate_traits(candidate: Mapping) -> dict[str, set[str]]:
    traits: dict[str, set[str]] = {}
    raw = candidate.get("traits")
    for name, value in (raw.items() if isinstance(raw, Mapping) else ()):
        dimension = _dimension(name)
        found = {_canon(dimension, item) for item in _values(value)}
        if found:
            traits.setdefault(dimension, set()).update(found)
    return traits


def _candidate_tags(candidate: Mapping) -> set[str]:
    tags = set()
    for item in _values(candidate.get("tags") or []):
        tags.add(item.casefold())
        for synonyms in _SYNONYMS.values():
            if item.casefold() in synonyms:
                tags.add(synonyms[item.casefold()])
    return tags


def _tokens(traits: Mapping[str, set[str]], tags: set[str]) -> frozenset[str]:
    return frozenset({value for values in traits.values() for value in values} | tags)


def _similarity(left: frozenset[str], right: frozenset[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def _identity_keys(item: Mapping) -> set[str]:
    keys = set()
    for field, prefix in (("library_voice_id", "lib"), ("key", "key"), ("candidate_id", "cand")):
        value = _text(item.get(field))
        if value:
            keys.add(f"{prefix}:{value}")
    return keys


def _age_ordinal(value: object) -> float | None:
    number = _number(value)
    if number is not None:
        return 0.0 if number < 13 else 1.0 if number < 20 else 2.0 if number < 30 else 5.0 if number >= 65 else 4.0 if number >= 45 else 3.0
    text = _text(value).casefold()
    for pattern, ordinal in _AGE_ORDER:
        if re.search(pattern, text):
            return ordinal
    return None


def _accent_tokens(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", value.casefold()) if word not in _ACCENT_STOP}


# --------------------------------------------------------------------------- analysis
def _canonical_segments(project, segments) -> list:
    aliases = getattr(project.character_cast, "role_aliases", {}) or {}
    policy = project.character_cast.policy

    def canonical(role_id):
        seen = set()
        while role_id in aliases and role_id not in seen:
            seen.add(role_id)
            role_id = aliases[role_id]
        return role_id

    allowed = {c.id for c in project.chapters
               if policy.allow_spoilers or policy.max_chapter is None or c.index <= policy.max_chapter}
    return [s.model_copy(update={"role_id": canonical(s.role_id)}) for s in segments
            if not (s.lineage and s.lineage.retired) and (s.chapter_id in allowed or s.chapter_id is None)]


def _fact_row(fact) -> dict:
    return {"fact_id": fact.id, "field": fact.field, "value": fact.value, "level": fact.level,
            "source_kind": fact.source_kind, "confidence": fact.confidence,
            "location": fact.location}


def analyze_characters(project, segments) -> list[dict]:
    """One evidence-bound casting analysis per speaking narrator/character role.

    ``traits`` holds only evidence-backed values (``trait_detail`` keeps level,
    evidence ids and basis); ``missing_evidence`` names key dimensions the book does
    not establish so ranking treats them as open, not as defaults.
    """
    state, policy = project.character_cast, project.character_cast.policy
    segments = _canonical_segments(project, list(segments))
    counts = Counter(s.role_id for s in segments if s.role_id)
    chapters: dict[str, set] = {}
    for segment in segments:
        chapters.setdefault(segment.role_id, set()).add(segment.chapter_id)
    contact = scene_cooccurrence(project, segments)
    profiles = build_profiles(project, segments)
    names = {role.id: role.name for role in project.roles}
    aliases = getattr(state, "role_aliases", {}) or {}
    analyses = []
    for role in sorted(project.roles, key=lambda r: (-counts[r.id], r.id)):
        if role.id in aliases or not counts[role.id] or role.role_type not in (RoleType.CHARACTER, RoleType.NARRATOR):
            continue
        dossier = state.dossiers.get(role.id)
        profile = profiles.get(role.id)
        supported = supported_traits(profile, dossier, policy=policy) if profile and dossier else {}
        resolved = {f.field: f for f in effective_facts(dossier, policy)} if dossier else {}
        count = counts[role.id]
        importance = "major" if count >= 6 else "recurring" if count >= 2 or len(chapters[role.id]) >= 2 else "minor"
        trait_detail = {name: {"value": row["value"], "level": row["level"],
                               "evidence_ids": list(row["evidence_ids"]), "basis": row["basis"]}
                        for name, row in sorted(supported.items())}
        evidence = [{"dimension": name, **row} for name, row in trait_detail.items()]
        group = lambda *fields: [_fact_row(resolved[f]) for f in fields if f in resolved]  # noqa: E731
        relationships = [
            {"role_id": other, "role_name": names.get(other, other), "scene_weight": weight,
             "basis": "scene co-occurrence within a short utterance window",
             "needs_contrast": weight >= CONTRAST_THRESHOLD}
            for other, weight in sorted(contact.get(role.id, {}).items(), key=lambda kv: (-kv[1], kv[0]))]
        trust = sum(_FIT_WEIGHT.get(name, 1.0) * _LEVEL_TRUST.get(row["level"], 0.0)
                    for name, row in supported.items() if name not in _FIT_SKIPPED)
        confidence = min(1.0, trust / 8.0)
        if dossier and dossier.conflicts:
            confidence = max(0.0, confidence - 0.25)
        reasons = list(dict.fromkeys([*(profile.review_reasons if profile else []),
                                      *(dossier.review_reasons if dossier else [])]))
        if dossier is None:
            reasons.append("No character dossier: nothing is established about this voice")
        analysis = {
            "algorithm": ALGORITHM,
            "role_id": role.id, "role_name": role.name, "role_type": role.role_type.value,
            "functions": ["narration"] if role.role_type == RoleType.NARRATOR else [],
            "evidence": evidence,
            "traits": {name: row["value"] for name, row in trait_detail.items()},
            "trait_detail": trait_detail,
            "temperament": group("temperament", "personality"),
            "habits": group("speech_patterns", "vocabulary", "cadence"),
            "relationship_facts": group("relationships", "social_context", "role"),
            "relationships": relationships,
            "importance": importance,
            "importance_score": round(1 - math.exp(-count / 12), 4),
            "spoken_lines": count,
            "chapters_spoken": len(chapters[role.id] - {None}),
            "missing_evidence": [d for d in _KEY_DIMENSIONS if d not in supported],
            "conflicts": [list(pair) for pair in (dossier.conflicts if dossier else [])],
            "review_reasons": reasons,
            "confidence": round(confidence, 3),
        }
        analysis["evidence_fingerprint"] = _digest({
            key: analysis[key] for key in ("role_id", "trait_detail", "temperament", "habits",
                                           "relationship_facts", "relationships", "importance",
                                           "missing_evidence", "conflicts")})
        analyses.append(analysis)
    return analyses


# --------------------------------------------------------------------------- fit
def _match(dimension: str, wanted: str, have: set[str]) -> float | None:
    """Score one evidenced trait against the candidate's stated values (None = unknown)."""
    wanted = _canon(dimension, wanted)
    if not have:
        return None
    if wanted in have:
        return 1.0
    if dimension == "pitch" and wanted in _PITCH_ORDER:
        known = [_PITCH_ORDER[v] for v in have if v in _PITCH_ORDER]
        return max(0.0, 1 - min(abs(_PITCH_ORDER[wanted] - k) for k in known) / 2) if known else 0.0
    if dimension == "perceived_vocal_age":
        target, known = _age_ordinal(wanted), [o for o in map(_age_ordinal, have) if o is not None]
        if target is None or not known:
            return None
        return max(0.0, 1 - min(abs(target - k) for k in known) / 3)
    if dimension == "presentation":
        return 0.5 if "androgynous" in have or wanted == "androgynous" else 0.0
    if dimension == "accent":
        mine = _accent_tokens(wanted)
        return 1.0 if any(mine & _accent_tokens(v) for v in have) else 0.0
    return 1.0 if any(wanted in v.split() or v in wanted.split() for v in have) else 0.0


def _fit(analysis: Mapping, traits: Mapping[str, set[str]], tags: set[str]) -> dict:
    detail = analysis.get("trait_detail") or {}
    matched = known = unknown = 0.0
    hits, misses, open_, conflicts = [], [], [], []
    for name, row in sorted(detail.items()):
        if name in _FIT_SKIPPED or not _text(row.get("value")):
            continue
        weight = _FIT_WEIGHT.get(name, 1.0) * _LEVEL_TRUST.get(row.get("level"), 0.4)
        wanted = _canon(name, row["value"])
        score = _match(name, wanted, traits.get(name, set()))
        if score is None and (wanted in tags or (name == "accent" and _accent_tokens(wanted) & tags)):
            score = 1.0
        if score is None:
            unknown += weight
            open_.append(name)
            continue
        known += weight
        matched += weight * score
        (hits if score >= 0.999 else misses).append(f"{name}: {row['value']}")
        if score == 0.0 and name in _HARD_DIMENSIONS:
            conflicts.append(f"{name} is {row['value']} in the evidence but the candidate states "
                             f"{', '.join(sorted(traits.get(name, set())))}")
    total = known + unknown
    if total == 0 or known == 0:
        basis = ("No supported character evidence to compare against; fit is not assessed" if total == 0
                 else "The candidate states none of the evidenced traits; fit is not assessed")
        return {"score": None, "status": "unmeasured", "weight": SCORE_WEIGHTS["fit"], "basis": basis,
                "hits": [], "misses": [], "open": open_, "conflicts": [], "coverage": 0.0}
    score = matched / (known + UNKNOWN_FIT_PENALTY * unknown)
    return {"score": round(score, 4), "status": "derived", "weight": SCORE_WEIGHTS["fit"],
            "basis": (f"{len(hits)} matched, {len(misses)} differing, {len(open_)} unstated of "
                      f"{len(hits) + len(misses) + len(open_)} evidenced traits; unstated traits count as "
                      f"{UNKNOWN_FIT_PENALTY:g} of a miss"),
            "hits": hits, "misses": misses, "open": open_, "conflicts": conflicts,
            "coverage": round(known / total, 4)}


# --------------------------------------------------------------------------- ranking
def _quality(candidate: Mapping) -> dict[str, dict]:
    raw = candidate.get("quality")
    raw = raw if isinstance(raw, Mapping) else {}
    result = {}
    for name in QUALITY_DIMENSIONS:
        entry = raw.get(name)
        source = entry.get("source") if isinstance(entry, Mapping) else None
        value = _unit(entry.get("value") if isinstance(entry, Mapping) else entry)
        if value is None:
            note = "no valid measurement supplied" if name not in raw else "supplied value is not a number in [0, 1]"
            result[name] = {"score": None, "status": "unmeasured", "weight": SCORE_WEIGHTS[name],
                            "basis": f"{name} is unmeasured ({note}); excluded from evidence, scored 0 overall"}
        else:
            result[name] = {"score": round(value, 4), "status": "measured", "weight": SCORE_WEIGHTS[name],
                            "basis": f"{name} measurement supplied" + (f" by {_text(source)}" if _text(source) else "")}
    return result


def _human_rating(candidate: Mapping) -> dict:
    rating = candidate.get("rating")
    if isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 5:
        return {"score": round((rating - 1) / 4, 4), "status": "human", "weight": SCORE_WEIGHTS["human_rating"],
                "basis": f"Human rating {rating}/5 recorded earlier; not an approval"}
    return {"score": None, "status": "unmeasured", "weight": SCORE_WEIGHTS["human_rating"],
            "basis": "No human rating recorded"}


def _single_analysis(analysis) -> Mapping:
    if isinstance(analysis, Sequence) and not isinstance(analysis, (str, bytes, Mapping)):
        if len(analysis) != 1:
            raise ValueError("rank_candidates needs exactly one role analysis")
        analysis = analysis[0]
    if not isinstance(analysis, Mapping) or not analysis.get("role_id"):
        raise ValueError("analysis must be an analyze_characters() entry with a role_id")
    return analysis


def _cast_entries(cast: Iterable, role_id: str) -> tuple[list[dict], list[dict]]:
    """Other roles' voices, and this role's current voice(s)."""
    others, current = [], []
    for entry in cast or ():
        if not isinstance(entry, Mapping):
            continue
        entry = {**(entry["candidate"] if isinstance(entry.get("candidate"), Mapping) else {}), **entry}
        traits, tags = _candidate_traits(entry), _candidate_tags(entry)
        row = {"role_id": _text(entry.get("role_id")), "identity": _identity_keys(entry),
               "tokens": _tokens(traits, tags), "name": _text(entry.get("name")) or _text(entry.get("role_id"))}
        (current if row["role_id"] == role_id else others).append(row)
    return others, current


def _evaluate(analysis: Mapping, raw: Mapping, index: int, others: list[dict], current: list[dict],
              contact: Mapping[str, float]) -> dict:
    candidate = copy.deepcopy(dict(raw))
    candidate_id = _text(candidate.get("candidate_id")) or "cand-" + _digest(raw)[:12]
    traits, tags = _candidate_traits(candidate), _candidate_tags(candidate)
    tokens = _tokens(traits, tags)
    identity = _identity_keys({**candidate, "candidate_id": candidate_id})
    reasons: list[str] = []
    category = None
    holds = [h for h in (candidate.get("holds") or []) if h]
    if candidate.get("eligible") is not True:
        category, reasons = "ineligible", ["Eligibility was not asserted for this candidate"]
    elif holds:
        category, reasons = "held", [f"Hold: {_text(h) if not isinstance(h, Mapping) else _text(h.get('reason') or h)}" for h in holds]
    elif not tokens:
        category, reasons = "unassessable", ["No descriptive traits or tags; cannot be justified for any character"]

    fit = _fit(analysis, traits, tags)
    if fit["conflicts"] and category is None:
        category, reasons = "conflict", ["Contradicts character evidence: " + "; ".join(fit["conflicts"])]

    # Distinctiveness against the other cast voices; co-occurrence raises the stakes.
    worst, worst_name, unknown_roles, same_voice = 0.0, None, [], None
    for member in others:
        weight = contact.get(member["role_id"], 0.0)
        sim = 1.0 if identity & member["identity"] else _similarity(tokens, member["tokens"])
        if sim is None:
            unknown_roles.append(member["role_id"])
            continue
        if sim == 1.0 and identity & member["identity"] and weight > 0 and same_voice is None:
            same_voice = member
        penalty = sim * (0.4 + 0.6 * weight)
        if penalty > worst:
            worst, worst_name = penalty, member
    if same_voice is not None and category is None:
        category = "same_voice_cooccurring"
        reasons = [f"Same voice already cast for {same_voice['name'] or same_voice['role_id']}, who shares scenes with this role"]
    if not others:
        distinct = {"score": 1.0, "status": "derived", "basis": "No other cast voices to collide with"}
    elif len(unknown_roles) == len(others):
        distinct = {"score": None, "status": "unmeasured",
                    "basis": "Cast voices carry no comparable traits; distinctiveness is unmeasured"}
    else:
        who = worst_name["name"] or worst_name["role_id"] if worst_name else None
        distinct = {"score": round(1 - worst, 4), "status": "derived",
                    "basis": ("Closest cast voice" + (f" ({who})" if who else "") +
                              f": weighted similarity {worst:.2f}; scene contact raises the penalty")
                             + (f"; {len(unknown_roles)} cast voice(s) not comparable" if unknown_roles else "")}
    distinct["weight"] = SCORE_WEIGHTS["distinctiveness"]

    dimensions = {"fit": {k: fit[k] for k in ("score", "status", "weight", "basis")},
                  "distinctiveness": distinct, **_quality(candidate), "human_rating": _human_rating(candidate)}
    total = sum(SCORE_WEIGHTS.values())
    score = sum(d["weight"] * d["score"] for d in dimensions.values() if d["score"] is not None) / total
    measured = sum(d["weight"] for d in dimensions.values() if d["score"] is not None)
    coverage = measured / total
    confidence = coverage * (0.5 + 0.5 * float(analysis.get("confidence") or 0.0))
    missing = [name for name in QUALITY_DIMENSIONS if dimensions[name]["score"] is None]

    concerns = list(fit["misses"] and [f"differs on {', '.join(fit['misses'])}"] or [])
    if worst >= 0.5 and worst_name:
        concerns.append(f"sounds similar to {worst_name['name'] or worst_name['role_id']}")
    if fit["open"]:
        concerns.append(f"library does not state {', '.join(fit['open'])}")
    if candidate.get("protected"):
        concerns.append("protected voice: recommendation never changes or replaces it")
    if not _text((candidate.get("provenance") or {}).get("rights_status") if isinstance(candidate.get("provenance"), Mapping) else None):
        concerns.append("rights status not recorded in candidate provenance; none is assumed")
    is_current = any(identity & member["identity"] for member in current)
    name = _text(candidate.get("name")) or candidate_id
    if fit["score"] is None:
        headline = f"{name}: no supported character evidence, so fit was not assessed"
    else:
        headline = f"{name}: fit {fit['score']:.2f} ({len(fit['hits'])} of {len(fit['hits']) + len(fit['misses']) + len(fit['open'])} evidenced traits matched)"
    explanation = {
        "summary": f"{headline}. Recommendation only; a human must audition and approve.",
        "strengths": [f"matches {hit}" for hit in fit["hits"]],
        "concerns": concerns,
        "unmeasured": missing + (["human_rating"] if dimensions["human_rating"]["score"] is None else []),
        "basis": {name_: d["basis"] for name_, d in dimensions.items()},
        "evidence_ids": sorted({e for row in (analysis.get("trait_detail") or {}).values() for e in row.get("evidence_ids", [])}),
    }
    return {"candidate": candidate, "candidate_id": candidate_id, "index": index, "tokens": tokens,
            "identity": identity, "category": category, "reasons": reasons, "score": score,
            "confidence": confidence, "coverage": coverage, "dimensions": dimensions,
            "missing_quality": missing, "explanation": explanation, "is_current": is_current}


def casting_recommendation(analysis, candidates: Iterable[Mapping], *, cast: Iterable = (), limit: int = MAX_RANKED) -> dict:
    """Rank, deduplicate and report what was excluded and whether casting is still needed."""
    analysis = _single_analysis(analysis)
    limit = max(1, min(MAX_RANKED, int(limit)))
    role_id = analysis["role_id"]
    contact = {r["role_id"]: float(r.get("scene_weight") or 0.0) for r in analysis.get("relationships") or []}
    others, current = _cast_entries(cast, role_id)
    evaluated, excluded, seen = [], [], set()
    for index, raw in enumerate(candidates or ()):
        if not isinstance(raw, Mapping):
            excluded.append({"candidate_id": None, "name": None, "category": "invalid",
                             "reasons": ["Candidate is not a JSON object"]})
            continue
        item = _evaluate(analysis, raw, index, others, current, contact)
        if item["candidate_id"] in seen:
            item["category"], item["reasons"] = "duplicate", ["Duplicate candidate_id; the first occurrence is used"]
        seen.add(item["candidate_id"])
        if item["category"]:
            excluded.append({"candidate_id": item["candidate_id"], "name": _text(item["candidate"].get("name")) or None,
                             "category": item["category"], "reasons": item["reasons"]})
        else:
            evaluated.append(item)

    # Greedy selection: best score, mildly preferring variety, never a near-duplicate.
    picked: list[dict] = []
    pool = sorted(evaluated, key=lambda i: (-i["score"], -(i["candidate"].get("rating") or 0) if isinstance(i["candidate"].get("rating"), int) else 0, i["candidate_id"]))
    while pool and len(picked) < limit:
        for item in pool:
            sims = [1.0 if item["identity"] & p["identity"] else (_similarity(item["tokens"], p["tokens"]) or 0.0) for p in picked]
            item["_sim"] = max(sims, default=0.0)
        duplicates = [i for i in pool if i["_sim"] >= NEAR_DUPLICATE]
        for item in duplicates:
            twin = next(p for p in picked if item["identity"] & p["identity"] or (_similarity(item["tokens"], p["tokens"]) or 0) >= NEAR_DUPLICATE)
            excluded.append({"candidate_id": item["candidate_id"], "name": _text(item["candidate"].get("name")) or None,
                             "category": "near_duplicate",
                             "reasons": [f"Near-duplicate of higher-ranked {twin['candidate_id']} (similarity {item['_sim']:.2f}); not a distinct audition"]})
        pool = [i for i in pool if i["_sim"] < NEAR_DUPLICATE]
        if not pool:
            break
        best = min(pool, key=lambda i: (-(i["score"] - DIVERSITY_WEIGHT * i["_sim"]), i["candidate_id"]))
        best["selection_adjustment"] = round(-DIVERSITY_WEIGHT * best["_sim"], 4)
        picked.append(best)
        pool.remove(best)

    needs_casting = len(picked) < MIN_DISTINCT
    note = (f"Only {len(picked)} distinct viable candidate(s); at least {MIN_DISTINCT} are needed for a meaningful audition"
            if needs_casting else None)
    ranked = []
    for rank, item in enumerate(picked, 1):
        candidate = item["candidate"]
        candidate.update({
            "rank": rank, "role_id": role_id, "score": round(item["score"], 4),
            "confidence": round(item["confidence"], 3), "evidence_coverage": round(item["coverage"], 3),
            "explanation": item["explanation"], "dimensions": item["dimensions"],
            "missing_quality": item["missing_quality"], "selection_adjustment": item.get("selection_adjustment", 0.0),
            "is_current_voice": item["is_current"], "needs_casting": needs_casting, "viable_count": len(picked),
            "recommendation_only": True, "approved": False, "human_review": "pending",
            "analysis_fingerprint": analysis.get("evidence_fingerprint"), "algorithm": ALGORITHM,
        })
        candidate.setdefault("candidate_id", item["candidate_id"])
        ranked.append(candidate)
    return {"role_id": role_id, "ranked": ranked, "needs_casting": needs_casting,
            "needs_casting_reason": note, "viable_count": len(picked), "minimum_distinct": MIN_DISTINCT,
            "excluded": excluded, "human_review": "pending", "algorithm": ALGORITHM,
            "analysis_fingerprint": analysis.get("evidence_fingerprint")}


def rank_candidates(analysis, candidates: Iterable[Mapping], *, cast: Iterable = (), limit: int = MAX_RANKED) -> list[dict]:
    """Up to ``limit`` (max 5) distinct viable candidates, best first.

    Each entry extends the candidate with ``score``, ``confidence``,
    ``explanation``, ``dimensions`` (unmeasured quality is ``None``) and
    ``needs_casting`` (true when fewer than three distinct viable candidates exist).
    Use :func:`casting_recommendation` to also receive the excluded list.
    """
    return casting_recommendation(analysis, candidates, cast=cast, limit=limit)["ranked"]
