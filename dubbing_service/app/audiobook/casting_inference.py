"""Casting inference for characters the book describes sparsely.

A dossier often establishes only one or two vocal facts — "young male", "an old
woman", nothing at all. The previous behaviour fell back to a generically
described synthetic voice, which is exactly the outcome a human reviewer cannot
judge: every option sounds like the same anonymous narrator.

This module replaces that fallback with *casting inference*. Missing dimensions
are supplied as an explicit, labelled interpretation with the concept that
requires them, and each concept becomes a deliberately differentiated performer
reference. Two rules keep this honest:

* a fact from the book and a casting interpretation are never mixed — the
  supported set keeps the dossier's own evidence ids and levels, and the inferred
  set carries ``level="CASTING_INFERENCE"`` plus the basis for the interpolation;
* an inferred value never silently overrides supported evidence.
"""
from __future__ import annotations

from typing import Mapping

SCHEMA = "casting-inference-v1"
INFERENCE_VERSION = "casting-inference-v1"

#: The level every inferred dimension carries, so no consumer can mistake it for
#: something the book said.
INFERENCE_LEVEL = "CASTING_INFERENCE"

#: Dimensions this layer may supply when the dossier is silent.
_INFERABLE = ("presentation", "perceived_vocal_age", "accent", "pitch", "weight", "resonance", "texture",
              "cadence", "tempo", "articulation", "warmth", "intensity", "confidence",
              "formality", "theatricality", "emotional_range", "breathiness", "roughness")

#: Voice presentations auditioned when the book establishes no gender at all.
#: Rotating them across the concepts is what turns the dimension the book left open
#: into something the human can actually decide by listening, instead of leaving
#: the reference search unconstrained and letting it land on one presentation.
_PRESENTATION_ROTATION = ("masculine", "feminine", "androgynous")

#: A casting concept: one deliberate interpretation of the character, its own
#: delivery emphasis, and the acoustic bias that makes it audibly distinct.
#:
#: ``dimensions`` fills *missing* dimensions only; ``accent_choices`` supplies an
#: accent when the book establishes none, which is what makes two otherwise
#: identical characters genuinely different auditions rather than two names for
#: one voice. ``prefer_tags`` ranks catalogue performances toward this concept.
CONCEPTS = (
    {"key": "warm-vulnerable", "label": "warmth and vulnerability",
     "emphasis": "softer, warmer delivery with an unguarded, vulnerable turn",
     "dimensions": {"warmth": "warm", "intensity": "restrained", "texture": "airy",
                    "cadence": "deliberate", "emotional_range": "expressive"},
     "accent_choices": ("British (neutral RP)", "American (General American)", None),
     "prefer_tags": ("warm", "soft", "earnest", "conversational", "vulnerable", "thoughtful", "measured"),
     "bias": {"pitch_semitones": -0.7, "rate": 0.94, "energy_db": -1.5, "brightness": 0.93,
              "breathiness": 0.25, "rasp": 0.06, "articulation": 0.96, "pause_ms": 1.15,
              "dynamic_range_db": 0.95}},
    {"key": "confident-witty", "label": "confidence and wit",
     "emphasis": "quicker, crisper delivery with dry confidence and comic timing",
     "dimensions": {"confidence": "controlled authority", "theatricality": "comic timing",
                    "articulation": "precise", "tempo": "brisk", "formality": "conversational"},
     "accent_choices": ("American (General American)", "British (neutral RP)", None),
     "prefer_tags": ("confident", "dry", "energetic", "clear", "theatrical", "lively", "expressive"),
     "bias": {"pitch_semitones": 0.4, "rate": 1.10, "energy_db": 1.5, "brightness": 1.07,
              "breathiness": 0.05, "rasp": 0.05, "articulation": 1.10, "pause_ms": 0.85,
              "dynamic_range_db": 1.15}},
    {"key": "deep-serious", "label": "depth and gravity",
     "emphasis": "lower, slower, weightier delivery with restrained dramatic authority",
     "dimensions": {"pitch": "low", "weight": "heavy", "resonance": "rounded",
                    "cadence": "deliberate", "intensity": "restrained"},
     "accent_choices": (None, "British (neutral RP)", "American (General American)"),
     "prefer_tags": ("resonant", "measured", "mature", "narrator", "authoritative", "warm"),
     "bias": {"pitch_semitones": -1.8, "rate": 0.88, "energy_db": 0.8, "brightness": 0.88,
              "breathiness": 0.02, "rasp": 0.12, "articulation": 1.04, "pause_ms": 1.25,
              "dynamic_range_db": 0.85}},
    {"key": "bright-youthful", "label": "youth and energy",
     "emphasis": "brighter, quicker delivery with youthful energy and light articulation",
     "dimensions": {"pitch": "mid-high", "weight": "light", "tempo": "brisk",
                    "intensity": "high energy", "texture": "clear"},
     "accent_choices": ("American (General American)", "British (neutral RP)", None),
     "prefer_tags": ("young", "energetic", "clear", "lively", "playful", "vulnerable"),
     "bias": {"pitch_semitones": 1.2, "rate": 1.06, "energy_db": 1.0, "brightness": 1.10,
              "breathiness": 0.12, "rasp": 0.0, "articulation": 1.02, "pause_ms": 0.90,
              "dynamic_range_db": 1.05}},
    {"key": "weathered-gravel", "label": "age and texture",
     "emphasis": "older, gravelled delivery with unhurried, worn-in phrasing",
     "dimensions": {"perceived_vocal_age": "older adult", "roughness": "rough",
                    "texture": "rough", "tempo": "slow", "breathiness": "airy"},
     "accent_choices": ("British (neutral RP)", None, "American (General American)"),
     "prefer_tags": ("mature", "resonant", "measured", "narrator", "gravel", "warm"),
     "bias": {"pitch_semitones": -1.2, "rate": 0.92, "energy_db": -0.5, "brightness": 0.85,
              "breathiness": 0.18, "rasp": 0.35, "articulation": 1.0, "pause_ms": 1.20,
              "dynamic_range_db": 0.90}},
)


def supported_traits(profile, dossier=None, *, policy=None) -> dict:
    """Resolved evidence and evidence-backed interpretations for this policy.

    Raw dossier facts never bypass the resolver's source priority, conflict
    abstention, adaptation opt-in or story boundary. Accent is read only from
    the profile, where it has already passed the voice-design safety filter.
    """
    from ..models.character import CharacterCastingPolicy
    from .casting_profiles import _acoustic_decisions, _age
    from .character_research import _visible, effective_facts

    policy = policy if policy is not None else CharacterCastingPolicy()
    facts = {fact.id: fact for fact in getattr(dossier, "facts", []) or []}
    resolved = ({fact.field: fact for fact in effective_facts(dossier, policy)}
                if dossier is not None else {})
    scalar_fields = {"gender", "pronouns", "age", "age_range", "apparent_age", "accent", "language"}
    # A replacement round can read a profile cached before the current policy
    # or evidence revision. Validate acoustic dimensions through the same
    # canonical interpreter that built the profile, preserving cumulative traits.
    acoustic_fields = ("voice_description", "cadence", "speech_patterns", "vocabulary")
    acoustic = _acoustic_decisions(dossier, policy, resolved) if dossier is not None else {}
    supported: dict = {}
    for decision in getattr(profile, "decisions", []) or []:
        if decision.level == "UNKNOWN" or not decision.value or decision.value == "unspecified":
            continue
        if not decision.evidence_ids:
            continue  # An editorial placeholder, not evidence.
        if dossier is not None:
            evidence = [facts.get(identifier) for identifier in decision.evidence_ids]
            if any(fact is None or not _visible(fact, policy) or fact.field not in resolved
                   or (fact.field in scalar_fields and resolved[fact.field].id != fact.id)
                   for fact in evidence):
                continue  # A profile cached under a broader policy is not current evidence.
            if any(fact.field in acoustic_fields for fact in evidence):
                current = acoustic.get(decision.dimension)
                if current is None or current.value != decision.value or current.evidence_ids != decision.evidence_ids:
                    continue  # New higher-priority evidence supersedes this cached dimension.
        supported[decision.dimension] = {"value": decision.value, "level": decision.level,
                                         "evidence_ids": list(decision.evidence_ids),
                                         "basis": decision.reason}
    for field in ("gender", "pronouns", "age", "age_range", "language"):
        fact = resolved.get(field)
        if fact is None or field in supported or fact.level not in {"EXPLICIT", "SUPPORTED"}:
            continue
        value = fact.value
        if not isinstance(value, str) and not (
                field in {"age", "age_range"} and isinstance(value, (int, float)) and not isinstance(value, bool)):
            continue
        if not str(value).strip():
            continue
        supported[field] = {"value": str(value).strip(), "level": fact.level,
                            "evidence_ids": [fact.id], "basis": f"{fact.level.lower()} {field} evidence"}
    if "perceived_vocal_age" not in supported:
        age_fact = resolved.get("apparent_age") or resolved.get("age") or resolved.get("age_range")
        if age_fact is not None and age_fact.level in {"EXPLICIT", "SUPPORTED"}:
            age = _age(age_fact.value)
            if age != "unspecified":
                supported["perceived_vocal_age"] = {
                    "value": age, "level": "INFERRED", "evidence_ids": [age_fact.id],
                    "basis": f"Perceived age interpretation from story-scoped {age_fact.field}: {age_fact.value}",
                }
    return supported


def infer_dimensions(profile, dossier=None, *, concept, accent_index=0, policy=None) -> dict:
    """The casting interpretation one concept adds where the book is silent."""
    supported = supported_traits(profile, dossier, policy=policy)
    inferred: dict = {}
    for dimension, value in (concept.get("dimensions") or {}).items():
        if dimension in supported or dimension not in _INFERABLE:
            continue
        inferred[dimension] = {"value": value, "level": INFERENCE_LEVEL,
                               "basis": (f"Casting inference: the {concept['label']} concept needs "
                                         f"{dimension.replace('_', ' ')} and the book does not establish it"),
                               "evidence_ids": []}
    choices = concept.get("accent_choices") or (None,)
    accent = choices[accent_index % len(choices)]
    if accent and "accent" not in supported and "accent" not in inferred:
        inferred["accent"] = {"value": accent, "level": INFERENCE_LEVEL,
                              "basis": (f"Casting inference: the book establishes no accent, so the "
                                        f"{concept['label']} concept casts this character as {accent}"),
                              "evidence_ids": []}
    if "perceived_vocal_age" not in supported and "perceived_vocal_age" not in inferred:
        inferred["perceived_vocal_age"] = {
            "value": "adult", "level": INFERENCE_LEVEL,
            "basis": (f"Casting inference: the book establishes no age, so the {concept['label']} concept "
                      "casts this character as an adult rather than inventing an age from a name"),
            "evidence_ids": []}
    if "presentation" not in supported and "presentation" not in inferred:
        # The book established no gender, so auditioning one presentation would
        # present an interpretation as the character instead of offering a choice.
        value = _PRESENTATION_ROTATION[accent_index % len(_PRESENTATION_ROTATION)]
        inferred["presentation"] = {
            "value": value, "level": INFERENCE_LEVEL,
            "basis": (f"Casting inference: the book establishes no gender, so the {concept['label']} concept "
                      f"auditions a {value} voice; the other options audition different presentations"),
            "evidence_ids": []}
    return inferred


def _age_band(supported: Mapping) -> str:
    value = str((supported.get("perceived_vocal_age") or {}).get("value")
                or (supported.get("age") or {}).get("value") or "").casefold()
    if any(word in value for word in ("child", "youth", "teen", "adolesc", "young")):
        return "young"
    if any(word in value for word in ("elder", "old", "senior", "mature", "aged")):
        return "old"
    return "unspecified"


def _is_narrator(role) -> bool:
    if role is None:
        return False
    from ..models.enums import RoleType
    role_type = getattr(role, "role_type", None)
    value = getattr(role_type, "value", None) or role_type
    return value == RoleType.NARRATOR.value


def concept_order(supported: Mapping, role=None) -> tuple:
    """Which concepts are worth proposing, most fitting first, for this character.

    A young character is not auditioned against a weathered older reference and
    an older character is not auditioned against a bright youthful one: the
    inferred dimensions would then contradict the one thing the book did say. A
    narrator additionally leads with sustained-delivery concepts, because a voice
    that carries a whole book is not cast on comic timing.
    """
    band = _age_band(supported)
    if band == "young":
        order = ("warm-vulnerable", "confident-witty", "bright-youthful", "deep-serious", "weathered-gravel")
    elif band == "old":
        order = ("deep-serious", "weathered-gravel", "warm-vulnerable", "confident-witty", "bright-youthful")
    elif _is_narrator(role):
        order = ("deep-serious", "warm-vulnerable", "confident-witty", "weathered-gravel", "bright-youthful")
    else:
        order = ("warm-vulnerable", "confident-witty", "deep-serious", "bright-youthful", "weathered-gravel")
    by_key = {concept["key"]: concept for concept in CONCEPTS}
    return tuple(by_key[key] for key in order)


def casting_concepts(profile, dossier=None, *, count: int = 3, role=None, policy=None) -> list[dict]:
    """``count`` differentiated casting concepts for one character, best first.

    Each concept carries its own inferred dimensions, its own acoustic bias and
    its own accent choice, so the resulting candidates differ in casting
    dimension rather than only in performer name.
    """
    count = max(1, int(count))
    supported = supported_traits(profile, dossier, policy=policy)
    concepts = []
    for index, concept in enumerate(concept_order(supported, role)[:count]):
        inferred = infer_dimensions(profile, dossier, concept=concept, accent_index=index, policy=policy)
        concepts.append({
            "schema": SCHEMA, "version": INFERENCE_VERSION, "key": concept["key"],
            "label": concept["label"], "emphasis": concept["emphasis"],
            "prefer_tags": list(concept["prefer_tags"]),
            "bias": dict(concept["bias"]),
            "supported": {name: dict(row) for name, row in supported.items()},
            "inferred": inferred,
            "differing_dimensions": sorted(set(inferred) | {"delivery"}),
        })
    return concepts


def query_traits(concept) -> dict:
    """The vocal query for one concept: the book's facts plus that concept's cast."""
    traits = {name: row["value"] for name, row in (concept.get("supported") or {}).items()}
    for name, row in (concept.get("inferred") or {}).items():
        traits.setdefault(name, row["value"])
    return traits


def supported_summary(concept) -> list[dict]:
    """The book-supported traits as a display list, clearly not the interpretation."""
    return [{"dimension": name, "value": row["value"], "level": row["level"],
             "evidence_ids": row["evidence_ids"], "basis": row["basis"]}
            for name, row in sorted((concept.get("supported") or {}).items())]


def inferred_summary(concept) -> list[dict]:
    """The casting interpretation as a display list, clearly not a book fact."""
    return [{"dimension": name, "value": row["value"], "level": row["level"],
             "basis": row["basis"], "evidence_ids": []}
            for name, row in sorted((concept.get("inferred") or {}).items())]
