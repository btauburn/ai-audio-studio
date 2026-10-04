"""Versioned editorial performance palettes, never celebrity audio or prompts.

Official links establish only the actor/character association. Acoustic traits
are explicitly editorial casting interpretations, not measured actor biometrics
or canonical facts about a book character. No audio download API exists here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from ..models.character import CastingProfile, CharacterDossier, InspirationMatch

PALETTE_VERSION = "editorial-palette-v1"
TRAIT_DIMENSIONS = frozenset({
    "perceived_vocal_age", "pitch", "weight", "resonance", "texture",
    "articulation", "cadence", "tempo", "formality", "warmth", "intensity",
    "confidence", "breathiness", "roughness", "theatricality", "emotional_range",
})
# A custom palette supplies descriptors, not arbitrary instructions. Keeping a
# vocabulary at this external boundary also rejects unrelated celebrity names
# whose identity was not repeated in the entry's metadata.
_TRAIT_WORDS = frozenset("""
    youthful child adolescent teen young mature adult older elderly ageless
    low lower mid middle high higher register pitch pitched light lighter heavy
    weight weighted rich thin full rounded round focused resonant resonance
    bright dark clear airy breathy rough lightly slightly subtly smooth smoky
    husky hoarse raspy gravelly dry edged grainy velvety nasal soft firm crisp
    precise careful sharp relaxed natural clean clipped articulate articulation
    deliberate measured unhurried brisk quick fast slow conversational direct
    rhythmic flowing halting even uneven mercurial steady varied melodic cadence
    tempo timing pauses short long formal informal refined casual neutral warmth
    warm cool reserved gentle kind restrained moderate intense intensity energy
    energetic calm lively animated quiet forceful assured confident cautious shy
    hesitant controlled authority confident confidence commanding theatrical
    theatricality comic dramatic understated expressive restrained broad limited
    emotional range playful serious deadpan flexible subtle balanced and with
    a an the of in very slightly lightly mostly but consistently
""".split())


@dataclass(frozen=True)
class PaletteEntry:
    character: str
    performer: str | None = None
    traits: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    source: str = "user-supplied editorial palette"


_YOUNG = "https://www.harrypotter.com/features/actors-of-the-wizarding-world-part-one-the-stars-that-rose"
_ADULT = "https://www.harrypotter.com/features/actors-of-the-wizarding-world-part-three-the-household-names"
HARRY_POTTER_FILMS = (
    PaletteEntry("Harry Potter", "Daniel Radcliffe", {
        "perceived_vocal_age": "youthful", "pitch": "mid", "texture": "clear",
        "cadence": "direct", "warmth": "warm", "confidence": "cautious",
    }, ("young", "protagonist", "earnest", "cautious", "brave"), ("Harry",), _YOUNG),
    PaletteEntry("Hermione Granger", "Emma Watson", {
        "perceived_vocal_age": "youthful", "pitch": "mid-high", "articulation": "precise",
        "cadence": "brisk", "confidence": "assured", "formality": "formal",
    }, ("young", "scholar", "intelligent", "precise", "confident"), ("Hermione",), _YOUNG),
    PaletteEntry("Ron Weasley", "Rupert Grint", {
        "perceived_vocal_age": "youthful", "pitch": "mid", "texture": "lightly rough",
        "cadence": "conversational", "warmth": "warm", "theatricality": "comic timing",
    }, ("young", "friend", "comic", "loyal", "informal"), ("Ron",), _YOUNG),
    PaletteEntry("Severus Snape", "Alan Rickman", {
        "perceived_vocal_age": "mature adult", "pitch": "low-mid", "resonance": "focused",
        "cadence": "deliberate", "intensity": "restrained", "confidence": "controlled authority",
    }, ("adult", "teacher", "authority", "stern", "controlled", "reserved"), ("Snape", "Professor Snape"), _ADULT),
    PaletteEntry("Minerva McGonagall", "Maggie Smith", {
        "perceived_vocal_age": "older adult", "pitch": "mid", "articulation": "crisp",
        "cadence": "measured", "confidence": "controlled authority", "warmth": "reserved",
    }, ("older", "teacher", "mentor", "stern", "formal", "authority"), ("McGonagall", "Professor McGonagall"), _ADULT),
    PaletteEntry("Rubeus Hagrid", "Robbie Coltrane", {
        "perceived_vocal_age": "mature adult", "pitch": "low-mid", "texture": "rough",
        "resonance": "rounded", "cadence": "unhurried", "warmth": "warm",
    }, ("adult", "guardian", "kind", "warm", "informal", "gentle"), ("Hagrid",),
        "https://www.harrypotter.com/features/10-things-you-might-not-know-about-hagrid"),
    PaletteEntry("Luna Lovegood", "Evanna Lynch", {
        "perceived_vocal_age": "youthful", "pitch": "mid-high", "texture": "airy",
        "cadence": "unhurried", "intensity": "gentle", "warmth": "warm",
    }, ("young", "reflective", "dreamy", "gentle", "calm", "quiet"), ("Luna",), _YOUNG),
    PaletteEntry("Bellatrix Lestrange", "Helena Bonham Carter", {
        "perceived_vocal_age": "mature adult", "pitch": "mid-high", "texture": "edged",
        "cadence": "mercurial", "intensity": "high energy", "theatricality": "dramatic",
    }, ("adult", "antagonist", "volatile", "intense", "dramatic"), ("Bellatrix",),
        "https://www.harrypotter.com/features/every-time-bellatrix-lestrange-gave-us-nightmares"),
)


def load_palette(palette: str | Sequence[PaletteEntry | Mapping] | Mapping) -> tuple[str, tuple[PaletteEntry, ...]]:
    """Resolve a named palette or a bounded custom ``{name, entries}`` value."""
    if isinstance(palette, str):
        key = re.sub(r"[^a-z0-9]", "", palette.lower())
        if key not in {"harrypotter", "harrypotterfilms", "harrypotterfilmcast", "hpfilms"}:
            raise ValueError(f"Unknown casting inspiration palette: {palette}")
        return "harry-potter-films-v1", HARRY_POTTER_FILMS
    name = "custom-palette-v1"
    if isinstance(palette, Mapping):
        name = str(palette.get("name", name))
        palette = palette.get("entries", ())
    if not palette or len(palette) > 64:
        raise ValueError("A palette requires between 1 and 64 entries")
    entries = tuple(item if isinstance(item, PaletteEntry) else PaletteEntry(**item) for item in palette)
    for entry in entries:
        safe_traits(entry)
    return name, entries


def safe_traits(entry: PaletteEntry) -> dict[str, str]:
    """Reject name-only or imitation instructions at the palette boundary."""
    result = {}
    forbidden = [name.lower() for name in (entry.character, entry.performer or "") if name]
    for key, raw in entry.traits.items():
        if key not in TRAIT_DIMENSIONS:
            raise ValueError(f"Unsupported inspiration dimension: {key}")
        value = str(raw).strip()
        words = set(re.findall(r"[a-z]+", value.lower()))
        if (not value or len(value) > 120 or words - _TRAIT_WORDS
                or any(re.search(r"\b" + re.escape(name) + r"\b", value.lower()) for name in forbidden)
                or re.search(r"https?://|www\.|clone|imitat|impersonat|sound(?:s|ing)? like|exact(?:ly)?|recording|audio file", value, re.I)):
            raise ValueError("Inspiration requires abstract performance traits, never identity or audio instructions")
        result[key] = value
    if not result:
        raise ValueError("A casting palette entry must contain acoustic/performance traits")
    return result


def _tokens(value: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", value.lower()))
    equivalents = {"youthful": "young", "child": "young", "adolescent": "young",
                    "elderly": "older", "deliberate": "measured", "unhurried": "measured",
                    "slow": "measured", "slowly": "measured", "brisk": "fast",
                    "assured": "confident", "cautious": "reserved", "quiet": "gentle"}
    return words | {equivalents[word] for word in words if word in equivalents}


def _age_band(value: str) -> str:
    words = _tokens(value)
    if words & {"young", "adolescent", "child", "teen"}:
        return "young"
    if words & {"older", "elderly", "senior"}:
        return "older"
    return "adult" if "adult" in words or "mature" in words else "unknown"


def match_inspiration(
    profile: CastingProfile, dossier: CharacterDossier,
    palette: str | Sequence[PaletteEntry | Mapping] | Mapping,
    adaptation_requested: bool = False,
    *, ensemble: Mapping[str, InspirationMatch] | None = None,
) -> InspirationMatch:
    """Match known performance dimensions; gender and names do not drive scoring.

    Only profile decisions enter semantic scoring: its dossier was already
    filtered at the configured story boundary. Raw dossier facts are not read.
    Names are used solely for opt-in direct adaptation matching.
    """
    name, entries = load_palette(palette)
    names = {x.casefold().strip() for x in (dossier.canonical_name, *dossier.aliases)}
    direct = [item for item in entries if names.intersection(
        x.casefold().strip() for x in (item.character, *item.aliases))] if adaptation_requested else []
    scores = []
    for entry in direct or entries:
        traits = safe_traits(entry)
        score, reasons = 0.0, []
        tags = _tokens(" ".join(entry.tags))
        for decision in profile.decisions:
            if decision.level == "UNKNOWN" or decision.dimension in {"presentation", "accent", "contrast"}:
                continue
            words = _tokens(decision.value)
            acoustic = _tokens(traits.get(decision.dimension, ""))
            matched = words & (acoustic | tags)
            weight = min(8, decision.weight) * (2 if decision.level == "EXPLICIT" else 1)
            if decision.dimension == "perceived_vocal_age":
                compatible_age = _age_band(decision.value) == _age_band(traits.get(decision.dimension, ""))
                score += weight * 2 if compatible_age else -weight * 3
                if compatible_age:
                    reasons.append(f"Vocal age fits {decision.value}")
                continue
            if matched:
                score += weight * (1.5 if acoustic & words else 0.5)
                reasons.append(f"{decision.dimension}: {decision.value} aligns with {', '.join(sorted(matched))}")
            elif decision.dimension in traits and decision.level == "EXPLICIT":
                score -= weight
        scores.append((score, entry, traits, reasons))
    # A bounded tie-break only operates among comparably fitting archetypes.
    # Strong evidence and direct adaptation mappings keep their priority. The
    # pressure is acoustic overlap, so different persona labels with identical
    # traits do not falsely count as an ensemble improvement.
    best_semantic_score = max(item[0] for item in scores)
    neighbors = {
        role_id: (match, profile.contrast_with[role_id])
        for role_id, match in (ensemble or {}).items()
        if role_id != profile.role_id and match.palette == name
        and profile.contrast_with.get(role_id, 0) >= 0.25
    }
    weighted = []
    for semantic_score, entry, traits, reasons in scores:
        penalty, contacts = 0.0, []
        if not direct and neighbors and semantic_score >= best_semantic_score - 1.0:
            for role_id, (neighbor, contact) in neighbors.items():
                dimensions = (set(traits) & set(neighbor.traits)) - {"perceived_vocal_age"}
                similarity = sum(traits[key] == neighbor.traits[key] for key in dimensions) / max(1, len(dimensions))
                if entry.character == neighbor.character:
                    similarity = 1.0
                pressure = 3.0 * contact * similarity
                penalty = max(penalty, pressure)
                if pressure:
                    contacts.append(f"{role_id} (shared-scene weight {contact:.2f})")
            weighted.append((semantic_score - penalty, semantic_score, entry, traits, reasons, contacts))
        elif direct or not neighbors:
            weighted.append((semantic_score, semantic_score, entry, traits, reasons, []))
    # At least the semantic leader is eligible. Stable palette order resolves
    # remaining ties without guessing a character's gender from their name.
    score, semantic_score, entry, traits, reasons, contacts = max(weighted, key=lambda item: item[0])
    base_entry = max(scores, key=lambda item: item[0])[1]
    if not direct and neighbors and entry.character != base_entry.character:
        reasons = reasons + [
            "INFERRED ensemble tie-break among similarly fitting archetypes: distinct performance traits "
            "reduce overlap with frequent scene partners " + ", ".join(sorted(neighbors))
            + f"; semantic score {semantic_score:.2f}, adjusted score {score:.2f}. This is casting interpretation, not character fact."
        ]
    # Inspiration cannot overwrite an established book vocal requirement.
    protected = {d.dimension for d in profile.decisions if d.level in {"EXPLICIT", "SUPPORTED"}
                 or (d.dimension == "perceived_vocal_age" and d.evidence_ids)}
    compatible = {key: value for key, value in traits.items() if key not in protected}
    return InspirationMatch(
        palette=name, character=entry.character, performer=entry.performer,
        mode="direct_adaptation" if direct else "archetype", traits=compatible,
        reasons=(["Explicitly requested direct adaptation mapping"] if direct else [])
        + reasons + (["Weak match: insufficient supported performance evidence; editorial suggestion only"]
                     if not any(d.evidence_ids and d.level != "UNKNOWN"
                                and d.dimension not in {"presentation", "accent", "contrast"}
                                for d in profile.decisions) else [])
        + ["Abstract editorial traits only; source establishes cast association, not measured vocal identity"],
        score=round(score, 4), source=entry.source,
    )
