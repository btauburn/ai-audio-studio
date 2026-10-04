"""Deterministic, evidence-first casting interpretations of Studio dossiers."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

from ..models.character import CastingDecision, CastingProfile, CharacterFact
from ..models.enums import RoleType
from ..models.project import Project, Segment
from .cast_palette import PaletteEntry, match_inspiration, safe_traits
from .character_research import (
    LEVEL_PRIORITY, SOURCE_PRIORITY, _active_aliases, _visible, effective_facts,
)

PROFILE_VERSION = "evidence-casting-v3"

# Controlled output vocabulary prevents source prose from becoming instructions.
_VOICE_RULES = (
    ("pitch", "high", r"\b(?:high[- ]pitched|high voice|squeak\w*|treble|soprano)\b"),
    ("pitch", "low", r"\b(?:deep|low[- ]pitched|low voice|bass|baritone)\b"),
    ("pitch", "mid-high", r"\b(?:higher|light[- ]pitched)\b"),
    ("texture", "rough", r"\b(?:rough|rasp\w*|gravel\w*|hoarse|husky)\b"),
    ("texture", "clear", r"\b(?:clear|bell[- ]like|crystalline)\b"),
    ("texture", "airy", r"\b(?:airy|breathy)\b"),
    ("resonance", "nasal", r"\b(?:nasal|nasally)\b"),
    ("resonance", "rounded", r"\b(?:resonant|rich|rounded|sonorous)\b"),
    ("cadence", "deliberate", r"\b(?:slow(?:ly)?|deliberate(?:ly)?|unhurried|measured)\b"),
    ("cadence", "brisk", r"\b(?:fast|quick(?:ly)?|rapid|brisk)\b"),
    ("articulation", "precise", r"\b(?:precise|crisp|clipped|careful|enunciated)\b"),
    ("warmth", "warm", r"\b(?:warm|gentle|kind|soft[- ]spoken)\b"),
    ("intensity", "restrained", r"\b(?:restrained|quiet|softly|soft[- ]spoken)\b"),
    ("confidence", "controlled authority", r"\b(?:authoritative|commanding|controlled|confident)\b"),
)


def scene_cooccurrence(project: Project, segments: list[Segment], window: int = 4) -> dict[str, dict[str, float]]:
    """Nearby utterances approximate scene contact; no whole-book co-presence.

    Windows include narration so a long intervening passage breaks contact.
    Chapter and source boundaries are hard stops. The public weight is bounded
    to [0, 1], increasing as repeated exchanges supply more evidence.
    """
    characters = {role.id for role in project.roles if role.role_type == RoleType.CHARACTER}
    chapters: dict[str | None, list[Segment]] = defaultdict(list)
    for segment in segments:
        if not (segment.lineage and segment.lineage.retired):
            chapters[segment.chapter_id].append(segment)
    counts: Counter = Counter()
    for chapter in chapters.values():
        chapter.sort(key=lambda s: s.index)
        for index, left in enumerate(chapter):
            if left.role_id not in characters:
                continue
            for distance, right in enumerate(chapter[index + 1:index + window + 1], 1):
                if right.role_id not in characters or left.role_id == right.role_id:
                    continue
                if right.index - left.index > window:
                    continue
                if left.source_anchor and right.source_anchor:
                    if left.source_anchor.source_revision_id != right.source_anchor.source_revision_id:
                        continue
                    start, end = left.source_anchor.start_offset, right.source_anchor.start_offset
                    if start is not None and end is not None and abs(end - start) > 2000:
                        continue
                pair = tuple(sorted((left.role_id, right.role_id)))
                counts[pair] += 1 / distance
    result: dict[str, dict[str, float]] = {role: {} for role in characters}
    for (left, right), count in counts.items():
        weight = round(1 - math.exp(-count / 4), 4)
        result[left][right] = result[right][left] = weight
    return result


def _age(value: object) -> str:
    try:
        age = float(value)
    except (ValueError, TypeError):
        text = str(value).lower()
        if re.search(r"child|teen|adolescent|young|youth", text):
            return "youthful"
        if re.search(r"elder|old|senior", text):
            return "older adult"
        return "adult" if "adult" in text else "unspecified"
    return "child" if age < 13 else "adolescent" if age < 20 else "young adult" if age < 30 else "older adult" if age >= 65 else "mature adult"


def _decision(dimension: str, value: str, fact: CharacterFact, reason: str,
              *, weight: float = 4, infer: bool = False) -> CastingDecision:
    return CastingDecision(dimension=dimension, value=value, reason=reason,
                           evidence_ids=[fact.id], level="INFERRED" if infer else fact.level,
                           weight=weight)


def _affirmed(pattern: str, value: str) -> bool:
    for match in re.finditer(pattern, value):
        prefix = re.split(r"[;,.]|\bbut\b|\bhowever\b", value[:match.start()])[-1]
        if not re.search(r"\b(?:not|never|no|without|neither)\s+(?:\w+[\s-]+){0,2}$", prefix):
            return True
    return False


def design_instruction(profile: CastingProfile) -> str:
    """Compile a traits-only prompt; metadata names never cross this boundary."""
    clauses = ["Create an original, consistent character voice."]
    acoustic = {"perceived_vocal_age", "pitch", "weight", "resonance", "texture",
                "articulation", "cadence", "tempo", "formality", "warmth", "intensity",
                "confidence", "breathiness", "roughness", "theatricality", "accent",
                "presentation", "emotional_range"}
    values = {d.dimension: d.value for d in profile.decisions
              if d.dimension in acoustic and d.level != "UNKNOWN" and d.value != "unspecified"}
    if profile.inspiration:
        # Explicit dimensions remain authoritative. Inferred neutral defaults
        # may be replaced by the editorial inspiration's abstract traits.
        protected = {d.dimension for d in profile.decisions if d.level in {"EXPLICIT", "SUPPORTED"}
                     or (d.dimension == "perceived_vocal_age" and d.evidence_ids)}
        if profile.inspiration.traits:
            traits = safe_traits(PaletteEntry(character=profile.inspiration.character,
                performer=profile.inspiration.performer, traits=profile.inspiration.traits))
            values.update({k: v for k, v in traits.items() if k not in protected})
    clauses.extend(f"{key.replace('_', ' ')}: {value}." for key, value in values.items())
    clauses.append("Keep this vocal identity stable across emotion and volume changes.")
    return " ".join(clauses)


def _acoustic_decisions(dossier, policy, facts) -> dict[str, CastingDecision]:
    """Canonical cumulative acoustic interpretations, shared by proposal validation."""
    decisions: dict[str, CastingDecision] = {}
    # A scalar resolver remains correct for gender/age. Acoustic traits are
    # cumulative: "deep" then "raspy" supplies pitch AND texture. A field
    # that the canonical resolver found contradictory remains excluded.
    # Source rank is global across fields, so official voice prose cannot
    # override cadence explicitly established by the source book.
    acoustic_fields = ("voice_description", "cadence", "speech_patterns", "vocabulary")
    acoustic_facts = [fact for fact in dossier.facts
                      if fact.field in acoustic_fields and fact.field in facts
                      and fact.level != "UNKNOWN" and fact.value is not None
                      and _visible(fact, policy)]
    acoustic_facts.sort(key=lambda fact: (
        SOURCE_PRIORITY[fact.source_kind], LEVEL_PRIORITY[fact.level],
        acoustic_fields.index(fact.field), -(fact.scope.chapter_start or 0),
        -fact.confidence, fact.id,
    ))
    for fact in acoustic_facts:
        field = fact.field
        value = str(fact.value).lower()
        for dimension, trait, pattern in _VOICE_RULES:
            if _affirmed(pattern, value) and dimension not in decisions:
                decisions[dimension] = _decision(dimension, trait, fact,
                    f"{fact.level.lower()} {field.replace('_', ' ')} evidence: {fact.value}", weight=8)
        if field == "vocabulary":
            for needle, trait in (("formal", "formal"), ("colloquial", "conversational")):
                if _affirmed(r"\b" + needle + r"\b", value) and "formality" not in decisions:
                    decisions["formality"] = _decision("formality", trait, fact,
                        f"Vocabulary evidence: {fact.value}", weight=5)
    return decisions


def build_profiles(project: Project, segments: list[Segment]) -> dict[str, CastingProfile]:
    """Return proposed profiles without changing locked identities or dossiers."""
    state, policy = project.character_cast, project.character_cast.policy
    role_aliases = getattr(state, "role_aliases", {})

    def canonical_role(role_id):
        seen = set()
        while role_id in role_aliases and role_id not in seen:
            seen.add(role_id)
            role_id = role_aliases[role_id]
        return role_id

    allowed_chapters = {c.id for c in project.chapters
                        if policy.allow_spoilers or policy.max_chapter is None or c.index <= policy.max_chapter}
    # Canonicalize on copies: source attribution and existing rendered takes
    # remain untouched while counts, auditions, and scene pressure share one ID.
    segments = [s.model_copy(update={"role_id": canonical_role(s.role_id)})
                for s in segments if not (s.lineage and s.lineage.retired)
                and (s.chapter_id in allowed_chapters or s.chapter_id is None)]
    counts = Counter(s.role_id for s in segments if s.role_id)
    chapter_counts: dict[str, set] = defaultdict(set)
    for segment in segments:
        chapter_counts[segment.role_id].add(segment.chapter_id)
    cooccurrence = scene_cooccurrence(project, segments)
    result = {}
    # Lock metadata informs proposals for the rest of the ensemble without
    # replacing an accepted identity. Principal voices get palette priority.
    ensemble = {role.id: role.voice_identity.inspiration for role in project.roles
                if role.voice_identity and role.voice_identity.inspiration}
    for role in sorted(project.roles, key=lambda r: (-counts[r.id], r.id)):
        dossier = state.dossiers.get(role.id)
        if role.id in role_aliases or role.role_type != RoleType.CHARACTER or dossier is None or not counts[role.id]:
            continue
        facts = {fact.field: fact for fact in effective_facts(dossier, policy)}
        decisions = _acoustic_decisions(dossier, policy, facts)
        gender = facts.get("gender")
        if gender and gender.level not in {"EXPLICIT", "SUPPORTED"}:
            gender = None
        if gender:
            # An established gender informs presentation, not an automatic pitch.
            known = {"male": "masculine", "man": "masculine", "boy": "masculine",
                     "female": "feminine", "woman": "feminine", "girl": "feminine",
                     "transgender woman": "feminine", "transgender man": "masculine",
                     "nonbinary": "androgynous", "non-binary": "androgynous"}
            presentation = known.get(str(gender.value).lower(), "unspecified")
            decisions["presentation"] = _decision("presentation", presentation, gender,
                "Presentation is a casting interpretation of established gender; it does not prescribe pitch", infer=True)
        else:
            pronouns = facts.get("pronouns")
            pronoun_presentation = {"he/him": "masculine", "she/her": "feminine"}.get(
                str(pronouns.value).lower() if pronouns else "")
            if pronouns and pronouns.level in {"EXPLICIT", "SUPPORTED"} and pronoun_presentation:
                decisions["presentation"] = _decision("presentation", pronoun_presentation, pronouns,
                    "Provisional casting presentation inferred from supported pronoun usage; gender remains unknown. "
                    "This interpretation establishes neither vocal pitch nor age", weight=1, infer=True)
            else:
                decisions["presentation"] = CastingDecision(dimension="presentation", value="unspecified",
                    reason="No supported gender or applicable pronoun evidence; names do not establish gender", level="UNKNOWN", weight=0)
        age_fact = facts.get("apparent_age") or facts.get("age") or facts.get("age_range")
        if age_fact and _age(age_fact.value) != "unspecified":
            decisions["perceived_vocal_age"] = _decision("perceived_vocal_age", _age(age_fact.value), age_fact,
                f"Perceived age interpretation from story-scoped {age_fact.field}: {age_fact.value}", weight=5, infer=True)
        accent = facts.get("accent")
        if accent and accent.level in {"EXPLICIT", "SUPPORTED"}:
            # An accent is copied only from an evidence field, never from social
            # class, nationality, a name, or an inspiration performance.
            value = str(accent.value).strip()
            if (len(value) <= 100 and re.fullmatch(r"[\w '\-]+", value)
                    and not re.search(r"\b(?:like|clone|voice|actor|performer|instructions?)\b|imitat", value, re.I)):
                decisions["accent"] = _decision("accent", value, accent, "Accent established by character evidence", weight=7)
        for field in ("temperament", "personality", "role", "relationships", "social_context"):
            fact = facts.get(field)
            if not fact:
                continue
            value = str(fact.value).lower()
            for pattern, dimension, trait in (
                (r"\b(?:shy|cautious|nervous|reserved)\b", "confidence", "cautious"),
                (r"\b(?:stern|authority|authoritative|confident|leader)\b", "confidence", "controlled authority"),
                (r"\b(?:kind|gentle|warm|compassionate)\b", "warmth", "warm"),
                (r"\b(?:volatile|excitable|energetic)\b", "intensity", "high energy"),
                (r"\b(?:formal|scholar|professor|teacher)\b", "formality", "formal"),
                (r"\b(?:comic|witty|humorous)\b", "theatricality", "comic timing"),
            ):
                if re.search(pattern, value) and dimension not in decisions:
                    decisions[dimension] = _decision(dimension, trait, fact,
                        f"Casting interpretation of {field}: {fact.value}", weight=2, infer=True)
        # Stable contrast defaults are editorial decisions, with no appearance
        # mapping. A large body never creates an unsubstantiated deep voice.
        if "pitch" not in decisions:
            decisions["pitch"] = CastingDecision(dimension="pitch", value="mid",
                reason="Neutral register pending audition; no explicit pitch evidence", weight=0.5)
        if "texture" not in decisions:
            decisions["texture"] = CastingDecision(dimension="texture", value="clear",
                reason="Neutral intelligible texture pending audition", weight=0.5)
        count = counts[role.id]
        importance = "major" if count >= 6 else "recurring" if count >= 2 or len(chapter_counts[role.id]) >= 2 else "minor"
        reasons = list(dossier.review_reasons)
        if dossier.conflicts:
            reasons.append("Contradictory character evidence requires review")
        if importance == "major":
            if not gender:
                reasons.append("Major-character gender is unknown; " +
                    ("presentation left unspecified" if decisions["presentation"].level == "UNKNOWN"
                     else "pronoun-based casting presentation is provisional"))
            if not age_fact:
                reasons.append("Major-character age is unknown; no vocal age asserted")
        if accent and accent.level == "INFERRED":
            reasons.append("Unsupported accent inference excluded from voice design")
        profile = CastingProfile(role_id=role.id, revision="", dossier_revision=dossier.revision,
            decisions=list(decisions.values()), design_instruction="", importance=importance,
            dialogue_count=count, contrast_with=cooccurrence.get(role.id, {}),
            review_reasons=list(dict.fromkeys(reasons)))
        if policy.inspiration_palette:
            try:
                # An alias can itself reveal identity. A policy lowered after
                # later chapters were read must also hide it from adaptation
                # matching, while retaining the full append-only dossier.
                scoped_dossier = dossier.model_copy(update={"aliases": _active_aliases(dossier, policy)})
                profile.inspiration = match_inspiration(profile, scoped_dossier, policy.inspiration_palette,
                    policy.adaptation_requested, ensemble=ensemble)
                if not role.voice_identity:
                    ensemble[role.id] = profile.inspiration
                ensemble_reasons = [reason for reason in profile.inspiration.reasons
                                    if reason.startswith("INFERRED ensemble tie-break")]
                if ensemble_reasons:
                    profile.decisions.append(CastingDecision(dimension="contrast", value="distinct performance palette",
                        reason=ensemble_reasons[0], level="INFERRED", weight=1))
            except ValueError as exc:
                profile.review_reasons.append(str(exc))
        profile.design_instruction = design_instruction(profile)
        payload = profile.model_dump(mode="json", exclude={"created_at", "revision"})
        payload["algorithm"] = PROFILE_VERSION
        profile.revision = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        result[role.id] = profile
    return result
