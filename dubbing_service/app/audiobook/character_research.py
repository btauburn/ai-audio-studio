"""Incremental, conservative character research over Studio's canonical text.

This extractor recognizes anchored statements, not names or broad word
co-occurrence. Unrecognized prose remains unknown. External providers return
evidence through an explicit boundary; they never write canonical state.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from dubbing_service.app.models.character import (
    CharacterCastingPolicy,
    CharacterDossier,
    CharacterFact,
    StoryScope,
    now,
)
from dubbing_service.app.models.project import Project, Role, Segment

EXTRACTOR_VERSION = "anchored-text-v3"
FACT_FIELDS = (
    "gender",
    "pronouns",
    "age",
    "age_range",
    "apparent_age",
    "appearance",
    "height",
    "build",
    "distinctive_features",
    "voice_description",
    "accent",
    "speech_patterns",
    "vocabulary",
    "cadence",
    "temperament",
    "personality",
    "social_context",
    "relationships",
    "role",
    "portrayal_changes",
    "scene_delivery",
)
SOURCE_PRIORITY = {
    "book": 0,
    "project": 1,
    "official": 2,
    "reference": 3,
    "adaptation": 4,
}
LEVEL_PRIORITY = {"EXPLICIT": 0, "SUPPORTED": 1, "INFERRED": 2, "UNKNOWN": 3}
_SAFE_EXTERNAL_FIELDS = {
    "appearance",
    "height",
    "build",
    "distinctive_features",
    "voice_description",
    "accent",
    "speech_patterns",
    "vocabulary",
    "cadence",
    "pronouns",
    "gender",
}
_AGE_FIELDS = {"age", "age_range", "apparent_age"}
_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER = (
    r"(?:\d{1,4}|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[ -](?:one|two|three|four|five|six|seven|eight|nine)|"
    + "|".join(_NUMBERS)
    + r")"
)
_GENDER = r"(?:non[- ]?binary|transgender woman|transgender man|woman|girl|female|man|boy|male)"
_MODIFIER = rf"(?:(?:{_NUMBER})[- ]year[- ]old|young|old|elderly|adult|middle-aged|tall|short|small|thin|large|slender|little|human)"
_PREDICATE = r"\s*(?:,\s*)?(?:(?:is|was|were|remained|had been)\s+)?(?:an?\s+|the\s+)?"
# New depictions are admitted by _statements only after "See NAME" or
# "Consider NAME". No unrestricted pronoun/coreference inference is added.
_DEPICTION = (
    rf"\s+as\s+(?:he|she|they)\s+(?:was|is|were|are),\s+(?:the|an?)\s+"
    rf"(?:(?:{_MODIFIER}|quiet)\s+){{0,3}}"
    rf"(?P<kind>girl|boy|child|teenager|woman|man|person|young adult),\s+"
    rf"(?P<age>{_NUMBER})\s+years?\s+old\s*$"
)
_CHILD_DESCRIPTION = (
    r"\s*(?:was|is|had been)\s+"
    r"(?:only\s+an?\s+(?:quiet\s+)?|an?\s+quiet\s+)"
    r"(?P<kind>child|teenager)\b"
)
_DESCRIPTION_LABELS = {
    "gender": "gender",
    "pronouns": "pronouns",
    "age": "age",
    "age range": "age_range",
    "apparent age": "apparent_age",
    "appearance": "appearance",
    "height": "height",
    "build": "build",
    "voice": "voice_description",
    "voice description": "voice_description",
    "accent": "accent",
    "dialect": "accent",
    "cadence": "cadence",
    "speech patterns": "speech_patterns",
    "vocabulary": "vocabulary",
    "temperament": "temperament",
    "personality": "personality",
    "role": "role",
    "relationships": "relationships",
    "social context": "social_context",
}


class ExternalResearchProvider(Protocol):
    """A provider must identify its version and return provenance-bearing facts.

    Only official/reference/adaptation facts are accepted. Providers should
    avoid retrieving spoilers at all; the boundary independently filters their
    result before it can reach dossiers, profiles, or the persistent cache.
    """

    provider_id: str
    version: str

    def research(
        self,
        canonical_name: str,
        *,
        aliases: Sequence[str],
        source_title: str | None,
        max_chapter: int | None,
    ) -> Sequence[CharacterFact]: ...


class ExternalResearchUnavailable(RuntimeError):
    pass


class UnavailableExternalResearchProvider:
    provider_id = "unavailable"
    version = "1"

    def research(self, canonical_name: str, **kwargs) -> Sequence[CharacterFact]:
        raise ExternalResearchUnavailable(
            "No external character research provider configured"
        )


def _key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:24]


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    parts = re.split(r"[ -]", value.casefold())
    return (
        sum(_NUMBERS[x] for x in parts) if all(x in _NUMBERS for x in parts) else None
    )


def _gender(value: str) -> str:
    value = value.casefold().strip()
    if value in {"woman", "girl", "female", "transgender woman"}:
        return "female"
    if value in {"man", "boy", "male", "transgender man"}:
        return "male"
    return "nonbinary" if re.fullmatch(r"non[- ]?binary", value) else value


def _visible(
    fact: CharacterFact, policy: CharacterCastingPolicy, chapter: int | None = None
) -> bool:
    if fact.source_kind == "adaptation" and not policy.adaptation_requested:
        return False
    boundary = chapter if chapter is not None else policy.max_chapter
    if policy.max_chapter is not None and boundary is not None:
        boundary = min(boundary, policy.max_chapter)
    # Explicit chapter requests are temporal selection, even when spoilers are allowed.
    if chapter is not None:
        if fact.scope.chapter_start is not None and fact.scope.chapter_start > chapter:
            return False
        if fact.scope.chapter_end is not None and fact.scope.chapter_end < chapter:
            return False
    if policy.allow_spoilers:
        return True
    for known_at in (fact.available_after_chapter, fact.scope.chapter_start):
        if boundary is not None and known_at is not None and known_at > boundary:
            return False
    if fact.source_kind not in {"book", "project"}:
        known_at = fact.available_after_chapter
        if known_at is None:
            return fact.spoiler_safe and fact.field in _SAFE_EXTERNAL_FIELDS
    return True


def _active_aliases(
    dossier: CharacterDossier, policy: CharacterCastingPolicy
) -> list[str]:
    evidence = [f for f in dossier.facts if f.field == "aliases"]
    return [
        alias
        for alias in dossier.aliases
        if not any(_key(str(f.value)) == _key(alias) for f in evidence)
        or any(
            _key(str(f.value)) == _key(alias) and _visible(f, policy) for f in evidence
        )
    ]


def _overlap(left: CharacterFact, right: CharacterFact) -> bool:
    a, b = left.scope, right.scope
    if a.book != b.book or (
        a.story_time and b.story_time and a.story_time != b.story_time
    ):
        return False
    return max(a.chapter_start or 0, b.chapter_start or 0) <= min(
        a.chapter_end if a.chapter_end is not None else float("inf"),
        b.chapter_end if b.chapter_end is not None else float("inf"),
    )


def _contradict(left: CharacterFact, right: CharacterFact) -> bool:
    if left.field != right.field or not _overlap(left, right):
        return False
    if left.level in {"UNKNOWN", "INFERRED"} or right.level in {"UNKNOWN", "INFERRED"}:
        return False
    a, b = _key(str(left.value)), _key(str(right.value))
    if a == b:
        return False
    if left.field in {"gender", "age", "apparent_age", "accent"}:
        return True
    if left.field == "voice_description":
        pairs = (
            (
                r"\b(?:deep|low[- ]pitched|low register)\b",
                r"\b(?:high[- ]pitched|high register)\b",
            ),
            (r"\braspy\b", r"\bsmooth\b"),
        )
        return any(
            (re.search(x, a) and re.search(y, b))
            or (re.search(y, a) and re.search(x, b))
            for x, y in pairs
        )
    return False


def resolve_fact(
    dossier: CharacterDossier,
    field: str,
    policy: CharacterCastingPolicy,
    chapter: int | None = None,
) -> CharacterFact | None:
    """Resolve usable evidence by source, level and story scope, never by name.

    Same-priority contradictions remain unresolved. Lower-priority conflicting
    evidence is retained/reported but cannot override the source book.
    """
    choices = [
        f
        for f in dossier.facts
        if f.field == field
        and f.level != "UNKNOWN"
        and f.value is not None
        and _visible(f, policy, chapter)
    ]
    if not choices:
        return None
    rank = min(
        (SOURCE_PRIORITY[f.source_kind], LEVEL_PRIORITY[f.level]) for f in choices
    )
    choices = [
        f
        for f in choices
        if (SOURCE_PRIORITY[f.source_kind], LEVEL_PRIORITY[f.level]) == rank
    ]
    if field in _AGE_FIELDS and chapter is None:
        latest = max(
            f.scope.chapter_start if f.scope.chapter_start is not None else -1
            for f in choices
        )
        choices = [
            f
            for f in choices
            if (f.scope.chapter_start if f.scope.chapter_start is not None else -1)
            == latest
        ]
    if any(
        _contradict(left, right)
        for index, left in enumerate(choices)
        for right in choices[index + 1 :]
    ):
        return None
    return max(
        choices,
        key=lambda f: (
            f.confidence,
            f.scope.chapter_start or 0,
            len(str(f.value)),
            f.id,
        ),
    )


def effective_facts(
    dossier: CharacterDossier,
    policy: CharacterCastingPolicy,
    chapter: int | None = None,
) -> list[CharacterFact]:
    return [
        fact
        for field in sorted({f.field for f in dossier.facts})
        if (fact := resolve_fact(dossier, field, policy, chapter)) is not None
    ]


def _statements(text: str, names: Sequence[str]):
    """Yield only clauses immediately governed by a known character name.

    Quoted speech is removed: a character's claims about somebody else are
    not automatically narrator-established facts. Matching never bridges a
    sentence or another named subject.
    """
    narrative = re.sub(r'["“][^"”]*["”]', lambda m: " " * len(m.group()), text)
    name_pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True) if n)
        + r")(?!\w)"
    )
    for match in re.finditer(name_pattern, narrative, re.IGNORECASE):
        tail = narrative[match.end() :]
        stop = re.search(r"[.!?;\n]", tail)
        tail = tail[: stop.start()] if stop else tail
        next_subject = re.search(
            r"\b(?:and|but|while|whereas)\s+[A-Z][\w'-]*(?: [A-Z][\w'-]*)?\s+(?:was|is|had|has|spoke|speaks|said|wore|looked)\b",
            tail,
        )
        if next_subject:
            tail = tail[: next_subject.start()]
        prefix = re.split(r"[.!?;\n]", narrative[:match.start()])[-1]
        if re.match(r"\s+as\s+(?:he|she|they)\s+(?:was|is|were|are)\b", tail, re.IGNORECASE):
            if not re.fullmatch(r"\s*(?:See|Consider)\s+", prefix, re.IGNORECASE):
                continue
        if re.match(_CHILD_DESCRIPTION, tail, re.IGNORECASE):
            # Only these new child predicates need this strict assertion gate;
            # existing direct-predicate extraction retains its prior behavior.
            if not re.fullmatch(
                r"\s*(?:(?:But|Yet|And|Then)\s+|After all,\s+|I said before that\s+)?",
                prefix, re.IGNORECASE,
            ):
                continue
        if tail.strip():
            yield (
                tail[:400],
                text[match.start() : match.end() + min(len(tail), 260)].strip(),
            )


def _contextual_evidence(text: str, names: Sequence[str]):
    """Bounded grammatical evidence, never unrestricted pronoun coreference.

    A direct self-directed body action supplies pronoun evidence. A following
    pronoun is accepted only after a sentence with one named subject, no other
    names, human referents, or personal pronouns. This deliberately abstains
    from ordinary multi-person scenes. Pronouns are not converted to gender.
    """
    narrative = re.sub(r'["“][^"”]*["”]|<<.*?>>', lambda m: " " * len(m.group()), text)
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
        + r")(?!\w)"
    )
    pronouns = {
        "he": "he/him",
        "his": "he/him",
        "she": "she/her",
        "her": "she/her",
        "they": "they/them",
        "their": "they/them",
    }
    for match in re.finditer(pattern, narrative, re.IGNORECASE):
        tail = narrative[match.end() :]
        sentence = re.split(r"[.!?\n]", tail, maxsplit=1)[0][:250]
        prefix = narrative[: match.start()]
        # A descriptive noun immediately identifying a named character is
        # useful even in a book's synopsis (e.g. "a teenager named Rowan").
        descriptor = re.search(
            r"\b(?:a|an)\s+(teenager|child|young adult|elderly (?:man|woman)|boy|girl|man|woman)\s+(?:named|called)\s*$",
            prefix,
            re.IGNORECASE,
        )
        if descriptor:
            value = descriptor[1].lower()
            excerpt = text[descriptor.start() : match.end()]
            if value in {"teenager", "child", "young adult"}:
                yield "age_range", value, "EXPLICIT", excerpt
            else:
                yield "gender", _gender(value.split()[-1]), "EXPLICIT", excerpt
                if value.startswith("elderly"):
                    yield "age_range", "elderly", "EXPLICIT", excerpt
        # Direct possessive descriptions avoid the need for any coreference.
        body = re.match(
            r"['’]s\s+((?:(?:black|brown|blond|blonde|red|white|grey|gray|blue|green|hazel|long|short|curly|straight|pale|dark|light)\s+){1,4}(?:hair|eyes|skin|beard))\b",
            sentence,
            re.IGNORECASE,
        )
        if body:
            yield (
                "appearance",
                body[1].lower(),
                "EXPLICIT",
                text[match.start() : match.end() + body.end()],
            )
        direct = re.match(
            r"\s+(?:closed|opened|lowered|raised|tilted|shook|clenched|folded|crossed|clucked|deepened|cleared)\s+(his|her|their)\s+(?:own\s+)?(?:eyes|voice|head|arms|legs|fists|throat|tongue)\b",
            sentence,
            re.IGNORECASE,
        )
        if direct is None:
            direct = re.match(
                r"\s+(?:rubbed|touched|scratched|covered)\s+(his|her|their)\s+own\s+(?:head|chin|cheek|face|hands)\b",
                sentence,
                re.IGNORECASE,
            )
        if direct:
            yield (
                "pronouns",
                pronouns[direct[1].lower()],
                "SUPPORTED",
                text[match.start() : match.end() + direct.end()],
            )
        speech_pronoun = re.match(
            r"\s+(?:said|asked|replied|answered|shouted)\s+from where\s+(he|she|they)\s+(?:stood|sat|waited)\b",
            sentence,
            re.IGNORECASE,
        )
        if speech_pronoun:
            yield (
                "pronouns",
                pronouns[speech_pronoun[1].lower()],
                "SUPPORTED",
                text[match.start() : match.end() + speech_pronoun.end()],
            )
        delivery = re.match(
            r"\s+(?:(?:raised|lowered|deepened|softened)\s+(?:his|her|their)\s+voice|(?:whispered|shouted|murmured|bellowed|stammered|stuttered|drawled)|(?:said|asked|replied|answered)\s+(?:softly|quietly|loudly|hoarsely|sharply|slowly))\b",
            sentence,
            re.IGNORECASE,
        )
        if delivery:
            # One temporary deepening or shout is evidence of scene delivery,
            # not a permanent low register, loud personality, or speech defect.
            yield (
                "scene_delivery",
                delivery[0].strip().lower(),
                "EXPLICIT",
                text[match.start() : match.end() + delivery.end()],
            )
        if re.match(
            r"\s*,?\s*(?:lowering|raising|deepening|softening)\s+(?:his|her|their)\s+voice\b",
            sentence,
            re.IGNORECASE,
        ):
            yield (
                "scene_delivery",
                sentence.strip(" ,").lower(),
                "EXPLICIT",
                text[match.start() : match.end() + len(sentence)],
            )
        # Restrict across-sentence bridging to a sentence beginning with the
        # character, a short simple predicate, and no competing referent.
        beginning = re.search(r"(?:^|[.!?\n])\s*$", prefix) is not None
        if not beginning or len(sentence) > 125:
            continue
        if re.search(
            r"\b(?:he|she|his|her|him|they|their|them|I|you|we|our|boy|girl|man|woman|person|people|someone|anyone|friend|mother|father|brother|sister|enemy|opponent|stranger|instructor|teacher|guard)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        if re.search(r"\b[A-Z][a-z]+\b", sentence):
            continue
        if not re.match(
            r"\s+(?:said|asked|replied|answered|smiled|laughed|nodded|sighed|waited|sat|stood|spun|bent|backed|stepped|walked|ran|sprang|was saying)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        following = re.match(
            r"[^.!?\n]*[.!?]\s+(He|She|They|His|Her|Their)\b([^.!?\n]{0,160})", tail
        )
        if not following:
            continue
        first = following[1].lower()
        yield (
            "pronouns",
            pronouns[first],
            "SUPPORTED",
            text[match.start() : match.end() + following.end()],
        )
        if first in {"his", "her", "their"}:
            appearance = re.match(
                r"\s+(hair|eyes|skin|beard)\s+(?:was|were|is|are)\s+([^,;]{1,120})",
                following[2],
                re.IGNORECASE,
            )
            if appearance and re.search(
                r"\b(?:black|brown|blond|blonde|red|white|grey|gray|blue|green|hazel|long|short|longer|curly|straight|pale|dark|light)\b",
                appearance[2],
                re.IGNORECASE,
            ):
                yield (
                    "appearance",
                    appearance[0].strip(),
                    "SUPPORTED",
                    text[match.start() : match.end() + following.end()],
                )


def _extract(tail: str) -> list[tuple[str, object]]:
    facts: list[tuple[str, object]] = []
    depiction = re.match(_DEPICTION, tail, re.IGNORECASE)
    if depiction:
        value = _number(depiction["age"])
        modifier_age = re.search(rf"({_NUMBER})[- ]year[- ]old\b", depiction[0], re.IGNORECASE)
        if modifier_age is None or _number(modifier_age[1]) == value:
            facts.append(("age", value))
            if depiction["kind"].lower() in {"girl", "boy", "woman", "man"}:
                facts.append(("gender", _gender(depiction["kind"])))
    child = re.match(_CHILD_DESCRIPTION, tail, re.IGNORECASE)
    if child and not re.search(r"\b(?:if|unless|would|might|could|not|never|imaginary|hypothetical)\b",
                               tail[child.end():], re.IGNORECASE):
        facts.append(("age_range", child["kind"].lower()))
    direct = re.match(
        rf"{_PREDICATE}(?:(?:{_MODIFIER})\s+){{0,3}}(?P<g>{_GENDER})\b",
        tail,
        re.IGNORECASE,
    )
    if direct:
        facts.append(("gender", _gender(direct["g"])))
    # The verb (or an age adjective before a noun) makes the number an age,
    # rather than a quantity mentioned in the character's vicinity.
    age = re.match(
        rf"\s*(?:(?:was|is|had been|turned|aged)\s+)(?:only |just |already )?({_NUMBER})(?:\s+years?\s+old)?(?=\s*(?:$|,|\band\b|\bwhen\b))",
        tail,
        re.IGNORECASE,
    )
    if age is None:
        age = re.match(
            rf"{_PREDICATE}({_NUMBER})[- ]year[- ]old\b", tail, re.IGNORECASE
        )
    if age:
        facts.append(("age", _number(age[1])))
    age_range = re.match(
        r"\s*(?:was|is|had been)\s+(?:a |an |in (?:his|her|their) )?(teenager|child|young adult|elderly|early twenties|late teens|middle-aged)\b",
        tail,
        re.IGNORECASE,
    )
    if age_range:
        facts.append(("age_range", age_range[1].lower()))
    apparent = re.match(
        rf"\s*(?:looked|appeared|seemed)\s+(?:to be\s+)?(?:about\s+)?({_NUMBER})\b",
        tail,
        re.IGNORECASE,
    )
    if apparent:
        facts.append(("apparent_age", _number(apparent[1])))
    voice = re.match(
        r"\s*['’]s\s+(?:speaking\s+)?voice\s+(?:was|is|sounded|had|remained)\s+([^,;]+)",
        tail,
        re.IGNORECASE,
    )
    if voice is None:
        voice = re.match(
            r"\s*(?:had|has)\s+(?:an?\s+)([^,;]*?\bvoice)\b", tail, re.IGNORECASE
        )
    if voice is None:
        voice = re.match(
            r"\s*(?:spoke|speaks|said|replied|answered)\s+in\s+(?:an?\s+)?([^,;]*?\bvoice)\b",
            tail,
            re.IGNORECASE,
        )
    if voice:
        facts.append(("voice_description", voice[1].strip()))
    accent = re.match(
        r"\s*(?:spoke|speaks|talked|talks)\s+with\s+(?:an?\s+)?([^,;]{1,70}?)\s+(?:accent|dialect)\b",
        tail,
        re.IGNORECASE,
    )
    if accent is None:
        accent = re.match(
            r"\s*['’]s\s+(?:accent|dialect)\s+(?:was|is)\s+([^,;]{1,70})",
            tail,
            re.IGNORECASE,
        )
    if accent:
        facts.append(("accent", accent[1].strip()))
    cadence = re.match(
        r"\s*(?:always |usually |often |now )?(?:spoke|speaks|talked|talks)\s+(slowly|quickly|deliberately|rapidly|hesitantly|softly|in clipped sentences|with measured pauses)\b",
        tail,
        re.IGNORECASE,
    )
    if cadence:
        facts.append(("cadence", cadence[1].lower()))
    speech = re.match(
        r"\s*(?:always |usually |often )?(stuttered|stutters|lisped|lisps|drawled|drawls|spoke formally|speaks formally|used formal language|used slang)\b",
        tail,
        re.IGNORECASE,
    )
    if speech:
        facts.append(("speech_patterns", speech[1].lower()))
    appearance = re.match(r"\s*(?:had|has|wore)\s+([^,;]{1,150})", tail, re.IGNORECASE)
    if (
        appearance
        and re.search(
            r"\b(?:hair|eyes|skin|scar|freckles|beard|glasses)\b",
            appearance[1],
            re.IGNORECASE,
        )
        and not re.search(
            r"\b(?:his|her|their|friend|mother|father)\b", appearance[1], re.IGNORECASE
        )
    ):
        facts.append(("appearance", appearance[1].strip()))
        if re.search(
            r"\b(?:scar|freckles|birthmark|tattoo)\b", appearance[1], re.IGNORECASE
        ):
            facts.append(("distinctive_features", appearance[1].strip()))
    height = re.match(
        rf"\s*(?:was|is|stood)\s+((?:{_NUMBER})\s+(?:feet|foot|centimetres|centimeters|metres|meters)(?:\s+(?:{_NUMBER})\s+inches)?)(?:\s+tall)?\b",
        tail,
        re.IGNORECASE,
    )
    if height:
        facts.append(("height", height[1].lower()))
    build = re.match(
        r"\s*(?:was|is)\s+(?:very |unusually )?((?:tall|short|thin|slender|small|large|stocky|muscular|broad-shouldered|slight)(?:\s+and\s+(?:tall|short|thin|slender|small|large|stocky|muscular|broad-shouldered|slight))?)\b",
        tail,
        re.IGNORECASE,
    )
    if build:
        facts.append(("build", build[1].lower()))
    personality = re.match(
        r"\s*(?:was|is|remained)\s+(?:usually |always |a notoriously |notoriously )?(patient|impatient|cautious|confident|shy|reserved|kind|stern|gentle|impulsive|cheerful|sarcastic|calm|irritable)\b",
        tail,
        re.IGNORECASE,
    )
    if personality:
        facts.append(("temperament", personality[1].lower()))
    role = re.match(
        r"\s*(?:was|is|served as)\s+(?:a |an |the )?(teacher|student|captain|doctor|king|queen|prince|princess|apprentice|mentor|guard|servant|noble|merchant|soldier)\b",
        tail,
        re.IGNORECASE,
    )
    if role:
        facts.append(("role", role[1].lower()))
    vocabulary = re.match(
        r"\s*['’]s\s+(?:vocabulary|language|diction)\s+(?:was|is|remained)\s+([^,;]{1,120})",
        tail,
        re.IGNORECASE,
    )
    if vocabulary:
        facts.append(("vocabulary", vocabulary[1].strip()))
    context = re.match(
        r"\s*(?:was raised|grew up|was born)\s+in\s+([^,;]{1,100})", tail, re.IGNORECASE
    )
    if context:
        facts.append(("social_context", context[0].strip()))
    relationship = re.match(
        r"\s*(?:was|is)\s+([A-Z][\w'-]*(?: [A-Z][\w'-]*)?['’]s\s+(?:brother|sister|mother|father|friend|mentor|student|rival|wife|husband|partner))\b",
        tail,
    )
    if relationship:
        facts.append(("relationships", relationship[1]))
    change = re.match(
        r"\s*(?:now\s+(?:spoke|speaks)\b[^,;]{0,100}|['’]s\s+voice\s+(?:had become|became)\b[^,;]{0,100})",
        tail,
        re.IGNORECASE,
    )
    if change:
        facts.append(("portrayal_changes", change[0].strip()))
    pronouns = re.match(
        r"\s*(?:used|uses)\s+(he/him|she/her|they/them|he/they|she/they)\s+pronouns\b",
        tail,
        re.IGNORECASE,
    )
    if pronouns:
        facts.append(("pronouns", pronouns[1].lower()))
    return [(field, value) for field, value in facts if value is not None]


def _append(dossier: CharacterDossier, fact: CharacterFact) -> None:
    if not any(existing.id == fact.id for existing in dossier.facts):
        dossier.facts.append(fact)


def _fact(
    dossier: CharacterDossier,
    field: str,
    value: object,
    *,
    source: str,
    kind: str,
    excerpt: str,
    chapter: int | None = None,
    segment: Segment | None = None,
    location: str | None = None,
    level: str = "EXPLICIT",
) -> CharacterFact:
    anchor = segment.source_anchor if segment is not None else None
    scope = StoryScope(
        chapter_start=chapter,
        chapter_end=chapter
        if field in _AGE_FIELDS
        or field == "scene_delivery"
        or (field == "pronouns" and level == "SUPPORTED")
        else None,
    )
    data = {
        "field": field,
        "value": value,
        "source_kind": kind,
        "source": source,
        "source_revision_id": anchor.source_revision_id if anchor else None,
        "chapter_id": segment.chapter_id if segment else None,
        "scope": scope,
        "location": location
        or (f"segment:{segment.id};line:{segment.index}" if segment else None),
        "excerpt": excerpt[:300],
        "available_after_chapter": chapter,
        "level": level,
        "confidence": 0.9 if level == "SUPPORTED" else 0.98 if kind == "book" else 0.9,
        "extractor": EXTRACTOR_VERSION,
    }
    return CharacterFact(id="fact_" + _digest([dossier.role_id, data]), **data)


def _research_description(
    dossier: CharacterDossier, role: Role, project: Project
) -> None:
    text = role.description or ""
    if not text.strip():
        return
    chunk = _digest(["role-description", role.id, text, EXTRACTOR_VERSION])
    if chunk in dossier.processed_chunks:
        return
    extracted = []
    for field_name, value in re.findall(
        r"(?:^|[;\n])\s*([\w ]{2,25})\s*:\s*([^;\n]+)", text
    ):
        field = _DESCRIPTION_LABELS.get(field_name.strip().casefold())
        if field and value.strip().casefold() not in {"unknown", "unspecified", "?"}:
            result: object = value.strip()
            if field == "gender":
                result = _gender(value)
            elif field in {"age", "apparent_age"}:
                result = _number(value.strip()) or value.strip()
            extracted.append((field, result))
    if not extracted:
        # Role descriptions have a known subject, unlike an arbitrary paragraph.
        for tail, _ in _statements(f"{role.name} {text}", [role.name]):
            extracted.extend(_extract(tail))
    for field, value in extracted:
        _append(
            dossier,
            _fact(
                dossier,
                field,
                value,
                source=f"project:{project.id}/roles/{role.id}",
                kind="project",
                excerpt=text,
                location="role.description",
            ),
        )
    dossier.processed_chunks.append(chunk)


def _refresh(dossier: CharacterDossier) -> None:
    known = [f for f in dossier.facts if f.level not in {"UNKNOWN", "INFERRED"}]
    dossier.conflicts = [
        [left.id, right.id]
        for index, left in enumerate(known)
        for right in known[index + 1 :]
        if _contradict(left, right)
    ]
    dossier.review_reasons = [
        reason
        for reason in dossier.review_reasons
        if not reason.startswith("Conflicting ")
    ]
    by_id = {f.id: f for f in dossier.facts}
    for pair in dossier.conflicts:
        left, right = (by_id[x] for x in pair)
        reason = f"Conflicting {left.field} evidence: {left.source_kind} vs {right.source_kind} ({left.id}, {right.id})"
        dossier.review_reasons.append(reason)
    revision = _digest(
        [
            dossier.canonical_name,
            sorted(dossier.aliases),
            sorted(f.id for f in dossier.facts),
        ]
    )
    if dossier.revision != revision:
        dossier.updated_at = now()
        dossier.revision = revision


def _retire_provider_facts(project, dossier, provider, *, active_ids=None) -> None:
    """Retain prior provider evidence outside effective casting after invalidation.

    Only the new provider's explicitly owned extractor namespace is managed;
    book, project and other providers' evidence is untouched.
    """
    prefix = getattr(provider, "managed_extractor_prefix", None)
    if prefix is None:
        return
    retired = [fact for fact in dossier.facts
               if fact.source_kind in {"reference", "official"}
               and (fact.extractor == prefix or fact.extractor.startswith(prefix + ":"))
               and (fact.extractor != provider.version
                    or (active_ids is not None and fact.id not in active_ids))]
    if not retired:
        return
    key = "external-evidence-history:" + dossier.role_id
    history = project.character_cast.external_cache.setdefault(key, {
        "status": "historical_evidence", "retired_facts": []})
    existing = {raw["id"] for raw in history["retired_facts"]}
    history["retired_facts"].extend(fact.model_dump(mode="json") for fact in retired if fact.id not in existing)
    retired_ids = {fact.id for fact in retired}
    dossier.facts = [fact for fact in dossier.facts if fact.id not in retired_ids]


def _external(
    project: Project,
    dossier: CharacterDossier,
    provider: ExternalResearchProvider,
    policy: CharacterCastingPolicy,
    canonical_name: str,
) -> None:
    state = project.character_cast
    _retire_provider_facts(project, dossier, provider)
    provider_id = getattr(provider, "provider_id", type(provider).__name__)
    version = getattr(provider, "version", "unknown")
    title = (project.metadata or {}).get("source_title") or project.name
    key = _digest(
        [
            provider_id,
            version,
            _key(canonical_name),
            title,
            policy.max_chapter,
            policy.allow_spoilers,
            policy.adaptation_requested,
        ]
    )
    # Concrete network providers expire successful and failed results. Existing
    # fixture/injected providers without TTL attributes keep their prior contract.
    cache_ttl = getattr(provider, "cache_ttl_seconds", None)
    failure_ttl = getattr(provider, "failure_ttl_seconds", cache_ttl)
    entry = state.external_cache.get(key)
    if cache_ttl is not None and entry is not None:
        expiry = entry.get("expires_at", 0)
        if not isinstance(expiry, (int, float)) or expiry <= now().timestamp():
            state.external_cache.pop(key, None)
    if key not in state.external_cache:
        try:
            results = provider.research(
                canonical_name,
                aliases=tuple(_active_aliases(dossier, policy)),
                source_title=str(title),
                max_chapter=policy.max_chapter,
            )
            accepted, rejected = [], 0
            for raw in results:
                fact = (
                    raw
                    if isinstance(raw, CharacterFact)
                    else CharacterFact.model_validate(raw)
                )
                if fact.source_kind in {"book", "project"} or not _visible(
                    fact, policy
                ):
                    rejected += 1
                    continue
                accepted.append(fact.model_dump(mode="json"))
            state.external_cache[key] = {
                "status": "ok",
                "provider": provider_id,
                "version": version,
                "facts": accepted,
                "rejected_count": rejected,
                "researched_at": now().isoformat(),
            }
        except Exception as error:  # noqa: BLE001 - optional provider boundary must degrade safely
            if getattr(error, "research_deferred", False):
                # Offline ingest may reuse a valid cache but must not poison the
                # explicit online refresh with a cached deliberate deferral.
                return
            # A cached failure cannot be mistaken for a researched fact, and
            # provider exception text (which may contain credentials) is not stored.
            state.external_cache[key] = {
                "status": "unavailable",
                "provider": provider_id,
                "version": version,
                "error_type": type(error).__name__,
                "facts": [],
                "researched_at": now().isoformat(),
            }
    entry = state.external_cache[key]
    if cache_ttl is not None and "expires_at" not in entry:
        ttl = cache_ttl if entry.get("status") == "ok" else failure_ttl
        entry["expires_at"] = now().timestamp() + float(ttl)
        report = getattr(provider, "last_report", None)
        if isinstance(report, dict):
            # Quarantined discovery reports are inert metadata, never facts.
            entry["report"] = json.loads(json.dumps(report))
            if entry.get("status") != "ok" and entry["report"].get("status") == "researching":
                entry["report"]["status"] = "unavailable"
    if entry.get("status") == "ok":
        _retire_provider_facts(project, dossier, provider,
                               active_ids={raw["id"] for raw in entry.get("facts", [])})
        unavailable = f"External research unavailable ({provider_id}); unsupported facts remain UNKNOWN"
        dossier.review_reasons = [reason for reason in dossier.review_reasons if reason != unavailable]
    for raw in entry.get("facts", []):
        fact = CharacterFact.model_validate(raw)
        if _visible(fact, policy):
            _append(dossier, fact)
    if entry.get("status") != "ok":
        reason = f"External research unavailable ({provider_id}); unsupported facts remain UNKNOWN"
        if reason not in dossier.review_reasons:
            dossier.review_reasons.append(reason)


def research_project(
    project: Project,
    segments: Sequence[Segment],
    *,
    external_provider: ExternalResearchProvider | None = None,
) -> None:
    """Accumulate evidence without changing any accepted casting or voice.

    Each source chunk is fingerprinted with its anchor and extractor version.
    Re-running unchanged material neither duplicates facts nor changes dossier
    revisions/timestamps. The caller owns the canonical persistence transaction.
    """
    state = project.character_cast
    if not state.policy.enabled:
        return
    chapter_indices = {chapter.id: chapter.index for chapter in project.chapters}
    highest = max(chapter_indices.values(), default=0)
    policy = state.policy.model_copy(
        update={
            "max_chapter": state.policy.max_chapter
            if state.policy.max_chapter is not None
            else highest
        }
    )
    source_uris = {revision.id: revision.uri for revision in project.source_revisions}
    narration_roles = {
        role.id
        for role in project.roles
        if getattr(role.role_type, "value", role.role_type) == "narrator"
    }
    for role in project.roles:
        dossier = state.dossiers.setdefault(
            role.id, CharacterDossier(role_id=role.id, canonical_name=role.name)
        )
        if not any(f.field == "canonical_name" for f in dossier.facts):
            identified = [
                segment
                for segment in segments
                if segment.role_id == role.id
                and (
                    policy.allow_spoilers
                    or policy.max_chapter is None
                    or chapter_indices.get(segment.chapter_id, 0) <= policy.max_chapter
                )
            ]
            attributions = [
                segment.attribution
                for segment in identified
                if segment.attribution is not None
            ]
            human = any(item.human_authoritative for item in attributions)
            strong = [
                item
                for item in attributions
                if item.source in {"tag", "inline", "user"}
                and (item.confidence is None or item.confidence >= 0.7)
            ]
            # A derived role label is project identity metadata. It is never
            # presented as an explicit statement by the book's narrator.
            evidence_level = (
                "EXPLICIT" if human else "SUPPORTED" if strong else "INFERRED"
            )
            confidence = max((item.confidence or 0.7 for item in strong), default=0.35)
            _append(
                dossier,
                CharacterFact(
                    id="name_" + _digest([role.id, role.name]),
                    field="canonical_name",
                    value=role.name,
                    level=evidence_level,
                    confidence=1 if human else confidence,
                    source_kind="project",
                    source=f"project:{project.id}/roles/{role.id}",
                    location="role.name;initial_attribution_count:"
                    + str(len(attributions))
                    + ";sources:"
                    + ",".join(sorted({item.source for item in attributions})),
                    extractor="project-identity-v1",
                ),
            )
        # Unknown is explicit state, not a guess hidden in a text prompt.
        for field in FACT_FIELDS:
            if not any(f.field == field for f in dossier.facts):
                _append(
                    dossier,
                    CharacterFact(
                        id="unknown_" + _digest([role.id, field]),
                        field=field,
                        source="no supported evidence",
                        confidence=0,
                    ),
                )
        _research_description(dossier, role, project)
        # Discover explicit aliases before extracting any chapter so a later
        # identity statement can safely resolve an earlier alias mention.
        for segment in segments:
            chapter = chapter_indices.get(segment.chapter_id, 0)
            if segment.lineage is not None and segment.lineage.retired_at is not None:
                continue
            if (
                not policy.allow_spoilers
                and policy.max_chapter is not None
                and chapter > policy.max_chapter
            ):
                continue
            for tail, excerpt in _statements(segment.text, [dossier.canonical_name]):
                alias = re.match(
                    r"\s*,?\s*(?:also known as|known as|called)\s+([A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2})(?=\s*(?:,|$))",
                    tail,
                )
                if alias and _key(alias[1]) != _key(dossier.canonical_name):
                    if alias[1] not in dossier.aliases:
                        dossier.aliases.append(alias[1])
                    revision_id = (
                        segment.source_anchor.source_revision_id
                        if segment.source_anchor
                        else None
                    )
                    source = source_uris.get(
                        revision_id or "", f"project:{project.id}/segments/{segment.id}"
                    )
                    _append(
                        dossier,
                        _fact(
                            dossier,
                            "aliases",
                            alias[1],
                            source=source,
                            kind="book",
                            excerpt=excerpt,
                            chapter=chapter,
                            segment=segment,
                        ),
                    )
        for segment in segments:
            if segment.lineage is not None and segment.lineage.retired_at is not None:
                continue
            chapter = chapter_indices.get(segment.chapter_id, 0)
            if (
                not policy.allow_spoilers
                and policy.max_chapter is not None
                and chapter > policy.max_chapter
            ):
                continue
            attribution = segment.attribution
            if (
                segment.role_id == role.id
                and attribution is not None
                and not attribution.human_authoritative
                and (
                    attribution.source in {"unknown", "turn"}
                    or (
                        attribution.confidence is not None
                        and attribution.confidence < 0.7
                    )
                )
            ):
                reason = "Speaker attribution is uncertain; inspect dialogue attribution before accepting casting"
                if reason not in dossier.review_reasons:
                    dossier.review_reasons.append(reason)
            anchor = (
                segment.source_anchor.model_dump(mode="json")
                if segment.source_anchor
                else None
            )
            aliases = _active_aliases(dossier, policy)
            chunk = _digest(
                [segment.id, segment.text, anchor, sorted(aliases), EXTRACTOR_VERSION]
            )
            if chunk in dossier.processed_chunks:
                continue
            revision_id = (
                segment.source_anchor.source_revision_id
                if segment.source_anchor
                else None
            )
            source = source_uris.get(
                revision_id or "", f"project:{project.id}/segments/{segment.id}"
            )
            for tail, excerpt in _statements(
                segment.text, [dossier.canonical_name, *aliases]
            ):
                # Only explicit identity statements create aliases. Co-occurrence
                # and surname similarity are not sufficient to merge people.
                alias = re.match(
                    r"\s*,?\s*(?:also known as|known as|called)\s+([A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2})(?=\s*(?:,|$))",
                    tail,
                )
                if alias and _key(alias[1]) != _key(dossier.canonical_name):
                    if alias[1] not in dossier.aliases:
                        dossier.aliases.append(alias[1])
                    _append(
                        dossier,
                        _fact(
                            dossier,
                            "aliases",
                            alias[1],
                            source=source,
                            kind="book",
                            excerpt=excerpt,
                            chapter=chapter,
                            segment=segment,
                        ),
                    )
                for field, value in _extract(tail):
                    _append(
                        dossier,
                        _fact(
                            dossier,
                            field,
                            value,
                            source=source,
                            kind="book",
                            excerpt=excerpt,
                            chapter=chapter,
                            segment=segment,
                        ),
                    )
            if segment.attribution is None or segment.role_id in narration_roles:
                for field, value, level, excerpt in _contextual_evidence(
                    segment.text, [dossier.canonical_name, *aliases]
                ):
                    # Repeated gestures add no new research after two anchored
                    # examples per chapter; keep bounded support, not a log of
                    # every blink or nod in a novel.
                    repeats = sum(
                        1
                        for f in dossier.facts
                        if f.field == field
                        and f.value == value
                        and f.chapter_id == segment.chapter_id
                        and f.level == level
                    )
                    if repeats >= 2:
                        continue
                    _append(
                        dossier,
                        _fact(
                            dossier,
                            field,
                            value,
                            source=source,
                            kind="book",
                            excerpt=excerpt,
                            chapter=chapter,
                            segment=segment,
                            level=level,
                        ),
                    )
            dossier.processed_chunks.append(chunk)
    # Resolve only unambiguous explicit aliases before any costly provider work.
    for dossier in state.dossiers.values():
        owners = {
            other.canonical_name
            for other in state.dossiers.values()
            if any(
                _key(alias) == _key(dossier.canonical_name)
                for alias in _active_aliases(other, policy)
            )
        }
        canonical_name = (
            next(iter(owners)) if len(owners) == 1 else dossier.canonical_name
        )
        if state.policy.external_research:
            _external(
                project,
                dossier,
                external_provider or UnavailableExternalResearchProvider(),
                policy,
                canonical_name,
            )
        _refresh(dossier)
