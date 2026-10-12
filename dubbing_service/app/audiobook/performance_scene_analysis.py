"""Performance Director scene-aware analysis (node C).

``analyze_scenes`` reads a project's utterances, groups them into scenes and
produces *recommendations*: per-line emotion, intention, pacing, emphasis,
boundaries, conversational response and scene endings, each bound to the exact
prose through character offsets.  It is CPU-only and deterministic: no network,
no model, no GPU, no synthesis, no download.  It never edits a segment, never
changes a voice identity, never writes a project and never approves anything.

Two layers
----------
* The **interpretation** depends only on the exact prose, attribution, alias
  data and each role's *type and name*.  It is the cacheable part, keyed by a
  content hash.  A role's voice is deliberately outside the key: recasting keeps
  every interpretation usable (the same rule as ``performance_director.scene_proofs``).
* The **resolution** is recomputed on every call: the compiled delivery profile
  (dossier habits), the live voice revision, the runtime capability record and
  the existing pure resolver ``performance_director.resolve_directions``.

Basis labels
------------
``explicit_human`` (a caller-supplied correction), ``explicit_source`` (the
author's own speech tag, adverb or markup in the prose) and ``inferred`` (the
analyzer's lexicon or punctuation heuristics).  Inferred recommendations are
never relabelled as explicit, and nothing here is human-approved: every row
carries ``human_review == "pending"`` and ``approved == False``.

Source proof
------------
A segment's ``source_anchor.source_revision_id`` is a registry *id*, not a hash.
The SHA-256 a scene plan is bound to is resolved through ``project.source_revisions``
and validated there; a missing, duplicated or malformed registry fails closed
(no plan, a typed hold), never falls back to treating the id as a hash.

Scene carry
-----------
An author-written cue (``she whispered``) describes more than the one line it
sits beside.  It is carried, as a lower-confidence ``inferred`` cue that still
points at the author's own words, to that speaker's following turns in the same
exchange until the author or the line itself says otherwise; a hushed register
also reaches the other participant's next turns.  No cue is invented from
knowledge of the book.

Optional experimental code (``performance_director``, ``provider_capabilities``)
is imported lazily.  When it is absent the analysis still runs and the
``integrations`` block says exactly what was unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "performance-scene-analysis-v1"
CACHE_SCHEMA_VERSION = "performance-scene-analysis-cache-v1"
ANALYZER_VERSION = "pd-scene-analysis-2"
CACHE_DIRECTORY = "performance-scene-analysis"
DEFAULT_RUNTIME_PROFILE = "faster-base-032"
EXPERIMENTAL_RUNTIME_PROFILE = "faster-base-032-direction-x1"
MAX_SCENE_UTTERANCES = 40
MAX_EVIDENCE = 12
MAX_PRIORITIES = 3
MAX_CARRY_TURNS = 3
MAX_PARTNER_CARRY_TURNS = 2
MAX_EXCHANGE_GAP_WORDS = 40
MAX_EXCHANGE_GAP_LINES = 2

BASIS_HUMAN = "explicit_human"
BASIS_SOURCE = "explicit_source"
BASIS_INFERRED = "inferred"
BASIS_NONE = "none"

#: Controls this analysis can recommend, and the capability name that governs each.
REQUESTED_CONTROLS = {
    "native_instruction": "native_instruction",
    "emotion_label": "emotion_label",
    "intensity": "intensity",
    "emphasis": "emphasis",
    "pace_hint": "pace",
    "role_delivery": "role_delivery",
}

LIMITATIONS = (
    "Lexicons are English only; punctuation cues are language neutral, and non-Latin text receives "
    "punctuation cues but no lexicon, word-count or tempo inference.",
    "Every row is a recommendation from deterministic heuristics. None is a listening verdict, "
    "none is human approved, and no audio quality is claimed or measured.",
    "Emphasis, emotion label and intensity have no native control on any known runtime; they can "
    "only be expressed inside the one native instruction, where effect is unmeasured.",
    "Boundary hints become assembly timing at best; this module applies none of them.",
    "Narrator lines are preserved: they receive pacing advice for assembly but never a forwarded "
    "native instruction.",
    "Dossier habits are evidence, not an approved profile; approval stays a human decision.",
)

# --------------------------------------------------------------------- helpers


class SceneAnalysisError(ValueError):
    """A refusal with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _canonical(value: Any) -> bytes:
    # Same bytes as fingerprint.canonical_json_bytes.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _value(obj: Any) -> Any:
    return getattr(obj, "value", obj)


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so the result is provably JSON-compatible."""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=False))


_SHA_RE = re.compile(r"^[a-f0-9]{64}$")

# ------------------------------------------------------------ optional seams


@dataclass
class _Seams:
    director: Any = None
    capabilities: Any = None
    fingerprint: Any = None
    reasons: dict[str, str] = field(default_factory=dict)


def _load_seams() -> _Seams:
    seams = _Seams()
    try:
        from dubbing_service.app.audiobook import performance_director as director

        seams.director = director
    except Exception as exc:  # noqa: BLE001 - optional experimental code must never break analysis
        seams.reasons["performance_director"] = f"{type(exc).__name__}: {exc}"
    try:
        from dubbing_service.app.audiobook import provider_capabilities as capabilities

        if not (hasattr(capabilities, "performance_capabilities")
                and hasattr(capabilities, "experimental_direction_capabilities")):
            raise AttributeError("capability owners are not present")
        seams.capabilities = capabilities
    except Exception as exc:  # noqa: BLE001
        seams.reasons["provider_capabilities"] = f"{type(exc).__name__}: {exc}"
    try:
        from dubbing_service.app.audiobook import fingerprint

        seams.fingerprint = fingerprint
    except Exception as exc:  # noqa: BLE001
        seams.reasons["fingerprint"] = f"{type(exc).__name__}: {exc}"
    return seams


def _integrations(seams: _Seams) -> dict[str, Any]:
    out = {}
    for name, module in (("performance_director", seams.director),
                         ("provider_capabilities", seams.capabilities),
                         ("fingerprint", seams.fingerprint)):
        out[name] = ({"available": True} if module is not None
                     else {"available": False, "reason": seams.reasons.get(name, "not importable")})
    return out


# ---------------------------------------------------------------- lexicons


def _phrases(*items: str) -> re.Pattern[str]:
    ordered = sorted(items, key=len, reverse=True)
    body = "|".join(r"\s+".join(re.escape(part) for part in item.split(" ")) for item in ordered)
    return re.compile(rf"(?<![\w'’])(?:{body})(?![\w'’])", re.IGNORECASE)


def _stems(*items: str) -> re.Pattern[str]:
    body = "|".join(re.escape(item) for item in sorted(items, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})\w*", re.IGNORECASE)


_EMOTION_LEXICON: dict[str, re.Pattern[str]] = {
    "anger": _phrases("damn", "hate", "furious", "liar", "idiot", "fool", "enough", "shut up",
                      "get out", "how dare you", "how dare", "stop it", "you bastard", "sick of"),
    "fear": _phrases("afraid", "scared", "terrified", "please don't", "don't hurt", "help me",
                     "help", "they're coming", "can't breathe", "oh god", "run"),
    "sadness": _phrases("gone", "lost", "alone", "died", "dead", "miss you", "goodbye", "grief",
                        "funeral", "crying", "tears", "never see", "too late"),
    "joy": _phrases("wonderful", "delighted", "splendid", "marvellous", "marvelous", "glad",
                    "happy", "love it", "hooray", "haha", "ha ha", "thank you", "at last"),
    "surprise": _phrases("no way", "you what", "can't be", "impossible", "good heavens",
                         "i can't believe", "what!", "really?"),
    "tenderness": _phrases("my dear", "darling", "sweetheart", "my love", "love you", "dear one"),
}
_EMOTION_ORDER = tuple(_EMOTION_LEXICON)

_COMIC_CUES = _phrases("haha", "ha ha", "joke", "joked", "grinned", "giggled", "giggle", "snorted",
                       "chuckled", "rolled his eyes", "rolled her eyes", "rolled their eyes",
                       "deadpan", "ridiculous", "walked into a bar", "punchline", "snickered")
_DRAMATIC_CUES = _phrases("silence", "silent", "tears", "blood", "dead", "dying", "sobbed", "gasped",
                          "grief", "for a long moment", "nobody spoke", "no one spoke", "heart pounded",
                          "trembling", "trembled", "screamed", "scream", "funeral", "terrified")
_BEAT_CUES = _phrases("silence", "silent", "for a long moment", "nobody spoke", "no one spoke",
                      "a long pause", "a pause", "paused", "stared", "waited", "neither spoke",
                      "the room went quiet", "fell quiet", "fell silent")

_INTENTION_LEXICON: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("challenge", _phrases("how dare you", "you lied", "prove it", "who do you think you are",
                           "what do you mean", "explain yourself", "i don't believe you",
                           "you expect me to believe")),
    ("deflect", _phrases("never mind", "it's nothing", "it is nothing", "forget it", "that's not important",
                         "i don't know what you mean", "i couldn't say", "let's not talk about",
                         "change the subject", "anyway")),
    ("concede", _phrases("you're right", "you are right", "very well", "i suppose", "all right then",
                         "fine, then", "i admit", "i was wrong", "you win", "have it your way")),
    ("reassure", _phrases("it's okay", "it's all right", "it is all right", "don't worry",
                          "you're safe", "everything will be", "i'm here", "it will be fine",
                          "it's going to be fine", "there, there")),
    ("recognize", _phrases("it's you", "so it was you", "i remember", "i know you", "i knew it",
                           "i know that voice", "it was you")),
    ("explain", _phrases("let me explain", "what happened was", "the reason is", "that's why",
                         "that is why", "because", "you see")),
    ("conceal", _phrases("i know nothing", "nothing happened", "i wasn't there", "i didn't see anything",
                         "i don't know anything", "no one told me")),
)

_FILLERS = _phrases("um", "uh", "er", "erm", "hmm", "well")

# verb -> (phrase, emotion, intensity, family)
_TAG_VERBS: dict[str, tuple[str, str | None, str | None, str]] = {
    "whispered": ("hushed and close", None, "low", "hushed"),
    "whispers": ("hushed and close", None, "low", "hushed"),
    "murmured": ("hushed and close", None, "low", "hushed"),
    "breathed": ("hushed and close", None, "low", "hushed"),
    "muttered": ("low, under the breath", None, "low", "muttered"),
    "mumbled": ("low, under the breath", None, "low", "muttered"),
    "shouted": ("raised voice", None, "high", "raised"),
    "yelled": ("raised voice", None, "high", "raised"),
    "bellowed": ("raised voice", None, "high", "raised"),
    "roared": ("raised voice", None, "high", "raised"),
    "cried": ("raised and urgent", None, "high", "raised"),
    "snapped": ("sharp and clipped", "anger", "moderate", "sharp"),
    "barked": ("sharp and clipped", "anger", "moderate", "sharp"),
    "hissed": ("low and sharp", "anger", "moderate", "sharp"),
    "growled": ("low and sharp", "anger", "moderate", "sharp"),
    "sighed": ("weary", "sadness", "low", "weary"),
    "laughed": ("amused", "joy", "moderate", "amused"),
    "chuckled": ("amused", "joy", "moderate", "amused"),
    "giggled": ("amused", "joy", "moderate", "amused"),
    "stammered": ("hesitant and halting", "fear", "moderate", "halting"),
    "stuttered": ("hesitant and halting", "fear", "moderate", "halting"),
    "sobbed": ("choked and tearful", "sadness", "high", "tearful"),
    "wept": ("choked and tearful", "sadness", "high", "tearful"),
    "pleaded": ("pleading", "fear", "high", "pleading"),
    "begged": ("pleading", "fear", "high", "pleading"),
}
_GENERIC_TAG_VERBS = frozenset({"said", "asked", "replied", "answered", "added", "continued", "called",
                                "remarked", "observed", "told", "says", "asks", "replies", "went on",
                                "offered", "suggested", "announced", "declared"})
_TAG_ADVERBS: dict[str, tuple[str, str | None, str | None]] = {
    "softly": ("soft", None, "low"), "quietly": ("quiet", None, "low"),
    "gently": ("gentle", "tenderness", "low"), "tenderly": ("tender", "tenderness", "low"),
    "warmly": ("warm", "tenderness", "moderate"), "angrily": ("angry", "anger", "high"),
    "sharply": ("sharp", "anger", "moderate"), "bitterly": ("bitter", "anger", "moderate"),
    "fiercely": ("fierce", "anger", "high"), "coldly": ("cold and controlled", None, "low"),
    "coolly": ("cold and controlled", None, "low"), "flatly": ("flat and deadpan", None, "low"),
    "dryly": ("dry", None, "low"), "calmly": ("calm and level", None, "low"),
    "firmly": ("firm", None, "moderate"), "nervously": ("nervous", "fear", "moderate"),
    "anxiously": ("anxious", "fear", "moderate"), "hesitantly": ("hesitant", "fear", "moderate"),
    "wearily": ("weary", "sadness", "low"), "sadly": ("sad", "sadness", "moderate"),
    "bleakly": ("bleak", "sadness", "moderate"), "cheerfully": ("cheerful", "joy", "moderate"),
    "excitedly": ("excited", "joy", "high"), "sarcastically": ("sarcastic", None, "moderate"),
    "urgently": ("urgent", None, "high"), "casually": ("casual", None, "low"),
}
_TAG_INTENT_VERBS: dict[str, str] = {
    "reassured": "reassure", "comforted": "reassure", "conceded": "concede", "admitted": "concede",
    "challenged": "challenge", "demanded": "challenge", "deflected": "deflect", "evaded": "deflect",
    "dodged": "deflect", "lied": "conceal", "pretended": "conceal", "explained": "explain",
    "recognized": "recognize", "recognised": "recognize",
}

_EMOTION_PHRASE = {
    ("anger", "high"): "heated and forceful", ("anger", "moderate"): "controlled anger",
    ("anger", "low"): "tightly controlled",
    ("fear", "high"): "frightened and urgent", ("fear", "moderate"): "tense and uneasy",
    ("fear", "low"): "uneasy",
    ("sadness", "high"): "heavy and tearful", ("sadness", "moderate"): "subdued and heavy",
    ("sadness", "low"): "quiet and weary",
    ("joy", "high"): "bright and delighted", ("joy", "moderate"): "bright and warm",
    ("joy", "low"): "lightly warm",
    ("surprise", "high"): "startled", ("surprise", "moderate"): "taken aback",
    ("surprise", "low"): "mildly surprised",
    ("tenderness", "high"): "soft and tender", ("tenderness", "moderate"): "soft and tender",
    ("tenderness", "low"): "gentle",
}
_INTENTION_PHRASE = {
    "reassure": "gently reassuring", "concede": "reluctantly giving way", "challenge": "pointed and challenging",
    "deflect": "light and evasive", "conceal": "guarded", "recognize": "dawning recognition",
    "explain": "clear and explanatory",
}

_ALLOWED_CAPS = frozenset({"OK", "TV", "USA", "UK", "US", "AI", "DNA", "FBI", "CIA", "NASA", "HQ", "ID", "UN",
                           "EU", "PC", "CD", "DVD", "PM", "AM", "II", "III", "IV", "VI", "VII", "VIII", "IX",
                           "XI", "XII"})

_QUOTE_RE = re.compile(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」')
_WORD_RE = re.compile(r"\w+(?:['’-]\w+)*")
_CAPS_RE = re.compile(r"(?<![\w'’])[A-Z]{2,}(?![\w'’])")
_ITALIC_RES = (
    re.compile(r"(?<![\w*])\*([^\s*][^*\n]{0,60}?)\*(?![\w*])"),
    re.compile(r"(?<![\w_])_([^\s_][^_\n]{0,60}?)_(?![\w_])"),
    re.compile(r"<(?:i|em)>([^<\n]{1,60}?)</(?:i|em)>", re.IGNORECASE),
)
_SCENE_BREAK_RE = re.compile(r"^\s*(?:(?:[*#~=_\-]\s*){3,}|⁂|\*)\s*$")
_LEADING_HESITATION_RE = re.compile(r"^[\s\"“'‘(\[]*(?:…|\.{3}|—|–|--)")
_STAMMER_RE = re.compile(r"(?<![\w])([A-Za-z])-\1\w*", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"(\.{2,}|…|——|—|–|--|[?？]+[!！]*|[!！]+[?？]*|[.。])$")
#: Speech-tag families that describe a register the speaker keeps, as opposed to a one-off event
#: (a laugh or a sigh) that only carries when the author attached an adverb to it.
_SUSTAINED_FAMILIES = frozenset({"hushed", "muttered", "raised", "sharp", "halting", "tearful", "pleading"})
_CLOSERS = "\"'”’)]»*_」 \t\r\n"
_PRONOUN_RE = re.compile(r"\b(?:he|she|they)\b", re.IGNORECASE)
_SPEECH_VERB_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(sorted([re.escape(v) for v in (*_TAG_VERBS, *_GENERIC_TAG_VERBS,
                                                           *_TAG_INTENT_VERBS)], key=len, reverse=True))
    + r")(?!\w)", re.IGNORECASE)
_ADVERB_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(_TAG_ADVERBS, key=len, reverse=True)) + r")(?!\w)",
                        re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?…。！？]+[.!?…。！？]*")


# --------------------------------------------------------------- data model


@dataclass
class _Utt:
    id: str
    index: int
    chapter_id: str | None
    role_id: str | None
    role_name: str | None
    role_type: str
    kind: str  # narration | dialogue | scene_break | unattributed
    text: str
    text_sha: str
    latin: bool
    source_revision: str | None
    speaker_id: str | None
    attribution_source: str
    anchor: dict[str, Any] | None = None


def _evidence(utt_id: str, kind: str, text: str, start: int, end: int) -> dict[str, Any]:
    return {"utterance_id": utt_id, "kind": kind, "text": text[start:end], "start": start, "end": end}


def _find_all(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _is_latin(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    ascii_letters = sum(1 for ch in letters if ch.isascii())
    return ascii_letters / len(letters) >= 0.6


def _terminal(text: str) -> dict[str, Any] | None:
    end = len(text)
    while end and text[end - 1] in _CLOSERS:
        end -= 1
    core = text[:end]
    match = _TERMINAL_RE.search(core)
    if match is None:
        return None
    mark = match.group(1)
    if "?" in mark or "？" in mark:
        kind = "question"
    elif "!" in mark or "！" in mark:
        kind = "exclaim"
    elif mark in ("…",) or (mark.startswith(".") and len(mark) >= 2):
        kind = "trail_off"
    elif mark in ("—", "——", "–", "--"):
        kind = "cutoff"
    else:
        kind = "period"
    return {"kind": kind, "text": mark, "start": match.start(1), "end": match.end(1)}


_CONTOUR = {"question": "rising", "exclaim": "emphatic", "trail_off": "trailing", "cutoff": "abrupt",
            "period": "falling"}


def _outside_quotes(text: str, position: int) -> bool:
    return not any(m.start() <= position < m.end() for m in _QUOTE_RE.finditer(text))


# ----------------------------------------------------------- tag detection


def _tag_cues(utt: _Utt) -> list[dict[str, Any]]:
    """Speech tags written by the author in this utterance's own prose."""

    cues: list[dict[str, Any]] = []
    if not utt.latin:
        return cues
    has_quotes = bool(_QUOTE_RE.search(utt.text))
    for match in _SPEECH_VERB_RE.finditer(utt.text):
        if utt.kind != "narration" and not (has_quotes and _outside_quotes(utt.text, match.start())):
            # A dialogue line only carries a tag when the verb sits outside the quoted speech.
            continue
        verb = match.group(0).casefold()
        window_start = max(0, match.start() - 28)
        window_end = min(len(utt.text), match.end() + 28)
        clause_start = max([window_start] + [window_start + m.end() for m in
                                             re.finditer(r"[.!?…;]", utt.text[window_start:match.start()])])
        clause_end = min([window_end] + [match.end() + m.start() for m in
                                         re.finditer(r"[.!?…;]", utt.text[match.end():window_end])])
        adverbs = [(window_start + m.start(), window_start + m.end(), m.group(1).casefold())
                   for m in _ADVERB_RE.finditer(utt.text[window_start:window_end])
                   if clause_start <= window_start + m.start() and window_start + m.end() <= clause_end]
        cue: dict[str, Any] = {"verb": verb, "verb_start": match.start(), "verb_end": match.end(),
                               "source_utterance_id": utt.id, "phrase": None, "emotion": None,
                               "intensity": None, "intention": None, "family": None, "adverb": None,
                               "evidence": [_evidence(utt.id, "speech_tag", utt.text, match.start(), match.end())]}
        if verb in _TAG_VERBS:
            phrase, emotion, intensity, family = _TAG_VERBS[verb]
            cue.update(phrase=phrase, emotion=emotion, intensity=intensity, family=family)
        if verb in _TAG_INTENT_VERBS:
            cue["intention"] = _TAG_INTENT_VERBS[verb]
        if adverbs:
            start, end, adverb = adverbs[0]
            phrase, emotion, intensity = _TAG_ADVERBS[adverb]
            # The adverb is the author's most specific statement; it overrides the verb's default.
            cue.update(phrase=phrase, emotion=emotion or cue["emotion"], intensity=intensity,
                       adverb=adverb, family=cue["family"] or "adverb")
            cue["evidence"].append(_evidence(utt.id, "speech_tag_adverb", utt.text, start, end))
        if cue["phrase"] is None and cue["intention"] is None:
            continue  # "said"/"asked" alone say nothing about delivery
        cues.append(cue)
    return cues


def _name_tokens(name: str | None) -> list[str]:
    if not name:
        return []
    tokens = [name.casefold()]
    first = name.split()[0].casefold() if name.split() else ""
    if len(first) >= 3 and first not in tokens:
        tokens.append(first)
    return tokens


def _mentions(text: str, name: str | None) -> bool:
    for token in _name_tokens(name):
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text.casefold()):
            return True
    return False


# -------------------------------------------------------- interpretation


def _own_emotion(utt: _Utt, terminal: dict[str, Any] | None, caps_line: bool) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    scores: dict[str, int] = {}
    if utt.latin:
        for label in _EMOTION_ORDER:
            hits = _find_all(_EMOTION_LEXICON[label], utt.text)
            if hits:
                scores[label] = len(hits)
                for start, end in hits[:3]:
                    evidence.append(_evidence(utt.id, f"lexicon:{label}", utt.text, start, end))
    loud = bool((terminal and terminal["kind"] == "exclaim") or caps_line)
    if terminal and terminal["kind"] == "exclaim":
        evidence.append(_evidence(utt.id, "punctuation:exclaim", utt.text, terminal["start"], terminal["end"]))
    if not scores:
        return {"label": None, "intensity": "high" if loud else None, "basis": BASIS_NONE,
                "confidence": None, "evidence": evidence[:MAX_EVIDENCE], "loud": loud}
    top = max(scores.values())
    leaders = [label for label in _EMOTION_ORDER if scores.get(label) == top]
    if len(leaders) > 1:
        return {"label": None, "intensity": "high" if loud else None, "basis": BASIS_NONE,
                "confidence": None, "conflict": leaders, "evidence": evidence[:MAX_EVIDENCE], "loud": loud}
    cues = sum(scores.values()) + (1 if loud else 0)
    return {"label": leaders[0], "intensity": "high" if loud else "moderate", "basis": BASIS_INFERRED,
            "confidence": "moderate" if cues >= 2 else "low", "evidence": evidence[:MAX_EVIDENCE], "loud": loud}


def _own_intention(utt: _Utt) -> dict[str, Any]:
    if utt.kind != "dialogue" or not utt.latin:
        return {"value": "unknown", "basis": BASIS_NONE, "confidence": None, "evidence": []}
    for intention, pattern in _INTENTION_LEXICON:
        hits = _find_all(pattern, utt.text)
        if hits:
            start, end = hits[0]
            return {"value": intention, "basis": BASIS_INFERRED, "confidence": "low",
                    "evidence": [_evidence(utt.id, f"lexicon:{intention}", utt.text, start, end)]}
    return {"value": "unknown", "basis": BASIS_NONE, "confidence": None, "evidence": []}


def _emphasis(utt: _Utt, caps_line: bool) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for pattern in _ITALIC_RES:
        for match in pattern.finditer(utt.text):
            found.append({"role": "marked_emphasis", "basis": BASIS_SOURCE, "markup": True,
                          **_evidence(utt.id, "markup", utt.text, match.start(1), match.end(1))})
    if not caps_line:
        for match in _CAPS_RE.finditer(utt.text):
            if match.group(0) in _ALLOWED_CAPS:
                continue
            found.append({"role": "capitalized_word", "basis": BASIS_SOURCE, "markup": False,
                          **_evidence(utt.id, "all_caps", utt.text, match.start(), match.end())})
    found.sort(key=lambda item: (item["start"], item["end"]))
    return found[:MAX_EVIDENCE]


def _is_caps_line(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 2]
    if len(words) < 2:
        return False
    caps = sum(1 for w in words if w.isupper() and w not in _ALLOWED_CAPS)
    return caps / len(words) > 0.6


def _word_count(utt: _Utt) -> int | None:
    return len(_WORD_RE.findall(utt.text)) if utt.latin else None


def _sentence_stats(text: str) -> tuple[int, float]:
    sentences = [s for s in _SENTENCE_RE.findall(text) if s.strip()]
    words = len(_WORD_RE.findall(text))
    return len(sentences), (words / len(sentences) if sentences else 0.0)


def _final_word_span(text: str) -> tuple[int, int] | None:
    end = len(text)
    while end and not text[end - 1].isalnum():
        end -= 1
    start = end
    while start and (text[start - 1].isalnum() or text[start - 1] in "'’-"):
        start -= 1
    return (start, end) if start < end else None


def _exchange_ids(scene_utts: list[_Utt], own: list[dict[str, Any]]) -> list[int | None]:
    """Group dialogue into exchanges: one conversation, broken by a scene break or a long narrative gap."""

    ids: list[int | None] = [None] * len(scene_utts)
    current = -1
    gap_lines = 0
    gap_words = 0
    have_dialogue = False
    for i, utt in enumerate(scene_utts):
        if utt.kind == "scene_break":
            have_dialogue, gap_lines, gap_words = False, 0, 0
        elif utt.kind in ("dialogue", "unattributed"):
            if not have_dialogue or gap_lines > MAX_EXCHANGE_GAP_LINES or gap_words > MAX_EXCHANGE_GAP_WORDS:
                current += 1
            have_dialogue, gap_lines, gap_words = True, 0, 0
            ids[i] = current
        elif have_dialogue:
            gap_lines += 1
            gap_words += own[i]["words"] or len(utt.text.split())
    return ids


def _carry_cues(scene_utts: list[_Utt], own: list[dict[str, Any]],
                exchange: list[int | None]) -> dict[int, list[dict[str, Any]]]:
    """Carry an author-written cue through the rest of its exchange.

    A cue stays attached to its speaker's following turns (``speaker_continuation``)
    until the author or the line itself says otherwise, and a hushed register also
    reaches the other participant's next turns (``exchange_register``).  Carried
    cues are ``inferred`` and low confidence, and every one points at the author's
    own words through its evidence.
    """

    carried: dict[int, list[dict[str, Any]]] = {}
    for i, utt in enumerate(scene_utts):
        if utt.kind != "dialogue" or exchange[i] is None:
            continue
        for cue in own[i]["bound_tags"]:
            if cue["binding"] == "adjacent_pronoun" or cue["ambiguous"] or not (cue["phrase"] or cue["emotion"]):
                continue  # a pronoun guess or a tag that could belong to either line is too weak to carry
            if not (cue["adverb"] or cue["family"] in _SUSTAINED_FAMILIES):
                continue  # a laugh or a sigh is an event, not a register
            own_turns = partner_turns = 0
            for j in range(i + 1, len(scene_utts)):
                other = scene_utts[j]
                if other.kind == "scene_break" or (other.kind in ("dialogue", "unattributed")
                                                   and exchange[j] != exchange[i]):
                    break
                if other.kind != "dialogue":
                    continue
                later = own[j]
                loud = bool(later["caps_line"] or (later["terminal"] and later["terminal"]["kind"] == "exclaim"))
                already_cued = any(t["phrase"] or t["emotion"] for t in later["bound_tags"])
                item = {"phrase": cue["phrase"], "emotion": cue["emotion"], "intensity": cue["intensity"],
                        "family": cue["family"], "from_utterance_id": utt.id, "binding": cue["binding"],
                        "basis": BASIS_INFERRED, "confidence": "low", "evidence": list(cue["evidence"])}
                if other.role_id == utt.role_id:
                    own_turns += 1
                    own_label = later["emotion"].get("label")
                    if (own_turns > MAX_CARRY_TURNS or already_cued
                            or (own_label and cue["emotion"] and own_label != cue["emotion"])
                            or (loud and cue["family"] in ("hushed", "muttered"))):
                        break  # superseded by the author, by the line's own evidence, or by a clash
                    carried.setdefault(j, []).append({**item, "scope": "speaker_continuation",
                                                      "turn_distance": own_turns})
                else:
                    partner_turns += 1
                    if (cue["family"] == "hushed" and partner_turns <= MAX_PARTNER_CARRY_TURNS
                            and not already_cued and not loud and later["emotion"].get("label") is None
                            and later["intention"]["value"] == "unknown"):
                        carried.setdefault(j, []).append({**item, "scope": "exchange_register",
                                                          "emotion": None, "intensity": "low",
                                                          "turn_distance": partner_turns})
    return carried


def _interpret_scene(scene_utts: list[_Utt]) -> dict[str, Any]:
    """The cacheable, voice-independent interpretation of one scene."""

    n = len(scene_utts)
    own: list[dict[str, Any]] = []
    for utt in scene_utts:
        terminal = _terminal(utt.text) if utt.kind != "scene_break" else None
        caps_line = _is_caps_line(utt.text)
        own.append({
            "terminal": terminal,
            "caps_line": caps_line,
            "words": _word_count(utt),
            "emotion": (_own_emotion(utt, terminal, caps_line)
                        if utt.kind in ("dialogue", "unattributed") else
                        {"label": None, "intensity": None, "basis": BASIS_NONE, "confidence": None,
                         "evidence": [], "loud": False}),
            "intention": _own_intention(utt) if utt.kind in ("dialogue", "unattributed") else
            {"value": "unknown", "basis": BASIS_NONE, "confidence": None, "evidence": []},
            "emphasis": _emphasis(utt, caps_line) if utt.kind != "scene_break" else [],
            "tags": _tag_cues(utt) if utt.kind != "scene_break" else [],
            "bound_tags": [],
            "beat": bool(utt.latin and utt.kind == "narration" and _BEAT_CUES.search(utt.text)),
        })

    # -- bind speech tags to the dialogue they describe -------------------
    for i, utt in enumerate(scene_utts):
        for cue in own[i]["tags"]:
            if utt.kind == "dialogue":
                own[i]["bound_tags"].append({**cue, "binding": "inline", "ambiguous": False})
                continue
            if utt.kind != "narration":
                continue
            targets = [j for j in (i - 1, i + 1)
                       if 0 <= j < n and scene_utts[j].kind == "dialogue"]
            named = [j for j in targets if _mentions(utt.text, scene_utts[j].role_name)]
            if named:
                for j in named:
                    # a tag between two lines by the same named speaker could belong to either
                    own[j]["bound_tags"].append({**cue, "binding": "named_role", "ambiguous": len(named) > 1})
            elif len(targets) == 1 and _PRONOUN_RE.search(utt.text):
                own[targets[0]]["bound_tags"].append({**cue, "binding": "adjacent_pronoun", "ambiguous": False})

    # -- carry author cues through their exchange --------------------------
    exchange = _exchange_ids(scene_utts, own)
    carry = _carry_cues(scene_utts, own, exchange)
    exchange_members: dict[int, list[int]] = {}
    for i, number in enumerate(exchange):
        if number is not None:
            exchange_members.setdefault(number, []).append(i)

    # -- scene tone ---------------------------------------------------------
    comic: list[dict[str, Any]] = []
    dramatic: list[dict[str, Any]] = []
    for utt in scene_utts:
        if utt.kind == "scene_break" or not utt.latin:
            continue
        comic += [_evidence(utt.id, "lexicon:comic", utt.text, s, e) for s, e in _find_all(_COMIC_CUES, utt.text)]
        dramatic += [_evidence(utt.id, "lexicon:dramatic", utt.text, s, e)
                     for s, e in _find_all(_DRAMATIC_CUES, utt.text)]
    if comic and dramatic:
        tone = "comic" if len(comic) >= 2 * len(dramatic) else ("dramatic" if len(dramatic) >= 2 * len(comic)
                                                                 else "mixed")
    elif comic:
        tone = "comic"
    elif dramatic:
        tone = "dramatic"
    else:
        tone = "neutral"

    spoken = [i for i, utt in enumerate(scene_utts) if utt.kind != "scene_break"]
    last_spoken = spoken[-1] if spoken else None

    # -- rapid exchange runs (dialogue turns of <= 8 words by alternating roles)
    brisk: set[int] = set()
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 3:
            brisk.update(run)
        run.clear()

    for i, utt in enumerate(scene_utts):
        words = own[i]["words"]
        if (utt.kind == "dialogue" and words is not None and words <= 8
                and (not run or scene_utts[run[-1]].role_id != utt.role_id)):
            run.append(i)
        else:
            flush()
            if utt.kind == "dialogue" and words is not None and words <= 8:
                run.append(i)
    flush()

    # -- per-utterance recommendations -------------------------------------
    records: list[dict[str, Any]] = []
    for i, utt in enumerate(scene_utts):
        o = own[i]
        if utt.kind == "scene_break":
            records.append({"utterance_id": utt.id, "kind": "scene_break", "terminal": None, "word_count": None,
                            "emotion": o["emotion"], "intention": o["intention"], "emphasis": [],
                            "tempo": {"value": "neutral", "basis": BASIS_NONE, "evidence": []},
                            "boundary": {"before": None, "after": None}, "response": None,
                            "ending": None, "tag_cues": [], "carried_cues": [], "exchange": None,
                            "delivery_phrases": []})
            continue
        dialogue = utt.kind in ("dialogue", "unattributed")
        prev = scene_utts[i - 1] if i > 0 and scene_utts[i - 1].kind != "scene_break" else None
        nxt = scene_utts[i + 1] if i + 1 < n and scene_utts[i + 1].kind != "scene_break" else None
        prev_o = own[i - 1] if prev is not None else None
        terminal = o["terminal"]
        tags = o["bound_tags"]
        carried = carry.get(i, []) if utt.kind == "dialogue" else []

        # emotion: an author-written tag outranks the analyzer's lexicon
        emotion = dict(o["emotion"])
        intention = dict(o["intention"])
        tag_phrases: list[dict[str, Any]] = []
        if dialogue:
            for cue in tags:
                if cue["emotion"] and emotion["basis"] != BASIS_SOURCE:
                    emotion = {"label": cue["emotion"], "intensity": cue["intensity"] or "moderate",
                               "basis": BASIS_SOURCE,
                               "confidence": ("low" if cue["binding"] == "adjacent_pronoun" or cue["ambiguous"]
                                              else "moderate"),
                               "binding": cue["binding"], "evidence": list(cue["evidence"]), "loud": emotion["loud"]}
                if cue["phrase"]:
                    tag_phrases.append({"phrase": cue["phrase"], "source": "speech_tag", "basis": BASIS_SOURCE,
                                        "binding": cue["binding"], "evidence": list(cue["evidence"])})
                    if emotion["basis"] != BASIS_SOURCE and cue["intensity"]:
                        emotion = {**emotion, "intensity": cue["intensity"]}
                if cue["intention"] and intention["basis"] != BASIS_SOURCE:
                    intention = {"value": cue["intention"], "basis": BASIS_SOURCE,
                                 "confidence": ("low" if cue["binding"] == "adjacent_pronoun" or cue["ambiguous"]
                                                else "moderate"),
                                 "binding": cue["binding"], "evidence": list(cue["evidence"])}

        # a carried cue fills an otherwise silent emotion; it never replaces the line's own evidence
        carried_emotion = False
        if carried and not tags:
            for item in carried:
                if (item["scope"] == "speaker_continuation" and item["emotion"]
                        and emotion["label"] is None and emotion["basis"] == BASIS_NONE):
                    emotion = {"label": item["emotion"], "intensity": item["intensity"] or "moderate",
                               "basis": BASIS_INFERRED, "confidence": "low",
                               "carried_from": item["from_utterance_id"],
                               "evidence": list(item["evidence"]), "loud": emotion["loud"]}
                    carried_emotion = True
                    break

        # response relation to the previous spoken line
        response = None
        if utt.kind == "dialogue":
            prev_dialogue_index = next((j for j in range(i - 1, -1, -1) if scene_utts[j].kind == "dialogue"), None)
            relation, responds_to, timing, ev = "opens_exchange", None, "natural", []
            if prev_dialogue_index is not None:
                pd = scene_utts[prev_dialogue_index]
                pt = own[prev_dialogue_index]["terminal"]
                between = i - prev_dialogue_index - 1
                responds_to = pd.id
                if pd.role_id == utt.role_id:
                    relation = "continuation"
                elif pt and pt["kind"] == "question":
                    relation = "answer"
                elif pt and pt["kind"] == "cutoff":
                    relation = "interruption"
                elif pt and pt["kind"] == "exclaim":
                    relation = "reaction"
                else:
                    relation = "reply"
                if pt:
                    ev.append(_evidence(pd.id, f"punctuation:{pt['kind']}", pd.text, pt["start"], pt["end"]))
                words = o["words"]
                hesitant_lead = bool(_LEADING_HESITATION_RE.search(utt.text) or (
                    utt.latin and _FILLERS.match(utt.text.lstrip("\"“'‘ ")) is not None))
                if relation == "interruption":
                    timing = "immediate"
                elif hesitant_lead:
                    timing = "after_beat"
                elif relation == "answer" and words is not None and words <= 5 and between == 0:
                    timing = "immediate"
                if between and any(own[j]["beat"] for j in range(prev_dialogue_index + 1, i)):
                    timing = "after_beat"
            response = {"relation": relation, "responds_to": responds_to, "timing": timing,
                        "basis": BASIS_INFERRED if responds_to else BASIS_NONE, "evidence": ev}

        # boundaries
        before = after = None
        if dialogue:
            lead = _LEADING_HESITATION_RE.search(utt.text)
            stammer = _STAMMER_RE.search(utt.text) if utt.latin else None
            filler = _FILLERS.match(utt.text.lstrip("\"“'‘ ")) if utt.latin else None
            if lead or stammer or filler:
                if lead:
                    ev = _evidence(utt.id, "punctuation:leading_hesitation", utt.text, lead.start(), lead.end())
                elif stammer:
                    ev = _evidence(utt.id, "stammer", utt.text, stammer.start(), stammer.end())
                else:
                    offset = len(utt.text) - len(utt.text.lstrip("\"“'‘ "))
                    ev = _evidence(utt.id, "filler", utt.text, offset + filler.start(), offset + filler.end())
                before = {"value": "hesitant", "basis": BASIS_INFERRED, "evidence": [ev]}
            elif response and response["relation"] == "interruption":
                before = {"value": "quick", "basis": BASIS_INFERRED, "evidence": list(response["evidence"])}
            elif (response and response["relation"] == "answer" and response["timing"] == "immediate"):
                before = {"value": "quick", "basis": BASIS_INFERRED, "evidence": list(response["evidence"])}
            if terminal and terminal["kind"] == "trail_off":
                after = {"value": "withheld", "basis": BASIS_INFERRED,
                         "evidence": [_evidence(utt.id, "punctuation:trail_off", utt.text,
                                                terminal["start"], terminal["end"])]}
            elif terminal and terminal["kind"] == "cutoff":
                after = {"value": "quick", "basis": BASIS_INFERRED,
                         "evidence": [_evidence(utt.id, "punctuation:cutoff", utt.text,
                                                terminal["start"], terminal["end"])]}
        else:
            if (utt.kind == "narration" and prev is not None and prev.kind == "dialogue" and utt.latin
                    and o["tags"] and (o["words"] or 99) <= 10 and nxt is not None and nxt.kind == "dialogue"):
                tag = o["tags"][0]
                before = {"value": "quick", "basis": BASIS_INFERRED, "evidence": list(tag["evidence"])}
            if o["beat"]:
                match = _BEAT_CUES.search(utt.text)
                after = {"value": "beat_after", "basis": BASIS_INFERRED,
                         "evidence": [_evidence(utt.id, "lexicon:beat", utt.text, match.start(), match.end())]}

        # ending and emphasis for the closing line of the scene
        ending = None
        emphasis = list(o["emphasis"])
        if terminal is not None:
            ending = {"terminal": terminal["kind"], "contour": _CONTOUR[terminal["kind"]], "style": "unmarked",
                      "basis": BASIS_INFERRED,
                      "evidence": [_evidence(utt.id, f"punctuation:{terminal['kind']}", utt.text,
                                             terminal["start"], terminal["end"])]}
        if i == last_spoken and utt.kind != "narration":
            style = "soft_close"
            if terminal and terminal["kind"] == "trail_off":
                style = "trail_off"
            elif terminal and terminal["kind"] == "cutoff":
                style = "cutoff"
            elif terminal and terminal["kind"] == "question":
                style = "open_question"
                if after is None:
                    after = {"value": "beat_after", "basis": BASIS_INFERRED, "evidence": list(ending["evidence"])}
            elif tone == "comic":
                style = "punchline_landing"
                span = _final_word_span(utt.text) if utt.latin else None
                if span is not None and not any(item["role"] == "punchline_final_word" for item in emphasis):
                    emphasis.append({"role": "punchline_final_word", "basis": BASIS_INFERRED, "markup": False,
                                     **_evidence(utt.id, "scene_ending", utt.text, span[0], span[1])})
                if after is None:
                    after = {"value": "beat_after", "basis": BASIS_INFERRED, "evidence": list(comic[:1])}
            elif tone == "dramatic":
                style = "dramatic_hold"
                if after is None:
                    after = {"value": "beat_after", "basis": BASIS_INFERRED, "evidence": list(dramatic[:1])}
            ending = (ending or {"terminal": None, "contour": "unmarked", "basis": BASIS_INFERRED,
                                 "evidence": []}) | {"style": style, "scene_final": True}
        elif ending is not None:
            ending["scene_final"] = False

        # tempo
        words = o["words"]
        tempo = {"value": "neutral", "basis": BASIS_NONE, "evidence": []}
        if utt.kind == "narration" and utt.latin:
            count, mean = _sentence_stats(utt.text)
            if count >= 1 and mean >= 25:
                tempo = {"value": "measured", "basis": BASIS_INFERRED,
                         "evidence": [{"utterance_id": utt.id, "kind": "sentence_length",
                                       "mean_words_per_sentence": round(mean, 1), "sentences": count}]}
            elif count >= 3 and mean <= 7:
                tempo = {"value": "brisk", "basis": BASIS_INFERRED,
                         "evidence": [{"utterance_id": utt.id, "kind": "sentence_length",
                                       "mean_words_per_sentence": round(mean, 1), "sentences": count}]}
        elif dialogue:
            if i in brisk:
                tempo = {"value": "brisk", "basis": BASIS_INFERRED,
                         "evidence": [{"utterance_id": utt.id, "kind": "rapid_exchange", "words": words}]}
            elif (before and before["value"] == "hesitant") or (after and after["value"] == "withheld"):
                tempo = {"value": "unhurried", "basis": BASIS_INFERRED,
                         "evidence": list((before or after)["evidence"])}

        # delivery phrases (dialogue only; narrator lines are never given an instruction)
        phrases: list[dict[str, Any]] = []
        if dialogue:
            def add(phrase: str, source: str, basis: str, evidence: list[dict[str, Any]], **extra: Any) -> None:
                if phrase and all(item["phrase"] != phrase for item in phrases):
                    phrases.append({"phrase": phrase, "source": source, "basis": basis,
                                    "evidence": evidence[:MAX_EVIDENCE], **extra})

            for item in tag_phrases:
                add(item["phrase"], "speech_tag", BASIS_SOURCE, item["evidence"], binding=item["binding"])
            if not tag_phrases:
                for item in carried:
                    add(item["phrase"], "carried_cue", BASIS_INFERRED, item["evidence"],
                        carried_from=item["from_utterance_id"], scope=item["scope"])
            if emotion["label"] and not tag_phrases and not carried_emotion:
                phrase = _EMOTION_PHRASE.get((emotion["label"], emotion["intensity"] or "moderate"))
                add(phrase or "", f"emotion:{emotion['label']}", emotion["basis"], emotion["evidence"])
            if intention["value"] != "unknown" and intention["value"] in _INTENTION_PHRASE:
                add(_INTENTION_PHRASE[intention["value"]], f"intention:{intention['value']}", intention["basis"],
                    intention["evidence"])
            if tempo["value"] == "brisk":
                add("quick and clipped", "tempo", BASIS_INFERRED, tempo["evidence"])
            elif tempo["value"] == "unhurried":
                add("unhurried, hesitant" if before and before["value"] == "hesitant" else "trailing off at the end",
                    "tempo", BASIS_INFERRED, tempo["evidence"])
            if ending and ending.get("style") == "punchline_landing":
                add("land the final word", "scene_ending", BASIS_INFERRED, ending["evidence"])
            elif ending and ending.get("style") == "cutoff":
                add("cut off abruptly", "scene_ending", BASIS_INFERRED, ending["evidence"])

        records.append({
            "utterance_id": utt.id, "kind": utt.kind, "terminal": terminal, "word_count": words,
            "emotion": {key: value for key, value in emotion.items() if key != "loud"},
            "intention": intention, "emphasis": emphasis,
            "tempo": tempo, "boundary": {"before": before, "after": after}, "response": response,
            "ending": ending,
            "tag_cues": [{key: cue[key] for key in ("verb", "adverb", "phrase", "emotion", "intensity",
                                                    "intention", "binding", "ambiguous", "source_utterance_id", "evidence")}
                         for cue in tags],
            "carried_cues": [{key: item[key] for key in ("phrase", "emotion", "intensity", "scope", "binding",
                                                         "from_utterance_id", "turn_distance", "basis",
                                                         "confidence", "evidence")} for item in carried],
            "exchange": (None if exchange[i] is None else {
                "id": f"x{exchange[i]}", "position": exchange_members[exchange[i]].index(i) + 1,
                "size": len(exchange_members[exchange[i]])}),
            "delivery_phrases": phrases,
        })

    # -- plan turns (what the existing Performance Director plan can carry) --
    turns = []
    for utt, record in zip(scene_utts, records):
        if utt.kind == "scene_break":
            continue
        intention = record["intention"]["value"]
        before = (record["boundary"]["before"] or {}).get("value")
        after = (record["boundary"]["after"] or {}).get("value")
        relation = (record["response"] or {}).get("relation")
        if intention == "unknown" and before is None and after is None and relation in (None, "opens_exchange"):
            continue
        turns.append({"utterance_id": utt.id, "intention": intention,
                      "transition": relation if relation and relation != "opens_exchange" else None,
                      "boundary_hint_before": before, "boundary_hint_after": after,
                      "confidence": "deterministic"})

    arc = []
    for utt, record in zip(scene_utts, records):
        label = record["emotion"]["label"]
        if label and (not arc or arc[-1]["label"] != label):
            arc.append({"utterance_id": utt.id, "label": label, "basis": record["emotion"]["basis"]})
    final = records[last_spoken] if last_spoken is not None else None
    summary = {
        "tone": tone,
        "tone_evidence": {"comic": comic[:MAX_EVIDENCE], "dramatic": dramatic[:MAX_EVIDENCE]},
        "emotion_arc": arc,
        "ending": (final["ending"] if final else None),
        "rapid_exchange_utterance_ids": [scene_utts[i].id for i in sorted(brisk)],
        "exchanges": [{"id": f"x{number}", "utterance_ids": [scene_utts[i].id for i in members],
                       "speakers": list(dict.fromkeys(scene_utts[i].role_id for i in members
                                                      if scene_utts[i].role_id))}
                      for number, members in sorted(exchange_members.items())],
        "dialogue_utterances": sum(1 for utt in scene_utts if utt.kind == "dialogue"),
        "narration_utterances": sum(1 for utt in scene_utts if utt.kind == "narration"),
        "basis": BASIS_INFERRED,
    }
    return {"utterances": records, "summary": summary, "turns": turns}


# ------------------------------------------------------------ project reads


def _normalize(project: Any, segments: Sequence[Any]) -> tuple[list[_Utt], list[dict[str, Any]], dict[str, Any]]:
    roles = {str(_get(role, "id")): role for role in (_get(project, "roles") or [])}
    holds: list[dict[str, Any]] = []
    out: list[_Utt] = []
    seen: set[str] = set()
    for position, segment in enumerate(segments):
        seg_id = str(_get(segment, "id") or "")
        text = _get(segment, "text")
        if not seg_id or not isinstance(text, str):
            holds.append({"code": "input_ineligible", "utterance_id": seg_id or None,
                          "reason": "a segment needs an id and a text string"})
            continue
        if seg_id in seen:
            holds.append({"code": "input_ineligible", "utterance_id": seg_id,
                          "reason": "duplicate segment id; the later occurrence was skipped"})
            continue
        seen.add(seg_id)
        role_id = _get(segment, "role_id")
        role = roles.get(str(role_id)) if role_id else None
        role_type = str(_value(_get(role, "role_type"))) if role is not None else "unknown"
        anchor = _get(segment, "source_anchor")
        attribution = _get(segment, "attribution")
        if _SCENE_BREAK_RE.match(text):
            kind = "scene_break"
        elif role is None:
            kind = "unattributed"
        elif role_type == "narrator":
            kind = "narration"
        else:
            kind = "dialogue"
        out.append(_Utt(
            id=seg_id, index=int(_get(segment, "index") or position), chapter_id=_get(segment, "chapter_id"),
            role_id=str(role_id) if role_id else None,
            role_name=str(_get(role, "name")) if role is not None and _get(role, "name") is not None else None,
            role_type=role_type, kind=kind, text=text, text_sha=_text_sha(text), latin=_is_latin(text),
            source_revision=_get(anchor, "source_revision_id"),
            speaker_id=_get(attribution, "speaker_id"),
            attribution_source=str(_get(attribution, "source") or "unknown"),
            anchor=(None if anchor is None else {
                "source_revision_id": _get(anchor, "source_revision_id"),
                "start_offset": _get(anchor, "start_offset"), "end_offset": _get(anchor, "end_offset"),
                "occurrence_index": _get(anchor, "occurrence_index")})))
        if kind == "unattributed":
            holds.append({"code": "attribution_unresolved", "utterance_id": seg_id,
                          "reason": "the segment has no known role; analysis is text-only and no instruction is proposed"
                          if not role_id else f"role {role_id} is not in the project"})
    return out, holds, roles


def _split_scenes(utts: list[_Utt], limit: int) -> list[list[_Utt]]:
    scenes: list[list[_Utt]] = []
    current: list[_Utt] = []
    for utt in utts:
        if current:
            only_breaks = all(item.kind == "scene_break" for item in current)
            if not only_breaks and (utt.chapter_id != current[-1].chapter_id or len(current) >= limit
                                    or current[-1].kind == "scene_break"):
                scenes.append(current)
                current = []
        current.append(utt)
    if current:
        scenes.append(current)
    return scenes


def _scene_id(scene: list[_Utt]) -> str:
    return "scene-" + _sha([scene[0].chapter_id or "", scene[0].id])[:16]


def _alias_projection(project: Any) -> dict[str, Any]:
    casting = _get(_get(project, "voice_lab"), "casting") or {}
    aliases = casting.get("aliases") if isinstance(casting, Mapping) else None
    aliases = aliases if isinstance(aliases, Mapping) else {}
    canonical = _get(_get(project, "character_cast"), "role_aliases") or {}
    return {"aliases": {str(k): str(v) for k, v in sorted(aliases.items())},
            "canonical_aliases": {str(k): str(v) for k, v in sorted(canonical.items())}}


@dataclass
class _Registry:
    """The project's source-revision registry, reduced to ``id -> sha256`` or a refusal."""

    shas: dict[str, str] = field(default_factory=dict)
    problem: str | None = None  # stable code; the registry is unusable when set
    detail: str = ""
    count: int = 0


_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,200}$")


def _source_registry(project: Any) -> _Registry:
    """Resolve revision ids to SHA-256 through ``project.source_revisions``, failing closed.

    Any defect makes the whole registry unusable: a registry that is absent, has a
    repeated id, or carries a malformed id or digest cannot prove which source a
    scene was read from.
    """

    revisions = _get(project, "source_revisions")
    if revisions is None or isinstance(revisions, (str, bytes, Mapping)) or not isinstance(revisions, Sequence):
        return _Registry(problem="source_registry_missing",
                         detail="the project has no source_revisions registry to resolve revision ids")
    if len(revisions) == 0:
        return _Registry(problem="source_registry_missing",
                         detail="the project's source_revisions registry is empty")
    shas: dict[str, str] = {}
    for position, revision in enumerate(revisions):
        revision_id, digest = _get(revision, "id"), _get(revision, "sha256")
        if not isinstance(revision_id, str) or not _ID_RE.fullmatch(revision_id):
            return _Registry(problem="source_registry_malformed", count=len(revisions),
                             detail=f"source_revisions[{position}] has no usable id")
        if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
            return _Registry(problem="source_registry_malformed", count=len(revisions),
                             detail=f"source revision {revision_id} has no lowercase 64-hex sha256")
        if revision_id in shas:
            return _Registry(problem="source_registry_duplicate", count=len(revisions),
                             detail=f"source revision id {revision_id} appears more than once")
        shas[revision_id] = digest
    return _Registry(shas=shas, count=len(revisions))


def resolve_source_revision_sha256(project: Any, revision_id: str) -> str:
    """The registered SHA-256 for a ``source_anchor.source_revision_id``, or a typed refusal.

    This is the one place an anchor id becomes a digest.  An id is never accepted
    because it merely looks like a hash.
    """

    registry = _source_registry(project)
    if registry.problem:
        raise SceneAnalysisError(registry.problem, registry.detail)
    if not isinstance(revision_id, str) or revision_id not in registry.shas:
        raise SceneAnalysisError("source_revision_unknown",
                                 f"source revision {revision_id!r} is not in the project registry")
    return registry.shas[revision_id]


def _source_proofs(scene: list[_Utt], alias_digest: str, registry: _Registry) -> dict[str, Any]:
    """Proofs a scene is bound to.  The SHA comes from the registry, never from the anchor id itself."""

    ids = sorted({utt.source_revision for utt in scene if utt.source_revision})
    unanchored = [utt.id for utt in scene if not utt.source_revision and utt.kind != "scene_break"]
    source: str | None = None
    problem: str | None = None
    detail = ""
    if registry.problem:
        problem, detail = registry.problem, registry.detail
    elif unanchored or not ids:
        problem = "source_anchor_missing"
        detail = f"utterance {(unanchored or [scene[0].id])[0]} has no source revision anchor"
    else:
        unknown = [item for item in ids if item not in registry.shas]
        resolved = sorted({registry.shas[item] for item in ids if item in registry.shas})
        if unknown:
            problem, detail = "source_revision_unknown", f"source revision {unknown[0]} is not in the project registry"
        elif len(resolved) != 1:
            problem, detail = "source_revision_mixed", "the scene spans more than one source revision"
        else:
            source = resolved[0]
    attribution = [{"id": utt.id, "speaker": utt.speaker_id, "source": utt.attribution_source} for utt in scene]
    return {"source_revision_ids": ids, "source_revision_sha256": source,
            "source_problem": problem, "source_problem_detail": detail,
            "utterances": [[utt.id, utt.text_sha, utt.role_id] for utt in scene],
            "attribution_digest": _sha(attribution), "alias_digest": alias_digest}


def _role_bindings(scene: list[_Utt]) -> list[list[str | None]]:
    """Role facts the interpretation depends on.  A voice is deliberately absent."""

    seen: dict[str, list[str | None]] = {}
    for utt in scene:
        if utt.role_id and utt.role_id not in seen:
            seen[utt.role_id] = [utt.role_id, utt.role_type, (utt.role_name or "").casefold()]
    return [seen[key] for key in sorted(seen)]


def _analysis_key(scene: list[_Utt], proofs: dict[str, Any]) -> str:
    return _sha({"analyzer_version": ANALYZER_VERSION, "proofs": proofs,
                 "role_bindings": _role_bindings(scene),
                 "kinds": [utt.kind for utt in scene]})


# ------------------------------------------------------------------- cache


def _cache_dir(cache_root: Path | str | None) -> Path | None:
    if cache_root is None:
        return None
    root = Path(cache_root)
    if not root.is_dir():
        raise SceneAnalysisError("input_ineligible", f"cache_root must be an existing directory: {root}")
    return root / CACHE_DIRECTORY


def _entry_hash(entry: Mapping[str, Any]) -> str:
    return _sha({key: value for key, value in entry.items() if key != "content_sha256"})


def _cache_read(directory: Path | None, key: str, scene: list[_Utt]) -> tuple[dict[str, Any] | None, str]:
    if directory is None:
        return None, "uncached"
    path = directory / f"{key}.json"
    if not path.exists():
        return None, "miss"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (entry.get("schema_version") != CACHE_SCHEMA_VERSION or entry.get("analysis_key") != key
                or entry.get("analyzer_version") != ANALYZER_VERSION
                or entry.get("content_sha256") != _entry_hash(entry)):
            return None, "invalid_entry_ignored"
        interpretation = entry["interpretation"]
        rows = interpretation["utterances"]
        if [row["utterance_id"] for row in rows] != [utt.id for utt in scene]:
            return None, "invalid_entry_ignored"
        if entry.get("text_sha256s") != [utt.text_sha for utt in scene]:
            return None, "invalid_entry_ignored"
        # Exact prose is stored in the entry; it must still equal the live prose.
        if entry.get("texts") != [utt.text for utt in scene]:
            return None, "invalid_entry_ignored"
        return interpretation, "hit"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, "invalid_entry_ignored"


def _cache_write(directory: Path, key: str, scene: list[_Utt], scene_id: str,
                 interpretation: dict[str, Any]) -> str:
    entry: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION, "analysis_key": key, "analyzer_version": ANALYZER_VERSION,
        "scene_id": scene_id, "texts": [utt.text for utt in scene],
        "text_sha256s": [utt.text_sha for utt in scene], "interpretation": interpretation,
    }
    entry["content_sha256"] = _entry_hash(entry)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    data = json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")
    try:
        with open(path, "xb") as handle:  # exclusive: an existing entry is never overwritten
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return "write_conflict_ignored"
    return "miss_written"


# ----------------------------------------------------------- capabilities


def _capability_for(seams: _Seams, profile: str, role: Any, memo: dict[Any, Any]) -> dict[str, Any] | None:
    if seams.capabilities is None:
        return None
    identity = _get(role, "voice_identity")
    dumped = identity.model_dump(mode="json") if hasattr(identity, "model_dump") else (
        dict(identity) if isinstance(identity, Mapping) else None)
    key = (profile, (dumped or {}).get("mechanism"), (dumped or {}).get("clone_mode") or (dumped or {}).get("mode"))
    if key not in memo:
        owner = (seams.capabilities.experimental_direction_capabilities
                 if profile == EXPERIMENTAL_RUNTIME_PROFILE else seams.capabilities.performance_capabilities)
        try:
            memo[key] = owner(profile, dumped)
        except Exception as exc:  # noqa: BLE001 - read-only probe; report, never raise
            memo[key] = {"error": f"{type(exc).__name__}: {exc}"}
    value = memo[key]
    return None if "error" in value else value


def _investigation(seams: _Seams, profile: str, memo: dict[Any, Any]) -> dict[str, Any]:
    """Facts about the installed Faster instruct seam, read through the capability owner."""

    if seams.capabilities is None:
        return {"available": False, "reason": seams.reasons.get("provider_capabilities", "unavailable")}
    out: dict[str, Any] = {"available": True, "profiles": {}}
    for name in dict.fromkeys((profile, DEFAULT_RUNTIME_PROFILE, EXPERIMENTAL_RUNTIME_PROFILE)):
        record = _capability_for(seams, name, {"voice_identity": {"mechanism": "qwen_reference"}}, memo)
        if record is None:
            out["profiles"][name] = {"available": False, "reason": "capability probe failed"}
            continue
        evidence = record.get("evidence", {})
        out["profiles"][name] = {
            "available": True, "capability_sha256": record.get("sha256"),
            "native_instruction": record["controls"]["native_instruction"],
            "seed": record["controls"].get("seed"),
            "evidence": {key: evidence.get(key) for key in (
                "faster_version", "library_accepts_instruct", "adapter_forwards_instruct",
                "direction_adapter_sha256", "adapter_sha256", "seed_policy")},
        }
    out["requires"] = ["identity mechanism qwen_reference", "conditioning mode icl",
                       "profile faster-base-032-direction-x1 (experimental, not registered by this module)"]
    out["quality_evidence"] = None
    out["note"] = ("Capability and wiring facts only. No audio was generated, so whether the model follows "
                   "an instruction is unmeasured; the library itself warns that Base-model instruct is "
                   "reliable only in full-reference (icl) mode, and that is untested here.")
    return out


# ------------------------------------------------------- resolution layer


def _habit_phrase(profile: Any) -> str | None:
    """One short, safe, supported habit usable beside scene phrases, or None."""

    if profile is None:
        return None
    for habit in profile.habits:
        if habit.level in ("EXPLICIT", "SUPPORTED") and len(habit.value) <= 60:
            return habit.value
    return None


def _compose(director: Any, phrases: list[str]) -> tuple[str | None, str | None]:
    if not phrases:
        return None, None
    text = "; ".join(phrases[:MAX_PRIORITIES])
    if not text.endswith("."):
        text += "."
    try:
        return director.validate_native_instruction(text), None
    except Exception as exc:  # noqa: BLE001 - PerformanceDirectorError carries the reason
        return None, str(exc)


def _merge_effective(record: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply a human correction over the cached recommendation.  Human > source > inferred."""

    effective: dict[str, Any] = {}
    boundary_after = (record["boundary"]["after"] or {}).get("value")
    boundary_before = (record["boundary"]["before"] or {}).get("value")
    emotion = record["emotion"]
    effective["emotion"] = {"label": emotion["label"], "intensity": emotion["intensity"], "basis": emotion["basis"]}
    effective["intention"] = {"value": record["intention"]["value"], "basis": record["intention"]["basis"]}
    effective["pace_hint"] = {"value": boundary_after or boundary_before,
                              "basis": (record["boundary"]["after"] or record["boundary"]["before"] or {}
                                        ).get("basis", BASIS_NONE)}
    effective["emphasis"] = [{"text": item["text"], "basis": item["basis"]} for item in record["emphasis"]]
    effective["delivery_phrases"] = [item["phrase"] for item in record["delivery_phrases"]]
    effective["delivery_basis"] = (BASIS_SOURCE if any(item["basis"] == BASIS_SOURCE
                                                       for item in record["delivery_phrases"])
                                   else (BASIS_INFERRED if record["delivery_phrases"] else BASIS_NONE))
    if override:
        if override.get("emotion_label") is not None:
            effective["emotion"] = {"label": override["emotion_label"], "intensity": override.get("intensity"),
                                    "basis": BASIS_HUMAN}
        elif override.get("intensity") is not None:
            effective["emotion"] = {**effective["emotion"], "intensity": override["intensity"], "basis": BASIS_HUMAN}
        if override.get("intention") is not None:
            effective["intention"] = {"value": override["intention"], "basis": BASIS_HUMAN}
        if override.get("pace_hint") is not None:
            effective["pace_hint"] = {"value": override["pace_hint"], "basis": BASIS_HUMAN}
        if override.get("emphasis") is not None:
            effective["emphasis"] = [{"text": override["emphasis"], "basis": BASIS_HUMAN}]
        if override.get("delivery"):
            effective["delivery_phrases"] = list(override["delivery"])
            effective["delivery_basis"] = BASIS_HUMAN
    return effective


def _validate_overrides(director: Any, raw: Mapping[str, Any] | None, known: set[str],
                        holds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Human corrections, validated by the existing override model when it is present."""

    accepted: dict[str, dict[str, Any]] = {}
    for uid, value in (raw or {}).items():
        if uid not in known:
            holds.append({"code": "input_ineligible", "utterance_id": uid,
                          "reason": "explicit direction names an utterance that is not in this analysis"})
            continue
        if not isinstance(value, Mapping):
            holds.append({"code": "schema_unknown", "utterance_id": uid,
                          "reason": "an explicit direction must be a mapping"})
            continue
        if director is not None:
            try:
                model = director.DirectionPolicyOverride.model_validate(dict(value))
            except Exception as exc:  # noqa: BLE001
                holds.append({"code": "schema_unknown", "utterance_id": uid,
                              "reason": f"explicit direction rejected by DirectionPolicyOverride: {exc}"[:240]})
                continue
            accepted[uid] = model.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        else:
            accepted[uid] = {k: v for k, v in dict(value).items() if v is not None}
    return accepted


def _profile_for(director: Any, role: Any, project: Any, max_chapter: int, memo: dict[Any, Any],
                 holds: list[dict[str, Any]], scene_id: str) -> Any:
    if director is None or role is None:
        return None
    key = (_get(role, "id"), max_chapter)
    if key in memo:
        return memo[key]
    cast = _get(project, "character_cast")
    dossier = (_get(cast, "dossiers") or {}).get(_get(role, "id"))
    casting = (_get(cast, "profiles") or {}).get(_get(role, "id"))
    narrator = str(_value(_get(role, "role_type"))) == "narrator"
    try:
        memo[key] = director.compile_profile(str(_get(role, "id")), narrator=narrator, dossier=dossier,
                                             casting_profile=casting, max_chapter=max_chapter)
    except Exception as exc:  # noqa: BLE001
        memo[key] = None
        holds.append({"code": getattr(exc, "code", "direction_conflict"), "scene_id": scene_id,
                      "role_id": _get(role, "id"),
                      "reason": f"delivery profile could not be compiled: {exc}"[:240]})
    return memo[key]


def _scene_intention(seams: _Seams, scene: list[_Utt], scene_id: str, proofs: dict[str, Any],
                     turns: list[dict[str, Any]]) -> tuple[Any, dict[str, Any] | None, str | None]:
    director = seams.director
    if director is None:
        return None, None, seams.reasons.get("performance_director", "performance_director unavailable")
    if proofs["source_revision_sha256"] is None:
        return None, None, (f"plan_stale: {proofs['source_problem'] or 'source_anchor_missing'}: "
                            f"{proofs['source_problem_detail'] or 'the scene has no single canonical source revision'}")
    try:
        utterances = [director.SceneUtterance(id=utt.id, text_sha256=utt.text_sha, role_id=utt.role_id)
                      for utt in scene]
        interpretation = director.SceneInterpretation(mode="deterministic", planner_version=ANALYZER_VERSION)
        key = director.scene_cache_key(
            source_revision_sha256=proofs["source_revision_sha256"], utterances=utterances,
            attribution_digest=proofs["attribution_digest"], alias_digest=proofs["alias_digest"],
            interpretation=interpretation)
        plan = director.SceneIntention(
            scene_id=scene_id, chapter_id=scene[0].chapter_id or "unassigned",
            source_revision_sha256=proofs["source_revision_sha256"], utterances=utterances,
            attribution_digest=proofs["attribution_digest"], alias_digest=proofs["alias_digest"],
            turns=[director.SceneTurn(**turn) for turn in turns], interpretation=interpretation,
            cache_key=key, author=director.RecordAuthor(origin="deterministic"), created_at=None)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"schema_unknown: {exc}"[:300]
    return plan, plan.model_dump(mode="json"), None


def analyze_scenes(project: Any, segments: Sequence[Any], *, cache_root: Path | str | None = None,
                   explicit_directions: Mapping[str, Mapping[str, Any]] | None = None,
                   runtime_profile: str | None = None,
                   max_scene_utterances: int = MAX_SCENE_UTTERANCES) -> dict[str, Any]:
    """Analyse ``segments`` of ``project`` into scenes and per-line recommendations.

    ``segments`` must be in reading order; scenes break at a chapter change, a
    scene-break marker line or ``max_scene_utterances``.  ``cache_root`` is an
    existing directory the caller owns (no default): interpretations are written
    there exclusively and never overwritten.  Without it nothing is cached.

    ``explicit_directions`` maps utterance ids to human corrections (the fields
    of ``performance_director.DirectionPolicyOverride``); they outrank every
    source or inferred cue and are labelled ``explicit_human``.  Nothing is
    applied, stored or approved by this function.
    """

    if not 1 <= int(max_scene_utterances) <= 2000:
        raise SceneAnalysisError("input_ineligible", "max_scene_utterances must be between 1 and 2000")
    seams = _load_seams()
    director = seams.director
    directory = _cache_dir(cache_root)
    metadata = _get(project, "metadata") or {}
    profile_name = str(runtime_profile or (metadata.get("requested_runtime") if isinstance(metadata, Mapping)
                                           else None) or DEFAULT_RUNTIME_PROFILE)
    profiles = list(dict.fromkeys((profile_name, EXPERIMENTAL_RUNTIME_PROFILE)))

    utts, holds, roles = _normalize(project, segments)
    scenes = _split_scenes(utts, int(max_scene_utterances))
    alias_digest = _sha(_alias_projection(project))
    registry = _source_registry(project)
    overrides = _validate_overrides(director, explicit_directions, {utt.id for utt in utts}, holds)
    chapters = {str(_get(c, "id")): _get(c, "index") for c in (_get(project, "chapters") or [])}

    capability_memo: dict[Any, Any] = {}
    profile_memo: dict[Any, Any] = {}
    revision_memo: dict[str, str | None] = {}
    scene_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    cache_scenes: list[dict[str, Any]] = []
    cache_written = False

    for ordinal, scene in enumerate(scenes):
        scene_id = _scene_id(scene)
        proofs = _source_proofs(scene, alias_digest, registry)
        key = _analysis_key(scene, proofs)
        interpretation, status = _cache_read(directory, key, scene)
        if interpretation is None:
            interpretation = _interpret_scene(scene)
            if directory is not None:
                written = _cache_write(directory, key, scene, scene_id, interpretation)
                # An invalid entry is never overwritten, so its write reports a conflict; keep both facts.
                status = written if status == "miss" else f"{status}:{written}"
                cache_written = cache_written or written == "miss_written"
        cache_scenes.append({"scene_id": scene_id, "analysis_key": key, "status": status})

        scene_holds: list[dict[str, Any]] = []
        if proofs["source_problem"]:
            scene_holds.append({"code": proofs["source_problem"], "scene_id": scene_id,
                                "reason": proofs["source_problem_detail"]})
        plan, plan_record, plan_reason = _scene_intention(seams, scene, scene_id, proofs, interpretation["turns"])
        records = {row["utterance_id"]: row for row in interpretation["utterances"]}

        chapter_index = chapters.get(str(scene[0].chapter_id))
        max_chapter = int(chapter_index) if isinstance(chapter_index, int) else -1

        # real resolution per role, per profile (the project's own and the experimental one)
        real_policy_overrides = {uid: ov for uid, ov in overrides.items() if uid in records}
        resolution: dict[str, dict[str, Any]] = {}
        what_if: dict[str, dict[str, dict[str, Any]]] = {}
        candidates: dict[str, dict[str, Any]] = {}
        for role_id in dict.fromkeys(utt.role_id for utt in scene if utt.role_id and utt.role_id in roles):
            role = roles[role_id]
            narrator = str(_value(_get(role, "role_type"))) == "narrator"
            profile = _profile_for(director, role, project, max_chapter, profile_memo, scene_holds, scene_id)
            habit = None if narrator else _habit_phrase(profile)
            role_utts = [utt for utt in scene if utt.role_id == role_id]
            for utt in role_utts:
                record = records[utt.id]
                effective = _merge_effective(record, overrides.get(utt.id))
                phrases = list(effective["delivery_phrases"])
                habit_used = False
                if not narrator and effective["delivery_basis"] != BASIS_HUMAN and habit and \
                        len(phrases) < MAX_PRIORITIES and habit not in phrases:
                    phrases.append(habit)
                    habit_used = True
                if narrator:
                    text, reason = None, "narrator preservation: no instruction is proposed"
                elif director is None:
                    text, reason = None, "performance_director unavailable"
                else:
                    text, reason = _compose(director, phrases)
                    if text is None and habit_used:
                        # Retry without the habit so one unsafe or long habit cannot suppress the scene phrases.
                        phrases, habit_used = phrases[:-1], False
                        text, reason = _compose(director, phrases)
                scene_phrases = [item for item in effective["delivery_phrases"]][:MAX_PRIORITIES]
                candidates[utt.id] = {"text": text, "chars": len(text) if text else 0,
                                      "phrases": phrases[:MAX_PRIORITIES],
                                      "habit_included": bool(habit_used and text), "reason": reason,
                                      "basis": effective["delivery_basis"],
                                      # scene intention and character habit stay separate for downstream composers
                                      "scene_phrases": scene_phrases, "habit_phrase": habit}
            if plan is None or director is None:
                reason = plan_reason or "performance_director unavailable"
                for utt in role_utts:
                    resolution[utt.id] = {"available": False, "reason": reason}
                    what_if[utt.id] = {}
                continue
            for name in profiles:
                capability = _capability_for(seams, name, role, capability_memo)
                if capability is None:
                    for utt in role_utts:
                        what_if.setdefault(utt.id, {})[name] = {"available": False,
                                                                "reason": "capability record unavailable"}
                        if name == profile_name:
                            resolution[utt.id] = {"available": False, "reason": "capability record unavailable"}
                    continue
                for variant in ("real", "candidate"):
                    line_overrides: dict[str, Any] = {}
                    for utt in role_utts:
                        base = dict(real_policy_overrides.get(utt.id) or {})
                        if variant == "candidate" and candidates[utt.id]["text"] and not base.get("delivery"):
                            base["delivery"] = list(candidates[utt.id]["phrases"])
                        if base:
                            line_overrides[utt.id] = base
                    try:
                        policy = director.DirectionPolicy(line_overrides={
                            k: director.DirectionPolicyOverride.model_validate(v) for k, v in line_overrides.items()})
                        rows = {row.utterance_id: row for row in director.resolve_directions(
                            profile, plan, capability, policy=policy)}
                    except Exception as exc:  # noqa: BLE001 - resolver refusals are typed holds
                        rows = {}
                        failure = {"available": False, "reason": str(exc)[:240],
                                   "code": getattr(exc, "code", "direction_conflict")}
                        for utt in role_utts:
                            if variant == "real" and name == profile_name:
                                resolution[utt.id] = failure
                            if variant == "candidate":
                                what_if.setdefault(utt.id, {})[name] = failure
                        continue
                    for utt in role_utts:
                        row = rows.get(utt.id)
                        if row is None:
                            continue
                        dump = row.model_dump(mode="json")
                        instruction = next((d for d in dump["dispositions"] if d["control"] == "native_instruction"),
                                           None)
                        if variant == "real" and name == profile_name:
                            resolution[utt.id] = {
                                "available": True, "runtime_profile": name,
                                "capability_sha256": capability.get("sha256"),
                                "requested": dump["requested"], "resolved": dump["resolved"],
                                "dispositions": dump["dispositions"], "precedence_trace": dump["precedence_trace"],
                                "sources": dump["sources"]}
                        if variant == "candidate":
                            state = instruction["state"] if instruction else "no_instruction"
                            what_if.setdefault(utt.id, {})[name] = {
                                "available": True, "capability_sha256": capability.get("sha256"),
                                "native_instruction_state": state,
                                "would_forward": state == "forward_native",
                                "reason": instruction["reason"] if instruction else "no delivery priority to forward",
                                "applied": False}

        # build direction rows
        scene_direction_ids = []
        for utt in scene:
            record = records[utt.id]
            override = overrides.get(utt.id)
            effective = _merge_effective(record, override)
            role = roles.get(utt.role_id) if utt.role_id else None
            narrator = utt.kind == "narration"
            if role is not None and utt.role_id not in revision_memo:
                revision_memo[utt.role_id] = (seams.fingerprint.voice_revision_for_role(role)
                                              if seams.fingerprint is not None else None)
            row_holds = [h for h in holds if h.get("utterance_id") == utt.id]
            if utt.kind == "scene_break":
                status_label = "scene_break"
            elif utt.kind == "unattributed":
                status_label = "held"
            elif narrator:
                status_label = "narrator_preserved"
            else:
                status_label = "recommendation"
            candidate = candidates.get(utt.id)
            direction_rows.append({
                "utterance_id": utt.id, "scene_id": scene_id, "index": utt.index,
                "role_id": utt.role_id, "role_name": utt.role_name, "role_type": utt.role_type, "kind": utt.kind,
                "text_sha256": utt.text_sha, "word_count": record["word_count"], "status": status_label, "human_review": "pending", "approved": False,
                "voice_revision": revision_memo.get(utt.role_id) if utt.role_id else None,
                "emotion": record["emotion"], "intention": record["intention"], "emphasis": record["emphasis"],
                "tempo": record["tempo"], "boundary": record["boundary"], "response": record["response"],
                "ending": record["ending"], "tag_cues": record["tag_cues"],
                "carried_cues": record["carried_cues"], "exchange": record["exchange"],
                "source_anchor": utt.anchor,
                "delivery_phrases": record["delivery_phrases"],
                "human_override": override, "effective": effective,
                "native_instruction_candidate": (None if candidate is None else {
                    key: candidate[key] for key in ("text", "chars", "phrases", "habit_included", "reason", "basis",
                                                "scene_phrases", "habit_phrase")}),
                "what_if": what_if.get(utt.id, {}),
                "resolution": resolution.get(utt.id, {"available": False,
                                                       "reason": "no resolvable role for this line"}),
                "preservation": {"narrator": narrator, "source_text_unchanged": True, "voice_unchanged": True,
                                 "forwarding": "never" if narrator else "only through the experimental profile"},
                "holds": row_holds,
            })
            scene_direction_ids.append(utt.id)

        scene_rows.append({
            "scene_id": scene_id, "chapter_id": scene[0].chapter_id, "ordinal": ordinal, "analysis_key": key,
            "cache": {"status": status}, "source": proofs,
            "utterances": [{"utterance_id": utt.id, "index": utt.index, "kind": utt.kind, "role_id": utt.role_id,
                            "role_name": utt.role_name, "role_type": utt.role_type, "text": utt.text,
                            "text_sha256": utt.text_sha} for utt in scene],
            "summary": interpretation["summary"], "turns": interpretation["turns"],
            "scene_intention": plan_record,
            "scene_intention_reason": plan_reason,
            "direction_ids": scene_direction_ids,
            "holds": scene_holds,
        })

    # roles: habits as evidence, never as approval
    role_rows = []
    used_roles = list(dict.fromkeys(utt.role_id for utt in utts if utt.role_id and utt.role_id in roles))
    for role_id in used_roles:
        role = roles[role_id]
        habits = []
        for (rid, _chapter), profile in profile_memo.items():
            if rid == role_id and profile is not None:
                habits = [h.model_dump(mode="json") for h in profile.habits]
                break
        role_rows.append({"role_id": role_id, "role_name": _get(role, "name"),
                          "role_type": str(_value(_get(role, "role_type"))),
                          "narrator_preservation": str(_value(_get(role, "role_type"))) == "narrator",
                          "habits": habits, "profile_status": "compiled_not_approved" if director else "unavailable",
                          "voice_revision": revision_memo.get(role_id)})

    controls_supported: dict[str, Any] = {}
    for name in profiles:
        record = _capability_for(seams, name, {"voice_identity": {"mechanism": "qwen_reference"}}, capability_memo)
        if record is None:
            controls_supported[name] = {"available": False}
            continue
        controls_supported[name] = {
            "available": True, "capability_sha256": record["sha256"],
            "controls": {control: {k: record["controls"].get(cap, {}).get(k) for k in ("state", "stage", "reason")}
                         for control, cap in REQUESTED_CONTROLS.items()}}

    cache_identity = _sha({"analyzer_version": ANALYZER_VERSION,
                           "keys": [entry["analysis_key"] for entry in cache_scenes]})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "analyzer_version": ANALYZER_VERSION,
        "project_id": _get(project, "id"),
        "runtime": {"requested_profile": profile_name, "profiles_evaluated": profiles},
        "source": {"registry_status": registry.problem or "ok", "registry_detail": registry.detail,
                   "revisions": [{"id": key, "sha256": value} for key, value in sorted(registry.shas.items())]},
        "scenes": scene_rows, "directions": direction_rows, "roles": role_rows,
        "cache": {"enabled": directory is not None, "identity": cache_identity, "scenes": cache_scenes,
                  "hits": sum(1 for e in cache_scenes if e["status"] == "hit"),
                  "written": sum(1 for e in cache_scenes if e["status"] == "miss_written")},
        "controls": {"requested": list(REQUESTED_CONTROLS), "supported": controls_supported,
                     "applied_by_analysis": []},
        "investigation": _investigation(seams, profile_name, capability_memo),
        "integrations": _integrations(seams),
        "holds": holds, "human_review": "pending", "limitations": list(LIMITATIONS),
        "side_effects": {"network": False, "gpu": False, "model_inference": False, "synthesis": False,
                         "project_written": False, "voice_changed": False, "cache_written": cache_written},
    }
    result = _jsonable(result)
    result["analysis_sha256"] = _sha(result)
    return result
