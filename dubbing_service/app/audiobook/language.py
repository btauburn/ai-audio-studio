"""The one rule that decides which language an utterance is synthesized in.

Renderer, synthesis requirement, QA and Take metadata used to each re-derive
this value with subtly different rules, so a Take could not prove what it was
actually made of.  They all call :func:`resolve_synthesis_language` now, and the
resolved value is carried through one render transaction.

Precedence, highest first:

1. ``segment.qwen3_language`` -- a one-off override for this utterance;
2. ``role.qwen3_language`` -- the role's stable voice profile;
3. ``default`` -- an explicit caller-supplied fallback, otherwise ``None``.

``None`` is a real answer: it means nothing declared a language, and it is
persisted as such rather than replaced by a guess.  The engine's own configured
default (``tts_params.qwen3_default_language``) is deliberately *not* consulted
here.  It is engine configuration, not part of the utterance's synthesis
requirement, and adopting it would re-window QA for every project that declares
no language -- turning today's accepted English fallback into a Chinese duration
window (see ``qa.coverage_profile_for``).
"""

from __future__ import annotations

from typing import Optional

# The spellings the Qwen3 engines accept, mirroring
# ``routers.settings.QWEN3_ALLOWED_LANGUAGES``.  The engine is the consumer of
# this value, so the engine's own spelling is the canonical form; introducing a
# second id alongside it would only create somewhere for the two to disagree.
SUPPORTED_LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)

_CANONICAL_BY_TOKEN = {option.lower(): option for option in SUPPORTED_LANGUAGES}

# Common ways the same language is written by hand or by another tool.  This is
# a lookup table, not a BCP-47 library: it exists so that ``english``, ``EN``
# and ``English`` all resolve to the one spelling the engine accepts.
_LANGUAGE_ALIASES = {
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "eng": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "zh-tw": "Chinese",
    "cmn": "Chinese",
    "mandarin": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Return the canonical engine spelling of ``value``.

    Case and surrounding whitespace are folded away.  An unrecognized non-empty
    value is returned stripped but otherwise untouched: it is what the user
    declared, and silently discarding it would hide a typo instead of showing it.
    """

    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    lowered = token.lower()
    if lowered in _CANONICAL_BY_TOKEN:
        return _CANONICAL_BY_TOKEN[lowered]
    if lowered in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[lowered]
    return token


def resolve_synthesis_language(
    segment: object,
    role: object = None,
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Resolve the language for one utterance, exactly once.

    ``segment`` and ``role`` are read through ``getattr`` so that a partially
    built object (a chunk copy, a legacy projection) resolves like the canonical
    one instead of raising.
    """

    for candidate in (
        getattr(segment, "qwen3_language", None),
        getattr(role, "qwen3_language", None),
        default,
    ):
        resolved = normalize_language(candidate)
        if resolved is not None:
            return resolved
    return None
