"""Web addresses in book text: found, and taken out of what is read aloud.

Books from web serials carry links ("Table of Contents URL: https://...", "Join the Discord:
https://..."). A voice model cannot read one, and trying stalls the recording: it runs out of
its generation budget and refuses the line. So an address is never spoken. A line that is
only a link is left out of the recording, and an address inside a sentence is removed from
the sentence.

The import spells numbers out before any of this sees the text, and that reaches into
addresses too: ``/fiction/104434/`` arrives as ``/fiction/one hundred four thousand four
hundred thirty-four/``, which breaks the address at every space. An address that ends in a
spelled-out number therefore runs on through the number words that follow it.
"""
from __future__ import annotations

import re

_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|million|billion"
)
_NUMBER = rf"(?:{_NUMBER_WORDS})(?:-(?:{_NUMBER_WORDS}))*"
_START = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# The last piece of an address (after a slash, dot, dash, or a space the spelling put
# there) is a spelled-out number.
_ENDS_IN_NUMBER = re.compile(rf"(?:^|[\s/=?&#_.-]){_NUMBER}$", re.IGNORECASE)
# More number words, and whatever is glued to the last of them ("thirty-four/the-elf-...").
_RUNS_ON = re.compile(rf"\s+(?:and\s+)?{_NUMBER}(?=[\s/?#&=]|$)\S*", re.IGNORECASE)
_TRAILING = ".,;:!?)]}'\"’”"
# What introduces a link: "URL:", "Link -", "... at", "see", "visit".
_LEAD_IN = re.compile(
    r"(?:[:\-–—]|\b(?:url|urls|link|links|website|site|page|here|at|on|via|see|visit|from))\s*$",
    re.IGNORECASE)
_LEAD_IN_WORD = re.compile(r"\s*\b(?:at|on|via|see|visit|from)\s*$", re.IGNORECASE)
_LETTER_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
# A line this short around an address is there for the address.
_FEW_WORDS = 3
_LABEL_WORDS = 8


def spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every web address in ``text``, in order."""
    found = []
    for match in _START.finditer(text or ""):
        start, end = match.span()
        if found and start < found[-1][1]:
            continue  # already inside an address that ran on through spelled-out numbers
        while _ENDS_IN_NUMBER.search(text[start:end]):
            more = _RUNS_ON.match(text, end)
            if not more:
                break
            end = more.end()
        while end > start and text[end - 1] in _TRAILING:
            end -= 1
        if end > start:
            found.append((start, end))
    return found


def has_web_address(text: str) -> bool:
    return bool(spans(text))


def _words(text: str) -> list[str]:
    return _LETTER_WORD.findall(text)


def is_link_line(text: str) -> bool:
    """Is this line only a link, with at most a short label for it?

    "https://...", "Table of Contents URL: https://...", "Join the Discord: https://...".
    A sentence that happens to contain an address is not a link line: it is read without it.
    """
    found = spans(text)
    if not found:
        return False
    rest = _remove(text, found)
    words = _words(rest)
    if len(words) <= _FEW_WORDS:
        return True
    last_start, last_end = found[-1]
    ends_with_address = not _words(text[last_end:])
    return ends_with_address and len(words) <= _LABEL_WORDS and bool(_LEAD_IN.search(text[:last_start]))


def _remove(text: str, found: list[tuple[int, int]]) -> str:
    pieces, cursor = [], 0
    for start, end in found:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return " ".join(piece.strip() for piece in pieces if piece.strip())


def without_web_addresses(text: str) -> str:
    """``text`` with its web addresses taken out and the sentence left readable.

    "You can find the maps at https://example.com/maps and more." reads
    "You can find the maps and more.": the word that pointed at the address goes with it.
    """
    found = spans(text)
    if not found:
        return text
    out = text
    for start, end in reversed(found):
        before, after = _LEAD_IN_WORD.sub("", out[:start]), out[end:]
        out = before.rstrip() + (" " if before.strip() and after[:1].isspace() else "") + after.lstrip()
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    out = re.sub(r"\s+([,.;:!?)\]])", r"\1", out)
    out = re.sub(r"([,;:])(?=[.!?])", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"^[\s,.;:\-–—]+", "", out)
    if text[:1].isupper() and out[:1].islower():
        out = out[:1].upper() + out[1:]  # "See https://... for the rest" reads "For the rest"
    return out
