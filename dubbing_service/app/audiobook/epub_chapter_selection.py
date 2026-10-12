"""Deterministic selection of explicitly numbered story chapters from an EPUB.

Additive layer over ``book_creator._parse_epub``: archive safety limits, spine
order, titles and the ``epub-xhtml-body-text-v1`` extraction all stay owned by
that parser. Chapters are chosen by the *number in their title*, never by
position, so covers, information pages and tables of contents can never be
mistaken for "the first chapters". Source bytes are only read, never rewritten
or repackaged, and nothing leaves this process.
"""
from __future__ import annotations

import hashlib
import re

from . import book_creator
from .book_creator import BookCreatorError

SELECTION_SCHEMA = "epub-chapter-selection-v1"
MAX_SELECTED_CHAPTERS = book_creator.MAX_CHAPTERS
MAX_CHAPTER_NUMBER = book_creator.MAX_CHAPTERS
# A whole title must be a decimal number ("001", "7") or "Chapter <decimal>",
# optionally followed by a subtitle after a colon, period or dash. Spelled-out
# numbers, ranges, roman numerals and embedded numbers ("Part 2 Chapter 1",
# "Chapter 1 and 2") are deliberately not story-chapter labels.
TITLE_RULE = "title == <decimal> | 'Chapter' <decimal> [(':'|'.'|'-'|en/em dash) subtitle]"
_NUMBER = r"[0-9]{1,4}"
_TITLE_RE = re.compile(
    rf"^(?:(?P<bare>{_NUMBER})|chapter\s+(?P<labelled>{_NUMBER})(?:\s*[:.–—-]\s*\S.*)?)$",
    re.IGNORECASE)


class EpubChapterSelectionError(BookCreatorError):
    """Raised when the requested chapters cannot be selected unambiguously."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_numbers(chapter_numbers) -> list[int]:
    if isinstance(chapter_numbers, (str, bytes)) or not hasattr(chapter_numbers, "__iter__"):
        raise EpubChapterSelectionError("chapter_numbers must be a list of positive integers")
    numbers = list(chapter_numbers)
    if not 1 <= len(numbers) <= MAX_SELECTED_CHAPTERS:
        raise EpubChapterSelectionError(f"Select between 1 and {MAX_SELECTED_CHAPTERS} chapters")
    for number in numbers:
        if type(number) is not int:  # rejects bool, float and numeric strings
            raise EpubChapterSelectionError("Chapter numbers must be integers")
        if not 1 <= number <= MAX_CHAPTER_NUMBER:
            raise EpubChapterSelectionError(f"Chapter numbers must be between 1 and {MAX_CHAPTER_NUMBER}")
    if len(set(numbers)) != len(numbers):
        raise EpubChapterSelectionError("Chapter numbers must not repeat")
    if numbers != sorted(numbers):
        raise EpubChapterSelectionError("Chapter numbers must be in ascending order")
    return numbers


def chapter_number_from_title(title) -> int | None:
    """Return the story-chapter number a parsed title declares, else ``None``."""
    match = _TITLE_RE.match(title.strip()) if isinstance(title, str) else None
    if not match:
        return None
    number = int(match.group("bare") or match.group("labelled"))
    return number if number >= 1 else None


def select_epub_chapters(data, filename, *, chapter_numbers) -> dict:
    """Select the chapters titled with exactly ``chapter_numbers`` from an EPUB.

    Returns the parsed book identity (title/author/language), the whole-file
    ``source_sha256``, the selected chapters with their exact parser text,
    member, extraction id and per-chapter ``text_sha256``, and a ``selection``
    manifest listing what was matched and what was left out. Raises
    ``EpubChapterSelectionError`` (a ``BookCreatorError``) when a requested
    number is missing or matches more than one document, or when the matching
    documents are not in ascending reading order. Archive safety failures are
    raised by the delegated parser unchanged.
    """
    numbers = _validated_numbers(chapter_numbers)
    if not isinstance(data, (bytes, bytearray)):
        raise EpubChapterSelectionError("EPUB data must be bytes")
    if not isinstance(filename, str) or not filename.strip():
        raise EpubChapterSelectionError("An EPUB filename is required")
    data = bytes(data)
    if not data or len(data) > book_creator.MAX_UPLOAD_BYTES:
        raise EpubChapterSelectionError("EPUB size is outside the bounded upload limit")

    parsed = book_creator._parse_epub(data, filename)
    candidates: dict[int, list[int]] = {}
    for position, chapter in enumerate(parsed["chapters"]):
        number = chapter_number_from_title(chapter["title"])
        if number is not None:
            candidates.setdefault(number, []).append(position)

    positions = []
    for number in numbers:
        found = candidates.get(number, [])
        if not found:
            raise EpubChapterSelectionError(f"No chapter titled as number {number} was found")
        if len(found) > 1:
            members = ", ".join(parsed["chapters"][p]["member"] for p in found)
            raise EpubChapterSelectionError(f"Chapter number {number} is ambiguous; it matches {members}")
        positions.append(found[0])
    if positions != sorted(positions):
        raise EpubChapterSelectionError("Numbered chapters are not in ascending reading order in this EPUB")

    chosen = set(positions)
    chapters, selected = [], []
    for number, position in zip(numbers, positions):
        source = parsed["chapters"][position]
        text_hash = _sha256(source["text"].encode("utf-8"))
        chapters.append({"chapter_number": number, "title": source["title"], "member": source["member"],
                         "extraction": source["extraction"], "text": source["text"], "text_sha256": text_hash})
        selected.append({"chapter_number": number, "title": source["title"], "member": source["member"],
                         "extraction": source["extraction"], "text_sha256": text_hash,
                         "reading_order_position": position})
    not_selected = [{"title": c["title"], "member": c["member"], "reading_order_position": p,
                     "chapter_number": chapter_number_from_title(c["title"])}
                    for p, c in enumerate(parsed["chapters"]) if p not in chosen]
    return {
        "title": parsed["title"], "author": parsed["author"], "language": parsed["language"],
        "source_sha256": _sha256(data), "chapters": chapters,
        "selection": {"schema": SELECTION_SCHEMA, "filename": filename, "title_rule": TITLE_RULE,
                      "requested_chapter_numbers": numbers, "selected": selected,
                      "parsed_chapter_count": len(parsed["chapters"]), "not_selected": not_selected},
    }
