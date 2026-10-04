"""Local-only source and rendered-coverage gates for the Voice Lab listening test.

This module owns no project state and performs no synthesis or network access.
The caller persists the returned content-addressed manifest through Studio before
synthesis. Offsets are Python Unicode character offsets in the complete extracted
chapter, while ``source_sha256`` identifies the immutable original source bytes.
Existing Segment IDs and SourceAnchor revisions are preserved by the adapter.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import posixpath
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence
from xml.etree import ElementTree as ET


SCHEMA = "voice-lab-listening-excerpt-v1"
MIN_SECONDS = 285.0
MAX_SECONDS = 315.0
TARGET_SECONDS = 300.0


class ExcerptValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise ExcerptValidationError(code, message)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False))


def _positive(value: float, field: str) -> float:
    if isinstance(value, bool):
        _fail("invalid_duration", f"{field} must be a finite positive duration")
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail("invalid_duration", f"{field} must be a finite positive duration")
    if not math.isfinite(result) or result <= 0:
        _fail("invalid_duration", f"{field} must be a finite positive duration")
    return result


_ONES = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))


def chapter_number(label: str | None) -> int | None:
    """Read an explicit number from a heading/TOC, never an import position.

    Supports numeric and spelled-out English chapter labels. Unrecognized labels
    remain unknown; e.g. Information and an EPUB member's 0001 prefix are not 1.
    """
    label = (label or "").strip()
    numeric = re.match(r"^(?:chapter\s+)?(\d+)\b(?:\s*[:.\-–—]|\s|$)", label, re.I)
    if numeric:
        return int(numeric.group(1))
    prefix = re.split(r"[:.\n–—]", label, maxsplit=1)[0].strip().lower()
    prefix = re.sub(r"^chapter\s+", "", prefix)
    if prefix in {"i", "ii", "iii", "iv", "v"}:
        return {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}[prefix]
    tokens = prefix.replace("-", " ").split()
    if not tokens or any(t not in {*_ONES, *_TENS, "hundred", "thousand", "and"} for t in tokens):
        return None
    current = total = 0
    for token in tokens:
        if token in _ONES:
            current += _ONES[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current = (current or 1) * 100
        elif token == "thousand":
            total += (current or 1) * 1000
            current = 0
    return total + current


@dataclass(frozen=True)
class ChapterIdentity:
    book_title: str
    source_identity: str
    source_revision_id: str
    source_sha256: str
    chapter_id: str
    chapter_number: int
    chapter_title: str
    toc_label: str
    heading: str
    chapter_text_sha256: str
    imported_index: int | None = None
    source_member: str | None = None
    extraction_method: str = "utf8-text-v1"


@dataclass(frozen=True)
class ExcerptUtterance:
    utterance_id: str
    source_revision_id: str
    chapter_id: str
    start_offset: int
    end_offset: int
    source_text_sha256: str
    kind: Literal["narration", "dialogue"]
    role_id: str | None
    attribution_evidence: str
    estimated_duration_seconds: float
    duration_basis: str
    speaker_resolved: bool = True
    complete_boundary: bool = True
    aliases: tuple[str, ...] = ()
    spoken_text_revision_id: str | None = None


def utterance_from_segment(segment: Any, *, kind: Literal["narration", "dialogue"],
                           attribution_evidence: str, estimated_duration_seconds: float,
                           duration_basis: str, complete_boundary: bool = True,
                           aliases: Sequence[str] = ()) -> ExcerptUtterance:
    """Adapt a canonical live Segment without minting IDs or inferring dialogue.

    ``kind`` and supporting evidence must come from source inspection. The mere
    presence of a character name or role assignment cannot establish speech.
    """
    anchor = segment.source_anchor
    if anchor is None or anchor.start_offset is None or anchor.end_offset is None:
        _fail("missing_anchor", f"Utterance {segment.id} has no exact source offsets")
    if segment.lineage and segment.lineage.retired_at is not None:
        _fail("retired_utterance", f"Utterance {segment.id} is retired")
    attribution = segment.attribution
    resolved = bool(attribution and attribution.speaker_id and attribution.source != "unknown")
    if resolved and attribution.speaker_id != segment.role_id:
        _fail("attribution_mismatch", f"Utterance {segment.id} has conflicting speaker identities")
    return ExcerptUtterance(
        utterance_id=segment.id, source_revision_id=anchor.source_revision_id,
        chapter_id=segment.chapter_id, start_offset=anchor.start_offset,
        end_offset=anchor.end_offset, source_text_sha256=text_hash(segment.text),
        kind=kind, role_id=segment.role_id, attribution_evidence=attribution_evidence,
        estimated_duration_seconds=estimated_duration_seconds, duration_basis=duration_basis,
        speaker_resolved=resolved, complete_boundary=complete_boundary,
        aliases=tuple(aliases), spoken_text_revision_id=segment.spoken_text_revision_id,
    )


def validate_chapter_identity(chapter: ChapterIdentity | None, *,
                              expected_book: str = "Super Supportive",
                              expected_chapter: int = 1) -> None:
    if chapter is None:
        _fail("missing_chapter", f"Literal Chapter {expected_chapter} of {expected_book} is unavailable locally")
    if " ".join(chapter.book_title.casefold().split()) != " ".join(expected_book.casefold().split()):
        _fail("wrong_book", f"Source metadata does not identify {expected_book}")
    values = [chapter.chapter_number, chapter_number(chapter.toc_label),
              chapter_number(chapter.heading), chapter_number(chapter.chapter_title)]
    if isinstance(chapter.chapter_number, bool) or any(value != expected_chapter for value in values):
        _fail("wrong_chapter", "Chapter number, TOC, heading and canonical chapter title must agree; import index is not a chapter number")
    if not all((chapter.source_identity, chapter.source_revision_id, chapter.chapter_id)):
        _fail("missing_provenance", "Source identity, canonical revision and chapter ID are required")
    if not all(re.fullmatch(r"[a-f0-9]{64}", h or "") for h in
               (chapter.source_sha256, chapter.chapter_text_sha256)):
        _fail("missing_provenance", "Original source and extracted chapter SHA256 hashes are required")


def extract_epub_chapter_text(source_bytes: bytes, *, member: str) -> str:
    """Deterministic full local body text for ``epub-xhtml-body-text-v1``.

    This is an extraction derivative, not a second canonical source authority.
    Preserve all body text including headings/notes; discard only nonspoken
    script/style elements. Block boundaries become newlines, inline text remains
    in source order, horizontal XML whitespace becomes a single space. Inspect
    the entire result before selecting source offsets and register anchors into
    this same derivative. No text is sent to any service.
    """
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        if archive.getinfo(member).file_size > 20 * 1024 * 1024:
            _fail("source_limit", "Source member exceeds bounded EPUB inspection limits")
        document = ET.fromstring(archive.read(member))
    body = next((node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "body"), None)
    if body is None:
        _fail("unverified_source_format", "A complete XHTML body is required for chapter extraction")
    chunks: list[str] = []
    blocks = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "br", "hr"}

    def visit(node):
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"script", "style"}:
            return
        if tag in blocks:
            chunks.append("\n")
        if node.text:
            chunks.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                chunks.append(child.tail)
        if tag in blocks:
            chunks.append("\n")

    visit(body)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in "".join(chunks).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def validate_source_bytes(chapter: ChapterIdentity, source_bytes: bytes) -> None:
    """Cross-check declared identity against original local snapshot metadata.

    EPUB chapter discovery requires matching TOC and document heading within the
    actual source bytes. A renamed canonical chapter or hand-written manifest
    cannot turn Chapter 256 into Chapter 1. Plain-text snapshots must be the
    complete exact chapter; other binary formats require an explicit extractor.
    """
    validate_chapter_identity(chapter)
    if hashlib.sha256(source_bytes).hexdigest() != chapter.source_sha256:
        _fail("source_hash_mismatch", "Original source bytes do not match the canonical revision")
    stream = io.BytesIO(source_bytes)
    if not zipfile.is_zipfile(stream):
        if chapter.source_sha256 != chapter.chapter_text_sha256 or chapter.extraction_method != "utf8-text-v1":
            _fail("unverified_source_format", "Non-EPUB source must be the exact complete chapter text snapshot")
        try:
            decoded = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            _fail("unverified_source_format", "Binary source requires a verified local chapter extractor")
        leading = [line.strip() for line in decoded.splitlines() if line.strip()][:8]
        actual_heading = next((line for line in leading if chapter_number(line) is not None), None)
        if (actual_heading is None or chapter_number(actual_heading) != chapter.chapter_number
                or " ".join(actual_heading.casefold().split()) != " ".join(chapter.heading.casefold().split())):
            _fail("wrong_chapter", "Plain-text snapshot must contain the matching literal Chapter 1 heading before story prose")
        return
    try:
        with zipfile.ZipFile(stream) as archive:
            if len(archive.infolist()) > 5000 or sum(m.file_size for m in archive.infolist()) > 100 * 1024 * 1024:
                _fail("source_limit", "Source exceeds bounded EPUB inspection limits")

            def xml(name):
                if archive.getinfo(name).file_size > 20 * 1024 * 1024:
                    _fail("source_limit", "Source member exceeds bounded EPUB inspection limits")
                return ET.fromstring(archive.read(name))

            container = xml("META-INF/container.xml")
            package_name = next(e.attrib["full-path"] for e in container.iter() if e.tag.endswith("rootfile"))
            package = xml(package_name)
            base = posixpath.dirname(package_name)
            normalize = lambda value: " ".join(value.split()).casefold()
            titles = [" ".join(e.itertext()) for e in package.iter() if e.tag == "{http://purl.org/dc/elements/1.1/}title"]
            if normalize(chapter.book_title) not in [normalize(t) for t in titles]:
                _fail("wrong_book", "EPUB metadata does not identify the requested work")
            toc_candidates = []
            for item in package.iter():
                if item.tag.rsplit("}", 1)[-1] != "item":
                    continue
                name = posixpath.normpath(posixpath.join(base, item.attrib["href"]))
                if item.attrib.get("media-type") == "application/x-dtbncx+xml":
                    for point in xml(name).iter():
                        if point.tag.rsplit("}", 1)[-1] == "navPoint":
                            label = next((" ".join(e.itertext()) for e in point if e.tag.endswith("navLabel")), "")
                            href = next((e.attrib.get("src", "") for e in point if e.tag.endswith("content")), "")
                            if normalize(label) == normalize(chapter.toc_label):
                                toc_candidates.append(posixpath.normpath(posixpath.join(posixpath.dirname(name), href.split("#")[0])))
                elif "nav" in item.attrib.get("properties", "").split():
                    for link in xml(name).iter():
                        if link.tag.rsplit("}", 1)[-1] == "a" and normalize(" ".join(link.itertext())) == normalize(chapter.toc_label):
                            toc_candidates.append(posixpath.normpath(posixpath.join(posixpath.dirname(name), link.attrib.get("href", "").split("#")[0])))
            found_members = set()
            for name in set(toc_candidates):
                for heading in xml(name).iter():
                    if heading.tag.rsplit("}", 1)[-1] in {"h1", "h2", "h3"}:
                        label = " ".join(heading.itertext())
                        if normalize(label) == normalize(chapter.heading) and chapter_number(label) == 1:
                            found_members.add(name)
            if not found_members:
                _fail("wrong_chapter", "Original EPUB TOC and corresponding document heading do not prove literal Chapter 1")
            if len(found_members) != 1 or chapter.source_member not in found_members:
                _fail("source_member_mismatch", "Manifest must identify the exact unambiguous Chapter 1 XHTML member")
            if chapter.extraction_method != "epub-xhtml-body-text-v1":
                _fail("unverified_extraction", "EPUB offsets require the explicit epub-xhtml-body-text-v1 extraction method")
            actual = extract_epub_chapter_text(source_bytes, member=chapter.source_member)
            if text_hash(actual) != chapter.chapter_text_sha256:
                _fail("source_extraction_mismatch", "Declared complete chapter text does not match the actual Chapter 1 XHTML body")
    except ExcerptValidationError:
        raise
    except (KeyError, ValueError, StopIteration, ET.ParseError, zipfile.BadZipFile) as exc:
        _fail("unverified_source_format", f"EPUB metadata could not be verified: {type(exc).__name__}")


def _resolve(role_id: str | None, role_aliases: Mapping[str, str], known: set[str]) -> str:
    if not role_id:
        _fail("unresolved_speaker", "Dialogue has no resolved speaker identity")
    seen: set[str] = set()
    while role_id in role_aliases and role_aliases[role_id] != role_id:
        if role_id in seen:
            _fail("alias_cycle", "Role aliases contain a cycle")
        seen.add(role_id)
        role_id = role_aliases[role_id]
    if role_id not in known:
        _fail("unresolved_speaker", f"Role {role_id} is not a resolved canonical character")
    return role_id


def build_excerpt_manifest(*, chapter: ChapterIdentity, source_text: str,
                           utterances: Sequence[ExcerptUtterance],
                           selected_utterance_ids: Sequence[str],
                           character_ids: Sequence[str], narrator_role_id: str,
                           role_aliases: Mapping[str, str] | None = None,
                           full_chapter_inspected_sha256: str,
                           selection_rationale: str) -> dict[str, Any]:
    """Validate a contiguous complete passage and return a text-free manifest.

    Pass the complete canonical chapter utterance sequence, in source order.
    Selection is a contiguous slice, never a set. All dialogue must be resolved;
    extra speakers are preserved. Estimates gate selection, decoded audio gates
    delivery separately. No source prose is included in this return value.
    """
    validate_chapter_identity(chapter)
    digest = text_hash(source_text)
    if digest != chapter.chapter_text_sha256:
        _fail("source_hash_mismatch", "Extracted chapter content changed")
    if full_chapter_inspected_sha256 != digest:
        _fail("full_chapter_uninspected", "Full-chapter inspection must bind to the current complete chapter hash")
    if not selection_rationale.strip():
        _fail("missing_rationale", "Explain source continuity, boundaries and meaningful role speech")
    ids = [u.utterance_id for u in utterances]
    selected_ids = list(selected_utterance_ids)
    if not ids or len(ids) != len(set(ids)) or any(not i for i in ids):
        _fail("invalid_utterance_ids", "Canonical live utterance IDs must be nonempty and unique")
    if not selected_ids or any(i not in ids for i in selected_ids):
        _fail("unknown_utterance", "Selection must use existing canonical utterance IDs")
    first = ids.index(selected_ids[0])
    if ids[first:first + len(selected_ids)] != selected_ids:
        _fail("noncontiguous_selection", "Selection must retain every interior canonical utterance in source order")
    selected = utterances[first:first + len(selected_ids)]
    aliases = dict(role_aliases or {})
    known = set(character_ids)
    if not narrator_role_id or narrator_role_id in known:
        _fail("invalid_narrator", "Narrator must be a separate canonical role")
    counts: Counter[str] = Counter()
    role_seconds: dict[str, float] = defaultdict(float)
    rows: list[dict[str, Any]] = []
    last_end: int | None = None
    narrator_count = 0
    for u in selected:
        if u.source_revision_id != chapter.source_revision_id or u.chapter_id != chapter.chapter_id:
            _fail("source_revision_mismatch", f"Utterance {u.utterance_id} belongs to another source revision or chapter")
        if (any(not isinstance(value, int) or isinstance(value, bool) for value in (u.start_offset, u.end_offset))
                or not 0 <= u.start_offset < u.end_offset <= len(source_text)):
            _fail("invalid_offsets", f"Utterance {u.utterance_id} has invalid exact source offsets")
        if last_end is not None and (u.start_offset < last_end or source_text[last_end:u.start_offset].strip()):
            _fail("source_gap", "The passage overlaps or omits interior source text")
        if text_hash(source_text[u.start_offset:u.end_offset]) != u.source_text_sha256:
            _fail("utterance_hash_mismatch", f"Utterance {u.utterance_id} does not match canonical source offsets")
        if not u.complete_boundary:
            _fail("incomplete_boundary", f"Utterance {u.utterance_id} is not a complete source utterance")
        if not u.attribution_evidence.strip() or not u.speaker_resolved:
            _fail("unresolved_speaker", f"Utterance {u.utterance_id} requires source-supported attribution")
        seconds = _positive(u.estimated_duration_seconds, "estimated_duration_seconds")
        if not u.duration_basis.strip():
            _fail("missing_timing_basis", "Each estimate requires its measured timing basis")
        if u.kind == "narration":
            if u.role_id != narrator_role_id:
                _fail("invalid_narrator", "Source narration must retain the narrator role")
            resolved = narrator_role_id
            narrator_count += 1
        elif u.kind == "dialogue":
            resolved = _resolve(u.role_id, aliases, known)
            counts[resolved] += 1
        else:
            _fail("invalid_kind", "Only source narration and actual spoken dialogue count")
        role_seconds[resolved] += seconds
        row = asdict(u)
        row["role_id"] = resolved
        row["source_role_id"] = u.role_id
        rows.append(row)
        last_end = u.end_offset
    if narrator_count == 0:
        _fail("missing_narration", "The excerpt requires existing interleaved narration")
    if len(counts) < 4:
        _fail("insufficient_speakers", f"Found {len(counts)} resolved speaking characters; four plus the narrator are required")
    dialogue_positions = [i for i, row in enumerate(rows) if row["kind"] == "dialogue"]
    if not any(row["kind"] == "narration" for row in rows[dialogue_positions[0] + 1:dialogue_positions[-1]]):
        _fail("missing_interleaved_narration", "Existing narration must flow between actual dialogue turns")
    # A measurable floor guards a token exclamation counting as a useful voice
    # sample. Source inspection still determines whether speech is meaningful.
    if any(seconds < 2.0 for seconds in role_seconds.values()):
        _fail("insufficient_role_speech", "Each role requires at least two estimated seconds of meaningful source speech")
    duration = sum(role_seconds.values())
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        _fail("duration_out_of_bounds", f"Estimated duration {duration:.3f}s is outside 285–315 seconds")
    manifest = {
        "schema": SCHEMA, "status": "selected_pending_synthesis",
        "chapter": asdict(chapter), "offset_unit": "unicode_codepoint",
        "full_chapter_inspected_sha256": digest,
        "start_offset": selected[0].start_offset, "end_offset": selected[-1].end_offset,
        "selected_source_sha256": text_hash(source_text[selected[0].start_offset:selected[-1].end_offset]),
        "utterance_ids": selected_ids, "utterances": rows,
        "chapter_utterances": [asdict(row) for row in utterances],
        "chapter_character_ids": sorted(known),
        "narrator_role_id": narrator_role_id, "narration_turn_count": narrator_count,
        "character_ids": sorted(counts), "speaking_turn_counts": dict(sorted(counts.items())),
        "role_aliases": aliases, "estimated_duration_seconds": duration,
        "estimated_role_seconds": dict(role_seconds), "selection_rationale": selection_rationale,
        "technical_qa": "pending", "human_review": "pending",
    }
    manifest["manifest_id"] = "excerpt_" + _digest(manifest)[:24]
    return manifest


def adjust_excerpt_to_measured_duration(manifest: Mapping[str, Any], *, source_text: str,
                                       measured_utterance_seconds: Mapping[str, float],
                                       max_neighbor_utterances: int = 8,
                                       max_added_seconds: float = 60.0) -> dict[str, Any]:
    """Bounded adjacent source adjustment using actual decoded Take durations.

    Keep every required character and narration. Trim only complete leading or
    trailing utterances, or add complete adjacent utterances; never splice a
    scene. Known audio uses measured seconds, neighbors use per-role calibration
    from those measurements. The caller renders only new/invalid Takes, then
    checks actual duration again. No audio is altered to force a fit.
    """
    if not 1 <= max_neighbor_utterances <= 12 or not 0 < max_added_seconds <= 60:
        _fail("adjustment_limit", "Timing adjustment permits at most 12 neighbors and 60 estimated new seconds")
    rows = [ExcerptUtterance(**row) for row in manifest.get("chapter_utterances", [])]
    if not rows:
        _fail("constraint_conflict", "Adjacent complete source utterances were not persisted during full-chapter inspection")
    ids = [row.utterance_id for row in rows]
    current = list(manifest["utterance_ids"])
    start = ids.index(current[0]) if current and current[0] in ids else -1
    stop = start + len(current)
    if start < 0 or ids[start:stop] != current:
        _fail("noncontiguous_selection", "Persisted full-chapter sequence does not contain the selected contiguous excerpt")
    actual = {i: _positive(measured_utterance_seconds.get(i), "decoded_utterance_seconds") for i in current}
    aliases = manifest.get("role_aliases", {})
    known = set(manifest.get("chapter_character_ids", manifest["character_ids"]))
    narrator = manifest["narrator_role_id"]
    role_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows[start:stop]:
        role = narrator if row.kind == "narration" else _resolve(row.role_id, aliases, known)
        role_totals[role][0] += actual[row.utterance_id]
        role_totals[role][1] += _positive(row.estimated_duration_seconds, "estimated_duration_seconds")
    global_ratio = sum(actual.values()) / sum(row.estimated_duration_seconds for row in rows[start:stop])
    calibrated = []
    for row in rows:
        role = row.role_id
        try:
            role = narrator if row.kind == "narration" else _resolve(role, aliases, known)
        except ExcerptValidationError:
            pass  # An unresolved neighbor remains unusable when validating candidates.
        measured = actual.get(row.utterance_id)
        total = role_totals.get(role)
        ratio = total[0] / total[1] if total else global_ratio
        seconds = measured if measured is not None else _positive(row.estimated_duration_seconds, "estimated_duration_seconds") * ratio
        calibrated.append(ExcerptUtterance(**{**asdict(row), "estimated_duration_seconds": seconds,
            "duration_basis": "Exact decoded selected Take duration" if measured is not None else "Adjacent source estimate calibrated from decoded role timing"}))
    best = None
    best_score = None
    for left in range(max(0, start - max_neighbor_utterances), min(stop, start + max_neighbor_utterances) + 1):
        for right in range(max(left + 1, stop - max_neighbor_utterances), min(len(rows), stop + max_neighbor_utterances) + 1):
            if left >= stop or right <= start or (left == start and right == stop):
                continue
            selected_ids = ids[left:right]
            added = [row for row in calibrated[left:right] if row.utterance_id not in actual]
            added_seconds = sum(row.estimated_duration_seconds for row in added)
            if len(added) > max_neighbor_utterances or added_seconds > max_added_seconds:
                continue
            seconds = sum(row.estimated_duration_seconds for row in calibrated[left:right])
            if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                continue
            try:
                candidate = build_excerpt_manifest(
                    chapter=ChapterIdentity(**manifest["chapter"]), source_text=source_text,
                    utterances=calibrated, selected_utterance_ids=selected_ids,
                    character_ids=sorted(known), narrator_role_id=narrator, role_aliases=aliases,
                    full_chapter_inspected_sha256=manifest["full_chapter_inspected_sha256"],
                    selection_rationale=f"Adjacent complete source adjustment from {manifest['manifest_id']} after {sum(actual.values()):.3f} decoded seconds; at most {max_neighbor_utterances} new utterances and {max_added_seconds:g} estimated new seconds; all original required speakers retained.",
                )
            except ExcerptValidationError:
                continue
            if not set(manifest["character_ids"]).issubset(candidate["character_ids"]):
                continue
            score = (bool(added), abs(seconds - TARGET_SECONDS), added_seconds, len(added))
            if best_score is None or score < best_score:
                best, best_score = candidate, score
    if best is None:
        _fail("constraint_conflict", "No bounded adjacent complete Chapter 1 passage fits 285–315 seconds while retaining every required speaker; no padding, speed change or unrelated scene is permitted")
    return best


def select_contiguous_excerpt(*, chapter: ChapterIdentity, source_text: str,
                              utterances: Sequence[ExcerptUtterance],
                              character_ids: Sequence[str], narrator_role_id: str,
                              full_chapter_inspected_sha256: str,
                              role_aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Choose the valid complete window nearest 300s; never synthesize to search.

    The complete sequence must come from prior full-chapter source inspection.
    At most 10,000 utterances are accepted; positive estimates bound each window.
    """
    validate_chapter_identity(chapter)
    if len(utterances) > 10_000:
        _fail("selection_limit", "Chapter exceeds the bounded 10,000-utterance selection limit")
    if not utterances:
        _fail("constraint_conflict", "Chapter has no canonical source utterances")
    durations = [_positive(u.estimated_duration_seconds, "estimated_duration_seconds") for u in utterances]
    best: dict[str, Any] | None = None
    reasons: Counter[str] = Counter()
    for start in range(len(utterances)):
        seconds = 0.0
        for end in range(start, len(utterances)):
            seconds += durations[end]
            if seconds > MAX_SECONDS:
                break
            if seconds < MIN_SECONDS:
                continue
            try:
                candidate = build_excerpt_manifest(
                    chapter=chapter, source_text=source_text, utterances=utterances,
                    selected_utterance_ids=[u.utterance_id for u in utterances[start:end + 1]],
                    character_ids=character_ids, narrator_role_id=narrator_role_id,
                    role_aliases=role_aliases, full_chapter_inspected_sha256=full_chapter_inspected_sha256,
                    selection_rationale="Contiguous complete source utterances nearest 300 seconds using recorded timing estimates; all source speakers retained.",
                )
            except ExcerptValidationError as exc:
                reasons[exc.code] += 1
                continue
            if best is None or abs(seconds - TARGET_SECONDS) < abs(best["estimated_duration_seconds"] - TARGET_SECONDS):
                best = candidate
    if best is None:
        _fail("constraint_conflict", "No contiguous complete 285–315s Chapter 1 passage satisfies four resolved speaking characters plus narration; validation reasons: " + json.dumps(dict(reasons), sort_keys=True))
    return best


def validate_rendered_excerpt(manifest: Mapping[str, Any], *,
                              audio_map: Sequence[Mapping[str, Any]],
                              decoded_duration_seconds: float,
                              audio_sha256: str) -> dict[str, Any]:
    """Check actual production coverage, stable distinct voices and decoded time.

    Rows require utterance_id, role_id, voice_id, voice_version,
    conditioning_sha256 (or immutable provider identity), start_seconds,
    end_seconds, take_id and audio_sha256. Different IDs around the same
    conditioning do not prove different voices. This supplements normal Take QA;
    it does not label technical or subjective performance checks as passed.
    """
    unbound = dict(manifest)
    identity = unbound.pop("manifest_id", None)
    if identity != "excerpt_" + _digest(unbound)[:24]:
        _fail("manifest_hash_mismatch", "Selection manifest changed after persistence")
    duration = _positive(decoded_duration_seconds, "decoded_duration_seconds")
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        _fail("duration_out_of_bounds", f"Decoded audio duration {duration:.3f}s is outside 285–315 seconds")
    if not re.fullmatch(r"[a-f0-9]{64}", audio_sha256 or ""):
        _fail("missing_audio_hash", "The assembled audio requires its actual SHA256")
    if [row.get("utterance_id") for row in audio_map] != manifest["utterance_ids"]:
        _fail("render_coverage", "Rendered audio must contain every selected utterance exactly once in order")
    identities: dict[str, tuple[str, str, str]] = {}
    actual_role_seconds: dict[str, float] = defaultdict(float)
    cursor = 0.0
    for source, row in zip(manifest["utterances"], audio_map):
        if row.get("role_id") != source["role_id"]:
            _fail("render_attribution", "Rendered speaker differs from the source-backed attribution")
        start, end = row.get("start_seconds"), row.get("end_seconds")
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or abs(start - cursor) > 0.02:
            _fail("render_gap", "Source-to-audio map has padding, overlap or an interior gap")
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start:
            _fail("render_gap", "Source-to-audio map has an invalid duration")
        cursor = end
        voice = (row.get("voice_id"), str(row.get("voice_version") or ""), row.get("conditioning_sha256") or row.get("provider_voice_identity"))
        if not all(voice) or not row.get("take_id") or not re.fullmatch(r"[a-f0-9]{64}", row.get("audio_sha256") or ""):
            _fail("missing_production_identity", "Every rendered utterance requires immutable Take, audio hash and final conditioning identity")
        role = source["role_id"]
        actual_role_seconds[role] += end - start
        if role in identities and identities[role] != voice:
            _fail("voice_identity_drift", f"Role {role} changed voice identity within the excerpt")
        identities[role] = voice
    if abs(cursor - duration) > 0.02:
        _fail("render_duration_mismatch", "Decoded audio duration differs from source-to-audio coverage")
    if len({voice[2] for voice in identities.values()}) != len(identities):
        _fail("shared_voice_conditioning", "Every character and the narrator require distinct production conditioning identities")
    if any(seconds < 2.0 for seconds in actual_role_seconds.values()):
        _fail("insufficient_role_speech", "Each role requires at least two decoded seconds of meaningful source speech")
    return {
        "manifest_id": identity, "audio_sha256": audio_sha256,
        "duration_seconds": duration, "speaking_character_count": len(manifest["character_ids"]),
        "rendered_voice_role_count": len(identities), "source_coverage": "passed",
        "actual_role_seconds": dict(actual_role_seconds),
        "technical_qa": "pending", "human_review": "pending",
    }


def blocked_source_report(*, available_chapters: Sequence[int], inspected_paths: Sequence[str],
                          reason: str | None = None) -> dict[str, Any]:
    """A durable local-source blocker suitable for Project.voice_lab.excerpt."""
    return {
        "status": "blocked_source", "code": "missing_chapter",
        "requested_book": "Super Supportive", "requested_chapter": 1,
        "reason": reason or "Literal Chapter 1 of Super Supportive was not found in the inspected local sources. Imported subset ordering cannot substitute for chapter identity.",
        "available_chapters": sorted(set(available_chapters)),
        "inspected_paths": list(inspected_paths), "manifest_id": None,
        "duration_seconds": None, "audio_url": None, "audio_sha256": None,
        "technical_qa": "not_run_source_unavailable", "human_review": "pending",
        "generation_totals": {"conditioning_reference_seconds": 0, "candidate_seconds": 0, "final_listening_seconds": 0},
    }
