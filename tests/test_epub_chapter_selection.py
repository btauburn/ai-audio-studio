"""Numbered-chapter selection over real EPUB packages: no positional guessing, exact bytes."""
import hashlib
import io
import zipfile

import pytest

from dubbing_service.app.audiobook import book_creator, epub_chapter_selection as selection
from dubbing_service.app.audiobook.epub_chapter_selection import EpubChapterSelectionError, select_epub_chapters

NS = 'xmlns="http://www.w3.org/1999/xhtml"'
CHAPTER_BODIES = {
    1: "<p>The first night was cold.</p><p>“Stay close,” she said.</p>",
    2: "<p>Morning came slowly.</p><p>Second   body  line.</p>",
    3: "<p>Third door, third key.</p><br/><p>Last line here.</p>",
    4: "<p>Fourth chapter text.</p>",
}


def xhtml(title, body):
    return f'<?xml version="1.0" encoding="utf-8"?><html {NS}><head><title>x</title></head><body><h1>{title}</h1>{body}</body></html>'


def default_docs():
    docs = [
        ("Cover.xhtml", "cover", f'<html {NS}><body><img src="cover.jpg" alt=""/></body></html>', None),
        ("0000_Information.xhtml", "info", xhtml("Information", "<p>Translated and published online.</p>"), "Information"),
    ]
    for number, body in CHAPTER_BODIES.items():
        docs.append((f"{number:04d}_{number:03d}.xhtml", f"c{number}", xhtml(f"{number:03d}", body), f"{number:03d}"))
    return docs


def epub(docs=None, *, extra=None, nav=True, encryption=False, raw_members=(), hrefs=None):
    docs = default_docs() if docs is None else docs
    hrefs = hrefs or {}
    manifest = "".join(f'<item id="{i}" href="{hrefs.get(i, "Text/" + name)}" media-type="application/xhtml+xml"/>' for name, i, _, _ in docs)
    manifest += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>' if nav else ""
    spine = "".join(f'<itemref idref="{i}"/>' for _, i, _, _ in docs)
    links = "".join(f'<li><a href="Text/{name}">{label}</a></li>' for name, _, _, label in docs if label)
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                         '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        archive.writestr("OEBPS/content.opf", '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                         'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture Saga</dc:title><dc:creator>Fixture Author</dc:creator>'
                         f'<dc:language>en</dc:language></metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        if nav:
            archive.writestr("OEBPS/nav.xhtml", f'<html {NS}><body><nav><ol>{links}</ol></nav></body></html>')
        for name, _, content, _ in docs:
            archive.writestr(f"OEBPS/Text/{name}", content)
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
        for name, content in raw_members:
            archive.writestr(name, content)
        if encryption:
            archive.writestr("META-INF/encryption.xml", "<encryption/>")
    return target.getvalue()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def test_first_three_numbered_chapters_skip_cover_and_information_with_exact_hashes():
    data = epub()
    result = select_epub_chapters(data, "saga.epub", chapter_numbers=[1, 2, 3])
    assert (result["title"], result["author"], result["language"]) == ("Fixture Saga", "Fixture Author", "en")
    assert result["source_sha256"] == sha(data)
    assert [c["member"] for c in result["chapters"]] == [f"OEBPS/Text/000{n}_00{n}.xhtml" for n in (1, 2, 3)]
    assert [c["chapter_number"] for c in result["chapters"]] == [1, 2, 3]
    for chapter in result["chapters"]:
        expected = book_creator._parse_epub(data, "saga.epub")["chapters"]
        source = next(c for c in expected if c["member"] == chapter["member"])
        assert chapter["text"] == source["text"]  # byte-identical parser output, nothing rewritten
        assert chapter["extraction"] == "epub-xhtml-body-text-v1"
        assert chapter["text_sha256"] == sha(chapter["text"])
    assert result["chapters"][0]["text"] == "001\n\nThe first night was cold.\n\n“Stay close,” she said."
    manifest = result["selection"]
    assert manifest["schema"] == selection.SELECTION_SCHEMA and manifest["requested_chapter_numbers"] == [1, 2, 3]
    assert [s["text_sha256"] for s in manifest["selected"]] == [c["text_sha256"] for c in result["chapters"]]
    assert [s["reading_order_position"] for s in manifest["selected"]] == [1, 2, 3]  # Cover has no text; Information is position 0
    left_out = {n["member"]: n["chapter_number"] for n in manifest["not_selected"]}
    assert left_out == {"OEBPS/Text/0000_Information.xhtml": None, "OEBPS/Text/0004_004.xhtml": 4}
    assert manifest["parsed_chapter_count"] == 5


def test_selection_is_deterministic_and_does_not_mutate_source_bytes():
    data = epub()
    before = bytes(data)
    first = select_epub_chapters(data, "saga.epub", chapter_numbers=[1, 2, 3])
    assert select_epub_chapters(bytearray(data), "saga.epub", chapter_numbers=(1, 2, 3)) == first
    assert data == before


def test_non_contiguous_selection_picks_by_number_not_position():
    result = select_epub_chapters(epub(), "saga.epub", chapter_numbers=[2, 4])
    assert [c["chapter_number"] for c in result["chapters"]] == [2, 4]
    assert [c["member"] for c in result["chapters"]] == ["OEBPS/Text/0002_002.xhtml", "OEBPS/Text/0004_004.xhtml"]


@pytest.mark.parametrize("title,number", [("001", 1), ("7", 7), ("Chapter 1", 1), ("CHAPTER 012", 12), ("Chapter 3: The Key", 3),
                                          ("Chapter 4 — Dusk", 4), (" 002 ", 2)])
def test_accepted_title_labels(title, number):
    assert selection.chapter_number_from_title(title) == number


@pytest.mark.parametrize("title", ["Information", "Cover", "Contents", "Table of Contents", "Chapter One", "Chapter 1 and 2", "Part 2 Chapter 1",
                                   "0", "000", "Chapter 0", "Chapter", "12345", "1a", "1.5", "Prologue", "Section 1", "Chapter 1:", "١", "", None])
def test_rejected_title_labels(title):
    assert selection.chapter_number_from_title(title) is None


def test_numbered_labels_in_a_contents_page_do_not_become_chapters():
    docs = default_docs()
    toc = xhtml("Contents", "<p>001</p><p>002</p><p>003</p>")
    docs.insert(2, ("Contents.xhtml", "toc", toc, "Table of Contents"))
    result = select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 2, 3])
    assert all(c["member"].endswith(f"_00{c['chapter_number']}.xhtml") for c in result["chapters"])


def test_heading_supplies_number_when_navigation_has_no_label():
    docs = [(n, i, c, None) for n, i, c, _ in default_docs()]
    result = select_epub_chapters(epub(docs, nav=False), "saga.epub", chapter_numbers=[1, 2, 3])
    assert [c["title"] for c in result["chapters"]] == ["001", "002", "003"]


def test_chapter_prefixed_titles_match():
    docs = [(n, i, xhtml(f"Chapter {k}", CHAPTER_BODIES[k]), f"Chapter {k}") if k else (n, i, c, l)
            for k, (n, i, c, l) in zip([0, 0, 1, 2, 3, 4], default_docs())]
    result = select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 2, 3])
    assert [c["title"] for c in result["chapters"]] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_missing_chapter_is_rejected_not_substituted():
    with pytest.raises(EpubChapterSelectionError, match="number 5"):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=[1, 5])
    docs = [d for d in default_docs() if d[1] != "c2"]
    with pytest.raises(EpubChapterSelectionError, match="number 2"):
        select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 2, 3])


def test_unnumbered_book_is_never_guessed_from_spine_order():
    docs = [(n, i, xhtml(f"Scene {k}", "<p>Text.</p>"), f"Scene {k}") for k, (n, i, _, _) in enumerate(default_docs())]
    with pytest.raises(EpubChapterSelectionError, match="number 1"):
        select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1])


def test_duplicate_labels_are_ambiguous():
    docs = default_docs()
    docs.append(("0005_Extra.xhtml", "extra", xhtml("Chapter 2", "<p>Bonus.</p>"), "Chapter 2"))
    with pytest.raises(EpubChapterSelectionError, match="number 2 is ambiguous.*0002_002.*0005_Extra"):
        select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 2])
    # An unrequested duplicate does not block unrelated, unambiguous requests.
    assert [c["chapter_number"] for c in select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 3])["chapters"]] == [1, 3]
    equivalent = default_docs()
    equivalent.append(("0005_Alt.xhtml", "alt", xhtml("1", "<p>Alt.</p>"), "1"))  # "1" and "001" are the same number
    with pytest.raises(EpubChapterSelectionError, match="ambiguous"):
        select_epub_chapters(epub(equivalent), "saga.epub", chapter_numbers=[1])


def test_source_order_must_match_requested_ascending_numbers():
    docs = default_docs()
    docs[2], docs[3] = docs[3], docs[2]  # spine reads 002 before 001
    with pytest.raises(EpubChapterSelectionError, match="ascending reading order"):
        select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[1, 2])
    assert [c["chapter_number"] for c in select_epub_chapters(epub(docs), "saga.epub", chapter_numbers=[2])["chapters"]] == [2]


@pytest.mark.parametrize("numbers", [[], [0], [-1], [1, 1], [2, 1], [True], [1, True], [1.0], ["1"], [None], [201], "123", b"1", None, 3])
def test_invalid_chapter_numbers_are_rejected_before_parsing(numbers, monkeypatch):
    monkeypatch.setattr(book_creator, "_parse_epub", lambda *a: pytest.fail("parser must not run for invalid numbers"))
    with pytest.raises(EpubChapterSelectionError):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=numbers)


def test_reusable_upper_bound_is_exactly_the_parser_chapter_limit():
    assert selection.MAX_SELECTED_CHAPTERS == selection.MAX_CHAPTER_NUMBER == book_creator.MAX_CHAPTERS == 200
    with pytest.raises(EpubChapterSelectionError, match="number 200"):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=[1, 200])
    with pytest.raises(EpubChapterSelectionError, match="between 1 and 200"):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=list(range(1, 202)))


@pytest.mark.parametrize("data,filename", [("text", "x.epub"), (b"", "x.epub"), (b"not a zip", "x.epub"), (epub(), ""), (epub(), None)])
def test_bad_inputs_are_rejected(data, filename):
    with pytest.raises(book_creator.BookCreatorError):  # selection errors subclass it; bad zip comes from the parser
        select_epub_chapters(data, filename, chapter_numbers=[1])


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(book_creator, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(EpubChapterSelectionError, match="upload limit"):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=[1])


@pytest.mark.parametrize("kind", ["traversal", "absolute", "backslash", "encryption", "duplicate", "symlink", "member-encrypted-flag"])
def test_unsafe_archives_are_rejected_by_the_delegated_parser(kind):
    if kind == "traversal":
        data = epub(extra={"../escape.txt": "x"})
    elif kind == "absolute":
        data = epub(extra={"/abs.txt": "x"})
    elif kind == "backslash":
        data = epub(extra={"OEBPS\\..\\..\\escape.txt": "x"})
    elif kind == "encryption":
        data = epub(encryption=True)
    elif kind == "duplicate":
        with pytest.warns(UserWarning):
            data = epub(raw_members=[("OEBPS/Text/0001_001.xhtml", "second copy")])
    elif kind == "symlink":
        info = zipfile.ZipInfo("OEBPS/link.xhtml")
        info.create_system = 3
        info.external_attr = (0o120777) << 16
        buffer = io.BytesIO(epub())
        with zipfile.ZipFile(buffer, "a") as archive:
            archive.writestr(info, "OEBPS/Text/0001_001.xhtml")
        data = buffer.getvalue()
    else:
        raw = bytearray(epub(extra={"OEBPS/secret.bin": "x"}))
        central = raw.index(b"PK", raw.rindex(b"PK", 0, raw.index(b"OEBPS/secret.bin", raw.index(b"PK"))))
        raw[central + 8] |= 1  # general-purpose flag bit 0 = encrypted, in the central directory entry
        data = bytes(raw)
    with pytest.raises(book_creator.BookCreatorError):
        select_epub_chapters(data, "saga.epub", chapter_numbers=[1, 2, 3])


def test_oversized_member_and_missing_spine_document_are_rejected_by_parser(monkeypatch):
    monkeypatch.setattr(book_creator, "MAX_MEMBER_BYTES", 100)
    with pytest.raises(book_creator.BookCreatorError, match="oversized"):
        select_epub_chapters(epub(), "saga.epub", chapter_numbers=[1])
    monkeypatch.undo()
    data = epub()
    broken = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(broken, "w") as target:
        for item in source.infolist():
            if item.filename != "OEBPS/Text/0003_003.xhtml":
                target.writestr(item, source.read(item.filename))
    with pytest.raises(book_creator.BookCreatorError, match="repeated or missing"):
        select_epub_chapters(broken.getvalue(), "saga.epub", chapter_numbers=[1])


def test_external_references_are_never_followed():
    docs = default_docs()
    docs.append(("0005_005.xhtml", "ext", xhtml("005", "<p>Remote.</p>"), "005"))
    data = epub(docs, hrefs={"ext": "https://example.invalid/0005_005.xhtml"})
    with pytest.raises(book_creator.BookCreatorError):
        select_epub_chapters(data, "saga.epub", chapter_numbers=[1])
