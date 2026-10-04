"""Web addresses are never read aloud: found (even with their numbers spelled out) and removed."""
from __future__ import annotations

from pathlib import Path

import pytest

from dubbing_service.app.audiobook import adapter, autocast_import
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import web_addresses as web
from dubbing_service.app.audiobook.provenance import spoken_text_for_segment
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments
from tests.audiobook_helpers import use_projects_root

# The three lines that stopped the Elf acceptance book, exactly as the import left them.
ROYAL_ROAD = ("Table of Contents URL: https://www.royalroad.com/fiction/one hundred four thousand four hundred "
              "thirty-four/the-elf-who-would-become-a-dragon-story-complete")
IMAGE_ONE = "https://ljamberfantasy.s-ul.eu/yuVbforty-sixWq"
IMAGE_TWO = "https://ljamberfantasy.s-ul.eu/lfiveyueCkA"


@pytest.mark.parametrize("line", [ROYAL_ROAD, IMAGE_ONE, IMAGE_TWO, "Join the Discord: https://discord.gg/abc123",
                                  "www.example.com", "Chapter art (link): https://example.com/art.png."])
def test_a_line_that_is_only_a_link_is_a_link_line(line):
    assert web.is_link_line(line)


def test_an_address_whose_number_was_spelled_out_is_found_whole():
    (start, end), = web.spans(ROYAL_ROAD)
    assert ROYAL_ROAD[start:end] == ROYAL_ROAD[ROYAL_ROAD.index("https"):], \
        "it runs on through 'one hundred four thousand ...' to the end of the address"


def test_an_address_stops_where_the_sentence_carries_on():
    line = "Go to https://x.com/page one more time, she said, and see what is there for yourself."
    (start, end), = web.spans(line)
    assert line[start:end] == "https://x.com/page", "'one' after a word that is not a number belongs to the sentence"
    line = "It was at https://x.com/a. Then she left."
    (start, end), = web.spans(line)
    assert line[start:end] == "https://x.com/a", "the full stop ends the sentence, not the address"


@pytest.mark.parametrize("line, spoken", [
    ("You can find the maps at https://example.com/maps and more.", "You can find the maps and more."),
    ("She drew the map herself (https://example.com/map), then slept for a week.",
     "She drew the map herself, then slept for a week."),
    ("See https://x.com/chapter/one hundred and two for the rest of it, which is long and involved.",
     "For the rest of it, which is long and involved."),
])
def test_an_address_inside_a_sentence_is_taken_out_and_the_sentence_still_reads(line, spoken):
    assert not web.is_link_line(line)
    assert web.without_web_addresses(line) == spoken


@pytest.mark.parametrize("line", [
    "Mr. Smith went to the market, e.g. for bread.", "The number one hundred four was carved over the door.",
    "She typed www into the box and waited.", "", "Hello."])
def test_text_without_an_address_is_left_exactly_as_it_is(line):
    assert web.spans(line) == [] and not web.is_link_line(line) and web.without_web_addresses(line) == line


# ---- cleanup: part of importing a book --------------------------------------------------------

SOURCE_LINES = ["Chapter 1", ROYAL_ROAD, "Mara saw the map at https://example.com/map and smiled.", "Mara walked home."]
SOURCE = ("\n".join(SOURCE_LINES) + "\n").encode("utf-8")


@pytest.fixture
def imported_book(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    text = SOURCE.decode("utf-8")

    def utterance(line_index, body):
        start = text.index(body)
        return {"text": body, "speaker": "Narrator", "utterance_type": "narration", "chapter_index": 0,
                "line_index": line_index, "start_pos": start, "end_pos": start + len(body),
                "attribution_source": "tag", "confidence": 0.99}

    payload = {"title": "Links", "author": "Test", "language": "en", "speakers": ["Mara"],
               "chapters": [{"index": 0, "title": "Chapter 1",
                             "utterances": [utterance(i, body) for i, body in enumerate(SOURCE_LINES[1:])]}]}
    monkeypatch.setattr(adapter, "availability", lambda refresh=False: adapter.Availability(
        True, "", str(Path(__file__).resolve()), adapter.PINNED_AUDIOBOOKER_VERSION))
    original = adapter.AudiobookerAdapter
    monkeypatch.setattr(autocast_import.adapter, "AudiobookerAdapter", lambda *, interpreter=None: original(
        interpreter=Path(interpreter) if interpreter else Path(__file__),
        runner=lambda request, interpreter: payload, scratch_root=tmp_path / "scratch"))
    result = autocast_import.import_autocast(SOURCE, "links.txt", "request-links", "Links")
    return result["project_id"]


def test_importing_a_book_skips_its_link_lines_and_takes_addresses_out_of_its_sentences(imported_book):
    project = load_project(imported_book)
    rows = {row.text: row for row in load_segments(imported_book)}
    link, sentence, plain = rows[ROYAL_ROAD], rows[SOURCE_LINES[2]], rows[SOURCE_LINES[3]]
    skipped = casting.skipped_lines(project)
    assert set(skipped) == {link.id}
    assert skipped[link.id]["reason"] == casting.SKIP_WEB_ADDRESS and skipped[link.id]["by"] == "cleanup"
    assert spoken_text_for_segment(project, sentence) == "Mara saw the map and smiled."
    assert sentence.text == SOURCE_LINES[2], "the source text stays exactly as imported; only what is spoken changes"
    assert spoken_text_for_segment(project, plain) == "Mara walked home."


def test_cleanup_leaves_alone_a_line_a_person_brought_back(imported_book):
    from dubbing_service.app.storage.projects import project_meta_lock, save_project
    segments = load_segments(imported_book)
    link = next(row for row in segments if row.text == ROYAL_ROAD)
    with project_meta_lock(imported_book):
        project = load_project(imported_book)
        casting.restore_skips(project, [link.id])
        assert casting.clean_web_addresses(project, segments) == ([], []), "it was brought back on purpose"
        save_project(project)
    assert link.id not in casting.skipped_lines(load_project(imported_book))


def test_the_book_panel_says_which_lines_were_left_out_for_being_links(imported_book):
    from dubbing_service.app.audiobook import autocast_book as book
    groups = book.book_status(imported_book)["skipped"]["groups"]
    assert [(group["reason"], group["label"], group["count"]) for group in groups] == [
        (casting.SKIP_WEB_ADDRESS, "Web addresses, which are not read aloud", 1)]
