"""Source-scope/coverage regressions; synthetic text fixtures never produce audio."""

from dataclasses import replace
from copy import deepcopy

import pytest

from dubbing_service.app.audiobook.listening_excerpt import (
    ChapterIdentity, ExcerptUtterance, ExcerptValidationError,
    blocked_source_report, build_excerpt_manifest, chapter_number,
    select_contiguous_excerpt, text_hash, utterance_from_segment,
    validate_chapter_identity, validate_rendered_excerpt, validate_source_bytes, extract_epub_chapter_text,
    adjust_excerpt_to_measured_duration,
)


@pytest.fixture
def source():
    # Deliberately artificial test prose, never a claimed book excerpt.
    lines = ["A fixture narrator mentions Alice, Bea, Charlie and Dani.",
             '"A complete fixture sentence from Alice."',
             '"A complete fixture reply from Bea."',
             "The fixture narrator supplies context between replies.",
             '"A complete fixture sentence from Charlie."',
             '"A complete fixture reply from Dani."']
    heading = "ONE: Fixture\n"
    text = heading + "\n".join(lines)
    chapter = ChapterIdentity(book_title="Super Supportive", source_identity="fixture:never-render",
                              source_revision_id="src_fixture", source_sha256="a" * 64,
                              chapter_id="chapter_fixture", chapter_number=1,
                              chapter_title="ONE: Fixture", toc_label="ONE: Fixture",
                              heading="ONE: Fixture", chapter_text_sha256=text_hash(text), imported_index=8)
    rows = []
    cursor = len(heading)
    for i, (line, role) in enumerate(zip(lines, ["narrator", "alice", "bea", "narrator", "charlie", "dani"])):
        rows.append(ExcerptUtterance(
            utterance_id=f"stable_{i}", source_revision_id=chapter.source_revision_id,
            chapter_id=chapter.chapter_id, start_offset=cursor, end_offset=cursor + len(line),
            source_text_sha256=text_hash(line), kind="narration" if role == "narrator" else "dialogue",
            role_id=role, attribution_evidence=f"fixture-source-anchor:{i}",
            estimated_duration_seconds=50, duration_basis="isolated fixture timing, no generated audio"))
        cursor += len(line) + 1
    return dict(chapter=chapter, source_text=text, utterances=rows,
                selected_utterance_ids=[r.utterance_id for r in rows],
                character_ids=["alice", "bea", "charlie", "dani"], narrator_role_id="narrator",
                full_chapter_inspected_sha256=text_hash(text), selection_rationale="Complete fixture passage.")


def assert_error(code, function, *args, **kwargs):
    with pytest.raises(ExcerptValidationError) as error:
        function(*args, **kwargs)
    assert error.value.code == code


@pytest.mark.parametrize("label,number", [
    ("ONE: Arrival", 1), ("Chapter 1: Arrival", 1), ("1. Arrival", 1),
    ("TWO HUNDRED FIFTY-SIX: A Pointy Thing on the Other", 256),
    ("TWO HUNDRED SIXTY: Snow II", 260), ("Chapter I", 1),
    ("Information", None), ("0001_Chapter.xhtml", None), ("A One-Time Thing", None),
])
def test_heading_numbers_never_infer_from_import_position(label, number):
    assert chapter_number(label) == number


def test_chapter_one_missing_is_a_durable_blocker():
    assert_error("missing_chapter", validate_chapter_identity, None)
    report = blocked_source_report(available_chapters=[260, 256, 256], inspected_paths=["local.epub"])
    assert report["available_chapters"] == [256, 260]
    assert report["audio_url"] is None and report["duration_seconds"] is None
    assert report["generation_totals"]["final_listening_seconds"] == 0


@pytest.mark.parametrize("field,value", [
    ("chapter_number", 256), ("chapter_title", "TWO HUNDRED FIFTY-SIX: Opening of subset"),
    ("toc_label", "TWO HUNDRED FIFTY-SIX: Opening of subset"),
    ("heading", "TWO HUNDRED FIFTY-SIX: Opening of subset"), ("heading", "Unknown"),
])
def test_imported_first_item_cannot_pass_as_literal_chapter_one(source, field, value):
    chapter = replace(source["chapter"], imported_index=0, **{field: value})
    assert_error("wrong_chapter", validate_chapter_identity, chapter)


def test_correct_chapter_can_have_nonzero_import_index(source):
    validate_chapter_identity(source["chapter"])
    assert_error("wrong_book", validate_chapter_identity, replace(source["chapter"], book_title="Other book"))


def test_manifest_has_exact_stable_ids_and_no_prose(source):
    result = build_excerpt_manifest(**source)
    assert result["character_ids"] == ["alice", "bea", "charlie", "dani"]
    assert result["narration_turn_count"] == 2
    assert result["utterance_ids"] == source["selected_utterance_ids"]
    assert result["start_offset"] == len("ONE: Fixture\n") and result["end_offset"] == len(source["source_text"])
    assert result["estimated_duration_seconds"] == 300
    assert result == build_excerpt_manifest(**source)
    assert "complete fixture reply" not in str(result)


def test_mentioned_characters_do_not_count_as_speakers(source):
    source["utterances"] = [replace(u, kind="narration", role_id="narrator") for u in source["utterances"]]
    assert_error("insufficient_speakers", build_excerpt_manifest, **source)


def test_aliases_do_not_manufacture_a_fourth_person(source):
    source["role_aliases"] = {"dani": "charlie"}
    assert_error("insufficient_speakers", build_excerpt_manifest, **source)


def test_transitive_aliases_resolve_to_actual_identity(source):
    source["utterances"][1] = replace(source["utterances"][1], role_id="Al")
    source["role_aliases"] = {"Al": "A", "A": "alice"}
    result = build_excerpt_manifest(**source)
    assert result["utterances"][1]["role_id"] == "alice"
    assert result["utterances"][1]["source_role_id"] == "Al"
    source["role_aliases"] = {"Al": "A", "A": "Al"}
    assert_error("alias_cycle", build_excerpt_manifest, **source)


@pytest.mark.parametrize("change", [{"speaker_resolved": False}, {"role_id": None},
                                      {"role_id": "unidentified"}, {"attribution_evidence": ""}])
def test_unresolved_dialogue_is_not_an_additional_person(source, change):
    source["utterances"][1] = replace(source["utterances"][1], **change)
    assert_error("unresolved_speaker", build_excerpt_manifest, **source)


def test_more_than_four_actual_characters_are_preserved(source):
    u = source["utterances"][0]
    source["utterances"][0] = replace(u, kind="dialogue", role_id="fifth")
    source["character_ids"].append("fifth")
    result = build_excerpt_manifest(**source)
    assert len(result["character_ids"]) == 5
    assert result["speaking_turn_counts"]["fifth"] == 1


def test_narrator_is_required_and_cannot_be_one_of_the_characters(source):
    source["utterances"] = [replace(u, kind="dialogue", role_id="alice") if u.kind == "narration" else u for u in source["utterances"]]
    assert_error("missing_narration", build_excerpt_manifest, **source)
    source["character_ids"].append("narrator")
    assert_error("invalid_narrator", build_excerpt_manifest, **source)


@pytest.mark.parametrize("duration,valid", [(285, True), (300, True), (315, True), (284.99, False), (315.01, False)])
def test_exact_estimated_duration_boundaries(source, duration, valid):
    source["utterances"] = [replace(u, estimated_duration_seconds=duration / 6) for u in source["utterances"]]
    if valid:
        assert build_excerpt_manifest(**source)["estimated_duration_seconds"] == pytest.approx(duration)
    else:
        assert_error("duration_out_of_bounds", build_excerpt_manifest, **source)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_invalid_duration_cannot_enter_manifest(source, duration):
    source["utterances"][0] = replace(source["utterances"][0], estimated_duration_seconds=duration)
    assert_error("invalid_duration", build_excerpt_manifest, **source)


def test_brief_token_speech_does_not_satisfy_meaningful_role(source):
    source["utterances"][1] = replace(source["utterances"][1], estimated_duration_seconds=1.9)
    source["utterances"][0] = replace(source["utterances"][0], estimated_duration_seconds=98.1)
    assert_error("insufficient_role_speech", build_excerpt_manifest, **source)


@pytest.mark.parametrize("ids", [["stable_0", "stable_2"], ["stable_2", "stable_1"], ["stable_0", "stable_0"]])
def test_selection_must_be_an_ordered_contiguous_slice(source, ids):
    source["selected_utterance_ids"] = ids
    assert_error("noncontiguous_selection", build_excerpt_manifest, **source)


def test_missing_interior_source_text_is_rejected_even_if_id_list_is_doctored(source):
    source["utterances"].pop(3)
    source["selected_utterance_ids"].pop(3)
    assert_error("source_gap", build_excerpt_manifest, **source)


def test_utterance_cannot_be_cut_mid_sentence(source):
    source["utterances"][2] = replace(source["utterances"][2], complete_boundary=False)
    assert_error("incomplete_boundary", build_excerpt_manifest, **source)


def test_hashes_bind_full_chapter_and_each_exact_span(source):
    original = dict(source)
    source["full_chapter_inspected_sha256"] = "b" * 64
    assert_error("full_chapter_uninspected", build_excerpt_manifest, **source)
    source = original
    source["utterances"][1] = replace(source["utterances"][1], source_text_sha256="b" * 64)
    assert_error("utterance_hash_mismatch", build_excerpt_manifest, **source)


def test_source_revision_cannot_be_mixed(source):
    source["utterances"][2] = replace(source["utterances"][2], source_revision_id="other_revision")
    assert_error("source_revision_mismatch", build_excerpt_manifest, **source)


def test_selector_reports_constraint_conflict_and_never_uses_other_chapters(source):
    options = {k: v for k, v in source.items() if k not in {"selected_utterance_ids", "selection_rationale"}}
    assert select_contiguous_excerpt(**options)["estimated_duration_seconds"] == 300
    options["role_aliases"] = {"dani": "charlie"}
    assert_error("constraint_conflict", select_contiguous_excerpt, **options)


def render_rows(manifest, duration=300):
    rows = []
    for i, u in enumerate(manifest["utterances"]):
        role = u["role_id"]
        rows.append(dict(utterance_id=u["utterance_id"], role_id=role, voice_id="voice_" + role,
                         voice_version=1, conditioning_sha256=text_hash(role), take_id="take_" + str(i),
                         audio_sha256="b" * 64, start_seconds=i * duration / 6,
                         end_seconds=(i + 1) * duration / 6))
    return rows


@pytest.mark.parametrize("duration,valid", [(285, True), (300, True), (315, True), (284.99, False), (315.01, False)])
def test_actual_decoded_duration_is_a_separate_gate(source, duration, valid):
    manifest = build_excerpt_manifest(**source)
    args = dict(audio_map=render_rows(manifest, duration), decoded_duration_seconds=duration, audio_sha256="c" * 64)
    if valid:
        result = validate_rendered_excerpt(manifest, **args)
        assert result["rendered_voice_role_count"] == 5 and result["speaking_character_count"] == 4
        assert result["technical_qa"] == "pending" and result["human_review"] == "pending"
    else:
        assert_error("duration_out_of_bounds", validate_rendered_excerpt, manifest, **args)


@pytest.mark.parametrize("change,code", [
    ("missing_line", "render_coverage"), ("wrong_speaker", "render_attribution"),
    ("padding", "render_gap"), ("overlap", "render_gap"),
    ("identity_drift", "voice_identity_drift"), ("same_conditioning", "shared_voice_conditioning"),
    ("no_take", "missing_production_identity"), ("tampered_manifest", "manifest_hash_mismatch"),
    ("trimmed_tail", "render_duration_mismatch"),
])
def test_actual_audio_coverage_and_distinct_voice_proof(source, change, code):
    manifest = build_excerpt_manifest(**source)
    rows = render_rows(manifest)
    if change == "missing_line": rows.pop(2)
    elif change == "wrong_speaker": rows[2]["role_id"] = "alice"
    elif change == "padding": rows[0]["start_seconds"] = 1
    elif change == "overlap": rows[1]["start_seconds"] = 49
    elif change == "identity_drift": rows[3]["voice_version"] = 2
    elif change == "same_conditioning": rows[1]["conditioning_sha256"] = rows[0]["conditioning_sha256"]
    elif change == "no_take": rows[0]["take_id"] = None
    elif change == "tampered_manifest": manifest["end_offset"] -= 1
    elif change == "trimmed_tail": rows[-1]["end_seconds"] -= 1
    assert_error(code, validate_rendered_excerpt, manifest, audio_map=rows,
                 decoded_duration_seconds=300, audio_sha256="c" * 64)


def test_segment_adapter_preserves_provenance_and_explicit_attribution(source):
    from dubbing_service.app.models.project import Segment
    from dubbing_service.app.models.audiobook import SourceAnchor, Attribution
    segment = Segment(id="canonical-id", chapter_id="chapter_fixture", role_id="alice", text="Fixture.",
                      source_anchor=SourceAnchor(source_revision_id="src_fixture", start_offset=10, end_offset=18),
                      attribution=Attribution(speaker_id="alice", source="tag"))
    row = utterance_from_segment(segment, kind="dialogue", attribution_evidence="anchored speech tag",
                                 estimated_duration_seconds=4, duration_basis="measured local role timing")
    assert row.utterance_id == segment.id and row.source_revision_id == "src_fixture"
    assert row.start_offset == 10 and row.speaker_resolved
    segment.attribution.speaker_id = "bea"
    assert_error("attribution_mismatch", utterance_from_segment, segment, kind="dialogue",
                 attribution_evidence="conflict", estimated_duration_seconds=4, duration_basis="measured")


def test_preflight_reads_epub_metadata_not_member_number(tmp_path):
    import zipfile
    from scripts.voice_lab_source_preflight import inspect_epub, inspect_roots
    path = tmp_path / "arbitrary-name.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfile full-path="book.opf"/></container>')
        archive.writestr("book.opf", '<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Super Supportive</dc:title></metadata><manifest><item id="n" href="toc.ncx" media-type="application/x-dtbncx+xml"/><item id="c" href="0001.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>')
        archive.writestr("toc.ncx", '<ncx><navPoint><navLabel><text>TWO HUNDRED FIFTY-SIX: Fixture</text></navLabel><content src="0001.xhtml"/></navPoint></ncx>')
        archive.writestr("0001.xhtml", '<html><h1>TWO HUNDRED FIFTY-SIX: Fixture</h1><p>Private fixture prose is never exported.</p></html>')
    evidence = inspect_epub(path)
    assert evidence["available_chapters"] == [256]
    assert not evidence["chapter_one_toc_and_heading"]
    assert "Private fixture prose" not in str(evidence)
    inventory = inspect_roots([tmp_path])
    assert inventory["excerpt"]["status"] == "blocked_source"
    assert inventory["available_chapters"] == [256]


def test_declared_first_chapter_is_cross_checked_against_original_epub_bytes(source, tmp_path):
    import zipfile
    import hashlib
    path = tmp_path / "arbitrary-name.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfile full-path="book.opf"/></container>')
        archive.writestr("book.opf", '<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Super Supportive</dc:title></metadata><manifest><item id="n" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest></package>')
        archive.writestr("toc.ncx", '<ncx><navPoint><navLabel><text>TWO HUNDRED FIFTY-SIX: Fixture</text></navLabel><content src="0001.xhtml"/></navPoint></ncx>')
        archive.writestr("0001.xhtml", '<html><body><h1>TWO HUNDRED FIFTY-SIX: Fixture</h1><p>Actual complete fixture chapter text.</p></body></html>')
    data = path.read_bytes()
    forged = replace(source["chapter"], source_sha256=hashlib.sha256(data).hexdigest())
    assert_error("wrong_chapter", validate_source_bytes, forged, data)
    # A genuine local source has mutually agreeing TOC and matching document
    # heading, independent of internal filename/order.
    actual = tmp_path / "actual-fixture.epub"
    with zipfile.ZipFile(actual, "w") as archive, zipfile.ZipFile(path) as old:
        for name in old.namelist():
            archive.writestr(name, old.read(name).replace(b"TWO HUNDRED FIFTY-SIX", b"ONE"))
    data = actual.read_bytes()
    actual_text = extract_epub_chapter_text(data, member="0001.xhtml")
    valid = replace(source["chapter"], source_sha256=hashlib.sha256(data).hexdigest(),
                    source_member="0001.xhtml", extraction_method="epub-xhtml-body-text-v1",
                    chapter_text_sha256=text_hash(actual_text))
    validate_source_bytes(valid, data)
    assert_error("source_extraction_mismatch", validate_source_bytes,
                 replace(valid, chapter_text_sha256=text_hash("Entirely different invented prose.")), data)
    assert_error("source_member_mismatch", validate_source_bytes, replace(valid, source_member="other.xhtml"), data)
    assert_error("unverified_extraction", validate_source_bytes, replace(valid, extraction_method="unverified"), data)


@pytest.mark.parametrize("text", ["Chapter 256\nThis is a later chapter, not Chapter 1.",
                                   "Unknown chapter\nNo verified heading is available.",
                                   "Chapter 256\nChapter 1\nA later mention cannot override the first chapter heading."])
def test_plain_text_cannot_bypass_actual_chapter_heading(source, text):
    identity = replace(source["chapter"], source_sha256=text_hash(text), chapter_text_sha256=text_hash(text))
    assert_error("wrong_chapter", validate_source_bytes, identity, text.encode())


def test_plain_text_requires_matching_real_heading(source):
    text = source["source_text"]
    identity = replace(source["chapter"], source_sha256=text_hash(text))
    validate_source_bytes(identity, text.encode())


def test_existing_narration_must_be_interleaved_not_just_an_intro(source):
    source["utterances"][3] = replace(source["utterances"][3], kind="dialogue", role_id="alice")
    assert_error("missing_interleaved_narration", build_excerpt_manifest, **source)


@pytest.mark.parametrize("offset", [-1, 0.5, True])
def test_offsets_are_exact_integer_codepoints(source, offset):
    source["utterances"][0] = replace(source["utterances"][0], start_offset=offset)
    assert_error("invalid_offsets", build_excerpt_manifest, **source)


def test_actual_speech_from_each_voice_must_be_audible_long_enough(source):
    manifest = build_excerpt_manifest(**source)
    rows = render_rows(manifest)
    rows[0]["end_seconds"] = rows[1]["start_seconds"] = 99
    assert_error("insufficient_role_speech", validate_rendered_excerpt, manifest, audio_map=rows,
                 decoded_duration_seconds=300, audio_sha256="c" * 64)


def test_measured_timing_can_trim_only_complete_edge_utterances(source):
    manifest = build_excerpt_manifest(**source)
    actual = {row.utterance_id: 30 if i == 0 else 60 for i, row in enumerate(source["utterances"])}
    adjusted = adjust_excerpt_to_measured_duration(manifest, source_text=source["source_text"], measured_utterance_seconds=actual)
    assert adjusted["utterance_ids"] == manifest["utterance_ids"][1:]
    assert adjusted["character_ids"] == manifest["character_ids"]
    assert adjusted["estimated_duration_seconds"] == 300
    assert adjusted["start_offset"] == source["utterances"][1].start_offset


def test_measured_timing_can_extend_only_adjacent_full_source(source):
    before = source["source_text"]
    tail = "A complete closing fixture narration paragraph."
    source["source_text"] = before + "\n" + tail
    source["chapter"] = replace(source["chapter"], chapter_text_sha256=text_hash(source["source_text"]))
    source["full_chapter_inspected_sha256"] = text_hash(source["source_text"])
    source["utterances"].append(replace(source["utterances"][0], utterance_id="stable_tail", start_offset=len(before) + 1,
                                         end_offset=len(source["source_text"]), source_text_sha256=text_hash(tail),
                                         estimated_duration_seconds=25))
    manifest = build_excerpt_manifest(**source)
    actual = {i: 45 for i in manifest["utterance_ids"]}
    adjusted = adjust_excerpt_to_measured_duration(manifest, source_text=source["source_text"], measured_utterance_seconds=actual)
    assert adjusted["utterance_ids"] == [*manifest["utterance_ids"], "stable_tail"]
    assert adjusted["character_ids"] == manifest["character_ids"]
    assert adjusted["estimated_duration_seconds"] == pytest.approx(292.5)


def test_timing_adjustment_cannot_drop_a_required_character_or_splice_scene(source):
    manifest = build_excerpt_manifest(**source)
    actual = {i: 60 for i in manifest["utterance_ids"]}
    # Trimming the initial narration gives 300s and is permitted; trimming the
    # terminal character would also give 300s but must never be selected.
    adjusted = adjust_excerpt_to_measured_duration(manifest, source_text=source["source_text"], measured_utterance_seconds=actual)
    assert adjusted["utterance_ids"][-1] == manifest["utterance_ids"][-1]
    actual = {i: 70 for i in manifest["utterance_ids"]}
    assert_error("constraint_conflict", adjust_excerpt_to_measured_duration, manifest,
                 source_text=source["source_text"], measured_utterance_seconds=actual)


def test_unbounded_neighbor_repair_is_refused(source):
    manifest = build_excerpt_manifest(**source)
    assert_error("adjustment_limit", adjust_excerpt_to_measured_duration, manifest, source_text=source["source_text"],
                 measured_utterance_seconds={i: 45 for i in manifest["utterance_ids"]}, max_neighbor_utterances=100)


def test_preflight_reports_bounded_search_and_missing_roots(tmp_path):
    from scripts.voice_lab_source_preflight import inspect_roots
    (tmp_path / "one.txt").write_text("fixture")
    evidence = inspect_roots([tmp_path / "absent", tmp_path], max_files=0)
    assert evidence["search_complete"] is False
    assert evidence["searched_roots"][0]["exists"] is False
    assert evidence["errors"][0]["error"] == "file_count_limit"
