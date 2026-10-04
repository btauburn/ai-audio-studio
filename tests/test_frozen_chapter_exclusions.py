"""Governed chapter derivation with existing metadata/signal fixtures only."""
import copy
import json
from types import SimpleNamespace

import pytest

from tests.test_voice_lab_chapter import source, fork, ready, run, mock_pipeline
from tests.test_voice_lab_chapter_export import chapter as export_chapter_fixture
from dubbing_service.app.audiobook import voice_lab_chapter as chapter
from dubbing_service.app.audiobook import voice_lab_chapter_export as export
from dubbing_service.app.audiobook import casting_review
from dubbing_service.app.models.audiobook import SourceAnchor, SpokenTextRevision
from dubbing_service.app.models.project import Segment
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments


def governed_source(source):
    project = load_project(source[0].id)
    rows = load_segments(project.id)
    text = "https://example.test/book/chapter"
    spoken = SpokenTextRevision(id="spoken_url", source_revision_id="source1",
        source_text_hash=chapter._text_hash(text), spoken_text=text,
        spoken_text_hash=chapter._text_hash(text))
    project.spoken_text_revisions.append(spoken)
    rows.append(Segment(id="url", chapter_id="chap1", index=3, text=text, role_id="narrator",
        source_anchor=SourceAnchor(source_revision_id="source1", chapter_id="chap1", occurrence_index=3),
        spoken_text_revision_id=spoken.id))
    casting_review.record_skips(project, ["seg0"], reason="human", detail="Existing reader choice", by="you")
    casting_review.record_skips(project, ["url"], reason="web_address",
        detail=casting_review.WEB_ADDRESS_DETAIL, by="cleanup")
    save_project(project)
    save_segments(project.id, rows)
    return project


def test_fork_retains_all_source_and_governed_decisions_without_touching_originals(source):
    original = governed_source(source)
    base = source[1] / original.id
    before = {path.name: path.read_bytes() for path in base.iterdir() if path.is_file()}
    result = fork(source, request_id="governed")
    target = load_project(result["project_id"])
    rows = load_segments(target.id)
    meta = target.metadata["chapter_rerun"]
    assert [row.id for row in rows] == ["seg0", "seg1", "seg2", "url"]
    assert meta["source_utterance_ids"] == [row.id for row in rows]
    assert meta["speech_utterance_ids"] == ["seg2"]
    assert result["excluded_utterance_count"] == 2
    assert result["scene_break_count"] == 1 and result["speech_utterance_count"] == 1
    assert meta["expected_source_hash"] == chapter._hash(chapter._source_proof(target, meta["source_utterances"]))
    assert casting_review.skipped_lines(target) == casting_review.skipped_lines(original)
    assert not target.metadata["autocast_import"].get("mode")
    receipt = meta["governed_exclusions"]
    assert (source[1] / target.id / receipt["source_project_metadata_file"]).read_bytes() == before["project.json"]
    assert receipt["source_segments_sha256"] == chapter.hashlib.sha256(before["segments.json"]).hexdigest()
    assert before == {path.name: path.read_bytes() for path in base.iterdir() if path.is_file()}
    assert all(not row.takes and row.selected_take_id is None for row in rows)
    assert fork(source, request_id="governed")["reused"]
    original.metadata["autocast_import"]["skipped"]["seg0"]["detail"] = "Different decision"
    save_project(original)
    with pytest.raises(chapter.ChapterRenditionError, match="changed source"):
        fork(source, request_id="governed")


def test_read_source_scope_before_cast_and_default_render_gate_stays_strict(source):
    governed_source(source)
    result = fork(source)
    project, rows, binding = chapter.source_scope(result["project_id"], "chap1")
    assert [row.id for row in rows] == ["seg2"]
    assert binding["identities"] == binding["requirements"] == {}
    with pytest.raises(chapter.ChapterRenditionError, match="locked, ready"):
        chapter._bound_state(project.id, "chap1")
    current = load_segments(project.id)
    next(row for row in current if row.id == "seg2").text = "Changed source after fork"
    save_segments(project.id, current)
    for verify_cast in [False, True]:
        with pytest.raises(chapter.ChapterRenditionError, match="source text"):
            chapter._bound_state(project.id, "chap1", verify_cast=verify_cast)


def test_render_dispatches_only_frozen_speech_and_preserves_excluded_raw_rows(source, monkeypatch):
    governed_source(source)
    project = ready(source)
    calls, generated = mock_pipeline(monkeypatch)
    result = run(project)
    assert result["job"]["status"] == "succeeded"
    assert calls == generated == ["seg2"]
    rows = load_segments(project.id)
    assert [row.id for row in rows] == ["seg0", "seg1", "seg2", "url"]
    assert all(row.selected_take_id is None for row in rows if row.id in {"seg0", "url"})


@pytest.mark.parametrize("change", ["current_decision", "frozen_row", "arbitrary_skip", "archive_bytes"])
def test_changed_exclusion_evidence_refuses_initial_scope(source, change):
    governed_source(source)
    result = fork(source)
    target = load_project(result["project_id"])
    receipt = target.metadata["chapter_rerun"]["governed_exclusions"]
    if change == "current_decision":
        target.metadata["autocast_import"]["skipped"]["seg0"]["detail"] = "Changed"
    elif change == "frozen_row":
        receipt["decisions"][0]["source_text_sha256"] = "0" * 64
        target.metadata["chapter_rerun"]["expected_exclusions_hash"] = chapter._hash(receipt)
    elif change == "arbitrary_skip":
        target.metadata["autocast_import"]["skipped"]["seg2"] = copy.deepcopy(receipt["decisions"][0]["decision"])
    else:
        path = source[1] / target.id / receipt["source_project_metadata_file"]
        path.write_bytes(path.read_bytes() + b" ")
    save_project(target)
    with pytest.raises(ValueError):
        chapter.source_scope(target.id, "chap1")


def test_per_line_guard_checks_current_decisions_without_reparsing_large_archive(source, monkeypatch):
    governed_source(source)
    target = ready(source)
    original_read = chapter._read_verified
    def read(path, expected):
        assert "source-exclusions" not in str(path)
        return original_read(path, expected)
    monkeypatch.setattr(chapter, "_read_verified", read)
    _, rows, _ = chapter._bound_state(target.id, "chap1", verify_files=False)
    assert [row.id for row in rows] == ["seg2"]
    target.metadata["autocast_import"]["skipped"]["seg0"]["detail"] = "Concurrent scope change"
    save_project(target)
    with pytest.raises(ValueError, match="exclusion decisions changed"):
        chapter._bound_state(target.id, "chap1", verify_files=False)


def test_unreceipted_skip_claim_cannot_shrink_legacy_derived_scope(source):
    result = fork(source)
    target = load_project(result["project_id"])
    casting_review.record_skips(target, ["seg2"], reason="human", detail="No source decision", by="you")
    save_project(target)
    with pytest.raises(ValueError, match="frozen source decision evidence"):
        chapter.source_scope(target.id, "chap1")


@pytest.mark.parametrize("reason, text", [("failed_qa", "Fixture narration."), ("web_address", "Fixture narration.")])
def test_fork_does_not_infer_new_exclusion_authority(source, reason, text):
    original = load_project(source[0].id)
    casting_review.record_skips(original, ["seg0"], reason=reason, detail="Not eligible for carry", by="you")
    save_project(original)
    with pytest.raises(ValueError):
        fork(source)
    assert len(list(source[1].glob("*/project.json"))) == 1


def add_export_exclusions(export_chapter_fixture):
    project_id, base, _ = export_chapter_fixture
    project = load_project(project_id)
    rows = load_segments(project_id)
    # This metadata-only fixture creates source evidence before its derivation;
    # the existing selected audio of every remaining line is unchanged.
    by_id = {row.id: row for row in rows}
    by_id["seg3"].text = "https://example.test/skipped-link"
    decisions = {
        "seg2": {"reason": "human", "detail": "Source choice", "by": "you", "at": "2026-10-04T00:00:00Z"},
        "seg3": {"reason": "web_address", "detail": casting_review.WEB_ADDRESS_DETAIL,
                 "by": "cleanup", "at": "2026-10-04T00:00:00Z"},
    }
    original = project.model_dump(mode="json")
    original["id"] = project.metadata["chapter_rerun"]["source_project_id"]
    original["metadata"] = {"autocast_import": {"skipped": decisions}}
    raw = json.dumps(original, sort_keys=True).encode()
    checksum = chapter.hashlib.sha256(raw).hexdigest()
    relative = f"chapter_rerun/source-exclusions/source-project-{checksum}.json"
    path = base / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    meta = project.metadata["chapter_rerun"]
    meta["source_utterances"] = [chapter._source_row(row) for row in rows]
    meta["expected_source_hash"] = chapter._hash(chapter._source_proof(project, meta["source_utterances"]))
    project.metadata["autocast_import"] = {"skipped": decisions}
    receipt = {"schema": "chapter-governed-exclusions-v1",
        "source_project_id": original["id"], "source_project_metadata_file": relative,
        "source_project_metadata_sha256": checksum, "source_segments_sha256": "1" * 64,
        "decisions": chapter._source_exclusions(SimpleNamespace(metadata=original["metadata"]), rows)}
    meta["governed_exclusions"] = receipt
    meta["expected_exclusions_hash"] = chapter._hash(receipt)
    meta["speech_utterance_ids"] = [row.id for row in rows if row.id not in {"seg1", "seg2", "seg3"}]
    for row in rows:
        if row.id in decisions:
            row.takes, row.selected_take_id, row.audio_file = [], None, None
    save_project(project)
    save_segments(project.id, rows)
    return project


def test_export_accounts_for_every_source_row_without_demanding_excluded_takes(export_chapter_fixture):
    project = add_export_exclusions(export_chapter_fixture)
    before = {path: path.read_bytes() for path in (export_chapter_fixture[1] / "audio").glob("*.wav")}
    result = export.export_chapter(project.id, "chap0")
    assert result["speech_utterance_count"] == 2 and result["excluded_utterance_count"] == 2
    assert result["scene_break_count"] == 1 and result["duration_seconds"] == 2.25
    assert result["utterance_ids"] == ["seg0", "seg1", "seg2", "seg3", "seg4"]
    assert result["speech_utterance_ids"] == ["seg0", "seg4"]
    assert [row["kind"] for row in result["map"]] == ["speech", "scene_break", "excluded", "excluded", "speech"]
    assert all(row["start_seconds"] == row["end_seconds"] and row["decision"]["reason"]
               for row in result["map"] if row["kind"] == "excluded")
    assert [row.id for row in load_segments(project.id)] == result["utterance_ids"]
    assert before == {path: path.read_bytes() for path in before}
    assert result["human_review"] == "pending" and result["generation_totals"]["tts_generated_by_export"] == 0
    assert export.export_chapter(project.id, "chap0")["audio_sha256"] == result["audio_sha256"]


@pytest.mark.parametrize("change", ["decision", "archive"])
def test_export_rechecks_frozen_exclusions_and_archive_before_delivery(export_chapter_fixture, change):
    project = add_export_exclusions(export_chapter_fixture)
    if change == "decision":
        project.metadata["autocast_import"]["skipped"]["seg3"]["reason"] = "human"
        save_project(project)
    else:
        receipt = project.metadata["chapter_rerun"]["governed_exclusions"]
        path = export_chapter_fixture[1] / receipt["source_project_metadata_file"]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        export.export_chapter(project.id, "chap0")
    assert not list(export_chapter_fixture[1].glob("voice_lab/chapters/**/export.json"))
