"""Phase 2 identity: text edits, reorder, split and merge lineage."""

from __future__ import annotations

from dubbing_service.app.audiobook import identity
from dubbing_service.app.models.project import Segment
from tests.audiobook_helpers import build_project, build_segments, reload, use_projects_root


def test_ordinary_text_edit_preserves_segment_id(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["Hello there.", "Second line."])

    target = segments[0]
    original_id = target.id
    target.text = "Hello there, friend."
    lineage = identity.note_content_edit(target)

    assert target.id == original_id
    assert lineage.content_revision == 2
    assert lineage.retired_at is None


def test_split_retires_original_and_mints_new_ids(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["First sentence. Second sentence.", "Tail."])
    tail_id = segments[1].id

    result = identity.split_segment(
        segments, segments[0].id, parts=["First sentence.", "Second sentence."]
    )

    assert len(result.new_segments) == 2
    original = result.original
    assert original.lineage is not None
    assert original.lineage.retired_at is not None
    assert original.lineage.split_into == [segment.id for segment in result.new_segments]
    for created in result.new_segments:
        assert created.lineage is not None
        assert created.lineage.split_from == original.id
        assert created.id != original.id
    assert [segment.id for segment in identity.live_segments(segments)] == [
        result.new_segments[0].id,
        result.new_segments[1].id,
        tail_id,
    ]
    # the retired original is retained, not deleted, so its history survives
    assert original.id in {segment.id for segment in segments}


def test_split_preserves_sibling_order_and_settings(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["Alpha. Beta.", "Gamma."])
    segments[0].emotion_text = "warm"

    result = identity.split_segment(segments, segments[0].id, parts=["Alpha.", "Beta."])

    for created in result.new_segments:
        assert created.emotion_text == "warm"
        assert created.chapter_id == segments[0].chapter_id
    assert [segment.text for segment in identity.live_segments(segments)] == [
        "Alpha.",
        "Beta.",
        "Gamma.",
    ]


def test_merge_retires_sources_and_records_provenance(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["One.", "Two.", "Three."])
    first_id, second_id, third_id = (segment.id for segment in segments)

    merged = identity.merge_segments(segments, [first_id, second_id])

    assert merged.lineage is not None
    assert merged.lineage.merged_from == [first_id, second_id]
    assert merged.text.strip() == "One. Two."
    for source in segments:
        if source.id in {first_id, second_id}:
            assert source.lineage is not None
            assert source.lineage.retired_at is not None
            assert source.lineage.merged_into == merged.id
    assert [segment.id for segment in identity.live_segments(segments)] == [
        merged.id,
        third_id,
    ]


def test_reorder_utterances_keeps_ids(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project(chapter_titles=("One", "Two"))
    first = build_segments(project, ["A1", "A2"], chapter_index=0)
    second = build_segments(project, ["B1"], chapter_index=1, start_index=10)

    all_segments = first + second
    first_ids = [segment.id for segment in first]
    second_id = second[0].id
    ids_before = {segment.id for segment in all_segments}
    positions = identity.reorder_utterances(
        all_segments, [second_id, first_ids[0], first_ids[1]]
    )

    assert {segment.id for segment in all_segments} == ids_before
    assert [segment.id for segment in identity.live_segments(all_segments)] == [
        second_id,
        first_ids[0],
        first_ids[1],
    ]
    assert positions[second_id] == 0
    assert identity.find_segment(all_segments, first_ids[0]).index == 1


def test_retired_segments_are_reported_separately(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["One.", "Two."])
    identity.retire_segment(segments[1], reason="no longer in source")

    assert [segment.id for segment in identity.retired_segments(segments)] == [
        segments[1].id
    ]
    assert identity.is_live(segments[0]) is True
    assert identity.is_live(segments[1]) is False


def test_persistence_round_trip_keeps_lineage(monkeypatch, tmp_path):
    from dubbing_service.app.storage.segments import save_segments

    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["First sentence. Second sentence."])
    identity.split_segment(
        segments, segments[0].id, parts=["First sentence.", "Second sentence."]
    )
    save_segments(project.id, segments)

    _project, stored = reload(project)
    retired = [
        segment
        for segment in stored
        if segment.lineage is not None and segment.lineage.retired_at is not None
    ]
    assert len(retired) == 1
    assert len(retired[0].lineage.split_into) == 2


def test_segment_ids_are_unique():
    assert len({identity.new_segment_id() for _ in range(50)}) == 50


def test_single_segment_object_can_be_retired_directly():
    segment = Segment(
        id="seg-x", chapter_id="c1", role_id="narrator", index=0, text="Hi"
    )
    identity.retire_segment(segment, reason="duplicate")
    assert segment.lineage is not None
    assert segment.lineage.retired is True
    assert segment.lineage.retired_reason == "duplicate"


def test_split_rejects_single_or_empty_parts(monkeypatch, tmp_path):
    from dubbing_service.app.audiobook.identity import LineageError

    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["Alpha. Beta."])

    for parts in (["only one"], ["Alpha.", "   "]):
        try:
            identity.split_segment(segments, segments[0].id, parts=parts)
        except LineageError:
            continue
        raise AssertionError(f"split 应当拒绝: {parts}")


def test_merge_rejects_single_source(monkeypatch, tmp_path):
    from dubbing_service.app.audiobook.identity import LineageError

    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["One.", "Two."])
    try:
        identity.merge_segments(segments, [segments[0].id])
    except LineageError:
        return
    raise AssertionError("merge 应当拒绝单个来源")
