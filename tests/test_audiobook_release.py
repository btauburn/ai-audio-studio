"""Phase 11 + 15: the release gate and utterance-anchored audio cues.

Coverage is the single refusal point before release: it must refuse a project
whose live utterances have no usable Take, and it must say exactly why.  Cues
stay anchored to utterance identity, so a re-rendered Take that changes
duration moves them with the utterance instead of invalidating them.
"""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import (
    cues as cues_module,
    identity,
    pipeline,
    release,
    renderer,
    takes,
)
from dubbing_service.app.models.audiobook import AudioCue
from tests.audiobook_helpers import build_project, build_segments, use_projects_root


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="release")
    segments = build_segments(project, ["First utterance.", "Second utterance."])
    return root, project, segments


def _render(project, segment, *, promote: bool = True):
    rendering = renderer.FixtureRenderer(
        project_root=pipeline.project_root(project.id), renderer_id="fixture"
    )
    record = pipeline.render_segment(
        project,
        segment,
        renderer=rendering,
        root=pipeline.project_root(project.id),
        promote=promote,
    )
    assert record.success is True, record.error
    return takes.find_take(segment, record.take_id)


def _audio(project, take):
    return pipeline.project_root(project.id) / take.audio_file


def _codes(report):
    return sorted(issue.code for issue in report.issues)


def test_release_is_refused_while_an_utterance_has_no_take(studio):
    _root, project, segments = studio

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is False
    assert report.total_live == 2
    assert report.covered == 0
    assert report.missing == 2
    assert _codes(report) == [release.RESOLUTION_NO_TAKE] * 2
    assert {issue.segment_id for issue in report.issues} == {
        segment.id for segment in segments
    }


def test_release_passes_once_every_live_utterance_has_a_qa_passed_take(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segment)

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is True
    assert report.covered == 2
    assert report.missing == 0
    assert report.issues == []
    assert report.as_dict()["ready"] is True


def test_release_refuses_a_take_whose_audio_disappeared(studio):
    _root, project, segments = studio
    take = _render(project, segments[0])
    _render(project, segments[1])
    _audio(project, take).unlink()

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is False
    assert release.RESOLUTION_MISSING_AUDIO in _codes(report)


def test_release_refuses_a_take_whose_qa_report_belongs_to_other_bytes(studio):
    _root, project, segments = studio
    take = _render(project, segments[0])
    _render(project, segments[1])
    assert take.qa_report is not None
    take.qa_report.output_audio_hash = "0" * 64

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is False
    assert _codes(report) == [release.RESOLUTION_BLOCKING_QA]
    assert "different bytes" in report.issues[0].detail


def test_release_refuses_a_stale_selected_take(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segment)
    segments[0].text = "Changed after rendering."
    pipeline.refresh_staleness(project, segments)

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is False
    assert _codes(report) == [release.RESOLUTION_STALE]


def test_release_refuses_a_live_utterance_with_no_text(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segment)
    segments[1].text = "   "

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.ready is False
    assert _codes(report) == [release.RESOLUTION_ZERO_LENGTH]


def test_release_ignores_retired_utterances(studio):
    _root, project, segments = studio
    _render(project, segments[0])
    identity.retire_segment(segments[1], reason="removed by author")

    report = release.release_report(
        segments, project_root=pipeline.project_root(project.id)
    )

    assert report.total_live == 1
    assert report.ready is True


def test_a_cue_keeps_its_offsets_when_the_take_changes_duration(studio):
    _root, project, segments = studio
    _render(project, segments[0])
    cue = cues_module.make_cue(
        utterance_id=segments[0].id,
        cue_type="sfx",
        relative_offset_ms=120,
        relative_end_ms=360,
        asset_id="asset_bell",
    )

    short = cues_module.resolve_cue_window(
        cue, utterance_start_ms=0, utterance_duration_ms=800
    )
    long = cues_module.resolve_cue_window(
        cue, utterance_start_ms=0, utterance_duration_ms=4000
    )

    assert (short.start_ms, short.duration_ms) == (120, 240)
    assert (long.start_ms, long.duration_ms) == (120, 240)
    assert short.clamped is False and long.clamped is False


def test_a_cue_past_the_end_of_a_shorter_take_is_clamped_not_dropped():
    cue = cues_module.make_cue(
        utterance_id="seg_x", cue_type="bgm", relative_offset_ms=3000
    )

    window = cues_module.resolve_cue_window(
        cue, utterance_start_ms=500, utterance_duration_ms=1000
    )

    assert window.clamped is True
    assert window.start_ms == 1500
    assert window.duration_ms == 0


def test_cues_resolve_offsets_against_their_own_utterance(studio):
    _root, project, segments = studio
    cues = [
        cues_module.make_cue(
            utterance_id=segments[0].id, cue_type="sfx", relative_offset_ms=100
        ),
        cues_module.make_cue(
            utterance_id=segments[1].id, cue_type="ambience", relative_offset_ms=50
        ),
    ]
    placements = {
        segments[0].id: (0, 1000),
        segments[1].id: (1000, 2000),
    }

    windows = cues_module.resolve_cues(cues, placements)

    assert [(window.utterance_id, window.start_ms) for window in windows] == [
        (segments[0].id, 100),
        (segments[1].id, 1050),
    ]


def test_release_reports_an_orphan_cue(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segment)
    cue = cues_module.make_cue(utterance_id="seg_gone", cue_type="sfx")

    report = release.release_report(
        segments,
        cues=[cue],
        project_root=pipeline.project_root(project.id),
    )

    assert report.ready is False
    assert [issue.code for issue in report.cue_issues] == ["orphan_cue"]
    assert report.issues == []


def test_release_reports_a_cue_window_that_ends_before_it_starts(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segment)
    cue = AudioCue(
        id="cue_bad",
        utterance_id=segments[0].id,
        cue_type="sfx",
        relative_offset_ms=500,
        relative_end_ms=100,
    )

    report = release.release_report(
        segments,
        cues=[cue],
        project_root=pipeline.project_root(project.id),
    )

    assert report.ready is False
    assert [issue.code for issue in report.cue_issues] == ["invalid_cue_window"]


def test_cues_stay_out_of_the_narration_lane(studio):
    _root, _project, segments = studio
    project_cues = [
        cues_module.make_cue(utterance_id=segments[0].id, cue_type="sfx"),
        cues_module.make_cue(utterance_id=segments[0].id, cue_type="bgm"),
        cues_module.make_cue(utterance_id=segments[0].id, cue_type="ambience"),
    ]

    grouped = cues_module.lanes(project_cues)

    assert len(grouped["sfx"]) == 1
    assert len(grouped["bgm"]) == 1
    assert len(grouped["ambience"]) == 1
    assert grouped["dialogue"] == []


def test_split_moves_cues_onto_the_new_utterance_id(studio):
    _root, _project, segments = studio
    original = segments[0]
    cue = cues_module.make_cue(utterance_id=original.id, cue_type="sfx")
    project_cues = [cue]

    result = identity.split_segment(
        segments, original.id, parts=["First half.", "Second half."]
    )
    new_id = result.new_segments[1].id
    moved = cues_module.reassign_cues(
        project_cues, from_utterance_id=original.id, to_utterance_id=new_id
    )

    assert moved == 1
    assert project_cues[0].utterance_id == new_id
    assert release.check_cues(project_cues, segments) == []
    assert release.is_live(original) is False
