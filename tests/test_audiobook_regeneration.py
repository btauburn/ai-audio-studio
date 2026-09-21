"""Phase 14: dependency-aware regeneration, one regression case per dependency."""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import cues, identity, regeneration, takes
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

NO_RENDER_CASES = (
    "book_title",
    "author",
    "cover_art",
    "description",
    "review_note",
    "ui_state",
    "project_name",
)
REMIX_ONLY_CASES = ("music_level", "cue_timing", "cue_lane")
PROJECT_WIDE_CASES = ("renderer", "model")
UTTERANCE_CASES = (
    "spoken_text",
    "pronunciation",
    "voice",
    "reference_audio",
    "emotion",
    "speed",
    "attribution",
)


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["One.", "Two.", "Three."])
    return root, project, segments


@pytest.mark.parametrize("kind", NO_RENDER_CASES)
def test_metadata_changes_never_regenerate_audio(studio, kind):
    _root, project, segments = studio
    plan = regeneration.plan_regeneration(kind, project=project, segments=segments)
    assert plan.classification == regeneration.NO_RENDER
    assert plan.requires_render is False
    assert plan.remix_only is False
    assert plan.utterance_ids == []
    assert plan.as_dict()["kind"] == kind


@pytest.mark.parametrize("kind", REMIX_ONLY_CASES)
def test_mix_changes_reuse_every_take(studio, kind):
    _root, project, segments = studio
    plan = regeneration.plan_regeneration(kind, project=project, segments=segments)
    assert plan.classification == regeneration.REMIX_ONLY
    assert plan.remix_only is True
    assert plan.requires_render is False
    assert plan.utterance_ids == []
    assert any("Take" in note for note in plan.notes)


def test_music_level_counts_only_bgm_and_ambience_cues(studio):
    _root, project, segments = studio
    cue_list = [
        cues.make_cue(utterance_id=segments[0].id, cue_type="bgm", asset_id="a1"),
        cues.make_cue(utterance_id=segments[1].id, cue_type="ambience", asset_id="a2"),
        cues.make_cue(utterance_id=segments[1].id, cue_type="sfx", asset_id="a3"),
    ]
    music = regeneration.plan_regeneration(
        "music_level", project=project, segments=segments, cues=cue_list
    )
    assert any("cues touched: 2" in note for note in music.notes)

    timing = regeneration.plan_regeneration(
        "cue_timing", project=project, segments=segments, cues=cue_list
    )
    assert any("cues touched: 3" in note for note in timing.notes)


@pytest.mark.parametrize("kind", UTTERANCE_CASES)
def test_utterance_scoped_changes_touch_only_named_utterances(studio, kind):
    _root, project, segments = studio
    target = segments[1].id
    plan = regeneration.plan_regeneration(
        kind, project=project, segments=segments, utterance_ids=[target]
    )
    assert plan.classification == regeneration.UTTERANCE_RENDER
    assert plan.utterance_ids == [target]
    assert plan.requires_render is True
    assert plan.notes == []


@pytest.mark.parametrize("kind", PROJECT_WIDE_CASES)
def test_project_wide_changes_ignore_the_requested_subset(studio, kind):
    _root, project, segments = studio
    plan = regeneration.plan_regeneration(
        kind,
        project=project,
        segments=segments,
        utterance_ids=[segments[0].id],
    )
    assert plan.utterance_ids == [segment.id for segment in segments]
    assert plan.requires_render is True


def test_retired_and_unknown_utterances_are_never_scheduled(studio):
    _root, project, segments = studio
    retired = identity.retire_segment(segments[2], reason="removed from source")
    assert retired is not None

    scoped = regeneration.plan_regeneration(
        "spoken_text",
        project=project,
        segments=segments,
        utterance_ids=[segments[0].id, segments[2].id, "seg_missing"],
    )
    assert scoped.utterance_ids == [segments[0].id]
    assert any("seg_missing" in note for note in scoped.notes)

    wide = regeneration.plan_regeneration("model", project=project, segments=segments)
    assert wide.utterance_ids == [segments[0].id, segments[1].id]


def test_utterances_without_takes_are_reported_separately(studio):
    _root, project, segments = studio
    live = fingerprint_requirement(project, segments[0])
    take = takes.build_take(
        segments[0],
        slot="a",
        take_id="take_a",
        audio_file="a.wav",
        audio_sha256="aaa",
        requirement=live,
    )
    takes.attach_take(segments[0], take, slot="a")

    missing = regeneration.utterance_ids_needing_render(
        segments, {"seg000": [take], "seg001": [], "seg002": []}
    )
    assert missing == [segments[1].id, segments[2].id]


def test_stale_utterances_use_the_single_staleness_rule(studio):
    _root, project, segments = studio
    live = fingerprint_requirement(project, segments[0])
    take = takes.build_take(
        segments[0],
        slot="a",
        take_id="take_a",
        audio_file="a.wav",
        audio_sha256="aaa",
        requirement=live,
    )
    takes.attach_take(segments[0], take, slot="a")

    segments[0].text = "One, edited."
    stale = regeneration.stale_utterance_ids(
        segments,
        requirement_for=lambda segment: (
            fingerprint_requirement(project, segment)
            if segment.takes
            else None
        ),
    )
    assert stale == [segments[0].id]


def test_unknown_change_kind_is_rejected(studio):
    _root, project, segments = studio
    with pytest.raises(KeyError):
        regeneration.plan_regeneration("quantum_entanglement", project=project)


def test_every_documented_change_kind_has_a_classification():
    documented = set(regeneration.CHANGE_KINDS)
    assert documented >= set(NO_RENDER_CASES) | set(REMIX_ONLY_CASES) | set(
        PROJECT_WIDE_CASES
    ) | set(UTTERANCE_CASES)
    for kind in documented:
        assert regeneration.classify_change(kind) in {
            regeneration.NO_RENDER,
            regeneration.REMIX_ONLY,
            regeneration.UTTERANCE_RENDER,
        }


def fingerprint_requirement(project, segment):
    from dubbing_service.app.audiobook import engines, fingerprint

    return fingerprint.requirement_for_segment(
        segment,
        project_id=project.id,
        engine_id=engines.default_engine_id(),
        renderer_id="fixture",
    )
