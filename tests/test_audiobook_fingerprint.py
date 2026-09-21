"""Phase 5/6: the synthesis requirement fingerprint and staleness rules."""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import engines, fingerprint, takes
from dubbing_service.app.models.enums import EmotionMode
from tests.audiobook_helpers import build_project, build_segments, use_projects_root


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["First sentence.", "Second sentence."])
    return root, project, segments


def requirement(project, segment, role=None, **overrides):
    params = {
        "engine_id": engines.default_engine_id(),
        "role": role,
        "renderer_id": "fixture",
    }
    params.update(overrides)
    return fingerprint.requirement_for_segment(segment, project_id=project.id, **params)


def test_requirement_is_stable_for_unchanged_input(studio):
    _root, project, segments = studio
    first = requirement(project, segments[0])
    second = requirement(project, segments[0])
    assert first.fingerprint() == second.fingerprint()
    assert first.components() == second.components()


def test_text_edit_changes_the_fingerprint(studio):
    _root, project, segments = studio
    before = requirement(project, segments[0])
    segments[0].text = "First sentence, revised."
    after = requirement(project, segments[0])
    assert before.fingerprint() != after.fingerprint()
    assert fingerprint.changed_components(before.components(), after.components()) == [
        "spoken_text_hash"
    ]


def test_emotion_and_speed_changes_change_the_fingerprint(studio):
    _root, project, segments = studio
    base = requirement(project, segments[0])

    segments[0].emotion_mode = EmotionMode.TEXT
    segments[0].emotion_text = "warm"
    emotional = requirement(project, segments[0])
    assert emotional.fingerprint() != base.fingerprint()
    assert "emotion_text" in fingerprint.changed_components(
        base.components(), emotional.components()
    )

    segments[0].playback_speed = 1.25
    faster = requirement(project, segments[0])
    assert faster.fingerprint() != emotional.fingerprint()
    assert "speed" in fingerprint.changed_components(
        emotional.components(), faster.components()
    )


def test_voice_change_changes_the_fingerprint(studio):
    _root, project, segments = studio
    role = project.roles[0]
    before = requirement(project, segments[0], role=role)

    role.qwen3_speaker = "Maria"
    role.voice_description = "a deeper narrated voice"
    after = requirement(project, segments[0], role=role)

    assert before.fingerprint() != after.fingerprint()
    assert "voice_revision" in fingerprint.changed_components(
        before.components(), after.components()
    )


def test_renderer_and_model_changes_change_the_fingerprint(studio):
    _root, project, segments = studio
    base = requirement(project, segments[0])

    other_renderer = requirement(project, segments[0], renderer_id="yovoice")
    assert base.fingerprint() != other_renderer.fingerprint()
    assert fingerprint.changed_components(
        base.components(), other_renderer.components()
    ) == ["renderer_id"]

    newer_model = requirement(
        project,
        segments[0],
        engine_components={"model_id": "indextts", "model_revision": "2024-10-01"},
    )
    assert newer_model.fingerprint() != base.fingerprint()
    changed = fingerprint.changed_components(base.components(), newer_model.components())
    assert set(changed) == {"model_id", "model_revision"}


def test_role_identity_fields_do_not_change_the_fingerprint(studio):
    _root, project, segments = studio
    role = project.roles[0]
    before = requirement(project, segments[0], role=role)
    role.name = "Narrator (renamed)"
    role.description = "purely cosmetic"
    after = requirement(project, segments[0], role=role)
    assert before.fingerprint() == after.fingerprint()


def test_non_audio_metadata_is_never_part_of_the_requirement(studio):
    _root, project, segments = studio
    components = set(requirement(project, segments[0]).components())
    assert components.isdisjoint(fingerprint.NON_AUDIO_METADATA_FIELDS)

    before = requirement(project, segments[0])
    project.name = "renamed book"
    project.description = "a new blurb"
    project.metadata["book_title"] = "A Title"
    project.metadata["cover_art"] = "cover.png"
    after = requirement(project, segments[0])
    assert before.fingerprint() == after.fingerprint()


def test_one_requirement_can_be_rendered_into_several_takes(studio):
    _root, project, segments = studio
    segment = segments[0]
    live = requirement(project, segment)

    first = takes.build_take(
        segment, slot="a", take_id="take_a", audio_file="a.wav", audio_sha256="aaa",
        requirement=live,
    )
    second = takes.build_take(
        segment, slot="b", take_id="take_b", audio_file="b.wav", audio_sha256="bbb",
        requirement=live,
    )

    assert first.input_fingerprint == second.input_fingerprint
    assert first.audio_sha256 != second.audio_sha256


def test_stale_take_reports_the_changed_components_and_cannot_be_selected(studio):
    _root, project, segments = studio
    segment = segments[0]
    old = requirement(project, segment)
    take = takes.build_take(
        segment, slot="a", take_id="take_old", audio_file="a.wav", audio_sha256="aaa",
        requirement=old,
    )
    takes.attach_take(segment, take, slot="a", promote=True)
    assert segment.selected_take_id == "take_old"

    segment.text = "First sentence, revised."
    live = requirement(project, segment)
    is_stale, changed = takes.take_is_stale(take, live)
    assert is_stale is True
    assert changed == ["spoken_text_hash"]

    stale_ids = takes.mark_stale_takes(segment, live)
    assert stale_ids == ["take_old"]
    assert take.stale is True

    with pytest.raises(takes.PromotionError):
        takes.promote_take(segment, "take_old", requirement=live)

    # the previous selection is preserved: promotion only ever moves a pointer
    assert segment.selected_take_id == "take_old"
    assert takes.find_take(segment, "take_old").audio_sha256 == "aaa"


def test_legacy_take_without_a_stamp_is_not_declared_stale(studio):
    _root, project, segments = studio
    segment = segments[0]
    legacy = takes.build_take(
        segment, slot="a", take_id="take_legacy", audio_file="a.wav",
        audio_sha256="aaa", requirement=requirement(project, segment),
    )
    legacy.input_fingerprint = ""
    legacy.fingerprint_components = {}
    is_stale, changed = takes.take_is_stale(legacy, requirement(project, segment))
    assert (is_stale, changed) == (False, [])


def test_requirement_for_segment_uses_the_spoken_text_revision(studio):
    _root, project, segments = studio
    from dubbing_service.app.audiobook import provenance

    segment = segments[0]
    before = requirement(project, segment)
    revision = provenance.set_spoken_text(
        project, segment, spoken_text="First sentance."
    )
    after = requirement(
        project, segment, spoken_text_hash=revision.spoken_text_hash
    )
    assert before.fingerprint() != after.fingerprint()
    assert after.components()["spoken_text_hash"] == revision.spoken_text_hash
