"""The per-line manifest guard: cheap enough for every line of a whole book, and still strict."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import pipeline
from dubbing_service.app.audiobook.fingerprint import voice_revision_for_role
from dubbing_service.app.audiobook.pipeline import RenderRecord, render_utterances
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import use_projects_root

LINES = [("line-1", "mara"), ("line-2", "joss"), ("line-3", "mara"), ("line-4", "joss")]


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = create_project("Guarded book")
    project.voice_lab.scope = "regression_fixture"
    project.roles = [Role(id=rid, name=rid.title(), role_type=RoleType.CHARACTER, tts_backend="qwen3tts")
                     for rid in ("mara", "joss")]
    state = casting._state(project)
    for role in project.roles:
        event_id = f"evt-{role.id}"
        state["roles"][role.id] = {"assignment_revision": 1, "round_revision": 0, "history": [], "round": {},
            "rejections": [], "assignment": {"key": f"key-{role.id}", "voice_revision": voice_revision_for_role(role),
                "approved": True, "rejected": False, "event_id": event_id, "stars": 5, "revision": 1}}
        state["events"][event_id] = {"actor": "brian", "origin": "SYNTHETIC", "stars": 5}
    save_project(project)
    save_segments(project.id, [Segment(id=uid, index=index, text=f"{uid} is spoken here.", role_id=rid)
                               for index, (uid, rid) in enumerate(LINES)])
    project = load_project(project.id)
    manifest = casting.freeze_manifest(project, load_segments(project.id), ["mara", "joss"])
    return SimpleNamespace(id=project.id, manifest=manifest)


def loaded(frozen):
    return load_project(frozen.id), load_segments(frozen.id)


def test_an_unchanged_book_passes_the_guard_for_every_line(frozen):
    project, segments = loaded(frozen)
    assert casting.manifest_still_current(project, segments, frozen.manifest)
    assert all(casting.manifest_still_current(project, segments, frozen.manifest, uid) for uid, _ in LINES)
    assert casting.manifest_current(frozen.id, frozen.manifest), "and the full check agrees"


def test_the_guard_reads_nothing_from_disk(frozen, monkeypatch):
    project, segments = loaded(frozen)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the per-line guard must work on state the caller already loaded")

    monkeypatch.setattr(casting, "load_project", forbidden)
    monkeypatch.setattr(casting, "load_segments", forbidden)
    assert casting.manifest_still_current(project, segments, frozen.manifest, "line-2")
    assert casting.cast_current(project, frozen.manifest)


def test_a_revoked_approval_stops_the_line(frozen):
    project, segments = loaded(frozen)
    casting._state(project)["roles"]["joss"]["assignment"].update(approved=False, rejected=True)
    assert not casting.cast_current(project, frozen.manifest)
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "line-1"), \
        "any frozen approval, not only this line's character"


def test_a_replaced_voice_stops_the_line_even_when_still_approved(frozen):
    project, segments = loaded(frozen)
    casting._state(project)["roles"]["mara"]["assignment"]["key"] = "another-voice"
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "line-1")


def test_an_edit_to_any_line_of_the_book_stops_the_run(frozen):
    project, segments = loaded(frozen)
    segments[3].text = "A different line, far from the one being recorded."
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "line-1")


def test_reassigning_a_line_to_another_character_stops_the_run(frozen):
    project, segments = loaded(frozen)
    segments[2].role_id = "joss"
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "line-1")


def test_a_change_to_this_lines_own_requirement_stops_only_that_line(frozen):
    project, segments = loaded(frozen)
    segments[1].qwen3_speaker = "Serena"
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "line-2")
    assert casting.manifest_still_current(project, segments, frozen.manifest, "line-1"), \
        "another line's requirement is checked when that line is reached, and again when its chapter is assembled"
    save_segments(frozen.id, segments)
    assert not casting.manifest_current(frozen.id, frozen.manifest), "the full check catches it wherever audio is published"


def test_a_line_outside_the_frozen_scope_is_refused(frozen):
    project, segments = loaded(frozen)
    assert not casting.manifest_still_current(project, segments, frozen.manifest, "not-in-the-manifest")


def test_a_malformed_manifest_fails_closed(frozen):
    project, segments = loaded(frozen)
    assert not casting.manifest_still_current(project, segments, {"fixture": True}, "line-1")
    assert not casting.cast_current(project, {"fixture": True})


class Engine:
    backend = engine_id = renderer_id = "qwen3tts"


def test_rendering_a_line_never_re_derives_the_whole_book(frozen, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("render_utterances re-validated the whole book for a single line")

    monkeypatch.setattr(casting, "manifest_current", forbidden)
    seen = []

    def render(_project, segment, **_kwargs):
        seen.append(segment.id)
        return RenderRecord(segment_id=segment.id, take_id="", success=False, error="stop after the guard")

    monkeypatch.setattr(pipeline, "render_segment", render)
    rows = render_utterances(frozen.id, ["line-1", "line-2"], renderer=Engine(), casting_manifest=frozen.manifest)
    assert seen == ["line-1", "line-2"] and [row.dispatched for row in rows] == [True, True]


def test_rendering_refuses_a_line_whose_book_changed(frozen, monkeypatch):
    rows = load_segments(frozen.id)
    rows[3].text = "Edited while the run was in flight."
    save_segments(frozen.id, rows)
    monkeypatch.setattr(pipeline, "render_segment", lambda *a, **k: pytest.fail("rendered against a stale manifest"))
    with pytest.raises(pipeline.PipelineError, match="no longer matches the frozen production manifest"):
        render_utterances(frozen.id, ["line-1"], renderer=Engine(), casting_manifest=frozen.manifest)
