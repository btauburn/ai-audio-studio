"""Pipeline authorization boundary regressions using disposable synthetic projects."""
from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import casting_review, pipeline
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import use_projects_root


class FakeRenderer:
    backend = "fixture-renderer"
    engine_id = "fixture-engine"
    renderer_id = "fixture-renderer"


@pytest.fixture
def project(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    item = create_project("Synthetic strict casting boundary")
    item.roles = [Role(id="speaker", name="Speaker", tts_backend="fixture-engine")]
    save_project(item)
    save_segments(item.id, [Segment(id="line", index=0, text="Synthetic test line.", role_id="speaker")])
    return item.id


def failed_render(project_id, calls, **kwargs):
    calls.append(kwargs.get("role"))
    return pipeline.RenderRecord(segment_id="line", take_id="", success=False, error="fixture stopped before synthesis")


def test_legacy_render_path_remains_available_for_unflagged_projects(project, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "render_segment", lambda p, s, **kw: failed_render(p, calls, **kw))
    rows = pipeline.render_utterances(project, ["line"], renderer=FakeRenderer(), promote=True)
    assert len(rows) == 1 and rows[0].dispatched
    assert len(calls) == 1


@pytest.mark.parametrize("flag", ["enabled", "autocast"])
def test_strict_or_autocast_project_cannot_render_without_manifest(project, monkeypatch, flag):
    item = load_project(project)
    if flag == "enabled":
        item.voice_lab.casting["enabled"] = True
    else:
        item.metadata["autocast_import"] = {"mode": "autocast"}
    save_project(item)
    calls = []
    monkeypatch.setattr(pipeline, "render_segment", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(pipeline.PipelineError, match="current approved cast manifest"):
        pipeline.render_utterances(project, ["line"], renderer=FakeRenderer())
    assert calls == []


def test_audition_child_is_exempt_from_parent_strict_casting_gate(project, monkeypatch):
    item = load_project(project)
    item.voice_lab.casting["enabled"] = True
    item.metadata["voice_lab_audition"] = {"parent_project_id": "parent-fixture", "script_version": "casting-60-v1"}
    save_project(item)
    calls = []
    monkeypatch.setattr(pipeline, "render_segment", lambda p, s, **kw: failed_render(p, calls, **kw))
    rows = pipeline.render_utterances(project, ["line"], renderer=FakeRenderer())
    assert len(rows) == 1 and rows[0].dispatched
    assert calls == [None]


def test_manifest_render_passes_canonical_role_to_renderer(project, monkeypatch):
    item = load_project(project)
    item.roles.append(Role(id="canonical-speaker", name="Canonical", tts_backend="fixture-engine"))
    item.character_cast.role_aliases["speaker"] = "canonical-speaker"
    item.voice_lab.casting["enabled"] = True
    save_project(item)
    monkeypatch.setattr(casting_review, "manifest_current", lambda *_: True)
    monkeypatch.setattr(casting_review, "manifest_still_current", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(casting_review, "cast_current", lambda *_: True)
    calls = []
    monkeypatch.setattr(pipeline, "render_segment", lambda p, s, **kw: failed_render(p, calls, **kw))
    rows = pipeline.render_utterances(project, ["line"], renderer=FakeRenderer(), casting_manifest={"fixture": True})
    assert len(rows) == 1 and calls[0].id == "canonical-speaker"
