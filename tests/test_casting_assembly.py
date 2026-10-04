"""Strict assembly guards over isolated, synthetic project fixtures."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from dubbing_service.app.audiobook import casting_assembly as assembly, casting_review, pipeline
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.audiobook import TakeQaReport
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.routers import synthesis
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root


@pytest.fixture
def project(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    item = create_project("Strict assembly synthetic fixture")
    item.voice_lab.scope = "regression_fixture"
    item.voice_lab.casting["enabled"] = True
    item.roles = [Role(id="speaker", name="Alex", tts_backend="fixture-tts")]
    save_project(item)
    save_segments(item.id, [Segment(id="line-1", index=0, text="Synthetic assembly line.", role_id="speaker")])
    return root, item.id


def add_frozen_approval(project_id, alias=False):
    project = load_project(project_id)
    role = project.roles[0]
    if alias:   # a character merged into another: its lines still carry the old id
        project.character_cast.role_aliases = {"speaker-dup": "speaker"}
        merged = load_segments(project_id)
        merged[0].role_id = "speaker-dup"
        save_segments(project_id, merged)
        save_project(project)
    revision = voice_revision_for_role(role)
    event_id = "evt-synthetic-approved"
    project.voice_lab.casting["roles"][role.id] = {
        "assignment_revision": 1, "round_revision": 0, "history": [], "round": {}, "rejections": [],
        "assignment": {"key": "fixture-voice", "voice_revision": revision, "approved": True,
            "rejected": False, "event_id": event_id, "stars": 5, "revision": 1},
    }
    project.voice_lab.casting["events"][event_id] = {
        "actor": "brian", "origin": "SYNTHETIC", "stars": 5,
    }
    rows = load_segments(project_id)
    manifest = casting_review.freeze_manifest(project, rows, [role.id])
    project.voice_lab.casting["productions"]["strict-fixture"] = {
        "request_id": "strict-fixture", "status": "succeeded", "manifest": manifest,
        "manifest_history": [manifest],
    }
    path = make_tone_wav(safe_project_root(project_id) / "audio" / "selected.wav", seconds=0.7)
    audio_hash = file_sha256(path)
    requirement = manifest["requirements"]["line-1"]
    take = SegmentTake(id="take-fixture", audio_file="audio/selected.wav", audio_sha256=audio_hash,
        text=rows[0].text, role_id=rows[0].role_id, tts_backend=role.tts_backend, input_fingerprint=requirement,
        source_text_hash=pipeline.fingerprint_module.text_sha256(rows[0].text),
        fingerprint_components={"voice_revision": revision},
        qa_report=TakeQaReport(take_id="take-fixture", output_audio_hash=audio_hash,
            qa_check_version="fixture-qa", passed=True))
    rows[0].takes.append(take)
    rows[0].selected_take_id = take.id
    save_segments(project_id, rows)
    save_project(project)
    return manifest


def test_legacy_projects_skip_strict_guard_and_keep_export_path_unchanged(project):
    _, project_id = project
    item = load_project(project_id)
    item.voice_lab.casting["enabled"] = False
    item.metadata.pop("autocast_import", None)
    save_project(item)
    assert assembly.current_frozen_binding(load_project(project_id), load_segments(project_id)) is None


def test_strict_project_without_current_frozen_manifest_is_rejected(project):
    _, project_id = project
    with pytest.raises(assembly.CastingAssemblyError, match="requires a current frozen cast manifest"):
        assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))


def test_current_approved_manifest_accepts_only_exact_selected_take(project):
    _, project_id = project
    manifest = add_frozen_approval(project_id)
    binding = assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))
    assert binding["manifest_id"] == manifest["id"]
    assert binding["selections"][0]["utterance_id"] == "line-1"
    assert binding["selections"][0]["role_id"] == "speaker"
    assert binding["selections"][0]["take_id"] == "take-fixture"
    assert binding["selections"][0]["input_fingerprint"] == manifest["requirements"]["line-1"]
    assert len(binding["selections"][0]["audio_sha256"]) == 64


@pytest.mark.parametrize("corruption", ["fingerprint", "unapproved", "audio"])
def test_strict_guard_rejects_changed_take_or_lost_human_approval(project, corruption):
    root, project_id = project
    add_frozen_approval(project_id)
    if corruption == "fingerprint":
        rows = load_segments(project_id)
        rows[0].takes[0].input_fingerprint = "stale-requirement"
        save_segments(project_id, rows)
    elif corruption == "unapproved":
        item = load_project(project_id)
        item.voice_lab.casting["roles"]["speaker"]["assignment"]["approved"] = False
        save_project(item)
    else:
        (root / project_id / "audio" / "selected.wav").write_bytes(b"changed audio")
    with pytest.raises(assembly.CastingAssemblyError):
        assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))


def test_binding_revalidation_refuses_mid_assembly_assignment_change(project):
    _, project_id = project
    add_frozen_approval(project_id)
    original = assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))
    item = load_project(project_id)
    item.voice_lab.casting["roles"]["speaker"]["assignment"]["approved"] = False
    save_project(item)
    with pytest.raises(assembly.CastingAssemblyError, match="current frozen cast manifest"):
        assembly.revalidate_binding(load_project(project_id), load_segments(project_id), original)


@pytest.mark.parametrize("endpoint", [synthesis.api_export_full_project, synthesis.api_export_full_project_with_edits])
def test_full_project_export_routes_refuse_strict_projects_before_assembly(project, endpoint, monkeypatch):
    _, project_id = project
    called = []
    monkeypatch.setattr(synthesis, "_probe_audio_format", lambda *_: called.append("assembly"))
    with pytest.raises(HTTPException) as error:
        endpoint(project_id)
    assert getattr(error.value, "status_code", None) == 409
    assert called == []


def test_a_take_recorded_under_a_merged_characters_old_id_still_assembles(project):
    """Recording stores the line's own role id, which is the old id after a merge; the voice is the merged character's."""
    _, project_id = project
    manifest = add_frozen_approval(project_id, alias=True)
    row = load_segments(project_id)[0]
    assert row.role_id == "speaker-dup" and row.takes[0].role_id == "speaker-dup"
    binding = assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))
    assert binding["manifest_id"] == manifest["id"]
    assert binding["selections"][0]["role_id"] == "speaker", "the selection is bound to the canonical character"


def test_a_take_for_a_different_character_is_still_refused_after_a_merge(project):
    _, project_id = project
    add_frozen_approval(project_id, alias=True)
    rows = load_segments(project_id)
    rows[0].takes[0].role_id = "someone-else"
    save_segments(project_id, rows)
    with pytest.raises(assembly.CastingAssemblyError, match="not the frozen approved"):
        assembly.current_frozen_binding(load_project(project_id), load_segments(project_id))
