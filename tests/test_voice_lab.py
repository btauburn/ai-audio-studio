"""Canonical Voice Lab state and API regressions; all audio is isolated signal data."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.models.audiobook import ChunkRecord
from dubbing_service.app.models.jobs import SynthesisJob
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments, load_segments
from dubbing_service.app.storage.jobs import upsert_job, load_jobs
from dubbing_service.app.routers.voice_lab import router
from tests.audiobook_helpers import use_projects_root, make_tone_wav


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    p = create_project("Isolated signal test, never a requested listening test")
    p.voice_lab.scope = "regression_fixture"
    r = Role(id="morgan", name="Morgan", tts_backend="qwen3tts_cv", qwen3_speaker="ryan", qwen3_language="English")
    p.roles = [r]
    save_project(p)
    path = make_tone_wav(root / p.id / "audio/segments/fixture.wav", seconds=2)
    take = SegmentTake(id="take_fixture", audio_file="audio/segments/fixture.wav", audio_sha256=file_sha256(path),
        role_id=r.id, engine_id="qwen3tts_customvoice", fingerprint_components={"voice_revision": voice_revision_for_role(r), "builtin_speaker": "ryan", "language": "English", "style_instruction": ""},
        generation_metadata={"builtin_voice_snapshot": r.model_dump(mode="json")},
        chunk_map=[ChunkRecord(index=0, text_start=0, text_end=10, audio_start_ms=0, audio_end_ms=2000)])
    segment = Segment(id="u1", index=0, role_id=r.id, text="Test tone.", takes=[take], selected_take_id=take.id)
    save_segments(p.id, [segment])
    app = FastAPI(); app.include_router(router, prefix="/api/v1")
    return root, p, r, TestClient(app), f"/api/v1/projects/{p.id}/voice-lab"


def voice(fixture):
    return fixture[3].get(fixture[4]).json()["voices"][0]


def test_preferences_survive_process_restart_and_preserve_takes(fixture):
    root, p, r, client, base = fixture
    v = voice(fixture)
    before = (root / p.id / "segments.json").read_bytes()
    response = client.post(base + "/preferences", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "favorite": True, "rating": 5})
    assert response.status_code == 200
    code = "from dubbing_service.app.storage.projects import load_project; import json; print(json.dumps(load_project('" + p.id + "').voice_lab.preferences))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    persisted = json.loads(result.stdout)[v["key"]]
    assert persisted["favorite"] and persisted["rating"] == 5
    assert persisted["audio_sha256"] == v["audio_sha256"]
    assert persisted["version"] == v["version"] and persisted["take_id"] == "take_fixture"
    assert (root / p.id / "segments.json").read_bytes() == before
    assert load_project(p.id).roles[0].qwen3_speaker == "ryan"
    assert client.post(base + "/preferences", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "favorite": False, "rating": None}).status_code == 200
    assert voice(fixture)["rating"] is None


def test_audio_version_mismatch_and_tampering_are_rejected(fixture):
    root, p, _, client, base = fixture; v = voice(fixture)
    payload = {"key": v["key"], "audio_sha256": "a"*64, "rating": 4}
    assert client.post(base + "/preferences", json=payload).status_code == 409
    path = root / p.id / "audio/segments/fixture.wav"
    with path.open("ab") as f: f.write(b"changed")
    payload["audio_sha256"] = v["audio_sha256"]
    assert client.post(base + "/preferences", json=payload).status_code == 409
    assert client.get(base + "/audio/" + v["key"]).status_code == 409


def test_range_audio_and_read_actions_do_not_write(fixture):
    root, p, _, client, base = fixture
    before = (root / p.id / "project.json").read_bytes()
    v = voice(fixture)
    response = client.get(v["audio_url"], headers={"range": "bytes=0-31"})
    assert response.status_code == 206 and len(response.content) == 32
    assert (root / p.id / "project.json").read_bytes() == before
    assert not load_project(p.id).voice_lab.preferences


def test_technical_issues_are_separate_from_preference(fixture):
    _, p, _, client, base = fixture; v = voice(fixture)
    body = {"audio_sha256": v["audio_sha256"], "seconds": 1.5, "text": "Fixture defect", "category": "technical"}
    assert client.post(base + "/issues", json=body).status_code == 200
    assert not load_project(p.id).voice_lab.preferences
    assert client.post(base + "/issues", json={**body, "seconds": 2.01}).status_code == 409
    assert client.post(base + "/position", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "seconds": 1.25, "filter": "new"}).status_code == 200
    assert client.get(base).json()["review_position"]["seconds"] == 1.25


def test_role_suitability_is_not_global_rating(fixture):
    _, p, r, client, base = fixture; v = voice(fixture)
    assert client.post(base + "/preferences", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "suitability": 2, "role_id": r.id}).status_code == 200
    pref = load_project(p.id).voice_lab.preferences[v["key"]]
    assert pref["suitability"] == {r.id: 2} and pref.get("rating") is None


def test_more_like_idempotent_no_ratings_or_audio(fixture):
    _, p, _, client, base = fixture; v = voice(fixture)
    body = {"key": v["key"], "request_id": "same-intent"}
    a = client.post(base + "/more-like", json=body).json()
    b = client.post(base + "/more-like", json=body).json()
    assert a == b and len(a["candidates"]) == 2
    assert all(c["rating"] is None and not c["favorite"] for c in a["candidates"])
    assert len(load_segments(p.id)[0].takes) == 1


def test_stale_explicit_cast_rejected(fixture):
    _, p, r, client, base = fixture; v = voice(fixture)
    assert client.post(base + "/assign", json={"key": v["key"], "role_id": r.id, "expected_identity_id": "old"}).status_code == 409
    assert client.post(base + "/assign", json={"key": v["key"], "role_id": r.id, "expected_identity_id": voice_revision_for_role(r)}).status_code == 200
    assert load_segments(p.id)[0].selected_take_id == "take_fixture"


def test_changing_role_does_not_relabel_heard_voice(fixture):
    _, p, _, client, base = fixture; old = voice(fixture)
    project = load_project(p.id); project.roles[0].qwen3_speaker = "aiden"; save_project(project)
    current = voice(fixture)
    assert current["key"] == old["key"]
    assert "ryan" in current["voice_id"]


def test_source_blocker_precedes_provider_and_gpu(fixture, monkeypatch):
    _, p, r, _, _ = fixture
    project = load_project(p.id); project.voice_lab.scope = "isolated_listening_test"
    project.voice_lab.excerpt = {"status": "blocked_source", "reason": "Only chapters 256-260 are local"}; save_project(project)
    monkeypatch.setattr(lab, "production_busy", lambda: pytest.fail("GPU inspected before source gate"))
    with pytest.raises(lab.VoiceLabError, match="256-260"):
        lab.design(p.id, r.id, "x", voice_revision_for_role(r), "req")


def test_discovery_yields_to_queued_production_and_resumes_idempotently(fixture, monkeypatch):
    _, p, _, _, _ = fixture
    lab.set_automation(p.id, False)
    upsert_job(p.id, SynthesisJob(id="production", project_id=p.id, job_type="batch_synthesize", segment_ids=["u1"]))
    calls = []
    monkeypatch.setattr(lab, "match", lambda *a, **kw: calls.append(a))
    assert lab.discovery_tick(p.id, online=False)["status"] == "waiting_for_production"
    assert calls == []
    from dubbing_service.app.storage.jobs import cancel_job
    cancel_job(p.id, "production")
    assert lab.discovery_tick(p.id, online=False)["cursor"] == 1
    assert len(calls) == 1
    assert any(j.job_type == "voice_lab_discovery" and j.status.value == "succeeded" for j in load_jobs(p.id))
    lab.set_automation(p.id, True)
    lab.discovery_tick(p.id, online=False)
    assert len(calls) == 1


def test_newer_match_request_wins(fixture, monkeypatch):
    _, p, _, _, _ = fixture
    from dubbing_service.app.audiobook import performer_discovery
    def slow_discovery(*a, **kw):
        project = load_project(p.id)
        project.voice_lab.shortlists["general"] = {"token": "newer", "matches": [{"id": "new"}]}
        save_project(project)
        return {"matches": [{"id": "old"}]}
    monkeypatch.setattr(performer_discovery, "discover", slow_discovery)
    assert lab.match(p.id, "young male", online=False)["stale"]
    assert load_project(p.id).voice_lab.shortlists["general"]["matches"] == [{"id": "new"}]


def test_conditioning_placeholder_binds_first_heard_take_durably(fixture):
    _, p, _, client, base = fixture
    v = voice(fixture)
    project = load_project(p.id)
    row = lab.voice_records(project)[v["key"]]
    row.update(audio_file=None, audio_sha256=None, duration_seconds=0)
    project.voice_lab.voices[v["key"]] = row
    save_project(project)
    promoted = voice(fixture)
    assert promoted["audio_sha256"] == v["audio_sha256"]
    assert client.post(base + "/preferences", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "favorite": True}).status_code == 200
    stored = load_project(p.id).voice_lab.voices[v["key"]]
    assert stored["audio_sha256"] == v["audio_sha256"] and stored["take_id"] == "take_fixture"
    segments = load_segments(p.id)
    segments[0].takes[0].fingerprint_components["voice_revision"] = "newrevision"
    save_segments(p.id, segments)
    assert lab.voice_records(load_project(p.id))[v["key"]]["audio_sha256"] == v["audio_sha256"]


def test_concurrent_discovery_resume_claim_is_atomic(fixture, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    _, p, _, _, _ = fixture
    lab.set_automation(p.id, False)
    entered, release = Event(), Event()
    calls = []
    def discover(*args, **kwargs):
        calls.append(args); entered.set(); assert release.wait(5)
    monkeypatch.setattr(lab, "match", discover)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lab.discovery_tick, p.id, online=False)
        assert entered.wait(5)
        second = pool.submit(lab.discovery_tick, p.id, online=False)
        assert second.result(timeout=5)["status"] == "already_running"
        release.set(); assert first.result(timeout=5)["cursor"] == 1
    assert len(calls) == 1


def test_restart_resumes_only_opted_in_discovery(fixture, monkeypatch):
    _, p, _, _, _ = fixture
    calls = []
    monkeypatch.setattr(lab, "discovery_tick", lambda project: calls.append(project))
    lab.resume_discoveries(); assert not calls
    lab.set_automation(p.id, False)
    lab.resume_discoveries(); assert calls == [p.id]


def test_ab_requires_same_passage_measured_hash_and_owner_routes(fixture):
    _, p, _, client, base = fixture; v = voice(fixture)
    project = load_project(p.id)
    original = lab.voice_records(project)[v["key"]]
    comparison = {**original, "key": "v_" + "b"*28, "passage_id": "manifest", "utterance_ids": ["u1"], "loudness_lufs": -20,
                  "loudness_measurement": {"audio_sha256": v["audio_sha256"], "integrated_lufs": -20}}
    project.voice_lab.excerpt = {**original, "manifest_id": "manifest", "map": [{"utterance_id": "u1"}], "comparisons": [comparison]}
    save_project(project)
    result = client.get(base).json()["excerpt"]["comparisons"][0]
    assert client.get(result["audio_url"], headers={"range": "bytes=0-15"}).status_code == 206
    assert client.post(base + "/position", json={"key": comparison["key"], "audio_sha256": v["audio_sha256"], "seconds": 1}).status_code == 200
    project = load_project(p.id); project.voice_lab.excerpt["comparisons"][0]["utterance_ids"] = ["other"]
    save_project(project)
    assert client.get(result["audio_url"]).status_code == 409


def test_ready_design_without_production_audio_cannot_be_favorited(fixture):
    _, p, _, client, base = fixture
    project = load_project(p.id)
    project.voice_lab.voices["draft"] = {"key": "draft", "voice_id": "draft", "version": "one", "audio_file": None, "audio_sha256": None}
    save_project(project)
    assert client.post(base + "/preferences", json={"key": "draft", "audio_sha256": "a"*64, "favorite": True}).status_code == 409


@pytest.mark.parametrize("rating", [0, 6, 2.5, True])
def test_bad_rating_inputs(fixture, rating):
    _, _, _, client, base = fixture; v = voice(fixture)
    assert client.post(base + "/preferences", json={"key": v["key"], "audio_sha256": v["audio_sha256"], "rating": rating}).status_code == 422
