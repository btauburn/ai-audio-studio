"""CPU regressions over real project/round/cache owners; synthetic voices only."""
import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import casting_director as director, casting_review
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.routers.directors import router


@pytest.fixture
def project():
    p = create_project("Synthetic directors integration")
    p.voice_lab.scope = "regression_fixture"
    p.roles = [Role(id="alex", name="Alex", tts_backend="qwen3tts")]
    save_project(p)
    save_segments(p.id, [Segment(id="line", index=0, role_id="alex", text="Do we still have time?")])
    return p


def report(p, protected=False):
    current = load_project(p.id)
    binding = director._binding(current, load_segments(p.id))
    payload = {"schema": director.SCHEMA, "project_id": p.id, "runtime": "faster-base-032",
               "binding_sha256": binding, "roles": [{"role_id": "alex", "protected": protected,
                    "candidates": [{"candidate_id": f"choice-{i}", "name": f"Original {i}",
                        "origin": "model_design", "eligible": True, "holds": [], "tags": [],
                        "design": {"design_instruction": f"Original clear adult voice, variation {i}."}}
                        for i in range(3)]}], "holds": [], "human_review": "pending"}
    return {**payload, "report_sha256": director._digest(payload)}


def test_get_is_read_only_and_does_not_synthesize(project, monkeypatch):
    before = load_project(project.id).model_dump_json()
    monkeypatch.setattr(director, "_compute", lambda *args: pytest.fail("GET must not analyze"))
    result = director.view(project.id)
    assert result["status"] == "needs_analysis"
    assert not director._cache_path(project.id, result["binding_sha256"]).exists()
    assert load_project(project.id).model_dump_json() == before


def test_cache_retry_does_not_recompute_or_change_project(project, monkeypatch):
    calls = []
    def compute(p, s, binding):
        calls.append(binding)
        return report(p)
    monkeypatch.setattr(director, "_compute", compute)
    before = load_project(project.id).model_dump_json()
    first = director.analyze(project.id)
    assert director.analyze(project.id) == first
    assert director.view(project.id) == first
    assert len(calls) == 1
    assert load_project(project.id).model_dump_json() == before


def test_prose_edit_invalidates_analysis(project, monkeypatch):
    monkeypatch.setattr(director, "_compute", lambda p, s, b: report(p))
    director.analyze(project.id)
    rows = load_segments(project.id)
    rows[0].text += " Exactly."
    save_segments(project.id, rows)
    assert director.view(project.id)["status"] == "needs_analysis"


def test_cache_tampering_refused(project, monkeypatch):
    monkeypatch.setattr(director, "_compute", lambda p, s, b: report(p))
    result = director.analyze(project.id)
    path = director._cache_path(project.id, result["binding_sha256"])
    data = json.loads(path.read_bytes())
    data["human_review"] = "approved"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(director.DirectorError, match="identity mismatch"):
        director.view(project.id)


def test_approved_voice_cannot_be_recast(project, monkeypatch):
    r = report(project, protected=True)
    monkeypatch.setattr(director, "analyze", lambda _: r)
    with pytest.raises(director.DirectorError, match="protected"):
        director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])
    assert not load_project(project.id).voice_lab.casting["intents"]


def test_ranked_auditions_use_canonical_round_and_safe_retry(project, monkeypatch):
    r = report(project)
    monkeypatch.setattr(director, "analyze", lambda _: r)
    before_lines = [s.model_dump_json() for s in load_segments(project.id)]
    first = director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])
    after = load_project(project.id)
    round_ = after.voice_lab.casting["roles"]["alex"]["round"]
    assert first["candidate_count"] == 3
    assert all(c["status"] == "pending" for c in round_["candidates"])
    assert after.roles[0].voice_identity is None
    assert after.voice_lab.casting["roles"]["alex"]["assignment"] is None
    assert not after.metadata[director.META]["round_plans"]
    assert [s.model_dump_json() for s in load_segments(project.id)] == before_lines
    monkeypatch.setattr(director, "analyze", lambda _: pytest.fail("Retry must reuse the frozen receipt"))
    assert director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"]) == first
    assert len(load_project(project.id).voice_lab.casting["intents"]) == 1


def test_request_identity_cannot_be_rebound(project, monkeypatch):
    r = report(project)
    monkeypatch.setattr(director, "analyze", lambda _: r)
    director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])
    with pytest.raises(director.DirectorError, match="another audition"):
        director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256="a" * 64)


def test_unprepared_external_source_is_held(project, monkeypatch):
    r = report(project)
    for c in r["roles"][0]["candidates"]:
        c["origin"] = "authorized_external"
        c.pop("design")
    monkeypatch.setattr(director, "analyze", lambda _: r)
    with pytest.raises(director.DirectorError, match="no viable prepared"):
        director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])


def test_stale_report_is_rejected_before_round(project, monkeypatch):
    r = report(project)
    monkeypatch.setattr(director, "analyze", lambda _: r)
    with pytest.raises(director.DirectorError, match="refresh"):
        director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256="b" * 64)


def test_failed_import_does_not_substitute_generic(project, monkeypatch):
    from dubbing_service.app.audiobook import preferred_voices
    r = report(project)
    for c in r["roles"][0]["candidates"]:
        c.update(origin="approved_library", library_voice_id="library", library_version="v1")
    monkeypatch.setattr(director, "analyze", lambda _: r)
    monkeypatch.setattr(preferred_voices, "prepare_prior_candidate", lambda *a: {"state": "blocked", "reason": "Consent held"})
    with pytest.raises(director.DirectorError, match="Consent held"):
        director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])


def test_first_library_candidate_retains_exact_reuse_key(project, monkeypatch):
    from dubbing_service.app.audiobook import preferred_voices
    r = report(project)
    for c in r["roles"][0]["candidates"]:
        c.update(origin="approved_library", library_voice_id="library", library_version="v1")
    monkeypatch.setattr(director, "analyze", lambda _: r)
    monkeypatch.setattr(preferred_voices, "prepare_prior_candidate", lambda *a: {"state": "ready", "key": "exact-library-reference"})
    director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])
    round_ = load_project(project.id).voice_lab.casting["roles"]["alex"]["round"]
    assert all(c["reuse_key"] == "exact-library-reference" for c in round_["candidates"])


def test_runtime_change_invalidates_report(project, monkeypatch):
    monkeypatch.setattr(director, "_compute", lambda p, s, b: report(p))
    director.analyze(project.id)
    p = load_project(project.id)
    p.metadata["requested_runtime"] = "native-current"
    save_project(p)
    assert director.view(project.id)["status"] == "needs_analysis"


def test_api_requires_explicit_review_action(project, monkeypatch):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    monkeypatch.setattr(director, "analyze", lambda _: report(project))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{project.id}/voice-lab/directors"
        assert client.get(base).status_code == 200
        assert client.post(base + "/analyze").status_code == 403
        assert client.post(base + "/analyze", headers={"X-Review-Request": "1"}).status_code == 200


def test_existing_casting_plan_unchanged_without_director_request(project):
    p = load_project(project.id)
    assert director.proposals_for_round(p, "alex") is None
    assert not p.metadata.get(director.META)


def test_no_candidate_is_automatically_promoted(project, monkeypatch):
    from dubbing_service.app.storage import voices
    r = report(project)
    monkeypatch.setattr(director, "analyze", lambda _: r)
    monkeypatch.setattr(voices, "save_voices", lambda *a: pytest.fail("Auditions cannot promote voices"))
    director.prepare_auditions(project.id, request_id="one", role_id="alex", report_sha256=r["report_sha256"])
    assert not load_project(project.id).voice_lab.casting["events"]


def test_analysis_extracts_current_evidence_without_editing_the_book(project):
    p = load_project(project.id)
    p.roles.append(Role(id="narrator", name="Narrator", role_type="narrator"))
    save_project(p)
    rows = [Segment(id="description", index=0, role_id="narrator", text="Alex was nineteen years old. Alex spoke in a clear voice."),
            Segment(id="dialogue", index=1, role_id="alex", text="We should wait.")]
    save_segments(p.id, rows)
    before = load_project(p.id).model_dump_json()
    result = director.analyze(p.id)
    analysis = next(r['analysis'] for r in result['roles'] if r['role_id'] == 'alex')
    assert analysis['evidence']
    assert load_project(p.id).model_dump_json() == before
    assert result['human_review'] == 'pending'


def test_role_rejection_is_not_erased_by_global_keeper_rating(project, monkeypatch):
    from dubbing_service.app.audiobook import casting_director_discovery as discovery
    from dubbing_service.app.audiobook import casting_director_matching as matching
    from dubbing_service.app.audiobook import performance_scene_analysis as scene
    p = load_project(project.id)
    p.voice_lab.voices['low'] = {'key': 'low', 'version': 'old', 'identity': {'id': 'same-reference'}, 'audio_sha256': 'c' * 64}
    p.voice_lab.casting['roles']['alex'] = {'rejections': [{'key': 'low', 'rating': 2}]}
    seen = []
    monkeypatch.setattr(discovery, 'discover_candidates', lambda *a: {'candidates': [
        {'candidate_id': 'keeper', 'key': 'global-key', 'identity_id': 'same-reference', 'rating': 5},
        {'candidate_id': 'new', 'key': 'new', 'identity_id': 'different-reference', 'rating': 4}], 'holds': []})
    monkeypatch.setattr(matching, 'analyze_characters', lambda *a: [{'role_id': 'alex'}])
    def rank(a, candidates, **kw):
        seen.extend(c['candidate_id'] for c in candidates)
        return []
    monkeypatch.setattr(matching, 'rank_candidates', rank)
    monkeypatch.setattr(scene, 'analyze_scenes', lambda *a, **kw: {})
    director._compute(p, load_segments(p.id), 'a' * 64)
    assert seen == ['new']


def test_keeper_approval_does_not_authorize_a_different_engine():
    preset = {'rating': 5, 'protected': True, 'provenance': {'rights_status': 'provider_preset'}}
    assert director._base_compatible(preset) is False
    assert preset['rating'] == 5 and preset['protected'] is True
    assert director._base_compatible({'reference_conditioned': True, 'conditioning': {'backend': 'qwen3tts'}}) is True
    assert director._base_compatible({'reference_conditioned': True, 'conditioning': {'backend': 'indextts'}}) is False
    assert director._base_compatible({'reference_conditioned': True, 'conditioning': {}}) is False
    assert director._base_compatible({'origin': 'model_design', 'reference_conditioned': False}) is True
