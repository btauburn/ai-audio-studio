"""Preview boundary regressions: actual pipeline/QA with a test-only signal provider.

All data lives in pytest temporary project roots. No real books, preferences,
workers, models, network, or GPU are used by these tests.
"""
import os
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_audition as preview, renderer
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.jobs import SynthesisJob
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.routers.voice_lab import router as lab_router
from dubbing_service.app.routers.voice_lab_audition import router
from dubbing_service.app.storage.jobs import cancel_job, load_jobs, mutate_job, upsert_job
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import anchored_candidate, make_tone_wav, make_silence_wav, use_projects_root


class SignalProvider:
    def __init__(self):
        self.designs = []
        self.calls = []
        self.silent = False
        self.fail = False

    def design_character_voice(self, project_id, identity_id, instruction, text, seed, conditioning=None):
        self.designs.append((project_id, identity_id))
        self.conditioning = conditioning
        root = safe_project_root(project_id)
        reference = make_tone_wav(root / "voices/reference.wav", seconds=1)
        prompt = root / "voices/prompt.npz"
        prompt.write_bytes(b"LOCAL-TEST-SIGNAL-NOT-A-MODEL-PROMPT")
        return {"mechanism": "qwen_reference", "backend": "qwen3tts",
                "model": "LOCAL-TEST-SIGNAL-NOT-TTS", "runtime_version": "test-only",
                "reference_audio": reference.relative_to(root).as_posix(), "reference_audio_sha256": file_sha256(reference),
                "reference_text": text, "prompt_artifact": prompt.relative_to(root).as_posix(),
                "prompt_sha256": file_sha256(prompt), "provenance": {"origin": "generated_original_voice", "fixture_only": True}}

    def synthesize_segment(self, project_id, role, segment):
        self.calls.append((project_id, segment.id))
        if self.fail:
            return TTSResult(success=False, raw={"error": "isolated transient test failure"})
        maker = make_silence_wav if self.silent else make_tone_wav
        path = maker(safe_project_root(project_id) / f"signals/{len(self.calls)}.wav", seconds=10)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id, "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": True}})


@pytest.fixture
def setup(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = create_project("Isolated short preview regression, never a listening test")
    role = Role(id="role1", name="Morgan", tts_backend="qwen3tts_cv", qwen3_speaker="ryan", qwen3_language="English")
    project.roles = [role]
    project.voice_lab.shortlists[role.id] = {"matches": [
        anchored_candidate("test-candidate", design_instruction="Clear natural delivery.")]}
    save_project(project)
    save_segments(project.id, [Segment(id="unchanged-book", index=0, text="Existing selected book state must not be replaced.", role_id=role.id)])
    fake = SignalProvider()
    monkeypatch.setattr(preview, "ensure_backend_ready", lambda *a, **kw: {"fixture_only": True})
    monkeypatch.setattr(renderer.TTSProviderFactory, "create", lambda backend: fake)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.include_router(lab_router, prefix="/api/v1")
    body = {"role_id": role.id, "candidate_id": "test-candidate", "expected_identity_id": voice_revision_for_role(role), "request_id": "preview-one"}
    return root, project, role, fake, body, TestClient(app)


def run(setup, **overrides):
    _, project, _, provider, body, _ = setup
    return preview.audition(project.id, **{**body, **overrides}, design_provider=provider)


def test_the_preview_request_budget_scales_with_the_cast_instead_of_blocking_a_recase():
    """A re-cast adds request history without removing any, so a fixed cap is wrong.

    The history is append-only because a retained request is what makes a retry
    idempotent. A ceiling that does not account for the cast therefore blocks a
    legitimate later casting generation of a multi-character book, which is what
    happened: the third generation of an eleven-character project hit the fixed
    hundred-request ceiling with five characters still to generate.
    """
    from dubbing_service.app.audiobook import voice_lab_audition as audition

    state = {"schema": "voice-lab-auditions-v1", "requests": {}, "intents": {}}
    project = SimpleNamespace(metadata={"voice_lab_auditions": state})
    assert audition._request_budget(project) == audition.MAX_REQUESTS + \
        audition.REQUESTS_PER_CAST_CHARACTER, "an empty history still reserves one character"

    state["intents"] = {f"intent-{index}": {"role_id": f"role-{index % 11}"} for index in range(33)}
    budget = audition._request_budget(project)
    assert budget == audition.MAX_REQUESTS + audition.REQUESTS_PER_CAST_CHARACTER * 11
    assert budget > audition.MAX_REQUESTS, "a real cast gets more than the baseline"
    # Three generations of an eleven-character cast fit inside the budget.
    assert budget >= 3 * 3 * 11
    # One character accumulating history alone stays at the baseline allowance.
    state["intents"] = {"intent-0": {"role_id": "role-0"}}
    assert audition._request_budget(project) == \
        audition.MAX_REQUESTS + audition.REQUESTS_PER_CAST_CHARACTER


def test_real_pipeline_preview_reuses_exact_identity_and_never_changes_parent_cast_or_takes(setup):
    root, project, role, provider, body, _ = setup
    before = (root / project.id / "segments.json").read_bytes()
    role_before = role.model_dump(mode="json")
    result = run(setup)
    assert result["status"] == "succeeded", result
    assert result["duration_seconds"] == 10
    assert len(provider.designs) == len(provider.calls) == 1
    parent = load_project(project.id)
    assert parent.roles[0].model_dump(mode="json") == role_before
    assert (root / project.id / "segments.json").read_bytes() == before
    assert not parent.voice_lab.preferences and not parent.voice_lab.preference_history
    record = parent.voice_lab.voices[result["key"]]
    child = load_project(record["audition_project_id"])
    segment = load_segments(child.id)[0]
    take = next(t for t in segment.takes if t.id == segment.selected_take_id)
    assert take.engine_id == take.renderer_id == "qwen3tts"
    assert take.qa_report.matches(take.id, take.audio_sha256) and take.qa_report.passed
    assert record["take_id"] == take.id and record["audio_sha256"] == take.audio_sha256
    assert record["version"] == voice_revision_for_role(child.roles[0])
    proof = json.loads((root / project.id / record["details"]["take_proof_file"]).read_text())
    assert proof["take"]["id"] == take.id and proof["source_text"] == preview.PREVIEW_TEXT
    same = run(setup)
    fresh_click = run(setup, request_id="preview-again")
    assert same["key"] == fresh_click["key"] == result["key"]
    assert len(provider.designs) == len(provider.calls) == 1
    assert len(load_jobs(project.id)) == 1
    assert len(preview.audition_summaries(load_project(project.id))) == 1
    # Selecting is explicit, uses the heard identity, and still preserves Takes.
    assigned = lab.assign(project.id, result["key"], role.id, body["expected_identity_id"])
    assert assigned["same_voice_revision"]
    assert load_project(project.id).roles[0].voice_identity.id == child.roles[0].voice_identity.id
    assert (root / project.id / "segments.json").read_bytes() == before


def test_scheduled_double_click_and_get_do_not_dispatch_twice(setup):
    root, project, _, provider, body, client = setup
    scheduled = []
    schedule = lambda *a, **kw: scheduled.append((a, kw))
    a = preview.audition(project.id, **body, schedule=schedule, design_provider=provider)
    b = preview.audition(project.id, **body, schedule=schedule, design_provider=provider)
    assert a["job_id"] == b["job_id"] and len(scheduled) == 1
    before = (root / project.id / "project.json").read_bytes()
    response = client.get(f"/api/v1/projects/{project.id}/voice-lab/auditions/{body['request_id']}")
    assert response.status_code == 200 and response.json()["key"] is None
    assert (root / project.id / "project.json").read_bytes() == before
    assert not provider.designs and not provider.calls
    args, kwargs = scheduled.pop()
    args[0](*args[1:], **kwargs)
    assert preview.audition_status(project.id, body["request_id"])["status"] == "succeeded"


def test_production_priority_returns_durable_queue_and_explicit_resume(setup):
    _, project, _, provider, body, _ = setup
    upsert_job(project.id, SynthesisJob(id="old-production", project_id=project.id, job_type="batch_synthesize", segment_ids=["unchanged-book"]))
    result = run(setup)
    assert result["status"] == "queued" and "priority" in result["reason"]
    assert not provider.designs and not provider.calls
    assert preview.audition_status(project.id, body["request_id"])["status"] == "queued"
    cancel_job(project.id, "old-production")
    assert run(setup)["status"] == "succeeded"
    assert len(provider.designs) == len(provider.calls) == 1


def test_stale_new_request_and_request_id_rebinding_rejected_before_generation(setup):
    _, project, _, provider, body, _ = setup
    with pytest.raises(preview.AuditionError, match="Casting changed"):
        run(setup, expected_identity_id="stale")
    scheduled = []
    preview.audition(project.id, **body, schedule=lambda *a, **kw: scheduled.append((a, kw)), design_provider=provider)
    with pytest.raises(preview.AuditionError, match="different preview"):
        run(setup, candidate_id="other")
    args, kwargs = scheduled.pop()
    # A late preview completes as its own version, never overwrites new casting.
    latest = load_project(project.id)
    latest.roles[0].qwen3_speaker = "aiden"
    save_project(latest)
    args[0](*args[1:], **kwargs)
    assert load_project(project.id).roles[0].qwen3_speaker == "aiden"
    result = preview.audition_status(project.id, body["request_id"])
    with pytest.raises(lab.VoiceLabError, match="Casting changed"):
        lab.assign(project.id, result["key"], "role1", body["expected_identity_id"])


def test_transient_failure_resumes_saved_conditioning_and_render(setup):
    _, project, _, provider, body, _ = setup
    provider.fail = True
    failed = run(setup)
    assert failed["status"] == "interrupted" and failed["key"] is None
    assert len(provider.designs) == 1
    failures = len(provider.calls)
    provider.fail = False
    result = run(setup)
    assert result["status"] == "succeeded", result
    assert len(provider.designs) == 1 and len(provider.calls) == failures + 1
    assert run(setup)["key"] == result["key"]
    assert len(provider.calls) == failures + 1


def test_crash_after_take_before_publication_resumes_with_no_duplicate_generation(setup, monkeypatch):
    _, project, _, provider, _, _ = setup
    publish = preview._publish
    monkeypatch.setattr(preview, "_publish", lambda *a: (_ for _ in ()).throw(OSError("simulated interruption before publication")))
    first = run(setup)
    assert first["status"] == "interrupted"
    assert len(provider.designs) == len(provider.calls) == 1
    monkeypatch.setattr(preview, "_publish", publish)
    recovered = run(setup)
    assert recovered["status"] == "succeeded"
    assert len(provider.designs) == len(provider.calls) == 1


def test_legacy_bound_preview_resumes_after_default_text_changes_without_new_audio(setup, monkeypatch):
    _, project, _, provider, body, _ = setup
    current_text = preview.PREVIEW_TEXT
    legacy_text = current_text.split(" Outside,")[0]
    assert len(legacy_text.split()) == 26 and len(current_text.split()) == 39
    monkeypatch.setattr(preview, "PREVIEW_TEXT", legacy_text)
    publish = preview._publish
    monkeypatch.setattr(preview, "_publish", lambda *a: (_ for _ in ()).throw(OSError("interrupted before legacy publication")))
    first = run(setup)
    assert first["status"] == "interrupted" and first["preview_text"] == legacy_text
    parent = load_project(project.id)
    child_id = next(iter(parent.metadata["voice_lab_auditions"]["intents"].values()))["audition_project_id"]
    old_take = load_segments(child_id)[0].selected_take_id
    monkeypatch.setattr(preview, "PREVIEW_TEXT", current_text)
    monkeypatch.setattr(preview, "_publish", publish)
    recovered = run(setup)
    assert recovered["status"] == "succeeded" and recovered["preview_text"] == legacy_text
    assert len(provider.designs) == len(provider.calls) == 1
    assert load_segments(child_id)[0].selected_take_id == old_take
    assert load_segments(child_id)[0].text == legacy_text
    # New explicit requests bind the new source; historical heard audio remains.
    new = run(setup, request_id="longer-preview-explicit")
    assert new["status"] == "succeeded" and new["preview_text"] == current_text
    assert new["key"] != recovered["key"] and len(provider.calls) == 2
    assert preview.audition_status(project.id, body["request_id"])["key"] == recovered["key"]


def test_valid_http_post_background_job_and_audio_use_existing_owner(setup, monkeypatch):
    _, project, _, provider, body, client = setup
    design = lab.design
    monkeypatch.setattr(lab, "design", lambda *a, **kw: design(*a, **{**kw, "provider": provider}))
    base = f"/api/v1/projects/{project.id}/voice-lab"
    posted = client.post(base + "/audition", json=body)
    assert posted.status_code == 200
    result = client.get(base + "/auditions/" + body["request_id"]).json()
    assert result["status"] == "succeeded"
    view = client.get(base).json()
    heard = next(v for v in view["voices"] if v["key"] == result["key"])
    assert heard["technical_qa"] == "passed" and not heard["favorite"] and heard["rating"] is None
    assert client.get(heard["audio_url"], headers={"Range": "bytes=0-31"}).status_code == 206
    assert len(provider.designs) == len(provider.calls) == 1


def test_blocking_qa_does_not_publish_or_autoretry_and_retries_are_bounded(setup):
    _, project, _, provider, _, _ = setup
    provider.silent = True
    for _ in range(preview.MAX_ATTEMPTS):
        result = run(setup)
        assert result["status"] == "interrupted" and result["key"] is None
    attempts = len(provider.calls)
    final = run(setup)
    assert "exhausted" in final["reason"] and len(provider.calls) == attempts
    assert len(provider.designs) == 1
    assert not load_project(project.id).voice_lab.voices


def test_favorite_changes_do_not_invalidate_preview_but_direction_changes_do(setup):
    _, project, _, provider, _, _ = setup
    first = run(setup)
    parent = load_project(project.id)
    parent.voice_lab.shortlists["role1"]["matches"][0]["score"] = 123
    save_project(parent)
    assert run(setup, request_id="rank-change")["key"] == first["key"]
    assert len(provider.calls) == 1
    parent = load_project(project.id)
    parent.voice_lab.shortlists["role1"]["matches"][0]["creative_direction"] = "Different original creative direction."
    save_project(parent)
    second = run(setup, request_id="direction-change")
    assert second["key"] != first["key"] and len(provider.calls) == 2
    assert len(load_project(project.id).voice_lab.voices) == 2


def test_original_preview_independent_of_missing_book_source_without_relabeling(setup):
    _, project, _, _, _, _ = setup
    project.voice_lab.scope = "isolated_listening_test"
    project.voice_lab.excerpt = {"status": "blocked_source", "reason": "Chapter 1 is missing"}
    save_project(project)
    result = run(setup)
    assert result["status"] == "succeeded"
    parent = load_project(project.id)
    assert parent.voice_lab.excerpt["status"] == "blocked_source"
    assert parent.voice_lab.voices[result["key"]]["details"]["source_kind"] == "original_neutral_preview"


def test_completed_audio_tamper_is_rejected_in_read_status(setup):
    root, project, _, provider, body, _ = setup
    result = run(setup)
    record = load_project(project.id).voice_lab.voices[result["key"]]
    (root / project.id / record["audio_file"]).write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="match"):
        preview.audition_status(project.id, body["request_id"])
    assert len(provider.calls) == 1


def test_changed_preview_source_cannot_resume_or_replace_original_material(setup):
    _, project, _, provider, _, _ = setup
    provider.fail = True
    result = run(setup)
    assert result["status"] == "interrupted"
    parent = load_project(project.id)
    intent = next(iter(parent.metadata["voice_lab_auditions"]["intents"].values()))
    child = load_project(intent["audition_project_id"])
    segments = load_segments(child.id)
    segments[0].text = "Different source text"
    save_segments(child.id, segments)
    count = len(provider.calls)
    provider.fail = False
    result = run(setup)
    assert result["status"] == "interrupted" and "source changed" in result["reason"]
    assert len(provider.calls) == count and result["key"] is None


def test_cancelled_queued_preview_cannot_generate_on_resume(setup):
    _, project, _, provider, body, _ = setup
    upsert_job(project.id, SynthesisJob(id="production", project_id=project.id, job_type="batch_synthesize"))
    pending = run(setup)
    cancel_job(project.id, pending["job_id"])
    cancel_job(project.id, "production")
    assert run(setup)["status"] == "cancelled"
    assert not provider.designs and not provider.calls


def test_unique_preview_budget_is_checked_before_any_model_work(setup):
    _, project, _, provider, body, _ = setup
    upsert_job(project.id, SynthesisJob(id="production", project_id=project.id, job_type="batch_synthesize"))
    for i in range(preview.MAX_AUDITIONS):
        latest = load_project(project.id)
        latest.voice_lab.shortlists["role1"]["matches"][0]["creative_direction"] = f"Original fixture direction {i}"
        save_project(latest)
        assert run(setup, request_id=f"budget-{i}")["status"] == "queued"
    latest = load_project(project.id)
    latest.voice_lab.shortlists["role1"]["matches"][0]["creative_direction"] = "One beyond the bounded budget"
    save_project(latest)
    with pytest.raises(preview.AuditionError, match="12 explicit"):
        run(setup, request_id="budget-overflow")
    assert not provider.designs and not provider.calls


@pytest.mark.parametrize("extra", [{"request_id": "../escape"}, {"unexpected": True}])
def test_api_strict_body_rejects_bad_requests_without_gpu(setup, extra):
    _, project, _, provider, body, client = setup
    response = client.post(f"/api/v1/projects/{project.id}/voice-lab/audition", json={**body, **extra})
    assert response.status_code == 422
    assert not provider.designs and not provider.calls


def test_reporting_previews_reads_the_jobs_file_once_not_once_per_preview(setup, monkeypatch):
    _, project, *_ = setup
    run(setup)
    reads, real = [], preview.load_jobs
    monkeypatch.setattr(preview, "load_jobs", lambda project_id: reads.append(project_id) or real(project_id))
    monkeypatch.setattr(preview, "get_job", lambda *args: (_ for _ in ()).throw(AssertionError("read the jobs file again")))
    rows = preview.audition_summaries(load_project(project.id))
    assert rows and rows[0]["status"] == "succeeded"
    assert reads == [project.id]


def test_a_ready_preview_is_hashed_once_while_its_file_is_unchanged_and_again_when_it_changes(setup, monkeypatch):
    root, project, _, _, body, _ = setup
    result = run(setup)
    audio = root / project.id / load_project(project.id).voice_lab.voices[result["key"]]["audio_file"]
    hashed, real = [], preview._read_verified
    monkeypatch.setattr(preview, "_read_verified", lambda path, expected: hashed.append(path) or real(path, expected))
    preview._CHECKED_AUDIO.clear()
    preview.audition_status(project.id, body["request_id"])
    preview.audition_status(project.id, body["request_id"])
    assert len(hashed) == 1, "opening the page again does not re-read every preview's audio"
    content = audio.read_bytes()
    audio.write_bytes(content)
    os.utime(audio, ns=(audio.stat().st_atime_ns, audio.stat().st_mtime_ns + 1_000_000))
    preview.audition_status(project.id, body["request_id"])
    assert len(hashed) == 2, "a file that changed is checked again"
    audio.write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="match"):
        preview.audition_status(project.id, body["request_id"])
