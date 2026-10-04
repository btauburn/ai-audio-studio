"""Casting workflow regressions over the real project store, isolated fixtures only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.audiobook.pipeline import RenderRecord, requirement_for, render_utterances
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root
from dubbing_service.app.routers.voice_lab import router as voice_lab_router


@pytest.fixture
def world(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = create_project("Casting review isolated regression fixture")
    project.voice_lab.scope = "regression_fixture"
    project.roles = [Role(id="role-a", name="Alex", tts_backend="qwen3tts")]
    save_project(project)
    save_segments(project.id, [Segment(id="line-a", index=0, text="A fixture line with exact source ownership.", role_id="role-a")])
    return root, project.id


def ready_candidate(world, *, role_id="role-a", round_revision=None, candidate_index=0, key="voice-a"):
    root, project_id = world
    project = load_project(project_id)
    item = project.voice_lab.casting["roles"][role_id]
    round_ = item["round"]
    candidate = round_["candidates"][candidate_index]
    if round_revision is not None:
        round_ = next(r for r in [item["round"], *item["history"]] if r["revision"] == round_revision)
        candidate = round_["candidates"][candidate_index]
    audio = make_tone_wav(root / project_id / "voices" / f"{key}.wav", seconds=0.8)
    reference = make_tone_wav(root / project_id / "voices" / f"{key}-ref.wav", seconds=0.8,
                              frequency=220 + sum((i + 1) * ord(c) for i, c in enumerate(key)))
    prompt = root / project_id / "voices" / f"{key}.npz"
    prompt.write_bytes(b"SYNTHETIC-FIXTURE-PROMPT-CONDITIONING")
    version = voice_revision_for_role(next(r for r in project.roles if r.id == role_id))
    identity = {
        "id": f"identity-{key}", "role_id": role_id, "status": "LOCKED", "mechanism": "qwen_reference",
        "dossier_revision": "fixture-dossier", "casting_profile_revision": "fixture-profile",
        "design_instruction": "Synthetic regression fixture voice.",
        "reference_audio": reference.relative_to(root / project_id).as_posix(),
        "reference_audio_sha256": file_sha256(reference), "reference_text": "Synthetic fixture reference.",
        "prompt_artifact": prompt.relative_to(root / project_id).as_posix(), "prompt_sha256": file_sha256(prompt),
        "backend": "qwen3tts", "model": "SYNTHETIC-FIXTURE-ONLY", "model_revision": "fixture",
        "runtime_version": "fixture", "provenance": {"origin": "generated_original_voice", "fixture_only": True},
    }
    audio_hash = file_sha256(audio)
    record = {"key": key, "voice_id": f"voice-id-{key}", "version": version, "audio_file": audio.relative_to(root / project_id).as_posix(),
        "audio_sha256": audio_hash, "identity": identity, "backend": "qwen3tts", "technical_qa": "passed",
        "take_id": f"take-{key}", "sections": [], "duration_seconds": 0.8}
    candidate.update(status="ready", key=key, audio_sha256=audio_hash, identity_id=f"effective-{key}", duration_seconds=0.8,
                     technical_qa="passed", sections=[])
    project.voice_lab.voices[key] = record
    save_project(project)
    return {"event_id": "evt-" + key, "role_id": role_id, "candidate_id": candidate["candidate_id"],
            "round_revision": round_["revision"], "assignment_revision": item["assignment_revision"],
            "key": key, "audio_sha256": audio_hash, "action": "save", "raw_direction": ""}


def prepare(world, request_id="prepare-one", utterance_ids=None):
    _, project_id = world
    return casting.prepare(project_id, request_id=request_id, utterance_ids=utterance_ids)


def test_prepare_canonicalizes_alias_and_excludes_non_speaker_rows(world):
    _, project_id = world
    project = load_project(project_id)
    project.roles.append(Role(id="canonical", name="Alex Canonical", tts_backend="qwen3tts"))
    project.character_cast.role_aliases["role-a"] = "canonical"
    save_project(project)
    save_segments(project_id, [Segment(id="alias-line", index=0, text="Speaking.", role_id="role-a"),
                               Segment(id="blank", index=1, text="  ", role_id="role-a")])
    result = prepare(world)
    assert [r["role_id"] for r in result["roles"] if r["required"]] == ["canonical"]
    assert "role-a" not in load_project(project_id).voice_lab.casting["roles"]


def test_unresolved_speaker_is_rejected_without_narrator_fallback(world):
    _, project_id = world
    save_segments(project_id, [Segment(id="unknown-line", index=0, text="Nobody is identified.", role_id="")])
    with pytest.raises(casting.CastingError, match="Unresolved speaker"):
        prepare(world)


@pytest.mark.parametrize("placeholder", ["Unknown", "Unattributed", "Speaker", "Speaker 1", "speaker_1"])
def test_placeholder_role_names_are_not_cast_without_human_authoritative_attribution(world, placeholder):
    _, project_id = world
    project = load_project(project_id)
    project.roles[0].name = placeholder
    save_project(project)
    with pytest.raises(casting.CastingError, match="canonical character"):
        prepare(world)


def test_human_authoritative_attribution_can_resolve_a_placeholder_role(world):
    _, project_id = world
    project = load_project(project_id)
    project.roles[0].name = "Speaker 1"
    save_project(project)
    save_segments(project_id, [Segment(id="reviewed-line", index=0, text="Human resolved speaker.", role_id="role-a",
        attribution=Attribution(speaker_id="role-a", source="user", reviewed=True))])
    result = prepare(world)
    assert [r["role_id"] for r in result["roles"] if r["required"]] == ["role-a"]


def test_distinct_characters_with_same_display_name_get_distinct_review_rounds(world):
    _, project_id = world
    project = load_project(project_id)
    project.roles.append(Role(id="role-b", name="Alex", tts_backend="qwen3tts"))
    save_project(project)
    save_segments(project_id, [Segment(id="a", index=0, text="A.", role_id="role-a"), Segment(id="b", index=1, text="B.", role_id="role-b")])
    result = prepare(world)
    assert {r["role_id"] for r in result["roles"] if r["required"]} == {"role-a", "role-b"}
    state = load_project(project_id).voice_lab.casting
    assert state["roles"]["role-a"]["round"]["id"] != state["roles"]["role-b"]["round"]["id"]


def test_favorite_and_library_stars_do_not_change_production_fingerprint_or_assignment(world):
    _, project_id = world
    casting.render_request(project_id, request_id="fingerprint-check", utterance_ids=["line-a"])
    project = load_project(project_id)
    before = voice_revision_for_role(project.roles[0])
    segment = load_segments(project_id)[0]
    fingerprint_before = requirement_for(project, segment, role=project.roles[0], engine_id="qwen3tts", renderer_id="qwen3tts").fingerprint()
    project.voice_lab.preferences["voice-a"] = {"favorite": False, "rating": 1}
    save_project(project)
    assert voice_revision_for_role(load_project(project_id).roles[0]) == before
    project = load_project(project_id)
    project.voice_lab.preferences["voice-a"] = {"favorite": True, "rating": 5}
    save_project(project)
    latest = load_project(project_id)
    assert voice_revision_for_role(latest.roles[0]) == before
    assert requirement_for(latest, segment, role=latest.roles[0], engine_id="qwen3tts", renderer_id="qwen3tts").fingerprint() == fingerprint_before
    assert load_project(project_id).voice_lab.casting["roles"]["role-a"]["assignment"] is None


def test_atomic_low_three_feedback_is_synthetic_deduplicated_and_queues_one_unrated_replacement(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    heard_round = load_project(project_id).voice_lab.casting["roles"]["role-a"]["round"]
    # The terminal projection the round worker writes once its auditions exist.
    project = load_project(project_id)
    project.voice_lab.casting["roles"]["role-a"]["round"].update(
        status="awaiting_review", completed_at=casting.lab.now())
    save_project(project)
    body.update(stars=3, raw_direction="A little calmer", action="recast")
    saved = casting.feedback(project_id, body)
    again = casting.feedback(project_id, body)
    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    event = project.voice_lab.casting["events"][body["event_id"]]
    assert saved == again and saved["round_revision"] == 1
    assert event["origin"] == "SYNTHETIC" and event["stars"] == 3
    assert event["preference"]["state"] == "skipped_fixture"
    # A low rating never disturbs the batch being scored: the same round, the same
    # candidates, the same audio, and the rating visible on the option it belongs to.
    assert item["round_revision"] == 1 and item["round"]["id"] == heard_round["id"]
    assert item["round"]["proposal_count"] == 3
    assert [c["candidate_id"] for c in item["round"]["candidates"]] == \
        [c["candidate_id"] for c in heard_round["candidates"]]
    assert item["round"]["candidates"][0]["rating"] == 3
    assert item["round"]["candidates"][0]["needs_replacement"]["rating"] == 3
    assert item["history"] == []
    assert not any(i["round_revision"] > 1 for i in project.voice_lab.casting["intents"].values())
    request = project.voice_lab.casting["replacement_queue"]["role-a"]
    assert request["schema"] == "casting-replacement-v1" and request["round_id"] == heard_round["id"]
    assert len(request["candidates"]) == 1 and request["candidates"][0]["feedback"] == "A little calmer"
    assert request["candidates"][0]["direction"]["requested"] == ["A little calmer"]
    assert request["status"] == "waiting_for_review"

    # Rating the rest of the heard set is what releases exactly one replacement
    # round, whose candidates are unrated and whose direction retains the feedback.
    def rating_body(index, **extra):
        current = load_project(project_id).voice_lab.casting["roles"]["role-a"]
        candidate = current["round"]["candidates"][index]
        return {"event_id": f"evt-rate-{index}", "role_id": "role-a",
                "candidate_id": candidate["candidate_id"], "round_revision": current["round"]["revision"],
                "assignment_revision": current["assignment_revision"], "key": candidate["key"],
                "audio_sha256": candidate["audio_sha256"], "action": "save", "stars": 4,
                "raw_direction": "", **extra}

    for index in (1, 2):
        ready_candidate(world, candidate_index=index, key=f"voice-rest-{index}")
        casting.feedback(project_id, rating_body(index))
    casting.maintenance_tick(project_ids=[project_id], schedule=lambda *args: None)
    item = load_project(project_id).voice_lab.casting["roles"]["role-a"]
    assert item["round_revision"] == 2 and len(item["history"]) == 1
    assert item["round"]["replacement"] is True
    assert item["round"]["proposal_count"] == 3 and len(item["round"]["candidates"]) == 3
    assert all(c["rating"] is None for c in item["round"]["candidates"])
    assert item["round"]["direction"]["requested"] == ["A little calmer"]
    assert load_project(project_id).voice_lab.casting["replacement_queue"]["role-a"]["status"] == "queued"
    assert len([i for i in load_project(project_id).voice_lab.casting["intents"].values()
                if i["status"] in {"pending", "running"} and i["round_revision"] == 2]) == 1


def test_unrated_save_records_event_without_round_churn(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    before = load_project(project_id).voice_lab.casting["roles"]["role-a"]["round_revision"]
    result = casting.feedback(project_id, body)
    after = load_project(project_id).voice_lab.casting["roles"]["role-a"]
    assert result["saved"] and after["round_revision"] == before
    assert after["round"]["candidates"][0]["rating"] is None


def test_old_tab_feedback_is_historical_conflict_and_preserves_current_assignment(world):
    _, project_id = world
    prepare(world)
    stale = ready_candidate(world)
    stale.update(stars=2, action="save")
    casting.feedback(project_id, stale)
    # An explicit recast carries no low rating, so it still replaces the round
    # immediately; the rejected rating stays durable history.
    recast = ready_candidate(world)
    casting.feedback(project_id, {**recast, "event_id": "evt-recast", "stars": None, "action": "recast"})
    current = ready_candidate(world, round_revision=2, key="voice-b")
    current.update(event_id="evt-current", stars=5, action="save")
    casting.feedback(project_id, current)
    stale["event_id"] = "evt-old-tab"
    old = casting.feedback(project_id, stale)
    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    assert old["conflict"] is True
    assert item["assignment"] == {"key": "voice-a", "voice_revision": item["assignment"]["voice_revision"],
        "approved": False, "rejected": True, "event_id": "evt-voice-a", "revision": 1}
    assert project.voice_lab.casting["events"]["evt-old-tab"]["origin"] == "SYNTHETIC"


def test_feedback_for_one_role_preserves_unrelated_approved_role(world):
    _, project_id = world
    project = load_project(project_id)
    approved_ids = [f"role-{letter}" for letter in "bcde"]
    for letter, role_id in zip("bcde", approved_ids):
        other = Role(id=role_id, name="Alex", tts_backend="qwen3tts")
        project.roles.append(other)
        version = voice_revision_for_role(other)
        event_id = f"evt-approved-{letter}"
        project.voice_lab.casting["roles"][role_id] = {"assignment_revision": 1, "round_revision": 0,
            "history": [], "round": {}, "rejections": [], "assignment": {"key": f"other-key-{letter}", "voice_revision": version,
                "approved": True, "rejected": False, "event_id": event_id, "stars": 5, "revision": 1}}
        project.voice_lab.casting["events"][event_id] = {"actor": "brian", "origin": "SYNTHETIC", "stars": 5}
    save_project(project)
    prepare(world)
    body = ready_candidate(world)
    body.update(stars=2, event_id="evt-low-for-a")
    casting.feedback(project_id, body)
    saved = load_project(project_id)
    assert all(casting.approved(saved, role_id) for role_id in approved_ids)
    assert [saved.voice_lab.casting["roles"][role_id]["assignment"]["key"] for role_id in approved_ids] == [f"other-key-{letter}" for letter in "bcde"]


def test_four_star_use_requires_exact_usable_identity_and_approves_fixture_only_scope(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    body.update(stars=4, action="use")
    result = casting.feedback(project_id, body)
    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    assert result["saved"] and item["assignment"]["approved"]
    assert item["assignment"]["key"] == body["key"]
    assert casting.approved(project, "role-a")


def test_use_materializes_exact_qwen_identity_assets_for_provider_namespace(world, monkeypatch):
    root, project_id = world
    prepare(world)
    body = ready_candidate(world, key="review-library-voice")
    casting.feedback(project_id, {**body, "stars": 5, "action": "use"})

    project = load_project(project_id)
    identity = project.roles[0].voice_identity.model_dump(mode="json")
    prefix = f"casting/voices/{identity['id']}/"
    assert identity["reference_audio"].startswith(prefix)
    assert identity["prompt_artifact"].startswith(prefix)
    assert (root / project_id / identity["reference_audio"]).read_bytes() == \
        (root / project_id / "voices" / "review-library-voice-ref.wav").read_bytes()
    assert (root / project_id / identity["prompt_artifact"]).read_bytes() == \
        (root / project_id / "voices" / "review-library-voice.npz").read_bytes()
    # Assignment rewrites only the role identity; the reviewed audition record
    # remains anchored to its original immutable artifacts.
    reviewed_identity = project.voice_lab.voices[body["key"]]["identity"]
    assert reviewed_identity["id"] == identity["id"]
    assert reviewed_identity["reference_audio"] == "voices/review-library-voice-ref.wav"

    # Exercise the real provider's project-local path and checksum boundary,
    # then stop before prompt parsing/model metadata or any model load.
    from service import qwen_character_voice
    project_root = root / project_id
    monkeypatch.setattr(qwen_character_voice, "safe_project_root", lambda _: project_root)
    reached_prompt_validation = RuntimeError("verified assets reached prompt validation")
    monkeypatch.setattr(qwen_character_voice, "load_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(reached_prompt_validation))
    runtime = qwen_character_voice.CharacterVoiceRuntime(project_root, lambda _: pytest.fail("no model load expected"))
    with pytest.raises(RuntimeError, match="verified assets reached prompt validation"):
        runtime.synthesize(project_id, identity["id"], identity["prompt_artifact"], identity["prompt_sha256"],
            identity["reference_audio"], identity["reference_audio_sha256"], "Synthetic fixture line.", "English")


def test_explicit_resume_resets_run_budget_and_preserves_attempt_history(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world, key="resume-budget-voice")
    casting.feedback(project_id, {**body, "stars": 5, "action": "use"})
    casting.render_request(project_id, request_id="strict-resume-budget", utterance_ids=["line-a"])
    project = load_project(project_id)
    receipt = project.voice_lab.casting["productions"]["strict-resume-budget"]
    receipt.update(status="failed", attempts=3, total_attempts=3,
                   reason="Bounded production attempts exhausted; last failure: provider contract mismatch",
                   last_failure_reason="provider contract mismatch",
                   manifest_history=[{"id": "previous-manifest"}])
    save_project(project)

    casting.resume_request(project_id, "strict-resume-budget")
    receipt = load_project(project_id).voice_lab.casting["productions"]["strict-resume-budget"]
    assert receipt["status"] == "awaiting_review"
    assert receipt["attempts"] == 0 and receipt["total_attempts"] == 3 and receipt["resume_count"] == 1
    assert receipt["last_failure_reason"] == "provider contract mismatch"
    assert "provider contract mismatch" in receipt["reason"]
    assert receipt["manifest_history"] == [{"id": "previous-manifest"}]

    from dubbing_service.app.storage.jobs import mark_job_active
    scheduled = []
    casting.maintenance_tick(project_ids=[project_id],
        schedule=lambda fn, pid, arg: scheduled.append((fn.__name__, pid, arg)))
    resumed = load_project(project_id).voice_lab.casting["productions"]["strict-resume-budget"]
    assert any(fn == "_production_worker" and arg == "strict-resume-budget" for fn, _, arg in scheduled)
    assert resumed["status"] == "queued" and resumed["attempts"] == 0
    assert resumed["total_attempts"] == 3
    assert resumed["manifest_history"][0] == {"id": "previous-manifest"}
    mark_job_active(project_id, receipt["job_id"], False)


@pytest.mark.parametrize("active_status", ["queued", "running"])
def test_resume_does_not_reset_active_production_attempt(world, active_status):
    _, project_id = world
    casting.render_request(project_id, request_id="strict-active-resume", utterance_ids=["line-a"])
    project = load_project(project_id)
    receipt = project.voice_lab.casting["productions"]["strict-active-resume"]
    receipt.update(status=active_status, attempts=2, total_attempts=5, resume_count=1,
                   manifest={"id": "active-manifest"}, manifest_history=[{"id": "older-manifest"}])
    save_project(project)
    before = load_project(project_id).voice_lab.casting["productions"]["strict-active-resume"]

    with pytest.raises(casting.CastingError, match="already queued or running"):
        casting.resume_request(project_id, "strict-active-resume")

    after = load_project(project_id).voice_lab.casting["productions"]["strict-active-resume"]
    assert after["status"] == active_status
    assert after["attempts"] == before["attempts"] == 2
    assert after["total_attempts"] == before["total_attempts"] == 5
    assert after["resume_count"] == before["resume_count"] == 1
    assert after["manifest"] == before["manifest"] == {"id": "active-manifest"}
    assert after["manifest_history"] == before["manifest_history"] == [{"id": "older-manifest"}]


def test_exhausted_production_keeps_last_provider_failure_reason(world):
    _, project_id = world
    project = load_project(project_id)
    role = project.roles[0]
    event_id = "evt-approved-for-exhaustion"
    project.voice_lab.casting["roles"]["role-a"] = {
        "assignment_revision": 1, "round_revision": 0, "history": [], "round": {},
        "rejections": [], "assignment": {"key": "approved-key", "voice_revision": voice_revision_for_role(role),
            "approved": True, "rejected": False, "event_id": event_id, "stars": 5, "revision": 1}}
    project.voice_lab.casting["events"][event_id] = {"actor": "brian", "origin": "SYNTHETIC", "stars": 5}
    save_project(project)
    casting.render_request(project_id, request_id="strict-exhausted", utterance_ids=["line-a"])
    project = load_project(project_id)
    receipt = project.voice_lab.casting["productions"]["strict-exhausted"]
    receipt.update(status="awaiting_review", attempts=3, total_attempts=3,
                   reason="provider refused audition artifact", last_failure_reason="provider refused audition artifact")
    save_project(project)

    casting.maintenance_tick(project_ids=[project_id], schedule=lambda *_: None)
    receipt = load_project(project_id).voice_lab.casting["productions"]["strict-exhausted"]
    assert receipt["status"] == "failed"
    assert "Bounded production attempts exhausted" in receipt["reason"]
    assert "provider refused audition artifact" in receipt["reason"]


def test_performance_only_base_direction_retains_identity_and_marks_unsupported():
    identity = {"mechanism": "qwen_reference", "backend": "qwen3tts", "reference_audio_sha256": "b" * 64,
                "model_revision": "fixture-model", "clone_mode": "icl"}
    parsed = casting._direction("Make it calmer and slower", backend="qwen3tts", identity=identity)
    assert parsed["performance_only"] is True
    assert parsed["unsupported"]
    assert parsed["applied"] == []
    assert parsed["retained"] == []


def test_relative_comparison_needs_two_exact_baselines():
    absent = casting._direction("Go back to the first voice", baselines=[])
    one = casting._direction("Go back to the first voice", baselines=["candidate-a"])
    exact = casting._direction("Go back to the first voice", baselines=["candidate-a", "candidate-b"])
    assert absent["unsupported"] and one["unsupported"]
    assert not exact["unsupported"]
    assert exact["baseline_candidate_ids"] == ["candidate-a", "candidate-b"]


def test_feedback_persists_recovery_intent_before_any_dispatch(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    body.update(stars=3)
    casting.feedback(project_id, body)
    state = load_project(project_id).voice_lab.casting
    assert any(i["status"] == "pending" for i in state["intents"].values())


def test_stale_round_worker_cannot_publish_its_late_audition(world, monkeypatch):
    _, project_id = world
    prepare(world)
    intent_id = next(iter(load_project(project_id).voice_lab.casting["intents"]))

    def late_success(pid, **kwargs):
        project = load_project(pid)
        project.voice_lab.casting["roles"]["role-a"]["round_revision"] += 1
        save_project(project)
        return {"status": "succeeded", "key": "late-audio", "audio_sha256": "a" * 64, "duration_seconds": 0.8}

    from dubbing_service.app.audiobook import voice_lab_audition
    monkeypatch.setattr(voice_lab_audition, "audition", late_success)
    casting._round_worker(project_id, intent_id)
    current = load_project(project_id).voice_lab.casting["roles"]["role-a"]
    assert current["round"]["candidates"][0]["status"] == "pending"


def test_comparison_feedback_without_exact_baselines_is_not_claimed_as_applied(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    body["raw_direction"] = "Go back to the first voice."
    result = casting.feedback(project_id, body)
    assert result["direction"]["unsupported"]
    assert result["direction"]["applied"] == []


def test_parked_production_source_edit_requires_explicit_revalidation(world):
    _, project_id = world
    request = casting.render_request(project_id, request_id="strict-one", utterance_ids=["line-a"])
    assert request["productions"][0]["status"] == "awaiting_review"
    rows = load_segments(project_id)
    rows[0].text = "Changed while the cast is parked."
    save_segments(project_id, rows)
    dispatched = []
    actions = casting.maintenance_tick(project_ids=[project_id], schedule=lambda fn, pid, arg: dispatched.append(fn.__name__))
    status = load_project(project_id).voice_lab.casting["productions"]["strict-one"]["status"]
    assert status == "source_changed" and "_production_worker" not in dispatched
    assert all(row["job_id"].startswith("cast_") for row in actions)


def test_resume_revalidates_source_without_auto_approving_or_inheriting_rating(world):
    _, project_id = world
    casting.render_request(project_id, request_id="strict-one", utterance_ids=["line-a"])
    rows = load_segments(project_id)
    rows[0].text = "Changed source."
    save_segments(project_id, rows)
    casting.maintenance_tick(project_ids=[project_id], schedule=lambda *_: None)
    casting.resume_request(project_id, "strict-one")
    receipt = load_project(project_id).voice_lab.casting["productions"]["strict-one"]
    assert receipt["status"] == "awaiting_review"
    assert not casting.approved(load_project(project_id), "role-a")


def test_inflight_strict_take_is_retained_unselected_when_approval_changes(world, monkeypatch):
    _, project_id = world
    project = load_project(project_id)
    state = project.voice_lab.casting
    event_id = "evt-approved"
    role = project.roles[0]
    state["roles"]["role-a"] = {"assignment_revision": 1, "round_revision": 0, "history": [], "round": {},
        "rejections": [], "assignment": {"key": "approved-key", "voice_revision": voice_revision_for_role(role),
            "approved": True, "rejected": False, "event_id": event_id, "stars": 5, "revision": 1}}
    state["events"][event_id] = {"actor": "brian", "origin": "SYNTHETIC", "stars": 5}
    rows = load_segments(project_id)
    manifest = casting.freeze_manifest(project, rows, ["role-a"])
    save_project(project)

    def finish_while_revoking(_project, segment, **kwargs):
        latest = load_project(project_id)
        latest.voice_lab.casting["roles"]["role-a"]["assignment"].update(approved=False, rejected=True)
        save_project(latest)
        segment.takes.append(SegmentTake(id="inflight-take", audio_file="voices/inflight.wav", audio_sha256="c" * 64,
                                        text=segment.text, role_id="role-a", tts_backend="qwen3tts"))
        return RenderRecord(segment_id=segment.id, take_id="inflight-take", success=True, audio_sha256="c" * 64, qa_passed=True)

    observed = {}
    def commit(*args, **kwargs):
        observed["promote"] = kwargs["promote"]
        return SimpleNamespace(selected=False, stale=False, stale_components=[], reason=None)

    class Renderer:
        backend = "qwen3tts"
        engine_id = "qwen3tts"
        renderer_id = "qwen3tts"

    from dubbing_service.app.audiobook import pipeline
    monkeypatch.setattr(pipeline, "render_segment", finish_while_revoking)
    monkeypatch.setattr(pipeline.persistence_module, "commit_take_result", commit)
    result = render_utterances(project_id, ["line-a"], renderer=Renderer(), promote=True, casting_manifest=manifest)
    assert len(result) == 1 and not result[0].selected
    assert observed["promote"] is False
    assert load_segments(project_id)[0].selected_take_id is None


def test_real_voice_lab_casting_routes_validate_and_surface_historical_conflict(world, monkeypatch):
    _, project_id = world
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    base = f"/api/v1/projects/{project_id}/voice-lab/casting"
    assert client.get(base).status_code == 200
    assert client.post(base + "/prepare", json={"request_id": "api-prepare"}).status_code == 200
    body = ready_candidate(world)
    body.update(stars=2, action="save")
    response = client.post(base + "/feedback", json=body)
    assert response.status_code == 200
    # A below-threshold rating is recorded and the reviewed round is left intact
    # for the remaining options; the replacement is a durable request, not a
    # disruptive rewrite of what the human is scoring.
    state = client.get(base).json()["roles"][0]
    assert state["round_revision"] == 1
    assert state["candidates"][0]["rating"] == 2
    assert state["replacement_request"]["status"] == "waiting_for_review"
    old = dict(body, event_id="api-old-tab")
    conflict = client.post(base + "/feedback", json=old)
    assert conflict.status_code == 409 and conflict.json()["conflict"] is True
    invalid = client.post(base + "/feedback", json={**body, "unexpected": True})
    assert invalid.status_code == 422


def test_role_round_generate_is_durable_and_duplicate_tabs_join_one_intent(world, monkeypatch):
    _, project_id = world
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    base = f"/api/v1/projects/{project_id}/voice-lab/casting/round"
    assert client.get(f"/api/v1/projects/{project_id}/voice-lab/casting").json()["capabilities"]["round_requests"] is True
    body = {"request_id": "generate-one", "role_id": "role-a", "expected_round_revision": 0,
        "expected_assignment_revision": 0, "intent": "generate"}

    first = client.post(base, json=body)
    assert first.status_code == 200
    receipt = first.json()["receipt"]
    assert receipt["created"] and receipt["round_revision"] == 1
    assert first.json()["casting"]["roles"][0]["round_status"] == "pending"

    replay = client.post(base, json=body)
    assert replay.status_code == 200 and replay.json()["receipt"] == receipt
    mismatch = client.post(base, json={**body, "intent": "none_fit"})
    assert mismatch.status_code == 409
    other_tab = client.post(base, json={**body, "request_id": "generate-two",
        "expected_round_revision": 1})
    assert other_tab.status_code == 200
    assert other_tab.json()["receipt"]["round_id"] == receipt["round_id"]
    assert other_tab.json()["receipt"]["created"] is False
    state = load_project(project_id).voice_lab.casting
    assert state["enabled"] is True
    assert state["roles"]["role-a"]["round_revision"] == 1
    assert len(state["intents"]) == 1


def test_role_round_cas_conflict_has_no_state_mutation(world, monkeypatch):
    _, project_id = world
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    before = load_project(project_id).voice_lab.casting.copy()
    response = client.post(f"/api/v1/projects/{project_id}/voice-lab/casting/round", json={
        "request_id": "stale-round", "role_id": "role-a", "expected_round_revision": 2,
        "expected_assignment_revision": 0, "intent": "generate"})
    assert response.status_code == 409
    assert load_project(project_id).voice_lab.casting == before


def test_none_fit_replaces_blocked_round_without_rating_or_assignment_mutation(world, monkeypatch):
    _, project_id = world
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    prepare(world)
    project = load_project(project_id)
    state = project.voice_lab.casting
    item = state["roles"]["role-a"]
    prior_candidates = item["round"]["candidates"]
    for candidate in prior_candidates:
        candidate.update(status="blocked", reason="No audio was produced")
    item["round"]["status"] = "blocked"
    state["intents"][item["round"]["intent_id"]]["status"] = "complete"
    save_project(project)
    before_assignment = item.get("assignment")

    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    base = f"/api/v1/projects/{project_id}/voice-lab/casting/round"
    body = {"request_id": "none-fit-one", "role_id": "role-a", "expected_round_revision": 1,
        "expected_assignment_revision": 0, "intent": "none_fit"}
    response = client.post(base, json=body)
    assert response.status_code == 200
    receipt = response.json()["receipt"]
    latest = load_project(project_id).voice_lab.casting
    current = latest["roles"]["role-a"]
    assert receipt["created"] and receipt["round_revision"] == 2
    assert current["history"][0]["candidates"] == prior_candidates
    assert all(candidate["rating"] is None for candidate in current["round"]["candidates"])
    assert current.get("assignment") == before_assignment is None
    assert latest["events"] == {}

    replay = client.post(base, json=body)
    assert replay.status_code == 200 and replay.json()["receipt"] == receipt
    assert load_project(project_id).voice_lab.casting["roles"]["role-a"]["round_revision"] == 2


def test_none_fit_excludes_ready_identity_without_creating_rating_event_or_assignment(world, monkeypatch):
    _, project_id = world
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    prepare(world)
    ready_candidate(world, candidate_index=0, key="heard-voice")
    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    item["round"]["status"] = "awaiting_review"
    project.voice_lab.casting["intents"][item["round"]["intent_id"]]["status"] = "complete"
    save_project(project)

    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    response = client.post(f"/api/v1/projects/{project_id}/voice-lab/casting/round", json={
        "request_id": "none-fit-heard", "role_id": "role-a", "expected_round_revision": 1,
        "expected_assignment_revision": 0, "intent": "none_fit"})
    assert response.status_code == 200
    project = load_project(project_id)
    state = project.voice_lab.casting
    item = state["roles"]["role-a"]
    exclusion = next(row for row in item["rejections"] if row.get("request_id") == "none-fit-heard")
    assert exclusion == {"request_id": "none-fit-heard", "reason": "none_fit", "key": "heard-voice",
        "version": None, "effective_identity": "effective-heard-voice"}
    assert item["history"][0]["candidates"][0]["rating"] is None
    assert item.get("assignment") is None and state["events"] == {}
    assert all(candidate["rating"] is None for candidate in item["round"]["candidates"])


def test_none_fit_refuses_active_round_and_allows_bounded_manual_replacement_rounds(world, monkeypatch):
    _, project_id = world
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    prepare(world)
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    base = f"/api/v1/projects/{project_id}/voice-lab/casting/round"
    pending = {"request_id": "too-soon", "role_id": "role-a", "expected_round_revision": 1,
        "expected_assignment_revision": 0, "intent": "none_fit"}
    before = load_project(project_id).voice_lab.casting.copy()
    response = client.post(base, json=pending)
    assert response.status_code == 409
    assert load_project(project_id).voice_lab.casting == before

    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    item["round"]["status"] = "blocked"
    project.voice_lab.casting["intents"][item["round"]["intent_id"]]["status"] = "complete"
    save_project(project)
    first = client.post(base, json={**pending, "request_id": "first-replacement"})
    assert first.status_code == 200

    project = load_project(project_id)
    item = project.voice_lab.casting["roles"]["role-a"]
    item["round"]["status"] = "blocked"
    project.voice_lab.casting["intents"][item["round"]["intent_id"]]["status"] = "complete"
    save_project(project)
    second = client.post(base, json={"request_id": "second-replacement", "role_id": "role-a",
        "expected_round_revision": 2, "expected_assignment_revision": 0, "intent": "none_fit"})
    assert second.status_code == 200
    state = load_project(project_id).voice_lab.casting
    item = state["roles"]["role-a"]
    assert item["round_revision"] == 3 and len(item["history"]) == 2
    assert all(r["proposal_count"] <= casting.MAX_PROPOSALS for r in [*item["history"], item["round"]])
    assert all(candidate["rating"] is None for candidate in item["round"]["candidates"])
