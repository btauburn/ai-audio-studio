"""Keeper-first behavior uses disposable voice/project stores, no inference."""
# ruff: noqa: F811 -- imported pytest fixture is intentionally used as an argument.
import copy

import pytest

from dubbing_service.app.audiobook import preferred_voices, voice_lab_audition
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.storage.projects import load_project
from tests.test_casting_voice_extension import (
    ELDER_TRAITS, build, dispatch, rate_single_voice, studio,
)


def test_keeper_round_hears_exact_historical_sample_without_any_audition(studio, monkeypatch):
    source_id = rate_single_voice(studio, name="Keeper source", stars=5)
    source_before = copy.deepcopy(load_project(source_id).model_dump(mode="json"))
    target_id = build(studio, "Keeper target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="keeper-first")
    project = load_project(target_id)
    item = project.voice_lab.casting["roles"]["role-b"]
    assert all(c["prior_match"] for c in item["round"]["candidates"])
    def forbidden(*args, **kwargs):
        pytest.fail("A historical keeper sample must not trigger model inference")
    monkeypatch.setattr(voice_lab_audition, "audition", forbidden)
    dispatch(target_id)
    project = load_project(target_id)
    item = project.voice_lab.casting["roles"]["role-b"]
    candidate = item["round"]["candidates"][0]
    assert candidate["status"] == "ready"
    assert candidate["sample_origin"] == "historical_approved_sample"
    assert candidate["production_runtime_validated"] is False
    assert candidate["audio_sha256"] == candidate["prior_evidence"]["promoting_rating"]["audio_sha256"]
    assert candidate["rating"] is None and item.get("assignment") is None
    assert next(role for role in project.roles if role.id == "role-b").voice_identity is None
    casting._round_worker(target_id, item["round"]["intent_id"])
    assert load_project(target_id).voice_lab.casting["roles"]["role-b"]["round"]["status"] == "awaiting_review"
    assert load_project(source_id).model_dump(mode="json") == source_before


@pytest.mark.parametrize("field,value", [("accent", "American"), ("perceived_vocal_age", "young adult"),
                                         ("presentation", "feminine"), ("language", "fr")])
def test_known_identity_conflicts_are_not_soft_matches(field, value):
    traits = {"presentation": "masculine", "perceived_vocal_age": "older adult", "accent": "British", "language": "en"}
    entry = {"traits": traits | {field: value}, "signal": 5}
    assert preferred_voices._score(traits, entry)[0] == 0


def test_promoting_event_remains_visible_after_more_than_five_later_ratings(studio):
    rate_single_voice(studio, name="Rating provenance", stars=5)
    entry = preferred_voices.preferred_entries()[0]
    event = entry["ratings"][0]
    for index in range(6):
        preferred_voices.record_rating(library_voice_id=entry["library_voice_id"], version=entry["version"],
            rating=2, project_id="later", role_id="later-role", key=event["key"],
            audio_sha256=event["audio_sha256"], evidence_id="later-event-" + str(index))
    proposals = preferred_voices.match_preferred(entry["traits"])
    assert proposals and proposals[0]["promoting_rating"]["event_id"] == event["event_id"]
    assert event["project_id"] in {row["project_id"] for row in proposals[0]["rated_roles"]}


def test_explicit_new_audition_bypasses_keeper_offer(studio):
    rate_single_voice(studio, name="Exploration source", stars=5)
    target_id = build(studio, "Exploration target", traits=ELDER_TRAITS, role_id="role-b")
    project = load_project(target_id)
    profile = project.character_cast.profiles["role-b"]
    plan = casting._proposal_plan(project, "role-b", profile,
        project.character_cast.dossiers.get("role-b"), profile.design_instruction,
        "Please make a new audition")
    assert plan and all(row["kind"] == "generated" for row in plan)


def test_unbound_five_star_history_cannot_inflate_bound_four_star_offer(studio):
    rate_single_voice(studio, name="Bound four source", stars=4)
    entry = preferred_voices.preferred_entries()[0]
    event = entry["ratings"][0]
    preferred_voices.record_rating(library_voice_id=entry["library_voice_id"], version=entry["version"],
        rating=5, project_id="wrong-sample", role_id="wrong-role", key="wrong-key",
        audio_sha256="0" * 64, evidence_id="unbound-five")
    proposal = preferred_voices.match_preferred(entry["traits"])[0]
    assert proposal["signal"] == proposal["highest_rating"] == proposal["promoting_rating"]["rating"] == 4
    assert preferred_voices.evidence(proposal)["prior_rating"] == 4


def test_legacy_pending_import_with_reuse_key_is_verified_without_rerender(studio, monkeypatch):
    from dubbing_service.app.storage.projects import save_project
    rate_single_voice(studio, name="Legacy keeper source", stars=5)
    target_id = build(studio, "Legacy keeper target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="legacy-keeper")
    dispatch(target_id)
    project = load_project(target_id)
    candidate = project.voice_lab.casting["roles"]["role-b"]["round"]["candidates"][0]
    assert candidate["reuse_key"]
    candidate["status"] = "pending"
    candidate.pop("sample_origin", None)
    save_project(project)
    monkeypatch.setattr(voice_lab_audition, "audition", lambda *a, **kw: pytest.fail("Legacy keeper rerendered"))
    casting._prepare_prior_imports(target_id)
    project = load_project(target_id)
    item = project.voice_lab.casting["roles"]["role-b"]
    assert item["round"]["candidates"][0]["sample_origin"] == "historical_approved_sample"
    casting._round_worker(target_id, item["round"]["intent_id"])
    assert load_project(target_id).voice_lab.casting["roles"]["role-b"]["round"]["status"] == "awaiting_review"


def test_structured_none_fit_explores_new_candidates_after_keeper_review(studio, monkeypatch):
    rate_single_voice(studio, name="None-fit source", stars=5)
    target_id = build(studio, "None-fit target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="none-fit-first")
    dispatch(target_id)
    item = load_project(target_id).voice_lab.casting["roles"]["role-b"]
    casting._round_worker(target_id, item["round"]["intent_id"])
    item = load_project(target_id).voice_lab.casting["roles"]["role-b"]
    casting.request_round(target_id, request_id="structured-none-fit", role_id="role-b",
        expected_round_revision=item["round_revision"], expected_assignment_revision=item["assignment_revision"],
        intent="none_fit")
    item = load_project(target_id).voice_lab.casting["roles"]["role-b"]
    assert all(not row["prior_match"] for row in item["round"]["candidates"])
    calls = []
    def attempted(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "failed", "job_id": "fixture-new-audition", "reason": "disposable probe"}
    monkeypatch.setattr(voice_lab_audition, "audition", attempted)
    casting._round_worker(target_id, item["round"]["intent_id"])
    assert calls and all(call.get("reuse_key") is None for call in calls)


def test_language_aliases_match_and_unknown_sample_language_is_held():
    assert preferred_voices.normalized_language("English") == "en"
    entry = {"traits": {"presentation": "masculine", "language": "English"}, "signal": 5}
    assert preferred_voices._score({"presentation": "masculine", "language": "en"}, entry)[0] > 0
    entry["traits"].pop("language")
    assert preferred_voices._score({"presentation": "masculine", "language": "en"}, entry)[0] == 0


@pytest.mark.parametrize("direction", ["Do not make a new audition", "Don't make a fresh audition",
                                      "No new audition", "Never explore new voices"])
def test_negated_exploration_keeps_heard_keeper_offer(studio, direction):
    rate_single_voice(studio, name="Retained source", stars=5)
    target_id = build(studio, "Retained target", traits=ELDER_TRAITS, role_id="role-b")
    project = load_project(target_id)
    profile = project.character_cast.profiles["role-b"]
    plan = casting._proposal_plan(project, "role-b", profile,
        project.character_cast.dossiers.get("role-b"), profile.design_instruction, direction)
    assert plan and all(row["kind"] == "prior_voice" for row in plan)
