"""Evidence policy regressions: synthetic dossiers, no model or live data."""
from __future__ import annotations

import copy

import pytest

from dubbing_service.app.audiobook import (
    casting_anchors,
    casting_inference,
    casting_review,
)
from dubbing_service.app.audiobook.casting_profiles import build_profiles
from dubbing_service.app.models.character import CastingDecision, StoryScope
from tests.test_character_casting import fact, project_and_segments


def bundle(facts, *, max_chapter=0, allow_spoilers=False, adaptation_requested=False):
    project, segments = project_and_segments(facts)
    policy = project.character_cast.policy
    policy.max_chapter = max_chapter
    policy.allow_spoilers = allow_spoilers
    policy.adaptation_requested = adaptation_requested
    profile = build_profiles(project, segments)["r0"]
    project.character_cast.profiles["r0"] = profile
    return project, profile, project.character_cast.dossiers["r0"]


def traits(project, profile, dossier):
    policy = project.character_cast.policy
    supported = casting_inference.supported_traits(profile, dossier, policy=policy)
    query = casting_anchors.vocal_traits(profile, dossier, policy=policy)
    assert query == {name: row["value"] for name, row in supported.items()}
    return supported, query


def test_book_supported_fact_beats_external_explicit_fact_in_both_paths():
    project, profile, dossier = bundle([
        fact("gender", "female", source_kind="official", identifier="external"),
        fact("gender", "male", level="SUPPORTED", identifier="book"),
    ])
    supported, query = traits(project, profile, dossier)
    assert query["gender"] == "male"
    assert query["presentation"] == "masculine"
    assert supported["gender"]["evidence_ids"] == ["book"]


def test_equal_priority_conflicting_evidence_abstains():
    project, profile, dossier = bundle([
        fact("gender", "female", identifier="first"),
        fact("gender", "male", identifier="second"),
    ])
    supported, query = traits(project, profile, dossier)
    assert "gender" not in supported and "presentation" not in query


@pytest.mark.parametrize("field,value", [("age", 17), ("accent", "British"), ("pronouns", "she/her")])
def test_future_evidence_is_not_reintroduced(field, value):
    project, profile, dossier = bundle([
        fact(field, value, scope=StoryScope(chapter_start=9), identifier="later"),
    ])
    supported, query = traits(project, profile, dossier)
    assert field not in query
    assert all("later" not in row["evidence_ids"] for row in supported.values())


def test_adaptation_evidence_requires_opt_in_even_with_spoiler_permission():
    project, profile, dossier = bundle([
        fact("gender", "female", source_kind="adaptation"),
    ], allow_spoilers=True)
    assert "gender" not in traits(project, profile, dossier)[1]
    project.character_cast.policy.adaptation_requested = True
    assert traits(project, profile, dossier)[1]["gender"] == "female"


@pytest.mark.parametrize("unsafe", ["clone actor voice", "British " * 20])
def test_raw_accent_cannot_bypass_profile_sanitization(unsafe):
    project, profile, dossier = bundle([fact("accent", unsafe)])
    assert "accent" not in traits(project, profile, dossier)[1]


def test_numeric_age_and_explicit_safe_accent_remain_usable():
    project, profile, dossier = bundle([fact("age", 17), fact("accent", "British")])
    supported, query = traits(project, profile, dossier)
    assert query["age"] == "17" and query["accent"] == "British"
    assert query["perceived_vocal_age"] == "adolescent"
    assert supported["age"]["evidence_ids"] == ["age"]


def test_lowered_policy_filters_cached_scalar_and_acoustic_profile_decisions():
    project, profile, dossier = bundle([
        fact("gender", "female", scope=StoryScope(chapter_start=9), identifier="later-gender"),
        fact("accent", "British", scope=StoryScope(chapter_start=9), identifier="later-accent"),
        fact("voice_description", "deep voice", scope=StoryScope(chapter_start=9), identifier="later-voice"),
    ], max_chapter=9)
    before = copy.deepcopy(profile.model_dump())
    assert traits(project, profile, dossier)[1]["pitch"] == "low"
    project.character_cast.policy.max_chapter = 0
    assert traits(project, profile, dossier)[1] == {}
    assert profile.model_dump() == before  # The durable reviewed profile is not rewritten.


def test_compatible_acoustic_evidence_still_combines():
    project, profile, dossier = bundle([
        fact("voice_description", "deep", identifier="book-pitch"),
        fact("cadence", "slowly", identifier="book-cadence"),
        fact("voice_description", "high-pitched, quick and raspy", source_kind="official", identifier="official"),
    ])
    query = traits(project, profile, dossier)[1]
    assert query["pitch"] == "low" and query["cadence"] == "deliberate"
    assert query["texture"] == "rough"


def test_production_generated_concepts_use_actual_project_policy():
    project, profile, dossier = bundle([fact("gender", "female")])
    # The actual policy permits future facts; a default policy would omit them.
    future = fact("age", 17, scope=StoryScope(chapter_start=9), identifier="later")
    dossier.facts.append(future)
    project.character_cast.policy.allow_spoilers = True
    proposals = casting_review._generated_proposals(
        project, profile, dossier, project.roles[0], "Create an original voice.", "", count=2, priors=[])
    assert len(proposals) == 2
    assert all(row["concept"]["supported"]["age"]["value"] == "17" for row in proposals)
    assert all(row["concept"]["supported"]["age"]["evidence_ids"] == ["later"] for row in proposals)


def test_production_preferred_voice_search_and_rating_capture_use_policy(monkeypatch):
    project, profile, dossier = bundle([fact("gender", "female")])
    dossier.facts.append(fact("age", 17, scope=StoryScope(chapter_start=9), identifier="later"))
    project.character_cast.policy.allow_spoilers = True
    captured = []
    from dubbing_service.app.audiobook import preferred_voices
    monkeypatch.setattr(preferred_voices, "match_preferred", lambda query, **kwargs: captured.append(query) or [])
    monkeypatch.setattr(casting_review, "_generated_proposals", lambda *args, **kwargs: [])
    casting_review._proposal_plan(project, "r0", profile, dossier, "Create an original voice.", "")
    assert captured[0]["age"] == "17"
    preference = casting_review._preference_write(
        project, "r0", project.roles[0], {"key": "synthetic"}, {"event_id": "fixture"}, 5, "approve")
    assert preference["traits"]["age"] == "17"


def test_profile_only_evidence_and_editorial_placeholders_keep_existing_behavior():
    _project, profile, _dossier = bundle([])
    profile.decisions = [
        CastingDecision(dimension="cadence", value="deliberate", level="EXPLICIT",
                        reason="fixture evidence", evidence_ids=["fixture"]),
        CastingDecision(dimension="pitch", value="mid", reason="placeholder"),
    ]
    assert casting_anchors.vocal_traits(profile, None) == {"cadence": "deliberate"}


def test_default_policy_is_safe_for_callers_without_project_context():
    _project, profile, dossier = bundle([fact("gender", "female", source_kind="adaptation")])
    assert casting_anchors.vocal_traits(profile, dossier) == {}
    assert casting_inference.supported_traits(profile, dossier) == {}


def test_supplied_empty_dossier_cannot_authorize_stale_profile_evidence():
    project, profile, dossier = bundle([fact("voice_description", "deep voice")])
    dossier.facts = []
    assert traits(project, profile, dossier)[1] == {}
    assert casting_anchors.vocal_traits(profile, None)["pitch"] == "low"


def test_new_book_pitch_invalidates_cached_external_pitch_without_dropping_texture():
    project, profile, dossier = bundle([
        fact("voice_description", "deep and raspy", source_kind="official", identifier="external"),
    ])
    assert traits(project, profile, dossier)[1]["pitch"] == "low"
    dossier.facts.append(fact("voice_description", "high-pitched", identifier="book"))
    query = traits(project, profile, dossier)[1]
    assert "pitch" not in query  # Stale interpretation is excluded pending profile refresh.
    assert query["texture"] == "rough"  # Compatible complementary evidence remains usable.


def test_new_numeric_age_cannot_be_reinterpreted_as_unspecified_adult():
    project, profile, dossier = bundle([fact("gender", "female")])
    dossier.facts.append(fact("age", 17))
    concepts = casting_inference.casting_concepts(profile, dossier, count=2,
                                                  policy=project.character_cast.policy)
    assert all(concept["supported"]["perceived_vocal_age"]["value"] == "adolescent"
               and "perceived_vocal_age" not in concept["inferred"] for concept in concepts)


def test_new_round_uses_current_policy_for_base_prompt_without_rewriting_profile(monkeypatch):
    project, profile, dossier = bundle([
        fact("voice_description", "deep voice", scope=StoryScope(chapter_start=9)),
        fact("age", 75, scope=StoryScope(chapter_start=9)),
        fact("accent", "British", scope=StoryScope(chapter_start=9)),
    ], max_chapter=9)
    _unused, segments = project_and_segments([])
    before_profile = copy.deepcopy(profile.model_dump())
    before_dossier = copy.deepcopy(dossier.model_dump())
    assert "pitch: low." in profile.design_instruction
    project.character_cast.policy.max_chapter = 0
    captured = []
    monkeypatch.setattr(casting_review, "load_segments", lambda _project_id: segments)
    def capture(_project, _role_id, current, _dossier, base, _retained, **_kwargs):
        captured.append((current, base))
        return []
    monkeypatch.setattr(casting_review, "_proposal_plan", capture)
    casting_review._new_round(project, "r0")
    current, base = captured[0]
    assert current is not profile
    assert "pitch: low." not in base
    assert "perceived vocal age: older adult." not in base
    assert "accent: British." not in base
    assert base == current.design_instruction
    assert profile.model_dump() == before_profile
    assert dossier.model_dump() == before_dossier


def test_new_round_reinterprets_new_book_gender_without_rotating_against_stale_external_gender(monkeypatch):
    project, profile, dossier = bundle([
        fact("gender", "female", source_kind="official", identifier="external"),
    ])
    dossier.facts.append(fact("gender", "male", level="SUPPORTED", identifier="book"))
    _unused, segments = project_and_segments([])
    before_profile = copy.deepcopy(profile.model_dump())
    before_roles = [role.model_dump() for role in project.roles]
    monkeypatch.setattr(casting_review, "load_segments", lambda _project_id: segments)
    def capture(current_project, role_id, current, current_dossier, base, retained, **kwargs):
        return casting_review._generated_proposals(current_project, current, current_dossier,
            current_project.roles[0], base, retained, count=2, priors=[])
    monkeypatch.setattr(casting_review, "_proposal_plan", capture)
    round_ = casting_review._new_round(project, "r0")
    assert len(round_["candidates"]) == 2
    for candidate in round_["candidates"]:
        assert candidate["concept"]["supported"]["presentation"]["value"] == "masculine"
        assert "presentation" not in candidate["concept"]["inferred"]
    assert profile.model_dump() == before_profile
    assert [role.model_dump() for role in project.roles] == before_roles


def test_ingested_book_evidence_reaches_distinct_performer_proposals(monkeypatch, tmp_path):
    from tests.test_character_casting_integration import ingested
    from dubbing_service.app.audiobook import preferred_voices
    from dubbing_service.app.storage.segments import load_segments

    # Project-root isolation does not isolate the global preferred-voice store.
    # Keep its real matcher, but give this synthetic ingest an empty library.
    monkeypatch.setattr(preferred_voices.library, "load_voices", lambda: [])
    _root, project, role = ingested(monkeypatch, tmp_path)
    dossier = project.character_cast.dossiers[role.id]
    gender = next(row for row in dossier.facts
                  if row.field == "gender" and row.source_kind == "book" and row.level == "EXPLICIT")
    assert gender.value == "female"
    assert gender.source_revision_id and gender.chapter_id and gender.location and gender.excerpt
    evidence_segment_id = gender.location.split(";", 1)[0].removeprefix("segment:")
    evidence_segment = next(row for row in load_segments(project.id) if row.id == evidence_segment_id)
    assert gender.excerpt in evidence_segment.text
    assert gender.source_revision_id == evidence_segment.source_anchor.source_revision_id
    assert gender.chapter_id == evidence_segment.chapter_id
    dossier.facts.insert(0, fact("gender", "male", source_kind="official", identifier="external-lore"))
    before_profile = project.character_cast.profiles[role.id].model_dump()
    before_role = role.model_dump()
    round_ = casting_review._new_round(project, role.id)
    candidates = round_["candidates"]
    assert len(candidates) >= 2
    assert len({row["inspiration"]["performer"] for row in candidates}) == len(candidates)
    for candidate in candidates:
        supported = candidate["concept"]["supported"]
        assert supported["gender"]["value"] == "female"
        assert supported["gender"]["evidence_ids"] == [gender.id]
        assert supported["presentation"]["value"] == "feminine"
        assert candidate["inspiration"]["performance"] and candidate["inspiration"]["source_urls"]
    assert role.model_dump() == before_role
    assert project.character_cast.profiles[role.id].model_dump() == before_profile
