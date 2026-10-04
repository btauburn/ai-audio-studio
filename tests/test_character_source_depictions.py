"""Conservative source depictions: synthetic prose and no model or live data."""
from __future__ import annotations

import copy

import pytest

from dubbing_service.app.audiobook import casting_review
from dubbing_service.app.audiobook import character_research as research
from dubbing_service.app.audiobook.character_casting import research_and_profile
from dubbing_service.app.models.character import CharacterVoiceIdentity
from tests.test_character_research import book, fact


@pytest.mark.parametrize("text,age,gender", [
    ("See Alice as she was, the quiet young girl, ten years old.", 10, "female"),
    ("Consider Alice as he is, a tall boy, twelve years old.", 12, "male"),
    ("See Alice as they were, the teenager, seventeen years old.", 17, None),
    ("See Alice as they are, a person, twenty-one years old.", 21, None),
    ("See Alice as she was, the ten-year-old girl, ten years old.", 10, "female"),
])
def test_named_depiction_extracts_only_explicit_age_and_noun_gender(text, age, gender):
    project, segments = book(text)
    research.research_project(project, segments)
    age_fact = fact(project, "age")
    assert age_fact.value == age and age_fact.level == "EXPLICIT"
    assert age_fact.source_kind == "book"
    assert age_fact.source_revision_id == "src1" and age_fact.chapter_id == "c0"
    assert age_fact.location == "segment:s0;line:0" and age_fact.excerpt == text[text.index("Alice"):-1]
    assert age_fact.scope.chapter_start == 0 and age_fact.scope.chapter_end == 0
    gender_fact = fact(project, "gender")
    assert (gender_fact.value if gender_fact else None) == gender


@pytest.mark.parametrize("text", [
    "Alice was only a child.",
    "But Alice was only a child, and the storm was overwhelming.",
    "Alice was a quiet child.",
    "I said before that Alice was a quiet child.",
    "Alice is only a quiet child.",
])
def test_verified_child_predicates_establish_age_range_without_numeric_age_or_gender(text):
    project, segments = book(text)
    research.research_project(project, segments)
    assert fact(project, "age_range").value == "child"
    assert fact(project, "age") is None and fact(project, "gender") is None


@pytest.mark.parametrize("text", [
    '"See Alice as she was, the quiet young girl, ten years old."',
    "\u201cSee Alice as she was, the quiet young girl, ten years old.\u201d",
    "Imagine Alice as she was, the quiet young girl, ten years old.",
    "If we see Alice as she was, the quiet young girl, ten years old.",
    "I wish we could consider Alice as she was, the quiet young girl, ten years old.",
    "See Alice as she would be, the quiet young girl, ten years old.",
    "See Alice as she was not, the quiet young girl, ten years old.",
    "See Alice as she was, not the quiet young girl, ten years old.",
    "See Alice as she was, the quiet young girl, not ten years old.",
    "See Alice as she was, the quiet young girl, ten years younger than Bob.",
    "See Alice as she was, the quiet young girl, ten years old if the tale were true.",
    "See Alice as she was, the ten-year-old girl, twelve years old.",
    "See Alice and Bob as she was, the quiet young girl, ten years old.",
    "See Alice watching Bob as he was, the quiet young boy, ten years old.",
    "See Alice as Bob was, the quiet young boy, ten years old.",
    "See Alice as she was, her friend's child, ten years old.",
    "Alice was not a quiet child.",
    "If Alice was a quiet child, the story would change.",
    "Imagine Alice was only a child.",
    "Bob claimed Alice was a quiet child.",
    "Alice was a quiet child only if the tale were true.",
    "Alice would be only a child.",
    "Alice was two years younger than Bob.",
])
def test_new_rules_abstain_from_quotes_negation_hypotheticals_and_other_referents(text):
    project, segments = book(text)
    research.research_project(project, segments)
    assert fact(project, "age") is None
    assert fact(project, "age_range") is None
    assert fact(project, "gender") is None


def test_depiction_age_remains_chapter_scoped_and_spoiler_filtered():
    project, segments = book(
        "See Alice as she was, the quiet young girl, ten years old.",
        "See Alice as she was, the teenager, seventeen years old.",
    )
    project.character_cast.policy.max_chapter = 0
    research.research_project(project, segments)
    assert fact(project, "age").value == 10
    assert not any(row.field == "age" and row.value == 17 for row in
                   project.character_cast.dossiers["alice"].facts)
    project.character_cast.policy.max_chapter = 1
    research.research_project(project, segments)
    assert fact(project, "age").value == 17
    assert fact(project, "age", chapter=0).value == 10
    assert fact(project, "age", chapter=1).chapter_id == "c1"
    project.character_cast.policy.max_chapter = 0
    assert fact(project, "age").value == 10


def test_v2_processed_cache_is_revisited_under_v3_then_idempotent(monkeypatch):
    text = "See Alice as she was, the quiet young girl, ten years old."
    project, segments = book(text)
    with monkeypatch.context() as patch:
        patch.setattr(research, "EXTRACTOR_VERSION", "anchored-text-v2")
        patch.setattr(research, "_extract", lambda _tail: [])
        research.research_project(project, segments)
    assert fact(project, "age") is None
    old_chunks = list(project.character_cast.dossiers["alice"].processed_chunks)
    assert research.EXTRACTOR_VERSION == "anchored-text-v3"
    research.research_project(project, segments)
    assert fact(project, "age").value == 10
    assert fact(project, "age").extractor == "anchored-text-v3"
    assert len(project.character_cast.dossiers["alice"].processed_chunks) == len(old_chunks) + 1
    snapshot = project.character_cast.model_dump_json()
    research.research_project(project, segments)
    assert project.character_cast.model_dump_json() == snapshot


def test_research_and_profile_preserve_locked_identity_and_human_rating_state():
    project, segments = book("See Alice as she was, the quiet young girl, ten years old.")
    role = project.roles[0]
    role.voice_identity = CharacterVoiceIdentity(
        id="locked-fixture", role_id=role.id, mechanism="fixture", model="fixture-no-model",
        dossier_revision="old", casting_profile_revision="old", design_instruction="Approved fixture voice.",
    )
    project.voice_lab.casting = {"roles": {role.id: {
        "assignment": {"key": "approved-fixture", "stars": 5, "approved": True},
        "round": {"id": "unrated-fixture", "candidates": [{"candidate_id": "same", "rating": None}]},
    }}}
    identity = copy.deepcopy(role.voice_identity.model_dump())
    ratings = copy.deepcopy(project.voice_lab.casting)
    research_and_profile(project, segments)
    assert fact(project, "age").value == 10
    assert role.voice_identity.model_dump() == identity
    assert project.voice_lab.casting == ratings


def test_explicit_depiction_evidence_reaches_distinct_performer_concepts():
    project, segments = book("See Alice as she was, the quiet young girl, ten years old.")
    research_and_profile(project, segments)
    profile = project.character_cast.profiles["alice"]
    dossier = project.character_cast.dossiers["alice"]
    proposals = casting_review._generated_proposals(
        project, profile, dossier, project.roles[0], profile.design_instruction, "", count=2, priors=[])
    assert len(proposals) == 2
    assert len({row["anchor"]["performer"] for row in proposals}) == 2
    for row in proposals:
        supported = row["concept"]["supported"]
        assert supported["age"]["value"] == "10"
        assert supported["age"]["evidence_ids"] == [fact(project, "age").id]
        assert supported["presentation"]["value"] == "feminine"
        assert supported["perceived_vocal_age"]["value"] == "child"
        assert "perceived_vocal_age" not in row["concept"]["inferred"]
