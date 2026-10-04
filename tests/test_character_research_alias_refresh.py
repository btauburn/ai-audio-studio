"""Research refresh preserves established identity merges and chapter policy."""
from __future__ import annotations

import copy

import pytest

from dubbing_service.app.audiobook.character_casting import research_and_profile
from dubbing_service.app.models.character import CharacterDossier, CharacterFact, StoryScope
from dubbing_service.app.models.project import Role
from tests.test_character_research import book


def linked_project():
    project, segments = book("Alice waited.")
    project.roles.extend([Role(id="alias", name="Allie"), Role(id="other", name="Beth")])
    project.character_cast.role_aliases = {"alias": "alice"}
    return project, segments


def test_research_refresh_keeps_established_merge_without_dossier_alias_evidence():
    project, segments = linked_project()
    project.voice_lab.casting = {"roles": {"alice": {"assignment": {"stars": 5}},
                                          "alias": {"round": {"rating": None}}}}
    cast = copy.deepcopy(project.voice_lab.casting)
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {"alias": "alice"}
    assert project.voice_lab.casting == cast


def test_research_refresh_drops_dangling_established_links():
    project, segments = linked_project()
    project.character_cast.role_aliases = {"missing-alias": "alice", "alias": "missing-owner"}
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {}


@pytest.mark.parametrize("source_kind", ["book", "project", "official", "reference", "adaptation"])
def test_invisible_alias_evidence_removes_prior_derived_link(source_kind):
    project, segments = linked_project()
    project.character_cast.policy.max_chapter = 0
    project.character_cast.dossiers["alice"] = CharacterDossier(
        role_id="alice", canonical_name="Alice", aliases=["Allie"], facts=[
            CharacterFact(id="later-alias", field="aliases", value="Allie", source="fixture:later",
                          source_kind=source_kind, level="EXPLICIT", available_after_chapter=2,
                          scope=StoryScope(chapter_start=2))])
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {}


def test_new_research_metadata_cannot_silently_override_established_merge():
    project, segments = linked_project()
    project.character_cast.dossiers["other"] = CharacterDossier(
        role_id="other", canonical_name="Beth", aliases=["Allie"])
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {"alias": "alice"}
    reason = "Research alias for Allie conflicts with an established identity merge; explicit identity review is required"
    assert project.character_cast.review_reasons.count(reason) == 1
    research_and_profile(project, segments)
    assert project.character_cast.review_reasons.count(reason) == 1


def test_established_chains_still_flatten_and_cycles_are_rejected():
    project, segments = linked_project()
    project.character_cast.role_aliases = {"alias": "other", "other": "alice"}
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {"alias": "alice", "other": "alice"}
    project.character_cast.role_aliases = {"alias": "other", "other": "alias"}
    research_and_profile(project, segments)
    for alias in project.character_cast.role_aliases:
        seen = set()
        current = alias
        while current in project.character_cast.role_aliases:
            assert current not in seen
            seen.add(current)
            current = project.character_cast.role_aliases[current]


def test_explicit_book_alias_still_creates_a_new_link():
    project, segments = book("Alice, also known as Allie, waited.")
    project.roles.append(Role(id="alias", name="Allie"))
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {"alias": "alice"}


def test_new_book_alias_cannot_reclassify_and_override_prior_established_merge():
    project, segments = linked_project()
    segments[0].text = "Beth, also known as Allie, waited."
    research_and_profile(project, segments)
    assert any(f.field == "aliases" and f.value == "Allie" and f.source_kind == "book"
               for f in project.character_cast.dossiers["other"].facts)
    assert project.character_cast.role_aliases == {"alias": "alice"}
    assert any("conflicts with an established identity merge" in reason
               for reason in project.character_cast.review_reasons)


def test_book_alias_conflict_preserves_established_merge_on_repeated_refresh():
    project, segments = linked_project()
    segments[0].text = "Beth, also known as Allie, waited."
    for _ in range(3):
        research_and_profile(project, segments)
        assert project.character_cast.role_aliases == {"alias": "alice"}
    assert len([r for r in project.character_cast.review_reasons
                if "conflicts with an established identity merge" in r]) == 1


def test_orphan_dossier_cannot_become_alias_owner():
    project, segments = linked_project()
    project.character_cast.role_aliases = {}
    project.character_cast.dossiers["ghost"] = CharacterDossier(
        role_id="ghost", canonical_name="Deleted role", aliases=["Allie"])
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {}


def test_prior_evidence_owner_is_resolved_through_established_chain():
    project, segments = linked_project()
    project.character_cast.role_aliases = {"alias": "alice", "other": "alice"}
    project.character_cast.policy.max_chapter = 0
    project.character_cast.dossiers["other"] = CharacterDossier(
        role_id="other", canonical_name="Beth", aliases=["Allie"], facts=[
            CharacterFact(id="later-owner-alias", field="aliases", value="Allie",
                          source="fixture:later", source_kind="book", level="EXPLICIT",
                          available_after_chapter=2, scope=StoryScope(chapter_start=2))])
    research_and_profile(project, segments)
    assert project.character_cast.role_aliases == {"other": "alice"}
