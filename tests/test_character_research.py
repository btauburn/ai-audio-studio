"""Evidence boundary tests use original synthetic prose, never actor recordings."""

from __future__ import annotations

from dubbing_service.app.audiobook.character_research import (
    effective_facts,
    research_project,
    resolve_fact,
)
from dubbing_service.app.models.audiobook import (
    Attribution,
    SourceAnchor,
    SourceRevision,
)
from dubbing_service.app.models.character import CharacterFact, StoryScope
from dubbing_service.app.models.project import Chapter, Project, Role, Segment


def book(*texts: str, name: str = "Alice", description: str | None = None):
    project = Project(
        id="research-fixture",
        name="The Test Harbour",
        roles=[
            Role(id="alice", name=name, description=description),
        ],
        chapters=[
            Chapter(id=f"c{i}", index=i, title=f"Chapter {i + 1}")
            for i in range(len(texts))
        ],
    )
    project.source_revisions = [
        SourceRevision(id="src1", uri="fixture:original-harbour", sha256="a" * 64)
    ]
    segments = [
        Segment(
            id=f"s{i}",
            chapter_id=f"c{i}",
            role_id="alice",
            index=i,
            text=text,
            source_anchor=SourceAnchor(
                source_revision_id="src1",
                chapter_id=f"c{i}",
                occurrence_index=i,
                start_offset=0,
                end_offset=len(text),
            ),
        )
        for i, text in enumerate(texts)
    ]
    return project, segments


def fact(project, field, *, chapter=None):
    return resolve_fact(
        project.character_cast.dossiers["alice"],
        field,
        project.character_cast.policy,
        chapter,
    )


class ReferenceProvider:
    provider_id = "synthetic-reference"
    version = "1"

    def __init__(self, facts=(), error=False):
        self.facts = list(facts)
        self.error = error
        self.calls = []

    def research(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error:
            raise OSError("Synthetic offline provider failure")
        return self.facts


def external(field="gender", value="female", **kwargs):
    return CharacterFact(
        id="external-" + field + str(value),
        field=field,
        value=value,
        source="fixture:reference",
        source_kind="reference",
        level="EXPLICIT",
        confidence=1,
        **kwargs,
    )


def test_explicit_gender_beats_name_and_project_description():
    project, segments = book(
        "Alice was a man. Alice spoke slowly.", description="Gender: female"
    )
    research_project(project, segments)
    chosen = fact(project, "gender")
    assert chosen.value == "male"
    assert chosen.level == "EXPLICIT"
    assert chosen.source_kind == "book"
    assert chosen.source_revision_id == "src1"
    assert chosen.chapter_id == "c0"
    assert "Alice was a man" in chosen.excerpt
    assert project.character_cast.dossiers["alice"].conflicts


def test_unsupported_gender_remains_unknown_despite_familiar_name_and_other_people():
    project, segments = book('Alice watched her brother. "He is a man," Alice said.')
    research_project(project, segments)
    assert fact(project, "gender") is None
    unknown = [
        f for f in project.character_cast.dossiers["alice"].facts if f.field == "gender"
    ]
    assert all(f.level == "UNKNOWN" and f.value is None for f in unknown)


def test_explicit_high_voice_is_separate_from_large_build():
    project, segments = book(
        "Alice was large and muscular. Alice's voice was high-pitched and clear."
    )
    research_project(project, segments)
    assert fact(project, "build").value == "large and muscular"
    assert fact(project, "voice_description").value == "high-pitched and clear"
    assert all(
        f.field != "pitch" for f in project.character_cast.dossiers["alice"].facts
    )
    assert fact(project, "voice_description").level == "EXPLICIT"


def test_ages_coexist_at_different_story_times():
    project, segments = book(
        "Alice was twelve years old.", "Alice was seventeen years old."
    )
    research_project(project, segments)
    assert fact(project, "age", chapter=0).value == 12
    assert fact(project, "age", chapter=1).value == 17
    assert fact(project, "age").value == 17
    assert not project.character_cast.dossiers["alice"].conflicts


def test_same_scope_contradictions_are_preserved_and_unresolved():
    project, segments = book(
        "Alice was twelve years old. Alice was seventeen years old."
    )
    research_project(project, segments)
    dossier = project.character_cast.dossiers["alice"]
    assert {
        f.value for f in dossier.facts if f.field == "age" and f.level == "EXPLICIT"
    } == {12, 17}
    assert fact(project, "age") is None
    assert dossier.conflicts
    assert any("Conflicting age" in reason for reason in dossier.review_reasons)


def test_external_cannot_override_book_even_at_higher_confidence():
    project, segments = book("Alice was a man.")
    project.character_cast.policy.external_research = True
    provider = ReferenceProvider([external(available_after_chapter=0)])
    research_project(project, segments, external_provider=provider)
    assert fact(project, "gender").value == "male"
    dossier = project.character_cast.dossiers["alice"]
    assert any(
        f.value == "female" and f.source_kind == "reference" for f in dossier.facts
    )
    assert dossier.conflicts
    assert len(provider.calls) == 1


def test_external_cannot_masquerade_as_canonical_book():
    project, segments = book("Alice waited.")
    project.character_cast.policy.external_research = True
    forged = external(available_after_chapter=0).model_copy(
        update={"source_kind": "book"}
    )
    research_project(project, segments, external_provider=ReferenceProvider([forged]))
    assert fact(project, "gender") is None
    assert (
        next(iter(project.character_cast.external_cache.values()))["rejected_count"]
        == 1
    )


def test_spoiler_boundary_blocks_future_book_and_future_or_unscoped_external():
    project, segments = book("Alice waited.", "Alice was a queen.")
    policy = project.character_cast.policy
    policy.max_chapter = 0
    policy.external_research = True
    provider = ReferenceProvider(
        [
            external("role", "secret heir", available_after_chapter=79),
            external("gender", "female"),
            external(
                "appearance",
                "a later scar",
                available_after_chapter=79,
                spoiler_safe=True,
            ),
            external("voice_description", "clear, light voice", spoiler_safe=True),
        ]
    )
    research_project(project, segments, external_provider=provider)
    assert fact(project, "role") is None
    assert fact(project, "gender") is None
    assert fact(project, "appearance") is None
    assert fact(project, "voice_description").value == "clear, light voice"
    # Rejected spoilers do not enter either dossiers or the persisted cache.
    serialized = project.character_cast.model_dump_json()
    assert "secret heir" not in serialized
    assert "a later scar" not in serialized


def test_explicit_spoiler_permission_does_not_bypass_adaptation_opt_in():
    project, segments = book("Alice waited.")
    policy = project.character_cast.policy
    policy.external_research = True
    policy.allow_spoilers = True
    adaptation = external(
        "voice_description", "forceful", available_after_chapter=50
    ).model_copy(update={"source_kind": "adaptation"})
    research_project(
        project, segments, external_provider=ReferenceProvider([adaptation])
    )
    assert fact(project, "voice_description") is None


def test_failed_external_research_is_cached_and_unknown():
    project, segments = book("Alice waited.")
    project.character_cast.policy.external_research = True
    provider = ReferenceProvider(error=True)
    research_project(project, segments, external_provider=provider)
    research_project(project, segments, external_provider=provider)
    assert len(provider.calls) == 1
    assert fact(project, "gender") is None
    assert [
        f.field
        for f in effective_facts(
            project.character_cast.dossiers["alice"], project.character_cast.policy
        )
    ] == ["canonical_name"]
    assert (
        next(iter(project.character_cast.external_cache.values()))["status"]
        == "unavailable"
    )


def test_unchanged_research_is_idempotent_and_survives_reload():
    project, segments = book("Alice was a young woman. Alice spoke slowly.")
    research_project(project, segments)
    first = project.character_cast.model_dump_json()
    restored = Project.model_validate_json(project.model_dump_json())
    research_project(restored, segments)
    assert restored.character_cast.model_dump_json() == first


def test_explicit_alias_deduplicates_external_work_and_not_shared_surnames():
    project, segments = book("Alice, also known as Allie, waited.")
    project.roles.append(Role(id="allie", name="Allie"))
    project.character_cast.policy.external_research = True
    provider = ReferenceProvider([])
    research_project(project, segments, external_provider=provider)
    assert project.character_cast.dossiers["alice"].aliases == ["Allie"]
    assert len(provider.calls) == 1


def test_age_gender_apparent_age_and_pronouns_have_independent_evidence():
    project, segments = book(
        "Alice was a twelve-year-old girl. Alice appeared about nine. Alice uses she/her pronouns."
    )
    research_project(project, segments)
    assert fact(project, "age").value == 12
    assert fact(project, "gender").value == "female"
    assert fact(project, "apparent_age").value == 9
    assert fact(project, "pronouns").value == "she/her"


def test_distinct_books_do_not_create_false_temporal_conflict():
    project, segments = book("Alice waited.")
    research_project(project, segments)
    dossier = project.character_cast.dossiers["alice"]
    dossier.facts.extend(
        [
            external(
                "age",
                12,
                scope=StoryScope(book="Book 1", chapter_start=0, chapter_end=0),
                available_after_chapter=0,
            ),
            external(
                "age",
                17,
                scope=StoryScope(book="Book 6", chapter_start=0, chapter_end=0),
                available_after_chapter=0,
            ),
        ]
    )
    research_project(project, segments)
    assert not dossier.conflicts


def test_source_book_boundary_can_expand_without_reprocessing_earlier_chunks():
    project, segments = book("Alice was twelve.", "Alice was seventeen.")
    project.character_cast.policy.max_chapter = 0
    research_project(project, segments)
    assert fact(project, "age").value == 12
    first = project.character_cast.dossiers["alice"].facts[-1].researched_at
    project.character_cast.policy.max_chapter = 1
    research_project(project, segments)
    assert fact(project, "age").value == 17
    assert any(
        f.researched_at == first and f.value == 12
        for f in project.character_cast.dossiers["alice"].facts
    )


def test_negation_and_other_characters_traits_do_not_leak():
    project, segments = book(
        "Alice was not a man. Alice watched Bob, a tall boy. Alice admired Bob's deep voice."
    )
    research_project(project, segments)
    assert fact(project, "gender") is None
    assert fact(project, "build") is None
    assert fact(project, "voice_description") is None


def test_alias_discovery_is_order_independent_and_idempotent():
    project, segments = book(
        "Allie was a young woman.", "Alice, also known as Allie, waited."
    )
    research_project(project, segments)
    assert fact(project, "gender").value == "female"
    before = project.character_cast.model_dump_json()
    research_project(project, segments)
    assert project.character_cast.model_dump_json() == before


def test_future_alias_is_not_used_when_policy_boundary_moves_back():
    project, segments = book(
        "Allie was a young woman.", "Alice, also known as Allie, waited."
    )
    project.character_cast.policy.max_chapter = 0
    research_project(project, segments)
    assert fact(project, "gender") is None
    assert project.character_cast.dossiers["alice"].aliases == []


def test_additional_explicit_dimensions_keep_their_source():
    project, segments = book(
        "Alice stood six feet tall. Alice had a scar across one cheek. "
        "Alice was Bob's sister. Alice grew up in the harbour district. "
        "Alice's vocabulary was formal. Alice now spoke slowly."
    )
    research_project(project, segments)
    assert fact(project, "height").value == "six feet"
    assert "scar" in fact(project, "distinctive_features").value
    assert fact(project, "relationships").value == "Bob's sister"
    assert "harbour district" in fact(project, "social_context").value
    assert fact(project, "vocabulary").value == "formal"
    assert fact(project, "portrayal_changes").value == "now spoke slowly"
    assert all(
        f.source_revision_id == "src1"
        for f in effective_facts(
            project.character_cast.dossiers["alice"], project.character_cast.policy
        )
        if f.field != "canonical_name"
    )


def test_a_second_named_subject_cannot_contribute_appearance_to_first():
    project, segments = book("Alice had black hair and Bob had green eyes.")
    research_project(project, segments)
    assert fact(project, "appearance").value == "black hair"


def test_uncertain_attribution_is_reviewable_without_inventing_character_facts():
    project, segments = book("The tide moved in.")
    segments[0].attribution = Attribution(
        speaker_id="alice", source="turn", confidence=0.4
    )
    research_project(project, segments)
    assert any(
        "attribution is uncertain" in reason
        for reason in project.character_cast.dossiers["alice"].review_reasons
    )
    assert fact(project, "gender") is None


def test_unconfigured_external_provider_gracefully_preserves_unknown():
    project, segments = book("Alice waited.")
    project.character_cast.policy.external_research = True
    research_project(project, segments)
    assert fact(project, "gender") is None
    assert (
        next(iter(project.character_cast.external_cache.values()))["provider"]
        == "unavailable"
    )


def test_explicit_contradictory_voice_traits_require_review():
    project, segments = book("Alice's voice was high-pitched. Alice's voice was deep.")
    research_project(project, segments)
    assert fact(project, "voice_description") is None
    assert any(
        "Conflicting voice_description" in reason
        for reason in project.character_cast.dossiers["alice"].review_reasons
    )


def test_character_name_records_derived_attribution_provenance():
    project, segments = book("The tide moved in.")
    segments[0].attribution = Attribution(
        speaker_id="alice", source="inline", confidence=1
    )
    research_project(project, segments)
    chosen = fact(project, "canonical_name")
    assert chosen.value == "Alice"
    assert chosen.source_kind == "project"
    assert chosen.level == "SUPPORTED"
    assert "initial_attribution_count:1" in chosen.location
    assert "inline" in chosen.location


def test_direct_body_actions_support_pronouns_without_inventing_gender():
    project, segments = book("Alice shook his head. Alice closed his eyes.")
    research_project(project, segments)
    chosen = fact(project, "pronouns")
    assert chosen.value == "he/him"
    assert chosen.level == "SUPPORTED"
    assert chosen.source_revision_id == "src1"
    assert fact(project, "gender") is None


def test_adjacent_single_subject_pronoun_supports_appearance():
    project, segments = book(
        "Alice said after introductions had been made. Her hair was turning bright red."
    )
    research_project(project, segments)
    assert fact(project, "pronouns").value == "she/her"
    chosen = fact(project, "appearance")
    assert chosen.level == "SUPPORTED"
    assert "bright red" in chosen.value


def test_adjacent_coreference_abstains_with_competing_named_or_unnamed_people():
    for text in (
        "Alice smiled at Bob. His hair was black.",
        "Alice stood beside a woman. Her hair was black.",
        "Alice watched her brother. His hair was black.",
        "Alice looked at a picture of a stranger. His hair was black.",
        "Alice touched her face.",
    ):
        project, segments = book(text)
        research_project(project, segments)
        assert fact(project, "pronouns") is None
        assert fact(project, "appearance") is None


def test_named_age_descriptor_does_not_require_copular_sentence():
    project, segments = book("This story follows a teenager named Alice.")
    research_project(project, segments)
    assert fact(project, "age_range").value == "teenager"
    assert fact(project, "age_range").level == "EXPLICIT"


def test_temporary_voice_changes_are_scene_evidence_not_stable_pitch():
    project, segments = book("Alice deepened his voice.", "Alice whispered.")
    research_project(project, segments)
    assert fact(project, "voice_description") is None
    observations = [
        f
        for f in project.character_cast.dossiers["alice"].facts
        if f.field == "scene_delivery" and f.level == "EXPLICIT"
    ]
    assert {f.value for f in observations} == {"deepened his voice", "whispered"}
    assert all(f.scope.chapter_start == f.scope.chapter_end for f in observations)


def test_contextual_observation_never_uses_dialogue_attribution_as_coreference():
    project, segments = book("Alice smiled. Her hair was red.")
    segments[0].attribution = Attribution(speaker_id="bob", source="tag", confidence=1)
    research_project(project, segments)
    assert fact(project, "pronouns") is None
    assert fact(project, "appearance") is None


def test_supported_pronoun_research_is_bounded_and_idempotent():
    project, segments = book(
        "Alice shook his head. Alice closed his eyes. Alice raised his voice."
    )
    research_project(project, segments)
    before = project.character_cast.model_dump_json()
    research_project(project, segments)
    assert project.character_cast.model_dump_json() == before
    supported = [
        f
        for f in project.character_cast.dossiers["alice"].facts
        if f.field == "pronouns" and f.level == "SUPPORTED"
    ]
    assert len(supported) == 2


def test_explicit_named_possessive_appearance_has_no_pronoun_guess():
    project, segments = book("Alice's green eyes opened.")
    research_project(project, segments)
    assert fact(project, "appearance").value == "green eyes"
    assert fact(project, "pronouns") is None
