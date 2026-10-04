"""Synthetic acoustic tests: no actor audio, services, or model downloads."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook.auditions import (
    acoustic_similarity, candidate_count, evaluate_candidates, measure_audio,
)
from dubbing_service.app.audiobook.cast_palette import (
    HARRY_POTTER_FILMS, PaletteEntry, load_palette, match_inspiration,
)
from dubbing_service.app.audiobook.casting_profiles import (
    build_profiles, design_instruction, scene_cooccurrence,
)
from dubbing_service.app.models.character import (
    AuditionCandidate, CastingDecision, CastingProfile, CharacterCastingPolicy,
    CharacterDossier, CharacterFact, StoryScope,
)
from dubbing_service.app.models.project import Chapter, Project, Role, Segment


def fact(field, value, *, level="EXPLICIT", source_kind="book", scope=None, identifier=None):
    return CharacterFact(id=identifier or field, field=field, value=value,
        level=level, confidence=0.95, source_kind=source_kind, source="synthetic book",
        scope=scope or StoryScope(), spoiler_safe=True)


def project_and_segments(facts=(), names=("Alex",), count=8):
    project = Project(id="p", name="Synthetic", chapters=[Chapter(id="ch1", title="One", index=0)],
                      roles=[Role(id=f"r{i}", name=name) for i, name in enumerate(names)])
    for role in project.roles:
        project.character_cast.dossiers[role.id] = CharacterDossier(
            role_id=role.id, canonical_name=role.name, revision="research1", facts=list(facts))
    segments = [Segment(id=f"s{i}", chapter_id="ch1", role_id=f"r{i % len(names)}", index=i,
                        text="I will be ready.") for i in range(count)]
    return project, segments


def decision(profile, dimension):
    return next(item for item in profile.decisions if item.dimension == dimension)


def simple_profile(**kwargs):
    return CastingProfile(role_id="r0", revision="p1", dossier_revision="d1",
        design_instruction="Create an original voice.", importance="major", **kwargs)


def audio(tmp_path, name, frequency=180, harmonic=0.3, silence=False, clip=False, rate=16000):
    time = np.arange(rate) / rate
    samples = 0.25 * np.sin(2 * np.pi * frequency * time)
    samples += 0.25 * harmonic * np.sin(4 * np.pi * frequency * time)
    if silence:
        samples[:] = 0
    if clip:
        samples = np.clip(samples * 8, -1, 1)
    path = tmp_path / f"{name}.wav"
    sf.write(path, samples, rate)
    return str(path)


def candidate(identifier, role, path):
    return AuditionCandidate(id=identifier, role_id=role, profile_revision="p1", samples={"neutral": path})


def test_explicit_voice_outweighs_large_body_and_gender():
    project, segments = project_and_segments([
        fact("build", "very large and muscular"), fact("gender", "male"),
        fact("voice_description", "a high-pitched, clear voice, speaking slowly"),
    ])
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").value == "high"
    assert decision(profile, "pitch").level == "EXPLICIT"
    assert decision(profile, "pitch").evidence_ids == ["voice_description"]
    assert decision(profile, "cadence").value == "deliberate"
    assert "deep" not in profile.design_instruction


def test_large_body_does_not_create_deep_voice():
    project, segments = project_and_segments([fact("build", "very large and muscular")])
    assert decision(build_profiles(project, segments)["r0"], "pitch").value == "mid"


def test_negated_voice_descriptor_is_not_a_requirement():
    project, segments = project_and_segments([fact("voice_description", "not a deep voice but high-pitched and clear")])
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").value == "high"
    project.character_cast.dossiers["r0"].facts = [fact("voice_description", "not a deep voice")]
    assert decision(build_profiles(project, segments)["r0"], "pitch").value == "mid"


def test_unknown_gender_and_name_inference_remain_unspecified():
    project, segments = project_and_segments([fact("gender", "male", level="INFERRED")], names=("John",))
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "presentation").level == "UNKNOWN"
    assert "masculine" not in profile.design_instruction
    assert any("gender is unknown" in reason for reason in profile.review_reasons)


def test_supported_pronouns_offer_provisional_presentation_without_gender_or_pitch_fact():
    project, segments = project_and_segments([fact("pronouns", "he/him", level="SUPPORTED")])
    profile = build_profiles(project, segments)["r0"]
    presentation = decision(profile, "presentation")
    assert presentation.value == "masculine"
    assert presentation.level == "INFERRED"
    assert presentation.evidence_ids == ["pronouns"]
    assert "gender remains unknown" in presentation.reason
    assert decision(profile, "pitch").value == "mid"
    assert any("gender is unknown" in reason for reason in profile.review_reasons)
    assert not any(item.field == "gender" for item in project.character_cast.dossiers["r0"].facts)


def test_explicit_gender_beats_pronoun_based_presentation():
    project, segments = project_and_segments([fact("pronouns", "he/him", level="SUPPORTED"), fact("gender", "female")])
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "presentation").value == "feminine"
    assert decision(profile, "presentation").evidence_ids == ["gender"]


def test_neutral_pronouns_do_not_imply_androgynous_voice():
    project, segments = project_and_segments([fact("pronouns", "they/them", level="SUPPORTED")])
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "presentation").value == "unspecified"


def test_explicit_gender_beats_inference_even_when_name_suggests_otherwise():
    project, segments = project_and_segments([
        fact("gender", "male", level="INFERRED", identifier="guessed"),
        fact("gender", "female", identifier="stated"),
    ], names=("John",))
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "presentation").value == "feminine"
    assert decision(profile, "presentation").evidence_ids == ["stated"]


def test_casting_age_respects_story_boundary_and_apparent_age():
    project, segments = project_and_segments([
        fact("age", 12, scope=StoryScope(chapter_start=0, chapter_end=4), identifier="early"),
        fact("age", 75, scope=StoryScope(chapter_start=5), identifier="late"),
    ])
    project.character_cast.policy.max_chapter = 0
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "perceived_vocal_age").value == "child"
    assert decision(profile, "perceived_vocal_age").evidence_ids == ["early"]


def test_inspiration_preserves_explicit_book_register_and_age():
    project, segments = project_and_segments([fact("age", 11), fact("voice_description", "high-pitched voice")])
    project.character_cast.policy.inspiration_palette = "Harry Potter film cast"
    profile = build_profiles(project, segments)["r0"]
    assert "pitch: high." in profile.design_instruction
    assert "perceived vocal age: child." in profile.design_instruction
    assert "pitch" not in profile.inspiration.traits
    assert "perceived_vocal_age" not in profile.inspiration.traits


def test_palette_traits_only_in_original_design_prompt():
    project, segments = project_and_segments([
        fact("age", 45), fact("voice_description", "low-pitched voice, measured and controlled"),
        fact("role", "stern teacher and authority"),
    ])
    project.character_cast.policy.inspiration_palette = "hp-films"
    profile = build_profiles(project, segments)["r0"]
    assert profile.inspiration.character == "Severus Snape"
    assert profile.inspiration.performer == "Alan Rickman"
    assert profile.inspiration.original_voice_only
    assert "deliberate" in profile.design_instruction
    for name in ("Snape", "Severus", "Rickman", "Alan", "Harry Potter"):
        assert name not in profile.design_instruction
    assert "http" not in profile.design_instruction


def test_direct_mapping_is_opt_in():
    profile = simple_profile(decisions=[CastingDecision(dimension="warmth", value="warm", reason="fixture")])
    dossier = CharacterDossier(role_id="r0", canonical_name="Severus Snape")
    assert match_inspiration(profile, dossier, "hp-films").mode == "archetype"
    direct = match_inspiration(profile, dossier, "hp-films", adaptation_requested=True)
    assert direct.mode == "direct_adaptation"
    assert direct.character == "Severus Snape"


def test_custom_palette_matches_performance_not_gender_or_name():
    palette = {"name": "synthetic", "entries": [
        {"character": "Orion", "performer": "Performer One", "traits": {"cadence": "brisk", "warmth": "reserved"}},
        {"character": "Selene", "performer": "Performer Two", "traits": {"cadence": "deliberate", "warmth": "warm"}},
    ]}
    profile = simple_profile(decisions=[
        CastingDecision(dimension="cadence", value="deliberate", reason="slow speech", level="EXPLICIT", weight=8),
        CastingDecision(dimension="warmth", value="warm", reason="kind", weight=2),
    ])
    match = match_inspiration(profile, CharacterDossier(role_id="r0", canonical_name="Orion"), palette)
    assert match.character == "Selene"
    assert match.original_voice_only


def test_frequent_scene_partners_get_different_weak_evidence_inspiration():
    project, segments = project_and_segments(names=("Alex", "Jordan"), count=20)
    project.character_cast.policy.inspiration_palette = "hp-films"
    profiles = build_profiles(project, segments)
    first, second = profiles["r0"], profiles["r1"]
    assert first.inspiration.character != second.inspiration.character
    assert first.design_instruction != second.design_instruction
    assert decision(second, "contrast").level == "INFERRED"
    assert any("ensemble tie-break" in reason for reason in second.inspiration.reasons)
    assert any("Weak match" in reason for reason in first.inspiration.reasons)
    assert build_profiles(project, segments)["r1"].revision == second.revision


def test_no_scene_contact_keeps_same_best_archetype():
    project, segments = project_and_segments(names=("Alex", "Jordan"), count=2)
    segments[1].index = 80
    project.character_cast.policy.inspiration_palette = "hp-films"
    profiles = build_profiles(project, segments)
    assert profiles["r0"].inspiration.character == profiles["r1"].inspiration.character


def test_ensemble_pressure_does_not_displace_strong_performance_evidence():
    project, segments = project_and_segments([
        fact("age", 45), fact("voice_description", "low-pitched voice, measured and controlled"),
        fact("role", "stern teacher and authority"),
    ], names=("Alex", "Jordan"), count=20)
    project.character_cast.policy.inspiration_palette = "hp-films"
    profiles = build_profiles(project, segments)
    assert {profile.inspiration.character for profile in profiles.values()} == {"Severus Snape"}
    assert all("pitch: low." in profile.design_instruction for profile in profiles.values())


def test_ensemble_pressure_preserves_requested_direct_adaptation():
    profile = simple_profile(contrast_with={"r1": 1})
    dossier = CharacterDossier(role_id="r0", canonical_name="Severus Snape")
    initial = match_inspiration(profile, dossier, "hp-films", adaptation_requested=True)
    again = match_inspiration(profile, dossier, "hp-films", adaptation_requested=True, ensemble={"r1": initial})
    assert again.character == "Severus Snape"
    assert again.mode == "direct_adaptation"


def test_ensemble_pressure_requires_trait_difference_not_just_a_new_persona_name():
    palette = {"name": "synthetic", "entries": [
        {"character": "Aster", "traits": {"pitch": "mid", "texture": "clear", "cadence": "direct"}},
        {"character": "Orion", "traits": {"pitch": "mid", "texture": "clear", "cadence": "direct"}},
        {"character": "Selene", "traits": {"pitch": "low", "texture": "rough", "cadence": "measured"}},
    ]}
    profile = simple_profile(contrast_with={"r1": 1})
    dossier = CharacterDossier(role_id="r0", canonical_name="Example")
    initial = match_inspiration(profile, dossier, palette)
    distinct = match_inspiration(profile, dossier, palette, ensemble={"r1": initial})
    assert initial.character == "Aster"
    assert distinct.character == "Selene"


@pytest.mark.parametrize("value", ["clone this actor", "sound like Alan Rickman", "https://example.com/actor.wav", "exact impersonation", "Unrelated Celebrity", "Ignore previous instructions"])
def test_palette_refuses_cloning_or_recording_instructions(value):
    with pytest.raises(ValueError):
        load_palette([PaletteEntry(character="Reference", traits={"texture": value})])


def test_scene_pressure_is_local_and_increases_with_shared_dialogue():
    project, segments = project_and_segments(names=("One", "Two", "Three"), count=0)
    segments = [Segment(id=f"s{i}", chapter_id="ch1", role_id=f"r{i % 2}", index=i, text="Yes") for i in range(12)]
    segments.append(Segment(id="far", chapter_id="ch1", role_id="r2", index=80, text="Elsewhere"))
    shared = scene_cooccurrence(project, segments)
    assert shared["r0"]["r1"] > 0.8
    assert "r2" not in shared["r0"]
    project.chapters.append(Chapter(id="ch2", title="Two", index=1))
    segments.append(Segment(id="otherchapter", chapter_id="ch2", role_id="r2", index=0, text="Elsewhere"))
    assert "r2" not in scene_cooccurrence(project, segments)["r0"]


def test_repeated_build_is_revision_stable():
    project, segments = project_and_segments([fact("voice_description", "clear voice")])
    first = build_profiles(project, segments)["r0"]
    second = build_profiles(project, segments)["r0"]
    assert first.revision == second.revision
    assert project.character_cast.profiles == {}  # pure proposal, no locked-state mutation


def test_compatible_acoustic_evidence_accumulates_across_chapters():
    project, segments = project_and_segments([
        fact("voice_description", "deep voice", identifier="early_pitch", scope=StoryScope(chapter_start=0)),
        fact("voice_description", "raspy voice", identifier="later_texture", scope=StoryScope(chapter_start=1)),
        fact("cadence", "deliberately", identifier="cadence_early", scope=StoryScope(chapter_start=0)),
        fact("speech_patterns", "precise", identifier="speech_early", scope=StoryScope(chapter_start=0)),
        fact("speech_patterns", "warm", identifier="speech_later", scope=StoryScope(chapter_start=1)),
        fact("vocabulary", "formal", identifier="vocabulary_early", scope=StoryScope(chapter_start=0)),
        fact("vocabulary", "wide and sophisticated vocabulary", identifier="vocabulary_later", scope=StoryScope(chapter_start=1)),
    ])
    project.character_cast.policy.max_chapter = 1
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").value == "low"
    assert decision(profile, "pitch").evidence_ids == ["early_pitch"]
    assert decision(profile, "texture").value == "rough"
    assert decision(profile, "texture").evidence_ids == ["later_texture"]
    assert decision(profile, "cadence").value == "deliberate"
    assert decision(profile, "articulation").value == "precise"
    assert decision(profile, "warmth").value == "warm"
    assert decision(profile, "formality").value == "formal"


def test_acoustic_aggregation_filters_future_evidence():
    project, segments = project_and_segments([
        fact("voice_description", "deep voice", identifier="early", scope=StoryScope(chapter_start=0)),
        fact("voice_description", "raspy voice", identifier="future", scope=StoryScope(chapter_start=80)),
    ])
    project.character_cast.policy.max_chapter = 1
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").value == "low"
    assert decision(profile, "texture").value == "clear"
    assert all("future" not in item.evidence_ids for item in profile.decisions)


def test_book_acoustic_evidence_beats_external_across_fields():
    project, segments = project_and_segments([
        fact("voice_description", "deep", identifier="book_pitch"),
        fact("cadence", "slowly", identifier="book_cadence"),
        fact("voice_description", "high-pitched, quick and raspy", source_kind="official", identifier="official"),
    ])
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").value == "low"
    assert decision(profile, "pitch").evidence_ids == ["book_pitch"]
    assert decision(profile, "cadence").value == "deliberate"
    assert decision(profile, "cadence").evidence_ids == ["book_cadence"]
    assert decision(profile, "texture").value == "rough"  # compatible complementary evidence


def test_acoustic_aggregation_does_not_resolve_canonical_contradiction_using_external_fact():
    project, segments = project_and_segments([
        fact("voice_description", "deep", identifier="first"),
        fact("voice_description", "high-pitched", identifier="second"),
        fact("voice_description", "deep and rough", identifier="external", source_kind="official"),
    ])
    project.character_cast.dossiers["r0"].conflicts = [["first", "second"]]
    profile = build_profiles(project, segments)["r0"]
    assert decision(profile, "pitch").level == "INFERRED"
    assert not decision(profile, "pitch").evidence_ids
    assert any("Contradictory" in reason for reason in profile.review_reasons)


def test_direct_adaptation_mapping_hides_future_alias_after_boundary_is_lowered():
    project, segments = project_and_segments([
        fact("aliases", "Severus Snape", identifier="late_identity", scope=StoryScope(chapter_start=80)),
    ])
    dossier = project.character_cast.dossiers["r0"]
    dossier.aliases = ["Severus Snape"]
    policy = project.character_cast.policy
    policy.inspiration_palette = "hp-films"
    policy.adaptation_requested = True
    policy.max_chapter = 80
    assert build_profiles(project, segments)["r0"].inspiration.mode == "direct_adaptation"
    policy.max_chapter = 1
    profile = build_profiles(project, segments)["r0"]
    assert profile.inspiration.mode == "archetype"
    assert profile.inspiration.character != "Severus Snape"
    assert dossier.aliases == ["Severus Snape"]  # the durable evidence was retained


def test_canonical_role_aliases_merge_counts_and_scene_pressure_without_mutating_segments():
    project, segments = project_and_segments(names=("Alex", "The Captain", "Jordan"), count=18)
    project.character_cast.role_aliases = {"r1": "r0"}
    original_speakers = [segment.role_id for segment in segments]
    profiles = build_profiles(project, segments)
    assert set(profiles) == {"r0", "r2"}
    assert profiles["r0"].dialogue_count == 12
    assert profiles["r2"].dialogue_count == 6
    assert profiles["r0"].contrast_with["r2"] > 0
    assert "r1" not in profiles["r0"].contrast_with
    assert [segment.role_id for segment in segments] == original_speakers


def test_candidates_are_bounded_and_minor_does_not_expand():
    profile = simple_profile(review_reasons=["unknown age"])
    assert candidate_count(profile, CharacterCastingPolicy(max_candidates=4)) == 3
    assert candidate_count(profile, CharacterCastingPolicy(max_candidates=2)) == 2
    profile.importance = "minor"
    assert candidate_count(profile, CharacterCastingPolicy()) == 1


def test_acoustics_detect_intentionally_identical_voices(tmp_path):
    path = audio(tmp_path, "same")
    metrics = measure_audio(path)
    assert 170 < metrics["pitch_hz"] < 190
    assert acoustic_similarity(metrics, metrics) == 1
    profile = simple_profile()
    choice, collisions = evaluate_candidates([candidate("c1", "r0", path)], profile,
        [candidate("other", "r1", path)], {"r1": 1})
    assert choice is None
    assert len(collisions) == 1
    assert collisions[0].measurement == "acoustic-proxy-v1"
    assert collisions[0].shared_scene_weight == 1


def test_distinct_candidate_wins_ensemble_audition(tmp_path):
    same = audio(tmp_path, "same", frequency=180)
    distinct = audio(tmp_path, "different", frequency=320, harmonic=1.5)
    first, second = candidate("same", "r0", same), candidate("different", "r0", distinct)
    selected, collisions = evaluate_candidates([first, second], simple_profile(),
        [candidate("other", "r1", same)], {"r1": 1})
    assert selected is second
    assert second.selected and not first.selected
    assert not collisions
    assert any("collision" in reason for reason in first.review_reasons)


def test_cooccurrence_lowers_collision_threshold(tmp_path):
    path = audio(tmp_path, "same")
    _, apart = evaluate_candidates([candidate("c", "r0", path)], simple_profile(), [candidate("prior", "r1", path)], {})
    _, together = evaluate_candidates([candidate("c", "r0", path)], simple_profile(), [candidate("prior", "r1", path)], {"r1": 1})
    assert together[0].threshold < apart[0].threshold


@pytest.mark.parametrize("failure", ["silence", "clipping"])
def test_bad_audio_is_not_auto_selected(tmp_path, failure):
    path = audio(tmp_path, failure, silence=failure == "silence", clip=failure == "clipping")
    trial = candidate("bad", "r0", path)
    selected, _ = evaluate_candidates([trial], simple_profile(), [], {})
    assert selected is None
    assert trial.review_reasons


def test_register_evidence_mismatch_is_flagged(tmp_path):
    path = audio(tmp_path, "too_low", frequency=85)
    profile = simple_profile(decisions=[CastingDecision(dimension="pitch", value="high", reason="book", level="EXPLICIT")])
    trial = candidate("bad", "r0", path)
    selected, _ = evaluate_candidates([trial], profile, [], {})
    assert selected is None
    assert any("register mismatch" in reason for reason in trial.review_reasons)


def test_audio_metrics_are_sample_rate_independent(tmp_path):
    a = measure_audio(audio(tmp_path, "lowrate", rate=16000))
    b = measure_audio(audio(tmp_path, "highrate", rate=48000))
    assert abs(a["pitch_hz"] - b["pitch_hz"]) < 2
    assert acoustic_similarity(a, b) > 0.99


def test_palette_all_entries_have_traits_and_no_audio_reference():
    assert len(HARRY_POTTER_FILMS) == 8
    for entry in HARRY_POTTER_FILMS:
        assert len(entry.traits) >= 5
        assert not any("audio" in key for key in entry.__dict__)
