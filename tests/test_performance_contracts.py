"""CPU-only contracts for the Performance Director records and pure helpers."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.models.character import (
    CastingDecision,
    CastingProfile,
    CharacterDossier,
    CharacterFact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SHA_A = "a" * 64
SHA_B = "b" * 64
VALID_INSTRUCTION = ("Measured and conversational; articulate the final words; restrained warmth.")


def make_habit(dimension: str = "cadence", value: str = "measured and unhurried",
               level: str = "EXPLICIT", evidence_ids: tuple[str, ...] = (), origin: str = "manual"):
    return director.DeliveryHabit(dimension=dimension, value=value, level=level,
                                  evidence_ids=list(evidence_ids), origin=origin)


def make_profile(**overrides):
    payload = {"role_id": "role-1", "author": {"origin": "deterministic"}}
    payload.update(overrides)
    return director.DeliveryProfile(**payload)


def make_interpretation(**overrides):
    payload = {"mode": "manual", "planner_version": "scene-planner-v1"}
    payload.update(overrides)
    return director.SceneInterpretation(**payload)


def make_utterances():
    return [director.SceneUtterance(id="u1", text_sha256=SHA_A, role_id="role-1"),
            director.SceneUtterance(id="u2", text_sha256=SHA_B, role_id="role-2")]


def make_intention(**overrides):
    utterances = overrides.pop("utterances", None) or make_utterances()
    raw_interpretation = overrides.pop("interpretation", None)
    interpretation = (make_interpretation(**raw_interpretation)
                      if isinstance(raw_interpretation, dict)
                      else raw_interpretation or make_interpretation())
    source = overrides.pop("source_revision_sha256", SHA_A)
    attribution = overrides.pop("attribution_digest", SHA_A)
    alias = overrides.pop("alias_digest", SHA_B)
    payload = {
        "scene_id": "scene-1",
        "chapter_id": "chapter-1",
        "source_revision_sha256": source,
        "utterances": utterances,
        "attribution_digest": attribution,
        "alias_digest": alias,
        "interpretation": interpretation,
        "author": {"origin": "manual"},
        "cache_key": director.scene_cache_key(source_revision_sha256=source, utterances=utterances,
                                              attribution_digest=attribution, alias_digest=alias,
                                              interpretation=interpretation),
    }
    payload.update(overrides)
    return director.SceneIntention(**payload)


def make_direction(**overrides):
    payload = {
        "utterance_id": "u1",
        "text_sha256": SHA_A,
        "role_id": "role-1",
        "sources": {"capability_sha256": SHA_B, "resolver_version": "resolver-v1"},
    }
    payload.update(overrides)
    return director.UtteranceDirection(**payload)


def make_dossier(**overrides):
    payload = {"role_id": "role-1", "canonical_name": "Rowan", "revision": "dossier-r1"}
    payload.update(overrides)
    return CharacterDossier(**payload)


def make_casting(**overrides):
    payload = {"role_id": "role-1", "revision": "casting-r1", "dossier_revision": "dossier-r1",
               "design_instruction": "kept for voice design, not delivery"}
    payload.update(overrides)
    return CastingProfile(**payload)


def test_content_sha256_ignores_created_at_and_tracks_habit_values() -> None:
    first = make_profile(habits=[make_habit(value="measured")],
                         created_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = make_profile(habits=[make_habit(value="measured")],
                          created_at=datetime(2026, 2, 2, tzinfo=UTC))
    third = make_profile(habits=[make_habit(value="brisk")],
                         created_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert director.content_sha256(first) == director.content_sha256(second)
    assert director.content_sha256(first) != director.content_sha256(third)
    assert director.content_sha256(second) == director.content_sha256(second.model_dump(mode="json"))


def test_records_reject_unknown_fields_wrong_schema_and_malformed_hashes() -> None:
    with pytest.raises(ValidationError):
        make_profile(unexpected="x")
    with pytest.raises(ValidationError):
        make_profile(schema_version="delivery-profile-v2")
    with pytest.raises(ValidationError):
        make_profile(parent_sha256="not-a-hash")
    with pytest.raises(ValidationError):
        make_profile(parent_sha256="A" * 64)


def test_compile_profile_without_sources_is_empty_and_deterministic() -> None:
    profile = director.compile_profile("role-1", narrator=False)
    assert isinstance(profile, director.DeliveryProfile)
    assert profile.habits == []
    assert profile.author.origin == "deterministic"
    assert profile.narrator_preservation is False
    assert profile.created_at is None


def test_compile_profile_records_narrator_preservation() -> None:
    assert director.compile_profile("role-1", narrator=True).narrator_preservation is True


def test_compile_profile_takes_a_dossier_cadence_fact() -> None:
    dossier = make_dossier(facts=[CharacterFact(id="fact-1", field="cadence", source="book",
                                                value="measured, unhurried", level="EXPLICIT")])
    profile = director.compile_profile("role-1", narrator=False, dossier=dossier)
    assert [(habit.dimension, habit.value, habit.level, habit.evidence_ids, habit.origin)
            for habit in profile.habits] == [("cadence", "measured, unhurried", "EXPLICIT",
                                              ["fact-1"], "dossier")]
    assert profile.dossier_revision == "dossier-r1"


def test_compile_profile_takes_only_delivery_casting_decisions() -> None:
    casting = make_casting(decisions=[
        CastingDecision(dimension="tempo", value="brisk but even", reason="r", level="SUPPORTED"),
        CastingDecision(dimension="pitch", value="low", reason="r", level="SUPPORTED"),
        CastingDecision(dimension="warmth", value="unspecified", reason="r", level="SUPPORTED"),
    ])
    profile = director.compile_profile("role-1", narrator=False, casting_profile=casting)
    assert [(habit.dimension, habit.origin) for habit in profile.habits] == [("tempo", "casting_profile")]
    assert profile.casting_profile_revision == "casting-r1"


def test_manual_edit_outranks_dossier_which_outranks_casting() -> None:
    dossier = make_dossier(facts=[CharacterFact(id="fact-1", field="cadence", source="book",
                                                value="from the dossier", level="EXPLICIT")])
    casting = make_casting(decisions=[
        CastingDecision(dimension="cadence", value="from casting", reason="r", level="EXPLICIT")])
    compiled = director.compile_profile("role-1", narrator=False, dossier=dossier,
                                        casting_profile=casting)
    assert [(habit.value, habit.origin) for habit in compiled.habits] == [("from the dossier", "dossier")]
    edited = director.compile_profile("role-1", narrator=False, dossier=dossier,
                                      casting_profile=casting,
                                      edits=[make_habit(value="from the editor")])
    assert [(habit.value, habit.origin) for habit in edited.habits] == [("from the editor", "manual")]
    assert edited.author.origin == "manual"


def test_dossier_fact_horizon_follows_max_chapter() -> None:
    dossier = make_dossier(facts=[CharacterFact(id="fact-1", field="cadence", source="book",
                                                value="measured", level="SUPPORTED",
                                                available_after_chapter=5)])
    assert director.compile_profile("role-1", narrator=False, dossier=dossier,
                                    max_chapter=3).habits == []
    assert len(director.compile_profile("role-1", narrator=False, dossier=dossier,
                                        max_chapter=5).habits) == 1
    assert len(director.compile_profile("role-1", narrator=False, dossier=dossier,
                                        max_chapter=None).habits) == 1


def test_compile_profile_refuses_forbidden_terms_and_non_manual_edits() -> None:
    with pytest.raises(director.PerformanceDirectorError) as forbidden:
        director.compile_profile("role-1", narrator=False,
                                 edits=[make_habit(value="a swaggering stride")],
                                 forbidden_terms=["Swagger"])
    assert forbidden.value.code == "direction_conflict"

    with pytest.raises(director.PerformanceDirectorError) as borrowed:
        director.compile_profile("role-1", narrator=False,
                                 edits=[make_habit(origin="dossier")])
    assert borrowed.value.code == "direction_conflict"


def test_compile_profile_is_deterministic_and_never_mutates_inputs() -> None:
    dossier = make_dossier(facts=[CharacterFact(id="fact-1", field="speech_patterns", source="book",
                                                value="clipped clauses", level="SUPPORTED")])
    casting = make_casting(decisions=[
        CastingDecision(dimension="tempo", value="brisk", reason="r", level="INFERRED")])
    edits = [make_habit(dimension="theatricality", value="restrained")]
    before = (dossier.model_dump(), casting.model_dump(), [edit.model_dump() for edit in edits])

    first = director.compile_profile("role-1", narrator=False, dossier=dossier,
                                     casting_profile=casting, edits=edits)
    second = director.compile_profile("role-1", narrator=False, dossier=dossier,
                                      casting_profile=casting, edits=edits)
    assert director.content_sha256(first) == director.content_sha256(second)
    assert (dossier.model_dump(), casting.model_dump(),
            [edit.model_dump() for edit in edits]) == before


def test_validate_native_instruction_accepts_and_refuses() -> None:
    assert director.validate_native_instruction(VALID_INSTRUCTION) == VALID_INSTRUCTION
    for rejected in ("", "   ", "a" * 241, "hold the line\nthen soften", "20% slower", "150 wpm",
                     "Sound like a famous narrator", "In the style of a newsreader"):
        with pytest.raises(director.PerformanceDirectorError) as refused:
            director.validate_native_instruction(rejected)
        assert refused.value.code == "direction_conflict"
    with pytest.raises(director.PerformanceDirectorError):
        director.validate_native_instruction("Deliver it with swagger", forbidden_terms=["Swagger"])


def test_direction_component_hashes_only_the_delivery_component() -> None:
    assert director.direction_component(director.ResolvedControls()) is None
    salted = director.ResolvedControls(seed_salt=7)
    assert director.direction_component(salted) is not None

    forward = director.ControlDisposition(control="native_instruction", state="forward_native")
    one = director.ResolvedControls(native_instruction="measured warmth", seed_salt=7)
    equal = director.ResolvedControls(native_instruction="measured warmth", seed_salt=7)
    other_instruction = director.ResolvedControls(native_instruction="brisk warmth", seed_salt=7)
    other_salt = director.ResolvedControls(native_instruction="measured warmth", seed_salt=8)

    assert director.direction_component(one) == director.direction_component(equal)
    assert director.direction_component(one) != director.direction_component(other_instruction)
    assert director.direction_component(one) != director.direction_component(other_salt)
    assert director.direction_component(one) == director.direction_component(
        make_direction(resolved={"native_instruction": "measured warmth", "seed_salt": 7},
                       dispositions=[forward]))


def test_utterance_direction_binds_the_instruction_to_its_disposition() -> None:
    forward = director.ControlDisposition(control="native_instruction", state="forward_native")
    accepted = make_direction(resolved={"native_instruction": "measured warmth"},
                              dispositions=[forward])
    assert accepted.resolved.native_instruction == "measured warmth"

    with pytest.raises(ValidationError):
        make_direction(resolved={"native_instruction": "measured warmth"})
    with pytest.raises(ValidationError):
        make_direction(dispositions=[forward])
    with pytest.raises(ValidationError):
        make_direction(resolved={"native_instruction": "measured warmth"},
                       dispositions=[forward, forward])
    with pytest.raises(ValidationError):
        make_direction(resolved={"native_instruction": "measured warmth"},
                       dispositions=[director.ControlDisposition(control="native_instruction",
                                                                 state="hold")])


def test_direction_set_rejects_duplicate_utterance_ids() -> None:
    first = make_direction()
    second = make_direction(text_sha256=SHA_B)
    assert first.utterance_id == second.utterance_id
    with pytest.raises(ValidationError):
        director.DirectionSet(project_id="project-1", directions=[first, second])
    accepted = director.DirectionSet(project_id="project-1",
                                     directions=[first, make_direction(utterance_id="u2")])
    assert [direction.utterance_id for direction in accepted.directions] == ["u1", "u2"]


def test_scene_intention_binds_turns_to_its_own_cache_key() -> None:
    accepted = make_intention(turns=[{"utterance_id": "u1", "confidence": "manual"}])
    assert accepted.cache_key == director.scene_cache_key(
        source_revision_sha256=accepted.source_revision_sha256,
        utterances=accepted.utterances,
        attribution_digest=accepted.attribution_digest,
        alias_digest=accepted.alias_digest,
        interpretation=accepted.interpretation,
    )

    with pytest.raises(ValidationError):
        make_intention(turns=[{"utterance_id": "u9", "confidence": "manual"}])
    with pytest.raises(ValidationError):
        make_intention(utterances=[director.SceneUtterance(id="u1", text_sha256=SHA_A)] * 2)
    with pytest.raises(ValidationError):
        make_intention(cache_key=SHA_A)
    with pytest.raises(ValidationError):
        make_intention(interpretation={"mode": "model", "planner_version": "scene-planner-v1"})


def test_scene_cache_key_tracks_attribution_role_and_order() -> None:
    utterances = make_utterances()
    interpretation = make_interpretation()
    base = {"source_revision_sha256": SHA_A, "attribution_digest": SHA_A, "alias_digest": SHA_B,
            "interpretation": interpretation}
    reference = director.scene_cache_key(utterances=utterances, **base)
    assert director.scene_cache_key(utterances=list(utterances), **base) == reference
    assert director.scene_cache_key(utterances=utterances,
                                    **{**base, "attribution_digest": SHA_B}) != reference
    reordered = [director.SceneUtterance(id="u1", text_sha256=SHA_A, role_id="role-9"),
                 utterances[1]]
    assert director.scene_cache_key(utterances=reordered, **base) != reference
    assert director.scene_cache_key(utterances=list(reversed(utterances)), **base) != reference


def test_read_index_round_trips_and_refuses_unknown_schemas() -> None:
    assert director.read_index(None) is None
    assert director.read_index({}) is None
    index = director.default_index()
    assert director.read_index({director.INDEX_KEY: index.model_dump(mode="json")}) == index

    with pytest.raises(director.PerformanceDirectorError) as version:
        director.read_index({director.INDEX_KEY: {"schema_version": "performance-director-index-v2"}})
    assert version.value.code == "schema_unknown"

    with pytest.raises(director.PerformanceDirectorError) as shape:
        director.read_index({director.INDEX_KEY: "not-a-mapping"})
    assert shape.value.code == "schema_unknown"


def test_performance_director_error_rejects_an_unknown_code() -> None:
    with pytest.raises(ValueError):
        director.PerformanceDirectorError("not_a_code", "x")
    error = director.PerformanceDirectorError("plan_stale", "the plan moved")
    assert error.code == "plan_stale"
    assert str(error) == "plan_stale: the plan moved"


def test_importing_the_module_pulls_no_boundary_dependency() -> None:
    watchers = ("torch", "dubbing_service.app.providers.tts",
                "dubbing_service.app.audiobook.renderer")
    code = (
        "import sys;"
        "import dubbing_service.app.audiobook.performance_director;"
        f"print([name for name in {watchers!r} if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
