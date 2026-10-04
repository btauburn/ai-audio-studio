"""Collision-directed candidate retries; synthetic audio and no model calls."""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import character_casting as casting
from dubbing_service.app.audiobook.auditions import measure_audio
from dubbing_service.app.models.character import (
    AuditionCandidate,
    CastCollision,
    CastingDecision,
    CastingProfile,
    CharacterVoiceIdentity,
)
from dubbing_service.app.models.project import Project, Role
from dubbing_service.app.storage.projects import load_project, save_project
from tests.audiobook_helpers import make_tone_wav, use_projects_root
from tests.test_character_casting_integration import OriginalVoiceTestProvider


def identity(identifier, role_id):
    return CharacterVoiceIdentity(
        id=identifier,
        role_id=role_id,
        mechanism="fixture",
        model="synthetic-signal",
        dossier_revision="d1",
        casting_profile_revision="p1",
        design_instruction="Create an original voice.",
    )


def collision_project(*, pitch=100, decisions=()):
    target = Role(id="target", name="Unestablished Name")
    neighbor = Role(
        id="neighbor", name="Other Name", voice_identity=identity("locked", "neighbor")
    )
    project = Project(id="retry", name="Synthetic collision", roles=[target, neighbor])
    profile = CastingProfile(
        role_id=target.id,
        revision="p1",
        dossier_revision="d1",
        importance="major",
        dialogue_count=8,
        design_instruction="Create an original voice.",
        decisions=list(decisions),
        contrast_with={"neighbor": 1},
    )
    project.character_cast.profiles[target.id] = profile
    project.character_cast.auditions[neighbor.id] = [
        AuditionCandidate(
            id="locked",
            role_id=neighbor.id,
            profile_revision="p1",
            identity=neighbor.voice_identity,
            selected=True,
            measurements={"neutral": {"pitch_hz": pitch}},
        )
    ]
    project.character_cast.collisions = [
        CastCollision(
            left=target.id,
            right=neighbor.id,
            similarity=0.98,
            threshold=0.85,
            shared_scene_weight=1,
            reason="Synthetic collision",
        )
    ]
    return project, target, profile


def test_third_candidate_has_traceable_inferred_contrast_without_changing_facts():
    project, role, profile = collision_project()
    before = project.model_dump(mode="json")
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert "higher-mid register" in instruction
    assert "brighter resonance" in instruction
    assert "forward, precise articulation" in instruction
    assert provenance["schema"] == "ensemble-collision-retry-v1"
    assert provenance["neighbors"][0]["candidate_id"] == "locked"
    assert provenance["neighbors"][0]["neutral_pitch_hz"] == 100
    assert all(
        d["level"] == "INFERRED" and d["evidence_ids"][0].startswith("collision_")
        for d in provenance["decisions"]
    )
    assert project.model_dump(mode="json") == before
    assert role.name not in instruction and "Other Name" not in instruction
    assert not any(
        word in instruction.lower()
        for word in ("male", "female", "child", "adult", "british", "actor")
    )


@pytest.mark.parametrize(
    "index,suffix",
    [
        (0, " Use a natural, distinct original timbre; avoid exaggerated delivery."),
        (
            1,
            " Use a subtly different original resonance and articulation, preserving all specified vocal traits.",
        ),
    ],
)
def test_first_two_candidate_instructions_are_unchanged(index, suffix):
    project, role, profile = collision_project()
    assert casting._candidate_design(project, role, profile, index) == (
        profile.design_instruction + suffix,
        None,
    )


@pytest.mark.parametrize("level", ["EXPLICIT", "SUPPORTED"])
def test_supported_voice_requirements_are_never_overridden(level):
    decisions = [
        CastingDecision(
            dimension=d,
            value=v,
            level=level,
            reason="book",
            evidence_ids=["book-voice"],
        )
        for d, v in [("pitch", "low"), ("texture", "rough"), ("cadence", "deliberate")]
    ]
    project, role, profile = collision_project(decisions=decisions)
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert provenance["decisions"] == []
    assert not any(
        word in instruction
        for word in ("higher-mid", "lower-mid", "brighter", "forward, precise")
    )
    assert set(provenance["protected_dimensions"]) == {"pitch", "texture", "cadence"}


@pytest.mark.parametrize("level,pitch", [("EXPLICIT", "high"), ("SUPPORTED", "low")])
def test_fixed_pitch_still_allows_other_unconstrained_traits(level, pitch):
    project, role, profile = collision_project(
        decisions=[
            CastingDecision(dimension="pitch", value=pitch, level=level, reason="book"),
            CastingDecision(
                dimension="presentation",
                value="masculine",
                level="INFERRED",
                reason="supported pronouns",
            ),
            CastingDecision(
                dimension="accent",
                value="established regional accent",
                level="EXPLICIT",
                reason="book",
            ),
        ]
    )
    profile.design_instruction += f" pitch: {pitch}. presentation: masculine. Retain the established regional accent."
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert instruction.startswith(profile.design_instruction)
    assert {d["dimension"] for d in provenance["decisions"]} == {
        "resonance",
        "articulation",
    }
    assert "higher-mid" not in instruction and "lower-mid" not in instruction


@pytest.mark.parametrize("age", ["child", "adolescent", "youthful", "young adult"])
def test_high_neighbor_cannot_force_youthful_voice_lower(age):
    project, role, profile = collision_project(
        pitch=230,
        decisions=[
            CastingDecision(
                dimension="perceived_vocal_age",
                value=age,
                reason="book",
                level="SUPPORTED",
            )
        ],
    )
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert "lower-mid" not in instruction
    assert all(d["dimension"] != "pitch" for d in provenance["decisions"])
    assert "rounded resonance" in instruction


def test_high_neighbor_allows_lower_mid_without_inventing_demographics():
    project, role, profile = collision_project(pitch=230)
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert "lower-mid register" in instruction
    assert {d["dimension"] for d in provenance["decisions"]} == {
        "pitch",
        "resonance",
        "articulation",
    }


@pytest.mark.parametrize(
    "missing", ["collision", "selected", "pitch", "locked_neighbor", "unlocked_target"]
)
def test_no_retry_clause_without_current_collision_and_selected_neighbor(missing):
    project, role, profile = collision_project()
    neighbor = project.character_cast.auditions["neighbor"][0]
    if missing == "collision":
        project.character_cast.collisions = []
    elif missing == "selected":
        neighbor.selected = False
    elif missing == "pitch":
        neighbor.measurements = {}
    elif missing == "locked_neighbor":
        project.roles[1].voice_identity = None
    else:
        role.voice_identity = identity("already-accepted", role.id)
    instruction, provenance = casting._candidate_design(project, role, profile, 2)
    assert provenance is None
    assert "inferred ensemble contrast" not in instruction


@pytest.mark.parametrize("previous_count,cap", [(2, 3), (3, 4)])
def test_one_additional_candidate_reuses_prior_audio_and_persists_retry_metadata(
    monkeypatch, tmp_path, previous_count, cap
):
    root = use_projects_root(monkeypatch, tmp_path)
    project, role, profile = collision_project()
    project.character_cast.policy.max_candidates = cap
    folder = root / project.id
    prior = []
    for index in range(previous_count):
        identifier = "voice_" + casting._key(
            [project.id, role.id, profile.revision, index]
        )
        voice = identity(identifier, role.id)
        path = make_tone_wav(
            folder / "audio" / identifier / "reference.wav", seconds=3, frequency=100
        )
        voice.reference_audio = path.relative_to(folder).as_posix()
        voice.reference_audio_sha256 = casting._sha(path)
        prompt = path.with_name("prompt.json")
        prompt.write_text("{}", encoding="utf-8")
        voice.prompt_artifact = prompt.relative_to(folder).as_posix()
        voice.prompt_sha256 = casting._sha(prompt)
        prior.append(
            AuditionCandidate(
                id=identifier,
                role_id=role.id,
                profile_revision=profile.revision,
                identity=voice,
                audition_revision=casting.AUDITION_REVISION,
                samples={
                    kind: voice.reference_audio for kind in casting.AUDITION_TEXTS
                },
                measurements={
                    kind: measure_audio(path) for kind in casting.AUDITION_TEXTS
                },
            )
        )
    project.character_cast.auditions[role.id] = prior
    project.character_cast.auditions["neighbor"][0].measurements = prior[
        0
    ].measurements.copy()
    snapshots = [c.identity.model_dump(mode="json") for c in prior]
    save_project(project)
    monkeypatch.setattr(casting, "refresh_cast", load_project)

    class Provider(OriginalVoiceTestProvider):
        def prepare_character_voice(self, project_id, identity_id, metadata):
            return {
                **super().prepare_character_voice(project_id, identity_id, metadata),
                "provenance": {"prepared": True},
                "generation_settings": {"base": "fixture"},
            }

    provider = Provider(root)
    casting.prepare_cast(project.id, provider=provider, role_ids=[role.id])
    saved = load_project(project.id)
    candidates = saved.character_cast.auditions[role.id]
    assert provider.design_calls == 1
    assert len(candidates) == cap
    assert [
        c.identity.model_dump(mode="json") for c in candidates[:previous_count]
    ] == snapshots
    assert all(not c.sample_history for c in candidates[:previous_count])
    chosen = next(r for r in saved.roles if r.id == role.id).voice_identity
    assert chosen.id == candidates[previous_count].id
    assert chosen.provenance["prepared"] is True
    assert chosen.provenance["ensemble_collision_retry"]["decisions"]
    if cap == 3:
        assert (
            "higher-mid register" in chosen.generation_settings["candidate_instruction"]
        )
    else:
        assert (
            "lightly grainy, dry timbre"
            in chosen.generation_settings["candidate_instruction"]
        )
        assert (
            chosen.provenance["ensemble_collision_retry"]["schema"]
            == "ensemble-collision-retry-v2"
        )
    assert saved.roles[1].voice_identity.id == "locked"
    casting.prepare_cast(project.id, provider=provider, role_ids=[role.id])
    assert provider.design_calls == 1


def completed_candidates(tmp_path, role_id, count=3):
    audio = make_tone_wav(tmp_path / "complete.wav", seconds=3, frequency=100)
    measurements = measure_audio(audio)
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}", encoding="utf-8")
    rows = []
    for index in range(count):
        voice = identity(f"candidate-{index}", role_id)
        voice.reference_audio = audio.name
        voice.reference_audio_sha256 = casting._sha(audio)
        voice.prompt_artifact = prompt.name
        voice.prompt_sha256 = casting._sha(prompt)
        rows.append(
            AuditionCandidate(
                id=voice.id,
                role_id=role_id,
                profile_revision="p1",
                identity=voice,
                audition_revision=casting.AUDITION_REVISION,
                samples={kind: audio.name for kind in casting.AUDITION_TEXTS},
                measurements={
                    kind: measurements.copy() for kind in casting.AUDITION_TEXTS
                },
            )
        )
    return rows


@pytest.mark.parametrize(
    "blocker",
    [
        "cap-three",
        "no-collision",
        "error",
        "two-complete",
        "missing-sample",
        "missing-prompt",
        "changed-reference",
        "changed-prompt",
        "audio-unusable",
        "locked",
        "stale-neighbor",
        "no-free-traits",
    ],
)
def test_fourth_budget_requires_three_complete_collision_failures(tmp_path, blocker):
    project, role, profile = collision_project()
    project.character_cast.policy.max_candidates = 4
    rows = completed_candidates(tmp_path, role.id)
    if blocker == "cap-three":
        project.character_cast.policy.max_candidates = 3
    elif blocker == "no-collision":
        project.character_cast.collisions = []
    elif blocker == "error":
        rows[0].error = "audition generation failed"
    elif blocker == "two-complete":
        rows.pop()
    elif blocker == "missing-sample":
        rows[0].samples.pop("quiet")
    elif blocker == "missing-prompt":
        rows[0].identity.prompt_artifact = "not-created"
    elif blocker == "changed-reference":
        rows[0].identity.reference_audio_sha256 = "0" * 64
    elif blocker == "changed-prompt":
        rows[0].identity.prompt_sha256 = "0" * 64
    elif blocker == "audio-unusable":
        rows[0].measurements["quiet"]["rms"] = 0
    elif blocker == "locked":
        role.voice_identity = identity("accepted", role.id)
    elif blocker == "stale-neighbor":
        project.roles[1].voice_identity = None
    elif blocker == "no-free-traits":
        profile.decisions = [
            CastingDecision(
                dimension=d, value="established", level="EXPLICIT", reason="book"
            )
            for d in ["pitch", "texture", "resonance", "articulation", "cadence"]
        ]
    assert casting._candidate_budget(project, role, profile, rows, tmp_path) < 4


def test_fourth_budget_is_allowed_only_at_four_and_never_exceeds_four(tmp_path):
    project, role, profile = collision_project()
    rows = completed_candidates(tmp_path, role.id, count=4)
    project.character_cast.policy.max_candidates = 4
    assert casting._candidate_budget(project, role, profile, rows, tmp_path) == 4
    assert casting._candidate_design(project, role, profile, 4)[1] is None


def test_fourth_timbre_variation_is_restrained_inferred_and_preserves_profile():
    project, role, profile = collision_project(
        decisions=[
            CastingDecision(
                dimension="texture",
                value="clear",
                level="INFERRED",
                reason="neutral default",
            ),
            CastingDecision(
                dimension="pitch",
                value="mid",
                level="INFERRED",
                reason="neutral default",
            ),
            CastingDecision(
                dimension="presentation",
                value="masculine",
                level="INFERRED",
                reason="supported pronouns",
            ),
        ]
    )
    before = profile.model_dump(mode="json")
    instruction, provenance = casting._candidate_design(project, role, profile, 3)
    assert "lightly grainy, dry timbre" in instruction
    assert "gently forward resonance with light nasality" in instruction
    assert "natural and understated" in instruction
    assert {d["dimension"] for d in provenance["decisions"]} == {
        "texture",
        "resonance",
        "articulation",
    }
    assert all(d["level"] == "INFERRED" for d in provenance["decisions"])
    assert profile.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "level,value",
    [("EXPLICIT", "clear"), ("SUPPORTED", "clear"), ("INFERRED", "breathy")],
)
def test_fourth_never_replaces_supported_or_distinctive_texture(level, value):
    project, role, profile = collision_project(
        decisions=[
            CastingDecision(
                dimension="texture",
                value=value,
                level=level,
                reason="existing requirement",
            ),
            CastingDecision(
                dimension="resonance",
                value="rounded",
                level=level,
                reason="existing requirement",
            ),
            CastingDecision(
                dimension="cadence",
                value="deliberate",
                level="SUPPORTED",
                reason="book",
            ),
        ]
    )
    instruction, provenance = casting._candidate_design(project, role, profile, 3)
    assert not provenance["decisions"]
    assert "grainy" not in instruction and "nasality" not in instruction
