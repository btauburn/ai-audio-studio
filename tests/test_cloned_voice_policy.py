"""Cloned-only voice policy: pure in-memory fixtures; no stores, models, network or GPU."""

import copy
import hashlib
from types import MappingProxyType, SimpleNamespace

import pytest

from dubbing_service.app.audiobook import cloned_voice_policy as policy
from dubbing_service.app.audiobook.cloned_voice_policy import ClonePolicyError
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Role

ONLY = "cloned_only"


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def clone_identity(**over) -> dict:
    identity = {
        "id": "voice_clone", "role_id": "r", "status": "LOCKED", "mechanism": "qwen_reference",
        "backend": "qwen3tts", "model": "m", "reference_audio": "casting/voices/v/reference.wav",
        "reference_audio_sha256": sha("ref"), "prompt_artifact": "casting/voices/v/prompt.npz",
        "prompt_sha256": sha("prompt"),
        "provenance": {"origin": "eligible_reference_clone", "actor_audio_used": True},
    }
    identity.update(over)
    return identity


def codes(holds) -> set:
    return {hold["code"] for hold in holds}


def model_identity(**over) -> CharacterVoiceIdentity:
    base = clone_identity(**over)
    base.update(dossier_revision="d", casting_profile_revision="c", design_instruction="x")
    return CharacterVoiceIdentity(**base)


def discovery_candidate(**over) -> dict:
    """Shape produced by casting_director_discovery (conditioning + stored provenance)."""
    row = {
        "schema": "casting-candidate-v1", "origin": "generic_library", "kind": "reference_clone",
        "reference_conditioned": True,
        "provenance": {"stored_origin": "eligible_reference_clone", "actor_audio_used": True},
        "conditioning": {"mechanism": "qwen_reference", "backend": "qwen3tts",
                         "reference_audio_sha256": sha("ref"), "prompt_sha256": sha("prompt")},
    }
    row.update(over)
    return row


# ---- policy_mode ----------------------------------------------------------------------

def test_absent_policy_is_unrestricted_for_objects_and_mappings():
    assert policy.policy_mode(SimpleNamespace(metadata={})) == "any"
    assert policy.policy_mode(SimpleNamespace(metadata=None)) == "any"
    assert policy.policy_mode(SimpleNamespace()) == "any"
    assert policy.policy_mode({"metadata": {"other": 1}}) == "any"


@pytest.mark.parametrize("mode", ["any", "cloned_only"])
def test_valid_modes_round_trip_without_mutating_metadata(mode):
    metadata = {"director_voice_policy": {"mode": mode, "note": "kept"}}
    before = copy.deepcopy(metadata)
    assert policy.policy_mode(SimpleNamespace(metadata=metadata)) == mode
    assert metadata == before


@pytest.mark.parametrize("bad", [
    "cloned_only", None, {}, {"mode": None}, {"mode": "clone_only"}, {"mode": "CLONED_ONLY"},
    {"mode": ["cloned_only"]}, {"mode": {"x": 1}}, ["cloned_only"], 1,
])
def test_present_but_invalid_policy_fails_closed_instead_of_relaxing(bad):
    with pytest.raises(ClonePolicyError):
        policy.policy_mode(SimpleNamespace(metadata={"director_voice_policy": bad}))


def test_non_mapping_metadata_is_rejected():
    with pytest.raises(ClonePolicyError):
        policy.policy_mode(SimpleNamespace(metadata=["director_voice_policy"]))


@pytest.mark.parametrize("fn", [policy.record_holds, policy.candidate_holds, policy.role_holds])
@pytest.mark.parametrize("bad_mode", ["clone", "", None, "ANY"])
def test_checks_reject_unknown_mode(fn, bad_mode):
    with pytest.raises(ClonePolicyError):
        fn({}, mode=bad_mode)


def test_any_mode_changes_nothing_even_for_presets_and_garbage():
    for value in ({}, {"kind": "provider_builtin", "builtin_speaker": "Ryan"}, None, 5):
        assert policy.record_holds(value, mode="any") == []
        assert policy.candidate_holds(value, mode="any") == []
        assert policy.role_holds(value, mode="any") == []
    policy.require_record({"kind": "provider_builtin"}, mode="any")


# ---- genuine clones -------------------------------------------------------------------

def test_genuine_clone_passes_all_entry_points():
    record = {"kind": "reference_clone", "identity": clone_identity()}
    assert policy.record_holds(record, mode=ONLY) == []
    assert policy.record_holds(clone_identity(), mode=ONLY) == []  # bare identity
    assert policy.candidate_holds({"id": "c", "identity": clone_identity()}, mode=ONLY) == []
    assert policy.candidate_holds(discovery_candidate(), mode=ONLY) == []
    assert policy.role_holds({"voice_identity": clone_identity()}, mode=ONLY) == []
    assert policy.require_record(record, mode=ONLY) is None


def test_pydantic_identity_and_role_are_judged_by_model_dump_json():
    identity = model_identity()
    role = Role(id="r", name="R", voice_identity=identity)
    assert policy.role_holds(role, mode=ONLY) == []
    assert policy.record_holds({"identity": identity}, mode=ONLY) == []
    assert policy.candidate_holds(SimpleNamespace(model_dump=lambda mode: {"identity": identity.model_dump(mode="json")}),
                                  mode=ONLY) == []


def test_pydantic_role_with_non_clone_identity_is_held():
    role = Role(id="r", name="R", voice_identity=model_identity(mechanism="builtin", builtin_speaker="Ryan", backend="qwen3tts_cv"))
    assert {"clone_policy_provider_preset", "clone_policy_backend"} <= codes(policy.role_holds(role, mode=ONLY))


def test_role_without_identity_is_held():
    assert codes(policy.role_holds(Role(id="r", name="R"), mode=ONLY)) == {"clone_policy_no_voice_identity"}
    assert codes(policy.role_holds({"voice_identity": None}, mode=ONLY)) == {"clone_policy_no_voice_identity"}


def test_approval_rating_and_missing_rights_evidence_are_not_consulted():
    """Historical approved clone without eligibility evidence stays allowed; the policy neither adds nor needs rights."""
    record = {"identity": clone_identity(), "rating": 5, "approved": True, "preferred": {"ratings": [{"rating": 5}]}}
    assert policy.record_holds(record, mode=ONLY) == []
    unrated = {"identity": clone_identity(), "rating": 1, "approved": False}
    assert policy.record_holds(unrated, mode=ONLY) == []  # rating is a different owner's gate


def test_consent_claims_cannot_make_a_non_clone_eligible():
    identity = clone_identity(provenance={"origin": "eligible_reference_clone", "actor_audio_used": False,
                                          "eligibility": {"acquisition_eligible": True, "intended_use_evidence": {
                                              "status": "approved", "evidence_ref": "x", "performer_consent_claimed": True}}})
    assert codes(policy.record_holds({"identity": identity, "approved": True}, mode=ONLY)) == {"clone_policy_actor_audio"}


# ---- negative: each requirement independently --------------------------------------------

@pytest.mark.parametrize("over,expected", [
    ({"mechanism": "builtin", "builtin_speaker": "Ryan"}, "clone_policy_provider_preset"),
    ({"mechanism": "existing_reference"}, "clone_policy_mechanism"),
    ({"mechanism": "fixture"}, "clone_policy_mechanism"),
    ({"backend": "qwen3tts_cv"}, "clone_policy_backend"),
    ({"backend": "indextts2"}, "clone_policy_backend"),
    ({"backend": None}, "clone_policy_backend"),
    ({"reference_audio": None}, "clone_policy_reference_audio_missing"),
    ({"reference_audio": "  "}, "clone_policy_reference_audio_missing"),
    ({"prompt_artifact": None}, "clone_policy_prompt_missing"),
    ({"reference_audio_sha256": None}, "clone_policy_reference_audio_hash"),
    ({"reference_audio_sha256": "abc123"}, "clone_policy_reference_audio_hash"),
    ({"reference_audio_sha256": sha("ref").upper()}, "clone_policy_reference_audio_hash"),
    ({"reference_audio_sha256": "0" * 64}, "clone_policy_reference_audio_hash"),
    ({"reference_audio_sha256": "ab" * 32}, "clone_policy_reference_audio_hash"),
    ({"reference_audio_sha256": "g" * 64}, "clone_policy_reference_audio_hash"),
    ({"prompt_sha256": None}, "clone_policy_prompt_hash"),
    ({"prompt_sha256": "f" * 64}, "clone_policy_prompt_hash"),
    ({"prompt_sha256": sha("ref")}, "clone_policy_hash_reuse"),
    ({"provenance": {}}, "clone_policy_origin"),
    ({"provenance": None}, "clone_policy_origin"),
    ({"provenance": {"origin": "unknown"}}, "clone_policy_origin"),
    ({"provenance": {"origin": "eligible_reference_clone"}}, "clone_policy_actor_audio"),
    ({"provenance": {"origin": "eligible_reference_clone", "actor_audio_used": False}}, "clone_policy_actor_audio"),
    ({"provenance": {"origin": "eligible_reference_clone", "actor_audio_used": "true"}}, "clone_policy_actor_audio"),
    ({"provenance": {"origin": "eligible_reference_clone", "actor_audio_used": 1}}, "clone_policy_actor_audio"),
])
def test_each_missing_clone_requirement_holds_every_entry_point(over, expected):
    identity = clone_identity(**over)
    assert expected in codes(policy.record_holds({"identity": identity}, mode=ONLY))
    assert expected in codes(policy.candidate_holds({"identity": identity}, mode=ONLY))
    assert expected in codes(policy.role_holds({"voice_identity": identity}, mode=ONLY))
    with pytest.raises(ClonePolicyError) as err:
        policy.require_record({"identity": identity}, mode=ONLY)
    assert expected in codes(err.value.holds) and expected in str(err.value)


@pytest.mark.parametrize("actor", [True, False, None])
def test_reference_conditioned_generated_original_voice_is_rejected(actor):
    identity = clone_identity(provenance={"origin": "generated_original_voice", "actor_audio_used": actor})
    assert codes(policy.record_holds({"identity": identity}, mode=ONLY)) == {"clone_policy_generated_original_voice"}
    assert codes(policy.role_holds({"voice_identity": identity}, mode=ONLY)) == {"clone_policy_generated_original_voice"}
    candidate = discovery_candidate(provenance={"stored_origin": "generated_original_voice", "actor_audio_used": actor})
    assert codes(policy.candidate_holds(candidate, mode=ONLY)) == {"clone_policy_generated_original_voice"}


def test_approved_non_clone_is_still_held():
    preset = {"kind": "provider_builtin", "builtin_speaker": "Ryan", "approved": True, "rating": 5,
              "identity": clone_identity(mechanism="builtin", backend="qwen3tts_cv", builtin_speaker="Ryan", provenance={})}
    assert "clone_policy_provider_preset" in codes(policy.record_holds(preset, mode=ONLY))
    original = {"kind": "original_inspired", "approved": True, "rating": 5,
                "identity": clone_identity(provenance={"origin": "generated_original_voice", "actor_audio_used": False})}
    assert "clone_policy_generated_original_voice" in codes(policy.record_holds(original, mode=ONLY))


def test_record_kind_cannot_be_overridden_by_a_clone_looking_identity():
    assert codes(policy.record_holds({"kind": "provider_builtin", "identity": clone_identity()}, mode=ONLY)) \
        == {"clone_policy_provider_preset"}
    assert codes(policy.record_holds({"kind": "original_inspired", "identity": clone_identity()}, mode=ONLY)) \
        == {"clone_policy_generated_original_voice"}
    assert codes(policy.record_holds({"builtin_speaker": "Ryan", "identity": clone_identity()}, mode=ONLY)) \
        == {"clone_policy_provider_preset"}


@pytest.mark.parametrize("record", [{}, {"identity": None}, {"identity": {}}, {"kind": "reference_clone"}, 5, "text", [], None])
def test_missing_or_unreadable_identity_is_unknown_provenance_and_rejected(record):
    assert policy.record_holds(record, mode=ONLY)
    with pytest.raises(ClonePolicyError):
        policy.require_record(record, mode=ONLY)


def test_preset_without_any_identity_is_held_as_preset():
    assert codes(policy.record_holds({"kind": "provider_builtin", "builtin_speaker": "Ryan"}, mode=ONLY)) \
        == {"clone_policy_provider_preset", "clone_policy_no_voice_identity"}


def test_model_design_candidate_without_audio_is_held():
    design = {"schema": "casting-candidate-v1", "origin": "model_design", "kind": "model_design", "conditioning": None,
              "provenance": {"rights_status": "original_synthetic_no_recording", "actor_audio_used": False}}
    assert {"clone_policy_model_design", "clone_policy_no_voice_identity"} <= codes(policy.candidate_holds(design, mode=ONLY))


def test_model_design_tier_is_held_even_with_clone_looking_conditioning():
    assert "clone_policy_model_design" in codes(policy.candidate_holds(discovery_candidate(origin="model_design"), mode=ONLY))


@pytest.mark.parametrize("candidate,expected", [
    (discovery_candidate(conditioning=None), "clone_policy_no_voice_identity"),
    (discovery_candidate(conditioning={"mechanism": "qwen_reference", "backend": "qwen3tts"}), "clone_policy_reference_audio_hash"),
    (discovery_candidate(provenance={}), "clone_policy_origin"),
    (discovery_candidate(provenance={"stored_origin": "eligible_reference_clone"}), "clone_policy_actor_audio"),
    (discovery_candidate(kind="provider_builtin"), "clone_policy_provider_preset"),
    ({"id": "c", "identity": None}, "clone_policy_no_voice_identity"),
])
def test_candidate_shapes_reject_incomplete_clones(candidate, expected):
    assert expected in codes(policy.candidate_holds(candidate, mode=ONLY))


def test_reference_conditioned_flag_alone_is_not_trusted():
    assert policy.candidate_holds({"reference_conditioned": True, "origin": "generic_library"}, mode=ONLY)


def test_holds_have_stable_shape_and_are_deduplicated():
    holds = policy.record_holds({"kind": "provider_builtin", "builtin_speaker": "Ryan",
                                 "identity": clone_identity(mechanism="builtin", builtin_speaker="Ryan")}, mode=ONLY)
    assert all(set(hold) == {"code", "detail"} and hold["code"].startswith("clone_policy_") for hold in holds)
    assert len({hold["code"] for hold in holds}) == len(holds)


# ---- purity ---------------------------------------------------------------------------------

def test_checks_never_mutate_inputs_or_their_nested_state():
    record = {"kind": "reference_clone", "approved": True, "identity": clone_identity(provenance={"origin": "x"})}
    candidate = discovery_candidate(provenance={})
    project = SimpleNamespace(metadata={"director_voice_policy": {"mode": ONLY}})
    snapshots = [copy.deepcopy(item) for item in (record, candidate, project.metadata)]
    policy.record_holds(record, mode=ONLY)
    policy.candidate_holds(candidate, mode=ONLY)
    policy.role_holds({"voice_identity": record["identity"]}, mode=ONLY)
    with pytest.raises(ClonePolicyError):
        policy.require_record(record, mode=ONLY)
    policy.policy_mode(project)
    assert [record, candidate, project.metadata] == snapshots


def test_read_only_mappings_are_accepted():
    frozen = MappingProxyType({"identity": MappingProxyType({**clone_identity(), "provenance": MappingProxyType(clone_identity()["provenance"])})})
    assert policy.record_holds(frozen, mode=ONLY) == []
    assert policy.policy_mode(SimpleNamespace(metadata=MappingProxyType({"director_voice_policy": MappingProxyType({"mode": ONLY})}))) == ONLY


def test_module_imports_no_storage_or_runtime_owner():
    import inspect
    source = inspect.getsource(policy)
    assert "import " in source and not any(token in source for token in (
        "storage", "open(", "Path(", "requests", "torch", "subprocess", "save_", "write_"))
