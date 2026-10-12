"""CPU-only tests for the two-arm scene-aware dialogue comparison harness.

All audio here is artificial (sine tones from ``build_synthetic_fixture``).  Nothing is
speech, nothing is a model result, and no test touches a live store, the network, a model
or the GPU.  Source/private mutations are simulated by editing files inside ``tmp_path``.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import director_evaluation as de
from dubbing_service.app.audiobook import performance_provider as pp
from dubbing_service.app.audiobook import scene_dialogue_evaluation as sde

DIRECTED_LINES = ("fx-02", "fx-03", "fx-04", "fx-06", "fx-07", "fx-08", "fx-10", "fx-11", "fx-12")
NARRATOR_LINES = ("fx-01", "fx-05", "fx-09")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def fixture_manifest(tmp_path):
    return sde.build_synthetic_fixture(tmp_path / "inputs")


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): sde.sha256_hex(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def root_of(manifest) -> Path:
    return Path(manifest["base_dir"])


def sample(manifest, arm, utterance_id):
    return manifest["arms"][arm]["samples"][utterance_id]


def edit_receipt(manifest, arm, utterance_id, change):
    """Rewrite one saved worker receipt in place (test-only mutation of a private source file)."""
    path = root_of(manifest) / sample(manifest, arm, utterance_id)["receipt"]["path"]
    body = json.loads(path.read_text("utf-8"))
    change(body)
    path.write_bytes(de._pretty(body))
    return path


def production_like(manifest):
    """A manifest without the fixture label, for exercising the human gate (test-only data)."""
    clone = copy.deepcopy(manifest)
    clone.pop("fixture")
    for arm in sde.ARMS:
        for u in clone["arms"][arm].get("samples", {}):
            edit_receipt(clone, arm, u, lambda body: body.pop("synthetic_fixture"))
    return clone


def refused(manifest, code, **kwargs):
    with pytest.raises(sde.EvaluationError) as raised:
        sde.evaluate_scene_pilot(manifest, **kwargs)
    assert raised.value.code == code, raised.value
    return raised.value


def make_review(result, scorer, *, kind="human"):
    """Build a completed review; ``scorer(arm, narrator) -> rating`` is applied via the sealed key."""
    key = result["ballot_key"]
    scenes = []
    for ballot in result["ballots"]["scenes"]:
        responses = []
        for item in ballot["items"]:
            options = key["scenes"][ballot["scene_id"]]["items"][item["item_id"]]
            ratings = {
                label: {d: scorer(options["options"][label]["arm"], options["narrator"]) for d in sde.RATING_DIMENSIONS}
                for label in options["options"]
            }
            responses.append({"item_id": item["item_id"], "ratings": ratings, "preferred_label": None})
        scenes.append({"ballot_id": ballot["ballot_id"], "responses": responses})
    return {"ballot_set_id": result["ballots"]["ballot_set_id"], "reviewer_id": "brian", "reviewer_kind": kind,
            "scenes": scenes}


# --------------------------------------------------------------------------- #
# the happy path and the labeled fixture
# --------------------------------------------------------------------------- #


def test_fixture_is_labeled_complete_review_only_and_excluded_from_readiness(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    assert result["fixture"] == {"is_fixture": True, "label": sde.FIXTURE_LABEL}
    assert result["status"] == "complete" and result["arm_status"] == {"baseline": "complete", "directed": "complete"}
    assert result["production_eligible"] is False and result["review_only"] is True
    assert result["human_gate"]["status"] == "fixture_excluded"
    assert result["human_gate"]["approval"] is False and result["human_gate"]["default_changed"] is False
    assert result["promotion"] == {"approval": False, "default_changed": False,
                                   "human_listening": "final and still required"}
    assert result["lines"]["count"] == 12 and result["lines"]["narrator_lines"] == 3
    assert {r["role_id"] for r in result["lines"]["roles"] if not r["narrator"]} == {"ada", "bo"}


def test_instruction_proof_per_line_matches_the_contract(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    baseline = {s["utterance_id"]: s for s in result["arms"]["baseline"]["samples"]}
    directed = {s["utterance_id"]: s for s in result["arms"]["directed"]["samples"]}
    assert result["arms"]["baseline"]["summary"]["forwarded_instructions"] == 0
    assert result["arms"]["directed"]["summary"]["forwarded_instructions"] == len(DIRECTED_LINES)
    assert result["arms"]["directed"]["summary"]["narrator_instructions"] == 0
    for u in baseline:
        assert baseline[u]["instruction"]["forwarded"] is False
        assert baseline[u]["instruction"]["direction_sha256"] == pp.component_sha256(None, 0)
        assert baseline[u]["voice"] == directed[u]["voice"]
    for u in NARRATOR_LINES:
        assert directed[u]["instruction"]["native_instruction"] is None
        assert directed[u]["instruction"]["forwarded"] is False
    for u in DIRECTED_LINES:
        text = directed[u]["instruction"]["native_instruction"]
        assert directed[u]["instruction"]["forwarded"] is True
        assert directed[u]["instruction"]["instruct_sha256"] == sde.sha256_hex(text.encode())
        assert directed[u]["instruction"]["direction_sha256"] == pp.component_sha256(text, 0)


def test_expected_receipt_helper_matches_the_real_provider_digest():
    for instruction in (None, "Speak quietly, with worry held back."):
        for salt in (0, 7):
            receipt = sde.expected_direction_receipt(instruction, identity_id="v", text="Hello.", seed_salt=salt)
            assert set(receipt) == pp.RECEIPT_KEYS
            assert receipt["direction_sha256"] == pp.component_sha256(instruction, salt)
            assert receipt["forwarded"] is (instruction is not None)


def test_unavailable_measures_are_null_never_estimated(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    assert result["unavailable_measures"]["asr"]["word_error_rate"] is None
    assert result["unavailable_measures"]["acoustic_voice_embedding"]["similarity"] is None
    for arm in sde.ARMS:
        for s in result["arms"][arm]["samples"]:
            assert s["asr"]["status"] == "unavailable" and s["asr"]["omissions"] is None
            assert s["acoustic_voice_embedding"]["similarity"] is None
            assert s["human_measures"] == {"status": "pending"}


def test_measurements_come_from_the_actual_audio_and_receipt(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    s = result["arms"]["directed"]["samples"][1]
    wav = (root_of(fixture_manifest) / sample(fixture_manifest, "directed", s["utterance_id"])["audio_file"]).read_bytes()
    assert s["measures"]["duration_seconds"] == de.measure_audio(wav)["duration_seconds"]
    assert s["receipt"]["latency_seconds"] == pytest.approx(0.6) and s["receipt"]["peak_vram_mb"] == 1001.0
    comparison = result["comparisons"]["directed_vs_baseline"]
    assert comparison["changed_audio_dialogue_lines"] == len(DIRECTED_LINES)
    assert comparison["changed_audio_narrator_lines"] == 0
    assert comparison["mean_latency_seconds_delta"] == pytest.approx(0.1)
    assert comparison["max_peak_vram_mb_delta"] == 1.0
    rows = {r["utterance_id"]: r for r in comparison["per_utterance"]}
    assert rows["fx-02"]["duration_delta_seconds"] == pytest.approx(0.1) and rows["fx-01"]["duration_delta_seconds"] == 0.0


def test_unknown_latency_and_vram_stay_unknown(fixture_manifest):
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: b.update(latency_seconds=None, peak_vram_mb=None))
    result = sde.evaluate_scene_pilot(fixture_manifest)
    s = next(s for s in result["arms"]["directed"]["samples"] if s["utterance_id"] == "fx-02")
    assert s["receipt"]["latency_seconds"] is None and s["receipt"]["peak_vram_mb"] is None
    summary = result["arms"]["directed"]["summary"]
    assert summary["latency_seconds"]["complete"] is False and summary["peak_vram_mb"]["complete"] is False
    assert result["comparisons"]["directed_vs_baseline"]["mean_latency_seconds_delta"] is None
    assert any("latency or VRAM is unknown" in w for w in result["warnings"])


@pytest.mark.parametrize("value", [-1.0, True, "1.2", float("nan"), float("inf")])
def test_malformed_latency_is_refused_not_nulled(fixture_manifest, value):
    edit_receipt(fixture_manifest, "baseline", "fx-01", lambda b: b.update(latency_seconds=value))
    refused(fixture_manifest, "receipt_malformed")


def test_manifest_object_is_not_mutated(fixture_manifest):
    before = copy.deepcopy(fixture_manifest)
    sde.evaluate_scene_pilot(fixture_manifest)
    assert fixture_manifest == before


def test_clean_measurement_helpers_are_the_existing_director_evaluation_ones():
    assert sde.measure_audio is de.measure_audio and sde.MEASUREMENT_POLICY is de.MEASUREMENT_POLICY


# --------------------------------------------------------------------------- #
# the frozen line set
# --------------------------------------------------------------------------- #


def test_requires_twelve_lines(fixture_manifest):
    fixture_manifest["lines"] = fixture_manifest["lines"][:11]
    refused(fixture_manifest, "manifest_invalid")


def test_requires_narration_and_two_dialogue_roles(fixture_manifest):
    no_narrator = copy.deepcopy(fixture_manifest)
    for line in no_narrator["lines"]:
        line["narrator"] = False
    refused(no_narrator, "manifest_invalid")
    one_role = copy.deepcopy(fixture_manifest)
    for line in one_role["lines"]:
        if line["role_id"] == "bo":
            line["role_id"], line["role_name"] = "ada", "Ada"
    refused(one_role, "manifest_invalid")


def test_a_role_cannot_be_narrator_and_dialogue(fixture_manifest):
    fixture_manifest["lines"][1]["role_id"] = "narrator"
    refused(fixture_manifest, "manifest_invalid")


def test_narrator_flag_must_be_a_real_boolean_and_ids_unique(fixture_manifest):
    bad = copy.deepcopy(fixture_manifest)
    bad["lines"][0]["narrator"] = "true"
    refused(bad, "manifest_invalid")
    dup = copy.deepcopy(fixture_manifest)
    dup["lines"][1]["utterance_id"] = dup["lines"][0]["utterance_id"]
    refused(dup, "manifest_invalid")


def test_dialogue_over_the_codec_budget_is_refused_but_narration_is_not_limited(fixture_manifest):
    long_text = "x" * (sde.MAX_DIALOGUE_CHARS + 1)
    dialogue = copy.deepcopy(fixture_manifest)
    dialogue["lines"][1]["text"] = long_text
    refused(dialogue, "line_too_long")
    narrator = copy.deepcopy(fixture_manifest)
    narrator["lines"][0]["text"] = long_text
    narrator["source"]["lines_sha256"] = sde.lines_sha256(narrator["lines"])
    # the narrator line's samples no longer match, which is a different refusal
    refused(narrator, "text_mismatch")


def test_scenes_must_be_contiguous_and_single_project(fixture_manifest):
    split = copy.deepcopy(fixture_manifest)
    split["lines"][4]["scene_id"] = "fx-scene-01"  # scene 1 reappears after being closed? no: contiguous with 02
    split["lines"][8]["scene_id"] = "fx-scene-01"
    refused(split, "scene_not_contiguous")
    mixed = copy.deepcopy(fixture_manifest)
    mixed["source"]["projects"].append({"project_id": "other", "binding_sha256": sde.sha256_hex(b"b")})
    mixed["lines"][1]["source"]["project_id"] = "other"
    refused(mixed, "scene_not_contiguous")


def test_source_binding_must_match_declared_projects_and_line_hash(fixture_manifest):
    unknown = copy.deepcopy(fixture_manifest)
    unknown["lines"][0]["source"]["project_id"] = "ghost"
    unknown["lines"][1]["source"]["project_id"] = "ghost"
    unknown["lines"][2]["source"]["project_id"] = "ghost"
    unknown["lines"][3]["source"]["project_id"] = "ghost"
    refused(unknown, "source_mismatch")
    stale = copy.deepcopy(fixture_manifest)
    stale["source"]["lines_sha256"] = sde.sha256_hex(b"other")
    refused(stale, "source_mismatch")
    rebound = copy.deepcopy(fixture_manifest)
    rebound["lines"][3]["source"]["source_sha256"] = sde.sha256_hex(b"mutated source")
    refused(rebound, "source_mismatch")
    no_projects = copy.deepcopy(fixture_manifest)
    no_projects["source"]["projects"] = []
    refused(no_projects, "manifest_invalid")


def test_mutating_a_source_line_is_caught_everywhere(fixture_manifest):
    mutated = copy.deepcopy(fixture_manifest)
    mutated["lines"][1]["text"] += " (edited)"
    refused(mutated, "source_mismatch")  # frozen line hash
    mutated["source"].pop("lines_sha256")
    refused(mutated, "text_mismatch")  # even without the hash, every sample still carries the old text


def test_sample_text_must_equal_the_source_line_in_either_arm(fixture_manifest):
    for arm in sde.ARMS:
        bad = copy.deepcopy(fixture_manifest)
        sample(bad, arm, "fx-03")["text"] += "!"
        refused(bad, "text_mismatch")
        bad = copy.deepcopy(fixture_manifest)
        sample(bad, arm, "fx-03")["text_sha256"] = sde.sha256_hex(b"other")
        refused(bad, "text_mismatch")


def test_receipt_must_name_the_same_text(fixture_manifest):
    edit_receipt(fixture_manifest, "directed", "fx-03", lambda b: b.update(text_sha256=sde.sha256_hex(b"other")))
    refused(fixture_manifest, "receipt_mismatch")


# --------------------------------------------------------------------------- #
# instructions: baseline / narrator / neutral / identity words
# --------------------------------------------------------------------------- #


def test_baseline_may_not_carry_an_instruction(fixture_manifest):
    s = sample(fixture_manifest, "baseline", "fx-02")
    s["instruction"] = {"native_instruction": "Speak quietly.", "forwarded": True}
    refused(fixture_manifest, "baseline_instruction")


def test_baseline_receipt_may_not_secretly_forward(fixture_manifest):
    # metadata is clean, but the saved worker receipt says the instruction was forwarded
    def forward(body):
        body["native_instruction"] = "Speak quietly."
        body["worker_receipt"]["generation"]["direction"] = sde.expected_direction_receipt(
            "Speak quietly.", identity_id="fixture-voice-ada", text=sample(fixture_manifest, "baseline", "fx-02")["text"])

    edit_receipt(fixture_manifest, "baseline", "fx-02", forward)
    refused(fixture_manifest, "forwarding_mismatch")


def test_narrator_may_not_carry_an_instruction_in_metadata_or_receipt(fixture_manifest):
    meta = copy.deepcopy(fixture_manifest)
    sample(meta, "directed", "fx-01")["instruction"] = {"native_instruction": "Speak warmly.", "forwarded": True}
    refused(meta, "narrator_instruction")

    def forward(body):
        body["native_instruction"] = "Speak warmly."
        body["worker_receipt"]["generation"]["direction"] = sde.expected_direction_receipt(
            "Speak warmly.", identity_id="fixture-voice-narrator", text=sample(fixture_manifest, "directed", "fx-01")["text"])

    edit_receipt(fixture_manifest, "directed", "fx-01", forward)
    refused(fixture_manifest, "forwarding_mismatch")


def test_narrator_line_plan_with_instruction_is_refused(fixture_manifest):
    fixture_manifest["lines"][0]["direction"]["native_instruction"] = "Speak warmly."
    refused(fixture_manifest, "narrator_instruction")


@pytest.mark.parametrize("text", ["Speak neutrally.", "neutral", "Neutral tone", "Say it in a plain voice.",
                                  "Read it normally.", "No emotion", "Speak in a flat tone.", "default"])
def test_neutral_placeholder_instructions_are_refused(fixture_manifest, text):
    sample(fixture_manifest, "directed", "fx-02")["instruction"] = {"native_instruction": text, "forwarded": True}
    fixture_manifest["lines"][1]["direction"]["native_instruction"] = text
    refused(fixture_manifest, "neutral_instruction")


@pytest.mark.parametrize("text", [
    "Speak in a deeper voice.", "Use a British accent.", "Raise the pitch slightly.", "Sound like a young woman.",
    "Speak as a male voice.", "Change the voice to another speaker.", "Imitate the actor.", "Speak in the voice of a pirate.",
    "Sound like an old man.", "Use a different speaker identity.",
])
def test_instructions_may_not_change_who_is_speaking(fixture_manifest, text):
    sample(fixture_manifest, "directed", "fx-02")["instruction"] = {"native_instruction": text, "forwarded": True}
    fixture_manifest["lines"][1]["direction"]["native_instruction"] = text
    error = refused(fixture_manifest, "").__class__ if False else None  # placeholder to keep flake quiet
    with pytest.raises(sde.EvaluationError) as raised:
        sde.evaluate_scene_pilot(fixture_manifest)
    assert raised.value.code in {"instruction_identity_words", "instruction_refused"}


@pytest.mark.parametrize("text", ["", "   ", " Speak quietly. ", "Speak\nquietly.", "x" * 241, "Speak at 120 words per minute."])
def test_malformed_instructions_are_refused(fixture_manifest, text):
    sample(fixture_manifest, "directed", "fx-02")["instruction"] = {"native_instruction": text, "forwarded": True}
    fixture_manifest["lines"][1]["direction"]["native_instruction"] = text
    refused(fixture_manifest, "instruction_refused")


def test_scene_hesitation_wording_is_allowed(fixture_manifest):
    text = "Speak with a held breath, as if afraid to be overheard."
    sample(fixture_manifest, "directed", "fx-02")["instruction"] = {"native_instruction": text, "forwarded": True}
    fixture_manifest["lines"][1]["direction"]["native_instruction"] = text
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: (
        b.update(native_instruction=text),
        b["worker_receipt"]["generation"].update(direction=sde.expected_direction_receipt(
            text, identity_id="fixture-voice-ada", text=sample(fixture_manifest, "directed", "fx-02")["text"]))))
    assert sde.evaluate_scene_pilot(fixture_manifest)["status"] == "complete"


def test_forwarded_flag_must_match_instruction_presence(fixture_manifest):
    claimed = copy.deepcopy(fixture_manifest)
    sample(claimed, "directed", "fx-02")["instruction"]["forwarded"] = False
    refused(claimed, "forwarding_mismatch")
    silent = copy.deepcopy(fixture_manifest)
    sample(silent, "directed", "fx-01")["instruction"]["forwarded"] = True
    refused(silent, "forwarding_mismatch")
    not_bool = copy.deepcopy(fixture_manifest)
    sample(not_bool, "directed", "fx-02")["instruction"]["forwarded"] = "yes"
    refused(not_bool, "manifest_invalid")


def test_directed_arm_must_forward_something(fixture_manifest):
    for u in DIRECTED_LINES:
        sample(fixture_manifest, "directed", u)["instruction"] = {"native_instruction": None, "forwarded": False}
        fixture_manifest["lines"][int(u[3:]) - 1]["direction"] = None
        edit_receipt(fixture_manifest, "directed", u, lambda b, u=u: (
            b.update(native_instruction=None),
            b["worker_receipt"]["generation"].update(direction=sde.expected_direction_receipt(
                None, identity_id=b["worker_receipt"]["identity_id"], text=sample(fixture_manifest, "directed", u)["text"]))))
    refused(fixture_manifest, "direction_mismatch")


def test_dialogue_without_a_grounded_direction_is_an_undirected_control_not_a_failure(fixture_manifest):
    u = "fx-04"
    sample(fixture_manifest, "directed", u)["instruction"] = {"native_instruction": None, "forwarded": False}
    fixture_manifest["lines"][3]["direction"]["native_instruction"] = None
    edit_receipt(fixture_manifest, "directed", u, lambda b: (
        b.update(native_instruction=None),
        b["worker_receipt"]["generation"].update(direction=sde.expected_direction_receipt(
            None, identity_id=b["worker_receipt"]["identity_id"], text=sample(fixture_manifest, "directed", u)["text"]))))
    result = sde.evaluate_scene_pilot(fixture_manifest)
    assert result["arms"]["directed"]["summary"]["forwarded_instructions"] == len(DIRECTED_LINES) - 1
    assert {"code": "dialogue_left_undirected", "utterance_ids": [u]}.items() <= next(
        f for f in result["findings"] if f["code"] == "dialogue_left_undirected").items()


def test_planned_instruction_must_be_the_forwarded_one(fixture_manifest):
    fixture_manifest["lines"][1]["direction"]["native_instruction"] = "Speak with open delight."
    refused(fixture_manifest, "direction_plan_mismatch")


@pytest.mark.parametrize("change", [
    {"approved": True}, {"human_review": "approved"}, {"origin": "human_override"}, {"origin": "approved_by_brian"},
    {"recommendation_only": False}, {"recommendation_only": None},
])
def test_plans_may_not_disguise_proposals_as_human_decisions(fixture_manifest, change):
    fixture_manifest["lines"][1]["direction"].update(change)
    refused(fixture_manifest, "direction_claim_refused")


def test_unknown_evidence_and_confidence_stay_unknown(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    plan = result["lines"]["utterances"][1]["direction_plan"]
    assert plan["evidence_status"] == "unknown" and plan["confidence"] == "unknown"
    assert plan["recommendation_only"] is True and plan["human_review"] == "pending"
    del fixture_manifest["lines"][1]["direction"]["evidence_status"]
    result = sde.evaluate_scene_pilot(fixture_manifest)
    assert result["lines"]["utterances"][1]["direction_plan"]["evidence_status"] == "unknown"


# --------------------------------------------------------------------------- #
# receipts: forwarding is proven by the saved worker receipt, never by metadata
# --------------------------------------------------------------------------- #


def _direction(body):
    return body["worker_receipt"]["generation"]["direction"]


@pytest.mark.parametrize("name,change,code", [
    ("missing_worker_receipt", lambda b: b.pop("worker_receipt"), "receipt_malformed"),
    ("missing_direction", lambda b: b["worker_receipt"]["generation"].pop("direction"), "receipt_malformed"),
    ("empty_direction", lambda b: b["worker_receipt"]["generation"].update(direction={}), "receipt_malformed"),
    ("extra_key", lambda b: _direction(b).update(extra=1), "receipt_malformed"),
    ("missing_key", lambda b: _direction(b).pop("seed_effective"), "receipt_malformed"),
    ("forwarded_false", lambda b: _direction(b).update(forwarded=False), "forwarding_mismatch"),
    ("forwarded_truthy_int", lambda b: _direction(b).update(forwarded=1), "forwarding_mismatch"),
    ("forwarded_string", lambda b: _direction(b).update(forwarded="true"), "forwarding_mismatch"),
    ("wrong_instruct_sha", lambda b: _direction(b).update(instruct_sha256=sde.sha256_hex(b"other")), "forwarding_mismatch"),
    ("null_instruct_sha", lambda b: _direction(b).update(instruct_sha256=None), "forwarding_mismatch"),
    ("wrong_chars", lambda b: _direction(b).update(instruct_chars=3), "forwarding_mismatch"),
    ("wrong_policy", lambda b: _direction(b).update(policy="other"), "forwarding_mismatch"),
    ("wrong_conditioning", lambda b: _direction(b).update(conditioning_mode="x-vector"), "forwarding_mismatch"),
    ("wrong_direction_sha", lambda b: _direction(b).update(direction_sha256=sde.sha256_hex(b"other")), "forwarding_mismatch"),
    ("seed_effective", lambda b: _direction(b).update(seed_effective=5), "forwarding_mismatch"),
    ("seed_salt_string", lambda b: _direction(b).update(seed_salt="0"), "receipt_malformed"),
    ("seed_salt_negative", lambda b: _direction(b).update(seed_salt=-1), "receipt_malformed"),
    ("top_level_instruction", lambda b: b.update(native_instruction="Speak differently."), "forwarding_mismatch"),
    ("fallback", lambda b: b["worker_receipt"].update(fallback={"reason": "cpu"}), "fallback_refused"),
    ("other_runtime", lambda b: b["worker_receipt"]["worker_runtime_identity"].update(sha256=sde.sha256_hex(b"x")),
     "receipt_mismatch"),
    ("other_profile", lambda b: b["worker_receipt"]["execution_profile"].update(runtime_profile="qwen-base"),
     "engine_mismatch"),
    ("other_voice", lambda b: b["worker_receipt"].update(identity_id="someone-else"), "identity_mismatch"),
    ("other_prompt", lambda b: b["worker_receipt"].update(prompt_sha256=sde.sha256_hex(b"p")), "identity_mismatch"),
    ("other_reference", lambda b: b.update(reference_audio_sha256=sde.sha256_hex(b"r")), "identity_mismatch"),
    ("other_audio", lambda b: b.update(audio_sha256=sde.sha256_hex(b"a")), "receipt_mismatch"),
    ("other_runtime_top", lambda b: b.update(runtime_sha256=sde.sha256_hex(b"a")), "receipt_mismatch"),
    ("other_utterance", lambda b: b.update(utterance_id="fx-09"), "receipt_mismatch"),
    ("bad_request_sha", lambda b: b.update(request_sha256="nothex"), "receipt_malformed"),
])
def test_malformed_or_contradicting_receipts_are_refused_even_if_metadata_claims_forwarded(
        fixture_manifest, name, change, code):
    assert sample(fixture_manifest, "directed", "fx-02")["instruction"]["forwarded"] is True  # metadata claims it
    edit_receipt(fixture_manifest, "directed", "fx-02", change)
    refused(fixture_manifest, code)


def test_receipt_proving_not_forwarded_contradicts_forwarded_metadata(fixture_manifest):
    text = sample(fixture_manifest, "directed", "fx-02")["text"]
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: (
        b["worker_receipt"]["generation"].update(direction=sde.expected_direction_receipt(
            None, identity_id="fixture-voice-ada", text=text)),
        b.update(native_instruction=None)))
    refused(fixture_manifest, "forwarding_mismatch")


def test_missing_non_json_and_non_object_receipts_are_refused(fixture_manifest):
    clone = copy.deepcopy(fixture_manifest)
    sample(clone, "baseline", "fx-01").pop("receipt")
    refused(clone, "receipt_missing")
    path = root_of(fixture_manifest) / sample(fixture_manifest, "baseline", "fx-01")["receipt"]["path"]
    path.write_text("not json")
    refused(fixture_manifest, "receipt_malformed")
    path.write_text("[1, 2]")
    refused(fixture_manifest, "receipt_malformed")
    path.write_bytes(b"\xff\xfe\x00")
    refused(fixture_manifest, "receipt_malformed")


def test_receipt_bytes_must_match_declared_hash(fixture_manifest):
    s = sample(fixture_manifest, "baseline", "fx-01")
    path = root_of(fixture_manifest) / s["receipt"]["path"]
    s["receipt"]["sha256"] = sde.sha256_hex(path.read_bytes())
    assert sde.evaluate_scene_pilot(fixture_manifest)["status"] == "complete"
    edit_receipt(fixture_manifest, "baseline", "fx-01", lambda b: b.update(latency_seconds=9.0))  # private file mutated
    refused(fixture_manifest, "receipt_mismatch")


def test_actual_parent_path_and_settings_are_checked(fixture_manifest):
    uid = "fx-01"
    def parent(body):
        body["worker_receipt"]["generation"].pop("direction")
        body["worker_receipt"]["performance"] = {}
        body["worker_receipt"]["generation_settings"] = {
            "seed": int(sde.sha256_hex((body["worker_receipt"]["identity_id"] + fixture_manifest["lines"][0]["text"]).encode())[:8], 16),
            "temperature": 0.9}
    for arm in sde.ARMS:
        edit_receipt(fixture_manifest, arm, uid, parent)
    result = sde.evaluate_scene_pilot(fixture_manifest)
    left = next(s for s in result["arms"]["baseline"]["samples"] if s["utterance_id"] == uid)
    assert left["receipt"]["direction_proof"] == "unchanged_parent_path"
    edit_receipt(fixture_manifest, "directed", uid,
                 lambda body: body["worker_receipt"]["generation_settings"].update(temperature=0.5))
    refused(fixture_manifest, "settings_mismatch")


def test_absent_neutral_receipt_does_not_allow_hidden_controls(fixture_manifest):
    def hidden(body):
        body["worker_receipt"]["generation"].pop("direction")
        body["worker_receipt"]["performance"] = {"native_instruction": "Speak loudly."}
    edit_receipt(fixture_manifest, "baseline", "fx-01", hidden)
    refused(fixture_manifest, "receipt_malformed")


def test_sample_claimed_latency_must_agree_with_the_receipt(fixture_manifest):
    sample(fixture_manifest, "baseline", "fx-01")["latency_seconds"] = 0.5
    assert sde.evaluate_scene_pilot(fixture_manifest)["status"] == "complete"
    sample(fixture_manifest, "baseline", "fx-01")["latency_seconds"] = 0.1
    refused(fixture_manifest, "receipt_mismatch")


def test_synthetic_receipts_cannot_sit_in_a_production_manifest(fixture_manifest):
    clone = copy.deepcopy(fixture_manifest)
    clone.pop("fixture")
    refused(clone, "receipt_mismatch")


def test_direction_hash_claim_in_metadata_must_match_the_receipt(fixture_manifest):
    sample(fixture_manifest, "directed", "fx-02")["instruction"]["direction_sha256"] = sde.sha256_hex(b"other")
    refused(fixture_manifest, "forwarding_mismatch")


# --------------------------------------------------------------------------- #
# identical identity / runtime / settings / boundary policy / engine / fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["voice_id", "reference_id", "prompt_id"])
def test_clone_identity_must_match_per_line_across_arms(fixture_manifest, field):
    sample(fixture_manifest, "directed", "fx-02")[field] = "someone-else"
    with pytest.raises(sde.EvaluationError) as raised:
        sde.evaluate_scene_pilot(fixture_manifest)
    assert raised.value.code == "identity_mismatch"


def test_a_role_cannot_change_clone_between_lines_of_one_arm(fixture_manifest):
    for field in ("voice_id", "reference_id", "prompt_id"):
        clone = copy.deepcopy(fixture_manifest)
        sample(clone, "baseline", "fx-04")[field] = "different"
        refused(clone, "identity_mismatch")


@pytest.mark.parametrize("field,code", [
    ("runtime_sha256", "engine_mismatch"), ("checkpoint_sha256", "engine_mismatch"),
    ("settings_sha256", "settings_mismatch"), ("boundary_policy_sha256", "boundary_policy_mismatch"),
    ("name", "engine_mismatch"),
])
def test_arm_runtime_must_equal_the_shared_pinned_runtime(fixture_manifest, field, code):
    fixture_manifest["arms"]["directed"]["runtime"][field] = sde.sha256_hex(b"other") if field != "name" else "other-engine"
    refused(fixture_manifest, code)


def test_other_runtime_profile_or_fallback_is_refused(fixture_manifest):
    clone = copy.deepcopy(fixture_manifest)
    clone["runtime"]["runtime_profile"] = "qwen-base"
    for arm in sde.ARMS:
        clone["arms"][arm]["runtime"]["runtime_profile"] = "qwen-base"
    refused(clone, "engine_mismatch")
    fallback = copy.deepcopy(fixture_manifest)
    fallback["arms"]["directed"]["runtime"]["fallback"] = "cpu-tts"
    refused(fallback, "fallback_refused")
    sample_fallback = copy.deepcopy(fixture_manifest)
    sample(sample_fallback, "directed", "fx-02")["fallback"] = {"engine": "other"}
    refused(sample_fallback, "fallback_refused")


def test_sample_level_runtime_settings_and_boundary_must_match(fixture_manifest):
    s = sample(fixture_manifest, "directed", "fx-02")
    s["runtime_sha256"] = sde.sha256_hex(b"other")
    refused(fixture_manifest, "engine_mismatch")
    s["runtime_sha256"] = fixture_manifest["runtime"]["runtime_sha256"]
    s["settings_sha256"] = sde.sha256_hex(b"other")
    refused(fixture_manifest, "settings_mismatch")
    del s["settings_sha256"]
    s["boundary_policy_sha256"] = sde.sha256_hex(b"other")
    refused(fixture_manifest, "boundary_policy_mismatch")


def test_seed_settings_must_match_across_arms(fixture_manifest):
    text = sample(fixture_manifest, "directed", "fx-02")["text"]
    instruction = sample(fixture_manifest, "directed", "fx-02")["instruction"]["native_instruction"]
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: b["worker_receipt"]["generation"].update(
        direction=sde.expected_direction_receipt(instruction, identity_id="fixture-voice-ada", text=text, seed_salt=9)))
    refused(fixture_manifest, "settings_mismatch")


def test_missing_shared_runtime_fields_are_invalid(fixture_manifest):
    for field in ("runtime_sha256", "checkpoint_sha256", "settings_sha256", "boundary_policy_sha256", "name"):
        clone = copy.deepcopy(fixture_manifest)
        del clone["runtime"][field]
        refused(clone, "manifest_invalid")
    clone = copy.deepcopy(fixture_manifest)
    del clone["arms"]["baseline"]["runtime"]
    refused(clone, "manifest_invalid")


# --------------------------------------------------------------------------- #
# coverage / arms
# --------------------------------------------------------------------------- #


def test_short_arm_coverage_is_refused(fixture_manifest):
    del fixture_manifest["arms"]["directed"]["samples"]["fx-12"]
    error = refused(fixture_manifest, "coverage_incomplete")
    assert error.details["missing_utterance_ids"] == ["fx-12"]


def test_sample_outside_the_line_set_and_key_mismatch_are_refused(fixture_manifest):
    extra = copy.deepcopy(fixture_manifest)
    extra["arms"]["baseline"]["samples"]["ghost"] = copy.deepcopy(sample(extra, "baseline", "fx-01"))
    refused(extra, "manifest_invalid")
    wrong_key = copy.deepcopy(fixture_manifest)
    sample(wrong_key, "baseline", "fx-01")["utterance_id"] = "fx-02"
    refused(wrong_key, "manifest_invalid")
    as_list = copy.deepcopy(fixture_manifest)
    as_list["arms"]["baseline"]["samples"] = list(as_list["arms"]["baseline"]["samples"].values())
    refused(as_list, "manifest_invalid")


def test_role_must_match_the_line(fixture_manifest):
    sample(fixture_manifest, "baseline", "fx-02")["role_id"] = "bo"
    refused(fixture_manifest, "manifest_invalid")


def test_missing_arm_stays_missing_holds_the_gate_and_borrows_nothing(tmp_path):
    manifest = sde.build_synthetic_fixture(tmp_path / "fx", omit_arms=["directed"])
    result = sde.evaluate_scene_pilot(production_like(manifest))
    assert result["status"] == "incomplete" and result["arm_status"]["directed"] == "missing"
    assert result["arms"]["directed"]["samples"] == [] and result["comparisons"] is None
    assert result["ballots"] is None and result["ballot_key"] is None and result["ballot_ref"] is None
    assert result["human_gate"]["status"] == "held" and result["human_gate"]["approval"] is False
    assert any("directed" in w for w in result["warnings"])


def test_missing_arm_cannot_satisfy_the_gate_even_with_a_review(tmp_path):
    full = sde.evaluate_scene_pilot(production_like(sde.build_synthetic_fixture(tmp_path / "full")))
    manifest = production_like(sde.build_synthetic_fixture(tmp_path / "part", omit_arms=["baseline"]))
    manifest["human_review"] = make_review(full, lambda arm, narrator: 5)
    result = sde.evaluate_scene_pilot(manifest)
    assert result["human_gate"]["status"] == "held" and result["human_gate"]["scores"] is None


def test_arm_marked_missing_with_samples_is_contradictory(fixture_manifest):
    fixture_manifest["arms"]["directed"]["status"] = "missing"
    refused(fixture_manifest, "manifest_invalid")


def test_unknown_arm_schema_and_ids_are_invalid(fixture_manifest):
    for mutate in (
        lambda m: m["arms"].update(casting={}),
        lambda m: m.update(schema_version="director-comparison-manifest-v1"),
        lambda m: m.update(comparison_id="bad id"),
        lambda m: m.update(fixture={"is_fixture": False}),
    ):
        clone = copy.deepcopy(fixture_manifest)
        mutate(clone)
        refused(clone, "manifest_invalid")
    with pytest.raises(sde.EvaluationError):
        sde.evaluate_scene_pilot([])


def test_directed_arm_that_is_a_byte_copy_of_baseline_is_refused(fixture_manifest):
    root = root_of(fixture_manifest)
    for u, s in fixture_manifest["arms"]["directed"]["samples"].items():
        data = (root / fixture_manifest["arms"]["baseline"]["samples"][u]["audio_file"]).read_bytes()
        (root / s["audio_file"]).write_bytes(data)
        s["audio_sha256"] = sde.sha256_hex(data)
        edit_receipt(fixture_manifest, "directed", u, lambda b, h=s["audio_sha256"]: b.update(audio_sha256=h))
    refused(fixture_manifest, "arm_copied")


def test_forwarded_lines_with_unchanged_audio_are_reported_not_hidden(fixture_manifest):
    root = root_of(fixture_manifest)
    s = sample(fixture_manifest, "directed", "fx-02")
    data = (root / sample(fixture_manifest, "baseline", "fx-02")["audio_file"]).read_bytes()
    (root / s["audio_file"]).write_bytes(data)
    s["audio_sha256"] = sde.sha256_hex(data)
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    result = sde.evaluate_scene_pilot(fixture_manifest)
    finding = next(f for f in result["findings"] if f["code"] == "directed_audio_unchanged")
    assert finding["utterance_ids"] == ["fx-02"]


def test_narrator_byte_differences_are_reported_as_nondeterminism_not_instruction(fixture_manifest):
    root = root_of(fixture_manifest)
    s = sample(fixture_manifest, "directed", "fx-01")
    data = de._tone_wav(333.0, 0.7, 0.4)
    (root / s["audio_file"]).write_bytes(data)
    s["audio_sha256"] = sde.sha256_hex(data)
    edit_receipt(fixture_manifest, "directed", "fx-01", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    result = sde.evaluate_scene_pilot(fixture_manifest)
    assert result["comparisons"]["directed_vs_baseline"]["changed_audio_narrator_lines"] == 1
    assert next(f for f in result["findings"] if f["code"] == "narrator_audio_differs")["utterance_ids"] == ["fx-01"]


def test_edge_silence_drift_between_arms_is_surfaced(fixture_manifest):
    root = root_of(fixture_manifest)
    s = sample(fixture_manifest, "directed", "fx-02")
    data = de._tone_wav(220.0, 0.7, 0.4, tail_silence=1.2)
    (root / s["audio_file"]).write_bytes(data)
    s["audio_sha256"] = sde.sha256_hex(data)
    edit_receipt(fixture_manifest, "directed", "fx-02", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    comparison = sde.evaluate_scene_pilot(fixture_manifest)["comparisons"]["directed_vs_baseline"]
    assert "fx-02" in comparison["edge_silence_drift_utterance_ids"]


# --------------------------------------------------------------------------- #
# audio / private source mutation and path safety
# --------------------------------------------------------------------------- #


def test_audio_bytes_must_match_declared_hash(fixture_manifest):
    s = sample(fixture_manifest, "directed", "fx-02")
    path = root_of(fixture_manifest) / s["audio_file"]
    path.write_bytes(path.read_bytes() + b"\x00\x00")  # private source file mutated after the manifest was written
    error = refused(fixture_manifest, "audio_hash_mismatch")
    assert error.details["declared"] == s["audio_sha256"]


def test_unreadable_audio_is_refused(fixture_manifest):
    s = sample(fixture_manifest, "baseline", "fx-01")
    path = root_of(fixture_manifest) / s["audio_file"]
    path.write_bytes(b"not a wave file")
    s["audio_sha256"] = sde.sha256_hex(b"not a wave file")
    edit_receipt(fixture_manifest, "baseline", "fx-01", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    refused(fixture_manifest, "audio_unreadable")


def test_same_file_for_two_samples_is_refused(fixture_manifest):
    fixture_manifest["arms"]["directed"]["samples"]["fx-01"]["audio_file"] = \
        fixture_manifest["arms"]["baseline"]["samples"]["fx-01"]["audio_file"]
    refused(fixture_manifest, "source_shared")


def test_missing_audio_or_receipt_file_is_refused(fixture_manifest):
    (root_of(fixture_manifest) / sample(fixture_manifest, "baseline", "fx-01")["audio_file"]).unlink()
    refused(fixture_manifest, "source_missing")


@pytest.mark.parametrize("bad", ["../escape.wav", "sub/../../escape.wav", "..", "C:\\Windows\\win.ini", "/etc/passwd",
                                 "baseline-fx-01.wav:stream", "\\\\server\\share\\a.wav", "\x00bad", "",
                                 "baseline-fx-01.wav\x00.txt"])
def test_unsafe_paths_are_refused(fixture_manifest, bad):
    sample(fixture_manifest, "baseline", "fx-01")["audio_file"] = bad
    with pytest.raises(sde.EvaluationError) as raised:
        sde.evaluate_scene_pilot(fixture_manifest)
    assert raised.value.code in {"path_refused", "manifest_invalid", "source_missing"}


def test_absolute_path_outside_base_dir_is_refused(fixture_manifest, tmp_path):
    outside = tmp_path / "outside.wav"
    outside.write_bytes((root_of(fixture_manifest) / sample(fixture_manifest, "baseline", "fx-01")["audio_file"]).read_bytes())
    sample(fixture_manifest, "baseline", "fx-01")["audio_file"] = str(outside)
    refused(fixture_manifest, "path_refused")


def test_absolute_path_inside_base_dir_is_accepted(fixture_manifest):
    s = sample(fixture_manifest, "baseline", "fx-01")
    s["audio_file"] = str(root_of(fixture_manifest) / s["audio_file"])
    assert sde.evaluate_scene_pilot(fixture_manifest)["status"] == "complete"


def test_links_are_refused(fixture_manifest, tmp_path):
    root = root_of(fixture_manifest)
    s = sample(fixture_manifest, "baseline", "fx-01")
    target = root / s["audio_file"]
    link = root / "linked.wav"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    s["audio_file"] = "linked.wav"
    refused(fixture_manifest, "path_refused")


def test_link_in_a_parent_directory_is_refused(fixture_manifest, tmp_path):
    root = root_of(fixture_manifest)
    real = tmp_path / "elsewhere"
    real.mkdir()
    s = sample(fixture_manifest, "baseline", "fx-01")
    (real / "a.wav").write_bytes((root / s["audio_file"]).read_bytes())
    try:
        os.symlink(real, root / "dir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    s["audio_file"] = "dir/a.wav"
    refused(fixture_manifest, "path_refused")


def test_base_dir_must_be_an_absolute_existing_directory(fixture_manifest, tmp_path):
    for value in (None, "relative/dir", str(tmp_path / "missing")):
        clone = copy.deepcopy(fixture_manifest)
        clone["base_dir"] = value
        with pytest.raises(sde.EvaluationError):
            sde.evaluate_scene_pilot(clone)


def test_inputs_are_left_untouched_and_reported_with_hashes(fixture_manifest, tmp_path):
    root = root_of(fixture_manifest)
    before = snapshot(root)
    result = sde.evaluate_scene_pilot(fixture_manifest, output_dir=tmp_path / "out")
    assert snapshot(root) == before
    integrity = result["source_integrity"]
    assert integrity["unchanged"] is True
    by_name = {Path(f["path"]).name: f["sha256"] for f in integrity["files"]}
    assert by_name == {Path(k).name: v for k, v in before.items()}  # every wav and receipt, with its hash


def test_input_changing_during_evaluation_is_detected(fixture_manifest, monkeypatch):
    victim = root_of(fixture_manifest) / sample(fixture_manifest, "baseline", "fx-01")["audio_file"]
    real = sde.measure_audio
    calls = {"n": 0}

    def tamper(data, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            victim.write_bytes(victim.read_bytes() + b"\x00\x00")
        return real(data, **kwargs)

    monkeypatch.setattr(sde, "measure_audio", tamper)
    refused(fixture_manifest, "source_changed")


def test_receipt_changing_during_evaluation_is_detected(fixture_manifest, monkeypatch):
    victim = root_of(fixture_manifest) / sample(fixture_manifest, "baseline", "fx-01")["receipt"]["path"]
    real = sde.measure_audio
    calls = {"n": 0}

    def tamper(data, **kwargs):
        calls["n"] += 1
        if calls["n"] == 20:
            victim.write_bytes(victim.read_bytes() + b" ")
        return real(data, **kwargs)

    monkeypatch.setattr(sde, "measure_audio", tamper)
    refused(fixture_manifest, "source_changed")


# --------------------------------------------------------------------------- #
# blind per-scene ballots
# --------------------------------------------------------------------------- #


def test_ballots_are_per_scene_deterministic_and_arm_blind(fixture_manifest):
    first = sde.evaluate_scene_pilot(fixture_manifest)
    again = sde.evaluate_scene_pilot(copy.deepcopy(fixture_manifest))
    assert first["ballots"] == again["ballots"] and first["ballot_key"] == again["ballot_key"]
    ballots = first["ballots"]
    assert [b["scene_id"] for b in ballots["scenes"]] == ["fx-scene-01", "fx-scene-02", "fx-scene-03"]
    assert [len(b["items"]) for b in ballots["scenes"]] == [4, 4, 4]
    text = json.dumps(ballots).lower()
    for forbidden in ("baseline", "directed", "speak quietly", "dry, restrained", ".wav", "receipt", "voice_id"):
        assert forbidden not in text, forbidden
    for ballot in ballots["scenes"]:
        for item in ballot["items"]:
            assert [o["label"] for o in item["options"]] == ["A", "B"]
            assert all(o["sample_ref"].startswith(item["item_id"]) for o in item["options"])
        assert ballot["rating_scale"]["dimensions"] == list(sde.RATING_DIMENSIONS)


def test_ballot_text_is_the_exact_source_text_and_narration_is_marked(fixture_manifest):
    ballots = sde.evaluate_scene_pilot(fixture_manifest)["ballots"]["scenes"]
    by_id = {i["utterance_id"]: i for b in ballots for i in b["items"]}
    for line in fixture_manifest["lines"]:
        assert by_id[line["utterance_id"]]["text"] == line["text"]
        assert by_id[line["utterance_id"]]["narrator"] is line["narrator"]


def test_label_assignment_is_balanced_and_the_key_is_complete(fixture_manifest):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    firsts = [item["options"]["A"]["arm"] for scene in result["ballot_key"]["scenes"].values()
              for item in scene["items"].values()]
    assert sorted(firsts).count("baseline") == 6 and sorted(firsts).count("directed") == 6
    for scene in result["ballot_key"]["scenes"].values():
        for item in scene["items"].values():
            assert {o["arm"] for o in item["options"].values()} == {"baseline", "directed"}
            for option in item["options"].values():
                assert sde.sha256_hex(Path(option["audio_path"]).read_bytes()) == option["audio_sha256"]
    assert result["ballot_key"]["ballot_set_id"] == result["ballots"]["ballot_set_id"]


def test_ballot_identity_changes_with_the_comparison_lines_or_audio(fixture_manifest, tmp_path):
    base = sde.evaluate_scene_pilot(fixture_manifest)
    renamed = copy.deepcopy(fixture_manifest)
    renamed["comparison_id"] = "another-comparison"
    assert sde.evaluate_scene_pilot(renamed)["ballots"]["ballot_set_id"] != base["ballots"]["ballot_set_id"]
    other = sde.build_synthetic_fixture(tmp_path / "other")
    s = sample(other, "directed", "fx-02")
    data = de._tone_wav(500.0, 0.9, 0.4)
    (Path(other["base_dir"]) / s["audio_file"]).write_bytes(data)
    s["audio_sha256"] = sde.sha256_hex(data)
    edit_receipt(other, "directed", "fx-02", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    changed = sde.evaluate_scene_pilot(other)
    assert changed["ballots"]["audio_set_sha256"] != base["ballots"]["audio_set_sha256"]
    assert changed["ballots"]["ballot_set_id"] != base["ballots"]["ballot_set_id"]
    assert {b["scene_id"]: b["ballot_id"] for b in changed["ballots"]["scenes"]}["fx-scene-01"] != \
        {b["scene_id"]: b["ballot_id"] for b in base["ballots"]["scenes"]}["fx-scene-01"]


def test_stage_blind_audio_uses_blind_names_and_verified_bytes(fixture_manifest, tmp_path):
    result = sde.evaluate_scene_pilot(fixture_manifest)
    staged = sde.stage_blind_audio(result, tmp_path / "blind")
    assert len(staged) == 24
    names = sorted(p.name for p in (tmp_path / "blind").iterdir())
    assert names[0] == "fx-scene-01-item-001-A.wav" and not any("baseline" in n or "directed" in n for n in names)
    assert {s["status"] for s in sde.stage_blind_audio(result, tmp_path / "blind")} == {"reused"}
    victim = Path(next(iter(next(iter(result["ballot_key"]["scenes"].values()))["items"].values()))["options"]["A"]["audio_path"])
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(sde.EvaluationError) as raised:
        sde.stage_blind_audio(result, tmp_path / "blind2")
    assert raised.value.code == "source_changed"


def test_staging_without_ballots_is_refused(tmp_path):
    result = sde.evaluate_scene_pilot(sde.build_synthetic_fixture(tmp_path / "fx", omit_arms=["baseline"]))
    with pytest.raises(sde.EvaluationError) as raised:
        sde.stage_blind_audio(result, tmp_path / "blind")
    assert raised.value.code == "ballots_unavailable"


# --------------------------------------------------------------------------- #
# human review gate
# --------------------------------------------------------------------------- #


def prod(manifest_tmp):
    return production_like(sde.build_synthetic_fixture(manifest_tmp))


def test_gate_is_pending_without_a_review_and_never_approves(tmp_path):
    result = sde.evaluate_scene_pilot(prod(tmp_path / "fx"))
    gate = result["human_gate"]
    assert gate["status"] == "pending" and gate["approval"] is False and gate["default_changed"] is False
    assert result["production_eligible"] is False and gate["scores"] is None and gate["verdicts"] is None


def test_gate_supports_experimental_review_when_directed_dialogue_improves(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    manifest["human_review"] = make_review(first, lambda arm, narrator: 5 if arm == "directed" and not narrator else 3)
    gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
    assert gate["status"] == "supports_experimental_review"
    assert gate["verdicts"]["dialogue"] == {"overall_delta": 2.0, "verdict": "improved"}
    assert gate["verdicts"]["narrator"]["verdict"] == "no_clear_change"
    assert gate["approval"] is False and gate["default_changed"] is False and gate["promotion"] == "none"
    assert gate["coverage"] == {"scenes_total": 3, "scenes_reviewed": 3, "complete": True}
    assert gate["human_review_sha256"] == sde.content_sha256(manifest["human_review"])


def test_gate_does_not_support_when_directed_dialogue_regresses(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    manifest["human_review"] = make_review(first, lambda arm, narrator: 2 if arm == "directed" and not narrator else 4)
    gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
    assert gate["status"] == "does_not_support" and gate["verdicts"]["dialogue"]["verdict"] == "regressed"


def test_narrator_regression_blocks_support_even_when_dialogue_improves(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    manifest["human_review"] = make_review(
        first, lambda arm, narrator: (5 if not narrator else 2) if arm == "directed" else 3)
    gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
    assert gate["status"] == "does_not_support" and "narration" in gate["reasons"][0]


def test_gate_no_clear_change(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    manifest["human_review"] = make_review(first, lambda arm, narrator: 3)
    assert sde.evaluate_scene_pilot(manifest)["human_gate"]["status"] == "no_clear_change"


def test_model_or_unattested_reviews_cannot_satisfy_the_gate(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    for kind in ("model", "automated", None, "Human"):
        review = make_review(first, lambda arm, narrator: 5 if arm == "directed" else 1, kind=kind)
        manifest["human_review"] = review
        gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
        assert gate["status"] == "held" and gate["scores"] is None and gate["verdicts"] is None
        assert any("reviewer_kind" in r for r in gate["reasons"])


def test_partial_review_is_pending_not_scored(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    review = make_review(first, lambda arm, narrator: 5 if arm == "directed" else 1)
    review["scenes"] = review["scenes"][:2]
    manifest["human_review"] = review
    gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
    assert gate["status"] == "pending" and gate["scores"] is None
    assert gate["coverage"]["scenes_reviewed"] == 2 and "2 of 3" in gate["reasons"][0]


def test_invalid_reviews_hold_the_gate_with_reasons(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)

    def fresh():
        return make_review(first, lambda arm, narrator: 4)

    cases = []
    review = fresh(); review["ballot_set_id"] = "other"; cases.append((review, "ballot_set_id"))
    review = fresh(); review["reviewer_id"] = " "; cases.append((review, "reviewer_id"))
    review = fresh(); review["scenes"] = "no"; cases.append((review, "scenes"))
    review = fresh(); review["scenes"][0]["ballot_id"] = "ghost"; cases.append((review, "unknown ballot"))
    review = fresh(); review["scenes"].append(copy.deepcopy(review["scenes"][0])); cases.append((review, "duplicate review"))
    review = fresh(); review["scenes"][0]["responses"].pop(); cases.append((review, "missing response"))
    review = fresh(); review["scenes"][0]["responses"][0]["item_id"] = "ghost"; cases.append((review, "unknown item"))
    review = fresh(); review["scenes"][0]["responses"].append(copy.deepcopy(review["scenes"][0]["responses"][0]))
    cases.append((review, "duplicate response"))
    review = fresh(); review["scenes"][0]["responses"][0]["ratings"].pop("A"); cases.append((review, "ratings must cover"))
    review = fresh(); review["scenes"][0]["responses"][0]["ratings"]["A"]["naturalness"] = 6; cases.append((review, "integer"))
    review = fresh(); review["scenes"][0]["responses"][0]["ratings"]["A"]["scene_fit"] = True; cases.append((review, "integer"))
    review = fresh(); review["scenes"][0]["responses"][0]["ratings"]["B"].pop("clarity"); cases.append((review, "clarity"))
    review = fresh(); review["scenes"][0]["responses"][0]["preferred_label"] = "C"; cases.append((review, "preferred_label"))
    review = fresh(); review["scenes"][0]["responses"][0]["defects"] = {"Z": ["x"]}; cases.append((review, "defects"))
    review = fresh(); review["scenes"][0]["responses"][0]["defects"] = {"A": "clipped"}; cases.append((review, "defects"))
    for review, fragment in cases:
        manifest["human_review"] = review
        gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
        assert gate["status"] == "held", fragment
        assert any(fragment in reason for reason in gate["reasons"]), (fragment, gate["reasons"])
        assert gate["scores"] is None and gate["approval"] is False
    manifest["human_review"] = "approved!"
    assert sde.evaluate_scene_pilot(manifest)["human_gate"]["status"] == "held"


def test_preferences_and_defects_are_decoded_through_the_key(tmp_path):
    manifest = prod(tmp_path / "fx")
    first = sde.evaluate_scene_pilot(manifest)
    review = make_review(first, lambda arm, narrator: 3)
    for scene in review["scenes"]:
        for response in scene["responses"]:
            options = first["ballot_key"]["scenes"][
                next(b["scene_id"] for b in first["ballots"]["scenes"] if b["ballot_id"] == scene["ballot_id"])
            ]["items"][response["item_id"]]["options"]
            directed_label = next(l for l, o in options.items() if o["arm"] == "directed")
            response["preferred_label"] = directed_label
            response["defects"] = {directed_label: ["breath"]}
    manifest["human_review"] = review
    scores = sde.evaluate_scene_pilot(manifest)["human_gate"]["scores"]["arms"]
    assert scores["directed"]["dialogue"]["preferred_count"] == 9 and scores["directed"]["narrator"]["preferred_count"] == 3
    assert scores["baseline"]["dialogue"]["preferred_count"] == 0
    assert scores["directed"]["defects_listed"] == 12 and scores["baseline"]["defects_listed"] == 0


def test_fixture_gate_ignores_reviews_and_technical_blockers_hold_it(tmp_path, fixture_manifest):
    first = sde.evaluate_scene_pilot(fixture_manifest)
    fixture_manifest["human_review"] = make_review(first, lambda arm, narrator: 5)
    assert sde.evaluate_scene_pilot(fixture_manifest)["human_gate"]["status"] == "fixture_excluded"
    manifest = prod(tmp_path / "fx")
    root = root_of(manifest)
    s = sample(manifest, "directed", "fx-02")
    silent = de._tone_wav(220.0, 0.5, 0.0)
    (root / s["audio_file"]).write_bytes(silent)
    s["audio_sha256"] = sde.sha256_hex(silent)
    edit_receipt(manifest, "directed", "fx-02", lambda b: b.update(audio_sha256=s["audio_sha256"]))
    base = sde.evaluate_scene_pilot(manifest)
    manifest["human_review"] = make_review(base, lambda arm, narrator: 5 if arm == "directed" else 1)
    gate = sde.evaluate_scene_pilot(manifest)["human_gate"]
    assert gate["status"] == "held" and "technical blockers in directed" in gate["reasons"][0]


# --------------------------------------------------------------------------- #
# immutable outputs and retry
# --------------------------------------------------------------------------- #


def test_outputs_are_exclusive_and_retry_is_safe(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    first = sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)
    assert {f["name"]: f["status"] for f in first["outputs"]["files"]} == {
        "evaluation.json": "created", "ballot_set.json": "created", "ballot_key.json": "created",
        "ballots/fx-scene-01.json": "created", "ballots/fx-scene-02.json": "created",
        "ballots/fx-scene-03.json": "created"}
    written = snapshot(out)
    again = sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)
    assert {f["status"] for f in again["outputs"]["files"]} == {"reused"}
    assert snapshot(out) == written
    assert not [p for p in out.rglob("*") if p.name.endswith(".stage")]
    evaluation = json.loads((out / "evaluation.json").read_text("utf-8"))
    assert "ballot_key" not in evaluation and "ballots" not in evaluation and evaluation["evaluation_id"] == first["evaluation_id"]
    for ballot in out.glob("ballots/*.json"):
        text = ballot.read_text("utf-8").lower()
        assert "baseline" not in text and "directed" not in text
    set_file = json.loads((out / "ballot_set.json").read_text("utf-8"))
    assert "scenes" not in set_file and [b["scene_id"] for b in set_file["ballots"]][0] == "fx-scene-01"


def test_interrupted_run_is_completed_on_retry_without_touching_existing_files(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)
    full = snapshot(out)
    (out / "ballot_key.json").unlink()
    (out / "ballots" / "fx-scene-03.json").unlink()
    retry = sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)
    statuses = {f["name"]: f["status"] for f in retry["outputs"]["files"]}
    assert statuses["ballot_key.json"] == "created" and statuses["ballots/fx-scene-03.json"] == "created"
    assert statuses["evaluation.json"] == "reused"
    assert snapshot(out) == full


def test_different_result_in_an_existing_output_dir_is_refused(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)
    other = sde.build_synthetic_fixture(tmp_path / "other", line_count=16)
    before = snapshot(out)
    refused(other, "output_conflict", output_dir=out)
    assert snapshot(out) == before


def test_conflict_is_checked_before_any_new_file_is_written(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "ballot_key.json").write_text("{}")  # foreign file in the way
    refused(fixture_manifest, "output_conflict", output_dir=out)
    assert sorted(p.name for p in out.iterdir()) == ["ballot_key.json"]
    assert (out / "ballot_key.json").read_text() == "{}"


def test_human_gate_file_is_added_only_with_a_review_and_is_immutable(tmp_path):
    manifest = prod(tmp_path / "fx")
    out = tmp_path / "out"
    first = sde.evaluate_scene_pilot(manifest, output_dir=out)
    assert not (out / "human_gate.json").exists()
    manifest["human_review"] = make_review(first, lambda arm, narrator: 5 if arm == "directed" else 3)
    second = sde.evaluate_scene_pilot(manifest, output_dir=out)
    statuses = {f["name"]: f["status"] for f in second["outputs"]["files"]}
    assert statuses.pop("human_gate.json") == "created" and set(statuses.values()) == {"reused"}
    other = copy.deepcopy(manifest)
    other["human_review"] = make_review(first, lambda arm, narrator: 2)
    refused(other, "output_conflict", output_dir=out)


def test_output_directory_may_not_contain_inputs(fixture_manifest):
    root = root_of(fixture_manifest)
    refused(fixture_manifest, "output_conflict", output_dir=root)
    assert not (root / "evaluation.json").exists()


def test_output_directory_inside_inputs_directory_is_allowed_when_it_holds_no_inputs(fixture_manifest):
    out = root_of(fixture_manifest) / "evaluation"
    assert sde.evaluate_scene_pilot(fixture_manifest, output_dir=out)["outputs"]["directory"] == str(out)
    assert (out / "evaluation.json").is_file()


def test_no_files_are_written_when_validation_fails(fixture_manifest, tmp_path):
    fixture_manifest["arms"]["directed"]["runtime"]["name"] = "different"
    out = tmp_path / "out"
    refused(fixture_manifest, "engine_mismatch", output_dir=out)
    assert not out.exists()


def test_incomplete_run_publishes_report_only(tmp_path):
    manifest = production_like(sde.build_synthetic_fixture(tmp_path / "fx", omit_arms=["directed"]))
    out = tmp_path / "out"
    result = sde.evaluate_scene_pilot(manifest, output_dir=out)
    assert [f["name"] for f in result["outputs"]["files"]] == ["evaluation.json"]
    assert not (out / "ballot_key.json").exists()


def test_symlinked_output_directory_is_refused(fixture_manifest, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-out"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    refused(fixture_manifest, "output_conflict", output_dir=link)
    assert not list(real.iterdir())


def test_summary_never_leaks_the_sealed_key_or_ballot_content(fixture_manifest, tmp_path):
    summary = sde.summarize(sde.evaluate_scene_pilot(fixture_manifest, output_dir=tmp_path / "out"))
    text = json.dumps(summary)
    assert "ballot_key" not in summary and "options" not in text and "audio_path" not in text
    assert summary["human_gate"]["approval"] is False and summary["ballot"]["items"] == 12


def test_report_is_stable_across_runs(fixture_manifest):
    a = sde.evaluate_scene_pilot(fixture_manifest)
    b = sde.evaluate_scene_pilot(copy.deepcopy(fixture_manifest))
    assert a["evaluation_id"] == b["evaluation_id"] and a["manifest_sha256"] == b["manifest_sha256"]
    assert math.isfinite(a["arms"]["baseline"]["summary"]["total_duration_seconds"])


def test_evaluation_changes_when_any_input_changes(fixture_manifest):
    a = sde.evaluate_scene_pilot(fixture_manifest)
    clone = copy.deepcopy(fixture_manifest)
    clone["comparison_id"] = "renamed"
    assert sde.evaluate_scene_pilot(clone)["evaluation_id"] != a["evaluation_id"]
