"""Blind X/Y pairs and the exact-artifact review gate (M3, section P11).

The pair key is private: the public index exposes two opaque audio hashes and a
label, never an arm name.  Only an independently authenticated review of the
exact artifact can support improvement, and this milestone has no live
verification bridge, so the default path is fail-closed.  A documented test
double exercises the seam; its fidelity is stated where it is used.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.storage.projects import load_project

from tests.test_performance_pilot_assembly import (  # noqa: F401
    build_inputs,
    make_source_project,
    scope_manifest,
    studio,
)

PAIR_A = "pair-a"
PAIR_B = "pair-b"


def _ensure_editions(studio) -> dict:
    """Publish real review editions for both arms and return their artifact index.

    Blind pairs bind exact published artifacts, so the fixture must actually
    assemble them rather than inventing digests.
    """

    editions = pilot.pilot_editions(studio["pilot_id"])
    if editions:
        return editions
    inputs = build_inputs(studio, klass="D_D_AFTER_TRAILOFF")
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=2)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"], allow_lead_offset=False)
    for arm in ("baseline", "variant"):
        pilot.assemble_review(studio["pilot_id"], timeline, arm=arm)
    return pilot.pilot_editions(studio["pilot_id"])


def _arm_digest(editions: dict, arm: str) -> str:
    rows = [(info["name"], digest) for digest, info in editions.items() if info["arm"] == arm]
    if not rows:
        raise AssertionError(f"no published edition for arm {arm}")
    return next((digest for name, digest in sorted(rows) if name == "review-master.wav"),
                sorted(rows)[0][1])


def pair_payload(pair_id: str = PAIR_A, *, baseline: str | None = None,
                 variant: str | None = None) -> list[dict]:
    return [{"pair_id": pair_id, "baseline_audio_sha256": baseline or "b" * 64,
             "variant_audio_sha256": variant or "c" * 64}]


def publish_pairs(studio, *, seed: int = 7, pairs=None, use_real_editions: bool = True) -> dict:
    """Publish pairs with the pilot's real edition digests unless told otherwise."""

    supplied = pairs or pair_payload()
    if not use_real_editions:
        return pilot.blind_pairs(studio["pilot_id"], supplied, seed=seed)
    editions = _ensure_editions(studio)
    baseline = _arm_digest(editions, "baseline")
    variant = _arm_digest(editions, "variant")
    resolved = [{"pair_id": item["pair_id"], "baseline_audio_sha256": baseline,
                 "variant_audio_sha256": variant} for item in supplied]
    return pilot.blind_pairs(studio["pilot_id"], resolved, seed=seed)


def review_for(studio, public: dict, **overrides) -> dict:
    row = public["pairs"][0]
    review = {
        "schema_version": pilot.REVIEW_SCHEMA,
        "pilot_id": studio["pilot_id"],
        "pair_id": row["pair_id"],
        "edition_sha256": public["editions"][row["pair_id"]],
        "heard": {"x_audio_sha256": row["x_audio_sha256"],
                  "y_audio_sha256": row["y_audio_sha256"], "coverage": "full"},
        "rubric": {"speaker_recognizable": "yes", "continuity": "yes"},
        "preference": "X",
        "confidence": "high",
        "stated_benefit": "the reply lands more naturally",
        "reviewer": "synthetic",
        "at": "2026-10-07T00:00:00+00:00",
    }
    review.update(overrides)
    for key, value in list(review.items()):
        if value is None:
            del review[key]
    return review


@pytest.fixture()
def verified_owner(monkeypatch):
    """Replace the PRIVATE owner boundary inside this disposable test only.

    This is the seam a real integration would fill with the authenticated owner.
    Monkeypatching the private function proves the gate's contract — a verified
    verdict opens it, an unverified one does not — and nothing else: it is not a
    public parameter, no caller can supply it, and in production the boundary
    ships unconfigured so every owner-authenticated review stays pending.
    """

    def trusted(evidence):
        if isinstance(evidence, dict) and evidence.get("owner_token") == "test-double":
            return {"status": "verified", "reason": "disposable test double",
                    "fingerprint": "f" * 64}
        return {"status": "unverified", "reason": "test double did not recognise the evidence",
                "fingerprint": "e" * 64}

    monkeypatch.setattr(pilot, "_owner_verification", trusted)
    return trusted


def authenticated(evidence: dict | None = None) -> dict:
    return {"kind": "live_owner", "evidence": evidence or {"owner_token": "test-double"}}


def variant_preference(key: dict) -> str:
    """The label the variant is hidden behind for this seed."""

    row = key["key"][0]
    return "X" if row["x_arm"] == "variant" else "Y"


def verified_review_for(studio, public: dict, key: dict, **overrides) -> dict:
    """A verified review whose preference resolves to the variant for this seed."""

    row = key["key"][0]
    preference = "X" if row["x_arm"] == "variant" else "Y"
    return review_for(studio, public, preference=preference, authentication=authenticated(),
                      **overrides)


# ------------------------------------------------------------------ public/private split


def test_the_public_index_never_reveals_an_arm_name(studio) -> None:
    public, private = publish_pairs(studio)
    blob = json.dumps(public)
    for arm in ("baseline", "variant"):
        assert arm not in blob
    for row in public["pairs"]:
        assert set(row) == {"label", "pair_id", "x_audio_sha256", "y_audio_sha256"}
    assert len(private["key"]) == len(public["pairs"])
    assert {row["x_arm"] for row in private["key"]} <= {"baseline", "variant"}
    key_blob = json.dumps(private)
    assert "baseline" in key_blob or "variant" in key_blob


def test_labels_and_order_are_stable_for_the_same_seed_and_inputs(studio) -> None:
    first, first_key = publish_pairs(studio, seed=11)
    second, second_key = publish_pairs(studio, seed=11)
    assert first == second
    assert first_key == second_key


def test_a_different_seed_changes_the_stored_key_not_the_contract(studio) -> None:
    one, key_one = publish_pairs(studio, seed=1)
    two, key_two = publish_pairs(studio, seed=2)
    assert len(one["pairs"]) == len(two["pairs"]) == 1
    # Either the assignment or the label order may differ; what must hold is that
    # each seed is internally consistent and reproducible.
    again, key_again = publish_pairs(studio, seed=2)
    assert two == again and key_two == key_again
    assert one["label_seed"] == 1 and two["label_seed"] == 2
    assert key_one["key"][0]["x_arm"] in ("baseline", "variant")


def test_the_private_key_is_persisted_only_on_the_pilot(studio) -> None:
    public, _private = publish_pairs(studio)
    project = load_project(studio["pilot_id"])
    stored = project.metadata[pilot.PILOT_KEYS_KEY]
    assert PAIR_A in stored
    assert stored[PAIR_A]["x_arm"] in ("baseline", "variant")
    assert "x_arm" not in json.dumps(public)


def test_identical_arms_and_malformed_pairs_are_refused(studio) -> None:
    same = "b" * 64
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.blind_pairs(studio["pilot_id"],
                          [{"pair_id": PAIR_A, "baseline_audio_sha256": same,
                            "variant_audio_sha256": same}], seed=1)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.blind_pairs(studio["pilot_id"], [{"pair_id": PAIR_A,
                                                "baseline_audio_sha256": "short",
                                                "variant_audio_sha256": "c" * 64}], seed=1)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.blind_pairs(studio["pilot_id"], [], seed=1)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.blind_pairs(studio["pilot_id"], pair_payload(), seed=-1)


def test_a_pilot_without_both_arms_cannot_build_pairs(studio) -> None:
    other = dict(studio["manifest"])
    other["operation_id"] = "op-single-arm"
    other["scope"] = {**studio["manifest"]["scope"], "arms": ["baseline"]}
    created = pilot.create_pilot("single", other)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.blind_pairs(created["pilot_id"], pair_payload(), seed=1)
    assert refused.value.code == "input_ineligible"


# ------------------------------------------------------------------ the review gate


def test_a_draft_review_is_history_and_the_gate_stays_closed(studio) -> None:
    public, _key = publish_pairs(studio)
    record = pilot.record_review(studio["pilot_id"], review_for(studio, public))
    assert record["status"] == "draft"
    assert record["production_enablement"] is False
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["supports_improvement"] is False
    assert gate["status"] == "review_pending"


def test_there_is_no_public_way_to_supply_a_verdict(studio) -> None:
    """A caller cannot inject the trusted verdict through any public mutation."""

    public, _key = publish_pairs(studio)
    review = review_for(studio, public, authentication=authenticated())
    # The removed parameter is gone, so an injected bridge is a TypeError.
    with pytest.raises(TypeError):
        pilot.record_review(studio["pilot_id"], review,
                            verification_bridge=lambda _: {"status": "verified"})
    # And the unconfigured private boundary refuses the claim outright.
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.record_review(studio["pilot_id"], review)
    assert refused.value.code == "review_pending"
    assert pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)["supports_improvement"] is False


def test_a_caller_naming_a_person_is_not_authentication(studio) -> None:
    public, _key = publish_pairs(studio)
    review = review_for(studio, public, authentication=authenticated({"by": "Brian"}))
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.record_review(studio["pilot_id"], review)
    assert refused.value.code == "review_pending"


def test_the_default_seam_reports_the_integration_limit(studio) -> None:
    verdict = pilot.verify_reviewer({"by": "Brian"})
    assert verdict["status"] == "unverified"
    assert "no in-process reviewer-verification bridge" in verdict["reason"]
    assert len(verdict["fingerprint"]) == 64


def test_the_public_index_binds_published_editions(studio) -> None:
    _public, private = publish_pairs(studio)
    row = private["key"][0]
    assert row["resolved"] is True
    editions = pilot.pilot_editions(studio["pilot_id"])
    for digest in (row["x_audio_sha256"], row["y_audio_sha256"]):
        assert digest in editions
    assert {row["x_edition"]["arm"], row["y_edition"]["arm"]} == {"baseline", "variant"}


def test_an_unresolved_digest_leaves_the_gate_closed(studio) -> None:
    public, private = publish_pairs(studio, use_real_editions=False)
    assert private["key"][0]["resolved"] is False
    record = pilot.record_review(studio["pilot_id"], review_for(studio, public))
    assert record["status"] == "stale"
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["supports_improvement"] is False
    assert gate["status"] == "review_pending"
    assert "resolved to published editions" in gate["reason"]


def test_a_verified_review_of_the_exact_artifact_can_open_the_gate(studio,
                                                                  verified_owner) -> None:
    public, key = publish_pairs(studio)
    review = review_for(studio, public, preference=variant_preference(key),
                        authentication=authenticated())
    record = pilot.record_review(studio["pilot_id"], review)
    assert record["status"] == "verified"
    assert record["verification"]["status"] == "verified"
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["supports_improvement"] is True
    assert gate["resolved_arm"] == "variant"


def test_a_verified_preference_for_the_baseline_is_not_an_improvement(studio,
                                                                    verified_owner) -> None:
    public, key = publish_pairs(studio)
    baseline_choice = "X" if key["key"][0]["x_arm"] == "baseline" else "Y"
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, public, preference=baseline_choice,
                                   authentication=authenticated()))
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["supports_improvement"] is False
    assert gate["status"] == "not_improved"


def test_a_tie_or_an_inconclusive_review_never_improves(studio, verified_owner) -> None:
    public, _key = publish_pairs(studio)
    for preference in ("tie", "inconclusive"):
        pilot.record_review(studio["pilot_id"],
                            review_for(studio, public, preference=preference,
                                       authentication=authenticated()))
        gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
        assert gate["status"] == "inconclusive"
        assert gate["supports_improvement"] is False


def test_a_defect_blocks_an_otherwise_verified_improvement(studio, verified_owner) -> None:
    public, key = publish_pairs(studio)
    row = key["key"][0]
    review = review_for(studio, public, preference=variant_preference(key),
                        defects=[{"audio_sha256": row["x_audio_sha256"], "seconds": 1.5,
                                  "note": "a clipped final consonant"}],
                        authentication=authenticated())
    pilot.record_review(studio["pilot_id"], review)
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["status"] == "blocked"
    assert gate["supports_improvement"] is False


def test_a_verified_improvement_without_a_stated_benefit_is_pending(studio,
                                                                    verified_owner) -> None:
    public, key = publish_pairs(studio)
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, public, preference=variant_preference(key),
                                   stated_benefit="", authentication=authenticated()))
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["status"] == "review_pending"
    assert "no stated benefit" in gate["reason"]


def test_other_evidence_cannot_support_an_improvement_gate(studio) -> None:
    public, _key = publish_pairs(studio)
    for key in ("notification", "qa_passed", "favorite", "rating", "baseline_acceptable",
                "human_verified"):
        review = review_for(studio, public, **{key: True})
        with pytest.raises(pilot.PerformanceDirectorError) as refused:
            pilot.record_review(studio["pilot_id"], review)
        assert refused.value.code == "direction_conflict"
        assert key in str(refused.value)


def test_an_unrecognised_owner_verdict_is_refused(studio, verified_owner) -> None:
    public, _key = publish_pairs(studio)
    review = review_for(studio, public, authentication=authenticated({"owner_token": "wrong"}))
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.record_review(studio["pilot_id"], review)
    assert refused.value.code == "review_pending"


# ------------------------------------------------------------------ staleness and validity


def republish_with_changed_labels(studio, old_public):
    """Change the X/Y edition without misrepresenting either artifact's arm."""
    for seed in range(8, 108):
        public, key = publish_pairs(studio, seed=seed)
        if public["pairs"][0]["x_audio_sha256"] != old_public["pairs"][0]["x_audio_sha256"]:
            return public, key
    raise AssertionError("No changed blind ordering in 100 distinct seeds")


def test_a_stale_edition_review_is_retained_and_excluded_from_the_gate(studio,
                                                                       verified_owner) -> None:
    public, key = publish_pairs(studio)
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, public, preference=variant_preference(key),
                                   authentication=authenticated(),
                                   at="2026-10-07T00:00:00+00:00"))
    assert pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)["supports_improvement"] is True

    # A new seed changes opaque labels while preserving each artifact's actual arm.
    new_public, new_key = republish_with_changed_labels(studio, public)
    assert new_public["pairs"][0]["x_audio_sha256"] != public["pairs"][0]["x_audio_sha256"]

    # The old review remains valid immutable history about the old edition...
    history = pilot.read_reviews(studio["pilot_id"], pair_id=PAIR_A)
    assert any(row["heard"] == review_for(studio, public)["heard"] for row in history)
    # ...and cannot accept the newer edition.
    gate = pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    assert gate["supports_improvement"] is False
    assert gate["status"] == "review_pending"
    assert "older edition" in gate["reason"]

    # A review of the new edition is what the gate now reads.
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, new_public, preference=variant_preference(new_key),
                                   authentication=authenticated(),
                                   at="2026-10-07T02:00:00+00:00"))
    assert pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)["supports_improvement"] is True
    assert len(pilot.read_reviews(studio["pilot_id"], pair_id=PAIR_A)) == 2


def test_the_prior_key_is_kept_so_history_stays_interpretable(studio, verified_owner) -> None:
    public, _ = publish_pairs(studio)
    republish_with_changed_labels(studio, public)
    stored = load_project(studio["pilot_id"]).metadata[pilot.PILOT_KEYS_KEY][PAIR_A]
    assert stored["history"], "the superseded edition must stay recorded"
    assert stored["history"][0]["x_audio_sha256"] != stored["x_audio_sha256"]


def test_artifact_arms_cannot_be_falsely_relabeled(studio) -> None:
    publish_pairs(studio)
    editions = pilot.pilot_editions(studio["pilot_id"])
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.blind_pairs(studio["pilot_id"], [{"pair_id": PAIR_A,
                         "baseline_audio_sha256": _arm_digest(editions, "variant"),
                         "variant_audio_sha256": _arm_digest(editions, "baseline")}], seed=7)


def test_reviews_are_validated_before_they_are_stored(studio) -> None:
    public, _key = publish_pairs(studio)
    for broken in (
        review_for(studio, public, schema_version="performance-review-v2"),
        review_for(studio, public, preference="maybe"),
        review_for(studio, public, confidence="certain"),
        review_for(studio, public, rubric={}),
        review_for(studio, public, defects=[{"audio_sha256": "d" * 64, "seconds": 1.0,
                                             "note": "not heard"}]),
    ):
        with pytest.raises(pilot.PerformanceDirectorError):
            pilot.record_review(studio["pilot_id"], broken)
    assert pilot.read_reviews(studio["pilot_id"], pair_id=PAIR_A) == []


def test_a_review_for_another_pilot_is_refused(studio) -> None:
    public, _key = publish_pairs(studio)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.record_review(studio["pilot_id"],
                            review_for(studio, public, pilot_id="somewhere-else"))
    assert refused.value.code == "direction_conflict"


def test_a_review_of_an_unknown_pair_is_refused(studio) -> None:
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.record_review(studio["pilot_id"], {
            "schema_version": pilot.REVIEW_SCHEMA, "pilot_id": studio["pilot_id"],
            "pair_id": "pair-unknown", "edition_sha256": hashlib.sha256(b"e").hexdigest(),
            "heard": {"x_audio_sha256": "b" * 64, "y_audio_sha256": "c" * 64,
                      "coverage": "full"},
            "rubric": {"continuity": "yes"}, "preference": "X", "confidence": "high",
            "stated_benefit": "b", "reviewer": "synthetic",
            "at": "2026-10-07T00:00:00+00:00"})
    assert refused.value.code == "input_ineligible"


def test_stored_reviews_are_immutable_and_owner_side(studio) -> None:
    public, _key = publish_pairs(studio)
    first = pilot.record_review(studio["pilot_id"], review_for(studio, public))
    again = pilot.record_review(studio["pilot_id"], review_for(studio, public))
    assert first["review_sha256"] == again["review_sha256"]
    assert first["production_enablement"] is False
    rows = pilot.read_reviews(studio["pilot_id"], pair_id=PAIR_A)
    assert len(rows) == 1
    assert rows[0]["review_sha256"] == first["review_sha256"]


# ------------------------------------------------------------------ false A/B/C gates


def test_a_combined_scene_cannot_be_proposed_on_unverified_evidence(studio) -> None:
    public, _key = publish_pairs(studio)
    pilot.record_review(studio["pilot_id"], review_for(studio, public))
    proposal = pilot.propose_scene(studio["pilot_id"], pairs_a=[PAIR_A], pairs_b=[PAIR_A])
    assert proposal["may_propose_scene"] is False
    assert proposal["requires_own_scope_grant"] is True


def test_a_combined_scene_needs_both_stages_to_support_improvement(studio,
                                                                   verified_owner) -> None:
    # Stage A is verified and improved; stage B is only a draft.
    public, key = publish_pairs(studio)
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, public, preference=variant_preference(key),
                                   authentication=authenticated(),
                                   at="2026-10-07T00:00:00+00:00"))
    editions = pilot.pilot_editions(studio["pilot_id"])
    second, second_key = pilot.blind_pairs(studio["pilot_id"], [{
        "pair_id": PAIR_B,
        "baseline_audio_sha256": _arm_digest(editions, "baseline"),
        "variant_audio_sha256": _arm_digest(editions, "variant")}], seed=4)
    pilot.record_review(studio["pilot_id"], review_for(studio, second))

    assert pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)["supports_improvement"] is True
    assert pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_B)["supports_improvement"] is False
    proposal = pilot.propose_scene(studio["pilot_id"], pairs_a=[PAIR_A], pairs_b=[PAIR_B])
    assert proposal["may_propose_scene"] is False
    assert proposal["requires_own_scope_grant"] is True

    # With both stages verified and improved, the scene may be proposed — and
    # still needs its own grant.  The second review reports a later time, which
    # is what makes it the current verdict for that pair.
    pilot.record_review(studio["pilot_id"],
                        review_for(studio, second, preference=variant_preference(second_key),
                                   authentication=authenticated(),
                                   at="2026-10-07T01:00:00+00:00"))
    proposal = pilot.propose_scene(studio["pilot_id"], pairs_a=[PAIR_A], pairs_b=[PAIR_B])
    assert proposal["may_propose_scene"] is True
    assert proposal["requires_own_scope_grant"] is True


def test_a_scene_proposal_needs_pairs_from_both_stages(studio) -> None:
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.propose_scene(studio["pilot_id"], pairs_a=[], pairs_b=[PAIR_A])
    assert refused.value.code == "input_ineligible"


def test_gates_read_and_write_nothing_outside_the_pilot(studio) -> None:
    source_root = pilot.safe_project_root(studio["project"].id)
    before = {str(path): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    public, _key = publish_pairs(studio)
    pilot.record_review(studio["pilot_id"], review_for(studio, public))
    pilot.improvement_gate(studio["pilot_id"], pair_id=PAIR_A)
    after = {str(path): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    assert after == before


def test_a_published_digest_binds_its_arm_and_timeline(studio) -> None:
    """A resolved pair names two published editions, not two arbitrary hashes."""

    public, key = publish_pairs(studio)
    editions = pilot.pilot_editions(studio["pilot_id"])
    row = key["key"][0]
    assert row["resolved"] is True
    assert row["x_audio_sha256"] == public["pairs"][0]["x_audio_sha256"]
    assert row["x_edition"] == editions[public["pairs"][0]["x_audio_sha256"]]
    assert row["y_edition"] == editions[public["pairs"][0]["y_audio_sha256"]]
    assert {row["x_edition"]["arm"], row["y_edition"]["arm"]} == {"baseline", "variant"}
    assert {row["x_edition"]["timeline_sha256"], row["y_edition"]["timeline_sha256"]} <= {
        info["timeline_sha256"] for info in editions.values()}
