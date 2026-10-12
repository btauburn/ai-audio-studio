"""CPU-only tests for the four-arm director comparison harness.

All audio here is artificial (sine tones and hand-built WAV bytes from
``build_synthetic_fixture`` or this file).  Nothing is speech, nothing is a
model result, and no test touches a live store, the network or the GPU.
"""
from __future__ import annotations

import copy
import io
import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from dubbing_service.app.audiobook import director_evaluation as de


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def pcm_wav(samples, rate=16000) -> bytes:
    pcm = np.clip(np.round(np.asarray(samples, dtype=np.float64) * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def raw_wav(payload: bytes, *, tag: int, channels: int, rate: int, bits: int, declared_size=None) -> bytes:
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    size = len(payload) if declared_size is None else declared_size
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", size) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def tone(seconds=1.0, amplitude=0.4, rate=16000, frequency=220.0):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


@pytest.fixture()
def fixture_manifest(tmp_path):
    return de.build_synthetic_fixture(tmp_path / "inputs")


def production_like(manifest):
    """A manifest without the fixture label, for exercising the human gate (test-only data)."""
    clone = copy.deepcopy(manifest)
    clone.pop("fixture")
    return clone


def rewrite_sample(manifest, arm, utterance_id, audio: bytes, root: Path):
    """Replace one sample's audio with new bytes, keeping the manifest hash honest."""
    sample = next(s for s in manifest["arms"][arm]["samples"] if s["utterance_id"] == utterance_id)
    (root / sample["audio_path"]).write_bytes(audio)
    sample["audio_sha256"] = de.sha256_hex(audio)
    sample.pop("receipt", None)
    return sample


def make_review(result, scorer):
    """Build a completed review; ``scorer(arm) -> rating`` is applied via the sealed key."""
    ballot, key = result["ballot"], result["ballot_key"]
    responses = []
    for item in ballot["items"]:
        options = key["items"][item["item_id"]]["options"]
        ratings = {label: {d: scorer(options[label]["arm"]) for d in de.RATING_DIMENSIONS} for label in options}
        responses.append({"item_id": item["item_id"], "ratings": ratings, "preferred_label": None})
    return {"ballot_id": ballot["ballot_id"], "reviewer_id": "reviewer-1", "responses": responses}


def snapshot(directory: Path):
    return {p.name: de.sha256_hex(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}


# --------------------------------------------------------------------------- #
# fixture and basic evaluation
# --------------------------------------------------------------------------- #


def test_fixture_is_labeled_complete_and_excluded_from_readiness(fixture_manifest):
    result = de.evaluate_comparison(fixture_manifest)
    assert result["fixture"]["is_fixture"] is True
    assert "SYNTHETIC" in result["fixture"]["label"]
    assert result["production_eligible"] is False
    assert result["status"] == "complete"
    assert result["arm_status"] == {arm: "complete" for arm in de.ARMS}
    assert result["lines"]["count"] >= de.MIN_UTTERANCES
    roles = result["lines"]["roles"]
    assert any(r["role_type"] == "narration" for r in roles)
    assert sum(r["role_type"] == "dialogue" for r in roles) >= 2
    assert result["human_gate"]["status"] == "fixture_excluded"
    assert result["human_gate"]["approval"] is False


def test_unavailable_measures_are_null_never_estimated(fixture_manifest):
    result = de.evaluate_comparison(fixture_manifest)
    assert result["unavailable_measures"]["asr"]["word_error_rate"] is None
    assert result["unavailable_measures"]["acoustic_voice_embedding"]["similarity"] is None
    for arm in result["arms"].values():
        for sample in arm["samples"]:
            assert sample["asr"]["status"] == "unavailable"
            assert sample["asr"]["omissions"] is None and sample["asr"]["intelligibility"] is None
            assert sample["acoustic_voice_embedding"]["similarity"] is None
            assert sample["human_measures"] == {"status": "pending"}


def test_measurements_reflect_the_actual_audio(fixture_manifest):
    result = de.evaluate_comparison(fixture_manifest)
    first = result["arms"]["baseline"]["samples"][0]["measures"]
    # 0.6s tone + 0.1s artificial tail silence at 16 kHz.
    assert first["duration_seconds"] == pytest.approx(0.7, abs=1e-6)
    assert first["peak"] == pytest.approx(0.5, abs=0.01)
    assert first["clipping"]["indicator"] == "none"
    assert first["tail"]["indicator"] == "ends_in_silence"
    assert first["silence"]["trailing_seconds"] == pytest.approx(0.1, abs=0.03)
    # the performance arm lengthens every fixture line; the comparison reports it as a number only
    rows = result["comparisons"]["performance_vs_baseline"]["per_utterance"]
    assert all(row["duration_delta_seconds"] == pytest.approx(0.1, abs=2e-3) for row in rows)
    assert "not a quality" in result["comparisons"]["interpretation"]


def test_receipt_metrics_come_only_from_bound_receipts(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    root = Path(manifest["base_dir"])
    manifest["arms"]["baseline"]["samples"][0].pop("receipt")
    unbound = manifest["arms"]["baseline"]["samples"][1]
    receipt_path = root / unbound["receipt"]["path"]
    body = json.loads(receipt_path.read_text())
    del body["audio_sha256"]
    receipt_path.write_text(json.dumps(body))
    result = de.evaluate_comparison(manifest)
    samples = result["arms"]["baseline"]["samples"]
    assert samples[0]["receipt"]["latency_seconds"] is None
    assert samples[0]["receipt"]["status"] == "absent"
    assert samples[1]["receipt"]["status"] == "unbound"
    assert samples[1]["receipt"]["latency_seconds"] is None and samples[1]["receipt"]["peak_vram_mb"] is None
    summary = result["arms"]["baseline"]["summary"]
    assert summary["latency_seconds"]["samples_with_receipt"] == 10
    assert summary["latency_seconds"]["complete"] is False
    # incomplete coverage must not produce a delta
    assert result["comparisons"]["casting_vs_baseline"]["mean_latency_seconds_delta"] is None
    assert result["comparisons"]["casting_vs_baseline"]["max_peak_vram_mb_delta"] is None


def test_complete_receipts_produce_latency_and_vram_deltas(fixture_manifest):
    delta = de.evaluate_comparison(fixture_manifest)["comparisons"]["casting_vs_baseline"]
    assert delta["mean_latency_seconds_delta"] == pytest.approx(0.1)
    assert delta["max_peak_vram_mb_delta"] == pytest.approx(1.0)


def test_receipt_for_other_audio_is_rejected(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    root = Path(manifest["base_dir"])
    path = root / manifest["arms"]["casting"]["samples"][0]["receipt"]["path"]
    body = json.loads(path.read_text())
    body["audio_sha256"] = "0" * 64
    path.write_text(json.dumps(body))
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "receipt_mismatch"


def test_manifest_object_is_not_mutated(fixture_manifest):
    fixture_manifest["human_review"] = {"ballot_id": "x"}
    snapshot_before = copy.deepcopy(fixture_manifest)
    de.evaluate_comparison(fixture_manifest)
    assert fixture_manifest == snapshot_before


# --------------------------------------------------------------------------- #
# audio measures
# --------------------------------------------------------------------------- #


def test_clipping_distinguishes_isolated_peaks_from_flat_tops():
    flat = tone(0.5, 0.3)
    flat[2000:2010] = 1.0
    isolated = tone(0.5, 0.3)
    isolated[2000] = 1.0
    flat_measures = de.measure_audio(pcm_wav(flat))
    isolated_measures = de.measure_audio(pcm_wav(isolated))
    assert flat_measures["clipping"]["indicator"] == "likely"
    assert isolated_measures["clipping"]["indicator"] == "possible"
    assert de.measure_audio(pcm_wav(tone(0.5, 0.3)))["clipping"]["indicator"] == "none"
    assert {f["code"]: f["severity"] for f in flat_measures["flags"]}["clipping_likely"] == "review"
    # ambiguity is not escalated to a defect
    assert {f["code"]: f["severity"] for f in isolated_measures["flags"]}["clipping_possible"] == "info"


def test_silence_leading_trailing_and_interior_pause():
    rate = 16000
    audio = np.concatenate([np.zeros(rate * 2), tone(0.5), np.zeros(int(rate * 0.4)), tone(0.5), np.zeros(rate * 2)])
    measures = de.measure_audio(pcm_wav(audio))
    silence = measures["silence"]
    assert silence["leading_seconds"] == pytest.approx(2.0, abs=0.03)
    assert silence["trailing_seconds"] == pytest.approx(2.0, abs=0.03)
    assert silence["longest_interior_pause_seconds"] == pytest.approx(0.4, abs=0.05)
    codes = {f["code"] for f in measures["flags"]}
    assert {"long_leading_silence", "long_trailing_silence"} <= codes
    assert all(f["severity"] == "info" for f in measures["flags"] if "silence" in f["code"])


def test_all_silent_and_empty_audio_are_blockers():
    silent = de.measure_audio(pcm_wav(np.zeros(8000)))
    assert silent["silence"]["all_silent"] is True
    assert silent["tail"]["indicator"] == "ambiguous"
    assert any(f["code"] == "silent_audio" and f["severity"] == "blocker" for f in silent["flags"])
    empty = de.measure_audio(pcm_wav([]))
    assert empty["duration_seconds"] == 0.0
    assert any(f["code"] == "empty_audio" and f["severity"] == "blocker" for f in empty["flags"])
    assert empty["silence"] is None and empty["peak"] is None


def test_tail_active_is_flagged_for_review_but_stays_an_indicator():
    measures = de.measure_audio(pcm_wav(tone(0.6, 0.4)))  # tone cut off at full level
    assert measures["tail"]["indicator"] == "active_at_end"
    assert "natural endings" in measures["tail"]["reason"]
    assert de.measure_audio(pcm_wav(tone(0.1, 0.4)))["tail"]["indicator"] == "ambiguous"


def test_float_wav_with_non_finite_samples_is_a_blocker():
    values = np.concatenate([tone(0.3), [np.nan, np.inf, -np.inf]]).astype("<f4")
    measures = de.measure_audio(raw_wav(values.tobytes(), tag=3, channels=1, rate=16000, bits=32))
    assert measures["non_finite_samples"] == 3
    assert measures["encoding"] == "float"
    assert any(f["code"] == "non_finite_samples" and f["severity"] == "blocker" for f in measures["flags"])


def test_24_bit_stereo_and_truncated_container_decoding():
    values = (tone(0.3, 0.5) * 8388607).astype(np.int32)
    payload = b"".join(int(v).to_bytes(3, "little", signed=True) for v in values)
    parsed = de.parse_wav(raw_wav(payload, tag=1, channels=1, rate=16000, bits=24))
    assert parsed["samples"].max() == pytest.approx(0.5, abs=0.01)
    stereo = np.stack([tone(0.2, 0.2), tone(0.2, 0.6)], axis=1)
    wav = raw_wav(np.round(stereo * 32767).astype("<i2").tobytes(), tag=1, channels=2, rate=16000, bits=16)
    assert de.measure_audio(wav)["peak"] == pytest.approx(0.6, abs=0.01)
    truncated = raw_wav(np.round(tone(0.2) * 32767).astype("<i2").tobytes(), tag=1, channels=1, rate=16000,
                        bits=16, declared_size=10_000_000)
    assert "container_truncated" in {f["code"] for f in de.measure_audio(truncated)["flags"]}


def test_unreadable_and_unsupported_audio_is_refused():
    with pytest.raises(de.EvaluationError) as raised:
        de.measure_audio(b"not a wave file at all")
    assert raised.value.code == "audio_unreadable"
    with pytest.raises(de.EvaluationError) as raised:
        de.parse_wav(raw_wav(b"\x00" * 8, tag=2, channels=1, rate=16000, bits=16))
    assert raised.value.code == "audio_unsupported_format"


# --------------------------------------------------------------------------- #
# manifest and cross-arm integrity
# --------------------------------------------------------------------------- #


def test_requires_twelve_lines(fixture_manifest):
    short = copy.deepcopy(fixture_manifest)
    short["utterances"] = short["utterances"][:11]
    with pytest.raises(de.EvaluationError, match="at least 12"):
        de.evaluate_comparison(short)


def test_requires_narration_and_two_dialogue_roles(fixture_manifest):
    no_narration = copy.deepcopy(fixture_manifest)
    for u in no_narration["utterances"]:
        u["role_type"] = "dialogue"
    with pytest.raises(de.EvaluationError, match="narration"):
        de.evaluate_comparison(no_narration)

    one_voice = copy.deepcopy(fixture_manifest)
    for u in one_voice["utterances"]:
        if u["role_id"] == "bo":
            u["role_id"], u["role_name"] = "ada", "Ada"
    with pytest.raises(de.EvaluationError, match="dialogue roles"):
        de.evaluate_comparison(one_voice)


def test_text_mismatch_between_arms_is_rejected(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["performance"]["samples"][3]["text"] += " (edited)"
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "text_mismatch"
    assert raised.value.details["arm"] == "performance"


def test_engine_mismatch_between_arms_is_rejected(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["combined"]["engine"]["runtime_sha256"] = "a" * 64
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "engine_mismatch"
    other = copy.deepcopy(fixture_manifest)
    other["arms"]["casting"]["engine"]["name"] = "some-other-engine"
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(other)
    assert raised.value.code == "engine_mismatch"


def test_source_provenance_must_match(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["casting"]["samples"][0]["provenance"]["source_sha256"] = "b" * 64
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "provenance_mismatch"


def test_audio_hash_must_match_declared_hash(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["baseline"]["samples"][2]["audio_sha256"] = "c" * 64
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "audio_hash_mismatch"


def test_voice_pairing_keeps_direction_arms_attributable(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    voice = manifest["arms"]["performance"]["samples"][0]["voice"]
    voice["voice_id"], voice["identity_sha256"] = "someone-else", "d" * 64
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "voice_pairing_mismatch"


def test_direction_rules_per_arm(fixture_manifest):
    leaked = copy.deepcopy(fixture_manifest)
    leaked["arms"]["baseline"]["samples"][0]["direction"] = {"direction_sha256": "e" * 64}
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(leaked)
    assert raised.value.code == "direction_mismatch"
    undirected = copy.deepcopy(fixture_manifest)
    for sample in undirected["arms"]["performance"]["samples"]:
        sample["direction"] = None
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(undirected)
    assert raised.value.code == "direction_mismatch"
    assert "relabeled" in raised.value.message


def test_arm_that_is_a_byte_copy_of_another_is_refused(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    root = Path(manifest["base_dir"])
    for src, dst in zip(manifest["arms"]["baseline"]["samples"], manifest["arms"]["performance"]["samples"]):
        (root / dst["audio_path"]).write_bytes((root / src["audio_path"]).read_bytes())
        dst["audio_sha256"] = src["audio_sha256"]
        dst["receipt"] = None
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "arm_copied"


def test_same_file_referenced_by_two_arms_is_refused(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    base = manifest["arms"]["baseline"]["samples"][0]
    cast = manifest["arms"]["casting"]["samples"][0]
    cast["audio_path"], cast["audio_sha256"] = base["audio_path"], base["audio_sha256"]
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "source_shared"


def test_unchanged_lines_between_arms_are_reported_not_hidden(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    root = Path(manifest["base_dir"])
    base = manifest["arms"]["baseline"]["samples"][4]
    target = manifest["arms"]["performance"]["samples"][4]
    (root / target["audio_path"]).write_bytes((root / base["audio_path"]).read_bytes())
    target["audio_sha256"] = base["audio_sha256"]
    target["receipt"] = None
    comparison = de.evaluate_comparison(manifest)["comparisons"]["performance_vs_baseline"]
    assert comparison["unchanged_audio_utterance_ids"] == [base["utterance_id"]]
    assert comparison["changed_audio_utterances"] == 11


def test_casting_arm_without_voice_change_is_a_finding(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    for base, cast in zip(manifest["arms"]["baseline"]["samples"], manifest["arms"]["casting"]["samples"]):
        cast["voice"] = copy.deepcopy(base["voice"])
    for perf, comb in zip(manifest["arms"]["performance"]["samples"], manifest["arms"]["combined"]["samples"]):
        comb["voice"] = copy.deepcopy(perf["voice"])
    result = de.evaluate_comparison(manifest)
    assert [f["code"] for f in result["findings"]] == ["casting_voices_unchanged"]
    assert any("baseline voice" in w for w in result["warnings"])


def test_unknown_arm_missing_hash_and_relative_base_dir_are_invalid(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["extra"] = {}
    with pytest.raises(de.EvaluationError):
        de.evaluate_comparison(manifest)
    manifest = copy.deepcopy(fixture_manifest)
    del manifest["arms"]["baseline"]["samples"][0]["audio_sha256"]
    with pytest.raises(de.EvaluationError, match="audio_sha256"):
        de.evaluate_comparison(manifest)
    manifest = copy.deepcopy(fixture_manifest)
    manifest["base_dir"] = "relative/dir"
    with pytest.raises(de.EvaluationError, match="absolute"):
        de.evaluate_comparison(manifest)


def test_paths_may_not_climb_out_of_base_dir(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["baseline"]["samples"][0]["audio_path"] = "../outside.wav"
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(manifest)
    assert raised.value.code == "path_refused"


# --------------------------------------------------------------------------- #
# missing arms
# --------------------------------------------------------------------------- #


def test_missing_arm_stays_missing_and_nothing_is_borrowed(tmp_path):
    manifest = de.build_synthetic_fixture(tmp_path / "inputs", omit_arms=["casting"])
    result = de.evaluate_comparison(manifest)
    assert result["status"] == "incomplete"
    assert result["arm_status"]["casting"] == "missing"
    assert result["arms"]["casting"]["samples"] == []
    assert result["comparisons"]["casting_vs_baseline"] is None
    assert result["comparisons"]["performance_vs_baseline"] is not None
    assert result["ballot"]["status"] == "partial"
    assert all(len(item["options"]) == 3 for item in result["ballot"]["items"])
    assert any("casting" in w for w in result["warnings"])
    assert result["production_eligible"] is False


def test_missing_arm_cannot_satisfy_gate_even_with_a_review(tmp_path):
    manifest = production_like(de.build_synthetic_fixture(tmp_path / "inputs", omit_arms=["combined"]))
    first = de.evaluate_comparison(manifest)
    assert first["production_eligible"] is False
    manifest["human_review"] = make_review(first, lambda arm: 5)
    gate = de.evaluate_comparison(manifest)["human_gate"]
    assert gate["status"] == "held"
    assert gate["scores"] is None
    assert any("combined" in reason for reason in gate["reasons"])


def test_partial_arm_is_incomplete_and_excluded(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    dropped = manifest["arms"]["performance"]["samples"].pop()
    result = de.evaluate_comparison(manifest)
    assert result["arm_status"]["performance"] == "incomplete"
    assert result["arms"]["performance"]["missing_utterance_ids"] == [dropped["utterance_id"]]
    assert result["comparisons"]["performance_vs_baseline"] is None
    assert result["ballot"]["status"] == "partial"


def test_arm_marked_missing_with_samples_is_contradictory(fixture_manifest):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["casting"]["status"] = "missing"
    with pytest.raises(de.EvaluationError, match="marked missing"):
        de.evaluate_comparison(manifest)


# --------------------------------------------------------------------------- #
# blind ballot and human gate
# --------------------------------------------------------------------------- #


def test_ballot_is_deterministic_and_arm_blind(fixture_manifest):
    first = de.evaluate_comparison(fixture_manifest)
    second = de.evaluate_comparison(copy.deepcopy(fixture_manifest))
    assert first["ballot"] == second["ballot"] and first["ballot_key"] == second["ballot_key"]
    text = json.dumps(first["ballot"]).lower()
    assert not any(arm in text for arm in de.ARMS)
    assert ".wav" not in text and "sha256" not in text
    labels_by_arm = {arm: [] for arm in de.ARMS}
    for item in first["ballot"]["items"]:
        assert [o["label"] for o in item["options"]] == list("ABCD")
        options = first["ballot_key"]["items"][item["item_id"]]["options"]
        assert sorted(o["arm"] for o in options.values()) == sorted(de.ARMS)
        for label, option in options.items():
            labels_by_arm[option["arm"]].append(label)
    assert any(len(set(labels)) > 1 for labels in labels_by_arm.values())  # not a fixed arm order
    assert first["ballot"]["ballot_id"] == first["ballot_key"]["ballot_id"]


def test_ballot_changes_when_the_line_set_changes(tmp_path):
    a = de.evaluate_comparison(de.build_synthetic_fixture(tmp_path / "a"))
    b = de.evaluate_comparison(de.build_synthetic_fixture(tmp_path / "b", utterance_count=15))
    assert a["ballot"]["ballot_id"] != b["ballot"]["ballot_id"]


def test_gate_pending_without_review_and_never_approves(fixture_manifest):
    gate = de.evaluate_comparison(production_like(fixture_manifest))["human_gate"]
    assert gate["status"] == "pending"
    assert gate["approval"] is False
    assert gate["scores"] is None and gate["verdicts"] is None


def test_gate_supports_adoption_review_when_combined_improves(fixture_manifest):
    manifest = production_like(fixture_manifest)
    first = de.evaluate_comparison(manifest)
    ratings = {"baseline": 3, "casting": 4, "performance": 3, "combined": 5}
    manifest["human_review"] = make_review(first, ratings.get)
    result = de.evaluate_comparison(manifest)
    gate = result["human_gate"]
    assert gate["status"] == "supports_adoption_review"
    assert gate["approval"] is False
    assert gate["verdicts"]["combined"] == {"overall_delta": 2.0, "verdict": "improved"}
    assert gate["verdicts"]["performance"]["verdict"] == "no_clear_change"
    assert gate["scores"]["arms"]["baseline"]["overall_mean"] == 3.0
    assert gate["human_review_sha256"] == de.content_sha256(manifest["human_review"])
    assert result["evaluation_id"] == first["evaluation_id"]  # review does not change the technical report


def test_gate_does_not_support_when_any_arm_regresses(fixture_manifest):
    manifest = production_like(fixture_manifest)
    first = de.evaluate_comparison(manifest)
    ratings = {"baseline": 4, "casting": 2, "performance": 4, "combined": 5}
    manifest["human_review"] = make_review(first, ratings.get)
    gate = de.evaluate_comparison(manifest)["human_gate"]
    assert gate["status"] == "does_not_support"
    assert gate["verdicts"]["casting"]["verdict"] == "regressed"


def test_gate_does_not_support_without_clear_combined_gain(fixture_manifest):
    manifest = production_like(fixture_manifest)
    first = de.evaluate_comparison(manifest)
    manifest["human_review"] = make_review(first, lambda arm: 3)
    assert de.evaluate_comparison(manifest)["human_gate"]["status"] == "does_not_support"


def test_invalid_reviews_hold_the_gate_with_reasons(fixture_manifest):
    manifest = production_like(fixture_manifest)
    first = de.evaluate_comparison(manifest)
    good = make_review(first, lambda arm: 3)

    wrong_ballot = copy.deepcopy(good)
    wrong_ballot["ballot_id"] = "0" * 32
    incomplete = copy.deepcopy(good)
    incomplete["responses"].pop()
    out_of_range = copy.deepcopy(good)
    out_of_range["responses"][0]["ratings"]["A"]["naturalness"] = 6
    boolean = copy.deepcopy(good)
    boolean["responses"][0]["ratings"]["A"]["clarity"] = True
    bad_preference = copy.deepcopy(good)
    bad_preference["responses"][0]["preferred_label"] = "Z"
    no_reviewer = copy.deepcopy(good)
    no_reviewer["reviewer_id"] = " "
    duplicate = copy.deepcopy(good)
    duplicate["responses"].append(copy.deepcopy(duplicate["responses"][0]))

    for review in (wrong_ballot, incomplete, out_of_range, boolean, bad_preference, no_reviewer, duplicate):
        manifest["human_review"] = review
        gate = de.evaluate_comparison(manifest)["human_gate"]
        assert gate["status"] == "held", review
        assert gate["scores"] is None
        assert len(gate["reasons"]) >= 2


def test_technical_blockers_hold_the_gate_despite_a_good_review(tmp_path):
    manifest = production_like(de.build_synthetic_fixture(tmp_path / "inputs"))
    root = Path(manifest["base_dir"])
    rewrite_sample(manifest, "combined", "fx-01", pcm_wav(np.zeros(8000)), root)
    first = de.evaluate_comparison(manifest)
    manifest["human_review"] = make_review(first, lambda arm: 5 if arm == "combined" else 2)
    gate = de.evaluate_comparison(manifest)["human_gate"]
    assert gate["status"] == "held"
    assert any("combined" in r and "blockers" in r for r in gate["reasons"])


def test_preference_and_defect_counts_are_decoded_through_the_key(fixture_manifest):
    manifest = production_like(fixture_manifest)
    first = de.evaluate_comparison(manifest)
    review = make_review(first, lambda arm: 3)
    for response in review["responses"]:
        options = first["ballot_key"]["items"][response["item_id"]]["options"]
        response["preferred_label"] = next(l for l, o in options.items() if o["arm"] == "performance")
        loser = next(l for l, o in options.items() if o["arm"] == "baseline")
        response["defects"] = {loser: ["clicks"]}
    manifest["human_review"] = review
    scores = de.evaluate_comparison(manifest)["human_gate"]["scores"]["arms"]
    assert scores["performance"]["preferred_count"] == len(review["responses"])
    assert scores["baseline"]["defects_listed"] == len(review["responses"])
    assert scores["combined"]["preferred_count"] == 0


# --------------------------------------------------------------------------- #
# immutability and outputs
# --------------------------------------------------------------------------- #


def test_inputs_are_left_untouched_and_reported_with_hashes(fixture_manifest, tmp_path):
    root = Path(fixture_manifest["base_dir"])
    before = snapshot(root)
    result = de.evaluate_comparison(fixture_manifest, output_dir=tmp_path / "out")
    assert snapshot(root) == before
    integrity = result["source_integrity"]
    assert integrity["unchanged"] is True
    by_name = {Path(f["path"]).name: f["sha256"] for f in integrity["files"]}
    assert by_name == before  # every wav and receipt is accounted for, with its hash


def test_input_changing_during_evaluation_is_detected(fixture_manifest, monkeypatch):
    root = Path(fixture_manifest["base_dir"])
    victim = root / fixture_manifest["arms"]["baseline"]["samples"][0]["audio_path"]
    real = de.measure_audio
    calls = {"n": 0}

    def tamper(data, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            victim.write_bytes(victim.read_bytes() + b"\x00\x00")
        return real(data, **kwargs)

    monkeypatch.setattr(de, "measure_audio", tamper)
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(fixture_manifest)
    assert raised.value.code == "source_changed"


def test_outputs_are_exclusive_and_retry_is_safe(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    first = de.evaluate_comparison(fixture_manifest, output_dir=out)
    assert {f["name"]: f["status"] for f in first["outputs"]["files"]} == {
        "evaluation.json": "created", "ballot.json": "created", "ballot_key.json": "created"}
    written = snapshot(out)
    again = de.evaluate_comparison(fixture_manifest, output_dir=out)
    assert {f["status"] for f in again["outputs"]["files"]} == {"reused"}
    assert snapshot(out) == written
    assert not [p for p in out.iterdir() if p.name.endswith(".stage")]
    evaluation = json.loads((out / "evaluation.json").read_text())
    assert "ballot_key" not in evaluation and "ballot" not in evaluation
    assert evaluation["evaluation_id"] == first["evaluation_id"]
    ballot_text = (out / "ballot.json").read_text().lower()
    assert not any(arm in ballot_text for arm in de.ARMS)


def test_different_result_in_an_existing_output_dir_is_refused(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    de.evaluate_comparison(fixture_manifest, output_dir=out)
    changed = copy.deepcopy(fixture_manifest)
    changed["utterances"][0]["text"] += "!"
    for arm in de.ARMS:
        changed["arms"][arm]["samples"][0]["text"] += "!"
    before = snapshot(out)
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(changed, output_dir=out)
    assert raised.value.code == "output_conflict"
    assert snapshot(out) == before


def test_conflict_is_checked_before_any_new_file_is_written(fixture_manifest, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "ballot_key.json").write_text("{}")  # foreign file in the way
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(fixture_manifest, output_dir=out)
    assert raised.value.code == "output_conflict"
    assert sorted(p.name for p in out.iterdir()) == ["ballot_key.json"]
    assert (out / "ballot_key.json").read_text() == "{}"


def test_human_gate_file_is_added_only_with_a_review_and_is_immutable(fixture_manifest, tmp_path):
    manifest = production_like(fixture_manifest)
    out = tmp_path / "out"
    first = de.evaluate_comparison(manifest, output_dir=out)
    assert not (out / "human_gate.json").exists()
    manifest["human_review"] = make_review(first, lambda arm: 5 if arm == "combined" else 3)
    second = de.evaluate_comparison(manifest, output_dir=out)
    assert {f["name"]: f["status"] for f in second["outputs"]["files"]} == {
        "evaluation.json": "reused", "ballot.json": "reused", "ballot_key.json": "reused",
        "human_gate.json": "created"}
    other = copy.deepcopy(manifest)
    other["human_review"] = make_review(first, lambda arm: 2)
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(other, output_dir=out)
    assert raised.value.code == "output_conflict"


def test_output_directory_may_not_contain_inputs(fixture_manifest):
    root = Path(fixture_manifest["base_dir"])
    with pytest.raises(de.EvaluationError) as raised:
        de.evaluate_comparison(fixture_manifest, output_dir=root)
    assert raised.value.code == "output_conflict"
    assert not (root / "evaluation.json").exists()


def test_no_files_are_written_when_validation_fails(fixture_manifest, tmp_path):
    manifest = copy.deepcopy(fixture_manifest)
    manifest["arms"]["casting"]["engine"]["name"] = "different"
    out = tmp_path / "out"
    with pytest.raises(de.EvaluationError):
        de.evaluate_comparison(manifest, output_dir=out)
    assert not out.exists()


def test_stage_blind_audio_uses_blind_names_and_verified_bytes(fixture_manifest, tmp_path):
    result = de.evaluate_comparison(fixture_manifest)
    staged = de.stage_blind_audio(result, tmp_path / "blind")
    assert len(staged) == 12 * 4
    assert not any(arm in Path(s["file"]).name for s in staged for arm in de.ARMS)
    option = result["ballot_key"]["items"]["item-001"]["options"]["A"]
    assert de.sha256_hex((tmp_path / "blind" / "item-001-A.wav").read_bytes()) == option["audio_sha256"]
    assert {s["status"] for s in de.stage_blind_audio(result, tmp_path / "blind")} == {"reused"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_cli():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_directors.py"
    spec = importlib.util.spec_from_file_location("compare_directors_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_fixture_run_publishes_results_and_hides_the_key(tmp_path, capsys):
    cli = load_cli()
    assert cli.main(["--write-fixture", str(tmp_path / "fx")]) == 0
    capsys.readouterr()
    out = tmp_path / "out"
    manifest = str(tmp_path / "fx" / "manifest.json")
    assert cli.main([manifest, "--output-dir", str(out), "--stage-blind-audio", str(tmp_path / "blind")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["fixture"]["is_fixture"] is True
    assert summary["production_eligible"] is False
    assert summary["human_gate"]["status"] == "fixture_excluded"
    assert "ballot_key" not in summary and "items" not in summary
    assert (out / "ballot_key.json").is_file() and len(summary["staged_blind_audio"]) == 48
    # an identical retry succeeds, and re-writing the identical fixture is a no-op
    assert cli.main([manifest, "--output-dir", str(out)]) == 0
    assert cli.main(["--write-fixture", str(tmp_path / "fx")]) == 0


def test_cli_reports_untrusted_input_with_exit_code_2(tmp_path, capsys):
    cli = load_cli()
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    assert cli.main([str(bad)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "manifest_invalid"
    assert cli.main([str(tmp_path / "missing.json")]) == 2


def test_cli_scores_a_human_review_file(tmp_path, capsys):
    cli = load_cli()
    manifest = production_like(de.build_synthetic_fixture(tmp_path / "fx"))
    manifest_path = tmp_path / "fx" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    first = de.evaluate_comparison(manifest)
    review = make_review(first, {"baseline": 3, "casting": 3, "performance": 3, "combined": 5}.get)
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    assert cli.main([str(manifest_path), "--human-review", str(review_path),
                     "--output-dir", str(tmp_path / "out")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["human_gate"]["status"] == "supports_adoption_review"
    assert summary["human_gate"]["approval"] is False
    assert (tmp_path / "out" / "human_gate.json").is_file()
