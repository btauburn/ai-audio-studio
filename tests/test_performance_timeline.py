"""Timeline-compilation contracts: integer frames, bound offsets, refused inputs.

The compiler is pure.  These tests build measurement and classification records
by hand so every frame count is arithmetic a reader can check.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries

RATE = 24000
AUDIO = "a" * 64
TAKE_FRAMES = 60000


def ms(value: float) -> int:
    return boundaries.frames_for_ms(value, RATE)


def measurement(*, trailing_content: int = 0, leading_content: int = 0, leading_silent: int = 0,
                status: str = "measured", flags=(), audio: str = AUDIO,
                sample_rate: int = RATE, frames: int = TAKE_FRAMES) -> dict:
    record = {
        "schema_version": boundaries.EDGE_MEASUREMENT_SCHEMA,
        "status": status,
        "audio_sha256": audio,
        "sample_rate": sample_rate,
        "sample_domain": boundaries.sample_domain(sample_rate),
        "frames": frames,
        "channels": 1,
        "detector": {"version": boundaries.DETECTOR_VERSION},
        "leading": {"content_frames": leading_content, "silent_frames": leading_silent,
                    "floor_dbfs": -80.0},
        "trailing": {"content_frames": trailing_content, "silent_frames": trailing_content,
                     "floor_dbfs": -80.0},
        "onset": {"first_sample": 0.0, "max_step_2ms": 0.0, "dc_removed": True},
        "lexical": {"status": "not_requested"},
        "flags": list(flags),
        "machine_generated": True,
        "human_verified": False,
    }
    record["content_sha256"] = boundaries.content_sha256(record)
    return record


def classification(klass: str, **features) -> dict:
    return {
        "schema_version": boundaries.BOUNDARY_CLASSIFICATION_SCHEMA,
        "classifier_version": boundaries.CLASSIFIER_VERSION,
        "class": klass,
        "features": features or {"a_end": ".", "b_lower": False, "para_break": False},
        "reason": None,
        "a_role_id": "narrator",
        "b_role_id": "rosa",
        "machine_generated": True,
        "human_verified": False,
    }


def row(utterance_id: str, record: dict, boundary: dict | None = None, *, start: int = 0,
        end: int = TAKE_FRAMES) -> dict:
    return {
        "project_id": "project-1",
        "utterance_id": utterance_id,
        "take_id": f"take-{utterance_id}",
        "audio_sha256": record["audio_sha256"],
        "measurement_sha256": record["content_sha256"],
        "sample_rate": record["sample_rate"],
        "source_start_frame": start,
        "source_end_frame": end,
        "measurement": record,
        "boundary": boundary,
    }


def pair(klass: str = "N_D_SAME_PARA", *, trailing: int = 0, leading: int = 0,
         silent: int | None = None, a_flags=(), b_flags=()) -> list[dict]:
    """One adjacent pair.  ``silent`` defaults to the whole leading edge region.

    The measurement contract requires ``silent_frames <= content_frames`` on the
    same edge: the strictly-silent run is a subset of the pre-content span.
    """

    leading_silent = leading if silent is None else silent
    first = row("u1", measurement(trailing_content=trailing, flags=a_flags))
    second = row("u2", measurement(leading_content=leading, leading_silent=leading_silent,
                                  flags=b_flags),
                 classification(klass))
    return [first, second]


def compile_rows(rows, *, policy=None, hints=None, allow_lead_offset=False) -> dict:
    return boundaries.compile_timeline(rows, policy=policy or boundaries.default_policy(),
                                       hints=hints or {}, allow_lead_offset=allow_lead_offset)


def spacers(timeline: dict) -> list[dict]:
    return [item for item in timeline["placements"] if item["kind"] == "spacer"]


def test_the_worked_case_gives_a_ninety_millisecond_spacer() -> None:
    policy = boundaries.default_policy()
    policy["targets_ms"]["N_D_SAME_PARA"] = 300.0
    rows = pair("N_D_SAME_PARA", trailing=ms(120.0), leading=ms(90.0))
    timeline = compile_rows(rows, policy=policy)
    assert timeline["schema_version"] == "boundary-timeline-v1"
    assert timeline["sample_rate"] == RATE
    assert timeline["sample_domain"] == boundaries.sample_domain(RATE)
    boundary = timeline["boundaries"][0]
    assert boundary["existing_frames"] == ms(210.0)
    assert boundary["target_frames"] == ms(300.0)
    assert boundary["spacer_frames"] == ms(90.0)
    assert boundary["effective_frames"] == ms(300.0)
    assert boundary["status"] == "met"
    assert spacers(timeline)[0]["frames"] == ms(90.0)


def test_recompiling_the_same_inputs_gives_the_same_hash() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(40.0), leading=ms(20.0))
    first = compile_rows(rows)
    second = compile_rows(rows)
    assert first["timeline_sha256"] == second["timeline_sha256"]
    assert first == second
    assert len(first["timeline_sha256"]) == 64


def test_a_dependency_change_changes_the_hash() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(40.0), leading=ms(20.0))
    baseline = compile_rows(rows)["timeline_sha256"]

    changed_measurement = pair("N_D_SAME_PARA", trailing=ms(41.0), leading=ms(20.0))
    assert compile_rows(changed_measurement)["timeline_sha256"] != baseline

    changed_class = pair("N_D_NEW_PARA", trailing=ms(40.0), leading=ms(20.0))
    assert compile_rows(changed_class)["timeline_sha256"] != baseline

    changed_policy = boundaries.default_policy()
    changed_policy["targets_ms"]["N_D_SAME_PARA"] = 999.0
    assert compile_rows(rows, policy=changed_policy)["timeline_sha256"] != baseline

    hinted = compile_rows(rows, hints={"u1": "hesitant"})
    assert hinted["timeline_sha256"] != baseline

    assert compile_rows(rows)["timeline_sha256"] \
        != compile_rows(rows, allow_lead_offset=True)["timeline_sha256"]

    other_rate = pair("N_D_SAME_PARA", trailing=ms(40.0), leading=ms(20.0))
    for item in other_rate:
        item["sample_rate"] = 48000
        item["measurement"]["sample_rate"] = 48000
        item["measurement"]["sample_domain"] = boundaries.sample_domain(48000)
        item["measurement"]["content_sha256"] = boundaries.content_sha256(item["measurement"])
        item["measurement_sha256"] = item["measurement"]["content_sha256"]
    assert compile_rows(other_rate)["timeline_sha256"] != baseline


def test_long_edges_are_limited_and_never_negative() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(300.0))
    timeline = compile_rows(rows, allow_lead_offset=False)
    boundary = timeline["boundaries"][0]
    assert boundary["spacer_frames"] == 0
    assert boundary["offset_frames"] == 0
    assert boundary["status"] == "limited_by_existing_audio"
    assert boundary["effective_frames"] == boundary["existing_frames"] == ms(900.0)
    assert spacers(timeline) == []

    exact = pair("D_D_REPLY", trailing=ms(200.0), leading=ms(200.0), silent=0)
    zero = compile_rows(exact)["boundaries"][0]
    assert zero["spacer_frames"] == 0
    # existing == target here (400 ms), so the gap is already exact and the
    # boundary is met even though nothing had to be adjusted.  The pre-follow-up
    # module reported limited_by_existing_audio for this case.
    assert zero["existing_frames"] == zero["target_frames"] == ms(400.0)
    assert zero["effective_frames"] == zero["target_frames"]
    assert zero["status"] == "met"


def test_an_explicit_separator_appears_once_and_takes_no_second_pad() -> None:
    rows = pair("SCENE_SEPARATOR", trailing=ms(120.0), leading=ms(90.0))
    timeline = compile_rows(rows, allow_lead_offset=True)
    separators = [item for item in timeline["placements"] if item["kind"] == "scene_separator"]
    assert len(separators) == 1
    assert separators[0]["frames"] == ms(boundaries.SCENE_SEPARATOR_MS)
    assert spacers(timeline) == []
    assert timeline["boundaries"][0]["spacer_frames"] == 0
    assert timeline["boundaries"][0]["offset_frames"] == 0
    assert timeline["boundaries"][0]["status"] == "met"


def test_unknown_boundaries_keep_baseline_timing() -> None:
    rows = pair("UNKNOWN", trailing=ms(50.0), leading=ms(40.0))
    timeline = compile_rows(rows, allow_lead_offset=True)
    boundary = timeline["boundaries"][0]
    assert boundary["status"] == "baseline"
    assert boundary["spacer_frames"] == 0
    assert boundary["offset_frames"] == 0
    assert boundary["effective_frames"] == boundary["existing_frames"] == ms(90.0)
    assert spacers(timeline) == []


def test_the_offset_respects_the_bound_and_the_forty_millisecond_guard() -> None:
    policy = boundaries.default_policy()
    policy["targets_ms"]["D_D_REPLY"] = 200.0
    # existing = 500 + 400 = 900 ms; target 200 ms; silent lead-in 400 ms.
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    boundary = compile_rows(rows, policy=policy, allow_lead_offset=True)["boundaries"][0]
    guard = ms(boundaries.LEAD_OFFSET_GUARD_MS)
    assert boundary["eligible_offset_frames"] == ms(400.0) - guard
    assert boundary["offset_frames"] == min(ms(700.0), ms(400.0) - guard)
    assert boundary["offset_frames"] == ms(400.0) - guard
    assert boundary["effective_frames"] == ms(900.0) - (ms(400.0) - guard) + 0
    # The offset is bounded by the guard, so it cannot reach a 200 ms target:
    # 360 ms is the most it may remove, leaving 540 ms against a 200 ms target.
    # A successful partial adjustment is not proof the target was reached, so the
    # status is limited rather than met.  The pre-follow-up module reported met.
    assert boundary["offset_frames"] == ms(360.0)
    assert boundary["effective_frames"] == ms(540.0)
    assert boundary["effective_frames"] > boundary["target_frames"] == ms(200.0)
    assert boundary["status"] == "limited_by_existing_audio"

    # A small shortfall is bound by the excess over the target, not by the guard.
    # (Before the repair this used an inconsistent fixture whose silent run
    # exceeded its own lead-in; the guard now correctly binds whenever the
    # strictly-silent run is the whole lead-in, so the excess is made smaller.)
    rows_small = pair("D_D_REPLY", trailing=ms(20.0), leading=ms(200.0), silent=ms(200.0))
    small = compile_rows(rows_small, policy=policy, allow_lead_offset=True)["boundaries"][0]
    assert small["existing_frames"] == ms(220.0)
    assert small["eligible_offset_frames"] == ms(200.0) - guard
    assert small["offset_frames"] == ms(20.0)
    # Here the offset does reach the target exactly, so this boundary is met.
    assert small["effective_frames"] == small["target_frames"] == ms(200.0)
    assert small["status"] == "met"

    # A silent lead-in shorter than the guard can never be shortened.
    rows_tight = pair("D_D_REPLY", trailing=ms(200.0), leading=ms(200.0), silent=ms(20.0))
    tight = compile_rows(rows_tight, policy=policy, allow_lead_offset=True)["boundaries"][0]
    assert tight["eligible_offset_frames"] == 0
    assert tight["offset_frames"] == 0


def test_offsets_are_refused_outside_the_eligible_classes() -> None:
    rows = pair("D_N_BEAT_SAME_PARA", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    boundary = compile_rows(rows, allow_lead_offset=True)["boundaries"][0]
    assert boundary["offset_frames"] == 0
    assert boundary["status"] == "limited_by_existing_audio"

    disabled = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    off = compile_rows(disabled, allow_lead_offset=False)["boundaries"][0]
    assert off["offset_frames"] == 0
    assert off["status"] == "limited_by_existing_audio"


def test_an_uncertain_edge_protects_the_boundary() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0),
                b_flags=(boundaries.EDGE_UNCERTAIN,))
    boundary = compile_rows(rows, allow_lead_offset=True)["boundaries"][0]
    assert boundary["status"] == "protected_uncertain_edge"
    assert boundary["offset_frames"] == 0
    assert boundary["uncertain_edges"] == ["leading"]

    trailing_only = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0),
                         a_flags=(boundaries.EDGE_UNCERTAIN,))
    protected = compile_rows(trailing_only, allow_lead_offset=True)["boundaries"][0]
    assert protected["status"] == "protected_uncertain_edge"
    assert protected["uncertain_edges"] == ["trailing"]

    all_silent = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    all_silent[1]["measurement"]["status"] = "all_silent"
    all_silent[1]["measurement"]["flags"] = [boundaries.ALL_SILENT_FLAG]
    all_silent[1]["measurement"]["content_sha256"] = boundaries.content_sha256(all_silent[1]["measurement"])
    all_silent[1]["measurement_sha256"] = all_silent[1]["measurement"]["content_sha256"]
    guarded = compile_rows(all_silent, allow_lead_offset=True)["boundaries"][0]
    assert guarded["status"] == "protected_uncertain_edge"
    assert guarded["offset_frames"] == 0


def test_uncertainty_never_blocks_a_safe_pad() -> None:
    rows = pair("D_D_REPLY", trailing=ms(50.0), leading=ms(40.0),
                a_flags=(boundaries.EDGE_UNCERTAIN,))
    boundary = compile_rows(rows, allow_lead_offset=True)["boundaries"][0]
    assert boundary["spacer_frames"] == ms(400.0) - ms(90.0)
    assert boundary["status"] == "met"


def test_hints_adjust_the_target_with_its_floor_and_cap() -> None:
    policy = boundaries.default_policy()
    # D_D_INTERRUPTION targets 120 ms; "quick" halves it and the floor holds it at 80 ms.
    rows = pair("D_D_INTERRUPTION", trailing=ms(20.0), leading=ms(10.0))
    assert compile_rows(rows)["boundaries"][0]["target_frames"] == ms(120.0)
    assert compile_rows(rows, hints={"u1": "quick"})["boundaries"][0]["target_frames"] == ms(80.0)

    # N_D_SAME_PARA targets 250 ms; "hesitant" adds 300 ms.
    normal = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    assert compile_rows(normal, hints={"u1": "hesitant"})["boundaries"][0]["target_frames"] == ms(550.0)
    assert compile_rows(normal, hints={"u1": "withheld"})["boundaries"][0]["target_frames"] == ms(850.0)
    assert compile_rows(normal, hints={"u1": "beat_after"})["boundaries"][0]["target_frames"] == ms(600.0)

    capped = boundaries.default_policy()
    capped["targets_ms"]["N_N_NEW_PARA"] = 1400.0
    tall = pair("N_N_NEW_PARA", trailing=ms(20.0), leading=ms(10.0))
    boundary = compile_rows(tall, policy=capped, hints={"u1": "withheld"})["boundaries"][0]
    assert boundary["target_frames"] == ms(boundaries.HINT_CAP_MS) == ms(1500.0)
    assert boundary["hint"] == "withheld"


def test_unknown_or_conflicting_hints_are_refused_not_hidden() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    for hints in ({"u9": "quick"}, {"u1": "louder"}, {"u2": "quick"}, {"u2": "beat_after"}):
        with pytest.raises(boundaries.PerformanceDirectorError) as refused:
            compile_rows(rows, hints=hints)
        assert refused.value.code == "direction_conflict"

    separator = pair("SCENE_SEPARATOR", trailing=ms(20.0), leading=ms(10.0))
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(separator, hints={"u1": "quick"})
    assert refused.value.code == "direction_conflict"
    assert "separator" in str(refused.value)


def test_total_frame_accounting_adds_up() -> None:
    rows = [row("u1", measurement(trailing_content=ms(30.0))),
            row("u2", measurement(leading_content=ms(20.0), leading_silent=ms(20.0)),
                classification("N_D_SAME_PARA")),
            row("u3", measurement(leading_content=ms(10.0), leading_silent=ms(10.0)),
                classification("N_D_SAME_PARA")),
            row("u4", measurement(leading_content=ms(0.0)), classification("SCENE_SEPARATOR"))]
    timeline = compile_rows(rows)
    total = 0
    cursor = 0
    for item in timeline["placements"]:
        if item["kind"] == "speech":
            assert item["derived_start_frame"] == cursor
            total += item["derived_frames"]
            cursor += item["derived_frames"]
        else:
            total += item["frames"]
            cursor += item["frames"]
    assert total == timeline["total_frames"] == cursor
    kinds = [item["kind"] for item in timeline["placements"]]
    assert kinds.count("scene_separator") == 1
    assert kinds.count("speech") == 4
    spacer_total = sum(item["frames"] for item in timeline["placements"] if item["kind"] == "spacer")
    speech_total = sum(item["derived_frames"] for item in timeline["placements"] if item["kind"] == "speech")
    separator_total = sum(item["frames"] for item in timeline["placements"]
                          if item["kind"] == "scene_separator")
    assert timeline["total_frames"] == spacer_total + speech_total + separator_total


def test_all_original_source_frames_remain_when_offsets_are_disabled() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    timeline = compile_rows(rows, allow_lead_offset=False)
    for item in timeline["placements"]:
        if item["kind"] == "speech":
            assert item["offset_frames"] == 0
            assert item["derived_frames"] == item["source_end_frame"] - item["source_start_frame"]
    assert timeline["total_frames"] == 2 * TAKE_FRAMES


def test_the_offset_only_removes_leading_frames_below_the_ceiling() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    timeline = compile_rows(rows, allow_lead_offset=True)
    speech = [item for item in timeline["placements"] if item["kind"] == "speech"]
    assert speech[0]["offset_frames"] == 0
    second = speech[1]
    assert second["offset_frames"] > 0
    assert second["offset_frames"] <= ms(400.0)
    assert second["derived_frames"] == (second["source_end_frame"] - second["source_start_frame"]) \
        - second["offset_frames"]


def test_the_compiler_states_its_eligibility_limit() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    timeline = compile_rows(rows)
    limitations = " ".join(timeline["limitations"])
    assert "eligibility" in limitations
    assert "qa_passed" in limitations
    assert "fingerprint" in limitations
    assert timeline["machine_generated"] is True and timeline["human_verified"] is False
    # A caller's eligibility flag is neither read nor trusted.
    trusting = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    for item in trusting:
        item["qa_passed"] = True
        item["missing"] = False
    assert compile_rows(trusting)["timeline_sha256"] == compile_rows(rows)["timeline_sha256"]


def test_derived_and_malformed_inputs_are_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    derived = copy.deepcopy(rows)
    derived[0]["derived"] = True
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(derived)
    assert refused.value.code == "input_ineligible"
    assert "derived" in str(refused.value)

    from_source = copy.deepcopy(rows)
    from_source[1]["source_take_id"] = "take-u1"
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(from_source)

    escaping = copy.deepcopy(rows)
    escaping[0]["utterance_id"] = "../../etc/passwd"
    with pytest.raises(boundaries.PerformanceDirectorError) as escaped:
        compile_rows(escaping)
    assert escaped.value.code == "input_ineligible"

    bad_kind = copy.deepcopy(rows)
    bad_kind[0]["kind"] = "scene_separator"
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(bad_kind)


def test_missing_or_mismatched_bindings_are_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))

    wrong_audio = copy.deepcopy(rows)
    wrong_audio[1]["audio_sha256"] = "b" * 64
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(wrong_audio)
    assert refused.value.code == "input_ineligible"

    wrong_measurement = copy.deepcopy(rows)
    wrong_measurement[0]["measurement_sha256"] = "c" * 64
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(wrong_measurement)

    tampered = copy.deepcopy(rows)
    tampered[0]["measurement"]["leading"]["silent_frames"] = 123
    with pytest.raises(boundaries.PerformanceDirectorError) as digest:
        compile_rows(tampered)
    assert "digest" in str(digest.value)

    unknown_schema = copy.deepcopy(rows)
    unknown_schema[0]["measurement"]["schema_version"] = "edge-measurement-v2"
    unknown_schema[0]["measurement"]["content_sha256"] = boundaries.content_sha256(
        unknown_schema[0]["measurement"])
    unknown_schema[0]["measurement_sha256"] = unknown_schema[0]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(unknown_schema)
    assert refused.value.code == "schema_unknown"

    missing_block = copy.deepcopy(rows)
    del missing_block[0]["measurement"]["trailing"]
    missing_block[0]["measurement"]["content_sha256"] = boundaries.content_sha256(
        missing_block[0]["measurement"])
    missing_block[0]["measurement_sha256"] = missing_block[0]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(missing_block)


def test_impossible_frame_ranges_and_foreign_audio_are_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))

    negative = copy.deepcopy(rows)
    negative[0]["source_start_frame"] = -1
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(negative)

    inverted = copy.deepcopy(rows)
    inverted[0]["source_end_frame"] = 0
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(inverted)
    assert "impossible" in str(refused.value)

    beyond = copy.deepcopy(rows)
    beyond[0]["source_end_frame"] = TAKE_FRAMES + 1
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(beyond)

    mismatched_domain = copy.deepcopy(rows)
    mismatched_domain[1]["sample_rate"] = 48000
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(mismatched_domain)
    assert refused.value.code == "input_ineligible"


def test_non_finite_and_unknown_policies_are_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    for bad in (float("nan"), float("inf")):
        policy = boundaries.default_policy()
        policy["targets_ms"]["N_D_SAME_PARA"] = bad
        with pytest.raises(boundaries.PerformanceDirectorError):
            compile_rows(rows, policy=policy)

    unknown_version = boundaries.default_policy()
    unknown_version["schema_version"] = "boundary-policy-v2"
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(rows, policy=unknown_version)
    assert refused.value.code == "schema_unknown"

    # A target class omitted by design is not a refusal: the boundary becomes a
    # baseline boundary that keeps its explicit baseline spacing.  (Before the
    # repair an omitted class was refused as schema_unknown.)
    omitted = boundaries.default_policy()
    del omitted["targets_ms"]["N_D_SAME_PARA"]
    baseline = compile_rows(rows, policy=omitted)
    assert baseline["boundaries"][0]["status"] == "baseline"
    assert baseline["boundaries"][0]["target_frames"] is None
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy=omitted, hints={"u1": "quick"})

    misspelled = boundaries.default_policy()
    misspelled["targets_ms"]["D_TAG_V2"] = 10.0
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(rows, policy=misspelled)
    assert refused.value.code == "schema_unknown"

    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy={"schema_version": boundaries.BOUNDARY_POLICY_SCHEMA})


def test_fixed_policy_invariants_cannot_be_relaxed() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    guarded = boundaries.default_policy()

    # The 40 ms guard is an invariant, not a target: a zero guard must not be
    # able to remove a whole silent lead-in.
    guarded["offset_guard_ms"] = 0.0
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(rows, policy=guarded, allow_lead_offset=True)
    assert refused.value.code == "direction_conflict"
    guarded["offset_guard_ms"] = boundaries.LEAD_OFFSET_GUARD_MS - 1.0
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy=guarded)
    guarded["offset_guard_ms"] = float(boundaries.LEAD_OFFSET_GUARD_MS)
    boundary = compile_rows(rows, policy=guarded, allow_lead_offset=True)["boundaries"][0]
    assert boundary["offset_frames"] == ms(400.0) - ms(boundaries.LEAD_OFFSET_GUARD_MS)

    floored = boundaries.default_policy()
    floored["hint_floor_ms"] = boundaries.HINT_FLOOR_MS - 1.0
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy=floored)

    capped = boundaries.default_policy()
    capped["hint_cap_ms"] = boundaries.HINT_CAP_MS + 1.0
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy=capped)

    inverted = boundaries.default_policy()
    inverted["hint_floor_ms"] = boundaries.HINT_CAP_MS
    inverted["hint_cap_ms"] = boundaries.HINT_FLOOR_MS
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(rows, policy=inverted)


def test_the_policy_and_boundary_records_are_reachable_from_the_output() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    timeline = compile_rows(rows)
    assert timeline["policy_sha256"] == boundaries.content_sha256(boundaries.default_policy())
    assert timeline["classifier_version"] == boundaries.CLASSIFIER_VERSION
    assert timeline["assembler_version"] == boundaries.ASSEMBLER_VERSION
    assert boundaries.default_policy()["schema_version"] == boundaries.BOUNDARY_POLICY_SCHEMA
    assert set(boundaries.POLICY_TARGET_CLASSES) <= set(boundaries.default_policy()["targets_ms"])
    assert boundaries.SCENE_SEPARATOR not in boundaries.default_policy()["targets_ms"]


def test_a_timing_plan_changes_no_take_identity() -> None:
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0))
    before = [copy.deepcopy(item["measurement"]) for item in rows]
    timeline = compile_rows(rows, allow_lead_offset=True)
    assert [item["measurement"] for item in rows] == before
    for placement in timeline["placements"]:
        if placement["kind"] == "speech":
            assert placement["audio_sha256"] == AUDIO
            assert placement["measurement_sha256"] == rows[0]["measurement_sha256"] or \
                len(placement["measurement_sha256"]) == 64
    assert timeline["boundaries"][0]["offset_frames"] > 0


def test_the_first_row_needs_no_boundary_and_an_empty_timeline_is_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    assert rows[0]["boundary"] is None
    assert compile_rows(rows)["boundaries"][0]["after_utterance_id"] == "u1"

    leading_boundary = copy.deepcopy(rows)
    leading_boundary[0]["boundary"] = classification("D_TAG")
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(leading_boundary)
    assert refused.value.code == "direction_conflict"

    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows([])

    missing_boundary = copy.deepcopy(rows)
    missing_boundary[1]["boundary"] = None
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows(missing_boundary)

    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.compile_timeline(rows, policy=boundaries.default_policy(), hints={},
                                    allow_lead_offset="yes")


def test_grid_and_digest_helpers_are_deterministic() -> None:
    assert boundaries.sample_domain(24000) == "pcm-mono-24000"
    assert boundaries.frames_for_ms(300.0, 24000) == 7200
    assert boundaries.frames_for_ms(0.0, 24000) == 0
    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.frames_for_ms(float("nan"), 24000)
    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.frames_for_ms(-1.0, 24000)
    assert hashlib.sha256(b"x").hexdigest() != boundaries.sample_domain(16000)


# ------------------------------------------------------------------ repair regressions
# Each of these fails against the original M2 module bytes.


def test_a_pre_cropped_source_range_is_refused() -> None:
    """M2's contract is whole original Takes; a crop is not a timeline input."""

    record = measurement(trailing_content=ms(20.0), leading_content=ms(10.0))
    whole = row("whole", record)
    assert compile_rows([whole])["total_frames"] == record["frames"]

    for start, end in ((240, 2400), (0, 2400), (240, record["frames"])):
        cropped = row("cropped", record, start=start, end=end)
        with pytest.raises(boundaries.PerformanceDirectorError) as refused:
            compile_rows([cropped])
        assert refused.value.code == "input_ineligible"
        assert "whole measured Take" in str(refused.value)


def test_a_cropped_take_cannot_shorten_the_plan() -> None:
    # Before the repair this produced a 2160-frame plan instead of 4800.
    record = measurement(trailing_content=ms(20.0))
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows([row("cropped", record, start=240, end=2400)])
    assert compile_rows([row("whole", record)])["total_frames"] == record["frames"]


def test_an_offset_cannot_drive_a_placement_negative() -> None:
    # A short selected range paired with a long leading-silence measurement used
    # to yield derived_frames = -10800.  The crop is now refused outright, and
    # the derived length is validated independently.
    short = measurement(trailing_content=ms(10.0))
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows([row("tiny", short, classification("N_D_SAME_PARA"), end=240)],
                     allow_lead_offset=True)
    with pytest.raises(boundaries.PerformanceDirectorError):
        compile_rows([row("tiny", short, classification("N_D_SAME_PARA"), end=240)])


def test_measurement_edge_bounds_are_validated_against_the_frame_count() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))

    silent_beyond_leading = copy.deepcopy(rows)
    silent_beyond_leading[1]["measurement"]["leading"]["silent_frames"] = ms(999.0)
    silent_beyond_leading[1]["measurement"]["content_sha256"] = boundaries.content_sha256(
        silent_beyond_leading[1]["measurement"])
    silent_beyond_leading[1]["measurement_sha256"] = \
        silent_beyond_leading[1]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(silent_beyond_leading)
    assert "exceeds its own" in str(refused.value)

    edge_beyond_take = copy.deepcopy(rows)
    edge_beyond_take[0]["measurement"]["trailing"]["content_frames"] = TAKE_FRAMES + 1
    edge_beyond_take[0]["measurement"]["content_sha256"] = boundaries.content_sha256(
        edge_beyond_take[0]["measurement"])
    edge_beyond_take[0]["measurement_sha256"] = edge_beyond_take[0]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(edge_beyond_take)
    assert "exceeds the Take length" in str(refused.value)

    edges_overflow = copy.deepcopy(rows)
    for item in edges_overflow:
        item["measurement"]["trailing"]["content_frames"] = TAKE_FRAMES - ms(200.0)
        item["measurement"]["leading"]["content_frames"] = TAKE_FRAMES - ms(100.0)
    edges_overflow[0]["measurement"]["content_sha256"] = boundaries.content_sha256(
        edges_overflow[0]["measurement"])
    edges_overflow[0]["measurement_sha256"] = edges_overflow[0]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(edges_overflow)
    assert "inconsistent" in str(refused.value)


def test_an_empty_measurement_is_refused() -> None:
    rows = pair("N_D_SAME_PARA", trailing=ms(20.0), leading=ms(10.0))
    empty = copy.deepcopy(rows)
    empty[0]["measurement"]["frames"] = 0
    empty[0]["measurement"]["content_sha256"] = boundaries.content_sha256(empty[0]["measurement"])
    empty[0]["measurement_sha256"] = empty[0]["measurement"]["content_sha256"]
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows(empty)
    assert "empty Take" in str(refused.value)


def test_unknown_preserves_explicit_baseline_spacing() -> None:
    """The documented ``baseline_spacer_frames`` row field."""

    previous = measurement(trailing_content=ms(100.0))
    following = measurement(leading_content=ms(50.0))
    baseline_spacer = ms(150.0)

    unknown = row("u2", following, classification("UNKNOWN"))
    unknown["baseline_spacer_frames"] = baseline_spacer
    timeline = compile_rows([row("u1", previous), unknown])
    assert timeline["total_frames"] == previous["frames"] + following["frames"] + baseline_spacer
    boundary = timeline["boundaries"][0]
    assert boundary["status"] == "baseline"
    assert boundary["spacer_frames"] == baseline_spacer
    assert boundary["target_frames"] is None
    assert boundary["offset_frames"] == 0
    assert boundary["effective_frames"] == ms(150.0) + baseline_spacer

    placed = [item for item in timeline["placements"] if item["kind"] == "spacer"]
    assert [item["frames"] for item in placed] == [baseline_spacer]

    # The field is part of the dependency identity.
    without = compile_rows([row("u1", previous), row("u2", following, classification("UNKNOWN"))])
    assert without["timeline_sha256"] != timeline["timeline_sha256"]
    assert without["total_frames"] == previous["frames"] + following["frames"]
    assert without["boundaries"][0]["spacer_frames"] == 0

    # Offsets are never applied on a baseline boundary, even when permitted.
    permitted = compile_rows([row("u1", previous), unknown], allow_lead_offset=True)
    assert permitted["boundaries"][0]["offset_frames"] == 0
    assert permitted["boundaries"][0]["spacer_frames"] == baseline_spacer


def test_baseline_spacing_is_validated() -> None:
    previous = measurement(trailing_content=ms(100.0))
    following = measurement(leading_content=ms(50.0))

    def rows_with(value):
        item = row("u2", following, classification("UNKNOWN"))
        item["baseline_spacer_frames"] = value
        return [row("u1", previous), item]

    assert compile_rows(rows_with(0))["boundaries"][0]["spacer_frames"] == 0
    for bad in (-1, 1.5, "150", True, None):
        if bad is None:
            continue
        with pytest.raises(boundaries.PerformanceDirectorError) as refused:
            compile_rows(rows_with(bad))
        assert refused.value.code == "input_ineligible"

    # The field belongs to a boundary, so the first row cannot carry it.
    first = row("u1", previous)
    first["baseline_spacer_frames"] = 3600
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows([first, row("u2", following, classification("UNKNOWN"))])
    assert refused.value.code == "direction_conflict"


def test_a_policy_omitted_target_uses_explicit_baseline_spacing() -> None:
    """A class omitted by design is baseline; a misspelled name is still refused."""

    previous = measurement(trailing_content=ms(100.0))
    following = measurement(leading_content=ms(50.0))
    known = classification("D_D_REPLY")
    following_row = row("u2", following, known)
    following_row["baseline_spacer_frames"] = ms(150.0)

    omitted = boundaries.default_policy()
    del omitted["targets_ms"]["D_D_REPLY"]
    timeline = compile_rows([row("u1", previous), following_row], policy=omitted)
    assert timeline["boundaries"][0]["status"] == "baseline"
    assert timeline["boundaries"][0]["spacer_frames"] == ms(150.0)
    assert timeline["total_frames"] == previous["frames"] + following["frames"] + ms(150.0)

    misspelled = boundaries.default_policy()
    misspelled["targets_ms"]["D_D_REPLYE"] = 10.0
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        compile_rows([row("u1", previous), following_row], policy=misspelled)
    assert refused.value.code == "schema_unknown"


# ------------------------------------------------------- final follow-up regressions
# Both fail against the first-repair module bytes.


def test_status_is_derived_from_achieved_arithmetic_not_from_adjustment() -> None:
    """``met`` means the effective gap EQUALS the target, nothing weaker."""

    policy = boundaries.default_policy()
    policy["targets_ms"]["N_D_SAME_PARA"] = 250.0
    guard = ms(boundaries.LEAD_OFFSET_GUARD_MS)
    # A whole-Take pair: 600 ms trailing edge and 500 ms lead-in.
    rows = pair("N_D_SAME_PARA", trailing=ms(600.0), leading=ms(500.0), silent=ms(500.0))

    disabled = compile_rows(rows, policy=policy, allow_lead_offset=False)["boundaries"][0]
    assert disabled["existing_frames"] == ms(1100.0)
    assert disabled["effective_frames"] == ms(1100.0) > disabled["target_frames"] == ms(250.0)
    assert disabled["status"] == "limited_by_existing_audio"

    # With the 40 ms guard the offset can only remove 460 ms, leaving 640 ms
    # against a 250 ms target.  The first-repair module called this "met".
    adjusted = compile_rows(rows, policy=policy, allow_lead_offset=True)["boundaries"][0]
    assert adjusted["eligible_offset_frames"] == ms(500.0) - guard
    assert adjusted["offset_frames"] == ms(500.0) - guard
    assert adjusted["effective_frames"] == ms(640.0)
    assert adjusted["effective_frames"] > adjusted["target_frames"]
    assert adjusted["status"] == "limited_by_existing_audio"


def test_an_exact_gap_is_met_without_any_adjustment() -> None:
    """The other side of the equality: no spacer and no offset, but still met."""

    policy = boundaries.default_policy()
    policy["targets_ms"]["N_D_SAME_PARA"] = 1100.0
    rows = pair("N_D_SAME_PARA", trailing=ms(600.0), leading=ms(500.0), silent=ms(500.0))
    boundary = compile_rows(rows, policy=policy, allow_lead_offset=False)["boundaries"][0]
    assert boundary["existing_frames"] == boundary["target_frames"] == ms(1100.0)
    assert boundary["spacer_frames"] == 0
    assert boundary["offset_frames"] == 0
    assert boundary["effective_frames"] == boundary["target_frames"]
    # The first-repair module reported limited_by_existing_audio here.
    assert boundary["status"] == "met"
    assert spacers(compile_rows(rows, policy=policy, allow_lead_offset=False)) == []


@pytest.mark.parametrize("klass", ["N_D_SAME_PARA", "D_D_REPLY", "N_N_SAME_PARA"])
def test_status_equality_holds_for_every_experimental_family(klass: str) -> None:
    """One frame above the target is limited; exactly the target is met."""

    policy = boundaries.default_policy()
    policy["targets_ms"][klass] = 400.0
    short = ms(300.0)
    exact_rows = pair(klass, trailing=short, leading=ms(100.0), silent=ms(100.0))
    exact = compile_rows(exact_rows, policy=policy)["boundaries"][0]
    assert exact["existing_frames"] == exact["target_frames"] == ms(400.0)
    assert exact["status"] == "met"

    one_frame_over = pair(klass, trailing=short + 1, leading=ms(100.0), silent=ms(100.0))
    over = compile_rows(one_frame_over, policy=policy)["boundaries"][0]
    assert over["existing_frames"] == ms(400.0) + 1 > over["target_frames"]
    assert over["effective_frames"] == over["existing_frames"]
    assert over["status"] == "limited_by_existing_audio"

    one_frame_under = pair(klass, trailing=short - 1, leading=ms(100.0), silent=ms(100.0))
    under = compile_rows(one_frame_under, policy=policy)["boundaries"][0]
    assert under["spacer_frames"] == 1
    assert under["effective_frames"] == under["target_frames"]
    assert under["status"] == "met"


def test_a_spacer_boundary_reports_met_only_because_it_reached_the_target() -> None:
    policy = boundaries.default_policy()
    policy["targets_ms"]["D_D_REPLY"] = 400.0
    rows = pair("D_D_REPLY", trailing=ms(50.0), leading=ms(40.0), silent=ms(40.0))
    timeline = compile_rows(rows, policy=policy)
    boundary = timeline["boundaries"][0]
    assert boundary["existing_frames"] == ms(90.0)
    assert boundary["spacer_frames"] == ms(310.0)
    assert boundary["effective_frames"] == boundary["target_frames"] == ms(400.0)
    assert boundary["status"] == "met"
    # A displayed spacer placement never claims more than the boundary reached.
    placed = spacers(timeline)
    assert len(placed) == 1
    assert placed[0]["boundary"]["status"] == boundary["status"]
    assert placed[0]["frames"] == boundary["spacer_frames"]


def test_uncertainty_still_wins_over_the_equality() -> None:
    policy = boundaries.default_policy()
    policy["targets_ms"]["D_D_REPLY"] = 200.0
    rows = pair("D_D_REPLY", trailing=ms(500.0), leading=ms(400.0), silent=ms(400.0),
                b_flags=(boundaries.EDGE_UNCERTAIN,))
    boundary = compile_rows(rows, policy=policy, allow_lead_offset=True)["boundaries"][0]
    # effective (900 ms) is not the target, but uncertainty is the reported cause.
    assert boundary["effective_frames"] > boundary["target_frames"]
    assert boundary["status"] == "protected_uncertain_edge"
    assert boundary["offset_frames"] == 0
