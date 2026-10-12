"""Pilot allocation and the durable budget ledger (M3, section P11).

Unit A allocates one isolated review-only pilot; unit B keeps the durable
pre-dispatch reservation that stops a second caller from spending the same
budget.  Everything runs against disposable synthetic projects under the
per-run pytest root, and the shared builders come from the assembly suite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.process_lock import acquire_projects_root
from dubbing_service.app.storage.projects import load_project, save_project

from tests.test_performance_pilot_assembly import (  # noqa: F401
    make_source_project,
    scope_manifest,
    studio,
)


# --------------------------------------------------------------------------- unit A


def test_create_pilot_allocates_an_isolated_review_only_project(studio) -> None:
    project = load_project(studio["pilot_id"])
    assert project.voice_lab.scope == "isolated_listening_test"
    index = project.metadata[pilot.PILOT_INDEX_KEY]
    assert index["kind"] == "review_only"
    assert index["human_review"] == "pending"
    assert index["production_enablement"] is False
    assert index["state"] == "approved_scope"
    assert project.metadata[pilot.PILOT_LEDGER_KEY]["schema_version"] == pilot.LEDGER_SCHEMA


def test_create_pilot_is_idempotent_and_a_changed_payload_conflicts(studio) -> None:
    same = pilot.create_pilot("pilot", studio["manifest"])
    assert same["pilot_id"] == studio["pilot_id"]
    # A changed canonical payload under the same operation identity conflicts
    # rather than allocating a second pilot.
    changed = dict(studio["manifest"])
    changed["scope"] = {**studio["manifest"]["scope"], "arms": ["baseline"]}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("pilot", changed)
    assert refused.value.code == "direction_conflict"
    assert len([item for item in Path(studio["root"]).iterdir() if item.is_dir()]) == 2


def test_a_different_operation_identity_allocates_separately(studio) -> None:
    other = dict(studio["manifest"])
    other["operation_id"] = "op-0002"
    created = pilot.create_pilot("second", other)
    assert created["pilot_id"] != studio["pilot_id"]
    assert created["state"] == "approved_scope"


def test_create_pilot_refuses_a_foreign_destination_root(studio, tmp_path) -> None:
    manifest = dict(studio["manifest"])
    manifest["authorized_destination_root"] = str(tmp_path / "elsewhere")
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("pilot", manifest)
    assert refused.value.code == "input_ineligible"


def test_create_pilot_refuses_a_live_grant_without_authentication(studio) -> None:
    manifest = dict(studio["manifest"])
    manifest["operation_id"] = "op-live"
    manifest["authorization"] = {"kind": "live_pilot", "by": "Brian", "event_id": "evt-live"}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("pilot", manifest)
    assert refused.value.code == "review_pending"
    # A caller string naming the owner is not authentication.
    manifest["authorization"] = {"kind": "live_pilot", "by": "Brian",
                                 "authentication": {"kind": "live_owner", "by": "Brian"}}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("pilot", manifest)
    assert refused.value.code == "review_pending"


def test_create_pilot_validates_scope_before_any_write(studio) -> None:
    for broken in (
        {**studio["manifest"], "schema_version": "performance-pilot-scope-v2"},
        {**studio["manifest"], "kind": "production"},
        {**studio["manifest"], "scope": {**studio["manifest"]["scope"], "arms": ["gamma"]}},
        {**studio["manifest"], "ceilings": {**studio["manifest"]["ceilings"],
                                            "deadline": "not-a-date"}},
    ):
        with pytest.raises(pilot.PerformanceDirectorError):
            pilot.create_pilot("pilot", broken)


# --------------------------------------------------------------------------- unit B


def test_duplicate_operations_return_the_receipt_and_changed_payloads_conflict(studio) -> None:
    first = pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=2)
    assert first["reused"] is False
    again = pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=2)
    assert again["reused"] is True
    assert again["reservation"]["operation_id"] == "op-a"
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=3)
    assert refused.value.code == "direction_conflict"


def test_reservation_is_durable_and_survives_a_restart(studio) -> None:
    pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=2)
    # A fresh read from disk, as a restarted process would see it.
    state = pilot.read_pilot(studio["pilot_id"])
    assert state["ledger"]["operations"]["op-a"]["state"] == "reserved"
    assert state["ledger"]["spend"]["assemblies"] == 2
    assert state["index"]["state"] == "reserved"
    assert state["ledger"]["ceilings"]["deadline"] == "2026-12-31T23:59:59+00:00"


def test_the_assembly_ceiling_is_enforced_and_never_released(studio) -> None:
    pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=3)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.reserve_operation(studio["pilot_id"], "op-b", assemblies=2)
    assert refused.value.code == "budget_exhausted"
    # A failure retains its spent reservation.
    pilot.complete_operation(studio["pilot_id"], "op-a", outcome="failed")
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.reserve_operation(studio["pilot_id"], "op-b", assemblies=2)


def test_unknown_outcomes_are_recorded_and_never_retried_automatically(studio) -> None:
    pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=1)
    pilot.complete_operation(studio["pilot_id"], "op-a", outcome="unknown_outcome")
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assert_operation_open(studio["pilot_id"], "op-a")
    assert refused.value.code == "plan_stale"
    pilot.complete_operation(studio["pilot_id"], "op-a", outcome="unknown_outcome")
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.complete_operation(studio["pilot_id"], "op-a", outcome="completed")


def test_a_competing_writer_owner_is_refused_across_processes(studio) -> None:
    """The reservation is refused while another process owns the root lock.

    An in-process lock is re-entrant, so only a second *process* can show that
    the durable boundary is real.  A short-lived child holds the projects-root
    lock and signals readiness on stdout; the reservation here must then fail
    with the owner's busy error rather than mutating any state.
    """

    import subprocess
    import sys
    import time

    from dubbing_service.app.storage.process_lock import LockBusyError

    holder = (
        "import sys, time;"
        "from pathlib import Path;"
        "from dubbing_service.app.storage.process_lock import acquire_projects_root;"
        "lock = acquire_projects_root(Path(sys.argv[1]), label='competing-owner');"
        "print('ready', flush=True);"
        "time.sleep(20);"
        "lock.release()"
    )
    child = subprocess.Popen([sys.executable, "-B", "-c", holder, str(studio["root"])],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        before = pilot.read_pilot(studio["pilot_id"])
        with pytest.raises((LockBusyError, pilot.PerformanceDirectorError)):
            pilot.reserve_operation(studio["pilot_id"], "op-blocked", assemblies=1)
        after = pilot.read_pilot(studio["pilot_id"])
        assert after == before
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            child.kill()
        time.sleep(0.2)


def test_a_denied_reservation_write_is_reported(studio, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise OSError("synthetic denied write")

    monkeypatch.setattr(pilot, "save_project", refuse)
    with pytest.raises(OSError):
        pilot.reserve_operation(studio["pilot_id"], "op-a", assemblies=1)
    monkeypatch.undo()
    # The reservation was never half-recorded.
    state = pilot.read_pilot(studio["pilot_id"])
    assert "op-a" not in (state["ledger"].get("operations") or {})




# --------------------------------------------------- repair regressions (M3 review)

def _scope(studio, **changes) -> dict:
    manifest = {**studio["manifest"], **changes}
    return manifest


def test_an_expired_original_deadline_is_refused_at_allocation(studio) -> None:
    """A deadline in the past never allocates a pilot."""

    expired = _scope(studio, operation_id="op-expired",
                     ceilings={**studio["manifest"]["ceilings"],
                               "deadline": "2000-01-01T00:00:00+00:00"})
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("expired", expired)
    assert refused.value.code == "budget_exhausted"


def test_a_caller_deadline_cannot_extend_the_operation_deadline(studio) -> None:
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=1)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.reserve_operation(studio["pilot_id"], "op-later", assemblies=1,
                                deadline="2031-01-01T00:00:00+00:00")
    assert refused.value.code == "budget_exhausted"
    shorter = pilot.reserve_operation(studio["pilot_id"], "op-sooner", assemblies=1,
                                      deadline="2026-11-01T00:00:00+00:00")
    assert shorter["reservation"]["outcome"] is None


def test_a_changed_expected_revision_conflicts_with_its_allocation(studio) -> None:
    changed = _scope(studio, expected_revision_sha256="a" * 64)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.create_pilot("pilot", changed)


def test_one_reservation_funds_only_the_assemblies_it_reserved(studio) -> None:
    """A second arm cannot be funded by a one-assembly reservation."""

    from tests.test_performance_pilot_assembly import build_inputs

    project_id = studio["project"].id
    studio = {**studio, "project": load_project(project_id),
              "segments": load_segments(project_id)}
    inputs = build_inputs(studio, klass="D_D_REPLY")
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=1)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"], allow_lead_offset=False)
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="variant")
    assert refused.value.code == "budget_exhausted"


def test_canonical_inputs_are_immutable_once_an_operation_completed(studio) -> None:
    from tests.test_performance_pilot_assembly import build_inputs

    project_id = studio["project"].id
    studio = {**studio, "project": load_project(project_id),
              "segments": load_segments(project_id)}
    inputs = build_inputs(studio, klass="D_D_REPLY")
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=1)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"], allow_lead_offset=False)
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    # Storing the identical inputs again is harmless.
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    changed = {**inputs, "rows": [{**inputs["rows"][0], "source_start_frame": 1},
                                  inputs["rows"][1]]}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.store_timeline_inputs(studio["pilot_id"], changed)
    assert refused.value.code == "plan_stale"


def test_a_cancelled_operation_keeps_its_budget(studio) -> None:
    pilot.reserve_operation(studio["pilot_id"], "op-cancel", assemblies=3)
    pilot.complete_operation(studio["pilot_id"], "op-cancel", outcome="cancelled")
    state = pilot.read_pilot(studio["pilot_id"])
    assert state["ledger"]["spend"]["assemblies"] == 3
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.reserve_operation(studio["pilot_id"], "op-next", assemblies=2)
    assert refused.value.code == "budget_exhausted"
    receipt = state["ledger"]["operations"]["op-cancel"]
    assert receipt["outcome"] == "cancelled"
    assert receipt["spent"]["assemblies"] == 3

def test_a_deadline_that_moves_into_the_past_before_publication_holds(studio, monkeypatch) -> None:
    """The deadline is re-read after encoding, immediately before publishing."""

    from tests.test_performance_pilot_assembly import build_inputs

    project_id = studio["project"].id
    studio = {**studio, "project": load_project(project_id),
              "segments": load_segments(project_id)}
    inputs = build_inputs(studio, klass="D_D_REPLY")
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=1)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"], allow_lead_offset=False)

    original = pilot._encode_mp3

    def expire_then_encode(master, staging):
        data = original(master, staging)
        project = load_project(studio["pilot_id"])
        ledger = dict(project.metadata[pilot.PILOT_LEDGER_KEY])
        ledger["ceilings"] = {**ledger["ceilings"], "deadline": "2000-01-01T00:00:00+00:00"}
        project.metadata[pilot.PILOT_LEDGER_KEY] = ledger
        save_project(project)
        return data

    monkeypatch.setattr(pilot, "_encode_mp3", expire_then_encode)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert refused.value.code == "plan_stale"
    directory = (pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots"
                 / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"])
    assert not (directory / "receipt.json").exists()
