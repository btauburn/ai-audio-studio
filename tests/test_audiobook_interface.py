"""Phase 16: the narrow agent capability boundary.

These tests pin the contract a future agent is allowed to rely on: a repeated
request never starts duplicate GPU work, status is answerable after the caller
has lost its connection, a stale or QA-blocked Take can never become selected,
and rejection reuses the existing Take fields instead of inventing a parallel
ledger.
"""

from __future__ import annotations

import json

import pytest

from dubbing_service.app.audiobook import interface, pipeline, renderer, takes
from dubbing_service.app.storage.jobs import load_jobs
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="interface")
    segments = build_segments(project, ["First utterance.", "Second utterance."])
    return root, project, segments


def _render(project, segments, segment, *, promote: bool = True):
    """Render one Take and persist it: the interface always reads from disk."""

    record = pipeline.render_segment(
        project,
        segment,
        renderer=renderer.FixtureRenderer(
            project_root=pipeline.project_root(project.id), renderer_id="fixture"
        ),
        root=pipeline.project_root(project.id),
        promote=promote,
    )
    assert record.success is True, record.error
    save_segments(project.id, segments)
    return takes.find_take(segment, record.take_id)


def _request(project, segments, **overrides):
    payload = {
        "segment_ids": [segment.id for segment in segments],
        "idempotency_key": "agent-key-1",
    }
    payload.update(overrides)
    return interface.create_request(project.id, **payload)


def test_a_repeated_request_reuses_the_job_instead_of_duplicating_work(studio):
    _root, project, segments = studio

    first = _request(project, segments)
    second = _request(project, segments)

    assert first.detail["created"] is True
    assert second.detail["created"] is False
    assert second.detail["job"]["id"] == first.detail["job"]["id"]
    assert len(load_jobs(project.id)) == 1


def test_the_same_key_with_a_different_intent_is_refused(studio):
    _root, project, segments = studio
    _request(project, segments)

    with pytest.raises(interface.InterfaceError):
        _request(project, segments[:1])

    assert len(load_jobs(project.id)) == 1


def test_request_identity_survives_a_new_process_view(studio):
    _root, project, segments = studio
    created = _request(project, segments, request_id="req_fixed")

    stored = json.loads(
        (pipeline.project_root(project.id) / "jobs.json").read_text(encoding="utf-8")
    )
    status = interface.get_status(project.id, request_id="req_fixed")

    assert [job["id"] for job in stored] == [created.detail["job"]["id"]]
    assert stored[0]["request_id"] == "req_fixed"
    assert status.detail["job"]["id"] == created.detail["job"]["id"]


def test_a_request_with_no_utterances_is_refused(studio):
    _root, project, _segments = studio

    with pytest.raises(interface.InterfaceError):
        interface.create_request(project.id, segment_ids=[])


def test_a_request_for_unknown_utterances_is_refused(studio):
    _root, project, _segments = studio

    with pytest.raises(interface.InterfaceError):
        interface.create_request(project.id, segment_ids=["seg_missing"])


def test_duplicate_utterance_ids_collapse_and_backends_are_filtered(studio):
    _root, project, segments = studio
    result = _request(
        project,
        segments,
        segment_ids=[segments[0].id, segments[0].id, segments[1].id],
        segment_backends={segments[0].id: "indextts", "seg_other": "indextts"},
    )

    job = result.detail["job"]
    assert job["segment_ids"] == [segments[0].id, segments[1].id]
    assert job["segment_backends"] == {segments[0].id: "indextts"}


def test_status_answers_by_request_id_after_the_caller_timed_out(studio):
    _root, project, segments = studio
    _request(project, segments, request_id="req_timeout")

    status = interface.get_status(project.id, request_id="req_timeout")

    assert status.ok is True
    assert status.detail["job"]["status"] == "queued"
    project_report = status.detail["project"]
    assert project_report["coverage"]["ready"] is False
    assert project_report["coverage"]["total_live"] == 2


def test_status_refuses_a_request_that_never_existed(studio):
    _root, project, _segments = studio

    with pytest.raises(interface.InterfaceError):
        interface.get_status(project.id, request_id="req_never")


def test_cancel_is_operational_and_does_not_delete_the_request(studio):
    _root, project, segments = studio
    created = _request(project, segments)
    job_id = created.detail["job"]["id"]
    interface.start(project.id, job_id)

    cancelled = interface.cancel(project.id, job_id)

    assert cancelled.detail["job"]["status"] == "cancelled"
    assert cancelled.detail["job"]["stop_requested"] is True
    assert len(load_jobs(project.id)) == 1
    assert interface.get_status(project.id, request_id=created.detail["request_id"]).ok


def test_inspect_failure_reports_the_failing_utterances_and_qa_findings(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0])
    take.qa_report.passed = False
    take.qa_report.results[0].blocking = True
    take.qa_report.results[0].status = "failed"
    assert take.qa_report.blocking_failures()
    save_segments(project.id, segments)
    created = _request(project, segments[:1])

    detail = interface.inspect_failure(
        project.id, request_id=created.detail["request_id"]
    ).detail

    assert detail["job"]["id"] == created.detail["job"]["id"]
    assert [item["segment_id"] for item in detail["qa_failures"]] == [segments[0].id]
    assert detail["qa_failures"][0]["take_id"] == take.id
    assert detail["qa_failures"][0]["blocking"]


def test_targeted_repair_plans_only_the_changed_utterances(studio):
    _root, project, segments = studio
    for segment in segments:
        _render(project, segments, segment)

    plan = interface.request_targeted_repair(
        project.id,
        kind="spoken_text",
        segment_ids=[segments[0].id],
        reason="typo fix",
    ).detail

    assert plan["kind"] == "spoken_text"
    assert plan["utterance_ids"] == [segments[0].id]
    assert plan["requires_render"] is True
    assert plan["reason"] == "typo fix"


def test_targeted_repair_refuses_an_unknown_change_kind(studio):
    _root, project, _segments = studio

    with pytest.raises(interface.InterfaceError):
        interface.request_targeted_repair(project.id, kind="teleport")


def test_select_take_persists_the_choice(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0], promote=False)

    interface.select_take(project.id, segment_id=segments[0].id, take_id=take.id)

    reloaded = next(
        item for item in load_segments(project.id) if item.id == segments[0].id
    )
    assert reloaded.selected_take_id == take.id
    assert takes.selected_take(reloaded).id == take.id


def test_select_take_refuses_a_stale_take(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0], promote=False)
    segments[0].text = "Rewritten after the Take was produced."
    pipeline.refresh_staleness(project, segments)
    save_segments(project.id, segments)

    with pytest.raises(interface.InterfaceError) as excinfo:
        interface.select_take(project.id, segment_id=segments[0].id, take_id=take.id)

    assert take.id in str(excinfo.value)


def test_select_take_refuses_a_take_with_blocking_qa_failures(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0])
    take.qa_report.results[0].blocking = True
    take.qa_report.results[0].status = "failed"
    take.qa_report.passed = False
    save_segments(project.id, segments)

    with pytest.raises(interface.InterfaceError):
        interface.select_take(project.id, segment_id=segments[0].id, take_id=take.id)


def test_reject_take_archives_it_and_detaches_every_slot(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0])
    assert segments[0].selected_take_id == take.id

    interface.reject_take(
        project.id, segment_id=segments[0].id, take_id=take.id, reason="mispronounced"
    )

    reloaded = next(
        item for item in load_segments(project.id) if item.id == segments[0].id
    )
    archived = takes.find_take(reloaded, take.id)
    assert archived.archived_at is not None
    assert archived.generation_metadata["rejected_reason"] == "mispronounced"
    assert reloaded.selected_take_id is None
    assert all(state.take_id != take.id for state in reloaded.take_slots.values())
    assert takes.selected_take(reloaded) is None


def test_rejection_does_not_delete_audio_or_create_a_second_ledger(studio):
    _root, project, segments = studio
    take = _render(project, segments, segments[0])
    audio = pipeline.project_root(project.id) / take.audio_file

    interface.reject_take(project.id, segment_id=segments[0].id, take_id=take.id)

    assert audio.is_file()
    assert len(load_segments(project.id)[0].takes) == 1


def test_export_reports_readiness_without_writing_anything(studio):
    _root, project, segments = studio
    before = interface.list_artifacts(project.id).detail["artifacts"]
    blocked = interface.export(project.id)

    for segment in segments:
        _render(project, segments, segment)

    ready = interface.export(project.id)
    after = interface.list_artifacts(project.id).detail["artifacts"]

    assert blocked.detail["ready"] is False
    assert ready.detail["ready"] is True
    assert ready.detail["coverage"]["covered"] == 2
    assert {item["path"] for item in before} <= {item["path"] for item in after}


def test_list_artifacts_stays_inside_the_project_root(studio):
    _root, project, segments = studio
    _render(project, segments, segments[0])

    artifacts = interface.list_artifacts(project.id).detail["artifacts"]

    paths = {item["path"] for item in artifacts}
    assert "project.json" in paths and "segments.json" in paths
    assert any(path.startswith("audio/") for path in paths)
    assert all(".." not in path and not path.startswith("/") for path in paths)
    assert all(item["bytes"] >= 0 for item in artifacts)


def test_the_agent_surface_never_forks_a_canonical_store(studio):
    _root, project, segments = studio
    _render(project, segments, segments[0])
    _request(project, segments)
    interface.export(project.id)

    root = pipeline.project_root(project.id)
    names = sorted(path.name for path in root.iterdir())
    forbidden = [name for name in names if name.endswith((".db", ".sqlite", ".sqlite3"))]

    assert forbidden == []
    assert "project.json" in names
    assert "segments.json" in names
