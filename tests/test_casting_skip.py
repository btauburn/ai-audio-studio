"""Skipping lines, and deciding unattributed ones automatically, through the real maintenance loop."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import autocast_book as book
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook.fingerprint import voice_revision_for_role
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Chapter, Role, Segment, SegmentTake
from dubbing_service.app.storage.jobs import get_job, mark_job_active
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.routers.voice_lab import router as voice_lab_router
from tests.audiobook_helpers import use_projects_root

# id, chapter, role, analyzer label, held for review
LINES = [("l1", "ch1", "mara", None, False), ("l2", "ch1", "narrator", None, False),
         ("l3", "ch1", None, "Joss", True), ("l4", "ch2", None, "unknown", True),
         ("l5", "ch2", None, None, False), ("l6", "ch2", "mara", None, False)]


@pytest.fixture
def blocked_book(monkeypatch, tmp_path):
    """An Auto Cast book whose cast is approved but which has three lines with no speaker."""
    use_projects_root(monkeypatch, tmp_path)
    monkeypatch.setattr(casting, "notify_ready", lambda project_id: {"state": "stubbed"})
    project = create_project("Skipping book")
    project.voice_lab.scope = "regression_fixture"
    project.roles = [Role(id="narrator", name="Narrator", role_type=RoleType.NARRATOR, tts_backend="qwen3tts"),
                     Role(id="mara", name="Mara", role_type=RoleType.CHARACTER, tts_backend="qwen3tts"),
                     Role(id="joss", name="Joss", role_type=RoleType.CHARACTER, tts_backend="qwen3tts")]
    project.chapters = [Chapter(id="ch1", title="The Gate", index=0), Chapter(id="ch2", title="The Road", index=1)]
    state = casting._state(project)
    for role in project.roles:
        state["roles"][role.id] = {"assignment_revision": 1, "round_revision": 1, "history": [], "rejections": [],
            "round": {"status": "resolved"},
            "assignment": {"key": f"key-{role.id}", "voice_revision": voice_revision_for_role(role), "approved": True,
                           "rejected": False, "event_id": f"evt-{role.id}", "stars": 5, "revision": 1}}
        state["events"][f"evt-{role.id}"] = {"actor": "brian", "origin": "SYNTHETIC", "stars": 5}
    state["intents"]["seed"] = {"status": "complete"}
    project.metadata["autocast_import"] = {"mode": "autocast", "unresolved_utterance_ids": ["l3", "l4", "l5"]}
    save_project(project)
    save_segments(project.id, [
        Segment(id=uid, index=index, chapter_id=chapter, text=f"{uid} is spoken here.", role_id=role,
                attribution=Attribution(label=label, source="nlp") if label else None)
        for index, (uid, chapter, role, label, _) in enumerate(LINES)])
    return SimpleNamespace(id=project.id)


def tick(book_project):
    started = []
    casting.maintenance_tick(project_ids=[book_project.id], schedule=lambda fn, *args: started.append(fn.__name__))
    return started


def receipt_of(book_project):
    return casting._state(load_project(book_project.id))["productions"].get(casting.AUTOMATIC_PRODUCTION_ID)


def park(book_project, status="awaiting_review"):
    """Stand in for a worker that has stopped: the production is parked, not running."""
    receipt = receipt_of(book_project)
    project = load_project(book_project.id)
    casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(status=status)
    save_project(project)
    mark_job_active(book_project.id, receipt["job_id"], False)


def test_the_book_is_blocked_until_someone_decides(blocked_book):
    assert tick(blocked_book) == []
    receipt = receipt_of(blocked_book)
    assert receipt["status"] == "blocked" and "3 of 6" in receipt["reason"]


def test_a_decision_restarts_the_blocked_book_by_itself(blocked_book):
    tick(blocked_book)
    outcome = book.resolve_unattributed(blocked_book.id, request_id="decide-1", mode="auto")
    assert (outcome["resolved"], outcome["skipped"], outcome["still_unresolved"]) == (1, 2, 0)
    assert receipt_of(blocked_book)["status"] == "awaiting_review", "re-planned, not left blocked or marked source_changed"
    assert tick(blocked_book) == ["_production_worker"], "and the next pass starts recording without a click"
    receipt = receipt_of(blocked_book)
    assert receipt["status"] == "queued"
    assert receipt["manifest"]["utterance_ids"] == ["l1", "l2", "l3", "l6"], "the two skipped lines are outside the frozen scope"
    assert receipt["utterance_ids"] == ["l1", "l2", "l3", "l6"] and receipt["required_roles"] == ["joss", "mara", "narrator"] or \
        sorted(receipt["required_roles"]) == ["joss", "mara", "narrator"]
    assert {row.id: row.role_id for row in load_segments(blocked_book.id)}["l3"] == "joss"
    job = get_job(blocked_book.id, receipt["job_id"])
    assert job.segment_ids == ["l1", "l2", "l3", "l6"], "the job is not measured against lines that will never be recorded"


def test_a_decision_that_leaves_a_line_unresolved_keeps_the_book_blocked(blocked_book):
    tick(blocked_book)
    book.resolve_unattributed(blocked_book.id, request_id="decide-2", mode="suggested")
    assert tick(blocked_book) == []
    assert receipt_of(blocked_book)["status"] == "blocked", "two lines still have no speaker"


def test_skip_leaves_every_unresolved_line_out_and_records_why(blocked_book):
    outcome = book.resolve_unattributed(blocked_book.id, request_id="decide-3", mode="skip")
    assert (outcome["resolved"], outcome["skipped"]) == (0, 3)
    project = load_project(blocked_book.id)
    assert {uid: row["reason"] for uid, row in casting.skipped_lines(project).items()} == {
        "l3": "unknown_speaker", "l4": "unknown_speaker", "l5": "unknown_speaker"}
    assert all(row["by"] == "human" and row["at"] for row in casting.skipped_lines(project).values())
    wanted, _, required, attention = casting._scope(project)
    assert wanted == ["l1", "l2", "l6"] and attention == [] and casting._held_ids(project) == set()


def test_a_skipped_line_is_never_offered_as_something_to_record(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-4", mode="skip")
    project = load_project(blocked_book.id)
    _, rows, _, _ = casting._scope(project)
    assert not {"l3", "l4", "l5"} & {row.id for row in rows}
    assert casting.unresolved_scope(project, rows) == ([], [])


# ---- automatic handling -------------------------------------------------

def test_with_automatic_handling_on_the_book_never_stops_for_these_lines(blocked_book):
    book.set_automatic_handling(blocked_book.id, True)
    assert tick(blocked_book) == ["_production_worker"], "decided and started in the very first pass"
    receipt = receipt_of(blocked_book)
    assert receipt["status"] == "queued" and receipt["manifest"]["utterance_ids"] == ["l1", "l2", "l3", "l6"]
    assert casting._production_event(load_project(blocked_book.id)) is None, "no false 'needs attention' push"
    decisions = load_project(blocked_book.id).metadata["autocast_import"]["attribution_decisions"]
    assert [row["by"] for row in decisions.values()] == ["policy"]
    row = next(row for row in load_segments(blocked_book.id) if row.id == "l3")
    assert row.attribution.human_authoritative and "automatic-handling setting" in row.attribution.note


def test_without_it_the_book_still_waits_for_a_person(blocked_book):
    assert tick(blocked_book) == [] and tick(blocked_book) == []
    assert receipt_of(blocked_book)["status"] == "blocked"
    assert casting.skipped_lines(load_project(blocked_book.id)) == {}


def test_automatic_handling_only_applies_to_auto_cast_books(blocked_book):
    project = load_project(blocked_book.id)
    project.metadata["autocast_import"]["mode"] = "manual"
    save_project(project)
    with pytest.raises(book.BookError, match="Auto Cast"):
        book.set_automatic_handling(blocked_book.id, True)


def test_a_line_the_person_brings_back_is_theirs_to_decide(blocked_book):
    book.set_automatic_handling(blocked_book.id, True)
    tick(blocked_book)
    park(blocked_book)
    outcome = book.restore_lines(blocked_book.id, request_id="restore-1", reason="unknown_speaker")
    assert outcome["restored"] == 2
    project = load_project(blocked_book.id)
    assert set(project.metadata["autocast_import"]["policy_exempt"]) == {"l4", "l5"}
    assert casting.skipped_ids(project) == set(), "l3 was accepted, l4 and l5 are back"
    assert tick(blocked_book) == [], "automatic handling does not skip them again"
    assert receipt_of(blocked_book)["status"] == "blocked"
    assert book.book_status(blocked_book.id)["block"]["unresolved"] == 2, "the person is asked about them"


def test_restoring_reopens_a_finished_book(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-5", mode="auto")
    tick(blocked_book)
    park(blocked_book, status="succeeded")
    book.restore_lines(blocked_book.id, request_id="restore-2", utterance_ids=["l4"])
    receipt = receipt_of(blocked_book)
    assert receipt["status"] == "awaiting_review" and "l4" in receipt["utterance_ids"]
    assert receipt["utterance_ids"] == ["l1", "l2", "l3", "l4", "l6"], "restored in reading order"


# ---- skipping by hand ---------------------------------------------------

def test_a_person_can_skip_a_line_and_the_book_carries_on_without_it(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-6", mode="auto")
    tick(blocked_book)
    park(blocked_book)
    outcome = book.skip_lines(blocked_book.id, request_id="skip-1", utterance_ids=["l6"])
    assert outcome["skipped"] == 1
    assert casting.skipped_lines(load_project(blocked_book.id))["l6"]["reason"] == "human"
    assert receipt_of(blocked_book)["utterance_ids"] == ["l1", "l2", "l3"]
    assert book.skip_lines(blocked_book.id, request_id="skip-1", utterance_ids=["l6"]) == outcome, "idempotent per request"


def test_skipping_refuses_what_would_change_a_chapter_you_can_hear(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-7", mode="auto")
    tick(blocked_book)
    park(blocked_book)
    rows = load_segments(blocked_book.id)
    line = next(row for row in rows if row.id == "l1")
    line.takes = [SegmentTake(id="take-1", audio_file="audio/t1.wav", audio_sha256="c" * 64, text=line.text,
                              role_id="mara", tts_backend="qwen3tts")]
    line.selected_take_id = "take-1"
    save_segments(blocked_book.id, rows)
    with pytest.raises(book.BookError, match="already recorded"):
        book.skip_lines(blocked_book.id, request_id="skip-2", utterance_ids=["l1"])
    with pytest.raises(book.BookError, match="not part of this book"):
        book.skip_lines(blocked_book.id, request_id="skip-3", utterance_ids=["l4"])
    with pytest.raises(book.BookError, match="between one and 100"):
        book.skip_lines(blocked_book.id, request_id="skip-4", utterance_ids=[])


def test_skipping_and_restoring_wait_while_the_book_is_being_recorded(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-8", mode="auto")
    assert tick(blocked_book) == ["_production_worker"]      # queued, and its job is claimed
    with pytest.raises(book.BookError, match="being recorded right now"):
        book.skip_lines(blocked_book.id, request_id="skip-5", utterance_ids=["l6"])
    with pytest.raises(book.BookError, match="being recorded right now"):
        book.restore_lines(blocked_book.id, request_id="restore-3", reason="unknown_speaker")
    park(blocked_book)


def test_restore_needs_to_be_told_what_to_bring_back(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-9", mode="skip")
    with pytest.raises(book.BookError, match="Choose which"):
        book.restore_lines(blocked_book.id, request_id="restore-4")
    with pytest.raises(book.BookError, match="not skipped"):
        book.restore_lines(blocked_book.id, request_id="restore-5", reason="unspeakable")


# ---- attributing speakers from a reading of the story -------------------------

@pytest.fixture
def cast_calls(monkeypatch):
    """A new character gets a casting round; here the round itself is only recorded."""
    calls = []
    monkeypatch.setattr(casting, "_new_round", lambda project, role_id, **kwargs: calls.append(role_id))
    return calls


def test_speakers_are_attributed_as_the_persons_own_decision_and_replace_a_wrong_guess(blocked_book):
    outcome = book.attribute_lines(blocked_book.id, request_id="attr-1", note="Read from the scene",
                                   assignments={"l3": "Joss", "l4": "Mara", "l5": "Mara", "l6": "Joss"})
    assert (outcome["attributed"], outcome["unchanged"], outcome["new_characters"]) == (4, 0, [])
    rows = {row.id: row for row in load_segments(blocked_book.id)}
    assert {uid: rows[uid].role_id for uid in ("l3", "l4", "l5", "l6")} == {"l3": "joss", "l4": "mara", "l5": "mara", "l6": "joss"}, \
        "l6 had been guessed as Mara and is corrected"
    attribution = rows["l4"].attribution
    assert attribution.human_authoritative and attribution.note == "Read from the scene" and attribution.label == "unknown"
    project = load_project(blocked_book.id)
    assert casting._held_ids(project) == set()
    _, _, required, attention = casting._scope(project)
    assert attention == [] and sorted(required) == ["joss", "mara", "narrator"]


def test_attributing_the_same_lines_again_changes_nothing(blocked_book):
    body = {"l3": "Joss", "l4": "Mara"}
    first = book.attribute_lines(blocked_book.id, request_id="attr-2", assignments=body)
    assert book.attribute_lines(blocked_book.id, request_id="attr-2", assignments=body) == first, "idempotent per request"
    again = book.attribute_lines(blocked_book.id, request_id="attr-3", assignments=body)
    assert (again["attributed"], again["unchanged"]) == (0, 2), "a line already spoken by that character is left alone"


def test_narration_is_never_handed_to_a_character(blocked_book):
    with pytest.raises(book.BookError, match="Narration"):
        book.attribute_lines(blocked_book.id, request_id="attr-4", assignments={"l3": "Joss", "l2": "Joss"})
    assert {row.id: row.role_id for row in load_segments(blocked_book.id)}["l3"] is None, "and nothing was half-applied"


@pytest.mark.parametrize("name", ["unknown", "Narrator", "Speaker 1", "Unattributed", "  "])
def test_a_placeholder_is_never_a_character(blocked_book, name):
    with pytest.raises(book.BookError, match="not a character"):
        book.attribute_lines(blocked_book.id, request_id="attr-5", assignments={"l3": name})
    with pytest.raises(book.BookError, match="not a character"):
        book.attribute_lines(blocked_book.id, request_id="attr-6", assignments={"l3": "Joss"}, new_roles=[name])


def test_a_character_must_exist_or_be_added_explicitly(blocked_book, cast_calls):
    with pytest.raises(book.BookError, match="is not a character in this book"):
        book.attribute_lines(blocked_book.id, request_id="attr-7", assignments={"l3": "Tavin"})
    with pytest.raises(book.BookError, match="not in this book"):
        book.attribute_lines(blocked_book.id, request_id="attr-8", assignments={"no-such-line": "Joss"})
    with pytest.raises(book.BookError, match="between one and"):
        book.attribute_lines(blocked_book.id, request_id="attr-9", assignments={})


def test_a_character_the_analyzer_missed_is_added_and_cast(blocked_book, cast_calls):
    outcome = book.attribute_lines(blocked_book.id, request_id="attr-10", new_roles=["Tavin"],
                                   assignments={"l3": "Joss", "l4": "Tavin", "l5": "Tavin"})
    assert outcome["new_characters"] == ["Tavin"]
    project = load_project(blocked_book.id)
    tavin = next(role for role in project.roles if role.name == "Tavin")
    assert tavin.role_type == RoleType.CHARACTER and tavin.tts_backend == "qwen3tts" and tavin.id.startswith("role_")
    assert cast_calls == [tavin.id], "a casting round is opened for the new character, and only for them"
    assert {row.id: row.role_id for row in load_segments(blocked_book.id)}["l4"] == tavin.id
    assert tick(blocked_book) == [], "the book waits for their voice rather than recording with a placeholder"


def test_the_casting_run_is_reopened_to_include_a_new_character(blocked_book, cast_calls):
    from dubbing_service.app.models.character import AutoCastRun
    project = load_project(blocked_book.id)
    project.character_cast.autocast = AutoCastRun(
        key="k", project_id=blocked_book.id, source_revision_id="r", source_sha256="s", pipeline_version="p",
        request_id="autocast_x", status="complete", trigger="source_import", requested_at="2026-09-30T00:00:00+00:00",
        required_role_ids=["narrator", "mara", "joss"])
    save_project(project)
    book.attribute_lines(blocked_book.id, request_id="attr-11", new_roles=["Tavin"], assignments={"l4": "Tavin", "l3": "Joss", "l5": "Mara"})
    run = load_project(blocked_book.id).character_cast.autocast
    assert run.status == "running" and len(run.required_role_ids) == 4, "so progress and pushes describe the whole cast"
    assert "Tavin" in run.reason


def test_a_duplicate_character_is_merged_into_the_one_it_is(blocked_book):
    project = load_project(blocked_book.id)
    project.roles.append(Role(id="mara-b", name="The Elder", role_type=RoleType.CHARACTER, tts_backend="qwen3tts"))
    save_project(project)
    rows = load_segments(blocked_book.id)
    next(row for row in rows if row.id == "l6").role_id = "mara-b"
    save_segments(blocked_book.id, rows)
    outcome = book.attribute_lines(blocked_book.id, request_id="attr-12", aliases={"The Elder": "Mara"},
                                   assignments={"l3": "Joss", "l4": "Mara", "l5": "Mara"})
    assert outcome["merged"] == {"The Elder": "Mara"}
    project = load_project(blocked_book.id)
    assert project.character_cast.role_aliases["mara-b"] == "mara"
    _, _, required, _ = casting._scope(project)
    assert "mara-b" not in required, "one voice for one character"
    with pytest.raises(book.BookError, match="Cannot merge"):
        book.attribute_lines(blocked_book.id, request_id="attr-13", aliases={"Nobody": "Mara"}, assignments={"l3": "Joss"})


def test_attributing_a_skipped_line_brings_it_back_into_the_book(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-a", mode="skip")
    assert casting.skipped_ids(load_project(blocked_book.id)) == {"l3", "l4", "l5"}
    book.attribute_lines(blocked_book.id, request_id="attr-14", assignments={"l3": "Joss", "l4": "Mara"})
    assert casting.skipped_ids(load_project(blocked_book.id)) == {"l5"}, "only the lines that were attributed come back"


def test_speakers_cannot_be_attributed_while_the_book_is_being_recorded(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-b", mode="auto")
    assert tick(blocked_book) == ["_production_worker"]
    with pytest.raises(book.BookError, match="being recorded right now"):
        book.attribute_lines(blocked_book.id, request_id="attr-15", assignments={"l6": "Joss"})
    park(blocked_book)


def test_attribution_restarts_a_blocked_book_by_itself(blocked_book):
    tick(blocked_book)
    assert receipt_of(blocked_book)["status"] == "blocked"
    book.attribute_lines(blocked_book.id, request_id="attr-16", assignments={"l3": "Joss", "l4": "Mara", "l5": "Mara"})
    assert receipt_of(blocked_book)["status"] == "awaiting_review"
    assert tick(blocked_book) == ["_production_worker"]
    assert receipt_of(blocked_book)["manifest"]["utterance_ids"] == ["l1", "l2", "l3", "l4", "l5", "l6"], "every line is recorded, none skipped"
    park(blocked_book)


# ---- the Studio routes ------------------------------------------------------

@pytest.fixture
def api(blocked_book, monkeypatch):
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    return SimpleNamespace(client=TestClient(app), base=f"/api/v1/projects/{blocked_book.id}/voice-lab/casting/production")


def test_the_routes_take_each_decision_and_answer_with_the_book_as_it_now_stands(blocked_book, api):
    decided = api.client.post(api.base + "/resolve", json={"request_id": "api-1", "mode": "auto"})
    assert decided.status_code == 200
    body = decided.json()
    assert (body["decision"]["resolved"], body["decision"]["skipped"]) == (1, 2)
    assert body["book"]["skipped"]["total"] == 2 and body["book"]["automatic"] is False
    on = api.client.post(api.base + "/policy", json={"enabled": True})
    assert on.status_code == 200 and on.json()["decision"] == {"enabled": True} and on.json()["book"]["automatic"] is True
    assert api.client.post(api.base + "/policy", json={"enabled": False}).json()["book"]["automatic"] is False


def test_the_attribute_route_answers_with_the_book_and_refuses_what_is_wrong(blocked_book, api, cast_calls):
    body = {"request_id": "api-a1", "assignments": {"l3": "Joss", "l4": "Tavin", "l5": "Mara"}, "new_roles": ["Tavin"]}
    response = api.client.post(api.base + "/attribute", json=body)
    assert response.status_code == 200
    decision = response.json()["decision"]
    assert decision["attributed"] == 3 and decision["new_characters"] == ["Tavin"]
    assert response.json()["book"]["schema"] == "casting-book-status-v1"
    assert api.client.post(api.base + "/attribute", json={**body, "request_id": "api-a2", "assignments": {"l3": "Nobody"}}).status_code == 409
    assert api.client.post(api.base + "/attribute", json={**body, "request_id": "api-a3", "assignments": {}}).status_code == 422
    assert api.client.post(api.base + "/attribute", json={**body, "request_id": "api-a4", "extra": 1}).status_code == 422
    assert api.client.post(api.base + "/attribute", json={**body, "request_id": "bad id!"}).status_code == 422


def test_the_routes_skip_and_restore_and_refuse_what_is_not_allowed(blocked_book, api):
    assert api.client.post(api.base + "/resolve", json={"request_id": "api-2", "mode": "auto"}).status_code == 200
    skipped = api.client.post(api.base + "/skip", json={"request_id": "api-3", "utterance_ids": ["l6"]})
    assert skipped.status_code == 200 and skipped.json()["book"]["skipped"]["total"] == 3
    assert api.client.post(api.base + "/skip", json={"request_id": "api-4", "utterance_ids": ["l4"]}).status_code == 409, "not in the recording"
    assert api.client.post(api.base + "/skip", json={"request_id": "api-5", "utterance_ids": []}).status_code == 422
    assert api.client.post(api.base + "/skip", json={"request_id": "api-6", "utterance_ids": ["u"] * 101}).status_code == 422
    restored = api.client.post(api.base + "/restore", json={"request_id": "api-7", "utterance_ids": ["l6"]})
    assert restored.status_code == 200 and restored.json()["decision"]["restored"] == 1
    assert api.client.post(api.base + "/restore", json={"request_id": "api-8"}).status_code == 409, "nothing chosen"
    assert api.client.post(api.base + "/restore", json={"request_id": "api-9", "reason": "made-up"}).status_code == 422
    assert api.client.post(api.base + "/policy", json={"enabled": "maybe"}).status_code == 422
    assert api.client.post(api.base + "/skip", json={"request_id": "api-10", "utterance_ids": ["l6"], "extra": 1}).status_code == 422


# ---- a line skipped while the book is being recorded -----------------------

def test_a_line_skipped_mid_run_moves_the_frozen_scope_with_it(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-10", mode="auto")
    tick(blocked_book)
    before = receipt_of(blocked_book)["manifest"]
    manifest = casting._skip_line_in_run(blocked_book.id, casting.AUTOMATIC_PRODUCTION_ID, "l2",
                                         reason="unspeakable", detail="reached its generation budget")
    receipt = receipt_of(blocked_book)
    assert manifest["utterance_ids"] == ["l1", "l3", "l6"] and receipt["manifest"] == manifest
    assert receipt["utterance_ids"] == ["l1", "l3", "l6"] and receipt["status"] == "queued", "a running book keeps running"
    assert casting.skipped_lines(load_project(blocked_book.id))["l2"]["reason"] == "unspeakable"
    project, segments = load_project(blocked_book.id), load_segments(blocked_book.id)
    assert casting.manifest_still_current(project, segments, manifest, "l3"), "the rest of the run is checked against the new scope"
    assert not casting.manifest_still_current(project, segments, before, "l3"), "and the old scope no longer holds"
    assert casting.manifest_current(blocked_book.id, manifest)
    assert get_job(blocked_book.id, receipt["job_id"]).segment_ids == ["l1", "l3", "l6"]
    park(blocked_book)


def test_a_mid_run_skip_that_cannot_be_frozen_changes_nothing(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-11", mode="auto")
    tick(blocked_book)
    project = load_project(blocked_book.id)
    casting._state(project)["roles"]["joss"]["assignment"].update(approved=False, rejected=True)
    save_project(project)
    with pytest.raises(casting.CastingError):
        casting._skip_line_in_run(blocked_book.id, casting.AUTOMATIC_PRODUCTION_ID, "l2", reason="unspeakable", detail="x")
    assert "l2" not in casting.skipped_lines(load_project(blocked_book.id)), "the skip and the new scope commit together or not at all"
    park(blocked_book)


def test_a_chapter_is_assembled_without_its_skipped_lines(blocked_book):
    from dubbing_service.app.audiobook.casting_assembly import CastingAssemblyError, current_frozen_binding
    book.resolve_unattributed(blocked_book.id, request_id="decide-12", mode="skip")
    project = load_project(blocked_book.id)
    only_skipped = [row for row in load_segments(blocked_book.id) if row.id in {"l3", "l4", "l5"}]
    with pytest.raises(CastingAssemblyError, match="no speaking utterances"):
        current_frozen_binding(project, only_skipped)


# ---- what the page is told --------------------------------------------------

def test_status_says_what_was_left_out_and_why(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-13", mode="auto")
    tick(blocked_book)
    casting._skip_line_in_run(blocked_book.id, casting.AUTOMATIC_PRODUCTION_ID, "l2", reason="unspeakable",
                              detail="persistent voice sample is overlong or reached its generation budget")
    status = book.book_status(blocked_book.id)
    assert status["skipped"]["total"] == 3
    groups = {group["reason"]: group for group in status["skipped"]["groups"]}
    assert [g["reason"] for g in status["skipped"]["groups"]] == ["unknown_speaker", "unspeakable"], "in a fixed, readable order"
    assert groups["unknown_speaker"]["count"] == 2 and groups["unknown_speaker"]["label"] == "No speaker could be found"
    assert [line["utterance_id"] for line in groups["unknown_speaker"]["lines"]] == ["l4", "l5"], "in reading order"
    assert groups["unspeakable"]["lines"] == [{"utterance_id": "l2", "text": "l2 is spoken here.", "chapter": "The Gate",
                                               "detail": "persistent voice sample is overlong or reached its generation budget"}]
    assert {row["chapter_id"]: row.get("lines_skipped") for row in status["chapters"]} == {"ch1": 1, "ch2": 2}
    assert status["production"]["lines_total"] == 3 and status["automatic"] is False
    park(blocked_book)


def test_status_shows_only_a_preview_of_a_long_list(blocked_book):
    rows = load_segments(blocked_book.id)
    extra = [Segment(id=f"x{i}", index=10 + i, chapter_id="ch2", text=f"Extra line {i}.", role_id="mara") for i in range(30)]
    project = load_project(blocked_book.id)
    casting.record_skips(project, [row.id for row in extra], reason="human", by="you")
    summary = book._skipped_summary(project, casting.skipped_lines(project), rows + extra)
    group = summary["groups"][0]
    assert summary["total"] == 30 and group["count"] == 30 and len(group["lines"]) == book.SKIP_LINES_SHOWN and group["more"] == 5


def test_a_book_with_every_line_skipped_is_refused_rather_than_planned_empty(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-16", mode="auto")
    tick(blocked_book)
    park(blocked_book)
    every = [row.id for row in load_segments(blocked_book.id) if row.id in {"l1", "l2", "l3", "l6"}]
    outcome_before = receipt_of(blocked_book)["utterance_ids"]
    with pytest.raises(book.BookError, match="nothing in this book to record"):
        book.skip_lines(blocked_book.id, request_id="skip-7", utterance_ids=every)
    assert receipt_of(blocked_book)["utterance_ids"] == outcome_before, "the receipt was not re-planned to nothing"
    assert casting.skipped_ids(load_project(blocked_book.id)) == {"l4", "l5"}, "and nothing was half-saved"


def test_status_names_the_line_that_stopped_the_book(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-14", mode="auto")
    tick(blocked_book)
    project = load_project(blocked_book.id)
    casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(
        status="awaiting_review", last_line_failure={"utterance_id": "l6", "error": "boom", "at": "now"})
    save_project(project)
    failed = book.book_status(blocked_book.id)["failed_line"]
    assert failed == {"utterance_id": "l6", "text": "l6 is spoken here.", "chapter": "The Road", "error": "boom"}
    park(blocked_book)
    book.skip_lines(blocked_book.id, request_id="skip-6", utterance_ids=["l6"])
    assert book.book_status(blocked_book.id)["failed_line"] is None, "once skipped it is no longer what is stopping the book"


def test_a_skipped_book_finishes_with_the_count_in_its_reason_and_no_chapter_of_only_skipped_lines(blocked_book):
    book.resolve_unattributed(blocked_book.id, request_id="decide-15", mode="skip")
    project = load_project(blocked_book.id)
    plan = book.chapter_plan(project, ["l1", "l2", "l6"])
    assert [(row["chapter_id"], row["utterance_ids"]) for row in plan] == [("ch1", ["l1", "l2"]), ("ch2", ["l6"])]
    plan = book.chapter_plan(project, ["l1", "l2"])
    assert [row["chapter_id"] for row in plan] == ["ch1"], "a chapter with nothing left to say disappears from the book"


# ---- the page stays responsive while auditions are being made ----------------------

def count_book_reads(monkeypatch):
    """Count every full read of the segment store made through the modules the view uses."""
    from dubbing_service.app.audiobook import voice_lab as lab
    reads = []
    for module in (casting, lab):
        real = module.load_segments
        monkeypatch.setattr(module, "load_segments", lambda project_id, real=real: reads.append(project_id) or real(project_id))
    return reads


def test_the_casting_view_reads_the_book_once_however_many_characters_have_no_voice_yet(blocked_book, cast_calls, monkeypatch):
    book.attribute_lines(blocked_book.id, request_id="speed-1", new_roles=["Tavin", "Wren", "Ossa"],
                         assignments={"l3": "Tavin", "l4": "Wren", "l5": "Ossa"})
    reads = count_book_reads(monkeypatch)
    view = casting.view(blocked_book.id)
    assert len(reads) == 1, "it used to read the whole book once for every character without an assigned voice"
    by_name = {row["name"]: row for row in view["roles"]}
    assert {"Tavin", "Wren", "Ossa", "Mara", "Joss"} <= set(by_name)
    assert by_name["Tavin"]["current_key"] is None and view["utterance_ids"] == [row[0] for row in LINES]


def test_a_voice_that_is_already_recorded_is_found_without_reading_the_book(blocked_book, monkeypatch):
    from dubbing_service.app.audiobook import voice_lab as lab
    project = load_project(blocked_book.id)
    project.voice_lab.voices["v-known"] = {"key": "v-known", "voice_id": "x", "version": "1", "audio_sha256": "a" * 64,
                                            "audio_file": "voices/a.wav", "name": "Known"}
    save_project(project)
    project = load_project(blocked_book.id)
    expected = lab.voice_records(project)["v-known"]
    monkeypatch.setattr(lab, "load_segments", lambda project_id: (_ for _ in ()).throw(AssertionError("read the whole book")))
    assert lab._record(project, "v-known") == expected, "the shortcut returns exactly what the full index would"
    with pytest.raises(AssertionError, match="read the whole book"):
        lab._record(project, "v-unknown")          # anything not already recorded still takes the full path


def test_saves_use_the_fast_compact_writer_and_read_back_identically(blocked_book):
    from dubbing_service.app.core.config import get_config
    from dubbing_service.app.storage.jobs import load_jobs
    root = get_config().projects_root_abs
    project = load_project(blocked_book.id)
    save_project(project)
    save_segments(blocked_book.id, load_segments(blocked_book.id))
    for name in ("project.json", "segments.json"):
        text = (root / blocked_book.id / name).read_text("utf-8")
        assert "\n" not in text.strip(), f"{name} is written compactly (the indented writer is pure Python and four times slower)"
    assert load_project(blocked_book.id).model_dump(mode="json") == project.model_dump(mode="json")
    assert [row.id for row in load_segments(blocked_book.id)] == [row[0] for row in LINES]
    assert load_jobs(blocked_book.id) == []


def test_the_view_flags_a_chosen_voice_by_the_same_test_the_book_counts_with(blocked_book, monkeypatch):
    rows = casting.view(blocked_book.id)["roles"]
    chosen_ids = {rows[0]["role_id"]}
    monkeypatch.setattr(casting, "approved", lambda project, role_id: role_id in chosen_ids)
    rows = casting.view(blocked_book.id)["roles"]
    assert {row["role_id"] for row in rows if row["chosen"]} == chosen_ids
    assert all(isinstance(row["chosen"], bool) for row in rows)
    # a stand-alone flag: the status follows the round, the flag follows the approval
    monkeypatch.setattr(casting, "approved", lambda project, role_id: False)
    assert not any(row["chosen"] for row in casting.view(blocked_book.id)["roles"])
