"""Where an Auto Cast book is, its chapters as they finish, and the unblocking decision.

Every case runs against a disposable projects root. Chapter assembly is real (WAV
joining, MP3 encoding, hashing); only the collaborators that need a model or the
full strict-cast proof are replaced by narrow doubles.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import autocast_book as book
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook.fingerprint import file_sha256
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.models.project import Chapter, Role, Segment
from dubbing_service.app.routers.voice_lab import router as voice_lab_router
from dubbing_service.app.storage.jobs import mark_job_active, upsert_job
from dubbing_service.app.storage.projects import create_project, load_project, project_meta_lock, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root

REQUEST = casting.AUTOMATIC_PRODUCTION_ID
LINES = [("line-1", "ch1", 0, "mara"), ("line-2", "ch1", 1, "narrator"),
         ("line-3", "ch2", 0, "narrator"), ("line-4", "ch2", 1, "narrator")]


@pytest.fixture
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = create_project("Fixture Book")
    project.roles = [Role(id="narrator", name="Narrator", role_type=RoleType.NARRATOR, tts_backend="qwen3tts"),
                     Role(id="mara", name="Mara", role_type=RoleType.CHARACTER, tts_backend="qwen3tts")]
    project.chapters = [Chapter(id="ch1", title="The Gate", index=0), Chapter(id="ch2", title="The Road", index=1)]
    save_project(project)
    save_segments(project.id, [Segment(id=uid, index=index, chapter_id=chapter, text=f"{uid} is spoken here.",
                                       role_id=role) for uid, chapter, index, role in LINES])
    return SimpleNamespace(root=root, id=project.id)


def write_state(studio, *, receipt=None, roles=None, approved=()):
    project = load_project(studio.id)
    state = casting._state(project)
    state.setdefault("productions", {})
    state.setdefault("roles", {})
    if receipt is not None:
        state["productions"][REQUEST] = receipt
    for role_id, item in (roles or {}).items():
        state["roles"][role_id] = item
    save_project(project)
    return set(approved)


def receipt(status="queued", **extra):
    return {"request_id": REQUEST, "status": status, "job_id": "strict_fixture", "reason": None,
            "utterance_ids": [row[0] for row in LINES], "required_roles": ["narrator", "mara"], **extra}


def approve(monkeypatch, approved):
    monkeypatch.setattr(casting, "approved", lambda project, role_id: role_id in approved)
    monkeypatch.setattr(casting, "_required_role_ids", lambda project: ["narrator", "mara"])


def job(studio, *, done=(), active=False, status=SynthesisJobStatus.RUNNING):
    project = load_project(studio.id)
    upsert_job(studio.id, SynthesisJob(id="strict_fixture", project_id=studio.id, project_name=project.name,
                                       job_type="batch_synthesize", request_id="strict_fixture",
                                       idempotency_key="strict_fixture", segment_ids=[row[0] for row in LINES],
                                       completed_segment_ids=list(done), status=status))
    if active:
        mark_job_active(studio.id, "strict_fixture", True)


def candidate(status="ready", index=0):
    return {"candidate_id": f"c{index}", "status": status, "audio_sha256": "a" * 64 if status == "ready" else None}


# --------------------------------------------------------------------------
# chapter plan
# --------------------------------------------------------------------------

def test_chapter_plan_orders_chapters_and_keeps_only_scoped_utterances(studio):
    project = load_project(studio.id)
    plan = book.chapter_plan(project, ["line-4", "line-3", "line-1"])
    assert [(row["chapter_id"], row["utterance_ids"]) for row in plan] == [
        ("ch1", ["line-1"]), ("ch2", ["line-4", "line-3"])]
    assert [row["title"] for row in plan] == ["The Gate", "The Road"]


# --------------------------------------------------------------------------
# book status: the three things a human can be looking at, and what blocks them
# --------------------------------------------------------------------------

def test_voices_waiting_for_the_human_come_first(studio, monkeypatch):
    approve(monkeypatch, {"narrator"})
    write_state(studio, roles={"mara": {"round": {"status": "awaiting_review",
                                                  "candidates": [candidate(index=0), candidate(index=1),
                                                                 candidate("pending", 2)]}}})
    status = book.book_status(studio.id)
    assert status["state"] == "voices_waiting"
    assert status["voices"]["waiting"] == [{"role_id": "mara", "name": "Mara", "options": 2, "replacement": False}]
    assert status["voices"]["approved"] == 1 and status["voices"]["total"] == 2
    assert status["production"] is None


def test_a_book_being_recorded_reports_real_progress_and_the_current_chapter(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("queued"))
    job(studio, done=["line-1", "line-2", "line-3"], active=True)
    status = book.book_status(studio.id)
    assert status["state"] == "processing"
    production = status["production"]
    assert (production["lines_done"], production["lines_total"]) == (3, 4)
    assert production["phase"] == "running" and production["current_chapter"] == "The Road"
    assert [(row["number"], row["title"], row["status"], row["lines_done"], row["lines_total"])
            for row in status["chapters"]] == [(1, "The Gate", "waiting", 2, 2), (2, "The Road", "rendering", 1, 2)]


def test_a_recorded_chapter_is_playable_before_the_book_is_finished(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    chapters = {"ch1": {"status": "ready", "chapter_id": "ch1", "title": "The Gate", "number": 1,
                        "export_id": "cast_" + "b" * 28, "audio_sha256": "c" * 64, "duration_seconds": 61.5}}
    write_state(studio, receipt=receipt("queued", chapters=chapters))
    job(studio, done=["line-1", "line-2"], active=True)
    status = book.book_status(studio.id)
    first, second = status["chapters"]
    assert first["status"] == "ready" and first["duration_seconds"] == 61.5
    assert first["audio_url"] == f"/api/v1/projects/{studio.id}/voice-lab/casting/chapters/cast_{'b' * 28}/audio"
    assert first["download_url"] == first["audio_url"] + "?download=true"
    assert second["status"] == "rendering" and "audio_url" not in second
    assert status["state"] == "processing" and status["production"]["chapters_ready"] == 1


def test_a_finished_book_lists_every_chapter(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    chapters = {cid: {"status": "ready", "chapter_id": cid, "title": title, "number": number,
                      "export_id": "cast_" + str(number) * 28, "audio_sha256": "d" * 64, "duration_seconds": 30.0}
                for number, (cid, title) in enumerate((("ch1", "The Gate"), ("ch2", "The Road")), start=1)}
    write_state(studio, receipt=receipt("succeeded", chapters=chapters))
    job(studio, done=[row[0] for row in LINES], status=SynthesisJobStatus.SUCCEEDED)
    status = book.book_status(studio.id)
    assert status["state"] == "finished"
    assert status["production"]["phase"] == "finished" and status["production"]["lines_done"] == 4
    assert [row["status"] for row in status["chapters"]] == ["ready", "ready"]


def test_a_succeeded_receipt_with_a_missing_chapter_is_not_called_finished(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("succeeded", chapters={
        "ch1": {"status": "ready", "chapter_id": "ch1", "title": "The Gate", "number": 1,
                "export_id": "cast_" + "1" * 28, "audio_sha256": "d" * 64, "duration_seconds": 30.0},
        "ch2": {"status": "failed", "chapter_id": "ch2", "reason": "QA failed"}}))
    status = book.book_status(studio.id)
    assert status["production"]["phase"] == "exporting" and status["state"] == "processing"
    assert [row["status"] for row in status["chapters"]] == ["ready", "failed"]
    assert status["chapters"][1]["reason"] == "QA failed"


def test_a_book_blocked_on_unattributed_lines_names_them_and_the_choices(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    rows = load_segments(studio.id)
    rows[2].role_id = None
    rows[3].role_id = None
    rows[3].attribution = Attribution(label="Mara", source="nlp")
    save_segments(studio.id, rows)
    project = load_project(studio.id)
    project.metadata["autocast_import"] = {"mode": "autocast", "unresolved_utterance_ids": ["line-4"]}
    save_project(project)
    write_state(studio, receipt=receipt("blocked", reason="2 of 4 source utterance(s) have no speaker"))
    job(studio)
    status = book.book_status(studio.id)
    assert status["state"] == "blocked" and status["production"]["phase"] == "blocked"
    assert status["block"] == {"unresolved": 2, "suggested": 1, "suggested_mappable": 1, "without_evidence": 1,
                               "narrator": "Narrator", "actions": ["auto", "suggested", "narrator", "skip"]}


# --------------------------------------------------------------------------
# chapter assembly
# --------------------------------------------------------------------------

@pytest.fixture
def assembling(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("queued"))
    for uid, *_ in LINES:
        make_tone_wav(studio.root / studio.id / "audio" / f"{uid}.wav", seconds=0.6, frequency=220)
    state = SimpleNamespace(binding={"manifest_id": "manifest_fixture", "selections": [{"utterance_id": "x"}]})
    monkeypatch.setattr(book.release, "check_coverage", lambda segments, **kwargs: SimpleNamespace(ready=True))
    monkeypatch.setattr(book, "current_frozen_binding", lambda project, segments: state.binding)
    monkeypatch.setattr(book.takes, "selected_take", lambda segment: SimpleNamespace(
        audio_file=f"audio/{segment.id}.wav", role_id=segment.role_id, id="take-" + segment.id,
        audio_sha256=file_sha256(studio.root / studio.id / "audio" / f"{segment.id}.wav")))
    monkeypatch.setattr(book.qa, "qa_take_audio", lambda **kwargs: SimpleNamespace(
        passed=True, blocking_failures=lambda: [], executed_check_ids=lambda: ["fixture"],
        model_dump=lambda mode="json": {"passed": True}))
    state.studio = studio
    return state


def test_a_finished_chapter_is_assembled_published_and_recorded(assembling):
    studio = assembling.studio
    record = book.export_chapter(studio.id, REQUEST, "ch1", 1)
    assert record["status"] == "ready" and record["export_id"].startswith("cast_") and record["title"] == "The Gate"
    path, sha256, label, number = book.chapter_artifact(studio.id, record["export_id"])
    assert path.suffix == ".mp3" and path.is_file() and file_sha256(path) == sha256 == record["audio_sha256"]
    assert (label, number) == ("The Gate", 1)
    assert 1.1 < record["duration_seconds"] < 1.4, "two 0.6 s utterances, joined losslessly, then encoded"
    stored = casting._state(load_project(studio.id))["productions"][REQUEST]["chapters"]["ch1"]
    assert stored["export_id"] == record["export_id"] and stored["utterance_count"] == 2


def _probe_mp3(path):
    import json
    import re
    import subprocess
    stream = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels,bit_rate",
                                        "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    peak = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                          capture_output=True, text=True, check=True).stderr
    return int(stream["sample_rate"]), int(stream["channels"]), int(stream["bit_rate"]), float(re.findall(r"Peak:\s*(-?[\d.]+) dBFS", peak)[-1])


def test_the_listening_file_meets_audiobook_platform_specs_even_for_a_hot_master(tmp_path):
    loud = make_tone_wav(tmp_path / "hot.wav", seconds=2.0, amplitude=0.95)     # about -0.4 dBFS
    out = tmp_path / "out.mp3"
    book._listening_mp3(loud, out)
    rate, channels, bit_rate, peak = _probe_mp3(out)
    assert (rate, channels, bit_rate) == (44100, 1, 192000)
    assert peak <= -3.0, "decoded peak must stay under the -3 dBFS platform ceiling"


def test_the_listening_file_does_not_change_the_level_of_ordinary_speech(tmp_path):
    quiet = make_tone_wav(tmp_path / "level.wav", seconds=2.0, amplitude=0.25)  # about -12 dBFS, well under the ceiling
    out = tmp_path / "level.mp3"
    book._listening_mp3(quiet, out)
    before, after = book._loudness(quiet), book._loudness(out)
    assert abs(before - after) < 0.5, f"mastering moved the loudness from {before} to {after}"


def test_assembling_the_same_chapter_again_returns_the_same_recording(assembling):
    studio = assembling.studio
    first = book.export_chapter(studio.id, REQUEST, "ch1", 1)
    second = book.export_chapter(studio.id, REQUEST, "ch1", 1)
    assert first["export_id"] == second["export_id"] and first["audio_sha256"] == second["audio_sha256"]
    assert len(list((studio.root / studio.id / "autocast" / "chapters").iterdir())) == 1


def test_a_scene_break_becomes_a_recorded_pause_not_speech(assembling):
    studio = assembling.studio
    rows = load_segments(studio.id)
    rows[1].text = "***"
    save_segments(studio.id, rows)
    record = book.export_chapter(studio.id, REQUEST, "ch1", 1)
    assert 1.8 < record["duration_seconds"] < 2.1, "one 0.6 s line plus a 1.25 s pause"
    assert record["utterance_count"] == 1


def test_a_chapter_is_refused_when_its_takes_are_not_the_current_approved_cast(assembling, monkeypatch):
    studio = assembling.studio
    monkeypatch.setattr(book.release, "check_coverage", lambda segments, **kwargs: SimpleNamespace(ready=False))
    with pytest.raises(book.BookError, match="missing, stale or QA-failing"):
        book.export_chapter(studio.id, REQUEST, "ch1", 1)
    assert "chapters" not in casting._state(load_project(studio.id))["productions"][REQUEST]


def test_a_cast_that_changes_while_assembling_does_not_publish(assembling, monkeypatch):
    studio = assembling.studio
    calls = []

    def moving(project, segments):
        calls.append(1)
        return assembling.binding if len(calls) == 1 else {"manifest_id": "manifest_other", "selections": []}

    monkeypatch.setattr(book, "current_frozen_binding", moving)
    with pytest.raises(book.BookError, match="changed while this chapter was assembled"):
        book.export_chapter(studio.id, REQUEST, "ch1", 1)
    assert "chapters" not in casting._state(load_project(studio.id))["productions"][REQUEST]


def test_a_chapter_failure_is_remembered_without_failing_the_book(assembling):
    studio = assembling.studio
    book.record_chapter_failure(studio.id, REQUEST, "ch2", "QA failed")
    stored = casting._state(load_project(studio.id))["productions"][REQUEST]["chapters"]["ch2"]
    assert stored["status"] == "failed" and stored["reason"] == "QA failed"
    record = book.export_chapter(studio.id, REQUEST, "ch2", 2)
    assert casting._state(load_project(studio.id))["productions"][REQUEST]["chapters"]["ch2"]["status"] == "ready"
    assert record["status"] == "ready", "a later successful assembly replaces the remembered failure"


def test_chapter_artifact_refuses_unknown_and_malformed_identities(studio):
    with pytest.raises(book.BookError):
        book.chapter_artifact(studio.id, "../../project")
    with pytest.raises(book.BookError, match="not available"):
        book.chapter_artifact(studio.id, "cast_" + "0" * 28)


# --------------------------------------------------------------------------
# the explicit decision for lines with no speaker
# --------------------------------------------------------------------------

@pytest.fixture
def unresolved(studio):
    rows = load_segments(studio.id)
    for row in rows[2:]:
        row.role_id = None
    rows[2].attribution = Attribution(label="Mara", source="nlp")
    save_segments(studio.id, rows)
    project = load_project(studio.id)
    project.metadata["autocast_import"] = {"mode": "autocast", "unresolved_utterance_ids": ["line-3"]}
    save_project(project)
    return studio


def roles_of(studio):
    return {row.id: row.role_id for row in load_segments(studio.id)}


def test_accepting_the_suggested_speakers_resolves_only_the_lines_that_have_one(unresolved):
    outcome = book.resolve_unattributed(unresolved.id, request_id="decide-1", mode="suggested")
    assert outcome["resolved"] == 1 and outcome["still_unresolved"] == 1
    assert roles_of(unresolved)["line-3"] == "mara" and roles_of(unresolved)["line-4"] is None
    row = next(row for row in load_segments(unresolved.id) if row.id == "line-3")
    assert row.attribution.human_authoritative and row.attribution.label == "Mara"
    assert casting._held_ids(load_project(unresolved.id)) == set()


def test_reading_the_rest_with_the_narrator_is_an_explicit_human_attribution(unresolved):
    outcome = book.resolve_unattributed(unresolved.id, request_id="decide-2", mode="narrator")
    assert outcome["resolved"] == 2 and outcome["still_unresolved"] == 0
    assert roles_of(unresolved)["line-3"] == roles_of(unresolved)["line-4"] == "narrator"
    assert all(row.attribution.human_authoritative for row in load_segments(unresolved.id) if row.id in {"line-3", "line-4"})
    _, rows, _, attention = casting._scope(load_project(unresolved.id))
    assert attention == [], "nothing is left for casting to hold back"


def test_the_same_decision_request_is_idempotent(unresolved):
    first = book.resolve_unattributed(unresolved.id, request_id="decide-3", mode="narrator")
    again = book.resolve_unattributed(unresolved.id, request_id="decide-3", mode="narrator")
    assert first == again


def test_a_decision_needs_a_known_mode_and_something_to_decide(unresolved, studio):
    with pytest.raises(book.BookError, match="suggested speakers"):
        book.resolve_unattributed(unresolved.id, request_id="decide-4", mode="everyone")
    book.resolve_unattributed(unresolved.id, request_id="decide-5", mode="narrator")
    with pytest.raises(book.BookError, match="already has a speaker"):
        book.resolve_unattributed(unresolved.id, request_id="decide-6", mode="narrator")


def test_narrator_reading_refuses_a_book_with_no_single_narrator(unresolved):
    project = load_project(unresolved.id)
    project.roles = [role for role in project.roles if role.role_type != RoleType.NARRATOR]
    save_project(project)
    with pytest.raises(book.BookError, match="no single narrator"):
        book.resolve_unattributed(unresolved.id, request_id="decide-7", mode="narrator")


@pytest.fixture
def placeholders(studio):
    """The shape of the real acceptance book: most held lines are labelled "unknown",
    and the project even carries a placeholder role of that name."""
    project = load_project(studio.id)
    project.roles += [Role(id="unknown", name="unknown", role_type=RoleType.NARRATOR, tts_backend="qwen3tts"),
                      Role(id="unattributed", name="Unattributed", role_type=RoleType.CHARACTER, tts_backend="qwen3tts")]
    project.metadata["autocast_import"] = {"mode": "autocast", "unresolved_utterance_ids": ["line-2", "line-3", "line-4"]}
    save_project(project)
    rows = load_segments(studio.id)
    for row, label in zip(rows[1:], ("unknown", "Mara", "Unattributed"), strict=True):
        row.role_id = None
        row.attribution = Attribution(label=label, source="nlp")
    save_segments(studio.id, rows)
    return studio


def test_an_unknown_label_is_not_a_suggested_speaker_even_when_a_role_has_that_name(placeholders):
    outcome = book.resolve_unattributed(placeholders.id, request_id="decide-8", mode="suggested")
    assert outcome["resolved"] == 1, "only Mara is a real proposal"
    assert roles_of(placeholders) == {"line-1": "mara", "line-2": None, "line-3": "mara", "line-4": None}
    assert casting._held_ids(load_project(placeholders.id)) == {"line-2", "line-4"}, "placeholders stay held for the human"


def test_the_block_report_counts_only_real_suggestions(placeholders):
    report = book._block_report(load_project(placeholders.id))
    assert report["unresolved"] == 3 and report["suggested_mappable"] == 1


def test_a_book_of_only_placeholder_labels_has_nothing_to_accept(placeholders):
    rows = load_segments(placeholders.id)
    rows[2].attribution = Attribution(label="unknown", source="nlp")
    save_segments(placeholders.id, rows)
    with pytest.raises(book.BookError, match="real suggested speaker"):
        book.resolve_unattributed(placeholders.id, request_id="decide-9", mode="suggested")
    assert book._block_report(load_project(placeholders.id))["suggested_mappable"] == 0
    book.resolve_unattributed(placeholders.id, request_id="decide-10", mode="narrator")
    assert set(roles_of(placeholders).values()) == {"narrator", "mara"}, "the narrator decision is still available"


# --------------------------------------------------------------------------
# the Studio routes
# --------------------------------------------------------------------------

def test_studio_routes_expose_status_the_decision_and_chapter_audio(assembling, monkeypatch):
    studio = assembling.studio
    monkeypatch.setattr(casting, "maintenance_tick", lambda **kwargs: [])
    app = FastAPI()
    app.include_router(voice_lab_router, prefix="/api/v1")
    client = TestClient(app)
    base = f"/api/v1/projects/{studio.id}/voice-lab/casting"
    assert client.get(base + "/book").json()["schema"] == "casting-book-status-v1"
    record = book.export_chapter(studio.id, REQUEST, "ch1", 1)
    audio = client.get(f"{base}/chapters/{record['export_id']}/audio", headers={"Range": "bytes=0-99"})
    assert audio.status_code == 206 and audio.headers["content-type"] == "audio/mpeg" and len(audio.content) == 100
    head = client.head(f"{base}/chapters/{record['export_id']}/audio")
    assert head.status_code == 200 and int(head.headers["content-length"]) > 100
    assert client.get(f"{base}/chapters/cast_{'0' * 28}/audio").status_code == 409
    assert client.post(base + "/production/resolve", json={"request_id": "r-1", "mode": "narrator"}).status_code == 409
    assert client.post(base + "/production/resolve", json={"request_id": "r-1", "mode": "bogus"}).status_code == 422


# --------------------------------------------------------------------------
# the production worker publishes and announces each chapter as it completes
# --------------------------------------------------------------------------

@pytest.fixture
def worker(studio, monkeypatch):
    from dubbing_service.app.audiobook import pipeline
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("queued", manifest={"id": "manifest_fixture"}, attempts=0, total_attempts=0))
    events = []
    state_checks = SimpleNamespace(full=0, book_reads=0)

    def full_check(project_id, manifest):
        state_checks.full += 1
        return True

    real_load_segments = casting.load_segments

    def counted_load_segments(project_id):
        state_checks.book_reads += 1
        return real_load_segments(project_id)

    monkeypatch.setattr(casting, "manifest_current", full_check)
    monkeypatch.setattr(casting, "load_segments", counted_load_segments)
    monkeypatch.setattr(casting, "_cast_settled", lambda project, required=None: (True, list(required or [])))

    def render(project_id, ids, **kwargs):
        events.append(("render", ids[0]))
        return [SimpleNamespace(success=True, selected=True, stale=False, qa_passed=True, error=None)]

    monkeypatch.setattr(pipeline, "render_utterances", render)
    monkeypatch.setattr(casting, "notify_ready", lambda project_id: events.append(("notify", None)) or {"state": "idle"})
    state = SimpleNamespace(events=events, failures={}, studio=studio, checks=state_checks)

    def export(project_id, request_id, chapter_id, number=None):
        if state.failures.get(chapter_id, 0) > 0:
            state.failures[chapter_id] -= 1
            raise book.BookError("assembly fixture failure")
        events.append(("export", chapter_id))
        record = {"status": "ready", "chapter_id": chapter_id, "title": chapter_id, "number": number,
                  "export_id": "cast_" + chapter_id.encode().hex().ljust(28, "0")[:28], "audio_sha256": "e" * 64,
                  "duration_seconds": 10.0}
        project = load_project(project_id)
        casting._state(project)["productions"][request_id].setdefault("chapters", {})[chapter_id] = record
        save_project(project)
        return record

    monkeypatch.setattr(book, "export_chapter", export)
    return state


def run_worker(worker):
    casting._production_worker(worker.studio.id, REQUEST, renderer_factory=lambda backend: object())
    return casting._state(load_project(worker.studio.id))["productions"][REQUEST]


def test_each_chapter_is_published_and_announced_the_moment_its_last_line_is_recorded(worker):
    result = run_worker(worker)
    assert worker.events == [("render", "line-1"), ("render", "line-2"), ("export", "ch1"), ("notify", None),
                             ("render", "line-3"), ("render", "line-4"), ("export", "ch2"), ("notify", None)], \
        "chapter 1 is available and announced before chapter 2 has even started"
    assert result["status"] == "succeeded" and result["reason"] == "All 2 chapters are recorded and ready to listen to."
    assert sorted(result["chapters"]) == ["ch1", "ch2"]


def test_a_chapter_that_fails_to_assemble_is_retried_before_the_book_is_finished(worker):
    worker.failures = {"ch1": 1}
    result = run_worker(worker)
    assert result["status"] == "succeeded"
    assert result["chapters"]["ch1"]["status"] == "ready", "the retry after the last line published it"
    assert [event for event in worker.events if event[0] == "export"] == [("export", "ch2"), ("export", "ch1")]


def test_a_book_is_never_called_finished_with_a_chapter_missing(worker):
    worker.failures = {"ch1": 5}
    result = run_worker(worker)
    assert result["status"] == "awaiting_review", "it stays resumable instead of succeeding"
    assert "Chapter 1 (The Gate) could not be assembled" in result["reason"]
    assert result["chapters"]["ch1"]["status"] == "failed" and result["chapters"]["ch2"]["status"] == "ready"


def test_a_notification_failure_never_undoes_a_recorded_chapter(worker, monkeypatch):
    def broken(project_id):
        raise ConnectionError("push service is down")

    monkeypatch.setattr(casting, "notify_ready", broken)
    result = run_worker(worker)
    assert result["status"] == "succeeded" and sorted(result["chapters"]) == ["ch1", "ch2"]


def test_the_whole_book_is_revalidated_only_where_audio_is_published_never_per_line(worker):
    run_worker(worker)
    assert worker.checks.full == 2, "the full re-derivation runs at the start and at the final receipt, not once per line"


def test_the_worker_reads_the_book_once_for_the_whole_run_not_once_per_line(worker):
    run_worker(worker)
    assert worker.checks.book_reads == 1, "each line is validated by render_utterances on the state it reads anyway"


def test_a_line_refused_by_the_manifest_guard_parks_the_run_without_recording_it(worker, monkeypatch):
    from dubbing_service.app.audiobook import pipeline

    def render(project_id, ids, **kwargs):
        if ids[0] == "line-2":
            raise pipeline.PipelineError("Current human cast no longer matches the frozen production manifest")
        worker.events.append(("render", ids[0]))
        return [SimpleNamespace(success=True, selected=True, stale=False, qa_passed=True, error=None)]

    monkeypatch.setattr(pipeline, "render_utterances", render)
    result = run_worker(worker)
    assert result["status"] == "awaiting_review" and "no longer matches the frozen production manifest" in result["reason"]
    assert worker.events == [("render", "line-1")], "nothing after the refused line was recorded, and no chapter was published"


def test_the_worker_measures_the_pace_of_every_line_it_synthesizes(worker):
    run_worker(worker)
    pace = book.read_pace(worker.studio.id, REQUEST)
    assert [row["characters"] for row in pace["lines"]] == [22, 22, 22, 22]
    assert all(row["seconds"] >= 0 for row in pace["lines"])


# ---- a line the voice model refuses is retried, then skipped; a broken service is not ----

RECORDED = SimpleNamespace(success=True, selected=True, stale=False, qa_passed=True, error=None)
REFUSED = SimpleNamespace(success=False, selected=False, stale=False, qa_passed=False, error=(
    'TTS 调用失败: character voice synthesize failed: {"detail":"persistent voice sample is overlong or reached its '
    'generation budget (11.68s, 147 tokens); review required"}'))
FAILED_QA = SimpleNamespace(success=True, selected=False, stale=False, qa_passed=False, error="blocking QA check failed")
UNREACHABLE = SimpleNamespace(success=False, selected=False, stale=False, qa_passed=False, error="Connection refused by the voice service")
STALE = SimpleNamespace(success=True, selected=False, stale=True, qa_passed=True, error="cast changed")


def scripted(worker, monkeypatch, outcomes, automatic=True):
    """The renderer answers each line from `outcomes`; the last answer for a line repeats."""
    from dubbing_service.app.audiobook import pipeline
    seen = SimpleNamespace(calls=[], manifests=[], skips=[])

    def render(project_id, ids, **kwargs):
        seen.calls.append(ids[0])
        seen.manifests.append(kwargs["casting_manifest"])
        queue = outcomes.get(ids[0]) or [RECORDED]
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if result is RECORDED:
            worker.events.append(("render", ids[0]))
        return [result]

    def skip(project_id, request_id, uid, *, reason, detail):
        seen.skips.append((uid, reason))
        worker.events.append(("skip", uid))
        with project_meta_lock(project_id):
            project = load_project(project_id)
            casting.record_skips(project, [uid], reason=reason, detail=detail, by="auto")
            save_project(project)
        return {"id": "manifest_after_" + uid}

    monkeypatch.setattr(pipeline, "render_utterances", render)
    monkeypatch.setattr(casting, "_skip_line_in_run", skip)
    monkeypatch.setattr(book, "automatic_handling", lambda project: automatic)
    return seen


@pytest.mark.parametrize("result, expected", [
    (RECORDED, None), (REFUSED, ("unspeakable", REFUSED.error)), (FAILED_QA, ("failed_qa", "blocking QA check failed")),
    (UNREACHABLE, ("other", UNREACHABLE.error)), (STALE, ("other", "cast changed")),
    (SimpleNamespace(success=False, selected=False, stale=False, qa_passed=False, error=None),
     ("other", "Production Take failed current-cast validation")),
])
def test_only_a_refusal_about_the_line_itself_is_worth_skipping(result, expected):
    assert casting.line_problem(result) == expected


def test_a_line_the_voice_model_refuses_twice_is_skipped_and_the_book_carries_on(worker, monkeypatch):
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED]})
    result = run_worker(worker)
    assert seen.calls.count("line-2") == casting.LINE_ATTEMPTS, "one retry before giving up on it"
    assert seen.skips == [("line-2", "unspeakable")]
    assert worker.events == [("render", "line-1"), ("skip", "line-2"), ("export", "ch1"), ("notify", None),
                             ("render", "line-3"), ("render", "line-4"), ("export", "ch2"), ("notify", None)]
    assert result["status"] == "succeeded"
    assert result["reason"] == "All 2 chapters are recorded and ready to listen to (1 line skipped)."
    assert seen.manifests[:2] == [{"id": "manifest_fixture"}] * 2 and seen.manifests[-1] == {"id": "manifest_after_line-2"}, \
        "lines after the skip are validated against the manifest re-frozen without it"


def test_with_automatic_handling_off_a_refused_line_waits_for_the_person(worker, monkeypatch):
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED]}, automatic=False)
    result = run_worker(worker)
    assert seen.skips == [] and seen.calls.count("line-2") == casting.LINE_ATTEMPTS, "it still tries twice"
    assert result["status"] == "awaiting_review" and "automatic handling is off" in result["reason"]
    assert result["last_line_failure"]["utterance_id"] == "line-2", "so the page can offer to skip or retry exactly that line"
    assert casting.skipped_lines(load_project(worker.studio.id)) == {}


def test_a_refusal_that_does_not_repeat_is_not_a_skip(worker, monkeypatch):
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED, RECORDED]})
    result = run_worker(worker)
    assert seen.skips == [] and seen.calls.count("line-2") == 2 and ("render", "line-2") in worker.events
    assert result["reason"] == "All 2 chapters are recorded and ready to listen to."


def test_a_recording_that_keeps_failing_its_quality_checks_is_skipped(worker, monkeypatch):
    seen = scripted(worker, monkeypatch, {"line-4": [FAILED_QA]})
    result = run_worker(worker)
    assert seen.skips == [("line-4", "failed_qa")] and result["status"] == "succeeded"


@pytest.mark.parametrize("problem", [UNREACHABLE, STALE])
def test_a_problem_that_is_not_about_the_line_stops_the_book_instead_of_skipping_it(worker, monkeypatch, problem):
    seen = scripted(worker, monkeypatch, {"line-2": [problem]})
    result = run_worker(worker)
    assert seen.skips == [] and seen.calls == ["line-1", "line-2"], "no retry loop and no skip for an outage"
    assert result["status"] == "awaiting_review" and result["last_line_failure"]["utterance_id"] == "line-2"
    assert casting.skipped_lines(load_project(worker.studio.id)) == {}


def test_a_chapter_whose_every_line_is_skipped_is_left_out_of_the_book(worker, monkeypatch):
    seen = scripted(worker, monkeypatch, {"line-3": [REFUSED], "line-4": [REFUSED]})
    result = run_worker(worker)
    assert [event for event in worker.events if event[0] == "export"] == [("export", "ch1")], "there is no second chapter to record"
    assert seen.skips == [("line-3", "unspeakable"), ("line-4", "unspeakable")]
    assert result["status"] == "succeeded" and sorted(result["chapters"]) == ["ch1"]
    assert result["reason"] == "All 1 chapters are recorded and ready to listen to (2 lines skipped)."


def test_the_run_stops_rather_than_skip_past_a_failing_voice_service(worker, monkeypatch):
    monkeypatch.setattr(casting, "MAX_SKIPS_IN_A_ROW", 1)
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED], "line-3": [REFUSED]})
    result = run_worker(worker)
    assert seen.skips == [("line-2", "unspeakable")], "one skip, then the next refusal in a row stops the run"
    assert result["status"] == "awaiting_review" and "1 lines in a row were refused" in result["reason"]
    assert "voice service may not be healthy" in result["reason"]
    assert result["last_line_failure"]["utterance_id"] == "line-3", "so a person can look at it and skip it themselves"


def test_the_run_stops_when_it_has_skipped_more_than_it_is_allowed_to(worker, monkeypatch):
    monkeypatch.setattr(casting, "MIN_SKIP_BUDGET", 1)
    monkeypatch.setattr(casting, "MAX_SKIP_SHARE", 0)
    seen = scripted(worker, monkeypatch, {"line-1": [REFUSED], "line-3": [REFUSED]})    # not consecutive: line-2 records
    result = run_worker(worker)
    assert seen.skips == [("line-1", "unspeakable")]
    assert result["status"] == "awaiting_review" and "1 lines have already been skipped in this run" in result["reason"]


def test_a_skipped_line_costs_no_job_progress_and_no_pace(worker, monkeypatch):
    scripted(worker, monkeypatch, {"line-2": [REFUSED]})
    run_worker(worker)
    assert [row["characters"] for row in book.read_pace(worker.studio.id, REQUEST)["lines"]] == [22, 22, 22], "only recorded lines"


def test_a_reused_take_is_not_counted_as_recording_pace(worker, monkeypatch):
    from dubbing_service.app.audiobook import pipeline
    monkeypatch.setattr(pipeline, "render_utterances", lambda project_id, ids, **kwargs: [SimpleNamespace(
        success=True, selected=True, stale=False, qa_passed=True, error=None, dispatched=False)])
    run_worker(worker)
    assert book.read_pace(worker.studio.id, REQUEST)["lines"] == [], "reusing a Take is instant and says nothing about synthesis speed"


# --------------------------------------------------------------------------
# time left: the measured pace applied to what is left, never a guess
# --------------------------------------------------------------------------

NOW = 1_800_000_000.0


def measured(studio, *, lines=3, seconds=11.0, characters=22, last_at=NOW - 4, request_id=REQUEST):
    book._write_pace(studio.id, {"schema": book.PACE_SCHEMA, "request_id": request_id, "started_at": NOW - 120,
        "lines": [{"characters": characters, "seconds": seconds, "at": last_at - (lines - 1 - index) * seconds}
                  for index in range(lines)]})


def test_no_time_is_promised_before_the_pace_has_been_measured(studio):
    assert book.estimate_seconds([22], None) is None
    measured(studio, lines=2)
    assert book.estimate_seconds([22], book.read_pace(studio.id, REQUEST)) is None
    assert book.read_pace(studio.id, "another-production") is None, "another production's pace is not this one's"
    book._pace_path(studio.id).write_text("{not json", "utf-8")
    assert book.read_pace(studio.id, REQUEST) is None


def test_time_left_is_the_measured_pace_applied_to_what_is_left(studio):
    measured(studio)                                    # 22 characters took 11 s: half a second each
    pace = book.read_pace(studio.id, REQUEST)
    assert book.estimate_seconds([22, 22], pace, now=NOW - 4) == 22
    assert book.estimate_seconds([22, 22], pace, now=NOW) == 18, "the line in flight has already been running 4 s"
    assert book.estimate_seconds([22, 22], pace, now=NOW + 96) == 11, "but never more than that line is expected to take"
    assert book.estimate_seconds([44], pace, now=NOW - 4) == 22, "longer lines take proportionally longer"
    assert book.estimate_seconds([], pace, now=NOW) == 0


def test_the_pace_is_drawn_from_recent_lines_only(studio):
    for index in range(book.PACE_WINDOW + 5):
        book.record_pace(studio.id, REQUEST, characters=10 + index, seconds=1.0)
    assert len(book.read_pace(studio.id, REQUEST)["lines"]) == book.PACE_WINDOW
    book.start_pace(studio.id, REQUEST)
    assert book.read_pace(studio.id, REQUEST)["lines"] == [], "a new run measures its own pace"


def test_a_failure_to_record_pace_never_fails_a_render(studio, monkeypatch):
    def broken(project_id, pace):
        raise OSError("disk full")

    monkeypatch.setattr(book, "_write_pace", broken)
    book.start_pace(studio.id, REQUEST)
    book.record_pace(studio.id, REQUEST, characters=22, seconds=3.0)


def test_status_reports_time_left_for_the_chapter_and_the_book(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    monkeypatch.setattr(book, "time", SimpleNamespace(time=lambda: NOW))
    write_state(studio, receipt=receipt("queued"))
    job(studio, done=["line-1"], active=True)
    measured(studio)
    production = book.book_status(studio.id)["production"]
    assert production["eta"] == {"chapter_seconds": 7, "book_seconds": 29, "measured_lines": 3}
    assert next(row for row in book.book_status(studio.id)["chapters"] if row["status"] == "rendering")["eta_seconds"] == 7


def test_status_admits_it_is_still_measuring(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("queued"))
    job(studio, done=["line-1"], active=True)
    measured(studio, lines=2)
    production = book.book_status(studio.id)["production"]
    assert production["eta"] == {"chapter_seconds": None, "book_seconds": None, "measured_lines": 2}


def test_status_gives_no_time_when_nothing_is_being_recorded(studio, monkeypatch):
    approve(monkeypatch, {"narrator", "mara"})
    write_state(studio, receipt=receipt("queued"))
    job(studio, done=["line-1"], active=False)
    measured(studio)
    production = book.book_status(studio.id)["production"]
    assert production["eta"] is None, "a queued run must not borrow the last run's pace"


# ---- web addresses are never read aloud, so they never stop a book ----------------------------

LINK = "Table of Contents URL: https://www.royalroad.com/fiction/one hundred four thousand/the-story"
SENTENCE_WITH_LINK = "Mara pinned the map from https://example.com/map to the wall and stared at it for an hour."


def write_line(worker, uid, text):
    rows = load_segments(worker.studio.id)
    for row in rows:
        if row.id == uid:
            row.text = text
    save_segments(worker.studio.id, rows)


def test_a_link_line_is_left_out_without_asking_the_voice_even_with_automatic_handling_off(worker, monkeypatch):
    write_line(worker, "line-2", LINK)
    seen = scripted(worker, monkeypatch, {}, automatic=False)
    result = run_worker(worker)
    assert "line-2" not in seen.calls, "the voice service is never asked to read an address"
    assert seen.skips == [("line-2", casting.SKIP_WEB_ADDRESS)]
    assert result["status"] == "succeeded" and result["reason"].endswith("(1 line skipped).")


def test_a_sentence_with_an_address_the_voice_refuses_is_left_out_even_with_automatic_handling_off(worker, monkeypatch):
    write_line(worker, "line-2", SENTENCE_WITH_LINK)
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED]}, automatic=False)
    result = run_worker(worker)
    assert seen.calls.count("line-2") == casting.LINE_ATTEMPTS, "a sentence is still tried: most of it can be read"
    assert seen.skips == [("line-2", casting.SKIP_WEB_ADDRESS)] and result["status"] == "succeeded"


def test_leaving_out_links_never_trips_the_guard_against_a_failing_voice_service(worker, monkeypatch):
    monkeypatch.setattr(casting, "MAX_SKIPS_IN_A_ROW", 1)
    monkeypatch.setattr(casting, "MIN_SKIP_BUDGET", 1)
    monkeypatch.setattr(casting, "MAX_SKIP_SHARE", 0)
    for uid in ("line-1", "line-2", "line-3"):
        write_line(worker, uid, LINK)
    seen = scripted(worker, monkeypatch, {})
    result = run_worker(worker)
    assert seen.skips == [(uid, casting.SKIP_WEB_ADDRESS) for uid in ("line-1", "line-2", "line-3")]
    assert result["status"] == "succeeded", "three links in a row are not three refusals"


def test_a_link_line_a_person_brought_back_is_theirs_to_decide_about(worker, monkeypatch):
    write_line(worker, "line-2", LINK)
    with project_meta_lock(worker.studio.id):
        project = load_project(worker.studio.id)
        casting.record_skips(project, ["line-2"], reason=casting.SKIP_WEB_ADDRESS, by="cleanup")
        casting.restore_skips(project, ["line-2"])
        save_project(project)
    seen = scripted(worker, monkeypatch, {"line-2": [REFUSED]}, automatic=False)
    result = run_worker(worker)
    assert seen.calls.count("line-2") == casting.LINE_ATTEMPTS, "it is tried, because they asked for it back"
    assert seen.skips == [] and result["status"] == "awaiting_review"
    assert result["last_line_failure"]["utterance_id"] == "line-2"


# ---- after a restart, lines already recorded are recognised without re-reading the book ---------

def select_takes(worker, monkeypatch, valid=True, current=True):
    """Every line already has a selected take; the take check itself is covered in test_audiobook_pipeline."""
    from dubbing_service.app.audiobook import pipeline
    rows = load_segments(worker.studio.id)
    for row in rows:
        row.audio_file = f"audio/{row.id}.wav"        # a recorded line: storage derives its selected take from this
    save_segments(worker.studio.id, rows)
    assert all(row.selected_take_id for row in load_segments(worker.studio.id)), "the fixture lines carry selected takes"
    reused = SimpleNamespace(success=True, selected=True, stale=False, qa_passed=True, error=None, reused=True, dispatched=False)
    checked = []

    def resume(project, segment, **kwargs):
        checked.append((segment.id, kwargs.get("audit_missing_qa")))
        return (reused, "selected_take_valid") if valid else (None, "stale_requirement")

    monkeypatch.setattr(pipeline, "_resume_selected_take", resume)
    monkeypatch.setattr(casting, "manifest_still_current", lambda *args, **kwargs: current)
    return checked


def test_lines_that_are_already_recorded_are_not_sent_back_through_rendering(worker, monkeypatch):
    checked = select_takes(worker, monkeypatch)
    result = run_worker(worker)
    assert [e for e in worker.events if e[0] == "render"] == [], "nothing is rendered again"
    assert [sid for sid, _ in checked] == ["line-1", "line-2", "line-3", "line-4"]
    assert all(audit is False for _, audit in checked), "the walk only reads; it never re-audits or writes"
    assert [e for e in worker.events if e[0] == "export"] == [("export", "ch1"), ("export", "ch2")]
    assert result["status"] == "succeeded"


def test_the_walk_reads_the_book_when_it_changes_not_once_per_line(worker, monkeypatch):
    select_takes(worker, monkeypatch)
    run_worker(worker)
    assert worker.checks.book_reads <= 3, "the run's own read, the snapshot, and one more after the first chapter was published"


def test_a_take_that_is_no_longer_valid_is_rendered_the_usual_way(worker, monkeypatch):
    select_takes(worker, monkeypatch, valid=False)
    run_worker(worker)
    assert [e for e in worker.events if e[0] == "render"] == [("render", f"line-{n}") for n in (1, 2, 3, 4)]


def test_a_manifest_that_no_longer_holds_sends_the_line_to_the_guarded_path(worker, monkeypatch):
    select_takes(worker, monkeypatch, current=False)
    run_worker(worker)
    assert [e for e in worker.events if e[0] == "render"] == [("render", f"line-{n}") for n in (1, 2, 3, 4)], \
        "render_utterances makes the refusal, not the shortcut"


def test_a_line_with_no_take_never_builds_the_snapshot(worker, monkeypatch):
    run_worker(worker)
    assert worker.checks.book_reads == 1


def test_the_snapshot_is_read_again_the_moment_the_book_or_project_changes(worker, monkeypatch):
    select_takes(worker, monkeypatch)
    pid, reads = worker.studio.id, []
    real = casting.load_segments
    monkeypatch.setattr(casting, "load_segments", lambda project_id: reads.append(1) or real(project_id))
    recorded = casting._RecordedLines(pid)
    manifest = {"id": "m"}
    assert recorded.record_for("line-1", object(), manifest) is not None and len(reads) == 1
    assert recorded.record_for("line-2", object(), manifest) is not None and len(reads) == 1, "unchanged files are not re-read"
    rows = load_segments(pid)
    rows[0].text = rows[0].text + " Edited."
    save_segments(pid, rows)
    reads.clear()
    assert recorded.record_for("line-2", object(), manifest) is not None and len(reads) == 1, "an edited book is re-read"
    project = load_project(pid)
    project.description = "changed"
    save_project(project)
    reads.clear()
    assert recorded.record_for("line-2", object(), manifest) is not None and len(reads) == 1, "an edited project is re-read"
