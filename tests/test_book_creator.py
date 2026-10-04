"""Isolated canonical import/job tests; signal fixtures, no model or GPU."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
import wave
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_voice_lab_library import source, add_reference_identity  # noqa: F401
from tests.audiobook_helpers import make_tone_wav
from dubbing_service.app.audiobook import book_creator as creator, voice_lab_library as shelf, qa
from dubbing_service.app.models.project import SegmentTake
from dubbing_service.app.models.jobs import SynthesisJob
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.storage.jobs import load_jobs, upsert_job
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from dubbing_service.app.routers.book_creator import router


def epub(*, unsafe=None, encryption=False, missing=False, repeated=False):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
        archive.writestr("OEBPS/book.opf", '<package><metadata><title>Local fixture book</title><creator>Fixture author</creator><language>en</language></metadata><manifest><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/><item id="two" href="two.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/><itemref idref="' + ('one' if repeated else 'absent' if missing else 'two') + '"/></spine></package>')
        archive.writestr("OEBPS/one.xhtml", '<html><head><title>Not spoken head</title></head><body><h1>Chapter 258</h1><p>Fixture one. Another sentence.</p><p>[Alice]: Keep this speaker prefix.</p></body></html>')
        archive.writestr("OEBPS/two.xhtml", '<html><body><h1>Chapter 259</h1><p>Fixture two.</p></body></html>')
        if unsafe:
            archive.writestr(unsafe, "unsafe")
        if encryption:
            archive.writestr("META-INF/encryption.xml", "<encryption/>")
    return target.getvalue()


@pytest.fixture
def store(source, monkeypatch):
    record = add_reference_identity(source)
    published = shelf.publish_version(source[2].id, record["key"])
    calls = {"render": [], "generated": [], "encode": 0}
    def render(project_id, ids, **kwargs):
        assert kwargs == {"renderer": kwargs["renderer"], "resume": True, "run_qa": True, "promote": True, "job_id": kwargs["job_id"]}
        linked_job = creator.get_job(project_id, kwargs["job_id"])
        assert linked_job is not None and set(ids).issubset(linked_job.segment_ids)
        assert linked_job.project_id == project_id and linked_job.status.value == "running"
        project = load_project(project_id); segments = load_segments(project_id)
        segment = next(s for s in segments if s.id == ids[0]); calls["render"].append(segment.id)
        role = next(r for r in project.roles if r.id == segment.role_id)
        requirement = creator.pipeline.requirement_for(project, segment, renderer_id=role.tts_backend)
        old = creator.takes.selected_take(segment)
        reused = bool(old and old.input_fingerprint == requirement.fingerprint() and old.qa_report and old.qa_report.passed)
        if not reused:
            calls["generated"].append(segment.id)
            base = source[0] / project_id
            path = make_tone_wav(base / f"audio/fixture-{segment.id}-{len(segment.takes)}.wav", seconds=max(.5, len(segment.text) * .03))
            identity = role.voice_identity
            take = SegmentTake(id="take_" + segment.id + "_" + str(len(segment.takes)), audio_file=path.relative_to(base).as_posix(),
                audio_sha256=creator.file_sha256(path), role_id=role.id, source_text_hash=creator.text_sha256(segment.text),
                spoken_text_revision_id=segment.spoken_text_revision_id, synthesis_text=segment.text,
                input_fingerprint=requirement.fingerprint(), fingerprint_components=requirement.components(), renderer_id=role.tts_backend,
                engine_id=role.tts_backend, generation_metadata={"character_voice_identity": identity.model_dump(mode="json"),
                    "provider_calls": [{"identity_id": identity.id, "prompt_sha256": identity.prompt_sha256}]})
            take.qa_report = qa.qa_take_audio(take_id=take.id, audio_path=path, text=segment.text, expected_sha256=take.audio_sha256)
            assert take.qa_report.passed
            segment.takes.append(take); segment.selected_take_id = take.id
            save_segments(project_id, segments)
        return [creator.pipeline.RenderRecord(segment_id=segment.id, take_id=segment.selected_take_id,
            success=True, selected=True, qa_passed=True, reused=reused, dispatched=not reused)]
    monkeypatch.setattr(creator.pipeline, "render_utterances", render)
    def encode(master, target):
        calls["encode"] += 1
        with target.open("xb") as stream:
            stream.write(master.read_bytes())  # Explicit PCM test fixture, not real MP3/speech.
    def duration(path):
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    monkeypatch.setattr(creator, "_mp3", encode)
    monkeypatch.setattr(creator, "_decoded_duration", duration)
    monkeypatch.setattr(creator, "_loudness", lambda _: -18.5)
    return source, published, calls


def imported(store, data=b"Chapter 1\nHello there.\nChapter 2\nAnother line.\n", filename="fixture.txt", request_id="import-one"):
    return creator.import_book(data, filename, request_id)


def render(store, book, **kwargs):
    return creator.render_book(book["project_id"], chapter_ids=[c["id"] for c in book["chapters"]],
        library_voice_id=store[1]["library_voice_id"], voice_version=store[1]["version"], request_id="render-one",
        renderer=SimpleNamespace(backend="qwen3tts"), **kwargs)


def test_utf8_import_exact_offsets_quotes_prefixes_crlf_and_single_narrator(store):
    data = 'Chapter 1\r\n [Alice]: “Hello.”  \r\nA second line.\r\n'.encode()
    book = imported(store, data)
    project, meta, rows = creator._source(book["project_id"])
    assert len(book["chapters"]) == 1 and book["mode"] == "single_narrator"
    assert any('[Alice]: “Hello.”' == s.text for s in rows)
    assert {s.role_id for s in rows} == {meta["narrator_role_id"]}
    assert all(s.attribution.source == "inline" and not s.attribution.reviewed for s in rows)
    assert not project.character_cast.policy.enabled and project.character_cast.policy.audition_mode == "off"
    assert not store[2]["generated"]


def test_chosen_voice_accepts_verified_legacy_or_canonical_identity_paths(store):
    book = imported(store)
    render(store, book)
    project, meta, _ = creator._source(book["project_id"])
    record = next(r for r in project.voice_lab.voices.values()
                  if r.get("library_voice_id") == meta["voice_selection"]["library_voice_id"]
                  and r.get("library_version") == meta["voice_selection"]["voice_version"])
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
    source_identity = CharacterVoiceIdentity.model_validate(record["identity"])

    # Existing projects can still carry the exact paths from the immutable
    # library record; new assignments use only the canonical relocation.
    role.voice_identity = source_identity.model_copy(update={"role_id": role.id})
    save_project(project)
    project, meta, _ = creator._source(book["project_id"])
    assert creator._chosen_voice(project, meta).voice_identity == role.voice_identity

    role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
    role.voice_identity = voice_lab_identity = creator.voice_lab.identity_for_assignment(source_identity).model_copy(
        update={"role_id": role.id})
    save_project(project)
    project, meta, _ = creator._source(book["project_id"])
    assert creator._chosen_voice(project, meta).voice_identity == voice_lab_identity

    role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
    role.voice_identity = role.voice_identity.model_copy(update={"id": "different-identity"})
    save_project(project)
    project, meta, _ = creator._source(book["project_id"])
    with pytest.raises(creator.BookCreatorError, match="assignment changed"):
        creator._chosen_voice(project, meta)

    project, meta, _ = creator._source(book["project_id"])
    role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
    role.voice_identity = voice_lab_identity
    save_project(project)
    assigned_reference = store[0][0] / project.id / voice_lab_identity.reference_audio
    assigned_reference.write_bytes(b"tampered test reference")
    project, meta, _ = creator._source(book["project_id"])
    with pytest.raises(creator.BookCreatorError, match="conditioning is missing or changed"):
        creator._chosen_voice(project, meta)


def test_epub_uses_actual_spine_heading_and_full_local_text_not_import_position(store):
    book = imported(store, epub(), "book.epub")
    assert [c["title"] for c in book["chapters"]] == ["Chapter 258", "Chapter 259"]
    assert book["name"] == "Local fixture book"
    _, _, rows = creator._source(book["project_id"])
    assert any(s.text == "[Alice]: Keep this speaker prefix." for s in rows)
    assert not any("Not spoken head" in s.text for s in rows)


@pytest.mark.parametrize("kind", ["unsafe", "encryption", "missing", "repeated", "badzip", "badutf8", "binary", "empty", "badextension"])
def test_bad_sources_rejected_before_new_project(store, kind):
    root = store[0][0]; before = set(root.glob("*/project.json"))
    data, filename = epub(), "book.epub"
    if kind == "unsafe": data = epub(unsafe="../escape.txt")
    elif kind == "encryption": data = epub(encryption=True)
    elif kind == "missing": data = epub(missing=True)
    elif kind == "repeated": data = epub(repeated=True)
    elif kind == "badzip": data = b"not a zip"
    elif kind == "badutf8": data, filename = b"\xff\xfe", "book.txt"
    elif kind == "binary": data, filename = b"hello\x00world", "book.txt"
    elif kind == "empty": data, filename = b"  \n", "book.txt"
    elif kind == "badextension": filename = "book.exe"
    with pytest.raises(ValueError): imported(store, data, filename)
    assert set(root.glob("*/project.json")) == before


def test_import_idempotency_and_explicit_new_recording(store):
    first = imported(store); again = imported(store)
    assert again["project_id"] == first["project_id"] and again["reused"]
    with pytest.raises(creator.BookCreatorError, match="different file"):
        imported(store, b"different", request_id="import-one")
    fresh = imported(store, request_id="import-two")
    assert fresh["project_id"] != first["project_id"]


def test_complete_multi_chapter_request_uses_existing_voice_takes_and_exports(store):
    source_root = store[0][0] / store[0][2].id
    before = {p.name: p.read_bytes() for p in source_root.glob("*.json")}
    book = imported(store)
    result = render(store, book)
    assert result["status"] == "succeeded", result
    assert result["completed"] == result["total"] == 4
    assert len(result["outputs"]) == 3  # each chapter and one continuous combined recording
    project = load_project(book["project_id"])
    assert project.metadata["book_creator"]["voice_selection"] == {"library_voice_id": store[1]["library_voice_id"], "voice_version": store[1]["version"]}
    assert not project.voice_lab.preferences and not project.voice_lab.preference_history
    assert all(o["human_review"] == "pending" and o["technical_qa"] == "passed" for o in result["outputs"])
    assert before == {p.name: p.read_bytes() for p in source_root.glob("*.json")}
    assert all("audio_file" not in o for o in result["outputs"])
    listed = creator.list_books()
    assert listed["books"][0]["project_id"] == book["project_id"]
    assert listed["voices"][0]["versions"][0]["audio_url"].startswith("/api/v1/voice-library/versions/")


def test_same_request_is_idempotent_no_duplicate_generation_or_export(store):
    book = imported(store)
    first = render(store, book)
    before = (list(store[2]["generated"]), store[2]["encode"])
    again = render(store, book)
    assert again["outputs"] == first["outputs"]
    assert before == (store[2]["generated"], store[2]["encode"])
    assert len(load_jobs(book["project_id"])) == 1


def test_running_request_duplicate_and_browser_restart_resume(store):
    book = imported(store); tasks = []
    response = render(store, book, schedule=lambda fn, *args: tasks.append((fn, args)))
    assert response["status"] == "queued"
    assert render(store, book)["job_id"] == response["job_id"] and len(tasks) == 1
    fn, args = tasks[0]
    final = fn(*args)
    assert final["status"] == "succeeded"
    assert creator.request_status(book["project_id"], "render-one")["outputs"] == final["outputs"]


def test_busy_production_yields_then_same_request_resumes(store):
    book = imported(store)
    other = store[0][2]
    lease = try_register_running_batch(other.id)
    try:
        result = render(store, book)
        assert result["status"] == "interrupted" and not store[2]["generated"]
    finally:
        unregister_running_batch(lease)
    assert render(store, book)["status"] == "succeeded"


def test_pending_production_has_priority_without_starting_worker(store):
    book = imported(store)
    upsert_job(store[0][2].id, SynthesisJob(id="already-queued", project_id=store[0][2].id,
        job_type="batch_synthesize", segment_ids=["utterance"]))
    result = render(store, book)
    assert result["status"] == "interrupted" and "priority" in result["message"]
    assert not store[2]["generated"]


@pytest.mark.parametrize("change", ["text", "snapshot", "offset", "retire", "chapter"])
def test_source_changes_block_before_synthesis(store, change):
    book = imported(store); project = load_project(book["project_id"]); rows = load_segments(project.id)
    if change == "text": rows[0].text += " changed"
    elif change == "offset": rows[0].source_anchor.start_offset = 2
    elif change == "snapshot": (store[0][0] / project.id / project.source_revisions[0].snapshot_file).write_bytes(b"changed isolated source")
    elif change == "retire": rows.pop()
    elif change == "chapter": project.chapters.pop()
    save_project(project); save_segments(project.id, rows)
    with pytest.raises(creator.BookCreatorError): render(store, book)
    assert not store[2]["generated"]


def test_recast_and_request_rebinding_are_explicitly_rejected(store):
    book = imported(store); render(store, book)
    with pytest.raises(creator.BookCreatorError, match="another chapter selection or voice"):
        creator.render_book(book["project_id"], chapter_ids=[book["chapters"][0]["id"]],
            library_voice_id=store[1]["library_voice_id"], voice_version=store[1]["version"], request_id="render-one")
    with pytest.raises(creator.BookCreatorError, match="another narrator"):
        creator.render_book(book["project_id"], chapter_ids=[book["chapters"][0]["id"]],
            library_voice_id="different", voice_version="different", request_id="render-two")


def test_natural_scene_pause_is_not_synthesized(store):
    book = imported(store, b"Chapter 1\nBefore.\n******\nAfter.\n")
    result = render(store, book)
    assert result["status"] == "succeeded", result
    project = load_project(book["project_id"])
    breaks = project.metadata["book_creator"]["scene_breaks"]
    assert len(breaks) == 1 and breaks[0] not in store[2]["generated"]
    exported = next(iter(project.metadata["book_creator"]["exports"].values()))
    gap = next(row for row in exported["map"] if row["kind"] == "scene_break")
    assert gap["end_seconds"] - gap["start_seconds"] == 1.25


def test_audio_route_range_and_hash_verification(store):
    book = imported(store); result = render(store, book)
    app = FastAPI(); app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        output = result["outputs"][0]
        response = client.get(output["audio_url"], headers={"Range": "bytes=0-15"})
        assert response.status_code == 206 and len(response.content) == 16
        assert response.headers["etag"] == '"' + output["audio_sha256"] + '"'
        path, _, _ = creator.audio_artifact(book["project_id"], output["export_id"])
        path.write_bytes(b"tampered isolated signal")
        assert client.get(output["audio_url"]).status_code == 409


def test_multipart_import_and_validation_through_normal_router(store):
    app = FastAPI(); app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        response = client.post("/api/v1/book-creator/import", files={"file": ("own.txt", b"Own tiny line.", "text/plain")}, data={"request_id": "upload-one", "name": "My local book"})
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "My local book"
        assert client.get("/api/v1/book-creator").json()["books"][0]["name"] == "My local book"
        bad = client.post("/api/v1/book-creator/import", files={"file": ("own.txt", b"Own tiny line.")}, data={"request_id": "../../outside"})
        assert bad.status_code == 409


def test_blocking_failure_resumes_only_unfinished_audio(store, monkeypatch):
    book = imported(store)
    normal = creator.pipeline.render_utterances
    calls = 0
    def fail_second(project_id, ids, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return [creator.pipeline.RenderRecord(segment_id=ids[0], take_id="rejected-fixture", success=False,
                selected=False, qa_passed=False, error="explicit blocking test failure")]
        return normal(project_id, ids, **kwargs)
    monkeypatch.setattr(creator.pipeline, "render_utterances", fail_second)
    first = render(store, book)
    assert first["status"] == "interrupted" and len(store[2]["generated"]) == 1
    monkeypatch.setattr(creator.pipeline, "render_utterances", normal)
    second = render(store, book)
    assert second["status"] == "succeeded"
    assert len(store[2]["generated"]) == len(set(store[2]["generated"])) == 4
    assert first["job_id"] == second["job_id"]


def test_stale_voice_during_render_stops_remaining_source_and_exports(store, monkeypatch):
    book = imported(store)
    normal = creator.pipeline.render_utterances
    def mutate_after_first(project_id, ids, **kwargs):
        record = normal(project_id, ids, **kwargs)
        project = load_project(project_id)
        project.roles[0].voice_identity.id = "changed-after-dispatch"
        save_project(project)
        return record
    monkeypatch.setattr(creator.pipeline, "render_utterances", mutate_after_first)
    result = render(store, book)
    assert result["status"] == "interrupted" and len(store[2]["generated"]) == 1
    assert not result["outputs"] and store[2]["encode"] == 0


def test_worker_is_ensured_only_for_explicit_render_and_receives_job_identity(store, monkeypatch):
    from dubbing_service.app.audiobook import voice_lab_runtime
    ensured = []
    # The pipeline is a signal fixture; readiness tests must not contact a real worker.
    monkeypatch.setattr(creator.BackendRenderer, "_get_provider", lambda self: SimpleNamespace(fixture_only=True))
    monkeypatch.setattr(voice_lab_runtime, "ensure_backend_ready", lambda backend, **kwargs: ensured.append((backend, kwargs)))
    book = imported(store)
    creator.view(book["project_id"]); creator.list_books()
    assert ensured == []
    result = creator.render_book(book["project_id"], chapter_ids=[book["chapters"][0]["id"]],
        library_voice_id=store[1]["library_voice_id"], voice_version=store[1]["version"], request_id="explicit-worker")
    assert result["status"] == "succeeded", result
    assert ensured == [("qwen3tts", {"job_id": result["job_id"], "project_id": book["project_id"]})]


def test_hard_interruption_during_export_never_reports_early_success_and_resumes(store, monkeypatch):
    book = imported(store)
    normal = creator._export
    exported = 0
    def crash_before_second(project_id, request_id, chapters, binding):
        nonlocal exported
        status = creator.request_status(project_id, request_id)
        assert status["status"] == "running" and status["completed"] == status["total"] == 4
        assert creator.get_job(project_id, status["job_id"]).finished_at is None
        if exported == 1:
            raise KeyboardInterrupt("isolated process-loss simulation")
        exported += 1
        return normal(project_id, request_id, chapters, binding)
    monkeypatch.setattr(creator, "_export", crash_before_second)
    with pytest.raises(KeyboardInterrupt):
        render(store, book)
    interrupted = creator.request_status(book["project_id"], "render-one")
    assert interrupted["status"] == "interrupted" and len(interrupted["outputs"]) == 1
    generated = list(store[2]["generated"])
    monkeypatch.setattr(creator, "_export", normal)
    final = render(store, book)
    assert final["status"] == "succeeded" and len(final["outputs"]) == 3
    assert store[2]["generated"] == generated and store[2]["encode"] == 3


def test_legacy_succeeded_job_with_incomplete_exports_is_resumable(store, monkeypatch):
    book = imported(store)
    normal = creator._export
    monkeypatch.setattr(creator, "_export", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt("legacy crash fixture")))
    with pytest.raises(KeyboardInterrupt):
        render(store, book)
    status = creator.request_status(book["project_id"], "render-one")
    creator.mutate_job(book["project_id"], status["job_id"], lambda job: setattr(job, "status", creator.SynthesisJobStatus.SUCCEEDED))
    retained = list(store[2]["generated"])
    pending = creator.request_status(book["project_id"], "render-one")
    assert pending["status"] == "interrupted" and not pending["outputs"]
    monkeypatch.setattr(creator, "_export", normal)
    final = render(store, book)
    assert final["status"] == "succeeded" and len(final["outputs"]) == 3
    assert store[2]["generated"] == retained


@pytest.mark.parametrize("change", ["identity", "backend"])
def test_new_request_cannot_relabel_external_recast_as_locked_library_voice(store, change):
    book = imported(store)
    first = render(store, book)
    retained = list(store[2]["generated"])
    project = load_project(book["project_id"])
    if change == "identity":
        project.roles[0].voice_identity.id = "external-recast"
    else:
        project.roles[0].tts_backend = "qwen3tts_cv"
    save_project(project)
    with pytest.raises(creator.BookCreatorError, match="assignment changed"):
        creator.render_book(project.id, chapter_ids=[book["chapters"][0]["id"]],
            library_voice_id=store[1]["library_voice_id"], voice_version=store[1]["version"], request_id="new-request",
            renderer=SimpleNamespace(backend="qwen3tts"))
    assert store[2]["generated"] == retained
    current = load_project(project.id)
    assert "new-request" not in current.metadata["book_creator"]["requests"]
    assert [r["export_id"] for r in creator.request_status(project.id, "render-one")["outputs"]] == [r["export_id"] for r in first["outputs"]]
