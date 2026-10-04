"""Catalog and application-maintenance boundaries; no live jobs or pushes."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_book_creator import source, store, imported, render  # noqa: F401
from tests.test_voice_lab_delivery import private_lab  # noqa: F401
from dubbing_service.app.audiobook import recordings_library as catalog
from dubbing_service.app.audiobook.book_maintenance import run_book_maintenance
from dubbing_service.app.routers.voice_library import router
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.audiobook.fingerprint import file_sha256


def test_creator_recordings_are_visible_without_opening_book_and_tamper_blocks_media(store, monkeypatch):
    monkeypatch.setattr(catalog, "load_config", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    book = imported(store)
    finished = render(store, book)
    rows = catalog.list_recordings()["recordings"]
    assert len(rows) == 3
    assert {r["export_id"] for r in rows} == {r["export_id"] for r in finished["outputs"]}
    assert all(r["project_id"] == book["project_id"] and r["kind"] == "book_creator" for r in rows)
    assert all("book=" + book["project_id"] in r["review_url"] for r in rows)
    assert not any(k.endswith("_path") or k.endswith("_file") for row in rows for k in row)
    app = FastAPI(); app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    row = rows[0]
    url = f"/api/v1/voice-library/recordings/{row['project_id']}/{row['export_id']}/audio"
    response = client.get(url, headers={"Range": "bytes=0-15"})
    assert response.status_code == 206 and len(response.content) == 16
    path, _ = catalog.audio_artifact(row["project_id"], row["export_id"])
    path.write_bytes(b"changed")
    assert client.get(url).status_code == 409


def test_private_catalog_filters_unshared_and_rewrites_only_links(private_lab):
    _, client, state = private_lab
    state.upstream.body = {"recordings": [
        {"project_id": "secret", "kind": "voice_lab", "label": "private"},
        {"project_id": "isolated", "kind": "voice_lab", "audio_url": "/api/v1/voice-library/recordings/isolated/lab_" + "a"*28 + "/audio",
         "review_url": "/web/voice-lab.html?project=isolated", "label": "/web/voice-lab.html?unchanged"},
        {"project_id": "newbook", "kind": "book_creator", "audio_url": "/api/v1/book-creator/newbook/audio/mp3_abc",
         "review_url": "/web/voice-lab.html?home=1&book=newbook&output=mp3_abc"}]}
    response = client.get("/api/voice-library/recordings")
    assert response.status_code == 200
    rows = response.json()["recordings"]
    assert [row["project_id"] for row in rows] == ["isolated", "newbook"]
    assert rows[0]["audio_url"].startswith("/api/voice-library/")
    assert rows[0]["review_url"] == "/voice-lab?project=isolated"
    assert rows[0]["label"] == "/web/voice-lab.html?unchanged"
    assert rows[1]["review_url"].startswith("/voice-lab?home=1&book=newbook")
    before = len(state.calls)
    assert client.get("/api/voice-library/recordings/secret/lab_" + "a"*28 + "/audio").status_code == 404
    assert len(state.calls) == before


def test_voice_lab_media_link_does_not_silently_follow_new_audio_version(source, monkeypatch):
    monkeypatch.setattr(catalog, "load_config", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    root, _, project, _, audio = source
    project.voice_lab.excerpt = {"audio_file": audio.relative_to(root / project.id).as_posix(),
        "audio_sha256": file_sha256(audio), "duration_seconds": .25, "chapter_title": "Fixture chapter"}
    save_project(project)
    row = catalog.list_recordings()["recordings"][0]
    project.voice_lab.excerpt["audio_sha256"] = "f" * 64
    save_project(project)
    with pytest.raises(ValueError, match="unavailable"):
        catalog.audio_artifact(project.id, row["export_id"])


def test_delivery_catalog_is_deduplicated_and_never_exposes_an_unapproved_root(source, monkeypatch):
    root, _, project, _, audio = source
    state = root / project.id / "delivery"
    batch = {"project_id": project.id, "book_title": "Fixture book", "chapters": [
        {"chapter_id": "chapter", "chapter_number": 259, "title": "Snow fixture", "revision": "one",
         "audio_path": str(audio), "audio_sha256": file_sha256(audio), "duration_seconds": .25}]}
    catalog.store.publish_batch(state, batch, "brian")
    catalog.store.publish_batch(state, batch, "mk")
    config = {"state_dir": str(state), "allowed_audio_roots": [str(audio.parent)],
              "listening_base_url": "https://private.example.ts.net", "recipients": {"brian": {"enabled": True}}}
    monkeypatch.setattr(catalog, "load_config", lambda: config)
    rows = catalog.list_recordings()["recordings"]
    assert len(rows) == 1 and rows[0]["kind"] == "delivery"
    assert rows[0]["label"] == "Snow fixture" and "audio_path" not in rows[0]
    config["allowed_audio_roots"] = [str(root / "unshared")]
    assert catalog.list_recordings()["recordings"] == []


@pytest.mark.parametrize("operation", ["automation", "notify"])
def test_private_request_controls_require_existing_identity_and_csrf(private_lab, operation):
    _, client, state = private_lab
    url = "/api/book-creator/newbook/requests/request1/" + operation
    assert client.post(url, json={"paused": True}).status_code == 403
    assert not state.calls
    assert client.post(url, json={"paused": True}, headers={"X-Review-Request": "1"}).status_code == 200
    assert state.calls[-1].url.endswith("/api/v1/book-creator/newbook/requests/request1/" + operation)


@pytest.mark.parametrize("url", ["/api/voice-library/recordings/isolated/lab_" + "a"*28 + "/audio",
                                "/api/book-creator/newbook/audio/mp3_" + "b"*28])
def test_private_catalog_download_permission_and_query_are_retained(private_lab, url):
    config, client, state = private_lab
    assert client.get(url + "?download=true").status_code == 403
    assert not state.calls
    config["recipients"]["brian"]["allow_download"] = True
    assert client.get(url + "?download=true").status_code == 200
    assert state.calls[-1].url.endswith("?download=true")


def test_maintenance_does_not_block_ticks_and_waits_for_its_render_before_shutdown():
    async def scenario():
        stop = asyncio.Event(); started = threading.Event(); finished = threading.Event(); release = threading.Event()
        calls = []
        def render():
            started.set(); release.wait(3); finished.set()
        def recovery_tick(*, schedule):
            if not calls:
                schedule(render)
            calls.append("recovery")
        def notify_tick():
            calls.append("notify")
        task = asyncio.create_task(run_book_maintenance(stop, interval=.01, recovery_tick=recovery_tick, notification_tick=notify_tick))
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.sleep(.05)
        assert calls.count("recovery") >= 2 and "notify" in calls
        stop.set(); await asyncio.sleep(.03)
        assert not task.done() and not finished.is_set()
        release.set(); await asyncio.wait_for(task, 2)
        assert finished.is_set()
    asyncio.run(scenario())
