"""Private AutoCast review grants and casting notification delivery boundaries."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import casting_review, delivery
from dubbing_service.app.audiobook.delivery import notify
from dubbing_service.app.audiobook.delivery.server import create_app, shared_lab_project
from dubbing_service.app.models.project import Role
from dubbing_service.app.routers import book_creator as book_creator_router
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from tests.audiobook_helpers import use_projects_root


def _receipt_project(project_id=None):
    project = create_project("Private AutoCast review")
    project.metadata["autocast_import"] = {
        "schema": "autocast-import-v1", "mode": "autocast",
        "private_review_recipient": "brian",
    }
    save_project(project)
    return project


def test_import_route_grants_only_trusted_loopback_proxy_autocast(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    calls = []

    def fake_import(data, filename, request_id, name=None):
        project = _receipt_project()
        project.metadata["autocast_import"].pop("private_review_recipient")
        save_project(project)
        calls.append((data, filename, request_id, name, project.id))
        return {"project_id": project.id, "mode": "autocast", "status": "ready",
                "review_url": f"/web/voice-lab.html?project={project.id}", "reused": False}

    monkeypatch.setattr("dubbing_service.app.audiobook.autocast_import.import_autocast", fake_import)
    app = FastAPI()
    app.include_router(book_creator_router.router, prefix="/api/v1")
    with TestClient(app, client=("127.0.0.1", 8001)) as client:
        response = client.post("/api/v1/book-creator/import",
            files={"file": ("book.txt", b"local book", "text/plain")},
            data={"request_id": "trusted", "mode": "autocast"},
            headers={"X-Private-Review-Recipient": "brian"})
    assert response.status_code == 200
    assert calls[0][:4] == (b"local book", "book.txt", "trusted", None)
    config = {"project_dir": str(root / "owner-project")}
    assert shared_lab_project(config, calls[0][4])
    assert load_project(calls[0][4]).metadata["autocast_import"]["private_review_recipient"] == "brian"

    app2 = FastAPI()
    app2.include_router(book_creator_router.router, prefix="/api/v1")
    with TestClient(app2, client=("198.51.100.40", 8001)) as client:
        response = client.post("/api/v1/book-creator/import",
            files={"file": ("book.txt", b"local book", "text/plain")},
            data={"request_id": "untrusted", "mode": "autocast"},
            headers={"X-Private-Review-Recipient": "brian"})
    assert response.status_code == 200
    assert not shared_lab_project(config, calls[1][4])
    assert "private_review_recipient" not in load_project(calls[1][4]).metadata["autocast_import"]


@pytest.mark.parametrize("receipt,expected", [
    ({"mode": "autocast", "private_review_recipient": "brian"}, True),
    ({"mode": "single_narrator", "private_review_recipient": "brian"}, False),
    ({"mode": "autocast", "private_review_recipient": "mk"}, False),
    ({"mode": "autocast"}, False),
])
def test_shared_project_requires_exact_canonical_autocast_receipt(monkeypatch, tmp_path, receipt, expected):
    root = use_projects_root(monkeypatch, tmp_path)
    owner = create_project("Owner")
    target = create_project("Target")
    target.metadata["autocast_import"] = receipt
    save_project(target)
    assert shared_lab_project({"project_dir": str(root / owner.id)}, target.id) is expected
    assert shared_lab_project({"project_dir": str(root / owner.id)}, "../" + target.id) is False


def test_shared_project_rejects_receipt_reached_through_symlink(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    owner = create_project("Owner")
    external = tmp_path / "external-project"
    external.mkdir()
    (external / "project.json").write_text(json.dumps({"metadata": {"autocast_import": {
        "mode": "autocast", "private_review_recipient": "brian"}}}), encoding="utf-8")
    link = root / "alias-project"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this Windows environment")
    assert shared_lab_project({"project_dir": str(root / owner.id)}, "alias-project") is False


def test_private_voice_lab_casting_routes_are_whitelisted_and_media_keys_are_bounded(tmp_path, monkeypatch):
    project_root = tmp_path / "canonical" / "project"
    project_root.mkdir(parents=True)
    config = {"project_dir": str(project_root), "state_dir": str(project_root / "delivery"),
        "port": 8778, "listening_base_url": "https://desktop.example.ts.net:8443",
        "allowed_audio_roots": [], "recipients": {"brian": {"id": "brian", "enabled": True,
            "tailscale_login": "brian@example.test", "allow_feedback": True, "allow_download": True}}}
    sent = []

    class Upstream:
        status_code = 200
        headers = {"content-type": "application/json"}
        async def aread(self): return b'{"accepted":true}'
        def json(self): return {"accepted": True}
        async def aclose(self): pass

    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        def build_request(self, method, url, **kwargs): return (method, url, kwargs)
        async def send(self, request, stream=True): sent.append(request); return Upstream()
        async def aclose(self): pass

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1:8778", client=("127.0.0.1", 9012),
                    headers={"Tailscale-User-Login": "brian@example.test"}) as client:
        assert client.get("/api/voice-lab/project/casting").json() == {"accepted": True}
        assert sent[-1][1].endswith("/api/v1/projects/project/voice-lab/casting")
        headers = {"X-Review-Request": "1"}
        payload = {"request_id": "round-request", "role_id": "role-a", "expected_round_revision": 0,
            "expected_assignment_revision": 0, "intent": "generate"}
        assert client.post("/api/voice-lab/project/casting/round", json=payload,
            headers=headers).json() == {"accepted": True}
        assert sent[-1][1].endswith("/api/v1/projects/project/voice-lab/casting/round")
        before = len(sent)
        assert client.get("/api/voice-lab/project/casting/private-data").status_code == 404
        assert len(sent) == before
        assert client.get("/api/voice-lab/project/audio/v_" + "a" * 28).status_code == 200
        before = len(sent)
        assert client.get("/api/voice-lab/project/audio/not-a-media-key").status_code == 404
        assert len(sent) == before


def test_casting_ready_notification_uses_normal_priority_and_existing_at_most_once_ledger(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    owner = create_project("Owner")
    project = _receipt_project()
    project.metadata["autocast_import"]["private_review_recipient"] = "brian"
    role_id = "role-mara"
    audio_hash = "a" * 64
    project.roles = [Role(id=role_id, name="Mara", tts_backend="qwen3tts")]
    project.voice_lab.casting["roles"][role_id] = {"round": {
        "id": "round-1", "revision": 1, "status": "awaiting_review",
        "replacement": False, "completed_at": "2026-01-01T00:00:00+00:00",
        "notification": {"state": "pending"},
        "candidates": [{"status": "ready", "key": "voice-a", "audio_sha256": audio_hash}],
    }}
    save_project(project)
    monkeypatch.setattr(casting_review.lab, "_verified_record", lambda *args, **kwargs: {"key": "voice-a"})
    recipient = {"id": "brian", "enabled": True, "notification_transport": "pushover",
        "allow_download": False, "allow_feedback": True}
    delivery_config = {"project_dir": str(root / owner.id), "notification_transport_root": str(tmp_path / "transport"),
        "listening_base_url": "https://desktop.example.ts.net:8443"}
    monkeypatch.setattr("dubbing_service.app.audiobook.book_creator_notifications._private_config",
                        lambda: (delivery_config, recipient))
    provider_calls = []
    safe_config = SimpleNamespace(app_token="app-secret", user_key="user-secret", device="phone", timeout_seconds=2.0)

    class Message:
        def __init__(self, **values): self.__dict__.update(values)

    class Publisher:
        def send(self, message, *, timeout):
            provider_calls.append((message, timeout))
            return SimpleNamespace(state="accepted", attempts=1, http_status=200, request_id="safe-request")

    publisher = Publisher()
    monkeypatch.setattr(notify, "_load_transport", lambda path: (safe_config, publisher, Message, object()))

    first = casting_review.notify_ready(project.id)
    assert first["state"] == "accepted"
    message, timeout = provider_calls[0]
    assert message.title == "🎙 Voice ready: Mara"
    assert message.priority == 0
    assert message.url.endswith(f"/voice-lab?project={project.id}&character={role_id}")
    assert timeout == 2.0

    # Simulate a crash after the ledger accepted the push but before its project
    # projection was updated. Retrying must read the prior ledger claim.
    project = load_project(project.id)
    project.voice_lab.casting["roles"][role_id]["round"]["notification"] = {"state": "pending"}
    save_project(project)
    second = casting_review.notify_ready(project.id)
    assert second["state"] == "accepted"
    assert len(provider_calls) == 1
    assert len(list((root / project.id / "delivery/notifications/intents").glob("*.json"))) == 1
    assert len(list((root / project.id / "delivery/notifications/outcomes").glob("*.json"))) == 1


def test_casting_notification_constructs_body_contract_before_at_most_once_claim(monkeypatch, tmp_path):
    """The shared trip RenderedNotification contract uses `body`, not `message`."""
    recipient = {"id": "brian", "enabled": True, "notification_transport": "pushover"}
    batch = {"project_id": "project-1", "batch_id": "batch-1", "recipient_id": "brian",
        "kind": "casting_review", "event": "CHARACTER_REVIEW_READY", "book_title": "Elf_1_10",
        "characters": [{"role_id": "role-a", "name": "Saphienne", "options": 3, "audio_sha256": "a" * 64}],
        "characters_total": 11}
    listening_url = "https://desktop.example.ts.net:8443/voice-lab?project=project-1&character=role-a"
    sent = []
    config = SimpleNamespace(app_token="unused", user_key="unused", timeout_seconds=3.0)

    class RenderedNotificationContract:
        """Mirror the public immutable constructor: title/body/url/url_title/priority."""
        def __init__(self, *, title, body, url=None, url_title=None, priority=0):
            self.title, self.body = title, body
            self.url, self.url_title, self.priority = url, url_title, priority

    class Publisher:
        def send(self, message, *, timeout):
            sent.append((message, timeout))
            return SimpleNamespace(state="accepted", attempts=1, http_status=200,
                request_id="safe-request")

    publisher = Publisher()
    monkeypatch.setattr(notify, "_load_transport",
        lambda path: (config, publisher, RenderedNotificationContract, object()))
    result = notify.send_notification(batch, recipient, state_dir=tmp_path / "project",
        transport_root=tmp_path / "transport", listening_url=listening_url)

    assert result["state"] == "accepted"
    assert len(sent) == 1
    message, timeout = sent[0]
    assert message.title == "🎙 Voice ready: Saphienne"
    assert message.body == ("3 auditions are ready.\nTap to listen, rate, and choose a voice.\n"
                            "Project: Elf_1_10")
    assert message.url == listening_url
    assert message.url_title == "Review Saphienne"
    assert message.priority == 0
    assert not hasattr(message, "attachment_path")
    assert timeout == 3.0
    state = tmp_path / "project" / "notifications"
    assert len(list((state / "intents").glob("*.json"))) == 1
    assert len(list((state / "outcomes").glob("*.json"))) == 1
