"""Casting notification retries use the existing transport and ledger boundary."""
from datetime import datetime, timedelta, timezone

import pytest

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import book_creator_notifications
from dubbing_service.app.audiobook.delivery import notify, server
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from tests.audiobook_helpers import use_projects_root


@pytest.fixture
def ready(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = create_project("TEST notification recovery only")
    project.voice_lab.casting["roles"]["test-role"] = {"round": {
        "id": "test-round", "revision": 1, "status": "awaiting_review",
        "candidates": [{"status": "ready", "key": "test-audio", "audio_sha256": "a" * 64}],
        "notification": {"state": "pending"},
    }}
    save_project(project)
    monkeypatch.setattr(casting.lab, "_verified_record", lambda *args: {})
    monkeypatch.setattr(book_creator_notifications, "_private_config", lambda: (
        {"notification_transport_root": str(tmp_path), "listening_base_url": "https://fixture.ts.net:8443"},
        {"id": "brian", "enabled": True}))
    monkeypatch.setattr(server, "shared_lab_project", lambda *args: True)
    return project.id


def state(pid):
    return load_project(pid).voice_lab.casting["roles"]["test-role"]["round"]["notification"]


def make_due(pid):
    project = load_project(pid)
    notification = project.voice_lab.casting["roles"]["test-role"]["round"]["notification"]
    notification["next_retry_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    save_project(project)


def test_skipped_configuration_recovers_with_same_batch_once_then_stops(ready, monkeypatch):
    batches = []
    def send(batch, *args, **kwargs):
        assert state(ready)["attempts"] == len(batches) + 1, "durable before transport"
        batches.append(batch["batch_id"])
        return {"state": "skipped", "detail": "TEST transport unavailable"} if len(batches) == 1 else {"state": "accepted"}
    monkeypatch.setattr(notify, "send_notification", send)
    assert casting.notify_ready(ready) == {"state": "skipped"}
    assert state(ready)["reason"] == "TEST transport unavailable"
    assert casting.notify_ready(ready) == {"state": "idle"}
    make_due(ready)
    assert casting.notify_ready(ready) == {"state": "accepted"}
    assert batches[0] == batches[1]
    assert state(ready)["next_retry_at"] is None
    assert casting.notify_ready(ready) == {"state": "idle"}


@pytest.mark.parametrize("outcome", [{"state": "delivery_unknown"}, {"duplicate": True, "previous_state": "accepted"}])
def test_ambiguous_or_ledger_accepted_delivery_is_not_resent(ready, monkeypatch, outcome):
    calls = []
    monkeypatch.setattr(notify, "send_notification", lambda *args, **kwargs: calls.append(1) or outcome)
    casting.notify_ready(ready)
    make_due(ready)
    assert casting.notify_ready(ready) == {"state": "idle"}
    assert len(calls) == 1


def test_known_failure_keeps_retrying_after_the_early_attempts(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "send_notification", lambda *args, **kwargs: calls.append(1) or {"state": "failed"})
    for attempt in range(casting.MAX_NOTIFICATION_ATTEMPTS + 2):
        assert casting.notify_ready(ready) == {"state": "failed"}
        assert state(ready)["attempts"] == attempt + 1
        assert state(ready)["next_retry_at"] is not None, "a waiting review is never abandoned"
        make_due(ready)
    assert len(calls) == casting.MAX_NOTIFICATION_ATTEMPTS + 2
    # Not yet due again: the capped interval is honoured rather than hammering the provider.
    assert casting.notify_ready(ready) == {"state": "failed"}
    assert casting.notify_ready(ready) == {"state": "idle"}


def test_corrupt_retry_clock_does_not_send(ready, monkeypatch):
    project = load_project(ready)
    project.voice_lab.casting["roles"]["test-role"]["round"]["notification"]["next_retry_at"] = "not-a-time"
    save_project(project)
    monkeypatch.setattr(notify, "send_notification", lambda *args, **kwargs: pytest.fail("must not send"))
    assert casting.notify_ready(ready) == {"state": "idle"}
