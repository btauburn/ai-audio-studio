"""Delivery safety checks: real disk claims, mocked provider and no live sends."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlunsplit

import httpx
import pytest

from dubbing_service.app.audiobook.delivery import notify


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Tests must never call real notification HTTP")
    monkeypatch.setattr(httpx, "post", refuse)


@pytest.fixture
def batch():
    return {"schema_version": 1, "project_id": "book-project", "batch_id": "batch-one",
            "recipient_id": "brian", "book_title": "The Test Book",
            "chapters": [
                {"chapter_id": "chapter-three", "chapter_number": 3,
                 "audio_sha256": "a" * 64, "duration_seconds": 125.0, "revision": "rev-a"},
                {"chapter_id": "chapter-seven", "chapter_number": 7,
                 "audio_sha256": "b" * 64, "duration_seconds": 361.0, "revision": "rev-b"},
            ]}


@pytest.fixture
def recipient():
    return {"id": "brian", "enabled": True, "notification_transport": "pushover",
            "allow_download": True, "allow_feedback": True}


@pytest.fixture
def sender(monkeypatch, tmp_path):
    config = SimpleNamespace(app_token="SECRET_APP_TOKEN", user_key="SECRET_USER_KEY",
                             device="phone", timeout_seconds=10.0)
    calls = []
    response = SimpleNamespace(state="accepted", detail="raw SECRET_APP_TOKEN", attempts=1,
                               http_status=200, request_id="provider-request-123")

    class Publisher:
        def send(self, message, *, timeout):
            # The claim must exist and be fully readable before entering transport.
            claims = list((tmp_path / "delivery" / "notifications" / "intents").glob("*.json"))
            assert claims
            assert all(json.loads(path.read_text())["state"] == "pending" for path in claims)
            calls.append((message, timeout))
            return response

    transport = SimpleNamespace(post_form=lambda *args, **kwargs: None)
    publisher = Publisher()
    monkeypatch.setattr(notify, "_load_transport", lambda root: (config, publisher, SimpleNamespace, transport))
    return SimpleNamespace(config=config, publisher=publisher, transport=transport,
                           calls=calls, response=response)


def send(batch, recipient, tmp_path, **kwargs):
    return notify.send_notification(batch, recipient, state_dir=tmp_path / "delivery",
                                    transport_root=tmp_path / "transport",
                                    listening_url="https://desktop.tail-example.ts.net:8443/listen/batch-one",
                                    **kwargs)


def test_success_formats_exact_chapters_and_records_safe_provider_evidence(batch, recipient, tmp_path, sender):
    result = send(batch, recipient, tmp_path)
    assert result["state"] == "accepted"
    assert result["accepted"] is True
    assert result["device_delivery_confirmed"] is False
    assert result["provider_request_id"] == "provider-request-123"
    assert result["outcome_recorded"] is True
    message, timeout = sender.calls[0]
    assert message.title == "THE TEST BOOK · 2 CHAPTERS"
    assert "Chapter 3 · ~2 min" in message.body
    assert "Chapter 7 · ~6 min" in message.body
    assert "~8 min total" in message.body
    assert message.url_title == "LISTEN & REVIEW"
    assert message.priority == 0
    assert timeout == 10
    records = "".join(path.read_text() for path in (tmp_path / "delivery" / "notifications").rglob("*.json"))
    assert "SECRET" not in records + json.dumps(result)
    assert "https://" not in records  # No private listening URL in the ledger.


def test_repeat_batch_and_rebatched_identical_audio_are_suppressed(batch, recipient, tmp_path, sender):
    send(batch, recipient, tmp_path)
    assert send(batch, recipient, tmp_path)["duplicate"] is True
    regrouped = copy.deepcopy(batch)
    regrouped["batch_id"] = "regrouped"
    regrouped["chapters"].reverse()
    regrouped["chapters"][0]["revision"] = "new-source-same-audio"
    assert send(regrouped, recipient, tmp_path)["duplicate"] is True
    assert len(sender.calls) == 1


def test_partial_audio_overlap_blocks_entire_batch(batch, recipient, tmp_path, sender):
    send(batch, recipient, tmp_path)
    regrouped = copy.deepcopy(batch)
    regrouped["batch_id"] = "partially-new"
    regrouped["chapters"][1]["audio_sha256"] = "c" * 64
    assert send(regrouped, recipient, tmp_path)["duplicate"] is True
    assert len(sender.calls) == 1


def test_new_audio_and_batch_are_eligible(batch, recipient, tmp_path, sender):
    send(batch, recipient, tmp_path)
    changed = copy.deepcopy(batch)
    changed["batch_id"] = "new-audio"
    changed["chapters"] = [changed["chapters"][0]]
    changed["chapters"][0]["audio_sha256"] = "c" * 64
    assert send(changed, recipient, tmp_path)["accepted"] is True
    assert len(sender.calls) == 2


def test_force_is_an_explicit_intentional_resend(batch, recipient, tmp_path, sender):
    send(batch, recipient, tmp_path)
    assert send(batch, recipient, tmp_path, force="yes")["duplicate"] is True
    result = send(batch, recipient, tmp_path, force=True)
    assert result["accepted"] is True and result["forced"] is True
    assert len(sender.calls) == 2
    assert send(batch, recipient, tmp_path)["duplicate"] is True


@pytest.mark.parametrize("state", ["delivery_unknown", "unknown-future-state"])
def test_uncertain_outcome_never_frees_the_intent(batch, recipient, tmp_path, sender, state):
    sender.response.state = state
    result = send(batch, recipient, tmp_path)
    assert result["state"] == "delivery_unknown"
    assert send(batch, recipient, tmp_path)["duplicate"] is True
    assert len(sender.calls) == 1


def test_unexpected_exception_after_claim_is_unknown_and_secret_safe(batch, recipient, tmp_path, sender, monkeypatch):
    monkeypatch.setattr(sender.publisher, "send", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SECRET_USER_KEY")))
    result = send(batch, recipient, tmp_path)
    assert result["state"] == "delivery_unknown"
    assert "SECRET" not in json.dumps(result)
    assert send(batch, recipient, tmp_path)["duplicate"] is True


def test_process_interruption_leaves_a_durable_unknown_claim(batch, recipient, tmp_path, sender, monkeypatch):
    monkeypatch.setattr(sender.publisher, "send", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        send(batch, recipient, tmp_path)
    result = send(batch, recipient, tmp_path)
    assert result["duplicate"] is True
    assert result["previous_state"] == "delivery_unknown"


def test_known_failure_allows_later_retry(batch, recipient, tmp_path, sender):
    sender.response.state = "failed"
    assert send(batch, recipient, tmp_path)["state"] == "failed"
    sender.response.state = "accepted"
    assert send(batch, recipient, tmp_path)["accepted"] is True
    assert len(sender.calls) == 2


def test_claim_write_failure_prevents_network(batch, recipient, tmp_path, sender, monkeypatch):
    monkeypatch.setattr(notify, "_write_once", lambda *args: (_ for _ in ()).throw(OSError("secret disk error")))
    result = send(batch, recipient, tmp_path)
    assert result["state"] == "skipped"
    assert sender.calls == []
    assert "secret disk error" not in json.dumps(result)


def test_outcome_write_failure_keeps_claim_as_duplicate_blocker(batch, recipient, tmp_path, sender, monkeypatch):
    original = notify._write_once

    def write(path, payload):
        if path.parent.name == "outcomes":
            raise OSError("disk full")
        original(path, payload)

    monkeypatch.setattr(notify, "_write_once", write)
    result = send(batch, recipient, tmp_path)
    assert result["accepted"] is True
    assert result["outcome_recorded"] is False
    assert send(batch, recipient, tmp_path)["duplicate"] is True
    assert len(sender.calls) == 1


@pytest.mark.parametrize("corrupt", ["{", "{}"])
def test_damaged_history_fails_closed_even_with_force(batch, recipient, tmp_path, sender, corrupt):
    intents = tmp_path / "delivery" / "notifications" / "intents"
    intents.mkdir(parents=True)
    (intents / "broken.json").write_text(corrupt)
    result = send(batch, recipient, tmp_path, force=True)
    assert result["state"] == "skipped"
    assert sender.calls == []


def test_concurrent_calls_claim_only_once(batch, recipient, tmp_path, sender):
    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        return send(batch, recipient, tmp_path)

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: worker(), range(5)))
    assert sum(result["accepted"] for result in results) == 1
    assert len(sender.calls) == 1


_CHILD_SEND = """
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
import httpx
from dubbing_service.app.audiobook.delivery import notify
httpx.post = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('no live HTTP'))
root = Path(sys.argv[1])
batch = json.loads(sys.argv[2])
recipient = json.loads(sys.argv[3])
def fake_send(message, *, timeout):
    if sys.argv[4] == 'crash':
        os._exit(79)
    return SimpleNamespace(state='accepted', attempts=1, http_status=200, request_id='request-test')
config = SimpleNamespace(app_token='test-token', user_key='test-user', timeout_seconds=10)
notify._load_transport = lambda root: (config, SimpleNamespace(send=fake_send), SimpleNamespace, None)
result = notify.send_notification(batch, recipient, state_dir=root / 'delivery', transport_root=root,
                                  listening_url='https://desktop.tail-test.ts.net:8443/listen')
print(json.dumps(result))
"""


def _child_send(tmp_path, batch, recipient, mode="send"):
    return subprocess.run([sys.executable, "-c", _CHILD_SEND, str(tmp_path),
                           json.dumps(batch), json.dumps(recipient), mode],
                          capture_output=True, text=True, timeout=20)


def test_independent_processes_claim_the_same_audio_only_once(batch, recipient, tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_child_send, tmp_path, batch, recipient) for _ in range(2)]
        children = [future.result() for future in futures]
    assert all(child.returncode == 0 for child in children), [child.stderr for child in children]
    results = [json.loads(child.stdout) for child in children]
    assert sum(result["accepted"] for result in results) == 1
    assert len(list((tmp_path / "delivery" / "notifications" / "intents").glob("*.json"))) == 1


def test_hard_process_exit_after_claim_blocks_later_send(batch, recipient, tmp_path, sender):
    child = _child_send(tmp_path, batch, recipient, "crash")
    assert child.returncode == 79, child.stderr
    result = send(batch, recipient, tmp_path)
    assert result["duplicate"] is True
    assert result["previous_state"] == "delivery_unknown"
    assert sender.calls == []


@pytest.mark.parametrize("change", ["disabled", "mk", "transport", "mismatched"])
def test_disabled_or_mismatched_recipient_never_sends(batch, recipient, tmp_path, sender, change):
    if change == "disabled":
        recipient["enabled"] = False
    elif change == "mk":
        recipient.update(id="mk", enabled=False)
    elif change == "transport":
        recipient["notification_transport"] = "none"
    else:
        batch["recipient_id"] = "someone-else"
    assert send(batch, recipient, tmp_path)["state"] == "skipped"
    assert sender.calls == []


@pytest.mark.parametrize("url", ["https://public.example/listen", "http://desktop.tail.ts.net/listen",
                                 # Synthetic userinfo is constructed to avoid a
                                 # credential scanner trying to validate this test URL.
                                 urlunsplit(("https", "user:password@desktop.tail.ts.net", "/listen", "", "")),
                                 "https://desktop.tail.ts.net/listen\nsecret"])
def test_private_url_validation_precedes_claim(batch, recipient, tmp_path, sender, url):
    result = notify.send_notification(batch, recipient, state_dir=tmp_path / "delivery",
                                       transport_root=tmp_path / "transport", listening_url=url)
    assert result["state"] == "skipped"
    assert sender.calls == []
    assert not (tmp_path / "delivery").exists()


@pytest.mark.parametrize("field,value", [("duration_seconds", float("nan")), ("duration_seconds", -1),
                                         ("chapter_number", True), ("audio_sha256", "missing")])
def test_bad_metadata_never_reaches_provider(batch, recipient, tmp_path, sender, field, value):
    batch["chapters"][0][field] = value
    assert send(batch, recipient, tmp_path)["state"] == "skipped"
    assert sender.calls == []


def test_provider_echoed_secrets_do_not_enter_outcome(batch, recipient, tmp_path, sender):
    sender.response.request_id = sender.config.app_token
    sender.response.detail = sender.config.user_key
    result = send(batch, recipient, tmp_path)
    assert result["provider_request_id"] is None
    assert "SECRET" not in json.dumps(result)


def test_validation_uses_validation_endpoint_and_exposes_only_presence(tmp_path, sender):
    calls = []

    def post(url, data, *, timeout):
        calls.append((url, set(data), timeout))
        return SimpleNamespace(status_code=200, payload=lambda: {"status": 1, "devices": ["private-device"],
                                                                "secret": "SECRET_APP_TOKEN"})

    sender.transport.post_form = post
    result = notify.validate_credentials(tmp_path / "transport")
    assert result["valid"] is True
    assert result["device_count"] == 1
    assert calls == [("https://api.pushover.net/1/users/validate.json", {"token", "user", "device"}, 10)]
    assert sender.calls == []
    assert "SECRET" not in json.dumps(result)
    assert "private-device" not in json.dumps(result)


@pytest.mark.parametrize("status,payload,expected", [(400, {"status": 0, "errors": ["SECRET"]}, "rejected"),
                                                    (200, None, "validation_unknown"),
                                                    (503, {"errors": ["SECRET"]}, "validation_unknown")])
def test_validation_rejections_and_ambiguity_are_safe(tmp_path, sender, status, payload, expected):
    sender.transport.post_form = lambda *args, **kwargs: SimpleNamespace(status_code=status, payload=lambda: payload)
    result = notify.validate_credentials(tmp_path)
    assert result["state"] == expected
    assert "SECRET" not in json.dumps(result)


def test_missing_configuration_never_validates_or_sends(batch, recipient, tmp_path, sender, monkeypatch):
    monkeypatch.setattr(notify, "_load_transport", lambda *args: (_ for _ in ()).throw(ValueError("SECRET")))
    assert notify.validate_credentials(tmp_path)["configured"] is False
    result = send(batch, recipient, tmp_path)
    assert result["state"] == "skipped"
    assert "SECRET" not in json.dumps(result)
    assert sender.calls == []


@pytest.fixture
def existing_transport(monkeypatch):
    """Exercise the real sibling publisher without loading credentials or HTTP."""
    transport_root = Path.home() / "vscode-mission-panel" / "trip-research"
    source = transport_root / "src"
    if not (source / "trip_research" / "notify" / "publishers.py").is_file():
        pytest.skip("The reusable trip Pushover publisher checkout is unavailable")
    monkeypatch.syspath_prepend(str(source))
    credentials = importlib.import_module("trip_research.credentials")
    monkeypatch.setattr(credentials, "merged_env", lambda **kwargs: {
        "TRIP_NOTIFY_PROVIDER": "pushover", "PUSHOVER_APP_TOKEN": "test-token",
        "PUSHOVER_USER_KEY": "test-user", "PUSHOVER_DEVICE": "test-phone",
    })
    publishers = importlib.import_module("trip_research.notify.publishers")

    def no_retry_sleep(*args):
        raise AssertionError("Ambiguous requests must not reach the retry branch")

    monkeypatch.setattr(publishers.PushoverPublisher, "_wait", no_retry_sleep)
    return transport_root


@pytest.mark.parametrize("status,body", [
    (500, '{"status": 0, "errors": ["test-token"]}'),
    (503, "upstream failure"),
    (200, "not json"),
    (200, "{}"),
    (200, '{"status": "1"}'),
    (200, '{"status": true}'),
])
def test_existing_publisher_makes_one_request_for_ambiguous_reply(
        batch, recipient, tmp_path, existing_transport, monkeypatch, status, body):
    requests = []

    def post(url, *, data, timeout):
        requests.append(url)
        return SimpleNamespace(status_code=status, text=body)

    monkeypatch.setattr(httpx, "post", post)
    arguments = {"state_dir": tmp_path / "delivery", "transport_root": existing_transport,
                 "listening_url": "https://desktop.tail-test.ts.net:8443/listen"}
    result = notify.send_notification(batch, recipient, **arguments)
    assert result["state"] == "delivery_unknown"
    assert result["attempts"] == 1
    assert result["http_status"] == status
    assert requests == ["https://api.pushover.net/1/messages.json"]
    assert "test-token" not in json.dumps(result)
    again = notify.send_notification(batch, recipient, **arguments)
    assert again["duplicate"] is True
    assert again["previous_state"] == "delivery_unknown"
    assert len(requests) == 1


def test_explicit_acceptance_without_request_id_still_blocks_resend(
        batch, recipient, tmp_path, existing_transport, monkeypatch):
    requests = []

    def post(url, *, data, timeout):
        requests.append(url)
        return SimpleNamespace(status_code=200, text='{"status":1}')

    monkeypatch.setattr(httpx, "post", post)
    arguments = {"state_dir": tmp_path / "delivery", "transport_root": existing_transport,
                 "listening_url": "https://desktop.tail-test.ts.net:8443/listen"}
    result = notify.send_notification(batch, recipient, **arguments)
    assert result["state"] == "accepted"
    assert result["provider_request_id"] is None
    assert result["device_delivery_confirmed"] is False
    assert notify.send_notification(batch, recipient, **arguments)["duplicate"] is True
    assert len(requests) == 1


def test_explicit_provider_refusal_remains_retryable_by_a_later_invocation(
        batch, recipient, tmp_path, existing_transport, monkeypatch):
    requests = []

    def post(url, *, data, timeout):
        requests.append(url)
        return SimpleNamespace(status_code=200, text='{"status":0}')

    monkeypatch.setattr(httpx, "post", post)
    arguments = {"state_dir": tmp_path / "delivery", "transport_root": existing_transport,
                 "listening_url": "https://desktop.tail-test.ts.net:8443/listen"}
    assert notify.send_notification(batch, recipient, **arguments)["state"] == "failed"
    assert notify.send_notification(batch, recipient, **arguments)["state"] == "failed"
    assert len(requests) == 2
