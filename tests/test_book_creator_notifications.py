"""Completion notification regressions; isolated signal exports and mocked push.

The real delivery claim/outcome ledger runs in disposable pytest projects.
No test invokes installed credentials, network, workers, or production books.
"""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.test_voice_lab_library import source  # noqa: F401
from tests.test_book_creator import store, imported, render  # noqa: F401
from dubbing_service.app.audiobook import book_creator_notifications as completion
from dubbing_service.app.audiobook.delivery import notify
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.jobs import mutate_job
from dubbing_service.app.models.jobs import SynthesisJobStatus


@pytest.fixture
def finished(store, monkeypatch):
    book = imported(store, b"Chapter 259: Snow I\nFirst original test line.\nChapter 260: Snow II\nSecond original test line.\n")
    result = render(store, book)
    assert result["status"] == "succeeded", result
    project_id = book["project_id"]
    root = store[0][0]
    config = {"project_dir": str(root / project_id), "listening_base_url": "https://desktop.tail-test.ts.net:8443",
        "notification_transport_root": str(root / "transport-not-real"),
        "recipients": {"brian": {"id": "brian", "enabled": True, "notification_transport": "pushover", "tailscale_login": "brian@fixture.invalid"},
                       "mk": {"id": "mk", "enabled": False}}}
    monkeypatch.setattr(completion.delivery_config, "load_config", lambda: config)
    transport_config = SimpleNamespace(app_token="SECRET_FIXTURE_TOKEN", user_key="SECRET_FIXTURE_USER", timeout_seconds=1)
    state = SimpleNamespace(calls=[], response="accepted", config=config, now=datetime(2026, 9, 27, tzinfo=timezone.utc))
    def send(message, *, timeout):
        assert list((root / project_id / "delivery/notifications/intents").glob("*.json"))
        state.calls.append(message)
        if state.response == "exception":
            raise RuntimeError("SECRET_FIXTURE_TOKEN should never appear in output")
        return SimpleNamespace(state=state.response, attempts=1, http_status=200, request_id="provider-fixture-id")
    monkeypatch.setattr(notify, "_load_transport", lambda _: (transport_config, SimpleNamespace(send=send), SimpleNamespace, None))
    monkeypatch.setattr(completion, "_now", lambda: state.now)
    # An old successful receipt begins without opt-in, even when creator's new
    # request defaults change. Historical projects must never become a backlog.
    project = load_project(project_id)
    project.metadata["book_creator"]["requests"]["render-one"].pop("notify_on_completion", None)
    save_project(project)
    return root, project_id, result, state


def opt_in(project_id):
    project = load_project(project_id)
    project.metadata["book_creator"]["requests"]["render-one"]["notify_on_completion"] = True
    save_project(project)


def notification(project_id):
    return load_project(project_id).metadata["book_creator"]["requests"]["render-one"].get("notification", {})


def test_historical_receipt_never_sends_without_explicit_opt_in(finished):
    _, project_id, _, state = finished
    assert completion.notify_request(project_id, "render-one")["status"] == "not_requested"
    assert completion.maintenance_tick() == []
    assert state.calls == [] and not notification(project_id)


def test_explicit_completion_push_uses_actual_titles_and_combined_deep_link_once(finished, monkeypatch):
    root, project_id, result, state = finished
    response = completion.request_notification(project_id, "render-one")
    assert response["status"] == response["state"] == "accepted"
    assert response["accepted"] and not response["device_delivery_confirmed"]
    assert len(state.calls) == 1
    message = state.calls[0]
    assert "Chapter 259: Snow I" in message.body and "Chapter 260: Snow II" in message.body
    assert "Chapter 0 ·" not in message.body and "Chapter 1 ·" not in message.body
    combined = next(row for row in result["outputs"] if len(row["chapter_ids"]) == 2)
    assert message.url.endswith(f"/voice-lab?home=1&book={project_id}&output={combined['export_id']}")
    assert response["review_url"].startswith("/voice-lab?home=1")
    assert "SECRET" not in json.dumps(response)
    assert "audio_file" not in json.dumps(response) and str(root) not in json.dumps(response)
    monkeypatch.setattr(completion, "_verified_exports", lambda *a: pytest.fail("Accepted completion must not rehash book"))
    assert completion.request_notification(project_id, "render-one")["accepted"]
    assert completion.maintenance_tick() == [] and len(state.calls) == 1


@pytest.mark.parametrize("mutation", ["unfinished_job", "missing_combined", "audio_changed", "master_changed", "receipt_changed", "qa_unbound", "map_gap"])
def test_no_push_until_all_expected_immutable_outputs_are_valid(finished, mutation):
    root, project_id, result, state = finished
    opt_in(project_id)
    project = load_project(project_id)
    meta = project.metadata["book_creator"]
    request = meta["requests"]["render-one"]
    output = meta["exports"][request["export_ids"][-1]]
    if mutation == "unfinished_job":
        mutate_job(project_id, result["job_id"], lambda j: setattr(j, "status", SynthesisJobStatus.RUNNING))
    elif mutation == "missing_combined": request["export_ids"].pop()
    elif mutation == "audio_changed": (root / project_id / output["audio_file"]).write_bytes(b"changed local test audio")
    elif mutation == "master_changed": (root / project_id / output["master_file"]).write_bytes(b"changed local test master")
    elif mutation == "receipt_changed": (root / project_id / output["audio_file"]).with_name("receipt.json").write_text('{}')
    elif mutation == "qa_unbound": output["qa_report"]["output_audio_hash"] = "f" * 64
    elif mutation == "map_gap": output["map"].pop()
    save_project(project)
    response = completion.notify_request(project_id, "render-one")
    assert response["status"] in {"waiting_for_exports", "blocked"}
    assert not state.calls


def test_known_failure_has_only_two_bounded_retries_with_backoff(finished):
    _, project_id, _, state = finished
    state.response = "failed"
    first = completion.request_notification(project_id, "render-one")
    assert first["status"] == "failed" and first["attempts"] == 1
    assert first["next_retry_at"] == (state.now + timedelta(seconds=30)).isoformat()
    assert completion.maintenance_tick() == [] and len(state.calls) == 1
    state.now += timedelta(seconds=30)
    completion.maintenance_tick()
    second = notification(project_id)
    assert second["attempts"] == 2 and second["next_retry_at"] == (state.now + timedelta(seconds=120)).isoformat()
    state.now += timedelta(seconds=120)
    completion.maintenance_tick()
    assert notification(project_id)["attempts"] == 3 and notification(project_id)["next_retry_at"] is None
    state.now += timedelta(days=1)
    assert completion.maintenance_tick() == []
    assert len(state.calls) == 3


@pytest.mark.parametrize("provider_state", ["delivery_unknown", "exception"])
def test_unknown_acceptance_is_held_without_automatic_retry(finished, provider_state):
    _, project_id, _, state = finished
    state.response = provider_state
    result = completion.request_notification(project_id, "render-one")
    assert result["status"] == "delivery_unknown" and not result["accepted"]
    state.now += timedelta(days=1)
    assert completion.maintenance_tick() == []
    completion.request_notification(project_id, "render-one")
    assert len(state.calls) == 1 and "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("provider_state", ["accepted", "delivery_unknown"])
def test_existing_delivery_ledger_recovers_lost_project_projection_without_resend(finished, provider_state):
    _, project_id, _, state = finished
    state.response = provider_state
    completion.request_notification(project_id, "render-one")
    project = load_project(project_id)
    project.metadata["book_creator"]["requests"]["render-one"].pop("notification")
    save_project(project)
    recovered = completion.notify_request(project_id, "render-one")
    assert recovered["status"] == provider_state and len(state.calls) == 1


def test_concurrent_completion_requests_submit_only_once(finished):
    _, project_id, _, state = finished
    opt_in(project_id)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: completion.notify_request(project_id, "render-one"), range(3)))
    assert all(row["accepted"] for row in results) and len(state.calls) == 1


def test_crash_after_provider_acceptance_recovers_pending_projection_from_ledger(finished, monkeypatch):
    _, project_id, _, state = finished
    original = completion._save_status
    def crash_after_send(project, request, result):
        if result["state"] == "accepted":
            raise OSError("simulated projection write failure")
        return original(project, request, result)
    monkeypatch.setattr(completion, "_save_status", crash_after_send)
    with pytest.raises(OSError):
        completion.request_notification(project_id, "render-one")
    assert notification(project_id)["state"] == "pending" and len(state.calls) == 1
    monkeypatch.setattr(completion, "_save_status", original)
    assert completion.notify_request(project_id, "render-one")["accepted"]
    assert len(state.calls) == 1


@pytest.mark.parametrize("change", ["mk_enabled", "brian_disabled", "other_root"])
def test_private_recipient_and_workspace_mismatch_never_use_transport(finished, change):
    root, project_id, _, state = finished
    if change == "mk_enabled": state.config["recipients"]["mk"]["enabled"] = True
    elif change == "brian_disabled": state.config["recipients"]["brian"]["enabled"] = False
    else: state.config["project_dir"] = str(root.parent / "another-root/old-project")
    result = completion.request_notification(project_id, "render-one")
    assert result["status"] == "skipped" and state.calls == []


def test_bad_request_cannot_enable_a_historical_project(finished):
    _, project_id, _, state = finished
    with pytest.raises(ValueError): completion.request_notification(project_id, "missing")
    assert not notification(project_id) and state.calls == []


def test_label_formatter_is_compatible_and_bounds_large_book_notifications():
    recipient = {"id": "brian"}
    batch = {"project_id": "book", "batch_id": "completed", "recipient_id": "brian", "book_title": "Long book",
        "chapters": [{"chapter_id": f"section_{i}", "order": i, "chapter_label": "Source title " + str(i) + "x" * 130,
                      "duration_seconds": 60, "audio_sha256": format(i, "064x")} for i in range(200)]}
    message, artifacts = notify._message(batch, recipient, "https://desktop.tail-test.ts.net/voice-lab?home=1")
    assert len(message["body"]) <= 1024 and "more chapters available" in message["body"]
    assert len(artifacts) == 200
    batch["chapters"][0]["chapter_label"] = []
    with pytest.raises(ValueError): notify._message(batch, recipient, "https://desktop.tail-test.ts.net/voice-lab")
