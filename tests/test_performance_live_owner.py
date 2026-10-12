"""CPU contract tests for the real owner-derived pilot authorization.

These drive the real private delivery server, its real identity middleware and
the real receipt store over an isolated projects root.  Nothing here starts a
service, opens a socket, uses a GPU or sends anything.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import performance_live_owner as owner
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.audiobook.delivery.server import create_app

LOGIN = "brian@example.test"
LISTENING_URL = "https://desktop.example.ts.net:8444"
PORT = 8779


def _iso(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


@pytest.fixture
def deployment(tmp_path):
    pilot_write_root = tmp_path / "elf-chapter6-pilot"
    destination = pilot_write_root / "projects"
    destination.mkdir(parents=True)
    default_project = tmp_path / "default-project"
    (default_project / "delivery").mkdir(parents=True)
    (default_project / "project.json").write_text('{"id":"default-project"}', encoding="utf-8")
    config = {
        "project_dir": str(default_project),
        "state_dir": str(default_project / "delivery"),
        "port": PORT,
        "listening_base_url": LISTENING_URL,
        "allowed_audio_roots": [str(destination)],
        "recipients": {
            "brian": {"id": "brian", "enabled": True, "tailscale_login": LOGIN,
                      "allow_download": True, "allow_feedback": True},
            "mk": {"id": "mk", "enabled": False},
        },
        "performance_live_owner": {
            "enabled": True,
            "principal": "brian",
            "pilot_write_root": str(pilot_write_root),
            "destination_root": str(destination),
            "grant_root": str(pilot_write_root / "owner-receipts"),
            "limits": {"tts_calls": 212, "model_calls": 0, "assemblies": 12, "retries": 12},
            "max_deadline_hours": 2,
        },
    }
    return {"config": config, "pilot_write_root": pilot_write_root,
            "destination": destination, "grant_root": pilot_write_root / "owner-receipts"}


def scope(deployment, **overrides) -> dict:
    value = {
        "operation_id": "elf6-timing-a1-20261008",
        "source": {"project_id": "ba9786ea", "chapter_id": "chap_b87eaf4549ed",
                   "utterance_ids": ["utt-0", "utt-1", "utt-2"]},
        "arms": ["baseline", "variant"],
        "revision": "a" * 64,
        "ceilings": {"tts_calls": 212, "model_calls": 0, "assemblies": 12, "retries": 12,
                     "deadline": _iso(1)},
        "destination_root": str(deployment["destination"]),
    }
    for key, value_ in overrides.items():
        if key == "ceilings":
            value["ceilings"] = {**value["ceilings"], **value_}
        elif key == "source":
            value["source"] = {**value["source"], **value_}
        else:
            value[key] = value_
    return value


def variant(base: dict, **overrides) -> dict:
    """A copy of one already-built scope with named fields replaced."""

    value = json.loads(json.dumps(base))
    for key, item in overrides.items():
        if key in ("ceilings", "source"):
            value[key] = {**value[key], **item}
        else:
            value[key] = item
    return value


def client_for(config, *, login: str | None = LOGIN, host: str | None = None,
               peer: tuple[str, int] = ("127.0.0.1", 54321),
               headers: dict | None = None):
    request_headers = {"Tailscale-User-Login": login} if login is not None else {}
    if host is not None:
        request_headers["Host"] = host
    request_headers.update(headers or {})
    return TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}", client=peer,
                      headers=request_headers)


def issue(client, body, **headers):
    return client.post(owner.GRANT_ROUTE, json=body,
                       headers={"X-Review-Request": "1", **headers})


# --------------------------------------------------------------------------- issuance


def test_a_real_authenticated_request_issues_one_owner_receipt(deployment) -> None:
    config = deployment["config"]
    with client_for(config) as client:
        response = issue(client, {"scope": scope(deployment)})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["purpose"] == "allocation"
    assert payload["listening_approval"] is False
    assert payload["production_enablement"] is False
    assert payload["reference"]["kind"] == owner.RECEIPT_REFERENCE_KIND

    receipt_path = deployment["grant_root"] / "receipts" / (payload["grant_id"] + ".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == owner.GRANT_SCHEMA
    assert receipt["origin"]["boundary"] == "delivery_tailscale_middleware"
    assert receipt["origin"]["peer"] == "127.0.0.1"
    assert receipt["origin"]["recipient_id"] == "brian"
    assert receipt["origin"]["login_sha256"] != LOGIN
    assert receipt["human_review"] == "pending"
    assert receipt["listening_approval"] is False
    assert receipt["production_enablement"] is False
    assert receipt["limits"]["tts_calls"] == 212
    # The store's own marker exists and binds the receipt to this root.
    marker = json.loads((deployment["grant_root"] / owner.ROOT_MARKER_NAME).read_text("utf-8"))
    assert marker["schema_version"] == owner.ROOT_MARKER_SCHEMA
    assert receipt["root_marker_sha256"] == marker["marker_sha256"]
    # An issuance event is recorded, and it never claims approval.
    events = list((deployment["grant_root"] / owner.EVENT_DIR).glob("*.json"))
    assert len(events) == 1
    event = json.loads(events[0].read_text(encoding="utf-8"))
    assert event["kind"] == "allocation_grant_issued"
    assert event["listening_approval"] is False


def test_the_verdict_comes_from_the_receipt_not_the_caller(deployment) -> None:
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": scope(deployment)}).json()
    reference = payload["reference"]
    verdict = owner.verify_receipt(reference)
    assert verdict is not None and verdict["status"] == "verified"
    assert verdict["fingerprint"] == payload["grant_id"]
    # A caller-supplied verdict, a Boolean or a bare string is not evidence.
    assert owner.verify_receipt({"kind": "live_owner", "by": "Brian"}) is None
    assert owner.verify_receipt({"kind": "live_owner", "status": "verified"}) is None
    assert owner.verify_receipt(True) is None
    assert owner.verify_receipt(None) is None
    assert pilot._owner_verification({"kind": "live_owner", "by": "Brian"})["status"] == "unverified"
    assert pilot._owner_verification(True)["status"] == "unverified"


# --------------------------------------------------------------------------- refusal


@pytest.mark.parametrize("kwargs,expected", [
    ({"login": None}, 403),
    ({"login": "mk@example.test"}, 403),
    ({"login": ""}, 403),
    ({"host": "evil.example"}, 403),
    ({"peer": ("192.168.1.25", 8888)}, 403),
    ({"headers": {"Origin": "https://evil.example"}}, 403),
])
def test_anonymous_or_spoofed_requests_are_refused(deployment, kwargs, expected) -> None:
    with client_for(deployment["config"], **kwargs) as client:
        assert issue(client, {"scope": scope(deployment)}).status_code == expected
    assert not list((deployment["grant_root"] / owner.RECEIPT_DIR).glob("*.json"))


def test_missing_review_header_and_wrong_content_type_are_refused(deployment) -> None:
    config = deployment["config"]
    with client_for(config) as client:
        assert client.post(owner.GRANT_ROUTE, json={"scope": scope(deployment)}).status_code == 403
        assert client.post(owner.GRANT_ROUTE, content=b"{}",
                           headers={"X-Review-Request": "1",
                                    "Content-Type": "text/plain"}).status_code == 415
        assert client.post(owner.GRANT_ROUTE, content=b"{}",
                           headers={"X-Review-Request": "1",
                                    "Content-Type": "application/json"}).status_code == 422
        assert issue(client, {"scope": scope(deployment), "extra": 1}).status_code == 422
    assert not list((deployment["grant_root"] / owner.RECEIPT_DIR).glob("*.json"))


def test_an_unconfigured_deployment_installs_no_route_and_stays_fail_closed(tmp_path) -> None:
    config = {"project_dir": str(tmp_path), "state_dir": str(tmp_path), "port": PORT,
              "listening_base_url": LISTENING_URL, "allowed_audio_roots": [str(tmp_path)],
              "recipients": {"brian": {"id": "brian", "enabled": True, "tailscale_login": LOGIN},
                             "mk": {"id": "mk", "enabled": False}}}
    assert owner.load_live_owner(config) is None
    assert owner.enabled(config) is False
    with pytest.raises(owner.LiveOwnerError) as refused:
        owner.issue_allocation_grant(config, request=None, scope={})
    assert refused.value.code == "capability_unsupported"
    with client_for(config) as client:
        assert issue(client, {"scope": {}}).status_code == 404
    # The M3 seam keeps its original, published refusal.
    verdict = pilot.verify_reviewer({"by": "Brian"})
    assert verdict["status"] == "unverified"
    assert "no in-process reviewer-verification bridge" in verdict["reason"]
    assert len(verdict["fingerprint"]) == 64


def test_a_disabled_block_is_the_same_as_no_block(tmp_path) -> None:
    config = {"performance_live_owner": {"enabled": False}}
    assert owner.load_live_owner(config) is None
    for broken in ({"performance_live_owner": {"enabled": True, "pilot_write_root": "relative"}},
                   {"performance_live_owner": {"enabled": True, "principal": "mk"}}):
        with pytest.raises(owner.LiveOwnerError):
            owner.load_live_owner(broken)


# --------------------------------------------------------------------------- tampering


def test_malformed_unknown_and_tampered_receipts_are_unverified(deployment) -> None:
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": scope(deployment)}).json()
    reference = payload["reference"]

    assert owner.verify_receipt({**reference, "receipt_id": "../escape"})["status"] == "unverified"
    assert owner.verify_receipt({**reference, "receipt_id": "b" * 64})["status"] == "unverified"
    assert owner.verify_receipt({**reference,
                                 "grant_root": str(deployment["pilot_write_root"] / "elsewhere")
                                 })["status"] == "unverified"
    assert owner.verify_receipt({**reference, "grant_root": str(Path.cwd())})["status"] == "unverified"

    path = deployment["grant_root"] / owner.RECEIPT_DIR / (payload["grant_id"] + ".json")
    original = path.read_bytes()
    tampered = json.loads(original.decode("utf-8"))
    tampered["scope"]["ceilings"]["tts_calls"] = 999
    path.write_text(json.dumps(tampered), encoding="utf-8")
    verdict = owner.verify_receipt(reference)
    assert verdict["status"] == "unverified"
    path.write_bytes(original)
    assert owner.verify_receipt(reference)["status"] == "verified"


def test_a_receipt_can_never_claim_approval_or_another_principal(deployment) -> None:
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": scope(deployment)}).json()
    path = deployment["grant_root"] / owner.RECEIPT_DIR / (payload["grant_id"] + ".json")
    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("grant_id")

    forged = {**body, "listening_approval": True}
    forged_id = owner.grant_digest(forged)
    (deployment["grant_root"] / owner.RECEIPT_DIR / (forged_id + ".json")).write_text(
        json.dumps({**forged, "grant_id": forged_id}), encoding="utf-8")
    verdict = owner.verify_receipt({**payload["reference"], "receipt_id": forged_id})
    assert verdict["status"] == "unverified"

    borrowed = {**body, "principal": "mk"}
    borrowed_id = owner.grant_digest(borrowed)
    (deployment["grant_root"] / owner.RECEIPT_DIR / (borrowed_id + ".json")).write_text(
        json.dumps({**borrowed, "grant_id": borrowed_id}), encoding="utf-8")
    assert owner.verify_receipt({**payload["reference"],
                                 "receipt_id": borrowed_id})["status"] == "unverified"


# --------------------------------------------------------------------------- replay and binding


def test_identical_issuance_replays_and_a_changed_scope_does_not(deployment) -> None:
    config = deployment["config"]
    base = scope(deployment)
    with client_for(config) as client:
        first = issue(client, {"scope": base}).json()
        second = issue(client, {"scope": base}).json()
        changed = issue(client, {"scope": variant(base, source={"utterance_ids": ["utt-9"]})}).json()
    assert first["grant_id"] == second["grant_id"]
    assert changed["grant_id"] != first["grant_id"]
    stored = sorted(path.name for path in
                    (deployment["grant_root"] / owner.RECEIPT_DIR).glob("*.json"))
    assert stored == sorted({first["grant_id"] + ".json", changed["grant_id"] + ".json"})


def test_a_grant_funds_exactly_one_scope(deployment) -> None:
    base = scope(deployment)
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": base}).json()
    reference = payload["reference"]
    receipt = owner.authorize_operation(reference, expected_scope=base)
    assert receipt["grant_id"] == payload["grant_id"]
    consumed = list((deployment["grant_root"] / owner.CONSUMED_DIR).glob("*.json"))
    assert len(consumed) == 1
    assert json.loads(consumed[0].read_text(encoding="utf-8"))["listening_approval"] is False
    # Consuming the same grant twice is idempotent, never a second authorization.
    owner.authorize_operation(reference, expected_scope=base)
    assert len(list((deployment["grant_root"] / owner.CONSUMED_DIR).glob("*.json"))) == 1

    for overrides, code in (
            ({"operation_id": "another-operation"}, "direction_conflict"),
            ({"source": {"utterance_ids": ["utt-0", "utt-1"]}}, "direction_conflict"),
            ({"source": {"chapter_id": "chap_other"}}, "direction_conflict"),
            ({"arms": ["baseline"]}, "direction_conflict"),
            ({"revision": "b" * 64}, "direction_conflict"),
            ({"ceilings": {"assemblies": 1}}, "direction_conflict"),
            ({"destination_root": str(deployment["pilot_write_root"] / "projects2")},
             "capability_unsupported"),
    ):
        with pytest.raises(owner.LiveOwnerError) as refused:
            owner.authorize_operation(reference, expected_scope=variant(base, **overrides))
        assert refused.value.code == code


def test_an_expired_grant_cannot_fund_or_extend_work(deployment) -> None:
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": scope(deployment, ceilings={"deadline": _iso(0.5)})}).json()
    later = datetime.now(timezone.utc) + timedelta(hours=3)
    with pytest.raises(owner.LiveOwnerError) as refused:
        owner.authorize_operation(payload["reference"], expected_scope=scope(deployment),
                                  now=later)
    assert refused.value.code == "budget_exhausted"


def test_ceilings_and_deadline_cannot_exceed_the_deployment_bound(deployment) -> None:
    config = deployment["config"]
    with client_for(config) as client:
        over_budget = issue(client, {"scope": scope(deployment, ceilings={"tts_calls": 500})})
        too_long = issue(client, {"scope": scope(deployment, ceilings={"deadline": _iso(5)})})
        negative = issue(client, {"scope": scope(deployment, ceilings={"retries": -1})})
        duplicate = issue(client, {"scope": scope(deployment, source={"utterance_ids": ["a", "a"]})})
        bad_arm = issue(client, {"scope": {**scope(deployment), "arms": ["gamma"]}})
    assert over_budget.status_code == 403
    assert too_long.status_code == 403
    assert negative.status_code == 403
    assert duplicate.status_code == 403
    assert bad_arm.status_code == 403
    assert owner.load_live_owner(config) is not None


# --------------------------------------------------------------------------- reviews


def test_an_allocation_grant_can_never_be_a_listening_review(deployment) -> None:
    with client_for(deployment["config"]) as client:
        payload = issue(client, {"scope": scope(deployment)}).json()
    reference = payload["reference"]
    assert pilot._owner_verification(reference)["status"] == "verified"

    with pytest.raises(owner.LiveOwnerError) as refused:
        owner.refuse_allocation_authority({"kind": "live_owner", "evidence": reference})
    assert refused.value.code == "review_pending"

    with pytest.raises(owner.LiveOwnerError) as refused:
        pilot.record_review("any-pilot", {"authentication": {"kind": "live_owner",
                                                             "evidence": reference}})
    assert refused.value.code == "review_pending"
    # The guard is specific: an unrelated claim is passed straight through.
    assert owner.refuse_allocation_authority({"kind": "synthetic_test"}) is None
    assert owner.refuse_allocation_authority({"kind": "live_owner",
                                              "evidence": {"owner_token": "x"}}) is None
    assert "review_pending" in owner.REVIEW_AUTHENTICATION_LIMIT


def _allocation_manifest(granted, reference):
    return {"schema_version": pilot.SCOPE_SCHEMA, "kind": "review_only",
            "operation_id": granted["operation_id"],
            "scope": {"source_projects": [granted["source"]["project_id"]],
                      "utterance_ids": granted["source"]["utterance_ids"],
                      "arms": granted["arms"]},
            "ceilings": granted["ceilings"],
            "expected_revision_sha256": granted["revision"],
            "authorized_destination_root": granted["destination_root"],
            "authorization": {"kind": "live_pilot", "by": "brian",
                              "event_id": "authenticated-grant", "authentication": reference}}


def test_real_create_pilot_consumes_only_the_exact_owner_scope(deployment, monkeypatch):
    from dubbing_service.app.core.config import clear_config_caches
    from dubbing_service.app.storage.projects import load_project
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(deployment["destination"]))
    clear_config_caches()
    granted = scope(deployment)
    with client_for(deployment["config"]) as client:
        response = issue(client, {"scope": granted})
        assert response.status_code == 200, response.text
    manifest = _allocation_manifest(granted, response.json()["reference"])
    changes = [
        {"operation_id": "a-different-operation"},
        {"scope": {**manifest["scope"], "source_projects": ["another-source"]}},
        {"scope": {**manifest["scope"], "utterance_ids": ["not-authorized"]}},
        {"scope": {**manifest["scope"], "arms": ["baseline"]}},
        {"expected_revision_sha256": "b" * 64},
        {"ceilings": {**manifest["ceilings"], "assemblies": 11}},
        {"ceilings": {**manifest["ceilings"], "deadline": _iso(0.5)}},
        {"authorized_destination_root": str(deployment["pilot_write_root"] / "other")},
    ]
    before = set(deployment["destination"].iterdir())
    for change in changes:
        with pytest.raises(pilot.PerformanceDirectorError):
            pilot.create_pilot("real-owner-contract", {**manifest, **change})
        assert set(deployment["destination"].iterdir()) == before
    result = pilot.create_pilot("real-owner-contract", manifest)
    assert load_project(result["pilot_id"]).voice_lab.scope == "isolated_listening_test"
    assert pilot.create_pilot("real-owner-contract", manifest)["pilot_id"] == result["pilot_id"]


def test_a_valid_receipt_in_an_unconfigured_store_cannot_enable_the_seam(deployment, monkeypatch):
    with client_for(deployment["config"]) as client:
        response = issue(client, {"scope": scope(deployment)})
        assert response.status_code == 200
    reference = response.json()["reference"]
    monkeypatch.setattr(owner, "_CONFIGURED_STORES", {})
    assert pilot._owner_verification(reference)["status"] == "unverified"
    with pytest.raises(owner.LiveOwnerError):
        owner.authorize_operation(reference, expected_scope=scope(deployment))
