"""Operator delivery gates with synthetic files and no subprocess/network sends."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from dubbing_service.app.audiobook.delivery import cli, notify, projection, store


@pytest.fixture(autouse=True)
def no_external_operations(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Delivery CLI tests must not perform external operations")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(notify, "send_notification", refuse)


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "test_project"
    project.mkdir()
    (project / "project.json").write_text('{"id":"test_project"}', encoding="utf-8")
    (project / "segments.json").write_text("[]", encoding="utf-8")
    exports = tmp_path / "exports"
    exports.mkdir()
    config = {
        "project_dir": str(project), "state_dir": str(project / "delivery"),
        "export_manifests": [str(exports / "*-manifest.json")],
        "allowed_audio_roots": [str(exports)], "notification_transport_root": str(tmp_path / "transport"),
        "port": 8778, "listening_base_url": "https://example.tail-test.ts.net:8443",
        "recipients": {
            "brian": {"id": "brian", "enabled": True, "tailscale_login": "brian@example.test",
                      "taildrop_dns": "brians-phone.tail-test.ts.net.", "allow_download": True},
            "mk": {"id": "mk", "enabled": False},
        },
    }
    chapters = []
    payloads = [b"first exact audio" * 120, b"second exact audio" * 90]
    for index, content in enumerate(payloads, 41):
        audio = exports / f"Chapter_{index}.mp3"
        audio.write_bytes(content)
        chapters.append({"chapter_id": f"chapter_{index}", "chapter_number": index,
                         "audio_path": str(audio), "audio_sha256": hashlib.sha256(content).hexdigest(),
                         "duration_seconds": 90.0, "revision": "verified-revision"})
    batch = {"schema_version": 1, "project_id": project.name, "book_title": "Fixture Book",
             "recipient_id": "brian", "batch_id": "fixture_batch", "chapters": chapters}
    status = {"Self": {"UserID": 101}, "Peer": {"phone": {
        "DNSName": "brians-phone.tail-test.ts.net.", "UserID": 101, "OS": "iOS", "Online": True,
        "TaildropTarget": 1, "NoFileSharingReason": "", "TailscaleIPs": ["100.65.10.2"],
    }}}
    return SimpleNamespace(config=config, batch=batch, status=status, payloads=payloads, exports=exports)


def _tailscale_mock(monkeypatch, setup, on_transfer=None):
    calls = []

    def run(command, **kwargs):
        assert not kwargs.get("shell", False)
        if command == ["tailscale", "status", "--json"]:
            return SimpleNamespace(stdout=json.dumps(setup.status), returncode=0)
        assert command[:3] == ["tailscale", "file", "cp"]
        content = kwargs["stdin"].read()
        calls.append({"command": command, "content": content, "kwargs": kwargs})
        if on_transfer:
            return on_transfer(command, kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _receipt(setup, name, *, current):
    path = setup.exports / f"{name}-manifest.json"
    hashes = {name: store.digest(Path(setup.config["project_dir"]) / name)
              for name in ("project.json", "segments.json")}
    if not current:
        hashes["segments.json"] = "0" * 64
    path.write_text(json.dumps({"project_id": setup.batch["project_id"],
                                "rendition_state_hashes": hashes}), encoding="utf-8")
    return path


def test_prepare_skips_stale_receipts_and_projects_only_current(setup, monkeypatch):
    _receipt(setup, "stale", current=False)
    current = _receipt(setup, "current", current=True)
    calls = []

    def discover(root, manifests):
        calls.append((root, manifests))
        return copy.deepcopy(setup.batch)

    monkeypatch.setattr(projection, "discover_reviewable", discover)
    result = cli.prepare(setup.config, "brian")
    assert calls == [(Path(setup.config["project_dir"]), [current.resolve()])]
    assert result["chapters"] == setup.batch["chapters"]
    assert result["recipient_id"] == "brian"


def test_prepare_fails_when_no_receipt_matches_current_state(setup, monkeypatch):
    _receipt(setup, "stale", current=False)
    monkeypatch.setattr(projection, "discover_reviewable", lambda *args: pytest.fail("Stale exports must not project"))
    with pytest.raises(ValueError, match="current Studio revision"):
        cli.prepare(setup.config, "brian")
    assert not Path(setup.config["state_dir"]).exists()


@pytest.mark.parametrize("operation", ["prepare", "taildrop"])
def test_mk_is_denied_before_external_or_state_operations(setup, operation):
    if operation == "prepare":
        action = lambda: cli.prepare(setup.config, "mk")
    else:
        setup.batch["recipient_id"] = "mk"
        action = lambda: cli.taildrop(setup.config, setup.batch)
    with pytest.raises(ValueError, match="disabled or unauthorized"):
        action()
    assert not Path(setup.config["state_dir"]).exists()


def _mock_main(monkeypatch, setup, *, changed=False):
    order = []
    prepare_count = 0
    monkeypatch.setattr(cli, "load_config", lambda path: setup.config)

    def prepare(config, recipient):
        nonlocal prepare_count
        prepare_count += 1
        order.append("prepare")
        result = copy.deepcopy(setup.batch)
        if changed and prepare_count == 2:
            result["batch_id"] = "changed_batch"
        return result

    monkeypatch.setattr(cli, "prepare", prepare)
    monkeypatch.setattr(cli, "start_service", lambda *args: order.append("start") or {"service": "running"})
    monkeypatch.setattr(cli, "verify_endpoint", lambda *args, private: order.append("private" if private else "local") or {})
    sends = []
    monkeypatch.setattr(notify, "send_notification", lambda *args, **kwargs: order.append("notify") or sends.append(kwargs) or {"state": "accepted"})
    return order, sends


def test_canonical_guard_reruns_after_endpoint_checks_before_notification(setup, monkeypatch, capsys):
    order, sends = _mock_main(monkeypatch, setup)
    assert cli.main(["deliver", "--ready"]) == 0
    assert order == ["prepare", "start", "local", "private", "prepare", "notify"]
    assert len(sends) == 1 and sends[0]["force"] is False
    assert json.loads(capsys.readouterr().out)["notification"]["state"] == "accepted"


def test_changed_canonical_guard_prevents_notification_and_taildrop(setup, monkeypatch, capsys):
    order, sends = _mock_main(monkeypatch, setup, changed=True)
    monkeypatch.setattr(cli, "taildrop", lambda *args, **kwargs: pytest.fail("Changed canonical batch must not transfer"))
    assert cli.main(["deliver", "--ready", "--taildrop"]) == 1
    assert order == ["prepare", "start", "local", "private", "prepare"]
    assert sends == []
    assert json.loads(capsys.readouterr().err)["status"] == "failed"


@pytest.mark.parametrize("flag", ["--force", "--resend"])
def test_cli_explicit_resend_flag_reaches_both_transports(setup, monkeypatch, flag):
    _, sends = _mock_main(monkeypatch, setup)
    direct = []
    monkeypatch.setattr(cli, "taildrop", lambda *args, **kwargs: direct.append(kwargs) or {"state": "accepted_by_device"})
    assert cli.main(["deliver", "--ready", "--taildrop", flag]) == 0
    assert sends[0]["force"] is True and direct == [{"force": True}]


@pytest.mark.parametrize("field,value", [
    ("UserID", 202), ("DNSName", "some-other-phone.tail-test.ts.net."), ("OS", "Windows"),
    ("Online", False), ("TaildropTarget", 0), ("NoFileSharingReason", "blocked"),
])
def test_taildrop_requires_exact_configured_same_owner_ios_online_device(setup, monkeypatch, field, value):
    setup.status["Peer"]["phone"][field] = value
    calls = _tailscale_mock(monkeypatch, setup)
    result = cli.taildrop(setup.config, setup.batch)
    assert result["attempted"] is False and result["state"] == "device_not_available"
    assert calls == []


def test_taildrop_rejects_missing_owner_identity_even_if_both_missing(setup, monkeypatch):
    setup.status["Self"].pop("UserID")
    setup.status["Peer"]["phone"].pop("UserID")
    calls = _tailscale_mock(monkeypatch, setup)
    result = cli.taildrop(setup.config, setup.batch)
    assert result["attempted"] is False and result["state"] == "device_not_available"
    assert calls == []


def test_taildrop_sends_exact_verified_snapshot_through_stdin(setup, monkeypatch):
    def replace_live_file_after_snapshot(command, kwargs):
        index = int(command[command.index("--name") + 1].split("_")[1].split(".")[0]) - 41
        Path(setup.batch["chapters"][index]["audio_path"]).write_bytes(b"replaced after snapshot")
        return SimpleNamespace(returncode=0)

    calls = _tailscale_mock(monkeypatch, setup, replace_live_file_after_snapshot)
    result = cli.taildrop(setup.config, setup.batch)
    assert result["attempted"] is True and result["state"] == "accepted_by_device"
    assert [c["content"] for c in calls] == setup.payloads
    for index, call in enumerate(calls):
        chapter = setup.batch["chapters"][index]
        assert hashlib.sha256(call["content"]).hexdigest() == chapter["audio_sha256"]
        assert call["command"][-2:] == ["-", "brians-phone.tail-test.ts.net:"]
        assert chapter["audio_path"] not in call["command"]
        assert call["kwargs"]["timeout"] == 180


@pytest.mark.parametrize("changed_index", [0, 1])
def test_altered_audio_anywhere_in_batch_sends_zero_cp(setup, monkeypatch, changed_index):
    Path(setup.batch["chapters"][changed_index]["audio_path"]).write_bytes(b"wrong audio")
    calls = _tailscale_mock(monkeypatch, setup)
    result = cli.taildrop(setup.config, setup.batch)
    assert result["attempted"] is False
    assert result["state"] == "transfer_unknown"
    assert calls == []


def test_first_cp_timeout_is_attempted_unknown_and_stays_suppressed(setup, monkeypatch):
    def timeout(command, kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    calls = _tailscale_mock(monkeypatch, setup, timeout)
    result = cli.taildrop(setup.config, setup.batch)
    assert result["attempted"] is True and result["state"] == "transfer_unknown"
    assert result["files"] == [] and len(calls) == 1
    again = cli.taildrop(setup.config, setup.batch)
    assert again["attempted"] is False and again["state"] == "duplicate_suppressed"
    assert len(calls) == 1


def test_repeat_taildrop_suppressed_until_explicit_force(setup, monkeypatch):
    calls = _tailscale_mock(monkeypatch, setup)
    assert cli.taildrop(setup.config, setup.batch)["state"] == "accepted_by_device"
    assert cli.taildrop(setup.config, setup.batch)["state"] == "duplicate_suppressed"
    assert cli.taildrop(setup.config, setup.batch, force=False)["state"] == "duplicate_suppressed"
    assert len(calls) == 2
    assert cli.taildrop(setup.config, setup.batch, force=True)["state"] == "accepted_by_device"
    assert len(calls) == 4


def test_studio_upstream_is_a_loopback_origin_reachable_from_this_host(setup, tmp_path):
    from dubbing_service.app.audiobook.delivery import config as delivery_config

    path = tmp_path / "config.json"
    base = {**setup.config, "listening_base_url": "https://example.tail-test.ts.net:8443"}
    for value in ("https://desktop.example.ts.net:8443", "http://10.0.0.5:9100",
                  "http://127.0.0.1:9100/voice-lab", "http://127.0.0.1:9100?x=1"):
        path.write_text(json.dumps({**base, "studio_base_url": value}), encoding="utf-8")
        with pytest.raises(ValueError):
            delivery_config.load_config(path)
    path.write_text(json.dumps({**base, "studio_base_url": "http://127.0.0.1:9100/"}), encoding="utf-8")
    assert delivery_config.load_config(path)["studio_base_url"] == "http://127.0.0.1:9100"


def test_delivery_configuration_path_is_selected_per_deployment(tmp_path, monkeypatch):
    from dubbing_service.app.audiobook.delivery import config as delivery_config

    monkeypatch.delenv(delivery_config.CONFIG_ENV, raising=False)
    assert delivery_config.configured_path() == delivery_config.DEFAULT_CONFIG
    selected = tmp_path / "acceptance-delivery.json"
    monkeypatch.setenv(delivery_config.CONFIG_ENV, str(selected))
    assert delivery_config.configured_path() == selected
