"""Exercise the default transport through DNS/peer/framing checks, offline wire."""

import hashlib
import json

import pytest

from dubbing_service.app.audiobook import performer_discovery as discovery
from dubbing_service.app.audiobook import public_discovery_tools as public
from tests.test_public_discovery_tools import PUBLIC, WireSocket


def install(monkeypatch, *sockets):
    queue = list(sockets)
    calls = []

    def connector(*args):
        calls.append(args)
        return queue.pop(0)

    client = public.PublicDiscoveryClient(
        resolver=lambda *args: (PUBLIC,), connector=connector
    )
    monkeypatch.setattr(public, "PublicDiscoveryClient", lambda: client)
    monkeypatch.setattr(discovery.time, "sleep", lambda seconds: None)
    return calls


@pytest.mark.parametrize(
    "first",
    [
        WireSocket(b"busy", status=429),
        WireSocket(b"busy", status=503),
        WireSocket(
            b'{"error":{"code":"maxlag","info":"PRIVATE_UNTRUSTED_DETAIL"}}',
            headers=[("Content-Type", "application/json")],
        ),
    ],
)
def test_default_adapter_retry_cache_and_offline_resume(monkeypatch, tmp_path, first):
    body = b'{"search":[]}'
    calls = install(
        monkeypatch,
        first,
        WireSocket(body, headers=[("Content-Type", "application/json")]),
    )
    c = discovery.MetadataClient(tmp_path)
    result = c.get(
        discovery.WIKIDATA_API,
        {"action": "wbsearchentities", "search": "Douglas Adams", "format": "json"},
    )
    assert (
        len(calls) == 2
        and c.stats["requests"] == 2
        and result["content_sha256"] == hashlib.sha256(body).hexdigest()
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    again = discovery.MetadataClient(tmp_path, online=False).get(
        discovery.WIKIDATA_API,
        {"action": "wbsearchentities", "search": "Douglas Adams", "format": "json"},
    )
    assert again == result and len(calls) == 2
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    assert all(b"PRIVATE_UNTRUSTED_DETAIL" not in b for b in before.values())


@pytest.mark.parametrize(
    "socket",
    [
        WireSocket(b"no", status=404),
        WireSocket(b"{}", peer="127.0.0.1"),
        WireSocket(b"x" * 262145, headers=[("Content-Type", "application/json")]),
        WireSocket(
            b"{}",
            headers=[
                ("Content-Type", "application/json"),
                ("Content-Length", "2"),
                ("Transfer-Encoding", "chunked"),
            ],
        ),
    ],
)
def test_policy_framing_404_and_body_cap_hold_without_retry(
    monkeypatch, tmp_path, socket
):
    calls = install(monkeypatch, socket)
    c = discovery.MetadataClient(tmp_path)
    with pytest.raises(discovery.DiscoveryHold):
        c.get(discovery.WIKIDATA_API, {"format": "json"})
    assert len(calls) == 1 and c.stats["requests"] == 1
    hold = next(tmp_path.glob("request-*.json"))
    saved = json.loads(hold.read_text())
    assert saved["retry_after"] > 0 and "payload" not in saved
    with pytest.raises(discovery.DiscoveryHold):
        discovery.MetadataClient(tmp_path).get(
            discovery.WIKIDATA_API, {"format": "json"}
        )
    assert len(calls) == 1


def test_fixed_endpoint_rejects_before_adapter(monkeypatch):
    monkeypatch.setattr(
        public, "PublicDiscoveryClient", lambda: pytest.fail("No external transport")
    )
    with pytest.raises(discovery.DiscoveryHold):
        discovery._request_json("https://127.0.0.1/api", timeout=8, max_bytes=2000000)


def test_typed_retry_never_uses_message_substrings():
    assert discovery._retryable_public_error(
        public.DiscoveryError("http", "safe", http_status=429)
    )
    assert discovery._retryable_public_error(
        public.DiscoveryError("upstream", "safe", api_error_code="maxlag")
    )
    for error in [
        public.DiscoveryError("peer", "HTTP 503 maxlag", http_status=503),
        public.DiscoveryError("parse", "maxlag", api_error_code="maxlag"),
        public.DiscoveryError("http", "safe", http_status=404),
    ]:
        assert not discovery._retryable_public_error(error)
