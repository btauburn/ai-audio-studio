"""Private Voice Lab proxy tests with no network, GPU, or production writes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook.delivery.server import create_app


class Upstream:
    def __init__(self, *, status=200, body=None, chunks=(), headers=None):
        self.status_code = status
        self.headers = headers or {"content-type": "application/json"}
        self.body = body if body is not None else {"saved": True}
        self.chunks = chunks
        self.closed = False
        self.buffered = False
        self.yielded = 0

    @property
    def content(self):
        raise AssertionError("The private proxy must not buffer audio via .content")

    async def aread(self):
        self.buffered = True
        return json.dumps(self.body).encode()

    def json(self):
        assert self.buffered
        return self.body

    async def aiter_bytes(self):
        for block in self.chunks:
            self.yielded += 1
            yield block

    async def aclose(self):
        self.closed = True


@pytest.fixture
def private_lab(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.json").write_text('{"id":"project"}', encoding="utf-8")
    config = {"project_dir": str(project), "state_dir": str(project / "delivery"), "port": 8778,
              "listening_base_url": "https://desktop.example.ts.net:8443", "allowed_audio_roots": [str(project)],
              "voice_lab_projects": ["isolated"], "voice_lab_default_project": "isolated",
              "recipients": {"brian": {"enabled": True, "tailscale_login": "brian@example.test", "allow_feedback": True},
                             "mk": {"enabled": False}}}
    state = SimpleNamespace(upstream=Upstream(), clients=[], calls=[], failure=None, get_status={})

    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            state.clients.append(self)

        def build_request(self, method, url, **kwargs):
            return SimpleNamespace(method=method, url=url, **kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.aclose()

        async def get(self, url, **kwargs):
            state.calls.append(SimpleNamespace(method="GET", url=url, **kwargs))
            if state.failure:
                raise state.failure
            status = next((code for suffix, code in state.get_status.items() if url.endswith(suffix)), 200)
            return httpx.Response(status, json=state.upstream.body, request=httpx.Request("GET", url))

        async def send(self, request, *, stream):
            state.calls.append(request)
            assert stream is True
            if state.failure:
                raise state.failure
            return state.upstream

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with TestClient(create_app(config), base_url="http://127.0.0.1:8778", client=("127.0.0.1", 54321),
                    headers={"Tailscale-User-Login": "brian@example.test"}) as client:
        yield config, client, state


def test_private_page_csp_and_allowed_assets_work_without_inline_bootstrap(private_lab):
    _, client, state = private_lab
    response = client.get("/voice-lab")
    assert response.status_code == 200
    assert str(response.url).endswith("/voice-lab?project=isolated")
    assert 'src="/assets/voice-lab.js"' in response.text
    assert 'href="/assets/voice-lab.css"' in response.text
    assert "<script>" not in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert "unsafe-inline" not in response.headers["content-security-policy"]
    # Character portraits and the audio strip are self-contained data: images. Without
    # this directive the private page silently drops every portrait; scripts, styles and
    # everything else stay confined to the origin.
    policy = response.headers["content-security-policy"]
    assert "img-src 'self' data:" in policy
    assert "script-src 'self'" in policy and "style-src 'self'" in policy and "media-src 'self'" in policy
    assert "data:" not in policy.replace("img-src 'self' data:", "")
    assert client.get("/assets/voice-lab.js").status_code == 200
    assert client.get("/assets/voice-lab.css").status_code == 200
    assert state.calls == []


@pytest.mark.parametrize("path", ["/voice-lab?project=secret", "/api/voice-lab/secret", "/api/voice-lab/project/synthesize", "/api/voice-lab/project/library/delete", "/api/voice-lab/project/audio/../../project.json"])
def test_private_allowlist_blocks_unknown_projects_and_operations_before_proxy(private_lab, path):
    _, client, state = private_lab
    assert client.get(path).status_code == 404
    assert state.calls == []


@pytest.mark.parametrize("headers", [{"Tailscale-User-Login": ""}, {"Tailscale-User-Login": "mk@example.test"}, {"Host": "evil.example"}])
def test_private_identity_and_host_are_required_for_voice_library(private_lab, headers):
    _, client, state = private_lab
    assert client.get("/api/voice-lab/project/library", headers=headers).status_code == 403
    assert state.calls == []


def test_private_json_proxy_only_rewrites_generated_audio_urls_and_closes_resources(private_lab):
    _, client, state = private_lab
    origin = "/api/v1/projects/project/voice-lab"
    state.upstream = Upstream(body={"voices": [{"audio_url": origin + "/audio/v_" + "a" * 28}],
                                   "explanation": origin, "external": {"audio_url": "https://official.example/reference"}})
    response = client.get("/api/voice-lab/project")
    assert response.status_code == 200
    assert response.json()["voices"][0]["audio_url"].startswith("/api/voice-lab/project/audio/")
    assert response.json()["explanation"] == origin
    assert response.json()["external"]["audio_url"] == "https://official.example/reference"
    assert state.calls[0].url == "http://127.0.0.1:9000" + origin
    assert state.upstream.closed, "the streamed upstream response is always released"
    assert not state.clients[0].closed, "the shared client stays open for the next request"


def test_private_library_mutation_requires_review_header_origin_and_feedback_authority(private_lab):
    config, client, state = private_lab
    endpoint = "/api/voice-lab/project/library/import"
    body = {"library_voice_id": "vl_fixture", "version": "v1"}
    assert client.post(endpoint, json=body).status_code == 403
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1", "Origin": "https://evil.example"}).status_code == 403
    config["recipients"]["brian"]["allow_feedback"] = False
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1"}).status_code == 403
    assert state.calls == []
    config["recipients"]["brian"]["allow_feedback"] = True
    response = client.post(endpoint, json=body, headers={"X-Review-Request": "1", "Origin": config["listening_base_url"], "Authorization": "must-not-forward"})
    assert response.status_code == 200
    request = state.calls[0]
    assert request.method == "POST" and request.url.endswith("/voice-lab/library/import")
    assert json.loads(request.content) == body
    assert set(request.headers) == {"content-type"}
    assert state.upstream.closed and not state.clients[0].closed


def test_private_proxy_enforces_request_size_before_forwarding(private_lab):
    _, client, state = private_lab
    response = client.post("/api/voice-lab/project/issues", content=b"x" * 16385,
                           headers={"X-Review-Request": "1", "Content-Type": "application/json"})
    assert response.status_code == 413
    assert state.calls == []


def test_range_audio_is_streamed_without_buffering_and_closes_upstream(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(status=206, chunks=[b"abcd", b"efgh"], headers={"content-type": "audio/wav", "content-length": "8", "content-range": "bytes 4-11/32", "accept-ranges": "bytes", "etag": '"exact-hash"'})
    response = client.get("/api/voice-lab/project/audio/v_" + "a" * 28, headers={"Range": "bytes=4-11", "If-Range": '"exact-hash"'})
    assert response.status_code == 206 and response.content == b"abcdefgh"
    assert response.headers["content-range"] == "bytes 4-11/32"
    assert response.headers["content-length"] == "8" and response.headers["etag"] == '"exact-hash"'
    assert state.calls[0].headers == {"range": "bytes=4-11", "if-range": '"exact-hash"'}
    assert not state.upstream.buffered and state.upstream.yielded == 2
    assert state.upstream.closed and not state.clients[0].closed


def test_head_preserves_audio_headers_and_releases_without_consuming_audio(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(headers={"content-type": "audio/wav", "content-length": "9000", "accept-ranges": "bytes"})
    response = client.head("/api/voice-lab/project/audio/excerpt")
    assert response.status_code == 200 and response.content == b""
    assert response.headers["content-length"] == "9000"
    assert state.calls[0].method == "HEAD"
    assert state.upstream.yielded == 0 and not state.upstream.buffered
    assert state.upstream.closed and not state.clients[0].closed


def test_owner_offline_is_explicit_and_the_shared_client_survives_it(private_lab):
    _, client, state = private_lab
    state.failure = httpx.ConnectError("fixture unavailable")
    response = client.get("/api/voice-lab/project/library")
    assert response.status_code == 503 and "Start AI Audio Studio" in response.json()["detail"]
    assert not state.clients[0].closed, "an outage must not tear down the client the next request needs"
    state.failure = None
    assert client.get("/api/voice-lab/project/library").status_code == 200
    assert len(state.clients) == 1, "recovery reuses the same client"


def test_every_proxied_request_reuses_one_client_and_keeps_its_own_timeout(private_lab):
    """Building an httpx client costs ~0.3s of blocking CPU, so it is built once.

    Per-request construction put that cost on the single event loop for every audio
    range request and casting poll, and audio queued behind unrelated calls.
    """
    _, client, state = private_lab
    key = "/api/voice-lab/project/audio/v_" + "a" * 28
    for _ in range(4):
        assert client.get(key, headers={"Range": "bytes=0-"}).status_code == 200
    assert client.get("/api/voice-lab/project/casting").status_code == 200
    assert client.post("/api/voice-lab/project/design", json={}, headers={"X-Review-Request": "1"}).status_code == 200
    assert len(state.clients) == 1
    assert [call.timeout for call in state.calls[:4]] == [90] * 4
    assert state.calls[-1].timeout == 900, "long design requests keep their longer timeout"


def test_private_creator_rewrites_only_owner_urls_and_keeps_identity_boundary(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(body={"audio_url": "/api/v1/book-creator/book/audio/export",
                                   "voices": [{"audio_url": "/api/v1/voice-library/versions/voice/v1/audio"}]})
    result = client.get("/api/book-creator/book")
    assert result.status_code == 200
    assert result.json()["audio_url"] == "/api/book-creator/book/audio/export"
    assert result.json()["voices"][0]["audio_url"] == "/api/voice-library/versions/voice/v1/audio"
    assert state.upstream.closed
    assert client.get("/api/book-creator/book", headers={"Tailscale-User-Login": ""}).status_code == 403
    assert client.post("/api/book-creator/book/render", json={}).status_code == 403
    assert client.post("/api/book-creator/book/delete", json={}, headers={"X-Review-Request": "1"}).status_code == 404


def test_private_book_upload_is_bounded_separately_from_feedback(private_lab):
    _, client, state = private_lab
    body = b"x" * 20000
    headers = {"X-Review-Request": "1", "Content-Type": "multipart/form-data; boundary=fixture"}
    assert client.post("/api/book-creator/import", content=body, headers=headers).status_code == 200
    assert state.calls[-1].content == body
    count = len(state.calls)
    assert client.post("/api/book-creator/import", content=b"", headers={**headers, "Content-Length": str(32*1024*1024+1)}).status_code == 413
    assert len(state.calls) == count
    assert client.post("/api/voice-lab/project/issues", content=body, headers=headers).status_code == 413


def test_private_audition_status_does_not_permit_write_or_other_actions(private_lab):
    _, client, state = private_lab
    assert client.get("/api/voice-lab/project/auditions/request-1").status_code == 200
    assert state.calls[-1].method == "GET"
    assert client.post("/api/voice-lab/project/auditions/request-1", json={}, headers={"X-Review-Request": "1"}).status_code == 404
    assert client.post("/api/voice-lab/project/audition", json={}, headers={"X-Review-Request": "1"}).status_code == 200


def test_private_home_and_project_picker_expose_only_shared_projects(private_lab):
    _, client, state = private_lab
    assert client.get("/voice-lab?home=1").status_code == 200
    state.upstream = Upstream(body=[{"id": "isolated", "name": "Shared"}, {"id": "secret", "name": "Private production"}])
    assert client.get("/api/voice-lab-projects").json() == [{"id": "isolated", "name": "Shared"}]
    assert client.get("/api/voice-lab-projects", headers={"Tailscale-User-Login": ""}).status_code == 403
    assert not state.clients[0].closed


def test_creator_download_forwards_only_supported_query_and_leaves_text_unmodified(private_lab):
    config, client, state = private_lab
    config["recipients"]["brian"]["allow_download"] = True
    state.upstream = Upstream(body={"name": "/api/v1/book-creator/example", "download_url": "/api/v1/book-creator/book/audio/export?download=true"})
    result = client.get("/api/book-creator/book/audio/export?download=true&ignored=secret")
    assert state.calls[-1].url.endswith("/audio/export?download=true")
    assert result.json()["name"] == "/api/v1/book-creator/example"
    assert result.json()["download_url"] == "/api/book-creator/book/audio/export?download=true"


# --------------------------------------------------------------------------
# the book status page a push opens: status, chapter audio and the decision
# --------------------------------------------------------------------------

CHAPTER = "/api/voice-lab/project/casting/chapters/cast_" + "a" * 28 + "/audio"


def test_private_book_status_is_readable_and_forwarded_read_only(private_lab):
    _, client, state = private_lab
    assert client.get("/api/voice-lab/project/casting/book").status_code == 200
    assert state.calls[-1].method == "GET" and state.calls[-1].url.endswith("/voice-lab/casting/book")
    calls = len(state.calls)
    assert client.post("/api/voice-lab/project/casting/book", json={}, headers={"X-Review-Request": "1"}).status_code == 404
    assert len(state.calls) == calls, "status is read-only: a write never reaches Studio"


def test_private_chapter_audio_streams_ranges_and_rejects_anything_else(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(status=206, chunks=[b"ab", b"cd"], headers={
        "content-type": "audio/mpeg", "content-length": "4", "content-range": "bytes 0-3/9", "accept-ranges": "bytes"})
    response = client.get(CHAPTER, headers={"Range": "bytes=0-3"})
    assert response.status_code == 206 and response.content == b"abcd"
    assert response.headers["content-range"] == "bytes 0-3/9"
    assert state.calls[-1].url.endswith("/voice-lab/casting/chapters/cast_" + "a" * 28 + "/audio")
    calls = len(state.calls)
    for bad in ("/api/voice-lab/project/casting/chapters/cast_" + "a" * 27 + "/audio",
                "/api/voice-lab/project/casting/chapters/cast_" + "g" * 28 + "/audio",
                "/api/voice-lab/project/casting/chapters/../../project.json/audio",
                "/api/voice-lab/project/casting/chapters/cast_" + "a" * 28 + "/audio/extra"):
        assert client.get(bad).status_code == 404, bad
    assert client.post(CHAPTER, json={}, headers={"X-Review-Request": "1"}).status_code == 404
    assert len(state.calls) == calls, "refused requests never reach Studio"


def test_private_chapter_download_needs_download_authority_and_is_forwarded(private_lab):
    config, client, state = private_lab
    assert client.get(CHAPTER + "?download=true").status_code == 403
    config["recipients"]["brian"]["allow_download"] = True
    assert client.get(CHAPTER + "?download=true").status_code == 200
    assert state.calls[-1].url.endswith("/audio?download=true")
    assert client.get(CHAPTER).status_code == 200 and not state.calls[-1].url.endswith("?download=true")


def test_private_unattributed_decision_needs_review_header_and_feedback_authority(private_lab):
    config, client, state = private_lab
    endpoint = "/api/voice-lab/project/casting/production/resolve"
    body = {"request_id": "decide-1", "mode": "narrator"}
    assert client.post(endpoint, json=body).status_code == 403
    config["recipients"]["brian"]["allow_feedback"] = False
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1"}).status_code == 403
    assert state.calls == []
    config["recipients"]["brian"]["allow_feedback"] = True
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1"}).status_code == 200
    assert state.calls[-1].method == "POST" and state.calls[-1].url.endswith("/casting/production/resolve")
    calls = len(state.calls)
    assert client.get(endpoint).status_code == 404 and len(state.calls) == calls, "the decision only answers to POST"


@pytest.mark.parametrize("operation, body", [
    ("skip", {"request_id": "skip-1", "utterance_ids": ["u1"]}),
    ("restore", {"request_id": "restore-1", "reason": "unknown_speaker"}),
    ("policy", {"enabled": True}),
])
def test_private_skip_restore_and_policy_are_writes_that_need_the_same_authority(private_lab, operation, body):
    config, client, state = private_lab
    endpoint = "/api/voice-lab/project/casting/production/" + operation
    assert client.post(endpoint, json=body).status_code == 403, "no review header"
    config["recipients"]["brian"]["allow_feedback"] = False
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1"}).status_code == 403, "no feedback authority"
    assert state.calls == []
    config["recipients"]["brian"]["allow_feedback"] = True
    assert client.post(endpoint, json=body, headers={"X-Review-Request": "1"}).status_code == 200
    assert state.calls[-1].method == "POST" and state.calls[-1].url.endswith("/casting/production/" + operation)
    calls = len(state.calls)
    assert client.get(endpoint).status_code == 404 and len(state.calls) == calls, "a decision only answers to POST"


def test_private_service_refuses_production_operations_it_does_not_know(private_lab):
    config, client, state = private_lab
    config["recipients"]["brian"]["allow_feedback"] = True
    for operation in ("delete", "resolve/extra", "../resolve", "skip/1"):
        response = client.post("/api/voice-lab/project/casting/production/" + operation, json={}, headers={"X-Review-Request": "1"})
        assert response.status_code == 404, operation
    assert state.calls == []


def test_private_json_rewrites_chapter_download_urls_to_the_private_origin(private_lab):
    _, client, state = private_lab
    origin = "/api/v1/projects/project/voice-lab"
    state.upstream = Upstream(body={"chapters": [{"audio_url": origin + "/casting/chapters/cast_" + "a" * 28 + "/audio",
                                                  "download_url": origin + "/casting/chapters/cast_" + "a" * 28 + "/audio?download=true"}]})
    chapter = client.get("/api/voice-lab/project/casting/book").json()["chapters"][0]
    assert chapter["audio_url"] == "/api/voice-lab/project/casting/chapters/cast_" + "a" * 28 + "/audio"
    assert chapter["download_url"] == chapter["audio_url"] + "?download=true"


def test_finished_recordings_accept_auto_cast_chapter_identities(private_lab):
    _, client, state = private_lab
    good = "/api/voice-library/recordings/isolated/cast_" + "b" * 28 + "/audio"
    assert client.get(good).status_code == 200
    assert client.get(good.replace("cast_", "book_")).status_code == 404


def test_private_project_picker_asks_studio_for_the_light_list_not_every_project_in_full(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(body=[{"id": "isolated", "name": "Shared"}, {"id": "secret", "name": "Private production"}])
    assert client.get("/api/voice-lab-projects").json() == [{"id": "isolated", "name": "Shared"}]
    assert [call.url for call in state.calls if call.method == "GET"][-1].endswith("/api/v1/projects/summaries")


def test_private_project_picker_falls_back_to_the_full_list_for_a_studio_without_the_light_one(private_lab):
    _, client, state = private_lab
    state.upstream = Upstream(body=[{"id": "isolated", "name": "Shared"}, {"id": "secret", "name": "Private production"}])
    state.get_status["/api/v1/projects/summaries"] = 404
    assert client.get("/api/voice-lab-projects").json() == [{"id": "isolated", "name": "Shared"}]
    asked = [call.url for call in state.calls if call.method == "GET"]
    assert asked[-2].endswith("/api/v1/projects/summaries") and asked[-1].endswith("/api/v1/projects")


def test_reference_prepare_forwards_exact_request_and_keeps_feedback_guards(private_lab):
    config, client, state = private_lab
    acquisition_id = "a" * 32
    endpoint = f"/api/voice-lab/project/references/{acquisition_id}/prepare"
    body = {"acquisition_sha256": "b" * 64, "expected_text": "Reviewed source words."}
    assert client.post(endpoint, json=body).status_code == 403
    assert state.calls == []
    headers = {"X-Review-Request": "1", "Origin": config["listening_base_url"]}
    response = client.post(endpoint, json=body, headers=headers)
    assert response.status_code == 200
    assert state.calls[0].method == "POST"
    assert state.calls[0].url.endswith(f"/voice-lab/references/{acquisition_id}/prepare")
    assert json.loads(state.calls[0].content) == body
    state.calls.clear()
    config["recipients"]["brian"]["allow_feedback"] = False
    assert client.post(endpoint, json=body, headers=headers).status_code == 403
    assert state.calls == []


@pytest.mark.parametrize("operation", ["references/" + "a" * 32 + "/prepare-extra", "references/invalid/prepare"])
def test_reference_prepare_does_not_widen_proxy_allowlist(private_lab, operation):
    config, client, state = private_lab
    response = client.post("/api/voice-lab/project/" + operation, json={},
                           headers={"X-Review-Request": "1", "Origin": config["listening_base_url"]})
    assert response.status_code == 404
    assert state.calls == []


def test_reference_prepare_is_post_only(private_lab):
    _, client, state = private_lab
    assert client.get("/api/voice-lab/project/references/" + "a" * 32 + "/prepare").status_code == 404
    assert state.calls == []
