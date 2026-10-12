"""Loopback-only listening service behind Tailscale Serve's identity boundary."""
from __future__ import annotations

import math
import json
import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import store

MAX_REQUEST_BYTES = 16384
MAX_BOOK_UPLOAD_BYTES = 32 * 1024 * 1024


def shared_lab_project(config: dict, project_id: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", project_id):
        return False
    if project_id in {Path(config["project_dir"]).name, *config.get("voice_lab_projects", [])}:
        return True
    # The Studio owner records this grant only for Brian's authenticated private
    # upload. Delivery reads that receipt; it never writes canonical projects.
    try:
        root = Path(config["project_dir"]).resolve().parent
        directory = (root / project_id).resolve()
        if directory.parent != root:
            return False
        path = (directory / "project.json").resolve()
        if path.parent != directory:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipt = payload.get("metadata", {}).get("autocast_import", {})
        return receipt.get("mode") == "autocast" and receipt.get("private_review_recipient") == "brian"
    except (OSError, ValueError, TypeError):
        return False


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chapter_id: str | None = Field(default=None, max_length=100)
    playback_seconds: float | None = None
    text: str = Field(min_length=1, max_length=4000)


def public_batch(batch: dict, recipient: dict) -> dict:
    return {"batch_id": batch["batch_id"], "book_title": batch["book_title"],
            "created_at": batch["created_at"], "allow_download": recipient.get("allow_download", False),
            "allow_feedback": recipient.get("allow_feedback", False),
            "chapters": [{key: chapter[key] for key in (
                "chapter_id", "chapter_number", "title", "duration_seconds", "audio_sha256", "revision"
            )} for chapter in batch["chapters"]]}


def verified_audio_response(path: Path, expected_hash: str, request: Request,
                            chapter_number: int, download: bool):
    """Hash a private byte snapshot and stream that same handle, never reopen source.

    Exports can be replaced concurrently. A spooled snapshot bounds memory to 8MiB
    and ensures that the bytes validated are the bytes returned for this request.
    """
    snapshot = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
        if digest.hexdigest() != expected_hash:
            raise HTTPException(409, "This audio version is unavailable; refresh your review link")
        size = snapshot.tell()
        if not size:
            raise HTTPException(409, "Audio version is empty")
        start, end, status = 0, size - 1, 200
        etag = '"' + expected_hash + '"'
        requested = request.headers.get("range")
        if requested and request.headers.get("if-range", etag) == etag:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size-1) if right else size-1
                else:
                    suffix = int(right)
                    start = max(0, size-suffix) if suffix else size
                status = 206
            else:
                start = size
            if start >= size or end < start:
                snapshot.close()
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"})
        headers = {"ETag": etag, "Accept-Ranges": "bytes", "Content-Length": str(end-start+1),
                   "Content-Disposition": f'{"attachment" if download else "inline"}; filename="Chapter_{chapter_number}{path.suffix}"'}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        media_type = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}[path.suffix.lower()]
        if request.method == "HEAD":
            snapshot.close()
            return Response(status_code=status, headers=headers, media_type=media_type)
        snapshot.seek(start)
        def chunks():
            remaining = end-start+1
            try:
                while remaining:
                    block = snapshot.read(min(256*1024, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block
            finally:
                snapshot.close()
        return StreamingResponse(chunks(), status_code=status, headers=headers, media_type=media_type)
    except BaseException:
        snapshot.close()
        raise


def create_app(config: dict) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    state_dir = Path(config["state_dir"])
    audio_roots = [Path(root).resolve() for root in config["allowed_audio_roots"]]
    public_origin = config["listening_base_url"].rstrip("/")
    # The canonical Studio this projection reads. Deployments that isolate an
    # acceptance projects root behind their own Studio set this explicitly; the
    # default keeps every existing deployment on the long-standing port.
    studio_origin = str(config.get("studio_base_url") or "http://127.0.0.1:9000").rstrip("/")
    allowed_hosts = {urlsplit(public_origin).netloc, f"127.0.0.1:{config['port']}", f"localhost:{config['port']}"}
    shared_clients: dict = {}

    def studio_client():
        """One HTTP client per running event loop, reused by every proxied request.

        Constructing an httpx client builds a TLS context, roughly a third of a
        second of blocking CPU even for a plain-HTTP upstream. Creating one per
        request put that cost on the single event loop for every audio range
        request, casting poll and save, so audio queued behind unrelated calls. The
        client is keyed to its loop because a loop cannot use another loop's client.
        """
        import asyncio

        import httpx
        loop = asyncio.get_running_loop()
        entry = shared_clients.get("studio")
        if entry is None or entry[0] is not loop or entry[0].is_closed():
            entry = (loop, httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_connections=64)))
            shared_clients["studio"] = entry
        return entry[1]

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        # Serve replaces identity headers. Only its loopback connection is trusted;
        # uvicorn must run with proxy_headers=False so clients cannot forge peer IP.
        peer = request.client.host if request.client else ""
        if peer not in {"127.0.0.1", "::1"} or request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"detail": "Private listening access required"}, status_code=403)
        login = request.headers.get("tailscale-user-login", "")
        matches = [recipient for ident, recipient in config["recipients"].items()
                   if ident == "brian" and recipient.get("enabled")
                   and recipient.get("tailscale_login") == login and login]
        if len(matches) != 1:
            return JSONResponse({"detail": "Brian's Tailscale connection is required"}, status_code=403)
        request.state.recipient = {**matches[0], "id": "brian"}
        if request.method not in {"GET", "HEAD"}:
            if request.headers.get("x-review-request") != "1":
                return JSONResponse({"detail": "Review request header required"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin != public_origin:
                return JSONResponse({"detail": "Origin refused"}, status_code=403)
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid request size"}, status_code=400)
            body_limit = MAX_BOOK_UPLOAD_BYTES if request.url.path == "/api/book-creator/import" else MAX_REQUEST_BYTES
            if length > body_limit or length < 0:
                return JSONResponse({"detail": "Feedback is too large"}, status_code=413)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                                 "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"})
        return response

    def get_batch(batch_id: str, request: Request) -> dict:
        try:
            batch = store.load_batch(state_dir, batch_id)
        except (OSError, ValueError, KeyError):
            raise HTTPException(404, "Review batch not found") from None
        if batch["recipient_id"] != request.state.recipient["id"]:
            raise HTTPException(403, "Recipient refused")
        return batch

    def get_chapter(batch: dict, chapter_id: str) -> dict:
        return next((item for item in batch["chapters"] if item["chapter_id"] == chapter_id), None)

    @app.get("/")
    def home(request: Request):
        batches = []
        for path in (state_dir / "manifests").glob("*.json"):
            try:
                batch = store.load_batch(state_dir, path.stem)
                if batch["recipient_id"] == request.state.recipient["id"]:
                    batches.append(batch)
            except (ValueError, OSError, KeyError):
                continue
        if not batches:
            return HTMLResponse("<h1>Audiobook review</h1><p>No chapters are ready yet.</p>")
        return RedirectResponse("/review/" + max(batches, key=lambda row: row["created_at"])["batch_id"])

    @app.get("/health")
    def health():
        return {"service": "audiobook-review", "status": "ok"}

    # Private review remains a projection. Mutations go to the one Studio
    # process holding the canonical projects-root lock, never to a second writer.
    def lab_project(project_id: str) -> str:
        if not shared_lab_project(config, project_id):
            raise HTTPException(404, "Voice Lab project is not shared for private review")
        return project_id

    @app.get("/voice-lab")
    def voice_lab_page(project: str | None = None, home: bool = False):
        if not project and not home:
            return RedirectResponse("/voice-lab?project=" + str(config.get("voice_lab_default_project") or Path(config["project_dir"]).name))
        if project:
            lab_project(project)
        web = Path(__file__).resolve().parents[3] / "webui"
        page = (web / "voice-lab.html").read_text(encoding="utf-8")
        return HTMLResponse(page.replace('/web/voice-lab.css', '/assets/voice-lab.css').replace('/web/voice-lab.js', '/assets/voice-lab.js').replace('/web/runtime-comparison.js', '/assets/runtime-comparison.js').replace('/web/reference-transcripts.js', '/assets/reference-transcripts.js').replace('/web/casting-director.js', '/assets/casting-director.js').replace('/web/casting-director.css', '/assets/casting-director.css'))

    @app.api_route("/api/voice-lab/{project_id}", methods=["GET", "HEAD", "POST"])
    @app.api_route("/api/voice-lab/{project_id}/{operation:path}", methods=["GET", "HEAD", "POST"])
    async def voice_lab_proxy(project_id: str, request: Request, operation: str = ""):
        import httpx
        project_id = lab_project(project_id)
        if request.method == "POST" and not request.state.recipient.get("allow_feedback"):
            raise HTTPException(403, "Feedback is disabled")
        allowed = {"", "preferences", "position", "match", "more-like", "assign", "automation", "issues", "design", "audition", "library", "library/publish", "library/import", "casting", "casting/book", "casting/prepare", "casting/round", "casting/feedback", "casting/render", "casting/resume", "casting/production/resolve", "casting/production/skip", "casting/production/restore", "casting/production/policy", "directors", "directors/analyze", "directors/auditions"}
        audition_status = request.method in {"GET", "HEAD"} and re.fullmatch(r"auditions/[A-Za-z0-9_-]{1,100}", operation)
        chapter_audio = request.method in {"GET", "HEAD"} and re.fullmatch(r"casting/chapters/cast_[0-9a-f]{28}/audio", operation)
        runtime_comparison = request.method in {"GET", "HEAD"} and (
            operation == "runtime-comparison" or re.fullmatch(r"runtime-comparison/audio/[a-z0-9-]{1,100}/(?:baseline|candidate)", operation))
        reference_audio = request.method in {"GET", "HEAD"} and re.fullmatch(r"references/[a-f0-9]{32}/audio", operation)
        reference_operation = reference_audio or (operation == "references" and request.method in {"GET", "HEAD"}) or (
            request.method == "POST" and re.fullmatch(r"references/[a-f0-9]{32}/(?:transcript(?:/correction)?|quality|prepare)", operation))
        if operation not in allowed and not audition_status and not chapter_audio and not runtime_comparison and not reference_operation and not re.fullmatch(r"audio/(?:v_[0-9a-f]{28}|excerpt)", operation):
            raise HTTPException(404)
        # These operations answer to exactly one method each: reading the book, or one decision.
        if (operation == "casting/book" and request.method not in {"GET", "HEAD"}) or (
                operation.startswith("casting/production/") and request.method != "POST"):
            raise HTTPException(404)
        if chapter_audio and request.query_params.get("download", "").lower() in {"1", "true"}:
            if not request.state.recipient.get("allow_download"):
                raise HTTPException(403, "Downloads are disabled")
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Voice Lab request is too large")
        origin = f"/api/v1/projects/{project_id}/voice-lab"
        url = studio_origin + origin + ("/" + operation if operation else "")
        if reference_audio:
            from urllib.parse import urlencode
            url += "?" + urlencode({"acquisition_sha256": request.query_params.get("acquisition_sha256", "")})
        if chapter_audio and request.query_params.get("download", "").lower() in {"1", "true"}:
            url += "?download=true"
        headers = {k: v for k, v in request.headers.items() if k.lower() in {"range", "if-range", "content-type"}}
        if operation.startswith("directors/") and request.headers.get("x-review-request") == "1":
            headers["x-review-request"] = "1"
        client = studio_client()
        try:
            upstream = await client.send(client.build_request(
                request.method, url, headers=headers, content=bytes(content),
                timeout=900 if operation == "design" else 90), stream=True)
        except httpx.HTTPError:
            raise HTTPException(503, "Start AI Audio Studio to save and review Voice Lab") from None
        media = upstream.headers.get("content-type", "application/json")
        if "application/json" in media:
            # Replace only locally generated URL values, never arbitrary text.
            def rewrite(value):
                if isinstance(value, dict):
                    return {k: (f"/api/voice-lab/{project_id}" + v[len(origin):] if k in {"audio_url", "status_url", "download_url"} and isinstance(v, str) and v.startswith(origin + "/") else rewrite(v)) for k, v in value.items()}
                if isinstance(value, list):
                    return [rewrite(v) for v in value]
                return value
            try:
                await upstream.aread()
                return JSONResponse(rewrite(upstream.json()), status_code=upstream.status_code)
            finally:
                await upstream.aclose()
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() in {"accept-ranges", "content-range", "etag", "content-disposition", "content-length"}}
        async def stream():
            try:
                async for block in upstream.aiter_bytes():
                    yield block
            finally:
                await upstream.aclose()
        return StreamingResponse(stream(), status_code=upstream.status_code, media_type=media, headers=response_headers)

    async def creator_proxy(request: Request, family: str, operation: str):
        """Brian-only projection of existing owner APIs; no second project writer."""
        import httpx
        if request.method == "POST" and not request.state.recipient.get("allow_feedback"):
            raise HTTPException(403, "Private creation controls are disabled")
        limit = MAX_BOOK_UPLOAD_BYTES if family == "book-creator" and operation == "import" else MAX_REQUEST_BYTES
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > limit:
                raise HTTPException(413, "Request exceeds the private upload limit")
        url = f"{studio_origin}/api/v1/{family}" + ("/" + operation if operation else "")
        if ("/audio/" in operation or operation.endswith("/audio")) and request.query_params.get("download", "").lower() in {"1", "true"}:
            if not request.state.recipient.get("allow_download"):
                raise HTTPException(403, "Downloads are disabled")
            url += "?download=true"
        headers = {k: v for k, v in request.headers.items() if k.lower() in {"range", "if-range", "content-type"}}
        if family == "book-creator" and operation == "import":
            headers["x-private-review-recipient"] = request.state.recipient["id"]
        client = studio_client()
        try:
            upstream = await client.send(client.build_request(
                request.method, url, headers=headers, content=bytes(content),
                timeout=900 if operation == "import" else 90), stream=True)
        except httpx.HTTPError:
            raise HTTPException(503, "Start AI Audio Studio to use the book creator") from None
        media = upstream.headers.get("content-type", "application/json")
        if "application/json" in media:
            def rewrite(value):
                if isinstance(value, dict):
                    return {k: rewrite_url(v) if k in {"audio_url", "download_url", "status_url", "review_url"} and isinstance(v, str) else rewrite(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [rewrite(v) for v in value]
                return value
            def rewrite_url(value):
                if value.startswith("/web/voice-lab.html?"):
                    return "/voice-lab?" + value.split("?", 1)[1]
                for name in ("book-creator", "voice-library"):
                    prefix = "/api/v1/" + name
                    if value == prefix or value.startswith(prefix + "/"):
                        return "/api/" + name + value[len(prefix):]
                return value
            try:
                await upstream.aread()
                payload = upstream.json()
                if family == "voice-library" and operation == "recordings" and upstream.status_code == 200:
                    allowed = {Path(config["project_dir"]).name, *config.get("voice_lab_projects", [])}
                    payload = {"recordings": [row for row in payload.get("recordings", [])
                                if row.get("kind") == "book_creator" or row.get("project_id") in allowed]}
                return JSONResponse(rewrite(payload), status_code=upstream.status_code)
            finally:
                await upstream.aclose()
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() in {
            "accept-ranges", "content-range", "etag", "content-disposition", "content-length"}}
        async def stream():
            try:
                async for block in upstream.aiter_bytes():
                    yield block
            finally:
                await upstream.aclose()
        return StreamingResponse(stream(), status_code=upstream.status_code, media_type=media, headers=response_headers)

    @app.api_route("/api/book-creator", methods=["GET"])
    @app.api_route("/api/book-creator/{operation:path}", methods=["GET", "HEAD", "POST"])
    async def book_creator(request: Request, operation: str = ""):
        ident = r"[A-Za-z0-9_-]{1,100}"
        readable = (not operation or re.fullmatch(ident, operation)
                    or re.fullmatch(ident + r"/(?:requests|audio)/" + ident, operation))
        writable = (operation == "import" or re.fullmatch(ident + r"/render", operation)
                    or re.fullmatch(ident + r"/requests/" + ident + r"/(?:automation|notify)", operation))
        if not (writable if request.method == "POST" else readable):
            raise HTTPException(404)
        return await creator_proxy(request, "book-creator", operation)

    @app.api_route("/api/voice-library", methods=["GET"])
    @app.api_route("/api/voice-library/{operation:path}", methods=["GET", "HEAD"])
    async def global_voice_library(request: Request, operation: str = ""):
        recording = re.fullmatch(r"recordings/([A-Za-z0-9_-]{1,100})/((?:lab|delivery|cast)_[a-f0-9]{28})/audio", operation)
        if recording:
            lab_project(recording.group(1))
            if request.query_params.get("download", "").lower() in {"1", "true"} and not request.state.recipient.get("allow_download"):
                raise HTTPException(403, "Downloads are disabled")
        elif operation not in {"", "recordings"} and not re.fullmatch(r"versions/[A-Za-z0-9_-]{1,100}/[A-Za-z0-9_-]{1,100}/audio", operation):
            raise HTTPException(404)
        return await creator_proxy(request, "voice-library", operation)

    @app.get("/api/voice-lab-projects")
    async def shared_voice_lab_projects():
        import httpx
        try:
            client = studio_client()
            # The picker only needs each book's id and name. The full list is tens of megabytes and
            # slow enough to time out while Studio is busy, which showed as an empty "Open an
            # existing book". A Studio from before the light list is asked for the full one.
            response = await client.get(f"{studio_origin}/api/v1/projects/summaries", timeout=30)
            if response.status_code == 404:
                response = await client.get(f"{studio_origin}/api/v1/projects", timeout=60)
            response.raise_for_status()
            projects = response.json()
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "Start AI Audio Studio to open shared projects") from None
        return [{"id": row["id"], "name": row["name"]} for row in projects if shared_lab_project(config, row.get("id", ""))]

    @app.get("/review/{batch_id}")
    def page(batch_id: str, request: Request):
        get_batch(batch_id, request)
        return HTMLResponse(Path(__file__).with_name("player.html").read_text(encoding="utf-8"))

    @app.get("/assets/{asset}")
    def asset(asset: str):
        if asset in {"voice-lab.js", "voice-lab.css", "runtime-comparison.js", "reference-transcripts.js", "casting-director.js", "casting-director.css"}:
            return FileResponse(Path(__file__).resolve().parents[3] / "webui" / asset)
        if asset not in {"player.js", "player.css"}:
            raise HTTPException(404)
        return FileResponse(Path(__file__).with_name(asset))

    @app.get("/api/batches/{batch_id}")
    def manifest(batch_id: str, request: Request):
        return public_batch(get_batch(batch_id, request), request.state.recipient)

    @app.api_route("/audio/{batch_id}/{chapter_id}", methods=["GET", "HEAD"])
    def audio(batch_id: str, chapter_id: str, request: Request, download: bool = False):
        batch = get_batch(batch_id, request)
        chapter = get_chapter(batch, chapter_id)
        if chapter is None:
            raise HTTPException(404, "Chapter not found")
        if download and not request.state.recipient.get("allow_download"):
            raise HTTPException(403, "Downloads are disabled")
        try:
            path = Path(chapter["audio_path"]).resolve(strict=True)
            if not any(path.is_relative_to(root) for root in audio_roots) or path.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
                raise ValueError("Path refused")
            return verified_audio_response(path, chapter["audio_sha256"], request,
                                           chapter["chapter_number"], download)
        except (OSError, ValueError):
            raise HTTPException(409, "This audio version is unavailable; refresh your review link") from None

    @app.post("/api/batches/{batch_id}/feedback")
    async def feedback(batch_id: str, request: Request):
        batch = get_batch(batch_id, request)
        if not request.state.recipient.get("allow_feedback"):
            raise HTTPException(403, "Feedback is disabled")
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "JSON feedback required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Feedback is too large")
        try:
            payload = FeedbackInput.model_validate_json(body)
        except ValueError:
            raise HTTPException(422, "Invalid feedback") from None
        text = payload.text.strip()
        if not text:
            raise HTTPException(422, "Enter feedback")
        chapter = get_chapter(batch, payload.chapter_id) if payload.chapter_id is not None else None
        position = payload.playback_seconds
        if payload.chapter_id is not None and chapter is None:
            raise HTTPException(404, "Chapter not found")
        if position is not None and (chapter is None or not math.isfinite(position) or position < 0 or position > chapter["duration_seconds"]):
            raise HTTPException(422, "Playback time is outside this chapter")
        result = store.save_feedback(state_dir, batch, chapter, text, position, request.state.recipient["id"])
        return {"feedback_id": result["feedback_id"], "created_at": result["created_at"], "saved": True}

    @app.get("/api/batches/{batch_id}/feedback")
    def feedback_list(batch_id: str, request: Request):
        get_batch(batch_id, request)
        return {"feedback": store.list_feedback(state_dir, batch_id)}

    return app


# ---------------------------------------------------------------------------
# Additive opt-in route installation for the live review pilot.
#
# The module above is preserved byte-for-byte as an exact prefix of this file.
# The wrapper below is the only addition: it returns the original app unchanged
# unless the operator explicitly enabled the pilot owner store.
# ---------------------------------------------------------------------------

_create_app_without_live_owner = create_app


def create_app(config: dict) -> FastAPI:
    """Build the private listener, adding the owner grant route only on explicit opt-in.

    The added route sits behind this module's existing boundary middleware, so it
    is reachable only from the Tailscale-authenticated loopback connection.  With
    no opt-in configuration the original app is returned untouched and the pilot
    owner seam stays fail-closed.
    """

    app = _create_app_without_live_owner(config)
    from dubbing_service.app.audiobook import performance_live_owner as live_owner
    if live_owner.configure_live_owner(config) is None:
        return app
    _install_live_owner_routes(app, config)
    return app


def _install_live_owner_routes(app: FastAPI, config: dict) -> None:
    """Install the one bounded grant route.  It can only issue, never approve."""

    from dubbing_service.app.audiobook import performance_live_owner as live_owner

    @app.post(live_owner.GRANT_ROUTE)
    async def issue_live_owner_grant(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "A JSON grant request is required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > live_owner.MAX_GRANT_BYTES:
                raise HTTPException(413, "The grant request is too large")
        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "Invalid grant request") from None
        if not isinstance(payload, dict) or set(payload) != {"scope"}:
            raise HTTPException(422, "A grant request carries exactly one scope")
        try:
            receipt = live_owner.issue_allocation_grant(config, request=request,
                                                        scope=payload["scope"])
        except live_owner.LiveOwnerError as exc:
            raise HTTPException(403, str(exc)) from None
        return {"grant_id": receipt["grant_id"], "issued_at": receipt["issued_at"],
                "scope_sha256": receipt["scope_sha256"], "purpose": receipt["purpose"],
                "reference": live_owner.receipt_reference(receipt),
                "listening_approval": False, "production_enablement": False}
