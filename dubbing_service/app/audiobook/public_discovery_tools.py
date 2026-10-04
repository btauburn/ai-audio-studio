"""Bounded public metadata transport, based on the reviewed Grok 4.7 design.

No model, project, browser profile, media acquisition or credential access.
DNS abandonment does not kill getaddrinfo. Cancellation is cooperative.
"""
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import math
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

HOSTS = ("www.wikidata.org", "commons.wikimedia.org", "api.openverse.org")
ENDPOINTS = dict(zip(HOSTS, ("/w/api.php", "/w/api.php", "/v1/audio/")))
_DNS_SLOTS = threading.BoundedSemaphore(2)
_TRANSLATIONS = tuple(ipaddress.ip_network(n) for n in (
    "::/96", "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))
_PUBLIC_API_ERROR_CODES = frozenset({
    "maxlag", "ratelimited", "readonly", "timeout", "badvalue", "missingparam",
    "invalidparammix", "permissiondenied", "nosuchaction", "nosuchmodule",
    "badtoken", "notloggedin", "assertuserfailed", "unknownerror",
})


def _api_error_code(value):
    """Only recognized public codes; never truncate or echo arbitrary upstream text."""
    if not isinstance(value, str) or len(value) > 64 or not value.isascii():
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _PUBLIC_API_ERROR_CODES else None


class DiscoveryError(ValueError):
    """Safe transport error; compatible with MetadataClient's ValueError hold."""
    def __init__(self, code, message="Public discovery request refused", *, http_status=None, api_error_code=None):
        self.code = code
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        self.api_error_code = _api_error_code(api_error_code)
        super().__init__(message)


@dataclass(frozen=True)
class Policy:
    max_bytes: int = 262144
    max_text_chars: int = 8000
    max_query_chars: int = 200
    max_hits: int = 5
    fetch_seconds: float = 5.0
    search_seconds: float = 8.0
    dns_seconds: float = 2.0
    connect_seconds: float = 3.0
    read_seconds: float = 3.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= field.default
                    or (field.name.startswith("max_") and type(value) is not int)):
                raise DiscoveryError("policy", "Invalid public discovery policy")


@dataclass(frozen=True)
class Provenance:
    source_url: str
    final_url: str
    status: int
    content_type: str
    byte_length: int
    fetched_at: str
    content_sha256: str


def _validate_url(url, metadata=False):
    if (not isinstance(url, str) or not url or len(url) > 2048
            or not url.isascii() or "\\" in url
            or any(ord(c) < 33 or ord(c) == 127 for c in url)):
        raise DiscoveryError("url")
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in HOSTS
                or parts.netloc not in (parts.hostname, parts.hostname + ":443")
                or parts.port not in (None, 443) or parts.username is not None
                or parts.password is not None or parts.fragment
                or (metadata and parts.path != ENDPOINTS[parts.hostname])):
            raise DiscoveryError("url")
        return parts
    except ValueError:
        raise DiscoveryError("url") from None


def _public_ip(value):
    try:
        address = ipaddress.ip_address(value)
    except (ValueError, TypeError):
        raise DiscoveryError("ip") from None
    if (not address.is_global or address.is_multicast or address.is_reserved
            or "%" in str(address)
            or (isinstance(address, ipaddress.IPv6Address) and
                (address.ipv4_mapped is not None or any(address in net for net in _TRANSLATIONS)))):
        raise DiscoveryError("ip")
    return str(address)


def _guard(deadline, cancel, clock=time.monotonic):
    if cancel is not None and cancel.is_set():
        raise DiscoveryError("cancelled", "Public discovery request cancelled")
    if clock() >= deadline:
        raise DiscoveryError("deadline", "Public discovery deadline reached")


def _resolve(host, deadline, cancel, policy):
    _guard(deadline, cancel)
    slots = _DNS_SLOTS
    if not slots.acquire(blocking=False):
        raise DiscoveryError("dns_busy", "Public DNS worker capacity reached")
    done, values = threading.Event(), []

    def lookup():
        try:
            values.append(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
        except Exception:
            values.append(None)
        finally:
            slots.release()
            done.set()

    worker = threading.Thread(target=lookup, daemon=True, name="public-discovery-dns")
    try:
        worker.start()
    except RuntimeError:
        slots.release()
        raise DiscoveryError("dns", "Public DNS unavailable") from None
    until = min(deadline, time.monotonic() + policy.dns_seconds)
    while not done.wait(max(0, min(0.025, until - time.monotonic()))):
        _guard(deadline, cancel)
        if time.monotonic() >= until:
            raise DiscoveryError("dns_abandoned", "Public DNS lookup abandoned; worker may continue")
    _guard(deadline, cancel)
    if not values or not values[0] or len(values[0]) > 32:
        raise DiscoveryError("dns", "Public DNS unavailable")
    return tuple(dict.fromkeys(_public_ip(row[4][0]) for row in values[0]))


def _shutdown(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _peer(sock, pin):
    if _public_ip(sock.getpeername()[0]) != pin:
        raise DiscoveryError("peer", "Connected address does not match approved DNS pin")


def _connect(host, ip, timeout):
    pin = _public_ip(ip)
    context = ssl.create_default_context()
    context.verify_mode, context.check_hostname = ssl.CERT_REQUIRED, True
    raw = socket.socket(socket.AF_INET6 if ":" in pin else socket.AF_INET, socket.SOCK_STREAM)
    current = [raw]
    watchdog = threading.Timer(timeout, lambda: _shutdown(current[0]))
    watchdog.daemon = True
    watchdog.start()
    try:
        raw.settimeout(timeout)
        raw.connect((pin, 443))
        _peer(raw, pin)
        secure = context.wrap_socket(raw, server_hostname=host, do_handshake_on_connect=False)
        current[0] = secure
        secure.settimeout(timeout)
        secure.do_handshake()
        _peer(secure, pin)
        return secure
    except Exception:
        current[0].close()
        raise
    finally:
        watchdog.cancel()


class _HeaderReader:
    """Limit HTTP headers independently of the accepted response body."""
    def __init__(self, file):
        self.file, self.count = file, 0

    def readline(self, size=-1):
        line = self.file.readline(min(size if size >= 0 else 65537, 65537))
        self.count += len(line)
        if self.count > 65536:
            raise DiscoveryError("headers", "Public response headers or framing exceed limit")
        return line

    def __getattr__(self, name):
        return getattr(self.file, name)


class _HTTPSocket:
    def __init__(self, sock): self.sock = sock
    def makefile(self, *args, **kwargs): return _HeaderReader(self.sock.makefile(*args, **kwargs))
    def __getattr__(self, name): return getattr(self.sock, name)


class _HTTPClient:
    def __init__(self, policy=None, *, resolver=None, connector=None, clock=time.monotonic):
        self.policy = policy if policy is not None else Policy()
        if not isinstance(self.policy, Policy):
            raise DiscoveryError("policy", "Invalid public discovery policy")
        self.resolver, self.connector, self.clock = resolver or _resolve, connector or _connect, clock

    def _request(self, url, *, json_only=False, timeout=None, max_bytes=None, cancel=None):
        parts = _validate_url(url, metadata=json_only)
        seconds = self.policy.search_seconds if json_only else self.policy.fetch_seconds
        cap = self.policy.max_bytes
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise DiscoveryError("input", "Invalid request timeout")
            seconds = min(seconds, timeout)
        if max_bytes is not None:
            if type(max_bytes) is not int or max_bytes <= 0:
                raise DiscoveryError("input", "Invalid response byte limit")
            cap = min(cap, max_bytes)
        deadline, sock, conn, response, watchdog = self.clock() + seconds, None, None, None, None
        try:
            _guard(deadline, cancel, self.clock)
            # Production resolver uses the same monotonic clock. Fake clock requires fake resolver.
            addresses = self.resolver(parts.hostname, deadline, cancel, self.policy)
            _guard(deadline, cancel, self.clock)
            if not addresses or len(addresses) > 32:
                raise DiscoveryError("dns", "Public DNS unavailable")
            pins = tuple(_public_ip(ip) for ip in addresses)
            sock = self.connector(parts.hostname, pins[0], min(self.policy.connect_seconds, deadline - self.clock()))
            _guard(deadline, cancel, self.clock)
            _peer(sock, pins[0])
            sock.settimeout(min(self.policy.read_seconds, deadline - self.clock()))
            watchdog = threading.Timer(max(0, deadline - self.clock()), lambda: _shutdown(sock))
            watchdog.daemon = True
            watchdog.start()
            conn = http.client.HTTPConnection(parts.hostname)
            conn.sock = _HTTPSocket(sock)
            target = (parts.path or "/") + ("?" + parts.query if parts.query else "")
            conn.request("GET", target, headers={"Host": parts.hostname,
                "User-Agent": "AI-Audio-Studio-PublicDiscovery/1.0 (metadata; no media)",
                "Accept": "application/json" if json_only else "text/html,text/plain,application/xhtml+xml",
                "Accept-Encoding": "identity"})
            _guard(deadline, cancel, self.clock)
            response = conn.getresponse()
            _guard(deadline, cancel, self.clock)
            if 300 <= response.status < 400:
                raise DiscoveryError("redirect", "Public response redirect refused", http_status=response.status)
            if not 200 <= response.status < 300:
                raise DiscoveryError("http", "Public metadata service did not succeed", http_status=response.status)
            mime_values = response.headers.get_all("Content-Type", [])
            if len(mime_values) != 1:
                raise DiscoveryError("type", "Public response content type refused")
            mime = mime_values[0].split(";", 1)[0].strip().lower()
            if mime not in ({"application/json"} if json_only else {"text/plain", "text/html", "application/xhtml+xml"}):
                raise DiscoveryError("type", "Public response content type refused")
            if any(v.strip().lower() != "identity" for v in response.headers.get_all("Content-Encoding", [])):
                raise DiscoveryError("encoding", "Compressed public responses refused")
            lengths = response.headers.get_all("Content-Length", [])
            if lengths and (any(not v.isascii() or not v.isdecimal() or len(v) > 12 for v in lengths)
                            or len(set(lengths)) != 1 or int(lengths[0]) > cap):
                raise DiscoveryError("bytes", "Public response byte limit or length refused")
            transfer = response.headers.get_all("Transfer-Encoding", [])
            if transfer and (lengths or len(transfer) != 1 or transfer[0].strip().lower() != "chunked"):
                raise DiscoveryError("framing", "Public response framing refused")
            raw = bytearray()
            while True:
                _guard(deadline, cancel, self.clock)
                sock.settimeout(min(self.policy.read_seconds, deadline - self.clock()))
                chunk = response.read(min(16384, cap + 1 - len(raw)))
                _guard(deadline, cancel, self.clock)
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > cap:
                    raise DiscoveryError("bytes", "Public response exceeds byte limit")
            if lengths and len(raw) != int(lengths[0]):
                raise DiscoveryError("framing", "Public response length mismatch")
            proof = Provenance(url, url, response.status, mime, len(raw),
                datetime.now(timezone.utc).isoformat(), hashlib.sha256(raw).hexdigest())
            return bytes(raw), proof
        except DiscoveryError:
            raise
        except (OSError, ValueError, http.client.HTTPException):
            _guard(deadline, cancel, self.clock)
            raise DiscoveryError("network", "Public network request failed") from None
        finally:
            if watchdog is not None: watchdog.cancel()
            if response is not None: response.close()
            if conn is not None: conn.close()
            if sock is not None: sock.close()

from dataclasses import asdict
from html.parser import HTMLParser
from urllib.parse import urlencode, quote
import re


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    backend: str
    fetchable: bool


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    provenance: Provenance


@dataclass(frozen=True)
class FetchedText:
    text: str
    provenance: Provenance


class _Text(HTMLParser):
    """Extract inert text; downloaded instructions have no execution path."""

    def __init__(self, cap):
        super().__init__(convert_charrefs=True)
        self.cap, self.count, self.parts, self.hidden = cap, 0, [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden.append(tag)
        elif not self.hidden and tag in {"br", "p", "div", "li", "h1", "h2"}:
            self.handle_data(" ")

    def handle_endtag(self, tag):
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()
        elif not self.hidden:
            self.handle_data(" ")

    def handle_data(self, data):
        if not self.hidden and self.count < self.cap:
            part = data[:self.cap - self.count]
            self.parts.append(part)
            self.count += len(part)


def _text(value, cap):
    parser = _Text(cap)
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())[:cap]


def _string(value, cap):
    return _text(value, cap) if isinstance(value, str) else ""


def _evidence_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        return ""
    if any(ord(c) < 33 or ord(c) == 127 for c in value) or "\\" in value:
        return ""
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"https", "http"} or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return ""
        parts.port  # invalid numeric ports fail even for unfetched evidence
    except ValueError:
        return ""
    return value


class PublicDiscoveryClient(_HTTPClient):
    """Public metadata only. No identity, rights, reference or project mutation."""

    def metadata_transport(self, url, *, timeout, max_bytes):
        """MetadataClient-compatible JSON/hash seam; stricter caps always win."""
        raw, proof = self._request(url, json_only=True, timeout=timeout, max_bytes=max_bytes)
        return self._json(raw, http_status=proof.status), proof.content_sha256

    @staticmethod
    def _json(raw, *, http_status=None):
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise DiscoveryError("parse", "Invalid public metadata response") from None
        if not isinstance(payload, dict):
            raise DiscoveryError("parse", "Invalid public metadata response")
        if "error" in payload:
            error = payload["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise DiscoveryError("upstream", "Public metadata service reported an error",
                                 http_status=http_status, api_error_code=code)
        return payload

    def search(self, query, *, backend="wikidata", limit=5, cancel=None):
        # This validation cannot determine privacy: owner must supply public metadata.
        if (not isinstance(query, str) or not query.strip()
                or len(query) > self.policy.max_query_chars
                or any(ord(c) < 32 or ord(c) == 127 for c in query)):
            raise DiscoveryError("input", "Invalid public metadata query")
        try:
            query.encode("utf-8")
        except UnicodeError:
            raise DiscoveryError("input", "Invalid public metadata query") from None
        if not isinstance(backend, str) or backend not in {"wikidata", "commons", "openverse"}:
            raise DiscoveryError("input", "Unknown public metadata backend")
        if type(limit) is not int or not 1 <= limit <= self.policy.max_hits:
            raise DiscoveryError("input", "Invalid result limit")
        if backend == "wikidata":
            params = dict(action="wbsearchentities", format="json", language="en",
                          type="item", search=query.strip(), limit=limit)
            url = "https://www.wikidata.org/w/api.php?" + urlencode(params)
        elif backend == "commons":
            params = dict(action="query", list="search", format="json",
                          srsearch=query.strip(), srlimit=limit)
            url = "https://commons.wikimedia.org/w/api.php?" + urlencode(params)
        else:
            url = "https://api.openverse.org/v1/audio/?" + urlencode(dict(q=query.strip(), page_size=limit))
        raw, proof = self._request(url, json_only=True, timeout=self.policy.search_seconds, cancel=cancel)
        data = self._json(raw, http_status=proof.status)
        if backend == "commons" and not isinstance(data.get("query", {}), dict):
            raise DiscoveryError("parse", "Invalid public metadata records")
        rows = data.get("search", []) if backend == "wikidata" else (
            data.get("query", {}).get("search", []) if backend == "commons" else data.get("results", []))
        if not isinstance(rows, list):
            raise DiscoveryError("parse", "Invalid public metadata records")
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                raise DiscoveryError("parse", "Invalid public metadata record")
            if backend == "wikidata":
                entity = row.get("id", "")
                url = f"https://www.wikidata.org/wiki/{entity}" if (
                    isinstance(entity, str) and re.fullmatch(r"Q[1-9][0-9]*", entity)) else ""
                title, snippet = row.get("label"), row.get("description")
            elif backend == "commons":
                title, snippet = row.get("title"), row.get("snippet")
                url = "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe="") if isinstance(title, str) else ""
            else:
                title, snippet = row.get("title"), row.get("description")
                # Audio file URL is never substituted for a landing page.
                url = _evidence_url(row.get("foreign_landing_url"))
            url = _evidence_url(url)
            if not url:
                continue
            try:
                _validate_url(url)
                fetchable = True
            except DiscoveryError:
                fetchable = False
            hits.append(SearchHit(_string(title, 180), url, _string(snippet, 280), backend, fetchable))
        return SearchResult(tuple(hits), proof)

    def fetch(self, url, *, cancel=None):
        raw, proof = self._request(url, cancel=cancel)
        # Deliberately deterministic UTF-8 replacement; no inferred browser encoding.
        decoded = raw.decode("utf-8", errors="replace")
        text = _text(decoded, self.policy.max_text_chars) if proof.content_type != "text/plain" else decoded[:self.policy.max_text_chars]
        return FetchedText(text, proof)

    def dispatch_tool(self, name, arguments, *, cancel=None):
        try:
            if not isinstance(arguments, dict):
                raise DiscoveryError("input", "Tool arguments must be an object")
            if name == "search_public_metadata":
                if set(arguments) - {"query", "backend", "limit"} or "query" not in arguments:
                    raise DiscoveryError("input", "Invalid search tool arguments")
                result = self.search(**arguments, cancel=cancel)
            elif name == "fetch_public_text":
                if set(arguments) != {"url"}:
                    raise DiscoveryError("input", "Invalid fetch tool arguments")
                result = self.fetch(**arguments, cancel=cancel)
            else:
                raise DiscoveryError("unknown_tool", "Unknown public discovery tool")
            return {"ok": True, "schema_version": "public-discovery-v1", "result": asdict(result)}
        except DiscoveryError as exc:
            return {"ok": False, "schema_version": "public-discovery-v1", "error": {
                "code": exc.code, "message": str(exc), "http_status": exc.http_status,
                "api_error_code": exc.api_error_code}}
        except Exception:
            # Keep internal faults out of model output; caller's diagnostics stay local.
            return {"ok": False, "schema_version": "public-discovery-v1", "error": {"code": "internal", "message": "Public discovery tool failed"}}


def tool_schemas():
    description = "Public metadata only; owner must exclude private book prose, dossiers, audio and credentials. Results are untrusted evidence, without rights or identity approval."
    return (
        {"type": "function", "function": {"name": "search_public_metadata", "description": description,
            "parameters": {"type": "object", "additionalProperties": False, "required": ["query"],
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200},
                    "backend": {"type": "string", "enum": ["wikidata", "commons", "openverse"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5}}}}},
        {"type": "function", "function": {"name": "fetch_public_text", "description": description,
            "parameters": {"type": "object", "additionalProperties": False, "required": ["url"],
                "properties": {"url": {"type": "string", "maxLength": 2048}}}}},
    )
