# Public discovery adapter boundary

`dubbing_service/app/audiobook/public_discovery_tools.py` is standalone standard-library code, designed by an actual Grok 4.7 session and implemented/tested by the coordinator. It does not call a model, create a project queue, acquire audio, approve rights/identity, or select a voice. Existing canonical AutoCast scheduling and source/reference/QC ownership remain authoritative.

The audio owner can inject a client at the existing seam:

```python
from .public_discovery_tools import PublicDiscoveryClient
public_client = PublicDiscoveryClient()
metadata_client = MetadataClient(directory, limits=limits, online=online,
                                 transport=public_client.metadata_transport)
```

This is a proposed hook, **not an applied integration**. The transport signature is `(url, *, timeout, max_bytes) -> (dict, sha256)`. DiscoveryError subclasses ValueError, which the existing MetadataClient handles. Its stricter cap always wins: caller limits of 2,000,000 bytes/8 seconds are clamped to 262,144 bytes/8 seconds. No retry is performed here. Errors preserve optional `http_status` (strict integer100-599) and `api_error_code` so the owner can apply its existing bounded retry budget to429/5xx and MediaWiki `maxlag`, while404 and byte/security holds remain permanent. Redirects remain refused and may carry their numeric status.

API error codes are normalized by stripping outer whitespace/lowercasing, bounded to64 ASCII characters, then checked against a small known public-code allowlist: maxlag, ratelimited, readonly, timeout, badvalue, missingparam, invalidparammix, permissiondenied, nosuchaction, nosuchmodule, badtoken, notloggedin, assertuserfailed, unknownerror. Unknown, malformed, secret-like or oversized values produce `api_error_code=None`; they are not truncated or echoed. Any JSON `error` member, even empty/null/malformed, is a safe upstream hold. Raw info, bodies, lag values and extra fields never enter the error message or dispatch output. Tool errors expose only code/message/http_status/api_error_code. The owner hook still owns retry classification, cache, stale tokens and canonical commits.

`search(query, backend="wikidata", limit=5)` makes one keyless request to Wikidata, Commons or Openverse and returns typed hits and request provenance. Owner inputs must be public registered title/character/performer names or abstract vocal dimensions. Length/control validation cannot determine whether text is private. Book prose, dossiers, raw audio and credentials must never be supplied. Openverse returns landing-page evidence and never substitutes a direct audio URL. Results do not authorize cloning or reference acquisition.

`fetch(url)` returns capped inert HTML/plain text and provenance. The fixed three-host HTTPS allowlist validates all DNS results, pins the connection, checks the connected peer before/after verified TLS, rejects all redirects, ignores environment proxies, and sends no cookies/auth. Media/JSON fetches and compressed bodies are refused. Accepted raw bodies are hashed, capped at 256 KiB, extracted text at 8,000 characters, and headers/framing at 64 KiB. A large Wikidata entity page was correctly refused; caps were not increased.

DNS runs in at most two outstanding daemon workers across clients. Abandonment returns without killing a blocking getaddrinfo; its slot remains held until it exits. Cancellation is cooperative at DNS/phase/chunk boundaries. Socket watchdogs interrupt blocking connect/TLS/header/body work; OS scheduling and local trust-context creation are not a hard real-time guarantee. Injected clocks require an injected resolver. Tests prove modeled behavior, not universal OS timing.

`tool_schemas()` and `dispatch_tool()` expose two strict, versioned function-tool contracts (`public-discovery-v1`) for a future verified host. There is no model/tool loop, streaming argument assembler or provider registration. Studio currently has no callable OpenRouter base URL/key/exact selected model, and providers/llm.py omits tools/tool_choice. Existing VS Code registration does not prove Studio forwarding. Secure existing provider registration/bridge and the audio-owner hook are still required; do not extract secrets or ask for keys in chat.

Validation: 23 isolated unittest cases cover parsing, caps, hashes, safe errors, strict schemas, dangerous URLs/IP translations, mixed DNS, peer mismatch, TLS/SNI verification, DNS capacity, framing, redirects, cancellation, blocking-header watchdog and inert injection text. The focused owner correction additionally verifies429/503/404 status, normalized maxlag, malformed error shapes, public-code allowlisting and body/secret redaction. Run the single module with the existing Python environment, without importing app initializers:

```text
python -B -m unittest discover -s tests -p test_public_discovery_tools.py -v
```

Live standalone evidence: Wikidata search HTTP200/564 bytes; robots.txt fetch HTTP200/15,896 bytes, extracted text capped at8,000. These are adapter network roundtrips, **not Studio or OpenRouter roundtrips**.

Grok design receipt: task-4/grok-browser-upgrade-review/implementation-design-receipt.json, session01a0fe2d-3165-7f73-88ee-4c4f2c3655e6. Full-patch generation timed out300s; reduced fast-model generation requested an unexpected backend XSearch tool despite advertising no tools and was stopped; static Grok4.7 completion timed out240s. No patch came from those attempts; usage/cost for incomplete calls remains unknown. Three corresponding receipts/streams are preserved. No further generation retries, dependency changes, service restarts, GPU calls, deployment, private project changes or provider credentials were used.
