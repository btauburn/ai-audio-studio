"""Real owner-derived authorization for one live review pilot (ELF chapter 6).

The pilot needs more than the synthetic seam M3 shipped with: an allocation must
be funded by an authorization a *real* owner actually issued.  This module is
that owner.  It is deliberately small and fail-closed.

What it does
------------
* Reads an explicit, opt-in configuration block.  Absent or disabled, every
  entry point behaves exactly as the unconfigured M3 seam did.
* Issues a durable, immutable **allocation grant** receipt, but only from inside
  an authenticated request that already passed the existing private delivery
  server's Tailscale identity middleware.  The caller cannot hand in a verdict,
  a Boolean, a researcher string or an evidence file path.
* Verifies a receipt by re-reading it from its own immutable store and
  re-deriving every binding, so a caller object is only ever a *reference*.
* Binds the authorization to the operation id, the destination root, the source
  project/chapter/utterances, the arms, the expected revision and the ceilings
  and deadline.  Consuming it re-checks all of those, which is what makes a
  transplanted or expanded token useless.

What it deliberately is not
---------------------------
A grant is permission to spend a bounded scope.  It is **never** listening
approval, a preference, a defect verdict or permission to enable production.
Those facts are recorded as literal ``false``/``pending`` in every receipt, and
no entry point in this module can change them.

The residual trust boundary is stated once, plainly: possession of write access
to the grant root is treated as operator-level trust, exactly as write access to
the projects root is.  What this module guarantees is that an HTTP caller, or an
in-process caller holding only strings and booleans, cannot mint or replay an
authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dubbing_service.app.audiobook.performance_director import (
    PerformanceDirectorError,
    content_sha256,
)

LIVE_OWNER_CONFIG_KEY = "performance_live_owner"
GRANT_ROUTE = "/api/performance-live/owner-grants"
GRANT_SCHEMA = "performance-live-owner-grant-v1"
ROOT_MARKER_SCHEMA = "performance-live-owner-root-v1"

#: The only reference shape this module accepts from a caller.
RECEIPT_REFERENCE_KIND = "live_owner_receipt"
PURPOSE_ALLOCATION = "allocation"

RECEIPT_DIR = "receipts"
EVENT_DIR = "events"
CONSUMED_DIR = "consumed"
ROOT_MARKER_NAME = "root.json"
MAX_GRANT_BYTES = 16384

COUNTERS = ("tts_calls", "model_calls", "assemblies", "retries")
ARMS = ("baseline", "variant")
PRINCIPAL = "brian"

# Stores are installed from the operator's deployment configuration, never from
# a receipt reference. A self-consistent marker at an arbitrary caller path is
# not an authorization owner.
_CONFIGURED_STORES: dict[str, "LiveOwnerStore"] = {}

REVIEW_AUTHENTICATION_LIMIT = (
    "no owner-authenticated listening decision is issued in this milestone.  A live-owner "
    "allocation grant is permission to run a bounded scope and is never a listening verdict, "
    "so a review that presents one is refused rather than promoted.  A real review decision "
    "would have to arise from an authenticated human decision bound to the exact published "
    "pair and audio edition, through its own owner route, which is not installed here.  A "
    "pilot therefore completes as review_pending."
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_LOOPBACK = {"127.0.0.1", "::1"}
_REPO_ROOT = Path(__file__).resolve().parents[3]


class LiveOwnerError(PerformanceDirectorError):
    """An owner-authorization refusal.  Same hold codes as M1-M3."""


def _refuse(message: str, code: str = "review_pending") -> LiveOwnerError:
    return LiveOwnerError(code, message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_once(path: Path, payload: Mapping[str, Any]) -> bool:
    """Publish a complete record exclusively; an existing record is never touched.

    Mirrors the delivery store's own primitive: stage, fsync, then hard-link into
    place so a crash can never publish partial JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / (".stage-" + uuid.uuid4().hex)
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(stage, path)
        except FileExistsError:
            return False
        return True
    finally:
        stage.unlink(missing_ok=True)


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class LiveOwnerStore:
    """The operator's explicit pilot-specific owner configuration."""

    pilot_write_root: Path
    grant_root: Path
    destination_root: Path
    principal: str
    limits: Mapping[str, int]
    max_deadline_hours: int

    def marker_path(self) -> Path:
        return self.grant_root / ROOT_MARKER_NAME

    def receipt_path(self, grant_id: str) -> Path:
        return self.grant_root / RECEIPT_DIR / (grant_id + ".json")

    def as_reference(self) -> dict:
        return {"grant_root": str(self.grant_root)}


def _absolute(value: Any, what: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _refuse(f"{what} must be an absolute path", "capability_unsupported")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise _refuse(f"{what} must be an absolute path without '..'", "capability_unsupported")
    return path.resolve()


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def load_live_owner(config: Mapping[str, Any] | None) -> LiveOwnerStore | None:
    """The opt-in owner configuration, or ``None`` when the operator has not enabled it.

    ``None`` is the shipped state and is what keeps every existing refusal intact.
    """

    if not isinstance(config, Mapping):
        return None
    block = config.get(LIVE_OWNER_CONFIG_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping) or block.get("enabled") is not True:
        return None
    principal = str(block.get("principal") or PRINCIPAL)
    if principal != PRINCIPAL:
        raise _refuse("only the configured principal may be issued a live-owner grant",
                      "capability_unsupported")
    pilot_write_root = _absolute(block.get("pilot_write_root"), "pilot_write_root")
    destination_root = _absolute(block.get("destination_root"), "destination_root")
    grant_root = _absolute(block.get("grant_root") or pilot_write_root / "owner-receipts", "grant_root")
    if _inside(grant_root, _REPO_ROOT) or _inside(destination_root, _REPO_ROOT):
        raise _refuse("a live-owner root may never live inside the source repository",
                      "capability_unsupported")
    if not _inside(grant_root, pilot_write_root) or not _inside(destination_root, pilot_write_root):
        raise _refuse("both the grant root and the destination root must sit inside the "
                      "pilot write root", "capability_unsupported")
    default_project = config.get("project_dir")
    if isinstance(default_project, str) and default_project.strip():
        existing = Path(default_project)
        if existing.is_absolute() and _inside(destination_root, existing.resolve()):
            raise _refuse("a live-owner pilot may never reuse the default delivery project",
                          "capability_unsupported")
    limits: dict[str, int] = {}
    declared = block.get("limits")
    if not isinstance(declared, Mapping):
        raise _refuse("limits must declare every counter", "capability_unsupported")
    for counter in COUNTERS:
        value = declared.get(counter)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(f"limits.{counter} must be a non-negative integer",
                          "capability_unsupported")
        limits[counter] = value
    if limits["tts_calls"] <= 0:
        raise _refuse("limits.tts_calls must allow at least one bounded call",
                      "capability_unsupported")
    hours = block.get("max_deadline_hours", 2)
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 24:
        raise _refuse("max_deadline_hours must be an integer between 1 and 24",
                      "capability_unsupported")
    return LiveOwnerStore(pilot_write_root=pilot_write_root, grant_root=grant_root,
                          destination_root=destination_root, principal=principal,
                          limits=limits, max_deadline_hours=hours)


def enabled(config: Mapping[str, Any] | None) -> bool:
    """Whether this deployment installs the live-owner routes at all."""

    return load_live_owner(config) is not None


# --------------------------------------------------------------------------- the middleware boundary


def _authenticated_recipient(config: Mapping[str, Any], request: Any) -> dict:
    """Re-derive Brian's identity from the existing private delivery boundary.

    A caller cannot supply the answer: the request must be a real Starlette
    request whose state was populated by the delivery server's own middleware,
    and every check that middleware performs is repeated here.
    """

    try:
        from starlette.requests import Request as _Request
    except Exception as exc:  # pragma: no cover - starlette ships with the app
        raise _refuse(f"the owner boundary needs the delivery server request type: {exc}",
                      "capability_unsupported") from exc
    if not isinstance(request, _Request):
        raise _refuse("a live-owner grant may only be issued from an authenticated request",
                      "capability_unsupported")
    state = getattr(request, "state", None)
    recipient = getattr(state, "recipient", None)
    if not isinstance(recipient, Mapping):
        raise _refuse("this request did not pass the private delivery identity boundary",
                      "capability_unsupported")
    peer = request.client.host if request.client else ""
    if peer not in _LOOPBACK:
        raise _refuse("only the local delivery boundary may issue a live-owner grant",
                      "capability_unsupported")
    port = config.get("port")
    base = str(config.get("listening_base_url") or "")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    from urllib.parse import urlsplit
    if base:
        allowed_hosts.add(urlsplit(base).netloc)
    host = request.headers.get("host", "")
    if host not in allowed_hosts:
        raise _refuse("the request host is not this private listener", "capability_unsupported")
    if request.method != "POST":
        raise _refuse("a live-owner grant is issued only by POST", "capability_unsupported")
    if request.headers.get("x-review-request") != "1":
        raise _refuse("the review request header is required", "capability_unsupported")
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != base.rstrip("/"):
        raise _refuse("the request origin is refused", "capability_unsupported")
    configured = (config.get("recipients") or {}).get(PRINCIPAL)
    if not isinstance(configured, Mapping) or configured.get("enabled") is not True:
        raise _refuse("Brian's private recipient is not enabled", "capability_unsupported")
    login = request.headers.get("tailscale-user-login", "")
    expected_login = configured.get("tailscale_login")
    if (recipient.get("id") != PRINCIPAL or recipient.get("enabled") is not True
            or not isinstance(expected_login, str) or not expected_login
            or login != expected_login or recipient.get("tailscale_login") != expected_login):
        raise _refuse("Brian's Tailscale connection is required", "capability_unsupported")
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type != "application/json":
        raise _refuse("a live-owner grant request must be JSON", "capability_unsupported")
    return {"boundary": "delivery_tailscale_middleware", "peer": peer, "host": host,
            "route": str(request.url.path), "method": "POST",
            "recipient_id": PRINCIPAL, "public_origin": base,
            "login_sha256": _sha256(expected_login)}


# --------------------------------------------------------------------------- the scope


def _require_identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise _refuse(f"{what} is not a valid identifier")
    return value


def _require_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _refuse(f"{what} must be a lowercase SHA-256 digest")
    return value


def _require_date(value: Any, what: str) -> datetime:
    if not isinstance(value, str) or not _ISO.fullmatch(value):
        raise _refuse(f"{what} must be an ISO-8601 timestamp")
    try:
        return _utc(value)
    except ValueError as exc:
        raise _refuse(f"{what} is not a parseable timestamp") from exc


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _refuse(f"{what} must be an object")
    return value


def validate_scope(scope: Any, store: LiveOwnerStore, *, now: datetime | None = None) -> dict:
    """Validate and canonicalise the exact scope one grant authorizes."""

    scope = _require_mapping(scope, "scope")
    operation_id = _require_identifier(scope.get("operation_id"), "operation_id")
    source = _require_mapping(scope.get("source"), "source")
    utterances = source.get("utterance_ids")
    if not isinstance(utterances, Sequence) or isinstance(utterances, (str, bytes)) or not utterances:
        raise _refuse("source.utterance_ids must be a non-empty list")
    checked_utterances = [_require_identifier(item, "source.utterance_ids entry") for item in utterances]
    if len(set(checked_utterances)) != len(checked_utterances):
        raise _refuse("source.utterance_ids must not repeat an utterance")
    arms = scope.get("arms")
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)) or not arms:
        raise _refuse("arms must be a non-empty list")
    checked_arms = list(arms)
    if any(arm not in ARMS for arm in checked_arms) or len(set(checked_arms)) != len(checked_arms):
        raise _refuse(f"arms must be a subset of {list(ARMS)} without repeats")
    revision = _require_sha256(scope.get("revision"), "revision")
    ceilings = _require_mapping(scope.get("ceilings"), "ceilings")
    checked_ceilings: dict[str, int] = {}
    for counter in COUNTERS:
        value = ceilings.get(counter)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(f"ceilings.{counter} must be a non-negative integer")
        if value > store.limits[counter]:
            raise _refuse(
                f"ceilings.{counter}={value} exceeds this deployment's bound "
                f"{store.limits[counter]}; a grant can never expand the budget",
                "budget_exhausted")
        checked_ceilings[counter] = value
    deadline_text = ceilings.get("deadline")
    deadline = _require_date(deadline_text, "ceilings.deadline")
    reference = now or datetime.now(timezone.utc)
    if deadline <= reference:
        raise _refuse("ceilings.deadline has already passed", "budget_exhausted")
    if deadline > reference + timedelta(hours=store.max_deadline_hours):
        raise _refuse(
            f"ceilings.deadline is further out than this deployment's "
            f"{store.max_deadline_hours}-hour bound", "budget_exhausted")
    destination = _absolute(scope.get("destination_root"), "destination_root")
    if destination != store.destination_root:
        raise _refuse("destination_root is not this deployment's authorized pilot root",
                      "capability_unsupported")
    return {
        "operation_id": operation_id,
        "source": {
            "project_id": _require_identifier(source.get("project_id"), "source.project_id"),
            "chapter_id": _require_identifier(source.get("chapter_id"), "source.chapter_id"),
            "utterance_ids": checked_utterances,
        },
        "arms": checked_arms,
        "revision": revision,
        "ceilings": checked_ceilings | {"deadline": deadline_text},
        "destination_root": str(destination),
    }


def scope_sha256(scope: Mapping[str, Any]) -> str:
    """The canonical digest of a validated scope, used for every binding check."""

    return content_sha256(dict(scope))


def grant_digest(body: Mapping[str, Any]) -> str:
    """A grant's identity: its content without its own digest or issuance time.

    Excluding the issuance time is what makes an identical replay idempotent — the
    same scope issued again resolves to the same receipt instead of minting a new
    authorization for the same work.
    """

    return content_sha256({key: value for key, value in body.items()
                           if key not in ("grant_id", "issued_at")})


# --------------------------------------------------------------------------- issuance


def _ensure_root_marker(store: LiveOwnerStore) -> dict:
    """Write the store's own marker exactly once, proving a real issuance happened."""

    path = store.marker_path()
    if path.is_file():
        marker = json.loads(path.read_text(encoding="utf-8"))
        if not _marker_valid(marker, store):
            raise _refuse("the live-owner root marker does not verify; the store may not be used",
                          "plan_stale")
        return marker
    marker = {
        "schema_version": ROOT_MARKER_SCHEMA,
        "created_at": _now(),
        "principal": store.principal,
        "pilot_write_root": str(store.pilot_write_root),
        "grant_root": str(store.grant_root),
        "destination_root": str(store.destination_root),
        "authorizes": PURPOSE_ALLOCATION,
        "limits": dict(store.limits),
        "max_deadline_hours": int(store.max_deadline_hours),
        "listening_approval": False,
        "production_enablement": False,
    }
    marker["marker_sha256"] = content_sha256(marker)
    if not _write_once(path, marker):
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not _marker_valid(existing, store):
            raise _refuse("a competing live-owner root marker does not verify", "plan_stale")
        return existing
    return marker


def _marker_valid(marker: Any, store: LiveOwnerStore) -> bool:
    if not isinstance(marker, Mapping) or marker.get("schema_version") != ROOT_MARKER_SCHEMA:
        return False
    if marker.get("principal") != store.principal:
        return False
    if (marker.get("grant_root"), marker.get("destination_root")) != (str(store.grant_root),
                                                                     str(store.destination_root)):
        return False
    if marker.get("authorizes") != PURPOSE_ALLOCATION or marker.get("listening_approval") is not False:
        return False
    limits = marker.get("limits")
    if not isinstance(limits, Mapping) or any(
            isinstance(limits.get(counter), bool) or not isinstance(limits.get(counter), int)
            or limits[counter] < 0 for counter in COUNTERS):
        return False
    hours = marker.get("max_deadline_hours")
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 24:
        return False
    recorded = marker.get("marker_sha256")
    return isinstance(recorded, str) and recorded == content_sha256(
        {key: value for key, value in marker.items() if key != "marker_sha256"})


def issue_allocation_grant(config: Mapping[str, Any], *, request: Any, scope: Any) -> dict:
    """Issue one durable allocation grant for an authenticated operator request."""

    store = load_live_owner(config)
    if store is None:
        raise _refuse("live-owner authorization is not configured in this deployment",
                      "capability_unsupported")
    origin = _authenticated_recipient(config, request)
    checked = validate_scope(scope, store)
    marker = _ensure_root_marker(store)
    body = {
        "schema_version": GRANT_SCHEMA,
        "purpose": PURPOSE_ALLOCATION,
        "authorizes": PURPOSE_ALLOCATION,
        "principal": store.principal,
        "issued_at": _now(),
        "origin": origin,
        "scope": checked,
        "scope_sha256": scope_sha256(checked),
        "grant_root": str(store.grant_root),
        "pilot_write_root": str(store.pilot_write_root),
        "destination_root": checked["destination_root"],
        "root_marker_sha256": marker["marker_sha256"],
        "limits": dict(store.limits),
        "authority": {"kind": "tailscale_owner_middleware", "request_state": "recipient",
                      "principal": store.principal},
        "listening_approval": False,
        "human_review": "pending",
        "production_enablement": False,
    }
    grant_id = grant_digest(body)
    receipt = {**body, "grant_id": grant_id}
    path = store.receipt_path(grant_id)
    if not _write_once(path, receipt):
        stored = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(stored, Mapping) or stored.get("grant_id") != grant_id
                or grant_digest(stored) != grant_id):
            raise _refuse("a stored grant with this identity has different content",
                          "plan_stale")
        return dict(stored)
    _write_once(store.grant_root / EVENT_DIR / (uuid.uuid4().hex + ".json"), {
        "schema_version": "performance-live-owner-event-v1", "kind": "allocation_grant_issued",
        "at": _now(), "grant_id": grant_id, "operation_id": checked["operation_id"],
        "scope_sha256": receipt["scope_sha256"], "principal": store.principal,
        "listening_approval": False, "production_enablement": False})
    return receipt


def receipt_reference(receipt: Mapping[str, Any]) -> dict:
    """The caller-facing reference to a stored grant.  Never the verdict itself."""

    return {"kind": RECEIPT_REFERENCE_KIND, "purpose": PURPOSE_ALLOCATION,
            "receipt_id": receipt["grant_id"], "grant_root": receipt["grant_root"]}


# --------------------------------------------------------------------------- verification


def _resolve_store(grant_root: Any) -> LiveOwnerStore:
    """Resolve a receipt's store from its own root marker, never from trust in the caller."""

    declared = _absolute(grant_root, "grant_root")
    configured = _CONFIGURED_STORES.get(str(declared))
    if configured is None:
        raise _refuse("this receipt store is not enabled in the operator configuration",
                      "capability_unsupported")
    if _inside(declared, _REPO_ROOT):
        raise _refuse("a live-owner receipt root may never live inside the source repository",
                      "capability_unsupported")
    marker_path = declared / ROOT_MARKER_NAME
    if not marker_path.is_file():
        raise _refuse("the referenced directory is not a live-owner receipt store",
                      "capability_unsupported")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"the live-owner root marker is unreadable: {exc}", "plan_stale") from exc
    if not isinstance(marker, Mapping) or marker.get("schema_version") != ROOT_MARKER_SCHEMA:
        raise _refuse("the live-owner root marker has an unknown schema", "schema_unknown")
    store = configured
    if not _marker_valid(marker, store):
        raise _refuse("the live-owner root marker does not verify", "plan_stale")
    return store


def _verify_receipt_file(store: LiveOwnerStore, receipt_id: str) -> dict:
    path = store.receipt_path(receipt_id)
    if not path.is_file():
        raise _refuse("no live-owner receipt with that identity exists", "review_pending")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"the live-owner receipt is unreadable: {exc}", "plan_stale") from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != GRANT_SCHEMA:
        raise _refuse("the live-owner receipt has an unknown schema", "schema_unknown")
    if receipt.get("grant_id") != receipt_id:
        raise _refuse("the live-owner receipt identity does not match its name", "plan_stale")
    if grant_digest(receipt) != receipt_id:
        raise _refuse("the live-owner receipt's own digest does not verify", "plan_stale")
    if receipt.get("purpose") != PURPOSE_ALLOCATION or receipt.get("authorizes") != PURPOSE_ALLOCATION:
        raise _refuse("this receipt does not authorize an allocation", "review_pending")
    origin = receipt.get("origin")
    if (not isinstance(origin, Mapping)
            or origin.get("boundary") != "delivery_tailscale_middleware"
            or origin.get("recipient_id") != PRINCIPAL
            or origin.get("method") != "POST"
            or not isinstance(origin.get("login_sha256"), str)):
        raise _refuse("the receipt does not record a real owner origin", "plan_stale")
    if receipt.get("principal") != store.principal:
        raise _refuse("the receipt was issued to another principal", "plan_stale")
    if receipt.get("grant_root") != str(store.grant_root):
        raise _refuse("the receipt belongs to another live-owner store", "plan_stale")
    if receipt.get("root_marker_sha256") != json.loads(
            store.marker_path().read_text(encoding="utf-8"))["marker_sha256"]:
        raise _refuse("the receipt's root marker binding does not verify", "plan_stale")
    if receipt.get("listening_approval") is not False or receipt.get("production_enablement") is not False:
        raise _refuse("an allocation grant may never carry listening or production approval",
                      "direction_conflict")
    if receipt.get("destination_root") != str(store.destination_root):
        raise _refuse("the receipt's destination root does not match its store", "plan_stale")
    checked = validate_scope(receipt.get("scope"), store)
    if scope_sha256(checked) != receipt.get("scope_sha256"):
        raise _refuse("the receipt's scope digest does not verify", "plan_stale")
    return receipt


def verify_receipt(evidence: Any) -> dict | None:
    """The owner verdict for a *receipt reference*, or ``None`` for anything else.

    ``None`` means "this is not a live-owner reference", which lets the caller
    fall back to its own unconfigured refusal.  A receipt-shaped reference always
    returns a verdict, so a malformed or tampered one can never silently pass.
    """

    if not isinstance(evidence, Mapping) or evidence.get("kind") != RECEIPT_REFERENCE_KIND:
        return None
    purpose = evidence.get("purpose")
    if purpose != PURPOSE_ALLOCATION:
        return None
    receipt_id = evidence.get("receipt_id")
    if not isinstance(receipt_id, str) or not _SHA256.fullmatch(receipt_id):
        return {"status": "unverified", "reason": "the receipt identity is malformed",
                "fingerprint": content_sha256({"evidence": "malformed"})}
    try:
        store = _resolve_store(evidence.get("grant_root"))
        receipt = _verify_receipt_file(store, receipt_id)
    except PerformanceDirectorError as exc:
        return {"status": "unverified", "reason": str(exc),
                "fingerprint": content_sha256({"evidence": receipt_id, "reason": exc.code})}
    return {"status": "verified", "reason": "a real owner-issued allocation grant for this scope",
            "purpose": PURPOSE_ALLOCATION, "receipt_id": receipt_id,
            "operation_id": receipt["scope"]["operation_id"],
            "scope_sha256": receipt["scope_sha256"],
            "fingerprint": receipt_id}


def refuse_allocation_authority(claim: Any, *, _depth: int = 0) -> None:
    """Raise when a listening review tries to lean on an allocation grant.

    A grant funds work; it never says audio was heard, preferred or better.  Any
    allocation reference anywhere inside a review's authentication claim is
    refused here, before the shared private seam can see it.
    """

    if _depth > 6:
        return
    if isinstance(claim, Mapping):
        if claim.get("kind") == RECEIPT_REFERENCE_KIND or claim.get("purpose") == PURPOSE_ALLOCATION:
            raise _refuse(
                "a live-owner allocation grant is permission to run a bounded scope; it is "
                "never a listening verdict", "review_pending")
        for value in claim.values():
            refuse_allocation_authority(value, _depth=_depth + 1)
    elif isinstance(claim, (list, tuple)):
        for value in claim:
            refuse_allocation_authority(value, _depth=_depth + 1)


def authorize_operation(reference: Any, *, expected_scope: Any,
                        now: datetime | None = None) -> dict:
    """Consume a grant for one exact scope, refusing every mismatch.

    This is the anti-transplantation check: operation, destination, source,
    utterances, arms, revision, ceilings and deadline all have to be the ones the
    receipt was issued for, and the deployment's own bounds still apply.
    """

    verdict = verify_receipt(reference)
    if verdict is None:
        raise _refuse("only a live-owner receipt reference can authorize a pilot operation",
                      "capability_unsupported")
    if verdict.get("status") != "verified":
        raise _refuse("the live-owner grant does not verify: " + str(verdict.get("reason")),
                      "review_pending")
    store = _resolve_store(reference.get("grant_root"))
    receipt = _verify_receipt_file(store, verdict["receipt_id"])
    wanted = validate_scope(expected_scope, store,
                            now=now or _utc(receipt["issued_at"]))
    if scope_sha256(wanted) != receipt["scope_sha256"]:
        raise _refuse("the requested scope is not the scope this grant authorized",
                      "direction_conflict")
    deadline = _utc(receipt["scope"]["ceilings"]["deadline"])
    if deadline <= (now or datetime.now(timezone.utc)):
        raise _refuse("the grant's deadline has passed", "budget_exhausted")
    _write_once(store.grant_root / CONSUMED_DIR / (receipt["grant_id"] + ".json"), {
        "schema_version": "performance-live-owner-consumption-v1", "at": _now(),
        "grant_id": receipt["grant_id"], "operation_id": receipt["scope"]["operation_id"],
        "scope_sha256": receipt["scope_sha256"], "listening_approval": False,
        "production_enablement": False})
    return receipt


def grant_child_root(store: LiveOwnerStore, operation_id: str) -> Path:
    """The isolated per-operation directory a consumed grant funds."""

    _require_identifier(operation_id, "operation_id")
    return store.destination_root / operation_id


def bounded_deadline_hours(store: LiveOwnerStore) -> int:
    """Exposed so a runner can state the deployment's own ceiling honestly."""

    return int(store.max_deadline_hours)


def configure_live_owner(config: Mapping[str, Any]) -> LiveOwnerStore | None:
    """Install the trusted operator configuration in a service/runner process.

    No HTTP request body or receipt supplies this configuration. Separate owner
    processes load the same explicitly chosen private config before consuming
    receipts issued by the authenticated delivery process.
    """
    store = load_live_owner(config)
    if store is not None:
        previous = _CONFIGURED_STORES.get(str(store.grant_root))
        if previous is not None and previous != store:
            raise _refuse("an enabled receipt store cannot change configuration in-process",
                          "direction_conflict")
        _CONFIGURED_STORES[str(store.grant_root)] = store
    return store


def authorize_manifest(manifest: Mapping[str, Any]) -> dict:
    """Bind a live grant at the actual M3 allocation boundary, before writes."""
    grant = _require_mapping(manifest.get("authorization"), "authorization")
    reference = grant.get("authentication")
    verdict = verify_receipt(reference)
    if verdict is None or verdict.get("status") != "verified":
        raise _refuse("allocation requires a configured real owner receipt", "review_pending")
    store = _resolve_store(reference.get("grant_root"))
    receipt = _verify_receipt_file(store, verdict["receipt_id"])
    scope = _require_mapping(manifest.get("scope"), "scope")
    source_ids = scope.get("source_projects")
    if source_ids != [receipt["scope"]["source"]["project_id"]]:
        raise _refuse("the allocation source projects differ from the owner grant",
                      "direction_conflict")
    expected = {"operation_id": manifest.get("operation_id"),
                "source": {"project_id": source_ids[0],
                           "chapter_id": receipt["scope"]["source"]["chapter_id"],
                           "utterance_ids": scope.get("utterance_ids")},
                "arms": scope.get("arms"),
                "revision": manifest.get("expected_revision_sha256"),
                "ceilings": manifest.get("ceilings"),
                "destination_root": manifest.get("authorized_destination_root")}
    return authorize_operation(reference, expected_scope=expected)
