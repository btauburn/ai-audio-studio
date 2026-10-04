"""Private audiobook notifications through the existing trip Pushover publisher.

Only immutable notification claims/outcomes are written here. A missing outcome
is deliberately ambiguous and blocks another send, including rebatched audio.
Neither the travel decision service nor either project's production state is used.
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import json
import math
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from dubbing_service.app.storage.process_lock import ProjectsRootLock


_IMPORT_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_VALIDATE_URL = "https://api.pushover.net/1/users/validate.json"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DETAIL = {
    "accepted": "Pushover accepted the request; phone display is not confirmed.",
    "failed": "The provider refused the message or the connection was not established.",
    "delivery_unknown": "The request may have reached the provider; automatic resend is blocked.",
    "skipped": "No notification was attempted.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid delivery identity")
    return value


def _private_url(value: str, *, mobile: bool = False) -> None:
    """Require a private HTTPS URL, and for a phone link a non-loopback host.

    ``mobile`` marks the link a Pushover notification carries: a loopback host is
    schema-valid and still useless on the phone, so it is refused rather than
    delivered as a knowingly unreachable review link.
    """
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        private = host.endswith(".ts.net")
        loopback = host == "localhost"
        try:
            address = ipaddress.ip_address(host)
            private = private or address in ipaddress.ip_network("100.64.0.0/10")
            loopback = address.is_loopback
        except ValueError:
            pass
        if (parsed.scheme != "https" or not private or parsed.username or parsed.password
                or len(value) > 512 or any(ord(char) < 33 for char in value)
                or (mobile and loopback)):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("A private HTTPS listening URL is required") from None


_CASTING_KIND = "casting_review"
_CASTING_EVENTS = ("CHARACTER_REVIEW_READY", "REPLACEMENT_REVIEW_READY",
                   "CASTING_COMPLETE", "CASTING_NEEDS_ATTENTION", "PRODUCTION_READY", "CHAPTER_READY")
_REVIEW_EVENTS = _CASTING_EVENTS[:2]
_TITLE_LIMIT = 250
_BODY_LIMIT = 1024
_URL_TITLE_LIMIT = 100


def _plain(value: object, limit: int) -> str:
    """Collapse horizontal whitespace, keep the author's line breaks, bound length."""
    text = "\n".join(" ".join(line.split()) for line in str(value).splitlines()).strip()
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _names(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _options(count: int) -> str:
    return f"{count} audition{'s' if count != 1 else ''}"


def _casting_rows(batch: dict) -> list[dict]:
    rows = batch.get("characters")
    if not isinstance(rows, list):
        raise ValueError("Invalid casting notification characters")
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid casting notification character")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid casting notification character name")
        options = row.get("options")
        if type(options) is not int or options < 0:
            raise ValueError("Invalid casting notification option count")
        round_id = row.get("round_id")
        parsed.append({"role_id": _safe_id(row.get("role_id")), "name": _plain(name, 60),
                       "options": options, "direction": _plain(row.get("direction") or "", 240),
                       "audio_sha256": row.get("audio_sha256"),
                       "round_id": round_id if isinstance(round_id, str) and _ID.fullmatch(round_id) else None})
    return parsed


def _casting_message(batch: dict, listening_url: str) -> tuple[dict, list[str]]:
    """Render one Auto Cast human-review push and its durable artifact identities.

    Every event but ``CASTING_COMPLETE`` must name at least one character whose
    audio the human can actually play; the badge text is never the evidence.
    """
    project_id = _safe_id(batch.get("project_id"))
    event = batch.get("event")
    if event not in _CASTING_EVENTS:
        raise ValueError("Unknown Auto Cast notification event")
    # The review link is opened on a phone, so a loopback host is refused here
    # rather than delivered as a link that cannot work.
    _private_url(listening_url, mobile=True)
    book_title = batch.get("book_title")
    if not isinstance(book_title, str) or not book_title.strip():
        raise ValueError("The casting notification has no project name")
    book_title = _plain(book_title, 120)
    total = batch.get("characters_total")
    if type(total) is not int or total < 0:
        raise ValueError("Invalid casting notification character total")
    progress = batch.get("progress")
    if progress is not None and not isinstance(progress, dict):
        raise ValueError("Invalid casting notification progress")
    rows = _casting_rows(batch)
    if event in _REVIEW_EVENTS:
        if not rows:
            raise ValueError("Invalid casting review batch")
        for row in rows:
            if not isinstance(row.get("audio_sha256"), str) or not _SHA.fullmatch(row["audio_sha256"]):
                raise ValueError("Invalid casting review audio revision")
        # A new review round is a new thing to be told about even when a re-cast
        # happens to reproduce identical audio, so the round is part of the identity.
        identities = [_digest([project_id, row["role_id"], row["audio_sha256"],
                               *([row["round_id"]] if row.get("round_id") else [])]) for row in rows]
    else:
        binding = batch.get("binding")
        if (not isinstance(binding, list) or not binding
                or any(not isinstance(item, str) or not _SHA.fullmatch(item) for item in binding)):
            raise ValueError("Invalid casting notification binding")
        identities = list(binding)
    if event == "CASTING_COMPLETE":
        title = f"✅ Casting complete: {book_title}"
        body = (f"All {total} required character{'s' if total != 1 else ''} have approved voices.\n"
                "Tap to see how your book is coming along.")
        url_title = "Review final cast"
    elif event == "CHAPTER_READY":
        ready, chapters_total = (batch.get(key) for key in ("chapters_ready", "chapters_total"))
        duration = batch.get("duration_seconds")
        chapter_title = _plain(batch.get("chapter_title") or "", 120)
        if (any(type(value) is not int or value < 0 for value in (ready, chapters_total))
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("Invalid chapter notification metadata")
        # The book's own chapter title says which chapter this is; a count of our own
        # (which need not match the book's numbering) would only contradict it.
        title = f"🎧 New chapter ready: {book_title}"
        body = (f"{chapter_title}\n" if chapter_title else "") + (
            f"~{max(1, math.floor(duration / 60 + .5))} min · {ready} of {chapters_total} "
            f"chapter{'s' if chapters_total != 1 else ''} ready.\nTap to listen now.")
        url_title = "Listen now"
    elif event == "PRODUCTION_READY":
        manifest_id = batch.get("manifest_id")
        utterances = batch.get("utterance_count")
        chapters_ready = batch.get("chapters_ready")
        if batch.get("status") == "succeeded":
            title = f"🎧 Audiobook ready: {book_title}"
            body = ("The whole book is recorded and ready to listen to.\n"
                    + (f"{chapters_ready} chapter{'s' if chapters_ready != 1 else ''} · " if type(chapters_ready) is int and chapters_ready else "")
                    + f"{utterances or total} utterance{'s' if (utterances or total) != 1 else ''} recorded"
                    + (f"\nManifest: {_plain(str(manifest_id), 40)}" if manifest_id else ""))
            url_title = "Listen now"
        else:
            reason = _plain(batch.get("reason") or "", 300)
            title = f"⚠️ Audiobook generation stopped: {book_title}"
            body = ((f"{reason}\n" if reason else "")
                    + "Tap to see what your book needs.")
            url_title = "Open Voice Lab"
    elif event == "CASTING_NEEDS_ATTENTION":
        reason = _plain(batch.get("reason") or "", 400)
        if not reason:
            raise ValueError("A needs-attention notification needs a reason")
        title = f"⚠️ Casting needs your decision: {book_title}"
        body = f"{reason}\nTap to open the casting review and decide."
        url_title = "Open Voice Lab"
    elif event == "REPLACEMENT_REVIEW_READY" or len(rows) == 1:
        row = rows[0]
        if event == "REPLACEMENT_REVIEW_READY":
            title = f"🎙 New {row['name']} voice ready"
            body = (f"Your feedback:\n\"{row['direction']}\"\n\nA new audition is ready to review."
                    if row["direction"] else "A genuinely different audition is ready to review.")
        else:
            title = f"🎙 Voice ready: {row['name']}"
            body = (f"{_options(row['options'])} {'are' if row['options'] != 1 else 'is'} ready.\n"
                    "Tap to listen, rate, and choose a voice.\n"
                    f"Project: {book_title}")
            if isinstance(progress, dict):
                approved = progress.get("characters_approved", 0)
                ready = progress.get("characters_review_ready", 0)
                if type(approved) is int and type(ready) is int and total:
                    remaining = max(0, total - approved - ready)
                    if remaining:
                        body += f"\n{remaining} / {total} characters remaining."
        url_title = f"Review {row['name']}"
    else:
        title = f"🎭 {len(rows)} characters ready to cast"
        body = (f"{_names([row['name'] for row in rows])} have new auditions ready.\n"
                f"{sum(row['options'] for row in rows)} voice options are waiting for your feedback.")
        url_title = "Open Voice Lab"
    message = {"title": _plain(title, _TITLE_LIMIT), "body": _plain(body, _BODY_LIMIT),
               "url": listening_url, "url_title": _plain(url_title, _URL_TITLE_LIMIT), "priority": 0}
    return message, sorted(identities)


def _message(batch: dict, recipient: dict, listening_url: str) -> tuple[dict, list[str]]:
    project_id = _safe_id(batch.get("project_id"))
    _safe_id(batch.get("batch_id"))
    recipient_id = _safe_id(recipient.get("id"))
    if batch.get("recipient_id") != recipient_id:
        raise ValueError("Batch recipient does not match notification recipient")
    if batch.get("kind") == _CASTING_KIND:
        return _casting_message(batch, listening_url)
    _private_url(listening_url)
    chapters = batch.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("The delivery batch has no chapters")
    identities: list[str] = []
    lines: list[tuple[int, str]] = []
    numbers: set[int] = set()
    chapter_ids: set[str] = set()
    total = 0.0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise ValueError("Invalid chapter notification metadata")
        chapter_id = _safe_id(chapter.get("chapter_id"))
        label = chapter.get("chapter_label")
        if label is not None:
            if (not isinstance(label, str) or not label.strip()
                    or any(ord(char) < 32 and not char.isspace() for char in label)):
                raise ValueError("Invalid chapter title for notification")
            label = " ".join(label.split())[:160]
            # This is only display ordering, never an inferred chapter number.
            number = chapter.get("order")
        else:
            number = chapter.get("chapter_number")
        duration = chapter.get("duration_seconds")
        audio_hash = chapter.get("audio_sha256")
        if (type(number) is not int or number < 0 or number in numbers
                or chapter_id in chapter_ids or not isinstance(audio_hash, str)
                or not _SHA.fullmatch(audio_hash) or isinstance(duration, bool)
                or not isinstance(duration, (int, float)) or not math.isfinite(duration)
                or duration <= 0):
            raise ValueError("Invalid chapter notification metadata")
        numbers.add(number)
        chapter_ids.add(chapter_id)
        total += duration
        # Same bytes for the same canonical chapter remain the same artifact even
        # if a batch, export manifest or source revision has been regrouped.
        identities.append(_digest([project_id, chapter_id, audio_hash]))
        lines.append((number, f"{label or f'Chapter {number}'} · ~{max(1, math.floor(duration / 60 + .5))} min"))
    book_title = batch.get("book_title")
    if not isinstance(book_title, str) or not book_title.strip():
        raise ValueError("The delivery batch has no book title")
    title = " ".join(book_title.upper().split())
    count = len(chapters)
    suffix = f" · {count} CHAPTER{'S' if count != 1 else ''}"
    title = title[:250 - len(suffix)] + suffix
    total_line = f"~{max(1, math.floor(total / 60 + .5))} min total. Ready to listen and review."
    rendered_lines = [line for _, line in sorted(lines)]
    if all(chapter.get("chapter_label") is not None for chapter in chapters):
        # A large completed book still receives one bounded notification. Its
        # private link exposes every export; omitted labels are never invented.
        kept = []
        for line in rendered_lines:
            if len("\n".join(kept + [line, total_line])) > 900:
                break
            kept.append(line)
        if len(kept) < len(rendered_lines):
            kept.append(f"{len(rendered_lines) - len(kept)} more chapters available in Voice Library.")
        rendered_lines = kept
    body = "\n".join(rendered_lines + [total_line])
    if len(body) > 1024:
        raise ValueError("Too many chapters for one notification")
    return {"title": title, "body": body, "url": listening_url,
            "url_title": "LISTEN & REVIEW", "priority": 0}, sorted(identities)


class _ConservativeTransport:
    """Preserve the shared sender while tightening audiobook retry semantics.

    A server can accept a POST before failing to return its successful response.
    Translate that uncertainty into the shared publisher's existing protocol-error
    result before its retry loop sees it. No HTTP implementation is duplicated.
    """

    def __init__(self, transport, response_type):
        self.transport = transport
        self.response_type = response_type

    def post_form(self, url: str, data: dict, *, timeout: float):
        response = self.transport.post_form(url, data, timeout=timeout)
        status = response.status_code
        ambiguous = type(status) is int and status >= 500
        if status == 200:
            payload = response.payload()
            # Only an explicit integer 0 is a provider refusal. Missing/unknown
            # status, invalid JSON, or an unexpected status type is ambiguous.
            ambiguous = (not isinstance(payload, dict)
                         or type(payload.get("status")) is not int
                         or payload["status"] not in (0, 1))
        if ambiguous:
            return self.response_type(
                status_code=status, error_kind="protocol",
                error_note="The provider response is ambiguous; automatic resend is blocked.",
            )
        return response


def _load_transport(transport_root: Path):
    """Import only the configured checkout, without invoking a travel command."""
    source = Path(transport_root).resolve() / "src"
    if not (source / "trip_research" / "notify" / "publishers.py").is_file():
        raise ValueError("Notification transport is unavailable")
    with _IMPORT_LOCK:
        loaded = sys.modules.get("trip_research")
        if loaded and Path(loaded.__file__).resolve().parent != source / "trip_research":
            raise ValueError("Notification transport checkout does not match")
        sys.path.insert(0, str(source))
        try:
            credentials = importlib.import_module("trip_research.credentials")
            configuration = importlib.import_module("trip_research.notify.config")
            publishers = importlib.import_module("trip_research.notify.publishers")
            formatting = importlib.import_module("trip_research.notify.format")
            pushover = importlib.import_module("trip_research.notify.pushover")
        finally:
            sys.path.remove(str(source))
    config = configuration.notification_config(credentials.merged_env(root=transport_root))
    if not config.enabled or config.notes:
        # An invalid device must not silently broaden an audiobook notification.
        raise ValueError("Notification credentials or device configuration are unavailable")
    transport = pushover.HttpxTransport()
    message_transport = _ConservativeTransport(transport, pushover.PushoverResponse)
    return config, publishers.PushoverPublisher(config, message_transport), formatting.RenderedNotification, transport


def validate_credentials(transport_root: Path) -> dict:
    """Validate account/device without sending a message or exposing raw replies."""
    result = {"provider": "pushover", "configured": False, "valid": None,
              "state": "disabled", "http_status": None, "device_count": None}
    try:
        config, _, _, transport = _load_transport(transport_root)
    except Exception:
        return result
    result["configured"] = True
    fields = {"token": config.app_token, "user": config.user_key}
    if config.device:
        fields["device"] = config.device
    try:
        response = transport.post_form(_VALIDATE_URL, fields, timeout=config.timeout_seconds)
        status = response.status_code
        result["http_status"] = status if type(status) is int and 100 <= status <= 599 else None
        payload = response.payload()
        if status == 200 and isinstance(payload, dict) and payload.get("status") == 1:
            result.update(valid=True, state="validated")
            devices = payload.get("devices")
            if isinstance(devices, list):
                result["device_count"] = len(devices)
        elif (isinstance(payload, dict) and payload.get("status") == 0
              or type(status) is int and 400 <= status < 500):
            result.update(valid=False, state="rejected")
        else:
            result["state"] = "validation_unknown"
    except Exception:
        result["state"] = "validation_unknown"
    return result


def _write_once(path: Path, payload: dict) -> None:
    """Never truncate an existing record; partial writes remain fail-closed."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _blocker(directory: Path, recipient_id: str, project_id: str,
             batch_id: str, artifacts: list[str]) -> str | None:
    for path in (directory / "intents").glob("*.json"):
        claim = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(claim, dict) or claim.get("schema_version") != 1
                or claim.get("attempt_id") != path.stem
                or not isinstance(claim.get("artifact_ids"), list)
                or not claim["artifact_ids"]
                or not all(isinstance(claim.get(key), str) and _ID.fullmatch(claim[key])
                           for key in ("recipient_id", "project_id", "batch_id"))
                or not all(isinstance(item, str) and _SHA.fullmatch(item)
                           for item in claim["artifact_ids"])):
            raise ValueError("Notification history is unreadable")
        if claim.get("recipient_id") != recipient_id or claim.get("project_id") != project_id:
            continue
        if claim.get("batch_id") != batch_id and not set(artifacts).intersection(claim["artifact_ids"]):
            continue
        outcome_path = directory / "outcomes" / path.name
        if not outcome_path.exists():
            return "delivery_unknown"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        if not isinstance(outcome, dict) or outcome.get("attempt_id") != path.stem:
            raise ValueError("Notification history is unreadable")
        state = outcome.get("state")
        if state not in {"failed", "skipped"}:
            return state if state in {"accepted", "pending", "delivery_unknown"} else "delivery_unknown"
    return None


def _safe_result(raw, config) -> dict:
    state = getattr(getattr(raw, "state", None), "value", getattr(raw, "state", None))
    if state not in _DETAIL:
        state = "delivery_unknown"
    attempts = getattr(raw, "attempts", None)
    http_status = getattr(raw, "http_status", None)
    request_id = getattr(raw, "request_id", None)
    if (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)
            or any(secret and secret in request_id for secret in (config.app_token, config.user_key))):
        request_id = None
    return {"state": state, "detail": _DETAIL[state], "provider": "pushover",
            "accepted": state == "accepted", "device_delivery_confirmed": False,
            "attempts": attempts if type(attempts) is int and 0 <= attempts <= 2 else None,
            "http_status": http_status if type(http_status) is int and 100 <= http_status <= 599 else None,
            "provider_request_id": request_id}


def send_notification(batch: dict, recipient: dict, *, state_dir: Path,
                      transport_root: Path, listening_url: str, force: bool = False) -> dict:
    """Claim, send once through the shared publisher, and preserve safe evidence.

    ``state_dir`` is the audiobook project's delivery directory. An overlapping
    batch is suppressed in full; callers can construct a batch of only new audio.
    ``force=True`` is reserved for an explicitly requested intentional resend.
    """
    base = {"provider": "pushover", "state": "skipped", "accepted": False,
            "device_delivery_confirmed": False, "attempts": 0,
            "http_status": None, "provider_request_id": None}
    if recipient.get("enabled") is not True:
        return {**base, "detail": "Recipient is disabled."}
    if recipient.get("notification_transport") != "pushover":
        return {**base, "detail": "Recipient notification transport is disabled."}
    try:
        message, artifacts = _message(batch, recipient, listening_url)
        config, publisher, message_type, _ = _load_transport(transport_root)
        rendered = message_type(**message)
    except Exception:
        return {**base, "detail": "Notification metadata or transport configuration is unavailable."}
    directory = Path(state_dir).resolve() / "notifications"
    attempt_id = uuid.uuid4().hex
    claim = {"schema_version": 1, "attempt_id": attempt_id,
             "intent_id": _digest([recipient["id"], batch["project_id"], batch["batch_id"]]),
             "recipient_id": recipient["id"], "project_id": batch["project_id"],
             "batch_id": batch["batch_id"], "artifact_ids": artifacts,
             "message_sha256": _digest(message), "created_at": _now(),
             "state": "pending", "forced": force is True}
    try:
        # The shared primitive is reentrant per process, so threads need their
        # own serialization too. Its root is delivery-local, never projects/.
        with _STATE_LOCK, ProjectsRootLock(directory, label="audiobook-notification").hold(timeout=5):
            for name in ("intents", "outcomes"):
                (directory / name).mkdir(exist_ok=True)
            blocked = _blocker(directory, recipient["id"], batch["project_id"], batch["batch_id"], artifacts)
            if blocked and force is not True:
                return {**base, "detail": "This batch or its audio already has a notification intent.",
                        "duplicate": True, "previous_state": blocked}
            _write_once(directory / "intents" / f"{attempt_id}.json", claim)
    except Exception:
        return {**base, "detail": "Notification history could not be safely claimed."}
    try:
        raw = publisher.send(rendered, timeout=config.timeout_seconds)
        outcome = _safe_result(raw, config)
    except Exception:
        # An unexpected exception may happen after submission. Never call it a
        # known failure, and never propagate the exception's possible secrets.
        outcome = {**base, "state": "delivery_unknown", "attempts": None,
                   "detail": _DETAIL["delivery_unknown"]}
    outcome.update(schema_version=1, attempt_id=attempt_id, intent_id=claim["intent_id"],
                   recipient_id=recipient["id"], project_id=batch["project_id"],
                   batch_id=batch["batch_id"], recorded_at=_now(), forced=force is True)
    try:
        _write_once(directory / "outcomes" / f"{attempt_id}.json", outcome)
        outcome["outcome_recorded"] = True
    except Exception:
        # The pre-send claim remains a durable blocker even when this write fails.
        outcome["outcome_recorded"] = False
    return outcome
