"""Opted-in completion notifications through the existing private delivery owner.

The project receipt contains only a safe status projection and bounded retry
counters. Existing delivery notification claims/outcomes remain authoritative
for at-most-once submission and ambiguous provider acceptance. This module has
no timer, queue, GPU work, export generation, or transport implementation.
"""
from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.models.audiobook import TakeQaReport
from dubbing_service.app.models.jobs import SynthesisJobStatus
from dubbing_service.app.storage.jobs import get_job
from dubbing_service.app.storage.projects import load_project, project_meta_lock, save_project
from . import book_creator as creator
from .delivery import config as delivery_config, notify
from .fingerprint import file_sha256

MAX_ATTEMPTS = 3
RETRY_DELAYS = (30, 120)
MAX_SENDS_PER_TICK = 2
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FINAL = {"accepted", "delivery_unknown", "blocked"}
_MESSAGES = {
    "not_requested": "Completion notification was not requested.",
    "waiting_for_exports": "Waiting for every requested MP3 export to finish and pass verification.",
    "pending": "Checking completed recordings and preparing the private notification.",
    "accepted": "Pushover accepted the notification. Display on your phone is not confirmed.",
    "delivery_unknown": "The notification may have reached Pushover. Automatic resend is held.",
    "failed": "The notification was not accepted. A bounded retry is scheduled if attempts remain.",
    "skipped": "Notification configuration or transport is unavailable. A bounded retry is scheduled if attempts remain.",
    "blocked": "Completed recording verification failed. No notification was sent.",
}


def _now():
    return datetime.now(timezone.utc)


def _status(state, *, attempts=0, **fields):
    message = _MESSAGES[state]
    return {"state": state, "status": state, "detail": message, "message": message,
            "accepted": state == "accepted", "device_delivery_confirmed": False,
            "attempts": attempts, "next_retry_at": None, "updated_at": _now().isoformat(), **fields}


def _request(project_id, request_id):
    creator._request_id(request_id)
    project, meta = creator._owned(project_id)
    receipt = meta["requests"].get(request_id)
    if receipt is None:
        raise creator.BookCreatorError("This book has no matching notification request")
    return project, meta, receipt


def _save_status(project_id, request_id, state):
    with project_meta_lock(project_id):
        project, _, receipt = _request(project_id, request_id)
        receipt["notification"] = copy.deepcopy(state)
        save_project(project)
    return state


def _due(state):
    if not isinstance(state, dict):
        return False
    attempts = state.get("attempts", 0)
    if type(attempts) is not int or attempts < 0 or state.get("state") in _FINAL or attempts >= MAX_ATTEMPTS:
        return False
    due = state.get("next_retry_at")
    if not due:
        return True
    try:
        return datetime.fromisoformat(due) <= _now()
    except (TypeError, ValueError, OverflowError):
        return False  # Corrupt retry metadata must not increase message traffic.


def _verified_exports(project, meta, receipt):
    job = get_job(project.id, receipt["job_id"])
    if (not job or job.status != SynthesisJobStatus.SUCCEEDED or job.stop_requested
            or job.failed or job.done != job.total or not creator._exports_complete(meta, receipt)):
        return None
    root = safe_project_root(project.id)
    records = [meta["exports"][key] for key in receipt["export_ids"]]
    for record in records:
        export_id = record["export_id"]
        if not re.fullmatch(r"mp3_[a-f0-9]{28}", export_id):
            raise ValueError("Invalid immutable export identity")
        audio = creator._safe(root, record["audio_file"])
        master = creator._safe(root, record["master_file"])
        if (audio.suffix != ".mp3" or master.suffix != ".wav" or audio.parent != master.parent
                or not _SHA.fullmatch(record.get("audio_sha256", ""))
                or not _SHA.fullmatch(record.get("master_sha256", ""))
                or file_sha256(audio) != record["audio_sha256"]
                or file_sha256(master) != record["master_sha256"]):
            raise ValueError("Completed recording bytes changed")
        immutable = json.loads((audio.parent / "receipt.json").read_text(encoding="utf-8"))
        if immutable != record:
            raise ValueError("Published recording differs from its immutable receipt")
        report = TakeQaReport.model_validate(record["qa_report"])
        if (not report.passed or not report.matches(export_id, record["master_sha256"])
                or report.blocking_failures() or not report.executed_check_ids()):
            raise ValueError("Completed recording lacks exact hash-bound technical QA")
        expected = [row["utterance_id"] for row in meta["source_rows"] if row["chapter_id"] in record["chapter_ids"]]
        mapping = record.get("map", [])
        duration = record.get("duration_seconds")
        if ([row.get("utterance_id") for row in mapping] != expected
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError("Completed recording source coverage or duration changed")
    return records


def _private_config():
    config = delivery_config.load_config()
    if Path(config["project_dir"]).resolve().parent != get_config().projects_root_abs.resolve():
        raise ValueError("Notification configuration belongs to a different Studio projects root")
    recipients = config.get("recipients", {})
    if (set(recipients) != {"brian", "mk"} or recipients["mk"].get("enabled") is not False
            or recipients["brian"].get("enabled") is not True):
        raise ValueError("Only Brian's configured private recipient may be notified")
    recipient = delivery_config.enabled_recipient(config, "brian")
    return config, {**recipient, "id": "brian"}


def _batch(project, receipt, records):
    chapters = {chapter.id: chapter for chapter in project.chapters}
    single = {record["chapter_ids"][0]: record for record in records if len(record["chapter_ids"]) == 1}
    combined = next(record for record in records if record["chapter_ids"] == receipt["chapter_ids"])
    batch_id = "creator_" + notify._digest([project.id, receipt["job_id"],
        [(record["export_id"], record["audio_sha256"]) for record in records]])[:48]
    batch = {"project_id": project.id, "batch_id": batch_id, "recipient_id": "brian", "book_title": project.name,
        "chapters": [{"chapter_id": chapter_id, "order": index, "chapter_label": chapters[chapter_id].title,
            "duration_seconds": single[chapter_id]["duration_seconds"],
            "audio_sha256": single[chapter_id]["audio_sha256"]} for index, chapter_id in enumerate(receipt["chapter_ids"])]}
    review_url = "/voice-lab?" + urlencode({"home": "1", "book": project.id, "output": combined["export_id"]})
    return batch, review_url


def notify_request(project_id: str, request_id: str) -> dict:
    """Notify an explicitly opted-in, fully verified completed request once."""
    creator._request_id(request_id)
    # This is a process-local execution lock, not another notification store.
    # Cross-process/overlap correctness belongs to send_notification's ledger.
    with project_meta_lock("creator_notify_" + project_id + "_" + request_id):
        project, meta, receipt = _request(project_id, request_id)
        previous = receipt.get("notification", {})
        if receipt.get("notify_on_completion") is not True:
            return _status("not_requested")
        if not _due(previous):
            return copy.deepcopy(previous)
        try:
            records = _verified_exports(project, meta, receipt)
            if records is None:
                # Do not write or spend retries on every maintenance pass while
                # GPU work or export assembly is still running.
                return _status("waiting_for_exports", attempts=previous.get("attempts", 0))
            batch, review_url = _batch(project, receipt, records)
        except (ValueError, OSError, KeyError, TypeError):
            return _save_status(project_id, request_id, _status("blocked", attempts=previous.get("attempts", 0)))
        attempts = previous.get("attempts", 0) + 1
        # Counters are persisted before entering the sender. On a restart its
        # durable claim/outcome suppresses accepted or ambiguous prior sends.
        projected = _status("pending", attempts=attempts, review_url=review_url,
                            export_ids=[record["export_id"] for record in records])
        _save_status(project_id, request_id, projected)
        try:
            config, recipient = _private_config()
        except Exception:
            outcome = {"state": "skipped"}
        else:
            try:
                outcome = notify.send_notification(batch, recipient,
                    state_dir=safe_project_root(project_id) / "delivery",
                    transport_root=Path(config["notification_transport_root"]),
                    listening_url=config["listening_base_url"] + review_url)
            except Exception:
                # An unexpected sender exception may follow submission. Do not
                # copy exception text or automatically retry an uncertain send.
                outcome = {"state": "delivery_unknown"}
        state = outcome.get("previous_state") if outcome.get("duplicate") else outcome.get("state")
        if state not in {"accepted", "failed", "skipped", "delivery_unknown"}:
            state = "delivery_unknown"
        result = _status(state, attempts=attempts, review_url=review_url, export_ids=projected["export_ids"])
        for field in ("provider_request_id", "attempt_id"):
            value = outcome.get(field)
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
                result[field] = value
        if state in {"failed", "skipped"} and attempts < MAX_ATTEMPTS:
            result["next_retry_at"] = (_now() + timedelta(seconds=RETRY_DELAYS[attempts - 1])).isoformat()
        elif state in {"failed", "skipped"}:
            result["message"] = result["detail"] = "Notification retries are exhausted. No further automatic sends will be attempted."
        return _save_status(project_id, request_id, result)


def request_notification(project_id: str, request_id: str) -> dict:
    """Explicit normal-owner operation; enables this request, never a backlog."""
    with project_meta_lock(project_id):
        project, _, receipt = _request(project_id, request_id)
        if receipt.get("notify_on_completion") is not True:
            receipt["notify_on_completion"] = True
            save_project(project)
    return notify_request(project_id, request_id)


def maintenance_tick() -> list[dict]:
    """One bounded pass for Studio's existing lifespan maintenance loop."""
    results = []
    for path in sorted(get_config().projects_root_abs.glob("*/project.json")):
        try:
            project = load_project(path.parent.name)
            meta = project.metadata.get("book_creator", {})
            if meta.get("schema") != "studio-book-creator-v1":
                continue
            for request_id, receipt in meta.get("requests", {}).items():
                if receipt.get("notify_on_completion") is not True or not _due(receipt.get("notification", {})):
                    continue
                job = get_job(project.id, receipt["job_id"])
                if (not job or job.status != SynthesisJobStatus.SUCCEEDED
                        or not creator._exports_complete(meta, receipt)):
                    continue
                state = notify_request(project.id, request_id)
                results.append({"project_id": project.id, "request_id": request_id, **state})
                if len(results) >= MAX_SENDS_PER_TICK:
                    return results
        except Exception:
            # No raw exception or project text enters shared maintenance logs.
            # One malformed project does not prevent another completed request.
            continue
    return results
