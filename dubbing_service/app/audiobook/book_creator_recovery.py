"""Bounded recovery inside Studio's existing lifespan and durable job owner.

No worker threads, daemon, or queue live here. The application injects its
existing background-task scheduler. A live or uncertain remote request is
never taken over merely because time has elapsed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dubbing_service.app.core.config import get_config
from dubbing_service.app.core.tts_operations import has_active_tts_operation
from dubbing_service.app.storage.projects import load_project, save_project, project_meta_lock
from dubbing_service.app.storage.jobs import get_job, is_job_active

BACKOFF_SECONDS = (30, 120)
MAX_PROJECTS_PER_TICK = 64
MAX_REQUESTS_PER_TICK = 100
# Ephemeral enumeration positions only: durable request receipts still own
# work/attempt state. Rotation prevents fixed-prefix starvation on large roots.
_project_after = None
_request_after = {}


def _rotate(items, after, key):
    if after is None or not items:
        return items
    for index, item in enumerate(items):
        if key(item) > after:
            return items[index:] + items[:index]
    return items


def _now(now=None):
    return now or datetime.now(timezone.utc)


def _date(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def defaults():
    return {"enabled": True, "paused": False, "retry_count": 0, "max_retries": 2,
            "next_retry_at": None, "state": "watching", "reason": None}


def public_automation(receipt):
    value = receipt.get("automation") or {}
    return {"enabled": value.get("enabled") is True, "paused": value.get("paused", False),
            "retry_count": value.get("retry_count", 0), "max_retries": 2,
            "next_retry_at": value.get("next_retry_at"), "state": value.get("state", "disabled"),
            "reason": value.get("reason")}


def public_notification(receipt):
    value = receipt.get("notification") or {}
    result = {key: value[key] for key in ("state", "status", "detail", "message", "sent_at", "accepted_at", "updated_at",
              "attempts", "next_retry_at", "accepted", "device_delivery_confirmed", "review_url", "export_ids") if key in value}
    result.setdefault("state", result.get("status", "pending" if receipt.get("notify_on_completion") is True else "disabled"))
    result.setdefault("status", result["state"])
    result.setdefault("detail", result.get("message"))
    result.setdefault("message", result["detail"])
    return result


def selections(segments):
    return {s.id: {"take_id": s.selected_take_id,
                  "audio_sha256": next((t.audio_sha256 for t in s.takes if t.id == s.selected_take_id), None)} for s in segments}


def set_automation(project_id, request_id, paused):
    from . import book_creator as creator
    creator._request_id(request_id)
    with project_meta_lock(project_id):
        project, meta = creator._owned(project_id)
        receipt = meta["requests"].get(request_id)
        if not receipt:
            raise creator.BookCreatorError("This book has no matching render request")
        if "automation" not in receipt:
            # Historical work requires explicit opt-in; never arm it in a scan.
            receipt["automation"] = {**defaults(), "enabled": not paused}
        automation = receipt["automation"]
        automation.update(paused=paused, enabled=automation.get("enabled", False) or not paused,
                          next_retry_at=None, state="paused" if paused else "watching",
                          reason="Paused; current utterance may finish before stopping" if paused else None)
        save_project(project)
    return creator.request_status(project_id, request_id)


def paused(project_id, request_id):
    from . import book_creator as creator
    _, meta = creator._owned(project_id)
    return meta["requests"][request_id].get("automation", {}).get("paused") is True


def mark_attempt(project_id, request_id, *, state, reason=None, inflight=None, segment=None):
    """Commit the uncertainty marker and selection checkpoint to the receipt."""
    from . import book_creator as creator
    with project_meta_lock(project_id):
        project, meta = creator._owned(project_id)
        receipt = meta["requests"][request_id]
        receipt["inflight_utterance_id"] = inflight
        if segment is not None:
            receipt.setdefault("selection_checkpoint", {}).update(selections([segment]))
        if "automation" in receipt:
            automation = receipt["automation"]
            automation.update(state="paused" if automation.get("paused") else state,
                              reason=reason, next_retry_at=None)
        save_project(project)


def retry_guard(project, meta, receipt, segments):
    from . import book_creator as creator
    creator._chosen_voice(project, meta)
    speech = [s for s in segments if s.id in receipt["segment_ids"]]
    intent = creator._hash({**{key: receipt[key] for key in ("chapter_ids", "library_voice_id", "voice_version")},
                           "source_rows": meta["source_rows"], "requirements": creator._binding(project, speech)})
    if intent != receipt["intent"]:
        raise creator.BookCreatorError("Source or narrator changed; automatic retry is held")
    if receipt.get("inflight_utterance_id"):
        raise creator.BookCreatorError("Worker outcome is uncertain; automatic retry is held to avoid duplicate audio. Inspect the worker before manually resuming")
    if receipt.get("selection_checkpoint") != selections(speech):
        raise creator.BookCreatorError("Selected Takes changed or no recovery checkpoint exists; automatic retry is held")


def recoverable_idle_book_job(project_id, job):
    """Read-only proof that a persisted peer is recovery work, not live priority.

    Any absent, malformed, stale, disabled, uncertain or active state keeps the
    normal production priority. This does not rewrite a job or clear a marker.
    """
    from . import book_creator as creator
    try:
        if (job.project_id != project_id or job.job_type != "batch_synthesize"
                or job.status.value not in {"queued", "running"} or job.stop_requested
                or is_job_active(project_id, job.id)):
            return False
        project, meta, segments = creator._source(project_id)
        matches = [receipt for receipt in meta["requests"].values() if receipt.get("job_id") == job.id]
        if len(matches) != 1:
            return False
        receipt = matches[0]
        value = receipt.get("automation") or {}
        if (value.get("enabled") is not True or value.get("paused") or value.get("retry_count", 0) >= 2
                or receipt.get("inflight_utterance_id") or job.intent_hash != receipt["intent"]
                or list(job.segment_ids) != receipt["segment_ids"]):
            return False
        retry_guard(project, meta, receipt, segments)
        return True
    except Exception:
        # Unprovable state must retain priority, never grant a lane exemption.
        return False


def reconcile_committed_utterance(project_id, request_id):
    """Recognize a fully committed exact Take after loss of its owner process."""
    from . import book_creator as creator
    from dubbing_service.app.storage.segments import project_segments_lock
    with project_meta_lock(project_id), project_segments_lock(project_id):
        project, meta, segments = creator._source(project_id)
        receipt = meta["requests"][request_id]
        sid = receipt.get("inflight_utterance_id")
        if not sid or is_job_active(project_id, receipt["job_id"]):
            return False
        checkpoint = receipt.get("selection_checkpoint") or {}
        speech = [s for s in segments if s.id in receipt["segment_ids"]]
        actual = selections(speech)
        if any(actual.get(key) != value for key, value in checkpoint.items() if key != sid) or set(actual) != set(checkpoint):
            return False
        segment = next((s for s in speech if s.id == sid), None)
        if segment is None or not segment.selected_take_id:
            return False
        creator._chosen_voice(project, meta)
        # This verifies decoded bytes, QA/hash, provenance, final conditioning,
        # natural speed and the exact current input fingerprint; cards or a
        # selected pointer alone do not prove completed external work.
        creator._selected_audio(project, [segment])
        trial = {**receipt, "inflight_utterance_id": None, "selection_checkpoint": actual}
        retry_guard(project, meta, trial, segments)
        receipt.update(inflight_utterance_id=None, selection_checkpoint=actual)
        save_project(project)
        return True


def claim_retry(project_id, request_id, now=None):
    """Called inside render_book's dispatch mutex, before reserving live work."""
    from . import book_creator as creator
    now = _now(now)
    with project_meta_lock(project_id):
        project, meta = creator._owned(project_id)
        receipt = meta["requests"][request_id]
        value = receipt.get("automation") or {}
        due = _date(value.get("next_retry_at"))
        job = get_job(project_id, receipt["job_id"])
        if (not value.get("enabled") or value.get("paused") or value.get("retry_count", 0) >= 2
                or due is None or due > now or not job or job.stop_requested or job.status.value == "cancelled"
                or is_job_active(project_id, receipt["job_id"])):
            return False
        value.update(retry_count=value.get("retry_count", 0) + 1, next_retry_at=None,
                     state="retrying", reason="Resuming valid unfinished audio", last_retry_at=now.isoformat())
        save_project(project)
        return True


def _set_state(project_id, request_id, state, reason=None, *, due=None):
    from . import book_creator as creator
    with project_meta_lock(project_id):
        project, meta = creator._owned(project_id)
        value = meta["requests"][request_id].get("automation")
        if not value or value.get("paused"):
            return
        patch = {"state": state, "reason": reason, "next_retry_at": due}
        if any(value.get(k) != v for k, v in patch.items()):
            value.update(patch)
            save_project(project)


def maintenance_tick(*, now=None, schedule=None):
    """One bounded scan; at most one retry dispatch, always through render_book.

    schedule=None records due work without starting it. The lifespan provides a
    thread-safe scheduler onto its own tracked asyncio tasks.
    """
    from . import book_creator as creator
    global _project_after
    now = _now(now)
    result = {"scanned": 0, "due": 0, "dispatched": 0, "held": 0, "errors": 0}
    paths = sorted(get_config().projects_root_abs.glob("*/project.json"))
    paths = _rotate(paths, _project_after, lambda path: path.parent.name)[:MAX_PROJECTS_PER_TICK]
    for path in paths:
        if result["scanned"] >= MAX_REQUESTS_PER_TICK:
            return result
        _project_after = path.parent.name
        try:
            project = load_project(path.parent.name)
            meta = project.metadata.get("book_creator") or {}
            if meta.get("import_status") != "ready":
                continue
            requests = sorted(meta.get("requests", {}).items())
            requests = _rotate(requests, _request_after.get(project.id), lambda row: row[0])
            for request_id, receipt in requests[:MAX_REQUESTS_PER_TICK]:
                if result["scanned"] >= MAX_REQUESTS_PER_TICK:
                    return result
                result["scanned"] += 1
                _request_after[project.id] = request_id
                value = receipt.get("automation") or {}
                if value.get("enabled") is not True or value.get("paused"):
                    continue
                job = get_job(project.id, receipt["job_id"])
                if is_job_active(project.id, receipt["job_id"]):
                    continue
                if not job or job.stop_requested or job.status.value == "cancelled":
                    _set_state(project.id, request_id, "blocked", "Cancelled, stopped, or missing job; no automatic retry")
                    result["held"] += 1
                    continue
                if job.status.value == "succeeded" and creator._exports_complete(meta, receipt):
                    _set_state(project.id, request_id, "completed")
                    continue
                if value.get("retry_count", 0) >= len(BACKOFF_SECONDS):
                    _set_state(project.id, request_id, "exhausted", "Two automatic retries used; inspect the issue before resuming manually")
                    result["held"] += 1
                    continue
                try:
                    if receipt.get("inflight_utterance_id"):
                        reconcile_committed_utterance(project.id, request_id)
                    fresh, fresh_meta, segments = creator._source(project.id)
                    retry_guard(fresh, fresh_meta, fresh_meta["requests"][request_id], segments)
                except (ValueError, OSError) as exc:
                    _set_state(project.id, request_id, "blocked", str(exc)[:500])
                    result["held"] += 1
                    continue
                if has_active_tts_operation() or creator._pending_production(project.id, receipt["job_id"]):
                    _set_state(project.id, request_id, "waiting_for_production", "Existing production has priority", due=value.get("next_retry_at"))
                    continue
                due = _date(value.get("next_retry_at"))
                if due is None:
                    due = now + timedelta(seconds=BACKOFF_SECONDS[value.get("retry_count", 0)])
                    _set_state(project.id, request_id, "waiting", job.message or "Interrupted work will resume", due=due.isoformat())
                if due > now:
                    continue
                result["due"] += 1
                if schedule is None:
                    continue
                outcome = creator.render_book(project.id, **{key: receipt[key] for key in ("chapter_ids", "library_voice_id", "voice_version")},
                                              request_id=request_id, schedule=schedule, automatic=True, recovery_now=now)
                if outcome.get("automation", {}).get("state") == "retrying" or is_job_active(project.id, receipt["job_id"]):
                    result["dispatched"] += 1
                return result
        except (ValueError, OSError, KeyError) as exc:
            result["errors"] += 1
    return result
