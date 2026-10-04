"""Immutable review projections and observations, separate from canonical Takes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError("Invalid review identifier")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_once(path: Path, value: dict) -> bool:
    """Publish complete JSON exclusively; a crash cannot publish partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / (".stage-" + uuid.uuid4().hex)
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
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


def batch_identity(projection: dict, recipient_id: str) -> str:
    identities = [{key: chapter[key] for key in (
        "chapter_id", "revision", "audio_sha256"
    )} for chapter in projection["chapters"]]
    payload = [projection["project_id"], recipient_id, identities]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def publish_batch(state_dir: Path, projection: dict, recipient_id: str) -> dict:
    if not projection.get("chapters"):
        raise ValueError("No reviewable chapters")
    batch_id = batch_identity(projection, recipient_id)
    batch = {**projection, "batch_id": batch_id, "recipient_id": recipient_id,
             "created_at": now()}
    target = state_dir / "manifests" / (batch_id + ".json")
    if not write_once(target, batch):
        previous = json.loads(target.read_text(encoding="utf-8"))
        if batch_identity(previous, recipient_id) != batch_id:
            raise ValueError("Stored delivery manifest identity mismatch")
        return previous
    record_event(state_dir, "artifact_reviewable", batch_id=batch_id,
                 project_id=batch["project_id"], recipient_id=recipient_id)
    return batch


def load_batch(state_dir: Path, batch_id: str) -> dict:
    safe_id(batch_id)
    value = json.loads((state_dir / "manifests" / (batch_id + ".json")).read_text(encoding="utf-8"))
    if value["batch_id"] != batch_id or batch_identity(value, value["recipient_id"]) != batch_id:
        raise ValueError("Stored delivery manifest identity mismatch")
    return value


def record_event(state_dir: Path, kind: str, **fields) -> dict:
    event = {"event_id": uuid.uuid4().hex, "kind": kind, "created_at": now(), **fields}
    write_once(state_dir / "events" / (event["event_id"] + ".json"), event)
    return event


def save_feedback(state_dir: Path, batch: dict, chapter: dict | None,
                  text: str, timestamp: float | None, recipient_id: str) -> dict:
    bindings = [{key: item[key] for key in (
        "chapter_id", "chapter_number", "revision", "audio_sha256", "export_manifest_sha256"
    )} for item in (batch["chapters"] if chapter is None else [chapter])]
    result = {"feedback_id": uuid.uuid4().hex, "created_at": now(),
              "project_id": batch["project_id"], "batch_id": batch["batch_id"],
              "recipient_id": recipient_id, "chapter_id": chapter["chapter_id"] if chapter else None,
              "playback_seconds": timestamp, "text": text, "artifact_bindings": bindings,
              "kind": "chapter" if chapter else "overall"}
    write_once(state_dir / "feedback" / (result["feedback_id"] + ".json"), result)
    record_event(state_dir, "feedback_received", feedback_id=result["feedback_id"],
                 batch_id=batch["batch_id"], chapter_id=result["chapter_id"])
    return result


def list_feedback(state_dir: Path, batch_id: str | None = None) -> list[dict]:
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in (state_dir / "feedback").glob("*.json")]
    return sorted((row for row in rows if batch_id is None or row["batch_id"] == batch_id),
                  key=lambda row: row["created_at"])
