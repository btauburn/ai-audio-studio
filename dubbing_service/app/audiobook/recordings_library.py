"""Read-only catalog of existing published recordings; never an audio authority."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.parse import urlencode

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.storage.projects import load_project, ProjectStorageError
from .delivery import store
from .delivery.config import load_config
from .voice_lab_render import _path

log = logging.getLogger(__name__)


def _identity(kind, *parts):
    return kind + "_" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:28]


def _records():
    """Yield public metadata with a private path used only by the media resolver."""
    for file in get_config().projects_root_abs.glob("*/project.json"):
        try:
            project = load_project(file.parent.name)
            root = safe_project_root(project.id)
            meta = project.metadata.get("book_creator") or {}
            candidates = [("book_creator", key, record) for key, record in meta.get("exports", {}).items()]
            excerpt = project.voice_lab.excerpt or {}
            if excerpt.get("audio_file") and excerpt.get("audio_sha256"):
                candidates.append(("voice_lab", _identity("lab", project.id, excerpt["audio_sha256"]), excerpt))
            # Auto Cast chapters are published one by one as each is recorded.
            for receipt in (project.voice_lab.casting.get("productions") or {}).values():
                if receipt.get("status") == "retracted":
                    continue
                for chapter in (receipt.get("chapters") or {}).values():
                    if (chapter.get("status") == "ready" and chapter.get("audio_file") and chapter.get("audio_sha256")
                            and str(chapter.get("export_id", "")).startswith("cast_")):
                        number = chapter.get("number")
                        label = (f"Chapter {number}: " if isinstance(number, int) else "") + str(chapter.get("title") or "Chapter")
                        candidates.append(("autocast", chapter["export_id"], {**chapter, "label": label}))
            for kind, ident, record in candidates:
                try:
                    path = _path(root, record["audio_file"])
                    if not path.is_file() or path.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
                        continue
                    if kind == "book_creator":
                        url = f"/api/v1/book-creator/{project.id}/audio/{ident}"
                        review = "/web/voice-lab.html?" + urlencode({"home": 1, "book": project.id, "output": ident})
                    elif kind == "autocast":
                        url = f"/api/v1/voice-library/recordings/{project.id}/{ident}/audio"
                        review = "/web/voice-lab.html?" + urlencode({"project": project.id, "view": "casting",
                                                                      "chapter": record.get("chapter_id") or ""})
                    else:
                        url = f"/api/v1/voice-library/recordings/{project.id}/{ident}/audio"
                        review = "/web/voice-lab.html?" + urlencode({"project": project.id})
                    public = {"project_id": project.id, "project_name": project.name, "kind": kind,
                              "export_id": ident, "label": record.get("label") or record.get("chapter_title") or project.name,
                              "duration_seconds": record["duration_seconds"], "audio_sha256": record["audio_sha256"],
                              "technical_qa": record.get("technical_qa", "unknown"), "human_review": record.get("human_review", "pending"),
                              "audio_url": url, "download_url": url + "?download=true", "review_url": review,
                              "created_at": (record.get("qa_report") or {}).get("evaluated_at") or record.get("created_at") or ""}
                    yield public, path
                except (OSError, ValueError, KeyError):
                    log.warning("Skipping unavailable published recording in %s", project.id)
        except (OSError, ValueError, ProjectStorageError):
            log.warning("Skipping unreadable catalog project %s", file.parent.name)
    # Existing private delivery manifests are already approved projections. Do
    # not discover arbitrary output directories or expose filesystem locations.
    try:
        config = load_config()
        roots = [Path(p).resolve() for p in config["allowed_audio_roots"]]
        state = Path(config["state_dir"])
    except (OSError, ValueError, KeyError):
        return
    for file in (state / "manifests").glob("*.json"):
        try:
            batch = store.load_batch(state, file.stem)
            if batch.get("recipient_id") != "brian" or not config["recipients"]["brian"].get("enabled"):
                continue
            for record in batch["chapters"]:
                path = Path(record["audio_path"]).resolve(strict=True)
                if not any(path.is_relative_to(p) for p in roots) or path.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
                    continue
                pid = batch["project_id"]
                ident = _identity("delivery", pid, record["chapter_id"], record["audio_sha256"])
                url = f"/api/v1/voice-library/recordings/{pid}/{ident}/audio"
                yield {"project_id": pid, "project_name": batch["book_title"], "export_id": ident,
                       "label": record["title"], "duration_seconds": record["duration_seconds"],
                       "audio_sha256": record["audio_sha256"], "technical_qa": "passed", "human_review": "pending",
                       "kind": "delivery", "created_at": batch["created_at"], "audio_url": url,
                       "download_url": url + "?download=true",
                       "review_url": config["listening_base_url"] + "/review/" + batch["batch_id"]}, path
        except (OSError, ValueError, KeyError):
            log.warning("Skipping unavailable published review batch %s", file.stem)


def list_recordings():
    seen, rows = set(), []
    for record, _ in _records():
        key = (record["project_id"], record["export_id"])
        if key not in seen:
            seen.add(key)
            rows.append(record)
    return {"recordings": sorted(rows, key=lambda row: row["created_at"], reverse=True)}


def audio_artifact(project_id, export_id):
    for record, path in _records():
        if record["project_id"] == project_id and record["export_id"] == export_id:
            return path, record["audio_sha256"]
    raise ValueError("This published recording is unavailable; refresh the library")
