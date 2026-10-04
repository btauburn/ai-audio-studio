"""Explicit AutoCast import for locally uploaded EPUB and UTF-8 text books.

This boundary stores an immutable source and a recoverable request receipt before
running the configured Audiobooker adapter. It never renders audio or invents
speaker identities.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from dubbing_service.app.audiobook import adapter
from dubbing_service.app.audiobook.provenance import register_source_revision
from dubbing_service.app.core.config import get_config
from dubbing_service.app.core.helpers import project_root as project_root_path
from dubbing_service.app.storage.projects import (
    create_project,
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.segments import load_segments, save_segments

from . import book_creator


class AutoCastImportError(ValueError):
    """The requested AutoCast import could not be safely accepted."""


def _upload_name(filename: str) -> tuple[str, str]:
    safe_name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(safe_name).suffix.lower()
    if (not safe_name or len(safe_name) > 240 or "\x00" in safe_name
            or suffix not in {".epub", ".txt", ".pdf"}):
        raise AutoCastImportError("Choose a local PDF, EPUB, or UTF-8 TXT file")
    return safe_name, suffix


def _copy_exact_source(project_id: str, digest: str, suffix: str, data: bytes) -> str:
    """Preserve adapter-readable source bytes at a fixed, project-local path."""
    relative = f"audiobook/source/{digest}{suffix}"
    target = project_root_path(project_id, ensure_dir=True) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if target.read_bytes() != data:
            raise AutoCastImportError("The retained source file does not match its content hash")
    return relative


def _find_receipt(request_id: str):
    for path in get_config().projects_root_abs.glob("*/project.json"):
        project = load_project(path.parent.name)
        receipt = (project.metadata or {}).get("autocast_import") or {}
        if receipt.get("request_id") == request_id:
            return project, receipt
    return None, None


def _response(project_id: str, receipt: dict, *, reused: bool) -> dict:
    return {
        "project_id": project_id,
        "mode": "autocast",
        "status": receipt["status"],
        "reason": receipt.get("reason"),
        "analysis_source": receipt.get("analysis_source"),
        "review_url": f"/web/voice-lab.html?project={project_id}",
        "autocast_run_key": receipt.get("autocast_run_key"),
        "autocast_status": receipt.get("autocast_status"),
        "reused": reused,
    }


def _role_sentinel(speaker: str | None) -> bool:
    return not speaker or speaker.strip().casefold() in {
        "narrator", "narration", "unknown", "unattributed", "none", "null",
        "speaker", "speaker 1", "speaker_1", "voice 1", "voice_1",
    }


def _unresolved_ids(project_id: str, analysis) -> tuple[list[str], list[str], list[str]]:
    """Bind analysis positions back to adapter-created canonical segment IDs."""
    project = load_project(project_id)
    chapters = {chapter.index: chapter for chapter in project.chapters}
    segments = load_segments(project_id)
    by_anchor = {}
    for segment in segments:
        anchor = segment.source_anchor
        if anchor is not None:
            by_anchor[(anchor.chapter_id or segment.chapter_id, anchor.occurrence_index)] = segment
    unresolved: list[str] = []
    speaking_ids: list[str] = []
    role_ids: list[str] = []
    roles = {role.name.strip().casefold(): role for role in project.roles}
    for chapter in analysis.chapters:
        studio_chapter = chapters.get(chapter.index)
        if studio_chapter is None:
            continue
        for utterance in chapter.utterances:
            if not utterance.text.strip() or str(utterance.utterance_type).casefold() != "dialogue":
                continue
            segment = by_anchor.get((studio_chapter.id, utterance.line_index))
            if segment is None:
                continue
            speaker = utterance.speaker.strip() if utterance.speaker else None
            source = (utterance.attribution_source or "unknown").casefold()
            confidence = utterance.confidence
            unresolved_here = (
                _role_sentinel(speaker)
                or confidence is None
                or confidence < 0.7
                or (source in {"turn", "unknown"} and source != "user")
            )
            if unresolved_here:
                unresolved.append(segment.id)
                # The adapter routes unattributed speech through its narrator
                # fallback for legacy renderability. AutoCast must not preserve
                # that fallback as a character attribution. The analyzer's own
                # proposal stays on the record as evidence so a reviewer can see
                # what was suggested rather than a blank.
                segment.role_id = None
                if segment.attribution is not None:
                    segment.attribution.speaker_id = None
                    segment.attribution.label = speaker
                continue
            speaking_ids.append(segment.id)
            role = roles.get((speaker or "").casefold())
            if role and role.id not in role_ids:
                role_ids.append(role.id)
    # Persist cleared fallback assignments. The review workflow will decide
    # how held utterances are resolved; import itself never picks a narrator.
    if unresolved:
        from dubbing_service.app.storage.segments import save_segments
        save_segments(project_id, segments)
    return sorted(set(unresolved)), speaking_ids, role_ids


def _import_autocast(data: bytes, filename: str, request_id: str, name: str | None = None) -> dict:
    try:
        book_creator._request_id(request_id)
    except book_creator.BookCreatorError as exc:
        raise AutoCastImportError(str(exc)) from exc
    safe_name, suffix = _upload_name(filename)
    if not isinstance(data, bytes) or not data or len(data) > book_creator.MAX_UPLOAD_BYTES:
        raise AutoCastImportError("Book upload must contain 1 byte–32 MiB")
    normalized_name = name.strip() if name is not None else None
    if name is not None and (not normalized_name or len(normalized_name) > 200):
        raise AutoCastImportError("Book name must contain 1–200 characters")
    try:
        parsed = (book_creator._parse_epub(data, safe_name) if suffix == ".epub"
                  else book_creator._parse_pdf(data, safe_name) if suffix == ".pdf"
                  else book_creator._parse_text(data, safe_name))
    except book_creator.BookCreatorError as exc:
        raise AutoCastImportError(str(exc)) from exc
    documents = parsed["chapters"]
    if (not documents or len(documents) > book_creator.MAX_CHAPTERS
            or sum(len(chapter["text"]) for chapter in documents) > book_creator.MAX_SOURCE_CHARACTERS):
        raise AutoCastImportError("Book must contain readable text within 200 chapters and two million characters")

    availability = adapter.availability(refresh=True)
    if not availability.available or availability.version != adapter.PINNED_AUDIOBOOKER_VERSION or not availability.interpreter:
        reason = availability.reason or f"AutoCast requires Audiobooker {adapter.PINNED_AUDIOBOOKER_VERSION}; found {availability.version or 'unavailable'}"
        raise AutoCastImportError(reason)

    digest = hashlib.sha256(data).hexdigest()
    intent_hash = hashlib.sha256(book_creator._json({
        "sha256": digest, "format": suffix[1:], "name": normalized_name,
    })).hexdigest()
    title = normalized_name or parsed["title"]
    media_type = {".epub": "application/epub+zip", ".pdf": "application/pdf"}.get(suffix, "text/plain")

    with project_meta_lock("autocast_import_requests"):
        project, receipt = _find_receipt(request_id)
        reused = project is not None
        if project is not None:
            if receipt.get("intent_hash") != intent_hash:
                raise AutoCastImportError("This import request ID already belongs to a different file or name")
            if receipt.get("status") in {"ready", "blocked"}:
                return _response(project.id, receipt, reused=True)
        else:
            project = create_project(title, description="Local AutoCast audiobook; speaker assignments await review")
            raw_revision, _ = register_source_revision(
                project.id, project, uri=f"local-upload:{digest}", data=data,
                media_type=media_type, title=title,
                notes="Exact local upload bytes; never fetched or sent to a cloud analyzer",
            )
            source_file = _copy_exact_source(project.id, digest, suffix, data)
            project.character_cast.policy.audition_mode = "off"
            receipt = {
                "schema": "autocast-import-v1", "request_id": request_id,
                "intent_hash": intent_hash, "mode": "autocast", "status": "analyzing",
                "reason": None, "analysis_source": "audiobooker", "unresolved_utterance_ids": [],
                "raw_source_revision_id": raw_revision.id,
                "source_file": source_file, "source_sha256": digest,
                "source_format": suffix[1:], "source_filename": safe_name,
                "speaking_utterance_ids": [], "speaking_role_ids": [],
                # A new book starts with automatic handling on: lines that would stop it are decided
                # (real suggested speakers accepted, the rest skipped) rather than waiting for a
                # click. It is a visible per-project setting, and turning it off is one tap.
                "attribution_policy": {"enabled": True, "set_by": "default", "at": datetime.now(timezone.utc).isoformat()},
            }
            project.metadata["autocast_import"] = receipt
            save_project(project)

        # A receipt left in analyzing state is a recoverable crash marker.
        # Reuse its already verified source and source revision on retry.
        if reused:
            source_file = receipt.get("source_file")
            if not source_file:
                raise AutoCastImportError("The retained AutoCast source path is missing")
            raw_revision = next((row for row in project.source_revisions if row.id == receipt.get("raw_source_revision_id")), None)
            if raw_revision is None or raw_revision.sha256 != digest:
                raise AutoCastImportError("The retained AutoCast source revision is missing or does not match")
            path = project_root_path(project.id) / source_file
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AutoCastImportError("The retained AutoCast source file is missing or changed")
        else:
            path = project_root_path(project.id) / source_file

    try:
        analyzer = adapter.AudiobookerAdapter(interpreter=Path(availability.interpreter))
        analysis = analyzer.analyze_source(path, title=title, author=parsed["author"], language=parsed["language"], keep_scratch=True)
        analyzer.reconcile_into_project(analysis, project_id=project.id, source_revision_id=raw_revision.id)
        unresolved_ids, speaking_ids, role_ids = _unresolved_ids(project.id, analysis)
        project = load_project(project.id)
        receipt = dict((project.metadata or {}).get("autocast_import") or receipt)
        receipt.update(
            status="blocked" if unresolved_ids else "ready",
            reason=(f"{len(unresolved_ids)} dialogue utterance(s) need attribution review" if unresolved_ids else None),
            analysis_source=analysis.schema,
            unresolved_utterance_ids=unresolved_ids,
            speaking_utterance_ids=speaking_ids,
            speaking_role_ids=role_ids,
        )
        project.metadata["autocast_import"] = receipt
        # Cleanup: web addresses are never read aloud. A link line is skipped before anything
        # waits on it, and an address inside a sentence is taken out of what is spoken.
        from . import casting_review as review
        segments = load_segments(project.id)
        _, rewritten = review.clean_web_addresses(project, segments)
        if receipt["status"] == "blocked" and not receipt.get("unresolved_utterance_ids"):
            receipt.update(status="ready", reason=None)  # the only lines waiting were links
        project.character_cast.policy.audition_mode = "off"
        save_project(project)
        if rewritten:
            save_segments(project.id, segments)
        # The trigger point. Speaker attribution and canonical character
        # resolution are now committed, so this durable state — not a page
        # view, a refresh or a timer — is what starts Auto Cast.  Recording the
        # intent costs no GPU work; the existing maintenance owner starts the
        # rounds, and the run key makes a duplicate completion event a no-op.
        from . import casting_review
        with project_meta_lock(project.id):
            current = load_project(project.id)
            current.metadata.setdefault("autocast_model_research", {"requested": True})
            save_project(current)
        reason = None if receipt["status"] == "ready" else receipt.get("reason")
        run, _created = casting_review.request_autocast_run(
            project.id, trigger="source_import", reason=reason)
        with project_meta_lock(project.id):
            current = load_project(project.id)
            receipt = dict((current.metadata or {}).get("autocast_import") or receipt)
            receipt["autocast_run_key"] = run["key"]
            receipt["autocast_status"] = run["status"]
            current.metadata["autocast_import"] = receipt
            save_project(current)
    except Exception:
        # Keep status=analyzing so an identical request can retry this same
        # project. No scratch directory is removed by this module.
        raise
    return _response(project.id, receipt, reused=reused)


def import_autocast(data: bytes, filename: str, request_id: str, name: str | None = None) -> dict:
    """Import through the exact runtime; serialize retries for the same ID."""
    # The storage lock registry is keyed by arbitrary names, so this per-request
    # mutex coordinates duplicate uploads while leaving other imports independent.
    with project_meta_lock(f"autocast-import-request:{request_id}"):
        return _import_autocast(data, filename, request_id, name)
