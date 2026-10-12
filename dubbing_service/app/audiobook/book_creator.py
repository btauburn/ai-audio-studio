"""Local book-to-MP3 workflow using Studio source, library, Take and job owners.

The simple mode deliberately reads all prose/dialogue with one chosen narrator.
Parsing is deterministic and offline; no source text leaves this application.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import posixpath
import re
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, quote
from uuid import uuid4
import xml.etree.ElementTree as ET

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Role
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.storage.projects import create_project, load_project, save_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments, project_segments_lock
from dubbing_service.app.storage.jobs import (get_job, load_jobs, upsert_job, mutate_job, start_job,
    try_mark_job_active, mark_job_active, is_job_active)
from . import adapter, pipeline, qa, release, takes, voice_lab, voice_lab_library
from .casting_assembly import current_frozen_binding
from .fingerprint import file_sha256, text_sha256, voice_revision_for_role
from .listening_excerpt import extract_epub_chapter_text
from .persistence import commit_files
from .provenance import register_source_revision
from .renderer import BackendRenderer
from .voice_lab_render import _immutable, _json, _assemble_pcm, _mp3, _decoded_duration, _loudness, _identity_for_take
from .voice_lab_chapter_export import _pause

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_CHAPTERS = 200
MAX_SOURCE_CHARACTERS = 2_000_000
MAX_BOOK_UTTERANCES = 20000
MAX_RENDER_UTTERANCES = 2000
MAX_RENDER_CHARACTERS = 250000
MAX_REQUESTS = 100
REQUEST_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


class BookCreatorError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _commit_locked(project, segments):
    """Caller already owns both canonical locks; do not re-enter their mutexes."""
    root = safe_project_root(project.id)
    return commit_files([(root / "project.json", _json(project.model_dump(mode="json"))),
                         (root / "segments.json", _json([s.model_dump(mode="json") for s in segments]))], root=root)


def _request_id(value):
    if not isinstance(value, str) or not REQUEST_RE.fullmatch(value):
        raise BookCreatorError("Use a request ID containing 1–100 letters, numbers, underscores or hyphens")
    return value


def _safe(root, relative):
    return voice_lab_library._safe_path(root, relative)


def _xml(data):
    if b"<!ENTITY" in data.upper():
        raise BookCreatorError("XML entity declarations are not accepted")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise BookCreatorError("The EPUB contains invalid XML") from exc


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _href(base, value):
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise BookCreatorError("EPUB reading-order content must be a local archive member")
    value = unquote(parsed.path)
    if not value or "\\" in value or ":" in value or value.startswith("/"):
        raise BookCreatorError("Invalid EPUB member reference")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(base), value))
    if path == ".." or path.startswith("../"):
        raise BookCreatorError("EPUB member reference escapes its archive")
    return path


def _parse_epub(data, filename):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise BookCreatorError("This file is not a valid EPUB archive") from exc
    with archive:
        members = archive.infolist()
        if len(members) > 4096 or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
            raise BookCreatorError("EPUB exceeds the bounded archive size limit")
        names = set()
        for item in members:
            parts = item.filename.replace("\\", "/").split("/")
            if (item.filename in names or item.filename.startswith(("/", "\\")) or ":" in item.filename
                    or ".." in parts or item.file_size > MAX_MEMBER_BYTES or item.flag_bits & 1
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise BookCreatorError("EPUB contains duplicate, encrypted, linked, oversized or unsafe members")
            names.add(item.filename)
        if "META-INF/encryption.xml" in names:
            raise BookCreatorError("Encrypted EPUB resources are unsupported; use an eligible unencrypted local source")
        if "META-INF/container.xml" not in names:
            raise BookCreatorError("EPUB has no package container")
        container = _xml(archive.read("META-INF/container.xml"))
        packages = [e.attrib.get("full-path", "") for e in container.iter() if _tag(e) == "rootfile"]
        if not packages or packages[0] not in names:
            raise BookCreatorError("EPUB package document is missing")
        package_path = packages[0]
        package = _xml(archive.read(package_path))
        metadata = {}
        for element in package.iter():
            if _tag(element) in {"title", "creator", "language"} and element.text:
                metadata.setdefault(_tag(element), element.text.strip())
        manifest = {e.attrib.get("id"): e.attrib for e in package.iter() if _tag(e) == "item"}
        spine = [e.attrib.get("idref") for e in package.iter() if _tag(e) == "itemref" and e.attrib.get("linear", "yes") != "no"]
        if not spine or len(spine) > MAX_CHAPTERS:
            raise BookCreatorError("EPUB must contain a reading order of 1–200 documents")
        labels = {}
        for item in manifest.values():
            if "nav" in item.get("properties", "").split() or item.get("media-type") == "application/x-dtbncx+xml":
                member = _href(package_path, item.get("href", ""))
                if member not in names:
                    continue
                toc = _xml(archive.read(member))
                for node in toc.iter():
                    if _tag(node) == "a" and node.attrib.get("href"):
                        try:
                            target = _href(member, node.attrib["href"])
                        except BookCreatorError:
                            continue  # External links are metadata only, never fetched.
                        label = " ".join("".join(node.itertext()).split())
                        if label:
                            labels.setdefault(target, label)
                    elif _tag(node) == "navPoint":
                        content = next((e for e in node if _tag(e) == "content"), None)
                        label = next((e for e in node if _tag(e) == "navLabel"), None)
                        if content is not None and label is not None:
                            labels.setdefault(_href(member, content.attrib.get("src", "")), " ".join("".join(label.itertext()).split()))
        chapters, seen = [], set()
        for ref in spine:
            item = manifest.get(ref)
            if not item:
                raise BookCreatorError("EPUB reading order refers to a missing document")
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                raise BookCreatorError("EPUB reading order includes unsupported non-text content")
            member = _href(package_path, item.get("href", ""))
            if member in seen or member not in names:
                raise BookCreatorError("EPUB reading order contains repeated or missing documents")
            seen.add(member)
            body = _xml(archive.read(member))
            text = extract_epub_chapter_text(data, member=member)
            if not text.strip():
                continue
            heading = next((" ".join("".join(e.itertext()).split()) for e in body.iter() if _tag(e) in {"h1", "h2", "h3"} and "".join(e.itertext()).strip()), None)
            chapters.append({"title": labels.get(member) or heading or f"Section {len(chapters) + 1}", "member": member,
                             "text": text, "extraction": "epub-xhtml-body-text-v1"})
        return {"title": metadata.get("title") or Path(filename).stem, "author": metadata.get("creator"),
                "language": metadata.get("language"), "chapters": chapters}


def _parse_text(data, filename):
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BookCreatorError("Text books must use UTF-8 encoding") from exc
    if "\x00" in text:
        raise BookCreatorError("Text book contains binary data")
    return _chapters_from_text(text, filename, extraction="utf8-text-chapter-v1")


def _chapters_from_text(text, filename, *, extraction):
    boundaries = [m.start() for m in re.finditer(r"(?m)^.*$", text) if adapter._looks_like_chapter(m.group())]
    if not boundaries or boundaries[0] > 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))
    chapters = []
    for start, end in zip(boundaries, boundaries[1:]):
        body = text[start:end]
        if body.strip():
            first = body.strip().splitlines()[0].strip()
            title = first if adapter._looks_like_chapter(first) else Path(filename).stem if not chapters else f"Section {len(chapters) + 1}"
            chapters.append({"title": title, "text": body, "member": None, "text_file_start_offset": start,
                             "extraction": extraction})
    return {"title": Path(filename).stem, "author": None, "language": None, "chapters": chapters}


def _parse_pdf(data, filename):
    """Extract readable text from a local PDF; same chapter-boundary rules as TXT.

    PDF layout carries no explicit chapter markup, so the exact heuristic that
    already splits plain-text books is reused rather than duplicated.
    """
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise BookCreatorError("PDF import requires the pypdf package to be installed") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise BookCreatorError("This file could not be read as a PDF") from exc
    if reader.is_encrypted:
        raise BookCreatorError("Encrypted or password-protected PDF files are not supported")
    if not reader.pages or len(reader.pages) > 4000:
        raise BookCreatorError("PDF must contain 1-4000 pages")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pypdf raises assorted errors on malformed page streams
            raise BookCreatorError("A page in this PDF could not be read as text") from exc
    text = "\n\n".join(pages)
    if not text.strip():
        raise BookCreatorError("No extractable text was found in this PDF. Scanned/image-only PDFs are not supported")
    return _chapters_from_text(text, filename, extraction="pdf-text-chapter-v1")


def _utterances(text, chapter_index):
    result = []
    # Complete nonblank source lines, split only at natural sentence whitespace.
    # Speaker-looking prefixes and quotation marks remain spoken source text.
    for line in re.finditer(r"[^\r\n]+", text):
        body = line.group().strip()
        leading = len(line.group()) - len(line.group().lstrip())
        cursor, spans = 0, []
        for separator in re.finditer(r"(?<=[.!?])\s+(?=[A-Z“\"])", body):
            spans.append((cursor, separator.start()))
            cursor = separator.end()
        spans.append((cursor, len(body)))
        for begin, finish in spans:
            start, end = line.start() + leading + begin, line.start() + leading + finish
            if not text[start:end].strip():
                continue
            result.append(adapter.AnalyzedUtterance(text=text[start:end], speaker="Narrator", utterance_type="narration",
                chapter_index=chapter_index, line_index=len(result), start_pos=start, end_pos=end,
                attribution_source="inline", confidence=1.0))
    # Verify no story characters disappeared through segmentation.
    cursor = 0
    for item in result:
        if text[cursor:item.start_pos].strip():
            raise BookCreatorError("Offline sentence segmentation could not preserve complete source text")
        cursor = item.end_pos
    if text[cursor:].strip():
        raise BookCreatorError("Offline sentence segmentation omitted source text")
    return result


def _scene(text):
    return bool(re.fullmatch(r"(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}", text.strip()))


def _rows(segments):
    return [{"utterance_id": s.id, "chapter_id": s.chapter_id, "index": s.index,
             "source_text_sha256": text_sha256(s.text), "source_anchor": s.source_anchor.model_dump(mode="json") if s.source_anchor else None,
             "spoken_text_revision_id": s.spoken_text_revision_id} for s in segments]


def _owned(project_id):
    project = load_project(project_id)
    meta = project.metadata.get("book_creator")
    if not meta or meta.get("schema") != "studio-book-creator-v1":
        raise BookCreatorError("This operation applies only to books imported through Create an audiobook")
    return project, meta


def import_book(data: bytes, filename: str, request_id: str, name: str | None = None) -> dict:
    _request_id(request_id)
    filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or len(filename) > 240 or "\x00" in filename or Path(filename).suffix.lower() not in {".epub", ".txt", ".pdf"}:
        raise BookCreatorError("Choose a local PDF, EPUB, or UTF-8 TXT file")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise BookCreatorError("Book upload must contain 1 byte–32 MiB")
    if name is not None and (not name.strip() or len(name.strip()) > 200):
        raise BookCreatorError("Book name must contain 1–200 characters")
    intent = _hash({"sha256": hashlib.sha256(data).hexdigest(), "filename": filename, "name": name})
    with project_meta_lock("book_creator_import_requests"):
        for path in get_config().projects_root_abs.glob("*/project.json"):
            previous = load_project(path.parent.name)
            meta = previous.metadata.get("book_creator", {})
            if meta.get("import_request_id") == request_id:
                if meta.get("import_intent") != intent:
                    raise BookCreatorError("This import request ID already belongs to a different file or name")
                if meta.get("import_status") != "ready":
                    raise BookCreatorError("An incomplete import is retained; use a new import request to retry")
                return {**view(previous.id), "reused": True}
        lowered = filename.lower()
        parsed = (_parse_epub(data, filename) if lowered.endswith(".epub")
                  else _parse_pdf(data, filename) if lowered.endswith(".pdf")
                  else _parse_text(data, filename))
        documents = parsed["chapters"]
        if not documents or len(documents) > MAX_CHAPTERS or sum(len(c["text"]) for c in documents) > MAX_SOURCE_CHARACTERS:
            raise BookCreatorError("Book must contain readable text within 200 chapters and two million characters")
        analysis = adapter.BookAnalysis(title=name.strip() if name else parsed["title"], author=parsed["author"],
            language=parsed["language"], speakers=["Narrator"], warnings=["offline_splitter; explicit single-narrator reading; no character attribution"])
        for index, document in enumerate(documents):
            utterances = _utterances(document["text"], index)
            if not utterances or not any(not _scene(u.text) for u in utterances):
                raise BookCreatorError("Every imported chapter needs readable spoken text")
            analysis.chapters.append(adapter.AnalyzedChapter(index=index, title=document["title"], utterances=utterances))
        if analysis.utterance_count() > MAX_BOOK_UTTERANCES:
            raise BookCreatorError("Book exceeds the 20,000 source utterance import limit")
        project = create_project(analysis.title, description="Local single-narrator audiobook")
        project.metadata["book_creator"] = {"schema": "studio-book-creator-v1", "import_request_id": request_id,
            "import_intent": intent, "import_status": "preparing", "mode": "single_narrator",
            "source": {"filename": filename, "format": Path(filename).suffix.lower()[1:], "sha256": hashlib.sha256(data).hexdigest()},
            "requests": {}, "exports": {}, "voice_selection": None}
        raw_revision, _ = register_source_revision(project.id, project, uri="local-upload:" + filename, data=data,
            title=analysis.title, media_type="application/epub+zip" if filename.lower().endswith(".epub") else "text/plain",
            notes="Exact local upload bytes; never fetched or sent to a cloud analyzer")
        save_project(project)
        adapter.AudiobookerAdapter().reconcile_into_project(analysis, project_id=project.id, source_revision_id=raw_revision.id)
        with project_meta_lock(project.id), project_segments_lock(project.id):
            project = load_project(project.id)
            segments = load_segments(project.id)
            narrator = next(r for r in project.roles if r.name == "Narrator")
            narrator.role_type, narrator.tts_backend = RoleType.NARRATOR, None
            chapter_proof = {}
            for chapter, document in zip(sorted(project.chapters, key=lambda c: c.index), documents):
                revision, _ = register_source_revision(project.id, project,
                    uri=f"local-extraction:{raw_revision.id}:{chapter.id}", data=document["text"], kind="inline",
                    media_type="text/plain", title=chapter.title, notes=f"Complete deterministic {document['extraction']} from raw source {raw_revision.sha256}")
                chapter_proof[chapter.id] = {"source_revision_id": revision.id, "source_sha256": revision.sha256,
                    "extraction": document["extraction"], "source_member": document["member"],
                    "text_file_start_offset": document.get("text_file_start_offset")}
                for segment in (s for s in segments if s.chapter_id == chapter.id):
                    segment.source_anchor.source_revision_id = revision.id
                    segment.role_id = narrator.id
                    segment.attribution = Attribution(speaker_id=narrator.id, source="inline", confidence=1,
                        reviewed=False, note="Explicit single-narrator reading; this is a narration route, not inferred character identity")
            project.character_cast.policy.enabled = False
            project.character_cast.policy.audition_mode = "off"
            project.character_cast.policy.max_audition_characters = 0
            project.voice_lab.scope = "production"
            ordered = sorted(segments, key=lambda s: (next(c.index for c in project.chapters if c.id == s.chapter_id), s.index, s.id))
            project.metadata["book_creator"].update(import_status="ready", narrator_role_id=narrator.id,
                raw_source_revision_id=raw_revision.id, chapters=chapter_proof, source_rows=_rows(ordered),
                source_revision_hashes={r.id: r.sha256 for r in project.source_revisions},
                scene_breaks=[s.id for s in ordered if _scene(s.text)])
            _commit_locked(project, ordered)
        return {**view(project.id), "reused": False}


def _source(project_id, verify_files=True):
    project, meta = _owned(project_id)
    order = {c.id: c.index for c in project.chapters}
    segments = sorted((s for s in load_segments(project_id) if release.is_live(s)), key=lambda s: (order.get(s.chapter_id, -1), s.index, s.id))
    if meta.get("import_status") != "ready" or _rows(segments) != meta.get("source_rows"):
        raise BookCreatorError("Imported source changed; create a new recording from the revised book")
    if {r.id: r.sha256 for r in project.source_revisions} != meta.get("source_revision_hashes"):
        raise BookCreatorError("Source revision history changed")
    if set(order) != set(meta["chapters"]) or set(meta.get("scene_breaks", [])) != {s.id for s in segments if _scene(s.text)}:
        raise BookCreatorError("Book chapter or scene-break coverage changed")
    if verify_files:
        root = safe_project_root(project_id)
        for revision in project.source_revisions:
            if not revision.snapshot_file or file_sha256(_safe(root, revision.snapshot_file)) != revision.sha256:
                raise BookCreatorError("Local source snapshot is missing or changed")
        revisions = {r.id: r for r in project.source_revisions}
        for chapter_id, proof in meta["chapters"].items():
            text = _safe(root, revisions[proof["source_revision_id"]].snapshot_file).read_bytes().decode("utf-8")
            for segment in (s for s in segments if s.chapter_id == chapter_id):
                anchor = segment.source_anchor
                if not anchor or text[anchor.start_offset:anchor.end_offset] != segment.text:
                    raise BookCreatorError("Exact chapter text offsets no longer match canonical utterances")
    return project, meta, segments


def _public_record(project_id, record):
    return {key: record.get(key) for key in ["export_id", "label", "chapter_ids", "duration_seconds", "audio_sha256",
            "technical_qa", "human_review", "library_voice_id", "voice_version"]} | {
            "audio_url": f"/api/v1/book-creator/{project_id}/audio/{record['export_id']}",
            "download_url": f"/api/v1/book-creator/{project_id}/audio/{record['export_id']}?download=true"}


def _export_groups(chapter_ids):
    return [[cid] for cid in chapter_ids] + ([chapter_ids] if len(chapter_ids) > 1 else [])


def _exports_complete(meta, receipt):
    records = [meta["exports"].get(key) for key in receipt.get("export_ids", [])]
    return (len(records) == len(_export_groups(receipt["chapter_ids"]))
        and all(row and row.get("technical_qa") == "passed" and row.get("audio_sha256")
                and row.get("master_sha256") and row.get("library_voice_id") == receipt["library_voice_id"]
                and row.get("voice_version") == receipt["voice_version"] for row in records)
        and sorted(tuple(row["chapter_ids"]) for row in records) == sorted(tuple(group) for group in _export_groups(receipt["chapter_ids"])))


def _chosen_voice(project, meta):
    """Check the actual assignment against the imported, exact library version."""
    selection = meta.get("voice_selection")
    records = [r for r in project.voice_lab.voices.values()
               if selection and r.get("library_voice_id") == selection["library_voice_id"]
               and r.get("library_version") == selection["voice_version"]]
    role = next((r for r in project.roles if r.id == meta["narrator_role_id"]), None)
    if len(records) != 1 or role is None:
        raise BookCreatorError("The chosen library narrator is no longer available in this recording")
    record = records[0]
    identity = CharacterVoiceIdentity.model_validate(record["identity"]) if record.get("identity") else None
    if identity and identity.mechanism != "builtin":
        legacy = identity.model_copy(update={"role_id": role.id})
        assigned = voice_lab.identity_for_assignment(identity).model_copy(update={"role_id": role.id})
        matches = role.voice_identity in (legacy, assigned) and role.tts_backend == identity.backend
        if matches and identity.mechanism == "qwen_reference":
            # Accept old projects whose paths still match the immutable record,
            # or the exact identity-scoped relocation used for new assignments.
            # In either case verify actual bytes before rendering.
            from .voice_lab_library import _read_verified, _safe_path
            root = safe_project_root(project.id)
            try:
                for field, hash_field in (("reference_audio", "reference_audio_sha256"),
                                          ("prompt_artifact", "prompt_sha256")):
                    _read_verified(_safe_path(root, getattr(role.voice_identity, field)),
                                   getattr(role.voice_identity, hash_field))
            except Exception as exc:
                raise BookCreatorError(f"The assigned narrator conditioning is missing or changed: {exc}") from exc
    else:
        snapshot = record.get("builtin_snapshot") or {}
        settings = snapshot.get("settings", {})
        if snapshot.get("schema") != "voice-lab-builtin-role-v1" or not settings:
            raise BookCreatorError("Saved narrator settings are incomplete")
        expected = Role(id=role.id, name=role.name, voice_identity=identity.model_copy(update={"role_id": role.id}) if identity else None,
                        **settings)
        matches = (role.voice_identity == expected.voice_identity
                   and all(getattr(role, field) == getattr(expected, field) for field in settings)
                   and voice_revision_for_role(role) == voice_revision_for_role(expected))
    if not matches:
        raise BookCreatorError("The narrator assignment changed outside this recording; restore the chosen library voice or import a new recording")
    return role


def request_status(project_id, request_id):
    from . import book_creator_recovery as recovery
    project, meta = _owned(project_id)
    receipt = meta["requests"].get(_request_id(request_id))
    if not receipt:
        raise BookCreatorError("This book has no matching render request")
    job = get_job(project_id, receipt["job_id"])
    status = job.status.value if job else "interrupted"
    message = job.message if job else "Resume this request to continue"
    if status == "running" and not is_job_active(project_id, receipt["job_id"]):
        status = "interrupted"
    if status == "succeeded" and not _exports_complete(meta, receipt):
        status = "interrupted"
        message = "Speech is retained; resume this request to finish its MP3 exports"
    return {"project_id": project_id, "request_id": request_id, "job_id": receipt["job_id"], "status": status,
            "completed": job.done if job else 0, "total": len(job.segment_ids) if job else len(receipt["segment_ids"]),
            "message": message,
            "automation": recovery.public_automation(receipt),
            "notify_on_completion": receipt.get("notify_on_completion") is True,
            "notification": recovery.public_notification(receipt),
            "chapter_ids": receipt["chapter_ids"], "library_voice_id": receipt["library_voice_id"], "voice_version": receipt["voice_version"],
            "outputs": [_public_record(project_id, meta["exports"][key]) for key in receipt.get("export_ids", []) if key in meta["exports"]]}


def view(project_id):
    project, meta = _owned(project_id)
    segments = [s for s in load_segments(project_id) if release.is_live(s)]
    requests = [request_status(project_id, key) for key in meta["requests"]]
    return {"project_id": project.id, "name": project.name, "mode": "single_narrator",
            "status": "rendering" if any(r["status"] in {"running", "queued"} for r in requests) else "ready" if meta["exports"] else "imported",
            "source": meta["source"], "voice_selection": meta.get("voice_selection"),
            "chapters": [{"id": c.id, "title": c.title, "index": c.index,
                "utterance_count": sum(s.chapter_id == c.id and s.id not in meta["scene_breaks"] for s in segments),
                "selected_audio_count": sum(s.chapter_id == c.id and bool(s.selected_take_id) for s in segments)} for c in sorted(project.chapters, key=lambda c: c.index)],
            "recordings": [_public_record(project_id, r) for r in meta["exports"].values()], "requests": requests,
            "limits": {"max_render_utterances": MAX_RENDER_UTTERANCES, "max_chapters": MAX_CHAPTERS,
                       "max_render_characters": MAX_RENDER_CHARACTERS, "max_upload_bytes": MAX_UPLOAD_BYTES}}


def list_books():
    books = []
    for path in get_config().projects_root_abs.glob("*/project.json"):
        project = load_project(path.parent.name)
        if project.metadata.get("book_creator", {}).get("import_status") == "ready":
            books.append(view(project.id))
    voices = voice_lab_library.list_versions()
    for voice in voices:
        for version in voice["versions"]:
            version["audio_url"] = f"/api/v1/voice-library/versions/{quote(voice['library_voice_id'], safe='')}/{quote(version['version'], safe='')}/audio"
    return {"books": books, "voices": voices}


def _binding(project, segments):
    return {s.id: pipeline.requirement_for(project, s, renderer_id=(s.tts_backend or next(r.tts_backend for r in project.roles if r.id == s.role_id))).fingerprint()
            for s in segments}


def _pending_production(project_id, job_id):
    from .book_creator_recovery import recoverable_idle_book_job
    for path in get_config().projects_root_abs.glob("*/project.json"):
        for job in load_jobs(path.parent.name):
            if ((job.project_id, job.id) != (project_id, job_id) and job.job_type in {"batch_synthesize", "single_synthesize"}
                    and job.status in {SynthesisJobStatus.QUEUED, SynthesisJobStatus.RUNNING}
                    and not recoverable_idle_book_job(path.parent.name, job)):
                return True
    return False


def render_book(project_id, *, chapter_ids, library_voice_id, voice_version, request_id, schedule=None, renderer=None,
                automatic=False, recovery_now=None):
    from . import book_creator_recovery as recovery
    _request_id(request_id)
    with project_meta_lock("book_creator_dispatch_" + project_id):
        project, meta, all_segments = _source(project_id)
        if not chapter_ids or len(chapter_ids) != len(set(chapter_ids)) or not set(chapter_ids).issubset({c.id for c in project.chapters}):
            raise BookCreatorError("Choose one or more distinct chapters from this book")
        chapter_ids = [c.id for c in sorted(project.chapters, key=lambda c: c.index) if c.id in chapter_ids]
        speech = [s for s in all_segments if s.chapter_id in chapter_ids and s.id not in meta["scene_breaks"]]
        if not speech or len(speech) > MAX_RENDER_UTTERANCES or sum(len(s.text) for s in speech) > MAX_RENDER_CHARACTERS:
            raise BookCreatorError("Select fewer chapters: each render supports 2,000 utterances and 250,000 characters")
        requested = {"chapter_ids": chapter_ids, "library_voice_id": library_voice_id, "voice_version": voice_version}
        receipt = meta["requests"].get(request_id)
        if receipt and any(receipt.get(k) != v for k, v in requested.items()):
            raise BookCreatorError("Request ID already belongs to another chapter selection or voice")
        selection = {"library_voice_id": library_voice_id, "voice_version": voice_version}
        if meta.get("voice_selection") and meta["voice_selection"] != selection:
            raise BookCreatorError("This recording already uses another narrator. Import the book as a new recording to choose a different voice")
        if meta.get("voice_selection"):
            _chosen_voice(project, meta)
        if receipt:
            current = request_status(project_id, request_id)
            if current["status"] in {"running", "queued", "succeeded"} and (current["status"] == "succeeded" or is_job_active(project_id, receipt["job_id"])):
                return current
            if receipt.get("inflight_utterance_id"):
                recovery.reconcile_committed_utterance(project_id, request_id)
                project, meta, all_segments = _source(project_id)
                receipt = meta["requests"][request_id]
                if receipt.get("inflight_utterance_id"):
                    raise BookCreatorError("Worker outcome is uncertain; resume is held to avoid duplicate audio. Inspect the worker's terminal outcome before retrying")
        if len(meta["requests"]) >= MAX_REQUESTS and not receipt:
            raise BookCreatorError("This book has reached its bounded render request limit")
        if not meta.get("voice_selection"):
            if any(s.takes for s in all_segments):
                raise BookCreatorError("Existing Takes prevent automatic narrator assignment")
            imported = voice_lab_library.import_version(project_id, library_voice_id, voice_version)
            project = load_project(project_id)
            role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
            voice_lab.assign(project_id, imported["key"], role.id, role.voice_identity.id if role.voice_identity else voice_revision_for_role(role))
            with project_meta_lock(project_id), project_segments_lock(project_id):
                project = load_project(project_id); all_segments = load_segments(project_id)
                role = next(r for r in project.roles if r.id == meta["narrator_role_id"])
                for segment in all_segments:
                    segment.tts_backend = role.tts_backend
                    # No Takes exist at first assignment. Rebuild empty legacy
                    # slot defaults so loading cannot restore their old backend.
                    segment.take_slots = {}
                project.metadata["book_creator"]["voice_selection"] = selection
                _commit_locked(project, all_segments)
        project, meta, all_segments = _source(project_id)
        _chosen_voice(project, meta)
        speech = [s for s in all_segments if s.chapter_id in chapter_ids and s.id not in meta["scene_breaks"]]
        binding = _binding(project, speech)
        intent = _hash({**requested, "source_rows": meta["source_rows"], "requirements": binding})
        job_id = "book_" + _hash([project_id, request_id])[:24]
        previous = get_job(project_id, job_id)
        if previous and previous.intent_hash != intent:
            raise BookCreatorError("Source or narrator changed since this request; its older result cannot replace current audio")
        if automatic:
            if not receipt:
                raise BookCreatorError("Automatic recovery requires an existing opted-in request")
            recovery.retry_guard(project, meta, receipt, all_segments)
            from dubbing_service.app.core.tts_operations import has_active_tts_operation
            if has_active_tts_operation() or _pending_production(project_id, job_id):
                return request_status(project_id, request_id)
            if not recovery.claim_retry(project_id, request_id, recovery_now):
                return request_status(project_id, request_id)
        if not try_mark_job_active(project_id, job_id):
            return request_status(project_id, request_id)
        try:
            with project_meta_lock(project_id):
                project = load_project(project_id)
                project.metadata["book_creator"]["requests"].setdefault(request_id,
                    {**requested, "job_id": job_id, "intent": intent, "segment_ids": [s.id for s in speech], "export_ids": [],
                     "automation": recovery.defaults(), "notify_on_completion": True,
                     "selection_checkpoint": recovery.selections(speech), "inflight_utterance_id": None})
                if not automatic:
                    # Explicit manual resume accepts the currently selected
                    # Takes; historical requests still remain automation-off.
                    current_receipt = project.metadata["book_creator"]["requests"][request_id]
                    current_receipt["selection_checkpoint"] = recovery.selections(speech)
                save_project(project)
            if not previous:
                upsert_job(project_id, SynthesisJob(id=job_id, project_id=project_id, project_name=project.name,
                    job_type="batch_synthesize", request_id=request_id, idempotency_key=job_id, intent_hash=intent,
                    segment_ids=[s.id for s in speech], segment_backends={s.id: s.tts_backend for s in speech}))
            if schedule:
                schedule(_run, project_id, request_id, binding, renderer)
                return request_status(project_id, request_id)
            return _run(project_id, request_id, binding, renderer)
        except BaseException:
            mark_job_active(project_id, job_id, False)
            raise


def _run(project_id, request_id, binding, renderer=None):
    from . import book_creator_recovery as recovery
    project, meta = _owned(project_id)
    receipt = meta["requests"][request_id]; job_id = receipt["job_id"]
    lease = None
    try:
        job = get_job(project_id, job_id)
        if recovery.paused(project_id, request_id) or job.status == SynthesisJobStatus.CANCELLED or job.stop_requested:
            if job.status != SynthesisJobStatus.CANCELLED:
                mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED), setattr(j, "message", "Paused before rendering")))
            return request_status(project_id, request_id)
        if _pending_production(project_id, job_id):
            raise BookCreatorError("Existing production has priority. Resume this request when it finishes")
        role = _chosen_voice(project, meta)
        if renderer is None:
            from .voice_lab_runtime import ensure_backend_ready
            ensure_backend_ready(role.tts_backend, job_id=job_id, project_id=project_id)
        lease = try_register_running_batch(project_id)
        if lease is None:
            raise BookCreatorError("Production currently owns the audio worker. Resume this request when idle")
        renderer = renderer or BackendRenderer(backend=role.tts_backend, engine_id=role.tts_backend, renderer_id=role.tts_backend, project_root=safe_project_root(project_id))
        if hasattr(renderer, "runtime_contract"):
            from .runtime_binding import bind_job
            bind_job(project_id, job_id, renderer, [s for s in load_segments(project_id) if s.id in binding], project.roles, project.character_cast.role_aliases)
        mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED),
            setattr(j, "completed_segment_ids", []), setattr(j, "failed_segments", {}), setattr(j, "message", "Preparing your chosen narrator")))
        start_job(project_id, job_id)
        for sid in receipt["segment_ids"]:
            job = get_job(project_id, job_id)
            if job.stop_requested or job.status == SynthesisJobStatus.CANCELLED or recovery.paused(project_id, request_id):
                if job.status != SynthesisJobStatus.CANCELLED:
                    mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED), setattr(j, "message", "Paused after the current utterance")))
                return request_status(project_id, request_id)
            current, current_meta, segments = _source(project_id, verify_files=False)
            _chosen_voice(current, current_meta)
            selected = [s for s in segments if s.id in binding]
            checkpoint = current_meta["requests"][request_id].get("selection_checkpoint")
            if checkpoint is not None and checkpoint != recovery.selections(selected):
                raise BookCreatorError("Selected Takes changed during rendering; older work cannot replace the newer selection")
            if _binding(current, selected) != binding or current_meta["voice_selection"] != meta["voice_selection"]:
                raise BookCreatorError("Narrator or source changed during rendering; older results cannot replace it")
            recovery.mark_attempt(project_id, request_id, state="running", inflight=sid)
            records = pipeline.render_utterances(project_id, [sid], renderer=renderer, resume=True, run_qa=True, promote=True, job_id=job_id)
            success = len(records) == 1 and records[0].success and records[0].selected and records[0].qa_passed and not records[0].stale
            error = (records[0].error if records else "Missing render result") or "Technical QA failed"
            current_segment = next(s for s in load_segments(project_id) if s.id == sid)
            # An HTTP failure's wording cannot establish that remote inference
            # stopped. Only a non-dispatched failure or an actually retained,
            # hash-bound candidate establishes a safe bounded repair retry.
            candidate = next((take for take in current_segment.takes
                              if records and take.id == records[0].take_id), None)
            confirmed_return = bool(candidate and candidate.audio_sha256 and candidate.audio_file
                                    and file_sha256(_safe(safe_project_root(project_id), candidate.audio_file)) == candidate.audio_sha256)
            uncertain = not success and (not records or (records[0].dispatched and not confirmed_return))
            recovery.mark_attempt(project_id, request_id, state="running" if success else "waiting",
                                  reason=None if success else error[:500], inflight=sid if uncertain else None,
                                  segment=current_segment)
            # The shared per-segment recorder completes ordinary synthesis jobs
            # at the last line. This composite job also owns MP3 publication, so
            # persist progress atomically without ever marking early success.
            def progress(job):
                if success:
                    if sid not in job.completed_segment_ids:
                        job.completed_segment_ids.append(sid)
                    job.failed_segments.pop(sid, None)
                else:
                    job.completed_segment_ids = [item for item in job.completed_segment_ids if item != sid]
                    job.failed_segments[sid] = (records[0].error if records else "Missing render result") or "Technical QA failed"
                job.finished_at = None
            mutate_job(project_id, job_id, progress)
            if not success:
                raise BookCreatorError("A source line needs repair after technical QA. Resume to retry only unfinished audio")
        mutate_job(project_id, job_id, lambda j: setattr(j, "message", "Checking and exporting MP3 audio"))
        for chapters in _export_groups(receipt["chapter_ids"]):
            job = get_job(project_id, job_id)
            if job.stop_requested or job.status == SynthesisJobStatus.CANCELLED or recovery.paused(project_id, request_id):
                if job.status != SynthesisJobStatus.CANCELLED:
                    mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED), setattr(j, "message", "Paused before MP3 export")))
                return request_status(project_id, request_id)
            _export(project_id, request_id, chapters, binding)
        _, completed_meta = _owned(project_id)
        if not _exports_complete(completed_meta, completed_meta["requests"][request_id]):
            raise BookCreatorError("MP3 publication is incomplete; resume this request to finish")
        job = get_job(project_id, job_id)
        if job.status == SynthesisJobStatus.CANCELLED or job.stop_requested or recovery.paused(project_id, request_id):
            if job.status != SynthesisJobStatus.CANCELLED:
                mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.INTERRUPTED), setattr(j, "message", "Paused after export; existing audio is retained")))
            return request_status(project_id, request_id)
        mutate_job(project_id, job_id, lambda j: (setattr(j, "status", SynthesisJobStatus.SUCCEEDED),
            setattr(j, "finished_at", datetime.now(timezone.utc)), setattr(j, "message", "MP3 ready to play; human listening review pending")))
        recovery.mark_attempt(project_id, request_id, state="completed")
    except Exception as exc:
        def failed(job):
            if job.status != SynthesisJobStatus.CANCELLED:
                job.status = SynthesisJobStatus.INTERRUPTED
                job.message = str(exc)[:1000]
        mutate_job(project_id, job_id, failed)
        _, fresh_meta = _owned(project_id)
        recovery.mark_attempt(project_id, request_id, state="waiting", reason=str(exc)[:500],
                              inflight=fresh_meta["requests"][request_id].get("inflight_utterance_id"))
    finally:
        if lease:
            unregister_running_batch(lease)
        mark_job_active(project_id, job_id, False)
    return request_status(project_id, request_id)


def _selected_audio(project, segments):
    from .runtime_binding import require_production_take, RuntimeBindingError
    try:
        for segment in segments:
            require_production_take(takes.selected_take(segment))
    except RuntimeBindingError as exc:
        raise BookCreatorError(str(exc)) from exc
    root = safe_project_root(project.id)
    casting_binding = current_frozen_binding(project, segments)
    coverage = release.check_coverage(segments, project_root=root, verify_bytes=True, require_qa=True)
    if not coverage.ready:
        raise BookCreatorError("Selected source audio has missing, stale or QA-failing Takes")
    result = []
    for segment in segments:
        take = takes.selected_take(segment)
        role = next(r for r in project.roles if r.id == segment.role_id)
        requirement = pipeline._requirement_for_take(project, segment, take)
        if (not take.audio_sha256 or take.role_id != segment.role_id or take.source_text_hash != text_sha256(segment.text)
                or take.spoken_text_revision_id != segment.spoken_text_revision_id or not requirement
                or take.input_fingerprint != requirement.fingerprint()
                or segment.playback_speed != 1 or take.playback_speed != 1 or take.baked_speed != 1
                or segment.edit_state is not None or take.edit_state is not None):
            raise BookCreatorError("Selected Take no longer matches its complete natural-speed source and narrator")
        if role.voice_identity and role.voice_identity.mechanism == "qwen_reference":
            identity, version, _ = _identity_for_take(take)
            if identity != role.voice_identity.id or version != voice_revision_for_role(role):
                raise BookCreatorError("Selected Take final conditioning differs from the chosen voice")
        else:
            version = voice_revision_for_role(role)
            if (take.fingerprint_components or {}).get("voice_revision") != version:
                raise BookCreatorError("Selected provider voice version changed")
            identity = "provider:" + str(take.engine_id) + ":" + str(role.qwen3_speaker)
        path = _safe(root, take.audio_file)
        duration = _decoded_duration(path)
        if not math.isfinite(duration) or duration <= 0 or file_sha256(path) != take.audio_sha256:
            raise BookCreatorError("Selected Take bytes cannot be verified and decoded")
        row = {"utterance_id": segment.id, "chapter_id": segment.chapter_id, "take_id": take.id,
            "audio_file": take.audio_file, "audio_sha256": take.audio_sha256, "duration_seconds": duration,
            "voice_id": identity, "voice_version": version, "role_id": segment.role_id}
        if casting_binding:
            row["casting_manifest_id"] = casting_binding["manifest_id"]
        result.append(row)
    return result


def _export(project_id, request_id, chapter_ids, binding):
    project, meta, segments = _source(project_id)
    _chosen_voice(project, meta)
    receipt = meta["requests"][request_id]
    rows = [s for s in segments if s.chapter_id in chapter_ids]
    speech = [s for s in rows if s.id not in meta["scene_breaks"]]
    if _binding(project, [s for s in segments if s.id in binding]) != binding:
        raise BookCreatorError("Source or narrator changed before MP3 export")
    selected = _selected_audio(project, speech)
    export_id = "mp3_" + _hash({"chapter_ids": chapter_ids, "voice": meta["voice_selection"], "selections": selected})[:28]
    root = safe_project_root(project_id)
    relative = "book_creator/exports/" + export_id
    path = _safe(root, relative + "/receipt.json")
    if path.is_file():
        result = json.loads(path.read_text("utf-8"))
        if any(file_sha256(_safe(root, result[k])) != result[h] for k, h in [("audio_file", "audio_sha256"), ("master_file", "master_sha256")]):
            raise BookCreatorError("Completed immutable MP3 export has changed")
    else:
        if path.parent.exists():
            relative += "/resume_" + uuid4().hex[:12]
            path = _safe(root, relative + "/receipt.json")
        selected_by_id = {r["utterance_id"]: r for r in selected}
        cursor, mapping, audio_paths = 0.0, [], []
        for segment in rows:
            if segment.id in selected_by_id:
                entry = dict(selected_by_id[segment.id]); audio = _safe(root, entry.pop("audio_file"))
                seconds = entry.pop("duration_seconds"); entry["kind"] = "speech"
            else:
                audio = _safe(root, relative + "/pauses/" + segment.id + ".wav")
                _pause(_safe(root, selected[0]["audio_file"]), audio, 1.25)
                seconds = _decoded_duration(audio)
                entry = {"utterance_id": segment.id, "chapter_id": segment.chapter_id, "kind": "scene_break", "role_id": None}
            entry.update(start_seconds=cursor, end_seconds=cursor + seconds)
            cursor += seconds; mapping.append(entry); audio_paths.append(audio)
        master = _safe(root, relative + "/clean-master.wav"); mp3 = _safe(root, relative + "/listening.mp3")
        _assemble_pcm(audio_paths, master)
        duration = _decoded_duration(master)
        if abs(duration - cursor) > 0.001:
            raise BookCreatorError("Assembled chapter duration differs from complete source coverage")
        master_hash = file_sha256(master)
        report = qa.qa_take_audio(take_id=export_id, audio_path=master, text="\n".join(s.text for s in speech),
                                 expected_sha256=master_hash, language=project.metadata.get("source_language") or "English")
        if not report.passed or report.blocking_failures() or not report.executed_check_ids():
            raise BookCreatorError("Assembled audio did not pass blocking technical QA")
        _mp3(master, mp3)
        delivered = _decoded_duration(mp3)
        loudness = _loudness(mp3)
        if abs(delivered - duration) > 0.1 or loudness is None or not math.isfinite(loudness):
            raise BookCreatorError("MP3 decoding or loudness measurement failed")
        result = {"export_id": export_id, "label": next(c.title for c in project.chapters if c.id == chapter_ids[0]) if len(chapter_ids) == 1 else project.name + " — selected chapters",
            "chapter_ids": chapter_ids, "audio_file": mp3.relative_to(root).as_posix(), "audio_sha256": file_sha256(mp3),
            "master_file": master.relative_to(root).as_posix(), "master_sha256": master_hash, "duration_seconds": delivered,
            "master_duration_seconds": duration, "integrated_loudness_lufs": loudness, "map": mapping,
            "selections": selected, "technical_qa": "passed", "human_review": "pending", "qa_report": report.model_dump(mode="json"),
            **meta["voice_selection"]}
        _immutable(path, _json(result))
    _immutable(path.with_name("source-to-audio-map.json"), _json({"audio_sha256": result["audio_sha256"], "map": result["map"]}))
    with project_meta_lock(project_id), project_segments_lock(project_id):
        current, current_meta, current_rows = _source(project_id)
        _chosen_voice(current, current_meta)
        if (_binding(current, [s for s in current_rows if s.id in binding]) != binding
                or _selected_audio(current, [s for s in current_rows if s.id in {x.id for x in speech}]) != selected
                or current_meta["requests"][request_id]["intent"] != receipt["intent"]):
            raise BookCreatorError("Newer source, voice or selected Takes superseded this MP3")
        current_meta["exports"][export_id] = result
        outputs = current_meta["requests"][request_id]["export_ids"]
        if export_id not in outputs:
            outputs.append(export_id)
        save_project(current)
    return result


def audio_artifact(project_id, export_id):
    if not re.fullmatch(r"mp3_[a-f0-9]{28}", export_id):
        raise BookCreatorError("Invalid recording identity")
    _project, meta = _owned(project_id)
    result = meta["exports"].get(export_id)
    if not result:
        raise BookCreatorError("Recording is not available")
    path = _safe(safe_project_root(project_id), result["audio_file"])
    if path.suffix != ".mp3":
        raise BookCreatorError("Recording is not an MP3")
    return path, result["audio_sha256"], result["label"]
