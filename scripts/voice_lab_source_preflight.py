"""Inventory local Chapter 1 evidence without extracting or transmitting prose.

Usage: python scripts/voice_lab_source_preflight.py --root PATH [--root PATH]
       --report NEW_JSON_PATH

Read-only to source/project locations. The optional report is created exclusively
and existing evidence is never overwritten. No network, import, rendering or GPU
initialization is performed. Search limits and unreadable locations are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import sys
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_service.app.audiobook.listening_excerpt import (  # noqa: E402
    blocked_source_report,
    chapter_number,
)

EXCLUDED_DIRS = {".git", "node_modules", "site-packages", "__pycache__", ".venv", "venv",
                 "models", "checkpoints", ".cache", "cache", ".next"}
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_FILES = 250_000


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Headings(HTMLParser):
    def __init__(self):
        super().__init__()
        self.level = 0
        self.current: list[str] = []
        self.headings: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self.level += 1

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3") and self.level:
            self.level -= 1
            if self.level == 0:
                self.headings.append(" ".join("".join(self.current).split())[:240])
                self.current = []

    def handle_data(self, data):
        if self.level:
            self.current.append(data)


def inspect_epub(path: Path) -> dict:
    """Inspect metadata, TOC and every spine heading; never emit chapter prose."""
    report = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
              "metadata": {}, "toc": [], "documents": []}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 5000 or sum(m.file_size for m in members) > MAX_FILE_BYTES:
            raise ValueError("EPUB inspection exceeds bounded expanded-size/member limits")

        def read(name):
            if archive.getinfo(name).file_size > MAX_MEMBER_BYTES:
                raise ValueError("EPUB member exceeds bounded size limit")
            return archive.read(name)

        container = ET.fromstring(read("META-INF/container.xml"))
        package_name = next(e.attrib["full-path"] for e in container.iter() if e.tag.endswith("rootfile"))
        package = ET.fromstring(read(package_name))
        base = posixpath.dirname(package_name)
        manifest = {}
        for entry in package.iter():
            tag = entry.tag.rsplit("}", 1)[-1]
            if tag in {"title", "creator", "identifier", "language", "date"}:
                report["metadata"].setdefault(tag, []).append((entry.text or "")[:500])
            elif tag == "item":
                manifest[entry.attrib["id"]] = entry.attrib
        for entry in manifest.values():
            if entry.get("media-type") == "application/x-dtbncx+xml":
                toc = ET.fromstring(read(posixpath.normpath(posixpath.join(base, entry["href"]))))
                for point in toc.iter():
                    if point.tag.rsplit("}", 1)[-1] != "navPoint":
                        continue
                    label = next((" ".join(e.itertext()).strip() for e in point if e.tag.endswith("navLabel")), "")
                    href = next((e.attrib.get("src", "") for e in point if e.tag.endswith("content")), "")
                    report["toc"].append({"label": label[:240], "href": href, "chapter_number": chapter_number(label)})
            elif "nav" in entry.get("properties", "").split():
                nav = ET.fromstring(read(posixpath.normpath(posixpath.join(base, entry["href"]))))
                for link in nav.iter():
                    if link.tag.rsplit("}", 1)[-1] == "a":
                        label = " ".join(link.itertext()).strip()
                        report["toc"].append({"label": label[:240], "href": link.attrib.get("href"), "chapter_number": chapter_number(label)})
        for entry in package.iter():
            if entry.tag.rsplit("}", 1)[-1] != "itemref":
                continue
            item = manifest.get(entry.attrib.get("idref"), {})
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                continue
            name = posixpath.normpath(posixpath.join(base, item["href"]))
            data = read(name)
            parser = Headings()
            parser.feed(data.decode("utf-8-sig"))
            report["documents"].append({"member": name, "sha256": hashlib.sha256(data).hexdigest(),
                                        "bytes": len(data), "headings": parser.headings,
                                        "heading_numbers": [chapter_number(h) for h in parser.headings]})
    report["is_super_supportive"] = any(t.casefold().strip() == "super supportive" for t in report["metadata"].get("title", []))
    report["available_chapters"] = sorted({t["chapter_number"] for t in report["toc"] if t["chapter_number"] is not None})
    heading_numbers = {n for d in report["documents"] for n in d["heading_numbers"] if n is not None}
    report["chapter_one_toc_and_heading"] = 1 in report["available_chapters"] and 1 in heading_numbers
    return report


def inspect_project(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if "super supportive" not in (str(data.get("name", "")) + json.dumps(data.get("source_revisions", []))).casefold():
        return None
    report = {"path": str(path), "project_id": data.get("id"), "name": data.get("name"),
              "chapters": [], "sources": []}
    for chapter in data.get("chapters", []):
        report["chapters"].append({"id": chapter.get("id"), "title": chapter.get("title"),
                                   "imported_index": chapter.get("index"),
                                   "chapter_number": chapter_number(chapter.get("title"))})
    for source in data.get("source_revisions", []):
        row = {k: source.get(k) for k in ("id", "uri", "sha256", "byte_length", "title", "snapshot_file")}
        relative = source.get("snapshot_file")
        if relative:
            snapshot = (path.parent / relative).resolve()
            if not snapshot.is_relative_to(path.parent.resolve()):
                row["error"] = "Snapshot path escapes canonical project"
            elif snapshot.is_file():
                row["snapshot_actual_sha256"] = sha256(snapshot)
                row["snapshot_hash_verified"] = row["snapshot_actual_sha256"] == source.get("sha256")
                if zipfile.is_zipfile(snapshot):
                    row["epub"] = inspect_epub(snapshot)
            else:
                row["error"] = "Snapshot missing"
        report["sources"].append(row)
    return report


def inspect_roots(roots: list[Path], *, max_files: int = MAX_FILES) -> dict:
    report = {"schema": "voice-lab-local-source-preflight-v1",
              "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "network_access": False, "source_text_exported": False,
              "searched_roots": [], "epubs": [], "projects": [], "other_candidates": [],
              "errors": [], "skipped_junctions": [], "limits": {"max_files": max_files, "max_file_bytes": MAX_FILE_BYTES,
                                         "excluded_directories": sorted(EXCLUDED_DIRS)},
              "files_seen": 0, "search_complete": True}
    seen = set()
    for root in roots:
        root = root.resolve()
        report["searched_roots"].append({"path": str(root), "exists": root.exists()})
        if not root.exists():
            continue

        def onerror(error):
            report["errors"].append({"path": str(error.filename), "error": type(error).__name__})
            report["search_complete"] = False

        for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
            kept = []
            for name in dirs:
                child = Path(directory) / name
                if child.is_symlink() or (hasattr(child, "is_junction") and child.is_junction()):
                    report["skipped_junctions"].append(str(child))
                elif name.casefold() not in EXCLUDED_DIRS:
                    kept.append(name)
            dirs[:] = kept
            for name in files:
                path = Path(directory) / name
                key = str(path).casefold()
                if key in seen:
                    continue
                seen.add(key)
                report["files_seen"] += 1
                if report["files_seen"] > max_files:
                    report["search_complete"] = False
                    report["errors"].append({"path": str(root), "error": "file_count_limit"})
                    return _finish(report)
                relevant = "supportive" in name.casefold()
                source_candidate = path.suffix.casefold() == ".epub" or name.casefold() == "project.json" or (
                    relevant and path.suffix.casefold() in {".txt", ".md", ".pdf", ".html", ".xhtml", ".json", ".docx", ".mobi", ".azw3"})
                if not source_candidate:
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        if source_candidate:
                            report["errors"].append({"path": str(path), "error": "file_size_limit"})
                            report["search_complete"] = False
                        continue
                    if path.suffix.casefold() == ".epub":
                        report["epubs"].append(inspect_epub(path))
                    elif name.casefold() == "project.json":
                        project = inspect_project(path)
                        if project:
                            report["projects"].append(project)
                    elif relevant and path.suffix.casefold() in {".txt", ".md", ".pdf", ".html", ".xhtml", ".json", ".docx", ".mobi", ".azw3"}:
                        report["other_candidates"].append({"path": str(path), "bytes": path.stat().st_size,
                                                           "sha256": sha256(path), "assessment": "Requires format/source inspection; filename is not chapter identity"})
                except (OSError, ValueError, KeyError, StopIteration, ET.ParseError, zipfile.BadZipFile) as error:
                    report["errors"].append({"path": str(path), "error": type(error).__name__})
                    report["search_complete"] = False
    return _finish(report)


def _finish(report: dict) -> dict:
    sources = [e for e in report["epubs"] if e["is_super_supportive"]]
    sources += [s["epub"] for p in report["projects"] for s in p["sources"] if s.get("epub", {}).get("is_super_supportive")]
    available = sorted({n for e in sources for n in e["available_chapters"]})
    chapter_one = [e["path"] for e in sources if e["chapter_one_toc_and_heading"]]
    report["available_chapters"] = available
    report["chapter_one_candidates"] = chapter_one
    report["source_scope"] = ("chapter_one_requires_full_local_inspection" if chapter_one else "chapter_one_not_found_in_inspected_epubs_and_canonical_projects")
    if not chapter_one:
        report["excerpt"] = blocked_source_report(available_chapters=available, inspected_paths=[e["path"] for e in sources])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = inspect_roots(args.root)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"report": str(args.report.resolve()), "source_scope": report["source_scope"],
                          "available_chapters": report["available_chapters"], "files_seen": report["files_seen"],
                          "search_complete": report["search_complete"], "errors": report["errors"]}))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
