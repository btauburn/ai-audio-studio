"""Audiobooker analysis subprocess bridge (Phase 3).

Runs inside the **isolated audiobooker environment**.  It must not import
ai-audio-studio, so it only depends on the standard library plus the pinned
``audiobooker`` package, and it writes a single JSON document to stdout.

Input is read from ``DUBBING_AUDIOBOOKER_PAYLOAD`` (JSON) so nothing has to be
quoted through the shell.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_audiobooker():
    import audiobooker  # type: ignore

    return audiobooker


def _utterance_to_dict(utterance: Any, chapter_index: int) -> Dict[str, Any]:
    utterance_type = getattr(utterance, "utterance_type", None)
    return {
        "text": getattr(utterance, "text", "") or "",
        "speaker": getattr(utterance, "speaker", None),
        "utterance_type": getattr(utterance_type, "value", utterance_type) or "narration",
        "chapter_index": int(getattr(utterance, "chapter_index", chapter_index) or 0),
        "line_index": int(getattr(utterance, "line_index", 0) or 0),
        "start_pos": getattr(utterance, "start_pos", None),
        "end_pos": getattr(utterance, "end_pos", None),
        "attribution_source": getattr(utterance, "attribution_source", None) or "unknown",
        "confidence": getattr(utterance, "confidence", None),
        "emotion": getattr(utterance, "emotion", None),
    }


def _chapter_to_dict(chapter: Any, index: int) -> Dict[str, Any]:
    utterances = []
    for utterance in getattr(chapter, "utterances", []) or []:
        payload = _utterance_to_dict(utterance, index)
        if payload["text"].strip():
            utterances.append(payload)
    return {
        "index": int(getattr(chapter, "index", index) or 0),
        "title": getattr(chapter, "title", "") or "",
        "source_file": getattr(chapter, "source_file", None),
        "utterances": utterances,
    }


def _analysis_from_project(project: Any, *, title: str = "", author: str = "") -> Dict[str, Any]:
    config = getattr(project, "config", None)
    chapters: List[Dict[str, Any]] = []
    for index, chapter in enumerate(getattr(project, "chapters", []) or []):
        chapters.append(_chapter_to_dict(chapter, index))

    speakers: List[str] = []
    casting = getattr(project, "casting", None)
    if casting is not None:
        try:
            speakers = [str(name) for name in casting.list_characters()]
        except Exception:  # noqa: BLE001 - surfaced as a warning instead
            speakers = []

    warnings: List[str] = []
    pronunciations: Dict[str, str] = {}
    overrides = getattr(config, "pronunciation_overrides", None)
    if isinstance(overrides, dict):
        pronunciations.update({str(k): str(v) for k, v in overrides.items()})
    else:
        warnings.append("pronunciation_overrides unavailable on the pinned version")

    return {
        "title": title or str(getattr(project, "title", "") or ""),
        "author": author or str(getattr(project, "author", "") or ""),
        "language": getattr(config, "language_code", None),
        "chapters": chapters,
        "speakers": speakers,
        "pronunciations": pronunciations,
        "warnings": warnings,
    }


def main() -> int:
    payload = json.loads(os.environ.get("DUBBING_AUDIOBOOKER_PAYLOAD") or "{}")
    audiobooker = _load_audiobooker()
    project_factory = audiobooker.AudiobookProject
    scratch = Path(payload.get("scratch_dir") or os.getcwd())
    scratch.mkdir(parents=True, exist_ok=True)

    source_path = payload.get("source_path")
    text = payload.get("text")
    title = payload.get("title") or ""
    author = payload.get("author") or ""
    language = payload.get("language") or None

    if text is not None:
        project = project_factory.from_string(
            text, title=title or "untitled", author=author, lang=language or "en"
        )
    elif source_path:
        path = Path(source_path)
        suffix = path.suffix.lower()
        factory = {
            ".epub": project_factory.from_epub,
            ".pdf": project_factory.from_pdf,
            ".docx": project_factory.from_docx,
            ".txt": project_factory.from_text,
            ".md": project_factory.from_text,
        }.get(suffix)
        if factory is None:
            raise SystemExit(f"unsupported source extension: {suffix}")
        kwargs: Dict[str, Any] = {}
        if title:
            kwargs["title"] = title
        if author:
            kwargs["author"] = author
        project = factory(str(path), **kwargs)
    else:
        raise SystemExit("payload must contain source_path or text")

    project.compile()

    result = _analysis_from_project(project, title=title, author=author)
    result["source_path"] = str(source_path) if source_path else None
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
