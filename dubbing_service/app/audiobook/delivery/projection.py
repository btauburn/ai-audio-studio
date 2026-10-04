"""Read-only projection of verified listening exports onto canonical Studio state.

The configured manifests are receipts, not a replacement project database. Every
discovery checks their selection, existing QA, source WAVs and exported bytes.
No loader which migrates, repairs or persists Studio state is used here.
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path

from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.models.project import Project, Segment
from dubbing_service.app.services.takes import take_is_selectable


class _NotReviewable(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _NotReviewable(reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, snapshots: dict[Path, str]):
    content = path.read_bytes()
    snapshots[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8"))


def _inside(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Invalid asset path")
    normalized = relative.replace("\\", "/")
    _require(not any(p in ("", ".", "..") for p in normalized.split("/")), "Invalid asset path")
    path = Path(normalized)
    _require(not path.is_absolute() and not path.drive and ":" not in normalized, "Invalid asset path")
    candidate = (root / path).resolve()
    _require(candidate.is_relative_to(root), "Asset escapes its configured directory")
    return candidate


def _positive(value) -> float:
    _require(isinstance(value, (float, int)) and not isinstance(value, bool), "Invalid audio duration")
    result = float(value)
    _require(math.isfinite(result) and result > 0, "Invalid audio duration")
    return result


def _fingerprint(manifest: dict) -> str:
    payload = {
        "version": manifest["exporter_version"],
        "settings": manifest["settings"],
        "project_id": manifest["project_id"],
        "source_hashes": manifest["source_hashes"],
        "speaker_corrections_sha256": manifest["speaker_corrections_sha256"],
        "rendition_state_hashes": manifest["rendition_state_hashes"],
        "ordered_rows": manifest["ordered_rows"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _speech_binding(root: Path, segment: Segment, row: dict, audio_hashes: dict[Path, str]) -> dict:
    take = next((t for t in segment.takes if t.id == segment.selected_take_id), None)
    _require(take_is_selectable(take) is None, "Selected Take is missing, stale, archived or QA blocked")
    _require(take is not None and take.id == row.get("take_id"), "Export references an unselected Take")
    _require(take.audio_file == row.get("audio_file") and take.audio_sha256 == row.get("audio_sha256"),
             "Export Take identity differs from current selection")
    _require(take.text == segment.text and take.role_id == segment.role_id,
             "Selected Take text or speaker differs from current segment")
    _require(take.spoken_text_revision_id == segment.spoken_text_revision_id,
             "Selected Take text revision is stale")
    _require(take.edit_state is None and take.playback_speed == take.baked_speed
             and take.volume_gain == take.baked_volume, "Selected Take has unapplied edits")
    audio = _inside(root, take.audio_file)
    actual = _sha(audio)
    audio_hashes[audio] = actual
    _require(bool(take.audio_sha256) and actual == take.audio_sha256, "Selected Take audio hash mismatch")
    qa = take.qa_report
    _require(qa is not None and qa.matches(take.id, actual) and qa.passed
             and bool(qa.executed_check_ids()) and not qa.blocking_failures(), "Selected Take lacks valid executed QA")
    _require(row.get("qa_passed") is True and row.get("qa_check_version") == qa.qa_check_version
             and sorted(row.get("qa_executed_checks", [])) == sorted(qa.executed_check_ids()),
             "Export QA receipt differs from current Take")
    with wave.open(str(audio), "rb") as stream:
        frames = stream.getnframes()
        _require(stream.getframerate() == 24000 and stream.getnchannels() == 1
                 and stream.getsampwidth() == 2 and frames > 0, "Selected audio is not a supported source WAV")
        _require(len(stream.readframes(frames)) == frames * 2, "Selected source WAV is truncated")
    _require(row.get("frames_24000") == frames
             and abs(_positive(row["duration_seconds"]) - frames / 24000) < 0.000001,
             "Export source duration differs from selected audio")
    return {"segment_id": segment.id, "take_id": take.id, "audio_sha256": actual}


def _discover(project_dir: Path, export_manifests: list[Path]) -> dict:
    root = project_dir.resolve()
    snapshots: dict[Path, str] = {}
    audio_hashes: dict[Path, str] = {}
    project = Project.model_validate(_json(root / "project.json", snapshots))
    validate_simple_component_id(project.id)
    _require(root.name == project.id, "Project directory identity mismatch")
    segments = [Segment.model_validate(s) for s in _json(root / "segments.json", snapshots)]
    _require(len({s.id for s in segments}) == len(segments), "Duplicate canonical segment IDs")
    chapters = {c.id: c for c in project.chapters}
    _require(len(chapters) == len(project.chapters), "Duplicate canonical chapter IDs")
    for chapter in chapters.values():
        validate_simple_component_id(chapter.id)
    for segment in segments:
        validate_simple_component_id(segment.id)
    live = [s for s in segments if s.lineage is None or s.lineage.retired_at is None]
    pauses = {}
    if project.metadata.get("scene_break_manifest"):
        pause_rows = _json(_inside(root, project.metadata["scene_break_manifest"]), snapshots)
        pauses = {p["segment_id"]: p for p in pause_rows}
        _require(len(pauses) == len(pause_rows), "Duplicate canonical scene breaks")
    _require(bool(export_manifests), "No configured listening exports")
    result = {}
    numbers = set()
    for configured in dict.fromkeys(Path(p).resolve() for p in export_manifests):
        manifest = _json(configured, snapshots)
        _require(manifest.get("schema_version") == 1 and manifest.get("status") == "verified_listening_export",
                 "Manifest is not a verified listening export")
        _require(manifest.get("project_id") == project.id, "Export belongs to another project")
        expected_state = {name: snapshots[root / name] for name in ("project.json", "segments.json")}
        _require(manifest.get("rendition_state_hashes") == expected_state, "Export canonical state has changed")
        revision = _fingerprint(manifest)
        _require(manifest.get("source_fingerprint") == revision, "Export source fingerprint mismatch")
        origin = project.metadata.get("derived_from_project")
        if origin:
            validate_simple_component_id(origin)
            original = root.parent / origin
            _require(manifest.get("source_hashes") == {
                name: _sha(original / name) for name in ("project.json", "segments.json")
            }, "Original source project has changed")
        exports = manifest["chapter_exports"]
        export_ids = [c["chapter_id"] for c in exports]
        _require(bool(exports) and len(set(export_ids)) == len(exports) and set(export_ids) <= chapters.keys(),
                 "Export has duplicate or unknown chapters")
        scoped = sorted((s for s in live if s.chapter_id in export_ids),
                        key=lambda s: (chapters[s.chapter_id].index, s.index))
        _require(len({(s.chapter_id, s.index) for s in scoped}) == len(scoped), "Ambiguous canonical segment order")
        rows = manifest["ordered_rows"]
        _require([r["segment_id"] for r in rows] == [s.id for s in scoped],
                 "Export chapter coverage or order is incomplete")
        bindings = {}
        chapter_offsets = {}
        combined_offset = 0.0
        for sequence, (segment, row) in enumerate(zip(scoped, rows)):
            _require(row.get("sequence") == sequence and row.get("chapter_id") == segment.chapter_id
                     and row.get("segment_index") == segment.index and row.get("role_id") == segment.role_id
                     and row.get("canonical_text") == segment.text
                     and row.get("canonical_text_sha256") == hashlib.sha256(segment.text.encode("utf-8")).hexdigest(),
                     "Export segment snapshot differs from canonical state")
            duration = _positive(row["duration_seconds"])
            if segment.id in pauses:
                pause = pauses[segment.id]
                _require(row.get("kind") == "scene_break" and row.get("take_id") is None
                         and segment.selected_take_id is None and not segment.takes
                         and not segment.audio_file and not segment.alt_audio_file
                         and pause["chapter_id"] == segment.chapter_id and pause["index"] == segment.index
                         and pause["source_text"] == segment.text
                         and abs(_positive(pause["seconds"]) - duration) < 0.000001
                         and row.get("frames_24000") == round(duration * 24000),
                         "Export scene break differs from its canonical definition")
            else:
                _require(row.get("kind") in ("reused_narration", "new_narration", "new_dialogue"),
                         "Speech cannot be omitted or replaced with an undeclared scene break")
                bindings[segment.id] = _speech_binding(root, segment, row, audio_hashes)
            chapter_start = chapter_offsets.get(segment.chapter_id, 0.0)
            for field, expected in (("chapter_start_seconds", chapter_start),
                                    ("chapter_end_seconds", chapter_start + duration),
                                    ("combined_start_seconds", combined_offset),
                                    ("combined_end_seconds", combined_offset + duration)):
                _require(math.isfinite(float(row[field])) and abs(float(row[field]) - expected) < 0.000001,
                         "Export timeline does not match selected audio")
            chapter_offsets[segment.chapter_id] = chapter_start + duration
            combined_offset += duration
        for exported in exports:
            chapter_id = exported["chapter_id"]
            number = exported["chapter_number"]
            _require(isinstance(number, int) and not isinstance(number, bool) and number >= 0,
                     "Invalid chapter number")
            chapter_rows = [r for r in rows if r["chapter_id"] == chapter_id]
            _require(chapter_rows and exported["ordered_segment_ids"] == [r["segment_id"] for r in chapter_rows]
                     and all(r["chapter_number"] == number for r in chapter_rows), "Chapter export order mismatch")
            duration = _positive(exported["duration_seconds"])
            _require(abs(duration - chapter_offsets[chapter_id]) < 0.000001, "Chapter duration mismatch")
            candidates = [a for a in exported["artifacts"] if Path(a["file"]).suffix.lower() == ".mp3"]
            _require(len(candidates) == 1, "Chapter needs exactly one verified MP3")
            artifact = candidates[0]
            _require(artifact.get("source_fingerprint") == revision and artifact.get("kind") == "listening_export"
                     and artifact.get("verified_loudness") is True and artifact["probe"]["codec_name"] == "mp3"
                     and abs(_positive(artifact["probe"]["duration_seconds"]) - duration) <= 0.15,
                     "MP3 receipt is not valid for this chapter revision")
            audio = _inside(configured.parent, artifact["file"])
            digest = _sha(audio)
            audio_hashes[audio] = digest
            _require(audio.stat().st_size > 0 and digest == artifact["sha256"], "Export MP3 hash mismatch")
            chapter = {"chapter_id": chapter_id, "chapter_number": number, "title": chapters[chapter_id].title,
                       "audio_path": str(audio), "audio_sha256": digest, "duration_seconds": duration,
                       "revision": revision, "selected_takes": [bindings[r["segment_id"]] for r in chapter_rows
                                                                  if r["segment_id"] in bindings],
                       "export_manifest_sha256": snapshots[configured]}
            if chapter_id in result:
                _require(result[chapter_id] == chapter, "Conflicting exports for the same chapter")
            else:
                _require(number not in numbers, "Duplicate chapter number")
                result[chapter_id] = chapter
                numbers.add(number)
    _require(bool(result), "No reviewable chapters")
    # Recheck bytes after all parsing/hash work, without acquiring writer locks or
    # touching canonical metadata. Service serving performs its own exact check.
    for path, digest in {**snapshots, **audio_hashes}.items():
        _require(_sha(path) == digest, "Project or audio changed during discovery")
    return {"schema_version": 1, "project_id": project.id,
            "book_title": str(project.metadata.get("source_title") or project.name),
            "chapters": sorted(result.values(), key=lambda c: chapters[c["chapter_id"]].index)}


def discover_reviewable(project_dir: Path, export_manifests: list[Path]) -> dict:
    """Return exact existing chapter artifacts, or a safe, text-free failure.

    Callers configure manifests explicitly (and may expand their own globs).
    Source text is inspected locally for identity but never enters this result.
    """
    try:
        return _discover(Path(project_dir), export_manifests)
    except _NotReviewable:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, wave.Error, EOFError):
        raise ValueError("Listening export validation failed; metadata or audio is invalid or unavailable") from None
