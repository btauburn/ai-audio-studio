"""Export an explicitly authorized derived chapter through Studio's Take owner.

This operation synthesizes nothing. It retains every imported source utterance,
uses only selected hash-bound QA-passed Takes, and publishes a human-review copy.
Only recorded source scene separators receive their explicit natural pause.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.core.tts_operations import try_acquire_project_export, release_project_export
from dubbing_service.app.storage.projects import load_project, project_meta_lock, save_project
from dubbing_service.app.storage.segments import load_segments, project_segments_lock
from . import pipeline, qa, release, takes
from .casting_assembly import current_frozen_binding
from .fingerprint import file_sha256, text_sha256, voice_revision_for_role
from .voice_lab_render import (_assemble_pcm, _decoded_duration, _identity_for_take,
                               _immutable, _json, _loudness, _mp3, _path, _transitions,
                               ListeningRenderError)


class ChapterExportError(ListeningRenderError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _source_snapshot(project_id: str, chapter_id: str) -> tuple[Any, list, dict]:
    project = load_project(project_id)
    meta = project.metadata.get("chapter_rerun") or {}
    if (meta.get("authorized") is not True or meta.get("chapter_id") != chapter_id
            or not meta.get("source_project_id") or meta["source_project_id"] == project_id):
        raise ChapterExportError("Chapter export requires an explicitly authorized derived chapter rendition")
    chapter = next((c for c in project.chapters if c.id == chapter_id), None)
    if chapter is None:
        raise ChapterExportError("Authorized chapter is no longer present")
    root = safe_project_root(project_id)
    revisions = {r.id: r for r in project.source_revisions}
    recorded_revisions = meta.get("source_revisions") or []
    if not recorded_revisions:
        raise ChapterExportError("Derived chapter has no immutable source snapshot evidence")
    for expected in recorded_revisions:
        revision = revisions.get(expected.get("id"))
        if (not revision or revision.sha256 != expected.get("sha256")
                or revision.snapshot_file != expected.get("snapshot_file") or not revision.snapshot_file):
            raise ChapterExportError("Canonical source revision changed since the chapter fork")
        path = _path(root, revision.snapshot_file)
        if not path.is_file() or file_sha256(path) != revision.sha256:
            raise ChapterExportError("Canonical source snapshot is missing or hash-mismatched")
    segments = sorted((s for s in load_segments(project_id) if s.chapter_id == chapter_id and release.is_live(s)),
                      key=lambda s: s.index)
    ids = [s.id for s in segments]
    if not ids or len(ids) != len(set(ids)) or ids != meta.get("source_utterance_ids"):
        raise ChapterExportError("Chapter no longer has exact complete ordered source coverage")
    original = meta.get("source_utterances") or []
    if [r.get("utterance_id") for r in original] != ids:
        raise ChapterExportError("Fork-time exact source utterance evidence is missing")
    from .voice_lab_chapter import _hash, _source_proof
    if len(project.chapters) != 1 or _hash(_source_proof(project, original)) != meta.get("expected_source_hash"):
        raise ChapterExportError("Fork-time chapter identity or source/spoken-text revision history changed")
    for segment, expected in zip(segments, original):
        anchor = segment.source_anchor.model_dump(mode="json") if segment.source_anchor else None
        if (expected.get("source_text_sha256") != text_sha256(segment.text)
                or expected.get("source_anchor") != anchor
                or expected.get("spoken_text_revision_id") != segment.spoken_text_revision_id
                or expected.get("index") != segment.index or expected.get("chapter_id") != segment.chapter_id
                or not anchor or anchor.get("source_revision_id") not in revisions):
            raise ChapterExportError("Source text, offsets or spoken-text revision changed since the chapter fork")
    breaks = meta.get("scene_breaks") or []
    break_ids = [b.get("utterance_id") for b in breaks]
    if len(break_ids) != len(set(break_ids)) or not set(break_ids).issubset(ids):
        raise ChapterExportError("Scene-break source records are invalid")
    by_id = {s.id: s for s in segments}
    for record in breaks:
        segment = by_id[record["utterance_id"]]
        if (not re.fullmatch(r"\s*\*{3,}\s*", segment.text)
                or record.get("source_text_sha256") != text_sha256(segment.text)
                or record.get("index") != segment.index or record.get("duration_seconds") != 1.25):
            raise ChapterExportError("Only explicit recorded star separators receive their 1.25-second scene pause")
    from .voice_lab_chapter import _validated_exclusions
    exclusions = _validated_exclusions(project, segments)
    excluded_ids = {row["utterance_id"] for row in exclusions}
    speech_ids = [sid for sid in ids if sid not in set(break_ids) | excluded_ids]
    if not speech_ids or speech_ids != meta.get("speech_utterance_ids"):
        raise ChapterExportError("Speech coverage does not match the explicit source scene-break manifest")
    if any(re.fullmatch(r"\s*\*{3,}\s*", by_id[sid].text) for sid in speech_ids):
        raise ChapterExportError("A source scene separator was incorrectly marked for synthesis")
    return project, segments, {"chapter_id": chapter_id, "chapter_title": chapter.title,
                               "metadata": meta, "source_utterance_ids": ids,
                               "speech_utterance_ids": speech_ids, "breaks": breaks,
                               **({"exclusions": exclusions} if exclusions else {})}


def _take_snapshot(project: Any, segments: list, source: dict) -> tuple[list, list, dict]:
    root = safe_project_root(project.id)
    speech = [s for s in segments if s.id in source["speech_utterance_ids"]]
    casting_binding = current_frozen_binding(project, speech, allow_experimental=True)
    coverage = release.check_coverage(speech, project_root=root, verify_bytes=True, require_qa=True)
    if not coverage.ready:
        raise ChapterExportError("Selected chapter Takes fail blocking release QA: " + json.dumps(coverage.as_dict()))
    roles = {r.id: r for r in project.roles}
    selections, paths, proof = [], [], {}
    for segment in speech:
        take = takes.selected_take(segment)
        role = roles.get(segment.role_id)
        attribution = segment.attribution
        if not role or not attribution or attribution.source == "unknown" or attribution.speaker_id != role.id:
            raise ChapterExportError("Every spoken line requires resolved canonical source attribution")
        if (not take or not take.audio_sha256 or take.role_id != role.id
                or take.source_text_hash != text_sha256(segment.text)
                or take.spoken_text_revision_id != segment.spoken_text_revision_id):
            raise ChapterExportError("Selected Take does not match exact source, role or spoken-text revision")
        if (segment.playback_speed != 1.0 or segment.edit_state is not None
                or take.playback_speed != 1.0 or take.baked_speed != 1.0 or take.edit_state is not None):
            raise ChapterExportError("Chapter review copy requires natural-speed complete unedited source audio")
        requirement = pipeline._requirement_for_take(project, segment, take)
        if requirement is None or take.input_fingerprint != requirement.fingerprint():
            raise ChapterExportError("Selected Take synthesis requirement is stale")
        if role.voice_identity is not None and role.voice_identity.mechanism == "qwen_reference":
            if role.voice_identity.status != "LOCKED":
                raise ChapterExportError("Current chapter cast identity is no longer locked")
            voice_id, version, conditioning = _identity_for_take(take)
            if (voice_id != role.voice_identity.id or conditioning != role.voice_identity.reference_audio_sha256
                    or version != voice_revision_for_role(role)):
                raise ChapterExportError("Final Take conditioning no longer matches its cast voice version")
        else:
            version = voice_revision_for_role(role)
            recorded = take.fingerprint_components or {}
            if recorded.get("voice_revision") != version or not recorded.get("builtin_speaker"):
                raise ChapterExportError("Take lacks a verified stable final provider voice identity")
            voice_id = "provider:" + str(take.engine_id) + ":" + recorded["builtin_speaker"]
            conditioning = None
        path = _path(root, take.audio_file)
        if file_sha256(path) != take.audio_sha256:
            raise ChapterExportError("Selected audio hash changed")
        duration = _decoded_duration(path)
        facts = qa.read_audio_facts(path)
        if not facts or not facts.decodable or not math.isfinite(duration) or duration <= 0:
            raise ChapterExportError("Selected audio cannot be decoded")
        selections.append({"utterance_id": segment.id, "take_id": take.id, "audio_sha256": take.audio_sha256,
                           "input_fingerprint": take.input_fingerprint, "role_id": role.id,
                           "voice_id": voice_id, "voice_version": version})
        paths.append(path)
        proof[segment.id] = {**selections[-1], "role_name": role.name, "conditioning_sha256": conditioning,
                            "source_text_sha256": take.source_text_hash,
                            "source_revision_id": segment.source_anchor.source_revision_id,
                            "spoken_text_revision_id": segment.spoken_text_revision_id,
                            "duration_seconds": duration,
                            "rms_dbfs": 20 * math.log10(max(float(facts.rms), 1e-6))}
    state = {"source": source, "selections": selections,
             "cast": [r.model_dump(mode="json") for r in project.roles if r.id in {s.role_id for s in speech}],
             "casting_binding": casting_binding}
    from .runtime_binding import take_runtime_profiles, PRODUCTION_RUNTIME_PROFILES
    experimental = any(take_runtime_profiles(takes.selected_take(segment)) - PRODUCTION_RUNTIME_PROFILES for segment in speech)
    if experimental:
        state['experimental_runtime'] = {segment.id: sorted(take_runtime_profiles(takes.selected_take(segment))) for segment in speech}
    return paths, selections, {"state_sha256": _digest(state), "proof": proof, "experimental_runtime": experimental}


def _pause(reference: Path, target: Path, seconds: float) -> None:
    """Only a previously recorded source scene separator, with matching PCM."""
    with wave.open(str(reference), "rb") as source:
        channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream, wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        # 8-bit PCM is unsigned; all other supported PCM widths use zero.
        value = b"\x80" if width == 1 else b"\x00" * width
        output.writeframes(value * channels * round(rate * seconds))


def _valid_receipt(root: Path, result: dict, state_sha256: str) -> bool:
    if result.get("state_sha256") != state_sha256:
        return False
    return all(result.get(field) and _path(root, result[field]).is_file()
               and file_sha256(_path(root, result[field])) == result.get(hash_field)
               for field, hash_field in [("audio_file", "audio_sha256"), ("master_file", "master_sha256")])


def export_chapter(project_id: str, chapter_id: str) -> dict:
    """Export selected chapter Takes without synthesis or human approval gates."""
    lease = try_acquire_project_export(project_id, kind="voice_lab_chapter_export")
    if lease is None:
        raise ChapterExportError("Project is busy rendering or changing; resume export after its existing work")
    try:
        with project_meta_lock(project_id), project_segments_lock(project_id):
            project, segments, source = _source_snapshot(project_id, chapter_id)
            paths, selections, snapshot = _take_snapshot(project, segments, source)
        root = safe_project_root(project_id)
        passage_id = _digest({"chapter_id": chapter_id, "source": source["metadata"]["source_utterances"]})[:32]
        relative = f"voice_lab/chapters/{chapter_id}/{snapshot['state_sha256'][:24]}"
        receipt = _path(root, relative + "/export.json")
        if receipt.is_file():
            result = json.loads(receipt.read_text("utf-8"))
            if (not _valid_receipt(root, result, snapshot["state_sha256"])
                    or result.get("selections") != selections
                    or result.get("utterance_ids") != source["source_utterance_ids"]):
                raise ChapterExportError("Immutable completed chapter export no longer matches its receipt")
        else:
            if receipt.parent.exists():
                relative += "/resume_" + uuid4().hex[:12]
                receipt = _path(root, relative + "/export.json")
            master = _path(root, relative + "/clean-master.wav")
            mp3 = _path(root, relative + "/listening.mp3")
            breaks = {b["utterance_id"]: b for b in source["breaks"]}
            speech_paths = dict(zip(source["speech_utterance_ids"], paths))
            exclusions = {row["utterance_id"]: row for row in source.get("exclusions", [])}
            ordered, mapping, cursor = [], [], 0.0
            for segment in segments:
                if segment.id in exclusions:
                    excluded = exclusions[segment.id]
                    mapping.append({"utterance_id": segment.id, "kind": "excluded", "role_id": None,
                        "source_text_sha256": excluded["source_text_sha256"],
                        "decision": excluded["decision"], "start_seconds": cursor, "end_seconds": cursor})
                    continue
                if segment.id in breaks:
                    path = _path(root, relative + f"/scene-breaks/{segment.id}.wav")
                    _pause(paths[0], path, breaks[segment.id]["duration_seconds"])
                    seconds = _decoded_duration(path)
                    row = {"utterance_id": segment.id, "kind": "scene_break", "role_id": None,
                           "source_text_sha256": breaks[segment.id]["source_text_sha256"],
                           "audio_sha256": file_sha256(path), "pause_policy": "recorded_source_separator_1.25s"}
                else:
                    path = speech_paths[segment.id]
                    row = {**snapshot["proof"][segment.id], "kind": "speech"}
                    seconds = row.pop("duration_seconds")
                row.update(start_seconds=cursor, end_seconds=cursor + seconds)
                mapping.append(row)
                ordered.append(path)
                cursor += seconds
            _assemble_pcm(ordered, master)
            duration = _decoded_duration(master)
            if abs(duration - cursor) > 0.001:
                raise ChapterExportError("Master decoded duration does not match complete source coverage")
            master_hash = file_sha256(master)
            text = "\n".join(s.text for s in segments if s.id in source["speech_utterance_ids"])
            report = qa.qa_take_audio(take_id="chapter_" + snapshot["state_sha256"][:24], audio_path=master,
                                      text=text, expected_sha256=master_hash, language="English")
            if not report.passed or report.blocking_failures() or not report.executed_check_ids():
                raise ChapterExportError("Assembled chapter fails existing blocking technical QA")
            _mp3(master, mp3)
            delivered = _decoded_duration(mp3)
            if not math.isfinite(delivered) or abs(delivered - duration) > 0.1:
                raise ChapterExportError("Decoded listening file differs from its clean natural-speed master")
            loudness = _loudness(mp3)
            if loudness is None or not math.isfinite(loudness):
                raise ChapterExportError("Listening file loudness could not be measured")
            result = {"schema": "voice-lab-chapter-review-v1", "status": "ready_for_listening",
                      "experimental_runtime": snapshot["experimental_runtime"],
                      "chapter_id": chapter_id, "chapter_title": source["chapter_title"],
                      "manifest_id": passage_id, "passage_id": passage_id,
                      "state_sha256": snapshot["state_sha256"], "selections": selections,
                      "utterance_ids": source["source_utterance_ids"], "speech_utterance_ids": source["speech_utterance_ids"],
                      "audio_file": mp3.relative_to(root).as_posix(), "audio_sha256": file_sha256(mp3),
                      "master_file": master.relative_to(root).as_posix(), "master_sha256": master_hash,
                      "duration_seconds": delivered, "master_duration_seconds": duration, "map": mapping,
                      "speech_utterance_count": len(paths), "scene_break_count": len(breaks),
                      "excluded_utterance_count": len(exclusions),
                      "rendered_voice_role_count": len({r["role_id"] for r in selections}),
                      "technical_qa": "passed", "human_review": "pending", "qa_report": report.model_dump(mode="json"),
                      "integrated_loudness_lufs": loudness,
                      "loudness_measurement": {"audio_sha256": file_sha256(mp3), "integrated_lufs": loudness, "method": "ffmpeg_ebur128"},
                      "transition_diagnostics": _transitions([r for r in mapping if r["kind"] == "speech"]),
                      "generation_totals": {"tts_generated_by_export": 0, "final_listening_seconds": delivered,
                                            "recorded_scene_pause_seconds": sum(b["duration_seconds"] for b in breaks.values())}}
            _immutable(receipt, _json(result))
        _immutable(receipt.parent / "source-to-audio-map.json", _json({"passage_id": passage_id,
                   "audio_sha256": result["audio_sha256"], "master_sha256": result["master_sha256"], "map": result["map"]}))
        with project_meta_lock(project_id), project_segments_lock(project_id):
            current, current_segments, current_source = _source_snapshot(project_id, chapter_id)
            _, _, current_snapshot = _take_snapshot(current, current_segments, current_source)
            if current_snapshot["state_sha256"] != snapshot["state_sha256"]:
                raise ChapterExportError("Newer source, cast or selected Takes superseded this chapter export")
            if not _valid_receipt(root, result, snapshot["state_sha256"]):
                raise ChapterExportError("Export bytes changed before publication")
            previous = current.voice_lab.excerpt or {}
            comparisons = {}
            from .voice_lab import digest
            for entry in [*previous.get("comparisons", []), previous, result]:
                measured = entry.get("loudness_measurement") or {}
                level = entry.get("loudness_lufs", entry.get("integrated_loudness_lufs"))
                if (entry.get("passage_id", entry.get("manifest_id")) != passage_id
                        or entry.get("utterance_ids") != result["utterance_ids"]
                        or not isinstance(level, (int, float)) or not math.isfinite(level)
                        or measured.get("audio_sha256") != entry.get("audio_sha256") or not entry.get("audio_file")):
                    continue
                path = _path(root, entry["audio_file"])
                if not path.is_file() or file_sha256(path) != entry.get("audio_sha256"):
                    continue
                key = "v_" + digest(entry["audio_sha256"])[:28]
                comparisons[key] = {"key": key, "label": f"Chapter edition {len(comparisons) + 1}",
                                    "audio_file": entry["audio_file"], "audio_sha256": entry["audio_sha256"],
                                    "passage_id": passage_id, "utterance_ids": result["utterance_ids"],
                                    "loudness_lufs": level, "loudness_measurement": measured,
                                    "map": entry["map"], "duration_seconds": entry["duration_seconds"]}
            current.voice_lab.excerpt = {**result, "comparisons": list(comparisons.values())}
            current.voice_lab.revision += 1
            save_project(current)
            return dict(current.voice_lab.excerpt)
    finally:
        release_project_export(lease)
