"""What is happening with an Auto Cast book, and its chapters as they finish.

A cast book goes through three stages a human cares about, in this order:

1. voices still waiting for the human's choice,
2. the book being recorded, chapter by chapter, and
3. the finished chapters, playable as soon as each one exists.

This module answers "where is my book?" cheaply from durable state (so a phone can
ask every few seconds), assembles each chapter from the frozen approved cast the
moment its last utterance is recorded, and applies the one explicit human decision
that can unblock a book whose lines have no speaker. It reads casting authority from
``casting_review`` and never keeps a second copy of it.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from uuid import uuid4

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role
from dubbing_service.app.storage.jobs import get_job, is_job_active
from dubbing_service.app.storage.projects import load_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments, project_segments_lock, save_segments

from . import casting_review as casting
from . import qa, release, takes
from .casting_assembly import current_frozen_binding
from .fingerprint import file_sha256
from .voice_lab_chapter_export import _pause
from .voice_lab_render import (
    _assemble_pcm,
    _decoded_duration,
    _immutable,
    _json,
    _loudness,
    _path,
)

CHAPTER_EXPORT_RE = re.compile(r"cast_[0-9a-f]{28}\Z")
SCENE_BREAK_RE = re.compile(r"\s*\*{3,}\s*\Z")
SCENE_PAUSE_SECONDS = 1.25
LISTENING_CEILING = 0.668   # linear, about -3.5 dBFS: the delivered file must decode at or under -3 dBFS
DECISION_MODES = ("suggested", "narrator", "skip", "auto")
POLICY_KEY = "attribution_policy"
SKIP_LABELS = {
    casting.SKIP_UNKNOWN_SPEAKER: "No speaker could be found",
    casting.SKIP_UNSPEAKABLE: "The voice model would not speak it",
    casting.SKIP_WEB_ADDRESS: "Web addresses, which are not read aloud",
    casting.SKIP_FAILED_QA: "The recording failed its quality checks",
    casting.SKIP_HUMAN: "Skipped by you",
}
SKIP_LINES_SHOWN = 25
ATTRIBUTION_LIMIT = 1200
ATTRIBUTION_NOTE = "Attributed from a reading of the surrounding story, at your request"
PACE_SCHEMA = "autocast-production-pace-v1"
PACE_WINDOW = 40          # recent lines the estimate is drawn from
PACE_MIN_LINES = 3        # fewer than this is a guess, so no time is promised


class BookError(casting.CastingError):
    """A chapter cannot be assembled or a decision cannot be applied."""


# --------------------------------------------------------------------------
# How fast the book is being recorded
# --------------------------------------------------------------------------
#
# The worker records how long each synthesized line really took, next to its length,
# in a small file beside the project rather than in ``project.json`` (a rewrite of
# which costs more than a line of speech). Time left is that measured pace applied to
# what is left; it is never a fixed guess, and with too few lines to measure it says so.

def _pace_path(project_id):
    return _path(safe_project_root(project_id), "casting/production_pace.json")


def _write_pace(project_id, pace):
    path = _pace_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    staged.write_text(json.dumps(pace, allow_nan=False), "utf-8")
    os.replace(staged, path)


def read_pace(project_id, request_id):
    """The measured pace of this production's current run, or ``None``."""
    try:
        pace = json.loads(_pace_path(project_id).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(pace, dict) or pace.get("schema") != PACE_SCHEMA or pace.get("request_id") != request_id:
        return None
    lines = [row for row in pace.get("lines") or []
             if isinstance(row, dict) and all(isinstance(row.get(key), (int, float)) for key in ("characters", "seconds", "at"))]
    return {**pace, "lines": lines}


def start_pace(project_id, request_id):
    """A new run measures its own pace: a model reload or a different machine load is not the last run's."""
    try:
        _write_pace(project_id, {"schema": PACE_SCHEMA, "request_id": request_id, "started_at": time.time(), "lines": []})
    except OSError:
        pass  # an estimate is never worth failing a render


def record_pace(project_id, request_id, *, characters, seconds):
    try:
        pace = read_pace(project_id, request_id) or {"schema": PACE_SCHEMA, "request_id": request_id,
                                                     "started_at": time.time(), "lines": []}
        row = {"characters": int(characters), "seconds": round(float(seconds), 3), "at": time.time()}
        _write_pace(project_id, {**pace, "lines": [*pace["lines"], row][-PACE_WINDOW:]})
    except (OSError, ValueError, TypeError):
        pass


def seconds_per_character(pace):
    lines = [row for row in (pace or {}).get("lines", []) if row["characters"] > 0 and row["seconds"] > 0]
    if len(lines) < PACE_MIN_LINES:
        return None
    return sum(row["seconds"] for row in lines) / sum(row["characters"] for row in lines)


def estimate_seconds(remaining, pace, now=None):
    """Seconds until the lines in ``remaining`` (their lengths, in reading order) are recorded.

    ``None`` when the pace has not been measured yet. The line being recorded right now
    started when the previous one ended, so the time it has already taken is not owed again.
    """
    rate = seconds_per_character(pace)
    if rate is None:
        return None
    if not remaining:
        return 0
    now = time.time() if now is None else now
    total = sum(remaining) * rate
    already = min(max(0.0, now - pace["lines"][-1]["at"]), remaining[0] * rate)
    return max(0, int(round(total - already)))


# --------------------------------------------------------------------------
# Reading the book's state
# --------------------------------------------------------------------------

def _production_receipt(state):
    """The production this book is following: the automatic one, else the newest."""
    productions = state.get("productions") or {}
    receipt = productions.get(casting.AUTOMATIC_PRODUCTION_ID)
    if receipt:
        return receipt
    return max(productions.values(), key=lambda row: str(row.get("created_at") or ""), default=None)


def chapter_plan(project, utterance_ids, segments=None):
    """The chapters a production covers, in reading order, with their utterances."""
    segments = segments if segments is not None else load_segments(project.id)
    chapter_of = {segment.id: segment.chapter_id for segment in segments}
    order = {chapter.id: chapter for chapter in project.chapters}
    grouped: dict[str, list[str]] = {}
    for utterance_id in utterance_ids:
        chapter_id = chapter_of.get(utterance_id)
        if chapter_id in order:
            grouped.setdefault(chapter_id, []).append(utterance_id)
    ranked = sorted(grouped, key=lambda cid: (order[cid].index, cid))
    return [{"chapter_id": cid, "title": order[cid].title, "index": order[cid].index,
             "utterance_ids": grouped[cid]} for cid in ranked]


def _waiting_voices(project, state):
    """Characters whose playable auditions are waiting for the human's choice."""
    waiting, generating = [], 0
    for role_id, item in state["roles"].items():
        round_ = item.get("round") or {}
        status = round_.get("status")
        if status not in {"awaiting_review", "pending", "running"} or casting.approved(project, role_id):
            continue
        ready = [c for c in round_.get("candidates", []) if c.get("status") == "ready" and c.get("audio_sha256")]
        if ready:
            waiting.append({"role_id": role_id, "name": casting._role_name(project, role_id),
                            "options": len(ready), "replacement": bool(round_.get("replacement"))})
        elif status in {"pending", "running"}:
            generating += 1
    return waiting, generating


def _suggested_role(project, segment, held):
    """The one real character the analyzer proposed for a held line, or ``None``.

    A placeholder is not a suggestion. The analyzer labels lines it could not place
    "unknown", and a project can even hold a placeholder role of that name, so a label
    that merely *resolves* to a role is not enough: it must name a real character.
    Accepting placeholders would hand those lines to a voice nobody chose.
    """
    from .character_identity import SPEAKER_SENTINELS, role_id_for_label
    label = segment.attribution.label if segment.attribution else None
    if segment.id not in held or not casting._clean_label(label):
        return None
    if label.strip().casefold() in SPEAKER_SENTINELS:
        return None
    role_id = role_id_for_label(project, label)
    role = next((row for row in project.roles if row.id == role_id), None)
    if role is None or role.role_type != RoleType.CHARACTER or (role.name or "").strip().casefold() in SPEAKER_SENTINELS:
        return None
    return role_id


def _block_report(project):
    """Why lines cannot be recorded yet, and which explicit decisions would unblock them."""
    try:
        _, rows, _, _ = casting._scope(project)
    except casting.CastingError:
        return None
    unresolved, with_proposal = casting.unresolved_scope(project, rows)
    if not unresolved:
        return None
    held = casting._held_ids(project)
    mappable = sum(1 for segment in unresolved if _suggested_role(project, segment, held))
    narrator = _narrator(project)
    return {"unresolved": len(unresolved), "suggested": len(with_proposal), "suggested_mappable": mappable,
            "without_evidence": len(unresolved) - len(with_proposal),
            "narrator": narrator.name if narrator else None,
            "actions": [name for name, ok in (("auto", mappable > 0), ("suggested", mappable > 0),
                                              ("narrator", narrator is not None), ("skip", True)) if ok]}


def _eta(receipt, plan, current, done_ids, segments, project_id):
    """Time left for the chapter being recorded and for the whole book, from measured pace."""
    length = {segment.id: len(segment.text) for segment in segments}
    pace = read_pace(project_id, receipt.get("request_id"))
    entry = next((item for item in plan if current and item["chapter_id"] == current["chapter_id"]), None)
    in_chapter = [length.get(uid, 0) for uid in (entry["utterance_ids"] if entry else []) if uid not in done_ids]
    in_book = [length.get(uid, 0) for uid in (receipt.get("utterance_ids") or []) if uid not in done_ids]
    return {"chapter_seconds": estimate_seconds(in_chapter, pace),
            "book_seconds": estimate_seconds(in_book, pace),
            "measured_lines": len((pace or {}).get("lines") or [])}


def _snippet(text, limit=120):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _skipped_summary(project, left_out, segments):
    """The lines left out of the recording and why, grouped, with a short preview of each."""
    if not left_out:
        return None
    order = {segment.id: (index, segment) for index, segment in enumerate(segments or [])}
    chapters = {chapter.id: chapter.title for chapter in project.chapters}
    groups = []
    for reason in casting.SKIP_REASONS:
        ids = sorted((uid for uid, entry in left_out.items() if entry.get("reason") == reason),
                     key=lambda uid: order.get(uid, (10 ** 9,))[0])
        if not ids:
            continue
        shown = []
        for uid in ids[:SKIP_LINES_SHOWN]:
            segment = order.get(uid, (0, None))[1]
            shown.append({"utterance_id": uid, "text": _snippet(segment.text if segment else ""),
                          "chapter": chapters.get(getattr(segment, "chapter_id", None)),
                          "detail": _snippet(left_out[uid].get("detail"), 160) or None})
        groups.append({"reason": reason, "label": SKIP_LABELS[reason], "count": len(ids), "lines": shown,
                       "more": max(0, len(ids) - SKIP_LINES_SHOWN)})
    return {"total": len(left_out), "groups": groups}


def _failed_line(project, receipt, left_out, segments):
    """The line a stopped run could not record, so that a person can skip it."""
    failure = (receipt or {}).get("last_line_failure")
    if not failure or (receipt or {}).get("status") in {"succeeded", "retracted"}:
        return None
    uid = failure.get("utterance_id")
    if not uid or uid in left_out:
        return None
    segment = next((s for s in (segments if segments is not None else load_segments(project.id)) if s.id == uid), None)
    if segment is None:
        return None
    chapter = next((c.title for c in project.chapters if c.id == segment.chapter_id), None)
    return {"utterance_id": uid, "text": _snippet(segment.text, 200), "chapter": chapter,
            "error": _snippet(failure.get("error"), 240)}


def _chapter_rows(project, receipt, plan, done_ids, running, left_out=None):
    records = (receipt.get("chapters") or {}) if receipt else {}
    left_out = left_out or {}
    rows, current_marked = [], False
    for entry in plan:
        cid = entry["chapter_id"]
        record = records.get(cid) or {}
        lines_done = sum(1 for uid in entry["utterance_ids"] if uid in done_ids)
        lines_total = len(entry["utterance_ids"])
        ready = record.get("status") == "ready" and bool(record.get("audio_sha256"))
        if ready:
            status = "ready"
        elif running and not current_marked and lines_done < lines_total:
            status, current_marked = "rendering", True
        elif record.get("status") == "failed":
            status = "failed"
        else:
            status = "waiting"
        row = {"chapter_id": cid, "title": entry["title"], "number": len(rows) + 1, "status": status,
               "lines_done": lines_done, "lines_total": lines_total}
        if left_out.get(cid):
            row["lines_skipped"] = left_out[cid]
        if ready:
            base = f"/api/v1/projects/{project.id}/voice-lab/casting/chapters/{record['export_id']}/audio"
            row.update(export_id=record["export_id"], duration_seconds=record["duration_seconds"],
                       audio_sha256=record["audio_sha256"], audio_url=base, download_url=base + "?download=true")
        elif status == "failed":
            row["reason"] = str(record.get("reason") or "")[:300]
        rows.append(row)
    return rows


def book_status(project_id):
    """One small, fast answer to "where is my book?" for the page a push opens.

    It is derived from the durable casting state, the production receipt and its job,
    so it costs a project read rather than a casting review, and it never starts work.
    """
    project = load_project(project_id)
    state = casting._state(project)
    required = casting._required_role_ids(project)
    waiting, generating = _waiting_voices(project, state)
    approved_count = sum(1 for role_id in required if casting.approved(project, role_id))
    receipt = _production_receipt(state)
    production, chapters, block, segments = None, [], None, None
    left_out = casting.skipped_lines(project)
    if receipt and receipt.get("status") != "retracted":
        job = get_job(project_id, receipt["job_id"]) if receipt.get("job_id") else None
        ids = receipt.get("utterance_ids") or []
        segments = load_segments(project_id)
        done_ids = set(job.completed_segment_ids) if job else set()
        per_chapter = {}
        for segment in segments:
            if segment.id in left_out and segment.chapter_id:
                per_chapter[segment.chapter_id] = per_chapter.get(segment.chapter_id, 0) + 1
        status = receipt.get("status")
        # The worker owns the job until it releases it, which is after the last chapter
        # is assembled, so "active" (not the job's own status) is what says it is running.
        running = bool(job and is_job_active(project_id, receipt["job_id"]))
        plan = chapter_plan(project, ids, segments)
        chapters = _chapter_rows(project, receipt, plan, done_ids, running or status == "queued", per_chapter)
        ready_count = sum(1 for row in chapters if row["status"] == "ready")
        finished = status == "succeeded" and ready_count == len(chapters)
        if status == "succeeded":
            done_ids = set(ids)
        if finished:
            phase = "finished"
        elif status == "succeeded":
            phase = "exporting"
        elif status == "blocked":
            phase = "blocked"
        elif running or status == "queued":
            phase = "running"
        else:
            phase = "waiting"
        if phase in {"blocked", "waiting"}:
            block = _block_report(project)
        current = next((row for row in chapters if row["status"] == "rendering"), None)
        production = {"request_id": receipt.get("request_id"), "status": status, "phase": phase,
                      "reason": (str(receipt.get("reason") or "")[:600] or None),
                      "lines_done": len(done_ids & set(ids)) if status != "succeeded" else len(ids),
                      "lines_total": len(ids), "chapters_ready": ready_count, "chapters_total": len(chapters),
                      "current_chapter": current["title"] if current else None, "eta": None}
        if phase == "running" and running:
            production["eta"] = _eta(receipt, plan, current, done_ids, segments, project_id)
            if current and production["eta"]:
                current["eta_seconds"] = production["eta"]["chapter_seconds"]
    if waiting:
        overall = "voices_waiting"
    elif production and production["phase"] in {"running", "exporting"}:
        overall = "processing"
    elif production and production["phase"] == "finished":
        overall = "finished"
    elif production:
        overall = "blocked" if production["phase"] == "blocked" else "waiting"
    elif generating:
        overall = "casting"
    else:
        overall = "starting" if required and approved_count == len(required) else "idle"
    if left_out and segments is None:
        segments = load_segments(project_id)
    return {"schema": "casting-book-status-v1", "project_id": project.id, "project_name": project.name,
            "state": overall,
            "skipped": _skipped_summary(project, left_out, segments),
            "automatic": automatic_handling(project),
            "failed_line": _failed_line(project, receipt, left_out, segments),
            "voices": {"waiting": waiting, "generating": generating, "approved": approved_count,
                       "total": len(required)},
            "production": production, "chapters": chapters, "block": block}


# --------------------------------------------------------------------------
# Chapters, assembled as soon as their last utterance is recorded
# --------------------------------------------------------------------------

def chapter_artifact(project_id, export_id):
    """The verified file behind one published chapter, and its display name."""
    if not CHAPTER_EXPORT_RE.fullmatch(export_id or ""):
        raise BookError("Invalid chapter recording identity")
    project = load_project(project_id)
    for receipt in (casting._state(project).get("productions") or {}).values():
        for record in (receipt.get("chapters") or {}).values():
            if record.get("export_id") == export_id and record.get("status") == "ready":
                path = _path(safe_project_root(project_id), record["audio_file"])
                if path.suffix.lower() != ".mp3":
                    raise BookError("Chapter recording is not an MP3")
                return path, record["audio_sha256"], record.get("title") or "Chapter", record.get("number", 1)
    raise BookError("This chapter recording is not available")


def _listening_mp3(master, destination):
    """Encode the clean master as an audiobook-platform MP3 (ACX: 44.1 kHz, 192 kbps CBR, peaks <= -3 dBFS).

    Only this delivered copy is touched; the master stays the untouched lossless join of the
    approved takes. The limiter ceiling sits a little under -3 dBFS so the MP3's own overshoot
    cannot push the decoded peak past it, and it does not act on speech below that level, so
    loudness is unchanged.
    """
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(master), "-vn",
                    "-af", f"aresample=resampler=soxr:precision=28,alimiter=limit={LISTENING_CEILING}:attack=5:release=60:level=disabled",
                    "-ac", "1", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", "192k", str(destination)],
                   check=True, capture_output=True, timeout=300)


def _chapter_segments(project_id, chapter_id):
    from .release import is_live
    segments = [s for s in load_segments(project_id) if s.chapter_id == chapter_id and is_live(s)]
    return sorted(segments, key=lambda s: s.index)


def export_chapter(project_id, request_id, chapter_id, number=None):
    """Assemble one finished chapter into a listening MP3 from the frozen approved cast.

    Nothing is synthesized here: it joins the QA-passing Takes the strict production
    already selected, proves each still belongs to the current approved cast, and
    publishes an immutable, hash-bound file. Calling it again for unchanged audio
    returns the same recording.
    """
    project = load_project(project_id)
    rows = _chapter_segments(project_id, chapter_id)
    scene_breaks = set(((project.metadata or {}).get("book_creator") or {}).get("scene_breaks", []))
    left_out = casting.skipped_ids(project)
    speech = [s for s in rows if s.id not in scene_breaks and s.id not in left_out and s.text.strip()
              and not SCENE_BREAK_RE.fullmatch(s.text)]
    if not speech:
        raise BookError("This chapter has no spoken utterances to assemble")
    root = safe_project_root(project_id)
    coverage = release.check_coverage(speech, project_root=root, verify_bytes=True, require_qa=True)
    if not coverage.ready:
        raise BookError("Selected chapter audio has missing, stale or QA-failing Takes")
    binding = current_frozen_binding(project, speech)
    if binding is None:
        raise BookError("A chapter can only be published from a frozen, approved cast")
    export_id = "cast_" + casting.lab.digest({"chapter": chapter_id, "manifest": binding["manifest_id"],
                                              "selections": binding["selections"]})[:28]
    relative = f"autocast/chapters/{export_id}"
    receipt_path = _path(root, relative + "/receipt.json")
    speech_ids = {s.id for s in speech}
    title = next((c.title for c in project.chapters if c.id == chapter_id), "Chapter")
    if receipt_path.is_file():
        import json
        result = json.loads(receipt_path.read_text("utf-8"))
        if any(file_sha256(_path(root, result[k])) != result[h]
               for k, h in (("audio_file", "audio_sha256"), ("master_file", "master_sha256"))):
            raise BookError("A completed chapter recording no longer matches its receipt")
    else:
        if receipt_path.parent.exists():
            relative += "/resume_" + casting.lab.digest([export_id, casting.lab.now()])[:12]
            receipt_path = _path(root, relative + "/receipt.json")
        by_id = {s.id: s for s in speech}
        first_audio = _path(root, takes.selected_take(speech[0]).audio_file)
        cursor, mapping, audio_paths = 0.0, [], []
        for segment in rows:
            if segment.id in speech_ids:
                take = takes.selected_take(by_id[segment.id])
                audio = _path(root, take.audio_file)
                seconds = _decoded_duration(audio)
                entry = {"utterance_id": segment.id, "kind": "speech", "role_id": take.role_id,
                         "take_id": take.id, "audio_sha256": take.audio_sha256}
            elif segment.id in scene_breaks or SCENE_BREAK_RE.fullmatch(segment.text or ""):
                audio = _path(root, relative + f"/pauses/{segment.id}.wav")
                _pause(first_audio, audio, SCENE_PAUSE_SECONDS)
                seconds = _decoded_duration(audio)
                entry = {"utterance_id": segment.id, "kind": "scene_break", "role_id": None}
            else:
                continue
            entry.update(start_seconds=cursor, end_seconds=cursor + seconds)
            cursor += seconds
            mapping.append(entry)
            audio_paths.append(audio)
        master = _path(root, relative + "/clean-master.wav")
        mp3 = _path(root, relative + "/listening.mp3")
        _assemble_pcm(audio_paths, master)
        duration = _decoded_duration(master)
        if abs(duration - cursor) > 0.001:
            raise BookError("Assembled chapter duration differs from complete source coverage")
        master_hash = file_sha256(master)
        report = qa.qa_take_audio(take_id=export_id, audio_path=master,
                                  text="\n".join(s.text for s in speech), expected_sha256=master_hash,
                                  language=(project.metadata or {}).get("source_language") or "English")
        if not report.passed or report.blocking_failures() or not report.executed_check_ids():
            raise BookError("Assembled chapter did not pass blocking technical QA")
        _listening_mp3(master, mp3)
        delivered = _decoded_duration(mp3)
        loudness = _loudness(mp3)
        if abs(delivered - duration) > 0.1 or loudness is None or not math.isfinite(loudness):
            raise BookError("Chapter MP3 decoding or loudness measurement failed")
        result = {"export_id": export_id, "chapter_id": chapter_id, "label": title,
                  "audio_file": mp3.relative_to(root).as_posix(), "audio_sha256": file_sha256(mp3),
                  "master_file": master.relative_to(root).as_posix(), "master_sha256": master_hash,
                  "duration_seconds": delivered, "integrated_loudness_lufs": loudness,
                  "manifest_id": binding["manifest_id"], "selections": binding["selections"],
                  "map": mapping, "technical_qa": "passed", "human_review": "pending",
                  "qa_report": report.model_dump(mode="json")}
        _immutable(receipt_path, _json(result))
    record = {"status": "ready", "chapter_id": chapter_id, "title": title, "number": number,
              "export_id": export_id, "audio_file": result["audio_file"], "audio_sha256": result["audio_sha256"],
              "duration_seconds": result["duration_seconds"], "utterance_count": len(speech),
              "manifest_id": binding["manifest_id"], "technical_qa": "passed", "created_at": casting.lab.now()}
    with project_meta_lock(project_id):
        current = load_project(project_id)
        again = current_frozen_binding(current, [s for s in _chapter_segments(project_id, chapter_id) if s.id in speech_ids])
        if again != binding:
            raise BookError("Source, approved cast or selected Takes changed while this chapter was assembled")
        production = casting._state(current)["productions"][request_id]
        chapters = production.setdefault("chapters", {})
        previous = chapters.get(chapter_id) or {}
        if previous.get("export_id") == export_id and previous.get("status") == "ready":
            return previous
        chapters[chapter_id] = record
        casting._save(current)
    return record


def record_chapter_failure(project_id, request_id, chapter_id, reason):
    """Remember a chapter that could not be assembled, without failing the whole book."""
    with project_meta_lock(project_id):
        project = load_project(project_id)
        chapters = casting._state(project)["productions"][request_id].setdefault("chapters", {})
        if (chapters.get(chapter_id) or {}).get("status") != "ready":
            chapters[chapter_id] = {"status": "failed", "chapter_id": chapter_id,
                                    "reason": str(reason)[:400], "failed_at": casting.lab.now()}
            casting._save(project)


# --------------------------------------------------------------------------
# The one explicit decision that unblocks lines with no speaker
# --------------------------------------------------------------------------

def _narrator(project):
    narrators = [role for role in project.roles if role.role_type == RoleType.NARRATOR]
    if len(narrators) == 1:
        return narrators[0]
    named = [role for role in project.roles if (role.name or "").strip().casefold() == "narrator"]
    return named[0] if len(named) == 1 else None


def _open_rounds_for_new_characters(project):
    """A decision can make a character part of the book that had no voice to choose yet."""
    try:
        _, _, required, _ = casting._scope(project)
    except casting.CastingError:
        return
    for role_id in required:
        if not casting.approved(project, role_id) and not casting._state(project)["roles"].get(role_id, {}).get("round"):
            casting._new_round(project, role_id)


def _refuse_an_empty_book(project, leaving_out):
    """A decision that would leave nothing to record is refused, not saved half-way."""
    wanted, _, _, _ = casting._scope(project)
    if not set(wanted) - set(leaving_out):
        raise BookError("That would leave nothing in this book to record")


def _recording_now(project_id, project):
    receipt = _production_receipt(casting._state(project))
    return bool(receipt and receipt.get("job_id") and is_job_active(project_id, receipt["job_id"]))


def resolve_unattributed(project_id, *, request_id, mode, by="human", exclude=()):
    """Apply a decision to lines that have no resolved speaker, then restart the book.

    ``suggested`` accepts the analyzer's proposed speaker where it names one real
    character. ``narrator`` reads every unresolved line in the narrator's voice.
    ``skip`` leaves every unresolved line out of the recording, and ``auto`` accepts the
    suggested speakers and skips whatever is still unresolved. Each is explicit, recorded as
    the person's own decision (``by`` says whether they made it now or by their standing
    automatic-handling setting), and idempotent for a given ``request_id``. The narrator is
    never a fallback: nothing here happens unless someone chose it.

    A production that was blocked on these lines is re-planned in the same step, because a
    blocked production is otherwise never looked at again.
    """
    from .provenance import human_correction
    casting._id(request_id)
    if mode not in DECISION_MODES:
        raise BookError("Choose to accept the suggested speakers, read the rest with the narrator, or skip them")
    accepting, skipping = mode in {"suggested", "auto"}, mode in {"skip", "auto"}
    notes = {"suggested": "the suggested speaker was accepted", "auto": "the suggested speaker was accepted",
             "narrator": "read by the narrator"}
    with project_meta_lock(project_id), project_segments_lock(project_id):
        project = load_project(project_id)
        decisions = (project.metadata.setdefault("autocast_import", {})).setdefault("attribution_decisions", {})
        if request_id in decisions:
            return dict(decisions[request_id])
        _, rows, _, _ = casting._scope(project)
        unresolved, _ = casting.unresolved_scope(project, rows)
        unresolved = [row for row in unresolved if row.id not in set(exclude)]
        if not unresolved:
            raise BookError("Every line in scope already has a speaker")
        narrator = _narrator(project) if mode == "narrator" else None
        if mode == "narrator" and narrator is None:
            raise BookError("This book has no single narrator character to read the remaining lines")
        held = casting._held_ids(project)
        segments = load_segments(project_id)
        by_id = {segment.id: segment for segment in segments}
        changed, left = [], []
        for row in unresolved:
            segment = by_id[row.id]
            label = segment.attribution.label if segment.attribution else None
            role_id = _suggested_role(project, segment, held) if accepting else narrator.id if narrator else None
            if role_id:
                attribution = human_correction(
                    segment, speaker_id=role_id,
                    note=f"Decided in Voice Lab{' by your automatic-handling setting' if by == 'policy' else ''}: {notes[mode]}")
                attribution.label = label
                segment.role_id = role_id
                changed.append(segment.id)
            elif skipping:
                left.append(segment.id)
        if not changed and not left:
            raise BookError("None of these lines has a real suggested speaker; read them with the narrator, skip them, or attribute them in Studio")
        if left:
            _refuse_an_empty_book(project, left)
        if changed:
            save_segments(project_id, segments)
        imported = project.metadata["autocast_import"]
        gone = set(changed)
        imported["unresolved_utterance_ids"] = [uid for uid in (imported.get("unresolved_utterance_ids") or []) if uid not in gone]
        imported["reason"] = (f"{len(imported['unresolved_utterance_ids'])} dialogue utterance(s) need attribution review"
                              if imported["unresolved_utterance_ids"] else None)
        if left:
            casting.record_skips(project, left, reason=casting.SKIP_UNKNOWN_SPEAKER, by=by,
                                 detail="No speaker could be found for this line")
        outcome = {"request_id": request_id, "mode": mode, "by": by, "resolved": len(changed), "skipped": len(left),
                   "still_unresolved": len(unresolved) - len(changed) - len(left), "decided_at": casting.lab.now()}
        decisions[request_id] = outcome
        _open_rounds_for_new_characters(project)
        casting.replan_automatic(project, reason="Your decision was saved; recording resumes.")
        casting._save(project)
        return dict(outcome)


def _character_named(project, name):
    """The one character with this name, or ``None``. A placeholder is never a character."""
    from .character_identity import SPEAKER_SENTINELS
    key = " ".join(str(name).split()).casefold()
    if not key or key in SPEAKER_SENTINELS:
        raise BookError(f"“{name}” is not a character")
    matches = [role for role in project.roles
               if role.role_type == RoleType.CHARACTER and role.name.strip().casefold() == key]
    if len(matches) > 1:
        raise BookError(f"More than one character is named “{name}”")
    return matches[0] if matches else None


def attribute_lines(project_id, *, request_id, assignments, new_roles=(), aliases=None, note=None):
    """Attribute the speaker of lines from a reading of the story, then restart the book.

    ``assignments`` maps a line to the character who says it. Each is recorded as the person's
    own attribution (they asked for it), replacing whatever guess the analyzer made, and
    only ever for spoken lines: narration is never handed to a character. ``new_roles``
    adds characters the analyzer never found; they get a dossier and a casting profile from
    the story and a casting round, so the book waits for their voices to be chosen and
    starts recording by itself once they are. ``aliases`` merges a duplicate character
    ("The Librarian") into the one it really is ("Filaurel"), and is applied after the
    profiles are refreshed because that refresh recomputes them. Idempotent per request id.
    """
    from .adapter import _role_id
    from .character_casting import research_and_profile
    from .provenance import human_correction
    casting._id(request_id)
    if not assignments or len(assignments) > ATTRIBUTION_LIMIT:
        raise BookError(f"Attribute between one and {ATTRIBUTION_LIMIT} lines at a time")
    note = " ".join(str(note or ATTRIBUTION_NOTE).split())[:200]
    with project_meta_lock(project_id), project_segments_lock(project_id):
        project = load_project(project_id)
        imported = project.metadata.setdefault("autocast_import", {})
        decisions = imported.setdefault("attribution_decisions", {})
        if request_id in decisions:
            return dict(decisions[request_id])
        if _recording_now(project_id, project):
            raise BookError("The book is being recorded right now; attribute speakers when it stops")
        backend = next((role.tts_backend for role in project.roles
                        if role.role_type == RoleType.CHARACTER and role.tts_backend), "qwen3tts")
        created = []
        for name in dict.fromkeys(new_roles or ()):
            if _character_named(project, name) is None:
                project.roles.append(Role(id=_role_id(name.strip()), name=" ".join(name.split()),
                                          role_type=RoleType.CHARACTER, tts_backend=backend))
                created.append(" ".join(name.split()))
        segments = load_segments(project_id)
        by_id = {segment.id: segment for segment in segments}
        targets = {}
        for uid, name in assignments.items():
            if uid not in by_id:
                raise BookError("A line to attribute is not in this book")
            role = _character_named(project, name)
            if role is None:
                raise BookError(f"“{name}” is not a character in this book; name them in new_roles to add them")
            targets[uid] = role
        changed = []
        for uid, role in targets.items():
            segment = by_id[uid]
            current = next((row for row in project.roles if row.id == casting._canonical(project, segment.role_id)), None)
            if current is not None and current.role_type == RoleType.NARRATOR:
                raise BookError("Narration is not attributed to a character")
            if current is not None and current.id == role.id:
                continue
            label = segment.attribution.label if segment.attribution else None
            human_correction(segment, speaker_id=role.id, note=note).label = label
            segment.role_id = role.id
            changed.append(uid)
        if changed:
            save_segments(project_id, segments)
        gone = set(targets)
        imported["unresolved_utterance_ids"] = [uid for uid in imported.get("unresolved_utterance_ids") or [] if uid not in gone]
        imported["reason"] = (f"{len(imported['unresolved_utterance_ids'])} dialogue utterance(s) need attribution review"
                              if imported["unresolved_utterance_ids"] else None)
        for uid in gone:
            (imported.get("skipped") or {}).pop(uid, None)
        # Dossiers and casting profiles come from the story, so they are refreshed for the
        # whole cast; that recomputes the aliases, so ours are set afterwards.
        research_and_profile(project, segments)
        merged = {}
        for source, target in (aliases or {}).items():
            source_role = next((row for row in project.roles if row.name.strip().casefold() == " ".join(str(source).split()).casefold()), None)
            target_role = _character_named(project, target)
            if source_role is None or target_role is None or source_role.id == target_role.id:
                raise BookError(f"Cannot merge “{source}” into “{target}”")
            project.character_cast.role_aliases[source_role.id] = target_role.id
            merged[source_role.name] = target_role.name
        _open_rounds_for_new_characters(project)
        run = project.character_cast.autocast
        if created and run is not None:
            # The casting run knows which characters it is casting; a character added after it
            # finished has to be part of it, or the progress, the "voices ready" pushes and the
            # completion event would all describe a cast that no longer exists.
            try:
                _, _, current_cast, _ = casting._scope(project)
            except casting.CastingError:
                current_cast = []
            run.required_role_ids = list(current_cast)
            run.queued_role_ids = [rid for rid in current_cast if (casting._state(project)["roles"].get(rid) or {}).get("round")]
            if run.status in {"complete", "partial"}:
                run.status, run.completed_at = "running", None
                run.reason = f"{len(created)} new character(s) are being cast: {', '.join(created)}"
        outcome = {"request_id": request_id, "mode": "attribute", "by": "you", "attributed": len(changed),
                   "unchanged": len(targets) - len(changed), "new_characters": created, "merged": merged,
                   "decided_at": casting.lab.now()}
        decisions[request_id] = outcome
        casting.replan_automatic(project, reason="Speakers were attributed; recording resumes once every voice is chosen.")
        casting._save(project)
        return dict(outcome)


def automatic_handling(project):
    """Whether lines that would stop the book are decided as the person said to decide them."""
    return bool(((project.metadata.get("autocast_import") or {}).get(POLICY_KEY) or {}).get("enabled"))


def set_automatic_handling(project_id, enabled):
    """Turn automatic handling on or off: accept real suggested speakers, skip lines with no
    speaker, and skip lines the voice model will not speak, rather than wait for a click."""
    with project_meta_lock(project_id):
        project = load_project(project_id)
        imported = project.metadata.get("autocast_import")
        if not imported or imported.get("mode") != "autocast":
            raise BookError("Automatic handling applies to Auto Cast books")
        imported[POLICY_KEY] = {"enabled": bool(enabled), "set_by": "you", "at": casting.lab.now()}
        casting._save(project)
    return {"enabled": bool(enabled)}


def apply_policy(project_id, project=None):
    """Decide the lines that would stop the book, if the person turned automatic handling on.

    Run by the maintenance pass before it looks at the production, so a book never reaches
    "needs attention" (and never sends that push) for lines the person already said how to
    handle. Lines a person restored are theirs to decide and are left alone.
    """
    project = project or load_project(project_id)
    if not automatic_handling(project):
        return None
    imported = project.metadata.get("autocast_import") or {}
    exempt = set(imported.get("policy_exempt") or [])
    receipt = _production_receipt(casting._state(project))
    pending = casting._held_ids(project) - exempt - casting.skipped_ids(project)
    if not pending and not (receipt and receipt.get("status") == "blocked"):
        return None
    try:
        _, rows, _, _ = casting._scope(project)
    except casting.CastingError:
        return None
    unresolved, _ = casting.unresolved_scope(project, rows)
    todo = sorted(row.id for row in unresolved if row.id not in exempt)
    if not todo:
        return None
    try:
        return resolve_unattributed(project_id, request_id="policy-" + casting.lab.digest(todo)[:24],
                                    mode="auto", by="policy", exclude=exempt)
    except BookError:
        return None


def skip_lines(project_id, *, request_id, utterance_ids):
    """Leave chosen lines out of the recording. They can be brought back."""
    casting._id(request_id)
    ids = list(dict.fromkeys(utterance_ids))
    if not ids or len(ids) > 100:
        raise BookError("Choose between one and 100 lines to skip")
    with project_meta_lock(project_id):
        project = load_project(project_id)
        decisions = (project.metadata.setdefault("autocast_import", {})).setdefault("attribution_decisions", {})
        if request_id in decisions:
            return dict(decisions[request_id])
        if _recording_now(project_id, project):
            raise BookError("The book is being recorded right now. It skips lines the voice model refuses by itself; you can skip more when it stops")
        wanted, _, _, _ = casting._scope(project)
        by_id = {segment.id: segment for segment in load_segments(project_id)}
        for uid in ids:
            if uid not in wanted:
                raise BookError("That line is not part of this book's recording")
            if by_id[uid].selected_take_id:
                raise BookError("That line is already recorded; skipping it now would change a chapter you can already hear")
        _refuse_an_empty_book(project, ids)
        casting.record_skips(project, ids, reason=casting.SKIP_HUMAN, by="you", detail="Skipped by you")
        outcome = {"request_id": request_id, "mode": "skip_lines", "by": "you", "skipped": len(ids), "decided_at": casting.lab.now()}
        decisions[request_id] = outcome
        casting.replan_automatic(project, reason="Lines were skipped; recording resumes.")
        casting._save(project)
        return dict(outcome)


def restore_lines(project_id, *, request_id, reason=None, utterance_ids=None):
    """Bring skipped lines back into the recording, as a group or one by one."""
    casting._id(request_id)
    if not reason and not utterance_ids:
        raise BookError("Choose which skipped lines to bring back")
    with project_meta_lock(project_id):
        project = load_project(project_id)
        decisions = (project.metadata.setdefault("autocast_import", {})).setdefault("attribution_decisions", {})
        if request_id in decisions:
            return dict(decisions[request_id])
        if _recording_now(project_id, project):
            raise BookError("The book is being recorded right now; bring lines back when it stops")
        skipped = casting.skipped_lines(project)
        ids = ([uid for uid in utterance_ids if uid in skipped] if utterance_ids
               else [uid for uid, entry in skipped.items() if entry.get("reason") == reason])
        if not ids:
            raise BookError("Those lines are not skipped")
        casting.restore_skips(project, ids)
        outcome = {"request_id": request_id, "mode": "restore", "by": "you", "restored": len(ids), "decided_at": casting.lab.now()}
        decisions[request_id] = outcome
        _open_rounds_for_new_characters(project)
        casting.replan_automatic(project, reason="Skipped lines were brought back; recording resumes.", add=ids, reopen=True)
        casting._save(project)
        return dict(outcome)


def chapter_progress_line(record, ready, total):
    """A short, honest line for a chapter notification (minutes are display only)."""
    minutes = max(1, math.floor(float(record.get("duration_seconds") or 0) / 60 + 0.5))
    return f"{minutes} min · {ready} of {total} chapters ready"
