"""Scoped human casting decisions and durable work intents in Studio project.json.

No second store or queue: a single atomic project save accepts the human event,
assignment decision and dispatch intent. Existing synthesis jobs own execution.
Read/poll operations never generate sound. Legacy library ratings stay legacy.
"""
from __future__ import annotations

import copy
import ipaddress
import logging
import math
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlsplit

from dubbing_service.app.core.config import get_config, safe_project_root
from dubbing_service.app.storage.projects import (
    load_project,
    project_meta_lock,
)
from dubbing_service.app.storage.segments import load_segments

from . import voice_lab as lab
from . import web_addresses
from .fingerprint import voice_revision_for_role

SCHEMA = "casting-review-v1"
MAX_PROPOSALS = 6
MAX_CANDIDATES = 3
_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_DISPATCH_LOCK = threading.Lock()

#: Durable human-review events. Auto Cast state is the source of truth for each.
CHARACTER_REVIEW_READY = "CHARACTER_REVIEW_READY"
REPLACEMENT_REVIEW_READY = "REPLACEMENT_REVIEW_READY"
CASTING_COMPLETE = "CASTING_COMPLETE"
CASTING_NEEDS_ATTENTION = "CASTING_NEEDS_ATTENTION"
#: Production reached a terminal state, so the finished audio is the next thing
#: the human looks at. Emitted for success and for a bounded failure alike.
PRODUCTION_READY = "PRODUCTION_READY"
#: One finished chapter is ready to listen to. Emitted per chapter, as each is
#: recorded, so listening can start long before the whole book is done.
CHAPTER_READY = "CHAPTER_READY"

#: The request id the automatic production receipt is keyed by. One per approved
#: cast, so re-running the auto-start cannot queue a second render of the same
#: approved scope.
AUTOMATIC_PRODUCTION_ID = "autocast_render"

#: A review push waits at most this long for other characters still being cast,
#: so several auditions finishing together become one useful message while a
#: lone character is never held behind an unrelated long generation.
COALESCE_SECONDS = 90

#: Bounded delivery retries, per event. Casting itself never depends on them.
MAX_NOTIFICATION_ATTEMPTS = 3

#: A character waiting for the human's feedback is announced until the provider
#: accepts it. After the quick early attempts the retry interval stops growing at
#: this cap instead of the notification being abandoned, so an outage never turns
#: a review into one nobody was told about.
EXHAUSTED_RETRY_SECONDS = 1800

#: Playable auditions already waiting on the human are announced even when a
#: sibling candidate is still generating (or stuck) after this long, so one slow
#: voice can never hold back the review of the ones that are ready.
PARTIAL_REVIEW_SECONDS = 600

#: A configuration problem is retried slowly, so a fixed link starts working
#: without a restart and a still-broken one stays out of the retry budget.
BLOCKED_RETRY_SECONDS = 300

#: Durable request for a replacement candidate after a below-threshold rating.
REPLACEMENT_SCHEMA = "casting-replacement-v1"

#: Bounded attempts to import a proposed prior voice before it is blocked.
MAX_IMPORT_ATTEMPTS = 3


class CastingError(lab.VoiceLabError):
    pass


class MobileReviewLinkUnavailable(CastingError):
    """The configured review URL cannot be opened from the end user's phone."""


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CastingError("Use a stable event/request ID of 1–100 letters, digits, underscores or hyphens")
    return value


def _state(project):
    return project.voice_lab.casting


def _save(project):
    _state(project)["revision"] += 1
    lab._save(project)


def _role(project, role_id):
    role = next((r for r in project.roles if r.id == role_id), None)
    if role is None:
        raise CastingError("Unknown canonical speaking character")
    return role


def _first_problem(attention):
    """The message that names the first affected utterance, or ``None``."""
    for entry in attention:
        message = entry.get("message")
        if message:
            return message
    return None


def _canonical(project, role_id):
    from .character_identity import canonical_role_id
    return canonical_role_id(project, role_id)


def _held_ids(project):
    return set((project.metadata.get("autocast_import") or {}).get("unresolved_utterance_ids") or [])


# --------------------------------------------------------------------------
# Skipped lines
# --------------------------------------------------------------------------
#
# A line can be left out of the recording on purpose: nobody could say who speaks it,
# the voice model would not speak it (a spoken web address, say), or a person chose to.
# That is a decision about the book's spoken scope, so it is recorded durably with its
# reason, is never silent, and can be undone. Skipped lines are outside the scope
# everywhere the scope is decided, so nothing downstream ever waits for their audio.

SKIP_UNKNOWN_SPEAKER = "unknown_speaker"
SKIP_UNSPEAKABLE = "unspeakable"
SKIP_FAILED_QA = "failed_qa"
SKIP_HUMAN = "human"
# A link ("Table of Contents URL: https://..."). It is never read aloud, so it is left out
# whatever automatic handling says, and it does not count as the voice service refusing.
SKIP_WEB_ADDRESS = "web_address"
SKIP_REASONS = (SKIP_UNKNOWN_SPEAKER, SKIP_UNSPEAKABLE, SKIP_WEB_ADDRESS, SKIP_FAILED_QA, SKIP_HUMAN)
WEB_ADDRESS_DETAIL = "A web address, which is not read aloud"
# A refused line is tried this many times before it is skipped.
LINE_ATTEMPTS = 2
# A voice service that is failing would "refuse" every line, and skipping a whole book
# would be far worse than stopping, so a run stops instead of skipping past these.
MAX_SKIPS_IN_A_ROW = 5
MAX_SKIP_SHARE = 0.05
MIN_SKIP_BUDGET = 10
# The voice service's own line-specific refusals, as distinct from it being unreachable.
_REFUSAL = re.compile(r"generation budget|overlong", re.IGNORECASE)


def skipped_lines(project):
    """``{utterance_id: {reason, detail, by, at}}`` for every line left out on purpose."""
    return dict((project.metadata.get("autocast_import") or {}).get("skipped") or {})


def skipped_ids(project):
    return set(skipped_lines(project))


def record_skips(project, utterance_ids, *, reason, detail=None, by):
    """Leave lines out of the recording. The caller holds the project lock and saves."""
    if reason not in SKIP_REASONS:
        raise CastingError("Unknown reason for skipping a line")
    imported = project.metadata.setdefault("autocast_import", {})
    skipped = imported.setdefault("skipped", {})
    now = lab.now()
    ids = list(dict.fromkeys(utterance_ids))
    for uid in ids:
        skipped[uid] = {"reason": reason, "detail": str(detail or "")[:300], "by": by, "at": now}
    gone = set(ids)
    held = [uid for uid in imported.get("unresolved_utterance_ids") or [] if uid not in gone]
    imported["unresolved_utterance_ids"] = held
    imported["reason"] = f"{len(held)} dialogue utterance(s) need attribution review" if held else None
    return ids


def restore_skips(project, utterance_ids):
    """Bring skipped lines back into the scope. The caller holds the project lock and saves.

    A line that was skipped for having no speaker is remembered as one the person
    wants to decide about themselves, so automatic handling does not skip it again.
    """
    imported = project.metadata.setdefault("autocast_import", {})
    skipped = imported.setdefault("skipped", {})
    exempt = list(imported.get("policy_exempt") or [])
    restored = []
    for uid in dict.fromkeys(utterance_ids):
        entry = skipped.pop(uid, None)
        if entry is None:
            continue
        restored.append(uid)
        if entry.get("reason") in {SKIP_UNKNOWN_SPEAKER, SKIP_WEB_ADDRESS} and uid not in exempt:
            exempt.append(uid)
    imported["policy_exempt"] = exempt
    return restored


def clean_web_addresses(project, segments):
    """Take web addresses out of what a book reads aloud. The caller holds the lock and saves both.

    A line that is only a link is skipped. An address inside a sentence is taken out of that
    sentence's spoken text; its source text stays exactly as imported. A line a person brought
    back is theirs to decide about and is left alone. Returns the skipped and rewritten ids.
    """
    from .provenance import set_spoken_text, spoken_text_for_segment
    imported = project.metadata.get("autocast_import") or {}
    leave_alone = skipped_ids(project) | set(imported.get("policy_exempt") or [])
    skipped, rewritten = [], []
    for segment in segments:
        if segment.id in leave_alone:
            continue
        spoken = spoken_text_for_segment(project, segment)
        if not web_addresses.has_web_address(spoken):
            continue
        if web_addresses.is_link_line(spoken):
            skipped.append(segment.id)
        else:
            set_spoken_text(project, segment, spoken_text=web_addresses.without_web_addresses(spoken),
                            reason="Web address taken out: addresses are not read aloud")
            rewritten.append(segment.id)
    if skipped:
        record_skips(project, skipped, reason=SKIP_WEB_ADDRESS, detail=WEB_ADDRESS_DETAIL, by="cleanup")
    return skipped, rewritten


def line_problem(result):
    """``None`` for a recorded line, else ``(kind, detail)`` describing why it was not.

    ``unspeakable`` and ``failed_qa`` are about this line's text and are worth a retry and,
    failing that, a skip. Anything else (the service unreachable, a cast that changed under
    the run) says nothing about the line and stops the run instead.
    """
    if result.success and result.selected and not result.stale and result.qa_passed:
        return None
    error = str(getattr(result, "error", None) or "")
    if not result.success and _REFUSAL.search(error):
        return SKIP_UNSPEAKABLE, error
    if result.success and not result.stale and not result.selected and not result.qa_passed:
        return SKIP_FAILED_QA, error or "The recording did not pass its quality checks"
    return "other", error or "Production Take failed current-cast validation"


def scoped_rows(project, utterance_ids=None, segments=None):
    """The current source utterances a request covers: live, spoken, not a scene break.

    ``segments`` lets a caller that has already read the book reuse those rows instead
    of parsing the whole segment store again.
    """
    from .release import is_live
    rows = [s for s in (segments if segments is not None else load_segments(project.id)) if is_live(s)]
    wanted = list(dict.fromkeys(utterance_ids if utterance_ids is not None else [s.id for s in rows]))
    by_id = {s.id: s for s in rows}
    if not wanted or any(i not in by_id for i in wanted):
        raise CastingError("Choose a nonempty scope of current source utterances")
    scene_breaks = project.metadata.get("book_creator", {}).get("scene_breaks", [])
    skipped = skipped_ids(project)
    selected = [by_id[i] for i in wanted if by_id[i].text.strip() and i not in scene_breaks and i not in skipped]
    return [s.id for s in selected], selected


def _scope(project, utterance_ids=None, segments=None):
    """The castable scope, plus the entries a human still has to decide.

    One ambiguous or unresolved segment must never blank the whole project: it
    is isolated into ``needs_attention`` with its evidence while every
    reconciliable segment continues into casting.
    """
    from .character_identity import (
        UNRESOLVED_HELD_BY_IMPORT,
        resolve_segment,
    )
    wanted, selected = scoped_rows(project, utterance_ids, segments)
    scene_breaks = project.metadata.get("book_creator", {}).get("scene_breaks", [])
    held = _held_ids(project)
    required = []
    needs_attention = []

    def flag(segment, rid, reason, **evidence):
        from .character_identity import (
            UNRESOLVED_HELD_BY_IMPORT as _HELD,
        )
        from .character_identity import (
            UNRESOLVED_NO_EVIDENCE as _NONE,
        )
        from .character_identity import (
            UNRESOLVED_REAL_CONFLICT as _CONFLICT,
        )
        message = {
            _NONE: f"Unresolved speaker for {segment.id}; narrator is not a fallback",
            _HELD: f"Unresolved speaker for {segment.id}; review source attribution before casting",
            _CONFLICT: f"Speaker attribution conflicts with the assigned character for {segment.id}",
        }.get(reason, f"Unresolved speaker for {segment.id}; review the canonical character before casting")
        needs_attention.append({
            "utterance_id": segment.id, "role_id": rid, "reason": reason,
            "message": message, "evidence": evidence,
        })

    for segment in selected:
        if not segment.text.strip() or segment.id in scene_breaks:
            continue
        resolution = resolve_segment(project, segment, held=segment.id in held)
        if not resolution.settled:
            flag(segment, resolution.role_id, resolution.reason, **resolution.evidence)
            continue
        rid = resolution.role_id
        role = _role(project, rid)
        attribution = segment.attribution
        if role.name.strip().casefold() in {"unknown", "unattributed", "speaker", "speaker 1", "speaker_1"} and not (attribution and attribution.human_authoritative):
            flag(segment, rid, "unresolved_sentinel_speaker", role_name=role.name)
            continue
        if segment.id in held and not (attribution and attribution.human_authoritative):
            flag(segment, rid, UNRESOLVED_HELD_BY_IMPORT, role_name=role.name)
            continue
        if rid not in required:
            required.append(rid)
    return wanted, selected, required, needs_attention


def source_binding(project, rows):
    return lab.digest([{"id": s.id, "text": s.text, "role_id": _canonical(project, s.role_id),
        "source": s.source_anchor.model_dump(mode="json") if s.source_anchor else None,
        "spoken_text_revision": s.spoken_text_revision_id} for s in rows])


# --------------------------------------------------------------------------
# Durable automatic Auto Cast runs
# --------------------------------------------------------------------------
#
# A run is bound to projectId + source revision/content hash + casting pipeline
# version.  The same completion event delivered twice therefore resolves to the
# same run key and produces exactly one effective casting run; a genuinely new
# or changed source produces a new key, so new/changed characters are processed
# while existing human ratings and accepted voices stay in force.

def _current_source_identity(project):
    """The ids that make a casting run current, from the durable import record."""
    receipt = (project.metadata or {}).get("autocast_import") or {}
    revision_id = receipt.get("raw_source_revision_id")
    sha256 = receipt.get("source_sha256")
    if not revision_id and project.source_revisions:
        latest = project.source_revisions[-1]
        revision_id, sha256 = latest.id, latest.sha256
    return revision_id, sha256


def autocast_run_key(project, *, extra=None):
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    revision_id, sha256 = _current_source_identity(project)
    return lab.digest({
        "project_id": project.id,
        "source_revision_id": revision_id,
        "source_sha256": sha256,
        "pipeline_version": AUTOCAST_PIPELINE_VERSION,
        # A re-run that re-casts approved characters is a different intent, so it
        # mints its own run rather than being absorbed by a completed one.
        **(extra or {}),
    })


def request_autocast_run(project_id, *, trigger="source_import", reason=None,
                         recast_approved=False):
    """Durably record that the current source should be cast, once.

    This is the trigger point: it is called from the successful ingestion
    completion, never from opening Voice Lab, a page refresh or a timer.  It
    performs no GPU work; the existing maintenance owner starts the rounds.

    ``recast_approved`` re-casts characters that already have an approved voice,
    for an explicit whole-pipeline re-run. It is off by default so a routine
    version bump cannot overturn a human approval.
    """
    from dubbing_service.app.models.character import (
        AUTOCAST_PIPELINE_VERSION,
        AutoCastRun,
    )
    recast_approved = bool(recast_approved)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        key = autocast_run_key(project, extra={"recast_approved": recast_approved})
        current = project.character_cast.autocast
        if current is not None and current.key == key:
            # Same source revision, pipeline version and intent: one effective run.
            return current.model_dump(mode="json"), False
        revision_id, sha256 = _current_source_identity(project)
        if current is not None and current.status in {"pending", "running"}:
            current.history.append(current.model_dump(mode="json"))
            current.status = "superseded"
            current.completed_at = lab.now()
        run = AutoCastRun(
            key=key, project_id=project_id, source_revision_id=revision_id,
            source_sha256=sha256, pipeline_version=AUTOCAST_PIPELINE_VERSION,
            request_id="autocast_" + lab.digest([project_id, key])[:28],
            status="pending", trigger=trigger, reason=reason, requested_at=lab.now(),
            recast_approved=recast_approved,
        )
        project.character_cast.autocast = run
        lab._save(project)
        return run.model_dump(mode="json"), True


_AUTOCAST_START_GUARD = threading.Lock()
_AUTOCAST_START_ACTIVE = set()


def start_pending_autocast(project_id):
    # Process-local ownership mutex only; durable state remains the existing run.
    with _AUTOCAST_START_GUARD:
        if project_id in _AUTOCAST_START_ACTIVE:
            return None
        _AUTOCAST_START_ACTIVE.add(project_id)
    try:
        return _start_pending_autocast(project_id)
    finally:
        with _AUTOCAST_START_GUARD:
            _AUTOCAST_START_ACTIVE.discard(project_id)


def _start_pending_autocast(project_id):
    """Establish scope and rounds for a pending run, then hand off to the owner.

    Idempotent: ``prepare`` is keyed by the run's stable request id, so a
    duplicate completion event, a page refresh or a server restart cannot
    produce a second round or duplicate candidates.
    """
    with project_meta_lock(project_id):
        project = load_project(project_id)
        run = project.character_cast.autocast
        if run is None or run.status not in {"pending", "running"}:
            return None
        if run.status == "running" and run.request_id in project.voice_lab.casting.get("requests", {}):
            return None
        run.status = "running"
        run.attempts += 1
        run.started_at = run.started_at or lab.now()
        project.character_cast.policy.audition_mode = "auto"
        request_id = run.request_id
        claimed_run_key = run.key
        recast_approved = bool(getattr(run, "recast_approved", False))
        lab._save(project)
    try:
        prepared = prepare(project_id, request_id=request_id,
                           recast_approved=recast_approved)
    except CastingError as exc:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            run = project.character_cast.autocast
            if run is not None and run.key == claimed_run_key and run.key == autocast_run_key(project, extra={"recast_approved": recast_approved}):
                run.status = "needs_attention"
                run.reason = str(exc)[:900]
                run.completed_at = lab.now()
                lab._save(project)
        return None
    with project_meta_lock(project_id):
        project = load_project(project_id)
        run = project.character_cast.autocast
        if run is None or run.key != claimed_run_key or run.status != "running":
            return None
        state = _state(project)
        receipt = state["requests"].get(request_id) or {}
        run.required_role_ids = list(receipt.get("required_roles") or [])
        run.queued_role_ids = [rid for rid in run.required_role_ids
                               if (state["roles"].get(rid) or {}).get("round")]
        lab._save(project)
    return prepared


def settle_autocast_run(project):
    """Close the run once every required character has reached a terminal round.

    A round still being produced keeps the run open even when the character holds an
    earlier approval: an explicit re-cast supersedes the round without revoking the
    human's decision, so reading the approval as "finished" would close a run whose
    auditions are still generating — and, downstream, would announce completion and
    start rendering the voices that are being replaced.
    """
    run = project.character_cast.autocast
    if run is None or run.status not in {"pending", "running"}:
        return run
    state = _state(project)
    outstanding, ready, blocked = [], 0, 0
    for rid in run.required_role_ids:
        item = state["roles"].get(rid) or {}
        round_status = (item.get("round") or {}).get("status")
        if round_status in {None, "pending", "running"}:
            outstanding.append(rid)
        elif round_status == "blocked":
            blocked += 1
        elif approved(project, rid) or round_status in {"awaiting_review", "resolved"}:
            ready += 1
        else:
            outstanding.append(rid)
    if outstanding:
        return run
    run.status = "complete" if ready and not blocked else "partial" if ready else "needs_attention"
    run.completed_at = lab.now()
    run.reason = (f"{ready} character(s) have playable auditions awaiting human review"
                  if ready else "No character reached a playable audition")
    if blocked:
        run.reason += f"; {blocked} character(s) need attention"
    return run


def _pipeline_state(project, *, required, attention, scope_error, state, roles, characters):
    """The real workflow stage, derived only from durable state."""
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    receipt = (project.metadata or {}).get("autocast_import") or {}
    run = project.character_cast.autocast
    cast = [rid for rid in required]
    ready = sum(1 for role in roles for candidate in role["candidates"]
                if candidate.get("status") == "ready")
    approved_count = sum(1 for role in roles if role["status"] == "APPROVED")

    def stage(name, label):
        return {"schema": "casting-pipeline-v1", "stage": name, "label": label}

    if receipt.get("status") == "analyzing":
        pipeline = stage("extracting", "Extracting source text and speaker attributions")
    elif not project.chapters or not project.roles:
        pipeline = stage("segmenting", "Segmenting the source into utterances")
    elif characters == 0:
        pipeline = stage("resolving_characters",
                         "No canonical speaking character yet; review source attribution")
        if attention:
            pipeline["stage"] = "needs_attention"
            pipeline["label"] = "Utterances need speaker review before characters can be resolved"
    elif run is not None and run.status in {"pending", "running"}:
        pipeline = stage("casting", f"Auto Cast is preparing {characters} character(s)")
    elif ready == 0 and attention and not cast:
        pipeline = stage("needs_attention", "Characters are resolved but auditions are not ready")
    elif ready == 0:
        pipeline = stage("cast_ready", "Characters are resolved and queued for Auto Cast")
    elif approved_count == characters:
        pipeline = stage("ready_for_audio", "Every character has an approved voice")
    elif any(role["status"] == "APPROVED" for role in roles):
        pipeline = stage("auditions_partial", "Some characters are approved; others await review")
    else:
        pipeline = stage("awaiting_review", "Playable auditions await your rating")
    pipeline.update({
        "characters": characters,
        "castable": len(cast),
        "ready_options": ready,
        "approved": approved_count,
        "needs_attention": len(attention),
        "pipeline_version": AUTOCAST_PIPELINE_VERSION,
        "scope_error": scope_error,
    })
    if run is not None:
        pipeline["run"] = {
            "key": run.key[:16], "status": run.status, "attempts": run.attempts,
            "trigger": run.trigger, "pipeline_version": run.pipeline_version,
            "required": len(run.required_role_ids), "queued": len(run.queued_role_ids),
            "reason": run.reason, "started_at": run.started_at,
            "completed_at": run.completed_at,
        }
    return pipeline


def _current_key(project, role, records=None):
    version = voice_revision_for_role(role)
    for key, record in (lab.voice_records(project) if records is None else records).items():
        if record.get("version") == version and record.get("audio_sha256"):
            return key
    return None


def approved(project, role_id):
    item = _state(project)["roles"].get(role_id, {})
    assignment = item.get("assignment") or {}
    event = _state(project)["events"].get(assignment.get("event_id"), {})
    return bool(assignment.get("approved") and not assignment.get("rejected")
        and assignment.get("voice_revision") == voice_revision_for_role(_role(project, role_id))
        and event.get("actor") == "brian"
        and ((event.get("stars") or 0) >= 4
             or event.get("action") == "use" and event.get("selection_confirmed") is True)
        and (event.get("origin") == "human" or project.voice_lab.scope == "regression_fixture"))


def _description(project, role_id):
    dossier = project.character_cast.dossiers.get(role_id)
    if not dossier:
        return "Source voice description is not yet established."
    facts = [f"{f.field}: {f.value}" for f in dossier.facts if f.level == "EXPLICIT"]
    return "; ".join(facts[:4]) or "No explicit vocal description in the current source horizon."


def _direction(raw, previous=None, *, backend="qwen3tts", baselines=None, identity=None):
    """Conservative versioned interpreter; raw text always remains authoritative."""
    from .provider_capabilities import capability_record
    raw = raw.strip()
    lower = raw.lower()
    previous = copy.deepcopy(previous or {})
    dims = dict(previous.get("dimensions", {}))
    identity_terms = {"accent": ("british", "accent", "northern", "english accent"),
        "depth": ("deep", "pitch"), "nasality": ("nasal",), "breathiness": ("breathy", "breathier"),
        "warmth": ("warm",), "texture": ("gravel", "announcer"), "contrast": ("similar", "distinct")}
    performance_only = bool(raw and any(t in lower for t in ("calmer", "calm", "quiet", "slower", "faster", "excited", "energy"))
        and not any(t in lower for terms in identity_terms.values() for t in terms))
    for dimension, terms in identity_terms.items():
        if any(t in lower for t in terms):
            dims[dimension] = raw
    from .named_reference_feedback import interpret
    qualitative = interpret(raw, previous)
    performance_only = performance_only and not qualitative.get("requested_performer")
    caps = capability_record(backend, identity)
    performance = caps.get("performance", {}).get("style_instruction", {})
    supported = performance.get("status") == "supported"
    requested = [raw] if raw else []
    unsupported = []
    unverified = []
    if performance_only and not supported:
        unsupported.append("This production cloning path does not support natural-language performance instructions; the voice is retained.")
    elif raw:
        unverified.append("Requested direction requires listening; design wording does not prove an audible result.")
    if "british" in lower:
        unverified.append("British does not identify a regional accent; no region has been inferred.")
    if re.search(r"\b[abc]['’]s\b|first voice|go back", lower) and len(baselines or []) < 2:
        unsupported.append("This comparison needs the exact earlier candidate IDs; choose comparison baselines before generating.")
    if "pitch shift" in lower:
        unsupported.append("Pitch shifting is not a casting control in this workflow.")
    return {"interpreter_version": "casting-direction-v1", "requested": requested, "applied": [],
        "unsupported": unsupported, "unverified": unverified, "dimensions": dims,
        "retained": list(dict.fromkeys(dims.values())), "performance_only": performance_only,
        "baseline_candidate_ids": list(baselines or []), "confidence": "explicit_text_only", **qualitative}


def _new_round(project, role_id, *, direction=None, replacement=False, reuse_key=None, explore=False):
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    state = _state(project)
    role = _role(project, role_id)
    item = state["roles"].setdefault(role_id, {"assignment_revision": 0, "round_revision": 0,
        "history": [], "assignment": None, "rejections": []})
    old = item.get("round")
    if old:
        item["history"].append(copy.deepcopy(old))
        intent = state["intents"].get(old.get("intent_id"))
        if intent and intent["status"] not in {"complete", "failed"}:
            intent["status"] = "superseded"
    item["round_revision"] += 1
    rev = item["round_revision"]
    intent_id = "cast_" + lab.digest([project.id, role_id, rev])[:28]
    round_id = "round_" + lab.digest([project.id, role_id, rev])[:24]
    direction = copy.deepcopy(direction or {"requested": [], "applied": [], "unsupported": [], "unverified": [], "retained": []})
    candidates = []
    current_key = reuse_key or (None if replacement else _current_key(project, role))
    profile = project.character_cast.profiles.get(role_id)
    dossier = project.character_cast.dossiers.get(role_id)
    if dossier is not None:
        from .casting_profiles import build_profiles

        # Only a genuinely new round reaches here. Reinterpret current evidence
        # on local profiles so old policy/source claims cannot enter its prompt;
        # durable profiles, approved identities and existing rounds stay intact.
        profile = build_profiles(project, load_segments(project.id)).get(role_id)
    base = profile.design_instruction if profile else "Natural conversational narration with clear articulation."
    retained = " ".join(direction.get("retained", []) + direction.get("requested", []))
    from . import original_inspired_design
    from .named_reference_feedback import requested_performer as resolve_performer
    original = original_inspired_design.target(project, role_id, retained, direction.get("requested_performer"))
    performer = None if original else resolve_performer(project, role_id, retained, direction.get("requested_performer"))
    if original:
        direction.update(reference_mode="original_inspired_design", performance_only=False,
                         original_design_event=original["requested_by_event"])
        current_key = reuse_key = None  # A new original never retains a rejected or unrelated reference.
    if performer:
        direction.update(requested_performer=performer, reference_mode="verified_recording", performance_only=False)
        reuse_key = None
    matches = project.voice_lab.shortlists.setdefault(role_id, {}).setdefault("matches", [])
    plan = _proposal_plan(project, role_id, profile, dossier, base, retained, reuse_key=reuse_key, requested_performer=direction.get("requested_performer"), explore=explore)
    if len(plan) > MAX_PROPOSALS:
        raise CastingError("The per-round proposal budget has been exceeded")
    for index, proposal in enumerate(plan):
        generated = proposal["kind"] == "generated"
        cid = "candidate_" + lab.digest([round_id, index])[:24]
        candidate = {"candidate_id": cid, "label": chr(65 + index), "key": None, "status": "blocked" if proposal.get("blocked") else "pending",
            "rating": None, "audio_sha256": None, "duration_seconds": None,
            "technical_qa": "pending", "sections": [],
            "reuse_key": proposal.get("reuse_key") or (current_key if index == 0 and generated else None),
            "request_id": "castaud_" + lab.digest([round_id, cid])[:28],
            "provenance": proposal["kind"],
            "anchor": copy.deepcopy(proposal.get("anchor")),
            "inspiration": copy.deepcopy(proposal.get("inspiration")),
            # The reference evidence and the acoustic characterization derived
            # from it travel with the candidate: what was measured, what was
            # inferred, and the exact generation parameters they produce.
            "concept": copy.deepcopy(proposal.get("concept")),
            "reference": copy.deepcopy(proposal.get("reference")),
            "conditioning": copy.deepcopy(proposal.get("conditioning")),
            "supported_traits": copy.deepcopy(proposal.get("supported") or []),
            "inferred_traits": copy.deepcopy(proposal.get("inferred") or []),
            # What the acoustic conditioning pass actually achieved, filled in when
            # the candidate becomes audible. It is a separate field from the target
            # conditioning above: the target is the request, this is the result.
            "acoustics": None,
            "blocked": bool(proposal.get("blocked")),
            "blocked_reason": proposal.get("blocked_reason"),
            "reason": proposal.get("blocked_reason") if proposal.get("blocked") else None,
            "match_rationale": proposal["rationale"],
            "prior_match": proposal["kind"] == "prior_voice",
            "prior_evidence": copy.deepcopy(proposal.get("prior_evidence")),
            "library_voice_id": proposal.get("library_voice_id"),
            "library_version": proposal.get("library_version"),
            "import_state": "not_required" if proposal["kind"] != "prior_voice" else "pending",
            "import_attempts": 0,
            "import_reason": None,
            "needs_replacement": None}
        candidates.append(candidate)
        matches.append({"candidate_id": cid, "original_casting": True,
            "design_instruction": proposal["design_instruction"],
            "inspiration": copy.deepcopy(proposal.get("inspiration")),
            "anchor": copy.deepcopy(proposal.get("anchor")),
            "concept": copy.deepcopy(proposal.get("concept")),
            "reference": copy.deepcopy(proposal.get("reference")),
            "conditioning": copy.deepcopy(proposal.get("conditioning")),
            "supported_traits": copy.deepcopy(proposal.get("supported") or []),
            "inferred_traits": copy.deepcopy(proposal.get("inferred") or []),
            "match_rationale": proposal["rationale"],
            "prior_match": candidate["prior_match"],
            "tags": list(proposal.get("tags") or []), "source_urls": list(proposal.get("source_urls") or []),
            "round_id": round_id,
            "delivery_instruction": " ".join(direction.get("requested", [])) if reuse_key and direction.get("performance_only") else None})
    round_ = {"id": round_id, "revision": rev, "status": "pending", "candidates": candidates,
        "direction": direction, "intent_id": intent_id, "created_at": lab.now(),
        "script_version": "casting-30-v1" if replacement else "casting-60-v1", "proposal_count": len(plan),
        "replacement": bool(replacement),
        # Which casting pipeline produced these candidates. A later version change
        # makes them stale, and only this field can tell that apart from a round the
        # human is simply still reviewing.
        "pipeline_version": AUTOCAST_PIPELINE_VERSION,
        "baseline_voice_revision": voice_revision_for_role(role), "notification": {"state": "pending"}}
    if candidates and all(c["status"] == "blocked" for c in candidates):
        round_["status"] = "blocked"
    item["round"] = round_
    state["intents"][intent_id] = {"id": intent_id, "role_id": role_id, "round_revision": rev,
        "status": "complete" if round_["status"] == "blocked" else "pending", "attempts": 0, "created_at": lab.now()}
    return round_


def _keeper_proposals(project, role_id, profile, dossier, retained="", *, requested_performer=None, explore=False):
    """Shared read-only keeper decision, used before public research and rounds."""
    from . import casting_anchors, preferred_voices
    from .named_reference_feedback import requested_performer as resolve_performer
    performer = resolve_performer(project, role_id, retained, requested_performer)
    traits = casting_anchors.vocal_traits(profile, dossier, policy=project.character_cast.policy)
    role = next((item for item in project.roles if item.id == role_id), None)
    language = preferred_voices.normalized_language(getattr(role, "qwen3_language", None))
    if language:
        traits["language"] = language
    plan = []
    # A voice this exact character has already rated is not offered back to it:
    # the rating is the human's answer about this voice for this character.
    # Structured exploration is authoritative. Free text must be affirmative:
    # a negated request must not discard compatible heard keeper samples.
    clauses = re.split(r"[.!?;\n]", retained.casefold().replace("’", "'"))
    fresh_requested = explore or any(
        any(phrase in clause for phrase in
            ("new audition", "fresh audition", "explore new", "none fit"))
        and not re.search(r"\b(?:not|no|never|don't|do not|without|avoid)\b", clause)
        for clause in clauses
    )
    for row in ([] if fresh_requested else preferred_voices.match_preferred(traits, limit=MAX_CANDIDATES * 3)):
        if performer and str(row.get("actual_source_performer") or "").casefold() != performer.casefold():
            continue
        if any(rating.get("project_id") == project.id and rating.get("role_id") == role_id
               for rating in row.get("rated_roles") or row.get("ratings") or []):
            continue
        evidence = preferred_voices.evidence(row)
        prior_performer = row["inspiration"].get("performer")
        prior_performance = row["inspiration"].get("performance")
        plan.append({
            "kind": "prior_voice", "rationale": evidence["rationale"],
            "design_instruction": (f"Reuse the exact previously approved voice {row.get('name') or ''}".strip()
                                   + f" ({row['match_percent']}% character fit)."),
            "anchor": None,
            # A partial reference is not a reference: only a known performer *and*
            # performance may be attached to a proposal.
            "inspiration": ({"performer": prior_performer, "performance": prior_performance,
                             "source_urls": []} if prior_performer and prior_performance else None),
            "prior_evidence": evidence, "library_voice_id": row["library_voice_id"],
            "library_version": row["library_version"], "tags": [], "source_urls": [],
            "concept": None, "reference": None, "conditioning": None,
            "supported": [], "inferred": [], "blocked": False, "blocked_reason": None})
        if len(plan) >= MAX_CANDIDATES:
            break
    return plan


def _proposal_plan(project, role_id, profile, dossier, base, retained, *, reuse_key=None, requested_performer=None, explore=False):
    """The ordered proposal set for one round: prior approved voices, then anchored new ones.

    A prior voice already rated by this character is not re-proposed to it.
    Compatible four/five-star versions are heard as preserved historical samples
    first. New auditions follow an explicit exploration request or no keeper fit;
    no role, rating or production assignment is inherited.

    Every generated candidate carries its performer reference *and* the acoustic
    characterization derived from it, so the numbers that shape the voice are
    fixed here, shown in the review surface, and re-used unchanged when the
    design is dispatched.
    """
    from .casting_director import proposals_for_round
    directed = proposals_for_round(project, role_id)
    if directed is not None:
        return directed
    from .cloned_voice_policy import policy_mode
    if policy_mode(project) == "cloned_only":
        raise CastingError("This book requires cloned voices. Use Review recommendations and Request fresh auditions; synthetic design fallback is disabled.")
    from . import original_inspired_design
    from .named_reference_feedback import proposals
    from .named_reference_feedback import requested_performer as resolve_performer
    original = original_inspired_design.target(project, role_id, retained, requested_performer)
    if original:
        return original_inspired_design.proposals(original)
    performer = resolve_performer(project, role_id, retained, requested_performer)
    if reuse_key:
        return [{"kind": "generated", "rationale": "Directed reuse of an explicitly chosen heard voice.",
                 "design_instruction": base + (" Human casting direction: " + retained if retained else ""),
                 "anchor": None, "inspiration": None, "tags": [], "source_urls": [],
                 "concept": None, "reference": None, "conditioning": None,
                 "supported": [], "inferred": [], "blocked": False, "blocked_reason": None}]
    role = next((item for item in project.roles if item.id == role_id), None)
    plan = _keeper_proposals(project, role_id, profile, dossier, retained,
                            requested_performer=performer, explore=explore)
    if plan:
        return plan  # Hear the exact rated samples before requesting exploration.
    if performer:
        return proposals(project, role_id, performer)
    plan.extend(_generated_proposals(project, profile, dossier, role, base, retained,
                                     count=MAX_CANDIDATES - len(plan), priors=plan))
    return plan


def _prior_dimensions(priors) -> set[str]:
    """The dimensions a previously approved voice already answers for this character."""
    dimensions = set()
    for proposal in priors:
        for reason in (proposal.get("prior_evidence") or {}).get("match_reasons") or []:
            dimensions.add(str(reason).split(":", 1)[0].strip().replace(" ", "_"))
    return dimensions


def _cast_performers(project, exclude_role_id) -> list[str]:
    """Performers this project has already cast or proposed to other characters.

    A book's characters should be read by different people: two characters sharing
    a reference performer is a casting decision nobody made. Both the approved
    assignment and the current round count, so a character already holding auditions
    is not proposed the same faces again under a replacement round.
    """
    names = []
    state = _state(project)
    for role_id, item in (state.get("roles") or {}).items():
        if role_id == exclude_role_id:
            continue
        assignment = item.get("assignment") or {}
        if assignment.get("key"):
            record = lab.voice_records(project).get(assignment["key"]) or {}
            inspiration = record.get("inspiration") or {}
            if inspiration.get("performer"):
                names.append(inspiration["performer"])
        for candidate in (item.get("round") or {}).get("candidates") or []:
            performer = ((candidate.get("anchor") or {}).get("performer")
                         or (candidate.get("inspiration") or {}).get("performer"))
            if performer:
                names.append(performer)
    return list(dict.fromkeys(names))


def _generated_proposals(project, profile, dossier, role, base, retained, *, count, priors):
    """Fresh anchored candidates, ordered so they complement any prior voice.

    A historical voice already covers the dimensions that made it match, so
    re-proposing those dimensions as a new voice would spend an audition slot on
    a near-duplicate. Concepts are therefore ordered least-overlapping first —
    ordered, never dropped, because a complementary concept set is still worth
    exploring even when the overlap happens to be high.
    """
    from . import casting_anchors, casting_inference, performer_reference
    if count <= 0:
        return []
    covered = _prior_dimensions(priors)
    policy = project.character_cast.policy if project is not None else None
    concepts = casting_inference.casting_concepts(profile, dossier, count=count, role=role, policy=policy)
    if covered:
        concepts.sort(key=lambda concept: len(set(concept["inferred"]) & covered))
    proposals = []
    taken = [row.get("performer") for row in priors if row.get("performer")]
    avoid = _cast_performers(project, getattr(role, "id", None)) if project is not None else []
    for index, concept in enumerate(concepts):
        research = ((project.voice_lab.shortlists.get(getattr(role, "id", None)) or {}).get("model_research") or {}) if project is not None else {}
        recommendations = research.get("recommendations") or []
        anchors = casting_anchors.anchors_for_role(profile, dossier, limit=1, role=role,
                                                   concept=concept, exclude_performers=taken,
                                                   avoid_performers=avoid, policy=policy,
                                                   preferred_candidate_ids=[r["candidate_id"] for r in recommendations])
        anchor = anchors[0] if anchors else casting_anchors.unanchored_candidate(
            index, casting_anchors.UNANCHORED_REASON)
        if recommendations:
            anchor["model_research"] = {"schema": research.get("schema"), "route": research.get("route"),
                "receipt_file": research.get("receipt_file"),
                "recommendations": [r for r in recommendations if r["candidate_id"] == anchor.get("catalog_candidate_id")],
                "voice_approval": False, "source_acquisition_authorized": False}
        anchor["concept"] = concept
        anchor["index"] = index
        if anchor.get("performer"):
            taken.append(anchor["performer"])
        evidence = (performer_reference.resolve(anchor, concept)
                    if anchor.get("anchored") and concept else None)
        proposals.append({
            "kind": "generated", "rationale": evidence["rationale"] if evidence else anchor["rationale"],
            "design_instruction": _anchored_instruction(base, anchor, index, retained, evidence),
            "anchor": anchor,
            "inspiration": ({"performer": anchor["performer"], "performance": anchor["performance"],
                             "source_urls": anchor["source_urls"]} if anchor["anchored"] else None),
            "tags": list(anchor.get("tags") or []), "source_urls": list(anchor.get("source_urls") or []),
            "concept": copy.deepcopy(concept),
            "reference": performer_reference.describe(evidence) if evidence else None,
            "conditioning": performer_reference.conditioning_payload(evidence) if evidence else None,
            "supported": casting_inference.supported_summary(concept) if concept else [],
            "inferred": casting_inference.inferred_summary(concept) if concept else [],
            "blocked": not anchor["anchored"],
            "blocked_reason": anchor.get("unanchored_reason") if not anchor["anchored"] else None,
            "prior_evidence": None, "library_voice_id": None, "library_version": None})
    return proposals


def _concept_summary(concept) -> dict | None:
    if not concept:
        return None
    return {"key": concept.get("key"), "label": concept.get("label"),
            "emphasis": concept.get("emphasis"),
            "differing_dimensions": list(concept.get("differing_dimensions") or [])}


def _anchored_instruction(base, anchor, contrast_index, retained, evidence=None):
    """A design instruction that names its reference and still demands an original voice."""
    from . import casting_anchors
    contrasts = ["Balanced conversational resonance and unhurried articulation.",
                 "A distinct lower-resonance identity with rounded articulation, without pitch processing.",
                 "A distinct brighter-resonance identity with crisp, flexible articulation."]
    parts = [base]
    if anchor.get("concept"):
        parts.append(f"Casting concept: {anchor['concept']['emphasis']}.")
    parts.append(casting_anchors.design_fragment(anchor))
    if evidence:
        parameters = evidence["conditioning"]["parameters"]
        parts.append(_acoustic_clause(parameters))
    parts.append(contrasts[contrast_index % len(contrasts)])
    if retained:
        parts.append("Human casting direction: " + retained)
    return " ".join(part for part in parts if part)


def _acoustic_clause(parameters) -> str:
    """The numeric targets the design must hit, stated without the performer's name."""
    bits = []
    if parameters.get("pitch_target_hz"):
        bits.append(f"{parameters['pitch_target_hz']:.0f} Hz median pitch")
    if parameters.get("pitch_range_semitones"):
        bits.append(f"{parameters['pitch_range_semitones']:.1f} semitone range")
    if parameters.get("rate_words_per_minute"):
        bits.append(f"{parameters['rate_words_per_minute']:.0f} words per minute")
    if parameters.get("pause_ms"):
        bits.append(f"{parameters['pause_ms']:.0f} ms mean pauses")
    if parameters.get("brightness_hz"):
        bits.append(f"{parameters['brightness_hz']:.0f} Hz spectral brightness")
    if parameters.get("breathiness") is not None:
        bits.append(f"breathiness {parameters['breathiness']:.2f}")
    if parameters.get("rasp") is not None:
        bits.append(f"rasp {parameters['rasp']:.2f}")
    if parameters.get("dynamic_range_db"):
        bits.append(f"{parameters['dynamic_range_db']:.1f} dB dynamic range")
    if not bits:
        return ""
    return ("Acoustic target extracted from that reference performance: "
            + ", ".join(bits) + f"; {parameters['resonance']} resonance, "
            f"{parameters['vocal_weight']} vocal weight"
            + (f", {parameters['age_impression']} impression"
               if parameters.get("age_impression") else "") + ".")


def _is_real_character(role):
    """A canonical role that is a speaker, not an attribution slot."""
    from .character_identity import SPEAKER_SENTINELS
    name = (role.name or "").strip().casefold()
    return name not in SPEAKER_SENTINELS - {"narrator", "narration"}


def _attention_characters(project, attention):
    """Canonical *character* roles named by needs-attention evidence.

    Attribution labels such as ``narrator``/``unknown`` are slots, not people,
    so they are excluded here even though they own a role record.
    """
    from dubbing_service.app.models.enums import RoleType

    from .character_identity import (
        SPEAKER_SENTINELS,
        canonical_role_id,
        role_id_for_label,
    )
    by_id = {role.id: role for role in project.roles}
    found = {}
    for entry in attention:
        rid = canonical_role_id(project, entry.get("role_id"))
        if not rid:
            label = (entry.get("evidence") or {}).get("attribution_label")
            if not label or label.strip().casefold() in SPEAKER_SENTINELS:
                continue
            rid = role_id_for_label(project, label)
        role = by_id.get(rid) if rid else None
        if role is None or role.role_type != RoleType.CHARACTER:
            continue
        if (role.name or "").strip().casefold() in SPEAKER_SENTINELS:
            continue
        found.setdefault(rid, []).append(entry["utterance_id"])
    return found


def _current_audio(project, role_id, key, records, base):
    """Expose existing character audio without creating a round or approval."""
    record = records.get(key) or {}
    if record.get("role_id") != role_id or not record.get("audio_sha256"):
        return None
    current_revision = voice_revision_for_role(_role(project, role_id))
    assignment = (_state(project)["roles"].get(role_id) or {}).get("assignment") or {}
    assigned_current = assignment.get("key") == key and assignment.get("voice_revision") == current_revision
    if record.get("version") != current_revision and not assigned_current:
        return None
    available, error = False, None
    try:
        from .voice_lab_audition import _audio_still_verified
        _audio_still_verified(lab._path(project.id, record["audio_file"]), record["audio_sha256"])
        available = True
    except (ValueError, OSError, KeyError) as exc:
        error = str(exc)
    provenance = (record.get("identity") or {}).get("provenance", {})
    preference = project.voice_lab.preferences.get(key, {})
    return {"key": key, "name": record.get("name"), "kind": record.get("kind"),
            "synthetic": bool(record.get("synthetic", True)),
            "provenance": provenance.get("origin"), "inspiration": record.get("inspiration"),
            "audio_sha256": record["audio_sha256"], "duration_seconds": record.get("duration_seconds"),
            "audio_available": available, "audio_error": error,
            "audio_url": f"{base}/audio/{key}" if available else None,
            "favorite": bool(preference.get("favorite") and preference.get("audio_sha256") == record["audio_sha256"])}


def view(project_id, base_url=None):
    from .provider_capabilities import capability_record
    project = load_project(project_id)
    state = _state(project)
    base = base_url or f"/api/v1/projects/{project_id}/voice-lab"
    result = []
    # The book is read once for the whole view. Every character that has no assigned voice
    # yet used to make its own full read of the segment store to look one up, so a cast
    # with a few new characters took seconds to describe, on a page that asks every few
    # seconds while auditions are being made.
    segments = load_segments(project_id)
    voices = {}

    def voice_index():
        if "records" not in voices:
            voices["records"] = lab.voice_records(project, segments)
        return voices["records"]

    try:
        _, _, required, attention = _scope(project, None, segments)
        scope_error = None
    except CastingError as exc:
        required, attention, scope_error = list(state["roles"]), [], str(exc)
    needs_attention = _attention_characters(project, attention)
    by_role = {role.id: role for role in project.roles}
    required_set = set(required)
    discovered = [rid for rid in dict.fromkeys([
        *required,
        *state["roles"],
        *project.character_cast.profiles,
        *needs_attention,
    ]) if rid in by_role and (rid in required_set or _is_real_character(by_role[rid]))]
    for rid in discovered:
        role = by_role[rid]
        item = state["roles"].get(rid, {})
        round_ = item.get("round", {})
        has_approval = approved(project, rid)
        rejected = (item.get("assignment") or {}).get("rejected")
        status = "APPROVED" if has_approval and (not round_ or round_.get("status") == "resolved") else "NEEDS_RECAST" if rejected else "NEEDS_REVIEW" if round_ else "NEW_CHARACTER"
        if round_.get("status") == "blocked" and not has_approval:
            status = "BLOCKED"
        held_ids = needs_attention.get(rid, [])
        castable = rid in required
        if held_ids and not round_ and not castable:
            # Every utterance for this character is parked, so nothing can be
            # cast yet. A character with at least one settled utterance keeps its
            # normal progress state; the held utterances are reported separately.
            status = "NEEDS_REVIEW"
        candidates = []
        for candidate in round_.get("candidates", []):
            row = copy.deepcopy(candidate)
            row["audio_url"] = f"{base}/audio/{row['key']}" if row.get("key") else None
            preference = project.voice_lab.preferences.get(row.get("key"), {})
            row["favorite"] = bool(preference.get("favorite")
                                   and preference.get("audio_sha256") == row.get("audio_sha256"))
            candidates.append(row)
        reason = round_.get("reason") or ("Current human approval applies." if has_approval else "Listen and save a character-fit decision.")
        if held_ids and not round_ and not castable:
            reason = (f"{len(held_ids)} utterance(s) for this character await attribution review "
                      "before an audition can be prepared.")
        assignment = item.get("assignment") or {}
        current_key = (assignment.get("key") if assignment.get("voice_revision") == voice_revision_for_role(role)
                       else None) or _current_key(project, role, voice_index())
        current_audio = _current_audio(project, rid, current_key, voice_index(), base)
        assignment_event = state["events"].get((item.get("assignment") or {}).get("event_id"), {})
        result.append({"role_id": rid, "name": role.name, "description": _description(project, rid),
            "status": status, "reason": reason,
            "assignment_revision": item.get("assignment_revision", 0), "round_revision": item.get("round_revision", 0),
            "current_key": current_key, "current_audio": current_audio,
            "selected_candidate_id": assignment_event.get("candidate_id") if has_approval else None,
            "candidates": candidates, "direction": round_.get("direction", {}),
            "history": copy.deepcopy(item.get("history", [])), "notification": round_.get("notification", {}),
            # ``chosen`` is the same test the book uses to count voices: a valid human approval,
            # whatever state a later round is in. ``status`` follows the round, so a character
            # whose approval stands but whose round was reopened reads NEEDS_REVIEW.
            "required": castable, "chosen": has_approval, "round_status": round_.get("status"),
            "replacement": bool(round_.get("replacement")),
            "replacement_request": copy.deepcopy(_replacement_queue(project).get(rid)),
            "needs_attention_utterance_ids": held_ids,
            "castable": castable})
    capabilities = dict(capability_record("qwen3tts"))
    capabilities["round_requests"] = True
    pipeline = _pipeline_state(project, required=required, attention=attention,
                               scope_error=scope_error, state=state, roles=result,
                               characters=len(discovered))
    return {"schema": SCHEMA, "revision": state["revision"], "roles": result,
        "productions": list(state["productions"].values()), "capabilities": capabilities,
        "scope_error": scope_error, "pipeline": pipeline,
        "needs_attention": attention,
        "utterance_ids": [s.id for s in segments]}


def round_is_stale(existing) -> bool:
    """Was this round produced by an earlier casting pipeline than the current one?

    A round records the pipeline version that produced its candidates. Within one
    version a round is the human's review surface and is never disturbed — which is
    what makes ``prepare`` idempotent across refreshes, restarts and duplicate
    completion events. The version change, the documented signal that automatic
    casting's observable behaviour changed, is the only thing that makes the current
    candidates stale.
    """
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    if not existing:
        return False
    return existing.get("pipeline_version") != AUTOCAST_PIPELINE_VERSION


def prepare(project_id, *, request_id, utterance_ids=None, recast_approved=False):
    """Establish scope and one round per uncast character.

    ``recast_approved`` is the explicit, opt-in re-cast: a caller re-running the
    whole pipeline over a project whose characters were cast by an earlier
    pipeline asks for every character to be re-cast. It is never implied by a
    version change, because an approval is the human's decision and a pipeline
    bump must not overturn it silently. The superseded round and every rating and
    direction on it move to history, never away.
    """
    _id(request_id)
    from .autocast_public_model_research import research_for_project
    model_research = research_for_project(project_id, request_id, utterance_ids, recast_approved=recast_approved)
    if model_research is not None and model_research.get("status") != "complete":
        raise CastingError("Public model research held: " + str(model_research.get("reason") or "Setup or public evidence required")[:400])
    with project_meta_lock(project_id):
        project = load_project(project_id)
        wanted, rows, required, attention = _scope(project, utterance_ids)
        if not required:
            raise CastingError(_first_problem(attention)
                               or "No actual speaking characters in the requested scope")
        state = _state(project)
        state["enabled"] = True
        binding = lab.digest(wanted)
        previous = state["requests"].get(request_id)
        if previous and previous["payload_hash"] != binding:
            raise CastingError("Request ID already belongs to another source scope")
        if not previous:
            if model_research is not None:
                from .autocast_public_model_research import _stamp
                if model_research.get("request_binding") != _stamp(project, rows, required, request_id):
                    raise CastingError("Canonical research inputs changed before casting; stale results held")
                if model_research.get("route_config_sha256"):
                    from .autocast_public_model_research import (
                        ResearchHold,
                        resolve_route,
                    )
                    from .autocast_public_model_research import (
                        digest as research_digest,
                    )
                    try:
                        route_matches = research_digest(resolve_route(project)) == model_research["route_config_sha256"]
                    except ResearchHold:
                        route_matches = False
                    if not route_matches:
                        raise CastingError("Selected model configuration changed before casting; stale results held")
                for rid in model_research.get("keeper_role_ids", []):
                    profile = project.character_cast.profiles.get(rid)
                    dossier = project.character_cast.dossiers.get(rid)
                    if dossier is not None:
                        from .casting_profiles import build_profiles
                        profile = build_profiles(project, rows).get(rid)
                    if not _keeper_proposals(project, rid, profile, dossier):
                        raise CastingError("Compatible keeper changed before casting; review research scope again")
            state["requests"][request_id] = {"payload_hash": binding, "utterance_ids": wanted,
                "source_binding": source_binding(project, rows), "required_roles": required}
            recast = []
            for rid in required:
                existing = (state["roles"].get(rid) or {}).get("round")
                if approved(project, rid) and not recast_approved:
                    continue  # An approved voice is the human's decision, not the pipeline's.
                if existing is None:
                    _new_round(project, rid)
                elif round_is_stale(existing) or recast_approved:
                    # A re-submitted run under a new pipeline version re-casts the
                    # character: the reviewed candidates are no longer what this
                    # pipeline would propose. The old round and every rating and
                    # direction on it move to history rather than being discarded.
                    _new_round(project, rid)
                    recast.append(rid)
            if recast:
                state["requests"][request_id]["recast_role_ids"] = sorted(recast)
                state["requests"][request_id]["recast_approved"] = bool(recast_approved)
            _save(project)
    return view(project_id)


def request_round(project_id, *, request_id, role_id, expected_round_revision,
                  expected_assignment_revision, intent):
    """Durably request one role's initial round or an explicit no-fit replacement.

    Unlike prepare(), this does not establish or change chapter/source scope. The
    request receipt, current-round transition, and synthesis intent are committed
    together so retries cannot queue a second round.
    """
    _id(request_id)
    _id(role_id)
    if intent not in {"generate", "none_fit"}:
        raise CastingError("Intent must be generate or none_fit")
    if type(expected_round_revision) is not int or expected_round_revision < 0:
        raise CastingError("Expected round revision must be a nonnegative integer")
    if type(expected_assignment_revision) is not int or expected_assignment_revision < 0:
        raise CastingError("Expected assignment revision must be a nonnegative integer")
    command = {"request_id": request_id, "role_id": role_id,
        "expected_round_revision": expected_round_revision,
        "expected_assignment_revision": expected_assignment_revision, "intent": intent}
    command_hash = lab.digest(command)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        state = _state(project)
        requests = state.setdefault("round_requests", {})
        previous = requests.get(request_id)
        if previous:
            if previous.get("payload_hash") != command_hash:
                raise CastingError("Request ID already belongs to a different casting-round command")
            receipt = copy.deepcopy(previous["receipt"])
        else:
            _, _, required, _ = _scope(project)
            if role_id not in required:
                raise CastingError("This character is outside the current speaking-character scope")
            item = state["roles"].get(role_id)
            current_round_revision = item.get("round_revision", 0) if item else 0
            assignment_revision = item.get("assignment_revision", 0) if item else 0
            if expected_round_revision != current_round_revision or expected_assignment_revision != assignment_revision:
                raise CastingError("Casting changed since this view was loaded; refresh before requesting audio")

            round_ = item.get("round") if item else None
            intent_row = state["intents"].get(round_.get("intent_id")) if round_ else None
            intent_status = (intent_row or {}).get("status")
            if intent == "generate":
                if round_ is None:
                    state["enabled"] = True
                    round_ = _new_round(project, role_id)
                    created = True
                elif round_.get("status") == "pending" and intent_status in {"pending", "running"}:
                    # A second tab can safely join the same in-flight generation.
                    created = False
                else:
                    raise CastingError("This character already has a round. Use Try new voices to request a replacement.")
            else:
                if round_ is None:
                    raise CastingError("Generate audio for this character before requesting new voices")
                if intent_status in {"pending", "running"} or round_.get("status") == "pending":
                    raise CastingError("Audio generation is already queued or running for this character")
                # "None fit" is an explicit identity-level exclusion, without
                # manufacturing a star rating, feedback event, or assignment.
                for candidate in round_.get("candidates", []):
                    if candidate.get("status") != "ready" or not candidate.get("identity_id"):
                        continue
                    if any(rejection.get("request_id") == request_id
                           and rejection.get("effective_identity") == candidate["identity_id"]
                           for rejection in item.get("rejections", [])):
                        continue
                    item.setdefault("rejections", []).append({
                        "request_id": request_id, "reason": "none_fit",
                        "key": candidate.get("key"),
                        "version": candidate.get("voice_version"),
                        "effective_identity": candidate["identity_id"],
                    })
                round_ = _new_round(project, role_id, replacement=True, explore=True)
                created = True

            receipt = {"request_id": request_id, "role_id": role_id, "intent": intent,
                "round_id": round_["id"], "round_revision": round_["revision"],
                "assignment_revision": assignment_revision, "intent_id": round_["intent_id"],
                "status": round_["status"], "created": created,
                "reason": "Audio generation queued for this character." if created
                    else "This character's current audio generation is already queued or running."}
            requests[request_id] = {"payload_hash": command_hash, "receipt": copy.deepcopy(receipt),
                "created_at": lab.now()}
            _save(project)
    return {"receipt": receipt, "reason": receipt["reason"], "casting": view(project_id)}


def prepare_imported_review(project_id, *, role_id, key, library_voice_id, version,
                            audio_sha256, expected_identity_id, request_id):
    """Stage an exact Library sample for the existing explicit-selection owner.

    This metadata-only preparation applies to an authorized, non-strict chapter
    fork. It assigns nothing, inherits no approval/rating and schedules no work.
    The caller must still submit a genuine current selection through feedback().
    """
    _id(request_id)
    _id(role_id)
    from . import voice_lab_chapter as chapter, voice_lab_library as shelf
    from dubbing_service.app.storage import voices as library
    from dubbing_service.app.storage.segments import project_segments_lock
    from dubbing_service.app.core.tts_operations import try_acquire_project_mutation, release_project_mutation
    with library.voices_lock():
        voice = next((v for v in library.load_voices() if v.id == library_voice_id), None)
        if voice is None or version not in voice.voice_lab_versions:
            raise CastingError("Exact Library version is unavailable")
        entry = shelf._validated_entry(version, voice.voice_lab_versions[version])
    if entry["record"]["key"] != key or entry["record"]["audio_sha256"] != audio_sha256:
        raise CastingError("Review must identify the exact immutable Library sample")
    command = {"role_id": role_id, "key": key, "library_voice_id": library_voice_id,
               "version": version, "audio_sha256": audio_sha256,
               "expected_identity_id": expected_identity_id, "request_id": request_id}
    command_hash = lab.digest(command)
    lease = try_acquire_project_mutation(project_id, kind="casting_imported_review")
    if lease is None:
        raise CastingError("Rendering owns this project; imported review must wait")
    try:
        initial = load_project(project_id)
        chapter_id = initial.metadata.get("chapter_rerun", {}).get("chapter_id")
        source_project, speech, binding = chapter.source_scope(project_id, chapter_id)
        if role_id not in {s.role_id for s in speech}:
            raise CastingError("This character is outside the exact chapter speech scope")
        with project_meta_lock(project_id), project_segments_lock(project_id):
            project = load_project(project_id)
            state = _state(project)
            if (state.get("enabled") or project.metadata.get("autocast_import", {}).get("mode") == "autocast"):
                raise CastingError("Strict casting uses its existing review rounds")
            if project.metadata.get("chapter_rerun") != source_project.metadata.get("chapter_rerun"):
                raise CastingError("Chapter source changed during review preparation")
            rows = sorted((s for s in load_segments(project_id) if chapter.release.is_live(s)), key=lambda s: (s.index, s.id))
            current_source = [chapter._source_row(s) for s in rows]
            if (current_source != source_project.metadata["chapter_rerun"]["source_utterances"]
                    or chapter._hash(chapter._source_proof(project, current_source)) != binding["source_hash"]
                    or project.character_cast.role_aliases != binding["aliases"]
                    or any(s.playback_speed != 1 or s.edit_state is not None for s in rows)
                    or project.character_cast.policy.enabled or project.character_cast.policy.audition_mode != "off"):
                raise CastingError("Chapter source changed during review preparation")
            requests = state.setdefault("imported_review_requests", {})
            previous = requests.get(request_id)
            if previous:
                if previous["command_hash"] != command_hash:
                    raise CastingError("Request ID already belongs to different imported review")
                return copy.deepcopy(previous["receipt"])
            role = _role(project, role_id)
            current = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
            if current != expected_identity_id:
                raise CastingError("Casting changed since review was requested; refresh")
            if role.voice_identity is not None or (state["roles"].get(role_id) or {}).get("assignment"):
                raise CastingError("Imported initial review requires an unassigned chapter role")
            record = lab._verified_record(project, key, audio_sha256)
            if (record.get("library_voice_id") != library_voice_id or record.get("library_version") != version
                    or record.get("version") != version
                    or shelf._profile_fingerprint(record) != shelf._profile_fingerprint(entry["record"])):
                raise CastingError("Imported review differs from the immutable Library version")
            # Revalidate immutable conditioning and any imported Take proof.
            shelf._publication_record(project_id, record)
            root = safe_project_root(project_id)
            for relative, checksum in shelf._assets(record).values():
                shelf._read_verified(shelf._safe_path(root, relative), checksum)
            qa = record.get("technical_qa")
            if not (qa == "passed" or isinstance(qa, dict) and (qa.get("passed") is True or qa.get("status") == "passed")):
                raise CastingError("Imported sample requires passing bound technical QA")
            item = state["roles"].setdefault(role_id, {"assignment_revision": 0, "round_revision": 0,
                "history": [], "assignment": None, "rejections": []})
            if item.get("round"):
                raise CastingError("This character already has a review round; use its current decision")
            revision = item["round_revision"] + 1
            round_id = "round_" + lab.digest([project_id, request_id, role_id, key])[:24]
            candidate_id = "candidate_" + lab.digest([round_id, key, audio_sha256])[:24]
            from .provider_capabilities import effective_identity
            candidate = {"candidate_id": candidate_id, "label": "A", "status": "ready", "key": key,
                "audio_sha256": audio_sha256, "identity_id": effective_identity(record), "voice_version": version,
                "duration_seconds": record.get("duration_seconds"), "technical_qa": copy.deepcopy(qa),
                "sections": copy.deepcopy(record.get("sections", record.get("lines", []))), "rating": None,
                "provenance": "preserved_library_sample", "library_voice_id": library_voice_id,
                "library_version": version, "new_audio_generated": False}
            item["round_revision"] = revision
            item["round"] = {"id": round_id, "revision": revision, "status": "awaiting_review",
                "candidates": [candidate], "direction": {}, "created_at": lab.now(),
                "baseline_voice_revision": voice_revision_for_role(role), "proposal_count": 1,
                "script_version": "preserved-library-sample", "source_hash": binding["source_hash"],
                "notification": {"state": "not_requested"}, "metadata_only": True}
            receipt = {"role_id": role_id, "candidate_id": candidate_id, "key": key,
                "audio_sha256": audio_sha256, "round_revision": revision,
                "assignment_revision": item["assignment_revision"], "new_audio_generated": False,
                "assigned": False, "approved": False}
            requests[request_id] = {"command_hash": command_hash, "receipt": copy.deepcopy(receipt)}
            _save(project)
            return receipt
    finally:
        release_project_mutation(lease)
def feedback(project_id, payload):
    """Commit one exact human event, CAS decision and follow-up intent together."""
    _id(payload.get("event_id"))
    stars = payload.get("stars")
    if stars is not None and (type(stars) is not int or not 1 <= stars <= 5):
        raise CastingError("Stars must be an explicit integer from 1 to 5")
    if "selection_confirmed" in payload and type(payload["selection_confirmed"]) is not bool:
        raise CastingError("Selection confirmation must be an explicit boolean")
    action = payload.get("action", "save")
    if action not in {"save", "use", "recast", "preview"}:
        raise CastingError("Unknown casting action")
    raw = payload.get("raw_direction", "")
    if not isinstance(raw, str) or len(raw) > 4000:
        raise CastingError("Direction must be at most 4000 characters")
    command = {**payload, "action": action, "raw_direction": raw}
    command_hash = lab.digest(command)
    lease = None
    if action == "use":
        from dubbing_service.app.core.tts_operations import try_acquire_project_mutation
        lease = try_acquire_project_mutation(project_id, kind="casting_approval")
        if lease is None:
            raise CastingError("Rendering owns this project; save feedback now or use this voice at the next safe boundary")
    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            state = _state(project)
            previous = state["events"].get(payload["event_id"])
            if previous:
                if previous["command_hash"] != command_hash:
                    raise CastingError("Event ID already belongs to different feedback")
                return copy.deepcopy(previous["result"])
            rid = payload["role_id"]
            role = _role(project, rid)
            item = state["roles"].get(rid)
            if not item:
                raise CastingError("Prepare a review round before submitting feedback")
            rounds = [item["round"], *item["history"]]
            heard_round = next((r for r in rounds if r["revision"] == payload["round_revision"]), None)
            candidate = next((c for c in (heard_round or {}).get("candidates", []) if c["candidate_id"] == payload["candidate_id"]), None)
            if not candidate or candidate.get("status") != "ready" or candidate.get("key") != payload["key"]:
                raise CastingError("Feedback must identify a ready audition from the exact round heard")
            record = lab._verified_record(project, payload["key"], payload["audio_sha256"])
            if candidate["audio_sha256"] != record["audio_sha256"]:
                raise CastingError("Audition version changed")
            baselines = payload.get("baseline_candidate_ids") or [candidate["candidate_id"]]
            known = {c["candidate_id"]: c for r in rounds for c in r["candidates"]}
            if any(cid not in known or known[cid].get("status") != "ready" for cid in baselines):
                raise CastingError("Comparison baseline must name an exact playable audition")
            direction = _direction(raw, heard_round.get("direction"), backend=record.get("backend", "qwen3tts"), baselines=baselines, identity=record.get("identity"))
            stale = (payload["round_revision"] != item["round_revision"]
                or payload["assignment_revision"] != item["assignment_revision"]
                or heard_round.get("baseline_voice_revision") != voice_revision_for_role(role)
                    and not ((item.get("assignment") or {}).get("key") == payload["key"]
                             and (item.get("assignment") or {}).get("voice_revision") == voice_revision_for_role(role))
                or (heard_round["status"] == "resolved" and (item.get("assignment") or {}).get("key") != payload["key"]))
            result = {"saved": True, "conflict": stale, "event_id": payload["event_id"],
                "reason": "Historical feedback saved; a newer casting decision wins. Refresh before changing the current cast." if stale else "Feedback saved.",
                "direction": direction}
            origin = "SYNTHETIC" if project.voice_lab.scope == "regression_fixture" else "human"
            event = {**command, "command_hash": command_hash, "actor": "brian", "origin": origin,
                "created_at": lab.now(), "voice_id": record["voice_id"], "voice_version": record["version"],
                "take_id": record.get("take_id"), "baselines": [{k: known[cid].get(k) for k in
                    ("candidate_id", "key", "audio_sha256")} for cid in baselines], "interpretation": direction, "result": result}
            if not stale:
                if action == "use":
                    if (stars is not None and stars < 4) or (stars is None and payload.get("selection_confirmed") is not True):
                        raise CastingError("Use this voice requires explicit selection confirmation or a 4- or 5-star character-fit rating")
                    if (record.get("identity") or {}).get("provenance", {}).get("fixture_only") and origin != "SYNTHETIC":
                        raise CastingError("Test-generated voices cannot receive production human approval")
                    from .provider_capabilities import effective_identity
                    selected_identity = effective_identity(record)
                    if any(other_id != rid and approved(project, other_id)
                           and effective_identity(lab._record(project, other["assignment"]["key"])) == selected_identity
                           for other_id, other in state["roles"].items()):
                        raise CastingError("This effective voice identity is already approved for another character; unique casting requires a distinct identity")
                    current = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
                    lab.apply_assignment(project, record["key"], rid, current)
                    item["assignment_revision"] += 1
                    item["assignment"] = {"key": record["key"], "voice_revision": voice_revision_for_role(role),
                        "approved": True, "rejected": False, "event_id": payload["event_id"], "stars": stars,
                        "revision": item["assignment_revision"],
                        "decision": "explicit_selection" if payload.get("selection_confirmed") is True else "rated_selection"}
                    heard_round["status"] = "resolved"
                    result["reason"] = "Feedback saved and this exact voice assigned."
                candidate["rating"] = stars if stars is not None else candidate.get("rating")
                candidate["feedback_event_id"] = payload["event_id"]
                heard_round["direction"] = direction
                is_current = record["version"] == voice_revision_for_role(role)
                if stars is not None and stars <= 3:
                    # Rating below the fit threshold records the decision and asks
                    # for a replacement; it never restarts this round. The human
                    # stays free to keep listening and scoring the other
                    # candidates, and the reviewed set stays exactly as heard.
                    item["rejections"].append({"key": record["key"], "version": record["version"],
                        "event_id": payload["event_id"], "rating": stars,
                        "feedback": raw, "effective_identity": candidate.get("identity_id")})
                    if is_current:
                        item["assignment_revision"] += 1
                        item["assignment"] = {"key": record["key"], "voice_revision": record["version"],
                            "approved": False, "rejected": True, "event_id": payload["event_id"],
                            "revision": item["assignment_revision"]}
                    candidate["needs_replacement"] = {
                        "schema": REPLACEMENT_SCHEMA, "rating": stars, "event_id": payload["event_id"],
                        "feedback": raw, "candidate_id": candidate["candidate_id"],
                        "key": record["key"], "requested_at": lab.now(),
                        "direction": copy.deepcopy(direction)}
                    _queue_replacement(project, rid, round_id=heard_round["id"],
                        candidate=candidate, direction=direction, stars=stars,
                        feedback=raw, event_id=payload["event_id"], key=record["key"])
                    result["reason"] = ("Feedback saved; this option is marked for replacement. "
                                        "Keep listening: the replacement arrives as a new round.")
                elif action in {"recast", "preview"} or (action == "save" and stars is None and direction.get("actionable")) or (stars is not None and stars >= 4 and raw and direction["performance_only"]):
                    if direction.get("unsupported"):
                        result["reason"] = "Feedback saved. Requested controls need clarification or are unsupported; no synthesis queued."
                    else:
                        followup = _new_round(project, rid, direction=direction, replacement=True,
                            reuse_key=record["key"] if direction["performance_only"] else None)
                        result["reason"] = ("Feedback saved without adding a rating. " + followup["candidates"][0]["blocked_reason"]
                            if followup["status"] == "blocked" else "Feedback saved; one directed audition round is queued.")

                result.update(assignment_revision=item["assignment_revision"], round_revision=item["round_revision"])
            state["events"][payload["event_id"]] = event
            _save(project)
            outcome = copy.deepcopy(result)
            preference = _preference_write(project, rid, role, record, payload, stars, action)
        if preference is not None:
            # The global voice library is a separate store with its own lock, so
            # this write must happen after the project lock has been released.
            # Its outcome is recorded on the event, never folded into the reply:
            # replaying the same event must return the same answer.
            _record_preference_outcome(project_id, payload["event_id"], _publish_preference(preference))
        return outcome
    finally:
        if lease:
            from dubbing_service.app.core.tts_operations import release_project_mutation
            release_project_mutation(lease)


def _preference_write(project, role_id, role, record, payload, stars, action):
    """The cross-project preference facts for one rating, captured under the lock.

    Returns ``None`` when the human gave no rating: only a rating states whether a
    voice is worth reusing elsewhere.
    """
    if stars is None:
        return None
    profile = project.character_cast.profiles.get(role_id)
    from . import casting_anchors
    traits = casting_anchors.vocal_traits(profile, project.character_cast.dossiers.get(role_id),
                                         policy=project.character_cast.policy)
    return {"project_id": project.id, "role_id": role_id, "character": role.name,
            "key": record["key"], "audio_sha256": record.get("audio_sha256"),
            "rating": stars, "action": action, "feedback": payload.get("raw_direction", ""),
            "event_id": payload["event_id"], "traits": traits,
            "scope": project.voice_lab.scope}


def _publish_preference(payload):
    """Record a rating on the canonical cross-project voice library, never fatally.

    A library failure must not lose the human's rating, which is already durable
    in the project: the outcome is reported and the rating stands regardless.
    """
    from . import preferred_voices
    from .voice_lab_library import publish_version
    if payload.get("scope") == "regression_fixture":
        return {"state": "skipped_fixture"}
    try:
        published = publish_version(payload["project_id"], payload["key"])
        entry = preferred_voices.record_rating(
            library_voice_id=published["library_voice_id"], version=published["version"],
            rating=payload["rating"], project_id=payload["project_id"], role_id=payload["role_id"],
            character=payload["character"], key=payload["key"], audio_sha256=payload["audio_sha256"],
            feedback=payload["feedback"], action=payload["action"], traits=payload["traits"],
            evidence_id=payload["event_id"])
    except Exception as exc:  # noqa: BLE001 - a library failure must not invalidate the rating
        return {"state": "unavailable", "reason": str(exc)[:400]}
    return {"state": "recorded", "library_voice_id": published["library_voice_id"],
            "version": published["version"], "promoted": bool(entry.get("promoted")),
            "signal": entry.get("signal"), "highest_rating": entry.get("highest_rating")}


def _replacement_queue(project):
    return _state(project).setdefault("replacement_queue", {})


def _record_preference_outcome(project_id, event_id, result):
    """Attach the library outcome to its event, so diagnostics can see it."""
    with project_meta_lock(project_id):
        project = load_project(project_id)
        event = _state(project)["events"].get(event_id)
        if event is None:
            return
        event["preference"] = copy.deepcopy(result)
        _save(project)


def _queue_replacement(project, role_id, *, round_id, candidate, direction, stars, feedback, event_id, key):
    """Durably ask for a replacement of one candidate, without disturbing the round.

    The request is keyed to the round that was actually reviewed. A later round
    replaces the entry rather than accumulating stale requests, and a request
    already recorded for the same event is never duplicated.
    """
    queue = _state(project).setdefault("replacement_queue", {})
    entry = queue.get(role_id)
    if not isinstance(entry, dict) or entry.get("round_id") != round_id:
        entry = {"schema": REPLACEMENT_SCHEMA, "role_id": role_id, "round_id": round_id,
                 "status": "waiting_for_review", "created_at": lab.now(),
                 "candidates": [], "events": []}
        queue[role_id] = entry
    if event_id not in entry["events"]:
        entry["events"].append(event_id)
        entry["candidates"].append({
            "schema": REPLACEMENT_SCHEMA, "candidate_id": candidate["candidate_id"],
            "key": key, "rating": stars, "event_id": event_id, "feedback": feedback,
            "direction": copy.deepcopy(direction), "requested_at": lab.now()})
    entry["updated_at"] = lab.now()
    return entry


def _rejection_direction(entry):
    """Merge the retained directions of every deferred rejection, once each."""
    merged = {"requested": [], "applied": [], "unsupported": [], "unverified": [], "retained": []}
    dimensions = {}
    for row in entry.get("candidates") or []:
        direction = row.get("direction") or {}
        for key, values in merged.items():
            for value in direction.get(key) or []:
                if isinstance(value, str) and value.strip() and value not in values:
                    values.append(value)
        for name, value in (direction.get("dimensions") or {}).items():
            dimensions.setdefault(name, value)
    merged["dimensions"] = dimensions
    if any(row.get("feedback") for row in entry.get("candidates") or []):
        merged["confidence"] = "explicit_text_only"
        merged["interpreter_version"] = "casting-direction-v1"
    return merged


def _round_fully_reviewed(round_):
    """Whether every playable candidate in this round already carries a rating."""
    ready = [candidate for candidate in round_.get("candidates", []) if candidate.get("status") == "ready"]
    if not ready:
        return False
    return all(candidate.get("rating") is not None for candidate in ready)


def _round_in_flight(project, role_id) -> bool:
    """Is this character's current round still being produced?

    A round in flight is the authoritative statement that the character's cast is
    not settled, even when an earlier approval still stands. An explicit re-cast
    supersedes the round without revoking the human's decision, so every consumer
    that asks "is this character finished?" must read the round, not the approval.
    """
    item = _state(project)["roles"].get(role_id) or {}
    return ((item.get("round") or {}).get("status") or "pending") in {"pending", "running"}


def _cast_settled(project, required=None):
    """``(settled, required)``: every required character approved and nothing in flight."""
    required = list(required if required is not None else _required_role_ids(project))
    if not required:
        return False, required
    if any(_round_in_flight(project, role_id) for role_id in required):
        return False, required
    return all(approved(project, role_id) for role_id in required), required


def _cast_binding(project, required) -> str:
    """A digest of the exact approved voices a production would render."""
    state = _state(project)
    return lab.digest([
        (role_id, (state["roles"].get(role_id, {}).get("assignment") or {}).get("key"))
        for role_id in sorted(required)])


def _automatic_production(project):
    """Start the audiobook render once the cast is complete, without a click.

    Called with this project's state held. The precondition is exactly the one the
    human cannot act on further: every required character in the current scope is
    approved *and* nothing is still being produced. Until then this does nothing, so
    a partly-reviewed cast is never rendered with placeholder voices.

    A re-cast in flight retracts an automatic receipt that has not started, because
    that receipt was created against a cast that is being replaced: promoting it
    would publish the superseded voices.

    Idempotent by construction: the receipt is keyed by a fixed request id, so a
    later tick, a restart or a duplicate completion event finds the existing receipt
    and leaves it alone. A receipt that reached a terminal failure is not retried
    here — that is the human's explicit resume, not an automatic loop.
    """
    state = _state(project)
    existing = state["productions"].get(AUTOMATIC_PRODUCTION_ID)
    settled, required = _cast_settled(project)
    if not settled:
        if existing and existing.get("status") in {"awaiting_review", "blocked", "queued"}:
            recasting = [rid for rid in required if _round_in_flight(project, rid)]
            existing.update(
                status="retracted",
                reason=("Superseded: "
                        + (f"{len(recasting)} character(s) are being re-cast" if recasting
                           else "the cast is no longer fully approved")))
            # Persist here rather than relying on the caller: this path can be the
            # only state change in a tick, and an unsaved retraction leaves a
            # renderable receipt over a cast that is being replaced.
            _save(project)
        return False
    if existing and existing.get("status") != "retracted":
        return False
    try:
        wanted, rows, _, _ = _scope(project)
    except CastingError:
        return False
    if not wanted:
        return False
    # The approval that completes the cast also removes the last thing awaiting a
    # human, so the receipt is created here rather than in the HTTP handler that
    # would have needed the same check.
    carried = list(existing.get("superseded") or []) if existing else []
    if existing is not None:
        # A retracted receipt belongs to a cast that was replaced. The cast has now
        # settled again, so a fresh receipt is created for it rather than reviving
        # the old one — and the retraction is carried forward as history so the
        # reason a render did not happen is still auditable.
        carried = (carried + [{"cast_binding": existing.get("cast_binding"),
                               "retracted_reason": existing.get("reason"),
                               "retracted_at": existing.get("completed_at") or lab.now()}])[-4:]
    state["productions"][AUTOMATIC_PRODUCTION_ID] = {
        "request_id": AUTOMATIC_PRODUCTION_ID, "payload_hash": lab.digest(wanted),
        "utterance_ids": wanted, "required_roles": required,
        "source_binding": source_binding(project, rows),
        "status": "awaiting_review",
        "job_id": "strict_" + lab.digest([project.id, AUTOMATIC_PRODUCTION_ID, _cast_binding(project, required)])[:28],
        "created_at": lab.now(), "attempts": 0, "total_attempts": 0,
        "resume_count": 0, "manifest_history": [],
        # The approved cast this receipt belongs to, so a receipt that outlives its
        # cast is identifiable as a superseded render rather than a pending one.
        "cast_binding": _cast_binding(project, required),
        "superseded": carried,
        "automatic": True,
        "reason": "Every required character is approved; the render owner queues production",
    }
    _save(project)
    return True


def _production_event(project):
    """The durable event a finished render implies, or ``None`` while it runs.

    Derived from the canonical production receipt alone, so it cannot fire twice
    for one receipt and cannot fire for a render that is still going. A retracted
    receipt is not an outcome to announce: the re-cast's own review push says what
    happened, and calling a superseded render "stopped" would ask for a resume that
    is already in progress.
    """
    state = _state(project)
    receipt = state["productions"].get(AUTOMATIC_PRODUCTION_ID)
    if not receipt:
        return None
    status = receipt.get("status")
    if status not in {"succeeded", "failed", "blocked"}:
        return None
    manifest = receipt.get("manifest") or {}
    key = "production:" + lab.digest([project.id, receipt["request_id"], status,
                                      manifest.get("manifest_id") or ""])[:32]
    return {"event": PRODUCTION_READY, "key": key, "status": status,
            "reason": receipt.get("reason"),
            "manifest_id": manifest.get("manifest_id"),
            "utterance_count": len(receipt.get("utterance_ids") or []),
            "chapters_ready": sum(1 for row in (receipt.get("chapters") or {}).values()
                                  if row.get("status") == "ready"),
            "attempts": receipt.get("attempts"),
            # The delivery layer identities each notification by a hash binding, so
            # the production event carries one derived from the same facts its key is.
            "binding": [lab.digest([project.id, PRODUCTION_READY, receipt["request_id"], status,
                                    manifest.get("manifest_id") or ""])]}


def _chapter_events(project):
    """One durable, idempotent event per chapter that is recorded and ready to play.

    Derived only from the production receipt, so a restart, a refresh or a second
    tick cannot announce a chapter twice, while a chapter whose audio genuinely
    changed is a new event.
    """
    receipt = _state(project)["productions"].get(AUTOMATIC_PRODUCTION_ID)
    if not receipt or receipt.get("status") == "retracted":
        return []
    chapters = receipt.get("chapters") or {}
    ready = {cid: row for cid, row in chapters.items() if row.get("status") == "ready" and row.get("audio_sha256")}
    if not ready:
        return []
    from .autocast_book import chapter_plan
    plan = chapter_plan(project, receipt.get("utterance_ids") or [])
    number = {entry["chapter_id"]: position + 1 for position, entry in enumerate(plan)}
    events = []
    for chapter_id in sorted(ready, key=lambda cid: number.get(cid, 10 ** 6)):
        row = ready[chapter_id]
        events.append({
            "event": CHAPTER_READY, "chapter_id": chapter_id,
            "key": "chapter:" + lab.digest([project.id, receipt["request_id"], chapter_id, row["audio_sha256"]])[:32],
            "binding": [lab.digest([project.id, CHAPTER_READY, chapter_id, row["audio_sha256"]])],
            "chapter_title": row.get("title"), "chapter_number": number.get(chapter_id, row.get("number") or 1),
            "chapters_ready": len(ready), "chapters_total": len(plan) or len(ready),
            "duration_seconds": row.get("duration_seconds")})
    return events


def _replacement_dispatch(project):
    """Queue one replacement round per character whose reviewed set is finished.

    Called with this project's state held, so it may create the round directly.
    An approved character, a superseded round or a character already holding a
    newer round all release their queued request instead of acting on it. The
    project is saved only when a queued request actually changed state.
    """
    state = _state(project)
    changed = False
    for role_id, entry in list(state.setdefault("replacement_queue", {}).items()):
        if not isinstance(entry, dict) or entry.get("schema") != REPLACEMENT_SCHEMA:
            del state["replacement_queue"][role_id]
            changed = True
            continue
        if entry.get("status") not in {"waiting_for_review", "queued"}:
            continue
        item = state["roles"].get(role_id)
        if item is None:
            entry.update(status="superseded", reason="This character is no longer in the casting scope")
            changed = True
            continue
        round_ = item.get("round") or {}
        if round_.get("id") != entry.get("round_id"):
            entry.update(status="superseded",
                         reason="A newer audition round was created; this request is historical")
            changed = True
            continue
        if approved(project, role_id) or round_.get("status") == "resolved":
            entry.update(status="superseded", reason="This character already has an approved voice")
            changed = True
            continue
        if round_.get("status") in {"pending", "running"}:
            reason = "The reviewed round is still being produced"
        elif not _round_fully_reviewed(round_):
            reason = "Waiting for the remaining auditions in this round to be rated"
        else:
            direction = _rejection_direction(entry)
            created = _new_round(project, role_id, direction=direction, replacement=True)
            entry.update(status="queued", new_round_id=created["id"],
                         reason="Replacement round created from the retained rating and feedback",
                         dispatched_at=lab.now())
            changed = True
            continue
        if entry.get("reason") != reason or entry.get("status") != "waiting_for_review":
            entry.update(status="waiting_for_review", reason=reason)
            changed = True
    if changed:
        _save(project)
    return project


def _pending_prior_candidates(project):
    pending = []
    for role_id, item in _state(project)["roles"].items():
        round_ = item.get("round") or {}
        if round_.get("status") != "pending":
            continue
        for candidate in round_.get("candidates", []):
            if not candidate.get("library_voice_id") or candidate.get("status") in {"ready", "blocked"}:
                continue
            if candidate.get("import_state") not in {"pending", "waiting", "ready"}:
                continue
            if candidate.get("import_attempts", 0) >= MAX_IMPORT_ATTEMPTS:
                continue
            pending.append({"role_id": role_id, "round_id": round_["id"],
                            "candidate_id": candidate["candidate_id"],
                            "library_voice_id": candidate["library_voice_id"],
                            "library_version": candidate["library_version"]})
    return pending


def _prepare_prior_imports(project_id):
    """Make each proposed prior voice locally auditable before its round runs.

    A failure retries within a bounded budget and then blocks that one candidate
    with its reason. It never substitutes a freshly generated voice, because the
    human asked to hear the approved voice.
    """
    from . import preferred_voices
    with project_meta_lock(project_id):
        pending = _pending_prior_candidates(load_project(project_id))
    for row in pending:
        result = preferred_voices.prepare_prior_candidate(project_id, row["library_voice_id"],
                                                          row["library_version"])
        with project_meta_lock(project_id):
            project = load_project(project_id)
            item = _state(project)["roles"].get(row["role_id"]) or {}
            round_ = item.get("round") or {}
            if round_.get("id") != row["round_id"]:
                continue
            candidate = next((c for c in round_.get("candidates", [])
                              if c["candidate_id"] == row["candidate_id"]), None)
            if candidate is None:
                continue
            attempts = candidate.get("import_attempts", 0) + 1
            if result["state"] == "ready":
                candidate.update(reuse_key=result["key"], import_state="ready",
                                 import_attempts=attempts, import_reason=None)
                # Verify and expose the preserved heard sample. Do not spend a
                # production inference merely to rediscover an approved voice.
                from .provider_capabilities import effective_identity
                expected_hash = (candidate.get("prior_evidence") or {}).get("sample_audio_sha256")
                try:
                    record = lab._verified_record(project, result["key"], expected_hash)
                    if not expected_hash or record["audio_sha256"] != expected_hash:
                        raise ValueError("Prior rating is not bound to this exact historical sample")
                    identity = effective_identity(record)
                    others = {c.get("identity_id") for c in round_.get("candidates", [])
                              if c["candidate_id"] != candidate["candidate_id"] and c.get("status") == "ready"}
                    rejected = {r.get("effective_identity") for r in item.get("rejections", [])}
                    for other_id, other in _state(project)["roles"].items():
                        if other_id != row["role_id"] and approved(project, other_id):
                            others.add(effective_identity(lab._record(project, other["assignment"]["key"])))
                    if identity in others | rejected:
                        raise ValueError("Historical voice collides with an existing or rejected identity")
                    candidate.update(status="ready", key=result["key"], audio_sha256=expected_hash,
                        duration_seconds=record.get("duration_seconds"), identity_id=identity,
                        technical_qa=copy.deepcopy(record.get("technical_qa")),
                        voice_version=record["version"], sections=copy.deepcopy(record.get("sections", record.get("lines", []))),
                        duration_status=record.get("duration_status", "measured"),
                        acoustics=copy.deepcopy(record.get("acoustics")), ready_at=lab.now(),
                        sample_origin="historical_approved_sample", production_runtime_validated=False,
                        source_generation_metadata=copy.deepcopy(record.get("generation_metadata")),
                        source_runtime_profile=record.get("execution_profile") or "unknown_historical_runtime")
                except (lab.VoiceLabError, ValueError, OSError) as exc:
                    candidate.update(status="blocked", import_state="blocked", reason=str(exc)[:400])
            else:
                candidate.update(import_attempts=attempts,
                                 import_state="blocked" if attempts >= MAX_IMPORT_ATTEMPTS else "waiting",
                                 import_reason=result["reason"])
            _save(project)
    return pending


def _job(project, job_id, *, kind, ids, message, status=None):
    from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
    from dubbing_service.app.storage.jobs import get_job, mutate_job, upsert_job
    existing = get_job(project.id, job_id)
    if existing is None:
        existing = upsert_job(project.id, SynthesisJob(id=job_id, project_id=project.id,
            project_name=project.name, job_type=kind, request_id=job_id, idempotency_key=job_id,
            segment_ids=ids, message=message, status=status or SynthesisJobStatus.QUEUED))
    elif status is not None:
        def change(job):
            job.status, job.message = status, message
        existing = mutate_job(project.id, job_id, change)
    return existing


def _round_worker(project_id, intent_id):
    from dubbing_service.app.models.jobs import SynthesisJobStatus
    from dubbing_service.app.storage.jobs import mark_job_active

    from . import voice_lab_audition as auditions
    from .provider_capabilities import effective_identity
    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            intent = _state(project)["intents"][intent_id]
            rid, revision = intent["role_id"], intent["round_revision"]
            item = _state(project)["roles"][rid]
            if item["round_revision"] != revision or item["round"].get("status") == "resolved":
                return
            current_round = item["round"]
            current_round.setdefault("started_at", lab.now())
            current_round["status"] = "running"
            _job(project, intent_id, kind="voice_lab_listening", ids=[],
                 message=f"Generating audio options for {rid}", status=SynthesisJobStatus.RUNNING)
            round_ = copy.deepcopy(current_round)
            _save(project)
        for candidate in round_["candidates"]:
            with project_meta_lock(project_id):
                project = load_project(project_id)
                item = _state(project)["roles"][rid]
                if item["round_revision"] != revision or item["round"].get("status") == "resolved":
                    return
                current = next(c for c in item["round"]["candidates"] if c["candidate_id"] == candidate["candidate_id"])
                if current["status"] in {"ready", "blocked"}:
                    continue
                if current.get("library_voice_id"):
                    # The approved prior voice could not be made locally audible.
                    # Generating a different voice here would answer a question the
                    # human did not ask, so this candidate stops with its reason.
                    current.update(status="blocked", import_state="blocked",
                        reason=current.get("import_reason")
                        or "The previously approved voice is not available in this project yet")
                    item["round"]["status"] = "pending"
                    _state(project)["intents"][intent_id]["status"] = "pending"
                    _job(project, intent_id, kind="voice_lab_listening", ids=[],
                         message="Prior voice unavailable; remaining options await resume",
                         status=SynthesisJobStatus.QUEUED)
                    _save(project)
                    return
                role = _role(project, rid)
                expected = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
            audition_started = time.perf_counter()
            result = auditions.audition(project_id, role_id=rid, candidate_id=candidate["candidate_id"],
                expected_identity_id=expected, request_id=candidate["request_id"],
                script_version=round_["script_version"], reuse_key=candidate.get("reuse_key"))
            with project_meta_lock(project_id):
                project = load_project(project_id)
                item = _state(project)["roles"][rid]
                if item["round_revision"] != revision or item["round"].get("status") == "resolved":
                    return  # Audition is immutable history, never current after supersession.
                current = next(c for c in item["round"]["candidates"] if c["candidate_id"] == candidate["candidate_id"])
                if result["status"] == "succeeded" and result.get("key"):
                    record = lab._verified_record(project, result["key"], result["audio_sha256"])
                    identity = effective_identity(record)
                    collisions = {c.get("identity_id") for c in item["round"]["candidates"] if c["status"] == "ready"}
                    rejected = {r.get("effective_identity") for r in item.get("rejections", [])}
                    other_approved = []
                    for other_id, other in _state(project)["roles"].items():
                        if other_id != rid and approved(project, other_id):
                            other_approved.append(effective_identity(lab._record(project, other["assignment"]["key"])))
                    if identity in collisions | rejected | set(other_approved):
                        current.update(status="blocked", reason="Effective voice identity collides with an existing/rejected voice; no automatic shared casting.")
                    else:
                        current.update(status="ready", key=result["key"], audio_sha256=result["audio_sha256"],
                            duration_seconds=result["duration_seconds"], technical_qa=record.get("technical_qa"),
                            identity_id=identity, sections=record.get("sections", record.get("lines", [])),
                            duration_status=record.get("duration_status", "measured"), voice_version=record["version"],
                            # The achieved acoustic profile travels with the audible
                            # candidate, so the review surface reports what the
                            # conditioning did rather than only what it targeted.
                            acoustics=copy.deepcopy(record.get("acoustics")),
                            dispatch_wall_seconds=round(time.perf_counter() - audition_started, 3), ready_at=lab.now())
                else:
                    attempts = project.metadata.get("voice_lab_auditions", {}).get("intents", {})
                    exhausted = any(i.get("job_id") == result["job_id"] and i.get("attempts", 0) >= 3 for i in attempts.values())
                    current.update(status="blocked" if exhausted else "pending", reason=result.get("reason"))
                    _save(project)
                    return
                _save(project)
        with project_meta_lock(project_id):
            project = load_project(project_id)
            item = _state(project)["roles"][rid]
            if item["round_revision"] != revision or item["round"].get("status") == "resolved":
                return
            ready = [c for c in item["round"]["candidates"] if c["status"] == "ready"]
            item["round"]["status"] = "awaiting_review" if ready else "blocked"
            item["round"]["completed_at"] = lab.now()
            item["round"]["reason"] = (f"{len(ready)} audio options ready. Historical samples retain their original runtime; new auditions use the production path. Human fit remains unverified."
                if ready else "No noncolliding usable candidate is available; inspect the capability/pool limitation.")
            _state(project)["intents"][intent_id]["status"] = "complete"
            settle_autocast_run(project)
            _save(project)
            _job(project, intent_id, kind="voice_lab_listening", ids=[], message=item["round"]["reason"],
                 status=SynthesisJobStatus.SUCCEEDED if ready else SynthesisJobStatus.FAILED)
    except Exception as exc:  # noqa: BLE001 - a worker must record its failure, not die unrecorded
        with project_meta_lock(project_id):
            project = load_project(project_id)
            intent = _state(project)["intents"][intent_id]
            intent["status"] = "failed"
            intent["reason"] = str(exc)[:1200]
            item = _state(project)["roles"][intent["role_id"]]
            if item["round_revision"] == intent["round_revision"]:
                item["round"].update(status="blocked", reason=str(exc)[:1200])
            _save(project)
            _job(project, intent_id, kind="voice_lab_listening", ids=[], message="Casting round needs attention",
                 status=SynthesisJobStatus.INTERRUPTED)
    finally:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            intent = _state(project)["intents"].get(intent_id, {})
            item = _state(project)["roles"].get(intent.get("role_id"), {})
            resolved = item.get("round", {}).get("status") == "resolved"
            superseded = intent.get("round_revision") != item.get("round_revision")
            if resolved or superseded:
                intent["status"] = "superseded" if superseded else "complete"
                _save(project)
                _job(project, intent_id, kind="voice_lab_listening", ids=[], message="Round resolved or superseded; immutable completed auditions retained",
                     status=SynthesisJobStatus.CANCELLED if superseded else SynthesisJobStatus.SUCCEEDED)
        mark_job_active(project_id, intent_id, False)


def _clean_label(value):
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.casefold() in {"", "unknown"} else text


# Bumped whenever the rejection text changes. A blocked receipt is skipped by the
# promotion loop, so without this a later, better diagnosis would never reach the
# human: the stored reason would keep describing the code that first blocked it.
BLOCK_REASON_VERSION = 2


def unresolved_scope(project, rows):
    """The utterances in scope whose speaker never reached one canonical role.

    ``_scope`` isolates these into needs-attention instead of blanking the
    project, so they stay inside ``rows``: anything building a manifest across
    the whole scope has to partition them out rather than look a role up and
    raise. The two populations are kept separate because they need different
    human actions -- a proposal awaiting confirmation is a one-tap review,
    while no evidence at all has to be attributed from the text.
    """
    unresolved = [s for s in rows if _canonical(project, s.role_id) is None]
    held = _held_ids(project)
    with_proposal = [s for s in unresolved if s.id in held
                     and _clean_label(s.attribution.label if s.attribution else None)]
    return unresolved, with_proposal


def _line_requirement(project, segment):
    """The synthesis requirement a manifest freezes for one line."""
    from .pipeline import requirement_for
    role = _role(project, _canonical(project, segment.role_id))
    return requirement_for(project, segment, role=role, engine_id=role.tts_backend,
                           renderer_id=role.tts_backend).fingerprint()


def freeze_manifest(project, rows, required):
    if not all(approved(project, rid) for rid in required):
        raise CastingError("Required current speaking assignments await genuine human approval")
    unresolved, with_proposal = unresolved_scope(project, rows)
    if unresolved:
        # A full-book render is impossible while any scoped line has no speaker,
        # and rendering only the attributed lines would ship a book silently
        # missing the rest, so this refuses. It names the population because
        # "Unknown canonical speaking character" gives a human nothing to act on.
        raise CastingError(
            f"{len(unresolved)} of {len(rows)} source utterance(s) in scope have no resolved "
            f"speaking character: {len(with_proposal)} carry a proposed speaker awaiting "
            f"attribution review, {len(unresolved) - len(with_proposal)} have no speaker evidence "
            "and must be attributed from the text. Review speaker attribution before rendering "
            "the whole book."
        )
    manifest = {"schema": "casting-production-manifest-v1", "project_id": project.id,
        "source_binding": source_binding(project, rows), "utterance_ids": [s.id for s in rows],
        "cast": {rid: copy.deepcopy(_state(project)["roles"][rid]["assignment"]) for rid in required},
        "requirements": {s.id: _line_requirement(project, s) for s in rows},
        "created_at": lab.now()}
    manifest["id"] = "manifest_" + lab.digest(manifest)[:28]
    return manifest


def manifest_current(project_id, manifest):
    project = load_project(project_id)
    try:
        _, rows, required, _ = _scope(project, manifest["utterance_ids"])
        fresh = freeze_manifest(project, rows, required)
        return all(fresh[k] == manifest[k] for k in ("source_binding", "cast", "requirements"))
    except (CastingError, KeyError, ValueError):
        return False


def _sync_job_ids(project, receipt):
    """Keep the job's own list of lines equal to the production's, so its progress is not
    measured against lines that will never be recorded."""
    from dubbing_service.app.storage.jobs import get_job, mutate_job
    job_id = receipt.get("job_id")
    if not job_id or get_job(project.id, job_id) is None:
        return
    ids = list(receipt.get("utterance_ids") or [])
    keep = set(ids)

    def change(job):
        job.segment_ids = ids
        job.completed_segment_ids = [i for i in job.completed_segment_ids if i in keep]
        job.failed_segments = {k: v for k, v in job.failed_segments.items() if k in keep}

    mutate_job(project.id, job_id, change)


def replan_scope(project, receipt, utterance_ids, *, refreeze=False):
    """Recompute a production's frozen scope from ``utterance_ids`` less the skipped lines.

    A production's scope and source binding describe the book as it was when the receipt
    was made, so a decision about which lines are spoken has to move them, or the
    promotion loop would call the receipt ``source_changed`` and wait for a click. With
    ``refreeze`` a manifest is frozen now (a run that is already going); otherwise the
    next maintenance pass freezes one. The caller holds the project lock and saves.
    """
    skipped = skipped_ids(project)
    wanted, rows, required, _ = _scope(project, [i for i in utterance_ids if i not in skipped])
    if not wanted:
        raise CastingError("Every line of this book is skipped, so there is nothing to record")
    receipt.update(payload_hash=lab.digest(wanted), utterance_ids=wanted, required_roles=required,
                   source_binding=source_binding(project, rows))
    if not refreeze:
        receipt["manifest"] = None
        return None
    manifest = freeze_manifest(project, rows, required)
    receipt["manifest"] = manifest
    return manifest


def replan_automatic(project, *, reason, add=(), reopen=False):
    """Point the automatic production at the book as it now stands, after a decision.

    Without this a blocked receipt is terminal: it keeps its old scope, the promotion loop
    never looks at it again, and a decision that unblocks the book would change nothing
    until someone found a resume button. ``add`` lines are brought back into the scope
    (a restore); ``reopen`` lets a finished production be re-planned for them. A running
    or queued production is left alone. The caller holds the project lock and saves.
    """
    receipt = _state(project)["productions"].get(AUTOMATIC_PRODUCTION_ID)
    open_states = {"blocked", "awaiting_review", "source_changed", "failed"} | ({"succeeded"} if reopen else set())
    if not receipt or receipt.get("status") not in open_states:
        return False
    ids = list(receipt.get("utterance_ids") or [])
    if add:
        order = {s.id: index for index, s in enumerate(load_segments(project.id))}
        ids = sorted(set(ids) | set(add), key=lambda uid: order.get(uid, 10 ** 9))
    try:
        replan_scope(project, receipt, ids)
    except CastingError:
        return False
    receipt["total_attempts"] = max(receipt.get("total_attempts", receipt.get("attempts", 0)), receipt.get("attempts", 0))
    receipt.update(status="awaiting_review", attempts=0, reason=reason, last_failure_reason=None, last_line_failure=None)
    receipt.pop("reason_version", None)
    receipt.pop("completed_at", None)
    _sync_job_ids(project, receipt)
    return True


def cast_current(project, manifest):
    """Every approval and assignment the manifest froze still stands, on this project."""
    try:
        roles = _state(project)["roles"]
        return all(approved(project, rid) and roles[rid].get("assignment") == frozen
                   for rid, frozen in manifest["cast"].items())
    except (CastingError, KeyError, TypeError, AttributeError):
        return False


def manifest_still_current(project, segments, manifest, utterance_id=None):
    """Is the frozen manifest still true for the line about to be rendered or promoted?

    ``manifest_current`` re-reads and re-derives the whole book: parse the project and
    every segment, resolve every speaker, rebuild every requirement. That is seconds on a
    real book, and run three times before each of ~1,800 lines it was the slowest part of
    the whole render. This is the per-line guard, and it works on state the caller has
    already loaded: the frozen approvals and assignments, one digest over the whole
    book's source (so an edit anywhere still stops the run), and this one line's own
    requirement. The full check still guards every place audio is published: the start
    of a run, each chapter as it is assembled, and the final receipt.
    """
    try:
        if not cast_current(project, manifest):
            return False
        _, rows = scoped_rows(project, manifest["utterance_ids"], segments)
        if source_binding(project, rows) != manifest["source_binding"]:
            return False
        if utterance_id is None:
            return True
        line = next((s for s in rows if s.id == utterance_id), None)
        return line is not None and _line_requirement(project, line) == manifest["requirements"][utterance_id]
    except (CastingError, KeyError, ValueError, TypeError):
        return False


def render_request(project_id, *, request_id, utterance_ids):
    _id(request_id)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        wanted, rows, required, attention = _scope(project, utterance_ids)
        if not required:
            raise CastingError(_first_problem(attention)
                               or "No actual speaking characters in the requested scope")
        state = _state(project)
        state["enabled"] = True
        prior = state["productions"].get(request_id)
        digest = lab.digest(wanted)
        if prior and prior["payload_hash"] != digest:
            raise CastingError("Production request ID already binds a different source scope")
        if not prior:
            state["productions"][request_id] = {"request_id": request_id, "payload_hash": digest,
                "utterance_ids": wanted, "required_roles": required, "source_binding": source_binding(project, rows),
                "status": "awaiting_review", "job_id": "strict_" + lab.digest([project_id, request_id])[:28],
                "created_at": lab.now(), "attempts": 0, "total_attempts": 0,
                "resume_count": 0, "manifest_history": []}
            for rid in required:
                if not approved(project, rid) and not state["roles"].get(rid, {}).get("round"):
                    _new_round(project, rid)
            _save(project)
    return view(project_id)


def resume_request(project_id, request_id):
    _id(request_id)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        state = _state(project)
        receipt = state["productions"].get(request_id)
        changed = False
        if receipt:
            status = receipt.get("status")
            if status in {"queued", "running"}:
                raise CastingError("Production is already queued or running; wait for its current attempt to finish")
            if status in {"failed", "blocked", "source_changed", "awaiting_review", "retracted"}:
                _, rows, required, _ = _scope(project, receipt["utterance_ids"])
                if request_id == AUTOMATIC_PRODUCTION_ID:
                    # An automatic render is only ever started by the automatic
                    # start, against the cast it was created for: resuming one by
                    # hand after its cast was replaced would publish superseded
                    # voices. The human resumes an explicit production instead.
                    settled, _ = _cast_settled(project, required)
                    if not settled:
                        raise CastingError(
                            "This automatic production belongs to a cast that has since changed; "
                            "re-cast and approve the characters, and production restarts on its own")
                # `attempts` is the current explicit run's bounded budget;
                # `total_attempts` and manifest_history retain prior runs.
                receipt["total_attempts"] = max(
                    receipt.get("total_attempts", receipt.get("attempts", 0)),
                    receipt.get("attempts", 0))
                receipt["attempts"] = 0
                receipt["resume_count"] = receipt.get("resume_count", 0) + 1
                previous_failure = receipt.get("last_failure_reason")
                resume_reason = "Explicit resume requested after revalidation."
                if previous_failure:
                    resume_reason += f" Previous failure: {str(previous_failure)[:900]}"
                receipt.update(source_binding=source_binding(project, rows), required_roles=required,
                               status="awaiting_review", reason=resume_reason,
                               cast_binding=_cast_binding(project, required))
                changed = True
            elif status != "succeeded":
                raise CastingError("Production request is not in a resumable state")
        else:
            intent = state["intents"].get(request_id)
            if not intent or intent["status"] == "superseded":
                raise CastingError("Unknown or superseded casting request")
            intent["status"] = "pending"
            changed = True
        if changed:
            _save(project)
    return view(project_id)


def _publish_chapter(project_id, request_id, chapter_id, number):
    """Assemble one finished chapter, then announce it right away.

    A chapter that cannot be assembled is recorded and retried, never allowed to
    lose the rest of the book; delivery never gates the recording either, because
    the maintenance tick sends whatever is still due.
    """
    from . import autocast_book
    try:
        autocast_book.export_chapter(project_id, request_id, chapter_id, number)
    except Exception as exc:  # noqa: BLE001 - one chapter must not lose the rest of the book
        autocast_book.record_chapter_failure(project_id, request_id, chapter_id, exc)
        return False
    try:
        notify_ready(project_id)
    except Exception:  # noqa: BLE001 - a notification failure never invalidates a recorded chapter
        pass
    return True


class _LineFailure(CastingError):
    """The run stopped at one line, which is remembered so that a person can skip it."""

    def __init__(self, utterance_id, message):
        super().__init__(message)
        self.utterance_id = utterance_id


def _skip_line_in_run(project_id, request_id, utterance_id, *, reason, detail):
    """Leave one refused line out of a run that is going, and re-freeze what is left.

    The line and the new frozen scope are committed together or not at all, so the
    manifest the rest of the run is validated against always describes what is recorded.
    """
    with project_meta_lock(project_id):
        project = load_project(project_id)
        receipt = _state(project)["productions"][request_id]
        record_skips(project, [utterance_id], reason=reason, detail=detail, by="auto")
        manifest = replan_scope(project, receipt, receipt["utterance_ids"], refreeze=True)
        _save(project)
        _sync_job_ids(project, receipt)
    return manifest


class _RecordedLines:
    """Lines whose selected take is already valid, recognised without re-reading the book.

    After a restart the worker walks every line it recorded before. Handing each one to
    ``render_utterances`` re-reads the project and the whole book per line, which is the cost
    of the walk and nothing else. Here the same take check (``_resume_selected_take``, which
    hashes the audio and compares the synthesis requirement) and the same per-line manifest
    guard run on one snapshot, read-only. The snapshot is taken lazily, so a run that records
    new lines never pays for it, and it is read again whenever either file changes on disk.
    Anything that is not cleanly valid returns ``None`` and is rendered the usual way, and
    every chapter is still re-proved from its takes when it is published.
    """

    def __init__(self, project_id):
        self.project_id = project_id
        self.stamp = self.project = self.lines = None

    def _stamp(self):
        from dubbing_service.app.storage.projects import _project_meta_path
        from dubbing_service.app.storage.segments import _segments_path
        try:
            return tuple((stat.st_mtime_ns, stat.st_size)
                         for stat in (_project_meta_path(self.project_id).stat(), _segments_path(self.project_id).stat()))
        except OSError:
            return None

    def _fresh(self):
        stamp = self._stamp()
        if stamp is None:
            return False
        if stamp != self.stamp or self.project is None:
            self.stamp = None
            # The stamp is taken before the read, so a write that lands during it is seen next time.
            self.project = load_project(self.project_id)
            self.lines = {row.id: row for row in load_segments(self.project_id)}
            self.stamp = stamp
        return True

    def record_for(self, utterance_id, renderer, manifest):
        from . import pipeline
        try:
            if not self._fresh():
                return None
            segment = self.lines.get(utterance_id)
            if segment is None or not segment.selected_take_id:
                return None
            if not manifest_still_current(self.project, list(self.lines.values()), manifest, utterance_id):
                return None
            record, _ = pipeline._resume_selected_take(
                self.project, segment, renderer=renderer, root=pipeline.project_root(self.project_id),
                project_id=self.project_id, audit_missing_qa=False)
        except Exception:
            return None
        return record      # a record only comes back for a take that is valid as it stands


def _production_worker(project_id, request_id, *, renderer_factory=None):
    from dubbing_service.app.core.tts_operations import (
        try_register_running_batch,
        unregister_running_batch,
    )
    from dubbing_service.app.models.jobs import SynthesisJobStatus
    from dubbing_service.app.storage.jobs import (
        get_job,
        mark_job_active,
        record_job_segment,
    )

    from .pipeline import render_utterances
    from .provenance import spoken_text_for_segment
    from .renderer import BackendRenderer
    from .voice_lab_runtime import ensure_backend_ready
    project = load_project(project_id)
    receipt = _state(project)["productions"][request_id]
    job_id, lease = receipt["job_id"], None
    try:
        manifest = receipt["manifest"]
        if request_id == AUTOMATIC_PRODUCTION_ID:
            # The last line of defence: an automatic render exists to publish a
            # settled cast, so a worker that starts after the cast was replaced
            # must stop rather than publish the voices being superseded. A queued
            # receipt can outlive its cast when a re-cast begins between the tick
            # that queued it and the worker actually starting.
            settled, _ = _cast_settled(project, receipt.get("required_roles"))
            if not settled or receipt.get("status") == "retracted":
                with project_meta_lock(project_id):
                    current = load_project(project_id)
                    parked = _state(current)["productions"][request_id]
                    parked.update(status="retracted",
                                  reason="Superseded: the approved cast changed before this render started")
                    _save(current)
                    _job(current, job_id, kind="batch_synthesize", ids=parked["utterance_ids"],
                         message=parked["reason"], status=SynthesisJobStatus.AWAITING_REVIEW)
                return
        if not manifest_current(project_id, manifest):
            raise CastingError("Source or casting approval changed; strict production is parked for revalidation")
        # Worker readiness can require model load; no lease is held while parked.
        backends = {_role(project, rid).tts_backend for rid in receipt["required_roles"]}
        if renderer_factory is None:
            for backend in backends:
                ensure_backend_ready(backend, job_id=job_id, project_id=project_id)
        lease = try_register_running_batch(project_id)
        if lease is None:
            with project_meta_lock(project_id):
                project = load_project(project_id)
                _state(project)["productions"][request_id].update(status="awaiting_review", reason="Waiting for the existing render owner to release this project.")
                _save(project)
            return
        with project_meta_lock(project_id):
            project = load_project(project_id)
            receipt = _state(project)["productions"][request_id]
            receipt["attempts"] += 1
            receipt["total_attempts"] = max(
                receipt.get("total_attempts", receipt["attempts"] - 1), receipt["attempts"] - 1) + 1
            _save(project)
        _job(project, job_id, kind="batch_synthesize", ids=receipt["utterance_ids"],
             message="Rendering frozen human-approved cast", status=SynthesisJobStatus.RUNNING)
        from . import autocast_book
        # The book is read once for the whole run. Each line is then validated against the
        # frozen manifest by ``render_utterances`` on the state it reads for that line, so the
        # worker itself never re-reads the project or the book between lines.
        lines = {row.id: row for row in load_segments(project_id)}
        plan = autocast_book.chapter_plan(project, receipt["utterance_ids"], list(lines.values()))
        chapter_of = {uid: entry["chapter_id"] for entry in plan for uid in entry["utterance_ids"]}
        chapter_number = {entry["chapter_id"]: position + 1 for position, entry in enumerate(plan)}
        remaining = {entry["chapter_id"]: len(entry["utterance_ids"]) for entry in plan}
        recorded = dict.fromkeys(remaining, 0)
        budget = max(MIN_SKIP_BUDGET, math.ceil(MAX_SKIP_SHARE * len(receipt["utterance_ids"])))
        skipped_here = skipped_in_a_row = 0
        # Lines a person brought back after they were skipped: theirs to decide about.
        brought_back = set((project.metadata.get("autocast_import") or {}).get("policy_exempt") or [])
        autocast_book.start_pace(project_id, request_id)
        recorded_lines = _RecordedLines(project_id)

        def finish(chapter_id):
            # A chapter whose every line was skipped has nothing to record.
            if recorded[chapter_id] == 0:
                plan[:] = [entry for entry in plan if entry["chapter_id"] != chapter_id]
            else:
                _publish_chapter(project_id, request_id, chapter_id, chapter_number[chapter_id])

        def left_out(chapter_id):
            if chapter_id is not None:
                remaining[chapter_id] -= 1
                if remaining[chapter_id] == 0:
                    finish(chapter_id)

        for sid in receipt["utterance_ids"]:
            line_started = time.monotonic()
            job = get_job(project_id, job_id)
            if job.stop_requested or job.status == SynthesisJobStatus.CANCELLED:
                raise CastingError("Stopped at a safe utterance boundary; completed Takes are preserved")
            segment = lines.get(sid)
            if segment is None:
                raise CastingError("Source or human casting approval changed during rendering")
            chapter_id = chapter_of.get(sid)
            spoken = spoken_text_for_segment(project, segment)
            if sid not in brought_back and web_addresses.is_link_line(spoken):
                # A link is never read aloud, so the voice service is not even asked. Leaving it
                # out is not a refusal: it needs no automatic handling and does not count against
                # the budget that stops a run when the voice service itself seems to be failing.
                manifest = _skip_line_in_run(project_id, request_id, sid, reason=SKIP_WEB_ADDRESS,
                                             detail=WEB_ADDRESS_DETAIL)
                left_out(chapter_id)
                continue
            backend = _role(project, _canonical(project, segment.role_id)).tts_backend
            renderer = renderer_factory(backend) if renderer_factory else BackendRenderer(backend=backend,
                engine_id=backend, renderer_id=backend, project_root=safe_project_root(project_id))
            from .runtime_binding import bind_job
            if hasattr(renderer, "runtime_contract"):
                bind_job(project_id, job_id, renderer, [lines[uid] for uid in receipt["utterance_ids"] if uid in lines], project.roles, project.character_cast.role_aliases)
            problem = None
            result = recorded_lines.record_for(sid, renderer, manifest) if segment.selected_take_id else None
            for _ in range(LINE_ATTEMPTS if result is None else 0):
                result = render_utterances(project_id, [sid], renderer=renderer, resume=True,
                                          promote=True, run_qa=True, casting_manifest=manifest, job_id=job_id)[0]
                problem = line_problem(result)
                if problem is None or problem[0] == "other":
                    break
            if problem is not None and problem[0] == "other":
                raise _LineFailure(sid, problem[1])
            if problem is not None:
                kind, detail = problem
                if kind == SKIP_UNSPEAKABLE and sid not in brought_back and web_addresses.has_web_address(spoken):
                    # A sentence with an address in it the voice would not read: the address is
                    # the reason, so this is left out like a link, whatever automatic handling says.
                    manifest = _skip_line_in_run(project_id, request_id, sid, reason=SKIP_WEB_ADDRESS,
                                                 detail=f"{WEB_ADDRESS_DETAIL}; the voice would not read the line with it. {detail}")
                    left_out(chapter_id)
                    continue
                if not autocast_book.automatic_handling(load_project(project_id)):
                    # Skipping by itself is what automatic handling means. With it off the run
                    # stops here and names the line, so the person decides.
                    raise _LineFailure(sid, f"The voice service would not record this line, and automatic handling is off. {detail}")
                if skipped_in_a_row >= MAX_SKIPS_IN_A_ROW or skipped_here >= budget:
                    raise _LineFailure(sid, (
                        f"{skipped_in_a_row} lines in a row were refused" if skipped_in_a_row >= MAX_SKIPS_IN_A_ROW
                        else f"{skipped_here} lines have already been skipped in this run")
                        + f", so the run stopped instead of skipping more; the voice service may not be healthy. Latest: {detail}")
                manifest = _skip_line_in_run(project_id, request_id, sid, reason=kind, detail=detail)
                skipped_here += 1
                skipped_in_a_row += 1
                left_out(chapter_id)
                continue
            skipped_in_a_row = 0
            record_job_segment(project_id, job_id, sid, success=True)
            if getattr(result, "dispatched", True):
                autocast_book.record_pace(project_id, request_id, characters=len(segment.text),
                                          seconds=time.monotonic() - line_started)
            if chapter_id is not None:
                remaining[chapter_id] -= 1
                recorded[chapter_id] += 1
                if remaining[chapter_id] == 0:
                    finish(chapter_id)
        # A chapter that could not be assembled is retried once before the book is
        # called finished, so nothing is ever reported done with a chapter missing.
        published = ((_state(load_project(project_id))["productions"][request_id]).get("chapters") or {})
        for entry in plan:
            if (published.get(entry["chapter_id"]) or {}).get("status") != "ready":
                _publish_chapter(project_id, request_id, entry["chapter_id"], chapter_number[entry["chapter_id"]])
        published = ((_state(load_project(project_id))["productions"][request_id]).get("chapters") or {})
        missing = [entry for entry in plan if (published.get(entry["chapter_id"]) or {}).get("status") != "ready"]
        if missing:
            first = missing[0]
            reason = (published.get(first["chapter_id"]) or {}).get("reason") or "assembly failed"
            raise CastingError(f"Chapter {chapter_number[first['chapter_id']]} ({first['title']}) could not be assembled: {reason}")
        with project_meta_lock(project_id):
            project = load_project(project_id)
            if not manifest_current(project_id, manifest):
                raise CastingError("Cast changed at final publication boundary")
            receipt = _state(project)["productions"][request_id]
            left_out = len(skipped_lines(project))
            receipt.update(status="succeeded", completed_at=lab.now(), last_line_failure=None,
                           reason=f"All {len(plan)} chapters are recorded and ready to listen to"
                                  + (f" ({left_out} line{'s' if left_out != 1 else ''} skipped)." if left_out else "."))
            _save(project)
            _job(project, job_id, kind="batch_synthesize", ids=receipt["utterance_ids"],
                 message=receipt["reason"], status=SynthesisJobStatus.SUCCEEDED)
    except Exception as exc:  # noqa: BLE001 - a worker must record its failure, not die unrecorded
        with project_meta_lock(project_id):
            project = load_project(project_id)
            receipt = _state(project)["productions"][request_id]
            failure = str(exc)[:1200]
            line = getattr(exc, "utterance_id", None)
            receipt.update(status="awaiting_review" if lease else "blocked",
                           reason=failure, last_failure_reason=failure,
                           last_line_failure={"utterance_id": line, "error": failure[:300], "at": lab.now()} if line else None)
            _save(project)
            _job(project, job_id, kind="batch_synthesize", ids=receipt["utterance_ids"],
                 message=receipt["reason"], status=SynthesisJobStatus.AWAITING_REVIEW)
    finally:
        if lease:
            unregister_running_batch(lease)
        mark_job_active(project_id, job_id, False)


def maintenance_tick(*, schedule=None, project_ids=None):
    # Both HTTP background tasks and lifespan recovery call this owner. Claim one
    # global dispatch under a process lock, then release it before model work.
    # Existing pending-run owner performs slow research outside the global
    # dispatcher lock. Concurrent ticks skip its incomplete durable request.
    ids = list(project_ids) if project_ids is not None else [p.parent.name for p in get_config().projects_root_abs.glob("*/project.json")]
    for project_id in ids:
        start_pending_autocast(project_id)
    actions = []
    def collect(fn, *args):
        actions.append((fn, args))
    with _DISPATCH_LOCK:
        result = _maintenance_dispatch(schedule=collect, project_ids=ids)
    for fn, args in actions:
        if schedule:
            schedule(fn, *args)
        else:
            fn(*args)
    return result


def _maintenance_dispatch(*, schedule=None, project_ids=None):
    """Bounded existing-owner recovery pass; durable intents close enqueue gaps."""
    from dubbing_service.app.models.jobs import SynthesisJobStatus
    from dubbing_service.app.storage.jobs import (
        is_job_active,
        load_jobs,
        mark_job_active,
        try_mark_job_active,
    )
    ids = project_ids if project_ids is not None else [p.parent.name for p in get_config().projects_root_abs.glob("*/project.json")]
    all_ids = [p.parent.name for p in get_config().projects_root_abs.glob("*/project.json")]
    if any(is_job_active(pid, j.id) for pid in all_ids for j in load_jobs(pid) if j.id.startswith(("cast_", "strict_"))):
        return []
    actions = []
    for project_id in ids:
        project = load_project(project_id)
        # Durable resume: a run recorded pending by an import (or left running
        # by a restart) establishes its scope and rounds here, outside the
        # per-project lock that prepare() takes for itself.
        run = project.character_cast.autocast
        if run is not None and run.status in {"pending", "running"} and run.request_id not in project.voice_lab.casting.get("requests", {}):
            continue  # Scope/research is still owned by start_pending_autocast.
        # Lines the person already said how to handle are decided before anything reads the
        # production, so a book never reaches "needs attention" over them. The pass must
        # outlive a failure here, and the failure must not be silent.
        try:
            from . import autocast_book
            if autocast_book.apply_policy(project_id, project):
                project = load_project(project_id)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("Automatic handling of unattributed lines failed for %s", project_id)
        if (not _state(project)["intents"] and not _state(project)["productions"]
                and project.character_cast.autocast is None):
            continue
        with project_meta_lock(project_id):
            project = load_project(project_id)
            state = _state(project)
            # Decide the automatic production's eligibility before anything is
            # promoted: a receipt belonging to a cast that is being replaced has to
            # be retracted here, or the promotion loop below would queue a render of
            # the superseded voices in the same tick.
            _automatic_production(project)
            for request_id, receipt in state["productions"].items():
                if receipt["status"] in {"succeeded", "source_changed", "failed", "retracted"}:
                    continue
                if receipt["status"] == "blocked":
                    # A block is terminal for the promotion loop, so a reason
                    # written by older code would otherwise keep describing the code
                    # that first blocked it rather than the current diagnosis.
                    # Re-derive once per reason version; the notification key is
                    # stable, so a corrected reason never re-announces an event the
                    # human has already been sent.
                    if receipt.get("reason_version") == BLOCK_REASON_VERSION:
                        continue
                    try:
                        _, rows, required, _ = _scope(project, receipt["utterance_ids"])
                    except CastingError as exc:
                        receipt.update(status="source_changed", reason=str(exc)[:900])
                        _save(project)
                        continue
                    try:
                        freeze_manifest(project, rows, required)
                    except CastingError as exc:
                        receipt.update(reason=str(exc)[:900], reason_version=BLOCK_REASON_VERSION)
                        _save(project)
                        continue
                    # The scope became freezable, so the block no longer holds. Clear
                    # it and fall through to the normal promotion path rather than
                    # duplicating that path here, so the render starts on this tick.
                    receipt.update(status="awaiting_review", manifest=None,
                                   reason="Blocked scope became freezable; production resumed",
                                   reason_version=BLOCK_REASON_VERSION)
                    _save(project)
                try:
                    _, rows, required, _ = _scope(project, receipt["utterance_ids"])
                except CastingError as exc:
                    receipt.update(status="source_changed", reason=str(exc))
                    _save(project)
                    continue
                if request_id == AUTOMATIC_PRODUCTION_ID:
                    settled, _ = _cast_settled(project, required)
                    if not settled:
                        # Belt and braces: an automatic render only ever runs for a
                        # settled cast, whatever retracted the receipt.
                        _job(project, receipt["job_id"], kind="batch_synthesize",
                             ids=receipt["utterance_ids"],
                             message="The approved cast is being re-cast; production restarts when it settles",
                             status=SynthesisJobStatus.AWAITING_REVIEW)
                        continue
                if source_binding(project, rows) != receipt["source_binding"]:
                    receipt.update(status="source_changed", reason="Source changed while parked. Resume explicitly to reconcile the current scope.")
                    _save(project)
                    continue
                if not all(approved(project, rid) for rid in required):
                    _job(project, receipt["job_id"], kind="batch_synthesize", ids=receipt["utterance_ids"],
                        message="Awaiting required human casting feedback; no GPU lease held", status=SynthesisJobStatus.AWAITING_REVIEW)
                    continue
                if receipt["attempts"] >= 3:
                    detail = receipt.get("last_failure_reason") or receipt.get("reason")
                    reason = "Bounded production attempts exhausted"
                    if detail and detail not in reason:
                        reason += f"; last failure: {detail[:900]}"
                    receipt.update(status="failed", reason=reason)
                    _save(project)
                    continue
                if try_mark_job_active(project_id, receipt["job_id"]):
                    try:
                        manifest = freeze_manifest(project, rows, required)
                    except CastingError as exc:
                        # Freezing fails when an utterance in scope has no resolved
                        # speaker, which a partially attributed project always has.
                        # Retrying cannot fix that and would raise on every tick, so
                        # the receipt records the block and the human's attention
                        # event covers it: rendering what can be rendered would
                        # silently ship a book missing those lines.
                        receipt.update(status="blocked", reason=str(exc)[:900],
                                       reason_version=BLOCK_REASON_VERSION)
                        _save(project)
                        _job(project, receipt["job_id"], kind="batch_synthesize",
                             ids=receipt["utterance_ids"],
                             message=receipt["reason"], status=SynthesisJobStatus.AWAITING_REVIEW)
                        # Nothing is running, so the claim taken above is released. Left set
                        # it made every later pass (for every project) stand down as if a
                        # render were in flight, and a decision that unblocked this book
                        # could not restart it until the service itself was restarted.
                        mark_job_active(project_id, receipt["job_id"], False)
                        continue
                    receipt["manifest_history"].append(manifest)
                    receipt.update(manifest=manifest, status="queued")
                    _save(project)
                    _job(project, receipt["job_id"], kind="batch_synthesize", ids=receipt["utterance_ids"],
                        message="Current human cast approved; queued", status=SynthesisJobStatus.QUEUED)
                    actions.append((_production_worker, project_id, request_id, receipt["job_id"]))
                    break
            # Promote a queued replacement only once the round the human was
            # reviewing has nothing left to score. This is the whole point of the
            # deferral: the reviewed batch stays intact while it is being scored,
            # and the replacement arrives as its own later round.
            _replacement_dispatch(project)
            # A replacement round may have been created just above, so the automatic
            # production is re-evaluated after it: this is the step that makes
            # "answer the casting review" the last manual action.
            _automatic_production(project)
            # Dispatch one round per tick, sequential candidate generation on the GPU.
            if not actions and not lab.production_busy():
                for intent_id, intent in state["intents"].items():
                    if intent["status"] in {"pending", "running"} and try_mark_job_active(project_id, intent_id):
                        intent["status"] = "running"
                        _save(project)
                        _job(project, intent_id, kind="voice_lab_listening", ids=[], message="Preparing bounded casting auditions")
                        actions.append((_round_worker, project_id, intent_id, intent_id))
                        break
        # Importing a proposed prior voice re-enters the project lock inside the
        # library owner, so it must run with this project's lock released.
        _prepare_prior_imports(project_id)
        if len(actions) >= 1:
            break
    for fn, project_id, argument, job_id in actions:
        try:
            if schedule:
                schedule(fn, project_id, argument)
            else:
                fn(project_id, argument)
        except Exception:
            mark_job_active(project_id, job_id, False)
            raise
    for project_id in ids:
        notify_ready(project_id)
    return [{"project_id": pid, "job_id": jid} for _, pid, _, jid in actions]


def _notification_due(notification):
    """Whether this durable event may attempt delivery now.

    ``blocked`` is a configuration problem rather than a provider failure, so it
    is retried on a long interval and clears itself once the configuration is
    fixed; a provider failure is bounded and then stops.
    """
    state = notification.get("state")
    attempts = notification.get("attempts", 0)
    if type(attempts) is not int or attempts < 0:
        return False
    if state not in {"pending", "failed", "skipped", "blocked"}:
        return False
    due = notification.get("retry_at") or notification.get("next_retry_at")
    if not due:
        # No retry clock means this event has never been scheduled; a legacy
        # record left by an older build must still be able to clear itself.
        return True
    try:
        return datetime.fromisoformat(due) <= datetime.now(UTC)
    except (TypeError, ValueError):
        return False


def _delivery_status(outcome):
    """One accepted/failed/ambiguous word from a sender outcome, never a guess."""
    status = outcome.get("previous_state") if outcome.get("duplicate") else outcome.get("state")
    return status if status in {"accepted", "delivery_unknown", "failed", "skipped"} else "delivery_unknown"


def _retry_at(attempts):
    """Quick early retries, then a steady capped interval that never gives up."""
    seconds = 60 * max(1, attempts) if attempts < MAX_NOTIFICATION_ATTEMPTS else EXHAUSTED_RETRY_SECONDS
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def review_base_url(config):
    """The configured private review origin a phone can actually open."""
    raw = config.get("listening_base_url")
    if not isinstance(raw, str) or not raw.strip():
        raise MobileReviewLinkUnavailable(
            "MOBILE REVIEW LINK NOT CONFIGURED: set listening_base_url to your private Tailscale "
            "Serve URL in the audiobook-delivery configuration")
    parsed = urlsplit(raw.strip())
    host = (parsed.hostname or "").strip()
    if parsed.scheme != "https" or not host:
        raise MobileReviewLinkUnavailable(
            "MOBILE REVIEW LINK NOT CONFIGURED: listening_base_url must be an https:// address")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if loopback:
        raise MobileReviewLinkUnavailable(
            "MOBILE REVIEW LINK NOT CONFIGURED: listening_base_url is a localhost address the phone "
            "cannot reach; configure the private Tailscale Serve URL instead")
    return raw.strip().rstrip("/")


def review_link(base_url, project_id, *, role_id=None, view=None, chapter=None):
    """A Voice Lab deep link that lands on the exact review the human must do."""
    query = {"project": project_id}
    if role_id:
        query["character"] = role_id
    elif chapter:
        query["view"] = "casting"
        query["chapter"] = chapter
    elif view:
        query["view"] = view
    return base_url.rstrip("/") + "/voice-lab?" + urlencode(query)


def casting_progress(project):
    """Authoritative casting counts, derived once from the canonical state.

    The notification layer reads these instead of projecting a second copy of
    the casting pipeline, so a count can never disagree with the review it
    describes.
    """
    state = _state(project)
    required = _required_role_ids(project)
    ready = _ready_rounds(project)
    approved_ids = {role_id for role_id in required if approved(project, role_id)}
    ready_ids = {row["role_id"] for row in ready} - approved_ids
    blocked = {role_id for role_id in required
               if ((state["roles"].get(role_id) or {}).get("round") or {}).get("status") == "blocked"
               and role_id not in approved_ids}
    running = {role_id for role_id in required
               if ((state["roles"].get(role_id) or {}).get("round") or {}).get("status")
               in {"pending", "running"} and role_id not in approved_ids}
    return {"characters_total": len(required), "characters_casting": len(running),
            "characters_review_ready": len(ready_ids), "characters_approved": len(approved_ids),
            "characters_needs_attention": len(blocked),
            "auditions_ready": sum(len(row["candidates"]) for row in ready)}


def _round_ready_at(round_):
    try:
        return datetime.fromisoformat(round_.get("completed_at") or "")
    except (TypeError, ValueError):
        return None


def _role_name(project, role_id):
    """Display name for a casting round's role, falling back to its stable id.

    Notification copy is not casting authority: a renamed or removed role must
    never fail delivery of a review the human is already waiting on.
    """
    role = next((candidate for candidate in project.roles if candidate.id == role_id), None)
    return role.name if role is not None else role_id


def _partial_ready_at(candidates):
    """When the newest playable audition of an unfinished round became playable."""
    stamps = []
    for candidate in candidates:
        try:
            stamps.append(datetime.fromisoformat(candidate.get("ready_at") or ""))
        except (TypeError, ValueError):
            return None
    return max(stamps) if stamps else None


def _ready_rounds(project, *, include_partial=False):
    """Characters whose playable auditions a human can genuinely review now.

    A round only reaches ``awaiting_review`` after its candidate audio exists and
    passed technical validation, so queued work never becomes a notification.
    With ``include_partial`` (notifications only), a round that is still producing a
    sibling also counts once its playable auditions have been waiting longer than
    ``PARTIAL_REVIEW_SECONDS``: the human is waiting on those, not on the straggler.
    """
    ready = []
    for role_id, item in _state(project)["roles"].items():
        round_ = item.get("round") or {}
        status = round_.get("status")
        candidates = [c for c in round_.get("candidates", [])
                      if c.get("status") == "ready" and c.get("audio_sha256")]
        if not candidates:
            continue
        if status == "awaiting_review":
            ready_at = _round_ready_at(round_)
        elif include_partial and status in {"pending", "running"} and not approved(project, role_id):
            ready_at = _partial_ready_at(candidates)
            if ready_at is None or (datetime.now(UTC) - ready_at).total_seconds() < PARTIAL_REVIEW_SECONDS:
                continue
        else:
            continue
        ready.append({"role_id": role_id, "round": round_, "candidates": candidates,
                      "notification": round_.get("notification") or {},
                      "event": REPLACEMENT_REVIEW_READY if round_.get("replacement") else CHARACTER_REVIEW_READY,
                      "ready_at": ready_at})
    return ready


def _defer_for_work_in_flight(state, due):
    """Coalesce only while other auditions are still being produced.

    Aggregation is bounded by a durable deadline, never by a sleep: a lone
    finished character is sent promptly, and a character that has already waited
    out the window is never held behind unrelated slow work.
    """
    rounds = [item.get("round") or {} for item in state["roles"].values()]
    if not any(round_.get("status") in {"pending", "running"} for round_ in rounds):
        return False
    now = datetime.now(UTC)
    for row in due:
        started = row["ready_at"]
        if started is None or (now - started).total_seconds() >= COALESCE_SECONDS:
            return False
    return True


def _project_notifications(state):
    return state.setdefault("notifications", {})


def _blocked_rounds(project_id, round_ids, reason):
    return _record_round_notification(project_id, round_ids, {
        "state": "blocked", "reason": reason, "updated_at": lab.now(),
        "next_retry_at": (datetime.now(UTC) + timedelta(seconds=BLOCKED_RETRY_SECONDS)).isoformat()})


def _blocked_event(project_id, key, reason):
    with project_meta_lock(project_id):
        project = load_project(project_id)
        _project_notifications(_state(project)).setdefault(key, {}).update(
            state="blocked", reason=reason, updated_at=lab.now(),
            next_retry_at=(datetime.now(UTC) + timedelta(seconds=BLOCKED_RETRY_SECONDS)).isoformat())
        _save(project)


def _record_round_notification(project_id, round_ids, values):
    with project_meta_lock(project_id):
        project = load_project(project_id)
        for item in _state(project)["roles"].values():
            if (item.get("round") or {}).get("id") in set(round_ids):
                item["round"].setdefault("notification", {}).update(values)
        _save(project)


def _required_role_ids(project):
    """The in-scope cast: the current run's characters, else the derived scope."""
    state = _state(project)
    run = project.character_cast.autocast
    known = [role_id for role_id in (run.required_role_ids if run else []) if role_id in state["roles"]]
    return known or list(state["roles"])


def _attention_signals(project):
    """Cheap durable reasons a human decision may be required.

    Reading the source scope means parsing every segment, so it is only reached
    when one of these already says a decision is plausibly outstanding.
    """
    state = _state(project)
    run = project.character_cast.autocast
    if run is not None and run.status == "needs_attention":
        return True
    if _held_ids(project):
        return True
    if any((item.get("round") or {}).get("status") == "blocked" for item in state["roles"].values()):
        return True
    return any(intent.get("status") == "failed" for intent in state["intents"].values())


def _needs_attention(project):
    """The one actionable message for casting that genuinely cannot continue."""
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    state = _state(project)
    reasons = []
    subjects = []
    run = project.character_cast.autocast
    if run is not None and run.status == "needs_attention":
        reasons.append(run.reason or "Automatic casting stopped and needs a human decision")
        subjects.append("automation")
    for role_id, item in sorted(state["roles"].items()):
        if (item.get("round") or {}).get("status") == "blocked" and not approved(project, role_id):
            reasons.append(f"{_role(project, role_id).name} has no usable non-colliding audition")
            subjects.append(role_id)
    failed = sorted(intent_id for intent_id, intent in state["intents"].items()
                    if intent.get("status") == "failed")
    if failed:
        reasons.append("Audition generation failed after its bounded retries")
        subjects.extend(failed)
    try:
        unresolved = _attention_characters(project, _scope(project)[3])
    except CastingError:
        unresolved = {}
    if unresolved:
        reasons.append(f"{sum(len(v) for v in unresolved.values())} utterance(s) need speaker "
                       "attribution review before those characters can be cast")
        subjects.extend(sorted(unresolved))
    if not reasons:
        return None
    return {"event": CASTING_NEEDS_ATTENTION,
            "key": "attention:" + lab.digest([project.id, sorted(set(subjects)),
                                              AUTOCAST_PIPELINE_VERSION])[:32],
            "binding": [lab.digest([project.id, CASTING_NEEDS_ATTENTION,
                                    sorted(set(subjects)), AUTOCAST_PIPELINE_VERSION])],
            "reason": "; ".join(reasons)[:400]}


def notification_events(project):
    """Every human-review event this casting state currently implies.

    Derived only from durable casting state: a browser refresh, opening Voice
    Lab or a server restart cannot add to it, and each event key is stable for
    the exact revision that produced it.
    """
    from dubbing_service.app.models.character import AUTOCAST_PIPELINE_VERSION
    events = [{"event": row["event"], "key": "round:" + row["round"]["id"], "role_id": row["role_id"],
               "role_name": _role_name(project, row["role_id"]), "round": row["round"],
               "candidates": row["candidates"]}
              for row in _ready_rounds(project, include_partial=True)]
    if _attention_signals(project):
        attention = _needs_attention(project)
        if attention:
            events.append(attention)
    settled, required = _cast_settled(project)
    if settled:
        revision_id, sha256 = _current_source_identity(project)
        # The cast is part of the completion identity. Keyed only by source and
        # pipeline version, a re-cast that produced a different approved cast would
        # find the earlier completion already delivered and never announce its own.
        cast = _cast_binding(project, required)
        events.append({
            "event": CASTING_COMPLETE,
            "key": "complete:" + lab.digest([project.id, revision_id, sha256,
                                             AUTOCAST_PIPELINE_VERSION, cast])[:32],
            "binding": [lab.digest([project.id, CASTING_COMPLETE, revision_id, sha256,
                                    AUTOCAST_PIPELINE_VERSION, cast])],
            "total": len(required),
        })
    events.extend(_chapter_events(project))
    production = _production_event(project)
    if production:
        events.append(production)
    return events


def notify_ready(project_id):
    """Deliver every durable Auto Cast review event through the existing ledger.

    Casting and delivery are separate durable concerns: this projects canonical
    casting state into Pushover batches, and a delivery failure never invalidates
    an audition, a rating or an approval.
    """
    from .book_creator_notifications import _private_config
    from .delivery import notify
    review = None
    with project_meta_lock("casting_notify_" + project_id):
        project = load_project(project_id)
        if project.voice_lab.scope == "regression_fixture":
            return {"state": "disabled_fixture"}
        state = _state(project)
        due = [row for row in _ready_rounds(project, include_partial=True)
               if _notification_due(row["notification"])]
        if due and not _defer_for_work_in_flight(state, due):
            round_ids = [row["round"]["id"] for row in due]
            try:
                config, recipient = _private_config()
                from .delivery.server import shared_lab_project
                if not shared_lab_project(config, project_id):
                    raise MobileReviewLinkUnavailable(
                        "Private review sharing has not been authorized for this project")
                for row in due:
                    for record in row["candidates"]:
                        lab._verified_record(project, record["key"], record["audio_sha256"])
                base = review_base_url(config)
            except (ValueError, OSError) as exc:
                _blocked_rounds(project_id, round_ids, str(exc))
                review = {"state": "blocked", "reason": str(exc)}
            else:
                review = {"state": _send_review_batch(project_id, due, config, recipient, base, notify)}
        terminal = _send_terminal_events(project_id, notify)
    result = review or {"state": "idle"}
    return {**result, "terminal": terminal} if terminal else result


def _casting_rows(project, ready):
    rows = []
    for role_id, round_, candidates in ready:
        direction = round_.get("direction") or {}
        rows.append({"role_id": role_id, "round_id": round_.get("id"),
                     "name": _role_name(project, role_id), "options": len(candidates),
                     "audio_sha256": max(c["audio_sha256"] for c in candidates),
                     "direction": (direction.get("requested") or [""])[0]
                                  if round_.get("replacement") else ""})
    return rows


def _send_review_batch(project_id, due, config, recipient, base, notify):
    """One coalesced review push for every character due right now."""
    from pathlib import Path
    project = load_project(project_id)
    ready = [(row["role_id"], row["round"], row["candidates"]) for row in due]
    characters = _casting_rows(project, ready)
    batch_id = "casting_" + lab.digest([row["round"]["id"] for row in due])[:32]
    url = (review_link(base, project_id, role_id=characters[0]["role_id"]) if len(characters) == 1
           else review_link(base, project_id, view="casting"))
    batch = {"kind": "casting_review", "project_id": project_id, "batch_id": batch_id,
             "recipient_id": "brian", "book_title": project.name, "characters": characters,
             "event": due[0]["event"] if len(due) == 1 else CHARACTER_REVIEW_READY,
             "characters_total": len(_required_role_ids(project)),
             "progress": casting_progress(project)}
    attempts = max(row["notification"].get("attempts", 0) for row in due) + 1
    retry_at = _retry_at(attempts)
    round_ids = [row["round"]["id"] for row in due]
    # The pending durable record precedes the network: the existing ledger then
    # handles ambiguous sends and blocks a duplicate after a process crash.
    _record_round_notification(project_id, round_ids, {"state": "pending", "batch_id": batch_id,
        "event": batch["event"], "attempts": attempts, "last_attempt_at": lab.now(),
        "next_retry_at": retry_at, "updated_at": lab.now()})
    try:
        outcome = notify.send_notification(batch, recipient,
            state_dir=safe_project_root(project_id) / "delivery",
            transport_root=Path(config["notification_transport_root"]),
            listening_url=url)
    except Exception:  # noqa: BLE001 - an ambiguous send must not invalidate the durable record
        outcome = {"state": "delivery_unknown"}
    status = _delivery_status(outcome)
    _record_round_notification(project_id, round_ids, {"state": status, "batch_id": batch_id,
        "event": batch["event"], "attempts": attempts, "review_url": url,
        "next_retry_at": retry_at if status in {"failed", "skipped"} else None,
        "reason": outcome.get("detail"), "updated_at": lab.now(), **_outcome_facts(outcome, status),
        "device_delivery": "unobserved", "page_opened": "unobserved",
        "audio_playback": "unobserved", "human_feedback": "pending"})
    return status


def _outcome_facts(outcome, status):
    """Safe provider facts for diagnostics: never a credential, never a guess."""
    return {"delivery_status": status, "http_status": outcome.get("http_status"),
            "provider_request_id": outcome.get("provider_request_id"),
            "last_attempt_at": lab.now(),
            "delivered_at": lab.now() if status == "accepted" else None}


def _send_terminal_events(project_id, notify):
    """At most one completion push and one actionable needs-attention push."""
    from pathlib import Path

    from .book_creator_notifications import _private_config
    from .delivery.server import shared_lab_project
    outcomes = {}
    project = load_project(project_id)
    if project.voice_lab.scope == "regression_fixture":
        return outcomes
    for event in notification_events(project):
        if event["event"] not in {CASTING_COMPLETE, CASTING_NEEDS_ATTENTION, PRODUCTION_READY, CHAPTER_READY}:
            continue
        record = _project_notifications(_state(project)).get(event["key"]) or {"state": "pending"}
        if not _notification_due(record):
            continue
        try:
            config, recipient = _private_config()
            if not shared_lab_project(config, project_id):
                raise MobileReviewLinkUnavailable(
                    "Private review sharing has not been authorized for this project")
            url = review_link(review_base_url(config), project_id, view="casting",
                              chapter=event.get("chapter_id"))
        except (ValueError, OSError) as exc:
            _blocked_event(project_id, event["key"], str(exc))
            outcomes[event["event"]] = "blocked"
            continue
        batch_id = "casting_" + lab.digest([event["key"]])[:32]
        batch = {"kind": "casting_review", "project_id": project_id, "batch_id": batch_id,
                 "recipient_id": "brian", "event": event["event"], "book_title": project.name,
                 "characters": [], "binding": event["binding"], "reason": event.get("reason"),
                 "status": event.get("status"), "manifest_id": event.get("manifest_id"),
                 "utterance_count": event.get("utterance_count"),
                 "chapters_ready": event.get("chapters_ready"), "chapters_total": event.get("chapters_total"),
                 "chapter_title": event.get("chapter_title"), "chapter_number": event.get("chapter_number"),
                 "duration_seconds": event.get("duration_seconds"),
                 "characters_total": event.get("total") or len(_required_role_ids(project))}
        attempts = record.get("attempts", 0) + 1
        retry_at = _retry_at(attempts)
        with project_meta_lock(project_id):
            current = load_project(project_id)
            _project_notifications(_state(current))[event["key"]] = {
                "event": event["event"], "state": "pending", "batch_id": batch_id,
                "attempts": attempts, "last_attempt_at": lab.now(),
                "next_retry_at": retry_at, "updated_at": lab.now()}
            _save(current)
        try:
            outcome = notify.send_notification(batch, recipient,
                state_dir=safe_project_root(project_id) / "delivery",
                transport_root=Path(config["notification_transport_root"]),
                listening_url=url)
        except Exception:  # noqa: BLE001 - an ambiguous send must not invalidate the durable record
            outcome = {"state": "delivery_unknown"}
        status = _delivery_status(outcome)
        with project_meta_lock(project_id):
            current = load_project(project_id)
            _project_notifications(_state(current))[event["key"]] = {
                "event": event["event"], "state": status, "batch_id": batch_id,
                "attempts": attempts, "reason": outcome.get("detail"), "updated_at": lab.now(),
                "review_url": url, **_outcome_facts(outcome, status),
                "next_retry_at": retry_at if status in {"failed", "skipped"} else None}
            _save(current)
        outcomes[event["event"]] = status
    return outcomes



def qualitative_rejection(project_id, *, event_id, role_id, candidate_id, key,
                          audio_sha256, raw_feedback, requested_performer):
    """Record exact human qualitative rejection; no inferred stars or synthesis."""
    _id(event_id)
    command = {"event_id": event_id, "role_id": role_id, "candidate_id": candidate_id,
               "key": key, "audio_sha256": audio_sha256, "raw_feedback": raw_feedback,
               "requested_performer": requested_performer, "kind": "qualitative_rejection"}
    command_hash = lab.digest(command)
    with project_meta_lock(project_id):
        project = load_project(project_id)
        state = _state(project)
        previous = state["events"].get(event_id)
        if previous:
            if previous.get("command_hash") != command_hash:
                raise CastingError("Event identity already belongs to different feedback")
            return copy.deepcopy(previous["result"])
        record = lab._verified_record(project, key, audio_sha256)
        item = state["roles"].get(role_id) or {}
        candidates = [candidate for round_ in [*item.get("history", []), item.get("round") or {}]
                      for candidate in round_.get("candidates", [])
                      if candidate.get("candidate_id") == candidate_id and candidate.get("key") == key
                      and candidate.get("audio_sha256") == audio_sha256]
        if not candidates or record.get("role_id") != role_id:
            raise CastingError("Qualitative feedback must name this character's exact heard candidate")
        result = {"saved": True, "event_id": event_id, "qualitatively_rejected": True,
                  "rating_invented": False, "assignment_changed": False, "tts_queued": False,
                  "requested_performer": requested_performer}
        event = {**command, "command_hash": command_hash, "actor": "brian", "origin": "human",
                 "created_at": lab.now(), "voice_id": record["voice_id"], "voice_version": record["version"],
                 "result": result}
        state["events"][event_id] = event
        for candidate in candidates:
            candidate["human_fit"] = {"status": "rejected", "event_id": event_id,
                                      "reason": raw_feedback, "stars": None}
        item.setdefault("rejections", []).append({"key": key, "version": record["version"],
            "event_id": event_id, "feedback": raw_feedback, "rating": None,
            "effective_identity": candidates[0].get("identity_id"), "kind": "qualitative_rejection"})
        project.metadata.setdefault("voice_lab_source_targets", {})[role_id] = {
            "performer": requested_performer, "requested_by_event": event_id,
            "delivery_requirement": "Calm sustained narration/storytelling; older first-film Dumbledore voice era",
            "source_status": "eligible_reference_required", "supersedes": candidates[0].get("inspiration")}
        if (item.get("round") or {}).get("candidates") == candidates:
            item["round"]["reason"] = "Human rejected this delivery; requested replacement performer: " + requested_performer
        _save(project)
    return result
