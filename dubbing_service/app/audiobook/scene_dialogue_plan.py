"""Scene-aware native dialogue plan (pilot contract A).

``build_scene_dialogue_plan`` turns the cached whole-scene interpretation from
``performance_scene_analysis.analyze_scenes`` into a *temporary, source-bound,
recommendation-only* plan: for each selected utterance, one short native
instruction a reviewer may later ask the experimental profile to try.

What the plan is not
--------------------
* It never edits prose, never changes a voice, never starts synthesis, never
  calls a model or the network, and never writes a project.  The only file it may
  cause is the interpretation cache ``analyze_scenes`` already owns.
* It is not an approval.  Every direction carries ``human_review == "pending"``,
  ``approved == False`` and ``recommendation_only == True``.
* It does not change the default runtime.  The current profile keeps its own
  forwarding behaviour; the plan only *declares* an experimental review request.
* The narrator is preserved: a narrator line never receives an instruction.

Provenance
----------
A direction is bound to the exact text hash, the role, the voice revision and the
source-revision SHA that the project's registry resolves for the scene.  The
interpretation (cached) is voice independent; the voice binding is checked on its
own, so recasting changes ``binding_sha256`` and ``verify_scene_dialogue_plan``
without invalidating a single cached scene.  Unknown stays unknown: a missing
cue, confidence or source yields a hold, never an invented value.

The instruction is composed only from cues the analysis found in the book's own
prose (author speech tags, carried through their exchange, punctuation, response
timing) plus, separately labelled, one supported character habit.  Words that
would change who is speaking (performer, accent, pitch, age, gender, borrowed
voice) are refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dubbing_service.app.audiobook import performance_scene_analysis as psa

SCHEMA_VERSION = "scene-dialogue-plan-v1"
PLAN_VERSION = "scene-dialogue-plan-1"
MAX_PHRASES = psa.MAX_PRIORITIES
MAX_EVIDENCE_PER_DIRECTION = 24
BASIS_RANK = {psa.BASIS_HUMAN: 3, psa.BASIS_SOURCE: 2, psa.BASIS_INFERRED: 1, psa.BASIS_NONE: 0}

LIMITATIONS = (
    "Every direction is a recommendation from deterministic reading of the book's own prose. "
    "None is human approved and no audio quality is claimed or measured.",
    "The default runtime does not forward a native instruction; this plan only declares an "
    "experimental review request and changes no default.",
    "Narrator lines never receive an instruction.",
    "Carried cues are inferred, low confidence, and always cite the author's words.",
    "Character habits are separate from scene intention and are never an approved profile.",
    "Unknown emotion, intention or confidence stays unknown; a line without scene evidence is held.",
)

#: Words that would instruct the model to change *who* is speaking rather than *how* a line is said.
_IDENTITY_TERMS = re.compile(
    r"\b(?:accent\w*|dialect\w*|pitch\w*|octave\w*|timbre|baritone|soprano|alto|tenor|falsetto|bass|"
    r"gravel\w*|rasp\w*|husky|gender\w*|male|female|man|men|woman|women|boy|girl|masculine|feminine|"
    r"young\w*|old\w*|elderly|child\w*|teen\w*|british|american|english|scottish|irish|welsh|"
    r"australian|southern|northern|cockney|foreign|performer|actor|actress|celebrity|narrator|"
    r"impersonat\w*|imitat\w*|mimic\w*|clone\w*|sound\w*|"
    r"(?:another|different|new|deeper|higher|lower|own)\s+voice|voice\s+(?:of|like)|"
    r"chang\w*\s+(?:\w+\s+)?voice)\b", re.IGNORECASE)


class SceneDialoguePlanError(ValueError):
    """A refusal with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


# ---------------------------------------------------------------------- helpers


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unsafe_instruction_terms(text: str) -> list[str]:
    """Words in ``text`` that would change the speaker rather than the delivery."""

    return sorted({match.group(0).casefold() for match in _IDENTITY_TERMS.finditer(text)})


def _director() -> tuple[Any, str | None]:
    try:
        from dubbing_service.app.audiobook import performance_director

        return performance_director, None
    except Exception as exc:  # noqa: BLE001 - optional experimental code must not break planning
        return None, f"{type(exc).__name__}: {exc}"


def _strongest(bases: Iterable[str]) -> str:
    best = psa.BASIS_NONE
    for basis in bases:
        if BASIS_RANK.get(basis, 0) > BASIS_RANK[best]:
            best = basis
    return best


# --------------------------------------------------------------------- evidence


def _evidence_rows(row: Mapping[str, Any], used_phrases: set[str], texts: Mapping[str, str]) -> list[dict[str, Any]]:
    """Real spans of the book's prose behind a direction, each verified against the live text."""

    found: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def take(items: Iterable[Mapping[str, Any]] | None, supports: str) -> None:
        for item in items or []:
            if not isinstance(item, Mapping) or not {"utterance_id", "start", "end", "text"} <= set(item):
                continue  # structural facts (sentence length) have no span to quote
            key = (item["utterance_id"], item.get("kind"), item["start"], item["end"], supports)
            if key in seen:
                continue
            seen.add(key)
            prose = texts.get(item["utterance_id"])
            if prose is None or prose[item["start"]:item["end"]] != item["text"]:
                raise SceneDialoguePlanError(
                    "evidence_mismatch", f"evidence for {row['utterance_id']} does not match the live prose")
            found.append({"utterance_id": item["utterance_id"], "kind": item.get("kind"), "text": item["text"],
                          "start": item["start"], "end": item["end"], "supports": supports,
                          "basis": item.get("basis")})

    for cue in row["tag_cues"]:
        take(cue["evidence"], "author_cue")
    for cue in row["carried_cues"]:
        take(cue["evidence"], f"carried_cue:{cue['scope']}")
    if row["emotion"]["label"]:
        take(row["emotion"]["evidence"], f"emotion:{row['emotion']['label']}")
    if row["intention"]["value"] != "unknown":
        take(row["intention"]["evidence"], f"intention:{row['intention']['value']}")
    for item in row["delivery_phrases"]:
        if item["phrase"] in used_phrases:
            take(item["evidence"], f"phrase:{item['phrase']}")
    for side in ("before", "after"):
        boundary = row["boundary"][side]
        if boundary:
            take(boundary["evidence"], f"boundary_{side}:{boundary['value']}")
    if row["response"]:
        take(row["response"]["evidence"], f"response:{row['response']['relation']}")
    take(row["emphasis"], "emphasis")
    return found[:MAX_EVIDENCE_PER_DIRECTION]


# ------------------------------------------------------------------ composition


def _compose_instruction(director: Any, scene_phrases: Sequence[str], habit: str | None,
                         holds: list[dict[str, Any]], utterance_id: str) -> tuple[str | None, list[str], str]:
    """Return ``(instruction, scene phrases used, habit status)``.

    Scene intention fills the instruction first; one supported character habit may follow it.
    A phrase that would change the speaker is dropped with a hold, never rewritten.
    """

    safe: list[str] = []
    for phrase in scene_phrases[:MAX_PHRASES]:
        terms = unsafe_instruction_terms(phrase)
        if terms:
            holds.append({"code": "instruction_phrase_unsafe", "utterance_id": utterance_id,
                          "reason": f"phrase dropped; it names {', '.join(terms)}"})
            continue
        safe.append(phrase)
    if not safe:
        return None, [], "none" if habit is None else "withheld_no_scene_evidence"
    habit_status = "none" if habit is None else "not_included"
    phrases = list(safe)
    if habit is not None and len(phrases) < MAX_PHRASES:
        terms = unsafe_instruction_terms(habit)
        if terms:
            habit_status = "omitted_unsafe"
            holds.append({"code": "instruction_habit_unsafe", "utterance_id": utterance_id,
                          "reason": f"character habit omitted; it names {', '.join(terms)}"})
        else:
            phrases.append(habit)
            habit_status = "included"
    for attempt in (phrases, safe):
        text = "; ".join(attempt)
        text += "" if text.endswith(".") else "."
        try:
            return director.validate_native_instruction(text), attempt[:len(safe)] if attempt is safe else safe, (
                habit_status if attempt is phrases else ("rejected" if habit is not None else "none"))
        except Exception as exc:  # noqa: BLE001 - PerformanceDirectorError carries the reason
            if attempt is safe:
                holds.append({"code": "instruction_rejected", "utterance_id": utterance_id,
                              "reason": str(exc)[:240]})
                return None, [], habit_status
    return None, [], habit_status  # pragma: no cover - the loop always returns


# ------------------------------------------------------------------------ build


def _selection(segments: Sequence[Any], selected_ids: Iterable[str] | None, row_ids: set[str]) -> list[str] | None:
    if selected_ids is None:
        return None
    if isinstance(selected_ids, (str, bytes)):
        raise SceneDialoguePlanError("input_ineligible", "selected_ids must be a collection of utterance ids")
    chosen = list(selected_ids)
    if any(not isinstance(item, str) or not item for item in chosen):
        raise SceneDialoguePlanError("input_ineligible", "every selected id must be a non-empty string")
    if len(set(chosen)) != len(chosen):
        raise SceneDialoguePlanError("input_ineligible", "selected_ids contains a duplicate")
    missing = [item for item in chosen if item not in row_ids]
    if missing:
        raise SceneDialoguePlanError("input_ineligible",
                                     f"selected utterance {missing[0]} is not an analysable segment")
    if not chosen:
        raise SceneDialoguePlanError("input_ineligible", "selected_ids is empty")
    return chosen


def build_scene_dialogue_plan(project: Any, segments: Sequence[Any], *, cache_root: Path | str | None = None,
                              selected_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Plan scene-aware native dialogue for ``segments`` of ``project``.

    ``segments`` are analysed whole (reading order), so a selected line keeps the
    scene context around it.  ``selected_ids`` restricts the *directions* returned,
    not the scenes read.  ``cache_root`` is the existing, caller-owned directory
    ``analyze_scenes`` caches interpretations in (``None`` caches nothing).
    """

    analysis = psa.analyze_scenes(project, segments, cache_root=cache_root)
    rows = analysis["directions"]
    chosen = _selection(segments, selected_ids, {row["utterance_id"] for row in rows})
    wanted = set(chosen) if chosen is not None else None
    texts = {str(getattr(segment, "id", None) or segment["id"]): (getattr(segment, "text", None)
                                                                  if not isinstance(segment, Mapping) else segment["text"])
             for segment in segments}
    scenes = {scene["scene_id"]: scene for scene in analysis["scenes"]}
    cache_status = {entry["scene_id"]: entry["status"] for entry in analysis["cache"]["scenes"]}
    director, director_reason = _director()
    current = analysis["runtime"]["requested_profile"]

    directions: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    for row in rows:
        uid = row["utterance_id"]
        if wanted is not None and uid not in wanted:
            continue
        scene = scenes[row["scene_id"]]
        row_holds: list[dict[str, Any]] = []
        narrator = row["kind"] == "narration"
        candidate = row["native_instruction_candidate"] or {}
        source_ok = scene["source"]["source_revision_sha256"] is not None
        instruction: str | None = None
        used: list[str] = []
        habit_status = "none"
        status = "held"
        reason: str | None = None

        if row["kind"] == "scene_break":
            status, reason = "scene_break", "a scene-break marker is not spoken"
        elif narrator:
            status, reason = "narrator_preserved", "narrator preservation: no instruction is proposed"
        elif row["kind"] == "unattributed":
            reason = "the line has no known role; no instruction is proposed"
            row_holds.append({"code": "attribution_unresolved", "utterance_id": uid, "reason": reason})
        elif not source_ok:
            reason = scene["scene_intention_reason"] or "plan_stale: the scene has no proven source revision"
            row_holds.append({"code": scene["source"]["source_problem"] or "source_unproven",
                              "utterance_id": uid, "reason": reason})
        elif row["voice_revision"] is None:
            reason = "the role has no voice binding to pair a direction with"
            row_holds.append({"code": "voice_unbound", "utterance_id": uid, "reason": reason})
        elif director is None:
            reason = f"performance_director unavailable: {director_reason}"
            row_holds.append({"code": "director_unavailable", "utterance_id": uid, "reason": reason})
        else:
            scene_phrases = list(candidate.get("scene_phrases") or [])
            if not scene_phrases:
                reason = "no cue in the scene's prose supports a delivery direction for this line"
                row_holds.append({"code": "no_scene_evidence", "utterance_id": uid, "reason": reason})
                habit_status = "withheld_no_scene_evidence" if candidate.get("habit_phrase") else "none"
            else:
                instruction, used, habit_status = _compose_instruction(
                    director, scene_phrases, candidate.get("habit_phrase"), row_holds, uid)
                if instruction is None:
                    reason = "the proposed instruction was refused"
                else:
                    status = "directed"

        phrase_basis = {item["phrase"]: item["basis"] for item in row["delivery_phrases"]}
        basis = _strongest(phrase_basis.get(phrase, psa.BASIS_INFERRED) for phrase in used) if instruction else psa.BASIS_NONE
        used_set = set(used)
        evidence = _evidence_rows(row, used_set, texts) if instruction else []
        habit_text = candidate.get("habit_phrase") if habit_status == "included" else None
        capability = (row["what_if"] or {}).get(psa.EXPERIMENTAL_RUNTIME_PROFILE) or {}
        current_state = (row["what_if"] or {}).get(current) or {}
        emotion = row["emotion"]
        grounded = bool(emotion["label"] or row["intention"]["value"] != "unknown")
        direction = {
            "utterance_id": uid, "role_id": row["role_id"], "role_name": row["role_name"],
            "text_sha256": row["text_sha256"], "voice_revision": row["voice_revision"],
            "narrator": narrator, "native_instruction": instruction, "basis": basis,
            "basis_detail": {"scene": basis, "character_habit": "character_habit" if habit_text else None},
            "evidence": evidence, "holds": row_holds,
            "recommendation_only": True, "human_review": "pending", "approved": False,
            "status": status, "status_reason": reason,
            "scene_id": row["scene_id"], "index": row["index"], "kind": row["kind"],
            "instruction_parts": {"scene_phrases": used if instruction else [],
                                  "character_habit": habit_text},
            "scene_intention": {
                "emotion": {key: emotion.get(key) for key in ("label", "intensity", "basis", "confidence")},
                "intention": {key: row["intention"].get(key) for key in ("value", "basis", "confidence")},
                "response": (None if not row["response"] else
                             {key: row["response"][key] for key in ("relation", "responds_to", "timing", "basis")}),
                "exchange": row["exchange"],
                "tempo": row["tempo"]["value"],
                "boundary": {side: (row["boundary"][side] or {}).get("value") for side in ("before", "after")},
                "ending": (row["ending"] or {}).get("style"),
                "carried": [{key: cue[key] for key in ("scope", "from_utterance_id", "confidence")}
                            for cue in row["carried_cues"]],
            },
            "character_habit": {"phrase": candidate.get("habit_phrase"), "status": habit_status},
            "emotion_grounded": grounded and instruction is not None,
            "source": {"source_revision_ids": scene["source"]["source_revision_ids"],
                       "source_revision_sha256": scene["source"]["source_revision_sha256"],
                       "anchor": row["source_anchor"]},
            "forwarding": {
                "current_profile": current,
                "current_profile_forwards": bool(current_state.get("would_forward")),
                "default_changed": False,
                "experimental_review_requested": instruction is not None,
                "experimental_profile": psa.EXPERIMENTAL_RUNTIME_PROFILE,
                "experimental_capability_state": capability.get("native_instruction_state"),
                "experimental_would_forward": capability.get("would_forward"),
            },
        }
        directions.append(direction)
        holds.extend(row_holds)

    if not directions:
        raise SceneDialoguePlanError("input_ineligible", "no direction rows were produced")

    used_scene_ids = list(dict.fromkeys(item["scene_id"] for item in directions))
    scene_rows = []
    for scene_id in used_scene_ids:
        scene = scenes[scene_id]
        scene_rows.append({
            "scene_id": scene_id, "chapter_id": scene["chapter_id"], "ordinal": scene["ordinal"],
            "analysis_key": scene["analysis_key"], "cache_status": cache_status.get(scene_id),
            "source": {key: scene["source"][key] for key in (
                "source_revision_ids", "source_revision_sha256", "source_problem", "source_problem_detail",
                "attribution_digest", "alias_digest")},
            "utterance_ids": [item["utterance_id"] for item in scene["utterances"]],
            "tone": scene["summary"]["tone"], "emotion_arc": scene["summary"]["emotion_arc"],
            "exchanges": scene["summary"]["exchanges"], "turns": scene["turns"],
            "scene_intention_reason": scene["scene_intention_reason"],
            "holds": scene["holds"],
        })
        holds.extend({**hold, "scene_id": scene_id} for hold in scene["holds"] if hold not in holds)

    registry = analysis["source"]
    proven = all(item["source"]["source_revision_sha256"] for item in scene_rows)
    used_shas = sorted({item["source"]["source_revision_sha256"] for item in scene_rows
                        if item["source"]["source_revision_sha256"]})
    used_ids = sorted({sid for item in scene_rows for sid in item["source"]["source_revision_ids"]})

    roles = {item["role_id"]: item for item in analysis["roles"]}
    voices = []
    for role_id in dict.fromkeys(item["role_id"] for item in directions if item["role_id"]):
        role = roles.get(role_id, {})
        voices.append({"role_id": role_id, "role_name": role.get("role_name"), "role_type": role.get("role_type"),
                       "narrator": bool(role.get("narrator_preservation")),
                       "voice_revision": role.get("voice_revision")})

    binding = _sha({
        "schema_version": SCHEMA_VERSION, "plan_version": PLAN_VERSION, "analyzer_version": analysis["analyzer_version"],
        "project_id": analysis["project_id"],
        "source": [[sid, registry_sha] for sid, registry_sha in
                   ((entry["id"], entry["sha256"]) for entry in registry["revisions"]) if sid in used_ids],
        "lines": [[d["utterance_id"], d["text_sha256"], d["role_id"], d["voice_revision"], d["scene_id"]]
                  for d in directions],
        "alias_and_attribution": [[item["scene_id"], item["source"]["attribution_digest"],
                                   item["source"]["alias_digest"]] for item in scene_rows],
    })

    dialogue = [d for d in directions if d["kind"] == "dialogue"]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "plan_version": PLAN_VERSION,
        "project_id": analysis["project_id"], "binding_sha256": binding,
        "recommendation_only": True, "human_review": "pending", "approved": False,
        "status": "ready" if proven and any(d["status"] == "directed" for d in directions) else "held",
        "source": {"registry_status": registry["registry_status"], "registry_detail": registry["registry_detail"],
                   "proven": proven, "revision_ids": used_ids, "revision_sha256": used_shas,
                   "revisions": [entry for entry in registry["revisions"] if entry["id"] in used_ids]},
        "scenes": scene_rows,
        "cache": {"enabled": analysis["cache"]["enabled"], "identity": analysis["cache"]["identity"],
                  "hits": analysis["cache"]["hits"], "written": analysis["cache"]["written"]},
        "analysis": {"schema_version": analysis["schema_version"], "analyzer_version": analysis["analyzer_version"],
                     "analysis_sha256": analysis["analysis_sha256"]},
        "runtime": {"current_profile": current, "current_profile_forwards_instructions": any(
                        d["forwarding"]["current_profile_forwards"] for d in directions),
                    "default_changed": False,
                    "experimental_review_request": {"profile": psa.EXPERIMENTAL_RUNTIME_PROFILE,
                                                    "scope": "experimental review copies only",
                                                    "requested": any(d["native_instruction"] for d in directions)}},
        "voices": voices, "directions": directions,
        "coverage": {"selected": len(directions), "dialogue": len(dialogue),
                     "narrator": sum(1 for d in directions if d["narrator"]),
                     "directed": sum(1 for d in directions if d["status"] == "directed"),
                     "held": sum(1 for d in dialogue if d["status"] != "directed"),
                     "emotion_grounded": sum(1 for d in directions if d["emotion_grounded"]),
                     "explicit_source": sum(1 for d in directions if d["basis"] == psa.BASIS_SOURCE),
                     "inferred_only": sum(1 for d in directions if d["basis"] == psa.BASIS_INFERRED),
                     "narrator_instructions": sum(1 for d in directions if d["narrator"] and d["native_instruction"])},
        "holds": holds, "limitations": list(LIMITATIONS),
        "side_effects": {**analysis["side_effects"], "plan_written": False, "default_changed": False,
                         "approval_recorded": False},
    }
    plan = _jsonable(plan)
    plan["plan_sha256"] = _sha(plan)
    return plan


# ----------------------------------------------------------------------- verify


def verify_scene_dialogue_plan(plan: Mapping[str, Any], project: Any, segments: Sequence[Any]) -> dict[str, Any]:
    """Check a plan against live state: source, prose, role and voice each on their own.

    Read only.  A plan is ``current`` only when its content hash holds and nothing it was
    bound to has moved.  A voice change is reported as ``voice`` without calling the
    interpretation stale.
    """

    problems: list[dict[str, Any]] = []
    checks = {"content": True, "source": True, "prose": True, "role": True, "voice": True}

    def fail(check: str, code: str, reason: str, utterance_id: str | None = None) -> None:
        checks[check] = False
        problems.append({"check": check, "code": code, "utterance_id": utterance_id, "reason": reason})

    if not isinstance(plan, Mapping) or plan.get("schema_version") != SCHEMA_VERSION:
        fail("content", "schema_unknown", "not a scene-dialogue-plan-v1 record")
        return {"current": False, "checks": checks, "problems": problems}
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _sha(body) != plan.get("plan_sha256"):
        fail("content", "plan_tampered", "the plan's content hash does not match its content")
    if plan.get("approved") is not False or plan.get("recommendation_only") is not True:
        fail("content", "plan_tampered", "a plan must stay an unapproved recommendation")
    if plan.get("human_review") != "pending" or plan.get("plan_version") != PLAN_VERSION:
        fail("content", "plan_stale", "the plan version or review state changed")
    if plan.get("project_id") != psa._get(project, "id"):
        fail("source", "plan_stale", "the plan belongs to another project")
    # A selected line depends on its complete scene, including unselected cues.
    current = psa.analyze_scenes(project, segments)
    current_scenes = {s["scene_id"]: s for s in current["scenes"]}
    for scene in plan.get("scenes", []):
        live_scene = current_scenes.get(scene.get("scene_id"))
        if not live_scene or live_scene["analysis_key"] != scene.get("analysis_key"):
            fail("source", "plan_stale", "scene context, attribution or source changed")

    live = {}
    for segment in segments:
        sid = str(getattr(segment, "id", None) if not isinstance(segment, Mapping) else segment.get("id"))
        live[sid] = segment
    roles = {str(psa._get(role, "id")): role for role in (psa._get(project, "roles") or [])}
    try:
        from dubbing_service.app.audiobook import fingerprint
    except Exception:  # noqa: BLE001
        fingerprint = None

    for direction in plan.get("directions", []):
        uid = direction.get("utterance_id")
        segment = live.get(uid)
        if segment is None:
            fail("prose", "plan_stale", "the utterance is no longer in the project", uid)
            continue
        text = psa._get(segment, "text")
        if not isinstance(text, str) or _text_sha(text) != direction.get("text_sha256"):
            fail("prose", "plan_stale", "the prose changed after the plan was made", uid)
        if psa._get(segment, "role_id") != direction.get("role_id"):
            fail("role", "plan_stale", "the line is attributed to a different role", uid)
        role = roles.get(str(direction.get("role_id"))) if direction.get("role_id") else None
        if direction.get("role_id") and role is None:
            fail("role", "plan_stale", "the role is no longer in the project", uid)
        elif role is not None and fingerprint is not None and \
                fingerprint.voice_revision_for_role(role) != direction.get("voice_revision"):
            fail("voice", "voice_changed", "the role's voice revision differs from the one the plan was bound to",
                 uid)
        expected = (direction.get("source") or {}).get("source_revision_sha256")
        anchor = psa._get(segment, "source_anchor")
        if expected is None and direction.get("kind") == "dialogue":
            fail("source", "source_unproven", "the dialogue has no proven source digest", uid)
        live_anchor = None if anchor is None else {key: psa._get(anchor, key) for key in
                     ("source_revision_id", "start_offset", "end_offset", "occurrence_index")}
        if live_anchor != (direction.get("source") or {}).get("anchor"):
            fail("source", "plan_stale", "the exact source anchor changed", uid)
        if expected is not None:
            try:
                live_sha = psa.resolve_source_revision_sha256(project, psa._get(anchor, "source_revision_id"))
            except psa.SceneAnalysisError as exc:
                fail("source", exc.code, str(exc), uid)
            else:
                if live_sha != expected:
                    fail("source", "plan_stale", "the registered source digest changed", uid)
    return {"current": all(checks.values()), "checks": checks, "problems": problems}
