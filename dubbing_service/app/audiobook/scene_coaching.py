"""Validate and cache whole-scene model coaching without approving or rendering it.

The caller supplies a bounded model response and its captured request. This
module makes no model/network call. Scene interpretation is voice independent;
the returned plan keeps its existing source, cast and voice-revision bindings.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .scene_dialogue_plan import unsafe_instruction_terms, _sha
from .performance_director import validate_native_instruction

BUNDLE_SCHEMA = "scene-coaching-bundle-v1"
RESPONSE_SCHEMA = "scene-coaching-proposals-v1"
VERSION = "scene-coaching-1"
FIELDS = {"utterance_id", "objective", "emotion", "intensity", "cadence", "emphasis", "ending",
          "native_instruction", "confidence", "evidence"}


class CoachingError(ValueError):
    pass


def _plain(value, field, limit=240):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        raise CoachingError(f"Invalid coaching {field}")
    return value


def _get(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def validate_bundle(bundle):
    """Check coverage and all evidence against the captured, unmodified request."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise CoachingError("Unknown coaching bundle")
    request, response = bundle.get("request"), bundle.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise CoachingError("Coaching request/response missing")
    if _sha(request) != bundle.get("request_sha256") or _sha(response) != bundle.get("response_sha256"):
        raise CoachingError("Coaching request/response hash mismatch")
    provenance = bundle.get("provenance", {})
    if (not isinstance(provenance, dict) or provenance.get("origin") != "model" or
            not provenance.get("model_requested") or not provenance.get("receipt_sha256")):
        raise CoachingError("Captured model provenance required")
    if response.get("schema_version") != RESPONSE_SCHEMA:
        raise CoachingError("Unknown coaching response")
    source_rows = request.get("scenes")
    if (not isinstance(source_rows, list) or not source_rows or any(
            not isinstance(s, dict) or not isinstance(s.get("scene_id"), str) or
            not isinstance(s.get("context"), list) or not isinstance(s.get("selected_dialogue_ids"), list)
            for s in source_rows)):
        raise CoachingError("Invalid requested scenes")
    requested = {s["scene_id"]: s for s in source_rows}
    rows = response.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(requested) or len(requested) != len(request.get("scenes", [])):
        raise CoachingError("Coaching scene coverage differs")
    seen, result = set(), {}
    for scene in rows:
        if not isinstance(scene, dict) or not isinstance(scene.get("directions"), list):
            raise CoachingError("Invalid response scene")
        sid = scene.get("scene_id")
        if sid not in requested or sid in seen:
            raise CoachingError("Unknown or duplicate coaching scene")
        seen.add(sid)
        source = requested[sid]
        if any(not isinstance(r, dict) or not {"utterance_id", "text", "role_id", "narrator"} <= set(r)
               or not isinstance(r["text"], str) or type(r["narrator"]) is not bool for r in source["context"]):
            raise CoachingError("Invalid requested context")
        context = {r["utterance_id"]: r for r in source["context"]}
        expected = set(source["selected_dialogue_ids"])
        if len(context) != len(source["context"]) or len(expected) != len(source["selected_dialogue_ids"]) or not expected <= context.keys():
            raise CoachingError("Duplicate context or unknown dialogue selection")
        directions, found = [], set()
        for row in scene.get("directions", []):
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise CoachingError("Coaching fields differ from the contract")
            uid = row["utterance_id"]
            if uid not in expected or uid in found or context[uid]["narrator"]:
                raise CoachingError("Coaching targets a narrator, unknown or duplicate dialogue")
            found.add(uid)
            for field in ["objective", "emotion", "cadence", "ending"]:
                _plain(row[field], field)
            if row["intensity"] not in {"moderate", "strong"} or row["confidence"] not in {"low", "moderate", "high"}:
                raise CoachingError("Unknown coaching strength/confidence")
            instruction = _plain(row["native_instruction"], "native_instruction", 220)
            if unsafe_instruction_terms(instruction):
                raise CoachingError("Coaching instruction names a speaker/identity change")
            validate_native_instruction(instruction)
            if not isinstance(row["emphasis"], list) or len(row["emphasis"]) > 3:
                raise CoachingError("At most three emphasis spans")
            for phrase in row["emphasis"]:
                if not isinstance(phrase, str) or not phrase or phrase not in context[uid]["text"]:
                    raise CoachingError("Emphasis differs from exact source wording")
            if not isinstance(row["evidence"], list) or not 1 <= len(row["evidence"]) <= 8:
                raise CoachingError("Source evidence required")
            for evidence in row["evidence"]:
                if not isinstance(evidence, dict):
                    raise CoachingError("Invalid coaching evidence")
                src = context.get(evidence.get("utterance_id"))
                start, end = evidence.get("start"), evidence.get("end")
                if (src is None or type(start) is not int or type(end) is not int or
                        not 0 <= start < end <= len(src["text"]) or src["text"][start:end] != evidence.get("text")):
                    raise CoachingError("Coaching evidence span differs from source")
            directions.append(copy.deepcopy(row))
        if found != expected:
            raise CoachingError("Coaching dialogue coverage incomplete")
        result[sid] = {"scene": source, "directions": directions}
    return result


def apply_coaching(plan, bundle, project, segments, *, cache_root=None):
    """Overlay temporary model interpretation only after checking live source context."""
    scenes = validate_bundle(bundle)
    if (plan.get("source", {}).get("proven") is not True or plan.get("recommendation_only") is not True or
            plan.get("approved") is True or any(h.get("code") != "no_scene_evidence" for h in plan.get("holds", []))):
        raise CoachingError("Coaching cannot clear source, approval or safety holds")
    live = {_get(s, "id"): s for s in segments}
    project_id = _get(project, "id")
    mine, cache_entries, pending_cache = {}, [], []
    for sid, entry in scenes.items():
        source = entry["scene"]
        if source["project_id"] != project_id:
            continue
        for row in source["context"]:
            segment = live.get(row["utterance_id"])
            if segment is None or _get(segment, "text") != row["text"] or _get(segment, "role_id") != row["role_id"]:
                raise CoachingError("Live source context/attribution changed")
        record = {"version": VERSION, "scene_id": sid,
                  "context_sha256": _sha(source), "model": bundle["provenance"]["model_requested"],
                  "policy_sha256": _sha({k: bundle["request"][k] for k in ["instruction", "constraints", "response_contract"]}),
                  "directions": entry["directions"]}
        key = _sha(record)
        pending_cache.append((sid, key, record))
        for row in entry["directions"]:
            mine[row["utterance_id"]] = (sid, row)
    result = copy.deepcopy(plan)
    dialogue = {d["utterance_id"] for d in result["directions"] if not d["narrator"]}
    if set(mine) != dialogue:
        raise CoachingError("Model coaching differs from the selected dialogue set")
    for row in result["directions"]:
        if row["narrator"]:
            if row["native_instruction"] is not None: raise CoachingError("Narrator must remain neutral")
            continue
        if not row["source"]["source_revision_sha256"] or row["voice_revision"] is None:
            raise CoachingError("Coaching needs a proven source and existing voice binding")
        sid, coach = mine[row["utterance_id"]]
        row["previous_interpretation"] = {k: copy.deepcopy(row[k]) for k in
            ["basis", "evidence", "emotion", "emotion_grounded", "scene_intention"] if k in row}
        row.update(native_instruction=coach["native_instruction"], basis="model", status="directed",
                   status_reason="Whole-scene model coaching; recommendation only", evidence=coach["evidence"],
                   holds=[], coaching_scene_id=sid, emotion_grounded=False,
                   coaching={k: coach[k] for k in ["objective", "emotion", "intensity", "cadence", "emphasis", "ending", "confidence"]})
    for sid, key, record in pending_cache:
        hit = False
        if cache_root is not None:
            path = Path(cache_root) / "scene-coaching" / (key + ".json")
            data = json.dumps(record, sort_keys=True, ensure_ascii=False, indent=2).encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream: stream.write(data)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise CoachingError("Scene coaching cache conflict")
                hit = True
        cache_entries.append({"scene_id": sid, "key": key, "hit": hit})
    result["holds"] = [h for h in result["holds"] if h.get("code") not in {"no_scene_evidence"}]
    result["coaching"] = {"version": VERSION, "request_sha256": bundle["request_sha256"],
                          "response_sha256": bundle["response_sha256"], "provenance": bundle["provenance"],
                          "cache": cache_entries, "recommendation_only": True, "human_review": "pending"}
    result["binding_sha256"] = _sha([plan["binding_sha256"], bundle["request_sha256"], bundle["response_sha256"]])
    result["status"] = "ready"
    result["coverage"]["directed"] = len(dialogue)
    result["coverage"]["held"] = 0
    result["coverage"].update(model_coached=len(dialogue), explicit_source=0, inferred_only=0,
                              emotion_grounded=0)  # model readings remain separate from author-written emotion cues
    result["limitations"].append("Model coaching is unapproved interpretation. Stronger instructions do not establish better audio.")
    result.pop("plan_sha256", None)
    result["plan_sha256"] = _sha(result)
    return result
