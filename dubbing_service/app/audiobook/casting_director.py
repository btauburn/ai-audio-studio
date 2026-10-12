"""Casting recommendations over Studio owners; only explicit auditions spend GPU.

Analysis never assigns a voice, changes a rating, publishes a library version or
rewrites a Take. Auditions are ordinary Casting Review rounds with a shared script
and fresh production-path audio, including when an existing reference is reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.storage.projects import load_project, project_meta_lock
from dubbing_service.app.storage.segments import load_segments
from . import voice_lab as lab

SCHEMA = "casting-performance-directors-v1"
ALGORITHM_VERSION = "casting-director-integration-v5"
META = "casting_director"


class DirectorError(lab.VoiceLabError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def _json(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _binding(project, segments):
    from dubbing_service.app.storage import voices as library
    from .fingerprint import voice_revision_for_role
    with library.voices_lock():
        shelf = [{"id": v.id, "versions": {k: e.get("manifest_sha256")
                  for k, e in v.voice_lab_versions.items()}, "preferences": v.voice_lab_preferred}
                 for v in library.load_voices()]
    return _digest({"schema": SCHEMA, "algorithm": ALGORITHM_VERSION, "project_id": project.id,
                    "source": [[s.id, s.text, s.role_id, s.index, s.chapter_id, _json(s.lineage),
                                _json(s.source_anchor)] for s in segments],
                    "roles": [[r.id, r.name, r.role_type, r.description,
                               voice_revision_for_role(r)] for r in project.roles],
                    "character_cast": _json(project.character_cast), "shelf": shelf,
                    "preferences": _json(project.voice_lab.preferences),
                    "casting": project.voice_lab.casting,
                    "requested_runtime": project.metadata.get("requested_runtime")})


def _current(project, role, records):
    from . import casting_review as casting
    from .fingerprint import voice_revision_for_role
    key = casting._current_key(project, role, records)
    record = records.get(key)
    item = casting._state(project)["roles"].get(role.id) or {}
    assignment = item.get("assignment") or {}
    preference = project.voice_lab.preferences.get(key) or {}
    stars = assignment.get("stars")
    if stars is None:
        stars = preference.get("rating")
    approved = casting.approved(project, role.id)
    return {"key": key, "name": (record or {}).get("name"),
            "voice_revision": voice_revision_for_role(role),
            "identity_id": role.voice_identity.id if role.voice_identity else None,
            "rating": stars, "approved": bool(approved),
            "protected": bool(approved or type(stars) is int and stars >= 4),
            "audio_sha256": (record or {}).get("audio_sha256")}


def _cache_path(project_id, binding):
    root = safe_project_root(project_id).resolve()
    path = (root / "voice_lab" / "directors" / f"{binding}.json").resolve()
    if not path.is_relative_to(root):
        raise DirectorError("Director cache must remain inside the project")
    return path


def _store(path, payload):
    """Exclusive immutable cache publication; retries validate rather than replace."""
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise DirectorError("Director cache conflict; existing evidence preserved")


def _read(path, binding):
    try:
        report = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise DirectorError("Director cache is unreadable") from exc
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("binding_sha256") != binding or _digest(payload) != report.get("report_sha256"):
        raise DirectorError("Director cache identity mismatch")
    return report


def _audio_links(project, ranked, records):
    for candidate in ranked:
        # A global library key is not a playable project artifact until imported.
        record = next((row for row in records.values() if row.get("key") == candidate.get("key")
                       and row.get("audio_sha256")), None)
        candidate["audio_url"] = (f"/api/v1/projects/{project.id}/voice-lab/audio/{record['key']}"
                                  if record else None)
        candidate["audition_status"] = "historical_sample" if record else "needs_audition"
        if record:
            candidate["audio_sha256"] = record["audio_sha256"]
        candidate["comparison_status"] = "fresh_matched_script_required"


def _base_compatible(candidate):
    # A saved approval remains on its original engine. It does not authorize
    # introducing CustomVoice or another engine into a Base casting comparison.
    if (candidate.get("provenance") or {}).get("rights_status") == "provider_preset":
        return False
    if candidate.get("reference_conditioned"):
        return (candidate.get("conditioning") or {}).get("backend") == "qwen3tts"
    return True  # Original design/source directions still require preparation.


def _compute(project, segments, binding):
    from .casting_director_discovery import discover_candidates
    from .casting_director_matching import analyze_characters, rank_candidates
    from .performance_scene_analysis import analyze_scenes
    from .project_qwen_runtime import project_profile
    from .character_research import research_project
    # Accumulate current book evidence on a private projection. Approval history,
    # identities and the persisted dossier remain owned by their original owners.
    project = project.model_copy(deep=True)
    research_project(project, segments)
    from .casting_profiles import build_profiles
    project.character_cast.profiles = build_profiles(project, segments)
    analyses = analyze_characters(project, segments)
    discovery = discover_candidates(project, segments)
    records = lab.voice_records(project)
    cast = []
    for role in project.roles:
        current = _current(project, role, records)
        if current["identity_id"]:
            match = next((c for c in discovery["candidates"]
                          if c.get("key") == current["key"]), {})
            cast.append({**match, "role_id": role.id, "identity_id": current["identity_id"]})
    roles = []
    for analysis in analyses:
        role = next(r for r in project.roles if r.id == analysis["role_id"])
        current = _current(project, role, records)
        rejected = (project.voice_lab.casting["roles"].get(role.id) or {}).get("rejections", [])
        rejected_keys = {row.get("key") for row in rejected}
        rejected_identities = {(records.get(key) or {}).get("identity", {}).get("id")
                               for key in rejected_keys if key in records}
        candidates = [c for c in discovery["candidates"]
                      if c.get("designed_for_role_id") in {None, role.id}
                      and _base_compatible(c)
                      and c.get("key") not in rejected_keys
                      and (not c.get("identity_id") or c["identity_id"] not in rejected_identities)]
        ranked = rank_candidates(analysis, candidates,
                                 cast=[c for c in cast if c.get("role_id") != role.id], limit=5)
        _audio_links(project, ranked, records)
        replacement = type(current["rating"]) is int and current["rating"] <= 3
        roles.append({"analysis": analysis, "role_id": role.id, "role_name": role.name,
                      "current_voice": current, "protected": current["protected"],
                      "needs_casting": not current["protected"],
                      "replacement_candidate": replacement, "candidates": ranked,
                      "coverage": "recommendations" if ranked else "needs_casting"})
    # POST analysis creates only immutable recommendations. Native forwarding still
    # belongs to the existing reviewed Performance Director runtime owner.
    scene = analyze_scenes(project, segments, cache_root=_cache_path(project.id, binding).parent)
    payload = {"schema": SCHEMA, "project_id": project.id, "runtime": project_profile(project.id),
               "binding_sha256": binding, "roles": roles, "scenes": scene,
               "holds": [*discovery.get("holds", []),
                         *[{"candidate_id": c["candidate_id"], "code": "incompatible_base_engine",
                            "reason": "Saved voice remains approved for its engine; this comparison requires Qwen Base."}
                           for c in discovery["candidates"] if not _base_compatible(c)]],
               "human_review": "pending",
               "quality_improvement": "not_established", "automatic_approval": False}
    return {**payload, "report_sha256": _digest(payload)}


def view(project_id):
    """Read-only: no model call, cache write, synthesis, approval or assignment."""
    with project_meta_lock(project_id):
        project = load_project(project_id)
        segments = load_segments(project_id)
        binding = _binding(project, segments)
        path = _cache_path(project_id, binding)
        if path.is_file():
            return _read(path, binding)
        return {"schema": SCHEMA, "project_id": project_id, "status": "needs_analysis",
                "binding_sha256": binding, "roles": [], "scenes": None, "holds": [],
                "human_review": "pending", "quality_improvement": "not_established"}


def analyze(project_id):
    with project_meta_lock(project_id):
        project = load_project(project_id)
        segments = load_segments(project_id)
        binding = _binding(project, segments)
        path = _cache_path(project_id, binding)
        if path.is_file():
            return _read(path, binding)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = _compute(project, segments, binding)
        _store(path, report)
        return report


def proposals_for_round(project, role_id):
    """Optional hook: only a root-staged, source-bound request alters a new round."""
    plan = (project.metadata.get(META) or {}).get("round_plans", {}).get(role_id)
    if plan is None:
        return None  # Preserve every ordinary existing casting pathway.
    from .fingerprint import voice_revision_for_role
    role = next((r for r in project.roles if r.id == role_id), None)
    if role is None or voice_revision_for_role(role) != plan["expected_voice_revision"]:
        raise DirectorError("Casting changed while ranked auditions were prepared")
    if _digest(plan["proposals"]) != plan["proposals_sha256"]:
        raise DirectorError("Ranked audition proposal evidence changed")
    return copy.deepcopy(plan["proposals"])


def prepare_auditions(project_id, *, request_id, role_id, report_sha256):
    from dubbing_service.app.core.tts_operations import try_acquire_project_mutation, release_project_mutation
    lease = try_acquire_project_mutation(project_id, kind="casting_director_auditions")
    if lease is None:
        raise DirectorError("Rendering owns this project; request auditions at the next safe boundary")
    try:
        return _prepare_auditions(project_id, request_id=request_id, role_id=role_id,
                                 report_sha256=report_sha256)
    finally:
        release_project_mutation(lease)


def _prepare_auditions(project_id, *, request_id, role_id, report_sha256):
    """Queue 3–5 identical-script auditions using existing reference/Take owners.

    Imported heard library samples remain historical evidence. Each candidate gets
    a fresh ordinary audition with the same text and settings; no approval travels
    with an import, and no selected cast member is silently replaced.
    """
    from . import casting_review as casting, preferred_voices
    from .fingerprint import voice_revision_for_role
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
        raise DirectorError("Invalid audition request identity")
    command = _digest([role_id, report_sha256])
    # A retry refers to its frozen operation, even after a new round has changed
    # the current analysis binding. Check it before recomputing current evidence.
    with project_meta_lock(project_id):
        project = load_project(project_id)
        prior = (project.metadata.get(META) or {}).get("requests", {}).get(request_id)
        if prior:
            if prior["command_sha256"] != command:
                raise DirectorError("Request ID belongs to another audition operation")
            return copy.deepcopy(prior["receipt"])
    report = analyze(project_id)
    if report["report_sha256"] != report_sha256:
        raise DirectorError("Casting evidence changed; refresh before requesting auditions")
    selected = next((r for r in report["roles"] if r["role_id"] == role_id), None)
    if selected is None:
        raise DirectorError("Unknown speaking role")
    if selected["protected"]:
        raise DirectorError("Approved casting is protected; this request cannot recast it")
    with project_meta_lock(project_id):
        project = load_project(project_id)
        prior = (project.metadata.get(META) or {}).get("requests", {}).get(request_id)
        if prior:
            if prior["command_sha256"] != command:
                raise DirectorError("Request ID belongs to another audition operation")
            return copy.deepcopy(prior["receipt"])
        role = next(r for r in project.roles if r.id == role_id)
        expected_voice = voice_revision_for_role(role)
        _, _, required, attention = casting._scope(project, None)
        if role_id not in required:
            raise DirectorError("Role needs source attribution before casting: " + str(attention)[:400])
    proposals = []
    holds = []
    for candidate in selected["candidates"][:5]:
        if not candidate.get("eligible", False) or candidate.get("holds"):
            continue
        reuse_key = candidate.get("key")
        library_id = candidate.get("library_voice_id")
        version = candidate.get("library_version")
        if library_id and version:
            result = preferred_voices.prepare_prior_candidate(project_id, library_id, version)
            if result["state"] != "ready":
                holds.append({"candidate_id": candidate["candidate_id"], "reason": result["reason"]})
                continue
            reuse_key = result["key"]
        if candidate.get("origin") == "authorized_external" and not reuse_key:
            # Discovery metadata is not a prepared hash-bound reference. Acquisition
            # and quality/consent verification remain with the existing source owner.
            holds.append({"candidate_id": candidate["candidate_id"],
                          "reason": "Authorized source needs verified acquisition/reference preparation"})
            continue
        instruction = (candidate.get("design_instruction") or candidate.get("instruction")
                       or (candidate.get("design") or {}).get("design_instruction"))
        if not reuse_key and not instruction:
            continue
        proposals.append({"kind": "generated", "reuse_key": reuse_key,
                          "design_instruction": instruction or "Reuse this exact reference-conditioned voice.",
                          "rationale": str(candidate.get("explanation") or candidate.get("name")),
                          "tags": candidate.get("tags", []), "source_urls": [],
                          "prior_evidence": {"director_recommendation": copy.deepcopy(candidate),
                                             "report_sha256": report_sha256,
                                             "comparison": "fresh_matched_script"},
                          "concept": candidate.get("concept"), "reference": None,
                          "conditioning": candidate.get("conditioning"),
                          "supported": [], "inferred": [], "blocked": False,
                          "blocked_reason": None})
    if not proposals:
        raise DirectorError("Role needs casting: no viable prepared candidate; " + str(holds)[:500])
    with project_meta_lock(project_id):
        project = load_project(project_id)
        role = next(r for r in project.roles if r.id == role_id)
        if casting.approved(project, role_id) or voice_revision_for_role(role) != expected_voice:
            raise DirectorError("Casting changed during audition preparation")
        if _binding(project, load_segments(project_id)) != report["binding_sha256"]:
            raise DirectorError("Source, runtime or casting evidence changed during preparation")
        metadata = project.metadata.setdefault(META, {"requests": {}, "round_plans": {}})
        # Competing imports may finish, but only one receipt can allocate a round.
        prior = metadata["requests"].get(request_id)
        if prior:
            if prior["command_sha256"] != command:
                raise DirectorError("Request ID belongs to another audition operation")
            return copy.deepcopy(prior["receipt"])
        metadata["round_plans"][role_id] = {"expected_voice_revision": expected_voice,
                                            "proposals": proposals,
                                            "proposals_sha256": _digest(proposals)}
        try:
            round_ = casting._new_round(project, role_id, replacement=True)
        finally:
            metadata["round_plans"].pop(role_id, None)
        receipt = {"project_id": project_id, "role_id": role_id, "request_id": request_id,
                   "round_id": round_["id"], "candidate_count": len(proposals), "status": "queued",
                   "holds": holds, "script_version": round_["script_version"],
                   "runtime": report["runtime"], "human_review": "pending"}
        metadata["requests"][request_id] = {"command_sha256": command, "receipt": receipt}
        casting._state(project)["enabled"] = True
        casting._save(project)
        return receipt
