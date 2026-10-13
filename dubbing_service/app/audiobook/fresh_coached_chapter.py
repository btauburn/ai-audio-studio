"""Fresh scene-coached review chapter over a root-prepared derived project that has approved clones but no Takes.

``coached_chapter_batch`` deliberately requires existing selected, QA-passed neutral Takes. This pipeline is the
separate path for a chapter that has none: every speech line, narration included, is newly synthesized for
listening review through ONE serial ``directors_vertical_slice.Worker(directed=True)``. Root prepares the derived
project (``chapter_rerun`` metadata, approved library clone files) with the existing import/fork helpers; nothing
here imports, drafts or infers a voice or an attribution.

``prepare_chapter``  proves the exact EPUB chapter, derived source/coverage/breaks/exclusions, resolved attribution,
                     every speech role's approved recorded-reference clone on the Qwen Base route and its exact
                     source asset/casting provenance, freezes every line, builds the base scene plan and writes one
                     whole-scene coaching request (optionally enriched by captured character performance memory).
                     Status ``awaiting_coaching``. Nothing is coached or synthesized.
``apply_bundle``     validates a captured model bundle against the frozen request, live source/cast and the pilot's
                     direction guards, then publishes an immutable coached plan. No instruction is manufactured.
``generate_chapter`` renders every speech line (narrator: ``native_instruction`` None; dialogue: the captured
                     instruction), verifies each raw receipt and technical QA, and assembles the chapter in source
                     order under the measured boundary policy.

Everything is review only. No Take, selection, human event, approval or production default is created and the
canonical project is never written. Narration here is NEW neutral review audio; it is never claimed to be a reused
or approved Take, and every receipt and map says so.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from . import epub_chapter_selection, performance_boundaries as boundaries, qa, scene_coaching
from . import scene_dialogue_plan as plans
from . import voice_lab_chapter_export as exporter
from .cloned_voice_policy import policy_mode
from .fingerprint import voice_revision_for_role
from .voice_lab_render import _decoded_duration, _loudness, _mp3

REPO = Path(__file__).resolve().parents[3]

STATE_SCHEMA = "fresh-coached-chapter-state-v1"
APPLIED_SCHEMA = "fresh-coached-chapter-applied-v1"
SAMPLE_SCHEMA = "fresh-coached-chapter-sample-v1"
AUDIT_SCHEMA = "fresh-coached-chapter-forwarding-audit-v1"
CHAPTER_SCHEMA = "fresh-coached-chapter-review-v1"
RESULT_SCHEMA = "fresh-coached-chapter-result-v1"
MEMORY_BINDING_SCHEMA = "fresh-coached-chapter-memory-binding-v1"
MEMORY_FILE = "memory/performance-memory.json"
MEMORY_SCENE_KEY = "character_performance_memory"
MEMORY_CONSTRAINT_PREFIX = "Character performance memory "
SEPARATOR_TARGET_MS = 1250
MAX_MEMORY_BYTES = 8_000_000
FIXTURE_LABEL = "Synthetic CPU test fixture; excluded from production readiness"
OWNER_REVIEW_SCHEMA = "fresh-chapter-owner-review-v1"
MODE_TARGET = "current_target_approval"
MODE_EXISTING = "existing_book_cast_private_review"
EXISTING_CAST = {"mode": MODE_EXISTING, "target_approvals_created": 0, "role_fit_approval": "not_established",
                 "human_listening": "pending",
                 "statement": "Retained existing cloned cast reused for private review under an explicit owner request. "
                              "Source-use permission is not voice-quality or role-fit approval; no target approval, "
                              "selection or human event was created or copied, and the original book-role project "
                              "remains the context that authorizes each backend call."}
FROZEN_KEYS = ("epub", "sources", "chapter", "roles", "providers", "owner_review", "lines", "request", "base_plan",
               "memory", "isolated", "boundary_policy_sha256", "prepare_holds")
REVIEW_FLAGS = {"review_only": True, "human_review": "pending", "production_ready": False, "default_changed": False,
                "approvals_created": False, "takes_created": False, "selections_created": False,
                "human_events_created": False, "quality_improvement": "not_established"}
NARRATION = {"mode": "fresh_neutral_review_synthesis", "native_instruction": None, "reused_approved_take": False,
             "selected_take_used": False,
             "route": "same approved Base clone route as dialogue (directed wrapper, no native instruction)",
             "statement": "Newly generated neutral review audio; it is not a reused, selected or approved Take."}
_ID = re.compile(r"[A-Za-z0-9_-]{1,120}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
REQUEST_INSTRUCTION = (
    "Read each complete scene below, including the narration, and propose one temporary native delivery "
    "instruction for every id in that scene's selected_dialogue_ids. Narration rows are context only and will be "
    "rendered neutrally: never propose a direction for them. Never rewrite, add or omit spoken words. Keep your "
    "reading of the scene (objective, emotion, intensity, cadence, emphasis, ending, confidence) separate from the "
    "short native_instruction that conditions the existing voice. Return only JSON that matches response_contract.")
REQUEST_CONSTRAINTS = [
    "Return exactly one direction for each selected_dialogue_ids entry of each scene and none for any "
    "other utterance, especially narration.",
    "Evidence quotes an exact substring of one context row's text, with 0-based [start, end) character "
    "offsets into that row; one to eight items per direction.",
    "emphasis lists at most three exact substrings of the direction's own utterance text.",
    "native_instruction is a delivery description of at most 220 characters. It must not name or change "
    "the speaker, voice identity, gender, accent, dialect, pitch or performer, must not imitate another "
    "voice and must not use numeric rates.",
    "intensity is moderate or strong; confidence is low, moderate or high; objective, emotion, cadence "
    "and ending are short plain strings of at most 240 characters.",
]


class FreshChapterError(RuntimeError):
    """A fail-closed refusal. Preserved output is reported, never repaired silently."""


# ---------------------------------------------------------------------------------------- plumbing

_PILOT = None


def pilot():
    """The scene pilot helper module (a script, loaded once by path)."""
    global _PILOT
    if _PILOT is None:
        spec = importlib.util.spec_from_file_location("_fresh_chapter_scene_dialogue_pilot",
                                                      REPO / "scripts" / "scene_dialogue_pilot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PILOT = module
    return _PILOT


@contextlib.contextmanager
def _fail_closed():
    try:
        yield
    except FreshChapterError:
        raise
    except (ValueError, pilot().PilotError, epub_chapter_selection.EpubChapterSelectionError) as exc:
        raise FreshChapterError(str(exc)) from exc


@contextlib.contextmanager
def _projects_root(path):
    from dubbing_service.app.core.config import clear_config_caches
    previous = os.environ.get("DUBBING_PROJECTS_ROOT")
    os.environ["DUBBING_PROJECTS_ROOT"] = str(path)
    clear_config_caches()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DUBBING_PROJECTS_ROOT", None)
        else:
            os.environ["DUBBING_PROJECTS_ROOT"] = previous
        clear_config_caches()


def _optional_path(value):
    return Path(value).resolve() if value else None


def _config(args):
    """Normalise and validate the arguments; every later stage derives from this."""
    project, chapter = str(args.project), args.chapter
    if not _ID.fullmatch(project):
        raise FreshChapterError("--project must be a valid project id")
    if type(chapter) is not int or chapter < 1:
        raise FreshChapterError("--chapter must be a positive integer")
    source_root, output = Path(args.source_root).resolve(), Path(args.output_dir).resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise FreshChapterError("Output must be a separate directory outside the canonical project store")
    return SimpleNamespace(epub=_optional_path(getattr(args, "epub", None)), source_root=source_root,
                           project=project, chapter=chapter, output_dir=output,
                           coaching_bundle=_optional_path(getattr(args, "coaching_bundle", None)),
                           performance_memory=_optional_path(getattr(args, "performance_memory", None)))


def _frozen_sha(state):
    return pilot().value_sha({key: state[key] for key in FROZEN_KEYS})


def _write_state(output, state):
    pilot().replace_atomic(Path(output) / "state.json", pilot().encoded(state))


def _relative(path, root):
    return Path(path).relative_to(root).as_posix()


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def _label():
    return {"is_fixture": True, "label": FIXTURE_LABEL}


def load_state(output_dir):
    """Read ``state.json`` and prove the frozen section and its published inputs are unchanged."""
    output = Path(output_dir)
    path = output / "state.json"
    if not path.is_file():
        raise FreshChapterError("No prepared fresh chapter state; run prepare first")
    state = json.loads(path.read_bytes())
    if state.get("schema_version") != STATE_SCHEMA:
        raise FreshChapterError("Unknown fresh chapter state schema")
    if _frozen_sha(state) != state.get("frozen_sha256"):
        raise FreshChapterError("Frozen chapter lines, plans, cast or sources changed since preparation")
    p = pilot()
    for label, row in (("request", state["request"]), ("base plan", state["base_plan"])):
        if row and p.sha((output / row["file"]).read_bytes()) != row["sha256"]:
            raise FreshChapterError(f"Published {label} changed since preparation")
    _verify_memory(output, state)
    return state


def _require_same(state, cfg):
    mine = (cfg.project, cfg.chapter, pilot().sha(cfg.epub.read_bytes()))
    source = state["sources"][0]
    theirs = (source["project_id"], source["chapter_number"], state["epub"]["sha256"])
    if mine != theirs:
        raise FreshChapterError("This output directory was prepared for a different EPUB, chapter or project")
    binding, supplied = state["memory"], cfg.performance_memory
    if binding is None:
        if supplied is not None:
            raise FreshChapterError("This output directory was prepared without performance memory; "
                                    "use a new output directory to add one")
    elif supplied is None:
        raise FreshChapterError("This chapter is bound to performance memory " + binding["memory_sha256"][:16]
                                + "; supply the same --performance-memory to resume (it is never silently reused)")
    elif _read_memory(supplied, state["epub"]["sha256"])["summary"]["memory_sha256"] != binding["memory_sha256"]:
        raise FreshChapterError("The supplied performance memory differs from the memory bound at preparation; holding")


def _build_plan(project, segments, *, cache_root, selected_ids):
    return plans.build_scene_dialogue_plan(project, segments, cache_root=cache_root, selected_ids=selected_ids)


def _verify_plan(plan, project, segments, label):
    result = plans.verify_scene_dialogue_plan(plan, project, segments)
    if not result["current"]:
        raise FreshChapterError(f"{label} is no longer current for the live source/cast: "
                                + json.dumps(result["problems"], ensure_ascii=False)[:600])


# ---------------------------------------------------------------------------------------- clone evidence


def _safe_file(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    try:
        path.relative_to(folder)
    except ValueError as exc:
        raise FreshChapterError(f"Evidence path escapes its project: {relative}") from exc
    if (folder / relative).is_symlink():
        raise FreshChapterError(f"Evidence path is a symlink: {relative}")
    return path


def _casting_evidence(folder, project, role_id, voice):
    """The current human casting approval for one role's clone, as the existing recorded-draft owner reads it.

    A status string is not approval: the assignment, the immutable heard-sample record, the revision it was made
    for and the positive human event that chose it must all agree and the heard sample's bytes must verify.
    """
    from dubbing_service.app.models.project import Role
    lab = project.get("voice_lab") or {}
    casting = lab.get("casting") or {}
    assignment = ((casting.get("roles") or {}).get(role_id) or {}).get("assignment") or {}
    key = assignment.get("key")
    record = (lab.get("voices") or {}).get(key) if key else None
    record = record or {}
    event = (casting.get("events") or {}).get(assignment.get("event_id")) or {}
    identity = record.get("identity") or {}
    role = next((r for r in project.get("roles") or [] if r.get("id") == role_id), None)
    stars = event.get("stars")
    positive = event.get("selection_confirmed") is True or (type(stars) is int and stars >= 4)
    problems = []

    def need(ok, text):
        if not ok:
            problems.append(text)

    need(assignment.get("approved") is True and assignment.get("rejected") is not True,
         "no current approved casting assignment")
    need(bool(key) and bool(record), "the assignment names no voice record")
    need(identity.get("id") == voice["id"] and record.get("voice_id") == voice["id"],
         "the approved record is for a different voice identity")
    need(identity.get("reference_audio_sha256") == voice["reference_audio_sha256"]
         and identity.get("prompt_sha256") == voice["prompt_sha256"],
         "the approved record's conditioning differs from the role's clone")
    need(assignment.get("voice_revision") == record.get("version"), "the approval was made for another voice revision")
    revision = None
    if role is not None:
        revision = voice_revision_for_role(Role.model_validate(role))
    need(revision is not None and revision == record.get("version"), "the role's voice revision no longer matches its approval")
    need(event.get("origin") == "human" and event.get("action") == "use"
         and isinstance(event.get("actor"), str) and event["actor"].strip(), "no human 'use' event backs the assignment")
    need(event.get("role_id") == role_id and event.get("key") == key and event.get("voice_id") == voice["id"]
         and event.get("voice_version") == record.get("version"), "the human event is for another role or voice")
    need(positive, "the human selection is not positive")
    need(isinstance(record.get("audio_sha256"), str) and event.get("audio_sha256") == record.get("audio_sha256"),
         "the human event is for another heard sample")
    if not problems:
        try:
            data = _safe_file(folder, record.get("audio_file") or "").read_bytes()
        except OSError:
            data = None
        need(data is not None and pilot().sha(data) == record["audio_sha256"], "the approved heard sample is missing or changed")
    if problems:
        raise FreshChapterError(f"{role_id}: approval evidence is not current: " + "; ".join(problems))
    return {"role_id": role_id, "key": key, "version": record["version"], "event_id": assignment["event_id"],
            "voice_revision": revision, "heard_sample_file": record["audio_file"], "heard_sample_sha256": record["audio_sha256"],
            "event": {k: event.get(k) for k in ("actor", "origin", "action", "stars", "selection_confirmed")}}


def _source_asset(folder, project, role_id):
    """Exact recorded-reference source asset: the acquisition receipt the clone's provenance names must verify."""
    role = next((r for r in project.get("roles") or [] if r.get("id") == role_id), {})
    provenance = (role.get("voice_identity") or {}).get("provenance") or {}
    acquisition, receipt_sha = provenance.get("acquisition_id"), provenance.get("acquisition_sha256")
    if not (isinstance(acquisition, str) and _HEX32.fullmatch(acquisition)
            and isinstance(receipt_sha, str) and _HEX64.fullmatch(receipt_sha)):
        raise FreshChapterError(f"{role_id}: the clone names no recorded acquisition; its source asset is unverifiable")
    relative = f"voice_lab/discovery/acquisitions/{acquisition}/acquisition.json"
    try:
        data = _safe_file(folder, relative).read_bytes()
    except OSError:
        data = None
    if data is None or pilot().sha(data) != receipt_sha:
        raise FreshChapterError(f"{role_id}: the acquisition receipt for the clone's source asset is missing or changed")
    return {"acquisition_id": acquisition, "acquisition_sha256": receipt_sha, "receipt_file": relative,
            "provenance_origin": provenance.get("origin"), "provenance_sha256": pilot().value_sha(provenance)}


def _reference_record(folder, voice):
    path = Path(folder) / "casting" / "voices" / voice["id"] / "reference.json"
    if not path.is_file():
        return None
    data = path.read_bytes()
    provenance = json.loads(data).get("provenance", {})
    return {"file": _relative(path, folder), "sha256": pilot().sha(data),
            "audition_draft_policy": bool(provenance.get("audition_draft_policy"))}


def _verify_assets(folder, voice):
    """Clone bytes as assets only: path scope and SHA-256 of the reference and prompt. No draft approval is consulted,
    because an existing-cast target holds no selection of its own."""
    for path_key, hash_key in (("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")):
        relative = voice[path_key]
        if relative.startswith(("/", "\\")) or ".." in Path(relative).parts \
                or not relative.startswith(f"casting/voices/{voice['id']}/"):
            raise FreshChapterError(f"Clone artifact path is outside its identity: {relative}")
        path = Path(folder) / relative
        if not path.is_file() or pilot().sha(path.read_bytes()) != voice[hash_key]:
            raise FreshChapterError(f"Clone artifact is missing or hash-mismatched: {relative}")


def _verify_target_roles(folder, project, roles, *, owner_review=None):
    """Re-derive and compare every frozen role's clone files, cast evidence and source asset against ``folder``.

    ``approved`` roles carry the current human casting approval; ``existing_book_cast`` roles are validated as clone
    assets only (their authorization is the original provider context, checked by ``_verify_providers``)."""
    p = pilot()
    table = p.role_table(project)
    metadata = project.get("metadata") or {}
    for role_id, role in roles.items():
        entry = table.get(role_id)
        if entry is None or entry["identity"] != role["voice"] or entry["name"] != role["name"]:
            raise FreshChapterError(f"{role_id}: role or cloned voice identity drifted since preparation")
        if role["mode"] == MODE_EXISTING:
            _verify_assets(folder, role["voice"])
            if (metadata.get("existing_book_cast_sources") or {}).get(role_id) != role["lineage"]["declaration"] \
                    or metadata.get("owner_review_authorization") != owner_review:
                raise FreshChapterError(f"{role_id}: existing-cast declaration or owner authorization drifted since preparation")
        else:
            p.verify_identity_files(Path(folder), role["voice"])
            if _casting_evidence(folder, project, role_id, role["voice"]) != role["casting"]:
                raise FreshChapterError(f"{role_id}: approval evidence drifted since preparation")
        if _source_asset(folder, project, role_id) != role["source_asset"]:
            raise FreshChapterError(f"{role_id}: clone source asset drifted since preparation")
        if _reference_record(folder, role["voice"]) != role["reference_record"]:
            raise FreshChapterError(f"{role_id}: clone reference record drifted since preparation")


# ---------------------------------------------------------------------------------------- existing book cast

_DECLARATION_KEYS = ("project_id", "role_id", "book_sha256", "voice_id", "reference_audio_sha256", "prompt_sha256",
                     "project_sha256", "origin_project_sha256", "origin_source_root", "target_approval_created")


def _owner_authorization(metadata, declared, epub_sha256, chapter_number):
    """The explicit private owner request that permits reusing the retained cast for this one chapter review."""
    auth = metadata.get("owner_review_authorization")
    if not isinstance(auth, dict):
        raise FreshChapterError("Existing-cast reuse needs metadata.owner_review_authorization (fresh-chapter-owner-review-v1)")
    binding = pilot().value_sha({rid: {"provider_project_id": e["project_id"], "voice_id": e["voice_id"],
                                       "reference_audio_sha256": e["reference_audio_sha256"],
                                       "prompt_sha256": e["prompt_sha256"]} for rid, e in declared.items()})
    problems = []
    if auth.get("schema") != OWNER_REVIEW_SCHEMA:
        problems.append("schema")
    if auth.get("authorized") is not True:
        problems.append("not authorized")
    if auth.get("scope") != "private_chapter_review":
        problems.append("scope")
    if auth.get("book_sha256") != epub_sha256:
        problems.append("another book")
    if type(auth.get("chapter_number")) is not int or auth.get("chapter_number") != chapter_number:
        problems.append("another chapter")
    if auth.get("author") != "user_request" or not isinstance(auth.get("instruction"), str) or not auth["instruction"].strip():
        problems.append("author or instruction")
    if auth.get("role_voice_bindings_sha256") != binding:
        problems.append("role/voice bindings differ from the declared cast")
    if auth.get("quality_approval_created", False) is not False:
        problems.append("it claims a quality approval")
    if problems:
        raise FreshChapterError("Owner review authorization refused: " + "; ".join(problems))
    return copy.deepcopy(auth)


def _declaration(role_id, entry, voice, epub_sha256, target_project_id):
    if not isinstance(entry, dict) or set(entry) - set(_DECLARATION_KEYS) - {"authorization"}:
        raise FreshChapterError(f"{role_id}: existing-cast declaration refused: it has unknown fields")
    missing = [k for k in _DECLARATION_KEYS if k not in entry]
    problems = [f"missing {k}" for k in missing]
    if not missing:
        if not isinstance(entry["project_id"], str) or not _ID.fullmatch(entry["project_id"]) or entry["project_id"] == target_project_id:
            problems.append("provider project is invalid or is the target itself")
        if entry["role_id"] != role_id:
            problems.append("declared for another role")
        if entry["book_sha256"] != epub_sha256:
            problems.append("declared for another book")
        if (entry["voice_id"], entry["reference_audio_sha256"], entry["prompt_sha256"]) != (
                voice["id"], voice["reference_audio_sha256"], voice["prompt_sha256"]):
            problems.append("declared voice or conditioning differs from the role's clone")
        if entry["target_approval_created"] is not False:
            problems.append("target_approval_created must be false")
        for key in ("project_sha256", "origin_project_sha256"):
            if not isinstance(entry[key], str) or not _HEX64.fullmatch(entry[key]):
                problems.append(f"{key} is not a SHA-256")
        if entry["project_sha256"] != entry["origin_project_sha256"]:
            problems.append("staged and origin project hashes were not recorded identical")
        if not isinstance(entry["origin_source_root"], str) or not entry["origin_source_root"].strip():
            problems.append("no origin source root")
    if problems:
        raise FreshChapterError(f"{role_id}: existing-cast declaration refused: " + "; ".join(problems))
    return copy.deepcopy(entry)


def _source_use(folder, asset, voice):
    """Eligible recorded source-use proven from the hash-bound acquisition receipt (the synthesis owner re-proves it
    from the audio). This is permission to use the recording, never a statement about voice quality or role fit."""
    from .performer_discovery import reference_eligibility
    receipt = json.loads(_safe_file(folder, asset["receipt_file"]).read_bytes())
    eligibility = reference_eligibility(receipt.get("source") or {}, intended_use_evidence=(receipt.get("eligibility") or {}).get(
        "intended_use_evidence"))
    technical = receipt.get("technical_qa") or {}
    problems = []
    if receipt.get("schema") != "voice-lab-acquisition-v1" or receipt.get("acquisition_id") != asset["acquisition_id"]:
        problems.append("receipt schema or identity")
    if not eligibility["acquisition_eligible"]:
        problems.append("no eligible recorded source-use evidence: " + "; ".join(eligibility["hold_reasons"])[:300])
    if receipt.get("clone_ready") is not True or technical.get("status") != "pass":
        problems.append("not clone-ready with passing technical QA")
    if receipt.get("reference_audio_sha256") != voice["reference_audio_sha256"]:
        problems.append("the receipt's reference differs from the clone")
    if problems:
        raise FreshChapterError(f"{voice['id']}: source-use is not proven: " + "; ".join(problems))
    evidence = eligibility["intended_use_evidence"] or {}
    return {"acquisition_id": asset["acquisition_id"], "acquisition_sha256": asset["acquisition_sha256"],
            "eligible_recorded_source": True, "clone_ready": True, "technical_qa": technical.get("status"),
            "intended_use": eligibility["intended_use"], "use_authority": evidence.get("authority"),
            "use_evidence_ref": evidence.get("evidence_ref"), "source_id": (receipt.get("source") or {}).get("source_id"),
            "meaning": "permission to use this recording for private synthetic voice; not voice-quality or role-fit approval"}


def _original_selection(folder, project, voice, acquisition_sha256):
    """The provider context's own positive human selection for a draft clone, as the synthesis owner reads it."""
    from . import recorded_audition_policy as draft
    if not draft.human_casting_context(Path(folder), voice["id"], acquisition_sha256):
        raise FreshChapterError(f"{voice['id']}: the original context has no current positive human selection for this "
                                "recorded draft")
    casting = (project.get("voice_lab") or {}).get("casting") or {}
    for role_id, item in sorted((casting.get("roles") or {}).items()):
        assignment = item.get("assignment") or {}
        event = (casting.get("events") or {}).get(assignment.get("event_id")) or {}
        if assignment.get("approved") is True and event.get("voice_id") == voice["id"]:
            return {"original_role_id": role_id, "key": assignment.get("key"), "event_id": assignment.get("event_id"),
                    "audio_sha256": event.get("audio_sha256"),
                    "event": {k: event.get(k) for k in ("actor", "origin", "action", "stars", "selection_confirmed")}}
    return {"verified_by": "recorded_audition_policy.human_casting_context"}


def _provider_evidence(folder, project, role_id, voice):
    """Everything about one retained clone's original context that can drift, derived from ``folder``."""
    from .cloned_voice_policy import role_holds
    p = pilot()
    policy_mode(project)
    raw = next((r for r in project.get("roles") or [] if r.get("id") == role_id), None)
    entry = p.role_table(project).get(role_id)
    if raw is None or entry is None or entry["identity"] != voice or role_holds(raw, mode="cloned_only") \
            or raw.get("tts_backend") != "qwen3tts":
        raise FreshChapterError(f"{role_id}: the original context does not hold this exact recorded clone on the Qwen Base route")
    p.verify_identity_files(Path(folder), voice)  # includes the recorded draft's positive human selection
    asset = _source_asset(folder, project, role_id)
    record = _reference_record(folder, voice)
    return {"source_asset": asset, "source_use": _source_use(folder, asset, voice), "reference_record": record,
            "original_selection": _original_selection(folder, project, voice, asset["acquisition_sha256"])
            if record and record["audition_draft_policy"] else None}


def _origin_status(origin_root, provider_id, declaration, staged_contract_sha256):
    """The origin must still be the project that was staged. A byte-identical project.json is exact; otherwise only the
    pilot's own delivery-notification bookkeeping may differ (identical semantic source contract)."""
    p = pilot()
    path = Path(origin_root) / provider_id / "project.json"
    if not path.is_file():
        raise FreshChapterError(f"{provider_id}: the origin book-role context is unavailable at {origin_root}")
    data = path.read_bytes()
    if p.sha(data) == declaration["origin_project_sha256"]:
        return {"origin_bytes_identical": True, "origin_contract_sha256": staged_contract_sha256}
    contract = p.value_sha(p.source_contract(json.loads(data)))
    if contract != staged_contract_sha256:
        raise FreshChapterError(f"{provider_id}: the origin book-role context changed (source contract differs)")
    return {"origin_bytes_identical": False, "origin_contract_sha256": contract}


def _provider_context(cfg, role_id, name, voice, identity_raw, declaration, epub_sha256, target_asset, target_reference):
    """Validate one retained cast member against its immutable original book-role context (read only)."""
    p = pilot()
    provider_id = declaration["project_id"]
    staged = p.read_source(cfg.source_root, provider_id)
    if staged["project_sha256"] != declaration["project_sha256"]:
        raise FreshChapterError(f"{provider_id}: the staged original context differs from its recorded hash")
    folder = cfg.source_root / provider_id
    project = staged["project"]
    if not any(r.get("sha256") == epub_sha256 for r in project.get("source_revisions") or []):
        raise FreshChapterError(f"{provider_id}: the original context is not bound to this book")
    original = next((r for r in project.get("roles") or [] if r.get("id") == role_id), None)
    if original is None or original.get("name") != name:
        raise FreshChapterError(f"{provider_id}: the original context has no role {role_id} named {name!r}")
    if original.get("voice_identity") != identity_raw:
        raise FreshChapterError(f"{provider_id}:{role_id}: the retained clone differs from the original book-role clone")
    contract_sha = p.value_sha(p.source_contract(project))
    origin = _origin_status(declaration["origin_source_root"], provider_id, declaration, contract_sha)
    evidence = _provider_evidence(folder, project, role_id, voice)
    origin_folder = Path(declaration["origin_source_root"]) / provider_id
    if _provider_evidence(origin_folder, json.loads((origin_folder / "project.json").read_bytes()), role_id, voice) != evidence:
        raise FreshChapterError(f"{provider_id}:{role_id}: the origin evidence differs from the staged evidence")
    if evidence["source_asset"] != target_asset or evidence["reference_record"] != target_reference:
        raise FreshChapterError(f"{role_id}: the target's clone source asset or reference record differs from the original context")
    return {"provider_project_id": provider_id, "project_sha256": staged["project_sha256"],
            "segments_sha256": staged["segments_sha256"], "contract_sha256": contract_sha, **origin, "evidence": evidence}


def _verify_tree(folder, tree):
    p = pilot()
    for relative, digest in tree.items():
        path = Path(folder) / relative
        if not path.is_file() or p.sha(path.read_bytes()) != digest:
            raise FreshChapterError(f"Provider context file changed or missing: {relative}")


def _verify_providers(cfg, state):
    """Staged, isolated and origin copies of every original book-role context against the frozen contract."""
    p, out = pilot(), cfg.output_dir
    for provider_id, provider in state["providers"].items():
        staged, isolated = cfg.source_root / provider_id, out / "projects" / provider_id
        live = p.read_source(cfg.source_root, provider_id)
        if live["project_sha256"] != provider["project_sha256"]:
            raise FreshChapterError(f"{provider_id}: the staged original book-role context changed since preparation")
        _verify_tree(staged, provider["tree"])
        _verify_tree(isolated, provider["tree"])
        copy_project = p.read_source(out / "projects", provider_id)
        if copy_project["project_sha256"] != provider["project_sha256"]:
            raise FreshChapterError(f"{provider_id}: the isolated provider context changed since preparation")
        # Only delivery-notification bookkeeping may move at the origin; its semantic source contract may not.
        _origin_status(provider["origin_source_root"], provider_id, provider["declaration"], provider["contract_sha256"])
        origin_folder = Path(provider["origin_source_root"]) / provider_id
        origin_project = json.loads((origin_folder / "project.json").read_bytes())
        for role_id in provider["roles"]:
            role = state["roles"][role_id]
            for where, project in ((staged, live["project"]), (isolated, copy_project["project"]),
                                   (origin_folder, origin_project)):
                if _provider_evidence(where, project, role_id, role["voice"]) != role["lineage"]["evidence"]:
                    raise FreshChapterError(f"{role_id}: provider approval/source/asset evidence drifted since preparation")


def _cast_reuse(owner_review, roles):
    """Chapter-level statement of how the cast was authorized; identical wording in every receipt."""
    existing = sorted(rid for rid, role in roles.items() if role["mode"] == MODE_EXISTING)
    return {**EXISTING_CAST, "existing_book_cast_roles": existing,
            "current_target_approval_roles": sorted(set(roles) - set(existing)),
            "owner_review_authorization": owner_review} if existing else {
        "mode": MODE_TARGET, "existing_book_cast_roles": [], "current_target_approval_roles": sorted(roles),
        "owner_review_authorization": None}


def _clone_binding(role):
    voice = role["voice"]
    binding = {"role_id": role["role_id"], "role_name": role["name"], "narrator": role["narrator"],
               "voice_id": voice["id"], "reference_audio": voice["reference_audio"],
               "reference_audio_sha256": voice["reference_audio_sha256"], "prompt_artifact": voice["prompt_artifact"],
               "prompt_sha256": voice["prompt_sha256"], "voice_identity_sha256": voice["voice_identity_sha256"],
               "voice_revision": role["voice_revision"], "backend": voice["backend"], "mechanism": voice["mechanism"],
               "source_asset": role["source_asset"], "cast_mode": role["mode"],
               "provider_project_id": role["provider_project_id"], "casting_approval": None, "existing_cast": None}
    if role["mode"] == MODE_EXISTING:
        lineage = role["lineage"]
        binding["existing_cast"] = {k: v for k, v in lineage.items() if k != "evidence"}
        binding["existing_cast"].update(source_use=lineage["evidence"]["source_use"],
                                        original_selection=lineage["evidence"]["original_selection"])
    else:
        binding["casting_approval"] = {k: role["casting"][k] for k in ("key", "version", "event_id", "heard_sample_sha256")}
    return binding


# ---------------------------------------------------------------------------------------- prepare


def _first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _verify_prose(epub_text, segments):
    """Every nonblank EPUB character must be present, in order, in the live source segments."""
    expected = re.sub(r"\s+", "", epub_text)
    actual = re.sub(r"\s+", "", "".join(segment.text for segment in segments))
    if expected != actual:
        at = _first_difference(expected, actual)
        raise FreshChapterError(f"Canonical chapter does not cover every EPUB prose character; first difference at "
                                f"nonblank character {at}: epub {expected[max(0, at - 20):at + 20]!r} vs "
                                f"source {actual[max(0, at - 20):at + 20]!r}")
    return len(expected)


def _build_request(project_id, plan, segments, roles, lines):
    """One whole-scene request: all context incl. narration; directions only for dialogue ids."""
    live = {s.id: s for s in segments}
    dialogue = {line["utterance_id"] for line in lines if not line["narrator"]}
    scenes, covered = [], []
    for scene in plan["scenes"]:
        selected = [uid for uid in scene["utterance_ids"] if uid in dialogue]
        if not selected:
            continue
        context = []
        for uid in scene["utterance_ids"]:
            segment = live.get(uid)
            if segment is None:
                raise FreshChapterError(f"Scene {scene['scene_id']} references {uid}, which is not live source")
            role = roles.get(segment.role_id)
            context.append({"utterance_id": uid, "text": segment.text, "role_id": segment.role_id,
                            "narrator": bool(role and role["narrator"]), "role_name": role["name"] if role else None})
        scenes.append({"scene_id": scene["scene_id"], "project_id": project_id, "context": context,
                       "selected_dialogue_ids": selected})
        covered += selected
    if sorted(covered) != sorted(dialogue) or len(set(covered)) != len(covered):
        raise FreshChapterError(f"{project_id}: scene plan does not cover each dialogue line exactly once")
    return {"instruction": REQUEST_INSTRUCTION, "constraints": list(REQUEST_CONSTRAINTS),
            "response_contract": {
                "schema_version": scene_coaching.RESPONSE_SCHEMA,
                "scenes": "[{scene_id, directions: [one object per selected_dialogue_ids entry]}]",
                "direction_fields": sorted(scene_coaching.FIELDS),
                "intensity": ["moderate", "strong"], "confidence": ["low", "moderate", "high"]},
            "scenes": scenes}


def _memory_module():
    try:
        from . import character_performance_memory
    except ImportError as exc:
        raise FreshChapterError("Character performance memory is unavailable (character_performance_memory); "
                                "prepare without --performance-memory or wait for its owner") from exc
    return character_performance_memory


def _read_memory(path, book_sha256):
    """Parse and fully validate an optional performance-memory bundle against this exact book."""
    path = Path(path)
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_MEMORY_BYTES:
        raise FreshChapterError("Performance memory is empty or exceeds its size bound")
    try:
        bundle = json.loads(raw)
    except ValueError as exc:
        raise FreshChapterError(f"Performance memory is not valid JSON: {exc}") from exc
    summary = _memory_module().validate_bundle(bundle)
    if summary["book_sha256"] != book_sha256:
        raise FreshChapterError("Performance memory was captured for a different book than this EPUB")
    return {"summary": summary, "raw": raw, "bundle": bundle}


def _memory_binding(memory):
    summary = memory["summary"]
    return {"schema_version": MEMORY_BINDING_SCHEMA, "memory_sha256": summary["memory_sha256"], "file": MEMORY_FILE,
            "file_sha256": pilot().sha(memory["raw"]), "book_sha256": summary["book_sha256"],
            "request_sha256": summary["request_sha256"], "response_sha256": summary["response_sha256"],
            "model_requested": summary["provenance"]["model_requested"], "profile_roles": sorted(summary["profiles"]),
            "narrator_roles": list(summary["narrator_role_ids"]), "interpretation": _memory_module().STATUS,
            "approved": False, "fixture": memory["bundle"]["provenance"].get("fixture") is True}


def _carried_memory(request_bytes):
    """The memory hash a frozen request carries (None for a plain request). No memory owner is needed to say 'none'."""
    request = json.loads(request_bytes)
    scenes = request.get("scenes") if isinstance(request.get("scenes"), list) else []
    constraints = request.get("constraints") if isinstance(request.get("constraints"), list) else []
    marked = any(isinstance(c, str) and c.startswith(MEMORY_CONSTRAINT_PREFIX) for c in constraints)
    keyed = any(isinstance(s, dict) and MEMORY_SCENE_KEY in s for s in scenes)
    if not marked and not keyed:
        return None
    try:
        return _memory_module().enriched_memory_sha256(request)
    except ValueError as exc:
        raise FreshChapterError(str(exc)) from exc


def _verify_memory(output, state):
    """A bound state must still hold its immutable private memory and carry it in the frozen request."""
    binding, row = state["memory"], state["request"]
    carried = _carried_memory((output / row["file"]).read_bytes()) if row else None
    if binding is None:
        if carried or (row and "performance_memory_sha256" in row):
            raise FreshChapterError("The frozen request carries a performance memory but the state has no memory binding")
        return
    if binding.get("schema_version") != MEMORY_BINDING_SCHEMA or row is None:
        raise FreshChapterError("Unknown performance memory binding")
    path = output / binding["file"]
    if not path.is_file() or pilot().sha(path.read_bytes()) != binding["file_sha256"]:
        raise FreshChapterError("The private performance memory bound to this chapter is missing or changed")
    if _read_memory(path, binding["book_sha256"])["summary"]["memory_sha256"] != binding["memory_sha256"]:
        raise FreshChapterError("The private performance memory no longer matches its binding")
    if carried != binding["memory_sha256"] or row.get("performance_memory_sha256") != binding["memory_sha256"]:
        raise FreshChapterError("The frozen request is not bound to the chapter's performance memory")


def _enrich(request, memory, book_sha256):
    """Apply captured performance memory to the request; it may add context, never change scenes' source or the contract."""
    module = _memory_module()
    try:
        enriched = module.enrich_coaching_request(copy.deepcopy(request), memory["bundle"], book_sha256=book_sha256)
    except ValueError as exc:
        raise FreshChapterError(f"Performance memory refused: {exc}") from exc
    scenes_ok = (isinstance(enriched, dict) and isinstance(enriched.get("scenes"), list)
                 and len(enriched["scenes"]) == len(request["scenes"])
                 and all(isinstance(new, dict) and all(k in new and new[k] == v for k, v in old.items())
                         for new, old in zip(enriched["scenes"], request["scenes"])))
    if (not scenes_ok or not isinstance(enriched.get("instruction"), str)
            or enriched.get("response_contract") != request["response_contract"]
            or not isinstance(enriched.get("constraints"), list)
            or any(item not in enriched["constraints"] for item in request["constraints"])):
        raise FreshChapterError("Performance memory may only add context: every scene's source fields, the response "
                                "contract and the base constraints must stay exactly as frozen")
    if module.enriched_memory_sha256(enriched) != memory["summary"]["memory_sha256"]:
        raise FreshChapterError("The enriched request does not carry the supplied performance memory")
    try:
        pilot().encoded(enriched)
    except (TypeError, ValueError) as exc:
        raise FreshChapterError(f"Enriched coaching request is not canonical JSON: {exc}") from exc
    return enriched


def _prepare(cfg, plan_builder):
    p, out = pilot(), cfg.output_dir
    project_id, source_folder = cfg.project, cfg.source_root / cfg.project
    data = cfg.epub.read_bytes()
    book = epub_chapter_selection.select_epub_chapters(data, cfg.epub.name, chapter_numbers=[cfg.chapter])
    chapter = book["chapters"][0]
    memory = None
    if cfg.performance_memory is not None:  # fully validated before any output exists
        memory = _read_memory(cfg.performance_memory, book["source_sha256"])
    with _projects_root(cfg.source_root):
        before = p.read_source(cfg.source_root, project_id)
        from dubbing_service.app.storage.projects import load_project
        probe = load_project(project_id)
        canonical = next((c for c in probe.chapters if c.title == chapter["title"]), None)
        if canonical is None or epub_chapter_selection.chapter_number_from_title(canonical.title) != chapter["chapter_number"]:
            raise FreshChapterError(f"{project_id} does not contain the exact numbered chapter {chapter['chapter_number']}")
        project, segments, source = exporter._source_snapshot(project_id, canonical.id)
        if not any(r.sha256 == book["source_sha256"] for r in project.source_revisions):
            raise FreshChapterError(f"{project_id} is not bound to this exact EPUB (source revision hash differs)")
        nonblank = _verify_prose(chapter["text"], segments)
        policy_mode(project)  # a malformed voice policy fails closed; cloned-only is enforced below regardless
        model_roles = {r.id: r for r in project.roles}
        speech_ids = set(source["speech_utterance_ids"])
        speech = [s for s in segments if s.id in speech_ids]
        table = p.role_table(before["project"])
        metadata = before["project"].get("metadata") or {}
        declared = metadata.get("existing_book_cast_sources")
        if declared is not None and not isinstance(declared, dict):
            raise FreshChapterError("metadata.existing_book_cast_sources must be an object keyed by canonical role id")
        declared, owner, roles, providers = declared or {}, None, {}, {}
        for segment in speech:
            role = model_roles.get(segment.role_id)
            attribution = segment.attribution
            if role is None or not attribution or attribution.source == "unknown" or attribution.speaker_id != role.id:
                raise FreshChapterError(f"{project_id}:{segment.id} needs resolved source-backed speaker attribution")
            if segment.playback_speed != 1.0 or segment.edit_state is not None:
                raise FreshChapterError(f"{project_id}:{segment.id} must be a complete unedited natural-speed utterance")
            if (segment.tts_backend or role.tts_backend) != "qwen3tts":
                raise FreshChapterError(f"{project_id}:{role.id} is not assigned to the Qwen Base route")
            if segment.role_id in roles:
                continue
            entry = table.get(role.id)
            if entry is None or entry["identity"] is None:
                raise FreshChapterError(f"{project_id}:{role.id} has no locked eligible cloned identity "
                                        f"({', '.join(entry['clone_holds']) if entry else 'unknown role'})")
            if role.voice_identity is None or role.voice_identity.role_id != role.id:
                raise FreshChapterError(f"{project_id}:{role.id} voice identity belongs to another role")
            voice = entry["identity"]
            base = {"role_id": role.id, "name": entry["name"], "narrator": entry["narrator"],
                    "tts_backend": role.tts_backend, "voice": voice, "voice_revision": voice_revision_for_role(role),
                    "source_asset": _source_asset(source_folder, before["project"], role.id),
                    "reference_record": _reference_record(source_folder, voice)}
            if role.id not in declared:  # current approved-role mode: the target's own human casting approval
                p.verify_identity_files(source_folder, voice)
                roles[role.id] = {**base, "mode": MODE_TARGET, "provider_project_id": None, "lineage": None,
                                  "casting": _casting_evidence(source_folder, before["project"], role.id, voice)}
                continue
            # Explicit owner-authorized private review of the retained cast: the target gets no approval; the
            # immutable original book-role context authorizes the backend call.
            if owner is None:
                owner = _owner_authorization(metadata, declared, book["source_sha256"], chapter["chapter_number"])
            declaration = _declaration(role.id, declared[role.id], voice, book["source_sha256"], project_id)
            _verify_assets(source_folder, voice)  # target clone bytes as assets; no target draft approval exists
            identity_raw = next(r for r in before["project"]["roles"] if r["id"] == role.id)["voice_identity"]
            context = _provider_context(cfg, role.id, entry["name"], voice, identity_raw, declaration, book["source_sha256"],
                                        base["source_asset"], base["reference_record"])
            provider = providers.setdefault(context["provider_project_id"], {
                "project_id": context["provider_project_id"], "project_sha256": context["project_sha256"],
                "segments_sha256": context["segments_sha256"], "contract_sha256": context["contract_sha256"],
                "origin_source_root": declaration["origin_source_root"],
                "origin_project_sha256": declaration["origin_project_sha256"],
                "origin_bytes_identical": context["origin_bytes_identical"],
                "declaration": declaration, "roles": [], "voices": []})
            if (provider["origin_source_root"], provider["origin_project_sha256"], provider["project_sha256"]) != (
                    declaration["origin_source_root"], declaration["origin_project_sha256"], context["project_sha256"]):
                raise FreshChapterError(f"{context['provider_project_id']}: declarations disagree about one original context")
            provider["roles"].append(role.id)
            provider["voices"] = sorted(set(provider["voices"]) | {voice["id"]})
            roles[role.id] = {**base, "mode": MODE_EXISTING, "provider_project_id": context["provider_project_id"],
                              "casting": None,
                              "lineage": {"mode": MODE_EXISTING, "source_project_id": project_id,
                                          "provider_project_id": context["provider_project_id"],
                                          "origin_source_root": declaration["origin_source_root"],
                                          "project_sha256": context["project_sha256"],
                                          "origin_project_sha256": declaration["origin_project_sha256"],
                                          "origin_bytes_identical": context["origin_bytes_identical"],
                                          "origin_contract_sha256": context["origin_contract_sha256"],
                                          "book_sha256": book["source_sha256"], "target_approval_created": False,
                                          "role_fit_approval": "not_established", "human_listening": "pending",
                                          "owner_authorization_sha256": p.value_sha(owner),
                                          "declaration": declaration, "evidence": context["evidence"]}}
        if not any(not r["narrator"] for r in roles.values()):
            raise FreshChapterError(f"{project_id} has no non-narrator speech to coach")
        fixture = project.metadata.get("fresh_chapter_fixture") is True
        if memory is not None:
            if memory["bundle"]["provenance"].get("fixture") is True and not fixture:
                raise FreshChapterError(f"{project_id}: a fixture performance memory may not condition a non-fixture chapter")
            try:
                _memory_module().require_role_coverage(memory["summary"], [
                    (rid, role["name"], role["narrator"]) for rid, role in sorted(roles.items())])
            except ValueError as exc:
                raise FreshChapterError(f"{project_id}: performance memory does not fit this cast: {exc}") from exc

        registry = {r.id: r.sha256 for r in project.source_revisions}
        lines = []
        for segment in speech:
            role, anchor = roles[segment.role_id], segment.source_anchor
            lines.append({
                "utterance_id": segment.id, "project_id": project_id, "chapter_number": chapter["chapter_number"],
                "chapter_id": canonical.id, "index": segment.index, "role_id": segment.role_id,
                "role_name": role["name"], "narrator": role["narrator"], "scene_id": None,
                "mode": "fresh_neutral_narration" if role["narrator"] else "fresh_directed_dialogue",
                "text": segment.text, "text_sha256": p.text_sha(segment.text), "chars": len(segment.text),
                "source": {"source_revision_id": anchor.source_revision_id,
                           "source_revision_sha256": registry[anchor.source_revision_id],
                           "start_offset": anchor.start_offset, "end_offset": anchor.end_offset,
                           "occurrence_index": anchor.occurrence_index},
                "voice": role["voice"], "voice_revision": role["voice_revision"],
                "provider_project_id": role["provider_project_id"] or project_id})

        cache = out / "scene-cache"
        cache.mkdir(parents=True, exist_ok=True)
        base_file = out / "plans" / "base.json"
        if base_file.is_file():  # crashed earlier preparation: the published plan is the frozen one
            plan_bytes = base_file.read_bytes()
            plan = json.loads(plan_bytes)
        else:
            plan = plan_builder(project, segments, cache_root=cache, selected_ids=[s.id for s in speech])
            plan_bytes = p.encoded(plan)
            p.publish(base_file, plan_bytes)
        _verify_plan(plan, project, segments, f"Base plan for {project_id}")
        by_direction = {d["utterance_id"]: d for d in plan["directions"]}
        for line in lines:
            direction = by_direction.get(line["utterance_id"])
            if direction is None:
                raise FreshChapterError(f"Base plan for {project_id} omits speech line {line['utterance_id']}")
            line["scene_id"] = direction["scene_id"]
        holds = [h for h in plan.get("holds", []) if h.get("code") != "no_scene_evidence"]
        if plan.get("source", {}).get("proven") is not True:
            holds.append({"code": "source_unproven", "detail": "base plan has no proven source revision"})

        memory_info = None
        request = None if holds else _build_request(project_id, plan, segments, roles, lines)
        if request is not None and memory is not None:
            base_request_sha = plans._sha(request)
            request = _enrich(request, memory, book["source_sha256"])
            p.publish(out / MEMORY_FILE, memory["raw"])  # exact immutable private capture; identical retry is a no-op
            memory_info = {**_memory_binding(memory), "base_request_sha256": base_request_sha}
        request_info = None
        if request is not None:
            request_file = out / "requests" / "request.json"
            p.publish(request_file, p.encoded(request))
            request_info = {"file": _relative(request_file, out), "sha256": p.sha(request_file.read_bytes()),
                            "request_sha256": plans._sha(request), "scenes": len(request["scenes"]),
                            "dialogue_ids": [line["utterance_id"] for line in lines if not line["narrator"]]}
            if memory_info is not None:
                request_info["performance_memory_sha256"] = memory_info["memory_sha256"]

        voices = {role["voice"]["id"] for role in roles.values()}
        copied = p.prepare_isolated_project(source_folder, out / "projects" / project_id, voices)
        for provider in providers.values():  # exact original contexts, byte for byte, beside the retained target
            tree = p.prepare_isolated_project(cfg.source_root / provider["project_id"],
                                              out / "projects" / provider["project_id"], set(provider["voices"]))
            provider["tree"] = tree
            provider["isolated"] = {"files": len(tree), "tree_sha256": p.value_sha(tree)}
        _verify_target_roles(out / "projects" / project_id, before["project"], roles, owner_review=owner)
        _verify_providers(cfg, {"providers": providers, "roles": roles})
        after = p.read_source(cfg.source_root, project_id)
        if (after["project_sha256"], after["segments_sha256"]) != (before["project_sha256"], before["segments_sha256"]):
            raise FreshChapterError(f"Canonical source {project_id} changed during preparation")

        breaks = {b["utterance_id"] for b in source["breaks"]}
        excluded = {row["utterance_id"] for row in source.get("exclusions", [])}
        rows = [{"utterance_id": s.id, "index": s.index, "text_sha256": p.text_sha(s.text), "chars": len(s.text),
                 "kind": "scene_break" if s.id in breaks else "excluded" if s.id in excluded else "speech"}
                for s in segments]
        chapter_row = {
            "project_id": project_id, "chapter_number": chapter["chapter_number"], "chapter_id": canonical.id,
            "chapter_title": canonical.title, "epub_member": chapter["member"], "epub_text_sha256": chapter["text_sha256"],
            "nonblank_characters": nonblank, "rows": rows, "scene_breaks": source["breaks"],
            "exclusions": source.get("exclusions", []),
            "role_types": {rid: getattr(r.role_type, "value", r.role_type) for rid, r in model_roles.items()},
            "role_aliases": dict(project.character_cast.role_aliases), "fixture": fixture,
            "derived_from": {"source_project_id": source["metadata"]["source_project_id"],
                             "expected_source_hash": source["metadata"].get("expected_source_hash")}}
        return {
            "sources": [{"project_id": project_id, "chapter_number": chapter["chapter_number"], "chapter_id": canonical.id,
                         "chapter_title": canonical.title, "project_sha256": before["project_sha256"],
                         "segments_sha256": before["segments_sha256"],
                         "source_contract_sha256": p.value_sha(p.source_contract(before["project"]))}],
            "chapter": chapter_row, "roles": roles, "providers": providers, "owner_review": owner, "lines": lines,
            "request": request_info, "memory": memory_info,
            "base_plan": {"file": _relative(base_file, out), "sha256": p.sha(plan_bytes),
                          "binding_sha256": plan.get("binding_sha256"), "scenes": len(plan.get("scenes", []))},
            "isolated": {project_id: {"files": len(copied), "tree_sha256": p.value_sha(copied)}},
            "holds": [{**h, "project_id": project_id} for h in holds], "book": book}


def prepare_chapter(args, *, plan_builder=None):
    """Freeze source, approved clones and the coaching request. CPU only; status ``awaiting_coaching``."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    if cfg.epub is None:
        raise FreshChapterError("--epub is required")
    with _fail_closed():
        if (out / "state.json").is_file():
            state = load_state(out)
            _require_same(state, cfg)
            return state
        built = _prepare(cfg, plan_builder or _build_plan)
        book, holds = built["book"], built["holds"]
        state = {
            "schema_version": STATE_SCHEMA, "pipeline": "fresh-coached-chapter-v1", **REVIEW_FLAGS,
            "fixture": built["chapter"]["fixture"],
            "runtime": {"directed_profile": p.DIRECTED_PROFILE, "policy": p.DIRECTION_POLICY,
                        "request": "experimental review only; every speech line is newly synthesized, narration "
                                   "neutrally (no native instruction) on the same Base clone route"},
            "narration": NARRATION, "cast_reuse": _cast_reuse(built["owner_review"], built["roles"]),
            "epub": {"name": cfg.epub.name, "sha256": book["source_sha256"], "title": book["title"],
                     "author": book["author"], "selection": book["selection"]},
            "sources": built["sources"], "chapter": built["chapter"], "roles": built["roles"], "providers": built["providers"],
            "owner_review": built["owner_review"], "lines": built["lines"], "request": built["request"], "base_plan": built["base_plan"], "memory": built["memory"],
            "isolated": built["isolated"], "boundary_policy_sha256": p.boundary_policy_sha(), "prepare_holds": holds}
        state["frozen_sha256"] = _frozen_sha(state)
        state.update(status="held" if holds else "awaiting_coaching", holds=holds, coaching={})
        _write_state(out, state)
        return state


# ---------------------------------------------------------------------------------------- coaching


def _check_coached(state, request, bundle, coached, coached_bytes):
    """Validate the coached plan against the frozen lines/request and the captured bundle."""
    p = pilot()
    lines = state["lines"]
    if bundle.get("request") != request or bundle.get("request_sha256") != plans._sha(request):
        raise FreshChapterError("the bundle was not captured for the frozen request")
    scene_coaching.validate_bundle(bundle)
    coaching = coached.get("coaching") or {}
    if (coaching.get("request_sha256") != bundle["request_sha256"]
            or coaching.get("response_sha256") != bundle["response_sha256"]):
        raise FreshChapterError("coached plan is not bound to the supplied bundle")
    if (coached.get("source", {}).get("proven") is not True or coached.get("approved") is True
            or coached.get("recommendation_only") is not True or coached.get("human_review") != "pending"
            or coached.get("holds")):
        raise FreshChapterError("coached plan carries a source, approval or safety hold")
    directions, holds = p.apply_plan(coached, p.sha(coached_bytes), lines)
    if holds:
        raise FreshChapterError("direction guards refused the plan: " + json.dumps(holds, ensure_ascii=False)[:600])
    scene_of = {uid: scene["scene_id"] for scene in request["scenes"] for uid in scene["selected_dialogue_ids"]}
    proposed = {d["utterance_id"]: d for scene in bundle["response"]["scenes"] for d in scene["directions"]}
    for line in lines:
        direction = directions[line["utterance_id"]]
        instruction = direction["native_instruction"]
        if line["narrator"]:
            if instruction is not None:
                raise FreshChapterError(f"{line['utterance_id']} narrator carries an instruction")
            continue
        if (not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 220
                or direction["basis"] != "model" or not direction.get("coaching")
                or direction.get("coaching_scene_id") != scene_of.get(line["utterance_id"])
                or instruction != proposed[line["utterance_id"]]["native_instruction"]):
            raise FreshChapterError(f"{line['utterance_id']} lacks a complete, bundle-bound model direction")
        if direction["voice_revision"] != line["voice_revision"]:
            raise FreshChapterError(f"{line['utterance_id']} direction is bound to a different voice revision")
    if set(proposed) != set(scene_of):
        raise FreshChapterError("bundle directions do not match the requested dialogue set")
    return directions


def _reprove_source(cfg, state):
    """Live derived-chapter proof (source, coverage, breaks, exclusions) against the frozen rows; CPU, read only."""
    chapter = state["chapter"]
    with _projects_root(cfg.source_root):
        project, segments, source = exporter._source_snapshot(chapter["project_id"], chapter["chapter_id"])
    if source["speech_utterance_ids"] != [line["utterance_id"] for line in state["lines"]]:
        raise FreshChapterError("speech coverage changed since preparation")
    if source["breaks"] != chapter["scene_breaks"] or source.get("exclusions", []) != chapter["exclusions"]:
        raise FreshChapterError("scene-break or exclusion records changed since preparation")
    return project, segments, source


def _verify_live(cfg, state):
    """Canonical + isolated clone files, approvals, source asset and source contract against the frozen state."""
    p, out = pilot(), cfg.output_dir
    pid = state["sources"][0]["project_id"]
    live = p.read_source(cfg.source_root, pid)
    if p.value_sha(p.source_contract(live["project"])) != state["sources"][0]["source_contract_sha256"]:
        raise FreshChapterError("Canonical source contract changed since preparation (only delivery notification "
                                "counters may differ)")
    owner = state["owner_review"]
    _verify_target_roles(cfg.source_root / pid, live["project"], state["roles"], owner_review=owner)
    isolated = p.read_source(out / "projects", pid)
    _verify_target_roles(out / "projects" / pid, isolated["project"], state["roles"], owner_review=owner)
    _verify_providers(cfg, state)
    p.verify_prepared_sources(SimpleNamespace(source_root=cfg.source_root, output_dir=out), state)


def _apply(cfg, state, bundle_path):
    p, out = pilot(), cfg.output_dir
    request = _read_json(out / state["request"]["file"])
    raw = Path(bundle_path).read_bytes()
    bundle = json.loads(raw)
    published = out / "coaching" / "bundle.json"
    if published.exists() and published.read_bytes() != raw:
        raise FreshChapterError("a different coaching bundle was already applied; output is immutable")
    base = _read_json(out / state["base_plan"]["file"])
    with _projects_root(cfg.source_root):
        project, segments, _ = _reprove_source(cfg, state)
        _verify_plan(base, project, segments, "Base plan")
        scene_coaching.validate_bundle(bundle)
        try:  # no cache root: the plan bytes must be identical on every safe retry
            coached = scene_coaching.apply_coaching(base, bundle, project, segments, cache_root=None)
        except scene_coaching.CoachingError as exc:
            raise FreshChapterError(f"coaching refused: {exc}") from exc
        _verify_plan(coached, project, segments, "Coached plan")
    coached_bytes = p.encoded(coached)
    directions = _check_coached(state, request, bundle, coached, coached_bytes)
    receipt = {"schema_version": APPLIED_SCHEMA, "project_id": cfg.project, "bundle_sha256": p.sha(raw),
               "request_sha256": bundle["request_sha256"], "response_sha256": bundle["response_sha256"],
               "model_provenance": bundle["provenance"], "base_plan_sha256": state["base_plan"]["sha256"],
               "coached_plan_sha256": p.sha(coached_bytes), "performance_memory": state["memory"],
               "directions_sha256": p.value_sha({uid: d["native_instruction"] for uid, d in directions.items()}),
               "dialogue_directed": sum(1 for line in state["lines"] if not line["narrator"]),
               "narration_lines": sum(1 for line in state["lines"] if line["narrator"]), "narration": NARRATION,
               "cast_reuse": _cast_reuse(state["owner_review"], state["roles"]),
               "recommendation_only": True, "approved": False, "human_review": "pending"}
    if state["fixture"]:
        receipt["fixture"] = _label()
    p.publish(published, raw)
    p.publish(out / "plans" / "coached.json", coached_bytes)
    p.publish(out / "applied.json", p.encoded(receipt))
    return receipt


def apply_bundle(args, state):
    """Validate and publish the captured coaching bundle; ``ready`` only when it is applied."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    with _fail_closed():
        supplied, state = state, load_state(out)
        if _frozen_sha(supplied) != state["frozen_sha256"]:
            raise FreshChapterError("Supplied state differs from the frozen prepared chapter")
        if state["prepare_holds"]:
            raise FreshChapterError("Preparation held and cannot be coached: "
                                    + json.dumps(state["prepare_holds"], ensure_ascii=False)[:600])
        if cfg.coaching_bundle is None:
            return state
        if not cfg.coaching_bundle.is_file():
            raise FreshChapterError(f"Coaching bundle not found: {cfg.coaching_bundle}")
        _verify_live(cfg, state)
        holds = []
        try:
            receipt = _apply(cfg, state, cfg.coaching_bundle)
            ledger = {"status": "applied", "applied_sha256": p.value_sha(receipt),
                      "coached_plan_sha256": receipt["coached_plan_sha256"]}
        except (FreshChapterError, ValueError, p.PilotError) as exc:
            holds.append({"code": "coaching_refused", "project_id": cfg.project, "detail": str(exc)})
            ledger = {"status": "refused", "detail": str(exc)}
        status = ("generated" if state.get("status") == "generated" else "held" if holds else "ready")
        state.update(coaching=ledger, holds=holds, status=status)
        _write_state(out, state)
        return state


def _applied_directions(out, state):
    """Re-derive directions from the immutable published files, never from state.json."""
    p = pilot()
    receipt_path = out / "applied.json"
    if not receipt_path.is_file():
        raise FreshChapterError("no coaching has been applied; the chapter is not ready")
    receipt = _read_json(receipt_path)
    raw = (out / "coaching" / "bundle.json").read_bytes()
    coached_bytes = (out / "plans" / "coached.json").read_bytes()
    if receipt["bundle_sha256"] != p.sha(raw) or receipt["coached_plan_sha256"] != p.sha(coached_bytes):
        raise FreshChapterError("published coaching bundle or coached plan changed after it was applied")
    request = _read_json(out / state["request"]["file"])
    directions = _check_coached(state, request, json.loads(raw), json.loads(coached_bytes), coached_bytes)
    if p.value_sha({uid: d["native_instruction"] for uid, d in directions.items()}) != receipt["directions_sha256"]:
        raise FreshChapterError("directions differ from the applied receipt")
    return directions, receipt


# ---------------------------------------------------------------------------------------- generation


def _sample_path(out, line):
    return out / "samples" / f"{line['utterance_id']}.wav"


def _pending_files(audio):
    if not audio.parent.is_dir():
        return []
    return sorted(f for f in audio.parent.iterdir()
                  if f.name.startswith(audio.stem + ".pending") and f.name.endswith(".json"))


def _render_line(line):
    """The line as the synthesis owner sees it: the request is made in the original provider context, so its project id
    (and so the expected request SHA) is the provider's. The source line itself is never altered."""
    return {**line, "project_id": line["provider_project_id"]}


def _record_sample(out, state, line, render_receipt, pinned):
    p = pilot()
    audio = _sample_path(out, line)
    verified = p.verify_sample("directed", _render_line(line), render_receipt, pinned)
    if p.sha(audio.read_bytes()) != render_receipt["audio_sha256"]:
        raise FreshChapterError(f"{line['utterance_id']} audio bytes differ from the render receipt")
    worker = render_receipt.get("worker_receipt") or {}
    if worker.get("identity_id") not in (None, line["voice"]["id"]) \
            or worker.get("prompt_sha256") not in (None, line["voice"]["prompt_sha256"]):
        raise FreshChapterError(f"{line['utterance_id']} worker used a different voice identity than the frozen clone")
    report = qa.qa_take_audio(take_id="sample_" + render_receipt["audio_sha256"][:24], audio_path=audio,
                              text=line["text"], expected_sha256=render_receipt["audio_sha256"], language="English")
    if not report.passed or report.blocking_failures() or not report.executed_check_ids():
        raise FreshChapterError(f"{line['utterance_id']} new sample fails blocking technical QA")
    raw = audio.with_suffix(".receipt.json")
    role = state["roles"][line["role_id"]]
    record = {**p.sample_receipt("directed", _render_line(line), render_receipt, _relative(audio, out), verified),
              "schema_version": SAMPLE_SCHEMA, **REVIEW_FLAGS,
              "mode": line["mode"], "project_id": line["project_id"], "source_project_id": line["project_id"],
              "provider_project_id": line["provider_project_id"], "request_project_id": line["provider_project_id"],
              "cast_mode": role["mode"], "cast_reuse": _cast_reuse(state["owner_review"], state["roles"]),
              "raw_render_receipt": {"file": _relative(raw, out), "sha256": p.sha(raw.read_bytes())},
              "technical_qa": {"passed": True, "executed_checks": sorted(report.executed_check_ids())},
              "clone_binding": _clone_binding(role)}
    if line["narrator"]:
        record.update(audio_provenance="fresh_neutral_review_synthesis", narration=NARRATION, coaching=None)
    else:
        direction = line["direction"]
        record.update(audio_provenance="fresh_directed_review_synthesis", narration=None,
                      coaching={"scene_id": direction.get("coaching_scene_id"), "reading": direction.get("coaching"),
                                "basis": direction["basis"], "plan_sha256": direction["plan_sha256"]})
    if state["fixture"]:
        record["fixture"] = _label()
    p.publish(out / "receipts" / f"{line['utterance_id']}.json", p.encoded(record))
    return record


def _render_speech(cfg, state, rlines, worker_class, render_fn):
    """Serial, resumable synthesis of the missing speech. At most one worker for the whole chapter."""
    p, out = pilot(), cfg.output_dir
    for line in rlines:
        p.resolve_approval_refusal(out, "directed", _render_line(line), _sample_path(out, line))
    pending = [l for l in rlines if not _sample_path(out, l).with_suffix(".receipt.json").exists()]
    for line in pending:  # an unresolved dispatch is ambiguous: hold before any worker exists
        if _pending_files(_sample_path(out, line)):
            raise FreshChapterError(f"{line['utterance_id']} has an ambiguous pending dispatch record; "
                                    "inspect it before any GPU call (no resend)")
    pin_path = out / "runtime-pin.json"
    pinned = _read_json(pin_path)["runtime_sha256"] if pin_path.exists() else None
    todo = {l["utterance_id"] for l in pending}
    workers, receipts, dispatched = [], {}, []
    if render_fn is None or (pending and worker_class is None):
        default_worker, default_render = p.default_render_tools()
        worker_class, render_fn = worker_class or default_worker, render_fn or default_render
    worker = None
    try:
        if pending:
            worker = worker_class(out, directed=True)
            workers.append(worker)
            worker.__enter__()
            pinned = p.pin_runtime(out, worker.identity["sha256"], worker.identity)["runtime_sha256"]
        for line in rlines:
            instruction = line["direction"]["native_instruction"]
            if line["narrator"]:
                instruction = None  # neutral: same clone, no native instruction, ever
            elif not instruction:
                raise FreshChapterError(f"{line['utterance_id']} may not be synthesized without a model direction")
            render_receipt = render_fn(worker, line["provider_project_id"], line["voice"], _render_line(line),
                                       _sample_path(out, line), instruction)
            if line["utterance_id"] in todo:
                dispatched.append(line["utterance_id"])
            if pinned is None:
                raise FreshChapterError("No runtime pin exists to verify the completed samples against")
            receipts[line["utterance_id"]] = _record_sample(out, state, line, render_receipt, pinned)
    finally:
        for item in reversed(workers):
            item.__exit__(None, None, None)
    return receipts, pinned, workers, dispatched


def _audit(rlines, receipts, dispatched, workers):
    narration = [l for l in rlines if l["narrator"]]
    dialogue = [l for l in rlines if not l["narrator"]]
    narration_ids = {l["utterance_id"] for l in narration}
    audit = {"schema_version": AUDIT_SCHEMA, "speech_lines": len(rlines), "samples_verified": len(receipts),
             "dialogue_expected": len(dialogue), "narration_lines": len(narration),
             "dialogue_instruction_forwarded": sum(1 for l in dialogue if receipts[l["utterance_id"]]["instruction_forwarded"]),
             "narration_instructions": sum(1 for l in narration if receipts[l["utterance_id"]]["instruction"] is not None
                                           or receipts[l["utterance_id"]]["instruction_forwarded"]),
             "narration_fresh_neutral_synthesis": sum(1 for l in narration
                                                      if receipts[l["utterance_id"]]["audio_provenance"] == "fresh_neutral_review_synthesis"),
             "narration_reused_takes": 0,
             "narration_dispatched_this_run": sum(1 for uid in dispatched if uid in narration_ids),
             "new_requests_this_run": len(dispatched), "workers_started_this_run": len(workers)}
    if (audit["samples_verified"] != audit["speech_lines"]
            or audit["dialogue_instruction_forwarded"] != audit["dialogue_expected"] or audit["narration_instructions"]
            or audit["narration_fresh_neutral_synthesis"] != audit["narration_lines"] or len(workers) > 1):
        raise FreshChapterError(f"Forwarding audit failed: {audit}")
    audit["verdict"] = ("instruction forwarded for every coached dialogue line; narration newly synthesized "
                        "neutrally on the same clone route and never reused as a Take")
    return audit


def _entry_for(line, sample, expected, rate, start, pcm_len, timing, role):
    entry = {"utterance_id": line["utterance_id"], "kind": "speech", "index": line["index"],
             "role_id": line["role_id"], "role_name": line["role_name"], "narrator": line["narrator"],
             "mode": line["mode"], "scene_id": line["scene_id"], "source_anchor": line["source"],
             "source_project_id": line["project_id"], "provider_project_id": line["provider_project_id"],
             "cast_mode": role["mode"], "text": line["text"], "text_sha256": line["text_sha256"], "voice_id": line["voice"]["id"],
             "reference_audio_sha256": line["voice"]["reference_audio_sha256"],
             "prompt_sha256": line["voice"]["prompt_sha256"], "cloned_reference": _clone_binding(role),
             "audio_sha256": expected, "audio_file": None, "start_seconds": start, "end_seconds": start + pcm_len / rate,
             "boundary": timing, "provenance": sample["audio_provenance"], "native_instruction": sample["instruction"],
             "instruction_sha256": sample["instruction_sha256"], "request_sha256": sample["request_sha256"],
             "runtime_sha256": sample["runtime_sha256"],
             "receipt": {"file": f"receipts/{line['utterance_id']}.json", "raw": sample["raw_render_receipt"]},
             "coaching": sample["coaching"], "narration": sample["narration"]}
    return entry


def _verify_published_chapter(out, state, saved, receipts):
    """A completed chapter is verified, never re-assembled: bytes, durations, clone bindings and per-line audio."""
    p = pilot()
    if saved["frozen_sha256"] != state["frozen_sha256"]:
        raise FreshChapterError("Published chapter belongs to a different frozen chapter")
    for kind in ("master", "audio"):
        if p.sha((out / saved[kind + "_file"]).read_bytes()) != saved[kind + "_sha256"]:
            raise FreshChapterError(f"Published chapter {kind} changed")
    if (abs(_decoded_duration(out / saved["master_file"]) - saved["master_duration_seconds"]) > 0.001
            or abs(_decoded_duration(out / saved["audio_file"]) - saved["duration_seconds"]) > 0.01):
        raise FreshChapterError("Published chapter durations changed")
    roles = {l["role_id"] for l in state["lines"]}
    if saved["clone_binding"] != [_clone_binding(state["roles"][rid]) for rid in sorted(roles)]:
        raise FreshChapterError("Published chapter clone bindings differ from the frozen approved clones")
    speech = [row for row in saved["source_to_audio"] if row["kind"] == "speech"]
    if [row["utterance_id"] for row in speech] != [l["utterance_id"] for l in state["lines"]]:
        raise FreshChapterError("Published chapter map differs from the frozen speech lines")
    for row in speech:
        sample = receipts[row["utterance_id"]]
        if (row["audio_sha256"] != sample["audio_sha256"]
                or p.sha((out / row["audio_file"]).read_bytes()) != row["audio_sha256"]
                or row["receipt"]["raw"] != sample["raw_render_receipt"]):
            raise FreshChapterError(f"Published chapter audio or receipt for {row['utterance_id']} changed")
    return saved


def _assemble_chapter(cfg, state, rlines, receipts, pinned):
    """All speech in source order; measured silence-aware padding, canonical separators, no trimming."""
    import numpy as np
    import soundfile as sf
    p, out = pilot(), cfg.output_dir
    chapter = state["chapter"]
    number = chapter["chapter_number"]
    folder = out / "chapter"
    receipt_path = folder / f"chapter-{number:03}.json"
    if receipt_path.is_file():
        return _verify_published_chapter(out, state, _read_json(receipt_path), receipts)
    policy = boundaries.default_policy()
    breaks = chapter["scene_breaks"]
    index_of = {l["utterance_id"]: l["index"] for l in rlines}
    chunks, mapping, previous, previous_class, rate = [], [], None, None, None
    elapsed = padded = speech_seconds = existing_credit = 0.0
    for line in rlines:
        sample = receipts[line["utterance_id"]]
        path, expected = out / sample["audio_file"], sample["audio_sha256"]
        if p.sha(path.read_bytes()) != expected:
            raise FreshChapterError(f"{line['utterance_id']} audio changed before assembly")
        pcm, this_rate = sf.read(path, dtype="float32", always_2d=True)
        if rate is not None and rate != this_rate:
            raise FreshChapterError("Mixed PCM sample domains; no implicit resampling")
        rate = this_rate
        measure = boundaries.measure_take(path, expected, text=line["text"])
        row = {"utterance_id": line["utterance_id"], "text": line["text"], "role_id": line["role_id"],
               "source_revision_id": line["source"]["source_revision_id"],
               "start_offset": line["source"]["start_offset"], "end_offset": line["source"]["end_offset"]}
        timing = None
        if previous:
            left, left_measure = previous
            row["previous_tag"] = previous_class
            classification = boundaries.classify_boundary(left, row, role_types=chapter["role_types"],
                                                          aliases=chapter["role_aliases"])
            klass = classification["class"]
            target = policy["targets_ms"].get(klass, 350)
            between = [b["utterance_id"] for b in breaks if index_of[left["utterance_id"]] < b["index"] < line["index"]]
            if between:
                target = max(target, SEPARATOR_TARGET_MS)
            existing = (left_measure["trailing"]["silent_frames"] + measure["leading"]["silent_frames"]) * 1000 / rate
            frames = max(0, round((target - existing) * rate / 1000))
            chunks.append(np.zeros((frames, pcm.shape[1]), dtype="float32"))
            elapsed += frames / rate
            padded += frames / rate
            existing_credit += min(existing, target) / 1000
            timing = {"left": left["utterance_id"], "classification": classification, "target_ms": target,
                      "existing_silence_ms": existing, "added_frames": frames, "trimming": False,
                      "source_separators": between}
            previous_class = {"class": klass, "role_id": left["role_id"]}
        entry = _entry_for(line, sample, expected, rate, elapsed, len(pcm), timing, state["roles"][line["role_id"]])
        entry["audio_file"] = _relative(path, out)
        mapping.append(entry)
        chunks.append(pcm)
        elapsed += len(pcm) / rate
        speech_seconds += len(pcm) / rate
        previous = (row, measure)
    for item in chapter["rows"]:  # breaks and exclusions stay visible in the source-to-audio map
        if item["kind"] != "speech":
            mapping.append({"utterance_id": item["utterance_id"], "kind": item["kind"], "index": item["index"],
                            "source_text_sha256": item["text_sha256"], "audio_file": None,
                            "pause_target_ms": SEPARATOR_TARGET_MS if item["kind"] == "scene_break" else 0})
    mapping.sort(key=lambda r: r["index"])
    master, mp3 = folder / f"chapter-{number:03}.wav", folder / f"chapter-{number:03}.mp3"
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), rate, format="WAV", subtype="PCM_16")
    p.publish(master, buffer.getvalue())
    master_sha = p.sha(master.read_bytes())
    duration = _decoded_duration(master)
    if abs(duration - elapsed) > 0.001 or abs(duration - (speech_seconds + padded)) > 0.001:
        raise FreshChapterError("Assembled master duration differs from its source-to-audio accounting")
    report = qa.qa_take_audio(take_id="chapter_" + master_sha[:24], audio_path=master,
                              text="\n".join(l["text"] for l in rlines), expected_sha256=master_sha, language="English")
    if not report.passed or report.blocking_failures() or not report.executed_check_ids():
        raise FreshChapterError(f"Assembled chapter {number} fails blocking technical QA")
    if not mp3.exists():
        _mp3(master, mp3)
    delivered = _decoded_duration(mp3)
    if abs(delivered - duration) > 0.1:
        raise FreshChapterError("Decoded listening file differs from its master")
    loudness = _loudness(mp3)
    mp3_sha = p.sha(mp3.read_bytes())
    applied = _read_json(out / "applied.json")
    result = {
        "schema_version": CHAPTER_SCHEMA, "status": "ready_for_listening", **REVIEW_FLAGS, "narration": NARRATION,
        "cast_reuse": _cast_reuse(state["owner_review"], state["roles"]),
        "chapter": {"number": number, "id": chapter["chapter_id"], "title": chapter["chapter_title"],
                    "project_id": chapter["project_id"]},
        "epub": {"sha256": state["epub"]["sha256"], "member": chapter["epub_member"],
                 "chapter_text_sha256": chapter["epub_text_sha256"], "nonblank_characters": chapter["nonblank_characters"]},
        "source": {"project_sha256": state["sources"][0]["project_sha256"],
                   "segments_sha256": state["sources"][0]["segments_sha256"],
                   "derived_from": chapter["derived_from"], "utterance_ids": [r["utterance_id"] for r in chapter["rows"]]},
        "clone_binding": [_clone_binding(state["roles"][rid]) for rid in sorted({l["role_id"] for l in rlines})],
        "frozen_sha256": state["frozen_sha256"],
        "counts": {"speech": len(rlines), "fresh_neutral_narration": sum(l["narrator"] for l in rlines),
                   "fresh_directed_dialogue": sum(not l["narrator"] for l in rlines),
                   "scene_breaks": sum(r["kind"] == "scene_break" for r in chapter["rows"]),
                   "excluded": sum(r["kind"] == "excluded" for r in chapter["rows"])},
        "coaching": {"request_sha256": applied["request_sha256"], "response_sha256": applied["response_sha256"],
                     "bundle_sha256": applied["bundle_sha256"], "coached_plan_sha256": applied["coached_plan_sha256"],
                     "model_provenance": applied["model_provenance"], "performance_memory": applied["performance_memory"]},
        "runtime": {"runtime_sha256": pinned, "profile": p.DIRECTED_PROFILE, "policy": p.DIRECTION_POLICY},
        "boundary_policy_sha256": state["boundary_policy_sha256"],
        "accounting": {"speech_seconds": speech_seconds, "added_padding_seconds": padded,
                       "existing_silence_credited_seconds": existing_credit, "separator_target_ms": SEPARATOR_TARGET_MS,
                       "trimming": False, "master_seconds": duration},
        "added_padding_seconds": padded,
        "master_file": _relative(master, out), "master_sha256": master_sha, "master_duration_seconds": duration,
        "audio_file": _relative(mp3, out), "audio_sha256": mp3_sha, "duration_seconds": delivered,
        "integrated_loudness_lufs": loudness,
        "loudness_measurement": {"audio_sha256": mp3_sha, "integrated_lufs": loudness, "method": "ffmpeg_ebur128"},
        "technical_qa": "passed", "qa_report": report.model_dump(mode="json"), "source_to_audio": mapping}
    if chapter["fixture"]:
        result["fixture"] = _label()
    p.publish(receipt_path, p.encoded(result))
    return result


def generate_chapter(args, state, worker_class=None, render_fn=None):
    """Render every speech line serially and assemble the chapter (review only)."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    with _fail_closed():
        disk = load_state(out)
        if state.get("frozen_sha256") != disk["frozen_sha256"] or _frozen_sha(state) != disk["frozen_sha256"]:
            raise FreshChapterError("Supplied state differs from the frozen prepared chapter")
        if disk["prepare_holds"]:
            raise FreshChapterError("Preparation held; the chapter will not generate")
        directions, _ = _applied_directions(out, disk)
        _verify_live(cfg, disk)
        _reprove_source(cfg, disk)
        rlines = [{**line, "direction": directions[line["utterance_id"]]} for line in disk["lines"]]
        with _projects_root(out / "projects"):
            receipts, pinned, workers, dispatched = _render_speech(cfg, disk, rlines, worker_class, render_fn)
        audit = _audit(rlines, receipts, dispatched, workers)
        stable = {k: v for k, v in audit.items() if k not in ("new_requests_this_run", "workers_started_this_run",
                                                              "narration_dispatched_this_run")}
        stable["narration"] = NARRATION
        stable["cast_reuse"] = _cast_reuse(disk["owner_review"], disk["roles"])
        if disk["fixture"]:
            stable["fixture"] = _label()
        p.publish(out / "forwarding-audit.json", p.encoded({"frozen_sha256": disk["frozen_sha256"], **stable}))
        chapter = _assemble_chapter(cfg, disk, rlines, receipts, pinned)
        summary = {"chapter": chapter["chapter"]["number"], "project_id": chapter["chapter"]["project_id"],
                   "audio_file": chapter["audio_file"], "audio_sha256": chapter["audio_sha256"],
                   "master_sha256": chapter["master_sha256"], "duration_seconds": chapter["duration_seconds"],
                   "counts": chapter["counts"]}
        result = {"schema_version": RESULT_SCHEMA, "frozen_sha256": disk["frozen_sha256"], **REVIEW_FLAGS,
                  "narration": NARRATION, "cast_reuse": _cast_reuse(disk["owner_review"], disk["roles"]), "chapter": summary, "fixture": _label() if disk["fixture"] else None}
        p.publish(out / "result.json", p.encoded(result))
        disk.update(status="generated", holds=[])
        _write_state(out, disk)
        return {"status": "generated", "audit": audit, "chapter": summary,
                "result_sha256": p.sha((out / "result.json").read_bytes()), "human_review": "pending"}


# ---------------------------------------------------------------------------------------- CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epub", type=Path, required=True, help="the exact EPUB the derived chapter project was built from")
    parser.add_argument("--source-root", type=Path, required=True, help="read-only store holding the prepared derived project")
    parser.add_argument("--project", required=True, metavar="ID", help="the root-prepared derived chapter project id")
    parser.add_argument("--chapter", type=int, required=True, metavar="NUMBER", help="the numbered EPUB chapter")
    parser.add_argument("--output-dir", type=Path, required=True, help="private output outside the source store")
    parser.add_argument("--coaching-bundle", type=Path, help="captured whole-scene model coaching bundle (JSON)")
    parser.add_argument("--performance-memory", type=Path,
                        help="captured character performance memory bundle used to enrich the coaching request")
    parser.add_argument("--generate", action="store_true", help="explicit serial GPU synthesis (root-owned lane)")
    args = parser.parse_args(argv)
    try:
        _config(args)
        for label, path in (("EPUB", args.epub), ("Coaching bundle", args.coaching_bundle),
                            ("Performance memory", args.performance_memory)):
            if path is not None and not path.is_file():
                raise FreshChapterError(f"{label} not found: {path}")
    except FreshChapterError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        state = prepare_chapter(args)
        if args.coaching_bundle and state["status"] in ("awaiting_coaching", "held", "ready") and not state["prepare_holds"]:
            state = apply_bundle(args, state)
        summary = {"status": state["status"], "chapter": state["sources"][0]["chapter_number"],
                   "project_id": state["sources"][0]["project_id"], "speech_lines": len(state["lines"]),
                   "dialogue_lines": sum(not l["narrator"] for l in state["lines"]),
                   "narration_lines": sum(l["narrator"] for l in state["lines"]), "coaching": state.get("coaching"),
                   "performance_memory": bool(state["memory"]),
                   "cast_modes": {m: sorted(r for r, v in state["roles"].items() if v["mode"] == m)
                                  for m in sorted({v["mode"] for v in state["roles"].values()})}, "holds": state["holds"], "output": str(args.output_dir),
                   "human_review": "pending", "production_ready": False}
        if args.generate:
            if state["status"] not in ("ready", "generated"):
                print(json.dumps(summary, ensure_ascii=True), flush=True)
                return 2
            summary["generation"] = generate_chapter(args, state)
            summary["status"] = summary["generation"]["status"]
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return 2 if state["status"] == "held" else 0
    except (FreshChapterError, RuntimeError) as exc:  # includes the render owner's refusal to resend
        print(json.dumps({"status": "held", "error": str(exc)}, ensure_ascii=True), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
