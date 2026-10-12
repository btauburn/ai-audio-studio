"""Resumable full-chapter scene-coached (V2) review renders over existing cloned chapter projects.

Three CPU/GPU-separated stages share one private output directory and one frozen ``state.json``:

``prepare_batch``  verifies the exact EPUB, canonical chapter, clone cast and existing neutral Takes,
                   freezes every speech line, copies verified neutral narrator audio, builds the base
                   scene-dialogue plans and writes one whole-scene coaching *request* per chapter.
                   Nothing is coached or synthesized; the status is ``awaiting_coaching``.
``apply_bundles``  validates captured model coaching bundles (``coaching-{project_id}.json``) against
                   the frozen requests, live source/cast and the pilot's direction guards, then
                   publishes immutable coached plans.  No instruction is ever manufactured here.
``generate_batch`` renders only the new dialogue through one serial directed worker, verifies each
                   receipt, and assembles every speech line (reused neutral narration included) in
                   source order under the measured semantic boundary policy.

Everything stays review only: no Take is created or selected, no approval is recorded, the production
default is untouched and chapter receipts keep ``human_review: pending`` / ``production_ready: false``.
Existing helpers own the engines: ``voice_lab_chapter_export`` (source and Take proof),
``scene_dialogue_plan`` / ``scene_coaching`` (plans and coaching), ``scripts/scene_dialogue_pilot`` (clone
files, direction guards, receipts) and ``scripts/directors_vertical_slice`` (worker and render).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from . import epub_chapter_selection, performance_boundaries as boundaries, qa, scene_coaching
from . import scene_dialogue_plan as plans, takes
from . import voice_lab_chapter_export as exporter
from .runtime_binding import take_runtime_profiles
from .voice_lab_render import _decoded_duration, _loudness, _mp3

REPO = Path(__file__).resolve().parents[3]

STATE_SCHEMA = "coached-chapter-batch-state-v1"
APPLIED_SCHEMA = "coached-chapter-applied-v1"
CHAPTER_SCHEMA = "coached-chapter-review-v1"
RESULT_SCHEMA = "coached-chapter-batch-result-v1"
NEUTRAL_PROFILE = "faster-base-032"
SEPARATOR_TARGET_MS = 1250
FIXTURE_LABEL = "Synthetic CPU test fixture; excluded from production readiness"
FROZEN_KEYS = ("epub", "sources", "chapters", "lines", "requests", "base_plans", "isolated",
               "boundary_policy_sha256", "prepare_holds")
_ID = re.compile(r"[A-Za-z0-9_-]{1,120}")
REQUEST_INSTRUCTION = (
    "Read each complete scene below, including the neutral narration, and propose one temporary native "
    "delivery instruction for every id in that scene's selected_dialogue_ids. Narration rows are context "
    "only: never propose a direction for them. Never rewrite, add or omit spoken words. Keep your reading "
    "of the scene (objective, emotion, intensity, cadence, emphasis, ending, confidence) separate from "
    "the short native_instruction that conditions the existing voice. Return only JSON that matches "
    "response_contract.")
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


class BatchError(RuntimeError):
    """A fail-closed refusal. Preserved output is reported, never repaired silently."""


# ---------------------------------------------------------------------------------------- plumbing


_PILOT = None


def pilot():
    """The scene pilot helper module (a script, loaded once by path)."""
    global _PILOT
    if _PILOT is None:
        path = REPO / "scripts" / "scene_dialogue_pilot.py"
        spec = importlib.util.spec_from_file_location("_coached_batch_scene_dialogue_pilot", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PILOT = module
    return _PILOT


@contextlib.contextmanager
def _fail_closed():
    try:
        yield
    except BatchError:
        raise
    except (ValueError, pilot().PilotError, epub_chapter_selection.EpubChapterSelectionError) as exc:
        raise BatchError(str(exc)) from exc


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


def _config(args):
    """Normalise and validate the batch arguments; every later stage derives from this."""
    projects = [str(p) for p in args.projects]
    chapters = list(args.chapters)
    if not projects or len(projects) != len(chapters):
        raise BatchError("--projects and --chapters must be non-empty and the same length")
    if any(type(c) is not int or c < 1 for c in chapters) or chapters != sorted(set(chapters)):
        raise BatchError("--chapters must be distinct positive integers in ascending reading order")
    if len(set(projects)) != len(projects) or any(not _ID.fullmatch(p) for p in projects):
        raise BatchError("--projects must be distinct valid project ids")
    source_root, output = Path(args.source_root).resolve(), Path(args.output_dir).resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise BatchError("Output must be a separate directory outside the canonical project store")
    coaching = getattr(args, "coaching_dir", None)
    epub = getattr(args, "epub", None)
    return SimpleNamespace(epub=Path(epub).resolve() if epub else None, source_root=source_root, projects=projects,
                           chapters=chapters, output_dir=output,
                           coaching_dir=Path(coaching).resolve() if coaching else None)


def _frozen_sha(state):
    return pilot().value_sha({key: state[key] for key in FROZEN_KEYS})


def _write_state(output, state):
    pilot().replace_atomic(Path(output) / "state.json", pilot().encoded(state))


def _relative(path, root):
    return Path(path).relative_to(root).as_posix()


def load_state(output_dir):
    """Read ``state.json`` and prove the frozen section and its published inputs are unchanged."""
    output = Path(output_dir)
    path = output / "state.json"
    if not path.is_file():
        raise BatchError("No prepared batch state; run prepare first")
    state = json.loads(path.read_bytes())
    if state.get("schema_version") != STATE_SCHEMA:
        raise BatchError("Unknown batch state schema")
    if _frozen_sha(state) != state.get("frozen_sha256"):
        raise BatchError("Frozen batch lines, plans or sources changed since preparation")
    for kind in ("requests", "base_plans"):
        for pid, row in state[kind].items():
            if pilot().sha((output / row["file"]).read_bytes()) != row["sha256"]:
                raise BatchError(f"Published {kind[:-1]} for {pid} changed since preparation")
    return state


def _require_same_batch(state, cfg):
    mine = (cfg.projects, cfg.chapters, pilot().sha(cfg.epub.read_bytes()))
    theirs = ([s["project_id"] for s in state["sources"]], [s["chapter_number"] for s in state["sources"]],
              state["epub"]["sha256"])
    if mine != theirs:
        raise BatchError("This output directory was prepared for a different EPUB, chapter or project batch")


def _build_plan(project, segments, *, cache_root, selected_ids):
    return plans.build_scene_dialogue_plan(project, segments, cache_root=cache_root, selected_ids=selected_ids)


def _verify_plan(plan, project, segments, label):
    result = plans.verify_scene_dialogue_plan(plan, project, segments)
    if not result["current"]:
        raise BatchError(f"{label} is no longer current for the live source/cast: "
                         + json.dumps(result["problems"], ensure_ascii=False)[:600])


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
        raise BatchError(f"Canonical chapter does not cover every EPUB prose character; first difference at "
                         f"nonblank character {at}: epub {expected[max(0, at - 20):at + 20]!r} vs "
                         f"source {actual[max(0, at - 20):at + 20]!r}")
    return len(expected)


def _prepare_chapter(cfg, book, chapter, project_id, plan_builder):
    p, output = pilot(), cfg.output_dir
    source_folder = cfg.source_root / project_id
    before = p.read_source(cfg.source_root, project_id)
    from dubbing_service.app.storage.projects import load_project
    probe = load_project(project_id)
    canonical = next((c for c in probe.chapters if c.title == chapter["title"]), None)
    if canonical is None or epub_chapter_selection.chapter_number_from_title(canonical.title) != chapter["chapter_number"]:
        raise BatchError(f"{project_id} does not contain the exact numbered chapter {chapter['chapter_number']}")
    project, segments, source = exporter._source_snapshot(project_id, canonical.id)
    if not any(r.sha256 == book["source_sha256"] for r in project.source_revisions):
        raise BatchError(f"{project_id} is not bound to this exact EPUB (source revision hash differs)")
    nonblank = _verify_prose(chapter["text"], segments)

    roles = {r.id: r for r in project.roles}
    table = p.role_table(before["project"])
    speech_ids = set(source["speech_utterance_ids"])
    speech = [s for s in segments if s.id in speech_ids]
    for role_id in sorted({s.role_id for s in speech}):
        entry = table.get(role_id)
        if entry is None or entry["identity"] is None:
            raise BatchError(f"{project_id}:{role_id} has no locked eligible cloned identity "
                             f"({', '.join(entry['clone_holds']) if entry else 'unknown role'})")
        if roles[role_id].tts_backend != "qwen3tts":
            raise BatchError(f"{project_id}:{role_id} is not assigned to the Qwen Base route")
        p.verify_identity_files(source_folder, entry["identity"])
    paths, selections, evidence = exporter._take_snapshot(project, segments, source)
    for segment in speech:
        if take_runtime_profiles(takes.selected_take(segment)) != {NEUTRAL_PROFILE}:
            raise BatchError(f"{project_id}:{segment.id} neutral Take does not prove the {NEUTRAL_PROFILE} route")

    registry = {r.id: r.sha256 for r in project.source_revisions}
    lines = []
    for segment, path, selection in zip(speech, paths, selections):
        entry = table[segment.role_id]
        anchor = segment.source_anchor
        proof = evidence["proof"][segment.id]
        narrator = entry["narrator"]
        copy = None
        if narrator:  # existing immutable audio, copied byte for byte after hash verification
            target = output / "neutral" / project_id / f"{segment.id}.wav"
            data = Path(path).read_bytes()
            if p.sha(data) != selection["audio_sha256"]:
                raise BatchError(f"{project_id}:{segment.id} neutral Take bytes differ from their hash")
            p.publish(target, data)
            copy = {"file": _relative(target, output), "sha256": p.sha(data)}
        lines.append({
            "utterance_id": segment.id, "project_id": project_id, "chapter_number": chapter["chapter_number"],
            "chapter_id": canonical.id, "index": segment.index, "role_id": segment.role_id,
            "role_name": entry["name"], "narrator": narrator, "scene_id": None,
            "mode": "neutral_reuse" if narrator else "directed_dialogue",
            "text": segment.text, "text_sha256": p.text_sha(segment.text), "chars": len(segment.text),
            "source": {"source_revision_id": anchor.source_revision_id,
                       "source_revision_sha256": registry[anchor.source_revision_id],
                       "start_offset": anchor.start_offset, "end_offset": anchor.end_offset,
                       "occurrence_index": anchor.occurrence_index},
            "voice": entry["identity"],
            "neutral_take": {"take_id": selection["take_id"], "audio_sha256": selection["audio_sha256"],
                             "input_fingerprint": selection["input_fingerprint"], "voice_id": selection["voice_id"],
                             "voice_version": selection["voice_version"], "duration_seconds": proof["duration_seconds"],
                             "runtime_profiles": sorted(take_runtime_profiles(takes.selected_take(segment))),
                             "source_audio_file": _relative(path, source_folder.resolve())},
            "neutral_copy": copy})
    dialogue = [line for line in lines if not line["narrator"]]
    if not dialogue:
        raise BatchError(f"{project_id} has no non-narrator speech to coach")

    cache = output / "scene-cache"
    cache.mkdir(parents=True, exist_ok=True)
    base_file = output / "plans" / f"base-{project_id}.json"
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
            raise BatchError(f"Base plan for {project_id} omits speech line {line['utterance_id']}")
        line["scene_id"] = direction["scene_id"]
    holds = [h for h in plan.get("holds", []) if h.get("code") != "no_scene_evidence"]
    if plan.get("source", {}).get("proven") is not True:
        holds.append({"code": "source_unproven", "detail": "base plan has no proven source revision"})
    request = None if holds else _build_request(project_id, plan, segments, table, lines)
    request_info = None
    if request is not None:
        request_file = output / "requests" / f"request-{project_id}.json"
        p.publish(request_file, p.encoded(request))
        request_info = {"file": _relative(request_file, output), "sha256": p.sha(request_file.read_bytes()),
                        "request_sha256": plans._sha(request), "scenes": len(request["scenes"]),
                        "dialogue_ids": [line["utterance_id"] for line in dialogue]}

    voices = {line["voice"]["id"] for line in dialogue}
    copied = p.prepare_isolated_project(source_folder, output / "projects" / project_id, voices)
    for line in dialogue:
        p.verify_identity_files(output / "projects" / project_id, line["voice"])
    after = p.read_source(cfg.source_root, project_id)
    if (after["project_sha256"], after["segments_sha256"]) != (before["project_sha256"], before["segments_sha256"]):
        raise BatchError(f"Canonical source {project_id} changed during preparation")

    breaks = {b["utterance_id"] for b in source["breaks"]}
    excluded = {row["utterance_id"] for row in source.get("exclusions", [])}
    rows = [{"utterance_id": s.id, "index": s.index, "text_sha256": p.text_sha(s.text), "chars": len(s.text),
             "kind": "scene_break" if s.id in breaks else "excluded" if s.id in excluded else "speech"}
            for s in segments]
    meta = {
        "project_id": project_id, "chapter_number": chapter["chapter_number"], "chapter_id": canonical.id,
        "chapter_title": canonical.title, "epub_member": chapter["member"],
        "epub_text_sha256": chapter["text_sha256"], "nonblank_characters": nonblank,
        "project_sha256": before["project_sha256"], "segments_sha256": before["segments_sha256"],
        "take_state_sha256": evidence["state_sha256"], "experimental_runtime": evidence["experimental_runtime"]}
    chapter_row = {
        **meta, "rows": rows, "scene_breaks": source["breaks"], "exclusions": source.get("exclusions", []),
        "role_types": {rid: getattr(r.role_type, "value", r.role_type) for rid, r in roles.items()},
        "role_aliases": dict(project.character_cast.role_aliases),
        "fixture": project.metadata.get("coached_batch_fixture") is True
        or project.metadata.get("scene_pilot_fixture") is True}
    base_info = {"file": _relative(base_file, output), "sha256": p.sha(plan_bytes),
                 "binding_sha256": plan.get("binding_sha256"), "scenes": len(plan.get("scenes", []))}
    isolated = {"files": len(copied), "tree_sha256": p.value_sha(copied)}
    source_row = {k: meta[k] for k in ("project_id", "chapter_number", "chapter_id", "chapter_title",
                                       "project_sha256", "segments_sha256", "take_state_sha256")}
    return {"source": source_row, "chapter": chapter_row, "lines": lines, "request": request_info,
            "base": base_info, "isolated": isolated, "holds": [{**h, "project_id": project_id} for h in holds]}


def _build_request(project_id, plan, segments, table, lines):
    """One whole-scene request: all context incl. neutral narration; directions only for dialogue ids."""
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
                raise BatchError(f"Scene {scene['scene_id']} references {uid}, which is not live source")
            role = table.get(segment.role_id)
            context.append({"utterance_id": uid, "text": segment.text, "role_id": segment.role_id,
                            "narrator": bool(role and role["narrator"]),
                            "role_name": role["name"] if role else None})
        scenes.append({"scene_id": scene["scene_id"], "project_id": project_id, "context": context,
                       "selected_dialogue_ids": selected})
        covered += selected
    if sorted(covered) != sorted(dialogue) or len(set(covered)) != len(covered):
        raise BatchError(f"{project_id}: scene plan does not cover each dialogue line exactly once")
    return {"instruction": REQUEST_INSTRUCTION, "constraints": list(REQUEST_CONSTRAINTS),
            "response_contract": {
                "schema_version": scene_coaching.RESPONSE_SCHEMA,
                "scenes": "[{scene_id, directions: [one object per selected_dialogue_ids entry]}]",
                "direction_fields": sorted(scene_coaching.FIELDS),
                "intensity": ["moderate", "strong"], "confidence": ["low", "moderate", "high"]},
            "scenes": scenes}


def prepare_batch(args, *, plan_builder=None):
    """Freeze source, cast, neutral Takes and requests. CPU only; status ``awaiting_coaching``."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    with _fail_closed():
        if (out / "state.json").is_file():
            state = load_state(out)
            _require_same_batch(state, cfg)
            return state
        data = cfg.epub.read_bytes()
        book = epub_chapter_selection.select_epub_chapters(data, cfg.epub.name, chapter_numbers=cfg.chapters)
        out.mkdir(parents=True, exist_ok=True)
        built = []
        with _projects_root(cfg.source_root):
            for chapter, project_id in zip(book["chapters"], cfg.projects):
                built.append(_prepare_chapter(cfg, book, chapter, project_id, plan_builder or _build_plan))
        holds = [h for item in built for h in item["holds"]]
        state = {
            "schema_version": STATE_SCHEMA, "batch": "coached-chapters-v2",
            "review_only": True, "human_review": "pending", "production_ready": False,
            "default_runtime_changed": False, "approvals_created": False, "takes_selected": False,
            "fixture": any(item["chapter"]["fixture"] for item in built),
            "runtime": {"directed_profile": p.DIRECTED_PROFILE, "policy": p.DIRECTION_POLICY,
                        "request": "experimental review only; narrator audio is reused, never synthesized"},
            "epub": {"name": cfg.epub.name, "sha256": book["source_sha256"], "title": book["title"],
                     "author": book["author"], "selection": book["selection"]},
            "sources": [item["source"] for item in built],
            "chapters": [item["chapter"] for item in built],
            "lines": [line for item in built for line in item["lines"]],
            "requests": {item["source"]["project_id"]: item["request"] for item in built if item["request"]},
            "base_plans": {item["source"]["project_id"]: item["base"] for item in built},
            "isolated": {item["source"]["project_id"]: item["isolated"] for item in built},
            "boundary_policy_sha256": p.boundary_policy_sha(), "prepare_holds": holds}
        state["frozen_sha256"] = _frozen_sha(state)
        state.update(status="held" if holds else "awaiting_coaching", holds=holds, coaching={})
        _write_state(out, state)
        return state


# ---------------------------------------------------------------------------------------- coaching


def _lines_for(state, project_id):
    return [line for line in state["lines"] if line["project_id"] == project_id]


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def _check_coached(state, project_id, request, bundle, bundle_raw, coached, coached_bytes):
    """Validate one coached plan against the frozen lines/request and the captured bundle."""
    p = pilot()
    lines = _lines_for(state, project_id)
    if bundle.get("request") != request or bundle.get("request_sha256") != plans._sha(request):
        raise BatchError(f"{project_id}: the bundle was not captured for the frozen request")
    scene_coaching.validate_bundle(bundle)
    coaching = coached.get("coaching") or {}
    if (coaching.get("request_sha256") != bundle["request_sha256"]
            or coaching.get("response_sha256") != bundle["response_sha256"]):
        raise BatchError(f"{project_id}: coached plan is not bound to the supplied bundle")
    if (coached.get("source", {}).get("proven") is not True or coached.get("approved") is True
            or coached.get("recommendation_only") is not True or coached.get("human_review") != "pending"
            or coached.get("holds")):
        raise BatchError(f"{project_id}: coached plan carries a source, approval or safety hold")
    directions, holds = p.apply_plan(coached, p.sha(coached_bytes), lines)
    if holds:
        raise BatchError(f"{project_id}: direction guards refused the plan: " + json.dumps(holds, ensure_ascii=False)[:600])
    scene_of = {uid: scene["scene_id"] for scene in request["scenes"] for uid in scene["selected_dialogue_ids"]}
    proposed = {d["utterance_id"]: d for scene in bundle["response"]["scenes"] for d in scene["directions"]}
    for line in lines:
        direction = directions[line["utterance_id"]]
        instruction = direction["native_instruction"]
        if line["narrator"]:
            if instruction is not None:
                raise BatchError(f"{project_id}:{line['utterance_id']} narrator carries an instruction")
            continue
        if (not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 220
                or direction["basis"] != "model" or not direction.get("coaching")
                or direction.get("coaching_scene_id") != scene_of.get(line["utterance_id"])
                or instruction != proposed[line["utterance_id"]]["native_instruction"]):
            raise BatchError(f"{project_id}:{line['utterance_id']} lacks a complete, bundle-bound model direction")
        if direction["voice_revision"] != line["neutral_take"]["voice_version"]:
            raise BatchError(f"{project_id}:{line['utterance_id']} direction is bound to a different voice revision")
    if set(proposed) != set(scene_of):
        raise BatchError(f"{project_id}: bundle directions do not match the requested dialogue set")
    return directions


def _apply_project(cfg, state, project_id, bundle_path):
    p, out = pilot(), cfg.output_dir
    request_bytes = (out / state["requests"][project_id]["file"]).read_bytes()
    request = json.loads(request_bytes)
    raw = Path(bundle_path).read_bytes()
    bundle = json.loads(raw)
    published = out / "coaching" / f"bundle-{project_id}.json"
    if published.exists() and published.read_bytes() != raw:
        raise BatchError(f"{project_id}: a different coaching bundle was already applied; output is immutable")
    base_row = state["base_plans"][project_id]
    base = _read_json(out / base_row["file"])
    chapter = next(c for c in state["chapters"] if c["project_id"] == project_id)
    with _projects_root(cfg.source_root):
        project, segments, source = exporter._source_snapshot(project_id, chapter["chapter_id"])
        if source["speech_utterance_ids"] != [l["utterance_id"] for l in _lines_for(state, project_id)]:
            raise BatchError(f"{project_id}: speech coverage changed since preparation")
        _verify_plan(base, project, segments, f"Base plan for {project_id}")
        scene_coaching.validate_bundle(bundle)
        try:  # no cache root: the plan bytes must be identical on every safe retry
            coached = scene_coaching.apply_coaching(base, bundle, project, segments, cache_root=None)
        except scene_coaching.CoachingError as exc:
            raise BatchError(f"{project_id}: coaching refused: {exc}") from exc
        _verify_plan(coached, project, segments, f"Coached plan for {project_id}")
    coached_bytes = p.encoded(coached)
    directions = _check_coached(state, project_id, request, bundle, raw, coached, coached_bytes)
    voices = {l["voice"]["id"]: l["voice"] for l in _lines_for(state, project_id) if not l["narrator"]}
    for voice in voices.values():  # current canonical clone files and the isolated worker copy
        p.verify_identity_files(cfg.source_root / project_id, voice)
        p.verify_identity_files(out / "projects" / project_id, voice)
    receipt = {"schema_version": APPLIED_SCHEMA, "project_id": project_id, "bundle_sha256": p.sha(raw),
               "request_sha256": bundle["request_sha256"], "response_sha256": bundle["response_sha256"],
               "model_provenance": bundle["provenance"], "base_plan_sha256": base_row["sha256"],
               "coached_plan_sha256": p.sha(coached_bytes),
               "directions_sha256": p.value_sha({uid: d["native_instruction"] for uid, d in directions.items()}),
               "dialogue_directed": sum(1 for l in _lines_for(state, project_id) if not l["narrator"]),
               "recommendation_only": True, "human_review": "pending", "approved": False}
    if state["fixture"]:
        receipt["fixture"] = {"is_fixture": True, "label": FIXTURE_LABEL}
    p.publish(published, raw)
    p.publish(out / "plans" / f"coached-{project_id}.json", coached_bytes)
    p.publish(out / "applied" / f"{project_id}.json", p.encoded(receipt))
    return receipt


def apply_bundles(args, state):
    """Validate and publish captured coaching bundles; ``ready`` only when every chapter is covered."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    with _fail_closed():
        supplied, state = state, load_state(out)
        if _frozen_sha(supplied) != state["frozen_sha256"]:
            raise BatchError("Supplied state differs from the frozen prepared batch")
        if state["prepare_holds"]:
            raise BatchError("Preparation held and cannot be coached: "
                             + json.dumps(state["prepare_holds"], ensure_ascii=False)[:600])
        if cfg.coaching_dir is None:
            return state
        bundles = {pid: cfg.coaching_dir / f"coaching-{pid}.json" for pid in state["requests"]}
        present = {pid: path for pid, path in bundles.items() if path.is_file()}
        if present:
            p.verify_prepared_sources(SimpleNamespace(source_root=cfg.source_root, output_dir=out), state)
        ledger, holds = {}, []
        for pid in state["requests"]:
            if pid not in present:
                ledger[pid] = {"status": "awaiting_bundle", "expected_file": f"coaching-{pid}.json"}
                continue
            try:
                receipt = _apply_project(cfg, state, pid, present[pid])
            except (BatchError, ValueError, p.PilotError) as exc:
                holds.append({"code": "coaching_refused", "project_id": pid, "detail": str(exc)})
                ledger[pid] = {"status": "refused", "detail": str(exc)}
                continue
            ledger[pid] = {"status": "applied", "applied_sha256": p.value_sha(receipt),
                           "coached_plan_sha256": receipt["coached_plan_sha256"]}
        status = ("generated" if state.get("status") == "generated" else "held" if holds else
                  "ready" if all(v["status"] == "applied" for v in ledger.values()) else "awaiting_coaching")
        state.update(coaching=ledger, holds=holds, status=status)
        _write_state(out, state)
        return state


def _applied_directions(out, state):
    """Re-derive every project's directions from the immutable published files, never from state.json."""
    p = pilot()
    directions = {}
    for pid, info in state["requests"].items():
        receipt_path = out / "applied" / f"{pid}.json"
        if not receipt_path.is_file():
            raise BatchError(f"{pid}: no coaching has been applied; the batch is not ready")
        receipt = _read_json(receipt_path)
        raw = (out / "coaching" / f"bundle-{pid}.json").read_bytes()
        coached_bytes = (out / "plans" / f"coached-{pid}.json").read_bytes()
        if receipt["bundle_sha256"] != p.sha(raw) or receipt["coached_plan_sha256"] != p.sha(coached_bytes):
            raise BatchError(f"{pid}: published coaching bundle or coached plan changed after it was applied")
        request = _read_json(out / info["file"])
        found = _check_coached(state, pid, request, json.loads(raw), raw, json.loads(coached_bytes), coached_bytes)
        if p.value_sha({uid: d["native_instruction"] for uid, d in found.items()}) != receipt["directions_sha256"]:
            raise BatchError(f"{pid}: directions differ from the applied receipt")
        directions.update({(pid, uid): d for uid, d in found.items()})
    return directions


# ---------------------------------------------------------------------------------------- generation


def _runtime_lines(state, directions):
    """Frozen lines plus their verified direction; the shape the pilot helpers expect."""
    return [{**line, "direction": directions[(line["project_id"], line["utterance_id"])]}
            for line in state["lines"]]


def _sample_path(out, line):
    return out / "samples" / line["project_id"] / f"{line['utterance_id']}.wav"


def _pending_files(audio):
    if not audio.parent.is_dir():
        return []
    return sorted(f for f in audio.parent.iterdir()
                  if f.name.startswith(audio.stem + ".pending") and f.name.endswith(".json"))


def _verify_neutral_and_clones(cfg, state):
    p, out = pilot(), cfg.output_dir
    checked = set()
    for line in state["lines"]:
        if line["narrator"]:
            if p.sha((out / line["neutral_copy"]["file"]).read_bytes()) != line["neutral_copy"]["sha256"]:
                raise BatchError(f"{line['project_id']}:{line['utterance_id']} private neutral copy changed")
            original = cfg.source_root / line["project_id"] / line["neutral_take"]["source_audio_file"]
            if not original.is_file() or p.sha(original.read_bytes()) != line["neutral_take"]["audio_sha256"]:
                raise BatchError(f"{line['project_id']}:{line['utterance_id']} canonical neutral Take changed")
        elif (line["project_id"], line["voice"]["id"]) not in checked:
            checked.add((line["project_id"], line["voice"]["id"]))
            p.verify_identity_files(out / "projects" / line["project_id"], line["voice"])
            p.verify_identity_files(cfg.source_root / line["project_id"], line["voice"])


def _record_sample(out, line, render_receipt, pinned, fixture=False):
    p = pilot()
    audio = _sample_path(out, line)
    verified = p.verify_sample("directed", line, render_receipt, pinned)
    if p.sha(audio.read_bytes()) != render_receipt["audio_sha256"]:
        raise BatchError(f"{line['utterance_id']} audio bytes differ from the render receipt")
    worker = render_receipt.get("worker_receipt") or {}
    if worker.get("identity_id") not in (None, line["voice"]["id"]) \
            or worker.get("prompt_sha256") not in (None, line["voice"]["prompt_sha256"]):
        raise BatchError(f"{line['utterance_id']} worker used a different voice identity than the frozen clone")
    report = qa.qa_take_audio(take_id="sample_" + render_receipt["audio_sha256"][:24], audio_path=audio,
                              text=line["text"], expected_sha256=render_receipt["audio_sha256"], language="English")
    if not report.passed or report.blocking_failures() or not report.executed_check_ids():
        raise BatchError(f"{line['project_id']}:{line['utterance_id']} new sample fails blocking technical QA")
    raw = audio.with_suffix(".receipt.json")
    direction = line["direction"]
    record = {**p.sample_receipt("directed", line, render_receipt, _relative(audio, out), verified),
              "mode": "experimental_directed_dialogue",
              "raw_render_receipt": {"file": _relative(raw, out), "sha256": p.sha(raw.read_bytes())},
              "technical_qa": {"passed": True, "executed_checks": sorted(report.executed_check_ids())},
              "coaching": {"scene_id": direction.get("coaching_scene_id"), "reading": direction.get("coaching"),
                           "basis": direction["basis"], "plan_sha256": direction["plan_sha256"]},
              "neutral_take_reference": line["neutral_take"], "neutral_take_replaced": False}
    if fixture:
        record["fixture"] = {"is_fixture": True, "label": FIXTURE_LABEL}
    p.publish(out / "receipts" / line["project_id"] / f"{line['utterance_id']}.json", p.encoded(record))
    return record


def _render_new_dialogue(cfg, state, rlines, worker_class, render_fn):
    """Serial, resumable synthesis of only the missing dialogue. At most one worker for the whole batch."""
    p, out = pilot(), cfg.output_dir
    dialogue = [line for line in rlines if not line["narrator"]]
    for line in dialogue:
        p.resolve_approval_refusal(out, "directed", line, _sample_path(out, line))
    pending = [l for l in dialogue if not _sample_path(out, l).with_suffix(".receipt.json").exists()]
    for line in pending:  # an unresolved dispatch is ambiguous: hold before any worker exists
        if _pending_files(_sample_path(out, line)):
            raise BatchError(f"{line['project_id']}:{line['utterance_id']} has an ambiguous pending dispatch "
                             "record; inspect it before any GPU call (no resend)")
    pin_path = out / "runtime-pin.json"
    pinned = _read_json(pin_path)["runtime_sha256"] if pin_path.exists() else None
    todo = {(l["project_id"], l["utterance_id"]) for l in pending}
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
        for line in dialogue:
            audio = _sample_path(out, line)
            instruction = line["direction"]["native_instruction"]
            if line["narrator"] or not instruction:
                raise BatchError(f"{line['utterance_id']} may not be synthesized without a model direction")
            key = (line["project_id"], line["utterance_id"])
            render_receipt = render_fn(worker, line["project_id"], line["voice"], line, audio, instruction)
            if key in todo:
                dispatched.append(key)
            if pinned is None:
                raise BatchError("No runtime pin exists to verify the completed samples against")
            receipts[key] = _record_sample(out, line, render_receipt, pinned, fixture=state["fixture"])
    finally:
        for item in reversed(workers):
            item.__exit__(None, None, None)
    return receipts, pinned, workers, dispatched


def _audit(rlines, receipts, dispatched, workers, out):
    p = pilot()
    narrators = [l for l in rlines if l["narrator"]]
    dialogue = [l for l in rlines if not l["narrator"]]
    audit = {"speech_lines": len(rlines), "dialogue_expected": len(dialogue), "neutral_narration_reused": len(narrators),
             "directed_forwarded": sum(1 for l in dialogue
                                       if receipts[(l["project_id"], l["utterance_id"])]["instruction_forwarded"]),
             "narrator_instructions": sum(1 for l in narrators if l["direction"]["native_instruction"]),
             "narrator_dispatched": sum(1 for pid, uid in dispatched
                                        if any(n["project_id"] == pid and n["utterance_id"] == uid for n in narrators)),
             "narrator_samples_on_disk": sum(1 for l in narrators if _sample_path(out, l).exists()
                                             or _sample_path(out, l).with_suffix(".receipt.json").exists()),
             "new_requests_this_run": len(dispatched), "workers_started_this_run": len(workers)}
    if (audit["directed_forwarded"] != audit["dialogue_expected"] or audit["narrator_instructions"]
            or audit["narrator_dispatched"] or audit["narrator_samples_on_disk"] or len(workers) > 1):
        raise BatchError(f"Forwarding audit failed: {audit}")
    audit["verdict"] = "instruction forwarded for every coached dialogue line; narration neutral and never synthesized"
    return audit


def _assemble_chapter(cfg, state, chapter, rlines, receipts, pinned):
    """All speech in source order; measured silence-aware padding, canonical separators, no trimming."""
    import numpy as np
    import soundfile as sf
    p, out = pilot(), cfg.output_dir
    number = chapter["chapter_number"]
    folder = out / "chapters" / chapter["project_id"]
    receipt_path = folder / f"chapter-{number:03}.json"
    if receipt_path.is_file():  # completed chapter: verify the published bytes, never re-assemble
        saved = _read_json(receipt_path)
        if saved["frozen_sha256"] != state["frozen_sha256"]:
            raise BatchError(f"Published chapter {number} belongs to a different frozen batch")
        for kind in ("master", "audio"):
            if p.sha((out / saved[kind + "_file"]).read_bytes()) != saved[kind + "_sha256"]:
                raise BatchError(f"Published chapter {number} {kind} changed")
        return saved
    mine = [l for l in rlines if l["project_id"] == chapter["project_id"]]
    policy = boundaries.default_policy()
    breaks = chapter["scene_breaks"]
    index_of = {l["utterance_id"]: l["index"] for l in mine}
    chunks, mapping, previous, previous_class, rate, elapsed, padded = [], [], None, None, None, 0.0, 0.0
    for line in mine:
        if line["narrator"]:
            path, expected = out / line["neutral_copy"]["file"], line["neutral_copy"]["sha256"]
        else:
            record = receipts[(line["project_id"], line["utterance_id"])]
            path, expected = out / record["audio_file"], record["audio_sha256"]
        if p.sha(path.read_bytes()) != expected:
            raise BatchError(f"{line['utterance_id']} audio changed before assembly")
        pcm, this_rate = sf.read(path, dtype="float32", always_2d=True)
        if rate is not None and rate != this_rate:
            raise BatchError("Mixed PCM sample domains; no implicit resampling")
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
            timing = {"left": left["utterance_id"], "classification": classification, "target_ms": target,
                      "existing_silence_ms": existing, "added_frames": frames, "trimming": False,
                      "source_separators": between}
            previous_class = {"class": klass, "role_id": left["role_id"]}
        entry = {"utterance_id": line["utterance_id"], "kind": "speech", "index": line["index"],
                 "role_id": line["role_id"], "role_name": line["role_name"], "narrator": line["narrator"],
                 "mode": line["mode"], "scene_id": line["scene_id"], "source_anchor": line["source"],
                 "text_sha256": line["text_sha256"], "voice_id": line["voice"]["id"],
                 "reference_audio_sha256": line["voice"]["reference_audio_sha256"],
                 "prompt_sha256": line["voice"]["prompt_sha256"], "audio_sha256": expected,
                 "audio_file": _relative(path, out), "start_seconds": elapsed,
                 "end_seconds": elapsed + len(pcm) / rate, "boundary": timing,
                 "neutral_take_reference": line["neutral_take"]}
        if line["narrator"]:
            entry.update(provenance="existing_neutral_take_reused_unchanged", native_instruction=None,
                         receipt=None)
        else:
            sample = receipts[(line["project_id"], line["utterance_id"])]
            entry.update(provenance="new_experimental_synthesis", native_instruction=sample["instruction"],
                         instruction_sha256=sample["instruction_sha256"], request_sha256=sample["request_sha256"],
                         runtime_sha256=sample["runtime_sha256"],
                         receipt={"file": f"receipts/{line['project_id']}/{line['utterance_id']}.json",
                                  "raw": sample["raw_render_receipt"]},
                         coaching=sample["coaching"])
        mapping.append(entry)
        chunks.append(pcm)
        elapsed += len(pcm) / rate
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
    if abs(duration - elapsed) > 0.001:
        raise BatchError("Assembled master duration differs from its source-to-audio map")
    report = qa.qa_take_audio(take_id="chapter_" + master_sha[:24], audio_path=master,
                              text="\n".join(l["text"] for l in mine), expected_sha256=master_sha, language="English")
    if not report.passed or report.blocking_failures() or not report.executed_check_ids():
        raise BatchError(f"Assembled chapter {number} fails blocking technical QA")
    if not mp3.exists():
        _mp3(master, mp3)
    delivered = _decoded_duration(mp3)
    if abs(delivered - duration) > 0.1:
        raise BatchError("Decoded listening file differs from its master")
    loudness = _loudness(mp3)
    mp3_sha = p.sha(mp3.read_bytes())
    clones = {}
    for line in mine:
        clones.setdefault(line["role_id"], {"role_id": line["role_id"], "role_name": line["role_name"],
                                            "narrator": line["narrator"], "voice_id": line["voice"]["id"],
                                            "reference_audio_sha256": line["voice"]["reference_audio_sha256"],
                                            "prompt_sha256": line["voice"]["prompt_sha256"],
                                            "voice_identity_sha256": line["voice"]["voice_identity_sha256"]})
    applied = _read_json(out / "applied" / f"{chapter['project_id']}.json")
    result = {
        "schema_version": CHAPTER_SCHEMA, "status": "ready_for_listening", "review_only": True,
        "human_review": "pending", "production_ready": False, "takes_selected": False, "approvals_created": False,
        "default_runtime_changed": False, "quality_improvement": "not_established",
        "chapter": {"number": number, "id": chapter["chapter_id"], "title": chapter["chapter_title"],
                    "project_id": chapter["project_id"]},
        "epub": {"sha256": state["epub"]["sha256"], "member": chapter["epub_member"],
                 "chapter_text_sha256": chapter["epub_text_sha256"], "nonblank_characters": chapter["nonblank_characters"]},
        "source": {"project_sha256": chapter["project_sha256"], "segments_sha256": chapter["segments_sha256"],
                   "take_state_sha256": chapter["take_state_sha256"],
                   "utterance_ids": [r["utterance_id"] for r in chapter["rows"]]},
        "clone_binding": list(clones.values()), "frozen_sha256": state["frozen_sha256"],
        "counts": {"speech": len(mine), "neutral_narration_reused": sum(l["narrator"] for l in mine),
                   "new_dialogue_synthesized": sum(not l["narrator"] for l in mine),
                   "scene_breaks": sum(r["kind"] == "scene_break" for r in chapter["rows"]),
                   "excluded": sum(r["kind"] == "excluded" for r in chapter["rows"])},
        "coaching": {"request_sha256": applied["request_sha256"], "response_sha256": applied["response_sha256"],
                     "bundle_sha256": applied["bundle_sha256"], "coached_plan_sha256": applied["coached_plan_sha256"],
                     "model_provenance": applied["model_provenance"]},
        "runtime": {"runtime_sha256": pinned, "profile": p.DIRECTED_PROFILE, "policy": p.DIRECTION_POLICY,
                    "neutral_narration_profile": NEUTRAL_PROFILE},
        "boundary_policy_sha256": state["boundary_policy_sha256"], "added_padding_seconds": padded,
        "master_file": _relative(master, out), "master_sha256": master_sha, "master_duration_seconds": duration,
        "audio_file": _relative(mp3, out), "audio_sha256": mp3_sha, "duration_seconds": delivered,
        "integrated_loudness_lufs": loudness,
        "loudness_measurement": {"audio_sha256": mp3_sha, "integrated_lufs": loudness, "method": "ffmpeg_ebur128"},
        "technical_qa": "passed", "qa_report": report.model_dump(mode="json"), "source_to_audio": mapping}
    if chapter["fixture"]:
        result["fixture"] = {"is_fixture": True, "label": FIXTURE_LABEL}
    p.publish(receipt_path, p.encoded(result))
    return result


def generate_batch(args, state, worker_class=None, render_fn=None):
    """Render the new coached dialogue serially and assemble every chapter (review only)."""
    cfg = _config(args)
    p, out = pilot(), cfg.output_dir
    with _fail_closed():
        disk = load_state(out)
        if state.get("frozen_sha256") != disk["frozen_sha256"] or _frozen_sha(state) != disk["frozen_sha256"]:
            raise BatchError("Supplied state differs from the frozen prepared batch")
        if disk["prepare_holds"]:
            raise BatchError("Preparation held; the batch will not generate")
        directions = _applied_directions(out, disk)
        p.verify_prepared_sources(SimpleNamespace(source_root=cfg.source_root, output_dir=out), disk)
        _verify_neutral_and_clones(cfg, disk)
        rlines = _runtime_lines(disk, directions)
        with _projects_root(out / "projects"):
            receipts, pinned, workers, dispatched = _render_new_dialogue(cfg, disk, rlines, worker_class, render_fn)
        audit = _audit(rlines, receipts, dispatched, workers, out)
        stable = {k: v for k, v in audit.items() if k not in ("new_requests_this_run", "workers_started_this_run")}
        if disk["fixture"]:
            stable["fixture"] = {"is_fixture": True, "label": FIXTURE_LABEL}
        p.publish(out / "forwarding-audit.json", p.encoded({"frozen_sha256": disk["frozen_sha256"], **stable}))
        results = [_assemble_chapter(cfg, disk, chapter, rlines, receipts, pinned) for chapter in disk["chapters"]]
        summary_rows = [{"chapter": r["chapter"]["number"], "project_id": r["chapter"]["project_id"],
                         "audio_file": r["audio_file"], "audio_sha256": r["audio_sha256"],
                         "master_sha256": r["master_sha256"], "duration_seconds": r["duration_seconds"],
                         "counts": r["counts"]} for r in results]
        result = {"schema_version": RESULT_SCHEMA, "frozen_sha256": disk["frozen_sha256"], "review_only": True,
                  "human_review": "pending", "production_ready": False, "takes_selected": False,
                  "approvals_created": False, "chapters": summary_rows,
                  "fixture": {"is_fixture": True, "label": FIXTURE_LABEL} if disk["fixture"] else None}
        p.publish(out / "batch-result.json", p.encoded(result))
        disk.update(status="generated", holds=[])
        _write_state(out, disk)
        return {"status": "generated", "audit": audit, "chapters": summary_rows,
                "result_sha256": p.sha((out / "batch-result.json").read_bytes()), "human_review": "pending"}


# ---------------------------------------------------------------------------------------- CLI


def _chapter_numbers(value):
    try:
        return [int(token) for token in value.replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a chapter number list: {value!r}") from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epub", type=Path, required=True, help="the exact EPUB the chapter projects were built from")
    parser.add_argument("--source-root", type=Path, required=True, help="read-only canonical project store")
    parser.add_argument("--projects", nargs="+", required=True, metavar="ID", help="canonical project ids, in chapter order")
    parser.add_argument("--chapters", nargs="+", type=_chapter_numbers, required=True, metavar="N",
                        help="chapter numbers in ascending order (spaces or commas), one per project")
    parser.add_argument("--output-dir", type=Path, required=True, help="private output outside the source store")
    parser.add_argument("--coaching-dir", type=Path, help="directory holding coaching-{project_id}.json bundles")
    parser.add_argument("--generate", action="store_true", help="explicit serial GPU synthesis (root-owned lane)")
    args = parser.parse_args(argv)
    args.chapters = [n for group in args.chapters for n in group]
    args.projects = [token for item in args.projects for token in item.replace(",", " ").split()]
    try:
        _config(args)
        if not args.epub.is_file():
            raise BatchError(f"EPUB not found: {args.epub}")
    except BatchError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        state = prepare_batch(args)
        if args.coaching_dir and state["status"] in ("awaiting_coaching", "held", "ready") and not state["prepare_holds"]:
            state = apply_bundles(args, state)
        summary = {"status": state["status"], "chapters": [s["chapter_number"] for s in state["sources"]],
                   "speech_lines": len(state["lines"]), "dialogue_lines": sum(not l["narrator"] for l in state["lines"]),
                   "coaching": state.get("coaching"), "holds": state["holds"], "output": str(args.output_dir),
                   "human_review": "pending", "production_ready": False}
        if args.generate:
            if state["status"] not in ("ready", "generated"):
                print(json.dumps(summary, ensure_ascii=True), flush=True)
                return 2
            summary["generation"] = generate_batch(args, state)
            summary["status"] = summary["generation"]["status"]
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return 2 if state["status"] == "held" else 0
    except (BatchError, RuntimeError) as exc:  # includes the render owner's refusal to resend
        print(json.dumps({"status": "held", "error": str(exc)}, ensure_ascii=True), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
