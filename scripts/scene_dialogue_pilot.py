"""Bounded cloned-voice scene pilot: 20 exact source lines, neutral baseline vs. native direction.

Prepare (CPU only, no model): python scripts/scene_dialogue_pilot.py --source-root ROOT \
    --projects ID1 ID2 ID3 --output-dir OUT
Generate (explicit, root-owned GPU lane): same command plus --generate.

Originals are read-only. ``prepare`` freezes contiguous source mini-scenes taken from the
canonical chapter projects, copies only the isolated state a worker needs (project.json,
segments.json, casting/voices, voice_lab/discovery) into OUT/projects, asks the scene dialogue
plan owner for recommendation-only directions, and writes OUT/state.json. Nothing is approved,
no default runtime changes and a missing or inadequate plan is a hold, never an invented
direction. ``--generate`` renders two arms (baseline, directed) through the same pinned
directed-wrapper worker, voice, text, settings and boundary policy. Only directed dialogue
carries a native instruction; narrator and neutral lines never do. Samples are review-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SELECTION_SCHEMA = "scene-dialogue-pilot-selection-v1"
STATE_SCHEMA = "scene-dialogue-pilot-state-v1"
PLAN_SCHEMA = "scene-dialogue-plan-v1"
MANIFEST_SCHEMA = "scene-dialogue-comparison-v1"
RECEIPT_SCHEMA = "scene-dialogue-sample-receipt-v1"
DIRECTED_PROFILE = "faster-base-032-direction-x1"
DIRECTION_POLICY = "native-instruct-icl-x1"
ARMS = ("baseline", "directed")

TARGET_LINES, MIN_NARRATOR, MIN_DIALOGUE = 20, 6, 14
MIN_SCENE_LINES, MAX_SCENE_LINES = 4, 7
MAX_CHARS = 250            # bounded codec budget: every synthesized line, narrator included
CONTEXT_LINES = 2
POOL_PER_BUCKET, POOL_LIMIT = 5, 60
MIN_DIRECTED_DIALOGUE, MIN_DIRECTED_SCENES = 6, 3
MAX_INSTRUCTION_CHARS = 240
PLAN_BUILDER_NAME = "build_scene_dialogue_plan"

_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,120}")
_BREAK = re.compile(r"\s*(?:[-*~=_#⁂]\s*){3,}\s*|\s*⁂\s*")
_HEADING = re.compile(r"\s*\d{1,4}\s*(?:\n|$)")
_NUMERIC_RATE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:%|percent\b|wpm\b|words per minute\b|x\b|times\b)")
_BORROWED_VOICE = re.compile(r"(?i)\b(?:sounds? like|sounding like|in the (?:voice|style) of|impersonat\w*|imitat\w*|clone)\b")
_IDENTITY_WORDS = re.compile(r"(?i)\b(?:accents?|dialects?|pitch|octave|gender|male|female|performer|actor|actress|"
                             r"timbre|baritone|soprano|tenor|voice identity)\b")

# Real source cues only. Each theme is a hypothesis about where the book supports a delivery
# contrast; the matched words are recorded verbatim as evidence, nothing is inferred beyond them.
THEMES = {
    "fear": re.compile(r"(?i)\b(?:scared|terrified|terrifying|afraid|fear\w*|frighten\w*|panic\w*|dread|horror|"
                       r"monster|danger\w*|dead|dying|scream\w*|yell\w*|shaking|trembl\w*)\b"),
    "worry": re.compile(r"(?i)\b(?:worr\w+|concern\w*|nervous|anxious|uneasy|what if|i don't know|not right|"
                        r"that's odd|hope no one|no signal)\b"),
    "reassurance": re.compile(r"(?i)\b(?:it's okay|it'll be fine|don't worry|i love you|you be good|i'm here|"
                              r"of course|it's fine|i'm sorry|more supportive|going to want you|take care)\b"),
    "dry_humor": re.compile(r"(?i)\b(?:joke\w*|jurassic park|haha|ha ha|promotion|advertisement|my old friend|"
                            r"smiled|grinned|giggl\w+|sarcas\w+|perfect of a set up|utterly terrified)\b"),
}
THEME_ORDER = tuple(THEMES)


class PilotError(RuntimeError):
    """A fail-closed refusal: the run is preserved and reported, never repaired silently."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return sha(text.encode("utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def value_sha(value):
    return sha(encoded(value))


def publish(path, data):
    """Create ``path`` exclusively; an identical re-publish is a no-op, a different one is a conflict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise PilotError(f"Existing immutable pilot artifact conflicts: {path}")


def replace_atomic(path, data):
    """Derived index files only (state.json); evidence artifacts always use publish()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hold(code, detail, **extra):
    return {"code": code, "detail": detail, **extra}


def direction_digest(instruction, seed_salt=0):
    """Same bytes as service.faster_qwen3tts_base_direction_adapter.component_sha256."""
    payload = {"control_policy": DIRECTION_POLICY, "native_instruction": instruction, "seed_salt": seed_salt}
    return sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


# ------------------------------------------------------------------------------ source reading


def read_source(source_root, project_id):
    """Read one canonical chapter project without touching it; return parsed JSON and file hashes."""
    if not _ID.fullmatch(project_id or ""):
        raise PilotError(f"Invalid project id: {project_id!r}")
    folder = Path(source_root) / project_id
    try:
        project_bytes = (folder / "project.json").read_bytes()
        segment_bytes = (folder / "segments.json").read_bytes()
        project = json.loads(project_bytes.decode("utf-8"))
        segments = json.loads(segment_bytes.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise PilotError(f"Canonical source project {project_id} is unreadable: {exc}") from exc
    if not isinstance(project, dict) or not isinstance(segments, list):
        raise PilotError(f"Canonical source project {project_id} is malformed")
    return {"id": project_id, "folder": folder, "project": project, "segments": segments,
            "project_sha256": sha(project_bytes), "segments_sha256": sha(segment_bytes)}


def source_registry(project):
    """Source revision id -> SHA-256. Missing, duplicate or malformed registries fail closed."""
    rows = project.get("source_revisions")
    if not isinstance(rows, list) or not rows:
        raise PilotError("Project has no source revision registry")
    registry = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] \
                or not isinstance(row.get("sha256"), str) or not _SHA.fullmatch(row["sha256"]):
            raise PilotError("Malformed source revision registry entry")
        if row["id"] in registry:
            raise PilotError(f"Duplicate source revision id {row['id']}")
        registry[row["id"]] = row["sha256"]
    return registry


def verify_canonical_chapter(source):
    """Exact selected source: ordered live segments must equal the chapter fork's recorded utterances."""
    project, project_id = source["project"], source["id"]
    meta = (project.get("metadata") or {}).get("chapter_rerun") or {}
    if meta.get("authorized") is not True:
        raise PilotError(f"{project_id} is not an authorized derived chapter rendition")
    live = [s for s in source["segments"]
            if not isinstance(s.get("lineage"), dict) or s["lineage"].get("retired_at") is None]
    live.sort(key=lambda s: s.get("index", 0))
    ids = [s.get("id") for s in live]
    if not ids or len(set(ids)) != len(ids) or ids != meta.get("source_utterance_ids"):
        raise PilotError(f"{project_id} segments no longer cover the recorded chapter source exactly")
    recorded = {r.get("utterance_id"): r for r in meta.get("source_utterances") or []}
    for segment in live:
        expected = recorded.get(segment["id"]) or {}
        if expected.get("source_text_sha256") != text_sha(segment.get("text") or ""):
            raise PilotError(f"{project_id}:{segment['id']} text differs from the recorded source")
        if expected.get("source_anchor") != segment.get("source_anchor"):
            raise PilotError(f"{project_id}:{segment['id']} anchor differs from the recorded source")
    return live, meta


def role_table(project):
    """role_id -> {name, narrator, identity (clone subset) | None, clone_holds}."""
    from dubbing_service.app.audiobook.cloned_voice_policy import role_holds
    table = {}
    for role in project.get("roles") or []:
        identity = role.get("voice_identity")
        holds = role_holds(role, mode="cloned_only")
        usable = bool(identity) and not holds and identity.get("status") == "LOCKED"
        subset = None
        if usable:
            subset = {key: identity[key] for key in ("id", "status", "mechanism", "backend", "reference_audio",
                                                     "reference_audio_sha256", "prompt_artifact", "prompt_sha256")}
            subset["voice_identity_sha256"] = value_sha(identity)
            subset["provenance_origin"] = (identity.get("provenance") or {}).get("origin")
        table[role["id"]] = {"name": role.get("name"), "narrator": role.get("role_type") == "narrator",
                             "identity": subset, "clone_holds": [h["code"] for h in holds]
                             if holds else ([] if usable else ["identity_not_locked"])}
    return table


# ------------------------------------------------------------------------------ scene selection


def source_cues(text, utterance_id, narrator):
    """Verbatim theme cues with exact character offsets. Dialogue cues weigh twice a narration cue."""
    flat = text.replace("’", "'").replace("‘", "'")  # same length, so offsets stay exact
    out = {}
    for theme, pattern in THEMES.items():
        hits = [{"theme": theme, "utterance_id": utterance_id, "cue": text[m.start():m.end()],
                 "start": m.start(), "end": m.end(), "weight": 1 if narrator else 2}
                for m in pattern.finditer(flat)]
        if hits:
            out[theme] = hits
    return out


def line_rows(source, live, roles, registry):
    rows = []
    for segment in live:
        text = segment.get("text") or ""
        role = roles.get(segment.get("role_id"))
        reasons = []
        if role is None:
            reasons.append("role_unknown")
        elif role["identity"] is None:
            reasons.append("no_locked_clone")
        if not text.strip():
            reasons.append("empty")
        if _BREAK.fullmatch(text):
            reasons.append("scene_break")
        if _HEADING.match(text):
            reasons.append("chapter_heading")
        if len(text) > MAX_CHARS:
            reasons.append("over_codec_budget")
        anchor = segment.get("source_anchor") or {}
        revision = anchor.get("source_revision_id")
        if revision not in registry:
            reasons.append("source_revision_unresolved")
        narrator = bool(role and role["narrator"])
        rows.append({"project_id": source["id"], "utterance_id": segment["id"], "index": segment.get("index"),
                     "chapter_id": segment.get("chapter_id"), "role_id": segment.get("role_id"),
                     "role_name": role["name"] if role else None, "narrator": narrator, "text": text,
                     "text_sha256": text_sha(text), "chars": len(text), "anchor": anchor,
                     "eligible": not reasons, "ineligible": reasons,
                     "cues": source_cues(text, segment["id"], narrator),
                     "voice_id": role["identity"]["id"] if role and role["identity"] else None})
    return rows


def candidate_windows(project_ids, rows_by_project):
    out = []
    for order, project_id in enumerate(project_ids):
        rows = rows_by_project[project_id]
        for start in range(len(rows)):
            for length in range(MIN_SCENE_LINES, MAX_SCENE_LINES + 1):
                chunk = rows[start:start + length]
                if len(chunk) < length or not all(r["eligible"] for r in chunk) \
                        or len({r["chapter_id"] for r in chunk}) != 1:
                    continue
                narrators = sum(r["narrator"] for r in chunk)
                if narrators < 1 or length - narrators < 2:
                    continue
                scores, evidence = {}, {}
                for theme in THEME_ORDER:
                    hits = [h for r in chunk for h in r["cues"].get(theme, [])]
                    if hits:
                        scores[theme] = sum(h["weight"] for h in hits)
                        evidence[theme] = hits
                if not scores or max(scores.values()) < 2:
                    continue
                theme = max(THEME_ORDER, key=lambda t: (scores.get(t, 0), -THEME_ORDER.index(t)))
                out.append({"project_id": project_id, "order": order, "start": start, "length": length,
                            "narrators": narrators, "dialogue": length - narrators, "theme": theme,
                            "score": min(scores[theme], 10), "theme_scores": scores,
                            "evidence": evidence[theme],
                            "voices": sorted({r["voice_id"] for r in chunk}),
                            "chapter_id": chunk[0]["chapter_id"]})
    return out


def _pool(windows):
    # Preserve length/narration combinations before ranking source cues. A few
    # high-scoring overlapping windows can crowd out every feasible 20-line set.
    buckets = {}
    for w in sorted(windows, key=lambda w: (-w["score"], w["order"], w["start"], w["length"])):
        key = (w["order"], w["theme"], w["length"], w["narrators"])
        buckets.setdefault(key, w)
    pooled = list(buckets.values())
    return sorted(pooled, key=lambda w: (w["order"], w["start"], w["length"]))


def _separate(a, b):
    return a["order"] != b["order"] or b["start"] >= a["start"] + a["length"] + 2


def _value(chosen, relax):
    voices = {v for w in chosen for v in w["voices"]}
    chapters = {w["chapter_id"] for w in chosen}
    themes = {w["theme"] for w in chosen}
    value = sum(w["score"] for w in chosen) + 6 * len(voices)
    if "chapters" not in relax:
        value += 5 * len(chapters)
    if "themes" not in relax:
        value += 3 * len(themes)
    return value


def choose_windows(pool, relax=()):
    """Exhaustive deterministic search over the pool for non-overlapping windows totalling 20 lines."""
    best = None
    for count in (4, 5, 3):
        candidates = pool if count < 5 else pool[:40]
        found = None

        def walk(position, chosen, lines):
            nonlocal found
            remaining = count - len(chosen)
            if remaining == 0:
                narrators = sum(w["narrators"] for w in chosen)
                if lines == TARGET_LINES and narrators >= MIN_NARRATOR and lines - narrators >= MIN_DIALOGUE:
                    if "themes" not in relax and len({w["theme"] for w in chosen}) < min(count, len(THEMES)):
                        return
                    value = _value(chosen, relax)
                    if found is None or value > found[0]:
                        found = (value, list(chosen))
                return
            for i in range(position, len(candidates)):
                w = candidates[i]
                used = lines + w["length"]
                if used + (remaining - 1) * MIN_SCENE_LINES > TARGET_LINES \
                        or used + (remaining - 1) * MAX_SCENE_LINES < TARGET_LINES:
                    continue
                if all(_separate(prior, w) for prior in chosen):
                    chosen.append(w)
                    walk(i + 1, chosen, used)
                    chosen.pop()

        walk(0, [], 0)
        if found and (best is None or found[0] > best[0]):
            best = found
            break
    return best[1] if best else None


def select_scenes(project_ids, rows_by_project):
    windows = candidate_windows(project_ids, rows_by_project)
    pool = _pool(windows)
    relaxed = ()
    chosen = choose_windows(pool)
    for relaxed in (("themes",), ("themes", "chapters")):
        if chosen:
            break
        chosen = choose_windows(pool, relax=relaxed)
    if not chosen:
        return None, {"candidates": len(windows), "pool": len(pool), "relaxed": list(relaxed)}
    chosen.sort(key=lambda w: (w["order"], w["start"]))
    return chosen, {"candidates": len(windows), "pool": len(pool), "relaxed": list(relaxed)}


def freeze_selection(project_ids, rows_by_project, chosen, roles_by_project, registries):
    """Complete contiguous scenes: every line of each window is kept, with read-only context around it."""
    scenes, lines = [], []
    for ordinal, window in enumerate(chosen):
        rows = rows_by_project[window["project_id"]]
        chunk = rows[window["start"]:window["start"] + window["length"]]
        scene_id = f"pilot-{ordinal + 1:02d}-" + value_sha([window["project_id"], chunk[0]["utterance_id"]])[:8]

        def context(items):
            return [{"utterance_id": r["utterance_id"], "role_id": r["role_id"], "role_name": r["role_name"],
                     "narrator": r["narrator"], "text": r["text"], "text_sha256": r["text_sha256"],
                     "synthesized": False} for r in items]

        before = rows[max(0, window["start"] - CONTEXT_LINES):window["start"]]
        after = rows[window["start"] + window["length"]:window["start"] + window["length"] + CONTEXT_LINES]
        scenes.append({"scene_id": scene_id, "project_id": window["project_id"], "chapter_id": window["chapter_id"],
                       "ordinal": ordinal, "theme": window["theme"], "theme_scores": window["theme_scores"],
                       "source_cues": window["evidence"], "utterance_ids": [r["utterance_id"] for r in chunk],
                       "context_before": context(before), "context_after": context(after),
                       "selection_basis": "deterministic lexical source cues; no model call, not a human decision"})
        registry = registries[window["project_id"]]
        roles = roles_by_project[window["project_id"]]
        for position, row in enumerate(chunk):
            anchor = row["anchor"]
            lines.append({
                "utterance_id": row["utterance_id"], "project_id": row["project_id"], "chapter_id": row["chapter_id"],
                "index": row["index"], "scene_id": scene_id, "position_in_scene": position,
                "role_id": row["role_id"], "role_name": row["role_name"], "narrator": row["narrator"],
                "text": row["text"], "text_sha256": row["text_sha256"], "chars": row["chars"],
                "source": {"source_revision_id": anchor.get("source_revision_id"),
                           "source_revision_sha256": registry[anchor["source_revision_id"]],
                           "start_offset": anchor.get("start_offset"), "end_offset": anchor.get("end_offset"),
                           "occurrence_index": anchor.get("occurrence_index")},
                "voice": roles[row["role_id"]]["identity"],
                "direction": {"native_instruction": None, "basis": "unavailable", "plan_sha256": None,
                              "recommendation_only": True, "human_review": "pending"}})
    return scenes, lines


def role_bindings(lines, roles_by_project):
    bindings = {}
    for line in lines:
        key = f"{line['project_id']}:{line['role_id']}"
        bindings.setdefault(key, {"project_id": line["project_id"], "role_id": line["role_id"],
                                  "role_name": line["role_name"], "narrator": line["narrator"],
                                  "voice": line["voice"], "lines": 0})["lines"] += 1
    available = {}
    for project_id, roles in roles_by_project.items():
        for role_id, role in roles.items():
            if role["identity"]:
                available.setdefault(role["identity"]["id"], role["name"])
    used = {b["voice"]["id"] for b in bindings.values()}
    return bindings, {"available_clones": len(available), "covered_clones": len(used & set(available)),
                      "available": available, "missing": {v: n for v, n in sorted(available.items()) if v not in used}}


# ------------------------------------------------------------------------------ isolated state


def copy_file(source, destination):
    if destination.exists():
        if sha(destination.read_bytes()) != sha(source.read_bytes()):
            raise PilotError(f"Isolated copy conflicts with its source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def prepare_isolated_project(source_folder, destination, voice_ids):
    """Copy only what the worker reads: metadata, frozen segments, needed clone folders, acquisitions."""
    sources = [source_folder / "project.json", source_folder / "segments.json"]
    for voice_id in sorted(voice_ids):
        if not _ID.fullmatch(voice_id):
            raise PilotError(f"Invalid voice identity id {voice_id!r}")
        folder = source_folder / "casting" / "voices" / voice_id
        if not folder.is_dir():
            raise PilotError(f"Clone folder is missing for {voice_id}")
        sources += [p for p in sorted(folder.rglob("*")) if p.is_file()]
    discovery = source_folder / "voice_lab" / "discovery"
    if discovery.is_dir():
        sources += [p for p in sorted(discovery.rglob("*")) if p.is_file()]
    project = json.loads((source_folder / "project.json").read_bytes())
    # Recorded-draft human selections are bound to exact audition bytes. Copy
    # their existing evidence, rather than inventing approval in the pilot.
    for record in project.get("voice_lab", {}).get("voices", {}).values():
        if (record.get("identity") or {}).get("id") not in voice_ids or not record.get("audio_file"):
            continue
        path = (source_folder / record["audio_file"]).resolve()
        if not path.is_relative_to(source_folder.resolve()) or path.is_symlink():
            raise PilotError("Approved audition evidence is outside the project")
        if not path.is_file() or sha(path.read_bytes()) != record.get("audio_sha256"):
            raise PilotError("Approved audition evidence is missing or changed")
        sources.append(path)
    copied = {}
    for path in sources:
        relative = path.relative_to(source_folder)
        copy_file(path, destination / relative)
        copied[relative.as_posix()] = sha((destination / relative).read_bytes())
    return copied


def verify_identity_files(project_root, voice):
    for path_key, hash_key in (("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")):
        relative = voice[path_key]
        if relative.startswith(("/", "\\")) or ".." in Path(relative).parts or not relative.startswith(f"casting/voices/{voice['id']}/"):
            raise PilotError(f"Clone artifact path is outside its identity: {relative}")
        path = project_root / relative
        if not path.is_file() or sha(path.read_bytes()) != voice[hash_key]:
            raise PilotError(f"Clone artifact is missing or hash-mismatched: {relative}")
    reference_record = project_root / f"casting/voices/{voice['id']}/reference.json"
    if reference_record.is_file():
        provenance = json.loads(reference_record.read_bytes()).get("provenance", {})
        if provenance.get("audition_draft_policy"):
            from dubbing_service.app.audiobook import recorded_audition_policy as draft
            if not draft.human_casting_context(project_root, voice["id"], provenance.get("acquisition_sha256")):
                raise PilotError("Recorded clone lacks its immutable positive human selection evidence")


# ------------------------------------------------------------------------------ scene plans


def discover_plan_module():
    """Locate the plan owner's module by its public function; None when it has not landed."""
    folder = REPO / "dubbing_service" / "app" / "audiobook"
    for path in sorted(folder.glob("*.py")):
        try:
            if f"def {PLAN_BUILDER_NAME}(" in path.read_text(encoding="utf-8"):
                return f"dubbing_service.app.audiobook.{path.stem}"
        except OSError:
            continue
    return None


def load_plan_builder(spec=None):
    spec = spec or os.environ.get("SCENE_DIALOGUE_PLAN_BUILDER") or discover_plan_module()
    if not spec:
        return None
    module_name, _, function = spec.partition(":")
    module = importlib.import_module(module_name)
    builder = getattr(module, function or PLAN_BUILDER_NAME, None)
    if not callable(builder):
        raise PilotError(f"{spec} does not expose {function or PLAN_BUILDER_NAME}")
    return builder


def instruction_problem(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        return "empty instruction"
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        return f"over {MAX_INSTRUCTION_CHARS} characters"
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in instruction):
        return "control or surrogate character"
    if _NUMERIC_RATE.search(instruction) or _BORROWED_VOICE.search(instruction):
        return "numeric rate or borrowed voice"
    if _IDENTITY_WORDS.search(instruction):
        return "performer, accent, pitch, gender or voice-identity wording"
    return None


def apply_plan(plan, plan_sha256, project_lines):
    """Validate one project's plan against its frozen lines; return (directions by id, holds)."""
    holds = []
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        return {}, [hold("plan_schema", f"plan is not {PLAN_SCHEMA}")]
    if plan.get("recommendation_only") is not True:
        holds.append(hold("plan_not_recommendation_only", "plan must declare recommendation_only=true"))
    if not isinstance(plan.get("binding_sha256"), str) or not _SHA.fullmatch(plan["binding_sha256"]):
        holds.append(hold("plan_binding", "plan has no valid binding_sha256"))
    rows = plan.get("directions")
    rows = {r.get("utterance_id"): r for r in rows} if isinstance(rows, list) else {}
    directions = {}
    for line in project_lines:
        row = rows.get(line["utterance_id"])
        if row is None:
            holds.append(hold("plan_line_missing", "no direction row", utterance_id=line["utterance_id"]))
            continue
        problems = []
        if row.get("text_sha256") != line["text_sha256"]:
            problems.append("text hash differs from frozen source")
        if row.get("role_id") != line["role_id"]:
            problems.append("role differs from frozen source")
        if bool(row.get("narrator")) != line["narrator"]:
            problems.append("narrator flag differs from frozen source")
        if row.get("recommendation_only") is not True or row.get("approved") is True:
            problems.append("direction is not recommendation-only")
        if row.get("basis") == "explicit_human":
            problems.append("no human override was supplied; basis explicit_human is a disguised proposal")
        instruction = row.get("native_instruction")
        if line["narrator"] and instruction is not None:
            problems.append("narrator carries a native instruction")
        if instruction is not None and not line["narrator"]:
            problem = instruction_problem(instruction)
            if problem:
                problems.append(f"unsafe instruction: {problem}")
        if problems:
            holds.append(hold("plan_line_refused", "; ".join(problems), utterance_id=line["utterance_id"]))
            continue
        directions[line["utterance_id"]] = {
            "native_instruction": instruction, "basis": row.get("basis"), "plan_sha256": plan_sha256,
            "evidence": row.get("evidence"), "holds": row.get("holds") or [],
            "voice_revision": row.get("voice_revision"), "recommendation_only": True, "human_review": "pending"}
        if row.get("coaching") is not None:
            directions[line["utterance_id"]].update(coaching=row["coaching"], coaching_scene_id=row["coaching_scene_id"])
    return directions, holds


def adequacy(scenes, lines, *, minimum_lines, minimum_scenes):
    directed = [l for l in lines if not l["narrator"] and l["direction"]["native_instruction"]]
    scene_ids = {l["scene_id"] for l in directed}
    holds = []
    if len(directed) < minimum_lines or len(scene_ids) < minimum_scenes:
        holds.append(hold("insufficient_automatic_directions",
                          f"{len(directed)} directed dialogue lines in {len(scene_ids)} scenes; "
                          f"need {minimum_lines} in {minimum_scenes}. No direction is invented and no human approval implied."))
    return {"directed_dialogue_lines": len(directed), "directed_scenes": len(scene_ids),
            "neutral_dialogue_lines": sum(1 for l in lines if not l["narrator"]) - len(directed),
            "narrator_lines": sum(1 for l in lines if l["narrator"])}, holds


# ------------------------------------------------------------------------------ prepare


def _models(source):
    from dubbing_service.app.models.project import Project, Segment
    return Project(**source["project"]), [Segment(**s) for s in source["segments"]]


def prepare(args, plan_builder=None, model_loader=_models):
    output = args.output_dir
    sources = [read_source(args.source_root, pid) for pid in args.projects]
    registries, roles_by_project, rows_by_project, chapters = {}, {}, {}, []
    for source in sources:
        live, meta = verify_canonical_chapter(source)
        registries[source["id"]] = source_registry(source["project"])
        roles_by_project[source["id"]] = role_table(source["project"])
        rows_by_project[source["id"]] = line_rows(source, live, roles_by_project[source["id"]], registries[source["id"]])
        chapters.append({"project_id": source["id"], "chapter_id": meta.get("chapter_id"),
                         "chapter_title": meta.get("chapter_title"), "source_sha256": meta.get("source_sha256"),
                         "project_sha256": source["project_sha256"], "segments_sha256": source["segments_sha256"],
                         "segment_count": len(live)})
    state = {"schema_version": STATE_SCHEMA, "pilot": "scene-aware-dialogue-v1",
             "fixture": any(s["project"].get("metadata", {}).get("scene_pilot_fixture") is True for s in sources),
             "human_review": "pending", "production_ready": False, "review_only": True,
             "default_runtime_changed": False, "approvals_created": False,
             "runtime": {"directed_profile": DIRECTED_PROFILE, "policy": DIRECTION_POLICY,
                         "request": "experimental review only; current Faster default keeps no forwarding"},
             "sources": chapters, "holds": [], "arms": list(ARMS)}
    prior = getattr(args, "selection_from", None)
    previous = load_state(prior) if prior else None
    chosen, search = (None, {"reused_selection": str(prior)}) if previous else select_scenes(args.projects, rows_by_project)
    state["selection_search"] = search
    if not chosen and previous is None:
        state.update(status="held", holds=[hold("selection_infeasible", "No complete scene set satisfies the line, "
                                                  "narrator, dialogue and codec-budget constraints", **search)],
                     lines=[], scenes=[])
        return finish(output, state, sources)
    if previous is None:
        scenes, lines = freeze_selection(args.projects, rows_by_project, chosen, roles_by_project, registries)
    else:
        import copy
        verify_prepared_sources(argparse.Namespace(source_root=args.source_root, output_dir=prior), previous)
        if {s["project_id"] for s in previous["sources"]} != set(args.projects):
            raise PilotError("Prior selection belongs to different source projects")
        scenes, lines = copy.deepcopy(previous["scenes"]), copy.deepcopy(previous["lines"])
        for line in lines:
            live = next((r for r in rows_by_project[line["project_id"]] if r["utterance_id"] == line["utterance_id"]), None)
            if (not live or not live["eligible"] or any(live[k] != line[k] for k in
                    ["text_sha256", "role_id", "role_name", "narrator", "chapter_id", "index"]) or
                    roles_by_project[line["project_id"]][line["role_id"]]["identity"] != line["voice"]):
                raise PilotError("Prior selection source or cloned cast changed")
        state["previous_pilot"] = {"directory": str(prior), "frozen_sha256": previous["frozen_sha256"]}
    bindings, coverage = role_bindings(lines, roles_by_project)
    state.update(scenes=scenes, lines=lines, role_bindings=bindings, clone_coverage=coverage)
    counts = {"lines": len(lines), "narrator": sum(l["narrator"] for l in lines),
              "dialogue": sum(not l["narrator"] for l in lines)}
    state["counts"] = counts
    if counts["lines"] != TARGET_LINES or counts["narrator"] < MIN_NARRATOR or counts["dialogue"] < MIN_DIALOGUE:
        state["holds"].append(hold("selection_counts", f"frozen counts {counts} violate the pilot contract"))
    selection = {"schema_version": SELECTION_SCHEMA, "lines": lines, "scenes": scenes, "role_bindings": bindings}
    state["selection_sha256"] = value_sha(selection)
    publish(output / "selection.json", encoded(selection))
    state["isolated"] = {}
    for source in sources:
        voices = {l["voice"]["id"] for l in lines if l["project_id"] == source["id"]}
        if voices:
            copied = prepare_isolated_project(source["folder"], output / "projects" / source["id"], voices)
            state["isolated"][source["id"]] = {"files": len(copied), "tree_sha256": value_sha(copied)}
            for line in (l for l in lines if l["project_id"] == source["id"]):
                verify_identity_files(output / "projects" / source["id"], line["voice"])
    state["boundary_policy_sha256"] = boundary_policy_sha()
    state["plans"] = {}
    builder = plan_builder if plan_builder is not None else load_plan_builder(getattr(args, "plan_builder", None))
    if builder is None:
        state["holds"].append(hold("plan_api_unavailable", f"{PLAN_BUILDER_NAME} is not importable yet; "
                                                            "directions stay unavailable and no instruction is invented"))
    else:
        cache = output / "scene-cache"
        cache.mkdir(parents=True, exist_ok=True)
        for source in sources:
            mine = [l for l in lines if l["project_id"] == source["id"]]
            if not mine:
                continue
            project, segments = model_loader(source)
            plan = builder(project, segments, cache_root=cache, selected_ids=[l["utterance_id"] for l in mine])
            coaching_file = getattr(args, "coaching_file", None)
            if coaching_file:
                from dubbing_service.app.audiobook.scene_coaching import apply_coaching, CoachingError
                bundle_bytes = coaching_file.read_bytes()
                try:
                    plan = apply_coaching(plan, json.loads(bundle_bytes), project, segments, cache_root=cache)
                except (CoachingError, ValueError) as exc:
                    raise PilotError(f"Scene coaching refused: {exc}") from exc
                publish(output / "coaching-bundle.json", bundle_bytes)
            plan_bytes = encoded(plan)
            publish(output / "plans" / f"plan-{source['id']}.json", plan_bytes)
            directions, plan_holds = apply_plan(plan, sha(plan_bytes), mine)
            state["holds"] += [{**h, "project_id": source["id"]} for h in plan_holds]
            for line in mine:
                if line["utterance_id"] in directions:
                    line["direction"] = directions[line["utterance_id"]]
            state["plans"][source["id"]] = {"file": f"plans/plan-{source['id']}.json", "sha256": sha(plan_bytes),
                                            "binding_sha256": plan.get("binding_sha256"),
                                            "scene_data": plan.get("scenes"), "cache": plan.get("cache")}
        state["adequacy"], adequacy_holds = adequacy(scenes, lines, minimum_lines=args.min_directed_dialogue,
                                                     minimum_scenes=args.min_directed_scenes)
        state["holds"] += adequacy_holds
    state["status"] = "held" if state["holds"] else "ready"
    state["frozen_sha256"] = value_sha({"lines": lines, "scenes": scenes, "role_bindings": bindings,
                                        "plans": {k: v["sha256"] for k, v in state["plans"].items()},
                                        "boundary_policy_sha256": state["boundary_policy_sha256"]})
    return finish(output, state, sources)


def boundary_policy_sha():
    from dubbing_service.app.audiobook import performance_boundaries as boundaries
    return value_sha(boundaries.default_policy())


def finish(output, state, sources):
    for source in sources:  # originals must be byte-identical after preparation
        after = read_source(source["folder"].parent, source["id"])
        if (after["project_sha256"], after["segments_sha256"]) != (source["project_sha256"], source["segments_sha256"]):
            raise PilotError(f"Canonical source {source['id']} changed during preparation")
    state["source_unchanged"] = True
    replace_atomic(output / "state.json", encoded(state))
    return state


# ------------------------------------------------------------------------------ receipts


def expected_instruction(arm, line):
    if arm == "directed" and not line["narrator"]:
        return line["direction"]["native_instruction"]
    return None


def expected_request_sha(project_id, voice, text, instruction):
    payload = {key: voice[key] for key in ("prompt_artifact", "prompt_sha256", "reference_audio", "reference_audio_sha256")}
    payload.update(project_id=project_id, identity_id=voice["id"], text=text, language="English", performance={})
    if instruction:
        payload["performance"] = {"control_policy": DIRECTION_POLICY, "native_instruction": instruction,
                                  "seed_salt": 0, "direction_sha256": direction_digest(instruction)}
    return value_sha_compact(payload)


def value_sha_compact(payload):
    # Mirrors scripts.directors_vertical_slice.encoded(): no trailing newline.
    return sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode())


def verify_sample(arm, line, render_receipt, pinned_runtime):
    """Check one render receipt against the frozen line and the arm's instruction rule."""
    instruction = expected_instruction(arm, line)
    problems = []
    if render_receipt.get("text_sha256") != line["text_sha256"]:
        problems.append("text hash differs")
    if render_receipt.get("native_instruction") != instruction:
        problems.append("recorded instruction differs from the arm rule")
    if render_receipt.get("request_sha256") != expected_request_sha(line["project_id"], line["voice"], line["text"], instruction):
        problems.append("request differs from the frozen identity, text and instruction")
    if pinned_runtime and render_receipt.get("runtime_sha256") != pinned_runtime:
        problems.append("runtime differs from the pinned worker")
    worker = render_receipt.get("worker_receipt") or {}
    direction = (worker.get("generation") or {}).get("direction")
    if instruction:
        wanted = {"forwarded": True, "instruct_sha256": text_sha(instruction), "instruct_chars": len(instruction),
                  "direction_sha256": direction_digest(instruction), "policy": DIRECTION_POLICY}
        if not isinstance(direction, dict) or any(direction.get(k) != v for k, v in wanted.items()):
            problems.append("worker did not confirm the exact forwarded instruction")
    elif isinstance(direction, dict) and direction.get("forwarded"):
        problems.append("worker forwarded an instruction that this line must not carry")
    if worker.get("fallback") not in (None, False):
        problems.append("worker reported a fallback")
    if problems:
        raise PilotError(f"{arm}:{line['utterance_id']} receipt refused: " + "; ".join(problems))
    return {"instruction": instruction, "forwarded": bool(instruction), "worker_direction_receipt": direction}


def sample_receipt(arm, line, render_receipt, audio_file, verified):
    return {"schema_version": RECEIPT_SCHEMA, "arm": arm, "utterance_id": line["utterance_id"],
            "project_id": line["project_id"], "scene_id": line["scene_id"], "narrator": line["narrator"],
            "text_sha256": line["text_sha256"], "audio_file": audio_file, "audio_sha256": render_receipt["audio_sha256"],
            "request_sha256": render_receipt["request_sha256"], "runtime_sha256": render_receipt["runtime_sha256"],
            "voice_id": line["voice"]["id"], "reference_id": line["voice"]["reference_audio_sha256"],
            "prompt_id": line["voice"]["prompt_sha256"],
            "instruction": verified["instruction"], "instruction_sha256": text_sha(verified["instruction"]) if verified["instruction"] else None,
            "instruction_forwarded": verified["forwarded"], "worker_direction_receipt": verified["worker_direction_receipt"],
            "generation_settings": (render_receipt.get("worker_receipt") or {}).get("generation_settings"),
            "latency_seconds": render_receipt["latency_seconds"], "peak_vram_mb": render_receipt["peak_vram_mb"],
            "vram_measurement": render_receipt.get("vram_measurement"),
            "review_only": True, "human_review": "pending", "production_ready": False}


def audit_forwarding(lines, receipts):
    """receipts[arm][utterance_id] -> sample receipt. Forwarded only for directed dialogue, never otherwise."""
    audit = {"baseline_instructions": 0, "narrator_instructions": 0, "neutral_instructions": 0,
             "directed_forwarded": 0, "directed_dialogue_expected": 0, "samples": 0}
    for arm in ARMS:
        for line in lines:
            receipt = receipts[arm][line["utterance_id"]]
            audit["samples"] += 1
            if receipt["instruction_forwarded"] != bool(expected_instruction(arm, line)):
                raise PilotError(f"{arm}:{line['utterance_id']} forwarding differs from the frozen plan")
            if not receipt["instruction_forwarded"]:
                continue
            if arm == "baseline":
                audit["baseline_instructions"] += 1
            elif line["narrator"]:
                audit["narrator_instructions"] += 1
            elif not line["direction"]["native_instruction"]:
                audit["neutral_instructions"] += 1
            else:
                audit["directed_forwarded"] += 1
    audit["directed_dialogue_expected"] = sum(1 for l in lines if not l["narrator"] and l["direction"]["native_instruction"])
    if audit["baseline_instructions"] or audit["narrator_instructions"] or audit["neutral_instructions"] \
            or audit["directed_forwarded"] != audit["directed_dialogue_expected"]:
        raise PilotError(f"Forwarding audit failed: {audit}")
    audit["verdict"] = "instruction forwarded true only for directed dialogue"
    return audit


# ------------------------------------------------------------------------------ assembly


def assemble_scene(output, arm, scene, lines, sample_paths, role_types, aliases):
    """Join one scene's samples under the shared semantic boundary policy (existing silence respected)."""
    import io
    import numpy as np
    import soundfile as sf
    from dubbing_service.app.audiobook import performance_boundaries as boundaries
    policy = boundaries.default_policy()
    chunks, timing, previous, previous_class, rate, elapsed = [], [], None, None, None, 0.0
    for line in lines:
        pcm, this_rate = sf.read(sample_paths[line["utterance_id"]]["path"], dtype="float32", always_2d=True)
        if rate is not None and rate != this_rate:
            raise PilotError("Sample domains differ; no implicit resampling")
        rate = this_rate
        measure = boundaries.measure_take(sample_paths[line["utterance_id"]]["path"],
                                          sample_paths[line["utterance_id"]]["audio_sha256"], text=line["text"])
        row = {"utterance_id": line["utterance_id"], "text": line["text"], "role_id": line["role_id"],
               "source_revision_id": line["source"]["source_revision_id"],
               "start_offset": line["source"]["start_offset"], "end_offset": line["source"]["end_offset"]}
        boundary = None
        if previous:
            row["previous_tag"] = previous_class
            classification = boundaries.classify_boundary(previous[0], row, role_types=role_types, aliases=aliases)
            klass = classification["class"]
            target = policy["targets_ms"].get(klass, 350)
            existing = (previous[1]["trailing"]["silent_frames"] + measure["leading"]["silent_frames"]) * 1000 / rate
            padding = max(0, round((target - existing) * rate / 1000))
            chunks.append(np.zeros((padding, pcm.shape[1]), dtype="float32"))
            elapsed += padding / rate
            boundary = {"left": previous[0]["utterance_id"], "right": line["utterance_id"], "classification": classification,
                        "target_ms": target, "existing_silence_ms": existing, "added_frames": padding, "trimming": False}
            previous_class = {"class": klass, "role_id": previous[0]["role_id"]}
        timing.append({"utterance_id": line["utterance_id"], "start_seconds": elapsed,
                       "end_seconds": elapsed + len(pcm) / rate, "boundary": boundary})
        chunks.append(pcm)
        elapsed += len(pcm) / rate
        previous = (row, measure)
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), rate, format="WAV", subtype="PCM_16")
    wav = output / "scenes" / arm / f"{scene['scene_id']}.wav"
    publish(wav, buffer.getvalue())
    publish(wav.with_suffix(".timing.json"), encoded({"policy_sha256": value_sha(policy), "timing": timing}))
    return {"audio_file": wav.relative_to(output).as_posix(), "audio_sha256": sha(buffer.getvalue()),
            "timing_file": wav.with_suffix(".timing.json").relative_to(output).as_posix(),
            "duration_seconds": elapsed, "policy_sha256": value_sha(policy)}


# ------------------------------------------------------------------------------ generation


def load_state(output):
    path = output / "state.json"
    if not path.is_file():
        raise PilotError("No prepared state; run without --generate first")
    state = json.loads(path.read_bytes())
    if state.get("schema_version") != STATE_SCHEMA:
        raise PilotError("Unknown state schema")
    if state.get("status") != "ready":
        raise PilotError("Pilot is held and will not generate: " + json.dumps(state.get("holds"), ensure_ascii=False)[:800])
    frozen = value_sha({"lines": state["lines"], "scenes": state["scenes"], "role_bindings": state["role_bindings"],
                        "plans": {k: v["sha256"] for k, v in state["plans"].items()},
                        "boundary_policy_sha256": state["boundary_policy_sha256"]})
    if frozen != state.get("frozen_sha256"):
        raise PilotError("Frozen lines, plans or policy changed since preparation")
    for project_id, plan in state["plans"].items():
        if sha((output / plan["file"]).read_bytes()) != plan["sha256"]:
            raise PilotError(f"Saved plan for {project_id} changed")
    for line in state["lines"]:
        verify_identity_files(output / "projects" / line["project_id"], line["voice"])
    return state


def default_render_tools():
    spec = importlib.util.spec_from_file_location("directors_vertical_slice", REPO / "scripts" / "directors_vertical_slice.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Worker, module.render


def pin_runtime(output, runtime_sha256, identity):
    contract = (identity or {}).get("contract") or {}
    if contract.get("runtime_profile") != DIRECTED_PROFILE:
        raise PilotError(f"Worker is not the pinned {DIRECTED_PROFILE} wrapper: {contract.get('runtime_profile')!r}")
    pin = {"runtime_sha256": runtime_sha256, "runtime_profile": contract["runtime_profile"],
           "checkpoint_manifest_sha256": contract.get("checkpoint_manifest_sha256"), "model_id": contract.get("model_id")}
    path = output / "runtime-pin.json"
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved != pin:
            raise PilotError("Worker runtime differs from the pinned run; preserve this output and choose a new directory")
    else:
        publish(path, encoded(pin))
    return pin


def verified_previous(state):
    """Read and validate the previous pilot without changing its artifacts."""
    prior = Path(state["previous_pilot"]["directory"])
    previous = load_state(prior)
    manifest_bytes = (prior / "scene-comparison.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (previous["frozen_sha256"] != state["previous_pilot"]["frozen_sha256"] or
            manifest["frozen_sha256"] != previous["frozen_sha256"] or
            Path(manifest["base_dir"]).resolve() != prior.resolve() or
            previous["boundary_policy_sha256"] != state["boundary_policy_sha256"]):
        raise PilotError("Prior pilot or boundary policy changed")
    old_lines = {l["utterance_id"]: l for l in previous["lines"]}
    if set(old_lines) != {l["utterance_id"] for l in state["lines"]}:
        raise PilotError("Prior pilot selection differs")
    for line in state["lines"]:
        old = old_lines[line["utterance_id"]]
        if any(old[k] != line[k] for k in ["text_sha256", "voice", "source", "role_id", "scene_id", "narrator"]):
            raise PilotError("Prior pilot source, scene or cast differs")
    from dubbing_service.app.audiobook.scene_dialogue_evaluation import evaluate_scene_pilot, EvaluationError
    try:
        evaluate_scene_pilot(manifest)  # verifies both arms' audio, raw receipts, settings and forwarding
    except EvaluationError as exc:
        raise PilotError(f"Prior pilot QA refused reuse: {exc}") from exc
    return prior, previous, manifest, sha(manifest_bytes)


def reuse_prior_neutral(output, state):
    """Reuse proven baseline clips and neutral narration; retain their original receipts."""
    prior, _, manifest, manifest_sha = verified_previous(state)
    pin_bytes = (prior / "runtime-pin.json").read_bytes()
    pin = json.loads(pin_bytes)
    if pin["runtime_sha256"] != manifest["runtime"]["runtime_sha256"]:
        raise PilotError("Prior runtime pin differs from receipts")
    pending, reused = [], []
    for line in state["lines"]:
        arms = ARMS if line["narrator"] else ("baseline",)
        src = prior / "samples" / "baseline" / f"{line['utterance_id']}.wav"
        audio, raw = src.read_bytes(), src.with_suffix(".receipt.json").read_bytes()
        receipt = json.loads(raw)
        for arm in arms:
            verify_sample(arm, line, receipt, pin["runtime_sha256"])
            if sha(audio) != receipt["audio_sha256"]:
                raise PilotError("Prior audio differs from its receipt")
            target = output / "samples" / arm / src.name
            pending.extend([(target, audio), (target.with_suffix(".receipt.json"), raw)])
            reused.append({"arm": arm, "utterance_id": line["utterance_id"],
                           "source_arm": "baseline", "audio_sha256": sha(audio), "receipt_sha256": sha(raw)})
    # Check every conflict before copying anything; exclusive publication protects retries.
    pending.append((output / "runtime-pin.json", pin_bytes))
    for path, data in pending:
        if path.exists() and path.read_bytes() != data:
            raise PilotError("Existing reuse artifact differs; choose a new output directory")
    for path, data in pending:
        publish(path, data)
    publish(output / "reused-samples.json", encoded({"source_manifest_sha256": manifest_sha,
            "source_directory": str(prior), "samples": reused,
            "latency_policy": "Reused receipts retain original generation measurements; no new inference occurred"}))


def run_generation(output, state, worker_class, render_fn):
    """Render both arms line by line, resuming verified samples; hold on any ambiguity."""
    os.environ["DUBBING_PROJECTS_ROOT"] = str(output / "projects")
    from dubbing_service.app.core.config import clear_config_caches
    clear_config_caches()
    lines = state["lines"]
    pin_path = output / "runtime-pin.json"
    pinned = json.loads(pin_path.read_bytes())["runtime_sha256"] if pin_path.exists() else None
    receipts = {arm: {} for arm in ARMS}
    worker, stack = None, []
    try:
        for line in lines:
            for arm in ARMS:
                audio = output / "samples" / arm / f"{line['utterance_id']}.wav"
                done = audio.with_suffix(".receipt.json").exists()
                if not done and worker is None:
                    worker = worker_class(output, directed=True)
                    stack.append(worker)
                    worker.__enter__()
                    pinned = pin_runtime(output, worker.identity["sha256"], worker.identity)["runtime_sha256"]
                resolve_approval_refusal(output, arm, line, audio)
                render_receipt = render_fn(worker, line["project_id"], line["voice"], line, audio, expected_instruction(arm, line))
                pinned = pinned or render_receipt["runtime_sha256"]
                verified = verify_sample(arm, line, render_receipt, pinned)
                record = sample_receipt(arm, line, render_receipt, audio.relative_to(output).as_posix(), verified)
                publish(output / "receipts" / arm / f"{line['utterance_id']}.json", encoded(record))
                receipts[arm][line["utterance_id"]] = record
    finally:
        for item in reversed(stack):
            item.__exit__(None, None, None)
    return receipts, pinned


def resolve_approval_refusal(output, arm, line, audio):
    """Retry a proven pre-inference approval-file refusal after evidence is restored.

    Preserve both the original reservation and failure. Every other incomplete
    dispatch remains ambiguous and the existing render owner refuses to resend.
    """
    if audio.with_suffix(".receipt.json").exists():
        return  # a completed render supersedes the retained earlier refusal
    failure_path = audio.with_suffix(".failure.json")
    if not failure_path.is_file():
        return
    failure = json.loads(failure_path.read_bytes())
    expected_detail = "unapproved recorded draft is restricted to its exact audition; human selection required for production"
    if failure != {"status": 400, "response": {"detail": expected_detail}}:
        return
    request_sha = expected_request_sha(line["project_id"], line["voice"], line["text"], expected_instruction(arm, line))
    pending = audio.with_suffix(f".pending-{request_sha[:16]}.json")
    if not pending.exists():
        return
    reservation = json.loads(pending.read_bytes())
    if reservation.get("request_sha256") != request_sha:
        raise PilotError("Refused request differs from the frozen pilot request")
    from dubbing_service.app.audiobook import recorded_audition_policy as draft
    root = output / "projects" / line["project_id"]
    reference = json.loads((root / "casting" / "voices" / line["voice"]["id"] / "reference.json").read_bytes())
    acquisition_sha = reference.get("provenance", {}).get("acquisition_sha256")
    if not draft.human_casting_context(root, line["voice"]["id"], acquisition_sha):
        raise PilotError("Recorded clone still lacks its exact positive human selection evidence")
    failure_sha = sha(failure_path.read_bytes())
    resolution = {"request_sha256": request_sha, "failure_sha256": failure_sha,
                  "outcome": "HTTP 400 before model dispatch; no audio returned",
                  "human_selection_evidence_verified": True, "approval_created": False}
    publish(output / "retry-resolutions" / f"{request_sha}-{failure_sha[:12]}.json", encoded(resolution))
    archive = pending.with_name(pending.name + f".refused-{failure_sha[:12]}")
    if archive.exists():
        if archive.read_bytes() != pending.read_bytes():
            raise PilotError("Refused dispatch archive conflict")
        raise PilotError("A second refusal for the same request needs inspection")
    os.rename(pending, archive)


def compare_arms(lines, receipts):
    """Both arms must differ only by the instruction: same voice, text, settings, runtime."""
    for line in lines:
        base, directed = (receipts[arm][line["utterance_id"]] for arm in ARMS)
        for key in ("text_sha256", "voice_id", "reference_id", "prompt_id", "runtime_sha256"):
            if base[key] != directed[key]:
                raise PilotError(f"Arms differ in {key} for {line['utterance_id']}")
        if base["generation_settings"] != directed["generation_settings"]:
            raise PilotError(f"Arms differ in generation settings for {line['utterance_id']}")


def build_manifest(output, state, receipts, pinned_runtime):
    lines = state["lines"]
    role_types = {l["role_id"]: "narrator" if l["narrator"] else "character" for l in lines}
    arms = {}
    pin = json.loads((output / "runtime-pin.json").read_bytes())
    settings_sha = value_sha([[l["utterance_id"], receipts["baseline"][l["utterance_id"]]["generation_settings"]]
                              for l in lines])
    runtime = {"name": "Faster Qwen3-TTS Base 0.3.2 isolated native-instruction wrapper",
               "runtime_profile": DIRECTED_PROFILE, "runtime_sha256": pinned_runtime,
               "checkpoint_sha256": pin["checkpoint_manifest_sha256"], "settings_sha256": settings_sha,
               "boundary_policy_sha256": state["boundary_policy_sha256"], "fallback": None}
    for arm in ARMS:
        samples = {}
        for line in lines:
            r = receipts[arm][line["utterance_id"]]
            raw_path = output / "samples" / arm / f"{line['utterance_id']}.receipt.json"
            samples[line["utterance_id"]] = {
                "text": line["text"], "role_id": line["role_id"],
                "audio_file": r["audio_file"], "audio_sha256": r["audio_sha256"], "text_sha256": r["text_sha256"],
                "voice_id": r["voice_id"], "reference_id": r["reference_id"], "prompt_id": r["prompt_id"],
                "runtime_sha256": r["runtime_sha256"], "generation_settings": r["generation_settings"],
                "instruction": {"native_instruction": r["instruction"], "forwarded": r["instruction_forwarded"],
                                "direction_sha256": (r["worker_direction_receipt"] or {}).get("direction_sha256")},
                "instruction_forwarded": r["instruction_forwarded"],
                "receipt": {"path": raw_path.relative_to(output).as_posix(), "sha256": sha(raw_path.read_bytes())},
                "request_sha256": r["request_sha256"],
                "latency_seconds": r["latency_seconds"], "peak_vram_mb": r["peak_vram_mb"],
                "vram_measurement": r["vram_measurement"]}
        scenes = {}
        for scene in state["scenes"]:
            scene_lines = [l for l in lines if l["scene_id"] == scene["scene_id"]]
            paths = {l["utterance_id"]: {"path": output / samples[l["utterance_id"]]["audio_file"],
                                         "audio_sha256": samples[l["utterance_id"]]["audio_sha256"]} for l in scene_lines}
            scenes[scene["scene_id"]] = assemble_scene(output, arm, scene, scene_lines, paths, role_types, {})
        arms[arm] = {"engine": {"profile": DIRECTED_PROFILE, "policy": DIRECTION_POLICY, "runtime_sha256": pinned_runtime},
                     "runtime": runtime,
                     "fallback": None, "boundary_policy_sha256": state["boundary_policy_sha256"],
                     "samples": samples, "scenes": scenes}
    from dubbing_service.app.audiobook.scene_dialogue_evaluation import lines_sha256
    qa_lines = [{**{k: l[k] for k in ("utterance_id", "text", "text_sha256", "role_id", "role_name", "narrator",
                                     "scene_id", "project_id", "chapter_id", "index", "direction")},
                 "source": {**l["source"], "project_id": l["project_id"],
                            "source_sha256": l["source"]["source_revision_sha256"]}} for l in lines]
    manifest = {"schema_version": MANIFEST_SCHEMA, "comparison_id": "scene-dialogue-pilot-v1", "base_dir": str(output),
                "frozen_sha256": state["frozen_sha256"], "review_only": True,
                "production_ready": False, "runtime": runtime,
                "source": {"projects": [{"project_id": s["project_id"], "binding_sha256": s["project_sha256"]}
                                        for s in state["sources"]], "lines_sha256": lines_sha256(qa_lines)},
                "lines": qa_lines,
                "scenes": [{k: s[k] for k in ("scene_id", "project_id", "chapter_id", "theme", "utterance_ids")}
                           for s in state["scenes"]],
                "arms": arms}
    if state.get("fixture"):
        manifest["fixture"] = {"is_fixture": True, "label": "Synthetic CPU test fixture; excluded from production readiness"}
    publish(output / "scene-comparison.json", encoded(manifest))
    publish_review(output, state, manifest)
    return manifest


def publish_review(output, state, manifest):
    """An export for listening, with only mixed scenes exposed by a static host."""
    import html
    previous = verified_previous(state) if state.get("previous_pilot") else None
    cards = []
    for index, scene in enumerate(state["scenes"], 1):
        players = []
        for arm in ARMS:
            mix = manifest["arms"][arm]["scenes"][scene["scene_id"]]
            audio = (output / mix["audio_file"]).read_bytes()
            if sha(audio) != mix["audio_sha256"]:
                raise PilotError("Scene mix hash changed before review publication")
            name = f"audio/scene-{index:02}-{arm}.wav"
            publish(output / "review" / name, audio)
            if arm == "directed" and previous:
                prior, _, prior_manifest, _ = previous
                old_mix = prior_manifest["arms"]["directed"]["scenes"][scene["scene_id"]]
                old_audio = (prior / old_mix["audio_file"]).read_bytes()
                if sha(old_audio) != old_mix["audio_sha256"]:
                    raise PilotError("Prior scene mix changed")
                old_name = f"audio/scene-{index:02}-previous.wav"
                publish(output / "review" / old_name, old_audio)
                players.append(f'<p>Previous scene pilot (v1)</p><audio controls preload="none" src="{old_name}"></audio>')
            label = "Baseline" if arm == "baseline" else ("Stronger scene coaching (v2)" if previous else "Scene-aware delivery")
            players.append(f'<p>{label}</p><audio controls preload="none" src="{name}"></audio>')
        transcript = []
        for line in state["lines"]:
            if line["scene_id"] != scene["scene_id"]:
                continue
            d = line["direction"]
            note = html.escape(d["native_instruction"] or "Neutral / no supported direction")
            if d.get("coaching"):
                note += ' · Objective: ' + html.escape(d["coaching"]["objective"]) + ' · Confidence: ' + html.escape(d["coaching"]["confidence"])
            transcript.append(f'<li><strong>{html.escape(line["role_name"])}</strong> · {html.escape(line["text"])}'
                              f'<br><small>{note} · {html.escape(d["basis"])} · Review pending</small></li>')
        cards.append(f'<section><h2>Scene {index} · {html.escape(scene["theme"].replace("_", " "))}</h2>'
                     + ''.join(players) + '<details><summary>Source lines and proposed directions</summary><ol>'
            + ''.join(transcript) + '</ol></details></section>')
    page = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>ArKendrithyst · Scene-aware listening pilot</title><style>'
            'body{font:18px/1.5 system-ui;background:#15131d;color:#eee;max-width:900px;margin:35px auto;padding:20px}'
            'section{border:1px solid #534c66;border-radius:12px;padding:22px;margin:22px 0}audio{width:100%}'
            'small,p{color:#c9c4d3}li{margin:18px 0}</style></head><body>'
            '<h1>Scene-aware listening pilot</h1><p>20 original lines from chapters 1–3. Existing cloned voices, exact prose, '
            'same experimental Base wrapper and generation settings in both arms. Narration stays neutral. '
            'Measured boundary timing accounts for silence already in the clips.</p><p>These labeled players are for direct inspection. '
            'Blind ballots for baseline versus the current pilot and their separate sealed key are in the evaluation directory. '
            'The previous pilot, when shown, is a labeled historical reference. New coaching is awaiting listening review. No voice approval, '
            'Take selection or default change has been recorded.</p>' + ''.join(cards) + '</body></html>')
    publish(output / "review" / "index.html", page.encode("utf-8"))


def verify_prepared_sources(args, state):
    # A prepared run does not authorize synthesis against changed source state.
    # Check the canonical store and the isolated copy before dispatching GPU work.
    source_audit = []
    for source in state["sources"]:
        live = read_source(args.source_root, source["project_id"])
        if live["segments_sha256"] != source["segments_sha256"]:
            raise PilotError(f"Canonical source changed: {source['project_id']} segments_sha256")
        if source["project_id"] in state["isolated"]:
            copy = read_source(args.output_dir / "projects", source["project_id"])
            for key in ("project_sha256", "segments_sha256"):
                if copy[key] != source[key]:
                    raise PilotError(f"Isolated source changed: {source['project_id']} {key}")
            if value_sha(source_contract(live["project"])) != value_sha(source_contract(copy["project"])):
                raise PilotError(f"Canonical source changed: {source['project_id']} project contract")
        elif live["project_sha256"] != source["project_sha256"]:
            raise PilotError(f"Canonical source changed: {source['project_id']} project_sha256")
        source_audit.append({"project_id": source["project_id"], "prepared_sha256": source["project_sha256"],
                             "current_sha256": live["project_sha256"], "contract_unchanged": True,
                             "ignored_fields": ["voice_lab.revision", "voice_lab.casting.revision",
                                                "voice_lab.casting.notifications"]})
    publish(args.output_dir / "source-checks" / f"{value_sha(source_audit)}.json", encoded(source_audit))
    if boundary_policy_sha() != state["boundary_policy_sha256"]:
        raise PilotError("Boundary policy changed since preparation; choose a new pilot directory")


def generate(args, state, worker_class=None, render_fn=None):
    verify_prepared_sources(args, state)
    if worker_class is None or render_fn is None:
        worker_class, render_fn = default_render_tools()
    from dubbing_service.app.audiobook.performance_provider import component_sha256
    if component_sha256("Speak gently.", 0) != direction_digest("Speak gently.", 0):
        raise PilotError("Local direction digest disagrees with the provider; refusing to generate")
    output = args.output_dir
    if state.get("previous_pilot"):
        reuse_prior_neutral(output, state)
    receipts, pinned = run_generation(output, state, worker_class, render_fn)
    compare_arms(state["lines"], receipts)
    audit = audit_forwarding(state["lines"], receipts)
    publish(output / "forwarding-audit.json", encoded({"frozen_sha256": state["frozen_sha256"], **audit}))
    manifest = build_manifest(output, state, receipts, pinned)
    from dubbing_service.app.audiobook.scene_dialogue_evaluation import evaluate_scene_pilot, summarize
    evaluation = evaluate_scene_pilot(manifest, output_dir=output / "evaluation")
    return {"audit": audit, "manifest": "scene-comparison.json", "manifest_sha256": value_sha(manifest),
            "evaluation": summarize(evaluation), "review_page": "review/index.html"}


def source_contract(project):
    """Exclude only independent notification delivery bookkeeping, never casting decisions.

    A running Studio can retry already-created completion notifications while the
    read-only pilot is prepared. Preserve that live work without treating it as a
    prose, cast or approval change; every other project field remains bound.
    """
    import copy
    result = copy.deepcopy(project)
    lab = result.get("voice_lab", {})
    lab.pop("revision", None)
    casting = lab.get("casting", {})
    casting.pop("revision", None)
    casting.pop("notifications", None)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--projects", nargs="+", required=True, metavar="ID")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate", action="store_true", help="explicit GPU synthesis through the root-owned lane")
    parser.add_argument("--plan-builder", help="module[:function] of the scene dialogue plan owner (default: discovered)")
    parser.add_argument("--selection-from", type=Path, help="reuse exact lines/scenes/cast from a verified prior pilot")
    parser.add_argument("--coaching-file", type=Path, help="captured and source-bound whole-scene model coaching bundle")
    parser.add_argument("--min-directed-dialogue", type=int, default=MIN_DIRECTED_DIALOGUE)
    parser.add_argument("--min-directed-scenes", type=int, default=MIN_DIRECTED_SCENES)
    args = parser.parse_args(argv)
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    if args.selection_from:
        args.selection_from = args.selection_from.resolve(strict=True)
        if args.selection_from == args.output_dir:
            parser.error("Preserve the prior pilot; use a separate output directory")
    if args.coaching_file:
        args.coaching_file = args.coaching_file.resolve(strict=True)
    if len(set(args.projects)) != len(args.projects):
        parser.error("project ids must be distinct")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.output_dir == args.source_root or args.source_root in args.output_dir.parents:
            raise PilotError("Pilot output must be outside the source project store")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if (args.output_dir / "state.json").is_file():
            state = load_state(args.output_dir)
        else:
            state = prepare(args)
        if state["status"] == "ready":
            verify_prepared_sources(args, state)
        summary = {"status": state["status"], "lines": len(state["lines"]), "counts": state.get("counts"),
                   "clone_coverage": {k: state.get("clone_coverage", {}).get(k) for k in ("available_clones", "covered_clones")},
                   "adequacy": state.get("adequacy"), "holds": state["holds"], "output": str(args.output_dir),
                   "human_review": "pending"}
        if args.generate:
            if state["status"] != "ready":
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                return 2
            summary["generation"] = generate(args, state)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if state["status"] == "ready" else 2
    except PilotError as exc:
        print(json.dumps({"status": "held", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    except RuntimeError as exc:  # render()/Worker refuse on busy GPU, port 8143 or ambiguous dispatch
        print(json.dumps({"status": "held", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
