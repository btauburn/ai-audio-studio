"""Two-arm comparative QA and blind review for the scene-aware dialogue pilot.

One manifest describes the same exact source lines rendered in two arms:

``baseline``  the current clones, no native instruction on any line
``directed``  the same clones, runtime, settings, text and boundary policy, with a
              scene-derived native instruction forwarded on directed dialogue only

The harness is a *reader*.  It never synthesizes, never calls a model, never uses
the GPU, never touches a live store and never changes a default.  It hashes every
immutable input before and after analysis, measures the audio that was actually
supplied, proves from the saved worker receipts what was really forwarded, builds
reproducible arm-blind ballots per scene and scores a human review when one is
supplied.

What it will not do:

* accept a sample whose receipt does not prove the instruction state it claims --
  metadata saying "forwarded" is never evidence, the hash-bound worker receipt is;
* accept a baseline or narrator instruction, a neutral placeholder instruction, a
  voice-identity-changing instruction, or any difference between the arms other than
  the instruction itself (clone, reference, prompt, runtime, settings, text, boundary
  policy, fallback);
* invent ASR omissions, intelligibility or speaker similarity -- those need models
  this harness does not run, so they are ``null`` with a reason;
* invent latency or VRAM -- only a receipt bound to the exact audio hash supplies
  them, and an absent value stays unknown;
* invent a review or a quality verdict -- the only gate that can say anything about
  preference is a completed human review of the ballots;
* promote anything -- ``approval`` is always false, no default is changed and the
  samples are review-only.  The gate is a recommendation to the existing Voice Lab
  human actions and never replaces them.

Fixtures (artificial tones from :func:`build_synthetic_fixture`) are labeled in the
manifest and in every output and are excluded from production readiness.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dubbing_service.app.audiobook import director_evaluation as de
from dubbing_service.app.audiobook.director_evaluation import (
    EvaluationError,
    MEASUREMENT_POLICY,
    UNAVAILABLE_MEASURES,
    canonical_json,
    content_sha256,
    measure_audio,
    sha256_hex,
)
from dubbing_service.app.audiobook.performance_director import PerformanceDirectorError, validate_native_instruction
from dubbing_service.app.audiobook.performance_provider import (
    MAX_INSTRUCTION_CHARS,
    POLICY as NATIVE_POLICY,
    PROFILE as DIRECTED_PROFILE,
    RECEIPT_KEYS as DIRECTION_RECEIPT_KEYS,
    DirectionRefused,
    component_sha256,
)

MANIFEST_SCHEMA = "scene-dialogue-comparison-v1"
REPORT_SCHEMA = "scene-dialogue-comparison-report-v1"
BALLOT_SCHEMA = "scene-dialogue-blind-ballot-v1"
BALLOT_SET_SCHEMA = "scene-dialogue-blind-ballot-set-v1"
BALLOT_KEY_SCHEMA = "scene-dialogue-blind-ballot-key-v1"
REVIEW_SCHEMA = "scene-dialogue-human-review-v1"
GATE_SCHEMA = "scene-dialogue-human-gate-v1"
EVALUATOR_VERSION = "scene-dialogue-evaluation-v1"

ARMS: Tuple[str, ...] = ("baseline", "directed")
_LABELS = "AB"

MIN_LINES = 12
MIN_DIALOGUE_ROLES = 2
#: Bounded codec budget for one rendered dialogue line (pilot contract).
MAX_DIALOGUE_CHARS = 250

RATING_DIMENSIONS: Tuple[str, ...] = ("naturalness", "role_fit", "clarity", "scene_fit")
RATING_MIN = 1
RATING_MAX = 5
#: Mean-score margin (1-5 scale) the directed arm must clear against baseline.
GATE_MARGIN = 0.25
#: Edge-silence drift between arms worth surfacing (descriptive only).
EDGE_DRIFT_SECONDS = 0.3

#: Direction origins that may never be presented as a human decision in this pilot.
_HUMAN_CLAIM = re.compile(r"(?i)human|approv|override|confirm")
_NEUTRAL_INSTRUCTION = re.compile(
    r"(?i)^\W*(?:(?:please\s+)?(?:speak|say|read|deliver|talk)(?:\s+it)?\s+)?(?:in\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:completely\s+|totally\s+|very\s+)?(?:neutral(?:ly)?|plain(?:ly)?|normal(?:ly)?|flat(?:ly)?|default|natural(?:ly)?|standard|regular|unmarked|"
    r"no\s+(?:particular\s+)?(?:emotion|expression|direction)|without\s+(?:any\s+)?(?:emotion|expression))"
    r"(?:\s+(?:tone|voice|manner|way|delivery|reading|style|emotion|expression))*\W*$"
)
#: The instruction may shape delivery only; it may never name or change who is speaking.
_IDENTITY_TERMS = re.compile(
    r"(?i)\b(?:accent(?:ed)?|dialect|pitch(?:ed)?|gender(?:ed)?|male|female|timbre|baritone|soprano|falsetto|"
    r"performer|actor|actress|celebrity|(?:young|old|elderly|deep|high|low|child|childlike)\s+(?:man|woman|boy|girl|voice)|"
    r"(?:deeper|higher|lower|different|another|new)\s+(?:voice|pitch|speaker|person)|"
    r"(?:change|alter|switch|swap|replace)\s+(?:the\s+|his\s+|her\s+|their\s+)?(?:voice|speaker|identity))\b"
)


# --------------------------------------------------------------------------- #
# Small validation helpers
# --------------------------------------------------------------------------- #


def _fail(code: str, message: str, **details: Any) -> EvaluationError:
    return EvaluationError(code, message, **details)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _hex64(value: Any, field: str, *, code: str = "manifest_invalid") -> str:
    if not _is_hex64(value):
        raise _fail(code, f"{field} must be a lowercase sha256 hex digest")
    return value


def _mapping(value: Any, field: str, *, code: str = "manifest_invalid") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(code, f"{field} must be an object")
    return value


def _strict_bool(value: Any, field: str, *, code: str = "manifest_invalid") -> bool:
    if type(value) is not bool:
        raise _fail(code, f"{field} must be a boolean")
    return value


def _measure(value: Any, field: str, *, where: Dict[str, Any]) -> Optional[float]:
    """Latency/VRAM: ``None`` stays unknown; anything non-numeric, negative or non-finite is malformed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise _fail("receipt_malformed", f"{field} must be a finite non-negative number or null", **where)
    return float(value)


def _text_sha(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Safe source registry
# --------------------------------------------------------------------------- #


class _SafeSources(de._Sources):
    """Every input must live inside ``base_dir``, without links, climbing or stream tricks."""

    def resolve(self, raw: Any, field: str) -> Path:
        text = de._text(raw, field)
        if self.base_dir is None:
            raise _fail("base_dir_required", f"{field} needs the manifest base_dir")
        if "\x00" in text:
            raise _fail("path_refused", f"{field} contains a NUL byte")
        path = Path(text)
        if ".." in path.parts:
            raise _fail("path_refused", f"{field} may not climb out of base_dir", value=text)
        if ":" in text[len(path.drive):]:
            raise _fail("path_refused", f"{field} may not name an alternate data stream", value=text)
        if not path.is_absolute():
            if path.drive or path.root:
                raise _fail("path_refused", f"{field} is a rooted path without a drive", value=text)
            path = self.base_dir / path
        base = Path(os.path.abspath(self.base_dir))
        candidate = Path(os.path.abspath(path))
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            raise _fail("path_refused", f"{field} is outside base_dir", value=text) from None
        if not relative.parts:
            raise _fail("path_refused", f"{field} names base_dir itself", value=text)
        # A link anywhere between base_dir and the file changes where the bytes really live.
        real_expected = Path(os.path.realpath(base)) / relative
        if os.path.normcase(os.path.realpath(candidate)) != os.path.normcase(str(real_expected)):
            raise _fail("path_refused", f"{field} passes through a link and is refused", value=text)
        if candidate.is_symlink():
            raise _fail("path_refused", f"{field} is a link and is refused", value=text)
        if not candidate.is_file():
            raise _fail("source_missing", f"{field} does not exist", value=text)
        return candidate


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #


def _validate_direction_plan(raw: Any, where: str, narrator: bool) -> Optional[Dict[str, Any]]:
    """Optional per-line plan from the scene planner: recommendation-only, review pending, never human."""
    if raw is None:
        return None
    plan = _mapping(raw, f"{where}.direction")
    if plan.get("recommendation_only") is not True:
        raise _fail("direction_claim_refused", f"{where}.direction must be recommendation_only", where=where)
    if plan.get("human_review", "pending") != "pending" or plan.get("approved") not in (None, False):
        raise _fail("direction_claim_refused", f"{where}.direction claims a human decision; review stays pending",
                    where=where)
    origin = plan.get("origin")
    if origin is not None and (not isinstance(origin, str) or _HUMAN_CLAIM.search(origin)):
        raise _fail("direction_claim_refused", f"{where}.direction.origin may not claim a human decision", where=where)
    instruction = plan.get("native_instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise _fail("manifest_invalid", f"{where}.direction.native_instruction must be a string or null")
    if narrator and instruction is not None:
        raise _fail("narrator_instruction", f"{where}: the narrator never receives a native instruction", where=where)
    return {
        "native_instruction": instruction,
        "origin": origin,
        "evidence_status": plan.get("evidence_status", "unknown"),
        "confidence": plan.get("confidence", "unknown"),
        "recommendation_only": True,
        "human_review": "pending",
    }


def _validate_lines(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) < MIN_LINES:
        raise _fail("manifest_invalid", f"at least {MIN_LINES} identical source lines are required")
    seen: set = set()
    lines: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        where = f"lines[{index}]"
        item = _mapping(item, where)
        utterance_id = de._ident(item.get("utterance_id"), f"{where}.utterance_id")
        if utterance_id in seen:
            raise _fail("manifest_invalid", f"duplicate utterance_id {utterance_id}")
        seen.add(utterance_id)
        narrator = _strict_bool(item.get("narrator"), f"{where}.narrator")
        text = de._text(item.get("text"), f"{where}.text")
        if not narrator and len(text) > MAX_DIALOGUE_CHARS:
            raise _fail("line_too_long", f"{where}: dialogue exceeds the {MAX_DIALOGUE_CHARS}-character codec budget",
                        utterance_id=utterance_id)
        declared = item.get("text_sha256")
        if declared is not None and _hex64(declared, f"{where}.text_sha256") != _text_sha(text):
            raise _fail("text_mismatch", f"{where}.text_sha256 does not match the line text", utterance_id=utterance_id)
        source = _mapping(item.get("source"), f"{where}.source")
        lines.append(
            {
                "utterance_id": utterance_id,
                "role_id": de._ident(item.get("role_id"), f"{where}.role_id"),
                "role_name": de._text(item.get("role_name"), f"{where}.role_name"),
                "narrator": narrator,
                "scene_id": de._ident(item.get("scene_id"), f"{where}.scene_id"),
                "text": text,
                "text_sha256": _text_sha(text),
                "source": {
                    "project_id": de._ident(source.get("project_id"), f"{where}.source.project_id"),
                    "source_sha256": _hex64(source.get("source_sha256"), f"{where}.source.source_sha256"),
                    **{k: source[k] for k in ("segment_id", "chapter_index", "segment_index") if k in source},
                },
                "direction": _validate_direction_plan(item.get("direction"), where, narrator),
            }
        )
    if not any(line["narrator"] for line in lines):
        raise _fail("manifest_invalid", "the line set must include neutral narration")
    dialogue_roles = {line["role_id"] for line in lines if not line["narrator"]}
    if len(dialogue_roles) < MIN_DIALOGUE_ROLES:
        raise _fail("manifest_invalid", f"the line set needs at least {MIN_DIALOGUE_ROLES} dialogue roles")
    narrator_roles = {line["role_id"] for line in lines if line["narrator"]}
    if narrator_roles & dialogue_roles:
        raise _fail("manifest_invalid", "a role cannot be both narrator and dialogue", roles=sorted(narrator_roles & dialogue_roles))
    # Scenes are contiguous runs of one project, in order.
    closed: set = set()
    previous: Optional[str] = None
    scene_project: Dict[str, str] = {}
    for line in lines:
        scene = line["scene_id"]
        if scene != previous:
            if scene in closed:
                raise _fail("scene_not_contiguous", f"scene {scene} reappears after another scene", scene_id=scene)
            if previous is not None:
                closed.add(previous)
            previous = scene
        if scene_project.setdefault(scene, line["source"]["project_id"]) != line["source"]["project_id"]:
            raise _fail("scene_not_contiguous", f"scene {scene} spans more than one project", scene_id=scene)
    return lines


def lines_sha256(lines: Sequence[Mapping[str, Any]]) -> str:
    """Identity of the frozen source line set (text, roles, scenes and source binding)."""
    return content_sha256(
        [
            [l["utterance_id"], _text_sha(l["text"]), l["role_id"], l["scene_id"], bool(l["narrator"]),
             l["source"]["project_id"], l["source"]["source_sha256"]]
            for l in lines
        ]
    )


def _validate_source(raw: Any, lines: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source = _mapping(raw, "source")
    projects_raw = source.get("projects")
    if not isinstance(projects_raw, list) or not projects_raw:
        raise _fail("manifest_invalid", "source.projects must list the frozen source projects")
    projects: Dict[str, str] = {}
    for index, item in enumerate(projects_raw):
        item = _mapping(item, f"source.projects[{index}]")
        project_id = de._ident(item.get("project_id"), f"source.projects[{index}].project_id")
        if project_id in projects:
            raise _fail("manifest_invalid", f"duplicate source project {project_id}")
        projects[project_id] = _hex64(item.get("binding_sha256"), f"source.projects[{index}].binding_sha256")
    for line in lines:
        if line["source"]["project_id"] not in projects:
            raise _fail("source_mismatch", "a line names a project outside source.projects",
                        utterance_id=line["utterance_id"], project_id=line["source"]["project_id"])
    actual = lines_sha256(lines)
    declared = source.get("lines_sha256")
    if declared is not None and _hex64(declared, "source.lines_sha256", code="source_mismatch") != actual:
        raise _fail("source_mismatch", "source.lines_sha256 does not match the frozen line set",
                    declared=declared, actual=actual)
    return {"projects": [{"project_id": p, "binding_sha256": h} for p, h in sorted(projects.items())], "lines_sha256": actual}


_RUNTIME_MISMATCH_CODES = {"settings_sha256": "settings_mismatch", "boundary_policy_sha256": "boundary_policy_mismatch"}


def _validate_runtime(raw: Any, where: str) -> Dict[str, Any]:
    runtime = _mapping(raw, where)
    de._text(runtime.get("name"), f"{where}.name")
    for key in ("runtime_sha256", "checkpoint_sha256", "settings_sha256", "boundary_policy_sha256"):
        _hex64(runtime.get(key), f"{where}.{key}")
    if runtime.get("runtime_profile") != DIRECTED_PROFILE:
        raise _fail("engine_mismatch", f"{where}.runtime_profile must be the pinned {DIRECTED_PROFILE} wrapper",
                    value=runtime.get("runtime_profile"))
    if runtime.get("fallback") is not None:
        raise _fail("fallback_refused", f"{where} declares a fallback; fallback audio is not comparable")
    return {k: runtime[k] for k in sorted(runtime)}


def _check_arm_runtimes(shared: Mapping[str, Any], arms_raw: Mapping[str, Any]) -> None:
    for name, spec in arms_raw.items():
        if not isinstance(spec, Mapping) or spec.get("status", "present") != "present":
            continue
        declared = _validate_runtime(spec.get("runtime"), f"arms.{name}.runtime")
        for key in sorted(set(shared) | set(declared)):
            if canonical_json(shared.get(key)) != canonical_json(declared.get(key)):
                raise _fail(_RUNTIME_MISMATCH_CODES.get(key, "engine_mismatch"),
                            f"the {name} arm differs from the shared pinned runtime in {key}", arm=name, field=key)


# --------------------------------------------------------------------------- #
# Receipts and instruction proof
# --------------------------------------------------------------------------- #


def expected_direction_receipt(
    native_instruction: Optional[str], *, identity_id: str, text: str, seed_salt: int = 0
) -> Dict[str, Any]:
    """The worker ``generation.direction`` receipt a faithful render of this request must carry."""
    forwarded = native_instruction is not None
    base_seed = int(sha256_hex((identity_id + text).encode("utf-8"))[:8], 16)
    return {
        "policy": NATIVE_POLICY,
        "forwarded": forwarded,
        "instruct_sha256": sha256_hex(native_instruction.encode("utf-8")) if forwarded else None,
        "instruct_chars": len(native_instruction) if forwarded else 0,
        "conditioning_mode": "icl",
        "seed_salt": seed_salt,
        "seed_effective": base_seed if seed_salt == 0 else (base_seed + seed_salt) % 4294967296,
        "direction_sha256": component_sha256(native_instruction, seed_salt),
    }


def _check_instruction_text(instruction: Any, where: Dict[str, Any], *, arm: str, narrator: bool) -> Optional[str]:
    """Validate the claimed instruction itself; returns the text or ``None``."""
    if instruction is None:
        return None
    if arm == "baseline":
        raise _fail("baseline_instruction", "the baseline arm may not carry a native instruction", **where)
    if narrator:
        raise _fail("narrator_instruction", "the narrator never receives a native instruction", **where)
    if not isinstance(instruction, str) or not instruction.strip():
        raise _fail("instruction_refused", "a native instruction must be a non-blank string or null", **where)
    if instruction != instruction.strip():
        raise _fail("instruction_refused", "a native instruction may not carry surrounding whitespace", **where)
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise _fail("instruction_refused", f"a native instruction exceeds {MAX_INSTRUCTION_CHARS} characters", **where)
    try:
        validate_native_instruction(instruction)
    except (PerformanceDirectorError, DirectionRefused) as exc:
        raise _fail("instruction_refused", f"the native instruction fails the director gate: {exc}", **where) from exc
    if _NEUTRAL_INSTRUCTION.match(instruction):
        raise _fail("neutral_instruction", "a neutral placeholder is not a direction; leave the line undirected", **where)
    if _IDENTITY_TERMS.search(instruction):
        raise _fail("instruction_identity_words",
                    "a native instruction may shape delivery only, not accent, pitch, gender, performer or voice identity",
                    **where)
    return instruction


def _load_receipt(
    sources: _SafeSources, raw: Any, where: str, ctx: Mapping[str, Any]
) -> Dict[str, Any]:
    """Load the hash-bound worker receipt and prove the instruction state from it."""
    ids = {"arm": ctx["arm"], "utterance_id": ctx["utterance_id"]}
    spec = _mapping(raw, f"{where}.receipt", code="receipt_missing") if raw is not None else None
    if spec is None:
        raise _fail("receipt_missing", f"{where} has no receipt; forwarding cannot be proven", **ids)
    path, data, digest = sources.read(spec.get("path"), f"{where}.receipt.path", "receipt")
    declared = spec.get("sha256")
    if declared is not None and _hex64(declared, f"{where}.receipt.sha256", code="receipt_mismatch") != digest:
        raise _fail("receipt_mismatch", f"{where} receipt bytes do not match the declared sha256", **ids)
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail("receipt_malformed", f"{where} receipt is not valid JSON", **ids) from exc
    if not isinstance(body, dict):
        raise _fail("receipt_malformed", f"{where} receipt must be a JSON object", **ids)
    for field, expected in (
        ("utterance_id", ctx["utterance_id"]), ("audio_sha256", ctx["audio_sha256"]),
        ("text_sha256", ctx["text_sha256"]), ("runtime_sha256", ctx["runtime_sha256"]),
    ):
        if body.get(field) != expected:
            raise _fail("receipt_mismatch", f"{where} receipt {field} does not match the sample", field=field, **ids)
    request_sha = body.get("request_sha256")
    if request_sha is not None and not _is_hex64(request_sha):
        raise _fail("receipt_malformed", f"{where} receipt request_sha256 is not a sha256", **ids)
    worker = body.get("worker_receipt")
    if not isinstance(worker, dict):
        raise _fail("receipt_malformed", f"{where} receipt carries no worker_receipt", **ids)
    identity = worker.get("worker_runtime_identity")
    if not isinstance(identity, dict) or identity.get("sha256") != ctx["runtime_sha256"]:
        raise _fail("receipt_mismatch", f"{where} worker runtime identity differs from the pinned runtime", **ids)
    if worker.get("fallback") is not None:
        raise _fail("fallback_refused", f"{where} worker reports fallback audio", **ids)
    profile = (worker.get("execution_profile") or {}).get("runtime_profile") if isinstance(worker.get("execution_profile"), dict) else None
    if profile != DIRECTED_PROFILE:
        raise _fail("engine_mismatch", f"{where} was not rendered by the pinned {DIRECTED_PROFILE} wrapper", profile=profile, **ids)
    if worker.get("identity_id") != ctx["voice_id"] or worker.get("prompt_sha256") != ctx["prompt_id"]:
        raise _fail("identity_mismatch", f"{where} worker rendered another voice or prompt than the sample claims", **ids)
    reference = body.get("reference_audio_sha256")
    if reference is not None and reference != ctx["reference_id"]:
        raise _fail("identity_mismatch", f"{where} receipt reference differs from the sample", **ids)
    if body.get("native_instruction") != ctx["native_instruction"]:
        raise _fail("forwarding_mismatch", f"{where} receipt instruction differs from the sample metadata", **ids)
    if ctx["sample"].get("request_sha256") is not None and request_sha != ctx["sample"]["request_sha256"]:
        raise _fail("receipt_mismatch", f"{where} request hash differs from the saved dispatch", **ids)
    settings = worker.get("generation_settings")
    if ctx["sample"].get("generation_settings") is not None and settings != ctx["sample"]["generation_settings"]:
        raise _fail("settings_mismatch", f"{where} generation settings differ from the worker receipt", **ids)

    generation = worker.get("generation")
    direction = generation.get("direction") if isinstance(generation, dict) else None
    parent_path = direction is None and ctx["native_instruction"] is None
    if parent_path:
        # The x1 wrapper deliberately uses its unchanged parent path when no
        # controls are requested. That path has no native direction receipt.
        # Verify its empty controls and actual seed; never invent a receipt.
        wanted_seed = int(sha256_hex((ctx["voice_id"] + ctx["text"]).encode())[:8], 16)
        if worker.get("performance") != {} or not isinstance(settings, dict) or settings.get("seed") != wanted_seed:
            raise _fail("receipt_malformed", f"{where} does not prove the unchanged parent path", **ids)
        salt = 0
    elif not isinstance(direction, dict) or set(direction) != DIRECTION_RECEIPT_KEYS:
        raise _fail("receipt_malformed", f"{where} direction receipt is missing or has the wrong keys; "
                    "claimed forwarding is not proven", **ids)
    else:
        salt = direction.get("seed_salt")
    if type(salt) is not int or not 0 <= salt <= 2147483647:
        raise _fail("receipt_malformed", f"{where} direction receipt seed_salt is not a valid integer", **ids)
    try:
        expected = expected_direction_receipt(
            ctx["native_instruction"], identity_id=ctx["voice_id"], text=ctx["text"], seed_salt=salt
        )
    except DirectionRefused as exc:
        raise _fail("instruction_refused", str(exc), **ids) from exc
    for key in (() if parent_path else sorted(DIRECTION_RECEIPT_KEYS)):
        same = direction[key] is expected[key] if isinstance(expected[key], bool) else (
            type(direction[key]) is type(expected[key]) and direction[key] == expected[key])
        if not same:
            raise _fail("forwarding_mismatch",
                        f"{where} direction receipt {key} does not match the claimed instruction state",
                        key=key, **ids)
    latency = _measure(body.get("latency_seconds"), f"{where}.receipt.latency_seconds", where=ids)
    vram = _measure(body.get("peak_vram_mb"), f"{where}.receipt.peak_vram_mb", where=ids)
    for field, value in (("latency_seconds", latency), ("peak_vram_mb", vram)):
        claimed = ctx["sample"].get(field)
        if claimed is not None and (value is None or de._number(claimed) != value):
            raise _fail("receipt_mismatch", f"{where}.{field} disagrees with the receipt", **ids)
    return {
        "status": "bound",
        "path": str(path),
        "sha256": digest,
        "request_sha256": request_sha,
        "latency_seconds": de._round(latency),
        "peak_vram_mb": de._round(vram),
        "vram_measurement": body.get("vram_measurement") if isinstance(body.get("vram_measurement"), str) else None,
        "synthetic_fixture": bool(body.get("synthetic_fixture")),
        "seed_salt": salt,
        "generation_settings": settings,
        "direction_receipt": direction,
        "direction_proof": "unchanged_parent_path" if parent_path else "native_worker_receipt",
    }


# --------------------------------------------------------------------------- #
# Arm analysis
# --------------------------------------------------------------------------- #


def _analyze_arm(
    arm: str, spec: Any, lines: List[Dict[str, Any]], runtime: Mapping[str, Any], sources: _SafeSources, fixture: bool
) -> Dict[str, Any]:
    if spec is None or spec == {}:
        return {"arm": arm, "status": "missing", "reason": "arm not supplied", "samples": []}
    spec = _mapping(spec, f"arms.{arm}")
    status = spec.get("status", "present")
    if status == "missing":
        if spec.get("samples"):
            raise _fail("manifest_invalid", f"arms.{arm} is marked missing but lists samples")
        return {"arm": arm, "status": "missing", "reason": str(spec.get("reason") or "arm marked missing"), "samples": []}
    if status != "present":
        raise _fail("manifest_invalid", f"arms.{arm}.status must be present or missing")
    raw_samples = spec.get("samples")
    if not isinstance(raw_samples, Mapping) or not raw_samples:
        raise _fail("manifest_invalid", f"arms.{arm}.samples must be an object keyed by utterance_id")
    by_id = {line["utterance_id"]: line for line in lines}
    unknown = sorted(set(raw_samples) - set(by_id))
    if unknown:
        raise _fail("manifest_invalid", f"arms.{arm}.samples names lines outside the source set", utterance_ids=unknown)
    missing = [line["utterance_id"] for line in lines if line["utterance_id"] not in raw_samples]
    if missing:
        raise _fail("coverage_incomplete", f"arms.{arm} lacks {len(missing)} of {len(lines)} source lines; "
                    "a partial arm cannot be compared", arm=arm, missing_utterance_ids=missing)

    samples: List[Dict[str, Any]] = []
    role_voice: Dict[str, Tuple[Any, Any, Any]] = {}
    for line in lines:
        utterance_id = line["utterance_id"]
        where = f"arms.{arm}.samples.{utterance_id}"
        ids = {"arm": arm, "utterance_id": utterance_id}
        raw = _mapping(raw_samples[utterance_id], where)
        if raw.get("utterance_id") not in (None, utterance_id):
            raise _fail("manifest_invalid", f"{where}.utterance_id disagrees with its key", **ids)
        if raw.get("role_id") not in (None, line["role_id"]):
            raise _fail("manifest_invalid", f"{where}.role_id differs from the line's role", **ids)
        if raw.get("text") != line["text"]:
            raise _fail("text_mismatch", f"{where} text differs from the frozen source line", **ids)
        if raw.get("text_sha256") != line["text_sha256"]:
            raise _fail("text_mismatch", f"{where}.text_sha256 differs from the source line", **ids)
        if raw.get("runtime_sha256") != runtime["runtime_sha256"]:
            raise _fail("engine_mismatch", f"{where}.runtime_sha256 differs from the pinned runtime", **ids)
        for key in ("settings_sha256", "boundary_policy_sha256"):
            if raw.get(key) is not None and raw.get(key) != runtime[key]:
                raise _fail(_RUNTIME_MISMATCH_CODES[key], f"{where}.{key} differs from the pinned runtime", **ids)
        if raw.get("fallback") is not None:
            raise _fail("fallback_refused", f"{where} reports fallback audio", **ids)
        voice = (
            de._text(raw.get("voice_id"), f"{where}.voice_id"),
            de._text(raw.get("reference_id"), f"{where}.reference_id"),
            de._text(raw.get("prompt_id"), f"{where}.prompt_id"),
        )
        if role_voice.setdefault(line["role_id"], voice) != voice:
            raise _fail("identity_mismatch", f"{where}: role {line['role_id']} changes its clone between lines", **ids)

        instruction_raw = _mapping(raw.get("instruction"), f"{where}.instruction")
        if set(instruction_raw) - {"native_instruction", "forwarded", "direction_sha256"}:
            raise _fail("manifest_invalid", f"{where}.instruction has unknown fields", **ids)
        claimed_forwarded = _strict_bool(instruction_raw.get("forwarded"), f"{where}.instruction.forwarded")
        instruction = _check_instruction_text(
            instruction_raw.get("native_instruction"), ids, arm=arm, narrator=line["narrator"]
        )
        if claimed_forwarded is not (instruction is not None):
            raise _fail("forwarding_mismatch", f"{where}: forwarded must be true exactly when an instruction exists", **ids)
        plan = line["direction"]
        if arm == "directed" and plan is not None and plan["native_instruction"] != instruction:
            raise _fail("direction_plan_mismatch", f"{where} does not carry the planned instruction", **ids)

        declared_audio = _hex64(raw.get("audio_sha256"), f"{where}.audio_sha256")
        path, data, digest = sources.read(raw.get("audio_file"), f"{where}.audio_file", "audio")
        if digest != declared_audio:
            raise _fail("audio_hash_mismatch", f"{where} audio bytes do not match audio_sha256", declared=declared_audio,
                        actual=digest, **ids)
        try:
            measures = measure_audio(data, text=line["text"])
        except EvaluationError as exc:
            raise _fail(exc.code, f"{where}: {exc.message}", **ids) from exc
        receipt = _load_receipt(
            sources, raw.get("receipt"), where,
            {**ids, "audio_sha256": digest, "text_sha256": line["text_sha256"], "text": line["text"],
             "runtime_sha256": runtime["runtime_sha256"], "voice_id": voice[0], "reference_id": voice[1],
             "prompt_id": voice[2], "native_instruction": instruction, "sample": raw},
        )
        if receipt["synthetic_fixture"] and not fixture:
            raise _fail("receipt_mismatch", f"{where} receipt is a synthetic fixture inside a production manifest", **ids)
        claimed_direction = instruction_raw.get("direction_sha256")
        if claimed_direction is not None and claimed_direction != receipt["direction_receipt"]["direction_sha256"]:
            raise _fail("forwarding_mismatch", f"{where}.instruction.direction_sha256 disagrees with the receipt", **ids)
        samples.append(
            {
                "utterance_id": utterance_id,
                "role_id": line["role_id"],
                "narrator": line["narrator"],
                "scene_id": line["scene_id"],
                "audio_file": str(raw.get("audio_file")),
                "resolved_audio_path": str(path.resolve()),
                "audio_sha256": digest,
                "audio_bytes": len(data),
                "text_sha256": line["text_sha256"],
                "voice": {"voice_id": voice[0], "reference_id": voice[1], "prompt_id": voice[2]},
                "instruction": {
                    "native_instruction": instruction,
                    "forwarded": instruction is not None,
                    "instruct_sha256": (receipt["direction_receipt"] or {}).get("instruct_sha256"),
                    "direction_sha256": (receipt["direction_receipt"] or {}).get("direction_sha256"),
                    "seed_salt": receipt["seed_salt"],
                },
                "measures": measures,
                "receipt": {k: v for k, v in receipt.items() if k != "direction_receipt"},
                "asr": copy.deepcopy(UNAVAILABLE_MEASURES["asr"]),
                "acoustic_voice_embedding": copy.deepcopy(UNAVAILABLE_MEASURES["acoustic_voice_embedding"]),
                "human_measures": {"status": "pending"},
            }
        )
    if arm == "directed" and not any(s["instruction"]["forwarded"] for s in samples):
        raise _fail("direction_mismatch", "the directed arm forwards no instruction on any dialogue line; "
                    "it would be a relabeled baseline", arm=arm)
    result = {
        "arm": arm,
        "status": "complete",
        "reason": None,
        "runtime": dict(runtime),
        "samples": samples,
    }
    result["summary"] = _summarize_arm(samples)
    return result


def _summarize_arm(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    measures = [s["measures"] for s in samples]
    peaks = [m["peak"] for m in measures if m.get("peak") is not None]
    silence = [m["silence"]["silent_ratio"] for m in measures if m.get("silence")]
    severities = {"blocker": 0, "review": 0, "info": 0}
    for m in measures:
        for item in m["flags"]:
            severities[item["severity"]] += 1
    latencies = [s["receipt"]["latency_seconds"] for s in samples if s["receipt"]["latency_seconds"] is not None]
    vram = [s["receipt"]["peak_vram_mb"] for s in samples if s["receipt"]["peak_vram_mb"] is not None]
    return {
        "sample_count": len(samples),
        "narrator_samples": sum(1 for s in samples if s["narrator"]),
        "dialogue_samples": sum(1 for s in samples if not s["narrator"]),
        "forwarded_instructions": sum(1 for s in samples if s["instruction"]["forwarded"]),
        "narrator_instructions": sum(1 for s in samples if s["narrator"] and s["instruction"]["forwarded"]),
        "total_duration_seconds": de._round(sum(m["duration_seconds"] or 0.0 for m in measures)),
        "mean_silent_ratio": de._round(sum(silence) / len(silence)) if silence else None,
        "max_peak": max(peaks) if peaks else None,
        "clipped_samples": sum(m["clipping"]["clipped_samples"] for m in measures),
        "samples_with_likely_clipping": sum(1 for m in measures if m["clipping"]["indicator"] == "likely"),
        "samples_with_tail_active": sum(1 for m in measures if m["tail"]["indicator"] == "active_at_end"),
        "non_finite_samples": sum(m["non_finite_samples"] for m in measures),
        "flag_counts": severities,
        "latency_seconds": {
            "samples_with_receipt": len(latencies),
            "total": de._round(sum(latencies)) if latencies else None,
            "mean": de._round(sum(latencies) / len(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
            "complete": len(latencies) == len(samples),
        },
        "peak_vram_mb": {
            "samples_with_receipt": len(vram),
            "max": max(vram) if vram else None,
            "complete": len(vram) == len(samples),
        },
    }


# --------------------------------------------------------------------------- #
# Cross-arm integrity and descriptive comparison
# --------------------------------------------------------------------------- #


def _check_cross_arm(arms: Mapping[str, Dict[str, Any]], lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not all(arms[a]["status"] == "complete" for a in ARMS):
        return findings
    base = {s["utterance_id"]: s for s in arms["baseline"]["samples"]}
    directed = {s["utterance_id"]: s for s in arms["directed"]["samples"]}
    for line in lines:
        u = line["utterance_id"]
        left, right = base[u], directed[u]
        if left["voice"] != right["voice"]:
            raise _fail("identity_mismatch", f"the arms use different clone identity for {u}", utterance_id=u)
        if left["receipt"]["seed_salt"] != right["receipt"]["seed_salt"]:
            raise _fail("settings_mismatch", f"the arms use different seed settings for {u}", utterance_id=u)
        if left["receipt"]["generation_settings"] != right["receipt"]["generation_settings"]:
            raise _fail("settings_mismatch", f"the arms use different generation settings for {u}", utterance_id=u)
        if left["instruction"]["forwarded"]:
            raise _fail("baseline_instruction", f"baseline forwarded an instruction for {u}", utterance_id=u)
        if line["narrator"] and (right["instruction"]["forwarded"] or left["instruction"]["forwarded"]):
            raise _fail("narrator_instruction", f"narrator line {u} carries an instruction", utterance_id=u)
    if [base[l["utterance_id"]]["audio_sha256"] for l in lines] == [directed[l["utterance_id"]]["audio_sha256"] for l in lines]:
        raise _fail("arm_copied", "the baseline and directed arms have byte-identical audio for every line; "
                    "an arm may not be copied or relabeled", arms=list(ARMS))
    unchanged = [
        l["utterance_id"] for l in lines
        if directed[l["utterance_id"]]["instruction"]["forwarded"]
        and directed[l["utterance_id"]]["audio_sha256"] == base[l["utterance_id"]]["audio_sha256"]
    ]
    if unchanged:
        findings.append({"code": "directed_audio_unchanged", "utterance_ids": unchanged,
                         "detail": "A forwarded instruction produced byte-identical audio on these lines."})
    held = [l["utterance_id"] for l in lines if not l["narrator"] and not directed[l["utterance_id"]]["instruction"]["forwarded"]]
    if held:
        findings.append({"code": "dialogue_left_undirected", "utterance_ids": held,
                         "detail": "No grounded direction was forwarded on these dialogue lines; they are control lines."})
    changed_narrators = [l["utterance_id"] for l in lines if l["narrator"] and
                         directed[l["utterance_id"]]["audio_sha256"] != base[l["utterance_id"]]["audio_sha256"]]
    if changed_narrators:
        findings.append({"code": "narrator_audio_differs", "utterance_ids": changed_narrators,
                         "detail": "Narrator received no instruction in either arm; byte differences come from "
                                   "runtime nondeterminism and are listened for, not assumed."})
    return findings


def _compare(arms: Mapping[str, Dict[str, Any]], lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not all(arms[a]["status"] == "complete" for a in ARMS):
        return None
    base = {s["utterance_id"]: s for s in arms["baseline"]["samples"]}
    directed = {s["utterance_id"]: s for s in arms["directed"]["samples"]}
    rows = []
    for line in lines:
        u = line["utterance_id"]
        b, d = base[u]["measures"], directed[u]["measures"]

        def delta(x: Optional[float], y: Optional[float]) -> Optional[float]:
            return None if x is None or y is None else de._round(x - y)

        lead = delta((d.get("silence") or {}).get("leading_seconds"), (b.get("silence") or {}).get("leading_seconds"))
        trail = delta((d.get("silence") or {}).get("trailing_seconds"), (b.get("silence") or {}).get("trailing_seconds"))
        rows.append({
            "utterance_id": u,
            "narrator": line["narrator"],
            "forwarded": directed[u]["instruction"]["forwarded"],
            "audio_changed": directed[u]["audio_sha256"] != base[u]["audio_sha256"],
            "duration_delta_seconds": delta(d.get("duration_seconds"), b.get("duration_seconds")),
            "silent_ratio_delta": delta((d.get("silence") or {}).get("silent_ratio"), (b.get("silence") or {}).get("silent_ratio")),
            "peak_delta": delta(d.get("peak"), b.get("peak")),
            "leading_silence_delta_seconds": lead,
            "trailing_silence_delta_seconds": trail,
            "edge_silence_drift": any(v is not None and abs(v) > EDGE_DRIFT_SECONDS for v in (lead, trail)),
        })
    a, b_summary = arms["directed"]["summary"], arms["baseline"]["summary"]
    both_latency = a["latency_seconds"]["complete"] and b_summary["latency_seconds"]["complete"]
    both_vram = a["peak_vram_mb"]["complete"] and b_summary["peak_vram_mb"]["complete"]

    def summary_delta(group: str, key: str) -> Optional[float]:
        x, y = a[group][key], b_summary[group][key]
        return None if x is None or y is None else de._round(x - y)

    return {
        "interpretation": "Descriptive acoustic differences only; not a quality or preference claim.",
        "directed_vs_baseline": {
            "total_duration_seconds_delta": None if a["total_duration_seconds"] is None else de._round(
                a["total_duration_seconds"] - b_summary["total_duration_seconds"]),
            "clipped_samples_delta": a["clipped_samples"] - b_summary["clipped_samples"],
            "blocker_flags_delta": a["flag_counts"]["blocker"] - b_summary["flag_counts"]["blocker"],
            "review_flags_delta": a["flag_counts"]["review"] - b_summary["flag_counts"]["review"],
            "mean_latency_seconds_delta": summary_delta("latency_seconds", "mean") if both_latency else None,
            "max_peak_vram_mb_delta": summary_delta("peak_vram_mb", "max") if both_vram else None,
            "changed_audio_dialogue_lines": sum(1 for r in rows if r["audio_changed"] and not r["narrator"]),
            "changed_audio_narrator_lines": sum(1 for r in rows if r["audio_changed"] and r["narrator"]),
            "edge_silence_drift_utterance_ids": [r["utterance_id"] for r in rows if r["edge_silence_drift"]],
            "per_utterance": rows,
        },
    }


# --------------------------------------------------------------------------- #
# Per-scene blind ballots
# --------------------------------------------------------------------------- #


def _scenes(lines: Sequence[Mapping[str, Any]]) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    grouped: List[Tuple[str, List[Mapping[str, Any]]]] = []
    for line in lines:
        if not grouped or grouped[-1][0] != line["scene_id"]:
            grouped.append((line["scene_id"], []))
        grouped[-1][1].append(line)
    return grouped


def build_blind_ballots(
    comparison_id: str, lines: List[Dict[str, Any]], arms: Mapping[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(ballot_set, key)``.  Ballots name no arm; the sealed key maps labels back.

    Label order is a pure function of the comparison identity and the frozen lines
    (balanced: half the lines list the baseline first), so identical inputs always
    give the same ballots.  Audio identity is bound through ``audio_set_sha256`` so a
    review of one render cannot be applied to another.
    """
    base = {s["utterance_id"]: s for s in arms["baseline"]["samples"]}
    directed = {s["utterance_id"]: s for s in arms["directed"]["samples"]}
    salt = content_sha256({"comparison_id": comparison_id, "lines": lines_sha256(lines)})[:16]
    ranked = sorted((l["utterance_id"] for l in lines), key=lambda u: sha256_hex(f"{salt}|{u}".encode()))
    baseline_first = {u for rank, u in enumerate(ranked) if rank % 2 == 0}
    audio_set = content_sha256([[l["utterance_id"], base[l["utterance_id"]]["audio_sha256"],
                                 directed[l["utterance_id"]]["audio_sha256"]] for l in lines])
    ballots: List[Dict[str, Any]] = []
    key_scenes: Dict[str, Any] = {}
    for scene_index, (scene_id, scene_lines) in enumerate(_scenes(lines), start=1):
        items, key_items = [], {}
        for number, line in enumerate(scene_lines, start=1):
            u = line["utterance_id"]
            item_id = f"{scene_id}-item-{number:03d}"
            order = ("baseline", "directed") if u in baseline_first else ("directed", "baseline")
            options, key_options = [], {}
            for label, arm in zip(_LABELS, order):
                sample = (base if arm == "baseline" else directed)[u]
                options.append({"label": label, "sample_ref": f"{item_id}-{label}"})
                key_options[label] = {"arm": arm, "audio_path": sample["resolved_audio_path"],
                                      "audio_sha256": sample["audio_sha256"]}
            items.append({"item_id": item_id, "utterance_id": u, "role_name": line["role_name"],
                          "narrator": line["narrator"], "text": line["text"], "options": options})
            key_items[item_id] = {"utterance_id": u, "narrator": line["narrator"], "options": key_options}
        body = {
            "schema_version": BALLOT_SCHEMA,
            "comparison_id": comparison_id,
            "scene_id": scene_id,
            "scene_index": scene_index,
            "audio_set_sha256": audio_set,
            "status": "ready",
            "rating_scale": {"min": RATING_MIN, "max": RATING_MAX, "dimensions": list(RATING_DIMENSIONS)},
            "instructions": (
                "Read the scene lines in order, then listen to both options for every line without opening the key. "
                "Rate naturalness, role fit, clarity and scene fit (1-5) for each label, optionally name a preferred "
                "label, and record any defects heard. Narration is meant to stay neutral. Your listening is the "
                "final decision; nothing here approves a voice or changes a default."
            ),
            "items": items,
        }
        ballot = {**body, "ballot_id": content_sha256(body)[:32]}
        ballots.append(ballot)
        key_scenes[scene_id] = {"ballot_id": ballot["ballot_id"], "items": key_items}
    set_body = {
        "schema_version": BALLOT_SET_SCHEMA,
        "comparison_id": comparison_id,
        "audio_set_sha256": audio_set,
        "status": "ready",
        "ballots": [{"scene_id": b["scene_id"], "ballot_id": b["ballot_id"], "items": len(b["items"])} for b in ballots],
    }
    ballot_set_id = content_sha256(set_body)[:32]
    ballot_set = {**set_body, "ballot_set_id": ballot_set_id, "scenes": ballots}
    key = {"schema_version": BALLOT_KEY_SCHEMA, "ballot_set_id": ballot_set_id, "comparison_id": comparison_id,
           "sealed": "Do not show reviewers.", "scenes": key_scenes}
    return ballot_set, key


def score_human_review(
    ballot_set: Mapping[str, Any], key: Mapping[str, Any], review: Any
) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Validate a review against the ballots.  Returns ``(scores, problems, coverage)``.

    Scores are ``None`` while the review is invalid or does not cover every scene.
    """
    coverage = {"scenes_total": len(ballot_set["scenes"]), "scenes_reviewed": 0, "complete": False}
    if not isinstance(review, Mapping):
        return None, ["review must be an object"], coverage
    problems: List[str] = []
    if review.get("ballot_set_id") != ballot_set["ballot_set_id"]:
        return None, ["review ballot_set_id does not match these ballots"], coverage
    reviewer = review.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        problems.append("reviewer_id is required")
    if review.get("reviewer_kind") != "human":
        problems.append("reviewer_kind must be 'human'; model or automated reviews cannot satisfy the gate")
    scenes_raw = review.get("scenes")
    if not isinstance(scenes_raw, list):
        return None, problems + ["scenes must be a list"], coverage
    by_ballot = {b["ballot_id"]: b for b in ballot_set["scenes"]}
    answered: Dict[str, Mapping[str, Any]] = {}
    for entry in scenes_raw:
        ballot_id = entry.get("ballot_id") if isinstance(entry, Mapping) else None
        if ballot_id not in by_ballot:
            problems.append(f"unknown ballot {ballot_id!r}")
        elif ballot_id in answered:
            problems.append(f"duplicate review of ballot {ballot_id}")
        else:
            answered[ballot_id] = entry
    totals: Dict[str, Dict[str, Dict[str, List[int]]]] = {
        arm: {seg: {d: [] for d in RATING_DIMENSIONS} for seg in ("dialogue", "narrator")} for arm in ARMS}
    wins = {arm: {"dialogue": 0, "narrator": 0} for arm in ARMS}
    defects = {arm: 0 for arm in ARMS}
    reviewed = 0
    for ballot_id, entry in answered.items():
        ballot = by_ballot[ballot_id]
        items = {i["item_id"]: i for i in ballot["items"]}
        responses = entry.get("responses")
        if not isinstance(responses, list):
            problems.append(f"{ballot['scene_id']}: responses must be a list")
            continue
        seen: Dict[str, Mapping[str, Any]] = {}
        for response in responses:
            item_id = response.get("item_id") if isinstance(response, Mapping) else None
            if item_id not in items:
                problems.append(f"{ballot['scene_id']}: unknown item {item_id!r}")
            elif item_id in seen:
                problems.append(f"{ballot['scene_id']}: duplicate response for {item_id}")
            else:
                seen[item_id] = response
        missing = [i for i in items if i not in seen]
        for item_id in missing:
            problems.append(f"{ballot['scene_id']}: missing response for {item_id}")
        scene_ok = not missing
        staged = []
        for item_id, response in seen.items():
            labels = [o["label"] for o in items[item_id]["options"]]
            ratings = response.get("ratings")
            if not isinstance(ratings, Mapping) or set(ratings) != set(labels):
                problems.append(f"{item_id}: ratings must cover exactly labels {labels}")
                scene_ok = False
                continue
            valid = True
            for label in labels:
                row = ratings[label]
                for dimension in RATING_DIMENSIONS:
                    value = row.get(dimension) if isinstance(row, Mapping) else None
                    if isinstance(value, bool) or not isinstance(value, int) or not RATING_MIN <= value <= RATING_MAX:
                        problems.append(f"{item_id}/{label}: {dimension} must be an integer {RATING_MIN}-{RATING_MAX}")
                        valid = False
            preferred = response.get("preferred_label")
            if preferred is not None and preferred not in labels:
                problems.append(f"{item_id}: preferred_label must be one of {labels} or null")
                valid = False
            listed = response.get("defects") or {}
            if not isinstance(listed, Mapping) or not set(listed) <= set(labels) or any(
                    not isinstance(v, list) for v in listed.values()):
                problems.append(f"{item_id}: defects must be lists keyed by ballot labels")
                valid = False
            if not valid:
                scene_ok = False
                continue
            staged.append((item_id, ratings, preferred, listed, labels))
        if not scene_ok:
            continue
        reviewed += 1
        for item_id, ratings, preferred, listed, labels in staged:
            options = key["scenes"][ballot["scene_id"]]["items"][item_id]
            segment = "narrator" if options["narrator"] else "dialogue"
            for label in labels:
                arm = options["options"][label]["arm"]
                for dimension in RATING_DIMENSIONS:
                    totals[arm][segment][dimension].append(ratings[label][dimension])
                defects[arm] += len(listed.get(label) or [])
            if preferred is not None:
                wins[options["options"][preferred]["arm"]][segment] += 1
    coverage.update(scenes_reviewed=reviewed, complete=reviewed == coverage["scenes_total"] and not problems)
    if problems:
        return None, problems, coverage
    if not coverage["complete"]:
        return None, [], coverage
    scores: Dict[str, Any] = {}
    for arm in ARMS:
        scores[arm] = {}
        for segment in ("dialogue", "narrator"):
            per_dimension = {d: de._round(sum(v) / len(v), 4) for d, v in totals[arm][segment].items() if v}
            scores[arm][segment] = {
                "dimension_means": per_dimension,
                "overall_mean": de._round(sum(per_dimension.values()) / len(per_dimension), 4) if per_dimension else None,
                "preferred_count": wins[arm][segment],
                "rated_items": len(totals[arm][segment]["naturalness"]),
            }
        scores[arm]["defects_listed"] = defects[arm]
    return {"reviewer_id": reviewer, "reviewer_kind": "human", "arms": scores}, [], coverage


def build_human_gate(
    *, fixture: bool, ballot_set: Mapping[str, Any], key: Mapping[str, Any], arms: Mapping[str, Dict[str, Any]],
    review: Any,
) -> Dict[str, Any]:
    """Gate status: fixture_excluded / held / pending / supports_experimental_review / no_clear_change / does_not_support.

    A recommendation to the existing Voice Lab human actions.  It never approves a
    voice, never changes a default and never starts synthesis.
    """
    gate: Dict[str, Any] = {
        "schema_version": GATE_SCHEMA,
        "ballot_set_id": ballot_set["ballot_set_id"],
        "margin": GATE_MARGIN,
        "approval": False,
        "default_changed": False,
        "promotion": "none",
        "note": "Recommendation only. Human listening is final; voice approval stays with Voice Lab casting feedback.",
        "reasons": [],
        "scores": None,
        "verdicts": None,
        "coverage": None,
    }
    if fixture:
        gate.update(status="fixture_excluded", reasons=["Synthetic fixture output is excluded from readiness claims."])
        return gate
    absent = [a for a in ARMS if arms[a]["status"] != "complete"]
    if absent:
        gate.update(status="held", reasons=[f"arm {a} is {arms[a]['status']}: {arms[a]['reason']}" for a in absent])
        return gate
    blockers = [(a, arms[a]["summary"]["flag_counts"]["blocker"]) for a in ARMS if arms[a]["summary"]["flag_counts"]["blocker"]]
    if blockers:
        gate.update(status="held", reasons=[f"technical blockers in {a} ({n})" for a, n in blockers])
        return gate
    if review is None:
        gate.update(status="pending", reasons=["No human review has been recorded for these ballots."])
        return gate
    scores, problems, coverage = score_human_review(ballot_set, key, review)
    gate["coverage"] = coverage
    if problems:
        gate.update(status="held", reasons=["review invalid"] + problems[:50])
        return gate
    if scores is None:
        gate.update(status="pending", reasons=[
            f"{coverage['scenes_reviewed']} of {coverage['scenes_total']} scene ballots are reviewed."])
        return gate
    gate["scores"] = scores
    d_base, d_dir = scores["arms"]["baseline"]["dialogue"]["overall_mean"], scores["arms"]["directed"]["dialogue"]["overall_mean"]
    n_base, n_dir = scores["arms"]["baseline"]["narrator"]["overall_mean"], scores["arms"]["directed"]["narrator"]["overall_mean"]
    dialogue_delta = de._round(d_dir - d_base, 4)
    narrator_delta = de._round(n_dir - n_base, 4)
    verdict = lambda diff: "improved" if diff >= GATE_MARGIN else "regressed" if diff <= -GATE_MARGIN else "no_clear_change"
    gate["verdicts"] = {
        "dialogue": {"overall_delta": dialogue_delta, "verdict": verdict(dialogue_delta)},
        "narrator": {"overall_delta": narrator_delta, "verdict": verdict(narrator_delta)},
    }
    if gate["verdicts"]["narrator"]["verdict"] == "regressed":
        gate.update(status="does_not_support", reasons=["narration regressed; it must stay neutral and unchanged"])
    elif gate["verdicts"]["dialogue"]["verdict"] == "regressed":
        gate.update(status="does_not_support", reasons=["directed dialogue regressed against baseline"])
    elif gate["verdicts"]["dialogue"]["verdict"] == "improved":
        gate.update(status="supports_experimental_review",
                    reasons=["directed dialogue improved against baseline with narration intact; "
                             "this supports further human review of the experimental profile only"])
    else:
        gate.update(status="no_clear_change", reasons=["directed dialogue did not clearly differ from baseline"])
    return gate


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def evaluate_scene_pilot(manifest: Mapping[str, Any], *, output_dir: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Evaluate a ``scene-dialogue-comparison-v1`` manifest.  Reads only; with ``output_dir`` publishes
    exclusive result files.

    Returns the report with ``ballots`` (arm-blind, per scene), ``ballot_key`` (sealed; never show
    reviewers) and ``human_gate`` attached.  An absent arm is reported as missing and holds the gate;
    an arm that is present but short, or any identity/text/runtime/settings/boundary/instruction
    discrepancy, is refused with an :class:`EvaluationError` before anything is written.
    """
    if not isinstance(manifest, Mapping):
        raise _fail("manifest_invalid", "manifest must be an object")
    manifest = copy.deepcopy(dict(manifest))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise _fail("manifest_invalid", f"schema_version must be {MANIFEST_SCHEMA}")
    comparison_id = de._ident(manifest.get("comparison_id"), "comparison_id")
    fixture_raw = manifest.get("fixture")
    if fixture_raw is not None and not (isinstance(fixture_raw, Mapping) and fixture_raw.get("is_fixture") is True):
        raise _fail("manifest_invalid", "fixture, when present, must be {'is_fixture': true, 'label': ...}")
    fixture = fixture_raw is not None
    fixture_label = de._text(fixture_raw.get("label"), "fixture.label") if fixture else None
    base_dir = manifest.get("base_dir")
    if not (isinstance(base_dir, str) and Path(base_dir).is_absolute()):
        raise _fail("manifest_invalid", "base_dir must be an absolute path")
    base_path = Path(base_dir)
    if base_path.is_symlink() or not base_path.is_dir():
        raise _fail("path_refused", "base_dir must be an existing directory, not a link", value=base_dir)
    lines = _validate_lines(manifest.get("lines"))
    source = _validate_source(manifest.get("source"), lines)
    runtime = _validate_runtime(manifest.get("runtime"), "runtime")
    arms_raw = manifest.get("arms")
    if not isinstance(arms_raw, Mapping) or set(arms_raw) - set(ARMS):
        raise _fail("manifest_invalid", f"arms may only contain {ARMS}")
    review = manifest.pop("human_review", None)
    manifest_sha256 = content_sha256(manifest)
    # Runtime identity is compared before any audio is read, so a mismatch is reported as such
    # rather than as a downstream receipt or hash disagreement.
    _check_arm_runtimes(runtime, arms_raw)

    sources = _SafeSources(base_path)
    arms = {name: _analyze_arm(name, arms_raw.get(name), lines, runtime, sources, fixture) for name in ARMS}
    findings = _check_cross_arm(arms, lines)
    comparisons = _compare(arms, lines)
    all_complete = all(arms[a]["status"] == "complete" for a in ARMS)
    ballot_set = key = None
    if all_complete:
        ballot_set, key = build_blind_ballots(comparison_id, lines, arms)
    gate_ballots = ballot_set or {"ballot_set_id": None, "scenes": []}
    gate = build_human_gate(fixture=fixture, ballot_set=gate_ballots, key=key or {"scenes": {}}, arms=arms, review=review)
    if review is not None:
        gate["human_review_sha256"] = content_sha256(review)
    integrity = sources.verify_unchanged()

    roles: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        entry = roles.setdefault(line["role_id"], {"role_id": line["role_id"], "role_name": line["role_name"],
                                                   "narrator": line["narrator"], "lines": 0})
        entry["lines"] += 1
    warnings = [f"arm {a} is {arms[a]['status']}: {arms[a]['reason']}" for a in ARMS if arms[a]["status"] != "complete"]
    warnings += [f["detail"] for f in findings]
    latency_unknown = [a for a in ARMS if arms[a]["status"] == "complete" and not (
        arms[a]["summary"]["latency_seconds"]["complete"] and arms[a]["summary"]["peak_vram_mb"]["complete"])]
    if latency_unknown:
        warnings.append(f"latency or VRAM is unknown for some samples in: {', '.join(latency_unknown)}")
    report = {
        "schema_version": REPORT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "comparison_id": comparison_id,
        "manifest_sha256": manifest_sha256,
        "fixture": {"is_fixture": fixture, "label": fixture_label},
        "review_only": True,
        "production_eligible": False,
        "production_note": (
            "Synthetic fixture: excluded from production readiness and selection claims." if fixture else
            "Pilot samples are review-only: they never replace canonical Takes, approve a voice or change a default."
        ),
        "status": "complete" if all_complete else "incomplete",
        "arm_status": {a: arms[a]["status"] for a in ARMS},
        "source": source,
        "runtime": runtime,
        "lines": {
            "count": len(lines),
            "narrator_lines": sum(1 for l in lines if l["narrator"]),
            "dialogue_lines": sum(1 for l in lines if not l["narrator"]),
            "scenes": [{"scene_id": s, "utterance_ids": [l["utterance_id"] for l in ls]} for s, ls in _scenes(lines)],
            "roles": list(roles.values()),
            "utterances": [
                {"utterance_id": l["utterance_id"], "role_id": l["role_id"], "scene_id": l["scene_id"],
                 "narrator": l["narrator"], "text_sha256": l["text_sha256"], "characters": len(l["text"]),
                 "source": l["source"], "direction_plan": l["direction"]}
                for l in lines
            ],
        },
        "arms": arms,
        "comparisons": comparisons,
        "findings": findings,
        "unavailable_measures": copy.deepcopy(UNAVAILABLE_MEASURES),
        "measurement_policy": MEASUREMENT_POLICY,
        "source_integrity": {"unchanged": True, "files": integrity},
        "ballot_ref": None if ballot_set is None else {
            "ballot_set_id": ballot_set["ballot_set_id"], "scenes": len(ballot_set["scenes"]),
            "items": sum(len(b["items"]) for b in ballot_set["scenes"]), "key_sha256": content_sha256(key)},
        "promotion": {"approval": False, "default_changed": False,
                      "human_listening": "final and still required"},
        "warnings": warnings,
    }
    report["evaluation_id"] = content_sha256(report)[:32]
    result = {**report, "ballots": ballot_set, "ballot_key": key, "human_gate": gate}
    if output_dir is not None:
        result["outputs"] = _publish_outputs(Path(output_dir), report, ballot_set, key, gate, review is not None, sources)
    return result


def summarize(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Terminal/log-safe view: never includes the sealed ballot key or ballot contents."""
    return {
        "comparison_id": result["comparison_id"],
        "evaluation_id": result["evaluation_id"],
        "status": result["status"],
        "fixture": result["fixture"],
        "review_only": result["review_only"],
        "production_eligible": result["production_eligible"],
        "arm_status": result["arm_status"],
        "ballot": result["ballot_ref"],
        "human_gate": {k: result["human_gate"][k] for k in ("status", "reasons", "approval", "default_changed")},
        "warnings": result["warnings"],
        "outputs": result.get("outputs"),
    }


# --------------------------------------------------------------------------- #
# Exclusive, immutable output
# --------------------------------------------------------------------------- #


def _publish_outputs(
    directory: Path, report: Dict[str, Any], ballot_set: Optional[Dict[str, Any]], key: Optional[Dict[str, Any]],
    gate: Dict[str, Any], has_review: bool, sources: _SafeSources,
) -> Dict[str, Any]:
    if directory.is_symlink():
        raise _fail("output_conflict", "output directory is a link", path=str(directory))
    resolved = directory.resolve()
    for entry in sources.entries.values():
        if Path(entry["path"]).resolve().is_relative_to(resolved):
            raise _fail("output_conflict", "output directory contains immutable input files", path=str(directory))
    files: Dict[str, bytes] = {"evaluation.json": de._pretty(report)}
    if ballot_set is not None and key is not None:
        files["ballot_set.json"] = de._pretty({k: v for k, v in ballot_set.items() if k != "scenes"})
        for ballot in ballot_set["scenes"]:
            files[f"ballots/{ballot['scene_id']}.json"] = de._pretty(ballot)
        files["ballot_key.json"] = de._pretty(key)
    if has_review:
        files["human_gate.json"] = de._pretty(gate)
    for name, payload in files.items():  # check every conflict before writing anything
        target = directory / name
        if target.exists() and (target.is_symlink() or target.read_bytes() != payload):
            raise _fail("output_conflict", "a different result already exists at this path; use a new output directory",
                        path=str(target))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ballots").mkdir(exist_ok=True)
    written = []
    for name, payload in files.items():
        written.append({"name": name, "sha256": sha256_hex(payload), "status": de._publish_file(directory / name, payload)})
    return {"directory": str(directory), "files": written}


def stage_blind_audio(result: Mapping[str, Any], destination: os.PathLike) -> List[Dict[str, Any]]:
    """Copy ballot audio to arm-blind file names for a reviewer player.  Hash-verified, exclusive."""
    key = result.get("ballot_key")
    if not key:
        raise _fail("ballots_unavailable", "no ballots exist because an arm is missing")
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    for scene in key["scenes"].values():
        for item_id, item in scene["items"].items():
            for label, option in item["options"].items():
                data = Path(option["audio_path"]).read_bytes()
                if sha256_hex(data) != option["audio_sha256"]:
                    raise _fail("source_changed", "staged audio no longer matches its recorded hash", item=item_id)
                target = directory / f"{item_id}-{label}.wav"
                staged.append({"sample_ref": target.stem, "file": str(target), "status": de._publish_file(target, data)})
    return staged


# --------------------------------------------------------------------------- #
# Labeled CPU-only artificial fixture
# --------------------------------------------------------------------------- #

FIXTURE_LABEL = "SYNTHETIC FIXTURE - artificial tones, not speech; excluded from production readiness"
_FIXTURE_SCENE_ROLES = (
    ("narrator", "Narrator", True),
    ("ada", "Ada", False),
    ("bo", "Bo", False),
    ("ada", "Ada", False),
)
_FIXTURE_INSTRUCTIONS = {
    "ada": "Speak quietly, with worry held back.",
    "bo": "Speak with dry, restrained amusement.",
}


def build_synthetic_fixture(
    directory: os.PathLike, *, omit_arms: Iterable[str] = (), line_count: int = MIN_LINES
) -> Dict[str, Any]:
    """Write artificial tone WAVs and receipts under ``directory`` and return a labeled manifest.

    Nothing here is speech or a model result; receipts' latency/VRAM are fake and flagged.
    Files are created exclusively.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    omit = set(omit_arms)
    runtime = {
        "name": "synthetic-fixture-engine",
        "runtime_profile": DIRECTED_PROFILE,
        "runtime_sha256": sha256_hex(b"synthetic-fixture-runtime"),
        "checkpoint_sha256": sha256_hex(b"synthetic-fixture-checkpoint"),
        "settings_sha256": sha256_hex(b"synthetic-fixture-settings"),
        "boundary_policy_sha256": sha256_hex(b"synthetic-fixture-boundary-policy"),
        "fallback": None,
    }
    project_id = "fixture-project"
    source_sha = sha256_hex(b"synthetic-fixture-source")
    lines = []
    for n in range(line_count):
        role_id, role_name, narrator = _FIXTURE_SCENE_ROLES[n % len(_FIXTURE_SCENE_ROLES)]
        utterance_id = f"fx-{n + 1:02d}"
        text = f"Synthetic fixture line {n + 1:02d} for the {role_name} role."
        lines.append({
            "utterance_id": utterance_id, "role_id": role_id, "role_name": role_name, "narrator": narrator,
            "scene_id": f"fx-scene-{n // len(_FIXTURE_SCENE_ROLES) + 1:02d}", "text": text,
            "source": {"project_id": project_id, "source_sha256": source_sha, "segment_id": f"seg-{n + 1:03d}"},
            "direction": {
                "native_instruction": None if narrator else _FIXTURE_INSTRUCTIONS[role_id],
                "origin": "engineering_comparator_hypothesis", "recommendation_only": True, "human_review": "pending",
                "evidence_status": "unknown", "confidence": "unknown",
            },
        })
    arms: Dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        if arm in omit:
            arms[arm] = {"status": "missing", "reason": "omitted from the synthetic fixture"}
            continue
        samples = {}
        for n, line in enumerate(lines):
            directed = arm == "directed" and not line["narrator"]
            instruction = line["direction"]["native_instruction"] if directed else None
            voice_id = f"fixture-voice-{line['role_id']}"
            reference_id = sha256_hex(f"fixture-reference-{line['role_id']}".encode())
            prompt_id = sha256_hex(f"fixture-prompt-{line['role_id']}".encode())
            audio = de._tone_wav(180.0 + 40.0 * (n % 3), 0.6 + 0.02 * n + (0.1 if directed else 0.0), 0.5)
            audio_path = root / f"{arm}-{line['utterance_id']}.wav"
            de._publish_file(audio_path, audio)
            audio_sha = sha256_hex(audio)
            receipt = {
                "synthetic_fixture": True,
                "utterance_id": line["utterance_id"],
                "audio_sha256": audio_sha,
                "text_sha256": _text_sha(line["text"]),
                "request_sha256": sha256_hex(f"{arm}|{line['utterance_id']}".encode()),
                "runtime_sha256": runtime["runtime_sha256"],
                "reference_audio_sha256": reference_id,
                "latency_seconds": 0.5 + arm_index * 0.1,
                "peak_vram_mb": 1000.0 + arm_index,
                "vram_measurement": "synthetic fixture value, not measured",
                "native_instruction": instruction,
                "worker_receipt": {
                    "identity_id": voice_id,
                    "prompt_sha256": prompt_id,
                    "fallback": None,
                    "worker_runtime_identity": {"sha256": runtime["runtime_sha256"]},
                    "execution_profile": {"runtime_profile": DIRECTED_PROFILE},
                    "generation": {"direction": expected_direction_receipt(
                        instruction, identity_id=voice_id, text=line["text"])},
                },
                "human_review": "pending",
                "production_ready": False,
            }
            receipt_path = root / f"{arm}-{line['utterance_id']}.receipt.json"
            de._publish_file(receipt_path, de._pretty(receipt))
            samples[line["utterance_id"]] = {
                "text": line["text"],
                "text_sha256": _text_sha(line["text"]),
                "audio_file": audio_path.name,
                "audio_sha256": audio_sha,
                "voice_id": voice_id,
                "reference_id": reference_id,
                "prompt_id": prompt_id,
                "runtime_sha256": runtime["runtime_sha256"],
                "instruction": {"native_instruction": instruction, "forwarded": instruction is not None},
                "receipt": {"path": receipt_path.name},
            }
        arms[arm] = {"status": "present", "runtime": dict(runtime), "samples": samples}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "comparison_id": "synthetic-scene-fixture",
        "fixture": {"is_fixture": True, "label": FIXTURE_LABEL},
        "base_dir": str(root.resolve()),
        "source": {"projects": [{"project_id": project_id, "binding_sha256": sha256_hex(b"synthetic-fixture-binding")}],
                   "lines_sha256": lines_sha256(lines)},
        "runtime": runtime,
        "lines": lines,
        "arms": arms,
    }
