"""Exact local Qwen provider capability and effective voice identity records.

This module reads checked-in manifests/runtime metadata as text and JSON only.
It never imports a TTS library or loads model weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = REPO_ROOT / "models" / "manifests"

_SPEAKERS = (
    ("vivian", "Vivian"),
    ("serena", "Serena"),
    ("uncle_fu", "Uncle Fu"),
    ("dylan", "Dylan"),
    ("eric", "Eric"),
    ("ryan", "Ryan"),
    ("aiden", "Aiden"),
    ("ono_anna", "Ono Anna"),
    ("sohee", "Sohee"),
)


def resolve_customvoice_speaker(value: str) -> str:
    """Return a canonical API speaker id, or reject an unknown speaker."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("speaker must be one of the supported CustomVoice presets")
    normalized = re.sub(r"[\s_-]+", " ", value.strip()).casefold()
    for speaker_id, label in _SPEAKERS:
        aliases = {speaker_id, label, label.replace(" ", "_")}
        if any(re.sub(r"[\s_-]+", " ", alias).casefold() == normalized for alias in aliases):
            return speaker_id
    raise ValueError(f"unsupported CustomVoice speaker: {value.strip()}")


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _checkpoint(manifest_name: str) -> dict[str, Any]:
    path = MANIFESTS / manifest_name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"model_id": None, "source_model_id": None, "manifest_revision": None,
                "manifest_sha256": None, "model_content_sha256": None}
    model_sha = next((str(item.get("sha256")) for item in payload.get("files", [])
                      if isinstance(item, dict) and str(item.get("path", "")).endswith("/model.safetensors")), None)
    return {
        "model_id": payload.get("model_id"),
        "source_model_id": (payload.get("sources", {}).get("model", {}) or {}).get("model_id"),
        "manifest_revision": payload.get("revision"),
        "manifest_sha256": _sha256(path),
        "model_content_sha256": model_sha,
        "content_hash_basis": "manifest-pinned expected SHA256; local weight bytes were not read",
    }


def _runtime() -> dict[str, Any]:
    env = REPO_ROOT / "environments" / "qwen3tts"
    pyproject = env / "pyproject.toml"
    lock = env / "uv.lock"
    site_packages = [env / ".venv" / "Lib" / "site-packages", *((env / ".venv" / "lib").glob("python*/site-packages"))]
    installed_version = None
    for site_dir in site_packages:
        for metadata_path in site_dir.glob("qwen_tts-*.dist-info/METADATA"):
            try:
                lines = metadata_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            headers = {line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
                       for line in lines if ":" in line and not line[:1].isspace()}
            if headers.get("name", "").casefold().replace("_", "-") == "qwen-tts":
                installed_version = headers.get("version")
                break
        if installed_version:
            break
    return {
        "environment": "environments/qwen3tts/.venv",
        "qwen_tts_version": installed_version,
        "pyproject_sha256": _sha256(pyproject),
        "lock_sha256": _sha256(lock),
        "locally_tested_model_execution": False,
        "local_test_scope": "provider/server contract tests use fake responses; no speech model/GPU execution",
    }


def _adapter(paths: tuple[str, ...]) -> dict[str, Any]:
    return {"files": [{"path": item, "sha256": _sha256(REPO_ROOT / item)} for item in paths]}


def _control(status: str, documentation: str, local: str = "not acoustically tested") -> dict[str, Any]:
    return {"status": status, "documented": documentation, "locally_tested": local}


def capability_record(backend: str, identity: dict | None = None) -> dict[str, Any]:
    """Describe installed Qwen checkpoints and truthful provider controls."""

    aliases = {"qwen3tts_cv": "customvoice", "qwen3tts_customvoice": "customvoice",
               "qwen3tts": "voicedesign", "qwen3tts_voicedesign": "voicedesign"}
    kind = aliases.get(str(backend).casefold())
    custom = kind == "customvoice"
    design = kind == "voicedesign"
    active = _checkpoint(f"qwen3_tts_{kind}.json") if kind else _checkpoint("__unknown__.json")
    base = _checkpoint("qwen3_tts_base.json")
    server_path = "service/qwen3tts_customvoice_server.py" if custom else "service/qwen3tts_server.py"
    paths = (server_path, "dubbing_service/app/providers/tts.py")
    if design:
        paths += ("service/qwen_character_voice.py",)
    mechanisms = {
        "voice_design": {"status": "supported" if design else "unsupported",
                         "checkpoint": active["model_id"], "basis": "Qwen VoiceDesign generate_voice_design API" if design else None},
        "reference_clone": {"status": "supported" if design else "unsupported",
                            "checkpoint": base["model_id"] if design else None,
                            "basis": "Base create_voice_clone_prompt + generate_voice_clone, ICL transcript mode" if design else None},
        "reusable_conditioning": {"status": "supported" if design else "unsupported",
                                  "checkpoint": base["model_id"] if design else None,
                                  "basis": "per-identity checksum-bound prompt.npz" if design else None},
    }
    performance = {
        "style_instruction": _control("supported" if (custom or design) else "unsupported",
            "CustomVoice instruct (1.7B); VoiceDesign instruct documented by installed qwen-tts 0.1.1" if (custom or design) else "No style instruction path"),
        "emotion_text": _control("supported" if (custom or design) else "unsupported",
            "mapped to instruct by local provider; acoustic effect not model-tested" if (custom or design) else "No emotion instruction path"),
        "intensity": _control("unsupported" if design else "unverified",
            "Base does not accept natural-language performance control" if design else "No direct intensity parameter in provider payload"),
        "emphasis": _control("unsupported" if design else "unverified",
            "Base does not accept natural-language performance control" if design else "No direct emphasis parameter in provider payload"),
        "pace": _control("supported" if design else "unverified",
            "Base uses project postprocessing for pace" if design else "No direct speed parameter in CustomVoice provider payload"),
        "volume": _control("supported" if design else "unverified",
            "Base uses project postprocessing for volume" if design else "No direct volume parameter in CustomVoice provider payload"),
        "customvoice_0_6b_instruct": _control("unsupported",
            "Installed qwen-tts 0.1.1 source clears instruct for tts_model_size in '0b6'; no 0.6B checkpoint installed"),
    }
    if identity and identity.get("mechanism") == "qwen_reference":
        performance.update({
            "style_instruction": _control("unsupported", "Base clone generation has no natural-language performance instruction argument"),
            "emotion_text": _control("unsupported", "Base clone generation has no natural-language emotion instruction argument"),
            "intensity": _control("unsupported", "Base clone generation has no intensity control"),
            "emphasis": _control("unsupported", "Base clone generation has no emphasis control"),
            "pace": _control("supported", "Project postprocessing changes pace after Base synthesis"),
            "volume": _control("supported", "Project postprocessing changes volume after Base synthesis"),
        })
    result = {
        "schema": "provider-capabilities-v1",
        "backend": backend,
        "checkpoint": active,
        "conditioning_checkpoint": base if design else None,
        "adapter": _adapter(paths),
        "runtime": _runtime(),
        "dispatch": {"mode": "serialized_single_utterance", "input_cardinality": 1,
                     "output_cardinality": 1, "mixed_batch": "unsupported_by_local_adapter",
                     "scalar_broadcast": "rejected_at_local_request_boundary", "streaming": "not_exposed"},
        "languages": {"documented": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                      "local_acoustic_verification": "not_run", "dialect_control": "unverified; design wording is not evidence of accent compliance"},
        "reference_requirements": {"base": ["immutable reference audio", "exact transcript", "checksum-bound conditioning", "compatible checkpoint and adapter"],
                                   "customvoice": ["explicit canonical preset speaker"],
                                   "cache_portability": "Derived conditioning requires matching model/runtime compatibility; source assets are retained"},
        "preset_inventory": ([{"id": speaker_id, "label": label} for speaker_id, label in _SPEAKERS]
                             if custom else []),
        "mechanisms": mechanisms,
        "performance": performance,
        "evidence": {"documentation": "installed qwen-tts 0.1.1 source + environment pyproject/lock + model manifests",
                     "locally_tested": "CPU contract tests only; model behavior and sound quality unverified"},
    }
    if identity is not None:
        result["active_identity_mechanism"] = identity.get("mechanism")
    if identity is not None:
        result["effective_identity"] = effective_identity({**result, "identity": identity})
    return result


def effective_identity(record: dict[str, Any]) -> str:
    """Hash the effective voice conditioning, excluding display/random labels."""

    identity = record.get("identity") or {}
    backend = str(record.get("backend") or identity.get("backend") or "")
    mechanism = str(identity.get("mechanism") or record.get("mechanism") or "")
    checkpoint = record.get("checkpoint") or {}
    if mechanism == "builtin" or backend in {"qwen3tts_cv", "qwen3tts_customvoice"}:
        speaker = resolve_customvoice_speaker(
            str(identity.get("builtin_speaker") or identity.get("speaker") or record.get("builtin_speaker") or record.get("speaker") or "")
        )
        identity_payload = {"backend": "qwen3tts_cv", "model_content_sha256": checkpoint.get("model_content_sha256") or identity.get("model_revision") or identity.get("model"),
                            "speaker": speaker}
    else:
        conditioning = record.get("conditioning_checkpoint") or checkpoint
        reference = identity.get("reference_audio_sha256") or record.get("reference_audio_sha256")
        if not reference:
            raise ValueError("reference voice identity requires reference_audio_sha256")
        identity_payload = {"backend": backend or "qwen3tts", "model_content_sha256": conditioning.get("model_content_sha256") or identity.get("model_revision") or identity.get("model"),
                            "reference_audio_sha256": str(reference).casefold(),
                            "clone_mode": identity.get("clone_mode") or identity.get("mode") or "icl"}
    encoded = json.dumps(identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthesis_components(backend: str, identity: dict | None = None) -> dict[str, str]:
    """Stable synthesis inputs only; no UI labels, ratings or diagnostic timings."""
    record = capability_record(backend, identity)
    checkpoint = record["conditioning_checkpoint"] if identity and identity.get("mechanism") == "qwen_reference" else record["checkpoint"]
    if not checkpoint or not checkpoint.get("model_id"):
        return {}
    from dubbing_service.app.routers.settings import _load_raw_file_config
    configured = _load_raw_file_config()
    runtime = record["runtime"]
    profile = configured.get("tts_params", {}).get("qwen3_execution_profile", "native-current") if "cv" in backend or "customvoice" in backend else "native-base"
    adapter = record["adapter"]
    if profile == "qwen-cudagraph" and ("cv" in backend or "customvoice" in backend):
        adapter = _adapter(tuple(item["path"] for item in adapter["files"]) + ("service/faster_qwen3tts_customvoice_server.py",))
        runtime = {**runtime, "lock_sha256": _sha256(REPO_ROOT / "environments/faster_qwen3tts/uv.lock")}
    payload = {"adapter": adapter, "qwen_tts_version": runtime.get("qwen_tts_version"),
        "lock": runtime.get("lock_sha256"), "profile": profile}
    return {"model_id": checkpoint.get("source_model_id") or checkpoint["model_id"],
        "model_revision": checkpoint.get("model_content_sha256") or checkpoint.get("manifest_revision") or "unknown",
        "runtime_version": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "decoder_version": runtime.get("qwen_tts_version") or "unknown"}
