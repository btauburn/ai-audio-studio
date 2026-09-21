"""Renderer capability + engine/model descriptor and license records.

This is the thin boundary the audiobook pipeline needs; it intentionally does
not become a generic TTS framework.  Capabilities are derived from the
platform's existing ``core.tts_backends.BACKEND_SPECS`` so there is exactly one
source of truth for local workers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from dubbing_service.app.core.tts_backends import BACKEND_SPECS
from dubbing_service.app.models.audiobook import EngineLicenseRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "models" / "manifests"


class RendererCapabilities(BaseModel):
    """What a renderer can actually do (Phase 9 contract)."""

    reference_clone: bool = False
    reference_transcript: bool = False
    builtin_voice: bool = False
    voice_design: bool = False
    emotion_text: bool = False
    emotion_vector: bool = False
    style_instruction: bool = False
    speed: bool = False
    language: bool = False
    seed: bool = False
    continuation: bool = False
    chunking: bool = Field(
        False,
        description="渲染器可返回可按文本区间归属的内部分块（提交前必须重组为一个 Take）",
    )


# Backend capability strings -> renderer capability fields.
_CAPABILITY_MAP = {
    "reference_audio": "reference_clone",
    "builtin_speaker": "builtin_voice",
    "voice_design": "voice_design",
    "emotion_text": "emotion_text",
    "emotion_vector": "emotion_vector",
    "instruction": "style_instruction",
    "speed": "speed",
    "language": "language",
}


def capabilities_from_backend(backend_id: str) -> RendererCapabilities:
    spec = BACKEND_SPECS.get(backend_id)
    if spec is None:
        return RendererCapabilities()
    fields = {"reference_transcript": True, "chunking": True}
    for capability in spec.capabilities:
        mapped = _CAPABILITY_MAP.get(capability)
        if mapped:
            fields[mapped] = True
    return RendererCapabilities(**fields)


def _manifest_payload(model_id: Optional[str]) -> Optional[dict]:
    if not model_id:
        return None
    path = MANIFEST_DIR / f"{model_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_revision(model_id: Optional[str]) -> Optional[str]:
    payload = _manifest_payload(model_id)
    if not payload:
        return None
    revision = payload.get("revision")
    return str(revision) if revision else None


def _manifest_license(model_id: Optional[str]) -> Optional[str]:
    payload = _manifest_payload(model_id)
    if not payload:
        return None
    sources = payload.get("sources")
    if isinstance(sources, dict):
        model = sources.get("model")
        if isinstance(model, dict) and model.get("license"):
            return str(model["license"])
    return None


@dataclass(frozen=True)
class EngineDescriptor:
    """Engine/model identity used by the synthesis fingerprint."""

    engine_id: str
    display_name: str
    backend_id: Optional[str] = None
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    quantization: Optional[str] = None
    runtime_version: Optional[str] = None
    decoder_version: Optional[str] = None
    capabilities: RendererCapabilities = field(default_factory=RendererCapabilities)
    license: Optional[EngineLicenseRecord] = None
    installed: bool = False
    model_directory: Optional[str] = None

    def identity_components(self) -> Dict[str, str]:
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id or "",
            "model_revision": self.model_revision or "",
            "quantization": self.quantization or "",
            "runtime_version": self.runtime_version or "",
            "decoder_version": self.decoder_version or "",
        }


# License records keep the four scopes separate: a permissive repository
# license never implies the weights or the cloned voice may be used
# commercially.
_LICENSES: Dict[str, EngineLicenseRecord] = {
    "indextts": EngineLicenseRecord(
        engine_id="indextts",
        display_name="IndexTTS2",
        platform_license="Apache-2.0 (AI Audio Studio)",
        engine_source_license="engine source per upstream repository",
        model_weight_license="per upstream model card",
        voice_rights="reference audio supplied by the user; user must hold rights",
        commercial_use="requires_review",
        backend_ids=["indextts"],
        notes="参考音频权属不随仓库许可证转移；导出商用前需人工确认。",
    ),
    "cosyvoice3": EngineLicenseRecord(
        engine_id="cosyvoice3",
        display_name="CosyVoice3",
        platform_license="Apache-2.0 (AI Audio Studio)",
        engine_source_license="per upstream CosyVoice repository",
        model_weight_license="per upstream model card",
        voice_rights="reference audio supplied by the user; user must hold rights",
        commercial_use="requires_review",
        backend_ids=["cosyvoice3"],
    ),
    "qwen3tts_voicedesign": EngineLicenseRecord(
        engine_id="qwen3tts_voicedesign",
        display_name="Qwen3-TTS VoiceDesign",
        platform_license="Apache-2.0 (AI Audio Studio)",
        engine_source_license="Qwen3-TTS runtime (upstream)",
        model_weight_license="Apache-2.0 (per models/manifests/qwen3_tts_voicedesign.json)",
        voice_rights="built-in / designed voice, no reference audio rights needed",
        commercial_use="allowed",
        backend_ids=["qwen3tts"],
    ),
    "qwen3tts_customvoice": EngineLicenseRecord(
        engine_id="qwen3tts_customvoice",
        display_name="Qwen3-TTS CustomVoice",
        platform_license="Apache-2.0 (AI Audio Studio)",
        engine_source_license="Qwen3-TTS runtime (upstream)",
        model_weight_license="Apache-2.0 (per models/manifests/qwen3_tts_customvoice.json)",
        voice_rights="built-in speakers; no user reference audio required",
        commercial_use="allowed",
        is_default=True,
        backend_ids=["qwen3tts_cv"],
    ),
    "fixture": EngineLicenseRecord(
        engine_id="fixture",
        display_name="Deterministic fixture renderer (development only)",
        platform_license="Apache-2.0 (AI Audio Studio)",
        engine_source_license="n/a — generated tone, no model",
        model_weight_license="n/a — no model weights",
        voice_rights="n/a — not speech",
        commercial_use="not_allowed",
        backend_ids=["fixture"],
        notes="仅用于无模型环境的确定性测试替身，不产生可发布语音。",
    ),
    "omnivoice": EngineLicenseRecord(
        engine_id="omnivoice",
        display_name="OmniVoice",
        platform_license="n/a — external engine",
        engine_source_license="external engine source",
        model_weight_license="non-commercial (OmniVoice weights)",
        voice_rights="reference voice rights not granted by weights license",
        commercial_use="not_allowed",
        backend_ids=[],
        notes="非商用权重；不得设为默认引擎。",
    ),
    "yovoice": EngineLicenseRecord(
        engine_id="yovoice",
        display_name="yovoice (optional external renderer, not integrated)",
        platform_license="Apache-2.0 (upstream yovoice)",
        engine_source_license="Apache-2.0 (upstream yovoice)",
        model_weight_license="depends on the provider the user configures",
        voice_rights="reference audio supplied by the user",
        commercial_use="unknown",
        backend_ids=[],
        notes="仅保留扩展接缝；本任务不实现 yovoice 守护进程。",
    ),
}

_BACKEND_ENGINE = {
    "indextts": "indextts",
    "cosyvoice3": "cosyvoice3",
    "qwen3tts": "qwen3tts_voicedesign",
    "qwen3tts_cv": "qwen3tts_customvoice",
}

_BACKEND_MODEL = {
    "indextts": "index_tts2",
    "cosyvoice3": "cosyvoice3",
    "qwen3tts": "qwen3_tts_voicedesign",
    "qwen3tts_cv": "qwen3_tts_customvoice",
}

# OmniVoice must never be the default (non-commercial weights).
NON_DEFAULT_ENGINES = frozenset({"omnivoice"})


class UnknownBackendError(ValueError):
    """Raised when a backend id has no licensed engine behind it."""


def license_records() -> List[EngineLicenseRecord]:
    return list(_LICENSES.values())


def license_record(engine_id: str) -> Optional[EngineLicenseRecord]:
    return _LICENSES.get(engine_id)


def default_engine_id() -> str:
    """Default engine for new work.

    Resolution order: configured ``tts_backend`` -> Qwen3-TTS CustomVoice.
    ``omnivoice`` is refused as a default even if configured.
    """

    backend_id = "qwen3tts_cv"
    try:
        from dubbing_service.app.routers.settings import _load_raw_file_config

        configured = _load_raw_file_config().get("tts_backend")
        if isinstance(configured, str) and configured:
            backend_id = configured
    except Exception:  # pragma: no cover - config surface unavailable
        pass
    engine_id = _BACKEND_ENGINE.get(backend_id, "qwen3tts_customvoice")
    if engine_id in NON_DEFAULT_ENGINES:
        return "qwen3tts_customvoice"
    return engine_id


def engine_id_for_backend(backend_id: str) -> str:
    """Map a TTS backend id onto its licensed engine id.

    An unknown backend is refused rather than silently rendered with the
    default engine: that would stamp the wrong engine onto the Take.
    """

    engine_id = _BACKEND_ENGINE.get(backend_id)
    if engine_id is None:
        raise UnknownBackendError(f"未注册的 TTS 后端: {backend_id}")
    return engine_id


def describe_engine(engine_id: str) -> EngineDescriptor:
    """Build a descriptor, reading manifest metadata when it is available."""

    if engine_id == "fixture":
        return EngineDescriptor(
            engine_id="fixture",
            display_name="Deterministic fixture renderer",
            capabilities=RendererCapabilities(
                builtin_voice=True,
                language=True,
                speed=True,
                chunking=True,
            ),
            license=_LICENSES["fixture"],
            installed=True,
        )

    backend_id = next(
        (bid for bid, eid in _BACKEND_ENGINE.items() if eid == engine_id), None
    )
    if backend_id is None:
        if engine_id in _LICENSES:
            return EngineDescriptor(
                engine_id=engine_id,
                display_name=_LICENSES[engine_id].display_name,
                license=_LICENSES[engine_id],
                installed=False,
            )
        raise ValueError(f"unknown engine_id: {engine_id}")

    spec = BACKEND_SPECS[backend_id]
    model_dir = REPO_ROOT / spec.model_directory
    model_id = _BACKEND_MODEL.get(backend_id)
    weight_license = _manifest_license(model_id)
    record = _LICENSES.get(engine_id)
    if record is not None and weight_license:
        record = record.model_copy(update={"model_weight_license": weight_license})
    return EngineDescriptor(
        engine_id=engine_id,
        display_name=spec.display_name,
        backend_id=backend_id,
        model_id=model_id,
        model_revision=_manifest_revision(model_id),
        capabilities=capabilities_from_backend(backend_id),
        license=record,
        installed=model_dir.is_dir(),
        model_directory=str(model_dir),
    )


def runtime_status(engine_id: Optional[str] = None) -> Dict[str, object]:
    """Report whether the weights needed for a real runtime test exist locally."""

    engine_id = engine_id or default_engine_id()
    descriptor = describe_engine(engine_id)
    return {
        "engine_id": descriptor.engine_id,
        "backend_id": descriptor.backend_id,
        "model_directory": descriptor.model_directory,
        "installed": descriptor.installed,
        "model_revision": descriptor.model_revision,
    }
