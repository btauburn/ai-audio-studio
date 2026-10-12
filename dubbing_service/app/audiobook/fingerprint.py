"""Canonical synthesis fingerprint (the *requirement*) — Phase 5.

``input_fingerprint`` answers "is this the same synthesis requirement?".
``audio_sha256`` answers "are these the same audio bytes?".  The two are never
conflated here: one input fingerprint may intentionally produce several Takes.

Only values capable of changing generated audio are included.  Book title,
cover art, UI state, review notes and other human metadata are deliberately
excluded so changing them never invalidates audio.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel

from dubbing_service.app.audiobook import language as language_module
from dubbing_service.app.models.audiobook import AttributionSource
from dubbing_service.app.models.enums import EmotionMode
from dubbing_service.app.models.project import Role, Segment, SegmentTake

FINGERPRINT_SCHEMA = "audiobook-synthesis-requirement-v1"

# Component names that can change generated audio.  Everything returned by
# SynthesisRequirement.components() is audio-affecting by construction; this
# set exists so regeneration code can classify changes without hardcoding.
AUDIO_AFFECTING_COMPONENTS = frozenset(
    {
        "utterance_id",
        "spoken_text_hash",
        "voice_revision",
        "reference_audio_sha256",
        "reference_transcript_hash",
        "renderer_id",
        "engine_id",
        "model_id",
        "model_revision",
        "quantization",
        "runtime_version",
        "decoder_version",
        "language",
        "emotion_mode",
        "emotion_text",
        "emotion_vector",
        "emotion_intensity",
        "speed",
        "volume",
        "pitch",
        "emphasis",
        "style_instruction",
        "builtin_speaker",
        "seed",
        "temperature",
        "top_p",
        "continuation_context_hash",
        "performance_direction",
    }
)

# Metadata that must never invalidate audio (documented for regression tests).
NON_AUDIO_METADATA_FIELDS = frozenset(
    {
        "book_title",
        "author",
        "cover_art",
        "description",
        "ui_state",
        "review_note",
        "review_status",
        "project_name",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    """Deterministic serialization: sorted keys, no whitespace, UTF-8."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _num(value: object) -> str:
    """Format a number so equal values always produce equal text."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return format(value, ".6f").rstrip("0").rstrip(".") or "0"
    return str(value).strip()


def _vector(value: Optional[Sequence[float]]) -> str:
    if not value:
        return ""
    return ",".join(_num(item) for item in value)


class SynthesisRequirement(BaseModel):
    """Every value that can change the audio generated for one utterance."""

    utterance_id: str
    spoken_text_hash: str = ""
    voice_revision: Optional[str] = None
    reference_audio_sha256: Optional[str] = None
    reference_transcript_hash: Optional[str] = None
    renderer_id: Optional[str] = None
    engine_id: Optional[str] = None
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    quantization: Optional[str] = None
    runtime_version: Optional[str] = None
    decoder_version: Optional[str] = None
    language: Optional[str] = None
    emotion_mode: str = EmotionMode.TEXT.value
    emotion_text: Optional[str] = None
    emotion_vector: Optional[List[float]] = None
    emotion_intensity: Optional[float] = None
    speed: float = 1.0
    volume: Optional[float] = None
    pitch: Optional[float] = None
    emphasis: Optional[str] = None
    style_instruction: Optional[str] = None
    builtin_speaker: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    continuation_context_hash: Optional[str] = None
    #: M4.  The Performance Director delivery component: the SHA-256 of the
    #: canonical ``{control_policy, native_instruction, seed_salt}``.  It is
    #: present only when a native instruction or a seed salt really reaches the
    #: runtime, so a project that forwards nothing keeps its existing
    #: fingerprints byte for byte.  See ``docs/performance_director.md``.
    performance_direction: Optional[str] = None

    def components(self) -> Dict[str, str]:
        """Normalized, stable component map (the hashed payload)."""

        components = {
            "utterance_id": self.utterance_id or "",
            "spoken_text_hash": self.spoken_text_hash or "",
            "voice_revision": self.voice_revision or "",
            "reference_audio_sha256": self.reference_audio_sha256 or "",
            "reference_transcript_hash": self.reference_transcript_hash or "",
            "renderer_id": self.renderer_id or "",
            "engine_id": self.engine_id or "",
            "model_id": self.model_id or "",
            "model_revision": self.model_revision or "",
            "quantization": self.quantization or "",
            "runtime_version": self.runtime_version or "",
            "decoder_version": self.decoder_version or "",
            "language": self.language or "",
            "emotion_mode": self.emotion_mode or "",
            "emotion_text": (self.emotion_text or "").strip(),
            "emotion_vector": _vector(self.emotion_vector),
            "emotion_intensity": _num(self.emotion_intensity),
            "speed": _num(self.speed),
            "pitch": _num(self.pitch),
            "emphasis": (self.emphasis or "").strip(),
            "style_instruction": (self.style_instruction or "").strip(),
            "builtin_speaker": self.builtin_speaker or "",
            "seed": _num(self.seed),
            "temperature": _num(self.temperature),
            "top_p": _num(self.top_p),
            "continuation_context_hash": self.continuation_context_hash or "",
        }
        # Legacy renderers leave volume to non-destructive playback. Preserve
        # their v1 fingerprints; include gain only when the renderer bakes it.
        if self.volume is not None:
            components["volume"] = _num(self.volume)
        # Same precedent as volume: a delivery direction is serialized only when
        # one is really forwarded, so adding this component cannot stale a Take
        # that no instruction reached.
        if self.performance_direction is not None:
            components["performance_direction"] = self.performance_direction
        return components

    def fingerprint(self) -> str:
        payload = {
            "schema": FINGERPRINT_SCHEMA,
            "components": self.components(),
        }
        return sha256_bytes(canonical_json_bytes(payload))


def reference_audio_sha256(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return file_sha256(Path(path))
    except OSError:
        return None


def voice_revision_for_role(role: Optional[Role]) -> Optional[str]:
    """Stable identifier for a role's voice configuration.

    Any role field that changes the produced voice must flow into this value;
    identity fields (display name, description, rights note) must not.
    """

    if role is None:
        return None
    if role.voice_identity is not None:
        identity = role.voice_identity
        return sha256_bytes(canonical_json_bytes({
            "id": identity.id, "mechanism": identity.mechanism,
            "reference": identity.reference_audio_sha256, "prompt": identity.prompt_sha256,
            "speaker": identity.builtin_speaker, "model": identity.model,
            "model_revision": identity.model_revision, "runtime": identity.runtime_version,
            "settings": identity.generation_settings,
        }))[:32]
    payload = {
        "role_id": role.id,
        "speaker_audio": role.speaker_audio or "",
        "speaker_voice_id": role.speaker_voice_id or "",
        "tts_backend": role.tts_backend or "",
        "qwen3_speaker": role.qwen3_speaker or "",
        "qwen3_language": role.qwen3_language or "",
        "voice_description": role.voice_description or "",
        "style_instruction": role.style_instruction or "",
        "default_emotion_mode": role.default_emotion_mode.value,
        "default_emotion_text": role.default_emotion_text or "",
        "default_emotion_vector": _vector(role.default_emotion_vector),
        "emotion_alpha": _num(role.emotion_alpha),
    }
    return sha256_bytes(canonical_json_bytes(payload))[:32]


def build_requirement(
    *,
    utterance_id: str,
    spoken_text_hash: str,
    engine_id: str,
    engine_components: Mapping[str, str] | None = None,
    renderer_id: Optional[str] = None,
    voice_revision: Optional[str] = None,
    reference_audio: Optional[str] = None,
    reference_transcript: Optional[str] = None,
    builtin_speaker: Optional[str] = None,
    language: Optional[str] = None,
    emotion_mode: str = EmotionMode.TEXT.value,
    emotion_text: Optional[str] = None,
    emotion_vector: Optional[Sequence[float]] = None,
    emotion_intensity: Optional[float] = None,
    speed: float = 1.0,
    volume: Optional[float] = None,
    pitch: Optional[float] = None,
    emphasis: Optional[str] = None,
    style_instruction: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    continuation_context_hash: Optional[str] = None,
    performance_direction: Optional[str] = None,
) -> SynthesisRequirement:
    engine_components = engine_components or {}
    return SynthesisRequirement(
        utterance_id=utterance_id,
        spoken_text_hash=spoken_text_hash,
        voice_revision=voice_revision,
        reference_audio_sha256=reference_audio,
        reference_transcript_hash=reference_transcript,
        renderer_id=renderer_id,
        engine_id=engine_id,
        model_id=engine_components.get("model_id"),
        model_revision=engine_components.get("model_revision"),
        quantization=engine_components.get("quantization"),
        runtime_version=engine_components.get("runtime_version"),
        decoder_version=engine_components.get("decoder_version"),
        language=language,
        emotion_mode=emotion_mode,
        emotion_text=emotion_text,
        emotion_vector=list(emotion_vector) if emotion_vector else None,
        emotion_intensity=emotion_intensity,
        speed=speed,
        volume=volume,
        pitch=pitch,
        emphasis=emphasis,
        style_instruction=style_instruction,
        builtin_speaker=builtin_speaker,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        continuation_context_hash=continuation_context_hash,
        performance_direction=performance_direction,
    )


def requirement_for_segment(
    segment: Segment,
    *,
    project_id: str,
    engine_id: str,
    engine_components: Mapping[str, str] | None = None,
    renderer_id: Optional[str] = None,
    role: Optional[Role] = None,
    spoken_text_hash: Optional[str] = None,
    reference_audio_path: Optional[Path] = None,
    reference_transcript: Optional[str] = None,
    continuation_context_hash: Optional[str] = None,
    seed: Optional[int] = None,
    pitch: Optional[float] = None,
    emphasis: Optional[str] = None,
    language: Optional[str] = None,
    performance_direction: Optional[str] = None,
) -> SynthesisRequirement:
    """Gather the live requirement for one utterance from canonical state.

    ``language`` lets a caller carry in the value it already resolved for this
    transaction instead of resolving a second time; when it is omitted the one
    shared rule (``language.resolve_synthesis_language``) is used, so the
    requirement can never disagree with the renderer or QA about the language.
    """

    effective_role = role
    style_instruction = effective_role.style_instruction if effective_role else None
    builtin_speaker = (
        (segment.qwen3_speaker or (effective_role.qwen3_speaker if effective_role else None))
    )
    language = language_module.normalize_language(language) or (
        language_module.resolve_synthesis_language(segment, effective_role)
    )
    return build_requirement(
        utterance_id=segment.id,
        spoken_text_hash=spoken_text_hash or text_sha256(segment.text),
        engine_id=engine_id,
        engine_components=engine_components,
        renderer_id=renderer_id,
        voice_revision=voice_revision_for_role(effective_role),
        reference_audio=reference_audio_sha256(reference_audio_path),
        reference_transcript=(
            text_sha256(reference_transcript) if reference_transcript else None
        ),
        builtin_speaker=builtin_speaker,
        language=language,
        emotion_mode=segment.emotion_mode.value,
        emotion_text=segment.emotion_text,
        emotion_vector=segment.emotion_vector,
        emotion_intensity=segment.emotion_alpha,
        speed=segment.playback_speed,
        volume=(segment.volume_gain if effective_role and effective_role.voice_identity
                and effective_role.voice_identity.mechanism == "qwen_reference" else None),
        style_instruction=style_instruction,
        seed=seed,
        pitch=pitch,
        emphasis=emphasis,
        continuation_context_hash=continuation_context_hash,
        performance_direction=performance_direction,
    )


def changed_components(
    take_components: Optional[Mapping[str, str]],
    live_components: Mapping[str, str],
) -> List[str]:
    """Component names whose value differs between a Take and live state."""

    stored = dict(take_components or {})
    changed = []
    for key in sorted(set(stored) | set(live_components)):
        if stored.get(key, "") != live_components.get(key, ""):
            changed.append(key)
    return changed


def is_stale(take: SegmentTake, live: SynthesisRequirement) -> bool:
    """True when the Take no longer satisfies the live requirement."""

    if not take.input_fingerprint:
        return False
    return take.input_fingerprint != live.fingerprint()


def requirement_stamps(
    take: SegmentTake,
    *,
    spoken_text_revision_id: Optional[str],
    renderer_id: Optional[str],
    engine_id: Optional[str],
    model_id: Optional[str],
    source_text_hash: Optional[str],
) -> Dict[str, object]:
    """Field updates applied to a Take when it is committed."""

    return {
        "spoken_text_revision_id": spoken_text_revision_id,
        "renderer_id": renderer_id,
        "engine_id": engine_id,
        "model_id": model_id,
        "source_text_hash": source_text_hash,
    }


def attribution_component(source: AttributionSource, confidence: Optional[float]) -> str:
    """Attribution provenance for diagnostics (never audio-affecting alone).

    Speaker identity reaches audio through ``voice_revision``; attribution
    provenance is recorded for audit but is not part of the requirement.
    """

    return f"{source}:{_num(confidence)}"


def summarize_components(components: Mapping[str, str]) -> Dict[str, str]:
    """Compact view for diagnostics/UI (drops empty values)."""

    return {key: value for key, value in components.items() if value}


def fingerprint_of_components(components: Mapping[str, str]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"schema": FINGERPRINT_SCHEMA, "components": dict(components)}
        )
    )


def iter_requirement_fields() -> Iterable[str]:
    return sorted(SynthesisRequirement.model_fields)
