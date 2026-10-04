"""Renderer seam and chunked-utterance reassembly (Phase 8 / Phase 10).

The studio already has one provider per engine.  This module keeps that intact
and adds a thin, replaceable seam above it:

* :class:`UtteranceRenderer` — the contract yovoice (or any other provider) can
  implement without touching the take/file/spoken-text lifecycle;
* :class:`BackendRenderer` — the adapter over the existing
  ``TTSProviderFactory`` providers, including chunk planning and reassembly;
* :class:`FixtureRenderer` — deterministic, no-model rendering for tests and
  for fixtures that must never be presented as a real engine run.

Invariant enforced here: **one listener-visible utterance yields exactly one
Take**.  Chunked generation is an internal implementation detail; its parts are
reassembled into a single audio file before a Take is ever built.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence

from dubbing_service.app.core.audio_validation import (
    normalize_audio_extension,
    validate_decodable_audio,
)
from dubbing_service.app.models.audiobook import ChunkRecord
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.providers.tts import TTSProvider, TTSProviderFactory

# Chunking exists only to work around engine input limits.  Requests at or
# below this size are never chunked, so short utterances are not fragmented by
# the chunker.
CHUNK_TRIGGER_CHARS = 400
CHUNK_MAX_CHARS = 220
CHUNK_MIN_TRAILING_CHARS = 40
CHUNK_POLICY_VERSION = "sentence-greedy-v2"
_CHUNK_BREAK_CHARS = ("。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " ")


def new_take_id() -> str:
    return uuid.uuid4().hex[:16]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_long_piece(piece: str, limit: int) -> List[str]:
    if len(piece) <= limit:
        return [piece]
    parts: List[str] = []
    remaining = piece
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = -1
        for index in range(len(window) - 1, max(0, limit // 3), -1):
            if window[index] in _CHUNK_BREAK_CHARS:
                cut = index + 1
                break
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    if len(parts) > 1 and len(parts[-1]) < min(CHUNK_MIN_TRAILING_CHARS, limit):
        combined = parts[-2] + parts[-1]
        lower = max(1, len(combined) - limit)
        upper = min(limit, len(combined) - min(CHUNK_MIN_TRAILING_CHARS, limit))
        cut = -1
        for index in range(upper - 1, lower - 1, -1):
            if combined[index] in _CHUNK_BREAK_CHARS:
                cut = index + 1
                break
        if cut <= 0:
            cut = upper
        parts[-2:] = [combined[:cut], combined[cut:]]
    return parts


def plan_chunks(
    text: str,
    *,
    trigger_chars: int = CHUNK_TRIGGER_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> List[str]:
    """Return ``[text]`` for short input, or sentence-aligned parts."""

    if not text:
        return [""]
    if len(text) <= trigger_chars:
        return [text]

    expanded = _split_long_piece(text, max_chars)
    if len(expanded) == 1:
        return [text]
    return [piece for piece in expanded if piece]


class ReassemblyError(RuntimeError):
    """Chunked audio could not be reassembled into one valid file."""


def _annotate_chunk_offsets(
    chunks: Sequence[RenderedChunk], parts: Sequence[str]
) -> None:
    """Record where each chunk's text sits in the utterance text.

    Offsets come from the parts that were actually rendered, never from a guess
    made after the fact.
    """

    position = 0
    for chunk, part in zip(chunks, parts):
        chunk.text_start = position
        chunk.text_end = position + len(part)
        position = chunk.text_end


@dataclass
class RenderRequest:
    project_id: str
    segment: Segment
    role: Role
    text: str
    engine_id: str
    backend: str
    requirement_fingerprint: str
    requirement_components: dict = field(default_factory=dict)
    idempotency_key: str = ""
    synthesis_text: Optional[str] = None
    take_id: Optional[str] = None
    language: Optional[str] = None
    # Resolved language for this transaction, decided once by the caller.
    # ``None`` means nothing declared a language; the provider then applies its
    # own configured default exactly as before.  When it is set, it is the value
    # the engine is given, so the Take that records it records what happened.


@dataclass
class RenderedChunk:
    index: int
    text: str
    audio_path: str
    audio_sha256: str
    duration_ms: int = 0
    text_start: int = 0
    text_end: int = 0
    source: Literal["provider", "chunk"] = "provider"
    provider_calls: int = 1
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    def as_record(self, *, audio_start_ms: int = 0) -> ChunkRecord:
        """Chunk provenance with text and audio offsets, never a guessed split."""

        return ChunkRecord(
            index=self.index,
            text_start=self.text_start,
            text_end=self.text_end,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_start_ms + max(0, self.duration_ms),
            chunk_sha256=self.audio_sha256,
        )


@dataclass
class RenderOutcome:
    success: bool
    chunks: List[RenderedChunk] = field(default_factory=list)
    audio_path: Optional[str] = None
    audio_sha256: Optional[str] = None
    duration_ms: int = 0
    provider_calls: int = 0
    chunked: bool = False
    error: Optional[str] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, Any] = field(default_factory=dict)

    def chunk_map(self) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        offset = 0
        for chunk in self.chunks:
            record = chunk.as_record(audio_start_ms=offset)
            offset = record.audio_end_ms
            records.append(record)
        return records

    def total_chars(self) -> int:
        return sum(len(chunk.text) for chunk in self.chunks)


class UtteranceRenderer(Protocol):
    """Contract for any engine/provider adapter (indextts, cosyvoice, yovoice)."""

    renderer_id: str

    def render(self, request: RenderRequest) -> RenderOutcome:  # pragma: no cover - protocol
        ...


def validate_reassembly(request_text: str, outcome: RenderOutcome) -> None:
    """Enforce the one-utterance-one-Take invariant for a chunked render."""

    if not outcome.chunked:
        if outcome.chunks and outcome.chunks[0].text != request_text:
            raise ReassemblyError("单块渲染的文本与请求文本不一致")
        return
    if len(outcome.chunks) < 2:
        raise ReassemblyError("chunked 渲染必须包含至少两个分块")
    if "".join(chunk.text for chunk in outcome.chunks) != request_text:
        raise ReassemblyError("分块文本拼接后与请求文本不一致")
    if outcome.audio_path is None:
        raise ReassemblyError("chunked 渲染没有产出单一音频文件")


def _concat_with_ffmpeg(paths: Sequence[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = destination.with_suffix(".concat.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for path in paths:
            escaped = str(Path(path).resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-ar",
        "24000",
        "-ac",
        "1",
        str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - env specific
        raise ReassemblyError(f"ffmpeg 拼接失败: {exc}") from exc
    finally:
        listing.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise ReassemblyError(f"ffmpeg 拼接失败: {completed.stderr.strip()[:400]}")


def _probe_duration_ms(path: Path | str) -> int:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - env specific
        return 0
    try:
        return int(float(probe.stdout.strip()) * 1000)
    except (TypeError, ValueError):
        return 0


def _silence_wav(destination: Path, seconds: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{seconds:.3f}",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise ReassemblyError(f"ffmpeg 生成音频失败: {completed.stderr.strip()[:400]}")


def _tone_wav(
    destination: Path,
    seconds: float,
    *,
    frequency: float = 220.0,
    amplitude: float = 0.35,
    rate: int = 24000,
) -> None:
    """Deterministic audible PCM WAV, written without external tools.

    Fixture audio must be audible: a silent placeholder would make every
    downstream QA and coverage gate fail and force callers to disable checks.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(rate * max(0.05, seconds)))
    samples = bytearray()
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * index / rate))
        samples += struct.pack("<h", value)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(samples))


@dataclass
class BackendRenderer:
    """Adapter over the existing per-engine providers."""

    backend: str
    engine_id: str = ""
    renderer_id: str = "backend-renderer"
    provider: Optional[TTSProvider] = None
    project_root: Optional[Path] = None
    concat: Callable[[Sequence[str], Path], None] = _concat_with_ffmpeg
    probe: Callable[[Path | str], int] = _probe_duration_ms
    validate_audio: bool = True

    def _get_provider(self) -> TTSProvider:
        if self.provider is None:
            from .runtime_binding import canonical_backend
            self.provider = TTSProviderFactory.create(canonical_backend(self.backend))
            if canonical_backend(self.backend) == "qwen3tts" and self.project_root is not None:
                from .project_qwen_runtime import project_profile
                self.provider._project_runtime_profile = project_profile(self.project_root.name)
        return self.provider

    def runtime_contract(self):
        if self.backend not in {"qwen3tts", "qwen3tts_voicedesign"}:
            return None
        provider = self._get_provider()
        if not callable(getattr(provider, "runtime_identity", None)):
            return None  # Injected local test providers have no remote Qwen worker.
        from .runtime_binding import digest, RuntimeBindingError, RENDERER_IMPLEMENTATION_SHA256
        worker = provider.runtime_identity()
        contract = {"worker": worker, "renderer_id": self.renderer_id, "engine_id": self.engine_id,
            "renderer_implementation_sha256": RENDERER_IMPLEMENTATION_SHA256,
            "chunking": {"version": CHUNK_POLICY_VERSION, "trigger_chars": CHUNK_TRIGGER_CHARS,
                         "max_chars": CHUNK_MAX_CHARS, "min_trailing_chars": CHUNK_MIN_TRAILING_CHARS}}
        binding = {"contract": contract, "worker": worker, "sha256": digest(contract)}
        prior = getattr(self, "_bound_runtime", None)
        if prior is not None and prior != binding:
            raise RuntimeBindingError("Worker changed within the renderer job; no mixed-runtime output is allowed")
        self._bound_runtime = binding
        provider._expected_runtime_identity = worker
        return binding

    def _chunk_target(self, request: RenderRequest, index: int) -> Path:
        if self.project_root is None:
            raise ReassemblyError("backend 渲染需要 project_root 以持久化分块音频")
        take = request.take_id or new_take_id()
        target = (
            Path(self.project_root)
            / "audio"
            / "segments"
            / "chunks"
            / take
            / f"chunk_{index:03d}.wav"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _synthesize_chunk(
        self, request: RenderRequest, text: str, index: int
    ) -> RenderedChunk:
        provider = self._get_provider()
        self.runtime_contract()
        chunk_segment = request.segment.model_copy(deep=True)
        chunk_segment.text = text
        # The language was resolved once for the whole transaction.  Presenting
        # it here is what makes the Take's recorded language the value the engine
        # actually received, instead of the provider re-deriving its own.
        if request.language is not None:
            chunk_segment.qwen3_language = request.language
        provider_started = time.perf_counter()
        result = provider.synthesize_segment(
            request.project_id, request.role, chunk_segment
        )
        provider_http_seconds = time.perf_counter() - provider_started
        if not result.success or not result.audio_path:
            message = "TTS 未返回音频路径" if result.success else "TTS 调用失败"
            detail = result.raw.get("error") if isinstance(result.raw, dict) else None
            raise ReassemblyError(f"{message}: {detail or request.segment.id}")

        source = Path(result.audio_path)
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        if not source.is_file():
            raise ReassemblyError(f"TTS 生成的音频不存在: {source}")

        destination = self._chunk_target(request, index).with_suffix(
            normalize_audio_extension(source.suffix)
        )
        file_copy_started = time.perf_counter()
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        identity = request.role.voice_identity if request.role else None
        performance_applied = False
        if identity and identity.mechanism == "qwen_reference":
            from .performance import apply_scene_audio
            performance_applied = apply_scene_audio(destination, pace=request.segment.playback_speed,
                                                    volume=request.segment.volume_gain)
        file_copy_seconds = time.perf_counter() - file_copy_started
        validation_started = time.perf_counter()
        if self.validate_audio:
            validate_decodable_audio(destination)
        validation_seconds = time.perf_counter() - validation_started
        facts_started = time.perf_counter()
        audio_sha256 = sha256_file(destination)
        duration_ms = self.probe(destination)
        facts_seconds = time.perf_counter() - facts_started
        raw = result.raw if isinstance(result.raw, dict) else {}
        provider_metadata = {
            key: raw[key]
            for key in (
                "request_id",
                "utterance_id",
                "execution_profile",
                "worker_runtime_identity",
                "stage_timings",
                "generation",
                "generation_settings",
                "inference_policy_revision",
                "identity_id",
                "effective_speaker",
                "provider_voice_resolution",
                "prompt_sha256",
                "unsupported_performance_fields",
                "performance",
            )
            if key in raw
        }
        provider_metadata["transport_and_renderer"] = {
            "schema": "backend-renderer-stage-timings-v1",
            "clock": "time.perf_counter",
            "exclusive": {
                "provider_http_round_trip_seconds": round(provider_http_seconds, 6),
                "immutable_file_copy_seconds": round(file_copy_seconds, 6),
                "audio_validation_seconds": round(validation_seconds, 6),
                "hash_and_duration_probe_seconds": round(facts_seconds, 6),
            },
            "provider_http_round_trip_contains": [
                "service_queue",
                "service_generation",
                "service_file_write",
                "local_http_transport",
            ],
        }
        provider_metadata["scene_audio_applied"] = performance_applied
        return RenderedChunk(
            index=index,
            text=text,
            audio_path=str(destination),
            audio_sha256=audio_sha256,
            duration_ms=duration_ms,
            source="chunk",
            provider_calls=1,
            generation_metadata=provider_metadata,
        )

    def render(self, request: RenderRequest) -> RenderOutcome:
        render_started = time.perf_counter()
        identity = request.role.voice_identity if request.role else None
        if identity and identity.mechanism != "fixture" and self.backend != identity.backend:
            return RenderOutcome(success=False, error="Locked character identity requires its recorded backend; propose a new identity before changing engines")
        parts = plan_chunks(request.text)
        chunked = len(parts) > 1
        try:
            chunks: List[RenderedChunk] = []
            for index, part in enumerate(parts):
                chunks.append(self._synthesize_chunk(request, part, index))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed render
            return RenderOutcome(
                success=False,
                chunks=[],
                provider_calls=len(parts),
                chunked=chunked,
                error=str(exc),
                generation_metadata={"retryable": False} if getattr(self, "_bound_runtime", None) else {},
            )

        _annotate_chunk_offsets(chunks, parts)

        if not chunked:
            single = chunks[0]
            single.source = "provider"
            outcome = RenderOutcome(
                success=True,
                chunks=chunks,
                audio_path=single.audio_path,
                audio_sha256=single.audio_sha256,
                duration_ms=single.duration_ms,
                provider_calls=1,
                chunked=False,
                generation_metadata={
                    "schema": "audiobook-generation-metadata-v1",
                    "scene_audio_applied": single.generation_metadata.get("scene_audio_applied", False),
                    "provider_calls": [single.generation_metadata],
                    "chunking_policy": {
                        "version": CHUNK_POLICY_VERSION,
                        "trigger_chars": CHUNK_TRIGGER_CHARS,
                        "max_chars": CHUNK_MAX_CHARS,
                        "chunk_count": 1,
                    },
                },
                stage_timings={
                    "schema": "backend-renderer-stage-timings-v1",
                    "clock": "time.perf_counter",
                    "inclusive": {
                        "renderer_seconds": round(time.perf_counter() - render_started, 6)
                    },
                },
            )
            validate_reassembly(request.text, outcome)
            return outcome

        destination = self._chunk_target(request, 0).parent / (
            f"{(request.take_id or new_take_id())}.wav"
        )
        concat_started = time.perf_counter()
        try:
            self.concat([chunk.audio_path for chunk in chunks], destination)
            if self.validate_audio:
                validate_decodable_audio(destination)
        except Exception as exc:  # noqa: BLE001
            return RenderOutcome(
                success=False,
                chunks=chunks,
                provider_calls=len(chunks),
                chunked=True,
                error=f"分块重建失败: {exc}",
            )
        concat_seconds = time.perf_counter() - concat_started

        outcome = RenderOutcome(
            success=True,
            chunks=chunks,
            audio_path=str(destination),
            audio_sha256=sha256_file(destination),
            duration_ms=self.probe(destination),
            provider_calls=len(chunks),
            chunked=True,
            generation_metadata={
                "schema": "audiobook-generation-metadata-v1",
                "scene_audio_applied": any(c.generation_metadata.get("scene_audio_applied") for c in chunks),
                "provider_calls": [chunk.generation_metadata for chunk in chunks],
                "batching_policy": "sequential-independent-provider-calls",
                "chunking_policy": {
                    "version": CHUNK_POLICY_VERSION,
                    "trigger_chars": CHUNK_TRIGGER_CHARS,
                    "max_chars": CHUNK_MAX_CHARS,
                    "chunk_count": len(chunks),
                },
            },
            stage_timings={
                "schema": "backend-renderer-stage-timings-v1",
                "clock": "time.perf_counter",
                "inclusive": {
                    "renderer_seconds": round(time.perf_counter() - render_started, 6)
                },
                "exclusive": {"reassembly_seconds": round(concat_seconds, 6)},
            },
        )
        validate_reassembly(request.text, outcome)
        return outcome


@dataclass
class FixtureRenderer:
    """Deterministic, engine-free renderer used by tests and fixture runs.

    It never claims to be a real engine: ``renderer_id`` is ``fixture`` and the
    recorded engine stays ``fixture`` so a fixture Take can never be mistaken
    for engine output.
    """

    project_root: Path
    renderer_id: str = "fixture"
    engine_id: str = "fixture"
    ms_per_char: float = 55.0
    concat: Callable[[Sequence[str], Path], None] = _concat_with_ffmpeg
    probe: Callable[[Path | str], int] = _probe_duration_ms

    def _write_chunk(self, request: RenderRequest, text: str, index: int) -> RenderedChunk:
        seconds = max(0.05, len(text) * self.ms_per_char / 1000.0)
        take = request.take_id or new_take_id()
        target = (
            Path(self.project_root)
            / "audio"
            / "segments"
            / "chunks"
            / take
            / f"chunk_{index:03d}.wav"
        )
        _tone_wav(target, seconds)
        return RenderedChunk(
            index=index,
            text=text,
            audio_path=str(target),
            audio_sha256=sha256_file(target),
            duration_ms=self.probe(target),
            source="chunk",
            provider_calls=0,
        )

    def render(self, request: RenderRequest) -> RenderOutcome:
        parts = plan_chunks(request.text)
        chunked = len(parts) > 1
        chunks = [self._write_chunk(request, part, index) for index, part in enumerate(parts)]
        _annotate_chunk_offsets(chunks, parts)
        if not chunked:
            single = chunks[0]
            single.source = "provider"
            outcome = RenderOutcome(
                success=True,
                chunks=chunks,
                audio_path=single.audio_path,
                audio_sha256=single.audio_sha256,
                duration_ms=single.duration_ms,
                provider_calls=0,
                chunked=False,
            )
            validate_reassembly(request.text, outcome)
            return outcome

        take = request.take_id or new_take_id()
        destination = (
            Path(self.project_root) / "audio" / "segments" / "chunks" / take / f"{take}.wav"
        )
        self.concat([chunk.audio_path for chunk in chunks], destination)
        outcome = RenderOutcome(
            success=True,
            chunks=chunks,
            audio_path=str(destination),
            audio_sha256=sha256_file(destination),
            duration_ms=self.probe(destination),
            provider_calls=0,
            chunked=True,
        )
        validate_reassembly(request.text, outcome)
        return outcome


def reassemble_with_retry(
    renderer: UtteranceRenderer,
    request: RenderRequest,
    *,
    attempts: int = 2,
) -> RenderOutcome:
    """Retry a failed chunked render once before surfacing the failure."""

    outcome = renderer.render(request)
    attempt = 1
    while not outcome.success and attempt < attempts and outcome.generation_metadata.get("retryable") is not False:
        attempt += 1
        retry = renderer.render(request)
        if retry.success:
            return retry
        outcome = retry
    return outcome


def temporary_chunk_dir(prefix: str = "aas-chunks-") -> Path:
    """Scratch directory used when a renderer must not persist chunk audio."""

    return Path(tempfile.mkdtemp(prefix=prefix))
