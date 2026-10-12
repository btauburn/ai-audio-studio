"""Reproducible four-arm audio QA comparison for the casting / performance directors.

One manifest describes the same source lines rendered in four arms:

``baseline``     current voices, no performance direction
``casting``      director-suggested voices, no performance direction
``performance``  baseline voices, performance direction applied
``combined``     director-suggested voices and performance direction

The harness is a *reader*.  It never synthesizes, never calls a model, never
uses the GPU and never touches a live store.  It hashes every immutable input
before and after analysis, measures the audio that was actually supplied, builds
a deterministic blind listening ballot and scores a human review of that ballot
when one is supplied.

What it will not do:

* copy, borrow or relabel an arm's audio -- a missing arm stays missing and
  every comparison involving it is ``null``;
* invent a quality verdict -- acoustic numbers are descriptive, and the only
  gate that can say anything about preference is the human review;
* invent ASR omissions, intelligibility or voice-embedding similarity -- those
  need models this harness does not run, so they are ``null`` with a reason;
* invent latency or VRAM -- only a receipt bound to the exact audio hash can
  supply them;
* grant approval -- ``human_gate`` is a recommendation for the existing Voice
  Lab human actions, never a replacement for them.

Fixtures (artificial tones from :func:`build_synthetic_fixture`) are labeled in
the manifest and in every output, and are excluded from production readiness.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import struct
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MANIFEST_SCHEMA = "director-comparison-manifest-v1"
REPORT_SCHEMA = "director-comparison-report-v1"
BALLOT_SCHEMA = "director-blind-ballot-v1"
BALLOT_KEY_SCHEMA = "director-blind-ballot-key-v1"
GATE_SCHEMA = "director-human-gate-v1"
EVALUATOR_VERSION = "director-evaluation-v1"

ARMS: Tuple[str, ...] = ("baseline", "casting", "performance", "combined")
#: Arms that must carry a performance direction / must not.
DIRECTED_ARMS = frozenset({"performance", "combined"})
#: Arm pairs that differ only in the performance direction, so they share voices.
SAME_VOICE_PAIRS: Tuple[Tuple[str, str], ...] = (("baseline", "performance"), ("casting", "combined"))

MIN_UTTERANCES = 12
MIN_DIALOGUE_ROLES = 2
ROLE_TYPES = ("narration", "dialogue")

#: Published measurement policy.  Recorded in every report so a number can be
#: re-derived and a threshold change is visible rather than silent.
MEASUREMENT_POLICY: Dict[str, Any] = {
    "version": "audio-measures-v1",
    "window_seconds": 0.02,
    "silence_threshold_dbfs": -50.0,
    "clip_threshold": 0.999,
    "likely_clip_run_frames": 3,
    "silent_audio_peak": 1e-4,
    "long_edge_silence_seconds": 1.0,
    "tail_probe_seconds": 0.01,
    "tail_active_dbfs": -40.0,
    "min_tail_judgement_seconds": 0.25,
    "interpretation": (
        "Descriptive indicators only. Pauses can be intentional, full-scale peaks can be "
        "natural, and speech can end on non-silent audio; none of these is a quality verdict."
    ),
}

RATING_DIMENSIONS: Tuple[str, ...] = ("naturalness", "role_fit", "clarity")
RATING_MIN = 1
RATING_MAX = 5
#: Mean-score margin (1-5 scale) an arm must clear against baseline to count as
#: improved or regressed.  Recorded in the gate.
GATE_MARGIN = 0.25

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LABELS = "ABCD"

UNAVAILABLE_MEASURES: Dict[str, Any] = {
    "asr": {
        "status": "unavailable",
        "reason": "No ASR model is run by this harness; omissions and WER are not estimated.",
        "word_error_rate": None,
        "omissions": None,
        "intelligibility": None,
    },
    "acoustic_voice_embedding": {
        "status": "unavailable",
        "reason": "No speaker-embedding model is run by this harness; voice similarity is not estimated.",
        "similarity": None,
    },
}


class EvaluationError(RuntimeError):
    """A manifest, source file or output cannot be trusted; nothing was written."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details


# --------------------------------------------------------------------------- #
# Hashing / JSON
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha256(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _db(power: float) -> Optional[float]:
    """Power to dBFS; digital silence is ``None`` rather than ``-inf``."""
    if not power > 0.0:
        return None
    return _round(10.0 * math.log10(power), 3)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _fail(code: str, message: str, **details: Any) -> "EvaluationError":
    return EvaluationError(code, message, **details)


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _fail("manifest_invalid", f"{field} must be a non-empty string")
    return value


def _ident(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise _fail("manifest_invalid", f"{field} must match {_ID.pattern}", value=value)
    return value


def _hex64(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise _fail("manifest_invalid", f"{field} must be a lowercase sha256 hex digest")
    return value


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


# --------------------------------------------------------------------------- #
# WAV decoding and acoustic measures (numpy, CPU only)
# --------------------------------------------------------------------------- #


def parse_wav(data: bytes) -> Dict[str, Any]:
    """Decode a RIFF/WAVE file (PCM 8/16/24/32 or IEEE float 32/64) to float64 frames."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise _fail("audio_unreadable", "not a RIFF/WAVE file")
    notes: List[str] = []
    fmt: Optional[Tuple[int, int, int, int, int]] = None
    payload: Optional[bytes] = None
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        (size,) = struct.unpack_from("<I", data, position + 4)
        body = position + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise _fail("audio_unreadable", "fmt chunk is truncated")
            tag, channels, rate, _byte_rate, block, bits = struct.unpack_from("<HHIIHH", data, body)
            if tag == 0xFFFE:
                if size < 40 or body + 26 > len(data):
                    raise _fail("audio_unreadable", "extensible fmt chunk is truncated")
                (tag,) = struct.unpack_from("<H", data, body + 24)
            fmt = (tag, channels, rate, block, bits)
        elif chunk_id == b"data":
            available = len(data) - body
            if size > available:
                notes.append("container_truncated")
                size = available
            payload = data[body : body + size]
            break
        position = body + size + (size & 1)
    if fmt is None:
        raise _fail("audio_unreadable", "missing fmt chunk")
    if payload is None:
        raise _fail("audio_unreadable", "missing data chunk")
    tag, channels, rate, block, bits = fmt
    if channels < 1 or rate < 1:
        raise _fail("audio_unreadable", "invalid channel count or sample rate")
    if tag == 1 and bits in (8, 16, 24, 32):
        kind = "pcm"
    elif tag == 3 and bits in (32, 64):
        kind = "float"
    else:
        raise _fail("audio_unsupported_format", f"unsupported WAV encoding tag={tag} bits={bits}")
    if block != channels * bits // 8:
        raise _fail("audio_unreadable", "block alignment does not match channels and bit depth")
    frames = len(payload) // block
    if frames * block != len(payload):
        notes.append("partial_frame_ignored")
        payload = payload[: frames * block]
    if kind == "float":
        raw = np.frombuffer(payload, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif bits == 8:
        raw = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif bits == 16:
        raw = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif bits == 24:
        triples = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        value = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        raw = ((value ^ 0x800000) - 0x800000).astype(np.float64) / 8388608.0
    else:
        raw = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    return {
        "samples": raw.reshape(frames, channels),
        "sample_rate": rate,
        "channels": channels,
        "bits_per_sample": bits,
        "encoding": kind,
        "frames": frames,
        "container_notes": notes,
    }


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max())


def measure_audio(data: bytes, *, text: Optional[str] = None) -> Dict[str, Any]:
    """Measure one WAV.  Every field is a descriptive indicator, never a verdict."""
    policy = MEASUREMENT_POLICY
    decoded = parse_wav(data)
    samples: np.ndarray = decoded["samples"]
    rate = decoded["sample_rate"]
    frames = decoded["frames"]
    flags: List[Dict[str, str]] = []

    def flag(code: str, severity: str, detail: str) -> None:
        flags.append({"code": code, "severity": severity, "detail": detail})

    for note in decoded["container_notes"]:
        flag(note, "review", "The container size disagrees with the bytes present.")

    finite = np.isfinite(samples)
    non_finite = int(finite.size - int(finite.sum()))
    if non_finite:
        samples = np.where(finite, samples, 0.0)
        flag("non_finite_samples", "blocker", f"{non_finite} NaN/Inf samples were present.")

    measures: Dict[str, Any] = {
        "sample_rate": rate,
        "channels": decoded["channels"],
        "bits_per_sample": decoded["bits_per_sample"],
        "encoding": decoded["encoding"],
        "frames": frames,
        "duration_seconds": _round(frames / rate),
        "non_finite_samples": non_finite,
    }
    if frames == 0:
        flag("empty_audio", "blocker", "The file contains no audio frames.")
        measures.update(
            peak=None, peak_dbfs=None, rms_dbfs=None, clipping=_empty_clipping(), silence=None,
            tail={"indicator": "ambiguous", "reason": "no audio", "rms_dbfs_last_probe": None},
            active_chars_per_second=None,
        )
        measures["flags"] = flags
        return measures

    magnitude = np.abs(samples)
    peak = float(magnitude.max())
    mean_square = float((samples**2).mean())
    measures["peak"] = _round(peak)
    measures["peak_dbfs"] = _db(peak * peak)
    measures["rms_dbfs"] = _db(mean_square)

    # Clipping: counts and run length separate "touched full scale" from "flat-topped".
    clipped = magnitude >= policy["clip_threshold"]
    clipped_frames = clipped.any(axis=1)
    longest = _longest_run(clipped_frames)
    clipped_count = int(clipped.sum())
    if clipped_count == 0:
        indicator = "none"
    elif longest >= policy["likely_clip_run_frames"]:
        indicator = "likely"
        flag("clipping_likely", "review", f"{clipped_count} samples at full scale; longest run {longest} frames.")
    else:
        indicator = "possible"
        flag("clipping_possible", "info", f"{clipped_count} isolated full-scale samples; may be natural peaks.")
    measures["clipping"] = {
        "indicator": indicator,
        "clipped_samples": clipped_count,
        "clipped_ratio": _round(clipped_count / clipped.size, 8),
        "longest_run_frames": longest,
    }

    # Windowed silence (per-frame mean power across channels).
    window = max(1, int(round(rate * policy["window_seconds"])))
    power = (samples**2).mean(axis=1)
    starts = np.arange(0, frames, window)
    counts = np.minimum(window, frames - starts)
    window_power = np.add.reduceat(power, starts) / counts
    threshold = 10.0 ** (policy["silence_threshold_dbfs"] / 10.0)
    silent = window_power < threshold
    total_windows = int(silent.size)
    all_silent = bool(silent.all())
    lead_windows = total_windows if all_silent else int(np.argmin(silent))
    trail_windows = total_windows if all_silent else int(np.argmin(silent[::-1]))
    lead_seconds = min(frames, int(starts[lead_windows]) if lead_windows < total_windows else frames) / rate
    trail_start = frames if trail_windows == 0 else int(starts[total_windows - trail_windows])
    trail_seconds = (frames - trail_start) / rate
    interior = silent[lead_windows : total_windows - trail_windows] if not all_silent else silent[:0]
    silent_frames = int(counts[silent].sum())
    measures["silence"] = {
        "threshold_dbfs": policy["silence_threshold_dbfs"],
        "silent_ratio": _round(silent_frames / frames),
        "leading_seconds": _round(lead_seconds),
        "trailing_seconds": _round(trail_seconds),
        "longest_interior_pause_seconds": _round(_longest_run(interior) * policy["window_seconds"]),
        "all_silent": all_silent,
    }
    if all_silent or peak < policy["silent_audio_peak"]:
        flag("silent_audio", "blocker", "The audio is effectively digital silence.")
    if not all_silent and lead_seconds > policy["long_edge_silence_seconds"]:
        flag("long_leading_silence", "info", f"{lead_seconds:.2f}s of leading silence.")
    if not all_silent and trail_seconds > policy["long_edge_silence_seconds"]:
        flag("long_trailing_silence", "info", f"{trail_seconds:.2f}s of trailing silence.")

    # Tail: an indicator with honest ambiguity, not a completeness test (no ASR here).
    probe = max(1, int(round(rate * policy["tail_probe_seconds"])))
    tail_db = _db(float(power[-probe:].mean()))
    if frames / rate < policy["min_tail_judgement_seconds"] or all_silent:
        tail = {"indicator": "ambiguous", "reason": "too short or silent to judge an ending"}
    elif trail_windows >= 1:
        tail = {"indicator": "ends_in_silence", "reason": "final windows are below the silence threshold"}
    elif tail_db is not None and tail_db > policy["tail_active_dbfs"]:
        tail = {
            "indicator": "active_at_end",
            "reason": "audio is still active in the final probe; truncation is possible but natural endings can look the same",
        }
        flag("tail_active_at_end", "review", tail["reason"])
    else:
        tail = {"indicator": "low_level_at_end", "reason": "final probe is quiet but above the silence threshold"}
    tail["rms_dbfs_last_probe"] = tail_db
    measures["tail"] = tail

    active_seconds = (frames - silent_frames) / rate
    chars = len(text) if text is not None else 0
    measures["active_chars_per_second"] = _round(chars / active_seconds, 3) if chars and active_seconds > 0 else None
    measures["flags"] = flags
    return measures


def _empty_clipping() -> Dict[str, Any]:
    return {"indicator": "none", "clipped_samples": 0, "clipped_ratio": None, "longest_run_frames": 0}


# --------------------------------------------------------------------------- #
# Source file registry (hash before, verify after)
# --------------------------------------------------------------------------- #


class _Sources:
    """Reads each immutable input once, records its hash, and re-verifies it at the end."""

    def __init__(self, base_dir: Optional[Path]) -> None:
        self.base_dir = base_dir
        self.entries: Dict[str, Dict[str, Any]] = {}

    def resolve(self, raw: Any, field: str) -> Path:
        text = _text(raw, field)
        path = Path(text)
        if not path.is_absolute():
            if self.base_dir is None:
                raise _fail("base_dir_required", f"{field} is relative but the manifest has no base_dir")
            if ".." in path.parts:
                raise _fail("path_refused", f"{field} may not climb out of base_dir", value=text)
            path = self.base_dir / path
        if path.is_symlink():
            raise _fail("path_refused", f"{field} is a link and is refused", value=text)
        if not path.is_file():
            raise _fail("source_missing", f"{field} does not exist", value=text)
        return path

    def read(self, raw: Any, field: str, kind: str) -> Tuple[Path, bytes, str]:
        path = self.resolve(raw, field)
        key = os.path.normcase(str(path.resolve()))
        if key in self.entries:
            raise _fail(
                "source_shared",
                f"{field} reuses a file already referenced by {self.entries[key]['field']}; "
                "arms and receipts must each have their own bytes",
                value=str(raw),
            )
        data = path.read_bytes()
        digest = sha256_hex(data)
        self.entries[key] = {
            "field": field, "kind": kind, "path": str(path), "sha256": digest, "bytes": len(data), "resolved": key,
        }
        return path, data, digest

    def verify_unchanged(self) -> List[Dict[str, Any]]:
        changed = []
        for entry in self.entries.values():
            path = Path(entry["path"])
            now = sha256_hex(path.read_bytes()) if path.is_file() and not path.is_symlink() else None
            if now != entry["sha256"]:
                changed.append({"path": entry["path"], "before": entry["sha256"], "after": now})
        if changed:
            raise _fail("source_changed", "an immutable input changed during evaluation", changed=changed)
        return [
            {k: e[k] for k in ("kind", "path", "sha256", "bytes")}
            for e in sorted(self.entries.values(), key=lambda e: (e["kind"], e["path"]))
        ]


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #


def _validate_utterances(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) < MIN_UTTERANCES:
        raise _fail("manifest_invalid", f"at least {MIN_UTTERANCES} utterances are required")
    seen: set = set()
    utterances = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise _fail("manifest_invalid", f"utterances[{index}] must be an object")
        utterance_id = _ident(item.get("utterance_id"), f"utterances[{index}].utterance_id")
        if utterance_id in seen:
            raise _fail("manifest_invalid", f"duplicate utterance_id {utterance_id}")
        seen.add(utterance_id)
        role_type = item.get("role_type")
        if role_type not in ROLE_TYPES:
            raise _fail("manifest_invalid", f"utterances[{index}].role_type must be one of {ROLE_TYPES}")
        utterances.append(
            {
                "utterance_id": utterance_id,
                "role_id": _ident(item.get("role_id"), f"utterances[{index}].role_id"),
                "role_name": _text(item.get("role_name"), f"utterances[{index}].role_name"),
                "role_type": role_type,
                "text": _text(item.get("text"), f"utterances[{index}].text"),
            }
        )
    if not any(u["role_type"] == "narration" for u in utterances):
        raise _fail("manifest_invalid", "the line set must include narration")
    dialogue_roles = {u["role_id"] for u in utterances if u["role_type"] == "dialogue"}
    if len(dialogue_roles) < MIN_DIALOGUE_ROLES:
        raise _fail("manifest_invalid", f"the line set needs at least {MIN_DIALOGUE_ROLES} dialogue roles")
    return utterances


def _validate_engine(raw: Any, arm: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail("manifest_invalid", f"arms.{arm}.engine must be an object")
    engine = {k: raw[k] for k in sorted(raw)}
    _text(engine.get("name"), f"arms.{arm}.engine.name")
    _hex64(engine.get("runtime_sha256"), f"arms.{arm}.engine.runtime_sha256")
    return engine


def _validate_voice(raw: Any, where: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail("manifest_invalid", f"{where}.voice must be an object")
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or version == "":
        raise _fail("manifest_invalid", f"{where}.voice.version must be a string or integer")
    return {
        "voice_id": _text(raw.get("voice_id"), f"{where}.voice.voice_id"),
        "version": version,
        "identity_sha256": _hex64(raw.get("identity_sha256"), f"{where}.voice.identity_sha256"),
        **{k: raw[k] for k in ("key", "name", "origin") if isinstance(raw.get(k), str)},
    }


def _voice_identity(voice: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return (voice["voice_id"], voice["version"], voice["identity_sha256"])


def _load_receipt(
    sources: _Sources, raw: Any, where: str, *, utterance_id: str, audio_sha256: str, text_sha256: str, runtime: str
) -> Dict[str, Any]:
    """Latency and peak VRAM come from a receipt bound to this exact audio, or not at all."""
    empty = {"status": "absent", "reason": "no receipt supplied", "latency_seconds": None, "peak_vram_mb": None}
    if raw is None:
        return empty
    if not isinstance(raw, Mapping):
        raise _fail("manifest_invalid", f"{where}.receipt must be an object")
    path, data, digest = sources.read(raw.get("path"), f"{where}.receipt.path", "receipt")
    declared = raw.get("sha256")
    if declared is not None and _hex64(declared, f"{where}.receipt.sha256") != digest:
        raise _fail("receipt_mismatch", f"{where} receipt bytes do not match the declared sha256")
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail("receipt_mismatch", f"{where} receipt is not valid JSON") from exc
    if not isinstance(body, dict):
        raise _fail("receipt_mismatch", f"{where} receipt must be a JSON object")
    for field, expected in (
        ("utterance_id", utterance_id), ("audio_sha256", audio_sha256),
        ("text_sha256", text_sha256), ("runtime_sha256", runtime),
    ):
        if field in body and body[field] != expected:
            raise _fail("receipt_mismatch", f"{where} receipt {field} does not match the sample", field=field)
    result = {"status": "bound", "path": str(path), "sha256": digest}
    if body.get("audio_sha256") != audio_sha256:
        # An unbound receipt could belong to any render; it supplies nothing.
        return {**empty, "status": "unbound", "reason": "receipt does not name this audio sha256", "path": str(path),
                "sha256": digest}
    latency = _number(body.get("latency_seconds"))
    vram = _number(body.get("peak_vram_mb"))
    result.update(
        latency_seconds=_round(latency),
        peak_vram_mb=_round(vram),
        synthetic_fixture=bool(body.get("synthetic_fixture")),
    )
    result["reason"] = None if latency is not None and vram is not None else "receipt omitted or invalidated a value"
    return result


def _analyze_arm(
    arm: str, spec: Any, utterances: List[Dict[str, Any]], source_sha256: str, sources: _Sources
) -> Dict[str, Any]:
    if spec is None or spec == {}:
        return {"arm": arm, "status": "missing", "reason": "arm not supplied", "samples": []}
    if not isinstance(spec, Mapping):
        raise _fail("manifest_invalid", f"arms.{arm} must be an object")
    status = spec.get("status", "present")
    if status == "missing":
        if spec.get("samples"):
            raise _fail("manifest_invalid", f"arms.{arm} is marked missing but lists samples")
        return {"arm": arm, "status": "missing", "reason": str(spec.get("reason") or "arm marked missing"), "samples": []}
    if status != "present":
        raise _fail("manifest_invalid", f"arms.{arm}.status must be present or missing")
    engine = _validate_engine(spec.get("engine"), arm)
    raw_samples = spec.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise _fail("manifest_invalid", f"arms.{arm}.samples must be a non-empty list")
    by_id = {u["utterance_id"]: u for u in utterances}
    seen: set = set()
    samples = []
    for index, raw in enumerate(raw_samples):
        where = f"arms.{arm}.samples[{index}]"
        if not isinstance(raw, Mapping):
            raise _fail("manifest_invalid", f"{where} must be an object")
        utterance_id = raw.get("utterance_id")
        if utterance_id not in by_id:
            raise _fail("manifest_invalid", f"{where}.utterance_id is not in the line set", value=utterance_id)
        if utterance_id in seen:
            raise _fail("manifest_invalid", f"{where} duplicates {utterance_id}")
        seen.add(utterance_id)
        utterance = by_id[utterance_id]
        if raw.get("text") != utterance["text"]:
            raise _fail("text_mismatch", f"{where} text differs from the manifest line", utterance_id=utterance_id, arm=arm)
        text_hash = sha256_hex(utterance["text"].encode("utf-8"))
        if raw.get("text_sha256") not in (None, text_hash):
            raise _fail("text_mismatch", f"{where}.text_sha256 differs from the line text", utterance_id=utterance_id, arm=arm)
        if raw.get("role_id") not in (None, utterance["role_id"]):
            raise _fail("manifest_invalid", f"{where}.role_id differs from the line's role", utterance_id=utterance_id)
        provenance = raw.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("source_sha256") != source_sha256:
            raise _fail("provenance_mismatch", f"{where}.provenance.source_sha256 must equal the manifest source",
                        utterance_id=utterance_id, arm=arm)
        voice = _validate_voice(raw.get("voice"), where)
        direction_raw = raw.get("direction")
        direction = None
        if direction_raw is not None:
            if not isinstance(direction_raw, Mapping):
                raise _fail("manifest_invalid", f"{where}.direction must be an object or null")
            direction = {"direction_sha256": _hex64(direction_raw.get("direction_sha256"), f"{where}.direction.direction_sha256")}
        if arm not in DIRECTED_ARMS and direction is not None:
            raise _fail("direction_mismatch", f"{where}: the {arm} arm may not carry a performance direction",
                        utterance_id=utterance_id, arm=arm)
        declared_audio = _hex64(raw.get("audio_sha256"), f"{where}.audio_sha256")
        path, data, digest = sources.read(raw.get("audio_path"), f"{where}.audio_path", "audio")
        if digest != declared_audio:
            raise _fail("audio_hash_mismatch", f"{where} audio bytes do not match audio_sha256",
                        utterance_id=utterance_id, arm=arm, declared=declared_audio, actual=digest)
        try:
            measures = measure_audio(data, text=utterance["text"])
        except EvaluationError as exc:
            raise _fail(exc.code, f"{where}: {exc.message}", utterance_id=utterance_id, arm=arm) from exc
        receipt = _load_receipt(
            sources, raw.get("receipt"), where, utterance_id=utterance_id, audio_sha256=digest,
            text_sha256=text_hash, runtime=engine["runtime_sha256"],
        )
        samples.append(
            {
                "utterance_id": utterance_id,
                "role_id": utterance["role_id"],
                "audio_path": str(raw.get("audio_path")),
                "resolved_audio_path": str(path.resolve()),
                "audio_sha256": digest,
                "audio_bytes": len(data),
                "text_sha256": text_hash,
                "source_sha256": source_sha256,
                "voice": voice,
                "direction": direction,
                "measures": measures,
                "receipt": receipt,
                "asr": copy.deepcopy(UNAVAILABLE_MEASURES["asr"]),
                "acoustic_voice_embedding": copy.deepcopy(UNAVAILABLE_MEASURES["acoustic_voice_embedding"]),
                "human_measures": {"status": "pending"},
            }
        )
    complete = seen == set(by_id)
    if arm in DIRECTED_ARMS and complete and not any(s["direction"] for s in samples):
        raise _fail("direction_mismatch", f"the {arm} arm carries no performance direction on any line; "
                    "it would be a relabeled baseline or casting arm", arm=arm)
    order = {u["utterance_id"]: i for i, u in enumerate(utterances)}
    samples.sort(key=lambda s: order[s["utterance_id"]])
    missing = [u["utterance_id"] for u in utterances if u["utterance_id"] not in seen]
    result = {
        "arm": arm,
        "status": "complete" if complete else "incomplete",
        "reason": None if complete else "arm lacks some of the shared lines and is excluded from comparison",
        "engine": engine,
        "runtime_sha256": engine["runtime_sha256"],
        "missing_utterance_ids": missing,
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
    latencies = [s["receipt"]["latency_seconds"] for s in samples if s["receipt"].get("latency_seconds") is not None]
    vram = [s["receipt"]["peak_vram_mb"] for s in samples if s["receipt"].get("peak_vram_mb") is not None]
    return {
        "sample_count": len(samples),
        "total_duration_seconds": _round(sum(m["duration_seconds"] or 0.0 for m in measures)),
        "mean_silent_ratio": _round(sum(silence) / len(silence)) if silence else None,
        "max_peak": max(peaks) if peaks else None,
        "clipped_samples": sum(m["clipping"]["clipped_samples"] for m in measures),
        "samples_with_likely_clipping": sum(1 for m in measures if m["clipping"]["indicator"] == "likely"),
        "samples_with_tail_active": sum(1 for m in measures if m["tail"]["indicator"] == "active_at_end"),
        "non_finite_samples": sum(m["non_finite_samples"] for m in measures),
        "flag_counts": severities,
        "latency_seconds": {
            "samples_with_receipt": len(latencies),
            "total": _round(sum(latencies)) if latencies else None,
            "mean": _round(sum(latencies) / len(latencies)) if latencies else None,
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


def _check_cross_arm(arms: Mapping[str, Dict[str, Any]], utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    present = [a for a in ARMS if arms[a]["status"] in ("complete", "incomplete")]
    if len(present) > 1:
        reference = canonical_json(arms[present[0]]["engine"])
        for name in present[1:]:
            if canonical_json(arms[name]["engine"]) != reference:
                raise _fail("engine_mismatch", f"the {name} arm used a different engine/runtime than {present[0]}",
                            arms=[present[0], name])
    findings: List[Dict[str, Any]] = []
    index = {a: {s["utterance_id"]: s for s in arms[a]["samples"]} for a in present}
    for left, right in SAME_VOICE_PAIRS:
        if left in index and right in index:
            for utterance_id in index[left].keys() & index[right].keys():
                if _voice_identity(index[left][utterance_id]["voice"]) != _voice_identity(index[right][utterance_id]["voice"]):
                    raise _fail("voice_pairing_mismatch",
                                f"{left} and {right} must use the same voice for {utterance_id}",
                                arms=[left, right], utterance_id=utterance_id)
    complete = [a for a in present if arms[a]["status"] == "complete"]
    for i, left in enumerate(complete):
        for right in complete[i + 1 :]:
            left_hashes = [index[left][u["utterance_id"]]["audio_sha256"] for u in utterances]
            right_hashes = [index[right][u["utterance_id"]]["audio_sha256"] for u in utterances]
            if left_hashes == right_hashes:
                raise _fail("arm_copied", f"the {left} and {right} arms have byte-identical audio for every line; "
                            "an arm may not be copied or relabeled", arms=[left, right])
    if "baseline" in index and "casting" in index:
        changed = [
            u for u in index["baseline"].keys() & index["casting"].keys()
            if _voice_identity(index["baseline"][u]["voice"]) != _voice_identity(index["casting"][u]["voice"])
        ]
        if not changed:
            findings.append({"code": "casting_voices_unchanged",
                             "detail": "The casting arm uses the baseline voice on every shared line."})
    return {"findings": findings}


def _compare(arms: Mapping[str, Dict[str, Any]], utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = arms["baseline"]
    result: Dict[str, Any] = {
        "interpretation": "Descriptive acoustic differences only; not a quality or preference claim.",
    }
    for name in ("casting", "performance", "combined"):
        arm = arms[name]
        key = f"{name}_vs_baseline"
        if base["status"] != "complete" or arm["status"] != "complete":
            result[key] = None
            continue
        b, a = base["summary"], arm["summary"]

        def delta(path: Sequence[str]) -> Optional[float]:
            x, y = a, b
            for part in path:
                x, y = x[part], y[part]
            return None if x is None or y is None else _round(x - y)

        base_samples = {s["utterance_id"]: s for s in base["samples"]}
        rows = []
        for sample in arm["samples"]:
            ref = base_samples[sample["utterance_id"]]
            d_a, d_b = sample["measures"]["duration_seconds"], ref["measures"]["duration_seconds"]
            rows.append({
                "utterance_id": sample["utterance_id"],
                "audio_changed": sample["audio_sha256"] != ref["audio_sha256"],
                "duration_delta_seconds": _round(d_a - d_b) if d_a is not None and d_b is not None else None,
            })
        both_latency = a["latency_seconds"]["complete"] and b["latency_seconds"]["complete"]
        both_vram = a["peak_vram_mb"]["complete"] and b["peak_vram_mb"]["complete"]
        result[key] = {
            "total_duration_seconds_delta": delta(["total_duration_seconds"]),
            "mean_silent_ratio_delta": delta(["mean_silent_ratio"]),
            "max_peak_delta": delta(["max_peak"]),
            "clipped_samples_delta": a["clipped_samples"] - b["clipped_samples"],
            "blocker_flags_delta": a["flag_counts"]["blocker"] - b["flag_counts"]["blocker"],
            "review_flags_delta": a["flag_counts"]["review"] - b["flag_counts"]["review"],
            "mean_latency_seconds_delta": delta(["latency_seconds", "mean"]) if both_latency else None,
            "max_peak_vram_mb_delta": delta(["peak_vram_mb", "max"]) if both_vram else None,
            "changed_audio_utterances": sum(1 for r in rows if r["audio_changed"]),
            "unchanged_audio_utterance_ids": [r["utterance_id"] for r in rows if not r["audio_changed"]],
            "per_utterance": rows,
        }
    return result


# --------------------------------------------------------------------------- #
# Blind ballot and human review gate
# --------------------------------------------------------------------------- #


def build_blind_ballot(
    comparison_id: str, utterances: List[Dict[str, Any]], arms: Mapping[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(ballot, key)``.  The ballot names no arm; the key maps labels back.

    Only complete arms take part.  Label order per line is a pure function of the
    comparison identity, so the same inputs always give the same ballot.
    """
    taking_part = [a for a in ARMS if arms[a]["status"] == "complete"]
    salt = content_sha256({
        "comparison_id": comparison_id,
        "lines": [[u["utterance_id"], sha256_hex(u["text"].encode("utf-8"))] for u in utterances],
        "arms": taking_part,
    })[:16]
    items, key_items = [], {}
    for number, utterance in enumerate(utterances, start=1):
        utterance_id = utterance["utterance_id"]
        item_id = f"item-{number:03d}"
        order = sorted(taking_part, key=lambda a: sha256_hex(f"{salt}|{utterance_id}|{a}".encode()))
        options = []
        key_options = {}
        for label, arm in zip(_LABELS, order):
            sample = next(s for s in arms[arm]["samples"] if s["utterance_id"] == utterance_id)
            options.append({"label": label, "sample_ref": f"{item_id}-{label}"})
            key_options[label] = {"arm": arm, "audio_path": sample["resolved_audio_path"], "audio_sha256": sample["audio_sha256"]}
        items.append({
            "item_id": item_id,
            "utterance_id": utterance_id,
            "role_name": utterance["role_name"],
            "role_type": utterance["role_type"],
            "text": utterance["text"],
            "options": options,
        })
        key_items[item_id] = {"utterance_id": utterance_id, "options": key_options}
    complete = len(taking_part) == len(ARMS)
    body = {
        "schema_version": BALLOT_SCHEMA,
        "comparison_id": comparison_id,
        "status": "ready" if complete else "partial",
        "partial_reason": None if complete else "Not every arm is available; this ballot cannot satisfy the human gate.",
        "rating_scale": {"min": RATING_MIN, "max": RATING_MAX, "dimensions": list(RATING_DIMENSIONS)},
        "instructions": (
            "Listen to every option for each line without opening the key. Rate naturalness, role fit and "
            "clarity (1-5) for each label, optionally name a preferred label, and record any defects heard."
        ),
        "items": items,
    }
    ballot_id = content_sha256(body)[:32]
    ballot = {**body, "ballot_id": ballot_id}
    key = {"schema_version": BALLOT_KEY_SCHEMA, "ballot_id": ballot_id, "comparison_id": comparison_id,
           "sealed": "Do not show reviewers.", "items": key_items}
    return ballot, key


def _blocker_arms(arms: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    found = []
    for name in ARMS:
        arm = arms[name]
        if arm["status"] in ("complete", "incomplete") and arm["summary"]["flag_counts"]["blocker"]:
            found.append({"arm": name, "blockers": arm["summary"]["flag_counts"]["blocker"]})
    return found


def score_human_review(
    ballot: Mapping[str, Any], key: Mapping[str, Any], review: Any
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate a review against the ballot and aggregate per arm.  Returns ``(scores, problems)``."""
    problems: List[str] = []
    if not isinstance(review, Mapping):
        return None, ["review must be an object"]
    if review.get("ballot_id") != ballot["ballot_id"]:
        problems.append("review ballot_id does not match this ballot")
        return None, problems
    reviewer = review.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        problems.append("reviewer_id is required")
    responses = review.get("responses")
    if not isinstance(responses, list):
        return None, problems + ["responses must be a list"]
    items = {i["item_id"]: i for i in ballot["items"]}
    answered: Dict[str, Mapping[str, Any]] = {}
    for response in responses:
        item_id = response.get("item_id") if isinstance(response, Mapping) else None
        if item_id not in items:
            problems.append(f"unknown item {item_id!r}")
        elif item_id in answered:
            problems.append(f"duplicate response for {item_id}")
        else:
            answered[item_id] = response
    for item_id in items:
        if item_id not in answered:
            problems.append(f"missing response for {item_id}")
    totals: Dict[str, Dict[str, List[int]]] = {a: {d: [] for d in RATING_DIMENSIONS} for a in ARMS}
    wins = {a: 0 for a in ARMS}
    defects = {a: 0 for a in ARMS}
    for item_id, response in answered.items():
        labels = [o["label"] for o in items[item_id]["options"]]
        ratings = response.get("ratings")
        if not isinstance(ratings, Mapping) or set(ratings) != set(labels):
            problems.append(f"{item_id}: ratings must cover exactly labels {labels}")
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
        if not isinstance(listed, Mapping) or not set(listed) <= set(labels):
            problems.append(f"{item_id}: defects must be keyed by ballot labels")
            valid = False
        if not valid:
            continue
        for label in labels:
            arm = key["items"][item_id]["options"][label]["arm"]
            for dimension in RATING_DIMENSIONS:
                totals[arm][dimension].append(ratings[label][dimension])
            defects[arm] += len(listed.get(label) or [])
        if preferred is not None:
            wins[key["items"][item_id]["options"][preferred]["arm"]] += 1
    if problems:
        return None, problems
    scores: Dict[str, Any] = {}
    for arm in ARMS:
        per_dimension = {d: _round(sum(v) / len(v), 4) for d, v in totals[arm].items() if v}
        scores[arm] = {
            "dimension_means": per_dimension,
            "overall_mean": _round(sum(per_dimension.values()) / len(per_dimension), 4) if per_dimension else None,
            "preferred_count": wins[arm],
            "defects_listed": defects[arm],
            "rated_items": len(totals[arm]["naturalness"]),
        }
    return {"reviewer_id": reviewer, "arms": scores}, []


def build_human_gate(
    *, fixture: bool, ballot: Mapping[str, Any], key: Mapping[str, Any], arms: Mapping[str, Dict[str, Any]],
    review: Any,
) -> Dict[str, Any]:
    """Gate status: fixture_excluded / held / pending / supports_adoption_review / does_not_support.

    The gate is a recommendation to the existing Voice Lab human actions.  It never
    approves a voice and never starts synthesis.
    """
    gate: Dict[str, Any] = {
        "schema_version": GATE_SCHEMA,
        "ballot_id": ballot["ballot_id"],
        "margin": GATE_MARGIN,
        "approval": False,
        "note": "Recommendation only. Voice approval stays with Voice Lab casting feedback.",
        "reasons": [],
        "scores": None,
        "verdicts": None,
    }
    if fixture:
        gate.update(status="fixture_excluded", reasons=["Synthetic fixture output is excluded from readiness claims."])
        return gate
    absent = [a for a in ARMS if arms[a]["status"] != "complete"]
    if absent:
        gate.update(status="held", reasons=[f"arm {a} is {arms[a]['status']}: {arms[a]['reason']}" for a in absent])
        return gate
    blockers = _blocker_arms(arms)
    if blockers:
        gate.update(status="held", reasons=[f"technical blockers in {b['arm']} ({b['blockers']})" for b in blockers])
        return gate
    if review is None:
        gate.update(status="pending", reasons=["No human review has been recorded for this ballot."])
        return gate
    scores, problems = score_human_review(ballot, key, review)
    if problems:
        gate.update(status="held", reasons=["review invalid"] + problems[:50])
        return gate
    gate["scores"] = scores
    base = scores["arms"]["baseline"]["overall_mean"]
    verdicts = {}
    for arm in ("casting", "performance", "combined"):
        diff = scores["arms"][arm]["overall_mean"] - base
        verdicts[arm] = {
            "overall_delta": _round(diff, 4),
            "verdict": "improved" if diff >= GATE_MARGIN else "regressed" if diff <= -GATE_MARGIN else "no_clear_change",
        }
    gate["verdicts"] = verdicts
    if any(v["verdict"] == "regressed" for v in verdicts.values()):
        gate.update(status="does_not_support", reasons=["at least one arm regressed against baseline"])
    elif verdicts["combined"]["verdict"] == "improved":
        gate.update(status="supports_adoption_review",
                    reasons=["combined improved against baseline with no regressed arm"])
    else:
        gate.update(status="does_not_support", reasons=["combined did not clearly improve against baseline"])
    return gate


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def evaluate_comparison(manifest: Mapping[str, Any], *, output_dir: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Evaluate a manifest.  Reads only; with ``output_dir`` publishes exclusive result files.

    Returns the report with ``ballot``, ``ballot_key`` and ``human_gate`` attached.  The
    ballot is arm-blind; ``ballot_key`` must not be shown to reviewers.
    """
    if not isinstance(manifest, Mapping):
        raise _fail("manifest_invalid", "manifest must be an object")
    manifest = copy.deepcopy(dict(manifest))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise _fail("manifest_invalid", f"schema_version must be {MANIFEST_SCHEMA}")
    comparison_id = _ident(manifest.get("comparison_id"), "comparison_id")
    fixture_raw = manifest.get("fixture")
    if fixture_raw is not None and not (isinstance(fixture_raw, Mapping) and fixture_raw.get("is_fixture") is True):
        raise _fail("manifest_invalid", "fixture, when present, must be {'is_fixture': true, 'label': ...}")
    fixture = fixture_raw is not None
    fixture_label = _text(fixture_raw.get("label"), "fixture.label") if fixture else None
    source = manifest.get("source")
    source_sha256 = _hex64(source.get("source_sha256") if isinstance(source, Mapping) else None, "source.source_sha256")
    utterances = _validate_utterances(manifest.get("utterances"))
    arms_raw = manifest.get("arms")
    if not isinstance(arms_raw, Mapping) or set(arms_raw) - set(ARMS):
        raise _fail("manifest_invalid", f"arms may only contain {ARMS}")
    base_dir = manifest.get("base_dir")
    if base_dir is not None and not (isinstance(base_dir, str) and Path(base_dir).is_absolute()):
        raise _fail("manifest_invalid", "base_dir must be an absolute path")
    sources = _Sources(Path(base_dir) if base_dir else None)
    review = manifest.pop("human_review", None)
    manifest_sha256 = content_sha256(manifest)

    # Engine identity is compared before any audio is read, so a mismatch is reported as such
    # rather than as a downstream receipt or hash disagreement.
    declared_engines = {
        name: canonical_json(_validate_engine(spec.get("engine"), name))
        for name, spec in arms_raw.items()
        if isinstance(spec, Mapping) and spec.get("status", "present") == "present"
    }
    if len(set(declared_engines.values())) > 1:
        raise _fail("engine_mismatch", "arms were rendered by different engines/runtimes and cannot be compared",
                    arms=sorted(declared_engines))

    arms = {name: _analyze_arm(name, arms_raw.get(name), utterances, source_sha256, sources) for name in ARMS}
    cross = _check_cross_arm(arms, utterances)
    comparisons = _compare(arms, utterances)
    ballot, key = build_blind_ballot(comparison_id, utterances, arms)
    gate = build_human_gate(fixture=fixture, ballot=ballot, key=key, arms=arms, review=review)
    if review is not None:
        gate["human_review_sha256"] = content_sha256(review)
    integrity = sources.verify_unchanged()

    roles = {}
    for u in utterances:
        roles.setdefault(u["role_id"], {"role_id": u["role_id"], "role_name": u["role_name"], "role_type": u["role_type"], "lines": 0})
        roles[u["role_id"]]["lines"] += 1
    statuses = {a: arms[a]["status"] for a in ARMS}
    all_complete = all(s == "complete" for s in statuses.values())
    warnings = [f"arm {a} is {arms[a]['status']}: {arms[a]['reason']}" for a in ARMS if arms[a]["status"] != "complete"]
    warnings += [f["detail"] for f in cross["findings"]]
    report = {
        "schema_version": REPORT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "comparison_id": comparison_id,
        "manifest_sha256": manifest_sha256,
        "fixture": {"is_fixture": fixture, "label": fixture_label},
        "production_eligible": False if fixture else all_complete,
        "production_note": (
            "Synthetic fixture: excluded from production readiness and selection claims." if fixture else
            "Acoustic measures never establish approval; see human_gate."
        ),
        "status": "complete" if all_complete else "incomplete",
        "arm_status": statuses,
        "source": {"source_sha256": source_sha256},
        "lines": {
            "count": len(utterances),
            "roles": list(roles.values()),
            "utterances": [
                {"utterance_id": u["utterance_id"], "role_id": u["role_id"], "role_type": u["role_type"],
                 "text_sha256": sha256_hex(u["text"].encode("utf-8")), "characters": len(u["text"])}
                for u in utterances
            ],
        },
        "arms": arms,
        "comparisons": comparisons,
        "findings": cross["findings"],
        "unavailable_measures": copy.deepcopy(UNAVAILABLE_MEASURES),
        "measurement_policy": MEASUREMENT_POLICY,
        "source_integrity": {"unchanged": True, "files": integrity},
        "ballot_ref": {"ballot_id": ballot["ballot_id"], "status": ballot["status"], "items": len(ballot["items"]),
                       "key_sha256": content_sha256(key)},
        "warnings": warnings,
    }
    report["evaluation_id"] = content_sha256(report)[:32]

    result = {**report, "ballot": ballot, "ballot_key": key, "human_gate": gate}
    if output_dir is not None:
        result["outputs"] = _publish_outputs(Path(output_dir), report, ballot, key, gate, review is not None, sources)
    return result


# --------------------------------------------------------------------------- #
# Exclusive output
# --------------------------------------------------------------------------- #


def _publish_file(path: Path, payload: bytes) -> str:
    """Create ``path`` exclusively.  Identical existing bytes are verified reuse; anything else conflicts."""
    if path.is_symlink():
        raise _fail("output_conflict", "output file is a link", path=str(path))
    if path.exists():
        if path.read_bytes() == payload:
            return "reused"
        raise _fail("output_conflict", "a different result already exists at this path; use a new output directory",
                    path=str(path))
    stage = path.with_name(f".{path.name}.{uuid.uuid4().hex}.stage")
    with open(stage, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(stage, path)
        outcome = "created"
    except FileExistsError:
        outcome = "reused" if path.read_bytes() == payload else None
        if outcome is None:
            raise _fail("output_conflict", "a different result appeared at this path", path=str(path))
    except OSError:
        with open(path, "xb") as handle:  # filesystem without hard links
            handle.write(payload)
        outcome = "created"
    finally:
        try:
            stage.unlink()
        except OSError:
            pass
    return outcome


def _publish_outputs(
    directory: Path, report: Dict[str, Any], ballot: Dict[str, Any], key: Dict[str, Any], gate: Dict[str, Any],
    has_review: bool, sources: _Sources,
) -> Dict[str, Any]:
    if directory.is_symlink():
        raise _fail("output_conflict", "output directory is a link", path=str(directory))
    resolved = directory.resolve()
    for entry in sources.entries.values():
        if Path(entry["path"]).resolve().is_relative_to(resolved):
            raise _fail("output_conflict", "output directory contains immutable input files", path=str(directory))
    files = {
        "evaluation.json": _pretty(report),
        "ballot.json": _pretty(ballot),
        "ballot_key.json": _pretty(key),
    }
    if has_review:
        files["human_gate.json"] = _pretty(gate)
    for name, payload in files.items():  # check every conflict before writing anything
        target = directory / name
        if target.exists() and (target.is_symlink() or target.read_bytes() != payload):
            raise _fail("output_conflict", "a different result already exists at this path; use a new output directory",
                        path=str(target))
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in files.items():
        status = _publish_file(directory / name, payload)
        written.append({"name": name, "sha256": sha256_hex(payload), "status": status})
    return {"directory": str(directory), "files": written}


def stage_blind_audio(result: Mapping[str, Any], destination: os.PathLike) -> List[Dict[str, Any]]:
    """Copy ballot audio to arm-blind file names for a reviewer player.  Hash-verified, exclusive."""
    key = result["ballot_key"]
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    for item_id, item in key["items"].items():
        for label, option in item["options"].items():
            data = Path(option["audio_path"]).read_bytes()
            if sha256_hex(data) != option["audio_sha256"]:
                raise _fail("source_changed", "staged audio no longer matches its recorded hash", item=item_id)
            target = directory / f"{item_id}-{label}.wav"
            staged.append({"sample_ref": target.stem, "file": str(target), "status": _publish_file(target, data)})
    return staged


# --------------------------------------------------------------------------- #
# Labeled CPU-only artificial fixture
# --------------------------------------------------------------------------- #

FIXTURE_LABEL = "SYNTHETIC FIXTURE - artificial tones, not speech; excluded from production readiness"
_FIXTURE_ROLES = (
    ("narrator", "Narrator", "narration"),
    ("ada", "Ada", "dialogue"),
    ("bo", "Bo", "dialogue"),
)


def _tone_wav(
    frequency: float, seconds: float, amplitude: float, *, rate: int = 16000, clip: bool = False, tail_silence: float = 0.1
) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    envelope = np.minimum(1.0, np.minimum(t / 0.05, (seconds - t) / 0.05))
    signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    if clip:
        signal = np.clip(signal * 4.0, -1.0, 1.0)
    signal = np.concatenate([signal, np.zeros(int(rate * tail_silence))])
    pcm = np.clip(np.round(signal * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def build_synthetic_fixture(
    directory: os.PathLike, *, omit_arms: Iterable[str] = (), utterance_count: int = MIN_UTTERANCES
) -> Dict[str, Any]:
    """Write artificial tone WAVs and receipts under ``directory`` and return a labeled manifest.

    Nothing here is speech or a model result; the receipts' latency/VRAM are fake and flagged.
    Files are created exclusively.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    omit = set(omit_arms)
    runtime = sha256_hex(b"synthetic-fixture-runtime")
    source_sha = sha256_hex(b"synthetic-fixture-source")
    utterances = []
    for n in range(utterance_count):
        role_id, role_name, role_type = _FIXTURE_ROLES[n % len(_FIXTURE_ROLES)]
        utterances.append({
            "utterance_id": f"fx-{n + 1:02d}", "role_id": role_id, "role_name": role_name, "role_type": role_type,
            "text": f"Synthetic fixture line {n + 1:02d} for the {role_name} role.",
        })
    arms: Dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        if arm in omit:
            arms[arm] = {"status": "missing", "reason": "omitted from the synthetic fixture"}
            continue
        samples = []
        for n, utterance in enumerate(utterances):
            cast = arm in ("casting", "combined")
            directed = arm in DIRECTED_ARMS
            voice_tag = "cast" if cast else "base"
            frequency = 180.0 + 40.0 * (n % 3) + (60.0 if cast else 0.0)
            seconds = 0.6 + 0.02 * n + (0.1 if directed else 0.0)
            audio = _tone_wav(frequency, seconds, 0.5)
            audio_path = root / f"{arm}-{utterance['utterance_id']}.wav"
            _publish_file(audio_path, audio)
            audio_sha = sha256_hex(audio)
            receipt = {
                "synthetic_fixture": True, "utterance_id": utterance["utterance_id"], "audio_sha256": audio_sha,
                "runtime_sha256": runtime, "latency_seconds": 0.5 + arm_index * 0.1, "peak_vram_mb": 1000.0 + arm_index,
            }
            receipt_path = root / f"{arm}-{utterance['utterance_id']}.receipt.json"
            _publish_file(receipt_path, _pretty(receipt))
            samples.append({
                "utterance_id": utterance["utterance_id"],
                "text": utterance["text"],
                "audio_path": audio_path.name,
                "audio_sha256": audio_sha,
                "provenance": {"source_sha256": source_sha},
                "voice": {
                    "voice_id": f"fixture-{voice_tag}-{utterance['role_id']}", "version": 1,
                    "identity_sha256": sha256_hex(f"fixture-{voice_tag}-{utterance['role_id']}".encode()),
                },
                "direction": {"direction_sha256": sha256_hex(f"fixture-direction-{utterance['utterance_id']}".encode())}
                if directed else None,
                "receipt": {"path": receipt_path.name},
            })
        arms[arm] = {"status": "present", "engine": {"name": "synthetic-fixture-engine", "runtime_sha256": runtime},
                     "samples": samples}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "comparison_id": "synthetic-fixture",
        "fixture": {"is_fixture": True, "label": FIXTURE_LABEL},
        "base_dir": str(root.resolve()),
        "source": {"source_sha256": source_sha},
        "utterances": utterances,
        "arms": arms,
    }
