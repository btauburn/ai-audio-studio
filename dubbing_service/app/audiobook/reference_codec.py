"""Bounded local decoding for immutable reference crops.

Existing soundfile inputs retain their exact sample path. WebM/Opus is an
explicit additional path using installed FFmpeg, with binary/version provenance
revalidated by the worker. No network protocols, model downloads or processing
other than decoding, contiguous sample selection and channel mean are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

MAX_ORIGINAL_BYTES = 32_000_000


@dataclass
class DecodedReference:
    wave: np.ndarray
    sample_rate: int
    channels: int
    duration: float
    decoder: dict | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(args: list[str], *, max_stdout_bytes: int = 512_000,
         timeout_seconds: float = 45) -> bytes:
    # Drain both pipes concurrently with hard byte bounds. A malformed container
    # must not grow diagnostics or probe output unbounded before validation.
    exceeded = Event()
    process = None

    def read_bounded(stream, limit):
        chunks, size = [], 0
        while block := stream.read(65536):
            size += len(block)
            if size > limit:
                exceeded.set()
                try:
                    process.kill()
                except OSError:
                    pass
                return b""
            chunks.append(block)
        return b"".join(chunks)

    try:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with ThreadPoolExecutor(max_workers=2) as pool:
            stdout = pool.submit(read_bounded, process.stdout, max_stdout_bytes)
            stderr = pool.submit(read_bounded, process.stderr, 64_000)
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise
            result = stdout.result(timeout=5)
            stderr.result(timeout=5)
        if exceeded.is_set():
            raise ValueError("Reference decoder exceeded its output byte bound")
        if process.returncode:
            raise ValueError("Installed reference decoder could not decode the recording")
        return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Installed reference decoder unavailable or timed out") from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _binary(name: str) -> Path:
    configured = os.environ.get("VOICE_LAB_" + name.upper())
    found = configured or shutil.which(name)
    if not found and os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        installed = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Links" / (name + ".exe")
        if installed.is_file():
            found = str(installed)
    if not found or not Path(found).is_file():
        raise ValueError("WebM reference needs existing local " + name)
    return Path(found).resolve()


def _webm(path: Path, start: float, end: float, expected_decoder: dict | None) -> DecodedReference:
    with path.open("rb") as stream:
        if stream.read(4) != b"\x1a\x45\xdf\xa3":
            raise ValueError("Unsupported reference recording format")
    ffmpeg, ffprobe = _binary("ffmpeg"), _binary("ffprobe")
    decoder = {"operation": "webm_opus_decode_float32", "decoder": "ffmpeg",
               "version": _run([str(ffmpeg), "-version"]).decode("utf-8", "replace").splitlines()[0],
               "ffmpeg_sha256": _sha256(ffmpeg), "ffprobe_sha256": _sha256(ffprobe)}
    if expected_decoder is not None and decoder != expected_decoder:
        raise ValueError("Reference decoder provenance changed")
    raw = _run([str(ffprobe), "-v", "error", "-protocol_whitelist", "file,pipe",
                "-show_entries", "stream=codec_type,codec_name,sample_rate,channels:format=format_name,duration",
                "-of", "json", str(path)])
    try:
        probe = json.loads(raw)
        streams = [row for row in probe["streams"] if row.get("codec_type") == "audio"]
        if len(streams) != 1 or streams[0]["codec_name"] != "opus":
            raise ValueError("WebM reference needs exactly one Opus audio stream")
        rate, channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
        duration = float(probe["format"]["duration"])
        if "webm" not in probe["format"]["format_name"].split(","):
            raise ValueError("Reference is not a WebM container")
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("Reference recording metadata is invalid") from exc
    if channels not in (1, 2) or not 8000 <= rate <= 96000 or not math.isfinite(duration) or not 0 < duration <= 600:
        raise ValueError("Reference recording channels, rate or duration unsuitable")
    if end > duration + 1 / rate:
        raise ValueError("Reference selection exceeds the original")
    first, last = round(start * rate), round(end * rate)
    raw = _run([str(ffmpeg), "-v", "error", "-nostdin", "-threads", "1",
                "-protocol_whitelist", "file,pipe", "-t", str(end + 0.5), "-i", str(path),
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-af",
                f"atrim=start_sample={first}:end_sample={last}",
                "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
               max_stdout_bytes=(last - first) * channels * 4)
    if len(raw) != (last - first) * channels * 4:
        raise ValueError("Reference decoder returned an incomplete crop")
    wave = np.frombuffer(raw, dtype="<f4").copy()
    if channels == 2:
        wave = wave.reshape(-1, 2)
    return DecodedReference(wave, rate, channels, duration, decoder)


def _mp3(path: Path, start: float, end: float, expected_decoder: dict | None) -> DecodedReference:
    """Explicit bounded MPEG Layer III path; never expand legacy PCM limits."""
    ffmpeg, ffprobe = _binary("ffmpeg"), _binary("ffprobe")
    decoder = {"operation": "mp3_decode_float32", "decoder": "ffmpeg",
               "version": _run([str(ffmpeg), "-version"]).decode("utf-8", "replace").splitlines()[0],
               "ffmpeg_sha256": _sha256(ffmpeg), "ffprobe_sha256": _sha256(ffprobe)}
    if expected_decoder is not None and decoder != expected_decoder:
        raise ValueError("Reference decoder provenance changed")
    raw = _run([str(ffprobe), "-v", "error", "-protocol_whitelist", "file,pipe",
                "-show_entries", "stream=codec_type,codec_name,sample_rate,channels:stream_disposition=attached_pic:format=format_name,duration",
                "-of", "json", str(path)])
    try:
        probe = json.loads(raw)
        all_streams = probe["streams"]
        streams = [row for row in all_streams if row.get("codec_type") == "audio"]
        artwork = [row for row in all_streams if row.get("codec_type") == "video"
                   and row.get("disposition", {}).get("attached_pic") == 1]
        if (len(streams) != 1 or len(streams) + len(artwork) != len(all_streams)
                or streams[0].get("codec_name") != "mp3" or probe["format"]["format_name"] != "mp3"):
            raise ValueError("MP3 reference needs exactly one MPEG Layer III audio stream")
        rate, channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
        duration = float(probe["format"]["duration"])
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("Reference recording metadata is invalid") from exc
    if (channels not in (1, 2) or not 8000 <= rate <= 96000
            or not math.isfinite(duration) or not 0 < duration <= 1800):
        raise ValueError("MP3 reference channels, rate or duration unsuitable")
    if end > duration + 1 / rate:
        raise ValueError("Reference selection exceeds the original")
    first, last = round(start * rate), round(end * rate)
    raw = _run([str(ffmpeg), "-v", "error", "-nostdin", "-threads", "1",
                "-protocol_whitelist", "file,pipe", "-t", str(end + 0.5), "-i", str(path),
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-af",
                f"atrim=start_sample={first}:end_sample={last}",
                "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
               max_stdout_bytes=(last - first) * channels * 4)
    if len(raw) != (last - first) * channels * 4:
        raise ValueError("Reference decoder returned an incomplete crop")
    wave = np.frombuffer(raw, dtype="<f4").copy()
    if channels == 2:
        wave = wave.reshape(-1, 2)
    return DecodedReference(wave, rate, channels, duration, decoder)


def decode_reference(path: Path, start: float, end: float, *,
                     expected_decoder: dict | None = None) -> DecodedReference:
    """Decode one exact crop, rejecting decoder changes on compressed replay."""
    import soundfile as sf

    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (start, end)) or start < 0 or not 3 <= end - start <= 30):
        raise ValueError("Reference crop must be finite and 3-30 seconds")
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_ORIGINAL_BYTES:
        raise ValueError("Reference original exceeds the byte limit or is missing")
    with path.open("rb") as stream:
        signature = stream.read(3)
    # libsndfile can open MP3 too: choose the recorded, replayable decoder before
    # trying it, so an installation difference cannot silently change PCM.
    is_mp3 = signature == b"ID3" or (len(signature) >= 2 and signature[0] == 255
                                   and signature[1] & 0xE0 == 0xE0)
    if is_mp3:
        result = _mp3(path, start, end, expected_decoder)
        if not len(result.wave) or not np.isfinite(result.wave).all():
            raise ValueError("Reference decoder returned invalid samples")
        return result
    try:
        info = sf.info(str(path))
    except (RuntimeError, ValueError):
        result = _webm(path, start, end, expected_decoder)
    else:
        if path.stat().st_size > 16_000_000:
            raise ValueError("Non-WebM reference exceeds the legacy byte limit")
        if expected_decoder is not None:
            raise ValueError("Recorded WebM decoder cannot be replaced by another decoder")
        if (info.channels not in (1, 2) or not 8000 <= info.samplerate <= 96000
                or not 0 < info.duration <= 600 or end > info.duration + 1 / info.samplerate):
            raise ValueError("Reference recording channels, rate, duration or crop unsuitable")
        wave, rate = sf.read(str(path), start=round(start * info.samplerate),
                             stop=round(end * info.samplerate), dtype="float32")
        result = DecodedReference(wave, rate, info.channels, info.duration)
    if len(result.wave) != round(end * result.sample_rate) - round(start * result.sample_rate):
        raise ValueError("Reference decoder returned an incomplete crop")
    if not len(result.wave) or not np.isfinite(result.wave).all():
        raise ValueError("Reference decoder returned invalid samples")
    return result
