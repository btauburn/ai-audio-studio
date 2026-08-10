"""Validation helpers for user-supplied audio files.

File extensions and MIME types are controlled by the client and therefore are
not evidence that an upload contains audio.  The service probes the staged
file before any metadata starts referring to it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
from typing import Collection


UPLOAD_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
)


class AudioValidationError(ValueError):
    """The supplied file is not a supported, decodable audio stream."""


class AudioProbeUnavailableError(RuntimeError):
    """The local ffprobe executable required for validation is unavailable."""


def normalize_audio_extension(
    filename: str | None,
    *,
    default: str = ".wav",
    allowed_extensions: Collection[str] = UPLOAD_AUDIO_EXTENSIONS,
) -> str:
    """Return a lower-case, allow-listed extension for an upload filename."""

    suffix = Path(filename or "").suffix.lower() or default.lower()
    allowed = {str(item).lower() for item in allowed_extensions}
    if suffix not in allowed:
        raise AudioValidationError(f"不支持的音频格式: {suffix}")
    return suffix


def validate_decodable_audio(path: Path, *, timeout_seconds: float = 15.0) -> None:
    """Require at least one audio stream that ffprobe can decode/inspect.

    The path is expected to be a newly written staging file.  Async routes
    should use :func:`validate_decodable_audio_async` so probing cannot block
    unrelated requests on the event loop.
    """

    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise AudioValidationError("音频文件为空")
    except OSError as exc:
        raise AudioValidationError(f"无法读取音频文件: {exc}") from exc

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise AudioProbeUnavailableError("未找到 ffprobe，无法验证上传音频")

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-count_packets",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,nb_read_packets",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioValidationError("音频验证超时，文件可能已损坏") from exc
    except OSError as exc:
        raise AudioProbeUnavailableError(f"无法运行 ffprobe: {exc}") from exc

    if result.returncode != 0:
        raise AudioValidationError("上传文件不是可解码的音频")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioValidationError("ffprobe 返回了无效结果") from exc

    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise AudioValidationError("上传文件不包含音频流")

    stream = streams[0] if isinstance(streams[0], dict) else {}
    if stream.get("codec_type") not in {None, "audio"}:
        raise AudioValidationError("上传文件不包含有效音频流")
    try:
        channels = int(stream.get("channels") or 0)
        sample_rate = int(stream.get("sample_rate") or 0)
    except (TypeError, ValueError) as exc:
        raise AudioValidationError("音频流参数无效") from exc
    if channels <= 0 or sample_rate <= 0:
        raise AudioValidationError("音频流缺少有效采样率或声道信息")

    try:
        packet_count = int(stream.get("nb_read_packets") or 0)
    except (TypeError, ValueError) as exc:
        raise AudioValidationError("音频文件缺少有效数据包") from exc
    if packet_count <= 0:
        raise AudioValidationError("音频文件不包含可播放内容")


async def validate_decodable_audio_async(
    path: Path,
    *,
    timeout_seconds: float = 15.0,
) -> None:
    """Run audio probing outside the async server event loop."""

    await asyncio.to_thread(
        validate_decodable_audio,
        path,
        timeout_seconds=timeout_seconds,
    )
