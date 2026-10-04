"""Apply supported scene controls without changing a persistent voice prompt."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path


def apply_scene_audio(path: Path, *, pace: float = 1.0, volume: float = 1.0) -> bool:
    """Transform a newly created, uncommitted sample; never a reference/old Take.

    Qwen Base has no instruction API. FFmpeg atempo preserves pitch while
    applying the existing project pace/gain fields. Callers record baked values
    so the UI does not apply the same controls twice.
    """
    if pace == 1.0 and volume == 1.0:
        return False
    if not 0.5 <= pace <= 2 or not 0 <= volume <= 2:
        raise ValueError("scene pace/volume outside supported range")
    stage = path.with_name(f"{path.stem}.performance-{uuid.uuid4().hex}.wav")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(path),
            "-af",
            f"atempo={pace:.6f},volume={volume:.6f}",
            "-c:a",
            "pcm_s16le",
            str(stage),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Scene audio processing failed: {result.stderr[-300:]}")
    stage.replace(path)
    return True
