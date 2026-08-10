from __future__ import annotations

import io
import struct
import wave


def make_wav_bytes(*, sample_value: int = 0, frames: int = 800) -> bytes:
    """Return a tiny, valid mono PCM WAV for upload tests."""

    sample = max(-32768, min(32767, int(sample_value)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<h", sample) * max(1, int(frames)))
    return buffer.getvalue()
