"""Real codec regression on tiny isolated signals; no TTS or story/demo audio.

All files live in pytest's temporary directory. These checks exercise the actual
production assembly/MP3/decode/loudness functions without using any project,
model, recording, copyrighted text, voice reference or GPU.
"""

import math
import shutil
import struct
import subprocess
import wave

import pytest

from dubbing_service.app.audiobook import voice_lab_render as render


def _signal(path, *, seconds, frequency):
    rate = 24000
    pcm = b"".join(struct.pack("<h", round(0.12 * 32767 * math.sin(2 * math.pi * frequency * i / rate)))
                   for i in range(round(seconds * rate)))
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(pcm)
    return pcm


def test_real_pcm_assembly_preserves_every_sample_and_adds_no_padding(tmp_path):
    left, right = tmp_path / "signal-a.wav", tmp_path / "signal-b.wav"
    a = _signal(left, seconds=0.7, frequency=220)
    b = _signal(right, seconds=0.9, frequency=440)
    source_hashes = [render.file_sha256(path) for path in (left, right)]
    master = tmp_path / "test-master.wav"
    render._assemble_pcm([left, right], master)
    with wave.open(str(master), "rb") as joined:
        assert joined.getnchannels() == 1
        assert joined.getsampwidth() == 2
        assert joined.getframerate() == 24000
        assert joined.getnframes() == (len(a) + len(b)) // 2
        assert joined.readframes(joined.getnframes()) == a + b
    assert render._decoded_duration(master) == 1.6
    assert [render.file_sha256(path) for path in (left, right)] == source_hashes
    before = render.file_sha256(master)
    with pytest.raises(render.ListeningRenderError, match="already exists"):
        render._assemble_pcm([left, right], master)
    assert render.file_sha256(master) == before


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is absent; real production codec is unavailable")
def test_real_mp3_decode_duration_loudness_and_immutable_hashes(tmp_path):
    left, right = tmp_path / "signal-a.wav", tmp_path / "signal-b.wav"
    _signal(left, seconds=0.7, frequency=220)
    _signal(right, seconds=0.9, frequency=440)
    master, mp3 = tmp_path / "clean-master.wav", tmp_path / "listening.mp3"
    render._assemble_pcm([left, right], master)
    master_hash = render.file_sha256(master)
    render._mp3(master, mp3)
    mp3_hash = render.file_sha256(mp3)
    assert mp3_hash != master_hash and mp3.stat().st_size > 1000
    decoded = render._decoded_duration(mp3)
    assert abs(decoded - render._decoded_duration(master)) <= 0.03
    original_loudness = render._loudness(master)
    delivered_loudness = render._loudness(mp3)
    assert original_loudness is not None and math.isfinite(original_loudness)
    assert delivered_loudness is not None and math.isfinite(delivered_loudness)
    assert -40 < original_loudness < -10
    assert abs(delivered_loudness - original_loudness) < 2
    assert render.file_sha256(master) == master_hash
    timestamp = mp3.stat().st_mtime_ns
    # Installed FFmpeg 9 emits "already exists" but can return zero for -n.
    # The contract is that the completed artifact remains byte-for-byte intact,
    # independent of the platform/version's overwrite-refusal exit status.
    try:
        render._mp3(master, mp3)
    except subprocess.CalledProcessError:
        pass
    assert render.file_sha256(mp3) == mp3_hash
    assert mp3.stat().st_mtime_ns == timestamp
