"""Bounded MP3 decode/replay using disposable locally generated fixtures."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import reference_codec as codec


@pytest.fixture
def mp3(tmp_path):
    try:
        binary = codec._binary("ffmpeg")
        codec._binary("ffprobe")
    except ValueError:
        pytest.skip("Existing FFmpeg unavailable; no installation attempted")
    source = tmp_path / "fixture.wav"
    t = np.arange(44100 * 6, dtype=np.float32) / 44100
    sf.write(source, np.column_stack((np.sin(t * 900) * .12, np.sin(t * 1300) * .08)),
             44100, subtype="FLOAT")
    target = tmp_path / "fixture.mp3"
    subprocess.run([str(binary), "-v", "error", "-nostdin", "-i", str(source),
                    "-c:a", "libmp3lame", str(target)], check=True, capture_output=True, timeout=15)
    return target


def test_exact_mp3_crop_replay_and_decoder_binding(mp3):
    first = codec.decode_reference(mp3, .75, 4.75)
    assert first.wave.shape == (44100 * 4, 2)
    assert first.decoder["operation"] == "mp3_decode_float32"
    again = codec.decode_reference(mp3, .75, 4.75, expected_decoder=first.decoder)
    assert np.array_equal(first.wave, again.wave)
    with pytest.raises(ValueError, match="provenance changed"):
        codec.decode_reference(mp3, .75, 4.75, expected_decoder=first.decoder | {"version": "changed"})


def test_mp3_crop_overrun_and_missing_binary_hold(mp3, monkeypatch):
    with pytest.raises(ValueError, match="exceeds"):
        codec.decode_reference(mp3, 4, 8)
    monkeypatch.setattr(codec, "_binary", lambda _: (_ for _ in ()).throw(ValueError("decoder missing")))
    with pytest.raises(ValueError, match="decoder missing"):
        codec.decode_reference(mp3, 0, 4)


def test_id3_spoof_does_not_open_an_arbitrary_format(tmp_path):
    path = tmp_path / "spoof.mp3"
    path.write_bytes(b"ID3" + b"not mp3" * 100)
    with pytest.raises(ValueError, match="could not decode"):
        codec.decode_reference(path, 0, 4)


@pytest.mark.parametrize("change", ["additional_audio", "playable_video", "codec", "duration"])
def test_mp3_probe_rejects_invalid_streams_and_duration(mp3, monkeypatch, change):
    import json
    original = codec._run
    def run(args, **kwargs):
        if "-show_entries" not in args:
            return original(args, **kwargs)
        probe = json.loads(original(args, **kwargs))
        if change == "additional_audio":
            probe["streams"].append(probe["streams"][0].copy())
        elif change == "playable_video":
            probe["streams"].append({"codec_type": "video", "disposition": {"attached_pic": 0}})
        elif change == "codec":
            probe["streams"][0]["codec_name"] = "aac"
        else:
            probe["format"]["duration"] = "1800.1"
        return json.dumps(probe).encode()
    monkeypatch.setattr(codec, "_run", run)
    with pytest.raises(ValueError, match="MP3 reference"):
        codec.decode_reference(mp3, 0, 4)


def test_attached_artwork_is_ignored_for_audio_not_treated_as_playable_video(mp3, monkeypatch):
    import json
    original = codec._run
    def run(args, **kwargs):
        raw = original(args, **kwargs)
        if "-show_entries" not in args:
            return raw
        probe = json.loads(raw)
        probe["streams"].append({"codec_type": "video", "codec_name": "mjpeg",
                                 "disposition": {"attached_pic": 1}})
        return json.dumps(probe).encode()
    monkeypatch.setattr(codec, "_run", run)
    assert codec.decode_reference(mp3, 0, 4).wave.shape == (44100 * 4, 2)
