from __future__ import annotations

import pytest
import torch

from indextts.utils.audio_safety import PCM16_MAX, float_waveform_to_pcm16_range


def test_in_range_waveform_is_not_rescaled() -> None:
    waveform = torch.tensor([[0.5, -0.25, 0.0]], dtype=torch.float32)

    pcm, raw_peak = float_waveform_to_pcm16_range(waveform)

    assert raw_peak == pytest.approx(0.5)
    torch.testing.assert_close(pcm, waveform * PCM16_MAX)


def test_over_range_waveform_gets_headroom_without_hard_clipping() -> None:
    waveform = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float32)

    pcm, raw_peak = float_waveform_to_pcm16_range(waveform, peak_headroom=0.95)

    assert raw_peak == pytest.approx(2.0)
    assert float(pcm.abs().max()) == pytest.approx(PCM16_MAX * 0.95, rel=1e-5)
    assert float(pcm.abs().max()) < PCM16_MAX
    assert float(pcm[0, 1]) == pytest.approx(-PCM16_MAX * 0.475, rel=1e-5)


def test_full_scale_waveform_gets_headroom() -> None:
    waveform = torch.tensor([[1.0, -1.0]], dtype=torch.float32)

    pcm, raw_peak = float_waveform_to_pcm16_range(waveform, peak_headroom=0.95)

    assert raw_peak == pytest.approx(1.0)
    assert float(pcm.abs().max()) == pytest.approx(PCM16_MAX * 0.95, rel=1e-5)


def test_non_finite_waveform_is_rejected() -> None:
    waveform = torch.tensor([[float("nan")]], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-finite"):
        float_waveform_to_pcm16_range(waveform)
