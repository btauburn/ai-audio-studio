"""Small waveform safety helpers shared by IndexTTS inference paths."""

from __future__ import annotations

import math

import torch


PCM16_MAX = 32767.0


def float_waveform_to_pcm16_range(
    waveform: torch.Tensor,
    *,
    peak_headroom: float = 0.95,
) -> tuple[torch.Tensor, float]:
    """Scale only over-range model output before conversion to PCM16 values.

    BigVGAN normally emits samples in [-1, 1], but some reference/emotion
    combinations reach or exceed full scale.  Those waveforms are
    peak-normalized with modest headroom before integer conversion.  Signals
    already below the requested headroom are left unchanged.
    """
    if waveform.numel() == 0:
        raise ValueError("cannot convert an empty waveform")
    if not 0.0 < peak_headroom <= 1.0:
        raise ValueError("peak_headroom must be in (0, 1]")

    raw_peak = float(waveform.detach().abs().amax().item())
    if not math.isfinite(raw_peak):
        raise ValueError("waveform contains non-finite samples")
    if raw_peak > peak_headroom:
        waveform = waveform * (peak_headroom / raw_peak)
    return torch.clamp(PCM16_MAX * waveform, -PCM16_MAX, PCM16_MAX), raw_peak
