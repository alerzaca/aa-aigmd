# Random Gaussian noise is a control: it does not use detector gradients.
from __future__ import annotations

from typing import Optional

import torch

from src.domains import WaveformDomain


def gaussian_noise_baseline(
    x: torch.Tensor,
    target_snr_db: float,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    generator: Optional[torch.Generator] = None,
    domain: Optional[WaveformDomain] = None,
) -> torch.Tensor:
    domain = domain or WaveformDomain(clip_min, clip_max)
    # SNR = 10*log10(signal energy / noise energy). Scale Gaussian samples
    # to the corresponding expected energy, independently for each waveform.
    # Random realization and final clipping can shift the achieved SNR.
    signal_power = (x**2).sum(dim=-1, keepdim=True)
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise_std = torch.sqrt(noise_power / x.shape[-1])
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return (x + noise * noise_std).clamp(domain.clip_min, domain.clip_max)
