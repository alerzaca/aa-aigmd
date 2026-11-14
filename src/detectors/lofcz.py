# lofcz: fixed spectral features ("fakeprint") followed by logistic regression.
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

N_FFT, SAMPLE_RATE, HULL, MIN_DB, MAX_DB = 8192, 16000, 10, -45.0, 5.0
# HULL is the min-filter width in frequency bins; the analyzed band is 1-8 kHz.


@dataclasses.dataclass
class LofczMeta:
    sample_rate: int = SAMPLE_RATE
    input_length_seconds: None = None
    max_len_samples: None = None
    num_classes: int = 2
    fake_label_idx: int = 1
    real_label_idx: int = 0


class LofczDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Buffers move with .to(device), but are not trainable parameters.
        # PyTorch operations below retain gradients with respect to the audio.
        params = np.load(hf_hub_download("lofcz/ai-music-detector", "weights.npz"))
        self.register_buffer("weight", torch.tensor(params["weights"][0]))
        self.register_buffer("bias", torch.tensor(params["bias"][0]))
        self.register_buffer("window", torch.hann_window(N_FFT))
        bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
        self.register_buffer(
            "band", torch.from_numpy(np.where((bins >= 1000) & (bins <= 8000))[0])
        )
        self.meta = LofczMeta()

    def prepare_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        # Unlike SONICS, this detector averages features over the whole clip.
        return waveform

    def fakeprint(self, waveform: torch.Tensor) -> torch.Tensor:
        # STFT -> power -> dB -> time average -> selected frequency band.
        # Averaging dB is intentional; it is not dB of the averaged power.
        power = (
            torch.stft(
                waveform,
                N_FFT,
                hop_length=N_FFT // 2,
                window=self.window,
                return_complex=True,
            )
            .abs()
            .square()
        )
        spectrum = (10 * torch.log10(power.clamp(1e-10, 1e6))).mean(-1)[..., self.band]
        # A local minimum estimates the lower spectral envelope ("hull").
        # min(x) = -max(-x) gives a differentiable PyTorch min-filter.
        # Asymmetric padding preserves the bin count with an even filter width.
        padded = F.pad(
            spectrum.unsqueeze(1), (HULL // 2, HULL // 2 - 1), mode="replicate"
        )
        hull = -F.max_pool1d(-padded, HULL, 1).squeeze(1)
        # Keep peaks above that envelope, cap their height and normalize per clip.
        residue = (spectrum - hull.clamp_min(MIN_DB)).clamp(0, MAX_DB)
        return residue / (residue.amax(-1, keepdim=True) + 1e-6)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        logit = self.fakeprint(waveform) @ self.weight + self.bias
        # softmax([0, z])[fake] equals the trained regressor's sigmoid(z).
        return torch.stack([torch.zeros_like(logit), logit], dim=-1)
