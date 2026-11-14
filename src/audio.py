## Shared audio decoding interface.
## Any additional crop/pad rules belong to each detector adapter.

from __future__ import annotations

from typing import Optional

import librosa
import torch
import torchaudio


def load_audio(path: str, target_sr: Optional[int] = None) -> tuple[torch.Tensor, int]:
    ## Decode and average channels to mono
    waveform, sample_rate = librosa.load(path, sr=None, mono=True)

    ## Resample explicitly, so every detector receives its configured sample rate.
    if target_sr is not None and target_sr != sample_rate:
        audio = torch.from_numpy(waveform).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sample_rate, target_sr)
        waveform = audio.squeeze(0).numpy()
        sample_rate = target_sr

    return torch.from_numpy(waveform).float(), sample_rate


def normalize_std(waveform: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ## Match SONICS' NumPy population std (ddof=0), without subtracting the mean.
    ## Keep it in autograd: waveform gradients must pass through this scaling.
    return waveform / waveform.std(correction=0).clamp_min(eps)
