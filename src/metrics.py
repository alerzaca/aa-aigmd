# Raw distortion/quality measurements; success rates are computed in analysis.
from __future__ import annotations

import numpy as np
import torch
import torchaudio

# Missing optional metrics retain the existing empty-cell behavior.
try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


def snr_db(clean: torch.Tensor, adv: torch.Tensor, eps: float = 1e-10) -> float:
    # Compare total energies, without gain alignment inside this function.
    # Any alignment of a clean mel reconstruction is the caller's responsibility.
    clean_np = clean.detach().cpu().numpy().flatten()
    adv_np = adv.detach().cpu().numpy().flatten()
    noise = clean_np - adv_np
    signal_power = np.sum(clean_np**2)
    noise_power = np.sum(noise**2) + eps
    return 10.0 * np.log10(signal_power / noise_power + eps)


def linf_norm(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return (clean - adv).abs().max().item()


def try_pesq(clean: torch.Tensor, adv: torch.Tensor, sr: int) -> float | None:
    # Wideband PESQ requires 16 kHz. It measures speech quality, not music fidelity.
    if pesq is None:
        return None
    try:
        target_sr = 16000
        c = clean.detach().cpu()
        a = adv.detach().cpu()
        if sr != target_sr:
            c = torchaudio.functional.resample(c.unsqueeze(0), sr, target_sr).squeeze(0)
            a = torchaudio.functional.resample(a.unsqueeze(0), sr, target_sr).squeeze(0)
        return float(pesq(target_sr, c.numpy(), a.numpy(), "wb"))
    except Exception:  # noqa: BLE001
        # Existing optional-metric policy: missing/unsupported PESQ becomes an
        # empty CSV cell, never a perfect-quality score. Failure reason is not saved.
        return None


def try_stoi(clean: torch.Tensor, adv: torch.Tensor, sr: int) -> float | None:
    # STOI estimates speech intelligibility; high values do not establish
    # inaudibility of adversarial changes in music. Keep missing values as None.
    if stoi is None:
        return None
    try:
        c = clean.detach().cpu().numpy().flatten()
        a = adv.detach().cpu().numpy().flatten()
        return float(stoi(c, a, sr, extended=False))
    except Exception:  # noqa: BLE001
        return None
