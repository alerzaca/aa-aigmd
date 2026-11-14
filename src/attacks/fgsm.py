## FGSM Attack: wykonuje jeden krok w kierunku znaku gradientu w ramach budżetu.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchattacks

from src.attacks import registry
from src.domains import AttackDomain, WaveformDomain


def fgsm_attack(
    model: registry.ModelFn,
    x: torch.Tensor,
    true_label: torch.Tensor,
    epsilon: float,
    target_label: int,
    domain: Optional[AttackDomain] = None,
) -> torch.Tensor:
    ## X = zmienna ataku: próbki waveform albo cechy mel.
    domain = domain or WaveformDomain()
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    target = torch.full_like(true_label, target_label)
    loss = F.cross_entropy(logits, target)

    ## PyTorch oblicza gradient straty względem wejścia ataku, nie wag modelu.
    gradient = torch.autograd.grad(loss, x)[0]

    ## Wykonujemy krok przeciwny do gradientu i ograniczamy zmianę do budżetu epsilon.
    return domain.project_linf(x - epsilon * gradient.sign(), x, epsilon).detach()


def fgsm_torchattacks(model, x, labels, epsilon, target_label: int) -> torch.Tensor:
    ## Gotowa implemetacja ataku z TorchAttacks.
    ## Skala wejścia [0, 1] jest dwukrotnie mniejsza niż [-1, 1], dlatego epsilon też dzielimy przez 2, reszta działa tak samo.
    return registry.run_torchattacks(lambda m: torchattacks.FGSM(m, eps=epsilon / 2), model, x, labels, target_label)
