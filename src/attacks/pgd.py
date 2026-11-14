## PGD Attack: wielokrotnie wykonuje krok gradientowy i pilnuje ustalonego budżetu zmian.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchattacks

from src.attacks import registry
from src.domains import AttackDomain, WaveformDomain


def pgd_attack(
    model: registry.ModelFn,
    x: torch.Tensor,
    true_label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int = 20,
    target_label: int = 0,
    domain: Optional[AttackDomain] = None,
) -> torch.Tensor:
    ## X = zmienna ataku: próbki waveform albo cechy mel.
    domain = domain or WaveformDomain()
    original = x.clone().detach()

    ## Atak zaczyna od losowego punktu w granicach epsilon.
    ## Bez ustawionego ziarna kolejne uruchomienia mogą dać inne wyniki.
    adversarial = domain.project_linf(
        original + torch.empty_like(original).uniform_(-epsilon, epsilon),
        original,
        epsilon,
    )

    for _ in range(steps):
        ## Odłącz poprzedni krok, żeby graf obliczeń nie rósł z każdą iteracją.
        ## Gradient tej iteracji wyznacza wpływ bieżącego wejścia na stratę.
        adversarial = adversarial.detach().requires_grad_(True)
        logits = model(adversarial)
        target = torch.full_like(true_label, target_label)
        loss = F.cross_entropy(logits, target)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        ## Zmniejsz stratę krokiem wzdłuż gradientu, a potem ogranicz zmianę względem czystego wejścia, by nie przekroczyć łącznego budżetu epsilon.
        adversarial = domain.project_linf(adversarial - alpha * gradient.sign(), original, epsilon)

    return adversarial.detach()


def pgd_torchattacks(model, x, labels, epsilon, alpha, steps, target_label: int) -> torch.Tensor:
    ## Gotowa implemetacja ataku z TorchAttacks.
    ## Skala wejścia [0, 1] jest dwukrotnie mniejsza niż [-1, 1], dlatego dzielimy przez 2 zarówno epsilon, jak i rozmiar kroku alpha.
    build = lambda m: torchattacks.PGD(m, eps=epsilon / 2, alpha=alpha / 2, steps=steps)
    return registry.run_torchattacks(build, model, x, labels, target_label)
