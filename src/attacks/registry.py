## Rejestr ataków zapewnia wszystkim ten sam interfejs.
## Importujemy moduły, a funkcje pomocnicze są rozwiązywane dopiero podczas uruchamiania ataku.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.attacks import cw, fgsm, noise, pgd

ModelFn = Callable[[torch.Tensor], torch.Tensor]


class _UnitRange(torch.nn.Module):
    ## torchattacks działa na wejściu [0, 1], a detektor oczekuje [-1, 1].
    def __init__(self, model: ModelFn):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## torchattacks wywołuje model na wejściu [0, 1]; tu przeliczamy je dla detektora na [-1, 1].
        return self.model(2 * x - 1)


def run_torchattacks(build, model: ModelFn, x, labels, target_label: int) -> torch.Tensor:
    ## Wejście eksperymentu x jest w [-1, 1]; torchattacks wymaga [0, 1], więc skalujemy je przed atakiem.
    attack = build(_UnitRange(model))
    ## Wymuś, by atak próbował doprowadzić model do wskazanej klasy docelowej.
    attack.set_mode_targeted_by_function(lambda _, y: torch.full_like(y, target_label), quiet=True)
    ## Wynik ataku jest w [0, 1]; przeliczamy go z powrotem na [-1, 1] dla reszty eksperymentu.
    return (2 * attack((x + 1) / 2, labels) - 1).detach()


@dataclass(frozen=True)
class AttackRegistration:
    budget_name: str                    ## Parametr CLI, z którego pochodzą wartości budżetu.
    run: Callable[..., torch.Tensor]    ## Funkcja uruchamiająca atak i zwracająca zaburzone wejście.
    waveform_only: bool = False         ## Waveform_only zapobiega traktowaniu cech mel jak audio przez biblioteki.


def _run_fgsm(model, x, labels, budget, target, alpha, steps, domain):
    return fgsm.fgsm_attack(model, x, labels, budget, target, domain=domain)


def _run_fgsm_torchattacks(model, x, labels, budget, target, alpha, steps, domain):
    return fgsm.fgsm_torchattacks(model, x, labels, budget, target)


def _run_pgd(model, x, labels, budget, target, alpha, steps, domain):
    return pgd.pgd_attack(model, x, labels, budget, alpha, steps, target, domain=domain)


def _run_pgd_torchattacks(model, x, labels, budget, target, alpha, steps, domain):
    return pgd.pgd_torchattacks(model, x, labels, budget, alpha, steps, target)


def _run_cw(model, x, _labels, budget, target, alpha, steps, domain):
    return cw.cw_l2_attack( model, x, target, c=budget, steps=steps, domain=domain).x_adv


def _run_cw_torchattacks(model, x, labels, budget, target, alpha, steps, domain):
    return cw.cw_torchattacks(model, x, labels, budget, steps, target)


def _run_noise(model, x, labels, budget, target, alpha, steps, domain):
    return noise.gaussian_noise_baseline(x, budget, domain=domain)


ATTACK_REGISTRY = {
    "fgsm": AttackRegistration("epsilon", _run_fgsm),
    "fgsm_torchattacks": AttackRegistration("epsilon", _run_fgsm_torchattacks, waveform_only=True),

    "pgd": AttackRegistration("epsilon", _run_pgd),
    "pgd_torchattacks": AttackRegistration("epsilon", _run_pgd_torchattacks, waveform_only=True),

    "cw": AttackRegistration("c", _run_cw),
    "cw_torchattacks": AttackRegistration("c", _run_cw_torchattacks, waveform_only=True),

    "noise_baseline": AttackRegistration("snr_db_target", _run_noise, waveform_only=True),
}
