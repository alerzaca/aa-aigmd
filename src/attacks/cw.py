## C&W Attack: optymalizuje wejście, minimalizując sumę kwadratów zmian i karę za nieosiągnięcie klasy docelowej.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchattacks

from src.attacks import registry
from src.domains import AttackDomain, WaveformDomain


@dataclass
class CWResult:
    x_adv: torch.Tensor ## Znalezione zaburzone wejście. Jeśli atak się nie powiedzie, zwraca ostatnią próbę.
    best_l2: float      ## Suma kwadratów zmian: pozostaje nieskończona, jeśli atak się nie powiedzie.
    success: bool       ## Wynik


def cw_l2_attack(
    model: registry.ModelFn,
    x: torch.Tensor,
    target_label: int,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 500,
    lr: float = 0.01,
    domain: Optional[AttackDomain] = None,
) -> CWResult:
    ## X = zmienna ataku: próbki waveform albo cechy mel.
    domain = domain or WaveformDomain()
    original = x.clone().detach()
    
    ## C&W optymalizuje zmienną `variable`. 
    ## Dla waveform funkcja tanh zamienia ją na próbki o dozwolonej amplitudzie. Dla mel zmienną są bezpośrednio cechy.
    ## Gradient przechodzi dalej przez model adversarial.
    variable = domain.to_optimizer_space(original).requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=lr)
    best_l2 = float("inf")
    best_adversarial = original.clone()
    success = False

    for _ in range(steps):
        adversarial = domain.from_optimizer_space(variable)
        logits = model(adversarial)
        distance = domain.distance_l2(original, adversarial)

        ## Model ma przypisać klasie docelowej wynik większy od pozostałych
        ## co najmniej o kappa (domyślnie zero).
        target_logit = logits[:, target_label]
        other_logits = logits.clone()
        other_logits[:, target_label] = -1e9
        margin = torch.clamp(other_logits.max(dim=1).values - target_logit, min=-kappa)

        ## c określa kompromis między małą zmianą wejścia a osiągnięciem celu. W tej implementacji c jest stałe.
        loss = (distance + c * margin).sum()
        optimizer.zero_grad()

        ## Oblicza gradient sumy wielkości zmiany i marginesu klasyfikacji względem `variable`; Adam aktualizuje tę zmienną.
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ## Zachowaj udany wynik o najmniejszej zmianie.
            predictions = logits.argmax(dim=1)
            succeeded = predictions.eq(target_label)
            for index in range(original.shape[0]):
                if succeeded[index] and distance[index].item() < best_l2:
                    best_l2 = distance[index].item()
                    best_adversarial = adversarial.detach().clone()
                    success = True

    if not success:
        ## Jeśli nie znaleziono udanego wyniku, zwróć ostatnią próbę do oceny.
        best_adversarial = domain.from_optimizer_space(variable).detach()

    return CWResult(best_adversarial, best_l2, success)


def cw_torchattacks(model, x, labels, c, steps, target_label: int) -> torch.Tensor:
    ## Gotowa implemetacja ataku z TorchAttacks.
    ## W zakresie [0, 1] suma kwadratów zmian jest czterokrotnie mniejsza niż w [-1, 1], dlatego dzielimy c przez 4.
    build = lambda m: torchattacks.CW(m, c=c / 4, steps=steps, lr=0.01)
    return registry.run_torchattacks(build, model, x, labels, target_label)
