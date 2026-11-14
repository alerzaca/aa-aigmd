## Interfejs dla przestrzeni ataku: audio (waveform) i spektrogram mel.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


class AttackDomain(Protocol):
    ## Atak może zmieniać próbki audio [partia, próbki] albo cechy mel [partia, pasma, ramki]. 
    ## Model ataku przyjmuje tę samą postać danych, którą atak modyfikuje. 

    ## Ogranicz każdą składową zmiany względem oryginału do [-epsilon, epsilon].
    def project_linf(self, candidate: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor: ...

    def to_optimizer_space(self, value: torch.Tensor) -> torch.Tensor: ...

    def from_optimizer_space(self, value: torch.Tensor) -> torch.Tensor: ...

    ## Zwróć sumę kwadratów zmian osobno dla każdego elementu partii.
    def distance_l2(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor: ...

    ## Zamień reprezentację ataku na audio do odsłuchu, zapisu lub oceny pełnym detektorem.
    def to_waveform(self, value: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class WaveformDomain:
    clip_min: float = -1.0
    clip_max: float = 1.0
    tanh_epsilon: float = 1e-6

    def project_linf(self, candidate: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor:
        ## Najpierw ogranicz zmianę każdej próbki do +/- epsilon względem czystego audio, potem przytnij amplitudę do [clip_min, clip_max].
        delta = (candidate - original).clamp(-epsilon, epsilon)
        return (original + delta).clamp(self.clip_min, self.clip_max)

    def to_optimizer_space(self, value: torch.Tensor) -> torch.Tensor:
        ## C&W optymalizuje nieograniczone wartości w, a nie bezpośrednio próbki.
        ## To przekształcenie (atanh) mapuje amplitudy do w, a później (tanh) mapuje je z powrotem do dozwolonego zakresu bez przycinania.
        ## Dokładne wartości graniczne są odsuwane, bo atanh(+-1) jest nieskończone.
        scaled = (value - self.clip_min) / (self.clip_max - self.clip_min) * 2 - 1
        scaled = scaled.clamp(-1 + self.tanh_epsilon, 1 - self.tanh_epsilon)
        return 0.5 * torch.log((1 + scaled) / (1 - scaled))

    def from_optimizer_space(self, value: torch.Tensor) -> torch.Tensor:
        ## Zamień nieograniczoną zmienną C&W na próbki w zakresie [clip_min, clip_max].
        return (self.clip_min + (self.clip_max - self.clip_min) * (torch.tanh(value) + 1) / 2)

    def distance_l2(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        ## Zwróć sumę kwadratów zmian osobno dla każdego elementu partii, bez pierwiastka i uśredniania.
        return ((candidate - original) ** 2).flatten(1).sum(dim=1)

    def to_waveform(self, value: torch.Tensor) -> torch.Tensor:
        ## Próbki są już audio, więc nie wymagają konwersji.
        return value


@dataclass(frozen=True)
class MelSpectrogramDomain:
    ## W tej domenie atak zmienia znormalizowane cechy log-mel, a nie amplitudy próbek audio. 
    ## Nie przycinamy ich do [-1, 1], a epsilon jest w jednostkach cech.
    reconstruct: Callable[[torch.Tensor], torch.Tensor]

    def project_linf(self, candidate: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor:
        ## Ogranicz zmianę każdej wartości cechy do +/- epsilon względem czystych cech.
        return original + (candidate - original).clamp(-epsilon, epsilon)

    def to_optimizer_space(self, value: torch.Tensor) -> torch.Tensor:
        ## C&W optymalizuje cechy mel bezpośrednio, bez dodatkowego przekształcenia tanh.
        return value

    def from_optimizer_space(self, value: torch.Tensor) -> torch.Tensor:
        ## Cechy mel są optymalizowane bezpośrednio, więc nie ma konwersji.
        return value

    def distance_l2(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        ## Mierz zmianę cech mel sumą kwadratów, osobno dla każdego nagrania.
        return ((candidate - original) ** 2).flatten(1).sum(dim=1)

    def to_waveform(self, value: torch.Tensor) -> torch.Tensor:
        ## Zamień zaatakowane cechy na audio do końcowej oceny.
        return self.reconstruct(value)
