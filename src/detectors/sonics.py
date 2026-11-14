## Adapter SONICS: metadane checkpointu, przygotowanie audio i dwie ścieżki predykcji.

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from sonics import HFAudioClassifier

from src.audio import normalize_std
from src.domains import MelSpectrogramDomain


@dataclasses.dataclass
class SonicsMeta:
    sample_rate: Optional[int] = None
    input_length_seconds: Optional[float] = None
    max_len_samples: Optional[int] = None
    num_classes: Optional[int] = None
    fake_label_idx: Optional[int] = None
    real_label_idx: Optional[int] = None
    checkpoint: Optional[str] = None


def load_sonics(checkpoint: str = "awsaf49/sonics-spectttra-gamma-5s"):
    ## Architektura i ustawienia ekstraktora cech są pobierane z checkpointu.
    ## eval() wyłącza dropout i augmentacje używane przez bibliotekę podczas treningu.
    model = HFAudioClassifier.from_pretrained(checkpoint)
    model.checkpoint = checkpoint
    model.eval()
    return model


def inspect_sonics_meta(model) -> SonicsMeta:
    ## SONICS jest klasyfikatorem binarnym. Wspólny interfejs programu zapisuje klasy jako 0 = real i 1 = fake.
    meta = SonicsMeta(
        fake_label_idx=1,
        real_label_idx=0,
        checkpoint=getattr(model, "checkpoint", None),
    )
    config = getattr(model, "config", None)
    if config is None:
        return meta

    audio_config = getattr(config, "audio", None)
    if audio_config is not None:
        if hasattr(audio_config, "sample_rate"):
            meta.sample_rate = audio_config.sample_rate
        if hasattr(audio_config, "max_len"):
            meta.input_length_seconds = (audio_config.max_len / meta.sample_rate if meta.sample_rate else None)
            meta.max_len_samples = audio_config.max_len
        elif hasattr(audio_config, "max_time") and meta.sample_rate:
            meta.input_length_seconds = audio_config.max_time
            meta.max_len_samples = int(audio_config.max_time * meta.sample_rate)

    if hasattr(config, "num_classes"):
        meta.num_classes = config.num_classes

    return meta


class SonicsDetector(nn.Module):
    def __init__(self, model, meta: Optional[SonicsMeta] = None):
        super().__init__()
        self.model = model
        self.meta = meta or inspect_sonics_meta(model)


    def prepare_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        target_len = self.meta.max_len_samples
        length = waveform.shape[-1]
        if target_len is None or length == target_len:
            return waveform
        if length < target_len:
            return nn.functional.pad(waveform, (0, target_len - length))
        
        ## Domyślnie wycinek jest dopasowany do sposobu działania walidacyjnego loadera SONICS: przed oknem zostaw 3/4 nadmiaru, za nim 1/4.
        start = int((length - target_len) / 4 * 3)
        return waveform[..., start : start + target_len]


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        ## Ścieżka dla waveform: model dostaje próbki audio, normalizuje je, wylicza cechy mel i przepuszcza je przez encoder oraz klasyfikator.
        ## Ponieważ operacje tej ścieżki są różniczkowalne, gradient straty może przejść od wyniku klasyfikatora z powrotem aż do próbek audio.
        output = self.model(normalize_std(waveform))
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        else:
            raise RuntimeError(f"Unsupported model output type: {type(output)}")

        ## SONICS zwraca jeden surowy wynik klasyfikatora (logit).
        ## Logit nie jest etykietą ani prawdopodobieństwem: sigmoid(logit) daje prawdopodobieństwo fake, a 1 - sigmoid(logit) prawdopodobieństwo real.
        ## BCE (binary cross-entropy) to funkcja straty używana do uczenia tego binarnego klasyfikatora.
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if logits.shape[-1] == 1:
            ## SONICS zwraca jeden logit z: sigmoid(z) to prawdopodobieństwo fake.
            ## Dodajemy logit 0 dla real, aby uzyskać format dwóch klas [0, z], którego oczekują pozostałe części programu. 
            ## Softmax([0, z]) daje te same prawdopodobieństwa: real = 1 - sigmoid(z), fake = sigmoid(z).
            logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
        return logits


    def extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        ## Zwróć bezpośrednio wynik frontendu: znormalizowane cechy log-mel o kształcie [partia, pasma mel, ramki], czyli zmienną ataku mel.
        return self.model.ft_extractor(normalize_std(waveform))


    def predict_mel_features(self, features: torch.Tensor) -> torch.Tensor:
        ## Ścieżka ataku mel dostaje gotowe cechy zamiast próbek audio, więc pomija normalizację audio i frontend mel, które są już wykonane.
        ## Gradient jest liczony względem tych cech. 
        ## Użycie forward() byłoby błędne: potraktowałoby wartości cech jak próbki i ponownie liczyło mel.
        spec = features.unsqueeze(1)

        ## Dopasuj rozmiar cech do siatki wejściowej encodera z checkpointu.
        spec = nn.functional.interpolate(spec, size=tuple(self.model.input_shape), mode="bilinear")

        encoded = self.model.encoder(spec)
        if "timm" not in self.model.model_name:
            ## Uśrednij reprezentacje po pozycjach wyjściowych encodera.
            embeddings = encoded.mean(dim=1)
        else:
            ## Modele timm zwracają reprezentację w formacie oczekiwanym przez klasyfikator.
            embeddings = encoded
        logits = self.model.classifier(embeddings)
        if logits.shape[-1] == 1:
            ## Zachowaj format dwóch klas jak w ścieżce waveform: pozycja 0 = real, 1 = fake.
            logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
        return logits


    def create_mel_domain(self, waveform: torch.Tensor, reconstruction: str = "source_phase") -> tuple[torch.Tensor, MelSpectrogramDomain]:
        config = self.model.config
        mel_config = config.melspec
        if mel_config.norm != "mean_std":
            raise ValueError("Mel reconstruction currently supports SONICS mean_std normalization only")

        ## Przygotuj czyste cechy, które będą wejściem ataku mel, zgodnie z frontendem SONICS.
        ## Atak będzie zmieniał tę macierz cech, a nie próbki audio.
        normalized_waveform = normalize_std(waveform)
        extractor = self.model.ft_extractor
        mel_power = extractor.audio2melspec(normalized_waveform)
        mel_db = extractor.amplitude_to_db(mel_power)

        ## Zapamiętaj średnią i odchylenie czystego nagrania, by użyć ich do odwrócenia normalizacji każdej zaatakowanej macierzy cech.
        ## Rekonstrukcja jest przybliżona: normalizator dzieli przez std + eps, a odwrócenie używa std; projekcja mel i ograniczenie top_db też tracą dane.
        mean = mel_db.mean((1, 2), keepdim=True)
        std = mel_db.reshape(mel_db.shape[0], -1).std(1, keepdim=True).unsqueeze(-1)
        features = extractor.normalizer(mel_db)

        sample_rate = config.audio.sample_rate
        n_fft = mel_config.n_fft
        hop_length = mel_config.hop_length
        win_length = mel_config.win_length
        window = torch.hann_window(win_length, device=waveform.device)

        ## Cechy mel opisują widmo, ale nie zachowują fazy potrzebnej do odtworzenia sygnału. 
        ## Zapisz fazę STFT czystego audio dla trybu source_phase.
        complex_stft = torch.stft(
            normalized_waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        phase = complex_stft / complex_stft.abs().clamp_min(1e-8)

        ## InverseMelScale przybliża przejście z pasm mel do częstotliwości liniowych.
        ## Dekoder działa na CPU i odłącza cechy od grafu autograd, więc nie uczestniczy w obliczaniu gradientu ataku.
        ## Gradient mel przechodzi wyłącznie przez predict_mel_features.
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=mel_config.n_mels,
            sample_rate=sample_rate,
            f_min=mel_config.f_min,
            f_max=mel_config.f_max,
        )
        reference_features = features.detach().cpu()
        reference_mean = mean.detach().cpu()
        reference_std = std.detach().cpu()
        reference_phase = phase.detach().cpu()
        cpu_window = torch.hann_window(win_length)

        ## decode() przekazuje moduł widma (pierwiastek z mocy), dlatego power=1.
        ## Griffin-Lim szacuje brakującą fazę iteracyjnie, zamiast używać fazy źródła.
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            n_iter=32,
            rand_init=False,
            length=self.meta.max_len_samples or waveform.shape[-1],
        )
        waveform_length = self.meta.max_len_samples or waveform.shape[-1]


        def decode(candidate: torch.Tensor) -> torch.Tensor:
            ## Odtwórz przybliżone widmo: odwróć normalizację cech i skalę dB, zamień pasma mel na widmo liniowe i oblicz jego moduł.
            ## Ponieważ mel nie przechowuje fazy, uzupełnij ją wybranym trybem poniżej.
            candidate_db = candidate.detach().cpu() * reference_std + reference_mean
            candidate_mel_power = torch.pow(10.0, candidate_db / 10.0)
            linear_power = inverse_mel(candidate_mel_power).clamp_min(0)
            magnitude = linear_power.sqrt()
            if reconstruction == "griffin_lim":
                return griffin_lim(magnitude)
            
            ## W trybie source_phase użyj modułu z cech-kandydatów, ale fazy zachowanej z oryginalnego nagrania.
            reconstructed_stft = magnitude * reference_phase
            audio = torch.istft(
                reconstructed_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=cpu_window,
                center=True,
                length=waveform_length,
            )

            return audio

        ## Wyznacz wzmocnienie na podstawie rekonstrukcji czystych cech i stosuj to samo dla każdego kandydata.
        peak_scale = (
            decode(reference_features)
            .abs()
            .amax(dim=-1, keepdim=True)
            .clamp_min(1e-8)
            .reciprocal()
        )


        def reconstruct(candidate: torch.Tensor) -> torch.Tensor:
            ## Zamień zaatakowane cechy na audio, zastosuj wspólne wzmocnienie i ogranicz próbki do [-1, 1]. 
            ## Rekonstrukcja mel jest przybliżona, więc po ponownym przejściu audio przez detektor efekt ataku może być inny niż predykcja na cechach.
            return (decode(candidate) * peak_scale).clamp(-1.0, 1.0)

        domain = MelSpectrogramDomain(reconstruct=reconstruct)
        return features, domain
