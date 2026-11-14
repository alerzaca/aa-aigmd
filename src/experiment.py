## Wykonanie i zapisanie surowych wyników baseline'u i ataków.

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

from src.attacks.registry import ATTACK_REGISTRY
from src.audio import load_audio
from src.detectors.registry import Detector, DetectorMetadata
from src.domains import WaveformDomain
from src.metrics import linf_norm, snr_db, try_pesq, try_stoi


@dataclass
class AttackSweepResult:
    ## Średnie prawdopodobieństwa fake dla CLI i ścieżka pełnych metryk CSV.
    mean_p_fake_by_budget: dict[float, float]
    metrics_path: Path


@dataclass
class BaselineResult:
    ## Podsumowanie grup oraz ścieżka pełnych wyników baseline'u CSV.
    summary: pd.DataFrame
    metrics_path: Path


ERROR_COLUMNS = ["stage", "file", "budget", "error_type", "error_message"]
ATTACK_COLUMNS = [
    "file",
    "budget",
    "audio",
    "p_fake_after",
    "snr_db",
    "linf",
    "stoi",
    "pesq",
]


def _start_csv(path: Path, columns: list[str]) -> None:
    ## Tworzy plik wyników i zapisuje nagłówek przed przetwarzaniem próbek.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()


def _append_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    ## Dopisuje jeden rekord; zamknięcie pliku opróżnia bufor po każdym wierszu.
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writerow(row)


def _check_representation(detector: Detector, representation: str) -> None:
    ## Sprawdzenie obsługiwanej reprezentacji.
    if representation not in {"waveform", "mel"}:
        raise ValueError(f"Unsupported attack representation: {representation}")
    if representation == "mel" and not hasattr(detector, "create_mel_domain"):
        raise TypeError("The selected detector does not expose a mel-feature attack path")


## ==================== Podstawowa detekcja bez ataku ====================

def run_baseline(
    detector: Detector,
    meta: DetectorMetadata,
    data_dir: Path,
    results_dir: Path,
    device: torch.device,
    representation: str = "waveform",
    reconstruction: str = "source_phase",
) -> BaselineResult:
    ## Przygotowanie przebiegu: sprawdzenie trybu, ścieżki wyniku i wejściowych plików.
    _check_representation(detector, representation)
    path = results_dir / representation / "baseline.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline: {path}")
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in {".mp3", ".wav"})
    if not files:
        raise FileNotFoundError(f"No WAV or MP3 audio files found under {data_dir}")

    baseline_columns = (
        ["file", "p_fake"]
        if representation == "waveform"
        else [
            "file",
            "p_fake",
            "feature_p_fake",
            "recon_snr_db",
            "recon_stoi",
        ]
    )
    errors_path = path.parent / "errors.csv"
    _start_csv(path, baseline_columns)
    _start_csv(errors_path, ERROR_COLUMNS)

    fake_label = meta.fake_label_idx if meta.fake_label_idx is not None else 1
    print(f"Found {len(files)} audio files. Starting baseline...", flush=True)
    successful_count = 0
    error_count = 0
    group_counts: dict[str, dict[str, int]] = {}
    
    ## Obejmujemy zarówno real, jak i fake, bez filtrowania po decyzji modelu (której jeszcze nie ma).
    for index, file in enumerate(files, start=1):
        relative_path = file.relative_to(data_dir).as_posix()
        print(f"[{index}/{len(files)}] Processing {relative_path}", flush=True)
        try:
            ## Dekodowanie, zmiana częstotliwości próbkowania oraz dopasowanie długości.
            waveform, sample_rate = load_audio(str(file), target_sr=meta.sample_rate)
            original = detector.prepare_audio(waveform).unsqueeze(0).to(device)
            row = {"file": relative_path}
            with torch.no_grad():
                ## Baseline nie liczy gradientów. 
                ## W trybie waveform zapisujemy tylko predykcję cech.
                if representation == "waveform":
                    row["p_fake"] = detector(original).softmax(-1)[0, fake_label].item()
                else:
                    ## W trybie mel zapisujemy zarówno predykcję cech, jak i predykcję audio po ich rekonstrukcji.
                    features, domain = detector.create_mel_domain(original, reconstruction)
                    audio = domain.to_waveform(features).to(device)

                    ## Dodatkowe wzmocnienie wyznaczone metodą najmniejszych kwadratów dopasowuje głośność rekonstrukcji tylko na potrzeby metryk JAKOŚCI.
                    ## Detektor dostaje niezrównane, zrekonstruowane audio.
                    gain = torch.dot(original[0], audio[0]) / audio[0].square().sum().clamp_min(1e-8)

                    row["p_fake"] = detector(audio).softmax(-1)[0, fake_label].item()
                    row["feature_p_fake"] = (detector.predict_mel_features(features).softmax(-1)[0, fake_label].item())
                    row["recon_snr_db"] = snr_db(original[0], audio[0] * gain)
                    row["recon_stoi"] = try_stoi(original[0], audio[0] * gain, sample_rate)

            _append_csv_row(path, row, baseline_columns)
            successful_count += 1
            group = "/".join(relative_path.split("/")[:2])
            counts = group_counts.setdefault(group, {"size": 0, "flagged_fake": 0})
            counts["size"] += 1
            counts["flagged_fake"] += int(row["p_fake"] > 0.5)
            print(f"[{index}/{len(files)}] Done {relative_path} (p_fake={row['p_fake']:.4f})", flush=True)

        except Exception as error:
            ## Błąd jednego pliku zapisujemy, ale nie przerywamy całego baseline'u.
            _append_csv_row(
                errors_path,
                {
                    "stage": "baseline",
                    "file": relative_path,
                    "budget": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                ERROR_COLUMNS,
            )
            error_count += 1
            print(f"[{index}/{len(files)}] ERROR {relative_path}: {type(error).__name__}: {error}", flush=True)

    ## W pamięci pozostaje tylko małe podsumowanie, pogrupowane po katalogu źródła.
    summary = pd.DataFrame.from_dict(group_counts, orient="index")
    summary.index.name = "group"
    print(f"Baseline complete: {successful_count}/{len(files)} succeeded, {error_count} failed. Errors: {errors_path}", flush=True)
    return BaselineResult(summary=summary, metrics_path=path)


## ==================== Wykonanie ataków dla każdej kombinacji budżetu i pliku ====================

def run_attack_sweep(
    detector: Detector,
    meta: DetectorMetadata,
    files: list[Path],
    data_dir: Path,
    attack: str,
    budgets: list[float],
    results_dir: Path,
    device: torch.device,
    representation: str = "waveform",
    reconstruction: str = "source_phase",
    alpha: float = 0.001,
    steps: int = 20,
    target_label: int = 0,
) -> AttackSweepResult:
    ## Przygotowanie przebiegów: sprawdzenie trybu, ścieżki wyniku i wejściowych plików.
    _check_representation(detector, representation)
    attack_entry = ATTACK_REGISTRY[attack]
    if representation == "mel" and attack_entry.waveform_only:
        raise ValueError(f"Attack {attack!r} is defined in waveform space only")
    representation_dir = results_dir / representation

    ## Przerwij przed zapisem, jeśli nadpisałby poprzednie wyniki.
    if not (representation_dir / "baseline.csv").exists():
        raise FileNotFoundError(f"Run `baseline --representation {representation}` with this --results-dir first")
    output_dir = representation_dir / attack
    metrics_path = output_dir / "metrics.csv"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing attack results: {output_dir}")
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    attack_columns = ATTACK_COLUMNS.copy()
    if representation == "mel":
        attack_columns.extend(["feature_p_fake_after", "feature_linf"])
    fake_label = meta.fake_label_idx if meta.fake_label_idx is not None else 1
    true_label = torch.tensor([fake_label], device=device)

    ## Metryki i błędy zapisywane po każdej próbie.
    _start_csv(metrics_path, attack_columns)
    errors_path = output_dir / "errors.csv"
    _start_csv(errors_path, ERROR_COLUMNS)
    successful_count = 0
    error_count = 0
    p_fake_sums = {budget: 0.0 for budget in budgets}
    p_fake_counts = {budget: 0 for budget in budgets}

    ## Zewnętrzna pętla porównuje budżety, a wewnętrzna przetwarza po jednym pliku.
    ## Atakujemy wszystkie przekazane nagrania fake.
    for budget in budgets:
        for index, path in enumerate(files, start=1):
            rel = path.relative_to(data_dir)
            print(f"[{attack} {budget}] [{index}/{len(files)}] Processing {rel.as_posix()}", flush=True)
            try:
                ## Każda próba zaczyna się od czystego audio.
                waveform, sample_rate = load_audio(str(path), target_sr=meta.sample_rate)
                original_waveform = (detector.prepare_audio(waveform).unsqueeze(0).to(device))

                if representation == "waveform":
                    original_representation = original_waveform     ## Zmienna ataku: oryginalne próbki audio
                    attack_model = detector                         ## Pełna ścieżka detektora = cały frontend
                    domain = WaveformDomain()                       ## Budżet perturbacji i projekcja w jednostkach waveform
                    clean_output = original_waveform                ## Czyste audio do odniesienia dla metryk jakości
                else:
                    ## Zamiana audio na cechy mel i domenę ich rekonstrukcji.
                    mel_features, mel_domain = detector.create_mel_domain(original_waveform, reconstruction)

                    original_representation = mel_features          ## Zmienna ataku: znormalizowane cechy mel
                    attack_model = detector.predict_mel_features    ## Tylko klasyfikator za frontendem: gradient względem cech
                    domain = mel_domain                             ## Budżet perturbacji i projekcja w jednostkach cech mel

                    ## Rekonstrukcja czystych cech jako odniesienie dla metryk
                    clean_output = domain.to_waveform(original_representation).to(device)

                ## Atak zwraca zaburzoną reprezentację: próbki w trybie waveform, komórki cech w trybie mel.
                ## Jawne nazwy pokazują, co dostaje implementacja ataku.
                adversarial = attack_entry.run(
                    model=attack_model,
                    x=original_representation,
                    labels=true_label,
                    budget=budget,
                    target=target_label,
                    alpha=alpha,
                    steps=steps,
                    domain=domain,
                )

                ## Dopiero teraz zamieniamy wynik na audio.
                adversarial_waveform = domain.to_waveform(adversarial).to(device)

                ## Zrekonstruowane audio ocenia pełny detektor. 
                ## W trybie mel zapisujemy też wynik klasyfikatora cech, by pokazać różnicę.
                ## Predykcje są liczone na tensorach w pamięci, przed kwantyzacją WAV.
                with torch.no_grad():
                    p_fake_after = (
                        detector(adversarial_waveform)
                        .softmax(-1)[0, fake_label]
                        .item()
                    )
                    feature_p_fake_after = (
                        attack_model(adversarial).softmax(-1)[0, fake_label].item()
                        if representation == "mel"
                        else None
                    )

                ## Metryki jakości do clean_output: waveformu wejściowego albo rekonstrukcji dla ataku mel.
                output_path = audio_dir / rel.parent / f"{rel.stem}_{budget}.wav"
                if output_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing audio: {output_path}")
                row = {
                    "file": rel.as_posix(),
                    "budget": budget,
                    "audio": output_path.relative_to(results_dir).as_posix(),
                    "p_fake_after": p_fake_after,
                    "snr_db": snr_db(clean_output[0], adversarial_waveform[0]),
                    "linf": linf_norm(clean_output[0], adversarial_waveform[0]),
                    "stoi": try_stoi(clean_output[0], adversarial_waveform[0], sample_rate),
                    "pesq": try_pesq(clean_output[0], adversarial_waveform[0], sample_rate),
                }
                if representation == "mel":
                    row["feature_p_fake_after"] = feature_p_fake_after
                    row["feature_linf"] = linf_norm(original_representation[0], adversarial[0])

                ## Zapis wyników
                clean_path = representation_dir / "clean" / rel.with_suffix(".wav")
                if not clean_path.exists():
                    clean_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        str(clean_path),
                        clean_output[0].detach().cpu().numpy(),
                        sample_rate,
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    str(output_path),
                    adversarial_waveform[0].detach().cpu().numpy(),
                    sample_rate,
                )
                _append_csv_row(metrics_path, row, attack_columns)
                successful_count += 1
                p_fake_sums[budget] += p_fake_after
                p_fake_counts[budget] += 1
                print(f"[{attack} {budget}] [{index}/{len(files)}] Done {rel.as_posix()}: P(fake) after = {p_fake_after:.3f}", flush=True)

            except Exception as error:
                ## Zachowujemy kontekst nieudanej próby i przechodzimy do następnej.
                _append_csv_row(
                    errors_path,
                    {
                        "stage": attack,
                        "file": rel.as_posix(),
                        "budget": budget,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    ERROR_COLUMNS,
                )
                error_count += 1
                print(f"[{attack} {budget}] [{index}/{len(files)}] ERROR {rel.as_posix()}: {type(error).__name__}: {error}", flush=True)

    print(f"{attack} complete: {successful_count}/{len(files) * len(budgets)} succeeded, {error_count} failed. Errors: {errors_path}", flush=True)

    ## Zapis konfiguracji pozwala odtworzyć ustawienia wspólne dla całego przebiegu.
    config = {
        "argv": sys.argv[1:],
        "time": datetime.now().isoformat(timespec="seconds"),
        "detector": type(detector).__name__,
        "attack": attack,
        "representation": representation,
        "reconstruction": reconstruction if representation == "mel" else None,
        "budget_name": attack_entry.budget_name,
        "budgets": budgets,
        "alpha": alpha,
        "steps": steps,
        "target_label": target_label,
        "n_files": len(files),
    }
    checkpoint = getattr(meta, "checkpoint", None)
    if checkpoint is not None:
        config["checkpoint"] = checkpoint
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    ## CLI pokazuje średnie P(fake) per budżet, pełne wiersze są w CSV.
    mean_p_fake_by_budget = {
        budget: p_fake_sums[budget] / p_fake_counts[budget]
        for budget in budgets
        if p_fake_counts[budget]
    }
    return AttackSweepResult(
        mean_p_fake_by_budget=mean_p_fake_by_budget,
        metrics_path=metrics_path,
    )
