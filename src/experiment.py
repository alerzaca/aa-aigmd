## Wykonanie i zapisanie surowych wyników baseline'u i ataków.

from __future__ import annotations

import csv
import hashlib
import json
import math
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


def _ensure_csv(path: Path, columns: list[str]) -> None:
    ## Zachowuje zapisane wyniki przy wznowieniu i sprawdza zgodność nagłówka.
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), [])
        if header:
            if header != columns:
                raise ValueError(f"Unexpected CSV columns in {path}: {header}")
            with path.open("r+b") as file:
                file.seek(0, 2)
                end = file.tell()
                if end:
                    file.seek(-1, 2)
                    if file.read(1) != b"\n":
                        position = end - 1
                        while position >= 0:
                            file.seek(position)
                            if file.read(1) == b"\n":
                                file.truncate(position + 1)
                                break
                            position -= 1
                        else:
                            file.truncate(0)
            if path.stat().st_size:
                return

    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()


def _is_number(value: str | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except ValueError:
        return False


def _input_manifest(files: list[Path], data_dir: Path) -> str:
    names = sorted(path.relative_to(data_dir).as_posix() for path in files)
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _ensure_config(
    path: Path,
    config: dict[str, object],
    fields: tuple[str, ...],
    results_path: Path,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Invalid run configuration in {path}")
        changed = [
            f"{field} (saved={existing.get(field)!r}, requested={config.get(field)!r})"
            for field in fields
            if existing.get(field) != config.get(field)
        ]
        if changed:
            raise ValueError(
                f"Cannot resume {results_path}; run settings changed: "
                f"{', '.join(changed)}. Use a new results directory."
            )
        return

    if results_path.exists():
        print(
            f"WARNING: {path} is missing; assuming current settings match "
            "these legacy results.",
            file=sys.stderr,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


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


def _record_baseline_result(
    group_counts: dict[str, dict[str, int]], relative_path: str, p_fake: float
) -> None:
    group = "/".join(relative_path.split("/")[:2])
    counts = group_counts.setdefault(group, {"size": 0, "flagged_fake": 0})
    counts["size"] += 1
    counts["flagged_fake"] += int(p_fake > 0.5)


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
    config = {
        "argv": sys.argv[1:],
        "detector": type(detector).__name__,
        "representation": representation,
        "reconstruction": reconstruction if representation == "mel" else None,
        "n_files": len(files),
        "input_manifest": _input_manifest(files, data_dir),
        "sample_rate": meta.sample_rate,
        "fake_label_idx": meta.fake_label_idx,
    }
    checkpoint = getattr(meta, "checkpoint", None)
    if checkpoint is not None:
        config["checkpoint"] = checkpoint
    _ensure_config(
        path.parent / "baseline_config.json",
        config,
        (
            "detector",
            "checkpoint",
            "representation",
            "reconstruction",
            "n_files",
            "input_manifest",
            "sample_rate",
            "fake_label_idx",
        ),
        path,
    )

    errors_path = path.parent / "errors.csv"
    _ensure_csv(path, baseline_columns)
    _ensure_csv(errors_path, ERROR_COLUMNS)

    fake_label = meta.fake_label_idx if meta.fake_label_idx is not None else 1
    print(f"Found {len(files)} audio files. Starting baseline...", flush=True)
    current_files = {file.relative_to(data_dir).as_posix() for file in files}
    completed_files = set()
    group_counts: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            relative_path = row.get("file")
            if (
                relative_path not in current_files
                or relative_path in completed_files
                or None in row
                or any(row.get(column) is None for column in baseline_columns)
                or not _is_number(row.get("p_fake"))
            ):
                continue
            completed_files.add(relative_path)
            _record_baseline_result(
                group_counts, relative_path, float(row["p_fake"])
            )

    successful_count = len(completed_files)
    error_count = 0
    
    ## Obejmujemy zarówno real, jak i fake, bez filtrowania po decyzji modelu (której jeszcze nie ma).
    for index, file in enumerate(files, start=1):
        relative_path = file.relative_to(data_dir).as_posix()
        if relative_path in completed_files:
            print(f"[{index}/{len(files)}] Skipping completed {relative_path}", flush=True)
            continue
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
            completed_files.add(relative_path)
            successful_count += 1
            _record_baseline_result(
                group_counts, relative_path, float(row["p_fake"])
            )
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

    if not (representation_dir / "baseline.csv").exists():
        raise FileNotFoundError(f"Run `baseline --representation {representation}` with this --results-dir first")
    output_dir = representation_dir / attack
    metrics_path = output_dir / "metrics.csv"
    attack_columns = ATTACK_COLUMNS.copy()
    if representation == "mel":
        attack_columns.extend(["feature_p_fake_after", "feature_linf"])

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
        "input_manifest": _input_manifest(files, data_dir),
        "sample_rate": meta.sample_rate,
        "fake_label_idx": meta.fake_label_idx,
    }
    checkpoint = getattr(meta, "checkpoint", None)
    if checkpoint is not None:
        config["checkpoint"] = checkpoint
    _ensure_config(
        output_dir / "config.json",
        config,
        (
            "detector",
            "checkpoint",
            "attack",
            "representation",
            "reconstruction",
            "budget_name",
            "budgets",
            "alpha",
            "steps",
            "target_label",
            "n_files",
            "input_manifest",
            "sample_rate",
            "fake_label_idx",
        ),
        metrics_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    _ensure_csv(metrics_path, attack_columns)
    errors_path = output_dir / "errors.csv"
    _ensure_csv(errors_path, ERROR_COLUMNS)

    fake_label = meta.fake_label_idx if meta.fake_label_idx is not None else 1
    true_label = torch.tensor([fake_label], device=device)

    ## Metryki i błędy zapisywane po każdej próbie.
    current_files = {path.relative_to(data_dir).as_posix() for path in files}
    allowed_budgets = set(budgets)
    completed_attempts: set[tuple[str, float]] = set()
    p_fake_sums = {budget: 0.0 for budget in budgets}
    p_fake_counts = {budget: 0 for budget in budgets}
    with metrics_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            relative_path = row.get("file")
            budget_value = row.get("budget")
            p_fake_value = row.get("p_fake_after")
            if (
                not isinstance(relative_path, str)
                or not isinstance(budget_value, str)
                or not isinstance(p_fake_value, str)
                or None in row
                or any(row.get(column) is None for column in attack_columns)
                or not _is_number(budget_value)
                or not _is_number(p_fake_value)
            ):
                continue
            budget_number = float(budget_value)
            key = (relative_path, budget_number)
            if (
                relative_path not in current_files
                or budget_number not in allowed_budgets
                or key in completed_attempts
            ):
                continue
            completed_attempts.add(key)
            p_fake_sums[budget_number] += float(p_fake_value)
            p_fake_counts[budget_number] += 1

    successful_count = len(completed_attempts)
    error_count = 0

    ## Zewnętrzna pętla porównuje budżety, a wewnętrzna przetwarza po jednym pliku.
    ## Atakujemy wszystkie przekazane nagrania fake.
    for budget in budgets:
        for index, path in enumerate(files, start=1):
            rel = path.relative_to(data_dir)
            key = (rel.as_posix(), budget)
            if key in completed_attempts:
                print(f"[{attack} {budget}] [{index}/{len(files)}] Skipping completed {rel.as_posix()}", flush=True)
                continue
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
                completed_attempts.add(key)
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
