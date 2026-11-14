from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import load_audio
from src.detectors.registry import load_detector
from analysis.sonics_miss_analysis import canonical_start, make_windows

RUN = ROOT / "results" / "sonics" / "sonics_baseline_2"
BASELINE = RUN / "waveform" / "baseline.csv"
DATA = ROOT / "data" / "fake" / "sonics"
OUTPUT = RUN / "analysis" / "all_files"
BATCH_SIZE = 32
CHECKPOINT_EVERY_FILES = 25
REPLAY_TOLERANCE = 1e-4

WINDOW_COLUMNS = [
    "file",
    "baseline_p_fake",
    "start_seconds",
    "is_baseline_crop",
    "logit",
    "p_fake",
]
SUMMARY_COLUMNS = [
    "file",
    "baseline_p_fake",
    "duration_seconds",
    "window_count",
    "baseline_crop_start_seconds",
    "replayed_baseline_crop_p_fake",
    "baseline_crop_abs_error",
    "min_window_p_fake",
    "min_window_start_seconds",
    "max_window_p_fake",
    "max_window_start_seconds",
    "fraction_windows_p_fake_le_0_5",
    "fraction_windows_p_fake_le_0_1",
    "fraction_windows_p_fake_lt_0_01",
    "windows_p_fake_le_0_5",
]


def read_baseline() -> pd.DataFrame:
    ## Plik baseline jest źródłem listy wejść i punktów odniesienia do kontroli ponownego wyniku.
    frame = pd.read_csv(BASELINE)
    if "file" not in frame.columns:
        candidate = frame.columns[0]
        if not frame[candidate].astype(str).str.endswith(".mp3").all():
            raise ValueError(f"Cannot identify baseline file column: {frame.columns}")
        frame = frame.rename(columns={candidate: "file"})
        print(
            f"Note: using first column {candidate!r} as the file list; "
            "baseline.csv was not modified.",
            flush=True,
        )

    required = {"file", "p_fake"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected baseline columns {required}, got {frame.columns}")
    if frame["file"].duplicated().any():
        duplicates = frame.loc[frame["file"].duplicated(), "file"].head().tolist()
        raise ValueError(f"Duplicate baseline file rows: {duplicates}")
    missing = [name for name in frame["file"] if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audio inputs for baseline rows: {missing[:5]}")
    return frame


def extract_batch_features(detector, waveforms: torch.Tensor) -> torch.Tensor:
    ## Próg normalizacji chroni przed dzieleniem przez zero dla cichych wycinków.
    extractor = detector.model.ft_extractor
    normalized = waveforms / waveforms.std(dim=-1, correction=0, keepdim=True).clamp_min(1e-6)
    mel_power = extractor.audio2melspec(normalized.float())
    mel_db = extractor.amplitude_to_db(mel_power.unsqueeze(1)).squeeze(1)
    return extractor.normalizer(mel_db)


def process_file(detector, file_name: str, baseline_score: float) -> tuple[list[dict], dict]:
    ## Oceniaj wszystkie 5-sekundowe wycinki danego nagrania, partiami dla oszczędności pamięci.
    audio, sample_rate = load_audio(str(DATA / file_name), target_sr=detector.meta.sample_rate)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz for {file_name}; got {sample_rate}")

    target_count = detector.meta.max_len_samples
    canonical = canonical_start(audio.numel(), target_count)
    windows = make_windows(audio, target_count, canonical)
    scores: list[float] = []
    logits: list[float] = []

    with torch.inference_mode():
        for offset in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[offset : offset + BATCH_SIZE]
            batch = torch.stack([window for _, window, _ in batch_windows])
            features = extract_batch_features(detector, batch)
            output = detector.predict_mel_features(features)
            batch_logits = output[:, detector.meta.fake_label_idx]
            logits.extend(batch_logits.tolist())
            scores.extend(torch.sigmoid(batch_logits).tolist())

    ## Dokładny wycinek baseline pozwala wykryć rozbieżność w preprocessingu lub modelu.
    canonical_indices = [index for index, (_, _, is_canonical) in enumerate(windows) if is_canonical]
    if len(canonical_indices) != 1:
        raise RuntimeError(f"Expected one baseline crop for {file_name}")
    
    canonical_index = canonical_indices[0]
    canonical_score = float(scores[canonical_index])
    replay_error = abs(canonical_score - baseline_score)
    if replay_error > REPLAY_TOLERANCE:
        raise RuntimeError(
            f"Baseline replay mismatch for {file_name}: "
            f"{canonical_score:.9g} vs {baseline_score:.9g}"
        )

    records = [
        {
            "file": file_name,
            "baseline_p_fake": baseline_score,
            "start_seconds": start / sample_rate,
            "is_baseline_crop": is_canonical,
            "logit": float(logit),
            "p_fake": float(probability),
        }
        for (start, _, is_canonical), logit, probability in zip(
            windows, logits, scores
        )
    ]

    min_index = int(np.argmin(scores))
    max_index = int(np.argmax(scores))
    summary = {
        "file": file_name,
        "baseline_p_fake": baseline_score,
        "duration_seconds": audio.numel() / sample_rate,
        "window_count": len(windows),
        "baseline_crop_start_seconds": canonical / sample_rate,
        "replayed_baseline_crop_p_fake": canonical_score,
        "baseline_crop_abs_error": replay_error,
        "min_window_p_fake": float(scores[min_index]),
        "min_window_start_seconds": windows[min_index][0] / sample_rate,
        "max_window_p_fake": float(scores[max_index]),
        "max_window_start_seconds": windows[max_index][0] / sample_rate,
        "fraction_windows_p_fake_le_0_5": float(np.mean(np.asarray(scores) <= 0.5)),
        "fraction_windows_p_fake_le_0_1": float(np.mean(np.asarray(scores) <= 0.1)),
        "fraction_windows_p_fake_lt_0_01": float(np.mean(np.asarray(scores) < 0.01)),
        "windows_p_fake_le_0_5": int(np.count_nonzero(np.asarray(scores) <= 0.5)),
    }
    return records, summary


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    ## Dopisywanie z nagłówkiem tylko dla nowego pliku umożliwia wznowienie po checkpointach.
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize_results(summary: pd.DataFrame) -> str:
    ## Statystyki są rozdzielone według tego, czy baseline wykrył plik jako fake.
    lines = [
        f"Files analyzed: {len(summary)}",
        f"Total scored windows: {int(summary['window_count'].sum())}",
        f"Maximum canonical replay error: {summary['baseline_crop_abs_error'].max():.8g}",
    ]
    for label, subset in (
        ("Baseline detected fake (P(fake) > 0.5)", summary["baseline_p_fake"] > 0.5),
        ("Baseline classified real (P(fake) <= 0.5)", summary["baseline_p_fake"] <= 0.5),
    ):
        group = summary.loc[subset]
        if group.empty:
            continue
        lines.extend(
            [
                "",
                f"{label}: {len(group)} files",
                f"  At least one window <= 0.5: {int((group['min_window_p_fake'] <= 0.5).sum())} ({(group['min_window_p_fake'] <= 0.5).mean():.1%})",
                f"  At least one window <= 0.1: {int((group['min_window_p_fake'] <= 0.1).sum())} ({(group['min_window_p_fake'] <= 0.1).mean():.1%})",
                f"  At least one window < 0.01: {int((group['min_window_p_fake'] < 0.01).sum())} ({(group['min_window_p_fake'] < 0.01).mean():.1%})",
                f"  Median minimum-window P(fake): {group['min_window_p_fake'].median():.6f}",
                f"  Median fraction of windows <= 0.5: {group['fraction_windows_p_fake_le_0_5'].median():.1%}",
            ]
        )
    return "\n".join(lines)


def write_overview(summary: pd.DataFrame, output: Path) -> None:
    ## Kolor oznacza decyzję baseline; wykres porównuje ją z najsłabszym wynikiem z całego nagrania.
    detected = summary.loc[summary["baseline_p_fake"] > 0.5]
    missed = summary.loc[summary["baseline_p_fake"] <= 0.5]
    ordered = summary.sort_values("baseline_p_fake").reset_index(drop=True)
    colors = np.where(ordered["baseline_p_fake"] > 0.5, "#175cd3", "#b42318")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].scatter(
        ordered["baseline_p_fake"],
        ordered["min_window_p_fake"],
        c=colors,
        s=10,
        alpha=0.55,
        linewidths=0,
    )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(0.5, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        title="Baseline score vs weakest 5-second window",
        xlabel="Baseline P(fake)",
        ylabel="Minimum window P(fake)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].scatter([], [], color="#175cd3", label=f"Baseline detected ({len(detected)})")
    axes[0].scatter([], [], color="#b42318", label=f"Baseline missed ({len(missed)})")
    axes[0].legend(loc="upper left")

    bins = np.linspace(0, 1, 41)
    if not detected.empty:
        axes[1].hist(
            detected["min_window_p_fake"],
            bins=bins,
            alpha=0.6,
            color="#175cd3",
            label=f"Baseline detected ({len(detected)})",
            density=True,
        )
    if not missed.empty:
        axes[1].hist(
            missed["min_window_p_fake"],
            bins=bins,
            alpha=0.6,
            color="#b42318",
            label=f"Baseline missed ({len(missed)})",
            density=True,
        )
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Distribution of weakest-window scores",
        xlabel="Minimum window P(fake)",
        ylabel="Density",
        xlim=(0, 1),
    )
    axes[1].legend()
    #fig.suptitle(
    #    "All files: a weak window is any 5-second window with P(fake) ≤ 0.5\n"
    #    "Overlapping windows use a 2.5-second hop and include the exact baseline crop"
    #)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    window_path = OUTPUT / "window_scores.csv"
    summary_path = OUTPUT / "per_file_summary.csv"
    notes_path = OUTPUT / "summary.txt"
    state_path = OUTPUT / "progress.txt"
    weak_path = OUTPUT / "baseline_detected_with_weak_window.csv"
    plot_path = OUTPUT / "weak_window_overview.png"

    ## Oba pliki checkpoint muszą istnieć razem, bo ich wiersze odpowiadają sobie 1:1.
    baseline = read_baseline()
    if window_path.exists() != summary_path.exists():
        raise RuntimeError("Only one checkpoint CSV exists.")

    done: set[str] = set()
    summary_rows: list[dict] = []
    if summary_path.exists():
        ## Podsumowanie określa już przeliczone pliki; sprawdzamy, czy pochodzą z tego baseline.
        completed = pd.read_csv(summary_path)
        done = set(completed["file"].astype(str))
        summary_rows = completed.to_dict("records")
        known_files = set(baseline["file"].astype(str))
        unexpected = done - known_files
        if unexpected:
            raise ValueError(f"Checkpoint has files absent from baseline: {unexpected}")

    ## Pomiń pliki ukończone wcześniej, zachowując możliwość bezpiecznego wznowienia sweepu.
    todo = baseline.loc[~baseline["file"].isin(done)]
    if todo.empty:
        summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
        report = summarize_results(summary)
        weak = summary.loc[
            (summary["baseline_p_fake"] > 0.5)
            & (summary["min_window_p_fake"] <= 0.5)
        ].sort_values(["min_window_p_fake", "baseline_p_fake"])
        weak.to_csv(weak_path, index=False)
        write_overview(summary, plot_path)
        notes_path.write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"Baseline-detected files with a window P(fake) <= 0.5: {len(weak)}")
        print(f"Output directory: {OUTPUT}")
        return

    ## Weryfikuj kontrakt audio modelu przed obliczeniami, bo pozycje wycinków zależą od długości wejścia.
    detector = load_detector("sonics").eval()
    if detector.meta.sample_rate != 16000 or detector.meta.max_len_samples != 80000:
        raise RuntimeError(f"Unexpected SONICS audio contract: {detector.meta}")

    pending_windows: list[dict] = []
    pending_summaries: list[dict] = []
    total = len(baseline)
    for offset, row in enumerate(todo.itertuples(index=False), start=1):
        file_name = str(row.file)
        baseline_score = float(row.p_fake)
        records, file_summary = process_file(detector, file_name, baseline_score)
        pending_windows.extend(records)
        pending_summaries.append(file_summary)
        summary_rows.append(file_summary)

        ## Zapisuj jednocześnie wyniki okien i podsumowania, aby ograniczyć utratę pracy po przerwaniu.
        if offset % CHECKPOINT_EVERY_FILES == 0 or offset == len(todo):
            append_csv(window_path, pending_windows, WINDOW_COLUMNS)
            append_csv(summary_path, pending_summaries, SUMMARY_COLUMNS)
            pending_windows.clear()
            pending_summaries.clear()

            completed_count = len(done) + offset
            summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
            partial_report = summarize_results(summary)
            notes_path.write_text(partial_report + "\n", encoding="utf-8")
            state_path.write_text(f"completed_files={completed_count}\ntotal_files={total}\n", encoding="utf-8")
            latest = summary_rows[-1]
            print(
                f"[{completed_count}/{total}] {file_name}: "
                f"{latest['window_count']} windows, "
                f"min P(fake)={latest['min_window_p_fake']:.4f}, "
                f"max P(fake)={latest['max_window_p_fake']:.4f}",
                flush=True,
            )

    ## Kontrole kompletności gwarantują dokładnie jeden wynik dla każdego rekordu baseline.
    final_summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if len(final_summary) != len(baseline):
        raise RuntimeError(f"Completed {len(final_summary)} of {len(baseline)} baseline files")
    if final_summary["file"].duplicated().any():
        raise RuntimeError("Duplicate file rows in full-sweep summary")
    
    report = summarize_results(final_summary)
    weak = final_summary.loc[
        (final_summary["baseline_p_fake"] > 0.5)
        & (final_summary["min_window_p_fake"] <= 0.5)
    ].sort_values(["min_window_p_fake", "baseline_p_fake"])

    weak.to_csv(weak_path, index=False)
    write_overview(final_summary, plot_path)
    notes_path.write_text(report + "\n", encoding="utf-8")
    report += (
        "\n\n"
        f"Baseline-detected files with at least one window P(fake) <= 0.5: "
        f"{len(weak)} ({len(weak) / max(len(final_summary.loc[final_summary['baseline_p_fake'] > 0.5]), 1):.1%} of baseline detections)"
    )

    notes_path.write_text(report + "\n", encoding="utf-8")
    state_path.write_text(f"completed_files={len(final_summary)}\ntotal_files={len(baseline)}\n", encoding="utf-8")
    print(report)
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
