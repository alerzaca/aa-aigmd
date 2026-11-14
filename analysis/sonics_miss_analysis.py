from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from mutagen.mp3 import MP3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import load_audio
from src.detectors.registry import load_detector

RUN = ROOT / "results" / "sonics" / "sonics_baseline_2"
BASELINE = RUN / "waveform" / "baseline.csv"
DATA = ROOT / "data" / "fake" / "sonics"
OUTPUT = RUN / "analysis"
TARGET_SECONDS = 5.0
WINDOW_HOP_SECONDS = 2.5
CONFIDENT_FAKE_THRESHOLD = 0.99


def duration_seconds(file_name: str, cache: dict[str, float]) -> float:
    ## Długość MP3 jest używana wielokrotnie przy dopasowywaniu; cache eliminuje ponowne parsowanie.
    if file_name not in cache:
        cache[file_name] = float(MP3(DATA / file_name).info.length)
    return cache[file_name]


def select_controls(results: pd.DataFrame, durations: dict[str, float]) -> dict[str, str]:
    ## Kontrolą jest pewnie wykryty fake z tego samego generatora i wariantu, o najbliższym czasie.
    candidates = results.loc[results["p_fake"] >= CONFIDENT_FAKE_THRESHOLD].copy()
    controls: dict[str, str] = {}
    used: set[str] = set()

    for (generator, variant), misses in results.loc[results["miss"]].groupby(["generator", "variant"]):
        candidates_in_group = candidates.loc[
            (candidates["generator"] == generator)
            & (candidates["variant"] == variant)
        ]
        misses = misses.sort_values("duration_seconds")
        for miss in misses.itertuples(index=False):
            ## Nie używaj tej samej próbki kontrolnej dla wielu przeoczeń.
            available = candidates_in_group.loc[~candidates_in_group["file"].isin(used)]
            if available.empty:
                raise RuntimeError(f"No unused high-confidence control for {generator}/{variant}")
            distances = available["duration_seconds"].sub(miss.duration_seconds).abs()
            control_index = distances.idxmin()
            control_file = str(available.loc[control_index, "file"])
            controls[miss.file] = control_file
            used.add(control_file)
    return controls


def canonical_start(sample_count: int, target_count: int) -> int:
    ## SONICS pobiera wycinek z 3/4 nadmiarowej długości; identyczny wybór jest potrzebny do replayu baseline.
    if sample_count <= target_count:
        return 0
    return int((sample_count - target_count) / 4 * 3)


def make_windows(waveform: torch.Tensor, target_count: int, canonical: int) -> list[tuple[int, torch.Tensor, bool]]:
    ## Pokryj całe nagranie oknami o 50% nakładaniu, zachowując też wycinek baseline i końcowy fragment.
    sample_count = waveform.numel()
    if sample_count <= target_count:
        padded = torch.nn.functional.pad(waveform, (0, target_count - sample_count))
        return [(0, padded, True)]

    last_start = sample_count - target_count
    hop = int(WINDOW_HOP_SECONDS * 16000)
    starts = list(range(0, last_start + 1, hop))
    starts.extend([last_start, canonical])
    starts = sorted(set(starts))
    return [(start, waveform[start : start + target_count], start == canonical)for start in starts]


def score_windows(detector, file_name: str, cohort: str, baseline_score: float) -> tuple[list[dict], torch.Tensor, int, float]:
    ## Zwróć wyniki wszystkich okien, dokładny wycinek baseline oraz różnicę replayu.
    audio, sample_rate = load_audio(str(DATA / file_name), target_sr=detector.meta.sample_rate)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz for {file_name}; got {sample_rate}")

    target_count = int(TARGET_SECONDS * sample_rate)
    canonical = canonical_start(audio.numel(), target_count)
    windows = make_windows(audio, target_count, canonical)
    records: list[dict] = []
    probabilities: list[float] = []
    logits: list[float] = []

    with torch.inference_mode():
        for offset in range(0, len(windows), 32):
            batch_windows = windows[offset : offset + 32]
            ## AmplitudeToDB(top_db) dzieli próg między elementami batcha 3D.
            ## Frontend liczony osobno dla każdego okna zachowuje semantykę baseline.
            features = torch.cat(
                [
                    detector.extract_mel_features(window.unsqueeze(0))
                    for _, window, _ in batch_windows
                ]
            )
            output = detector.predict_mel_features(features)
            batch_logits = output[:, detector.meta.fake_label_idx]
            batch_probabilities = torch.sigmoid(batch_logits)

            for (start, _, is_canonical), logit, probability in zip(
                batch_windows, batch_logits.tolist(), batch_probabilities.tolist()
            ):
                probabilities.append(float(probability))
                logits.append(float(logit))
                records.append(
                    {
                        "file": file_name,
                        "cohort": cohort,
                        "baseline_p_fake": baseline_score,
                        "start_seconds": start / sample_rate,
                        "is_baseline_crop": is_canonical,
                        "logit": float(logit),
                        "p_fake": float(probability),
                    }
                )

    if sum(record["is_baseline_crop"] for record in records) != 1:
        raise RuntimeError(f"Expected one baseline crop for {file_name}")
    canonical_index = next(index for index, record in enumerate(records) if record["is_baseline_crop"])

    replay_error = abs(probabilities[canonical_index] - baseline_score)
    canonical_audio = windows[canonical_index][1]
    return records, canonical_audio, len(windows), replay_error


def crop_features(detector, waveform: torch.Tensor) -> dict[str, float]:
    ## Cechy opisują sygnał i mel wycinka baseline; są porównawcze, a nie dowodem przyczynowym.
    samples = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    normalized = samples / max(float(np.std(samples)), 1e-6)
    stft = librosa.stft(normalized, n_fft=1024, hop_length=256)
    magnitude = np.abs(stft)
    power = np.square(magnitude)
    mel = detector.extract_mel_features(waveform.unsqueeze(0))[0].detach().cpu().numpy()

    return {
        "crop_rms": rms,
        "crop_peak": peak,
        "crop_crest_db": float(20 * np.log10(max(peak, 1e-12) / max(rms, 1e-12))),
        "near_silence_fraction": float(np.mean(np.abs(samples) < 1e-4)),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(normalized).mean()),
        "spectral_centroid_hz": float(librosa.feature.spectral_centroid(S=magnitude, sr=16000).mean()),
        "spectral_bandwidth_hz": float(librosa.feature.spectral_bandwidth(S=magnitude, sr=16000).mean()),
        "spectral_rolloff_85_hz": float(librosa.feature.spectral_rolloff(S=magnitude, sr=16000, roll_percent=0.85).mean()),
        "spectral_flatness": float(librosa.feature.spectral_flatness(S=power).mean()),
        "sonics_mel_mean": float(mel.mean()),
        "sonics_mel_std": float(mel.std()),
    }


def write_pair_pdf(detector, summaries: pd.DataFrame, windows: pd.DataFrame, canonical_audio: dict[str, torch.Tensor], path: Path) -> None:
    ## Każda strona zestawia przebieg wyników całego nagrania z melami dopasowanej pary.
    with PdfPages(path) as pdf:
        for row in summaries.sort_values("miss_baseline_p_fake").itertuples(index=False):
            miss_windows = windows.loc[windows["file"] == row.miss_file]
            control_windows = windows.loc[windows["file"] == row.control_file]
            miss_mel = (
                detector.extract_mel_features(
                    canonical_audio[row.miss_file].unsqueeze(0)
                )[0].detach().cpu().numpy()
            )
            control_mel = (
                detector.extract_mel_features(
                    canonical_audio[row.control_file].unsqueeze(0)
                )[0].detach().cpu().numpy()
            )
            lower = float(np.quantile(np.concatenate([miss_mel.ravel(), control_mel.ravel()]), 0.02))
            upper = float(np.quantile(np.concatenate([miss_mel.ravel(), control_mel.ravel()]), 0.98))

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            axes[0, 0].plot(
                miss_windows["start_seconds"] / 60,
                miss_windows["p_fake"],
                label=f"miss (baseline {row.miss_baseline_p_fake:.3f})",
                color="#b42318",
            )
            axes[0, 0].plot(
                control_windows["start_seconds"] / 60,
                control_windows["p_fake"],
                label=f"confident control (baseline {row.control_baseline_p_fake:.3f})",
                color="#175cd3",
            )
            axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
            axes[0, 0].axvline(
                row.miss_baseline_crop_start_seconds / 60,
                color="#b42318",
                linestyle=":",
                linewidth=1,
            )
            axes[0, 0].axvline(
                row.control_baseline_crop_start_seconds / 60,
                color="#175cd3",
                linestyle=":",
                linewidth=1,
            )
            axes[0, 0].set(
                title="P(fake) across overlapping 5-second windows",
                xlabel="Window start (minutes)",
                ylabel="P(fake)",
                ylim=(-0.03, 1.03),
            )
            axes[0, 0].legend(fontsize=8, loc="best")

            for ax, mel, title in (
                (axes[0, 1], miss_mel, f"Miss crop (p={row.miss_baseline_p_fake:.3f})"),
                (
                    axes[1, 0],
                    control_mel,
                    f"Control crop (p={row.control_baseline_p_fake:.3f})",
                ),
            ):
                image = ax.imshow(
                    mel,
                    origin="lower",
                    aspect="auto",
                    extent=(0, TARGET_SECONDS, 0, mel.shape[0]),
                    cmap="coolwarm",
                    vmin=lower,
                    vmax=upper,
                )
                ax.set(
                    title=title,
                    xlabel="Seconds within the baseline crop",
                    ylabel="SONICS mel bin",
                )
            fig.colorbar(image, ax=[axes[0, 1], axes[1, 0]], shrink=0.78, label="Normalized SONICS mel")

            axes[1, 1].axis("off")
            axes[1, 1].text(
                0,
                1,
                "\n".join(
                    [
                        f"Miss: {row.miss_file}",
                        f"Matched control: {row.control_file}",
                        f"Generator / suffix: {row.generator} / {row.variant}",
                        f"Duration (miss/control): {row.miss_duration_seconds:.1f} / {row.control_duration_seconds:.1f} s",
                        f"Canonical crop start (miss/control): {row.miss_baseline_crop_start_seconds:.2f} / {row.control_baseline_crop_start_seconds:.2f} s",
                        f"Miss max-window P(fake): {row.miss_max_window_p_fake:.4f}",
                        f"Miss windows above 0.5: {row.miss_fraction_windows_detected:.1%}",
                        f"Control max-window P(fake): {row.control_max_window_p_fake:.4f}",
                        f"Control windows above 0.5: {row.control_fraction_windows_detected:.1%}",
                        f"Miss/control duration difference: {row.duration_difference_seconds:+.2f} s",
                    ]
                ),
                va="top",
                fontsize=10,
                family="monospace",
            )
            fig.suptitle(f"Miss vs nearest-duration confident fake — pair {row.pair_id}")
            pdf.savefig(fig)
            plt.close(fig)


def write_overview(summaries: pd.DataFrame, path: Path) -> None:
    ## Wykresy porównują przeoczenia z dopasowanymi kontrolami w tej samej kolejności.
    ordered = summaries.sort_values("miss_baseline_p_fake").reset_index(drop=True)
    x = np.arange(len(ordered))
    features = [
        ("spectral_centroid_hz", "Spectral centroid (Hz)"),
        ("spectral_bandwidth_hz", "Spectral bandwidth (Hz)"),
        ("spectral_flatness", "Spectral flatness"),
        ("zero_crossing_rate", "Zero-crossing rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for axis, miss_col, control_col, title in (
        (
            axes[0, 0],
            "miss_baseline_p_fake",
            "control_baseline_p_fake",
            "Baseline crop score",
        ),
        (
            axes[0, 1],
            "miss_max_window_p_fake",
            "control_max_window_p_fake",
            "Maximum score over full recording",
        ),
        (
            axes[1, 0],
            "miss_fraction_windows_detected",
            "control_fraction_windows_detected",
            "Share of windows with P(fake) > 0.5",
        ),
    ):
        axis.plot(x, ordered[miss_col], "o-", color="#b42318", label="Miss")
        axis.plot(x, ordered[control_col], "o-", color="#175cd3", label="Matched control")
        axis.set(title=title, xlabel="Misses ordered by baseline P(fake)", ylabel="Value")
        axis.set_ylim(-0.03, 1.03)
        axis.legend()

    axis = axes[1, 1]
    for feature, label in features:
        delta = ordered[f"miss_{feature}"] - ordered[f"control_{feature}"]
        scale = max(float(ordered[f"control_{feature}"].median()), 1e-12)
        axis.plot(x, delta / scale, marker=".", linewidth=0.8, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title="Paired feature differences (miss − control, divided by control median)",
        xlabel="Misses ordered by baseline P(fake)",
        ylabel="Relative difference",
    )
    axis.legend(fontsize=7)
    max_duration_gap = summaries["duration_difference_seconds"].abs().max()
    #fig.suptitle(
    #    "SONICS baseline misses vs nearest-duration confident-fake controls\n"
    #    "Controls match generator and filename suffix; P(fake) ≥ 0.99; "
    #    f"largest duration gap = {max_duration_gap:.1f} s"
    #)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_short_udio_cohort(detector, samples: pd.DataFrame, windows: pd.DataFrame, canonical_audio: dict[str, torch.Tensor], path: Path) -> None:
    ## Osobny widok pokazuje krótką grupę Udio, dla której brak kontroli o tej samej długości.
    ordered = samples.sort_values("baseline_p_fake")
    mel_by_file = {}
    for row in ordered.itertuples(index=False):
        mel_by_file[row.file] = (detector.extract_mel_features(canonical_audio[row.file].unsqueeze(0))[0].detach().cpu().numpy())
    all_mels = np.concatenate([mel.ravel() for mel in mel_by_file.values()])
    lower, upper = np.quantile(all_mels, [0.02, 0.98])

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.8, 1, 1))
    timeline = fig.add_subplot(grid[0, :])
    for row in ordered.itertuples(index=False):
        clip_windows = windows.loc[windows["file"] == row.file].sort_values("start_seconds")
        color = "#b42318" if row.baseline_p_fake <= 0.5 else "#175cd3"
        timeline.plot(
            clip_windows["start_seconds"],
            clip_windows["p_fake"],
            marker="o",
            markersize=3,
            linewidth=1,
            color=color,
            label=f"{row.file} (baseline {row.baseline_p_fake:.3f})",
        )

    timeline.axhline(0.5, color="black", linestyle="--", linewidth=1)
    crop_start = float(ordered["baseline_crop_start_seconds"].median())
    timeline.axvline(crop_start, color="#667085", linestyle=":", linewidth=1)
    timeline.set(
        title="All 32–34 second Udio clips: P(fake) over overlapping windows",
        xlabel="Window start (seconds); dotted line = baseline crop start",
        ylabel="P(fake)",
        xlim=(0, None),
        ylim=(-0.03, 1.03),
    )
    timeline.legend(fontsize=7, ncol=2, loc="best")

    for index, row in enumerate(ordered.itertuples(index=False)):
        axis = fig.add_subplot(grid[1 + index // 3, index % 3])
        mel = mel_by_file[row.file]
        axis.imshow(
            mel,
            origin="lower",
            aspect="auto",
            extent=(0, TARGET_SECONDS, 0, mel.shape[0]),
            cmap="coolwarm",
            vmin=lower,
            vmax=upper,
        )
        axis.set(
            title=f"{row.file}\nbaseline P(fake)={row.baseline_p_fake:.3f}",
            xlabel="Seconds within baseline crop",
            ylabel="SONICS mel bin",
        )

    fig.suptitle("Short Udio subgroup: five baseline misses and one baseline detection")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ## Walidacja wejść i nazw plików chroni przed analizą niekompletnego lub innego zestawu danych.
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(BASELINE)
    parsed = results["file"].str.extract(r"fake_(\d+)_(suno|udio)_(\d+)\.mp3")
    if parsed.isna().any().any():
        raise ValueError("Unexpected filename format in baseline.csv")
    results["generator"] = parsed[1]
    results["variant"] = parsed[2].astype(int)
    results["miss"] = results["p_fake"] <= 0.5
    if not results["miss"].any():
        raise RuntimeError("No baseline false negatives found")

    ## Wczytaj długości raz, a następnie dobierz do każdego przeoczenia niepowtarzalną kontrolę.
    duration_cache: dict[str, float] = {}
    files_for_duration = set(results["file"])
    for file_name in files_for_duration:
        duration_seconds(file_name, duration_cache)
    results["duration_seconds"] = results["file"].map(duration_cache)

    controls = select_controls(results, duration_cache)
    pairs = []
    for pair_id, miss in enumerate(results.loc[results["miss"]].sort_values("p_fake").itertuples(index=False), 1):
        control_file = controls[miss.file]
        control = results.loc[results["file"] == control_file].iloc[0]
        pairs.append(
            {
                "pair_id": pair_id,
                "generator": miss.generator,
                "variant": miss.variant,
                "miss_file": miss.file,
                "miss_baseline_p_fake": float(miss.p_fake),
                "miss_duration_seconds": float(miss.duration_seconds),
                "control_file": control_file,
                "control_baseline_p_fake": float(control["p_fake"]),
                "control_duration_seconds": float(control["duration_seconds"]),
                "duration_difference_seconds": float(
                    control["duration_seconds"] - miss.duration_seconds
                ),
            }
        )

    ## Replay wycinka baseline stanowi warunek wiarygodności późniejszej analizy okien.
    detector = load_detector("sonics")
    detector.eval()
    if detector.meta.sample_rate != 16000 or detector.meta.max_len_samples != 80000:
        raise RuntimeError(f"Unexpected SONICS audio contract: {detector.meta}")

    all_windows: list[dict] = []
    canonical_audio: dict[str, torch.Tensor] = {}
    replay_errors: dict[str, float] = {}
    sample_summaries = []
    ## Analizuj zarówno przeoczenie, jak i jego kontrolę tym samym okienkowaniem i frontendem.
    for pair in pairs:
        for file_name, cohort, baseline_score, prefix in (
            (
                pair["miss_file"],
                "miss",
                pair["miss_baseline_p_fake"],
                "miss",
            ),
            (
                pair["control_file"],
                "control",
                pair["control_baseline_p_fake"],
                "control",
            ),
        ):
            window_records, canonical, window_count, replay_error = score_windows(detector, file_name, cohort, baseline_score)
            all_windows.extend(window_records)
            canonical_audio[file_name] = canonical
            replay_errors[file_name] = replay_error
            stats = crop_features(detector, canonical)
            p_values = np.asarray([record["p_fake"] for record in window_records])
            max_index = int(np.argmax(p_values))
            max_record = window_records[max_index]
            canonical_record = next(record for record in window_records if record["is_baseline_crop"])
            sample_summaries.append(
                {
                    "file": file_name,
                    "cohort": cohort,
                    "baseline_p_fake": baseline_score,
                    "replayed_canonical_p_fake": canonical_record["p_fake"],
                    "canonical_replay_abs_error": replay_error,
                    "duration_seconds": pair[f"{prefix}_duration_seconds"],
                    "baseline_crop_start_seconds": canonical_record["start_seconds"],
                    "window_count": window_count,
                    "max_window_p_fake": float(p_values[max_index]),
                    "max_window_start_seconds": max_record["start_seconds"],
                    "fraction_windows_detected": float(np.mean(p_values > 0.5)),
                    "fraction_windows_high_confidence_fake": float(
                        np.mean(p_values >= 0.9)
                    ),
                    **stats,
                }
            )

    ## Dodatkowo uwzględnij całą grupę krótkich nagrań Udio, nie tylko wykryte przeoczenia.
    short_udio = results.loc[(results["generator"] == "udio") & results["duration_seconds"].between(32, 34)]
    processed_files = {sample["file"] for sample in sample_summaries}
    for sample in short_udio.itertuples(index=False):
        if sample.file in processed_files:
            continue
        window_records, canonical, window_count, replay_error = score_windows(detector, sample.file, "short_detected", float(sample.p_fake))
        all_windows.extend(window_records)
        canonical_audio[sample.file] = canonical
        replay_errors[sample.file] = replay_error
        stats = crop_features(detector, canonical)
        p_values = np.asarray([record["p_fake"] for record in window_records])
        max_index = int(np.argmax(p_values))
        max_record = window_records[max_index]
        canonical_record = next(record for record in window_records if record["is_baseline_crop"])
        sample_summaries.append(
            {
                "file": sample.file,
                "cohort": "short_detected",
                "baseline_p_fake": float(sample.p_fake),
                "replayed_canonical_p_fake": canonical_record["p_fake"],
                "canonical_replay_abs_error": replay_error,
                "duration_seconds": float(sample.duration_seconds),
                "baseline_crop_start_seconds": canonical_record["start_seconds"],
                "window_count": window_count,
                "max_window_p_fake": float(p_values[max_index]),
                "max_window_start_seconds": max_record["start_seconds"],
                "fraction_windows_detected": float(np.mean(p_values > 0.5)),
                "fraction_windows_high_confidence_fake": float(
                    np.mean(p_values >= 0.9)
                ),
                **stats,
            }
        )

    ## Nie interpretuj wyników, jeśli lokalne odtworzenie baseline różni się od zapisanych predykcji.
    samples = pd.DataFrame(sample_summaries)
    windows = pd.DataFrame(all_windows)
    replay_max = max(replay_errors.values())
    if replay_max > 1e-4:
        raise RuntimeError(f"The exact baseline crops did not replay within tolerance: {replay_max:.8g}")

    ## Spłaszcz statystyki obu elementów każdej pary do jednego wiersza raportu.
    summary_rows = []
    samples_by_file = samples.set_index("file")
    for pair in pairs:
        miss = samples_by_file.loc[pair["miss_file"]]
        control = samples_by_file.loc[pair["control_file"]]
        row = dict(pair)
        for cohort, sample in (("miss", miss), ("control", control)):
            for name, value in sample.items():
                if name in {"file", "cohort"}:
                    continue
                row[f"{cohort}_{name}"] = value
        summary_rows.append(row)
    summaries = pd.DataFrame(summary_rows)

    ## Zapisuj dane surowe i zestawienia oddzielnie, by umożliwić ponowną analizę bez inferencji.
    summaries.to_csv(OUTPUT / "paired_summary.csv", index=False)
    windows.to_csv(OUTPUT / "window_scores.csv", index=False)
    short_samples = samples.loc[samples["file"].isin(short_udio["file"])].sort_values("baseline_p_fake")
    short_windows = windows.loc[windows["file"].isin(short_udio["file"])]
    short_samples.to_csv(OUTPUT / "short_udio_samples.csv", index=False)
    short_windows.to_csv(OUTPUT / "short_udio_window_scores.csv", index=False)

    write_overview(summaries, OUTPUT / "overview.png")
    write_short_udio_cohort(
        detector,
        short_samples,
        short_windows,
        canonical_audio,
        OUTPUT / "short_udio_cohort.png",
    )
    write_pair_pdf(detector, summaries, windows, canonical_audio, OUTPUT / "pair_spectrograms.pdf")

    ## Notatki definiują ograniczenia eksperymentu oraz sposób interpretacji nakładających się okien.
    miss_samples = samples.loc[samples["cohort"] == "miss"]
    control_samples = samples.loc[samples["cohort"] == "control"]
    duration_gaps = summaries["duration_difference_seconds"].abs()
    any_miss_rescued = int((miss_samples["max_window_p_fake"] > 0.5).sum())
    notes = [
        f"Baseline rows: {len(results)}",
        f"Baseline misses (P(fake) <= 0.5): {int(results['miss'].sum())}",
        f"Nearest-duration controls: {len(controls)} (same generator and filename suffix, baseline P(fake) >= {CONFIDENT_FAKE_THRESHOLD})",
        f"Median absolute duration difference within pairs: {summaries['duration_difference_seconds'].abs().median():.2f} s",
        f"Pairs within 1 s / 10 s of the miss duration: {int((duration_gaps <= 1).sum())} / {int((duration_gaps <= 10).sum())} of {len(summaries)}",
        f"All canonical crops replayed; maximum absolute score difference from baseline: {replay_max:.8g}",
        f"Misses with at least one 2.5-second-hop window above 0.5: {any_miss_rescued}/{len(miss_samples)}",
        f"Misses with maximum window P(fake) >= 0.9: {int((miss_samples['max_window_p_fake'] >= 0.9).sum())}/{len(miss_samples)}",
        f"Matched controls with maximum window P(fake) >= 0.9: {int((control_samples['max_window_p_fake'] >= 0.9).sum())}/{len(control_samples)}",
        f"Median full-recording maximum P(fake), misses: {miss_samples['max_window_p_fake'].median():.4f}",
        f"Median full-recording maximum P(fake), controls: {control_samples['max_window_p_fake'].median():.4f}",
        f"32–34 s Udio clips: {len(short_samples)}; baseline misses: {int((short_samples['baseline_p_fake'] <= 0.5).sum())}; baseline detections: {int((short_samples['baseline_p_fake'] > 0.5).sum())}",
        f"32–34 s Udio baseline misses with no window above 0.5: {int(((short_samples['baseline_p_fake'] <= 0.5) & (short_samples['max_window_p_fake'] <= 0.5)).sum())}",
        f"32–34 s Udio clips with baseline P(fake) >= 0.99: {int(((short_udio['p_fake'] >= CONFIDENT_FAKE_THRESHOLD)).sum())}",
        f"Maximum duration gap among high-confidence matched controls: {summaries['duration_difference_seconds'].abs().max():.2f} s",
    ]
    (OUTPUT / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
