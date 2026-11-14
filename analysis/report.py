## Post-processing of one run: tables + figures from raw results
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram

REPS = ["waveform", "mel"]
KEYS = ["representation", "attack", "budget"]
ATTACKS = ["fgsm", "pgd", "cw"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def _with_groups(frame: pd.DataFrame) -> pd.DataFrame:
    ## Pierwszy segment ścieżki określa klasę, a dwa pierwsze — grupę źródłową próbki.
    parts = frame["file"].str.split("/")
    frame["label"], frame["group"] = parts.str[0], parts.str[0] + "/" + parts.str[1]
    return frame


def _load(run: Path):
    ## Wczytaj baseline i wyniki ataków, dołączając do każdego ataku predykcję próbki przed modyfikacją.
    baselines, attacks = {}, []
    for rep in REPS:
        path = run / rep / "baseline.csv"
        if not path.exists():
            continue
        baseline = _with_groups(pd.read_csv(path))
        baselines[rep] = baseline
        before = baseline.rename(columns={"p_fake": "p_fake_before", "feature_p_fake": "feature_p_fake_before"})

        for metrics in sorted((run / rep).glob("*/metrics.csv")):
            attack = metrics.parent.name
            config = json.loads((metrics.parent / "config.json").read_text())

            ## Walidacja many-to-one wymaga dokładnie jednego wyniku baseline dla każdego pliku ataku.
            frame = pd.read_csv(metrics).merge(before.drop(columns=["label", "group"]), on="file", how="left", validate="many_to_one")
            if frame["p_fake_before"].isna().any():
                raise ValueError(f"Missing baseline predictions for {metrics}")
            frame.insert(0, "representation", rep)
            frame.insert(1, "attack", attack)
            frame["budget_name"] = config["budget_name"]
            attacks.append(frame)

    if not attacks:
        raise FileNotFoundError(f"No <representation>/<attack>/metrics.csv found under {run}")
    data = _with_groups(pd.concat(attacks, ignore_index=True))

    ## Sukces oznacza obejście tylko wtedy, gdy próbka była wykryta przed atakiem.
    data["detected"] = data["p_fake_before"] > 0.5
    data["success"] = data["detected"] & (data["p_fake_after"] < 0.5)
    if "feature_p_fake_after" in data:
        data["feature_success"] = ((data["feature_p_fake_before"] > 0.5) & (data["feature_p_fake_after"] < 0.5)).astype(float).where(data["feature_p_fake_after"].notna())
    else:
        data["feature_success"] = np.nan
    return baselines, data


def _baseline_table(baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ## Zestawienie jakości predykcji bazowych osobno dla reprezentacji i źródła próbek.
    rows = []
    for rep, baseline in baselines.items():
        for group, d in baseline.groupby("group"):
            correct = (d["p_fake"] > 0.5) == (d["label"] == "fake")
            row = {
                "representation": rep,
                "source_group": group,
                "actual_class": d["label"].iloc[0],
                "sample_count": len(d),
                "correct_predictions": int(correct.sum()),
                "accuracy": correct.mean(),
            }
            if "recon_snr_db" in d:
                row["mean_reconstruction_snr_db"] = d["recon_snr_db"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def _attack_table(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    data = data.copy()
    ## Liczniki pokazują zarówno liczbę atakowanych plików, jak i ile z nich detektor nadal wykrywa.
    data["still_detected_after"] = data["p_fake_after"] > 0.5
    data["detected_before"] = data["p_fake_before"] > 0.5
    table = data.groupby(keys).agg(
        attacked_sample_count=("file", "size"),
        fake_detected_after_count=("still_detected_after", "sum"),
        detected_as_fake_before_count=("detected_before", "sum"),
        successful_evasion_count=("success", "sum"),
        mean_fake_probability_after=("p_fake_after", "mean"),
        mean_snr_db=("snr_db", "mean"),
        mean_stoi=("stoi", "mean"),
    )
    if "feature_success" in data:
        data["feature_detected_before"] = data["feature_p_fake_before"] > 0.5
        table = table.join(data.groupby(keys).agg(
            feature_detected_before_count=("feature_detected_before", "sum"),
            feature_successful_evasion_count=("feature_success", "sum"),
        ))

    ## Mianownikiem skuteczności są tylko próbki wykryte w baseline; brak takich próbek daje NaN.
    table["attack_success_rate"] = (
        table["successful_evasion_count"]
        / table["detected_as_fake_before_count"].replace(0, np.nan)
    )
    if "feature_successful_evasion_count" in table:
        table["feature_attack_success_rate"] = (
            table["feature_successful_evasion_count"]
            / table["feature_detected_before_count"].replace(0, np.nan)
        )
    return table.reset_index()


def _attack_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = KEYS + ["budget_name"]
    ## Oprócz wyniku łącznego tworzony jest wynik dla każdej grupy źródłowej.
    overall = _attack_table(data, keys)
    overall["source_group"] = "ALL_FAKE"
    by_group = _attack_table(data, keys + ["group"]).rename(columns={"group": "source_group"})
    return pd.concat([overall, by_group], ignore_index=True).rename(
        columns={"budget_name": "budget_type", "budget": "budget_value"}
    ).sort_values(
        ["representation", "attack", "budget_value", "source_group"],
        ignore_index=True,
    )


def _confusion_matrix(labels: pd.Series, predictions: pd.Series) -> np.ndarray:
    ## Wiersze macierzy odpowiadają klasie rzeczywistej, kolumny predykcji (real=0, fake=1).
    matrix = np.zeros((2, 2), dtype=int)
    for label, prediction in zip(labels, predictions):
        matrix[int(label == "fake"), int(prediction > 0.5)] += 1
    return matrix


def _confusion_figures(baselines: dict[str, pd.DataFrame], data: pd.DataFrame, out: Path) -> None:
    ## Dla realnych plików zachowujemy predykcje baseline, bo ataki są uruchamiane na fake.
    for rep, baseline in baselines.items():
        baseline = baseline.set_index("file")
        panels = [("Baseline", baseline["p_fake"])]
        attacks = data[
            (data["representation"] == rep) & data["attack"].isin(ATTACKS)
        ]

        for attack in ATTACKS:
            for budget in sorted(attacks.loc[attacks["attack"] == attack, "budget"].unique()):
                frame = attacks[(attacks["attack"] == attack) & (attacks["budget"] == budget)]
                predictions = baseline["p_fake"].copy()
                ## Ataki modyfikują tylko fake; predykcje utworów real pozostają jak w baseline.
                fake_files = baseline.index[baseline["label"] == "fake"]
                updates = frame.set_index("file")["p_fake_after"].reindex(fake_files)
                predictions.update(updates)
                budget_name = frame["budget_name"].iloc[0]
                panels.append((f"{attack.upper()} {budget_name}={budget:g}", predictions))

        matrices = [_confusion_matrix(baseline["label"], pred) for _, pred in panels]
        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.7 * rows), squeeze=False)
        vmax = max(1, max(matrix.max() for matrix in matrices))

        for ax, (title, _), matrix in zip(axes.flat, panels, matrices):
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            ax.set(title=title, xticks=[0, 1], yticks=[0, 1])
            ax.set_xticklabels(["Real", "Fake"])
            ax.set_yticklabels(["Real", "Fake"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (row, column), value in np.ndenumerate(matrix):
                ax.text(column, row, str(value), ha="center", va="center",
                        color="white" if value > vmax / 2 else "black")

        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(rep)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / f"confusion_matrix_{rep}.png", bbox_inches="tight")
        plt.close(fig)


def _spectrogram_figure(run: Path, data: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    ## Ten sam utwór pokazany dla waveform i mel; dla każdego wybierz jeden budżet ataku.
    attack = next((name for name in ATTACKS if name in set(data["attack"])), None)
    if attack is None:
        return
    reps = [r for r in REPS if ((summary["representation"] == r) & (summary["attack"] == attack)).any()]
    if not reps:
        return

    overall = summary[summary["source_group"] == "ALL_FAKE"]
    choices = {}
    for rep in reps:
        rows = overall[(overall["representation"] == rep) & (overall["attack"] == attack)].sort_values("budget_value")
        successful = rows[rows["attack_success_rate"] >= 0.5]

        ## Wybierz najmniejszy budżet osiągający co najmniej 50% skuteczności; gdy nie ma takiego budżetu, pokaż największy dostępny.
        budget = (successful if not successful.empty else rows).iloc[
            0 if not successful.empty else -1
        ]["budget_value"]
        choices[rep] = data[
            (data["representation"] == rep)
            & (data["attack"] == attack)
            & (data["budget"] == budget)
        ]

    ## Wybierz wspólną próbkę udanego ataku; kolejne zbiory są awaryjnymi kryteriami wyboru.
    candidate_sets = [
        [set(frame.loc[frame["success"], "file"]) for frame in choices.values()],
        [set(frame.loc[frame["detected"], "file"]) for frame in choices.values()],
        [set(frame["file"]) for frame in choices.values()],
    ]
    common_files = next((set.intersection(*sets) for sets in candidate_sets if sets and set.intersection(*sets)),set(),)
    if not common_files:
        return
    sample = sorted(common_files)[0]

    fig, axes = plt.subplots(len(reps), 3, figsize=(11, 3.2 * len(reps)), sharex=True, sharey=True, squeeze=False)
    for row, rep in zip(axes, reps):
        r = choices[rep].set_index("file").loc[sample]
        adv, sr = sf.read(str(run / r["audio"]))
        clean = sf.read(str(run / rep / "clean" / Path(sample).with_suffix(".wav")))[0]
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if adv.ndim > 1:
            adv = adv.mean(axis=1)

        ## Ograniczenie do wspólnego środkowego fragmentu ułatwia porównanie czystego i zmodyfikowanego audio.
        length = min(len(clean), len(adv), 5 * sr)
        start = max(0, (min(len(clean), len(adv)) - length) // 2)
        signals = [x[start : start + length] for x in (clean, adv, adv - clean)]
        specs = [
            10 * np.log10(spectrogram(x, sr, nperseg=min(512, len(x)))[2] + 1e-12)
            for x in signals
        ]
        top = specs[0].max()
        for ax, spec, title in zip(row, specs, ["Clean", "Attacked", "Difference"]):
            image = ax.imshow(
                spec - top, origin="lower", aspect="auto", vmin=-90, vmax=0,
                extent=(0, length / sr, 0, sr / 2000),
            )
            ax.set_title(title)
            ax.grid(False)
        row[0].set_ylabel(f"{rep}\nFrequency [kHz]")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"Sample: {textwrap.fill(sample, width=100)}  |  {attack.upper()}", fontsize=9)
    fig.subplots_adjust(top=0.88, right=0.9, hspace=0.35, wspace=0.12)
    colorbar_axis = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(image, cax=colorbar_axis, label="dB relative to clean peak")
    fig.savefig(out / "spectrograms.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="results/<detector>/<run>")
    run = parser.parse_args().run
    out = run / "report"
    out.mkdir(exist_ok=True)
    
    ## Kolejne tabele i wykresy korzystają z jednego wspólnego zestawu wczytanych danych.
    baselines, data = _load(run)
    baseline_table = _baseline_table(baselines)
    baseline_table.to_csv(out / "baseline_summary.csv", index=False)
    summary = _attack_summary(data)
    summary.to_csv(out / "attack_summary.csv", index=False)
    _confusion_figures(baselines, data, out)
    _spectrogram_figure(run, data, summary, out)
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
