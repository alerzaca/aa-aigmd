from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

## Porównanie dwóch uruchomień SONICS na wspólnym korpusie.
BASELINE_5S = ROOT / "results" / "sonics" / "sonics_baseline_2" / "waveform" / "baseline.csv"
BASELINE_120S = (
    ROOT
    / "results"
    / "sonics-gamma-120s"
    / "sonics_baseline_120s"
    / "waveform"
    / "baseline.csv"
)
DURATION_SUMMARY = (
    ROOT
    / "results"
    / "sonics"
    / "sonics_baseline_2"
    / "analysis"
    / "all_files"
    / "per_file_summary.csv"
)
OUTPUT = BASELINE_120S.parent.parent / "analysis"
THRESHOLD = 0.5


def read_baseline(path: Path, score_name: str) -> pd.DataFrame:
    ## Ujednolicenie nazw kolumn.
    frame = pd.read_csv(path)
    if "file" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "file"})
    if not {"file", "p_fake"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain a file column and p_fake")
    if frame["file"].duplicated().any():
        raise ValueError(f"{path} contains duplicate file rows")
    return frame[["file", "p_fake"]].rename(columns={"p_fake": score_name})


def main() -> None:
    ## Łączenie z walidacją.
    scores = read_baseline(BASELINE_5S, "p_fake_5s").merge(
        read_baseline(BASELINE_120S, "p_fake_120s"),
        on="file",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scores["_merge"].eq("both").all():
        raise ValueError(
            "The two baselines do not contain the same files: "
            f"{scores['_merge'].value_counts().to_dict()}"
        )
    scores = scores.drop(columns="_merge")
    
    ## Ta sama granica klasyfikacji do porównania zarówno zmiany wyniku, jak i decyzji.
    scores["detected_5s"] = scores["p_fake_5s"] > THRESHOLD
    scores["detected_120s"] = scores["p_fake_120s"] > THRESHOLD
    scores["score_change_120s_minus_5s"] = (scores["p_fake_120s"] - scores["p_fake_5s"])
    scores["decision_changed"] = scores["detected_5s"] != scores["detected_120s"]
    scores["source"] = scores["file"].str.extract(r"^fake_\d+_(.+)_\d+\.mp3$")

    durations = pd.read_csv(DURATION_SUMMARY)[["file", "duration_seconds"]]
    scores = scores.merge(durations, on="file", how="left", validate="one_to_one")

    ## Przedziały długości pokazują, czy różnice między kontekstami zależą od czasu nagrania.
    scores["duration_group"] = pd.cut(
        scores["duration_seconds"],
        bins=[0, 60, 120, float("inf")],
        labels=["<=60s", "60-120s", ">120s"],
        include_lowest=True,
    )

    ## Pliki per-próbka zachowują szczegóły potrzebne do dalszej kontroli przypadków.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.sort_values("file").to_csv(OUTPUT / "per_file_comparison.csv", index=False)
    scores.loc[~scores["detected_120s"]].sort_values("file").to_csv(OUTPUT / "gamma120_false_negatives.csv", index=False)
    scores.loc[scores["decision_changed"]].sort_values("file").to_csv(OUTPUT / "decision_changes.csv", index=False)

    ## Podsumowanie agreguje zmianę skuteczności ogółem oraz według czasu i źródła z nazwy pliku.
    lines = [
        f"Files compared: {len(scores)}",
        f"Threshold: P(fake) > {THRESHOLD}",
        "",
        "Overall fake-file detection:",
        f"  Gamma-5s: {scores['detected_5s'].sum()} detected, "
        f"{(~scores['detected_5s']).sum()} missed",
        f"  Gamma-120s: {scores['detected_120s'].sum()} detected, "
        f"{(~scores['detected_120s']).sum()} missed",
        f"  Decisions changed: {scores['decision_changed'].sum()}",
        f"  5s detected -> 120s missed: "
        f"{(scores['detected_5s'] & ~scores['detected_120s']).sum()}",
        f"  5s missed -> 120s detected: "
        f"{(~scores['detected_5s'] & scores['detected_120s']).sum()}",
        f"  Median P(fake), Gamma-5s: {scores['p_fake_5s'].median():.6f}",
        f"  Median P(fake), Gamma-120s: {scores['p_fake_120s'].median():.6f}",
        "",
        "Detection by recording duration:",
    ]
    for group, frame in scores.groupby("duration_group", observed=True):
        lines.append(
            f"  {group}: n={len(frame)}, "
            f"Gamma-5s={frame['detected_5s'].mean():.1%}, "
            f"Gamma-120s={frame['detected_120s'].mean():.1%}, "
            f"decisions changed={frame['decision_changed'].sum()}"
        )
    lines.extend(["", "Detection by filename source:",])
    for source, frame in scores.groupby("source", dropna=False):
        lines.append(
            f"  {source}: n={len(frame)}, "
            f"Gamma-5s={frame['detected_5s'].mean():.1%}, "
            f"Gamma-120s={frame['detected_120s'].mean():.1%}, "
            f"decisions changed={frame['decision_changed'].sum()}"
        )
    summary = "\n".join(lines) + "\n"
    (OUTPUT / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
