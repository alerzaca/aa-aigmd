#!/usr/bin/env bash
# Run one detector job; scripts/run-slurm.sh submits both alpha models.
# Data: data/fake/sonics-part1 and data/fake/sonics-part2.
set -Eeuo pipefail

SCRATCH_DIR="${SCRATCH:-/net/tscratch/people/${USER:?USER is not set}}"
PROJECT_DIR="$SCRATCH_DIR/aa-aigmd"
SCRIPT_DIR="$PROJECT_DIR/scripts"
cd "$PROJECT_DIR"

DETECTOR="${DETECTOR:-sonics-alpha-5s}"
RUN_NAME="${RUN_NAME:-sonics_parts}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
PYTHON="${PYTHON:-python3}"
RESULTS_DIR="$PROJECT_DIR/results/$DETECTOR/$RUN_NAME"

for part in sonics-part1 sonics-part2; do
    part_dir="$DATA_DIR/fake/$part"
    [[ -d "$part_dir" ]] || {
        echo "Missing dataset directory: $DATA_DIR/fake/$part" >&2
        exit 1
    }
    [[ -n "$(find "$part_dir" -type f \( -iname '*.wav' -o -iname '*.mp3' \) -print -quit)" ]] || {
        echo "No WAV/MP3 files in: $part_dir" >&2
        exit 1
    }
done

SAMPLE="$(find "$DATA_DIR/fake" -type f \( -iname '*.wav' -o -iname '*.mp3' \) -print -quit)"
[[ ! -e "$RESULTS_DIR" ]] || { echo "Results already exist: $RESULTS_DIR" >&2; exit 1; }

run() { "$PYTHON" -m src.main "$@"; }
common=(--detector "$DETECTOR" --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR")

run check-env
mkdir -p "$RESULTS_DIR"
run infer --detector "$DETECTOR" --sample-file "$SAMPLE"
run check-gradient --detector "$DETECTOR" --sample-file "$SAMPLE"

for representation in waveform mel; do
    run baseline --representation "$representation" "${common[@]}"
done

attacks=(
    'waveform fgsm --epsilon 0.001 0.002 0.005 0.01'
    'waveform pgd --epsilon 0.001 0.002 0.005 0.01 --alpha 0.001 --steps 20'
    'waveform cw --c 1 --steps 100'
    'waveform noise_baseline --snr-db 45 33 28 24'
    'mel fgsm --epsilon 0.02 0.05 0.1'
    'mel pgd --epsilon 0.02 0.05 0.1 --alpha 0.01 --steps 20'
    'mel cw --c 1 --steps 25'
)

for attack in "${attacks[@]}"; do
    read -r -a args <<< "$attack"
    run attack \
        --representation "${args[0]}" \
        --attack "${args[1]}" \
        "${args[@]:2}" \
        "${common[@]}"
done

echo "Report: python analysis/report.py results/$DETECTOR/$RUN_NAME"
