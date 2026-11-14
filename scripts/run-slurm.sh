#!/usr/bin/env bash
# From the project root, run: bash scripts/run-slurm.sh
# Edit RUN_NAME and DATA_DIR before each run; both SONICS parts are included.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SCRATCH_DIR="${SCRATCH:-/net/tscratch/people/${USER:?USER is not set}}"

RUN_NAME="sonics_parts"
DATA_DIR="$PROJECT_DIR/data"
PYTHON="$SCRATCH_DIR/conda-envs/aa-aigmd-py312/bin/python"
DETECTORS=(sonics-alpha-5s sonics-alpha-120s)

[[ -x "$PYTHON" ]] || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

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

for detector in "${DETECTORS[@]}"; do
    results_dir="$PROJECT_DIR/results/$detector/$RUN_NAME"
    [[ ! -e "$results_dir" ]] || {
        echo "Results already exist; change RUN_NAME: $results_dir" >&2
        exit 1
    }
done

mkdir -p "$PROJECT_DIR/results"
for detector in "${DETECTORS[@]}"; do
    DETECTOR="$detector" RUN_NAME="$RUN_NAME" DATA_DIR="$DATA_DIR" PYTHON="$PYTHON" \
        sbatch --parsable \
            --chdir="$PROJECT_DIR" \
            --job-name="aa-aigmd-$detector" \
            --output="$PROJECT_DIR/results/slurm-%x-%j.out" \
            --export=ALL \
            "$SCRIPT_DIR/run_detector.sbatch"
done
