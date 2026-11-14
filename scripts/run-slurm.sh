#!/usr/bin/env bash
set -Eeuo pipefail

SCRATCH_DIR="${SCRATCH:-/net/tscratch/people/${USER:?USER is not set}}"
RUN_NAME="sonics_parts"

PROJECT_DIR="$SCRATCH_DIR/aa-aigmd"
DATA_DIR="$PROJECT_DIR/data"
SCRIPT_DIR="$PROJECT_DIR/scripts"

PYTHON="$SCRATCH_DIR/conda-envs/aa-aigmd-py312/bin/python"
DETECTORS=(sonics-alpha-5s sonics-alpha-120s)

while (($#)); do
    case "$1" in
        --detectors)
            shift
            DETECTORS=()
            while (($#)) && [[ "$1" != --* ]]; do
                DETECTORS+=("$1")
                shift
            done
            [[ ${#DETECTORS[@]} -gt 0 ]] || {
                echo "--detectors requires at least one detector" >&2
                exit 1
            }
            ;;
        --run-name)
            (($# >= 2)) || { echo "--run-name requires a value" >&2; exit 1; }
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

[[ "$RUN_NAME" =~ ^[[:alnum:]_.-]+$ && "$RUN_NAME" != "." && "$RUN_NAME" != ".." ]] || {
    echo "RUN_NAME may contain only letters, numbers, '.', '_' and '-'" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || {
    echo "Python environment not found: $PYTHON" >&2
    exit 1
}

for detector in "${DETECTORS[@]}"; do
    case "$detector" in
        sonics-alpha-5s|sonics-alpha-120s) ;;
        *) echo "Unsupported detector: $detector" >&2; exit 1 ;;
    esac
done

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

mkdir -p "$PROJECT_DIR/results"
for detector in "${DETECTORS[@]}"; do
    DETECTOR="$detector" RUN_NAME="$RUN_NAME" DATA_DIR="$DATA_DIR" PYTHON="$PYTHON" \
        sbatch --parsable \
            --chdir="$PROJECT_DIR" \
            --job-name="aa-aigmd-$detector" \
            --output="$PROJECT_DIR/results/slurm-%x-%j.out" \
            --error="$PROJECT_DIR/results/slurm-%x-%j.err" \
            --export=ALL \
            "$SCRIPT_DIR/run_detector.sbatch"
done
