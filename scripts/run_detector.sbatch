#!/usr/bin/env bash
## Submitted by scripts/run-slurm.sh. Set CPU/RAM/time and add the GPU partition
## and GRES directives required by your grant before submitting.
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --account=plgregevgauss-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SCRATCH_DIR="${SCRATCH:-/net/tscratch/people/${USER:?USER is not set}}"
module load Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH_DIR/conda-envs/aa-aigmd-py312"

exec bash ./scripts/run.sh
