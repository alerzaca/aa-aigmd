#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --account=plgregevgauss-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1

set -Eeuo pipefail

SCRATCH_DIR="${SCRATCH:-/net/tscratch/people/${USER:?USER is not set}}"
module load Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH_DIR/conda-envs/aa-aigmd-py312"

exec bash "$SCRATCH_DIR/aa-aigmd/scripts/run.sh"
