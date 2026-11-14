#!/usr/bin/env bash
## Run from the project root: bash scripts/create-env.sh
## Creates the Python environment and caches on $SCRATCH; install CUDA PyTorch separately.
set -Eeuo pipefail

CONDA_MODULE="Miniconda3/25.7.0-2"
ENV_NAME="aa-aigmd-py312"

if ! type module >/dev/null 2>&1; then
    echo "Run this on Athena's login node." >&2
    exit 1
fi
if [[ -z "${SCRATCH:-}" || ! -d "$SCRATCH" || ! -w "$SCRATCH" ]]; then
    echo "SCRATCH must be set to a writable directory on Athena." >&2
    exit 1
fi

module load "$CONDA_MODULE"
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found after loading $CONDA_MODULE." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_PREFIX="$SCRATCH/conda-envs/$ENV_NAME"
export CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs"
export PIP_CACHE_DIR="$SCRATCH/pip-cache"
export HF_HOME="$SCRATCH/hf-cache"
mkdir -p "$SCRATCH/conda-envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

if [[ -e "$ENV_PREFIX" ]]; then
    if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
        echo "Environment path exists but has no Python executable: $ENV_PREFIX" >&2
        echo "Inspect it before removing or reusing it." >&2
        exit 1
    fi
    echo "Using existing environment: $ENV_PREFIX"
else
    conda create --yes --prefix "$ENV_PREFIX" python=3.12 pip
fi

ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
{
    printf 'export CONDA_PKGS_DIRS=%q\n' "$CONDA_PKGS_DIRS"
    printf 'export PIP_CACHE_DIR=%q\n' "$PIP_CACHE_DIR"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
} > "$ACTIVATE_DIR/aa-aigmd-scratch-caches.sh"

conda activate "$ENV_PREFIX"
python -c 'import sys; print(sys.version); sys.exit(sys.version_info[:2] != (3, 12))'

printf '\nEnvironment ready: %s\n' "$ENV_PREFIX"
