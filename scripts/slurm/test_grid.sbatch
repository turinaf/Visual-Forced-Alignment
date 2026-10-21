#!/bin/bash
#SBATCH --job-name=vfa-test
#SBATCH --partition=a3m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_grid_%j.out
#SBATCH --error=logs/test_grid_%j.err

# Submit with overrides, e.g.:
#   mkdir -p logs && sbatch scripts/slurm/test_grid.sbatch --export=SAVE_NAME=grid_v1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data"
GRID_ROOT="${GRID_ROOT:-$DATA_ROOT/GRID_processed}"
SAVE_NAME="${SAVE_NAME:-grid_v1}"
CHECKPOINT="${CHECKPOINT:-$DATA_ROOT/checkpoints/VFA/$SAVE_NAME/$SAVE_NAME.pkl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_ALIGN="${NUM_ALIGN:-200}"
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"

cd "$REPO_DIR"
mkdir -p logs

# Use the pyenv interpreter directly (no shims/init needed in a batch shell)
PYTHON="$PYENV_ROOT/versions/$PYENV_VERSION/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: python not found at $PYTHON" >&2
    echo "Set PYENV_VERSION (or PYENV_ROOT) to a version with torch installed." >&2
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found at $CHECKPOINT" >&2
    echo "Set CHECKPOINT to the trained .pkl file." >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo: $REPO_DIR  Root: $GRID_ROOT  Checkpoint: $CHECKPOINT"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

"$PYTHON" test_phoneme_split.py \
    --gpu 0 \
    --dataset GRID \
    --grid_root "$GRID_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --num_align "$NUM_ALIGN"

echo "Testing done."
