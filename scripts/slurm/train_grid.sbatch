#!/bin/bash
#SBATCH --job-name=vfa-train
#SBATCH --partition=a3m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_grid_%j.out
#SBATCH --error=logs/train_grid_%j.err

# Submit with overrides, e.g.:
#   mkdir -p logs && sbatch scripts/slurm/train_grid.sbatch --export=SAVE_NAME=grid_v1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data"
GRID_ROOT="${GRID_ROOT:-$DATA_ROOT/GRID_processed}"   # output of the preprocessing job
SAVE_NAME="${SAVE_NAME:-grid_v1}"
DATASET="${DATASET:-GRID}"                            # GRID or LRS2
BATCH_SIZE="${BATCH_SIZE:-64}"                        # sized for one H100
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"              # pyenv version with torch, opencv, einops, editdistance

cd "$REPO_DIR"
mkdir -p logs

# Use the pyenv interpreter directly (no shims/init needed in a batch shell)
PYTHON="$PYENV_ROOT/versions/$PYENV_VERSION/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: python not found at $PYTHON" >&2
    echo "Set PYENV_VERSION (or PYENV_ROOT) to a version with torch installed." >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo: $REPO_DIR  Dataset: $DATASET  Root: $GRID_ROOT  Save: $SAVE_NAME"
echo "Checkpoints: ${VFA_CHECKPOINT_ROOT:-<default from data/consts.py>}"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

"$PYTHON" train_phoneme_split.py \
    --gpu 0 \
    --dataset "$DATASET" \
    --grid_root "$GRID_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --epoch_num 60 \
    --save_epochs 3 \
    --save_name "$SAVE_NAME"

echo "Training done."
