#!/bin/bash
#SBATCH --job-name=vfa-preproc
#SBATCH --partition=a3m              # GPU nodes: NVIDIA drivers provide libEGL.so.1, which mediapipe needs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_grid_%j.out
#SBATCH --error=logs/preprocess_grid_%j.err

# Submit with overrides, e.g.:
#   mkdir -p logs && sbatch scripts/slurm/preprocess_grid.sbatch
# (mkdir first so slurm can open the job log)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
RAW_DIR="${RAW_DIR:-/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/grid-data}"
OUT_DIR="${OUT_DIR:-/mnt/lustre/ifm-lustre-prod-us-west1-a/av-data/GRID_processed}"
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"   # pyenv version with mediapipe, opencv-python, tqdm

cd "$REPO_DIR"
mkdir -p logs

# mediapipe's native lib needs libEGL.so.1, absent on headless CPU nodes.
# If extracted user-space GL libs exist (see README), put them on the path.
if [ -d "${GL_LIBS_DIR:-$HOME/gl-libs/usr/lib/x86_64-linux-gnu}" ]; then
    export LD_LIBRARY_PATH="${GL_LIBS_DIR:-$HOME/gl-libs/usr/lib/x86_64-linux-gnu}:$LD_LIBRARY_PATH"
    echo "Using user-space GL libs from $LD_LIBRARY_PATH"
fi

# Use the pyenv interpreter directly (no shims/init needed in a batch shell)
PYTHON="$PYENV_ROOT/versions/$PYENV_VERSION/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: python not found at $PYTHON" >&2
    echo "Set PYENV_VERSION (or PYENV_ROOT) to a version with mediapipe installed." >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Repo: $REPO_DIR  Raw: $RAW_DIR  Out: $OUT_DIR"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

"$PYTHON" scripts/preprocess_grid.py \
    --raw_dir "$RAW_DIR" \
    --out_dir "$OUT_DIR"

echo "Preprocessing done."
