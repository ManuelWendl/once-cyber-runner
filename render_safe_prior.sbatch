#!/bin/bash
#
#SBATCH --job-name=render_safe_prior
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/render_safe_prior_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
CHECKPOINT=${CHECKPOINT:-$REPO_DIR/.vendor/cyberrunner_ppo/checkpoints/final.pkl}
OUTDIR=${OUTDIR:-$REPO_DIR/render_out}
EPISODES=${EPISODES:-10}

mkdir -p "$REPO_DIR/logs" "$OUTDIR"

nvidia-smi || true

. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export MUJOCO_GL=egl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_ppo

cd "$REPO_DIR"

python render_safe_prior.py \
    --checkpoint "$CHECKPOINT" \
    --mode video \
    --outdir "$OUTDIR" \
    --episodes "$EPISODES"
