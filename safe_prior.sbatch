#!/bin/bash
#
# Safe-prior PPO training on the vendored Brax env.
# Strategy is selectable via STRATEGY env var:
#   sbatch safe_prior.sbatch                                  # exp_d (default)
#   STRATEGY=exp_d_sigma SIGMA=0.02 sbatch safe_prior.sbatch  # steeper basin
#   STRATEGY=survival sbatch safe_prior.sbatch                # +1 per alive step
# Wandb project is tagged with the strategy so curves stay separable.
#
#SBATCH --job-name=safe_prior
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/safe_prior_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
VENDOR_DIR=$REPO_DIR/.vendor/cyberrunner_ppo

STRATEGY=${STRATEGY:-exp_d}
SIGMA=${SIGMA:-0.02}

mkdir -p "$REPO_DIR/logs"

nvidia-smi

. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_ppo

export WANDB_ENTITY=${WANDB_ENTITY:-filippostaffoni-eth-z-rich}
export WANDB_PROJECT=${WANDB_PROJECT:-cyberrunner_safe_prior_${STRATEGY}}

cd "$VENDOR_DIR"

python train.py --wandb --prior-strategy "$STRATEGY" --prior-sigma "$SIGMA"
