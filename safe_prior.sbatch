#!/bin/bash
#
# Safe-prior PPO training on the vendored Brax env.
# Reward = exp(-‖ball - frozen_target‖) to the closest backward strict-corner.
# Config: .vendor/cyberrunner_ppo/config.yaml (env.safe_prior: true).
#
#SBATCH --job-name=safe_prior
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/safe_prior_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
VENDOR_DIR=$REPO_DIR/.vendor/cyberrunner_ppo

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
export WANDB_PROJECT=${WANDB_PROJECT:-cyberrunner_safe_prior}

cd "$VENDOR_DIR"

python train.py --wandb
