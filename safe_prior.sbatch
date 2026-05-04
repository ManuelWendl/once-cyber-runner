#!/bin/bash
#
# Safe-prior PPO training on the vendored Brax env.
# Strategy is selectable via STRATEGY env var:
#   sbatch safe_prior.sbatch                                  # exp_d (default)
#   STRATEGY=exp_d_sigma SIGMA=0.02 sbatch safe_prior.sbatch  # steeper basin
#   STRATEGY=survival sbatch safe_prior.sbatch                # +1 per alive step
# Wandb project is tagged with the strategy so curves stay separable.
#
# Optional difficulty knobs (env vars):
#   INIT_BALL_SPEED   max spawn ball speed (m/s). Empty → use config default.
#   INIT_TILT_FRAC    fraction of joint range used for spawn tilt.
#   TILT_BUMPS        "1" to enable random mid-episode tilt bumps.
#   TILT_BUMP_PROB    per-step bump probability.
#   TILT_BUMP_MAG     bump magnitude (fraction of half-joint-range).
#   RUN_NAME          custom subdir under checkpoints/ for this run.
#   RESUME            absolute path to a prior best.pkl/final.pkl to warm-start
#                     from (curriculum fine-tuning). Optimizer state restarts.
#   LR                override training learning rate (e.g. 5e-5 for fine-tune).
#   ENTROPY_COST      override entropy bonus (e.g. 1e-4 for fine-tune).
#   NUM_TIMESTEPS     override training.num_timesteps (e.g. 200000000).
#
# Each run lands in checkpoints/run_<timestamp>_<strategy>_seed<seed>/ by
# default, so consecutive runs do NOT overwrite each other.
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

EXTRA_ARGS=()
[[ -n "${INIT_BALL_SPEED:-}" ]] && EXTRA_ARGS+=(--init-ball-speed "$INIT_BALL_SPEED")
[[ -n "${INIT_TILT_FRAC:-}" ]]  && EXTRA_ARGS+=(--init-tilt-frac  "$INIT_TILT_FRAC")
if [[ "${TILT_BUMPS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--tilt-bumps)
  [[ -n "${TILT_BUMP_PROB:-}" ]] && EXTRA_ARGS+=(--tilt-bump-prob "$TILT_BUMP_PROB")
  [[ -n "${TILT_BUMP_MAG:-}" ]]  && EXTRA_ARGS+=(--tilt-bump-magnitude "$TILT_BUMP_MAG")
fi
[[ -n "${RUN_NAME:-}" ]]     && EXTRA_ARGS+=(--run-name      "$RUN_NAME")
[[ -n "${RESUME:-}" ]]       && EXTRA_ARGS+=(--resume        "$RESUME")
[[ -n "${LR:-}" ]]            && EXTRA_ARGS+=(--learning-rate "$LR")
[[ -n "${ENTROPY_COST:-}" ]]  && EXTRA_ARGS+=(--entropy-cost  "$ENTROPY_COST")
[[ -n "${NUM_TIMESTEPS:-}" ]] && EXTRA_ARGS+=(--num-timesteps "$NUM_TIMESTEPS")

python train.py --wandb --prior-strategy "$STRATEGY" --prior-sigma "$SIGMA" "${EXTRA_ARGS[@]}"
