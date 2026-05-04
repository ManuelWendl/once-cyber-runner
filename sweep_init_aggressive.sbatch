#!/bin/bash
#
# Single-allocation sweep over increasingly aggressive spawn init levels.
# Loops train.py serially inside ONE slurm job (1 entry in the queue) so we
# don't trip QOSMaxSubmitJobPerUserLimit (3-job cap) — each run is short.
#
# Usage:
#   sbatch sweep_init_aggressive.sbatch                  # exp_d, default LEVELS
#   STRATEGY=survival sbatch sweep_init_aggressive.sbatch
#   LEVELS="mild:0.05:0.05 hard:0.20:0.15" sbatch sweep_init_aggressive.sbatch
#
# Each level lands in its own checkpoint subdir + wandb run name so curves
# stay separable. If one level crashes, the loop continues (set +e around
# the python call) so the rest of the sweep still runs.
#
#SBATCH --job-name=sp_sweep
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sp_sweep_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
VENDOR_DIR=$REPO_DIR/.vendor/cyberrunner_ppo

STRATEGY=${STRATEGY:-exp_d}
SIGMA=${SIGMA:-0.02}

# Default sweep ladder. Override via LEVELS env var (space-separated).
DEFAULT_LEVELS="mild:0.05:0.05 med:0.10:0.10 hard:0.20:0.15 harder:0.40:0.25 extreme:0.80:0.40"
LEVELS=${LEVELS:-$DEFAULT_LEVELS}

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
export WANDB_PROJECT=${WANDB_PROJECT:-cyberrunner_safe_prior_${STRATEGY}_init_sweep}

cd "$VENDOR_DIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Sweep over init levels: $LEVELS"
echo "Strategy: $STRATEGY  Sigma: $SIGMA"
echo "Wandb project: $WANDB_PROJECT"
echo "Timestamp: $TS"
echo "============================================================"

for level in $LEVELS; do
  IFS=':' read -r label speed tilt <<< "$level"
  run_name="run_${TS}_${STRATEGY}_${label}_b${speed}_t${tilt}"
  echo
  echo ">>> [$(date +%H:%M:%S)] level=$label  ball_speed=$speed  tilt_frac=$tilt"
  echo ">>> run_name=$run_name"
  set +e
  python train.py \
    --wandb \
    --prior-strategy "$STRATEGY" \
    --prior-sigma "$SIGMA" \
    --init-ball-speed "$speed" \
    --init-tilt-frac "$tilt" \
    --run-name "$run_name"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo ">>> level=$label FAILED (exit $rc). Continuing with next level."
  else
    echo ">>> level=$label done."
  fi
done

echo
echo "Sweep complete."
