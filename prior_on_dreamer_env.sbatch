#!/bin/bash
#
# Render the survival prior driving the DreamerV3-side CPU env, via the
# SOOPER PriorObsAdapter. Companion to render_safe_prior.sbatch — that one
# rolls out env_mjx with env_mjx._build_obs; this one rolls out the CPU
# CyberRunnerEnv with the adapter. If the marble dies here but lives there,
# the adapter is the SOOPER bug.
#
# Env vars:
#   CHECKPOINT  absolute path to the survival-prior best.pkl (required).
#   LAYOUT      easy | medium | hard. Empty -> read from checkpoint config.
#   OUTDIR      where to write mp4s. Default: $REPO_DIR/render_out_adapter.
#   EPISODES    number of episodes to roll out (default 10).
#
#SBATCH --job-name=prior_on_dreamer_env
#SBATCH --account=pmlr_jobs
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prior_on_dreamer_env_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
CHECKPOINT=${CHECKPOINT:?must set CHECKPOINT=path/to/best.pkl}
OUTDIR=${OUTDIR:-$REPO_DIR/render_out_adapter}
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
# Use the sooper env: it has dreamerv3 + brax + flax + mujoco. The
# cyberrunner_ppo env would also work for loading the prior, but lacks
# the dreamerv3 package needed to import PriorObsAdapter.
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_sooper

cd "$REPO_DIR"

EXTRA_ARGS=()
[[ -n "${LAYOUT:-}" ]] && EXTRA_ARGS+=(--layout "$LAYOUT")

python prior_on_dreamer_env.py \
    --checkpoint "$CHECKPOINT" \
    --outdir "$OUTDIR" \
    --episodes "$EPISODES" \
    "${EXTRA_ARGS[@]}"
