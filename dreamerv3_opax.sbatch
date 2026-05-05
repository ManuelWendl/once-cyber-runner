#!/bin/bash
#
# OPAX (DreamerV3 + DisagreementEnsemble) on the cyberrunner CPU env.
# Ported from the world-model-luca branch — see .claude/rules/sooper-plan.md.
#
# Usage:
#   sbatch dreamerv3_opax.sbatch                                # easy maze, defaults
#   LAYOUT=medium sbatch dreamerv3_opax.sbatch
#   EXTRA="--run.steps 200000 --run.envs 4" sbatch dreamerv3_opax.sbatch
#
# Optional env vars:
#   LAYOUT      easy | medium | hard           (default: easy)
#   RUN_NAME    custom logdir name             (default: opax_<ts>_<layout>)
#   CONFIGS     space-separated config presets (default: cyberrunner_state explore_disagreement)
#   EXTRA       any extra `--key value` flags forwarded to main.py
#
# The DreamerV3 stack uses a separate cluster conda env from the Brax PPO
# stack — JAX/elements/portal pinning differs. Provision once with
# `conda env create -f environment-dreamerv3.yml -p /cluster/courses/pmlr/teams/team06/cyberrunner_dreamerv3`.
#
#SBATCH --job-name=opax
#SBATCH --account=pmlr_jobs
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/opax_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

LAYOUT=${LAYOUT:-easy}
CONFIGS=${CONFIGS:-"cyberrunner_state explore_disagreement"}
TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-opax_${TS}_${LAYOUT}}
LOGDIR=$REPO_DIR/logdir/$RUN_NAME

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/logdir"

nvidia-smi

. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_triton_gemm_any=true"
# Headless GPU rendering for the CPU env's vision pipeline.
export MUJOCO_GL=${MUJOCO_GL:-egl}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_dreamerv3

cd "$REPO_DIR"

echo "============================================================"
echo "DreamerV3 OPAX run"
echo "  configs:  $CONFIGS"
echo "  layout:   $LAYOUT"
echo "  logdir:   $LOGDIR"
echo "  extra:    ${EXTRA:-<none>}"
echo "============================================================"

python dreamerv3/dreamerv3/main.py \
  --configs $CONFIGS \
  --layout "$LAYOUT" \
  --logdir "$LOGDIR" \
  ${EXTRA:-}
