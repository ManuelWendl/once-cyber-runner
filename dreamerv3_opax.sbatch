#!/bin/bash
#
# OPAX (DreamerV3 + DisagreementEnsemble) on the cyberrunner CPU env.
# Mirrors world-model-luca:train.sbatch — the canonical "known working"
# OPAX recipe. See .claude/rules/sooper-plan.md.
#
# Usage:
#   sbatch dreamerv3_opax.sbatch                                # easy, size25m, pure exploration
#   LAYOUT=medium sbatch dreamerv3_opax.sbatch
#   SIZE=size12m sbatch dreamerv3_opax.sbatch                   # smaller model
#   EXTRA="--run.steps 200000 --run.envs 4" sbatch dreamerv3_opax.sbatch
#
# Optional env vars (defaults match luca's known-working setup):
#   LAYOUT      easy | medium | hard                 (default: easy)
#   SIZE        size1m | size12m | size25m | size50m | size100m | size200m | size400m
#                                                    (default: size25m)
#   EXPLORE     explore_disagreement | explore_mixed (default: explore_disagreement)
#   RUN_NAME    custom logdir name                   (default: opax_<ts>_<layout>)
#   CONFIGS     full --configs preset list           (default: defaults <SIZE> <EXPLORE>)
#   TASK        Dreamer task name                    (default: cyberrunner_default)
#   EPISODE_LEN env.cyberrunner.episode_length       (default: 6000 — matches our MJX env)
#   EXTRA       any extra `--key value` flags forwarded to main.py
#
# The DreamerV3 stack uses a separate cluster conda env from the Brax PPO
# stack — JAX/elements/portal pinning differs. Provision once with
# `conda env create -f environment-dreamerv3.yml -p /cluster/courses/pmlr/teams/team06/cyberrunner_sooper`.
#
#SBATCH --job-name=opax
#SBATCH --account=pmlr_jobs
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/opax_%j.out

# Force line-buffered stdout/stderr so partial output survives even if the
# script aborts in the first few seconds (slurm block-buffers redirects).
exec 1> >(stdbuf -oL cat)
exec 2> >(stdbuf -oL cat >&2)

# Print every command to stderr BEFORE running it. With set -e below, the
# last echoed line is the one that failed.
set -x
set -euo pipefail

echo "[boot] $(date)  jobid=${SLURM_JOB_ID:-?}  node=${SLURM_NODELIST:-?}  cwd=$(pwd)"

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

# Defaults match world-model-luca:train.sbatch (the known-working recipe).
LAYOUT=${LAYOUT:-easy}
SIZE=${SIZE:-size25m}
EXPLORE=${EXPLORE:-explore_disagreement}
TASK=${TASK:-cyberrunner_default}
EPISODE_LEN=${EPISODE_LEN:-2000}
CONFIGS=${CONFIGS:-"defaults $SIZE $EXPLORE"}

TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-opax_${TS}_${LAYOUT}_${SIZE}}
LOGDIR=$REPO_DIR/logdir/$RUN_NAME

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/logdir"

nvidia-smi

. /etc/profile.d/modules.sh
module load cuda/12.9

# Point JAX at the system CUDA toolkit (newer ptxas) instead of the
# pip-shipped one inside the conda env, which is too old for the GPU
# and triggers `ptxas too old. Falling back to the driver` → UNIMPLEMENTED.
CUDA_DIR=/cluster/data/cuda/x86_64/12.9
export PATH=$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=$CUDA_DIR

# Match luca's setup: let JAX manage GPU memory itself, force fp32 matmul
# (the default tf32 was numerically unstable for the world-model loss).
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_DIR"
export JAX_DEFAULT_MATMUL_PRECISION=float32
# Headless GPU rendering for the CPU env's vision pipeline.
export MUJOCO_GL=${MUJOCO_GL:-egl}
# Wandb tolerance for cluster startup latency.
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_sooper

# JAX picks the pip-shipped nvidia-cuda-nvcc/bin/ptxas via Python package
# discovery, ignoring XLA_FLAGS. That ptxas is too old for the GPU →
# JAX warns "ptxas too old", falls back to the driver, then driver-side
# compilation fails with UNIMPLEMENTED. Replace the bundled ptxas with a
# symlink to the system (cuda/12.9) ptxas. Idempotent; the .bak guard means
# subsequent runs are no-ops.
PTXAS_BUNDLED=/cluster/courses/pmlr/teams/team06/cyberrunner_sooper/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas
PTXAS_SYSTEM=$CUDA_DIR/bin/ptxas
if [ -e "$PTXAS_BUNDLED" ] && [ ! -L "$PTXAS_BUNDLED" ] && [ -x "$PTXAS_SYSTEM" ]; then
  echo "[ptxas-fix] backing up bundled ptxas → ${PTXAS_BUNDLED}.bak"
  mv "$PTXAS_BUNDLED" "${PTXAS_BUNDLED}.bak"
  ln -s "$PTXAS_SYSTEM" "$PTXAS_BUNDLED"
fi
echo "[ptxas-check] which: $(command -v ptxas || echo none)"
echo "[ptxas-check] bundled resolves to: $(readlink -f "$PTXAS_BUNDLED" || echo missing)"
"$PTXAS_BUNDLED" --version 2>&1 | head -2 || true

cd "$REPO_DIR"

echo "============================================================"
echo "DreamerV3 OPAX run (mirrors world-model-luca:train.sbatch)"
echo "  configs:        $CONFIGS"
echo "  task:           $TASK"
echo "  layout:         $LAYOUT"
echo "  episode_length: $EPISODE_LEN"
echo "  logdir:         $LOGDIR"
echo "  extra:          ${EXTRA:-<none>}"
echo "============================================================"

python dreamerv3/dreamerv3/main.py \
    --configs $CONFIGS \
    --task "$TASK" \
    --logdir "$LOGDIR" \
    --seed 0 \
    --batch_size 16 \
    --batch_length 64 \
    --vision True \
    --layout "$LAYOUT" \
    --random_start False \
    --run.steps 1.1e6 \
    --run.envs 16 \
    --run.train_ratio 512 \
    --replay.size 5e5 \
    --agent.opt.lr 4e-5 \
    --env.cyberrunner.episode_length "$EPISODE_LEN" \
    ${EXTRA:-}
