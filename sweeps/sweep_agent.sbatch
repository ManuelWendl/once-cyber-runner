#!/bin/bash

#SBATCH --job-name=ppo_sweep_agent
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%j.out

# Single long-lived W&B sweep agent for the student cluster (one job at a
# time per user). Mirrors train_ppo_gpu.sbatch's environment exactly so the
# sweep behaves identically to a manual run.
#
# Required env: SWEEP_ID (e.g. "entity/project/abc123")
# Optional env: COUNT (number of sequential trials before the agent exits;
#               default 20). With 1M steps per trial and 12h walltime, ~15-20
#               trials normally fit comfortably; the agent exits cleanly when
#               either COUNT or the walltime is reached.
#
# Submit with sweeps/launch_sweep.sh, or directly:
#   SWEEP_ID=<entity/project/id> sbatch sweeps/sweep_agent.sbatch

set -e

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: SWEEP_ID env var is required (entity/project/sweepid)." >&2
  exit 1
fi

COUNT=${COUNT:-20}

# ==== Paths ====
REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

# ==== Derived names ====
RUN_NAME=ppo_sweep_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}
SCRATCH_DIR=/work/scratch/$USER/ppo_gpu_sweep/$RUN_NAME
PERMA_DIR=$HOME/r2dreamer_results/$RUN_NAME

mkdir -p $SCRATCH_DIR $PERMA_DIR logs

# ==== Cleanup hook ====
cleanup() {
  echo "[cleanup] copying essential files to $PERMA_DIR"
  cp $SCRATCH_DIR/*.pkl $PERMA_DIR/ 2>/dev/null || true
}
trap cleanup EXIT

# ==== GPU info ====
nvidia-smi

# ==== CUDA setup ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== Rendering (MuJoCo headless; sweep agents do not render videos) ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/envs/cyberrunner_gpu

# ==== PYTHONPATH ====
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

# Keep wandb metadata under scratch so the home quota does not balloon.
export WANDB_DIR=$SCRATCH_DIR

cd $REPO_DIR

echo "Starting wandb sweep agent"
echo "  SWEEP_ID=$SWEEP_ID  COUNT=$COUNT"
echo "  scratch=$SCRATCH_DIR  perma=$PERMA_DIR"

# Sequentially pick up COUNT trials (or until walltime). Each trial runs
# train_ppo_gpu.py with the hyperparams the sweep controller suggests.
wandb agent --count $COUNT "$SWEEP_ID"

echo "Agent finished."
