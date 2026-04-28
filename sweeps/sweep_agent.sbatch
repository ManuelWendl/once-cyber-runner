#!/bin/bash

#SBATCH --job-name=ppo_sweep_agent
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%j.out

# W&B sweep agent runner for the GPU PPO prior.
#
# Required env: SWEEP_ID (e.g. "filippostaffoni-eth-z-rich/cyberrunner-prior-gpu-sweep/abc123")
# Optional env: COUNT (number of trials this agent will pick up before exiting; default 1).
#
# Submit with:
#   SWEEP_ID=<entity/project/id> sbatch --array=0-15 sweeps/sweep_agent.sbatch
# (or use sweeps/launch_sweep.sh which wraps this).

set -e

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: SWEEP_ID env var is required (entity/project/sweepid)." >&2
  exit 1
fi

COUNT=${COUNT:-1}

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
RUN_NAME=ppo_sweep_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
SCRATCH_DIR=/work/scratch/$USER/ppo_gpu_sweep/$RUN_NAME

mkdir -p $SCRATCH_DIR logs

# ==== GPU info ====
nvidia-smi

# ==== CUDA setup ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== Rendering off (sweep agents skip the post-training video) ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/envs/cyberrunner_gpu

# ==== PYTHONPATH ====
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

# Force every trial in this agent to write into its own scratch logdir.
# `--logdir` overrides the value baked into the sweep YAML, which is fine
# because wandb agent appends extra `--key=value` pairs at the end.
export WANDB_DIR=$SCRATCH_DIR

cd $REPO_DIR

echo "Starting wandb agent for sweep $SWEEP_ID (count=$COUNT)"
echo "  scratch=$SCRATCH_DIR"

# `wandb agent` will keep launching trials until $COUNT is reached or the
# sbatch time limit is hit, whichever comes first.
wandb agent --count $COUNT "$SWEEP_ID"

echo "Agent finished."
