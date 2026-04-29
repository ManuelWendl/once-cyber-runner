#!/bin/bash

#SBATCH --job-name=render_gpu_policy
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/render_%j.out

set -e

# ==== Required ====
: "${PARAMS:?set PARAMS=/path/to/brax_ppo_params*.pkl}"
OUTDIR=${OUTDIR:-eval_videos}
MAX_SPAWNS=${MAX_SPAWNS:-32}
INIT_BALL_SPEED=${INIT_BALL_SPEED:-0.05}
INIT_TILT_FRAC=${INIT_TILT_FRAC:-0.05}
SEED=${SEED:-0}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
WANDB_PROJECT=${WANDB_PROJECT:-cyberrunner-prior-gpu}
WANDB_ENTITY=${WANDB_ENTITY:-filippostaffoni-eth-z-rich}

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

mkdir -p logs $OUTDIR

# ==== CUDA ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== MuJoCo headless ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/envs/cyberrunner_gpu

export PYTHONPATH=$REPO_DIR:$PYTHONPATH

cd $REPO_DIR
echo "Render params=$PARAMS  outdir=$OUTDIR  max_spawns=$MAX_SPAWNS"

WANDB_ARGS=""
[ -n "$WANDB_RUN_ID" ] && WANDB_ARGS="--wandb_run_id $WANDB_RUN_ID"
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="$WANDB_ARGS --wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"

python render_gpu_policy.py \
  --params "$PARAMS" \
  --outdir "$OUTDIR" \
  --max_spawns "$MAX_SPAWNS" \
  --seed "$SEED" \
  --init_ball_speed "$INIT_BALL_SPEED" \
  --init_tilt_frac "$INIT_TILT_FRAC" \
  $WANDB_ARGS

echo "Render done."
