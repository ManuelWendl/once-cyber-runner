#!/bin/bash

#SBATCH --job-name=ppo_prior
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out

set -e

# ==== Configurable settings ====
STEPS=${STEPS:-1000000}
N_ENVS=${N_ENVS:-16}
SEED=${SEED:-0}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}

# ==== Paths ====
REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

# ==== Derived names ====
RUN_NAME=ppo_prior_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}
SCRATCH_DIR=/work/scratch/$USER/ppo/$RUN_NAME
PERMA_DIR=$HOME/r2dreamer_results/$RUN_NAME

mkdir -p $SCRATCH_DIR $PERMA_DIR logs

# ==== Cleanup hook ====
cleanup() {
  echo "[cleanup] copying essential files to $PERMA_DIR"
  cp $SCRATCH_DIR/*.zip $PERMA_DIR/ 2>/dev/null || true
  cp $SCRATCH_DIR/evaluations.npz $PERMA_DIR/ 2>/dev/null || true
}
trap cleanup EXIT

# ==== GPU info ====
nvidia-smi

# ==== CUDA setup ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== Rendering (MuJoCo headless) ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/r2dreamer_env

# ==== PYTHONPATH ====
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

# ==== Train ====
echo "Starting PPO prior training"
echo "  steps=$STEPS  n_envs=$N_ENVS  seed=$SEED"
echo "  scratch=$SCRATCH_DIR"
echo "  perma=$PERMA_DIR"

cd $REPO_DIR

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="--wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY"  ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"

python train_ppo.py \
  --logdir $SCRATCH_DIR \
  --steps $STEPS \
  --n_envs $N_ENVS \
  --seed $SEED \
  $WANDB_ARGS

echo "Training complete."
