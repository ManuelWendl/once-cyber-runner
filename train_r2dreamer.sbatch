#!/bin/bash

#SBATCH --job-name=r2dreamer
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_%j.out

set -e

# ==== Configurable settings ====
#   - Data augmentation = env.mirror_augment, default False (matches env yaml).
ENV_CONFIG=${ENV_CONFIG:-cyberrunner_vision}    # cyberrunner_vision | cyberrunner_state
MODEL_SIZE=${MODEL_SIZE:-size50M}               # size12M | size25M | size50M | size100M | ...
REP_LOSS=${REP_LOSS:-r2dreamer}                 # r2dreamer | dreamer | infonce | dreamerpro
MIRROR_AUGMENT=${MIRROR_AUGMENT:-false}         # Replay-buffer mirror augmentation (CyberRunner-specific)
SEED=${SEED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}                      # Additional Hydra overrides

# ==== Paths ====
REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

# ==== Derived names ====
RUN_NAME=r2dreamer_${ENV_CONFIG}_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}
SCRATCH_DIR=/work/scratch/$USER/r2dreamer/$RUN_NAME
PERMA_DIR=$HOME/r2dreamer_results/$RUN_NAME

mkdir -p $SCRATCH_DIR $PERMA_DIR logs

# ==== Cleanup hook ====
cleanup() {
  echo "[cleanup] copying essential files to $PERMA_DIR"
  cp $SCRATCH_DIR/config.yaml $PERMA_DIR/ 2>/dev/null || true
  cp $SCRATCH_DIR/*.jsonl $PERMA_DIR/ 2>/dev/null || true
  cp $SCRATCH_DIR/console.log $PERMA_DIR/ 2>/dev/null || true
  cp $SCRATCH_DIR/latest.pt $PERMA_DIR/ 2>/dev/null || true
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
echo "Starting R2Dreamer training"
echo "  env=$ENV_CONFIG  model=$MODEL_SIZE  rep_loss=$REP_LOSS  mirror_augment=$MIRROR_AUGMENT  seed=$SEED"
echo "  scratch=$SCRATCH_DIR"
echo "  perma=$PERMA_DIR"

cd $REPO_DIR
python train.py \
  env=$ENV_CONFIG \
  model=$MODEL_SIZE \
  model.rep_loss=$REP_LOSS \
  env.mirror_augment=$MIRROR_AUGMENT \
  seed=$SEED \
  logdir=$SCRATCH_DIR \
  $EXTRA_ARGS

echo "Training complete."
