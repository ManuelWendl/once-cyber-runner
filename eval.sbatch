#!/bin/bash

#SBATCH --job-name=r2dreamer_eval
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/eval_%j.out

set -e

# ==== Required ====
: "${CHECKPOINT:?set CHECKPOINT=/path/to/latest.pt}"
ENV_CONFIG=${ENV_CONFIG:-cyberrunner_prior_state}
MODEL_SIZE=${MODEL_SIZE:-size12M}
NUM_EPISODES=${NUM_EPISODES:-5}
VIDEO_PATH=${VIDEO_PATH:-$(dirname "$CHECKPOINT")/eval.mp4}
EXTRA_ARGS=${EXTRA_ARGS:-}

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

mkdir -p logs

# ==== CUDA ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== MuJoCo headless ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/courses/pmlr/teams/team06/r2dreamer_env

export PYTHONPATH=$REPO_DIR:$PYTHONPATH

cd $REPO_DIR
echo "Eval checkpoint=$CHECKPOINT -> $VIDEO_PATH"
python eval.py \
  env=$ENV_CONFIG \
  model=$MODEL_SIZE \
  checkpoint=$CHECKPOINT \
  video_path=$VIDEO_PATH \
  num_episodes=$NUM_EPISODES \
  wandb.enabled=False \
  $EXTRA_ARGS

echo "Saved video to $VIDEO_PATH"
