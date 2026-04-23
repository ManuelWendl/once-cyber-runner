#!/bin/bash

#SBATCH --job-name=ppo_prior_gpu
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out

set -e

# ==== Configurable settings ====
STEPS=${STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-2048}
UNROLL_LENGTH=${UNROLL_LENGTH:-20}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-20}
NUM_UPDATES_PER_BATCH=${NUM_UPDATES_PER_BATCH:-4}
SEED=${SEED:-0}
INIT_BALL_SPEED=${INIT_BALL_SPEED:-0.05}
INIT_TILT_FRAC=${INIT_TILT_FRAC:-0.05}
NUM_EVALS=${NUM_EVALS:-100}
MAX_VIDEO_SPAWNS=${MAX_VIDEO_SPAWNS:-32}
WANDB_PROJECT=${WANDB_PROJECT:-cyberrunner-prior-gpu}
WANDB_ENTITY=${WANDB_ENTITY:-filippostaffoni-eth-z-rich}

# ==== Paths ====
REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr

# ==== Derived names ====
RUN_NAME=ppo_prior_gpu_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}
SCRATCH_DIR=/work/scratch/$USER/ppo_gpu/$RUN_NAME
PERMA_DIR=$HOME/r2dreamer_results/$RUN_NAME

mkdir -p $SCRATCH_DIR $PERMA_DIR logs eval_videos

# ==== Cleanup hook ====
cleanup() {
  echo "[cleanup] copying essential files to $PERMA_DIR"
  cp $SCRATCH_DIR/*.pkl $PERMA_DIR/ 2>/dev/null || true
  cp $SCRATCH_DIR/wandb_run_id.txt $PERMA_DIR/ 2>/dev/null || true
  cp eval_videos/eval_all_spawns_gpu.mp4 $PERMA_DIR/ 2>/dev/null || true
  cp eval_videos/summary_gpu.csv $PERMA_DIR/ 2>/dev/null || true
}
trap cleanup EXIT

# ==== GPU info ====
nvidia-smi

# ==== CUDA setup ====
. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

# ==== Rendering (MuJoCo headless, for the post-training video) ====
export MUJOCO_GL=egl

# ==== Conda ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/envs/cyberrunner_gpu

# ==== PYTHONPATH ====
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

# ==== Train ====
echo "Starting PPO prior (GPU) training"
echo "  steps=$STEPS  num_envs=$NUM_ENVS  seed=$SEED"
echo "  init_ball_speed=$INIT_BALL_SPEED  init_tilt_frac=$INIT_TILT_FRAC"
echo "  scratch=$SCRATCH_DIR  perma=$PERMA_DIR"

cd $REPO_DIR

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="--wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY"  ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"

python train_ppo_gpu.py \
  --logdir $SCRATCH_DIR \
  --steps $STEPS \
  --num_envs $NUM_ENVS \
  --unroll_length $UNROLL_LENGTH \
  --batch_size $BATCH_SIZE \
  --num_minibatches $NUM_MINIBATCHES \
  --num_updates_per_batch $NUM_UPDATES_PER_BATCH \
  --num_evals $NUM_EVALS \
  --seed $SEED \
  --init_ball_speed $INIT_BALL_SPEED \
  --init_tilt_frac $INIT_TILT_FRAC \
  $WANDB_ARGS

echo "Training complete. Rendering policy video..."

WANDB_ID_FILE=$SCRATCH_DIR/wandb_run_id.txt
WANDB_ID_ARG=""
[ -f "$WANDB_ID_FILE" ] && WANDB_ID_ARG="--wandb_run_id $(cat $WANDB_ID_FILE)"

python render_gpu_policy.py \
  --params $SCRATCH_DIR/brax_ppo_params.pkl \
  --outdir eval_videos \
  --max_spawns $MAX_VIDEO_SPAWNS \
  --init_ball_speed $INIT_BALL_SPEED \
  --init_tilt_frac $INIT_TILT_FRAC \
  $WANDB_ID_ARG

echo "Rendering complete."
