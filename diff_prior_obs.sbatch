#!/bin/bash
#
# Dim-by-dim diff between env_mjx._build_obs (the prior's training obs)
# and PriorObsAdapter.transform (the SOOPER gate's reconstructed obs).
# Both pipelines are run on the same spawn with bias + per-step noise
# zeroed; non-zero diffs are the adapter bug.
#
# Env vars:
#   LAYOUT   easy | medium | hard (default: easy)
#   STEPS    number of control steps to compare (default: 8)
#   SEED     RNG seed (only affects MuJoCo solver micro-state; default: 0)
#
#SBATCH --job-name=diff_prior_obs
#SBATCH --account=pmlr_jobs
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/diff_prior_obs_%j.out

set -euo pipefail

REPO_DIR=$HOME/efficient-exploration-for-cyberrunner-pmlr
LAYOUT=${LAYOUT:-easy}
STEPS=${STEPS:-8}
SEED=${SEED:-0}

mkdir -p "$REPO_DIR/logs"

nvidia-smi || true

. /etc/profile.d/modules.sh
module load cuda/12.9
export PATH=/cluster/data/cuda/x86_64/12.9/bin:$PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

source ~/miniconda3/etc/profile.d/conda.sh
# Need both dreamerv3 (for PriorObsAdapter) and the vendored env_mjx
# (for CyberrunnerMJXEnv). The sooper env has both.
conda activate /cluster/courses/pmlr/teams/team06/cyberrunner_sooper

cd "$REPO_DIR"

python diff_prior_obs.py --layout "$LAYOUT" --steps "$STEPS" --seed "$SEED"
