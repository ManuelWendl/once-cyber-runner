#!/bin/bash
# Create the W&B sweep and submit ONE long-lived agent on the cluster.
#
# The student cluster only allows one job at a time per user, so this
# script does NOT use sbatch arrays — it submits a single agent that
# sequentially picks up COUNT trials inside the same Slurm job.
#
# Usage:
#   ./sweeps/launch_sweep.sh                      # creates sweep + 1 agent (20 trials)
#   COUNT=10 ./sweeps/launch_sweep.sh             # cap trials per submission
#   SWEEP_ID=existing/sweep/id ./sweeps/launch_sweep.sh   # reuse a sweep
#
# When the agent's job ends (walltime or COUNT reached), just run this
# script again with SWEEP_ID=... to dispatch the next batch of trials.
#
# Requirements:
#   - `wandb` CLI logged in (run `wandb login` once on the cluster).
#   - `sbatch` available on the host running this script.

set -e

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SWEEP_YAML=${SWEEP_YAML:-$REPO_DIR/sweeps/ppo_gpu_sweep.yaml}
COUNT=${COUNT:-20}

if [ -z "$SWEEP_ID" ]; then
  echo "Creating W&B sweep from $SWEEP_YAML ..."
  SWEEP_OUT=$(wandb sweep "$SWEEP_YAML" 2>&1 | tee /dev/tty)
  SWEEP_ID=$(echo "$SWEEP_OUT" | grep -oE 'wandb agent [^ ]+' | tail -1 | awk '{print $3}')
  if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: could not parse sweep id from wandb output." >&2
    exit 1
  fi
  echo "Sweep id: $SWEEP_ID"
else
  echo "Reusing existing sweep id: $SWEEP_ID"
fi

echo "Submitting one sweep-agent sbatch job (count=$COUNT trials)."
echo "Re-run this script with SWEEP_ID=$SWEEP_ID after it finishes for more."
sbatch \
  --export=ALL,SWEEP_ID=$SWEEP_ID,COUNT=$COUNT \
  $REPO_DIR/sweeps/sweep_agent.sbatch
