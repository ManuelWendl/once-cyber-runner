#!/bin/bash
# Create the W&B sweep and dispatch a fleet of sbatch agents on the cluster.
#
# Usage:
#   ./sweeps/launch_sweep.sh                      # default 16 agents, 1 trial each
#   NUM_AGENTS=32 TRIALS_PER_AGENT=2 ./sweeps/launch_sweep.sh
#   SWEEP_ID=existing/sweep/id ./sweeps/launch_sweep.sh   # skip creation
#
# Requirements:
#   - `wandb` CLI logged in (run `wandb login` once on the cluster).
#   - `sbatch` available on the host running this script.

set -e

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SWEEP_YAML=${SWEEP_YAML:-$REPO_DIR/sweeps/ppo_gpu_sweep.yaml}
NUM_AGENTS=${NUM_AGENTS:-16}
TRIALS_PER_AGENT=${TRIALS_PER_AGENT:-1}

if [ -z "$SWEEP_ID" ]; then
  echo "Creating W&B sweep from $SWEEP_YAML ..."
  # `wandb sweep` prints "Created sweep with ID: <id>" and "Run sweep agent with:
  # wandb agent <entity>/<project>/<id>" on stderr. We grep the agent line.
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

echo "Submitting $NUM_AGENTS sbatch agents (count=$TRIALS_PER_AGENT each) ..."
sbatch \
  --array=0-$((NUM_AGENTS - 1)) \
  --export=ALL,SWEEP_ID=$SWEEP_ID,COUNT=$TRIALS_PER_AGENT \
  $REPO_DIR/sweeps/sweep_agent.sbatch
