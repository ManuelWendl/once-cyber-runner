#!/bin/bash
#
# Submit a sweep of safe_prior runs with increasingly aggressive spawn init
# (ball speed + tilt). Each run goes to its own checkpoint subdir and its
# own wandb run name so the curves stay separable on the dashboard.
#
# Usage:
#   ./sweep_init_aggressive.sh                  # default exp_d strategy
#   STRATEGY=survival ./sweep_init_aggressive.sh
#
# Tweak the LEVELS array below to pick how many points / how aggressive.
# Format: "label:ball_speed:tilt_frac"
#   - ball_speed: m/s, max planar marble speed at spawn
#   - tilt_frac:  fraction of joint range used for spawn tilt
# The config defaults are 0.05 / 0.05 (already "mild"), so this sweep
# starts there and pushes up by ~2× each step.

set -euo pipefail

STRATEGY=${STRATEGY:-exp_d}
SIGMA=${SIGMA:-0.02}

LEVELS=(
  "mild:0.05:0.05"        # baseline (matches current config)
  "med:0.10:0.10"
  "hard:0.20:0.15"
  "harder:0.40:0.25"
  "extreme:0.80:0.40"
)

TS=$(date +%Y%m%d_%H%M%S)

for level in "${LEVELS[@]}"; do
  IFS=':' read -r label speed tilt <<< "$level"
  run_name="run_${TS}_${STRATEGY}_${label}_b${speed}_t${tilt}"
  echo "Submitting $run_name (ball_speed=$speed tilt_frac=$tilt)"
  STRATEGY="$STRATEGY" \
    SIGMA="$SIGMA" \
    INIT_BALL_SPEED="$speed" \
    INIT_TILT_FRAC="$tilt" \
    RUN_NAME="$run_name" \
    WANDB_PROJECT="cyberrunner_safe_prior_${STRATEGY}_init_sweep" \
    sbatch --job-name="sp_${label}" safe_prior.sbatch
done

echo
echo "Submitted ${#LEVELS[@]} jobs."
echo "Watch them with: squeue -u \$USER"
echo "Wandb project:    cyberrunner_safe_prior_${STRATEGY}_init_sweep"
echo "Checkpoints:      .vendor/cyberrunner_ppo/checkpoints/run_${TS}_${STRATEGY}_<label>_b<speed>_t<tilt>/"
