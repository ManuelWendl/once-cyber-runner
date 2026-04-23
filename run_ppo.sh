#!/bin/bash
set -e
mkdir -p logs
exec python -u train_ppo.py \
  --logdir logdir/ppo_prior_stabilize_v2 \
  --steps 6000000 \
  --n_envs 16 \
  --seed 0 \
  --prior_task stabilize \
  --prior_spawn_source waypoints \
  --wandb_project cyberrunner-prior \
  --wandb_entity filippostaffoni-eth-z-rich
