# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A MuJoCo-based Gymnasium environment simulating the CyberRunner labyrinth puzzle robot. The project focuses on efficient exploration algorithms for navigating a marble through a physical maze using two actuated joints (alpha/beta).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cyberrunner
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE  # required on macOS
conda deactivate && conda activate cyberrunner
```

## Common Commands

```bash
# Visualize environment with random actions (click to step)
python test_visualize_2d.py

# Train PPO or SAC
python train.py --algo ppo --timesteps 1000000 --n-envs 8
python train.py --algo sac --timesteps 2000000

# Test a trained model (MuJoCo viewer requires mjpython on macOS)
mjpython test_model.py --algo ppo
mjpython test_model.py --algo sac --no-render --episodes 20
mjpython test_model.py --algo ppo --model-path ./models/ppo_final.zip

# Hyperparameter sweep
python sweep.py --algo ppo --trials 30 --timesteps 500000 --n-envs 8
python sweep.py --algo sac --trials 30 --timesteps 500000

# Monitor training
tensorboard --logdir ./logs
optuna-dashboard sqlite:///sweep_ppo.db
optuna-dashboard sqlite:///sweep_sac.db
```

## Training Pipeline

- `train.py` — trains PPO or SAC via `--algo`. Saves best model to `./models/<algo>_best/` and final to `./models/<algo>_final.zip`. Logs to `./logs/<algo>/` for TensorBoard.
- `sweep.py` — Optuna hyperparameter sweep for PPO or SAC. Persists results to `sweep_<algo>.db` (resumable). Tighten search ranges based on best trials found.
- `test_model.py` — loads a saved model and runs episodes. Use `--no-render` for stats only.

## Architecture

### `cyberrunner_env.py`
Single-file environment implementing the full simulation stack:

- **Physics constants** (top): System-identified parameters for actuators, joints, board, and marble
- **`get_hard_layout()`**: Returns the hard maze configuration as numpy arrays — `walls_h` (horizontal walls `[x_start, x_end, y]`), `walls_v` (vertical walls `[y_start, y_end, x]`), `holes`, and `waypoints`
- **`compute_path_progress()`**: Vectorized raycasting to find the closest visible path point from the marble, with wall/hole occlusion. Falls back to direct projection when marble is within 2mm of path. Returns `(progress*10, seg_idx, param, closest_point)`
- **`build_model()` + helpers**: Constructs the MuJoCo `MjSpec` programmatically — board with two-joint linkage (alpha=Y-axis, beta=X-axis), maze walls with rounded caps, holes (visual only), and actuators
- **`CyberRunnerEnv`**: Gymnasium env with 10-dim observation space and 2-dim action space

**Observation space** (10 dims): `[alpha_joint, beta_joint, ball_x, ball_y, vec_to_closest_path(2), vec_to_next_wp(2), vec_to_next_next_wp(2)]`

**Action space**: `[-1, 1]^2` for alpha/beta motor commands

**Reward**: Change in path progress + `GOAL_BONUS=10.0` on reaching the final waypoint (within 4mm)

**Simulation**: 600Hz physics (`TIMESTEP=0.00166...`), 60Hz control (`FRAME_SKIP=10`). Board dimensions: 276×231mm.

### `test_visualize_2d.py`
Interactive matplotlib visualization — draws board layout, marble position, and observation vectors (closest path point, next waypoint, waypoint+1). Click to advance steps; close window to exit.

## Key Implementation Notes

- **MuJoCo body hierarchy**: `world → link (alpha_joint) → board (beta_joint)`. The marble is a free-floating body in `world`.
- **Coordinate system**: Board coordinates are in meters. qpos layout is `[alpha, beta, marble_x, marble_y, marble_z, qw, qx, qy, qz]`
- **Collision groups**: Board floor/edges use `contype=2`, marble uses `contype=4, conaffinity=7`. Holes are visual-only (`contype=0`).
- **Observation noise**: Per-episode bias + per-step noise applied to ball position (1mm) and joint angles (0.25°)
- **Path progress raycasting**: Uses 32 rays from marble position; checks wall and hole occlusion before accepting path intersections. Returns -1.0 if no visible path found.
- **Wall caps**: `check_endpoint_connected()` determines if wall endpoints need rounded cylinder caps (only when not connected to another wall or board edge)
- **SAC vs PPO**: SAC is off-policy (replay buffer) and more sample-efficient but always uses 1 env. PPO is on-policy and benefits from `--n-envs` parallelism.
- **Sweep best practices**: High learning rate trials that score well are usually reward hacking. Verify with `ep_len_mean` in TensorBoard — real learning shows increasing episode length. `gamma` should be close to 1 (≥0.99) for this long-horizon task. `ent_coef` sweet spot is typically `0.005–0.03`.
