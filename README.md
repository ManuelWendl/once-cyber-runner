# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides a PyTorch implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a computationally efficient world model that achieves high performance on continuous control benchmarks. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Installing via a virtual env like uv is recommended.
pip install -r requirements.txt
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Headless rendering

If you run MuJoCo-based environments on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)


## Uncertainty-driven exploration on R2-Dreamer

This branch adds uncertainty-driven exploration machinery onto **R2-Dreamer**. The implementation is based on how Vass did this in his branch `world-model-vass` for DreamerV3. It also parametrizes the CyberRunner reward so waypoint density, hole penalty, and the exploration bonus are all config knobs.

### Optimistic R2-Dreamer

Plan2Explore-style intrinsic reward added on top of R2-Dreamer.

- [`optimistic.py`](optimistic.py)
  - `DisagreementEnsemble`: `K` parallel MLPs that predict the next encoder embedding from `(h_t, z_t, a_{t+1})`.
  - `disagreement(preds)`: L2 norm of per-dimension variance across ensemble members — a scalar `σ`.
- [`dreamer.py`](dreamer.py) integration (guarded by `model.optimistic`):
  - Ensemble is built alongside the RSSM; optimizer and module dict pick it up automatically.
  - Training loss `losses["ensemble"]`: MSE between ensemble prediction and the *stop-gradiented* next encoder embedding, so no gradients flow back through the encoder via this path.
  - Imagination-time reward is augmented: `imag_reward += optimistic_lambda * σ_π`, with `σ_π` computed under `no_grad` so the policy treats the bonus as an exogenous reward.
  - `compute_sigma(data)`: offline hook used by the visualizer to map σ onto replay rollouts.

Relevant config (`configs/model/_base_.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `model.optimistic` | `False` | Turn exploration bonus on/off |
| `model.optimistic_lambda` | `1.0` | Bonus weight added to imagined reward |
| `model.opt_ensemble.K` | `5` | Number of MLP members |
| `model.opt_ensemble.{layers,units,act}` | `2, 256, SiLU` | Member MLP shape |
| `model.loss_scales.ensemble` | `1.0` | Scale for the ensemble training loss |

### Parametrized CyberRunner reward

[`envs/cyberrunner.py`](envs/cyberrunner.py)'s `_compute_reward` replaces the original dense progress reward with a parametrized version:

- **Sparse checkpoint reward** — `CHECKPOINT_REWARD = 1.0` is paid once per `reward_every_n_waypoints` waypoints crossed, tracked via `_max_checkpoint_reached` so it's never double-paid.
- **Goal bonus** — unchanged: `GOAL_BONUS` when the ball is within `GOAL_THRESHOLD` of the goal.
- **Hole penalty** — `-hole_penalty` whenever the ball overlaps any hole (`‖ball − hole‖ < HOLE_RADIUS`).

The two knobs are plumbed end-to-end (`envs/__init__.py` → `CyberRunner.__init__` → `CyberRunnerEnv.__init__`) and surfaced in the env configs:

```yaml
# configs/env/cyberrunner_state.yaml (and cyberrunner_vision.yaml)
reward_every_n_waypoints: 3
hole_penalty: 5.0
```

### Running

SLURM (see [`train_r2dreamer.sbatch`](train_r2dreamer.sbatch)):

```bash
ENV_CONFIG=cyberrunner_vision MODEL_SIZE=size50M REP_LOSS=r2dreamer \
MIRROR_AUGMENT=false SEED=0 EXTRA_ARGS="model.optimistic=True" \
sbatch train_r2dreamer.sbatch
```
