"""Roll out the survival prior side-by-side on the CPU CyberRunnerEnv
(via PriorObsAdapter) AND env_mjx (native obs), from the same fixed
spawn at waypoints[0]. Used to visually diagnose where the two pipelines
diverge.

Three rollouts:

  - env_mjx native (render_safe_prior.py-equivalent): the prior reads
    env_mjx._build_obs() directly. Confirmed working in prior tests.
  - CPU CyberRunnerEnv + PriorObsAdapter: the SOOPER plumbing the gate
    uses. Same prior weights, same maze, same physics constants — only
    the obs construction differs.

If the env_mjx rollout survives and the CPU rollout consistently fails
in the same direction, the bug is in the CPU pipeline (adapter, CPU env
path-progress, noise model, etc.) — and the side-by-side videos reveal
which.

Usage (cluster, headless):

    MUJOCO_GL=egl python prior_on_dreamer_env.py \\
        --checkpoint .vendor/cyberrunner_ppo/checkpoints/<run>/best.pkl \\
        --layout easy --episodes 5 \\
        --outdir render_out_adapter
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
VENDOR_DIR = REPO_DIR / ".vendor" / "cyberrunner_ppo"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(VENDOR_DIR))

# cyberrunner_env_vision.py lives at the repo root — same module DreamerV3's
# embodied wrapper imports.
from cyberrunner_env_vision import CyberRunnerEnv  # noqa: E402

# env_mjx is the vendored MJX env the prior was trained on.
from env_mjx import CyberrunnerMJXEnv  # noqa: E402

# PriorObsAdapter + load_survival_prior live in the SOOPER module. Importing
# triggers brax/flax/jax import chains (handled gracefully by sooper.py).
from dreamerv3.dreamerv3.sooper import (  # noqa: E402
    PriorObsAdapter,
    load_survival_prior,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a survival-prior best.pkl trained with .vendor/cyberrunner_ppo/train.py.",
    )
    p.add_argument(
        "--layout",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Maze layout. Default: read from checkpoint config (env.maze_layout).",
    )
    p.add_argument("--outdir", default="render_out_adapter", type=Path)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument(
        "--episode-length",
        type=int,
        default=2000,
        help="Max env steps per episode. The episode also ends on hole/goal.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Replay frame rate. Matches the env's 60 Hz control rate.",
    )
    p.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip the CPU CyberRunnerEnv + PriorObsAdapter rollouts.",
    )
    p.add_argument(
        "--skip-mjx",
        action="store_true",
        help="Skip the env_mjx native rollouts.",
    )
    return p.parse_args()


def rollout_cpu(
    env: CyberRunnerEnv,
    prior_fn,
    adapter: PriorObsAdapter,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one CPU-env episode driven by prior_fn(adapter(obs))."""
    import jax.numpy as jnp

    obs, _ = env.reset(seed=seed)
    adapter.reset_envs(np.array([True]))
    prev_action = np.zeros(2, dtype=np.float32)

    qpos_buf: List[np.ndarray] = [env.data.qpos.copy()]
    qvel_buf: List[np.ndarray] = [env.data.qvel.copy()]
    termination = "timeout"
    steps = 0

    for _ in range(max_steps):
        prior_obs = adapter.transform(
            {"states": obs["states"][None, :]},
            prev_action[None, :],
        )
        action = np.asarray(prior_fn(jnp.asarray(prior_obs)))[0]
        obs, _, terminated, truncated, info = env.step(action)
        qpos_buf.append(env.data.qpos.copy())
        qvel_buf.append(env.data.qvel.copy())
        prev_action = action
        steps += 1
        if terminated or truncated:
            termination = info.get("termination_reason", "unknown")
            break

    return {
        "qpos": np.stack(qpos_buf),
        "qvel": np.stack(qvel_buf),
        "termination": termination,
        "steps": steps,
    }


def rollout_mjx(
    env: CyberrunnerMJXEnv,
    prior_fn,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one env_mjx episode driven by the prior reading the env's
    native 36-dim obs. Same prior network as rollout_cpu — only the
    obs construction differs."""
    import jax
    import jax.numpy as jnp

    state = env.reset(jax.random.PRNGKey(seed))
    qpos_buf: List[np.ndarray] = [np.asarray(state.pipeline_state.qpos)]
    qvel_buf: List[np.ndarray] = [np.asarray(state.pipeline_state.qvel)]
    termination = "timeout"
    steps = 0

    for _ in range(max_steps):
        obs_36 = state.obs["state"]
        if obs_36.ndim == 1:
            obs_36 = obs_36[None, :]
        action = prior_fn(jnp.asarray(obs_36))[0]
        state = env.step(state, action)
        qpos_buf.append(np.asarray(state.pipeline_state.qpos))
        qvel_buf.append(np.asarray(state.pipeline_state.qvel))
        steps += 1
        if float(state.done) > 0.5:
            had_hole = float(state.info.get("had_hole", 0.0))
            termination = "hole" if had_hole > 0.5 else "goal"
            break

    return {
        "qpos": np.stack(qpos_buf),
        "qvel": np.stack(qvel_buf),
        "termination": termination,
        "steps": steps,
    }


def render_video(
    mj_model,
    traj: Dict[str, Any],
    path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Offscreen mp4 of a single trajectory. Needs MUJOCO_GL=egl headless."""
    import imageio
    import mujoco

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    data = mujoco.MjData(mj_model)
    frames = []
    for q, v in zip(traj["qpos"], traj["qvel"]):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera="board")
        frames.append(renderer.render())
    imageio.mimsave(path, frames, fps=fps)
    print(f"  wrote {path} ({len(frames)} frames, term={traj['termination']})")


def _print_summary(
    label: str,
    n: int,
    term_counts: Dict[str, int],
    steps_by_term: Dict[str, List[int]],
) -> None:
    print(f"\n{label} over {n} episodes:")
    for k, v in term_counts.items():
        if v == 0:
            continue
        mean_steps = float(np.mean(steps_by_term[k])) if steps_by_term[k] else 0.0
        print(f"  {k:10s} {v:3d} ({v/n:5.1%})  mean_steps={mean_steps:.0f}")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.skip_cpu and args.skip_mjx:
        raise SystemExit("Both --skip-cpu and --skip-mjx set; nothing to do.")

    print(f"Loading {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        blob = pickle.load(f)
    cfg = blob.get("config", {})
    env_cfg = cfg.get("env", {})
    layout = args.layout or env_cfg.get("maze_layout", "hard")
    print(
        f"  step={blob.get('step', '?')} "
        f"trained_layout={env_cfg.get('maze_layout', '?')} "
        f"strategy={env_cfg.get('safe_prior_strategy', '?')} "
        f"init_ball_speed={env_cfg.get('init_ball_speed', 0.0)} "
        f"init_tilt_frac={env_cfg.get('init_tilt_frac', 0.0)} "
        f"DR={env_cfg.get('domain_randomization', False)} "
        f"  rendering on layout={layout}"
    )

    cpu_env = None
    mjx_env = None
    adapter = None

    if not args.skip_cpu:
        print("\nBuilding CPU CyberRunnerEnv (randomize_init_pos=False) ...")
        cpu_env = CyberRunnerEnv(
            layout=layout,
            episode_length=args.episode_length,
            randomize_init_pos=False,
            include_vision=False,
        )
        adapter = PriorObsAdapter(num_envs=1)
        print(f"  states_dim={cpu_env.observation_space['states'].shape}")

    if not args.skip_mjx:
        print("Building env_mjx (randomize_init_pos=False, training-time init noise) ...")
        mjx_env = CyberrunnerMJXEnv(
            episode_length=args.episode_length,
            randomize_init_pos=False,
            num_envs_hint=1,
            history_length=env_cfg.get("history_length", 5),
            maze_layout=layout,
            safe_prior=True,
            safe_prior_strategy=env_cfg.get("safe_prior_strategy", "survival"),
            safe_prior_sigma=env_cfg.get("safe_prior_sigma", 0.02),
            init_ball_speed=env_cfg.get("init_ball_speed", 0.0),
            init_tilt_frac=env_cfg.get("init_tilt_frac", 0.0),
            tilt_bumps=env_cfg.get("tilt_bumps", False),
            tilt_bump_prob=env_cfg.get("tilt_bump_prob", 0.0),
            tilt_bump_magnitude=env_cfg.get("tilt_bump_magnitude", 0.0),
            domain_randomization=env_cfg.get("domain_randomization", False),
            domain_randomization_pct=env_cfg.get("domain_randomization_pct", 0.15),
        )
        print(f"  mjx backend: {mjx_env._mjx_impl}")

    print("\nLoading survival prior (jit warm-up — first call is slow) ...")
    prior_fn = load_survival_prior(str(args.checkpoint))

    rng = np.random.default_rng(args.seed)
    cpu_trajs: List[Dict[str, Any]] = []
    mjx_trajs: List[Dict[str, Any]] = []
    cpu_term: Dict[str, int] = {"hole": 0, "goal": 0, "timeout": 0, "unknown": 0}
    mjx_term: Dict[str, int] = {"hole": 0, "goal": 0, "timeout": 0, "unknown": 0}
    cpu_steps: Dict[str, List[int]] = {k: [] for k in cpu_term}
    mjx_steps: Dict[str, List[int]] = {k: [] for k in mjx_term}

    for ep in range(args.episodes):
        seed_i = int(rng.integers(0, 2**31 - 1))
        print(f"\n=== episode {ep+1}/{args.episodes} (seed={seed_i}) ===")

        if cpu_env is not None:
            t0 = time.time()
            tr = rollout_cpu(cpu_env, prior_fn, adapter, seed_i, args.episode_length)
            cpu_term[tr["termination"]] = cpu_term.get(tr["termination"], 0) + 1
            cpu_steps[tr["termination"]].append(tr["steps"])
            cpu_trajs.append(tr)
            print(f"  CPU  steps={tr['steps']:4d} term={tr['termination']:8s} ({time.time()-t0:.1f}s)")

        if mjx_env is not None:
            t0 = time.time()
            tr = rollout_mjx(mjx_env, prior_fn, seed_i, args.episode_length)
            mjx_term[tr["termination"]] = mjx_term.get(tr["termination"], 0) + 1
            mjx_steps[tr["termination"]].append(tr["steps"])
            mjx_trajs.append(tr)
            print(f"  MJX  steps={tr['steps']:4d} term={tr['termination']:8s} ({time.time()-t0:.1f}s)")

    n = args.episodes
    if cpu_env is not None:
        _print_summary("CPU env (via PriorObsAdapter)", n, cpu_term, cpu_steps)
    if mjx_env is not None:
        _print_summary("env_mjx (native obs)", n, mjx_term, mjx_steps)

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering videos to {args.outdir} ...")
    for i, tr in enumerate(cpu_trajs):
        path = args.outdir / f"episode_{i+1:02d}_cpu_{tr['termination']}.mp4"
        render_video(cpu_env.model, tr, path, args.fps, args.width, args.height)
    for i, tr in enumerate(mjx_trajs):
        path = args.outdir / f"episode_{i+1:02d}_mjx_{tr['termination']}.mp4"
        render_video(mjx_env.mj_model, tr, path, args.fps, args.width, args.height)
    print("Done.")


if __name__ == "__main__":
    main()
