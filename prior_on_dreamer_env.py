"""Roll out the survival prior on the CPU env DreamerV3 uses, via PriorObsAdapter.

This is the discriminating test for whether SOOPER's prior failure is the
obs adapter, OOD states, or a late trigger.

  - render_safe_prior.py rolls out env_mjx with the prior reading
    env_mjx._build_obs() directly. That works (manual inspection of the
    render confirmed the marble stays alive). So the prior itself is
    competent.
  - This script keeps the prior fixed but swaps in the SOOPER plumbing:
      CPU CyberRunnerEnv  ->  PriorObsAdapter  ->  load_survival_prior
    Same prior, same physics constants (env_mjx.py and env_mujoco.py
    share build_model + layout + path-progress code), only the obs
    construction differs.

If the marble survives here, the adapter is fine and SOOPER's failure
must be OOD states (OPAX drives the marble into states the prior never
saw during its curriculum) or a too-late trigger. If the marble dies,
the adapter is broken and a unit-level dim-by-dim diff against
env_mjx._build_obs is the next step.

Usage (cluster, headless):

    MUJOCO_GL=egl python prior_on_dreamer_env.py \\
        --checkpoint .vendor/cyberrunner_ppo/checkpoints/<run>/best.pkl \\
        --layout easy --episodes 10 \\
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
sys.path.insert(0, str(REPO_DIR))

# cyberrunner_env_vision.py lives at the repo root — same module DreamerV3's
# embodied wrapper imports.
from cyberrunner_env_vision import CyberRunnerEnv  # noqa: E402

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
    p.add_argument("--episodes", type=int, default=10)
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
        "--no-randomize-init",
        dest="randomize_init",
        action="store_false",
        help="Spawn at waypoints[0] each episode (default: random waypoint).",
    )
    p.set_defaults(randomize_init=True)
    return p.parse_args()


def rollout_episode(
    env: CyberRunnerEnv,
    prior_fn,
    adapter: PriorObsAdapter,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode driven by the prior + adapter; record qpos/qvel."""
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


def render_videos(
    env: CyberRunnerEnv,
    episodes_traj: List[Dict[str, Any]],
    outdir: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Offscreen mp4 rendering by replaying qpos/qvel through a renderer.

    Needs MUJOCO_GL=egl on headless GPUs.
    """
    import imageio
    import mujoco

    outdir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    data = mujoco.MjData(env.model)
    for i, traj in enumerate(episodes_traj):
        frames = []
        for q, v in zip(traj["qpos"], traj["qvel"]):
            data.qpos[:] = q
            data.qvel[:] = v
            mujoco.mj_forward(env.model, data)
            renderer.update_scene(data, camera="board")
            frames.append(renderer.render())
        path = outdir / f"episode_{i+1:02d}_{traj['termination']}.mp4"
        imageio.mimsave(path, frames, fps=fps)
        print(
            f"  wrote {path} ({len(frames)} frames, term={traj['termination']})"
        )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        blob = pickle.load(f)
    cfg = blob.get("config", {})
    layout = args.layout or cfg.get("env", {}).get("maze_layout", "hard")
    print(
        f"  step={blob.get('step', '?')} "
        f"trained_layout={cfg.get('env', {}).get('maze_layout', '?')} "
        f"strategy={cfg.get('env', {}).get('safe_prior_strategy', '?')} "
        f"domain_rand={cfg.get('env', {}).get('domain_randomization', False)} "
        f"  rendering on layout={layout}"
    )

    print("Building CPU CyberRunnerEnv (the env DreamerV3 uses, vision off)...")
    env = CyberRunnerEnv(
        layout=layout,
        episode_length=args.episode_length,
        randomize_init_pos=args.randomize_init,
        include_vision=False,
    )
    print(f"  states_dim={env.observation_space['states'].shape}")
    print(f"  action_dim={env.action_space.shape}")
    print(f"  randomize_init_pos={args.randomize_init}")

    print("Loading survival prior (jit warm-up — first call is slow) ...")
    prior_fn = load_survival_prior(str(args.checkpoint))
    adapter = PriorObsAdapter(num_envs=1)

    rng = np.random.default_rng(args.seed)
    episodes_traj: List[Dict[str, Any]] = []
    term_counts: Dict[str, int] = {"hole": 0, "goal": 0, "timeout": 0, "unknown": 0}
    steps_by_term: Dict[str, List[int]] = {k: [] for k in term_counts}

    for ep in range(args.episodes):
        seed_i = int(rng.integers(0, 2**31 - 1))
        t0 = time.time()
        traj = rollout_episode(
            env, prior_fn, adapter,
            seed=seed_i, max_steps=args.episode_length,
        )
        elapsed = time.time() - t0
        term = traj["termination"]
        term_counts[term] = term_counts.get(term, 0) + 1
        steps_by_term[term].append(traj["steps"])
        print(
            f"  ep {ep+1}/{args.episodes}: steps={traj['steps']} "
            f"term={term} (seed={seed_i}, {elapsed:.1f}s)"
        )
        episodes_traj.append(traj)

    n = args.episodes
    print()
    print(f"Summary over {n} episodes (layout={layout}):")
    for k, v in term_counts.items():
        if v == 0:
            continue
        mean_steps = float(np.mean(steps_by_term[k])) if steps_by_term[k] else 0.0
        print(f"  {k:10s} {v:3d} ({v/n:5.1%})  mean_steps={mean_steps:.0f}")

    print(f"\nRendering videos to {args.outdir} ...")
    render_videos(
        env, episodes_traj, args.outdir, args.fps, args.width, args.height,
    )
    print("Done.")


if __name__ == "__main__":
    main()
