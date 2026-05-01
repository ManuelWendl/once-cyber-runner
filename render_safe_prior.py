"""Render a safe-prior PPO policy trained on the vendored Brax MJX env.

Why a separate script: the vendor's eval.py runs on env_mujoco.py (CPU env),
which still uses the upstream forward-next-waypoint obs scheme. Our safe-prior
policy was trained on env_mjx.py with backward-next-waypoint obs, so the CPU
env feeds the policy the wrong observation distribution. This script rolls
out on the SAME env we trained on (env_mjx.py with safe_prior=True), records
qpos / qvel per step, and replays the trajectory through the underlying
mj_model in either an interactive MuJoCo viewer (local) or as an offscreen
mp4 video (cluster, headless).

Usage:
    # Local Mac with display (interactive window)
    python render_safe_prior.py \\
        --checkpoint .vendor/cyberrunner_ppo/checkpoints/final.pkl \\
        --mode viewer --episodes 3

    # Cluster (headless) — set MUJOCO_GL=egl before invoking
    MUJOCO_GL=egl python render_safe_prior.py \\
        --checkpoint .vendor/cyberrunner_ppo/checkpoints/final.pkl \\
        --mode video --outdir render_out --episodes 3

Requires the same conda env that trained the policy
(/cluster/courses/pmlr/teams/team06/cyberrunner_ppo on the cluster, or any
local install of the vendor's requirements.txt).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.nn
import jax.numpy as jnp
import mujoco
import numpy as np

REPO_DIR = Path(__file__).resolve().parent
VENDOR_DIR = REPO_DIR / ".vendor" / "cyberrunner_ppo"
if not VENDOR_DIR.is_dir():
    raise SystemExit(
        f"Vendor dir not found at {VENDOR_DIR}. Run from the repo root."
    )
sys.path.insert(0, str(VENDOR_DIR))

from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from env_mjx import CyberrunnerMJXEnv  # noqa: E402


_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "leaky_relu": jax.nn.leaky_relu,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="Path to final.pkl from train.py"
    )
    p.add_argument("--mode", choices=["viewer", "video"], default="viewer")
    p.add_argument(
        "--outdir",
        default="render_out",
        help="Where to write mp4s in video mode",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Steps per episode (default: read from checkpoint config)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Replay frame rate (matches training control rate)",
    )
    return p.parse_args()


def build_env(cfg: Dict[str, Any]) -> CyberrunnerMJXEnv:
    env_cfg = cfg["env"]
    return CyberrunnerMJXEnv(
        episode_length=env_cfg["episode_length"],
        randomize_init_pos=env_cfg["randomize_init_pos"],
        num_rays=env_cfg.get("num_rays", 32),
        # Single-env rollout — small warp buffer is enough.
        num_envs_hint=1,
        history_length=env_cfg.get("history_length", 5),
        safe_prior=env_cfg.get("safe_prior", False),
        init_ball_speed=env_cfg.get("init_ball_speed", 0.0),
        init_tilt_frac=env_cfg.get("init_tilt_frac", 0.0),
    )


def build_policy(checkpoint: Dict[str, Any], env: CyberrunnerMJXEnv):
    cfg = checkpoint["config"]
    net_cfg = cfg["training"]["brax_ppo"]["network"]
    activation = _ACTIVATIONS[net_cfg["activation"]]
    hidden_sizes = tuple(net_cfg["hidden_sizes"])

    network = ppo_networks.make_ppo_networks(
        observation_size={"state": (env.observation_size,)},
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        activation=activation,
        policy_obs_key="state",
        value_obs_key="state",
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)

    params = checkpoint["params"]
    if not isinstance(params, (tuple, list)) or len(params) < 2:
        raise ValueError(
            f"Unexpected params layout: {type(params)} len={len(params) if hasattr(params, '__len__') else '?'}"
        )
    inference_params = (params[0], params[1])
    return make_inference_fn(inference_params, deterministic=True)


def rollout(
    env: CyberrunnerMJXEnv, policy_fn, key: jax.Array, max_steps: int
) -> Dict[str, np.ndarray]:
    """Roll out one episode. Records qpos/qvel + reward + target_xy."""
    reset_jit = env.reset
    step_jit = env.step

    state = reset_jit(key)
    qpos_buf: List[np.ndarray] = [np.asarray(state.pipeline_state.qpos)]
    qvel_buf: List[np.ndarray] = [np.asarray(state.pipeline_state.qvel)]
    reward_buf: List[float] = []

    for _ in range(max_steps):
        key, k_act = jax.random.split(key)
        action, _ = policy_fn(state.obs, k_act)
        state = step_jit(state, action)
        qpos_buf.append(np.asarray(state.pipeline_state.qpos))
        qvel_buf.append(np.asarray(state.pipeline_state.qvel))
        reward_buf.append(float(state.reward))
        if float(state.done) > 0.5:
            break

    return {
        "qpos": np.stack(qpos_buf),
        "qvel": np.stack(qvel_buf),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "target_xy": np.asarray(state.info["safe_prior_target_xy"]),
    }


def play_viewer(mj_model, episodes_traj: List[Dict[str, np.ndarray]], fps: int) -> None:
    """Interactive MuJoCo viewer (requires display)."""
    import mujoco.viewer  # noqa: WPS433

    mj_data = mujoco.MjData(mj_model)
    dt = 1.0 / max(fps, 1)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for ep_idx, traj in enumerate(episodes_traj):
            print(
                f"Replaying episode {ep_idx + 1}/{len(episodes_traj)} "
                f"({len(traj['qpos'])} frames, return={traj['rewards'].sum():.2f}, "
                f"target={traj['target_xy'].tolist()})"
            )
            for q, v in zip(traj["qpos"], traj["qvel"]):
                if not viewer.is_running():
                    return
                mj_data.qpos[:] = q
                mj_data.qvel[:] = v
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(dt)


def play_video(
    mj_model,
    episodes_traj: List[Dict[str, np.ndarray]],
    outdir: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Offscreen mp4 rendering. Set MUJOCO_GL=egl on headless machines."""
    import imageio  # noqa: WPS433

    outdir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    mj_data = mujoco.MjData(mj_model)
    for ep_idx, traj in enumerate(episodes_traj):
        frames = []
        for q, v in zip(traj["qpos"], traj["qvel"]):
            mj_data.qpos[:] = q
            mj_data.qvel[:] = v
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render())
        out_path = outdir / f"episode_{ep_idx + 1}.mp4"
        imageio.mimsave(out_path, frames, fps=fps)
        print(
            f"Wrote {out_path} ({len(frames)} frames, "
            f"return={traj['rewards'].sum():.2f}, "
            f"target={traj['target_xy'].tolist()})"
        )


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    cfg = checkpoint["config"]
    print(
        f"  trained {checkpoint.get('step', '?')} steps; "
        f"safe_prior={cfg['env'].get('safe_prior')} "
        f"init_ball_speed={cfg['env'].get('init_ball_speed')} "
        f"init_tilt_frac={cfg['env'].get('init_tilt_frac')}"
    )

    print("Building env (this triggers JIT/Warp compile — first call is slow) ...")
    env = build_env(cfg)
    policy_fn = jax.jit(build_policy(checkpoint, env))

    max_steps = (
        args.episode_length
        if args.episode_length is not None
        else cfg["env"]["episode_length"]
    )

    rng = jax.random.PRNGKey(args.seed)
    episodes_traj: List[Dict[str, np.ndarray]] = []
    for ep in range(args.episodes):
        rng, k_reset = jax.random.split(rng)
        t0 = time.time()
        traj = rollout(env, policy_fn, k_reset, max_steps)
        print(
            f"  episode {ep + 1}/{args.episodes}: steps={len(traj['qpos']) - 1} "
            f"return={traj['rewards'].sum():.2f} ({time.time() - t0:.1f}s)"
        )
        episodes_traj.append(traj)

    if args.mode == "viewer":
        play_viewer(env.mj_model, episodes_traj, fps=args.fps)
    else:
        play_video(
            env.mj_model,
            episodes_traj,
            Path(args.outdir),
            fps=args.fps,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
