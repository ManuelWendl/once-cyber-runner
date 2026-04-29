"""Render the GPU-trained (Brax PPO / MJX) prior on the CPU CyberRunner env.

The Brax-trained policy is a pure JAX function. Rendering reuses the existing
CPU `CyberRunnerEnv(render_mode='rgb_array')` to produce frames that match
`eval_prior.py` visually. Per-frame obs is built from the CPU env's DENSE
dict obs (states + checkpoint → 11-dim) and externally frame-stacked to match
training (N_STACK=3 → 33-dim).

Usage:
    python render_gpu_policy.py \
        --params logdir/ppo_gpu_stabilize/brax_ppo_params.pkl \
        --outdir eval_videos --max_spawns 32

Optional: pass --wandb_run_id to upload the video to that W&B run.
"""
from __future__ import annotations

import argparse
import csv
import functools
import pathlib
import pickle
from collections import deque
from dataclasses import dataclass, field

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from envs.cyberrunner import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DENSE_OBS_DIM,
    PRIOR_VERSION_DENSE,
    WALL_RADIUS,
    CyberRunnerEnv,
    _point_to_segment_distance,
)

N_STACK = 3

VIDEO_FPS = 30
COL_WHITE = (255, 255, 255)
COL_GREEN = (80, 220, 80)
COL_RED = (80, 80, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class EpisodeResult:
    spawn_idx: int
    spawn_point: np.ndarray
    success: bool
    total_reward: float
    steps: int
    final_ball_speed: float
    final_hole_dist: float
    termination_reason: str
    frames: list = field(default_factory=list, repr=False)


def _draw_hud(frame, spawn_idx, n_spawns, step, reward, speed, hole_margin, done, success):
    out = frame.copy()
    h, w = out.shape[:2]
    scale = w / 640.0

    def txt(text, x, y, colour, size=0.5):
        cv2.putText(out, text, (x, y), FONT, size * scale, (0, 0, 0),
                    max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, text, (x, y), FONT, size * scale, colour,
                    max(1, int(2 * scale)), cv2.LINE_AA)

    txt(f"Spawn {spawn_idx + 1}/{n_spawns}", 8, int(20 * scale), COL_WHITE)
    txt(f"Step {step:4d}  R {reward:6.1f}", 8, int(38 * scale), COL_WHITE)
    txt(f"Speed {speed:.3f}  Margin {hole_margin:.3f}", 8, int(56 * scale), COL_WHITE)
    if done:
        label = "SUCCESS" if success else "FAILED"
        colour = COL_GREEN if success else COL_RED
        txt(label, int(w * 0.55), int(24 * scale), colour, size=0.7)
    return out


def _dense_per_frame_obs(raw_obs: dict) -> np.ndarray:
    """Concatenate `states` (8) + `checkpoint` (3) → 11-dim DENSE obs frame.

    Matches the per-frame layout produced by `CyberRunnerMJXEnv._obs_per_frame`
    in PRIOR_VERSION_DENSE mode. The CPU env emits the same fields; we just
    flatten the dict here.
    """
    return np.concatenate(
        [raw_obs["states"], raw_obs["checkpoint"]], axis=0
    ).astype(np.float32)


def _build_eval_env(init_ball_speed: float, init_tilt_frac: float) -> CyberRunnerEnv:
    return CyberRunnerEnv(
        render_mode="rgb_array",
        episode_length=500,
        randomize_init_pos=False,
        include_vision=False,
        reward_every_n_waypoints=3,
        hole_penalty=10.0,
        checkpoint_radius=0.015,
        checkpoint_hold_steps=6,
        checkpoint_speed_threshold=0.05,
        checkpoint_arrival_reward=0.0,
        checkpoint_stabilize_reward=10.0,
        checkpoint_hold_reward=1.0,
        safe_hole_margin=0.004,
        checkpoint_speed_ema_alpha=0.8,
        checkpoint_include_corridors=True,
        prior_mode=True,
        prior_task="stabilize",
        prior_version=PRIOR_VERSION_DENSE,
        prior_obs_mode=PRIOR_VERSION_DENSE,
        prior_spawn_source="waypoints",
        prior_start_waypoint_window=3,
        prior_init_ball_speed=init_ball_speed,
        prior_init_tilt_frac=init_tilt_frac,
        prior_min_checkpoint_start_dist=0.02,
        prior_max_checkpoint_start_dist=0.12,
        prior_spawn_min_hole_margin=0.012,
        prior_start_point_spacing=0.01,
        prior_spawn_merge_radius=0.02,
        checkpoint_progress_reward_scale=2.0,
        terminate_on_checkpoint_stabilized=True,
    )


def _split_inference_params(params) -> tuple[tuple, int, tuple[int, ...]]:
    """Pull (normalizer, policy_only) out of the saved (normalizer, PPONetworkParams).

    Saved pickle stores `(running_statistics, PPONetworkParams(policy, value))`
    — the value head is needed for training but not for inference.
    `make_inference_fn` calls `policy_network.apply(*params, obs)` with
    params=(normalizer, policy), so we drop the value side here.

    Also returns `obs_size` (from the normalizer state shape) and
    `policy_hidden_sizes` inferred from the policy Dense kernel chain.
    """
    normalizer, network_params = params
    policy_params = network_params.policy
    obs_size = int(jax.tree_util.tree_leaves(normalizer)[0].shape[0])
    kernels = [
        l for l in jax.tree_util.tree_leaves(policy_params) if l.ndim == 2
    ]
    hidden = tuple(int(k.shape[1]) for k in kernels[:-1])
    return (normalizer, policy_params), obs_size, hidden


def _make_inference_fn(
    inference_params: tuple, obs_size: int, action_size: int,
    hidden_sizes: tuple[int, ...],
):
    """Rebuild the policy used in `train_ppo_gpu.py` and load saved params.

    Training uses a custom factory: separate (h, h) MLPs for policy/value, tanh
    activation, DiagGaussian (NormalDistribution, no tanh squashing). The
    inference fn here must match that factory or the saved params won't bind.
    """
    from brax.training import distribution as brax_distribution
    from brax.training import networks as brax_networks
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    class _IdentityBijector:
        # Brax accepts any object with these three methods as a postprocessor;
        # avoids depending on a specific base class name across versions.
        def forward(self, x):
            return x

        def inverse(self, y):
            return y

        def forward_log_det_jacobian(self, x):
            return jnp.zeros_like(x[..., 0])

    class NormalDiagDistribution(brax_distribution.ParametricDistribution):
        def __init__(self, event_size, min_std=0.001, var_scale=1.0):
            super().__init__(
                param_size=2 * event_size,
                postprocessor=_IdentityBijector(),
                event_ndims=1,
                reparametrizable=True,
            )
            self._min_std = min_std
            self._var_scale = var_scale

        def create_dist(self, parameters):
            loc, scale = jnp.split(parameters, 2, axis=-1)
            scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
            return brax_distribution.NormalDistribution(loc=loc, scale=scale)

    def make_ppo_networks_normal(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        activation=jax.nn.tanh,
    ):
        dist = NormalDiagDistribution(event_size=action_size)
        policy_network = brax_networks.make_policy_network(
            dist.param_size,
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation,
        )
        value_network = brax_networks.make_value_network(
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
        )
        return ppo_networks.PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=dist,
        )

    network = make_ppo_networks_normal(
        observation_size=obs_size, action_size=action_size,
    )
    make_inference = ppo_networks.make_inference_fn(network)
    return make_inference(inference_params, deterministic=True)


def _run_episode(
    env: CyberRunnerEnv, inference_fn, spawn_point: np.ndarray,
    spawn_idx: int, n_spawns: int, rng: jax.Array, capture_frames: bool,
) -> EpisodeResult:
    raw_obs, _info = env.reset(seed=int(spawn_idx), options={"spawn_point": spawn_point})
    # Frame stack mirrors SB3 VecFrameStack / MJX `_stack_with_buffer`:
    # zeros for the first (N_STACK - 1) frames, oldest..newest along axis.
    buf: deque = deque(
        [np.zeros(DENSE_OBS_DIM, dtype=np.float32)] * (N_STACK - 1),
        maxlen=N_STACK - 1,
    )
    total_reward = 0.0
    done = False
    last_info: dict = {}
    frames = []
    step_count = 0
    while not done:
        frame_obs = _dense_per_frame_obs(raw_obs)
        stacked = np.concatenate([*buf, frame_obs], axis=0)
        rng, sub = jax.random.split(rng)
        action, _ = inference_fn(jnp.asarray(stacked), sub)
        buf.append(frame_obs)
        raw_obs, reward, terminated, truncated, info = env.step(np.asarray(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        last_info = info
        step_count += 1
        if capture_frames:
            frame = env.render()
            if frame is not None:
                speed = float(last_info.get("ball_speed", 0.0))
                margin = float(last_info.get("safe_hole_margin", 0.0))
                success = bool(last_info.get("success", False))
                frames.append(_draw_hud(frame, spawn_idx, n_spawns, step_count,
                                         total_reward, speed, margin, done, success))
    if frames:
        hold = int(VIDEO_FPS * 0.5)
        frames.extend([frames[-1]] * hold)

    hole_d = float(last_info.get("min_hole_distance", np.nan))
    return EpisodeResult(
        spawn_idx=spawn_idx,
        spawn_point=np.asarray(spawn_point, dtype=np.float32),
        success=bool(last_info.get("success", False)),
        total_reward=total_reward,
        steps=step_count,
        final_ball_speed=float(last_info.get("ball_speed", np.nan)),
        final_hole_dist=hole_d,
        termination_reason=str(last_info.get("termination_reason", "unknown")),
        frames=frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, required=True, help="Path to brax_ppo_params.pkl")
    parser.add_argument("--outdir", type=str, default="eval_videos")
    parser.add_argument("--max_spawns", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_ball_speed", type=float, default=0.05)
    parser.add_argument("--init_tilt_frac", type=float, default=0.05)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cyberrunner-prior-gpu")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    params_path = pathlib.Path(args.params)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading params: {params_path}")
    with open(params_path, "rb") as f:
        params_pkl = pickle.load(f)
    inference_params, obs_size, hidden_sizes = _split_inference_params(
        params_pkl["params"],
    )
    action_size = int(params_pkl.get("action_size") or 2)
    print(f"  obs_size={obs_size}  action_size={action_size}  hidden={hidden_sizes}")
    if obs_size != N_STACK * DENSE_OBS_DIM:
        raise ValueError(
            f"Saved checkpoint expects obs_size={obs_size}, but render builds "
            f"{N_STACK * DENSE_OBS_DIM}-dim obs (N_STACK={N_STACK} × DENSE_OBS_DIM={DENSE_OBS_DIM}). "
            "This script only supports DENSE prior checkpoints."
        )

    inference_fn = _make_inference_fn(
        inference_params, obs_size, action_size, hidden_sizes,
    )

    # CPU env for rendering; uses the same simplified init as training.
    env = _build_eval_env(args.init_ball_speed, args.init_tilt_frac)
    _ = env.reset(seed=args.seed)

    spawn_points = np.asarray(env.prior_start_points, dtype=np.float32)
    if args.max_spawns > 0 and len(spawn_points) > args.max_spawns:
        idxs = np.linspace(0, len(spawn_points) - 1, args.max_spawns, dtype=int)
        spawn_points = spawn_points[idxs]

    n = len(spawn_points)
    print(f"Rendering {n} spawn points (capture_frames={not args.no_video})")

    rng = jax.random.PRNGKey(args.seed)
    results = []
    all_frames = []
    for idx, sp in enumerate(spawn_points):
        rng, sub = jax.random.split(rng)
        r = _run_episode(
            env, inference_fn, spawn_point=sp,
            spawn_idx=idx, n_spawns=n, rng=sub,
            capture_frames=not args.no_video,
        )
        results.append(r)
        all_frames.extend(r.frames)
        status = "OK " if r.success else "---"
        print(f"  [{status}] spawn {idx+1:3d}/{n}  steps={r.steps:3d}  "
              f"reward={r.total_reward:6.1f}  reason={r.termination_reason}")

    env.close()

    video_path = None
    if all_frames:
        video_path = outdir / "eval_all_spawns_gpu.mp4"
        with imageio.get_writer(str(video_path), fps=args.fps, format="ffmpeg",
                                codec="libx264", quality=7) as writer:
            for frame in all_frames:
                writer.append_data(frame)
        print(f"Video saved → {video_path}  ({len(all_frames)} frames)")

    csv_path = outdir / "summary_gpu.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "spawn_idx", "spawn_x", "spawn_y",
            "success", "total_reward", "steps",
            "final_ball_speed", "final_hole_dist", "termination_reason",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "spawn_idx": r.spawn_idx,
                "spawn_x": f"{r.spawn_point[0]:.4f}",
                "spawn_y": f"{r.spawn_point[1]:.4f}",
                "success": int(r.success),
                "total_reward": f"{r.total_reward:.2f}",
                "steps": r.steps,
                "final_ball_speed": f"{r.final_ball_speed:.4f}",
                "final_hole_dist": f"{r.final_hole_dist:.4f}",
                "termination_reason": r.termination_reason,
            })
    print(f"CSV saved   → {csv_path}")

    n_success = sum(r.success for r in results)
    rate = n_success / len(results) if results else 0.0
    print(f"\n── Aggregate ─────────────────")
    print(f"  Spawns evaluated : {len(results)}")
    print(f"  Success rate     : {n_success}/{len(results)} = {rate:.1%}")
    print(f"  Mean reward      : {float(np.mean([r.total_reward for r in results])):.2f}")

    if args.wandb_run_id and video_path is not None:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                id=args.wandb_run_id,
                resume="allow",
            )
            wandb.log({
                "eval/video": wandb.Video(str(video_path), fps=args.fps, format="mp4"),
                "eval/final_success_rate": rate,
                "eval/final_spawns": len(results),
            })
            wandb.finish()
            print(f"Uploaded video to W&B run {args.wandb_run_id}")
        except Exception as exc:
            print(f"W&B upload failed: {exc}")


if __name__ == "__main__":
    main()
