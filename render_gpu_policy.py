"""Render the GPU-trained (Brax PPO / MJX) prior on the CPU CyberRunner env.

The Brax-trained policy is a pure JAX function. Rendering reuses the existing
CPU `CyberRunnerEnv(render_mode='rgb_array')` to produce frames that match
`eval_prior.py` visually; the only cross-backend coupling is a ~20-line
observation adapter from the CPU dict obs to the 10-dim MJX layout.

Usage:
    python render_gpu_policy.py \
        --params logdir/ppo_gpu_stabilize/brax_ppo_params.pkl \
        --outdir eval_videos --max_spawns 32

Optional: pass --wandb_run_id to upload the video to that W&B run.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import pickle
from dataclasses import dataclass, field

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from envs.cyberrunner import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    WALL_RADIUS,
    CyberRunnerEnv,
    _point_to_segment_distance,
)

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


def _cpu_to_mjx_obs(env: CyberRunnerEnv, raw_obs: dict, info: dict) -> np.ndarray:
    """Build the 10-dim MJX obs from the CPU env state.

    CPU env keeps the geometric ground truth (no noise-free speed/margin
    needed — the MJX obs is built from noise-free qpos for policy-consistent
    evaluation). To keep the policy input distribution close to training,
    mirror the MJX `_obs` formula exactly using the CPU env's `self.data`.
    """
    qpos = env.unwrapped.data.qpos
    alpha = float(qpos[0])
    beta = float(qpos[1])
    ball_pos = env.unwrapped._get_ball_pos_board_frame()
    # velocity: use the env's cached true ball speed vector via finite diff
    ball_vel = (ball_pos - env.unwrapped._prev_ball_pos).astype(np.float32)
    dt_ctrl = 1.0 / 60.0
    ball_vel = ball_vel / max(dt_ctrl, 1e-8)
    speed = float(np.linalg.norm(ball_vel))

    hole_d = float(np.min(np.linalg.norm(env.unwrapped.holes - ball_pos[None, :], axis=1)))
    wall_seg_d = float(_point_to_segment_distance(
        ball_pos[None, :], env.unwrapped._wall_starts, env.unwrapped._wall_ends).min())
    edge_d = float(min(ball_pos[0], BOARD_WIDTH - ball_pos[0],
                       ball_pos[1], BOARD_HEIGHT - ball_pos[1])) + WALL_RADIUS
    wall_d = min(wall_seg_d, edge_d)

    abs_tilt = float(np.sqrt(alpha * alpha + beta * beta))
    return np.asarray([
        alpha, beta,
        float(ball_pos[0]), float(ball_pos[1]),
        float(ball_vel[0]), float(ball_vel[1]),
        hole_d, wall_d,
        abs_tilt, speed,
    ], dtype=np.float32)


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


def _make_inference_fn(params_pkl: dict, obs_size: int, action_size: int):
    from brax.training.agents.ppo import networks as ppo_networks

    network_factory = ppo_networks.make_ppo_networks
    network = network_factory(observation_size=obs_size, action_size=action_size)
    make_inference = ppo_networks.make_inference_fn(network)
    return make_inference(params_pkl["params"], deterministic=True)


def _run_episode(
    env: CyberRunnerEnv, inference_fn, spawn_point: np.ndarray,
    spawn_idx: int, n_spawns: int, rng: jax.Array, capture_frames: bool,
) -> EpisodeResult:
    raw_obs, _info = env.reset(seed=int(spawn_idx), options={"spawn_point": spawn_point})
    total_reward = 0.0
    done = False
    last_info: dict = {}
    frames = []
    step_count = 0
    while not done:
        obs = _cpu_to_mjx_obs(env, raw_obs, last_info)
        rng, sub = jax.random.split(rng)
        action, _ = inference_fn(jnp.asarray(obs), sub)
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
    obs_size = int(params_pkl.get("obs_size", 10))
    action_size = int(params_pkl.get("action_size", 2))

    inference_fn = _make_inference_fn(params_pkl, obs_size, action_size)

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
