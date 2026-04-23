"""Evaluate a trained PPO prior on every spawn point and save a video.

Usage (defaults work out of the box for logdir/ppo_prior_stabilize/):
    python eval_prior.py

Custom model:
    python eval_prior.py --model logdir/ppo_prior_stabilize/ppo_5600000_steps.zip

All options:
    python eval_prior.py --help
"""
import argparse
import csv
import pathlib
from collections import deque
from dataclasses import dataclass, field

import pickle

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

# ── constants matching training ──────────────────────────────────────────────
OBS_DIM = 13  # 10 states + 3 checkpoint
DEFAULT_N_STACK = 4
DEFAULT_LOGDIR = "logdir/ppo_prior_stabilize"
VIDEO_FPS = 30

# HUD colours (BGR for cv2)
COL_WHITE = (255, 255, 255)
COL_GREEN = (80, 220, 80)
COL_RED = (80, 80, 220)
COL_YELLOW = (80, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    spawn_idx: int
    spawn_point: np.ndarray
    success: bool
    total_reward: float
    steps: int
    final_ball_speed: float
    final_checkpoint_dist: float
    final_safe_margin: float
    termination_reason: str
    frames: list = field(default_factory=list, repr=False)


# ── helpers ───────────────────────────────────────────────────────────────────
def _auto_vecnorm(model_path: pathlib.Path) -> pathlib.Path:
    candidate = model_path.parent / "vecnormalize.pkl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find vecnormalize.pkl next to {model_path}. "
        "Pass --vecnorm explicitly."
    )


def _normalize_obs(
    stacked_obs: np.ndarray,
    obs_rms_mean: np.ndarray,
    obs_rms_var: np.ndarray,
    clip_obs: float = 10.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    normed = (stacked_obs - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
    return np.clip(normed, -clip_obs, clip_obs).astype(np.float32)


def _draw_hud(
    frame: np.ndarray,
    spawn_idx: int,
    n_spawns: int,
    step: int,
    cum_reward: float,
    ball_speed: float,
    safe_margin: float,
    done: bool,
    success: bool,
) -> np.ndarray:
    """Overlay telemetry on a copy of the frame (RGB in, RGB out)."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = w / 640.0  # normalise to reference width

    def txt(text: str, x: int, y: int, colour: tuple, size: float = 0.5) -> None:
        cv2.putText(out, text, (x, y), FONT, size * scale, (0, 0, 0), max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, text, (x, y), FONT, size * scale, colour,    max(1, int(2 * scale)), cv2.LINE_AA)

    txt(f"Spawn {spawn_idx + 1}/{n_spawns}", 8, int(20 * scale), COL_WHITE)
    txt(f"Step {step:4d}  R {cum_reward:6.1f}", 8, int(38 * scale), COL_WHITE)
    txt(f"Speed {ball_speed:.3f}  Margin {safe_margin:.3f}", 8, int(56 * scale), COL_WHITE)

    if done:
        label = "SUCCESS" if success else "FAILED"
        colour = COL_GREEN if success else COL_RED
        txt(label, int(w * 0.55), int(24 * scale), colour, size=0.7)

    return out


def _run_episode(
    env,
    model: PPO,
    obs_rms_mean: np.ndarray,
    obs_rms_var: np.ndarray,
    spawn_point: np.ndarray,
    spawn_idx: int,
    n_spawns: int,
    n_stack: int,
    seed: int,
    capture_frames: bool = True,
) -> EpisodeResult:
    buf: deque[np.ndarray] = deque(
        [np.zeros(OBS_DIM, dtype=np.float32)] * n_stack, maxlen=n_stack
    )
    raw_obs, _ = env.reset(seed=seed, options={"spawn_point": spawn_point})
    total_reward = 0.0
    done = False
    last_info: dict = {}
    frames: list[np.ndarray] = []

    while not done:
        buf.append(raw_obs)
        obs_in = _normalize_obs(np.concatenate(list(buf)), obs_rms_mean, obs_rms_var)
        action, _ = model.predict(obs_in, deterministic=True)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info

        if capture_frames:
            frame = env.render()
            if frame is not None:
                safe_margin = float(last_info.get("safe_hole_margin", 0.0))
                ball_speed = float(last_info.get("ball_speed", 0.0))
                step = int(last_info.get("step", env.unwrapped._step_count))
                annotated = _draw_hud(
                    frame, spawn_idx, n_spawns, step,
                    total_reward, ball_speed, safe_margin,
                    done, bool(last_info.get("success", False)),
                )
                frames.append(annotated)

    # hold last frame for 0.5 s so the outcome is readable
    if frames:
        hold = int(VIDEO_FPS * 0.5)
        frames.extend([frames[-1]] * hold)

    return EpisodeResult(
        spawn_idx=spawn_idx,
        spawn_point=spawn_point,
        success=bool(last_info.get("success", False)),
        total_reward=total_reward,
        steps=env.unwrapped._step_count,
        final_ball_speed=float(last_info.get("ball_speed", np.nan)),
        final_checkpoint_dist=float(last_info.get("checkpoint_dist", np.nan)),
        final_safe_margin=float(last_info.get("safe_hole_margin", np.nan)),
        termination_reason=str(last_info.get("termination_reason", "unknown")),
        frames=frames,
    )


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO prior on all spawn points")
    parser.add_argument(
        "--model",
        type=str,
        default=str(pathlib.Path(DEFAULT_LOGDIR) / "best_model.zip"),
        help="Path to .zip model file",
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl (auto-detected from model dir if omitted)",
    )
    parser.add_argument("--outdir", type=str, default="eval_videos", help="Output directory")
    parser.add_argument("--n_stack", type=int, default=DEFAULT_N_STACK)
    parser.add_argument("--seed", type=int, default=42, help="Base seed (each spawn gets seed+spawn_idx)")
    parser.add_argument(
        "--max_spawns", type=int, default=0, help="Limit spawn count (0 = all)"
    )
    parser.add_argument(
        "--prior_task",
        type=str,
        default="stabilize",
        choices=["stabilize", "checkpoint"],
    )
    parser.add_argument(
        "--prior_spawn_source",
        type=str,
        default="waypoints",
        choices=["waypoints", "dense_path"],
    )
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument(
        "--no_video", action="store_true", help="Skip frame capture (stats only)"
    )
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vecnorm_path = pathlib.Path(args.vecnorm) if args.vecnorm else _auto_vecnorm(model_path)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model  : {model_path}")
    print(f"Loading vecnorm: {vecnorm_path}")
    model = PPO.load(str(model_path))
    with open(vecnorm_path, "rb") as fh:
        vec_norm = pickle.load(fh)
    obs_rms_mean = vec_norm.obs_rms.mean.astype(np.float32)
    obs_rms_var = vec_norm.obs_rms.var.astype(np.float32)

    # The saved stats cover (n_stack * obs_dim,) — broadcast mean/var to match.
    expected_len = args.n_stack * OBS_DIM
    if obs_rms_mean.shape[0] != expected_len:
        # Tile single-frame stats if saved from a non-stacked env (fallback)
        repeats = expected_len // obs_rms_mean.shape[0]
        obs_rms_mean = np.tile(obs_rms_mean, repeats)
        obs_rms_var = np.tile(obs_rms_var, repeats)

    # ── build eval env ────────────────────────────────────────────────────────
    from train_ppo import FlattenObsWrapper
    from envs.cyberrunner import CyberRunnerEnv

    env = FlattenObsWrapper(
        CyberRunnerEnv(
            render_mode="rgb_array",
            episode_length=500,
            randomize_init_pos=False,
            include_vision=False,
            reward_every_n_waypoints=3,
            hole_penalty=3.0,
            checkpoint_radius=0.015,
            checkpoint_hold_steps=6,
            checkpoint_speed_threshold=0.05,
            checkpoint_arrival_reward=0.0,
            checkpoint_stabilize_reward=6.0,
            checkpoint_hold_reward=1.0,
            safe_hole_margin=0.004,
            checkpoint_speed_ema_alpha=0.8,
            checkpoint_include_corridors=True,
            prior_mode=True,
            prior_task=args.prior_task,
            prior_spawn_source=args.prior_spawn_source,
            prior_start_waypoint_window=3,
            prior_init_ball_speed=0.2,
            prior_init_tilt_frac=0.25,
            prior_min_checkpoint_start_dist=0.02,
            prior_max_checkpoint_start_dist=0.12,
            prior_spawn_min_hole_margin=0.012,
            prior_start_point_spacing=0.01,
            prior_spawn_merge_radius=0.02,
            checkpoint_progress_reward_scale=2.0,
            terminate_on_checkpoint_stabilized=True,
        ),
        seed=args.seed,
    )

    spawn_points = np.asarray(env.unwrapped.prior_start_points, dtype=np.float32)
    if args.max_spawns > 0:
        idxs = np.linspace(0, len(spawn_points) - 1, min(args.max_spawns, len(spawn_points)), dtype=int)
        spawn_points = spawn_points[idxs]

    n_spawns = len(spawn_points)
    print(f"Evaluating {n_spawns} spawn points  (capture_frames={not args.no_video})")

    # ── run episodes ──────────────────────────────────────────────────────────
    results: list[EpisodeResult] = []
    all_frames: list[np.ndarray] = []

    for idx, spawn_pt in enumerate(spawn_points):
        result = _run_episode(
            env=env,
            model=model,
            obs_rms_mean=obs_rms_mean,
            obs_rms_var=obs_rms_var,
            spawn_point=spawn_pt,
            spawn_idx=idx,
            n_spawns=n_spawns,
            n_stack=args.n_stack,
            seed=args.seed + idx,
            capture_frames=not args.no_video,
        )
        results.append(result)
        all_frames.extend(result.frames)

        status = "OK " if result.success else "---"
        print(
            f"  [{status}] spawn {idx + 1:3d}/{n_spawns}  "
            f"steps={result.steps:3d}  "
            f"reward={result.total_reward:6.1f}  "
            f"reason={result.termination_reason}"
        )

    env.close()

    # ── save video ────────────────────────────────────────────────────────────
    if all_frames:
        video_path = outdir / "eval_all_spawns.mp4"
        with imageio.get_writer(str(video_path), fps=args.fps, format="ffmpeg", codec="libx264", quality=7) as writer:
            for frame in all_frames:
                writer.append_data(frame)
        print(f"\nVideo saved → {video_path}  ({len(all_frames)} frames)")

    # ── save CSV summary ──────────────────────────────────────────────────────
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "spawn_idx", "spawn_x", "spawn_y",
                "success", "total_reward", "steps",
                "final_ball_speed", "final_checkpoint_dist",
                "final_safe_margin", "termination_reason",
            ],
        )
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
                "final_checkpoint_dist": f"{r.final_checkpoint_dist:.4f}",
                "final_safe_margin": f"{r.final_safe_margin:.4f}",
                "termination_reason": r.termination_reason,
            })
    print(f"CSV saved      → {csv_path}")

    # ── print aggregate summary ───────────────────────────────────────────────
    n_success = sum(r.success for r in results)
    success_rate = n_success / len(results) if results else 0.0
    mean_reward = float(np.mean([r.total_reward for r in results]))
    mean_steps = float(np.mean([r.steps for r in results]))
    reasons = {}
    for r in results:
        reasons[r.termination_reason] = reasons.get(r.termination_reason, 0) + 1

    print("\n── Aggregate ────────────────────────────────")
    print(f"  Spawns evaluated : {len(results)}")
    print(f"  Success rate     : {n_success}/{len(results)} = {success_rate:.1%}")
    print(f"  Mean reward      : {mean_reward:.2f}")
    print(f"  Mean episode len : {mean_steps:.1f}")
    print(f"  Termination breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:20s}: {count:3d}  ({count/len(results):.1%})")
    print("─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
