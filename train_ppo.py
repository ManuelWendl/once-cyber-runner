"""PPO training for the CyberRunner prior (stabilization) task using stable-baselines3."""
import argparse
import pathlib
import time
from collections import deque

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3.common.vec_env import sync_envs_normalization


def make_prior_env(
    seed: int = 0,
    render_mode: str | None = None,
    progress_scale: float = 2.0,
    include_corridors: bool = True,
    prior_task: str = "stabilize",
    prior_spawn_source: str = "waypoints",
    prior_version: str = "legacy",
):
    from envs.cyberrunner import (
        CHECKPOINT_RECOVERY_OBS_DIM,
        LEGACY_PRIOR_OBS_DIM,
        PRIOR_VERSION_CHECKPOINT_RECOVERY,
        PRIOR_VERSION_DENSE,
        DENSE_OBS_DIM,
        CyberRunnerEnv,
    )

    # Safe spawn applies to ALL prior versions: never start the ball within
    # 12 mm of a hole — early policies cannot recover from spawn-adjacent
    # hole failures and the resulting -hole_penalty events drown the shaping.
    prior_spawn_min_hole_margin = 0.012
    if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY:
        checkpoint_hold_steps = 25
        hole_penalty = 50.0
    elif prior_version == PRIOR_VERSION_DENSE:
        checkpoint_hold_steps = 6
        # Tuned down from 100: a 100-cost terminal made the agent freeze
        # anywhere safely far from holes rather than risk the maze to reach
        # the target. 25 is large enough that a fall costs roughly half a
        # good Phase B return, but small enough that the +arrival bonus
        # dominates the calculus over hole avoidance.
        hole_penalty = 25.0
    else:  # legacy survival reward
        checkpoint_hold_steps = 6
        hole_penalty = 50.0
    env = CyberRunnerEnv(
        render_mode=render_mode,
        episode_length=500,
        randomize_init_pos=False,
        include_vision=False,
        reward_every_n_waypoints=3,
        hole_penalty=hole_penalty,
        checkpoint_radius=0.015,
        checkpoint_hold_steps=checkpoint_hold_steps,
        checkpoint_speed_threshold=0.05,
        checkpoint_arrival_reward=0.0,
        checkpoint_stabilize_reward=10.0,
        checkpoint_hold_reward=1.0,
        safe_hole_margin=0.004,
        checkpoint_speed_ema_alpha=0.8,
        checkpoint_include_corridors=include_corridors,
        prior_mode=True,
        prior_task=prior_task,
        prior_spawn_source=prior_spawn_source,
        prior_start_waypoint_window=3,
        # Gentle init kept across all prior versions (less aggressive than Wed's
        # 0.2/0.25 — the dense module restores Wed's REWARD, not its init).
        prior_init_ball_speed=0.05,
        prior_init_tilt_frac=0.05,
        prior_min_checkpoint_start_dist=0.02,
        prior_max_checkpoint_start_dist=0.12,
        prior_spawn_min_hole_margin=prior_spawn_min_hole_margin,
        prior_start_point_spacing=0.01,
        prior_spawn_merge_radius=0.0,
        checkpoint_progress_reward_scale=progress_scale,
        # Survival prior: keep the ball alive for the full episode rather than
        # ending early on first stabilization. Matches GPU env behavior.
        terminate_on_checkpoint_stabilized=False,
        prior_version=prior_version,
    )
    if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY:
        obs_dim = CHECKPOINT_RECOVERY_OBS_DIM
    elif prior_version == PRIOR_VERSION_DENSE:
        obs_dim = DENSE_OBS_DIM
    else:
        obs_dim = LEGACY_PRIOR_OBS_DIM
    return FlattenObsWrapper(env, seed=seed, prior_version=prior_version, obs_dim=obs_dim)


class FlattenObsWrapper(gym.Wrapper):
    """Flatten Dict obs {states(10), checkpoint(3)} → Box(13,)."""

    def __init__(self, env, seed: int = 0, prior_version: str = "legacy", obs_dim: int = 13):
        super().__init__(env)
        self.prior_version = prior_version
        self.obs_dim = int(obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.env.reset(seed=seed)

    def _flatten(self, obs: dict) -> np.ndarray:
        if "prior_state" in obs:
            return np.asarray(obs["prior_state"], dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(obs["states"], dtype=np.float32),
                np.asarray(obs["checkpoint"], dtype=np.float32),
            ]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logdir/ppo")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--n_stack", type=int, default=3)
    parser.add_argument("--progress_scale", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="corners_and_corridors",
        choices=["corners", "corners_and_corridors"],
    )
    parser.add_argument(
        "--prior_task",
        type=str,
        default="stabilize",
        choices=["checkpoint", "stabilize"],
    )
    parser.add_argument(
        "--prior_spawn_source",
        type=str,
        default="waypoints",
        choices=["waypoints", "dense_path"],
    )
    parser.add_argument(
        "--prior_version",
        type=str,
        default="legacy",
        choices=["legacy", "checkpoint_recovery", "dense"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10_000)
    parser.add_argument("--robust_eval_every", type=int, default=50_000)
    parser.add_argument("--robust_eval_repeats", type=int, default=3)
    parser.add_argument("--robust_eval_max_spawns", type=int, default=32)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps: int):
            super().__init__()
            self._pbar = tqdm(total=total_timesteps, unit="step", dynamic_ncols=True)
            self._prev = 0

        def _on_step(self) -> bool:
            delta = self.num_timesteps - self._prev
            self._pbar.update(delta)
            self._prev = self.num_timesteps
            return True

        def _on_training_end(self) -> None:
            self._pbar.close()

    env_kwargs = {
        "progress_scale": args.progress_scale,
        "include_corridors": args.checkpoint_mode == "corners_and_corridors",
        "prior_task": args.prior_task,
        "prior_spawn_source": args.prior_spawn_source,
        "prior_version": args.prior_version,
    }
    info_keywords = (
        "success",
        "stable_steps",
        "stable_steps_final",
        "stable_steps_max",
        "quiet_steps",
        "quiet_frac",
        "mean_observed_speed",
        "checkpoint_dist_final",
        "checkpoint_dist_min",
        "inside_checkpoint_frac",
        "observed_speed",
        "quiet_threshold_speed",
        "ep_quiet_steps",
        "ep_ball_speed_sum",
        "ep_safe_hole_margin_sum",
        "termination_reason",
    )

    # Reward normalization OFF for prior versions that emit large terminal
    # spikes (hole_penalty=50 and/or +100 survival lump): the running std
    # then gets dominated by those events, crushing the dense per-step
    # signal. dense has bounded dense rewards (hole_penalty=25, no
    # survival bonus) and benefits from VecNormalize.
    norm_reward = args.prior_version == "dense"
    vec_env = VecNormalize(
        VecFrameStack(
            VecMonitor(
                make_vec_env(make_prior_env, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs),
                info_keywords=info_keywords,
            ),
            n_stack=args.n_stack,
        ),
        norm_obs=True, norm_reward=norm_reward, clip_obs=10.0,
    )

    eval_env = VecNormalize(
        VecFrameStack(
            VecMonitor(
                make_vec_env(make_prior_env, n_envs=4, seed=args.seed + 1000, env_kwargs=env_kwargs),
                info_keywords=info_keywords,
            ),
            n_stack=args.n_stack,
        ),
        norm_obs=True, norm_reward=False, training=False,
    )

    if args.wandb_project:
        import wandb

        class WandbCallback(BaseCallback):
            """Emit the unified CPU/GPU metric set on a fixed step cadence.

            Metric keys here MUST match those emitted by `train_ppo_gpu.py`'s
            `progress_fn` so panels overlay across backends.
            """

            EP_LEN_CAP = 500.0

            def __init__(self, log_every: int):
                super().__init__()
                self._log_every = int(log_every)
                self._last_log_step = 0
                self._t_start = time.time()
                self._ep_rewards: list[float] = []
                self._ep_lengths: list[float] = []
                self._ep_success: list[float] = []
                self._ep_stable_steps_final: list[float] = []
                self._ep_stable_steps_max: list[float] = []
                self._ep_quiet_frac: list[float] = []
                self._ep_observed_speed_mean: list[float] = []
                self._ep_safe_margin_mean: list[float] = []
                self._ep_checkpoint_dist_final: list[float] = []
                self._ep_checkpoint_dist_min: list[float] = []
                self._ep_inside_checkpoint_frac: list[float] = []
                self._termination_counts = {
                    "hole": 0, "timeout": 0, "survived": 0, "other": 0,
                }

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                for info in (i for i in infos if "episode" in i):
                    ep_len = float(info["episode"]["l"])
                    ep_len_safe = max(ep_len, 1.0)
                    self._ep_rewards.append(float(info["episode"]["r"]))
                    self._ep_lengths.append(ep_len)
                    self._ep_success.append(float(info.get("success", 0.0)))
                    self._ep_stable_steps_final.append(float(info.get("stable_steps_final", info.get("stable_steps", 0.0))))
                    self._ep_stable_steps_max.append(float(info.get("stable_steps_max", info.get("stable_steps", 0.0))))
                    self._ep_quiet_frac.append(float(info.get("quiet_frac", float(info.get("ep_quiet_steps", 0.0)) / ep_len_safe)))
                    self._ep_observed_speed_mean.append(float(info.get("mean_observed_speed", float(info.get("ep_ball_speed_sum", 0.0)) / ep_len_safe)))
                    self._ep_safe_margin_mean.append(
                        float(info.get("ep_safe_hole_margin_sum", 0.0)) / ep_len_safe
                    )
                    self._ep_checkpoint_dist_final.append(float(info.get("checkpoint_dist_final", np.nan)))
                    self._ep_checkpoint_dist_min.append(float(info.get("checkpoint_dist_min", np.nan)))
                    self._ep_inside_checkpoint_frac.append(float(info.get("inside_checkpoint_frac", 0.0)))
                    reason = str(info.get("termination_reason", "other"))
                    if reason not in self._termination_counts:
                        reason = "other"
                    self._termination_counts[reason] += 1

                if self.num_timesteps - self._last_log_step < self._log_every:
                    return True

                payload = {"train/step": self.num_timesteps}
                if self._ep_rewards:
                    n_eps = len(self._ep_rewards)
                    mean_reward = float(np.mean(self._ep_rewards))
                    mean_length = float(np.mean(self._ep_lengths))
                    success_rate = float(np.mean(self._ep_success))
                    length_frac = mean_length / self.EP_LEN_CAP
                    reward_per_step = mean_reward / max(mean_length, 1.0)
                    mean_stable_final = float(np.mean(self._ep_stable_steps_final))
                    mean_stable_max = float(np.mean(self._ep_stable_steps_max))
                    quiet_frac = float(np.mean(self._ep_quiet_frac))
                    mean_observed_speed = float(np.mean(self._ep_observed_speed_mean))
                    mean_safe_margin = float(np.mean(self._ep_safe_margin_mean))
                    checkpoint_dist_final = float(np.nanmean(self._ep_checkpoint_dist_final))
                    checkpoint_dist_min = float(np.nanmean(self._ep_checkpoint_dist_min))
                    inside_checkpoint_frac = float(np.mean(self._ep_inside_checkpoint_frac))
                    payload.update({
                        # Core episode metrics — same keys on GPU.
                        "episode/mean_reward": mean_reward,
                        "episode/mean_length": mean_length,
                        "episode/length_frac": length_frac,
                        "episode/success_rate": success_rate,
                        "episode/stable_steps_final": mean_stable_final,
                        "episode/stable_steps_max": mean_stable_max,
                        "episode/mean_stable_steps": mean_stable_max,
                        "episode/quiet_frac": quiet_frac,
                        "episode/mean_observed_speed": mean_observed_speed,
                        "episode/mean_ball_speed": mean_observed_speed,
                        "episode/mean_safe_hole_margin": mean_safe_margin,
                        "episode/checkpoint_dist_final": checkpoint_dist_final,
                        "episode/checkpoint_dist_min": checkpoint_dist_min,
                        "episode/inside_checkpoint_frac": inside_checkpoint_frac,
                        "episode/reward_per_step": reward_per_step,
                        "episode/deployment_score": length_frac * success_rate,
                        # Rollout aliases (kept for SB3-style panels).
                        "rollout/ep_rew_mean": mean_reward,
                        "rollout/ep_len_mean": mean_length,
                        "rollout/episodes": n_eps,
                    })
                    timeout_count = (
                        self._termination_counts.get("timeout", 0)
                        + self._termination_counts.get("survived", 0)
                    )
                    self._termination_counts["timeout"] = timeout_count
                    for reason, count in self._termination_counts.items():
                        if reason == "survived":
                            continue
                        payload[f"episode/termination_{reason}_rate"] = float(
                            count / max(n_eps, 1)
                        )
                    print(
                        f"[step {self.num_timesteps}] "
                        f"len={mean_length:.1f}/{int(self.EP_LEN_CAP)} "
                        f"({length_frac:.0%}) "
                        f"success_rate={success_rate:.3f} "
                        f"quiet_frac={quiet_frac:.3f} "
                        f"stable_max={mean_stable_max:.1f} "
                        f"reward/step={reward_per_step:.3f} "
                        f"eps={n_eps}",
                        flush=True,
                    )
                    self._ep_rewards.clear()
                    self._ep_lengths.clear()
                    self._ep_success.clear()
                    self._ep_stable_steps_final.clear()
                    self._ep_stable_steps_max.clear()
                    self._ep_quiet_frac.clear()
                    self._ep_observed_speed_mean.clear()
                    self._ep_safe_margin_mean.clear()
                    self._ep_checkpoint_dist_final.clear()
                    self._ep_checkpoint_dist_min.clear()
                    self._ep_inside_checkpoint_frac.clear()
                    for key in self._termination_counts:
                        self._termination_counts[key] = 0

                elapsed = time.time() - self._t_start
                payload["perf/walltime_s"] = elapsed
                payload["perf/sps_overall"] = self.num_timesteps / max(elapsed, 1e-6)

                wandb.log(payload, step=self.num_timesteps)
                self._last_log_step = self.num_timesteps
                return True

        class SyncNormCallback(BaseCallback):
            """Keep eval VecNormalize stats in sync with training env."""
            def _on_step(self) -> bool:
                sync_envs_normalization(vec_env, eval_env)
                return True

        class WandbVideoCallback(BaseCallback):
            def __init__(self, video_freq: int = 50_000):
                super().__init__()
                self._video_freq = video_freq
                self._last_video_step = -video_freq

            def _normalize(self, stacked_obs: np.ndarray) -> np.ndarray:
                normed = (stacked_obs - vec_env.obs_rms.mean) / np.sqrt(vec_env.obs_rms.var + vec_env.epsilon)
                return np.clip(normed, -vec_env.clip_obs, vec_env.clip_obs)

            def _on_step(self) -> bool:
                if self.num_timesteps - self._last_video_step >= self._video_freq:
                    self._last_video_step = self.num_timesteps
                    from collections import deque
                    env = make_prior_env(
                        render_mode="rgb_array",
                        progress_scale=args.progress_scale,
                        include_corridors=args.checkpoint_mode == "corners_and_corridors",
                        prior_task=args.prior_task,
                        prior_spawn_source=args.prior_spawn_source,
                        prior_version=args.prior_version,
                    )
                    obs_dim = env.observation_space.shape[0]
                    buf = deque([np.zeros(obs_dim, dtype=np.float32)] * args.n_stack, maxlen=args.n_stack)
                    raw_obs, _ = env.reset(seed=self.num_timesteps)
                    frames, done = [], False
                    while not done:
                        buf.append(raw_obs)
                        obs_in = self._normalize(np.concatenate(list(buf)))
                        action, _ = self.model.predict(obs_in, deterministic=True)
                        raw_obs, _, terminated, truncated, _ = env.step(action)
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                        done = terminated or truncated
                    env.close()
                    if frames:
                        video = np.stack(frames).transpose(0, 3, 1, 2)
                        # Save to disk with a zero-padded step in the filename
                        # so the W&B media browser sorts videos chronologically.
                        # Width 10 ⇒ supports up to 10B steps without re-padding.
                        videos_dir = logdir / "wandb_videos"
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_path = videos_dir / (
                            f"eval_step_{self.num_timesteps:010d}.mp4"
                        )
                        # Encode via wandb.Video by writing the frames to disk
                        # using imageio (already a wandb video dep).
                        import imageio
                        imageio.mimwrite(
                            str(video_path),
                            np.stack(frames),
                            fps=30,
                            macro_block_size=None,
                        )
                        # Write to run.summary instead of run.history so the
                        # dashboard panel shows ONE video that refreshes,
                        # rather than appending a clip every video_freq steps.
                        # The on-disk file keeps the step in its name so the
                        # full archive is sortable in the media tab.
                        wandb.run.summary["eval/video"] = wandb.Video(
                            str(video_path), fps=30, format="mp4",
                        )
                        wandb.run.summary["eval/video_step"] = self.num_timesteps
                return True

        class WandbRobustEvalCallback(BaseCallback):
            def __init__(self, eval_every: int, repeats: int, max_spawns: int):
                super().__init__()
                self._eval_every = int(eval_every)
                self._repeats = int(repeats)
                self._max_spawns = int(max_spawns)
                self._last_eval_step = -eval_every
                self._eval_env = make_prior_env(
                    render_mode=None,
                    progress_scale=args.progress_scale,
                    include_corridors=args.checkpoint_mode == "corners_and_corridors",
                    prior_task=args.prior_task,
                    prior_spawn_source=args.prior_spawn_source,
                    prior_version=args.prior_version,
                )
                raw_spawns = np.asarray(self._eval_env.unwrapped.prior_start_points, dtype=np.float32)
                if len(raw_spawns) > self._max_spawns > 0:
                    idxs = np.linspace(0, len(raw_spawns) - 1, self._max_spawns, dtype=int)
                    self._spawn_bank = raw_spawns[idxs]
                else:
                    self._spawn_bank = raw_spawns

            def _normalize(self, stacked_obs: np.ndarray) -> np.ndarray:
                normed = (stacked_obs - vec_env.obs_rms.mean) / np.sqrt(vec_env.obs_rms.var + vec_env.epsilon)
                return np.clip(normed, -vec_env.clip_obs, vec_env.clip_obs)

            def _run_episode(self, spawn_point: np.ndarray, seed: int) -> dict[str, float]:
                obs_dim = self._eval_env.observation_space.shape[0]
                buf = deque([np.zeros(obs_dim, dtype=np.float32)] * args.n_stack, maxlen=args.n_stack)
                raw_obs, _ = self._eval_env.reset(seed=seed, options={"spawn_point": spawn_point})
                total_reward = 0.0
                done = False
                final_info = {}
                while not done:
                    buf.append(raw_obs)
                    obs_in = self._normalize(np.concatenate(list(buf)))
                    action, _ = self.model.predict(obs_in, deterministic=True)
                    raw_obs, reward, terminated, truncated, info = self._eval_env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated
                    final_info = info
                return {
                    "success": float(final_info.get("success", 0.0)),
                    "stable_steps": float(final_info.get("stable_steps_max", final_info.get("stable_steps", 0.0))),
                    "checkpoint_dist": float(final_info.get("checkpoint_dist_final", final_info.get("checkpoint_dist", np.nan))),
                    "ball_speed": float(final_info.get("mean_observed_speed", final_info.get("ball_speed", np.nan))),
                    "safe_hole_margin": float(final_info.get("safe_hole_margin", np.nan)),
                    "inside_checkpoint_frac": float(final_info.get("inside_checkpoint_frac", 0.0)),
                    "reward": total_reward,
                }

            def _on_step(self) -> bool:
                if self.num_timesteps - self._last_eval_step < self._eval_every:
                    return True
                self._last_eval_step = self.num_timesteps

                per_spawn_success = []
                per_spawn_reward = []
                per_spawn_dist = []
                per_spawn_stable = []
                per_spawn_speed = []
                per_spawn_margin = []
                for spawn_idx, spawn_point in enumerate(self._spawn_bank):
                    spawn_results = [
                        self._run_episode(
                            spawn_point=spawn_point,
                            seed=int(self.num_timesteps + 1000 * spawn_idx + rep),
                        )
                        for rep in range(self._repeats)
                    ]
                    per_spawn_success.append(float(np.mean([r["success"] for r in spawn_results])))
                    per_spawn_reward.append(float(np.mean([r["reward"] for r in spawn_results])))
                    per_spawn_dist.append(float(np.mean([r["checkpoint_dist"] for r in spawn_results])))
                    per_spawn_stable.append(float(np.mean([r["stable_steps"] for r in spawn_results])))
                    per_spawn_speed.append(float(np.mean([r["ball_speed"] for r in spawn_results])))
                    per_spawn_margin.append(float(np.mean([r["safe_hole_margin"] for r in spawn_results])))

                success_arr = np.asarray(per_spawn_success, dtype=np.float32)
                reward_arr = np.asarray(per_spawn_reward, dtype=np.float32)
                dist_arr = np.asarray(per_spawn_dist, dtype=np.float32)
                stable_arr = np.asarray(per_spawn_stable, dtype=np.float32)
                speed_arr = np.asarray(per_spawn_speed, dtype=np.float32)
                margin_arr = np.asarray(per_spawn_margin, dtype=np.float32)
                payload = {
                    "train/step": self.num_timesteps,
                    "eval/spawn_success_rate_mean": float(success_arr.mean()),
                    "eval/spawn_success_rate_median": float(np.median(success_arr)),
                    "eval/spawn_success_rate_p10": float(np.percentile(success_arr, 10)),
                    "eval/spawn_success_rate_min": float(success_arr.min()),
                    "eval/spawn_reward_mean": float(reward_arr.mean()),
                    "eval/spawn_checkpoint_dist_mean": float(dist_arr.mean()),
                    "eval/spawn_stable_steps_mean": float(stable_arr.mean()),
                    "eval/spawn_ball_speed_mean": float(speed_arr.mean()),
                    "eval/spawn_safe_hole_margin_mean": float(margin_arr.mean()),
                    "eval/spawn_count": int(len(success_arr)),
                    "eval/episodes_per_spawn": int(self._repeats),
                }
                print(
                    f"[step {self.num_timesteps}] "
                    f"spawn_success_mean={payload['eval/spawn_success_rate_mean']:.3f} "
                    f"spawn_success_p10={payload['eval/spawn_success_rate_p10']:.3f} "
                    f"spawn_success_min={payload['eval/spawn_success_rate_min']:.3f} "
                    f"spawn_reward_mean={payload['eval/spawn_reward_mean']:.3f} "
                    f"spawn_count={payload['eval/spawn_count']}",
                    flush=True,
                )
                wandb.log(payload, step=self.num_timesteps)
                return True

            def _on_training_end(self) -> None:
                self._eval_env.close()

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=vars(args),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("episode/*", step_metric="train/step")
        wandb.define_metric("rollout/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="train/step")
        extra_callbacks = [
            WandbCallback(args.wandb_log_every),
            SyncNormCallback(),
            WandbVideoCallback(video_freq=50_000),
            WandbRobustEvalCallback(
                eval_every=args.robust_eval_every,
                repeats=args.robust_eval_repeats,
                max_spawns=args.robust_eval_max_spawns,
            ),
        ]
    else:
        extra_callbacks = []

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,
        learning_rate=1e-4,
        verbose=0,
        tensorboard_log=None,
        seed=args.seed,
    )

    callbacks = [
        ProgressBarCallback(args.steps),
        CheckpointCallback(save_freq=50_000, save_path=str(logdir), name_prefix="ppo"),
        EvalCallback(
            eval_env,
            best_model_save_path=str(logdir),
            log_path=str(logdir),
            eval_freq=10_000 // args.n_envs,
            n_eval_episodes=10,
            deterministic=True,
        ),
        *extra_callbacks,
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(str(logdir / "final_model"))
    vec_env.save(str(logdir / "vecnormalize.pkl"))
    print(f"Saved final model to {logdir}/final_model.zip")
    print(f"Saved normalization stats to {logdir}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
