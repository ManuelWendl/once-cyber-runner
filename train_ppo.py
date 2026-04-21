"""PPO training for the CyberRunner prior (stabilization) task using stable-baselines3."""
import argparse
import pathlib
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
):
    from envs.cyberrunner import CyberRunnerEnv

    env = CyberRunnerEnv(
        render_mode=render_mode,
        episode_length=500,
        randomize_init_pos=False,
        include_vision=False,
        reward_every_n_waypoints=3,
        hole_penalty=0.0,
        checkpoint_radius=0.015,
        checkpoint_hold_steps=6,
        checkpoint_speed_threshold=0.05,
        checkpoint_arrival_reward=0.0,
        checkpoint_stabilize_reward=3.0,
        checkpoint_hold_reward=1.5,
        safe_hole_margin=0.004,
        checkpoint_speed_ema_alpha=0.8,
        checkpoint_include_corridors=include_corridors,
        prior_mode=True,
        prior_start_waypoint_window=3,
        prior_init_ball_speed=0.15,
        prior_init_tilt_frac=0.2,
        prior_min_checkpoint_start_dist=0.02,
        prior_max_checkpoint_start_dist=0.12,
        prior_spawn_min_hole_margin=0.012,
        prior_start_point_spacing=0.01,
        prior_spawn_merge_radius=0.02,
        checkpoint_progress_reward_scale=progress_scale,
        terminate_on_checkpoint_stabilized=True,
    )
    return FlattenObsWrapper(env, seed=seed)


class FlattenObsWrapper(gym.Wrapper):
    """Flatten Dict obs {states(10), checkpoint(3)} → Box(13,)."""

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.env.reset(seed=seed)

    def _flatten(self, obs: dict) -> np.ndarray:
        return np.concatenate([obs["states"], obs["checkpoint"]], axis=-1)

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
    parser.add_argument("--n_stack", type=int, default=4)
    parser.add_argument("--progress_scale", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="corners_and_corridors",
        choices=["corners", "corners_and_corridors"],
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
    }

    vec_env = VecNormalize(
        VecFrameStack(
            VecMonitor(make_vec_env(make_prior_env, n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs)),
            n_stack=args.n_stack,
        ),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )

    eval_env = VecNormalize(
        VecFrameStack(
            VecMonitor(make_vec_env(make_prior_env, n_envs=4, seed=args.seed + 1000, env_kwargs=env_kwargs)),
            n_stack=args.n_stack,
        ),
        norm_obs=True, norm_reward=False, training=False,
    )

    if args.wandb_project:
        import wandb

        class WandbCallback(BaseCallback):
            def __init__(self, log_every: int):
                super().__init__()
                self._log_every = int(log_every)
                self._last_log_step = 0
                self._ep_rewards = []
                self._ep_lengths = []
                self._ep_success = []
                self._ep_stable_steps = []

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                ep_infos = [i for i in infos if "episode" in i]
                for info in ep_infos:
                    self._ep_rewards.append(float(info["episode"]["r"]))
                    self._ep_lengths.append(float(info["episode"]["l"]))
                    self._ep_success.append(float(info.get("success", 0.0)))
                    self._ep_stable_steps.append(float(info.get("stable_steps", 0.0)))

                if self.num_timesteps - self._last_log_step >= self._log_every:
                    payload = {"train/step": self.num_timesteps}
                    if self._ep_rewards:
                        mean_reward = float(np.mean(self._ep_rewards))
                        mean_length = float(np.mean(self._ep_lengths))
                        mean_success = float(np.mean(self._ep_success))
                        mean_stable = float(np.mean(self._ep_stable_steps))
                        payload.update({
                            "episode/mean_reward": mean_reward,
                            "episode/mean_length": mean_length,
                            "episode/success_rate": mean_success,
                            "episode/mean_stable_steps": mean_stable,
                            "rollout/ep_rew_mean": mean_reward,
                            "rollout/ep_len_mean": mean_length,
                            "rollout/episodes": len(self._ep_rewards),
                        })
                        print(
                            f"[step {self.num_timesteps}] "
                            f"ep_rew_mean={mean_reward:.3f} "
                            f"ep_len_mean={mean_length:.1f} "
                            f"success_rate={mean_success:.3f} "
                            f"stable_steps={mean_stable:.2f} "
                            f"episodes={len(self._ep_rewards)}",
                            flush=True,
                        )
                        self._ep_rewards.clear()
                        self._ep_lengths.clear()
                        self._ep_success.clear()
                        self._ep_stable_steps.clear()
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
                    )
                    buf = deque([np.zeros(13, dtype=np.float32)] * args.n_stack, maxlen=args.n_stack)
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
                        wandb.log({
                            "eval/video": wandb.Video(video, fps=30, format="mp4"),
                            "train/step": self.num_timesteps,
                        }, step=self.num_timesteps)
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
                buf = deque([np.zeros(13, dtype=np.float32)] * args.n_stack, maxlen=args.n_stack)
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
                    "stable_steps": float(final_info.get("stable_steps", 0.0)),
                    "checkpoint_dist": float(final_info.get("checkpoint_dist", np.nan)),
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

                success_arr = np.asarray(per_spawn_success, dtype=np.float32)
                reward_arr = np.asarray(per_spawn_reward, dtype=np.float32)
                dist_arr = np.asarray(per_spawn_dist, dtype=np.float32)
                stable_arr = np.asarray(per_spawn_stable, dtype=np.float32)
                payload = {
                    "train/step": self.num_timesteps,
                    "eval/spawn_success_rate_mean": float(success_arr.mean()),
                    "eval/spawn_success_rate_median": float(np.median(success_arr)),
                    "eval/spawn_success_rate_p10": float(np.percentile(success_arr, 10)),
                    "eval/spawn_success_rate_min": float(success_arr.min()),
                    "eval/spawn_reward_mean": float(reward_arr.mean()),
                    "eval/spawn_checkpoint_dist_mean": float(dist_arr.mean()),
                    "eval/spawn_stable_steps_mean": float(stable_arr.mean()),
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
        ent_coef=0.01,
        learning_rate=3e-4,
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
