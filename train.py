from __future__ import annotations

import json
import os
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.cyberrunner import CyberRunnerEnv


class WandbCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000, verbose=0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[float] = []
        self._log_interval = log_interval
        self._last_log_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(float(info["episode"]["r"]))
                self._ep_lengths.append(float(info["episode"]["l"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_log_step < self._log_interval:
            return
        self._last_log_step = self.num_timesteps

        log = {}
        if len(self._ep_rewards) >= 10:
            log["train/ep_rew_mean"] = np.mean(self._ep_rewards[-100:])
            log["train/ep_len_mean"] = np.mean(self._ep_lengths[-100:])
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            for k, v in self.model.logger.name_to_value.items():
                if v is not None:
                    log[k] = v
        if log:
            wandb.log(log, step=self.num_timesteps)


def make_env(cfg):
    def _init():
        return FlattenObservation(CyberRunnerEnv(
            include_vision=False,
            reward_every_n_waypoints=cfg.env.reward_every_n_waypoints,
            hole_penalty=cfg.env.hole_penalty,
            episode_length=cfg.env.episode_length,
            randomize_init_pos=cfg.env.randomize_init_pos,
            backup_mode=cfg.env.backup_mode,
            recovery_speed_threshold=cfg.env.recovery_speed_threshold,
            recovery_tilt_threshold=cfg.env.recovery_tilt_threshold,
            recovery_hole_margin_factor=cfg.env.recovery_hole_margin_factor,
            backup_init_max_speed=cfg.env.get("backup_init_max_speed", 0.2),
        ))
    return _init


def save_artifact(run: wandb.Run, artifact_name: str, file_paths: list[str]) -> None:
    artifact = wandb.Artifact(name=artifact_name, type="model")
    for path in file_paths:
        if os.path.exists(path):
            artifact.add_file(path)
    run.log_artifact(artifact)


def eval_and_log_video(
    run: wandb.Run,
    model,
    vecnorm_path: str,
    env_cfg_path: str,
    n_episodes: int = 3,
    fps: int = 20,
) -> None:
    with open(env_cfg_path) as f:
        ec = json.load(f)

    backup_mode = ec.get("backup_mode", False)
    eval_env = VecNormalize.load(
        vecnorm_path,
        DummyVecEnv([lambda: FlattenObservation(CyberRunnerEnv(
            render_mode="rgb_array",
            include_vision=False,
            episode_length=ec["episode_length"],
            randomize_init_pos=True,
            backup_mode=backup_mode,
            recovery_speed_threshold=ec.get("recovery_speed_threshold", 0.03),
            recovery_tilt_threshold=ec.get("recovery_tilt_threshold", 0.02),
            recovery_hole_margin_factor=ec.get("recovery_hole_margin_factor", 3.0),
            backup_init_max_speed=ec.get("backup_init_max_speed", 0.2),
            reward_every_n_waypoints=ec.get("reward_every_n_waypoints", 3),
            hole_penalty=ec.get("hole_penalty", 5.0),
        ))]),
    )
    eval_env.training = False
    eval_env.norm_reward = False

    all_frames: list[np.ndarray] = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        while True:
            frame = eval_env.venv.envs[0].unwrapped.render()
            if frame is not None:
                all_frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            if done[0]:
                break
    eval_env.close()

    # wandb.Video expects (T, C, H, W)
    frames = np.stack(all_frames).transpose(0, 3, 1, 2)
    run.log({"eval/video": wandb.Video(frames, fps=fps, format="mp4")})


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wb_cfg = cfg.get("wandb", {})
    run = None
    if wb_cfg.get("enabled", False):
        run = wandb.init(
            project=wb_cfg.get("project", "cyberrunner"),
            name=wb_cfg.get("name", None) or None,
            tags=list(wb_cfg.get("tags", [])),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    env = VecNormalize(
        make_vec_env(make_env(cfg), n_envs=cfg.algo.n_envs),
        norm_obs=True,
        norm_reward=not cfg.env.get("backup_mode", False),
        gamma=cfg.algo.gamma,
    )

    algo = cfg.algo.name.lower()
    if algo == "ppo":
        model = PPO(
            "MlpPolicy", env, verbose=1, device=cfg.device,
            learning_rate=cfg.algo.learning_rate,
            n_steps=cfg.algo.n_steps,
            batch_size=cfg.algo.batch_size,
            n_epochs=cfg.algo.n_epochs,
            gamma=cfg.algo.gamma,
            gae_lambda=cfg.algo.gae_lambda,
            clip_range=cfg.algo.clip_range,
            ent_coef=cfg.algo.ent_coef,
        )
    elif algo == "sac":
        model = SAC(
            "MlpPolicy", env, verbose=1, device=cfg.device,
            learning_rate=cfg.algo.learning_rate,
            buffer_size=cfg.algo.buffer_size,
            batch_size=cfg.algo.batch_size,
            tau=cfg.algo.tau,
            gamma=cfg.algo.gamma,
            learning_starts=cfg.algo.learning_starts,
            gradient_steps=cfg.algo.get("gradient_steps", 1) * cfg.algo.n_envs,
            ent_coef=cfg.algo.ent_coef,
        )
    elif algo == "mbpo":
        from mbpo import MBPOTrainer
        trainer = MBPOTrainer(env, cfg, device=cfg.device)
        trainer.learn(cfg.total_timesteps, wandb_run=run)
        trainer.save("mbpo_cyberrunner")
        env.save("mbpo_cyberrunner_vecnormalize.pkl")
        with open("mbpo_cyberrunner_env_cfg.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg.env, resolve=True), f, indent=2)
        if run is not None:
            save_artifact(run, "mbpo_cyberrunner", [
                "mbpo_cyberrunner_policy.zip",
                "mbpo_cyberrunner_dynamics.pt",
                "mbpo_cyberrunner_vecnormalize.pkl",
                "mbpo_cyberrunner_env_cfg.json",
            ])
            eval_and_log_video(run, trainer.sac,
                               "mbpo_cyberrunner_vecnormalize.pkl",
                               "mbpo_cyberrunner_env_cfg.json")
            run.finish()
        return
    else:
        raise ValueError(f"Unknown algo: {algo}")

    log_interval = wb_cfg.get("log_interval", 10_000)
    callbacks = [WandbCallback(log_interval=log_interval)] if run is not None else []
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True, callback=callbacks)

    suffix = "backup" if cfg.env.get("backup_mode", False) else "cyberrunner"
    model.save(f"{algo}_{suffix}")
    env.save(f"{algo}_{suffix}_vecnormalize.pkl")
    with open(f"{algo}_{suffix}_env_cfg.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg.env, resolve=True), f, indent=2)

    if run is not None:
        save_artifact(run, f"{algo}_{suffix}", [
            f"{algo}_{suffix}.zip",
            f"{algo}_{suffix}_vecnormalize.pkl",
            f"{algo}_{suffix}_env_cfg.json",
        ])
        eval_and_log_video(run, model,
                           f"{algo}_{suffix}_vecnormalize.pkl",
                           f"{algo}_{suffix}_env_cfg.json")
        run.finish()


if __name__ == "__main__":
    main()
