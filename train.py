import hydra
from omegaconf import DictConfig
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from envs.cyberrunner import CyberRunnerEnv


def make_env(cfg):
    def _init():
        return FlattenObservation(CyberRunnerEnv(
            include_vision=False,
            reward_every_n_waypoints=cfg.env.reward_every_n_waypoints,
            hole_penalty=cfg.env.hole_penalty,
            episode_length=cfg.env.episode_length,
        ))
    return _init


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env = VecNormalize(
        make_vec_env(make_env(cfg), n_envs=cfg.algo.n_envs),
        norm_obs=True, norm_reward=True,
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
            ent_coef=cfg.algo.ent_coef,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
    model.save(f"{algo}_cyberrunner")


if __name__ == "__main__":
    main()
