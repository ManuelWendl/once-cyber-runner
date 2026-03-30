"""
Train PPO or SAC on CyberRunnerEnv.

Usage:
    python train.py --algo ppo
    python train.py --algo sac
    python train.py --algo ppo --timesteps 5000000 --n-envs 8
    python train.py --algo sac --timesteps 2000000
"""
import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from cyberrunner_env import CyberRunnerEnv


def make_env():
    return CyberRunnerEnv(render_mode=None, randomize_init_pos=True)


ALGO_DEFAULTS = {
    "ppo": dict(
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=16,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.25,
        ent_coef=0.01,
    ),
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=10_000,
        ent_coef="auto",
    ),
}

ALGO_CLS = {"ppo": PPO, "sac": SAC}


def main(args):
    algo = args.algo.lower()
    algo_cls = ALGO_CLS[algo]

    # SAC is off-policy: multiple envs give limited benefit, 1 is standard
    n_envs = args.n_envs if algo == "ppo" else 1

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = algo_cls(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=f"./logs/{algo}",
        **ALGO_DEFAULTS[algo],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{algo}_best",
        log_path=f"./logs/{algo}_eval",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(f"./models/{algo}_final")
    print(f"Saved final model to ./models/{algo}_final.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel envs (PPO only; SAC always uses 1)")
    args = parser.parse_args()
    main(args)
