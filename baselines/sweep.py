"""
Optuna hyperparameter sweep for PPO or SAC on CyberRunnerEnv.

Each trial trains for a short budget and reports mean eval reward.
Optuna uses Bayesian optimization (TPE sampler) to pick the next trial.
Results persist in a SQLite DB and can be resumed at any time.

Usage:
    python sweep.py --algo ppo
    python sweep.py --algo sac
    python sweep.py --algo ppo --trials 50 --timesteps 200000 --n-envs 8
    python sweep.py --algo sac --trials 30 --timesteps 200000
    optuna-dashboard sqlite:///sweep_ppo.db   # live dashboard
    optuna-dashboard sqlite:///sweep_sac.db
"""
import argparse

import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from cyberrunner_env import CyberRunnerEnv
from maxinfosac_compat import MaxInfoSAC

ALGO_CLS = {"ppo": PPO, "sac": SAC, "maxinfosac": MaxInfoSAC}

NET_ARCHS = {
    "small":  dict(pi=[64, 64],        vf=[64, 64]),
    "medium": dict(pi=[256, 256],      vf=[256, 256]),
    "large":  dict(pi=[256, 256, 256], vf=[256, 256, 256]),
}


def make_env():
    return CyberRunnerEnv(render_mode=None, randomize_init_pos=True)


def sample_ppo_params(trial: optuna.Trial) -> dict:
    return {
        # Anchored near best known value
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 9.99e-4, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [2048, 4096]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs":      trial.suggest_int("n_epochs", 10, 20),
        # Widened: long maze needs higher gamma to value distant rewards
        "gamma":         trial.suggest_float("gamma", 0.97, 0.9999, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.95, 1.0),
        "clip_range":    trial.suggest_float("clip_range", 0.2, 0.35),
        # Widened: explore whether more entropy helps navigate past holes
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
        "net_arch":      trial.suggest_categorical("net_arch", ["small", "medium"]),
    }


def sample_sac_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "buffer_size":     trial.suggest_categorical("buffer_size", [100_000, 300_000, 1_000_000]),
        "batch_size":      trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau":             trial.suggest_float("tau", 0.001, 0.05, log=True),
        "gamma":           trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        "learning_starts": trial.suggest_categorical("learning_starts", [1_000, 5_000, 10_000]),
        # "auto" lets SAC tune entropy automatically; fixed value disables that
        "ent_coef":        trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.1]),
        "net_arch":        trial.suggest_categorical("net_arch", ["small", "medium", "large"]),
    }


def objective(trial: optuna.Trial, algo: str, timesteps: int, n_envs: int,
              frame_stack: int) -> float:
    use_pre_norm_stack = algo != "ppo" and frame_stack > 1
    if algo == "ppo":
        params = sample_ppo_params(trial)
        if params["n_steps"] < params["batch_size"]:
            raise optuna.TrialPruned()
        train_env = make_vec_env(make_env, n_envs=n_envs)
    else:
        params = sample_sac_params(trial)
        train_env = make_vec_env(make_env, n_envs=1)

    net_arch = NET_ARCHS[params.pop("net_arch")]
    eval_env = make_vec_env(make_env, n_envs=1)

    if use_pre_norm_stack:
        train_env = VecFrameStack(train_env, n_stack=frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        if frame_stack > 1:
            train_env = VecFrameStack(train_env, n_stack=frame_stack)
            eval_env = VecFrameStack(eval_env, n_stack=frame_stack)

    try:
        model = ALGO_CLS[algo](
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
            tensorboard_log=None,
            **params,
        )
        model.learn(total_timesteps=timesteps, progress_bar=True)

        # Sync normalization stats from train to eval
        if frame_stack > 1 and not use_pre_norm_stack:
            eval_env.venv.obs_rms = train_env.venv.obs_rms
            eval_env.venv.ret_rms = train_env.venv.ret_rms
        else:
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    finally:
        train_env.close()
        eval_env.close()

    return mean_reward


def main(args):
    algo = args.algo.lower()
    storage = f"sqlite:///sweep_{algo}.db"

    study = optuna.create_study(
        study_name=f"{algo}_cyberrunner",
        storage=storage,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, algo, args.timesteps, args.n_envs, args.frame_stack),
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    print(f"\n=== Best {algo.upper()} trial ===")
    best = study.best_trial
    print(f"  Mean reward: {best.value:.3f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Training budget per trial")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel envs (PPO only; SAC always uses 1)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel Optuna trials (n_envs × n_jobs <= n_cores)")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack (>1 supported for all algos)")
    args = parser.parse_args()
    main(args)
