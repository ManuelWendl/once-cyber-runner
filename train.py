"""
Train PPO or SAC on CyberRunnerEnv.

Usage:
    python train.py --algo ppo
    python train.py --algo ppo --run-name ppo_3stack --frame-stack 3
    python train.py --algo ppo --run-name ppo_curriculum --frame-stack 3 \
        --start-from-beginning-prob 0.7 --final-start-prob 0.2
    python train.py --algo ppo --run-name ppo_3stack --resume --frame-stack 3
"""
import argparse
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from maxinfosac_compat import MaxInfoSAC
from cyberrunner_env import CyberRunnerEnv


def make_env(start_from_beginning_prob: float = 0.0):
    return lambda: CyberRunnerEnv(
        render_mode=None,
        randomize_init_pos=True,
        start_from_beginning_prob=start_from_beginning_prob,
    )


class CurriculumCallback(BaseCallback):
    """Decay start_from_beginning_prob to 0 over a warmup fraction of training."""

    def __init__(self, initial_prob: float, warmup_frac: float = 0.3, verbose: int = 0):
        super().__init__(verbose)
        self.initial_prob = initial_prob
        self.warmup_frac = warmup_frac
        self._done = False

    def _on_step(self) -> bool:
        if self._done or self.num_timesteps % 10_000 != 0:
            return True

        total = self.locals["total_timesteps"]
        warmup_end = total * self.warmup_frac

        if self.num_timesteps >= warmup_end:
            current_prob = 0.0
            self._done = True
        else:
            current_prob = self.initial_prob * (1.0 - self.num_timesteps / warmup_end)

        for env in self.training_env.envs:
            env.start_from_beginning_prob = current_prob

        if self.verbose > 0 and self._done:
            print(f"  [curriculum] warmup done at step {self.num_timesteps}, back to uniform")

        return True


ALGO_DEFAULTS = {
    # Pre-tuning defaults:
    # learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=16,
    # gamma=0.99, gae_lambda=0.98, clip_range=0.2, ent_coef=0.001
    "ppo": dict(
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=17,
        gamma=0.997,
        gae_lambda=0.974,
        clip_range=0.21,
        ent_coef=0.0037,
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
    "maxinfosac": dict(
        learning_rate=1e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=10_000,
        ent_coef="auto",
        ensemble_model_kwargs=dict(
            learn_std=False,
            features=(256, 256),
            optimizer_kwargs={"lr": 3e-4, "weight_decay": 0.0},
        ),
        normalize_ensemble_training=True,
        pred_diff=True,
        learn_rewards=True,
        dyn_entropy_scale="auto",
    ),
}

ALGO_CLS = {"ppo": PPO, "sac": SAC, "maxinfosac": MaxInfoSAC}


def main(args):
    algo = args.algo.lower()
    algo_cls = ALGO_CLS[algo]
    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{algo}_{timestamp}"

    model_dir = f"./models/{run_name}"
    log_dir = f"./logs/{run_name}"
    vecnorm_path = f"{model_dir}/vecnormalize.pkl"
    final_path = f"{model_dir}/final"
    best_path = f"{model_dir}/best"

    n_envs = args.n_envs
    n_stack = args.frame_stack
    use_pre_norm_stack = algo != "ppo" and n_stack > 1

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        make_env(args.start_from_beginning_prob),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )
    eval_env = make_vec_env(make_env(0.0), n_envs=1)
    if algo != "ppo" and n_envs > 1:
        print(
            f"Using n_envs={n_envs} for {algo.upper()}. This is valid, but speedups "
            "depend on CPU/IPC overhead and environment step cost."
        )
    if use_pre_norm_stack:
        print(
            f"Using frame_stack={n_stack} for {algo.upper()} with VecFrameStack before "
            "VecNormalize (required for off-policy replay-buffer compatibility)."
        )

    if args.resume:
        if use_pre_norm_stack:
            train_env = VecFrameStack(train_env, n_stack=n_stack)
            eval_env = VecFrameStack(eval_env, n_stack=n_stack)
            train_env = VecNormalize.load(vecnorm_path, train_env)
            train_env.training = True
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            train_env = VecNormalize.load(vecnorm_path, train_env)
            train_env.training = True
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            if n_stack > 1:
                train_env = VecFrameStack(train_env, n_stack=n_stack)
                eval_env = VecFrameStack(eval_env, n_stack=n_stack)
        model = algo_cls.load(f"{final_path}.zip", env=train_env,
                              tensorboard_log=log_dir)
        print(f"Resumed run '{run_name}' from {final_path}.zip")
    else:
        if use_pre_norm_stack:
            train_env = VecFrameStack(train_env, n_stack=n_stack)
            eval_env = VecFrameStack(eval_env, n_stack=n_stack)
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        else:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
            if n_stack > 1:
                train_env = VecFrameStack(train_env, n_stack=n_stack)
                eval_env = VecFrameStack(eval_env, n_stack=n_stack)
        model = algo_cls(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=log_dir,
            **ALGO_DEFAULTS[algo],
        )
        print(f"Starting new run '{run_name}'")

    callbacks = []

    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=f"{log_dir}/eval",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    ))

    if args.start_from_beginning_prob > 0:
        callbacks.append(CurriculumCallback(
            initial_prob=args.start_from_beginning_prob,
            warmup_frac=args.warmup_frac,
            verbose=1,
        ))
        print(f"Curriculum: start_prob {args.start_from_beginning_prob} → 0.0 "
              f"over first {int(args.warmup_frac * 100)}% of training")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(final_path)
    train_env.save(vecnorm_path)
    print(f"Saved final model to {final_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "maxinfosac"])
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name. Auto-generated as <algo>_<timestamp> if not provided.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel envs for data collection")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from ./models/<run-name>/final.zip")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack (>1 supported for all algos)")
    parser.add_argument("--start-from-beginning-prob", type=float, default=0.0,
                        help="Initial probability of starting at waypoint 0")
    parser.add_argument("--warmup-frac", type=float, default=0.3,
                        help="Fraction of training over which curriculum decays to uniform (default: 0.3)")
    args = parser.parse_args()
    main(args)
