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
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from cyberrunner_env import CyberRunnerEnv


def make_env(start_from_beginning_prob: float = 0.0):
    return lambda: CyberRunnerEnv(
        render_mode=None,
        randomize_init_pos=True,
        start_from_beginning_prob=start_from_beginning_prob,
    )


class CurriculumCallback(BaseCallback):
    """Linearly decay start_from_beginning_prob during training."""

    def __init__(self, initial_prob: float, final_prob: float, verbose: int = 0):
        super().__init__(verbose)
        self.initial_prob = initial_prob
        self.final_prob = final_prob

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.locals["total_timesteps"]
        current_prob = self.initial_prob + (self.final_prob - self.initial_prob) * progress

        # Update all training envs
        for env in self.training_env.envs:
            env.start_from_beginning_prob = current_prob

        if self.verbose > 0 and self.num_timesteps % 50_000 < self.locals.get("n_steps", 2048):
            print(f"  [curriculum] step {self.num_timesteps}: start_prob={current_prob:.3f}")

        return True


ALGO_DEFAULTS = {
    "ppo": dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=16,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.001,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{algo}_{timestamp}"

    model_dir = f"./models/{run_name}"
    log_dir = f"./logs/{run_name}"
    vecnorm_path = f"{model_dir}/vecnormalize.pkl"
    final_path = f"{model_dir}/final"
    best_path = f"{model_dir}/best"

    # SAC is off-policy: multiple envs give limited benefit, 1 is standard
    n_envs = args.n_envs if algo == "ppo" else 1
    n_stack = args.frame_stack

    train_env = make_vec_env(make_env(args.start_from_beginning_prob), n_envs=n_envs)
    eval_env = make_vec_env(make_env(0.0), n_envs=1)

    if args.resume:
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
        n_eval_episodes=5,
        deterministic=True,
    ))

    if args.start_from_beginning_prob > 0:
        callbacks.append(CurriculumCallback(
            initial_prob=args.start_from_beginning_prob,
            final_prob=args.final_start_prob,
            verbose=1,
        ))
        print(f"Curriculum: start_prob {args.start_from_beginning_prob} → {args.final_start_prob}")

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
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name. Auto-generated as <algo>_<timestamp> if not provided.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel envs (PPO only; SAC always uses 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from ./models/<run-name>/final.zip")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of frames to stack (>1 gives velocity info)")
    parser.add_argument("--start-from-beginning-prob", type=float, default=0.0,
                        help="Initial probability of starting at waypoint 0")
    parser.add_argument("--final-start-prob", type=float, default=0.0,
                        help="Final probability of starting at waypoint 0 (decays linearly)")
    args = parser.parse_args()
    main(args)
