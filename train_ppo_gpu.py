"""GPU-native PPO training for the CyberRunner stabilize prior via Brax.

Optional, additive. Does not replace train_ppo.py (SB3/CPU path).

Uses `brax.training.agents.ppo.train` with the MJX env `CyberRunnerMJXEnv`.
Logs to wandb and periodically pickles flax params to disk.
"""
from __future__ import annotations

import argparse
import functools
import os
import pathlib
import pickle
import time

import jax

from envs.cyberrunner_mjx import CyberRunnerMJXEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logdir/ppo_gpu")
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--num_updates_per_batch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--episode_length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_ball_speed", type=float, default=0.05)
    parser.add_argument("--init_tilt_frac", type=float, default=0.05)
    parser.add_argument("--num_evals", type=int, default=20)
    parser.add_argument("--save_every_evals", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"[gpu-train] jax devices: {jax.devices()}")
    print(f"[gpu-train] default backend: {jax.default_backend()}")

    env = CyberRunnerMJXEnv(
        episode_length=args.episode_length,
        init_ball_speed=args.init_ball_speed,
        init_tilt_frac=args.init_tilt_frac,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=vars(args),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("eval/*", step_metric="train/step")
        wandb.define_metric("training/*", step_metric="train/step")
        (logdir / "wandb_run_id.txt").write_text(str(wandb_run.id))

    eval_counter = {"n": 0}

    def progress_fn(num_steps: int, metrics: dict) -> None:
        eval_counter["n"] += 1
        flat = {k: float(v) for k, v in metrics.items() if _is_scalar(v)}
        summary = " ".join(
            f"{k.split('/')[-1]}={v:.3f}"
            for k, v in flat.items()
            if k.startswith("eval/")
        )
        print(f"[step {num_steps}] {summary}", flush=True)

        if wandb_run is not None:
            import wandb
            payload = {"train/step": int(num_steps)}
            for k, v in flat.items():
                payload[k] = v
            wandb.log(payload, step=int(num_steps))

    # Import here so the module-load cost is not paid unless this script is run.
    from brax.training.agents.ppo import train as ppo_train

    t0 = time.time()
    make_inference_fn, params, _ = ppo_train.train(
        environment=env,
        num_timesteps=args.steps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        batch_size=args.batch_size,
        clipping_epsilon=args.clip_eps,
        gae_lambda=args.gae_lambda,
        seed=args.seed,
        progress_fn=progress_fn,
    )
    print(f"[gpu-train] done in {time.time() - t0:.1f}s", flush=True)

    # Save params + make_inference_fn closure ingredients.
    out_path = logdir / "brax_ppo_params.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "params": params,
            "args": vars(args),
            "obs_size": int(env.observation_size),
            "action_size": int(env.action_size),
        }, f)
    print(f"[gpu-train] saved params → {out_path}", flush=True)

    if wandb_run is not None:
        import wandb
        wandb.finish()


def _is_scalar(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
