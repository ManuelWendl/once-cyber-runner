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
    parser.add_argument("--num_evals", type=int, default=100)
    parser.add_argument("--save_every_evals", type=int, default=10)
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
        # Use env steps as the global x-axis.
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        # Explicit groupings so the UI auto-charts them together.
        for prefix in ("eval/", "training/", "perf/", "reward/"):
            wandb.define_metric(f"{prefix}*", step_metric="train/step")
        (logdir / "wandb_run_id.txt").write_text(str(wandb_run.id))

    eval_counter = {"n": 0}
    t_start = time.time()
    _latest_params = {"p": None}

    def policy_params_fn(current_step, make_policy, params):
        _latest_params["p"] = params

    def progress_fn(num_steps: int, metrics: dict) -> None:
        eval_counter["n"] += 1
        flat = {k: float(v) for k, v in metrics.items() if _is_scalar(v)}

        def _get(k: str, d: float = float("nan")) -> float:
            return flat.get(k, d)

        # Brax's EpisodeWrapper reports per-step metrics SUMMED over the
        # episode. Convert to interpretable fractions/averages using the
        # average episode length.
        ep_len = _get("eval/avg_episode_length")
        ep_len_safe = ep_len if (ep_len and ep_len > 0) else float("nan")
        ep_len_cap = float(args.episode_length)

        reward_sum = _get("eval/episode_reward")
        success_sum = _get("eval/episode_success")
        stable_sum = _get("eval/episode_stable_steps")

        # Per-step reward (max achievable ~ k_v·v_max + k_m·m_max = 1.5).
        reward_per_step = reward_sum / ep_len_safe
        # Fraction of episode spent in success (sticky flag) state.
        success_frac = success_sum / ep_len_safe
        # Avg value of the running stable_steps counter during the episode.
        stable_avg = stable_sum / ep_len_safe
        # Fraction of max episode length actually survived. Rises toward 1.0
        # once the agent stops falling in holes.
        length_frac = ep_len / ep_len_cap if ep_len == ep_len else float("nan")

        # Human summary on stdout (interpretable, not raw Brax sums).
        summary = (
            f"len={ep_len:.1f}/{int(ep_len_cap)} ({length_frac:.0%}) "
            f"success_frac={success_frac:.3f} "
            f"stable_avg={stable_avg:.1f} "
            f"reward/step={reward_per_step:.3f} "
            f"sps={_get('training/sps'):.0f}"
        )
        print(f"[step {num_steps}] eval#{eval_counter['n']} {summary}", flush=True)

        if wandb_run is not None:
            import wandb
            elapsed = time.time() - t_start
            payload = {"train/step": int(num_steps)}
            # Pass through every scalar Brax gives us (eval/*, training/*).
            for k, v in flat.items():
                payload[k] = v
            # ---- Interpretable derived metrics (what to watch) ----
            payload["perf/walltime_s"] = elapsed
            payload["perf/sps_overall"] = num_steps / max(elapsed, 1e-6)
            payload["eval_norm/length_frac"] = length_frac
            payload["eval_norm/success_frac"] = success_frac
            payload["eval_norm/stable_avg"] = stable_avg
            payload["eval_norm/reward_per_step"] = reward_per_step
            # Composite "health": episode survives AND is mostly in success.
            payload["eval_norm/deployment_score"] = length_frac * success_frac
            wandb.log(payload, step=int(num_steps))

        # Periodic checkpointing to survive crashes.
        if args.save_every_evals > 0 and eval_counter["n"] % args.save_every_evals == 0:
            try:
                ckpt_path = logdir / f"brax_ppo_params_step{int(num_steps)}.pkl"
                with open(ckpt_path, "wb") as f:
                    pickle.dump(
                        {"step": int(num_steps), "params": _latest_params["p"]},
                        f,
                    )
                print(f"[gpu-train] ckpt → {ckpt_path}", flush=True)
            except Exception as e:
                print(f"[gpu-train] ckpt failed: {e}", flush=True)

    # Import here so the module-load cost is not paid unless this script is run.
    from brax.training.agents.ppo import train as ppo_train

    # Auto-tune num_evals so `args.steps` is honored faithfully.
    #
    # Brax's PPO does exactly one "training step" worth of env collection per
    # training step, where
    #   env_step_per_ts = batch_size * unroll_length * num_minibatches
    # and the number of env steps it runs in total is
    #   actual = (num_evals - 1) * steps_per_epoch * env_step_per_ts
    # with `steps_per_epoch = ceil(num_timesteps / ((num_evals - 1) * env_step_per_ts))`.
    #
    # If we pick steps_per_epoch=1 and choose num_evals to absorb the count,
    # the math collapses to `actual = (num_evals - 1) * env_step_per_ts`,
    # which we can match to `args.steps` within ±½ of env_step_per_ts.
    _env_step_per_ts = (
        args.batch_size * args.unroll_length * args.num_minibatches
    )
    _evals_after_init = max(1, round(args.steps / _env_step_per_ts))
    _num_evals = _evals_after_init + 1
    _actual_steps = _evals_after_init * _env_step_per_ts
    _delta_pct = 100.0 * (_actual_steps / args.steps - 1.0)
    print(
        f"[gpu-train] STEPS auto-tune: requested={args.steps:,} "
        f"→ running={_actual_steps:,} ({_delta_pct:+.2f}%)  "
        f"env_step/ts={_env_step_per_ts:,}  "
        f"num_evals: {args.num_evals} → {_num_evals} "
        f"(eval every {_env_step_per_ts:,} steps)",
        flush=True,
    )

    t0 = time.time()
    make_inference_fn, params, _ = ppo_train.train(
        environment=env,
        num_timesteps=_actual_steps,
        num_evals=_num_evals,
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
        policy_params_fn=policy_params_fn,
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
