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

# Force MJX/JAX into float32 with TF32 matmuls. MJX defaults to f64, which is
# 2x-60x slower than f32 on most GPUs (worst on consumer cards). This is the
# single biggest GPU PPO speedup. Must run BEFORE any jax.numpy import that
# would lock the dtype config.
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

import jax.numpy as jnp
from brax.envs.base import State, Wrapper

from envs.cyberrunner_mjx import CyberRunnerMJXEnv


class RunningRewardNormalizationWrapper(Wrapper):
    """Per-env running estimate of discounted-return std; normalizes reward.

    Mirrors SB3 ``VecNormalize(norm_reward=True)``: tracks the discounted
    return ``R_t = γ·R_{t-1} + r_t`` (reset to 0 when the previous step ended
    an episode) and divides outgoing reward by ``sqrt(var(R) + eps)``,
    clipping to ``±clip``.

    Brax env state is per-env (vmapped), so each env carries its own Welford
    running stats. After ~thousand steps per env this converges to the true
    discounted-return variance — adequate for the millions-of-steps regime.
    """

    def __init__(self, env, gamma: float = 0.99, eps: float = 1e-8,
                 clip: float = 10.0):
        super().__init__(env)
        self._gamma = float(gamma)
        self._eps = float(eps)
        self._clip = float(clip)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        z = jnp.zeros((), dtype=jnp.float32)
        info = {
            **state.info,
            "_rn_ret": z,
            "_rn_mean": z,
            "_rn_m2": z,
            "_rn_count": z,
        }
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        new_state = self.env.step(state, action)
        # NOTE: `state.done` here is ALWAYS 0 because brax's AutoResetWrapper
        # zeros it before delegating into the inner env stack. We instead
        # schedule the reset using `new_state.done` (the just-emitted done)
        # at the END of this method, so the NEXT call sees a fresh `_rn_ret`.
        ret_prev = state.info["_rn_ret"]
        ret = self._gamma * ret_prev + new_state.reward
        # Welford online update on `ret` (single sample per env per step).
        count = state.info["_rn_count"] + 1.0
        delta = ret - state.info["_rn_mean"]
        mean = state.info["_rn_mean"] + delta / count
        delta2 = ret - mean
        m2 = state.info["_rn_m2"] + delta * delta2
        var = m2 / jnp.maximum(count, 1.0)
        # SB3 convention: divide by std but DO NOT subtract the mean.
        norm = new_state.reward / jnp.sqrt(var + self._eps)
        norm = jnp.clip(norm, -self._clip, self._clip)
        # If THIS step ended the episode, zero the carried return so the
        # NEXT step starts a fresh discounted-return roll-up.
        next_ret = jnp.where(new_state.done > 0.5,
                             jnp.zeros_like(ret), ret)
        info = {
            **new_state.info,
            "_rn_ret": next_ret,
            "_rn_mean": mean,
            "_rn_m2": m2,
            "_rn_count": count,
        }
        return new_state.replace(reward=norm.astype(jnp.float32), info=info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logdir/ppo_gpu")
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--unroll_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--num_updates_per_batch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy_cost", type=float, default=8.9e-3)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--clip_eps", type=float, default=0.23)
    parser.add_argument("--episode_length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init_ball_speed", type=float, default=0.05)
    parser.add_argument("--init_tilt_frac", type=float, default=0.05)
    parser.add_argument(
        "--prior_version",
        type=str,
        default="legacy",
        choices=["legacy", "checkpoint_recovery", "dense"],
    )
    parser.add_argument("--mjx_smoke", action="store_true")
    parser.add_argument("--num_evals", type=int, default=100)
    parser.add_argument("--save_every_evals", type=int, default=10)
    # MJX solver budget — dominant lever for GPU rollout throughput.
    parser.add_argument("--solver_iterations", type=int, default=6)
    parser.add_argument("--ls_iterations", type=int, default=6)
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
        prior_version=args.prior_version,
        solver_iterations=args.solver_iterations,
        ls_iterations=args.ls_iterations,
    )
    print(
        f"[gpu-train] prior_version={args.prior_version} "
        f"obs_dim={env.obs_dim} observation_size={env.observation_size}",
        flush=True,
    )
    if args.mjx_smoke:
        rng = jax.random.PRNGKey(args.seed)
        state = env.reset(rng)
        action = jnp.zeros((env.action_size,), dtype=jnp.float32)
        state = env.step(state, action)
        print(f"[mjx-smoke] prior_version={args.prior_version}")
        print(f"[mjx-smoke] OBS_DIM={env.obs_dim}")
        print(f"[mjx-smoke] observation_size={env.observation_size}")
        print(f"[mjx-smoke] reset/step obs shape={state.obs.shape}")
        print(f"[mjx-smoke] first_step_reward={float(state.reward):.6f}")
        print(f"[mjx-smoke] metric_keys={sorted(state.metrics.keys())}")
        for key in ("success", "quiet_step", "checkpoint_dist_final", "checkpoint_dist_min", "term_hole", "term_timeout"):
            if key in state.metrics:
                print(f"[mjx-smoke] {key}={float(state.metrics[key]):.6f}")
        return
    # Mirror CPU `VecNormalize(norm_reward=True)`: divide reward by running
    # std of the discounted return. Per-env Welford in JAX, see class doc.
    # Skip for prior versions with large terminal spikes (legacy: -50 hole
    # + +100 survival; checkpoint_recovery: -50 hole) — the running std
    # gets dominated by those events and crushes the dense per-step signal.
    if args.prior_version == "dense":
        env = RunningRewardNormalizationWrapper(
            env, gamma=args.discounting, clip=10.0,
        )
    else:
        print(
            f"[gpu-train] reward normalization DISABLED for "
            f"prior_version={args.prior_version} (large terminal spikes)",
            flush=True,
        )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=vars(args),
        )
        # Use env steps as the global x-axis. Group keys to mirror the CPU
        # PPO run (train_ppo.py) so panels overlay cleanly when both jobs
        # share a wandb project.
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        for prefix in ("episode/", "rollout/", "eval/", "training/",
                       "perf/", "reward/", "eval_norm/"):
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
        # `success` fires once per episode → wrapper-summed value is 0 or 1
        # per episode → mean across episodes = success RATE ∈ [0, 1].
        # Matches CPU `episode/success_rate` semantics.
        success_rate = _get("eval/episode_success")
        quiet_sum = _get("eval/episode_quiet_step")
        stable_sum = _get("eval/episode_stable_steps")
        stable_final = _get("eval/episode_stable_steps_final")
        stable_max = _get("eval/episode_stable_steps_max")
        ball_speed_sum = _get("eval/episode_ball_speed")
        observed_speed_sum = _get("eval/episode_observed_speed", ball_speed_sum)
        safe_margin_sum = _get("eval/episode_safe_hole_margin")
        checkpoint_dist_final_sum = _get("eval/episode_checkpoint_dist_final")
        checkpoint_dist_min_sum = _get("eval/episode_checkpoint_dist_min")
        inside_checkpoint_sum = _get("eval/episode_inside_checkpoint")

        # Per-step reward (post-normalization, ~unit scale once Welford warms up).
        reward_per_step = reward_sum / ep_len_safe
        # Fraction of steps the ball spent in a quiet (low-velocity) state.
        quiet_frac = quiet_sum / ep_len_safe
        # Avg value of the running stable_steps counter during the episode.
        stable_avg = stable_sum / ep_len_safe
        # Episode-mean ball speed and hole margin.
        ball_speed_mean = observed_speed_sum / ep_len_safe
        safe_margin_mean = safe_margin_sum / ep_len_safe
        checkpoint_dist_final = checkpoint_dist_final_sum
        checkpoint_dist_min = checkpoint_dist_min_sum
        inside_checkpoint_frac = inside_checkpoint_sum / ep_len_safe
        # Fraction of max episode length actually survived. Rises toward 1.0
        # once the agent stops falling in holes.
        length_frac = ep_len / ep_len_cap if ep_len == ep_len else float("nan")

        # Human summary on stdout (interpretable, not raw Brax sums).
        summary = (
            f"len={ep_len:.1f}/{int(ep_len_cap)} ({length_frac:.0%}) "
            f"success_rate={success_rate:.3f} "
            f"quiet_frac={quiet_frac:.3f} "
            f"stable_avg={stable_avg:.1f} "
            f"reward/step={reward_per_step:.3f} "
            f"sps={_get('training/sps'):.0f}"
        )
        print(f"[step {num_steps}] eval#{eval_counter['n']} {summary}", flush=True)

        if wandb_run is not None:
            import wandb
            elapsed = time.time() - t_start
            term_hole = _get("eval/episode_term_hole")
            term_timeout = _get("eval/episode_term_timeout")
            payload = {"train/step": int(num_steps)}
            # Pass through every scalar Brax gives us (eval/*, training/*).
            for k, v in flat.items():
                payload[k] = v
            # ---- Unified metric set (must match train_ppo.py keys) ----
            payload["episode/mean_reward"] = reward_sum
            payload["episode/mean_length"] = ep_len
            payload["episode/length_frac"] = length_frac
            payload["episode/success_rate"] = success_rate
            payload["episode/stable_steps_final"] = stable_final
            payload["episode/stable_steps_max"] = stable_max
            payload["episode/mean_stable_steps"] = stable_max
            payload["episode/quiet_frac"] = quiet_frac
            payload["episode/mean_observed_speed"] = ball_speed_mean
            payload["episode/mean_ball_speed"] = ball_speed_mean
            payload["episode/mean_safe_hole_margin"] = safe_margin_mean
            payload["episode/checkpoint_dist_final"] = checkpoint_dist_final
            payload["episode/checkpoint_dist_min"] = checkpoint_dist_min
            payload["episode/inside_checkpoint_frac"] = inside_checkpoint_frac
            payload["episode/reward_per_step"] = reward_per_step
            payload["episode/deployment_score"] = length_frac * success_rate
            payload["episode/termination_hole_rate"] = term_hole
            payload["episode/termination_timeout_rate"] = term_timeout
            payload["episode/termination_other_rate"] = max(
                0.0, 1.0 - term_hole - term_timeout
            )
            payload["rollout/ep_rew_mean"] = reward_sum
            payload["rollout/ep_len_mean"] = ep_len
            payload["perf/walltime_s"] = elapsed
            payload["perf/sps_overall"] = num_steps / max(elapsed, 1e-6)
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
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training import distribution as brax_distribution
    from brax.training import networks as brax_networks
    from brax.training.acme import running_statistics

    # SB3 `MlpPolicy` uses a DiagGaussian (no tanh-squash). Brax ships only
    # NormalTanhDistribution as a parametric wrapper, so we build a parallel
    # class that swaps TanhBijector for an identity bijector.
    class _IdentityBijector:
        def forward(self, x):
            return x

        def inverse(self, y):
            return y

        def forward_log_det_jacobian(self, x):
            return jnp.zeros_like(x)

    class NormalDiagDistribution(brax_distribution.ParametricDistribution):
        """DiagGaussian over R^action_size — matches SB3 MlpPolicy."""

        def __init__(self, event_size, min_std=0.001, var_scale=1.0):
            super().__init__(
                param_size=2 * event_size,
                postprocessor=_IdentityBijector(),
                event_ndims=1,
                reparametrizable=True,
            )
            self._min_std = min_std
            self._var_scale = var_scale

        def create_dist(self, parameters):
            loc, scale = jnp.split(parameters, 2, axis=-1)
            scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
            return brax_distribution.NormalDistribution(loc=loc, scale=scale)

    # Mimic SB3 `MlpPolicy` for continuous Box actions:
    #   - two hidden layers of 64, tanh activation, separate policy/value MLPs
    #   - DiagGaussian (NOT tanh-squashed) action distribution
    def make_ppo_networks_normal(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
        activation=jax.nn.tanh,
    ):
        parametric_action_distribution = NormalDiagDistribution(
            event_size=action_size,
        )
        policy_network = brax_networks.make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation,
        )
        value_network = brax_networks.make_value_network(
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
        )
        return ppo_networks.PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=parametric_action_distribution,
        )

    network_factory = functools.partial(
        make_ppo_networks_normal,
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
        activation=jax.nn.tanh,
    )

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
        network_factory=network_factory,
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
