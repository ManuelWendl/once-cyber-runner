"""Brax PPO training for Cyberrunner.

Usage:
    python train.py                          # use config.yaml defaults
    python train.py --debug                  # quick smoke run (small num_envs, 1M steps)
    python train.py --num-envs 2048          # override num_envs
"""

import argparse
import functools
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import jax
import jax.nn
import jax.numpy as jnp
import yaml
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train

from env_mjx import CyberrunnerMJXEnv


_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "leaky_relu": jax.nn.leaky_relu,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-timesteps", type=int)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Small-scale smoke test: 256 envs, 1M steps, 2 evals.")
    p.add_argument("--wandb", action="store_true", help="Force-enable wandb logging.")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument(
        "--prior-strategy",
        choices=["exp_d", "exp_d_sigma", "survival"],
        default=None,
        help="Override env.safe_prior_strategy from the config.",
    )
    p.add_argument(
        "--prior-sigma",
        type=float,
        default=None,
        help="Override env.safe_prior_sigma (only used by exp_d_sigma).",
    )
    p.add_argument("--init-ball-speed", type=float, default=None,
                   help="Override env.init_ball_speed (max spawn ball speed, m/s).")
    p.add_argument("--init-tilt-frac", type=float, default=None,
                   help="Override env.init_tilt_frac (fraction of joint range at spawn).")
    p.add_argument("--tilt-bumps", dest="tilt_bumps", action="store_true",
                   default=None, help="Enable random mid-episode tilt bumps.")
    p.add_argument("--no-tilt-bumps", dest="tilt_bumps", action="store_false",
                   default=None, help="Disable random mid-episode tilt bumps.")
    p.add_argument("--tilt-bump-prob", type=float, default=None,
                   help="Per-step probability of a tilt bump.")
    p.add_argument("--tilt-bump-magnitude", type=float, default=None,
                   help="Tilt-bump magnitude (fraction of half-joint-range).")
    p.add_argument("--run-name", default=None,
                   help="Custom subdir name appended to checkpoint_dir. "
                        "If unset, a timestamp + strategy + seed is used.")
    p.add_argument("--resume", default=None,
                   help="Path to a .pkl checkpoint (saved by this script) to "
                        "warm-start training from. Loads params only — "
                        "optimizer state restarts. Useful for curriculum "
                        "fine-tuning (e.g. survival → bumps).")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_network_factory(net_cfg: Dict[str, Any]):
    activation = _ACTIVATIONS[net_cfg["activation"]]
    hidden_sizes = tuple(net_cfg["hidden_sizes"])

    def factory(observation_size, action_size, preprocess_observations_fn=lambda x, y: x):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=hidden_sizes,
            value_hidden_layer_sizes=hidden_sizes,
            activation=activation,
            policy_obs_key="state",
            value_obs_key="state",
        )

    return factory


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs
    if args.num_timesteps is not None:
        config["training"]["num_timesteps"] = args.num_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.prior_strategy is not None:
        config["env"]["safe_prior_strategy"] = args.prior_strategy
    if args.prior_sigma is not None:
        config["env"]["safe_prior_sigma"] = args.prior_sigma
    if args.init_ball_speed is not None:
        config["env"]["init_ball_speed"] = args.init_ball_speed
    if args.init_tilt_frac is not None:
        config["env"]["init_tilt_frac"] = args.init_tilt_frac
    if args.tilt_bumps is not None:
        config["env"]["tilt_bumps"] = args.tilt_bumps
    if args.tilt_bump_prob is not None:
        config["env"]["tilt_bump_prob"] = args.tilt_bump_prob
    if args.tilt_bump_magnitude is not None:
        config["env"]["tilt_bump_magnitude"] = args.tilt_bump_magnitude

    if args.debug:
        config["env"]["num_envs"] = 256
        config["training"]["num_timesteps"] = 1_000_000
        config["training"]["num_evals"] = 2

    use_wandb = config.get("wandb", {}).get("enabled", False) or args.wandb
    if args.no_wandb:
        use_wandb = False

    seed = config["seed"]
    # Per-run checkpoint dir so successive runs don't overwrite each other.
    # If --checkpoint-dir was passed, treat it as the exact path; otherwise
    # build <config.checkpoint_dir>/<run_name>.
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        base_dir = Path(config["checkpoint_dir"])
        if args.run_name is not None:
            run_name = args.run_name
        else:
            strategy = config["env"].get("safe_prior_strategy", "exp_d")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{ts}_{strategy}_seed{seed}"
        checkpoint_dir = base_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------ Optional warm-start (curriculum fine-tuning) ------
    # `restore_params` accepts the same params tuple our _save_checkpoint
    # writes — Brax PPO uses it to seed normalizer + policy + value at the
    # start of training. Optimizer state is NOT restored; the run begins a
    # fresh PPO loop with the loaded weights.
    restore_params = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"--resume path not found: {resume_path}")
        with open(resume_path, "rb") as f:
            resume_blob = pickle.load(f)
        if "params" not in resume_blob:
            raise SystemExit(
                f"{resume_path} does not look like a train.py checkpoint "
                f"(missing 'params'). Keys: {list(resume_blob.keys())}"
            )
        restore_params = resume_blob["params"]
        prev_step = resume_blob.get("step", "?")
        prev_strategy = (
            resume_blob.get("config", {}).get("env", {}).get("safe_prior_strategy")
        )
        print(
            f"  resume:          {resume_path}  "
            f"(prev_step={prev_step}, prev_strategy={prev_strategy})"
        )

    # ------ Banner ------
    print("=" * 60)
    print("Cyberrunner Brax PPO training")
    print("=" * 60)
    print(f"  num_envs:        {config['env']['num_envs']}")
    print(f"  episode_length:  {config['env']['episode_length']}")
    print(f"  num_timesteps:   {config['training']['num_timesteps']:,}")
    print(f"  num_evals:       {config['training']['num_evals']}")
    print(f"  jax devices:     {jax.devices()}")
    print(f"  checkpoint_dir:  {checkpoint_dir}")
    print(f"  seed:            {seed}")
    print(f"  safe_prior:      {config['env'].get('safe_prior', False)}")
    print(f"  prior_strategy:  {config['env'].get('safe_prior_strategy', 'exp_d')}")
    if config['env'].get('safe_prior_strategy', 'exp_d') == 'exp_d_sigma':
        print(f"  prior_sigma:     {config['env'].get('safe_prior_sigma', 0.02)}")
    print(f"  init_ball_speed: {config['env'].get('init_ball_speed', 0.0)}")
    print(f"  init_tilt_frac:  {config['env'].get('init_tilt_frac', 0.0)}")
    print(f"  tilt_bumps:      {config['env'].get('tilt_bumps', False)}")
    if config['env'].get('tilt_bumps', False):
        print(f"  tilt_bump_prob:  {config['env'].get('tilt_bump_prob', 0.0)}")
        print(f"  tilt_bump_mag:   {config['env'].get('tilt_bump_magnitude', 0.0)}")
    print("=" * 60)

    # ------ Environment ------
    env = CyberrunnerMJXEnv(
        episode_length=config["env"]["episode_length"],
        randomize_init_pos=config["env"]["randomize_init_pos"],
        num_rays=config["env"].get("num_rays", 32),
        num_envs_hint=config["env"]["num_envs"],
        history_length=config["env"].get("history_length", 5),
        safe_prior=config["env"].get("safe_prior", False),
        safe_prior_strategy=config["env"].get("safe_prior_strategy", "exp_d"),
        safe_prior_sigma=config["env"].get("safe_prior_sigma", 0.02),
        init_ball_speed=config["env"].get("init_ball_speed", 0.0),
        init_tilt_frac=config["env"].get("init_tilt_frac", 0.0),
        tilt_bumps=config["env"].get("tilt_bumps", False),
        tilt_bump_prob=config["env"].get("tilt_bump_prob", 0.0),
        tilt_bump_magnitude=config["env"].get("tilt_bump_magnitude", 0.0),
    )
    print(f"  mjx backend:     {env._mjx_impl}")
    print(f"  obs size:        {env.observation_size}")
    print(f"  action size:     {env.action_size}")

    # ------ Wandb (optional) ------
    if use_wandb:
        import wandb
        wandb.init(
            project=config.get("wandb", {}).get("project", "cyberrunner_ppo"),
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # ------ Progress + best-checkpoint callbacks ------
    # progress_fn(num_steps, metrics) is called after each eval — we record
    # the latest eval/episode_reward into a closure so policy_params_fn can
    # decide whether the corresponding params are a new best and pickle them.
    t0 = time.time()
    best_state: Dict[str, Any] = {
        "best_reward": float("-inf"),
        "latest_reward": None,
        "latest_step": 0,
    }

    def progress(num_steps: int, metrics: Dict[str, Any]):
        elapsed = time.time() - t0
        sps = num_steps / max(elapsed, 1e-6)
        line = f"step {num_steps:>11,}/{config['training']['num_timesteps']:,}  ({elapsed:6.1f}s, {sps:>7,.0f} sps)"
        for k in (
            "eval/episode_reward",
            "eval/avg_episode_length",
            "training/policy_loss",
            "training/v_loss",
            "training/entropy_loss",
            "training/sps",
        ):
            if k in metrics:
                v = float(metrics[k])
                line += f"  {k}={v:.3f}"
        print(line, flush=True)

        if "eval/episode_reward" in metrics:
            best_state["latest_reward"] = float(metrics["eval/episode_reward"])
            best_state["latest_step"] = int(num_steps)

        if use_wandb:
            wandb.log(
                {k: float(v) for k, v in metrics.items()
                 if hasattr(v, "__float__") or isinstance(v, (int, float))},
                step=num_steps,
            )

    best_path = checkpoint_dir / "best.pkl"

    def policy_params_fn(current_step: int, make_policy, params) -> None:
        """Save params to best.pkl whenever the most recent eval improves.

        Brax invokes this with the just-evaluated params right after each
        eval (and once before training starts). progress() has already run
        for that step, so best_state["latest_reward"] reflects this params.
        """
        latest = best_state["latest_reward"]
        if latest is None:
            return
        if latest > best_state["best_reward"]:
            best_state["best_reward"] = latest
            _save_checkpoint(
                best_path, params=params, step=int(current_step), config=config
            )
            print(
                f"  [best] step {current_step:,}  "
                f"eval/episode_reward={latest:.3f} → {best_path.name}",
                flush=True,
            )

    # ------ Brax PPO train ------
    brax_cfg = config["training"]["brax_ppo"]
    network_factory = make_network_factory(brax_cfg["network"])

    train_kwargs: Dict[str, Any] = {}
    if restore_params is not None:
        train_kwargs["restore_params"] = restore_params

    train_fn = functools.partial(
        ppo_train.train,
        environment=env,
        num_timesteps=config["training"]["num_timesteps"],
        num_evals=config["training"]["num_evals"],
        episode_length=config["env"]["episode_length"],
        num_envs=config["env"]["num_envs"],
        learning_rate=brax_cfg["learning_rate"],
        entropy_cost=brax_cfg["entropy_cost"],
        discounting=brax_cfg["discounting"],
        unroll_length=brax_cfg["unroll_length"],
        num_minibatches=brax_cfg["num_minibatches"],
        num_updates_per_batch=brax_cfg["num_updates_per_batch"],
        batch_size=brax_cfg["batch_size"],
        reward_scaling=brax_cfg["reward_scaling"],
        normalize_observations=brax_cfg["normalize_observations"],
        action_repeat=1,
        seed=seed,
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        **train_kwargs,
    )

    make_inference_fn, params, final_metrics = train_fn()

    final_path = checkpoint_dir / "final.pkl"
    _save_checkpoint(final_path, params=params, step=config["training"]["num_timesteps"], config=config)
    print(f"\nDone. Saved final checkpoint: {final_path}")
    if best_state["best_reward"] > float("-inf"):
        print(
            f"Best eval/episode_reward: {best_state['best_reward']:.3f}  → {best_path}"
        )
    if use_wandb:
        import wandb
        wandb.finish()


def _save_checkpoint(path: Path, params, step: int, config: Dict[str, Any]):
    with open(path, "wb") as f:
        pickle.dump({"params": params, "step": step, "config": config}, f)


if __name__ == "__main__":
    main()
