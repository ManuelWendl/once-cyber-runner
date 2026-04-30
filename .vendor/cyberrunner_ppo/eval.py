"""Evaluate a Brax PPO checkpoint on the CPU CyberRunnerEnv (env_mujoco.py).

Loads `<checkpoint>.pkl` from train.py, rebuilds the actor via Brax's PPO
network factory, and rolls out deterministically on the CPU MuJoCo env. The
CPU env handles rendering (much simpler than rendering MJX).

Usage:
    python eval.py --checkpoint checkpoints/final.pkl --episodes 5 --render human
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from env_mujoco import CyberRunnerEnv


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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", choices=["human", "none"], default="human")
    p.add_argument("--episode-length", type=int, default=2000)
    p.add_argument("--realtime", action="store_true",
                   help="Sleep 1/60s between steps to render at 60Hz")
    p.add_argument("--no-randomize-init", action="store_true",
                   help="Always start at the path origin (waypoints[0]) instead of a random waypoint")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_policy_fn(checkpoint: Dict[str, Any]) -> Tuple[Any, Any]:
    """Reconstruct policy from checkpoint, return (policy_fn, params)."""
    cfg = checkpoint.get("config", {})
    net_cfg = cfg.get("training", {}).get("brax_ppo", {}).get("network", {})
    activation = _ACTIVATIONS[net_cfg.get("activation", "swish")]
    hidden_sizes = tuple(net_cfg.get("hidden_sizes", [256, 256]))

    obs_size = 10
    action_size = 2

    network = ppo_networks.make_ppo_networks(
        observation_size={"state": (obs_size,)},
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        activation=activation,
        policy_obs_key="state",
        value_obs_key="state",
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)

    params = checkpoint["params"]
    if not isinstance(params, (tuple, list)):
        raise ValueError(f"Unrecognised params type: {type(params)}")
    # make_inference_fn expects (normalizer_params, policy_params)
    inference_params = (params[0], params[1])
    policy_fn = make_inference_fn(inference_params, deterministic=True)
    return policy_fn, inference_params


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    print(f"Loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    policy_fn, _ = build_policy_fn(checkpoint)
    policy_fn_jit = jax.jit(policy_fn)

    render_mode = None if args.render == "none" else args.render
    env = CyberRunnerEnv(
        render_mode=render_mode,
        episode_length=args.episode_length,
        randomize_init_pos=not args.no_randomize_init,
        include_vision=False,
    )

    rng = jax.random.PRNGKey(args.seed)
    summaries = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_progress = info.get("path_progress", 0.0)
        last_progress = ep_progress
        steps = 0
        reason = "running"

        while True:
            # CPU env returns {'states': ...}, policy expects {'state': ...}
            policy_obs = {"state": jnp.asarray(obs["states"], dtype=jnp.float32)}
            rng, k = jax.random.split(rng)
            action, _ = policy_fn_jit(policy_obs, k)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            last_progress = info.get("path_progress", last_progress)
            steps += 1

            if render_mode is not None:
                env.render()
                if args.realtime:
                    time.sleep(1 / 60)

            if terminated or truncated:
                reason = info.get("termination_reason", "done")
                break

        print(
            f"  episode {ep + 1}/{args.episodes}: "
            f"reward={ep_reward:8.3f}  steps={steps:>4}  "
            f"path_progress={last_progress:7.3f}  reason={reason}"
        )
        summaries.append((ep_reward, steps, last_progress, reason))

    env.close()

    # Aggregate
    rewards = [s[0] for s in summaries]
    steps_list = [s[1] for s in summaries]
    print()
    print(f"Mean reward:   {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Mean steps:    {np.mean(steps_list):.1f}")
    print(f"Goal reached:  {sum(1 for s in summaries if s[3] == 'goal')}/{len(summaries)}")


if __name__ == "__main__":
    main()
