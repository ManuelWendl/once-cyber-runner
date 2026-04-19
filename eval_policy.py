"""
Evaluate a trained DreamerV3 policy on CyberRunner.

Usage:
    python eval_policy.py --logdir path/to/logdir/train --episodes 10

The script runs N episodes from the fixed start position (waypoint 0),
renders the environment, prints per-episode score, and shows a summary.

Arguments:
    --logdir      Path to the training logdir (contains ckpt/ subfolder)
    --episodes    Number of evaluation episodes (default: 10)
    --random_start  Use random start positions instead of fixed start
    --no_render   Disable rendering (headless, faster)
"""

import argparse
import importlib
import pathlib
import sys

# Make dreamerv3 and repo root importable
repo_root = pathlib.Path(__file__).resolve().parent
dreamerv3_root = repo_root / 'dreamerv3'
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(dreamerv3_root))

import numpy as np
import elements
import embodied


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True,
                        help='Training logdir containing ckpt/ subfolder')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--random_start', action='store_true',
                        help='Randomize starting position (default: fixed start)')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering')
    return parser.parse_args()


def make_config(logdir):
    """Load DreamerV3 config matching the training run."""
    from dreamerv3 import main as dreamer_main
    # Load defaults then override to match training
    config = elements.Config(dreamer_main.configs['defaults'])
    config = config.update(dreamer_main.configs['size50m'])
    config = config.update({
        'task': 'cyberrunner_default',
        'logdir': logdir,
        'jax.platform': 'cpu',  # Use CPU for local eval; change to 'gpu' if available
        'jax.prealloc': False,
    })
    return config


def make_env(config, randomize_init_pos=False, render=True):
    """Create the CyberRunner env."""
    from embodied.envs.cyberrunner import CyberRunner
    render_mode = 'human' if render else None
    env = CyberRunner(
        task='default',
        episode_length=2000,
        randomize_init_pos=randomize_init_pos,
        include_vision=True,
    )
    # Apply standard wrappers
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def run_episode(env, agent, render=True):
    """Run one episode and return total score and length."""
    carry = agent.init_policy(1)
    obs = env.step({'action': np.zeros(2, np.float32), 'reset': True})
    total_reward = 0.0
    length = 0
    done = False
    while not done:
        obs_batch = {k: v[None] for k, v in obs.items()}
        carry, acts, _ = agent.policy(carry, obs_batch, mode='eval')
        act = {k: v[0] for k, v in acts.items()}
        act['reset'] = False
        obs = env.step(act)
        total_reward += float(obs['reward'])
        length += 1
        done = bool(obs['is_last'])
        if render:
            env.render()
    return total_reward, length


def main():
    args = parse_args()
    logdir = pathlib.Path(args.logdir)
    ckpt_path = logdir / 'ckpt'
    assert ckpt_path.exists(), f'Checkpoint not found at {ckpt_path}'

    print(f'Loading checkpoint from {ckpt_path}')
    config = make_config(str(logdir))

    # Build agent
    from dreamerv3.main import make_agent as _make_agent
    from functools import partial as bind
    agent = _make_agent(config)()

    # Load weights
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(str(ckpt_path), keys=['agent'])
    print('Checkpoint loaded.')

    render = not args.no_render
    env = make_env(config, randomize_init_pos=args.random_start, render=render)

    scores = []
    lengths = []
    print(f'\nRunning {args.episodes} episodes '
          f'({"random" if args.random_start else "fixed"} start)...\n')

    for ep in range(args.episodes):
        score, length = run_episode(env, agent, render=render)
        scores.append(score)
        lengths.append(length)
        term = 'timeout' if length >= 2000 else ('goal' if score > 5 else 'hole')
        print(f'  Episode {ep+1:2d}: score={score:.3f}  length={length:4d}  [{term}]')

    env.close()

    print(f'\n--- Summary ({args.episodes} episodes) ---')
    print(f'  Mean score:   {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    print(f'  Max score:    {np.max(scores):.3f}')
    print(f'  Mean length:  {np.mean(lengths):.0f}')
    print(f'  Goal reached: {sum(s > 5 for s in scores)}/{args.episodes}')


if __name__ == '__main__':
    main()
