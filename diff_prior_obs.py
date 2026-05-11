"""Dim-by-dim diff between env_mjx._build_obs and PriorObsAdapter.

Decisive test for SOOPER's adapter bug. Two pipelines feed the prior:

  env_mjx          ->  state.obs["state"]   (36-dim, built inside the JIT)
  CPU env + adapt  ->  PriorObsAdapter.transform(obs["states"], prev_action)

Both should produce identical 36-dim vectors when fed the same physical
state, biases, and per-step noise. This script runs both from the same
spawn with action=0 for `--steps` control steps, prints a per-step
side-by-side diff, and summarises mean(|diff|) per obs dim at the end.

Noise sources are zeroed where possible:
  - Per-episode obs bias: replaced post-reset with zeros on both sides.
  - Per-step noise (BALL_POS_NOISE, JOINT_ANGLE_NOISE): monkey-patched
    to 0 in both modules BEFORE building the envs. This is the cleanest
    way; the constants are module-level np scalars in both env files.

Spawn is forced to waypoints[0] (`randomize_init_pos=False`); ball
velocity is zero (`init_ball_speed=0, init_tilt_frac=0`); action is
zero every step. Anything left in the diff is an obs-construction
discrepancy.

Usage (cluster, headless or login node — no GPU needed):

    python diff_prior_obs.py --layout easy --steps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
VENDOR_DIR = REPO_DIR / ".vendor" / "cyberrunner_ppo"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(VENDOR_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--layout",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Maze layout to compare on.",
    )
    p.add_argument("--steps", type=int, default=8, help="Number of control steps to compare.")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Both envs reset with this seed (irrelevant after noise is zeroed, "
        "but pinned for reproducibility of MuJoCo solver micro-state).",
    )
    return p.parse_args()


def _zero_noise_constants() -> None:
    """Monkey-patch BALL_POS_NOISE and JOINT_ANGLE_NOISE to 0 in every
    module that resolves the names at obs-construction time.

    Caveat: `env_mjx` imports both constants BY VALUE at the top
    (`from env_mujoco import BALL_POS_NOISE, ...`), so they live in
    `env_mjx`'s globals as independent floats. Patching only
    `env_mujoco` is a no-op for env_mjx — we have to patch env_mjx
    directly. Same for cyberrunner_env_vision, which defines the
    constants in its own module.

    Must run BEFORE the first JIT trace of env_mjx.reset / step,
    because JAX bakes module-globals into the traced graph by value.
    """
    import env_mujoco
    import env_mjx
    import cyberrunner_env_vision

    for mod in (env_mujoco, env_mjx, cyberrunner_env_vision):
        mod.BALL_POS_NOISE = 0.0
        mod.JOINT_ANGLE_NOISE = 0.0


def _build_envs(layout: str):
    """Build both envs with deterministic spawn / no bumps / no DR / no
    init wobble. Returns (env_mjx, cpu_env)."""
    from env_mjx import CyberrunnerMJXEnv
    from cyberrunner_env_vision import CyberRunnerEnv

    env_mjx = CyberrunnerMJXEnv(
        episode_length=2000,
        randomize_init_pos=False,
        maze_layout=layout,
        safe_prior=True,
        safe_prior_strategy="survival",
        init_ball_speed=0.0,
        init_tilt_frac=0.0,
        tilt_bumps=False,
        domain_randomization=False,
        num_envs_hint=1,
    )
    cpu_env = CyberRunnerEnv(
        layout=layout,
        episode_length=2000,
        randomize_init_pos=False,
        include_vision=False,
    )
    return env_mjx, cpu_env


def _zero_episode_bias_mjx(state):
    """env_mjx samples per-episode obs bias inside reset() and stashes
    them in info. We replace with zeros to match the zeroed CPU env."""
    import jax.numpy as jnp

    new_info = dict(state.info)
    new_info["obs_bias_ball"] = jnp.zeros(2, dtype=jnp.float32)
    new_info["obs_bias_joint"] = jnp.zeros(2, dtype=jnp.float32)
    return state.replace(info=new_info)


def _zero_episode_bias_cpu(cpu_env) -> None:
    cpu_env._obs_bias = {
        "ball": np.zeros(2, dtype=np.float64),
        "joint": np.zeros(2, dtype=np.float64),
    }


def _diff_step(
    step_idx: int,
    env_mjx_obs36: np.ndarray,
    adapter_obs36: np.ndarray,
) -> np.ndarray:
    """Print a per-step block. Returns abs diff (36,)."""
    diff = env_mjx_obs36 - adapter_obs36
    LABELS = [
        # 5 history frames × 6 fields
        *(
            f"hist[{f}].{name}"
            for f in range(5)
            for name in ("alpha", "beta", "ball_x", "ball_y", "act_x", "act_y")
        ),
        "vec_close_x",
        "vec_close_y",
        "vec_next_x",
        "vec_next_y",
        "vec_nn_x",
        "vec_nn_y",
    ]
    print(f"\n--- step {step_idx} ---")
    print(f"{'dim':>3} {'label':>18} {'env_mjx':>10} {'adapter':>10} {'diff':>10}")
    for i, lab in enumerate(LABELS):
        marker = " *" if abs(diff[i]) > 1e-4 else ""
        print(
            f"{i:>3} {lab:>18} {env_mjx_obs36[i]:>+10.4f} "
            f"{adapter_obs36[i]:>+10.4f} {diff[i]:>+10.4f}{marker}"
        )
    return np.abs(diff)


def main() -> None:
    args = parse_args()

    _zero_noise_constants()  # MUST happen before env construction

    import jax
    import jax.numpy as jnp

    from dreamerv3.dreamerv3.sooper import PriorObsAdapter

    print(f"layout={args.layout}  steps={args.steps}")
    env_mjx, cpu_env = _build_envs(args.layout)

    # Reset both. env_mjx's reset is jitted; the rng key just decides spawn
    # waypoint (forced to 0 here) and per-episode biases (zeroed below).
    mjx_state = env_mjx.reset(jax.random.PRNGKey(args.seed))
    mjx_state = _zero_episode_bias_mjx(mjx_state)

    obs, _ = cpu_env.reset(seed=args.seed)
    _zero_episode_bias_cpu(cpu_env)
    # The bias was used INSIDE reset's _get_obs call before we zeroed it,
    # so the obs we just got might carry the pre-zero bias. Replay
    # _get_obs to get a clean one. (Safe to call: no env state mutated.)
    obs = cpu_env._get_obs()

    adapter = PriorObsAdapter(num_envs=1)
    adapter.reset_envs(np.array([True]))

    # Match env_mjx's reset-time history initialisation: it tiles the
    # spawn frame 5x. PriorObsAdapter starts with all zeros and rolls in.
    # If we don't pre-fill, dims 0..23 will trivially diverge at step 1.
    # Pre-fill by calling adapter.transform 5 times on the spawn obs +
    # zero prev_action — same content env_mjx would have produced.
    for _ in range(5):
        adapter.transform({"states": obs["states"][None, :]}, np.zeros((1, 2), dtype=np.float32))

    # Sanity print: spawn-state physical readings from both sides.
    qpos_mjx = np.asarray(mjx_state.pipeline_state.qpos[:5])
    qpos_cpu = np.asarray(cpu_env.data.qpos[:5])
    print("\nspawn state check (alpha, beta, ball_xyz):")
    print(f"  env_mjx qpos[:5] = {qpos_mjx}")
    print(f"  cpu_env qpos[:5] = {qpos_cpu}")

    # Compare obs at the spawn step (before any action). env_mjx's reset
    # already produced obs; CPU env returned obs from reset() too.
    env_mjx_obs36 = np.asarray(mjx_state.obs["state"], dtype=np.float64)
    adapter_obs36 = adapter.transform(
        {"states": obs["states"][None, :]},
        np.zeros((1, 2), dtype=np.float32),
    )[0].astype(np.float64)
    abs_diffs = [_diff_step(0, env_mjx_obs36, adapter_obs36)]

    # Now step both with zero action for --steps steps.
    step_jit = jax.jit(env_mjx.step)
    zero_act_mjx = jnp.zeros(2, dtype=jnp.float32)
    zero_act_np = np.zeros(2, dtype=np.float32)
    prev_action = np.zeros(2, dtype=np.float32)

    for t in range(1, args.steps + 1):
        mjx_state = step_jit(mjx_state, zero_act_mjx)
        obs, _, term, trunc, _ = cpu_env.step(zero_act_np)
        adapter_obs36 = adapter.transform(
            {"states": obs["states"][None, :]},
            prev_action[None, :],
        )[0].astype(np.float64)
        env_mjx_obs36 = np.asarray(mjx_state.obs["state"], dtype=np.float64)
        abs_diffs.append(_diff_step(t, env_mjx_obs36, adapter_obs36))

        prev_action = zero_act_np.copy()
        if term or trunc:
            print(f"\n(CPU env terminated at step {t}; stopping)")
            break

    abs_diffs_arr = np.stack(abs_diffs, axis=0)  # (T+1, 36)
    print("\n=== Mean |diff| per dim over all steps ===")
    LABELS = [
        *(
            f"hist[{f}].{name}"
            for f in range(5)
            for name in ("alpha", "beta", "ball_x", "ball_y", "act_x", "act_y")
        ),
        "vec_close_x",
        "vec_close_y",
        "vec_next_x",
        "vec_next_y",
        "vec_nn_x",
        "vec_nn_y",
    ]
    means = abs_diffs_arr.mean(axis=0)
    for i, lab in enumerate(LABELS):
        marker = " *" if means[i] > 1e-3 else ""
        print(f"  {i:>3} {lab:>18}  mean|diff|={means[i]:.6f}{marker}")

    n_diverging = int((means > 1e-3).sum())
    print(f"\nDims with mean|diff| > 1e-3: {n_diverging}/36")


if __name__ == "__main__":
    main()
