"""GPU-native CyberRunner env via MuJoCo MJX + Brax.

Optional, additive backend. `envs/cyberrunner.py` is untouched.

Scope (per the approved plan):
    - prior_task = "stabilize" only
    - NO hand-crafted safe-checkpoint target (pure stabilization)
    - same MuJoCo geometry: walls, holes, waypoints reused verbatim
    - 10-dim state obs (no target / checkpoint field)
    - simplified initial conditions (low ball speed, low tilt)
    - spawn from the existing `prior_start_points` bank on CPU, sampled
      per-episode via JAX rng at reset

Reward (per-step, "safe basin" shaping):
    safe  = tanh(hole_margin / d_ref)                # [0, 1], gradient everywhere
    quiet = exp(-(speed / v_ref)**2)                 # [0, 1], aggressive at 0
    level = exp(-(abs_tilt / tilt_ref)**2)           # [0, 1]

    r_step = w_shape · (safe + quiet)                # dense additive floor
           + w_sweet · safe · quiet · (0.5 + 0.5·level)  # multiplicative sweet-spot
           − k_a · ‖action‖²                         # smoothness
           − P_hole  if ball falls in a hole         # terminal penalty

Walls are NOT rewarded or penalized. The agent is free to discover that
resting against walls / in corners is a stable strategy through physics.

Termination: hole fall OR timeout (episode_length). No early success exit —
the agent gets to keep accruing sweet-spot reward for the full 500 steps.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from flax import struct
from mujoco import mjx

from envs.cyberrunner import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FRAME_SKIP,
    HOLE_RADIUS,
    MARBLE_RADIUS,
    RANGE_ALPHA,
    RANGE_BETA,
    TIMESTEP,
    WALL_RADIUS,
    build_model,
    compute_waypoint_distances,
    get_hard_layout,
)


@struct.dataclass
class _EpStats:
    prev_ball_pos: jnp.ndarray  # (2,)
    stable_steps: jnp.ndarray   # ()
    step_count: jnp.ndarray     # ()
    success: jnp.ndarray        # ()


class CyberRunnerMJXEnv(PipelineEnv):
    """Brax-compatible MJX env for the stabilize prior task.

    Observation (10-dim):
        [alpha, beta, bx, by, vbx, vby, min_hole_dist, min_wall_dist,
         abs_tilt_norm, ball_speed]

    Action: (2,) in [-1, 1] — alpha, beta motor commands (same as CPU env).
    """

    def __init__(
        self,
        episode_length: int = 500,
        init_ball_speed: float = 0.05,
        init_tilt_frac: float = 0.05,
        prior_spawn_source: str = "waypoints",
        prior_start_point_spacing: float = 0.01,
        prior_spawn_min_hole_margin: float = 0.012,
        # "Safe basin" reward constants.
        d_ref: float = 0.015,          # hole-margin reference (m) — tanh scale
        v_ref: float = 0.03,           # speed reference (m/s) — exp scale
        tilt_ref: float = 0.08,        # tilt reference (rad) — exp scale
        w_shape: float = 0.3,          # additive-floor weight
        w_sweet: float = 1.4,          # multiplicative sweet-spot weight
        k_a: float = 0.005,            # action ℓ2 cost
        p_hole: float = 50.0,          # terminal hole penalty
        # Sweet-spot flag thresholds (for the `success` metric only; NOT reward).
        safe_th: float = 0.5,
        quiet_th: float = 0.5,
        # MJX solver budget. MuJoCo's defaults (100/50) are overkill for a
        # marble-on-board system that converges in < 5 Newton steps. These
        # knobs are the dominant lever for GPU rollout throughput.
        solver_iterations: int = 6,
        ls_iterations: int = 6,
    ):
        walls_h, walls_v, holes, waypoints = get_hard_layout()
        seg_lengths, cum_distances = compute_waypoint_distances(waypoints)

        # Build the SAME MjModel the CPU env builds.
        # No hand-crafted safe-checkpoint geometry is registered here (empty
        # array) — the `build_model` signature accepts an empty checkpoint_points.
        checkpoint_points_for_model = np.zeros((0, 2), dtype=np.float32)
        mj_model = build_model(walls_h, walls_v, holes, waypoints, checkpoint_points_for_model)
        mjx_model = mjx.put_model(mj_model)

        # Override MJX solver iteration budget. MuJoCo's defaults
        # (iterations=100, ls_iterations=50) are wildly conservative for this
        # system — contacts are simple (sphere + ~53 capsule walls) and
        # typically converge in < 5 Newton steps. Reducing here is the primary
        # GPU-throughput lever; the CPU `mj_model` is NOT touched.
        mjx_model = mjx_model.replace(
            opt=mjx_model.opt.replace(
                iterations=int(solver_iterations),
                ls_iterations=int(ls_iterations),
            )
        )

        # Set n_frames=FRAME_SKIP so one env step == FRAME_SKIP physics steps,
        # matching the CPU env's 600 Hz physics / 60 Hz control cadence.
        super().__init__(sys=mjx_model, backend="mjx", n_frames=FRAME_SKIP)

        self._mj_model = mj_model
        self._marble_body_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "marble")
        )
        self._board_body_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "board")
        )

        # Walls as segment endpoints (identical to CPU construction).
        wall_starts = np.vstack([
            np.stack([walls_h[:, 0], walls_h[:, 2]], axis=1),
            np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1),
        ]).astype(np.float32)
        wall_ends = np.vstack([
            np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1),
            np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1),
        ]).astype(np.float32)
        self._wall_starts = jnp.asarray(wall_starts)
        self._wall_ends = jnp.asarray(wall_ends)

        self._holes = jnp.asarray(holes, dtype=jnp.float32)
        self._waypoints = jnp.asarray(waypoints, dtype=jnp.float32)

        # Spawn bank: reuse the CPU waypoints-based spawn logic without
        # checkpoint filtering (pure stabilization). Keep only spawns that
        # clear holes by a safe margin.
        spawn_np = self._build_spawn_bank(
            waypoints, holes, prior_spawn_source, prior_start_point_spacing,
            prior_spawn_min_hole_margin,
        )
        self._spawn_bank = jnp.asarray(spawn_np, dtype=jnp.float32)
        self._num_spawns = int(spawn_np.shape[0])

        # qpos layout: [alpha, beta, bx, by, bz, qw, qx, qy, qz]
        self._ball_height = float(0.0793)

        self._episode_length = int(episode_length)
        self._init_ball_speed = float(init_ball_speed)
        self._init_tilt_frac = float(init_tilt_frac)
        self._range_alpha = (float(RANGE_ALPHA[0]), float(RANGE_ALPHA[1]))
        self._range_beta = (float(RANGE_BETA[0]), float(RANGE_BETA[1]))

        # Reward constants ("safe basin" shaping).
        self._d_ref = float(d_ref)
        self._v_ref = float(v_ref)
        self._tilt_ref = float(tilt_ref)
        self._w_shape = float(w_shape)
        self._w_sweet = float(w_sweet)
        self._k_a = float(k_a)
        self._p_hole = float(p_hole)
        self._safe_th = float(safe_th)
        self._quiet_th = float(quiet_th)
        self._solver_iterations = int(solver_iterations)
        self._ls_iterations = int(ls_iterations)

    # ---- Brax API --------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, k_spawn, k_theta, k_speed, k_alpha, k_beta = jax.random.split(rng, 6)

        idx = jax.random.randint(k_spawn, (), 0, self._num_spawns)
        spawn_xy = self._spawn_bank[idx]

        tilt_frac = self._init_tilt_frac
        alpha0 = jax.random.uniform(
            k_alpha, (), minval=self._range_alpha[0] * tilt_frac,
            maxval=self._range_alpha[1] * tilt_frac,
        )
        beta0 = jax.random.uniform(
            k_beta, (), minval=self._range_beta[0] * tilt_frac,
            maxval=self._range_beta[1] * tilt_frac,
        )
        theta = jax.random.uniform(k_theta, (), minval=0.0, maxval=2.0 * jnp.pi)
        speed = jax.random.uniform(k_speed, (), minval=0.0, maxval=self._init_ball_speed)

        nq = self.sys.nq
        nv = self.sys.nv
        qpos = jnp.zeros(nq, dtype=jnp.float32)
        qpos = qpos.at[0].set(alpha0)
        qpos = qpos.at[1].set(beta0)
        qpos = qpos.at[2].set(spawn_xy[0])
        qpos = qpos.at[3].set(spawn_xy[1])
        qpos = qpos.at[4].set(self._ball_height)
        qpos = qpos.at[5].set(1.0)  # quat w

        qvel = jnp.zeros(nv, dtype=jnp.float32)
        qvel = qvel.at[2].set(speed * jnp.cos(theta))
        qvel = qvel.at[3].set(speed * jnp.sin(theta))

        pipeline_state = self.pipeline_init(qpos, qvel)
        ball_pos = pipeline_state.xpos[self._marble_body_id, :2]

        stats = _EpStats(
            prev_ball_pos=ball_pos,
            stable_steps=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            success=jnp.asarray(0.0, dtype=jnp.float32),
        )
        ball_vel = jnp.zeros(2, dtype=jnp.float32)
        obs = self._obs(pipeline_state, ball_pos, ball_vel)
        # All per-step metric keys must exist in reset for the EpisodeWrapper
        # to sum them consistently across the rollout.
        _zero = jnp.asarray(0.0, dtype=jnp.float32)
        metrics = {
            "success": stats.success,
            "stable_steps": stats.stable_steps.astype(jnp.float32),
            "r_safe": _zero,
            "r_quiet": _zero,
            "r_level": _zero,
            "r_sweet": _zero,
        }
        info = {"stats": stats, "rng": rng}
        return State(pipeline_state, obs, jnp.asarray(0.0, dtype=jnp.float32),
                     jnp.asarray(0.0, dtype=jnp.float32), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        ball_pos = pipeline_state.xpos[self._marble_body_id, :2]
        dt = TIMESTEP * FRAME_SKIP
        stats: _EpStats = state.info["stats"]
        ball_vel = (ball_pos - stats.prev_ball_pos) / dt
        ball_speed = jnp.linalg.norm(ball_vel)

        hole_d = jnp.min(jnp.linalg.norm(self._holes - ball_pos[None, :], axis=1))
        hole_margin = hole_d - HOLE_RADIUS

        # Tilt magnitude for the "level" term.
        alpha = pipeline_state.qpos[0]
        beta = pipeline_state.qpos[1]
        abs_tilt = jnp.sqrt(alpha * alpha + beta * beta)

        # Shaped potentials — all in [0, 1]. Walls are NOT in the reward;
        # the agent is free to use them as physical brakes.
        safe = jnp.tanh(jnp.maximum(hole_margin, 0.0) / self._d_ref)
        quiet = jnp.exp(-(ball_speed / self._v_ref) ** 2)
        level = jnp.exp(-(abs_tilt / self._tilt_ref) ** 2)

        # Additive floor (dense early signal) + multiplicative sweet-spot bonus.
        r_shape = self._w_shape * (safe + quiet)
        r_sweet = self._w_sweet * safe * quiet * (0.5 + 0.5 * level)
        r_action = -self._k_a * jnp.sum(action * action)

        # Events.
        in_hole = hole_d < (HOLE_RADIUS + MARBLE_RADIUS * 0.2)

        reward = (
            r_shape + r_sweet + r_action
            - jnp.where(in_hole, self._p_hole, 0.0)
        )

        # Sweet-spot flag (diagnostic metric only — never fed into reward).
        # Tracks fraction of episode the ball spent in a "basin"-like state.
        in_sweet = (safe > self._safe_th) & (quiet > self._quiet_th)
        stable_steps = jnp.where(
            in_sweet, stats.stable_steps + 1, jnp.int32(0)
        )

        step_count = stats.step_count + 1
        timeout = step_count >= self._episode_length
        done = (in_hole | timeout).astype(jnp.float32)
        # `success` is a PER-STEP indicator (0/1). Brax's EpisodeWrapper sums
        # it across the rollout, so `success_sum / ep_len` is the fraction of
        # the episode the ball spent in a safe basin (always in [0, 1]).
        success = in_sweet.astype(jnp.float32)

        new_stats = _EpStats(
            prev_ball_pos=ball_pos,
            stable_steps=stable_steps,
            step_count=step_count,
            success=success,
        )
        obs = self._obs(pipeline_state, ball_pos, ball_vel)
        metrics = dict(state.metrics)
        metrics.update({
            "success": success,
            "stable_steps": stable_steps.astype(jnp.float32),
            "r_safe": safe,
            "r_quiet": quiet,
            "r_level": level,
            "r_sweet": r_sweet,
        })
        info = {**state.info, "stats": new_stats}
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done,
            metrics=metrics, info=info,
        )

    # ---- obs / geometry --------------------------------------------------

    def _obs(self, pipeline_state, ball_pos: jax.Array, ball_vel: jax.Array) -> jax.Array:
        alpha = pipeline_state.qpos[0]
        beta = pipeline_state.qpos[1]
        hole_d = jnp.min(jnp.linalg.norm(self._holes - ball_pos[None, :], axis=1))
        wall_d = self._min_wall_dist(ball_pos)
        abs_tilt = jnp.sqrt(alpha * alpha + beta * beta)
        speed = jnp.linalg.norm(ball_vel)
        return jnp.stack([
            alpha, beta,
            ball_pos[0], ball_pos[1],
            ball_vel[0], ball_vel[1],
            hole_d, wall_d,
            abs_tilt, speed,
        ]).astype(jnp.float32)

    def _min_wall_dist(self, ball_pos: jax.Array) -> jax.Array:
        """Min distance from ball center to any wall segment OR board edge."""
        seg_d = _point_to_segments_dist(ball_pos, self._wall_starts, self._wall_ends)
        edge_d = jnp.minimum(
            jnp.minimum(ball_pos[0], BOARD_WIDTH - ball_pos[0]),
            jnp.minimum(ball_pos[1], BOARD_HEIGHT - ball_pos[1]),
        ) + WALL_RADIUS
        return jnp.minimum(jnp.min(seg_d), edge_d)

    @staticmethod
    def _build_spawn_bank(
        waypoints: np.ndarray,
        holes: np.ndarray,
        source: str,
        spacing: float,
        min_hole_margin: float,
    ) -> np.ndarray:
        if source == "waypoints":
            candidates = np.asarray(waypoints, dtype=np.float32)
        else:
            pts = [waypoints[0]]
            for i in range(len(waypoints) - 1):
                s, e = waypoints[i], waypoints[i + 1]
                seg = e - s
                L = float(np.linalg.norm(seg))
                if L < 1e-8:
                    continue
                num = max(1, int(np.ceil(L / spacing)))
                ts = np.linspace(0.0, 1.0, num + 1, endpoint=False)[1:]
                for t in ts:
                    pts.append((1.0 - t) * s + t * e)
            pts.append(waypoints[-1])
            candidates = np.asarray(pts, dtype=np.float32)
        dists = np.linalg.norm(candidates[:, None, :] - holes[None, :, :], axis=2).min(axis=1)
        keep = dists > (HOLE_RADIUS + min_hole_margin)
        filtered = candidates[keep]
        if len(filtered) == 0:
            return candidates
        return filtered.astype(np.float32)

    # Brax uses these for policy/value network construction.
    @property
    def observation_size(self) -> int:
        return 10

    @property
    def action_size(self) -> int:
        return 2


def _point_to_segments_dist(
    p: jax.Array, starts: jax.Array, ends: jax.Array
) -> jax.Array:
    """Distance from point p (2,) to each of N line segments. Returns (N,)."""
    seg = ends - starts
    seg_len_sq = jnp.sum(seg * seg, axis=1)
    seg_len_sq = jnp.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
    t = jnp.sum((p[None, :] - starts) * seg, axis=1) / seg_len_sq
    t = jnp.clip(t, 0.0, 1.0)
    proj = starts + seg * t[:, None]
    return jnp.linalg.norm(proj - p[None, :], axis=1)
