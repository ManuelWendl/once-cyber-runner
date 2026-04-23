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

Reward (per-step):
    r = + k_v · max(0, v_max − speed)          # lower speed is better, bounded
      + k_m · clip(hole_margin, 0, m_max)      # safe-margin bonus, capped
      − k_a · ‖action‖₂                        # small control penalty
      + B_stable  if stable_steps ≥ H          # one-shot stabilization bonus
      − P_hole    if ball falls in a hole      # one-shot terminal penalty

Termination: hole fall OR stabilization reached OR timeout (episode_length).
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
        # Reward weights.
        k_v: float = 1.0,
        v_max: float = 1.0,
        k_m: float = 50.0,
        m_max: float = 0.01,
        m_safe: float = 0.004,
        k_a: float = 0.01,
        b_stable: float = 10.0,
        p_hole: float = 10.0,
        hold_steps: int = 30,
        v_th: float = 0.03,
    ):
        walls_h, walls_v, holes, waypoints = get_hard_layout()
        seg_lengths, cum_distances = compute_waypoint_distances(waypoints)

        # Build the SAME MjModel the CPU env builds.
        # No hand-crafted safe-checkpoint geometry is registered here (empty
        # array) — the `build_model` signature accepts an empty checkpoint_points.
        checkpoint_points_for_model = np.zeros((0, 2), dtype=np.float32)
        mj_model = build_model(walls_h, walls_v, holes, waypoints, checkpoint_points_for_model)
        mjx_model = mjx.put_model(mj_model)

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

        # Reward constants.
        self._k_v = float(k_v)
        self._v_max = float(v_max)
        self._k_m = float(k_m)
        self._m_max = float(m_max)
        self._m_safe = float(m_safe)
        self._k_a = float(k_a)
        self._b_stable = float(b_stable)
        self._p_hole = float(p_hole)
        self._hold_steps = int(hold_steps)
        self._v_th = float(v_th)

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
        metrics = {
            "success": stats.success,
            "stable_steps": stats.stable_steps.astype(jnp.float32),
            "episode_reward": jnp.asarray(0.0, dtype=jnp.float32),
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

        wall_d = self._min_wall_dist(ball_pos)  # clamped distance to wall set
        _ = wall_d  # used only for obs; not rewarded directly

        # Per-step reward.
        r_speed = self._k_v * jnp.maximum(0.0, self._v_max - ball_speed)
        r_margin = self._k_m * jnp.clip(hole_margin, 0.0, self._m_max)
        r_action = -self._k_a * jnp.linalg.norm(action)

        # Events.
        in_hole = hole_d < (HOLE_RADIUS + MARBLE_RADIUS * 0.2)
        safe = hole_margin > self._m_safe
        slow = ball_speed < self._v_th
        stable_here = slow & safe
        stable_steps = jnp.where(stable_here, stats.stable_steps + 1, jnp.int32(0))
        stabilized = stable_steps >= self._hold_steps

        reward = (
            r_speed + r_margin + r_action
            + jnp.where(stabilized, self._b_stable, 0.0)
            - jnp.where(in_hole, self._p_hole, 0.0)
        )

        step_count = stats.step_count + 1
        timeout = step_count >= self._episode_length
        done = (in_hole | stabilized | timeout).astype(jnp.float32)
        success = jnp.where(stabilized, 1.0, stats.success)

        new_stats = _EpStats(
            prev_ball_pos=ball_pos,
            stable_steps=stable_steps,
            step_count=step_count,
            success=success,
        )
        obs = self._obs(pipeline_state, ball_pos, ball_vel)
        prev_ep_reward = state.metrics.get("episode_reward", jnp.asarray(0.0, dtype=jnp.float32))
        metrics = {
            "success": success,
            "stable_steps": stable_steps.astype(jnp.float32),
            "episode_reward": prev_ep_reward + reward,
        }
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
