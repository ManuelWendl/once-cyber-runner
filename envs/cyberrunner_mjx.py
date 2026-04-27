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

Reward (per-step, survival shaping — identical to CPU stabilize branch):
    quiet = exp(-(speed / v_ref)**2)                 # [0, 1], peak at 0 speed

    r_step = w_quiet · quiet                         # low velocity = +reward
           − k_a · ‖action‖²                         # smoothness
           − P_hole  if ball falls in a hole         # terminal penalty

Distance to holes is NOT in the reward. If the ball stabilizes anywhere safe
(even near a hole edge), that is acceptable. The hole penalty itself is
large enough (50) to dominate the long-horizon return.

Termination: hole fall OR timeout (episode_length). No early success exit —
the agent gets to keep accruing reward for the full 500 steps.
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
    BALL_POS_NOISE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FRAME_SKIP,
    HOLE_RADIUS,
    JOINT_ANGLE_NOISE,
    RANGE_ALPHA,
    RANGE_BETA,
    TIMESTEP,
    WALL_RADIUS,
    _project_points_to_path,
    build_model,
    compute_waypoint_distances,
    get_hard_layout,
    select_safe_checkpoints,
)


N_STACK = 4   # frame stacking, matches CPU `train_ppo.py --n_stack 4`
OBS_DIM = 13  # per-frame obs: 10-dim states + 3-dim checkpoint (matches CPU)


@struct.dataclass
class _EpStats:
    prev_ball_pos_clean: jnp.ndarray   # (2,)  true position, for clean speed
    prev_ball_pos_noisy: jnp.ndarray   # (2,)  noisy obs position, for EMA speed
    ema_speed: jnp.ndarray             # ()    EMA-smoothed noisy speed (reward source)
    obs_buf: jnp.ndarray               # (N_STACK - 1, OBS_DIM) — last 3 frames
    bias_ball: jnp.ndarray             # (2,)  per-episode ball-pos bias
    bias_joint: jnp.ndarray            # (2,)  per-episode joint-angle bias
    stable_steps: jnp.ndarray          # ()
    step_count: jnp.ndarray            # ()
    success: jnp.ndarray               # ()    sticky "ever crossed" flag


class CyberRunnerMJXEnv(PipelineEnv):
    """Brax-compatible MJX env for the stabilize prior task.

    Observation matches the CPU env (`envs/cyberrunner.py::_get_obs::states`):
        per-frame (10-dim, all NOISY):
            [α, β, bx, by, vec_to_closest_path(2), vec_to_next_wp(2),
             vec_to_next_next_wp(2)]
        policy input is a 4-frame stack along the last axis (40-dim total),
        replicating SB3 `VecFrameStack(n_stack=4)` with zero-init older frames.

    Reward:
        Identical "safe basin" formula to the CPU env's stabilize branch.
        `quiet` uses the EMA-smoothed noisy speed (matches CPU sensor model).

    Action: (2,) in [-1, 1] — alpha, beta motor commands (same as CPU env).
    """

    def __init__(
        self,
        episode_length: int = 500,
        init_ball_speed: float = 0.05,
        init_tilt_frac: float = 0.05,
        prior_spawn_source: str = "waypoints",
        prior_start_point_spacing: float = 0.01,
        # Match CPU `prior_spawn_min_hole_margin=0.0` — spawn from ALL waypoints.
        prior_spawn_min_hole_margin: float = 0.0,
        # EMA smoothing α for the noisy ball speed used by the reward
        # (matches CPU `checkpoint_speed_ema_alpha=0.8`).
        speed_ema_alpha: float = 0.8,
        # Survival reward constants.
        v_ref: float = 0.03,           # speed reference (m/s) — exp scale
        w_quiet: float = 1.0,          # weight on the low-velocity term
        k_a: float = 0.005,            # action ℓ2 cost
        p_hole: float = 50.0,          # terminal hole penalty
        p_survival: float = 100.0,     # one-shot bonus on full-episode survival
        # `success` metric threshold on the quiet term.
        quiet_th: float = 0.5,
        # Episode-level success criterion — matches CPU
        # `checkpoint_hold_steps=6`. Once stable_steps crosses this, the
        # `success` metric fires once for the episode.
        success_hold_steps: int = 6,
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
        # Path geometry (numpy → JAX) for in-graph path projection.
        wp = np.asarray(waypoints, dtype=np.float32)
        seg_starts_np = wp[:-1]
        seg_ends_np = wp[1:]
        seg_vecs_np = seg_ends_np - seg_starts_np
        seg_lens_np = np.linalg.norm(seg_vecs_np, axis=1).astype(np.float32)
        cum_dists_np = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.cumsum(seg_lens_np)]
        ).astype(np.float32)
        self._seg_starts = jnp.asarray(seg_starts_np)
        self._seg_vecs = jnp.asarray(seg_vecs_np)
        self._seg_lens = jnp.asarray(seg_lens_np)
        self._cum_dists = jnp.asarray(cum_dists_np)
        self._num_waypoints = int(wp.shape[0])

        # Safe corner targets — corners only (no single-wall fallback), same
        # as CPU `_corner_points`. These are static per maze layout.
        corners_np = select_safe_checkpoints(
            wp, np.asarray(holes, dtype=np.float32),
            walls_h, walls_v,
            reward_every_n_waypoints=3,
            include_corridors=False,
        )
        if len(corners_np) == 0:
            corners_np = np.zeros((1, 2), dtype=np.float32)
            corner_progresses_np = np.zeros((1,), dtype=np.float32)
        else:
            cp_raw, _ = _project_points_to_path(corners_np, wp)
            # CPU multiplies by 10 to match its progress scale; replicate.
            corner_progresses_np = (cp_raw.astype(np.float32) * 10.0)
        self._corners = jnp.asarray(corners_np.astype(np.float32))
        self._corner_progresses = jnp.asarray(corner_progresses_np)

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

        # Reward constants (survival shaping).
        self._v_ref = float(v_ref)
        self._w_quiet = float(w_quiet)
        self._k_a = float(k_a)
        self._p_hole = float(p_hole)
        self._p_survival = float(p_survival)
        self._quiet_th = float(quiet_th)
        self._success_hold_steps = int(success_hold_steps)
        self._solver_iterations = int(solver_iterations)
        self._ls_iterations = int(ls_iterations)
        self._speed_ema_alpha = float(speed_ema_alpha)

    # ---- Brax API --------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        (rng, k_spawn, k_theta, k_speed, k_alpha, k_beta,
         k_bias_ball, k_bias_joint) = jax.random.split(rng, 8)

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

        # Per-episode sensor biases (matches CPU `_obs_bias`).
        bias_ball = jax.random.uniform(
            k_bias_ball, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE,
        ).astype(jnp.float32)
        bias_joint = jax.random.uniform(
            k_bias_joint, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE,
        ).astype(jnp.float32)

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
        # Initial noisy ball pos (no per-step noise, just bias).
        ball_pos_noisy0 = ball_pos + bias_ball

        # First-step observation: use per-episode bias only (no per-step
        # sensor noise at reset — matches CPU which adds it inside step()).
        per_frame = self._build_per_frame_obs(
            pipeline_state.qpos[0], pipeline_state.qpos[1],
            ball_pos, ball_pos_noisy0, bias_joint,
            jnp.zeros(2, dtype=jnp.float32),
        )
        # Seed the buffer with [zero, zero, per_frame_reset] so the spawn
        # frame enters the stack history, matching SB3 VecFrameStack which
        # writes the reset obs into the freshest slot and then shifts.
        # Without this the first 3 GPU step obs lose the spawn frame.
        obs_buf0 = jnp.concatenate(
            [
                jnp.zeros((N_STACK - 2, OBS_DIM), dtype=jnp.float32),
                per_frame[None, :],
            ],
            axis=0,
        )
        stats = _EpStats(
            prev_ball_pos_clean=ball_pos,
            prev_ball_pos_noisy=ball_pos_noisy0,
            ema_speed=jnp.asarray(0.0, dtype=jnp.float32),
            obs_buf=obs_buf0,
            bias_ball=bias_ball,
            bias_joint=bias_joint,
            stable_steps=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            success=jnp.asarray(0.0, dtype=jnp.float32),
        )
        obs = self._stack_with_buffer(
            jnp.zeros((N_STACK - 1, OBS_DIM), dtype=jnp.float32), per_frame,
        )
        # All per-step metric keys must exist in reset for the EpisodeWrapper
        # to sum them consistently across the rollout.
        _zero = jnp.asarray(0.0, dtype=jnp.float32)
        metrics = {
            "success": _zero,        # per-step, fires once per episode
            "quiet_step": _zero,     # per-step indicator
            "stable_steps": stats.stable_steps.astype(jnp.float32),
            "r_quiet": _zero,
            "ball_speed": _zero,
            "safe_hole_margin": _zero,
            "term_hole": _zero,
            "term_timeout": _zero,
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

        # ---- Sample per-step sensor noise (matches CPU env). ----
        rng = state.info["rng"]
        rng, k_ball, k_joint = jax.random.split(rng, 3)
        ball_noise = jax.random.uniform(
            k_ball, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE,
        )
        joint_noise = jax.random.uniform(
            k_joint, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE,
        )

        # Noisy ball position (per-episode bias + per-step noise).
        ball_pos_noisy = ball_pos + stats.bias_ball + ball_noise
        # Noisy obs speed → EMA-smoothed → consumed by `quiet` term.
        obs_ball_speed = jnp.linalg.norm(
            ball_pos_noisy - stats.prev_ball_pos_noisy
        ) / jnp.maximum(dt, 1e-8)
        ema_speed = (
            self._speed_ema_alpha * stats.ema_speed
            + (1.0 - self._speed_ema_alpha) * obs_ball_speed
        )

        # Hole distance from CLEAN ball position — used for termination only,
        # NOT for shaping. (Diagnostic `safe_hole_margin` still logged.)
        hole_d = jnp.min(jnp.linalg.norm(self._holes - ball_pos[None, :], axis=1))
        hole_margin = hole_d - HOLE_RADIUS

        # Tilt magnitude from CLEAN qpos (diagnostic).
        alpha = pipeline_state.qpos[0]
        beta = pipeline_state.qpos[1]
        abs_tilt = jnp.sqrt(alpha * alpha + beta * beta)

        # Survival shaping: reward stays small per step but ramps up over the
        # 500-step episode horizon if the ball remains slow ("jiggly is fine").
        quiet = jnp.exp(-(ema_speed / self._v_ref) ** 2)
        r_action = -self._k_a * jnp.sum(action * action)

        # Hole termination: identical geometry to CPU env (`HOLE_RADIUS`).
        in_hole = hole_d < HOLE_RADIUS

        # Provisional reward (survival bonus added below once `timeout` known).
        reward = (
            self._w_quiet * quiet + r_action
            - jnp.where(in_hole, self._p_hole, 0.0)
        )

        # Quiet indicator (diagnostic + success counter).
        in_quiet = quiet > self._quiet_th
        stable_steps = jnp.where(
            in_quiet, stats.stable_steps + 1, jnp.int32(0)
        )

        step_count = stats.step_count + 1
        timeout = step_count >= self._episode_length
        done = (in_hole | timeout).astype(jnp.float32)
        # Survival bonus: full episode survived without falling in a hole.
        survived = (timeout & ~in_hole).astype(jnp.float32)
        reward = reward + self._p_survival * survived

        # Success fires ONCE per episode at the first crossing of
        # `success_hold_steps`. Matches CPU `episode/success_rate` semantics.
        crossed = (
            (stats.stable_steps < self._success_hold_steps)
            & (stable_steps >= self._success_hold_steps)
            & (stats.success < 0.5)
        )
        success_metric = crossed.astype(jnp.float32)
        success_ever = jnp.maximum(stats.success, success_metric)

        # Build the new per-frame obs (13-dim: 10-dim states + 3-dim ckpt).
        per_frame = self._build_per_frame_obs(
            alpha, beta, ball_pos, ball_pos_noisy,
            stats.bias_joint, joint_noise,
        )
        # Stack with the carried buffer of last (N_STACK - 1) frames → (40,).
        obs = self._stack_with_buffer(stats.obs_buf, per_frame)
        # Shift buffer: drop oldest, append current.
        new_obs_buf = jnp.concatenate(
            [stats.obs_buf[1:], per_frame[None, :]], axis=0,
        )

        new_stats = _EpStats(
            prev_ball_pos_clean=ball_pos,
            prev_ball_pos_noisy=ball_pos_noisy,
            ema_speed=ema_speed,
            obs_buf=new_obs_buf,
            bias_ball=stats.bias_ball,
            bias_joint=stats.bias_joint,
            stable_steps=stable_steps,
            step_count=step_count,
            success=success_ever,
        )
        # Per-step metrics. Brax's EpisodeWrapper sums them across the
        # rollout — divide by avg_episode_length in progress_fn for means
        # (or interpret directly when summing makes semantic sense, like
        # `success`, which fires once per episode).
        in_quiet_f = in_quiet.astype(jnp.float32)
        in_hole_f = in_hole.astype(jnp.float32)
        timeout_f = timeout.astype(jnp.float32)
        metrics = dict(state.metrics)
        metrics.update({
            "success": success_metric,        # 0/1, fires once per episode
            "quiet_step": in_quiet_f,         # 0/1 per step (fraction-quiet)
            "stable_steps": stable_steps.astype(jnp.float32),
            "r_quiet": quiet,
            # `ball_speed` reports the EMA-smoothed noisy speed used by the
            # reward, matching CPU's `_ball_speed`.
            "ball_speed": ema_speed.astype(jnp.float32),
            "safe_hole_margin": jnp.maximum(hole_margin, 0.0).astype(jnp.float32),
            # Termination indicators (Brax sums per-step → 0/1 per episode →
            # mean across episodes = termination rate).
            "term_hole": in_hole_f,
            "term_timeout": timeout_f,
        })
        info = {**state.info, "stats": new_stats, "rng": rng}
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done,
            metrics=metrics, info=info,
        )

    # ---- obs / geometry --------------------------------------------------

    def _project_to_path(self, ball_pos: jax.Array):
        """JAX equivalent of the CPU env's `compute_path_progress` direct
        projection branch (no raycasting / wall-occlusion). Returns
        ``(progress * 10, seg_idx, closest_pt)`` for parity with CPU
        progress scale.
        """
        seg_len_sq = jnp.maximum(
            jnp.sum(self._seg_vecs * self._seg_vecs, axis=-1), 1e-10,
        )
        rel = ball_pos[None, :] - self._seg_starts
        t = jnp.sum(rel * self._seg_vecs, axis=-1) / seg_len_sq
        t = jnp.clip(t, 0.0, 1.0)
        closest = self._seg_starts + t[:, None] * self._seg_vecs
        offsets = jnp.linalg.norm(closest - ball_pos[None, :], axis=-1)
        seg_idx = jnp.argmin(offsets)
        closest_pt = closest[seg_idx]
        progress = self._cum_dists[seg_idx] + t[seg_idx] * self._seg_lens[seg_idx]
        return progress * 10.0, seg_idx, closest_pt

    def _first_backward_safe_corner(self, ball_progress: jax.Array) -> jax.Array:
        """Largest-progress corner with progress ≤ ball_progress + ε.

        Falls back to the corner with smallest |gap| if no backward corner
        exists. Implementation is jit-compatible (no dynamic slicing).
        """
        eps = 0.005
        backward_mask = self._corner_progresses <= (ball_progress + eps)
        masked_progress = jnp.where(
            backward_mask, self._corner_progresses,
            jnp.full_like(self._corner_progresses, -jnp.inf),
        )
        backward_idx = jnp.argmax(masked_progress)
        forward_gap = jnp.abs(self._corner_progresses - ball_progress)
        forward_idx = jnp.argmin(forward_gap)
        has_backward = jnp.any(backward_mask)
        idx = jnp.where(has_backward, backward_idx, forward_idx)
        return self._corners[idx]

    def _build_per_frame_obs(
        self,
        alpha_clean: jax.Array,
        beta_clean: jax.Array,
        ball_pos_clean: jax.Array,
        ball_pos_noisy: jax.Array,
        bias_joint: jax.Array,
        joint_noise: jax.Array,
    ) -> jax.Array:
        """Build a 13-dim per-frame obs that mirrors CPU's {states, checkpoint}.

        Layout (all NOISY for ball-derived terms):
            [0:2]  joint_pos = (α, β) + bias_joint + joint_noise
            [2:4]  ball_pos_noisy
            [4:6]  vec from noisy ball to closest path point
            [6:8]  vec from noisy ball to next waypoint
            [8:10] vec from noisy ball to waypoint after next
            [10:12] vec from noisy ball to first safe corner BACKWARD
            [12]   distance to that backward corner
        """
        joint_pos = jnp.stack([alpha_clean, beta_clean]) + bias_joint + joint_noise
        # Path projection on CLEAN ball pos (matches CPU which uses true pos
        # internally for `compute_path_progress`; obs vec is then taken to
        # NOISY ball, so the sim-to-real noise still propagates into obs).
        ball_progress, seg_idx, closest_pt = self._project_to_path(ball_pos_clean)
        next_idx = jnp.minimum(seg_idx + 1, self._num_waypoints - 1)
        next_next_idx = jnp.minimum(seg_idx + 2, self._num_waypoints - 1)
        next_wp = self._waypoints[next_idx]
        next_next_wp = self._waypoints[next_next_idx]
        vec_closest = closest_pt - ball_pos_noisy
        vec_next = next_wp - ball_pos_noisy
        vec_next_next = next_next_wp - ball_pos_noisy
        backward_corner = self._first_backward_safe_corner(ball_progress)
        ckpt_vec = backward_corner - ball_pos_noisy
        ckpt_dist = jnp.linalg.norm(ckpt_vec)
        return jnp.stack([
            joint_pos[0], joint_pos[1],
            ball_pos_noisy[0], ball_pos_noisy[1],
            vec_closest[0], vec_closest[1],
            vec_next[0], vec_next[1],
            vec_next_next[0], vec_next_next[1],
            ckpt_vec[0], ckpt_vec[1], ckpt_dist,
        ]).astype(jnp.float32)

    @staticmethod
    def _stack_with_buffer(obs_buf: jax.Array, current_frame: jax.Array) -> jax.Array:
        """Concat last (N_STACK - 1) frames + current → (N_STACK · OBS_DIM,).

        Matches SB3 `VecFrameStack` ordering: oldest..newest along last axis.
        On reset the buffer is zero-filled, exactly as `StackedObservations`
        in stable-baselines3 does on its first step.
        """
        return jnp.concatenate([obs_buf.reshape(-1), current_frame], axis=0)

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
    # Frame-stacked obs: N_STACK frames × OBS_DIM features.
    @property
    def observation_size(self) -> int:
        return N_STACK * OBS_DIM

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
