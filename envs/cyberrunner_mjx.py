"""GPU-native CyberRunner env via MuJoCo MJX + Brax.

Optional, additive backend. `envs/cyberrunner.py` is untouched.

Scope:
    - prior_task = "stabilize" only
    - selectable prior_version: legacy or checkpoint_recovery
    - same MuJoCo geometry: walls, holes, waypoints reused verbatim
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
    CHECKPOINT_RECOVERY_OBS_DIM,
    FRAME_SKIP,
    HOLE_RADIUS,
    JOINT_ANGLE_NOISE,
    LEGACY_PRIOR_OBS_DIM,
    MARBLE_RADIUS,
    PRIOR_RECOVERY_ACTION_DELTA_PENALTY,
    PRIOR_RECOVERY_ACTION_PENALTY,
    PRIOR_RECOVERY_ALIVE_REWARD,
    PRIOR_RECOVERY_QUIET_THRESHOLD_SPEED,
    PRIOR_RECOVERY_TILT_PENALTY,
    PRIOR_RECOVERY_V_REF,
    PRIOR_RECOVERY_W_BASIN,
    PRIOR_RECOVERY_W_HOLD,
    PRIOR_RECOVERY_W_PROGRESS,
    PRIOR_RECOVERY_W_QUIET,
    PRIOR_VERSION_CHECKPOINT_RECOVERY,
    PRIOR_VERSION_LEGACY,
    PRIOR_VERSION_DENSE,
    PRIOR_VERSIONS,
    PRIOR_DENSE_ARRIVAL_BONUS,
    PRIOR_DENSE_PROGRESS_SCALE,
    PRIOR_DENSE_QUIET_SPEED,
    PRIOR_DENSE_SAFE_MARGIN_CLIP,
    PRIOR_DENSE_SAFE_MARGIN_COEF,
    PRIOR_DENSE_SPEED_COEF,
    PRIOR_DENSE_TOUCHING_WALL_BONUS,
    PRIOR_DENSE_WALL_CONTACT_MARGIN,
    RANGE_ALPHA,
    RANGE_BETA,
    TIMESTEP,
    WALL_RADIUS,
    DENSE_OBS_DIM,
    _project_points_to_path,
    build_model,
    compute_waypoint_distances,
    get_hard_layout,
    select_safe_checkpoints,
)


N_STACK = 3   # frame stacking, matches CPU `train_ppo.py --n_stack 3`
OBS_DIM = LEGACY_PRIOR_OBS_DIM  # legacy default; env.obs_dim is mode-specific


@struct.dataclass
class _EpStats:
    prev_ball_pos_clean: jnp.ndarray   # (2,)  true position, for clean speed
    prev_ball_pos_noisy: jnp.ndarray   # (2,)  noisy obs position, for EMA speed
    ema_speed: jnp.ndarray             # ()    EMA-smoothed noisy speed (reward source)
    obs_buf: jnp.ndarray               # (N_STACK - 1, OBS_DIM) — last 3 frames
    bias_ball: jnp.ndarray             # (2,)  per-episode ball-pos bias
    bias_joint: jnp.ndarray            # (2,)  per-episode joint-angle bias
    stable_steps: jnp.ndarray          # ()
    stable_steps_max: jnp.ndarray      # ()
    step_count: jnp.ndarray            # ()
    success: jnp.ndarray               # ()    sticky "ever crossed" flag
    prev_action: jnp.ndarray           # (2,)
    target: jnp.ndarray                # (2,)
    prev_target_dist: jnp.ndarray      # ()
    checkpoint_dist_min: jnp.ndarray   # ()
    inside_checkpoint_steps: jnp.ndarray
    speed_sum: jnp.ndarray
    # Sticky 0/1 flag: ball has entered the corner basin at least once
    # this episode. Used by the phased dense reward (approach vs stabilize).
    dense_arrived: jnp.ndarray         # ()  float32, 0.0 or 1.0
    # Path-segment index of the frozen dense target. Used by the dense obs
    # to expose the next waypoint along the path going toward the target.
    target_seg_idx: jnp.ndarray        # ()  int32


class CyberRunnerMJXEnv(PipelineEnv):
    """Brax-compatible MJX env for the stabilize prior task.

    Observation matches the CPU env (`envs/cyberrunner.py::_get_obs::states`):
        legacy per-frame is 13-dim: states(10) + checkpoint(3).
        checkpoint_recovery per-frame is 12-dim:
            [α, β, bx, by, ckpt_vec(2), ckpt_dist, vec_to_closest_path(2),
             prev_action(2), inside_ckpt]
        policy input is a 4-frame stack along the last axis,
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
        # Safe spawn for ALL prior versions: never start adjacent to a hole.
        # Matches CPU `train_ppo.py` default of 12 mm.
        prior_spawn_min_hole_margin: float = 0.012,
        prior_version: str = PRIOR_VERSION_LEGACY,
        # EMA smoothing α for the noisy ball speed used by the reward
        # (matches CPU `checkpoint_speed_ema_alpha=0.8`).
        speed_ema_alpha: float = 0.8,
        # Survival reward constants.
        v_ref: float = 0.03,           # legacy speed reference (m/s)
        w_quiet: float = 1.0,          # weight on the low-velocity term
        k_a: float = 0.005,            # action ℓ2 cost
        p_hole: float = 50.0,          # terminal hole penalty
        p_survival: float = 100.0,     # one-shot bonus on full-episode survival
        # `success` metric threshold on the quiet term.
        quiet_th: float = 0.5,
        quiet_threshold_speed: float = PRIOR_RECOVERY_QUIET_THRESHOLD_SPEED,
        w_progress: float = PRIOR_RECOVERY_W_PROGRESS,
        w_basin: float = PRIOR_RECOVERY_W_BASIN,
        w_hold: float = PRIOR_RECOVERY_W_HOLD,
        alive_reward: float = PRIOR_RECOVERY_ALIVE_REWARD,
        action_delta_penalty: float = PRIOR_RECOVERY_ACTION_DELTA_PENALTY,
        tilt_penalty: float = PRIOR_RECOVERY_TILT_PENALTY,
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
        if prior_version not in PRIOR_VERSIONS:
            raise ValueError(f"Unknown prior_version={prior_version!r}; expected one of {PRIOR_VERSIONS}")
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
        # clear holes by `prior_spawn_min_hole_margin` (12 mm by default for
        # every prior version — early policies cannot recover from a spawn
        # right next to a hole).
        spawn_np = self._build_spawn_bank(
            waypoints, holes, prior_spawn_source, prior_start_point_spacing,
            prior_spawn_min_hole_margin,
        )
        self._spawn_bank = jnp.asarray(spawn_np, dtype=jnp.float32)
        self._num_spawns = int(spawn_np.shape[0])

        # qpos layout: [alpha, beta, bx, by, bz, qw, qx, qy, qz]
        self._ball_height = float(0.0793)

        self._episode_length = int(episode_length)
        self._prior_version = prior_version
        if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY:
            self.obs_dim = CHECKPOINT_RECOVERY_OBS_DIM
        elif prior_version == PRIOR_VERSION_DENSE:
            self.obs_dim = DENSE_OBS_DIM
        else:
            self.obs_dim = LEGACY_PRIOR_OBS_DIM
        self._init_ball_speed = float(init_ball_speed)
        self._init_tilt_frac = float(init_tilt_frac)
        self._range_alpha = (float(RANGE_ALPHA[0]), float(RANGE_ALPHA[1]))
        self._range_beta = (float(RANGE_BETA[0]), float(RANGE_BETA[1]))

        # Reward constants (survival shaping).
        self._v_ref = float(PRIOR_RECOVERY_V_REF if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY else v_ref)
        self._w_quiet = float(PRIOR_RECOVERY_W_QUIET if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY else w_quiet)
        self._k_a = float(PRIOR_RECOVERY_ACTION_PENALTY if prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY else k_a)
        # dense uses a moderate hole penalty so the policy is willing to
        # navigate risky parts of the maze to reach the target. With p_hole
        # too high (e.g. 100) the agent freezes at a safe far-from-holes
        # spot rather than approach. The +arrival bonus is sized to
        # dominate the hole-penalty hit on the path to the target.
        if prior_version == PRIOR_VERSION_DENSE and p_hole == 50.0:
            p_hole = 25.0
        self._p_hole = float(p_hole)
        # Survival bonus is only meaningful for the legacy sparse reward —
        # dense has dense per-step shaping and would be destabilized
        # by a +100 terminal lump sum.
        if prior_version == PRIOR_VERSION_DENSE:
            p_survival = 0.0
        self._p_survival = float(p_survival)
        self._quiet_th = float(quiet_th)
        self._quiet_threshold_speed = float(quiet_threshold_speed)
        self._w_progress = float(w_progress)
        self._w_basin = float(w_basin)
        self._w_hold = float(w_hold)
        self._alive_reward = float(alive_reward)
        self._action_delta_penalty = float(action_delta_penalty)
        self._tilt_penalty = float(tilt_penalty)
        self._success_hold_steps = int(success_hold_steps)
        self._solver_iterations = int(solver_iterations)
        self._ls_iterations = int(ls_iterations)
        self._speed_ema_alpha = float(speed_ema_alpha)

        # dense reward constants (constant across episodes).
        self._dense_hold_reward = 1.0
        self._dense_stabilize_reward = 10.0
        self._dense_safe_hole_margin = 0.004

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
        ball_progress0, ball_seg_idx0, ball_on_path0 = self._project_to_path(ball_pos)
        target0 = self._first_backward_safe_corner(ball_progress0)
        # Project the frozen target onto the path to cache its segment index.
        # The dense obs uses this to select the next waypoint along the path
        # going toward the target.
        _, target_seg_idx0, _ = self._project_to_path(target0)
        # Seed `prev_target_dist` with the SAME metric used in step. For
        # dense (path-aware), match step's formula so the first
        # progress_delta is 0 and not a spurious cliff at episode start.
        eucl_to_target0 = jnp.linalg.norm(target0 - ball_pos)
        if self._prior_version == PRIOR_VERSION_DENSE:
            tgt_seg_start0 = self._seg_starts[target_seg_idx0]
            tgt_seg_vec0 = self._seg_vecs[target_seg_idx0]
            tgt_seg_len_sq0 = jnp.maximum(
                jnp.sum(tgt_seg_vec0 * tgt_seg_vec0), 1e-10,
            )
            t_target0 = jnp.clip(
                jnp.sum((target0 - tgt_seg_start0) * tgt_seg_vec0)
                / tgt_seg_len_sq0,
                0.0,
                1.0,
            )
            target_on_path0 = tgt_seg_start0 + t_target0 * tgt_seg_vec0
            s_target_raw0 = (
                self._cum_dists[target_seg_idx0]
                + t_target0 * self._seg_lens[target_seg_idx0]
            )
            tail_target0 = jnp.linalg.norm(target0 - target_on_path0)
            s_ball_raw0 = ball_progress0 / 10.0
            d_path0 = jnp.abs(s_ball_raw0 - s_target_raw0)
            d_off0 = jnp.linalg.norm(ball_pos - ball_on_path0)
            same_seg0 = ball_seg_idx0 == target_seg_idx0
            target_dist0 = jnp.where(
                same_seg0, eucl_to_target0, d_path0 + d_off0 + tail_target0,
            )
        else:
            target_dist0 = eucl_to_target0
        per_frame = self._build_per_frame_obs(
            pipeline_state.qpos[0], pipeline_state.qpos[1],
            ball_pos, ball_pos_noisy0, bias_joint,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            target0,
            target_seg_idx0,
        )
        # Seed the buffer with [zero, zero, per_frame_reset] so the spawn
        # frame enters the stack history, matching SB3 VecFrameStack which
        # writes the reset obs into the freshest slot and then shifts.
        # Without this the first 3 GPU step obs lose the spawn frame.
        obs_buf0 = jnp.concatenate(
            [
                jnp.zeros((N_STACK - 2, self.obs_dim), dtype=jnp.float32),
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
            stable_steps_max=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            success=jnp.asarray(0.0, dtype=jnp.float32),
            prev_action=jnp.zeros(2, dtype=jnp.float32),
            target=target0.astype(jnp.float32),
            prev_target_dist=target_dist0.astype(jnp.float32),
            checkpoint_dist_min=target_dist0.astype(jnp.float32),
            inside_checkpoint_steps=jnp.asarray(0, dtype=jnp.int32),
            speed_sum=jnp.asarray(0.0, dtype=jnp.float32),
            dense_arrived=jnp.asarray(0.0, dtype=jnp.float32),
            target_seg_idx=target_seg_idx0.astype(jnp.int32),
        )
        obs = self._stack_with_buffer(
            jnp.zeros((N_STACK - 1, self.obs_dim), dtype=jnp.float32), per_frame,
        )
        # All per-step metric keys must exist in reset for the EpisodeWrapper
        # to sum them consistently across the rollout.
        _zero = jnp.asarray(0.0, dtype=jnp.float32)
        metrics = {
            "success": _zero,        # per-step, fires once per episode
            "quiet_step": _zero,     # per-step indicator
            "stable_steps": stats.stable_steps.astype(jnp.float32),
            "stable_steps_final": _zero,
            "stable_steps_max": _zero,
            "r_quiet": _zero,
            "r_progress": _zero,
            "r_basin": _zero,
            "r_hold": _zero,
            "r_action": _zero,
            "r_action_delta": _zero,
            "r_tilt": _zero,
            "ball_speed": _zero,
            "observed_speed": _zero,
            "checkpoint_dist_final": _zero,
            "checkpoint_dist_min": _zero,
            "inside_checkpoint": _zero,
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
        target = stats.target
        checkpoint_dist = jnp.linalg.norm(target - ball_pos_noisy)
        checkpoint_dist_min = jnp.minimum(stats.checkpoint_dist_min, checkpoint_dist)
        inside_checkpoint = checkpoint_dist < 0.015
        r_progress = self._w_progress * jnp.maximum(stats.prev_target_dist - checkpoint_dist, 0.0)
        r_basin = jnp.where(inside_checkpoint, self._w_basin, 0.0)
        r_hold = jnp.asarray(0.0, dtype=jnp.float32)
        r_action_delta = -self._action_delta_penalty * jnp.sum((action - stats.prev_action) ** 2)
        r_tilt = -self._tilt_penalty * abs_tilt * abs_tilt

        # Hole termination: identical geometry to CPU env (`HOLE_RADIUS`).
        in_hole = hole_d < HOLE_RADIUS

        # Provisional reward (survival bonus added below once `timeout` known).
        r_hole = -jnp.where(in_hole, self._p_hole, 0.0)

        # Quiet indicator (diagnostic + success counter).
        if self._prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY:
            in_quiet = ema_speed < self._quiet_threshold_speed
            stable_condition = inside_checkpoint & in_quiet
            r_hold = jnp.where(stable_condition, self._w_hold, 0.0)
            reward = (
                self._alive_reward + r_progress + r_basin
                + self._w_quiet * quiet + r_hold
                + r_action + r_action_delta + r_tilt + r_hole
            )
            stable_steps_pre = jnp.where(
                stable_condition, stats.stable_steps + 1, jnp.int32(0)
            )
            stable_steps = stable_steps_pre
        elif self._prior_version == PRIOR_VERSION_DENSE:
            # Dense reward, single-phase, PATH-AWARE: progress shaping along
            # the safe waypoint polyline + one-time arrival bonus + hole
            # penalty. Straight-line Euclidean shaping was telling the agent
            # to dive across walls/holes when the corner sat off-polyline a
            # few segments back. Now ``dense_target_dist`` is path-distance
            # along the polyline plus a constant off-path tail, switched to
            # direct Euclidean once the ball reaches the target's segment
            # (so it can leave the path to enter the corner basin).
            dense_target = stats.target
            target_seg_idx = stats.target_seg_idx

            # Project the (frozen) corner onto its waypoint segment to get
            # path arc length and the constant off-path tail. Cheap O(1).
            tgt_seg_start = self._seg_starts[target_seg_idx]
            tgt_seg_vec = self._seg_vecs[target_seg_idx]
            tgt_seg_len_sq = jnp.maximum(
                jnp.sum(tgt_seg_vec * tgt_seg_vec), 1e-10,
            )
            t_target = jnp.clip(
                jnp.sum((dense_target - tgt_seg_start) * tgt_seg_vec)
                / tgt_seg_len_sq,
                0.0,
                1.0,
            )
            target_on_path = tgt_seg_start + t_target * tgt_seg_vec
            s_target_raw = (
                self._cum_dists[target_seg_idx]
                + t_target * self._seg_lens[target_seg_idx]
            )
            tail_target = jnp.linalg.norm(dense_target - target_on_path)

            # Project ball onto the polyline (CLEAN ball pos for parity
            # with the obs and CPU env). `_project_to_path` returns
            # ``progress * 10`` for parity with CPU's scaled progress.
            ball_progress, ball_seg_idx, ball_on_path = self._project_to_path(
                ball_pos
            )
            s_ball_raw = ball_progress / 10.0
            d_path = jnp.abs(s_ball_raw - s_target_raw)
            d_off = jnp.linalg.norm(ball_pos - ball_on_path)

            # Same-segment → direct Euclidean (final leg into the corner).
            # Different segments → path-distance + off-path offset + tail.
            eucl_to_target = jnp.linalg.norm(dense_target - ball_pos)
            same_seg = ball_seg_idx == target_seg_idx
            dense_target_dist = jnp.where(
                same_seg, eucl_to_target, d_path + d_off + tail_target,
            ).astype(jnp.float32)

            progress_delta = stats.prev_target_dist - dense_target_dist
            # Progress shaping gated on `dense_arrived`: fires only BEFORE
            # the first basin entry (see CPU comment for rationale — kills
            # the entry/exit oscillation incentive observed in videos).
            not_arrived_yet = stats.dense_arrived < 0.5
            r_dense_progress = jnp.where(
                not_arrived_yet,
                PRIOR_DENSE_PROGRESS_SCALE * progress_delta,
                jnp.float32(0.0),
            ).astype(jnp.float32)

            # Arrival uses Euclidean distance to the actual corner (the
            # corner lives off-path, so path-distance is wrong here).
            arrival_radius = jnp.asarray(0.015, dtype=jnp.float32)  # = checkpoint_radius
            arrived_now = eucl_to_target < arrival_radius
            fresh_arrival = arrived_now & (stats.dense_arrived < 0.5)
            r_arrival = jnp.where(
                fresh_arrival, PRIOR_DENSE_ARRIVAL_BONUS, 0.0,
            ).astype(jnp.float32)
            new_dense_arrived = jnp.maximum(
                stats.dense_arrived, arrived_now.astype(jnp.float32)
            )

            # No Phase B: keep stable_steps zeroed (still carried in the
            # pytree for parity with other branches and diagnostics).
            # Drive `success` off fresh_arrival by spoofing
            # `stable_steps_pre` to the threshold on the arrival step —
            # the shared crossed-threshold check below then fires once.
            in_quiet = ema_speed < PRIOR_DENSE_QUIET_SPEED
            stable_condition = arrived_now & in_quiet  # diagnostic only
            stable_steps = jnp.int32(0)
            stable_steps_pre = jnp.where(
                fresh_arrival, self._success_hold_steps, jnp.int32(0)
            )

            reward = r_dense_progress + r_arrival + r_hole
        else:
            in_quiet = quiet > self._quiet_th
            stable_condition = in_quiet
            reward = self._w_quiet * quiet + r_action + r_hole
            stable_steps_pre = jnp.where(
                stable_condition, stats.stable_steps + 1, jnp.int32(0)
            )
            stable_steps = stable_steps_pre
        stable_steps_max = jnp.maximum(stats.stable_steps_max, stable_steps)

        step_count = stats.step_count + 1
        timeout = step_count >= self._episode_length
        done = (in_hole | timeout).astype(jnp.float32)
        # Survival bonus: full episode survived without falling in a hole.
        survived = (timeout & ~in_hole).astype(jnp.float32)
        reward = reward + self._p_survival * survived

        # Success fires ONCE per episode at the first crossing of
        # `success_hold_steps`. Matches CPU `episode/success_rate` semantics.
        # Use `stable_steps_pre` (before any dense post-saturation
        # reset) so the threshold-crossing edge is detected on all branches.
        crossed = (
            (stats.stable_steps < self._success_hold_steps)
            & (stable_steps_pre >= self._success_hold_steps)
            & (stats.success < 0.5)
        )
        success_metric = crossed.astype(jnp.float32)
        success_ever = jnp.maximum(stats.success, success_metric)

        # Build the new per-frame obs (legacy 13-dim: states(10)+ckpt(3);
        # dense 11-dim: states(8)+ckpt(3); checkpoint_recovery 12-dim).
        per_frame = self._build_per_frame_obs(
            alpha, beta, ball_pos, ball_pos_noisy,
            stats.bias_joint, joint_noise, action, stats.target,
            stats.target_seg_idx,
        )
        # Stack with the carried buffer of last (N_STACK - 1) frames → (40,).
        obs = self._stack_with_buffer(stats.obs_buf, per_frame)
        # Shift buffer: drop oldest, append current.
        new_obs_buf = jnp.concatenate(
            [stats.obs_buf[1:], per_frame[None, :]], axis=0,
        )

        # Dense version keeps the target frozen at reset (the spawn-time
        # first backward safe corner) and tracks the sticky `dense_arrived`
        # flag. Other versions also keep the target frozen.
        if self._prior_version == PRIOR_VERSION_DENSE:
            next_target = stats.target
            next_prev_target_dist = dense_target_dist.astype(jnp.float32)
            next_dense_arrived = new_dense_arrived
        else:
            next_target = stats.target
            next_prev_target_dist = checkpoint_dist.astype(jnp.float32)
            next_dense_arrived = stats.dense_arrived
        new_stats = _EpStats(
            prev_ball_pos_clean=ball_pos,
            prev_ball_pos_noisy=ball_pos_noisy,
            ema_speed=ema_speed,
            obs_buf=new_obs_buf,
            bias_ball=stats.bias_ball,
            bias_joint=stats.bias_joint,
            stable_steps=stable_steps,
            stable_steps_max=stable_steps_max,
            step_count=step_count,
            success=success_ever,
            prev_action=action,
            target=next_target,
            prev_target_dist=next_prev_target_dist,
            checkpoint_dist_min=checkpoint_dist_min.astype(jnp.float32),
            inside_checkpoint_steps=stats.inside_checkpoint_steps + inside_checkpoint.astype(jnp.int32),
            speed_sum=stats.speed_sum + ema_speed,
            dense_arrived=next_dense_arrived,
            target_seg_idx=stats.target_seg_idx,
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
            "stable_steps_final": jnp.where(done > 0.5, stable_steps, 0).astype(jnp.float32),
            "stable_steps_max": jnp.where(done > 0.5, stable_steps_max, 0).astype(jnp.float32),
            "r_quiet": quiet,
            "r_progress": r_progress.astype(jnp.float32),
            "r_basin": r_basin.astype(jnp.float32),
            "r_hold": r_hold.astype(jnp.float32),
            "r_action": r_action.astype(jnp.float32),
            "r_action_delta": r_action_delta.astype(jnp.float32),
            "r_tilt": r_tilt.astype(jnp.float32),
            # `ball_speed` reports the EMA-smoothed noisy speed used by the
            # reward, matching CPU's `_ball_speed`.
            "ball_speed": ema_speed.astype(jnp.float32),
            "observed_speed": ema_speed.astype(jnp.float32),
            "checkpoint_dist_final": jnp.where(done > 0.5, checkpoint_dist, 0.0).astype(jnp.float32),
            "checkpoint_dist_min": jnp.where(done > 0.5, checkpoint_dist_min, 0.0).astype(jnp.float32),
            "inside_checkpoint": inside_checkpoint.astype(jnp.float32),
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
        prev_action: jax.Array,
        target_override: jax.Array | None = None,
        target_seg_idx_override: jax.Array | None = None,
    ) -> jax.Array:
        """Build a per-frame obs that mirrors the selected CPU prior mode.

        Legacy layout (all NOISY for ball-derived terms):
            [0:2]  joint_pos = (α, β) + bias_joint + joint_noise
            [2:4]  ball_pos_noisy
            [4:6]  vec from noisy ball to closest path point
            [6:8]  vec from noisy ball to next waypoint
            [8:10] vec from noisy ball to waypoint after next
            [10:12] vec from noisy ball to first safe corner BACKWARD
            [12]   distance to that backward corner

        checkpoint_recovery replaces the next-waypoint vectors with previous
        action and an inside-checkpoint flag, for a 12-dim per-frame obs.
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
        if target_override is not None:
            backward_corner = target_override
        ckpt_vec = backward_corner - ball_pos_noisy
        ckpt_dist = jnp.linalg.norm(ckpt_vec)
        if self._prior_version == PRIOR_VERSION_CHECKPOINT_RECOVERY:
            inside = (ckpt_dist < 0.015).astype(jnp.float32)
            return jnp.stack([
                joint_pos[0], joint_pos[1],
                ball_pos_noisy[0], ball_pos_noisy[1],
                ckpt_vec[0], ckpt_vec[1], ckpt_dist,
                vec_closest[0], vec_closest[1],
                prev_action[0], prev_action[1],
                inside,
            ]).astype(jnp.float32)
        if self._prior_version == PRIOR_VERSION_DENSE:
            # 11-dim: states(8) + ckpt(3). The "next waypoint" vector points
            # to the next path node going TOWARD the (frozen) target:
            #   ball_seg > target_seg → waypoints[ball_seg]   (step backward)
            #   ball_seg < target_seg → waypoints[ball_seg+1] (step forward)
            #   equal                 → target itself
            tgt_seg = (
                target_seg_idx_override
                if target_seg_idx_override is not None
                else seg_idx
            )
            wp_at_seg = self._waypoints[seg_idx]
            wp_at_seg_p1 = self._waypoints[
                jnp.minimum(seg_idx + 1, self._num_waypoints - 1)
            ]
            target_for_obs = (
                target_override if target_override is not None else backward_corner
            )
            next_wp_toward_target = jnp.where(
                seg_idx > tgt_seg, wp_at_seg,
                jnp.where(seg_idx < tgt_seg, wp_at_seg_p1, target_for_obs),
            )
            vec_to_next_wp_to_target = next_wp_toward_target - ball_pos_noisy
            return jnp.stack([
                joint_pos[0], joint_pos[1],
                ball_pos_noisy[0], ball_pos_noisy[1],
                vec_closest[0], vec_closest[1],
                vec_to_next_wp_to_target[0], vec_to_next_wp_to_target[1],
                ckpt_vec[0], ckpt_vec[1], ckpt_dist,
            ]).astype(jnp.float32)
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
        return N_STACK * self.obs_dim

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
