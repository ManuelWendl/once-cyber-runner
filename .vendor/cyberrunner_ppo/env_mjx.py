"""MJX/Brax environment for Cyberrunner.

Mirrors env_mujoco.py logic exactly (physics, maze, reward, observation,
termination), but expressed in JAX/MJX so Brax PPO can jit + vmap the rollout
across thousands of envs in parallel.

Source-of-truth reference is env_mujoco.py in this repo. Path-progress
raycast structure is taken verbatim from cyberrunner_mjx/utils/raycast.py
(proven fast vmap-of-vmap), augmented with the cleaned env's direct-projection
fallback and reward-tracking bugfixes.
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import envs
from mujoco import mjx

from env_mujoco import (
    BALL_POS_NOISE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    FRAME_SKIP,
    GOAL_BONUS,
    GOAL_THRESHOLD,
    HOLE_RADIUS,
    JOINT_ANGLE_NOISE,
    MARBLE_RADIUS,
    PROGRESS_SCALE,
    RANGE_ALPHA,
    RANGE_BETA,
    WALL_RADIUS,
    build_model,
    compute_waypoint_distances,
    get_hard_layout,
    get_layout,
)


# ============================================================================
# SAFE-PRIOR HELPERS (NumPy, host-side; called once at __init__)
# Ported from envs/cyberrunner.py to keep the corner set bit-for-bit
# comparable to the project's existing dense-prior runs. Only the
# `include_corridors=False` path is needed here, so the implementation below
# keeps the corner branch and drops the single-wall NMS pass.
# ============================================================================


def _project_points_to_path(points: np.ndarray, waypoints: np.ndarray):
    """Project points onto the polyline. Returns (progress, offset, seg_idx).

    Verbatim algorithm from envs/cyberrunner.py:_project_points_to_path,
    trimmed to the (return_closest=False, return_seg_idx=True) caller.
    """
    seg_starts = waypoints[:-1]
    seg_ends = waypoints[1:]
    seg_vecs = seg_ends - seg_starts
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    cum_distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    rel = points[:, None, :] - seg_starts[None, :, :]
    seg_len_sq = np.maximum(np.sum(seg_vecs**2, axis=1), 1e-10)
    t = np.sum(rel * seg_vecs[None, :, :], axis=-1) / seg_len_sq[None, :]
    t_clipped = np.clip(t, 0.0, 1.0)
    closest = seg_starts[None, :, :] + t_clipped[..., None] * seg_vecs[None, :, :]
    offsets = np.linalg.norm(points[:, None, :] - closest, axis=-1)
    best_seg = np.argmin(offsets, axis=1)
    best_offset = offsets[np.arange(len(points)), best_seg]
    best_t = t_clipped[np.arange(len(points)), best_seg]
    progress = cum_distances[best_seg] + best_t * seg_lengths[best_seg]
    return progress.astype(np.float32), best_offset.astype(np.float32), best_seg.astype(np.int32)


def _point_to_segment_distance(points, starts, ends):
    seg_vecs = ends - starts
    seg_len_sq = np.maximum(np.sum(seg_vecs**2, axis=1), 1e-10)
    rel = points[:, None, :] - starts[None, :, :]
    t = np.sum(rel * seg_vecs[None, :, :], axis=-1) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[..., None] * seg_vecs[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def _segment_crosses_walls(p, q, wall_starts, wall_ends) -> bool:
    r = q - p
    s = wall_ends - wall_starts
    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    nonpar = np.abs(rxs) > 1e-10
    qmp = wall_starts - p
    t = (qmp[:, 0] * s[:, 1] - qmp[:, 1] * s[:, 0]) / np.where(nonpar, rxs, 1.0)
    u = (qmp[:, 0] * r[1] - qmp[:, 1] * r[0]) / np.where(nonpar, rxs, 1.0)
    return bool((nonpar & (t > 0) & (t < 1) & (u > 0) & (u < 1)).any())


def select_corner_checkpoints(
    waypoints: np.ndarray,
    holes: np.ndarray,
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    grid_res: float = 0.002,
) -> np.ndarray:
    """Return the strict-corner safe checkpoints (sorted by path progress).

    Port of envs/cyberrunner.py:select_safe_checkpoints with
    include_corridors=False. Only the corner branch is materialised; the
    `reward_every_n_waypoints` knob folds into the corner NMS spacing as
    `min_sep_c = max(0.010, 0.018 - 0.001 * 3) = 0.015` (the project's
    `reward_every_n_waypoints=3` default).
    """
    corner_slack = 0.003
    corridor_slack = 0.010
    touch_wall = WALL_RADIUS + MARBLE_RADIUS
    touch_edge = MARBLE_RADIUS

    xs = np.arange(MARBLE_RADIUS, BOARD_WIDTH - MARBLE_RADIUS + 1e-9, grid_res, dtype=np.float32)
    ys = np.arange(MARBLE_RADIUS, BOARD_HEIGHT - MARBLE_RADIUS + 1e-9, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    cx = candidates[:, 0]
    cy = candidates[:, 1]

    hx0 = walls_h[:, 0][None, :]
    hx1 = walls_h[:, 1][None, :]
    hy = walls_h[:, 2][None, :]
    in_x = (cx[:, None] >= hx0 - 1e-6) & (cx[:, None] <= hx1 + 1e-6)
    dy = hy - cy[:, None]
    ray_up = np.where(in_x & (dy > 0), dy, np.inf).min(axis=1)
    ray_down = np.where(in_x & (dy < 0), -dy, np.inf).min(axis=1)

    vy0 = walls_v[:, 0][None, :]
    vy1 = walls_v[:, 1][None, :]
    vx = walls_v[:, 2][None, :]
    in_y = (cy[:, None] >= vy0 - 1e-6) & (cy[:, None] <= vy1 + 1e-6)
    dx = vx - cx[:, None]
    ray_right = np.where(in_y & (dx > 0), dx, np.inf).min(axis=1)
    ray_left = np.where(in_y & (dx < 0), -dx, np.inf).min(axis=1)

    gap_up = np.minimum(ray_up - touch_wall, BOARD_HEIGHT - cy - touch_edge)
    gap_down = np.minimum(ray_down - touch_wall, cy - touch_edge)
    gap_right = np.minimum(ray_right - touch_wall, BOARD_WIDTH - cx - touch_edge)
    gap_left = np.minimum(ray_left - touch_wall, cx - touch_edge)

    close_up = gap_up < corner_slack
    close_down = gap_down < corner_slack
    close_right = gap_right < corner_slack
    close_left = gap_left < corner_slack

    in_h_corridor = (gap_up < corridor_slack) & (gap_down < corridor_slack)
    in_v_corridor = (gap_left < corridor_slack) & (gap_right < corridor_slack)

    is_corner = (
        (close_up | close_down) & (close_right | close_left)
        & ~(in_h_corridor & in_v_corridor)
    )

    h_starts = np.stack([walls_h[:, 0], walls_h[:, 2]], axis=1)
    h_ends = np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1)
    v_starts = np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1)
    v_ends = np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1)
    all_wall_dist = _point_to_segment_distance(
        candidates, np.vstack([h_starts, v_starts]), np.vstack([h_ends, v_ends])
    )
    wall_clearance = all_wall_dist - touch_wall
    edge_clearance = np.minimum.reduce(
        [cx, BOARD_WIDTH - cx, cy, BOARD_HEIGHT - cy]
    ) - touch_edge

    clearance_margin = HOLE_RADIUS + MARBLE_RADIUS + np.where(is_corner, 0.006, 0.010)
    hole_dist_mat = np.linalg.norm(candidates[:, None, :] - holes[None, :, :], axis=-1)
    hole_clearance = hole_dist_mat.min(axis=1) - clearance_margin
    at_risk = np.where(hole_clearance < 0)[0]
    if len(at_risk) > 0:
        wall_s = np.vstack([h_starts, v_starts])
        wall_e = np.vstack([h_ends, v_ends])
        for i in at_risk:
            margin_i = clearance_margin[i]
            close_hole_idxs = np.where(hole_dist_mat[i] < margin_i)[0]
            worst = -np.inf
            for j in close_hole_idxs:
                if _segment_crosses_walls(candidates[i], holes[j], wall_s, wall_e):
                    continue
                worst = max(worst, margin_i - hole_dist_mat[i, j])
            hole_clearance[i] = -worst if worst > -np.inf else margin_i

    progress, path_offset, _ = _project_points_to_path(candidates, waypoints)
    max_path_offset = 0.065
    pre_valid = (
        (hole_clearance > 0.0)
        & (wall_clearance >= -1e-6)
        & (edge_clearance >= -1e-6)
        & (path_offset < max_path_offset)
        & is_corner
    )
    if not np.any(pre_valid):
        return np.zeros((0, 2), dtype=np.float32)

    candidates = candidates[pre_valid]
    progress = progress[pre_valid]
    hole_clearance = hole_clearance[pre_valid]
    gap_up_v = gap_up[pre_valid]
    gap_down_v = gap_down[pre_valid]
    gap_left_v = gap_left[pre_valid]
    gap_right_v = gap_right[pre_valid]
    touch_score = np.minimum.reduce([gap_up_v, gap_down_v, gap_left_v, gap_right_v])
    quality = 0.5 * hole_clearance - 10.0 * touch_score

    # Greedy NMS (corners-only spacing, matches reward_every_n_waypoints=3 default).
    min_sep = 0.015
    selected_idx: list[int] = []
    for idx in np.argsort(-quality):
        cand = candidates[idx]
        if not selected_idx:
            selected_idx.append(int(idx))
            continue
        sel = candidates[np.asarray(selected_idx)]
        if np.all(np.linalg.norm(sel - cand, axis=1) >= min_sep):
            selected_idx.append(int(idx))
    sel_arr = np.asarray(selected_idx, dtype=np.int32)
    progress_order = np.argsort(progress[sel_arr])
    return candidates[sel_arr][progress_order].astype(np.float32)


# ============================================================================
# RAYCAST HELPERS — copied verbatim from cyberrunner_mjx/utils/raycast.py:7-100
# ============================================================================


def ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end):
    """Distance from ray_origin to ray-segment intersection, or inf."""
    seg_vec = seg_end - seg_start
    denom = ray_dir[0] * seg_vec[1] - ray_dir[1] * seg_vec[0]
    parallel = jnp.abs(denom) < 1e-8
    diff = seg_start - ray_origin
    t = (diff[0] * seg_vec[1] - diff[1] * seg_vec[0]) / (denom + 1e-10)
    s = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / (denom + 1e-10)
    valid = (~parallel) & (t > 0) & (s >= 0) & (s <= 1)
    return jnp.where(valid, t, jnp.inf)


def ray_circle_intersection(ray_origin, ray_dir, circle_center, radius):
    """Distance from ray_origin to ray-circle entry, or inf."""
    oc = ray_origin - circle_center
    a = jnp.dot(ray_dir, ray_dir)
    b = 2.0 * jnp.dot(oc, ray_dir)
    c = jnp.dot(oc, oc) - radius**2
    disc = b * b - 4 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, jnp.inf))
    return jnp.where(disc < 0, jnp.inf, t)


# ============================================================================
# PATH PROGRESS — JAX port of env_mujoco.py:264-434
# ============================================================================


@partial(jax.jit, static_argnames=("num_rays",))
def compute_path_progress_jax(
    marble_pos,
    waypoints,
    seg_lengths,
    cum_distances,
    walls_h,
    walls_v,
    holes,
    num_rays: int = 32,
    hole_radius: float = HOLE_RADIUS,
    proj_threshold: float = 0.002,
):
    """Path progress with direct-projection fallback + raycast occlusion check.

    Mirrors the algorithm in env_mujoco.py:264-434:
      1. Direct projection onto every segment; if marble is within `proj_threshold`
         of the path, return the projection result directly (cleaned env L319-323).
      2. Otherwise, cast `num_rays` rays. For each ray, find the closest path
         intersection that is closer than the nearest wall/hole obstacle
         (cleaned env L401). Pick the ray with the minimum visible path distance.
      3. If no ray finds a visible path, return -1 sentinel for progress.

    Returns:
        progress: scalar float (path-distance * 10), -1 if not detected.
        seg_idx: int — which path segment was hit (0 if not detected).
        param: float in [0, 1] along that segment.
        closest_point: [2] float position on the path.
        found: bool — whether a valid path point was located.
    """
    seg_starts = waypoints[:-1]                       # [S, 2]
    seg_ends = waypoints[1:]                          # [S, 2]
    seg_vecs = seg_ends - seg_starts                  # [S, 2]

    # === DIRECT PROJECTION (cleaned env L294-323) ===
    to_marble = marble_pos - seg_starts                                  # [S, 2]
    seg_len_sq = jnp.sum(seg_vecs**2, axis=1)                            # [S]
    t_proj = jnp.sum(to_marble * seg_vecs, axis=1) / jnp.maximum(seg_len_sq, 1e-10)
    t_proj_clipped = jnp.clip(t_proj, 0.0, 1.0)                           # [S]
    closest_on_seg = seg_starts + t_proj_clipped[:, None] * seg_vecs      # [S, 2]
    dist_to_seg = jnp.linalg.norm(marble_pos - closest_on_seg, axis=1)    # [S]
    proj_seg = jnp.argmin(dist_to_seg)
    proj_min_dist = dist_to_seg[proj_seg]
    proj_param = t_proj_clipped[proj_seg]
    proj_progress = cum_distances[proj_seg] + proj_param * seg_lengths[proj_seg]
    proj_closest = closest_on_seg[proj_seg]

    # === RAYCAST (vmap-of-vmap, mirroring cyberrunner_mjx/utils/raycast.py:145-221) ===
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_rays, endpoint=False)
    ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)      # [R, 2]

    def cast_single_ray(direction):
        # Walls — vmap over horizontal walls (x_start, x_end, y)
        h_dists = jax.vmap(
            lambda w: ray_segment_intersection(
                marble_pos, direction,
                jnp.array([w[0], w[2]]), jnp.array([w[1], w[2]]),
            )
        )(walls_h)
        # Walls — vmap over vertical walls (y_start, y_end, x)
        v_dists = jax.vmap(
            lambda w: ray_segment_intersection(
                marble_pos, direction,
                jnp.array([w[2], w[0]]), jnp.array([w[2], w[1]]),
            )
        )(walls_v)
        wall_dist = jnp.min(jnp.concatenate([h_dists, v_dists]))

        # Holes — vmap over hole centers
        hole_dists = jax.vmap(
            lambda hp: ray_circle_intersection(marble_pos, direction, hp, hole_radius)
        )(holes)
        hole_dist = jnp.min(hole_dists)

        obstacle_dist = jnp.minimum(wall_dist, hole_dist)

        # Path intersections — vmap over segments, return (dist_along_ray, seg_param)
        def intersect_seg(start, end):
            d = ray_segment_intersection(marble_pos, direction, start, end)
            seg_v = end - start
            seg_l_sq = jnp.sum(seg_v**2)
            hit = marble_pos + direction * d
            s = jnp.dot(hit - start, seg_v) / jnp.maximum(seg_l_sq, 1e-8)
            return d, jnp.clip(s, 0.0, 1.0)

        path_dists, path_params = jax.vmap(intersect_seg)(seg_starts, seg_ends)
        # Occlusion: discard intersections behind walls/holes (cleaned env L401)
        valid = path_dists < obstacle_dist
        path_dists_filtered = jnp.where(valid, path_dists, jnp.inf)
        min_idx = jnp.argmin(path_dists_filtered)
        return path_dists_filtered[min_idx], min_idx, path_params[min_idx]

    ray_min_dists, ray_seg_idxs, ray_params_per_ray = jax.vmap(cast_single_ray)(ray_dirs)
    # Across all rays, pick the one with the closest visible path hit
    closest_ray = jnp.argmin(ray_min_dists)
    ray_dist = ray_min_dists[closest_ray]
    ray_seg = ray_seg_idxs[closest_ray]
    ray_param = ray_params_per_ray[closest_ray]
    ray_found = ray_dist < jnp.inf

    ray_progress = cum_distances[ray_seg] + ray_param * seg_lengths[ray_seg]
    ray_closest = seg_starts[ray_seg] + ray_param * (seg_ends[ray_seg] - seg_starts[ray_seg])

    # === SELECT BRANCH (replaces both Python `if`s in cleaned env L319, L414) ===
    use_proj = proj_min_dist < proj_threshold
    progress = jnp.where(
        use_proj,
        proj_progress * 10.0,
        jnp.where(ray_found, ray_progress * 10.0, -1.0),
    )
    seg_idx = jnp.where(use_proj, proj_seg, jnp.where(ray_found, ray_seg, 0))
    param = jnp.where(use_proj, proj_param, jnp.where(ray_found, ray_param, 0.0))
    closest = jnp.where(
        use_proj,
        proj_closest,
        jnp.where(ray_found, ray_closest, jnp.zeros(2)),
    )
    found = use_proj | ray_found
    return progress, seg_idx, param, closest, found


# ============================================================================
# MJX ENVIRONMENT — Brax envs.Env interface
# ============================================================================


class CyberrunnerMJXEnv(envs.Env):
    """MJX env for Brax PPO. Reuses build_model + maze data from env_mujoco.py.

    Single marble, hard layout, 10-dim state observation. Reset/step are jitted.
    Brax's wrappers (VmapWrapper, EpisodeWrapper, AutoResetWrapper) handle
    batching and timeout truncation; we only emit `done=1.0` on hole or goal.
    """

    def __init__(
        self,
        episode_length: int = 2000,
        randomize_init_pos: bool = True,
        num_rays: int = 32,
        num_envs_hint: int = 4096,
        history_length: int = 5,
        maze_layout: str = "hard",
        safe_prior: bool = False,
        safe_prior_strategy: str = "exp_d",
        safe_prior_sigma: float = 0.02,
        init_ball_speed: float = 0.0,
        init_tilt_frac: float = 0.0,
        tilt_bumps: bool = False,
        tilt_bump_prob: float = 0.0,
        tilt_bump_magnitude: float = 0.0,
    ):
        self.episode_length = episode_length
        self.randomize_init_pos = randomize_init_pos
        self.num_rays = num_rays
        self.num_envs_hint = num_envs_hint
        self.history_length = history_length
        self._frame_dim = 6  # joint(2) + ball(2) + action(2)
        self.maze_layout = str(maze_layout)

        # Safe-prior task: when enabled, the env spawns with mild random tilt
        # and ball velocity. The reward depends on which strategy is selected:
        #   - "exp_d":       reward = exp(-‖ball - frozen_target‖). Flat over
        #                    board scale; baseline.
        #   - "exp_d_sigma": reward = exp(-‖ball - frozen_target‖ / sigma).
        #                    Sharp basin around the corner; forces the policy
        #                    to actually reach the target.
        #   - "survival":    reward = 1.0 per alive step; no target, no obs
        #                    flip. Hole termination is the only signal —
        #                    truncation of future reward is the implicit
        #                    penalty (no per-step zeroing needed).
        # The two next-waypoint vectors in the obs flip from forward (toward
        # maze end) to backward (toward the frozen target) for strategies
        # that USE a target (exp_d, exp_d_sigma); survival keeps the upstream
        # forward obs since there's no semantic target.
        valid_strategies = ("exp_d", "exp_d_sigma", "survival")
        if safe_prior_strategy not in valid_strategies:
            raise ValueError(
                f"safe_prior_strategy must be one of {valid_strategies}, "
                f"got {safe_prior_strategy!r}"
            )
        self.safe_prior = bool(safe_prior)
        self.safe_prior_strategy = safe_prior_strategy
        self.safe_prior_sigma = float(safe_prior_sigma)
        # Strategies that need a frozen target → corner caching + obs flip.
        self._uses_target = self.safe_prior and self.safe_prior_strategy in (
            "exp_d", "exp_d_sigma",
        )
        self.init_ball_speed = float(init_ball_speed)
        self.init_tilt_frac = float(init_tilt_frac)
        # Mid-episode tilt bumps (challenge perturbations). Each step, with
        # probability `tilt_bump_prob`, both joint angles get an additive
        # delta uniform in [-mag, +mag] × half_joint_range, clipped to the
        # joint range. JIT-safe: `tilt_bumps` is a compile-time branch via
        # static_argnums=(0,).
        self.tilt_bumps = bool(tilt_bumps)
        self.tilt_bump_prob = float(tilt_bump_prob)
        self.tilt_bump_magnitude = float(tilt_bump_magnitude)
        self._range_alpha_lo = float(RANGE_ALPHA[0])
        self._range_alpha_hi = float(RANGE_ALPHA[1])
        self._range_beta_lo = float(RANGE_BETA[0])
        self._range_beta_hi = float(RANGE_BETA[1])

        # Build maze and MuJoCo model on the host (numpy + mjSpec).
        # Layout selectable: easy | medium | hard. Validated via get_layout.
        walls_h, walls_v, holes, waypoints = get_layout(self.maze_layout)
        seg_lengths, cum_distances = compute_waypoint_distances(waypoints)
        mj_model = build_model(walls_h, walls_v, holes, waypoints)
        self.mj_model = mj_model

        # Body indices (worldbody=0, base=1, link=2, board=3, marble=4)
        self.board_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "board"
        )
        self.marble_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "marble"
        )

        # Move model to MJX. Try fast warp backend; fall back to jax.
        try:
            self.mjx_model = mjx.put_model(mj_model, impl="warp")
            self._mjx_impl = "warp"
        except Exception:
            self.mjx_model = mjx.put_model(mj_model)
            self._mjx_impl = "jax"

        # Cache JAX views of the maze (frozen, baked into JIT closure)
        self._waypoints = jnp.asarray(waypoints, dtype=jnp.float32)
        self._seg_lengths = jnp.asarray(seg_lengths, dtype=jnp.float32)
        self._cum_distances = jnp.asarray(cum_distances, dtype=jnp.float32)
        self._walls_h = jnp.asarray(walls_h, dtype=jnp.float32)
        self._walls_v = jnp.asarray(walls_v, dtype=jnp.float32)
        self._holes = jnp.asarray(holes, dtype=jnp.float32)
        self._goal_pos = self._waypoints[-1]
        self._num_waypoints = int(waypoints.shape[0])

        # Strict-corner safe checkpoints + their progresses + segment indices.
        # Computed once at init (NumPy host code), then baked into JIT closures
        # as JAX constants. Path progress is scaled ×10 to match the convention
        # used by `compute_path_progress_jax` so backward-corner selection at
        # reset can compare directly against the ball's progress output.
        # Skipped when the strategy does not use a target (e.g. "survival").
        if self._uses_target:
            corners_np = select_corner_checkpoints(waypoints, holes, walls_h, walls_v)
            if corners_np.shape[0] == 0:
                raise RuntimeError(
                    "select_corner_checkpoints returned 0 corners — maze geometry "
                    "is unexpected. Cannot run with safe_prior=True."
                )
            corner_progress_np, _, corner_seg_np = _project_points_to_path(
                corners_np, waypoints
            )
            self._corners = jnp.asarray(corners_np, dtype=jnp.float32)
            self._corner_progresses = jnp.asarray(
                corner_progress_np * 10.0, dtype=jnp.float32
            )
            self._corner_seg_idx = jnp.asarray(corner_seg_np, dtype=jnp.int32)
            self._num_corners = int(corners_np.shape[0])
        else:
            # Dummy-but-typed placeholders so JIT trace doesn't see Nones if
            # the attributes are referenced somewhere. They are NEVER read
            # when safe_prior=False (compile-time branch in step/reset).
            self._corners = jnp.zeros((1, 2), dtype=jnp.float32)
            self._corner_progresses = jnp.zeros((1,), dtype=jnp.float32)
            self._corner_seg_idx = jnp.zeros((1,), dtype=jnp.int32)
            self._num_corners = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_data(self) -> mjx.Data:
        if self._mjx_impl == "warp":
            # mujoco_warp.make_data docstring (verbatim):
            #   naconmax: "Number of contacts to allocate FOR ALL WORLDS"  (total)
            #   njmax:    "Number of constraints to allocate PER WORLD"   (per-world)
            # Single marble: ≪1 active contact per env on average → naconmax = 8×num_envs.
            # Constraints per env are bounded by physical DoFs (~ 6 for free joint
            # + 2 hinges + a few contact constraints) → njmax = 8 is plenty.
            return mjx.make_data(
                self.mj_model, impl="warp",
                naconmax=8 * self.num_envs_hint,
                njmax=32,
            )
        return mjx.make_data(self.mj_model)

    def _ball_pos_board_frame(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Match env_mujoco.py L1003-1014: marble pos in board frame, [2]."""
        board_pos = mjx_data.xpos[self.board_body_id]
        board_mat = mjx_data.xmat[self.board_body_id].reshape(3, 3)
        marble_pos = mjx_data.xpos[self.marble_body_id]
        return (board_mat.T @ (marble_pos - board_pos))[:2]

    def _path_progress(self, ball_pos):
        return compute_path_progress_jax(
            ball_pos,
            self._waypoints,
            self._seg_lengths,
            self._cum_distances,
            self._walls_h,
            self._walls_v,
            self._holes,
            num_rays=self.num_rays,
        )

    def _build_frame(
        self,
        mjx_data: mjx.Data,
        ball_pos_clean: jnp.ndarray,
        action: jnp.ndarray,
        obs_bias_ball: jnp.ndarray,
        obs_bias_joint: jnp.ndarray,
        rng_step: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build a single per-step "frame" [joint(2), ball(2), action(2)] (6 dims)
        plus the noisy ball pos used for path-relative vectors. Returns
        `(frame, ball_pos_noisy)`.
        """
        k_ball, k_joint = jax.random.split(rng_step)
        step_ball_noise = jax.random.uniform(
            k_ball, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE
        ).astype(jnp.float32)
        step_joint_noise = jax.random.uniform(
            k_joint, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE
        ).astype(jnp.float32)
        joint_pos = (
            mjx_data.qpos[:2].astype(jnp.float32) + obs_bias_joint + step_joint_noise
        )
        ball_pos_noisy = (
            ball_pos_clean.astype(jnp.float32) + obs_bias_ball + step_ball_noise
        )
        action = action.astype(jnp.float32)
        frame = jnp.concatenate([joint_pos, ball_pos_noisy, action])  # (6,)
        return frame, ball_pos_noisy

    def _forward_next_waypoints(
        self, ball_pos_noisy: jnp.ndarray, seg_idx: jnp.ndarray
    ):
        """Upstream behavior: vec_to_next_wp + vec_to_next_next_wp pointing
        forward along the polyline (toward the maze end). Used when
        safe_prior=False.
        """
        safe_seg = jnp.maximum(seg_idx, 0)
        next_idx = jnp.minimum(safe_seg + 1, self._num_waypoints - 1)
        next_next_idx = jnp.minimum(safe_seg + 2, self._num_waypoints - 1)
        return (
            self._waypoints[next_idx] - ball_pos_noisy,
            self._waypoints[next_next_idx] - ball_pos_noisy,
        )

    def _backward_next_waypoints(
        self,
        ball_pos_noisy: jnp.ndarray,
        seg_idx: jnp.ndarray,
        target_xy: jnp.ndarray,
        target_seg_idx: jnp.ndarray,
    ):
        """Safe-prior obs flip: vec_to_first_back + vec_to_second_back.

        "First waypoint backward toward target" follows the polyline going back
        toward the frozen target:
          - if ball is past target's segment (seg > target_seg): waypoints[seg]
            is one polyline node closer to target.
          - if ball is on target's segment (seg == target_seg): the target
            itself is the closest polyline-aligned cue.
          - if ball drifted before target (seg < target_seg): waypoints[seg+1]
            is the next node toward target.
        "Second waypoint backward" is one more polyline step toward target,
        clamped at the target itself.
        """
        safe_seg = jnp.maximum(seg_idx, 0)
        seg_minus = jnp.maximum(safe_seg - 1, 0)
        seg_plus = jnp.minimum(safe_seg + 1, self._num_waypoints - 1)

        first_back = jnp.where(
            safe_seg > target_seg_idx,
            self._waypoints[safe_seg],
            jnp.where(
                safe_seg == target_seg_idx,
                target_xy,
                self._waypoints[seg_plus],
            ),
        )
        second_back = jnp.where(
            safe_seg > target_seg_idx + 1,
            self._waypoints[seg_minus],
            target_xy,
        )
        return first_back - ball_pos_noisy, second_back - ball_pos_noisy

    def _build_obs(
        self,
        history: jnp.ndarray,
        ball_pos_noisy: jnp.ndarray,
        closest_point: jnp.ndarray,
        seg_idx: jnp.ndarray,
        found: jnp.ndarray,
        target_xy: jnp.ndarray,
        target_seg_idx: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Build the full observation dict from a rolled history buffer.

        Layout:
            [0 : 6*H]      history.flatten()  — last H frames of [joint, ball, action]
            [6*H : 6*H+2]  vec_to_closest    = closest_point - ball_pos_noisy
            [6*H+2 : +4]   vec_to_next_a     = forward next or first-back
            [6*H+4 : +6]   vec_to_next_b     = forward next-next or second-back

        When `safe_prior=False`, vec_to_next_{a,b} point forward along the
        polyline toward the maze end (upstream behavior). When True, they
        point backward toward the frozen safe-prior target.

        Path-relative vectors use the CURRENT-step noisy ball pos. When
        `found=False`, all three are zeroed.
        """
        vec_to_closest = closest_point - ball_pos_noisy
        if self._uses_target:
            vec_to_next_a, vec_to_next_b = self._backward_next_waypoints(
                ball_pos_noisy, seg_idx, target_xy, target_seg_idx
            )
        else:
            vec_to_next_a, vec_to_next_b = self._forward_next_waypoints(
                ball_pos_noisy, seg_idx
            )

        zero2 = jnp.zeros(2, dtype=jnp.float32)
        vec_to_closest = jnp.where(found, vec_to_closest, zero2)
        vec_to_next_a = jnp.where(found, vec_to_next_a, zero2)
        vec_to_next_b = jnp.where(found, vec_to_next_b, zero2)

        state_obs = jnp.concatenate(
            [history.flatten(), vec_to_closest, vec_to_next_a, vec_to_next_b]
        ).astype(jnp.float32)
        return {"state": state_obs}

    # ------------------------------------------------------------------
    # Brax envs.Env interface
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> envs.State:
        rng, k_init, k_bias_b, k_bias_j, k_step = jax.random.split(rng, 5)

        mjx_data = self._make_data()

        # Initial marble position (uniform over waypoints when randomize_init_pos)
        if self.randomize_init_pos:
            idx = jax.random.randint(k_init, (), 0, self._num_waypoints)
            init_pos = self._waypoints[idx]
        else:
            init_pos = self._waypoints[0]

        # qpos layout: [alpha, beta, marble_x, marble_y, marble_z, qw, qx, qy, qz]
        qpos = mjx_data.qpos
        qpos = qpos.at[2:4].set(init_pos)
        qpos = qpos.at[4].set(0.0793)
        qpos = qpos.at[5:9].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        # Safe-prior: mild random tilt + ball velocity at spawn. Compile-time
        # branch via static_argnums=(0,) — no per-step Python overhead.
        if self.safe_prior:
            rng, k_alpha, k_beta, k_speed, k_dir = jax.random.split(rng, 5)
            alpha = jax.random.uniform(
                k_alpha,
                minval=self._range_alpha_lo * self.init_tilt_frac,
                maxval=self._range_alpha_hi * self.init_tilt_frac,
            )
            beta = jax.random.uniform(
                k_beta,
                minval=self._range_beta_lo * self.init_tilt_frac,
                maxval=self._range_beta_hi * self.init_tilt_frac,
            )
            qpos = qpos.at[0].set(alpha).at[1].set(beta)

            theta = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jnp.pi)
            speed = jax.random.uniform(
                k_speed, minval=0.0, maxval=self.init_ball_speed
            )
            qvel = mjx_data.qvel
            qvel = qvel.at[2].set(speed * jnp.cos(theta))
            qvel = qvel.at[3].set(speed * jnp.sin(theta))
            mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        else:
            mjx_data = mjx_data.replace(qpos=qpos)

        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        # Per-episode bias (frozen for whole episode), env_mujoco.py L949-952
        obs_bias_ball = jax.random.uniform(
            k_bias_b, (2,), minval=-BALL_POS_NOISE, maxval=BALL_POS_NOISE
        ).astype(jnp.float32)
        obs_bias_joint = jax.random.uniform(
            k_bias_j, (2,), minval=-JOINT_ANGLE_NOISE, maxval=JOINT_ANGLE_NOISE
        ).astype(jnp.float32)

        ball_pos_clean = self._ball_pos_board_frame(mjx_data)
        progress, seg_idx, _, closest, found = self._path_progress(ball_pos_clean)
        progress = progress.astype(jnp.float32)
        prev_progress = jnp.where(found, progress, jnp.float32(-1.0))

        # Freeze the safe-prior target: closest backward strict-corner relative
        # to spawn progress, fallback to closest-forward if no backward exists.
        # Stored in info so step() can read without recomputing argmax.
        # Strategies without a target (e.g. "survival") skip this and store
        # dummy zeros — never read from step().
        if self._uses_target:
            EPS = jnp.float32(0.005)
            le_mask = self._corner_progresses <= progress + EPS
            masked = jnp.where(le_mask, self._corner_progresses, -jnp.inf)
            backward_idx = jnp.argmax(masked)
            fwd_fallback = jnp.argmin(jnp.abs(self._corner_progresses - progress))
            target_idx = jnp.where(jnp.any(le_mask), backward_idx, fwd_fallback)
            target_idx = target_idx.astype(jnp.int32)
            target_xy = self._corners[target_idx]
            target_seg_idx = self._corner_seg_idx[target_idx]
        else:
            target_idx = jnp.int32(0)
            target_xy = jnp.zeros(2, dtype=jnp.float32)
            target_seg_idx = jnp.int32(0)

        # Initial frame uses zero action (matches cyberrunner_mjx history reset).
        zero_action = jnp.zeros(2, dtype=jnp.float32)
        frame0, ball_pos_noisy = self._build_frame(
            mjx_data, ball_pos_clean, zero_action, obs_bias_ball, obs_bias_joint, k_step,
        )
        history = jnp.tile(frame0[None, :], (self.history_length, 1))  # [H, 6]

        obs = self._build_obs(
            history, ball_pos_noisy, closest, seg_idx, found,
            target_xy, target_seg_idx,
        )

        info: Dict[str, Any] = {
            "rng": rng,
            "obs_bias_ball": obs_bias_ball,
            "obs_bias_joint": obs_bias_joint,
            "prev_progress": prev_progress,
            "obs_history": history,
            # Frozen safe-prior target (read-only after reset). Always present
            # in info — values are dummies when safe_prior=False, never read.
            "safe_prior_target_idx": target_idx,
            "safe_prior_target_xy": target_xy,
            "safe_prior_target_seg_idx": target_seg_idx,
        }
        metrics = {
            "reward": jnp.float32(0.0),
            "path_progress": prev_progress,
        }
        return envs.State(
            pipeline_state=mjx_data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics=metrics,
            info=info,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: envs.State, action: jnp.ndarray) -> envs.State:
        action = jnp.clip(action, -1.0, 1.0)
        mjx_data = state.pipeline_state.replace(ctrl=action)

        # Optional mid-episode tilt bumps: random additive perturbation on
        # joint angles before the physics scan. Compile-time branch via
        # static_argnums=(0,) — costs nothing when disabled.
        rng_in = state.info["rng"]
        if self.tilt_bumps:
            rng_in, k_trig, k_da, k_db = jax.random.split(rng_in, 4)
            bump_active = jax.random.uniform(k_trig) < self.tilt_bump_prob
            half_a = 0.5 * (self._range_alpha_hi - self._range_alpha_lo)
            half_b = 0.5 * (self._range_beta_hi - self._range_beta_lo)
            delta_a = jax.random.uniform(k_da, minval=-1.0, maxval=1.0) \
                * self.tilt_bump_magnitude * half_a
            delta_b = jax.random.uniform(k_db, minval=-1.0, maxval=1.0) \
                * self.tilt_bump_magnitude * half_b
            delta_a = jnp.where(bump_active, delta_a, 0.0)
            delta_b = jnp.where(bump_active, delta_b, 0.0)
            qpos = mjx_data.qpos
            new_alpha = jnp.clip(
                qpos[0] + delta_a, self._range_alpha_lo, self._range_alpha_hi
            )
            new_beta = jnp.clip(
                qpos[1] + delta_b, self._range_beta_lo, self._range_beta_hi
            )
            qpos = qpos.at[0].set(new_alpha).at[1].set(new_beta)
            mjx_data = mjx_data.replace(qpos=qpos)

        # Step physics FRAME_SKIP times via scan (avoids unrolled compile bloat)
        def physics_step(d, _):
            return mjx.step(self.mjx_model, d), None

        mjx_data, _ = jax.lax.scan(physics_step, mjx_data, None, length=FRAME_SKIP)

        # Path progress + termination all use CLEAN ball pos
        ball_pos_clean = self._ball_pos_board_frame(mjx_data)
        progress, seg_idx, _, closest, found = self._path_progress(ball_pos_clean)
        progress = progress.astype(jnp.float32)
        prev_progress = state.info["prev_progress"]

        # === REWARD ===
        target_xy = state.info["safe_prior_target_xy"]
        target_seg_idx = state.info["safe_prior_target_seg_idx"]
        if self.safe_prior:
            # Strategy dispatch — Python `if` over a string set at __init__,
            # so JIT traces only the selected branch.
            if self.safe_prior_strategy == "exp_d":
                # Flat reward landscape (baseline). Target frozen at reset.
                d_to_target = jnp.linalg.norm(ball_pos_clean - target_xy)
                reward = jnp.exp(-d_to_target)
            elif self.safe_prior_strategy == "exp_d_sigma":
                # Steepened reward: exp(-d/sigma). Sharp basin at the corner.
                d_to_target = jnp.linalg.norm(ball_pos_clean - target_xy)
                reward = jnp.exp(-d_to_target / self.safe_prior_sigma)
            else:  # "survival"
                # +1 per alive step. Hole termination cuts off future return,
                # which is the implicit penalty — no per-step zeroing needed.
                reward = jnp.float32(1.0)
            reward = reward.astype(jnp.float32)
        else:
            # Upstream reward (env_mujoco.py L1079-1091).
            both_valid = (progress >= 0) & (prev_progress >= 0)
            progress_reward = jnp.where(
                both_valid, (progress - prev_progress) * PROGRESS_SCALE, 0.0
            )
            dist_to_goal = jnp.linalg.norm(ball_pos_clean - self._goal_pos)
            goal_reward = jnp.where(dist_to_goal < GOAL_THRESHOLD, GOAL_BONUS, 0.0)
            reward = (progress_reward + goal_reward).astype(jnp.float32)

        # === TERMINATION (env_mujoco.py L1093-1117) ===
        hole_dists = jnp.linalg.norm(self._holes - ball_pos_clean, axis=1)
        in_hole = jnp.any(hole_dists < HOLE_RADIUS)
        dist_to_goal = jnp.linalg.norm(ball_pos_clean - self._goal_pos)
        at_goal = dist_to_goal < GOAL_THRESHOLD
        done = jnp.where(in_hole | at_goal, 1.0, 0.0).astype(jnp.float32)

        # prev_progress only updates when curr is valid (env_mujoco.py L995)
        new_prev_progress = jnp.where(found, progress, prev_progress)

        # Step RNG split for per-step noise (rng_in already advanced by the
        # tilt-bump branch when enabled; otherwise it's just state.info["rng"]).
        rng, k_step = jax.random.split(rng_in)

        # Build current frame (joint+ball+action), then roll the history buffer.
        new_frame, ball_pos_noisy = self._build_frame(
            mjx_data, ball_pos_clean, action,
            state.info["obs_bias_ball"], state.info["obs_bias_joint"], k_step,
        )
        prev_history = state.info["obs_history"]  # [H, 6]
        history = jnp.concatenate([prev_history[1:], new_frame[None, :]], axis=0)
        obs = self._build_obs(
            history, ball_pos_noisy, closest, seg_idx, found,
            target_xy, target_seg_idx,
        )

        info = {
            **state.info,
            "rng": rng,
            "prev_progress": new_prev_progress,
            "obs_history": history,
        }
        metrics = {
            "reward": reward,
            "path_progress": progress,
        }
        return state.replace(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    # Brax interface properties
    @property
    def observation_size(self) -> int:
        # H frames × 6 dims/frame  +  6 path-relative dims (vec_to_closest, next, next_next)
        return self.history_length * self._frame_dim + 6

    @property
    def action_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "mjx"


def make_env(**kwargs) -> CyberrunnerMJXEnv:
    return CyberrunnerMJXEnv(**kwargs)
