import tempfile
import time
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

# ============================================================================
# PHYSICS PARAMETERS (from system identification)
# ============================================================================

# Actuator parameters
GEAR_ALPHA = 0.6378096703815441
GEAR_BETA = 1.1503276919996794
DYNPRM_TAU_ALPHA = 0.017652846705560145
DYNPRM_TAU_BETA = 0.0381459302540759

# Joint parameters (angles in radians)
RANGE_ALPHA = (-0.15847916128302914, 0.15847916128302914)  # ~±9.08°
RANGE_BETA = (-0.10424974775885551, 0.10424974775885551)  # ~±5.97°
DAMPING_ALPHA = 0.13716407911582046
DAMPING_BETA = 0.4263556183589426
FRICTIONLOSS_ALPHA = 0.0496853803940722
FRICTIONLOSS_BETA = 0.09905565254081455

# Board parameters
BOARD_MASS = 1.5
BOARD_INERTIA = (0.021, 0.021, 0.021)

# Marble parameters
MARBLE_MASS = 0.009
MARBLE_RADIUS = 0.0063
MARBLE_FRICTION = (0.250129382, 0.07549734, 0.00275288)  # slide, spin, roll
MARBLE_SOLREF = (0.02, 1.25)  # timeconst, dampingratio

# Geometry
BOARD_WIDTH = 0.276
BOARD_HEIGHT = 0.231
WALL_RADIUS = 0.0025
HOLE_RADIUS = 0.0075

# Simulation
TIMESTEP = 0.00166666666  # 600Hz physics
FRAME_SKIP = 10  # 60Hz control

# Observation noise
BALL_POS_NOISE = 0.001  # 1mm
JOINT_ANGLE_NOISE = 0.25 * np.pi / 180  # 0.25 degrees

# Reward parameters
PROGRESS_SCALE = 1.0
GOAL_BONUS = 10.0
GOAL_THRESHOLD = 0.004  # 4mm
CHECKPOINT_REWARD = 1.0  # paid per reward-waypoint crossed under modulo-N sparsification


# ============================================================================
# HARD MAZE LAYOUT
# ============================================================================


def get_hard_layout() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the hard maze layout arrays."""
    walls_h = np.array(
        [
            [0.0, 0.024, 0.2055],
            [0.0, 0.025, 0.159],
            [0.0135, 0.025, 0.182],
            [0.0945, 0.1595, 0.2045],
            [0.0945, 0.1135, 0.1555],
            [0.0135, 0.0425, 0.1370],
            [0.0135, 0.0415, 0.0795],
            [0.0, 0.0135, 0.0365],
            [0.0945, 0.1185, 0.058],
            [0.1185, 0.1415, 0.0795],
            [0.1645, 0.2105, 0.1555],
            [0.1645, 0.1825, 0.1845],
            [0.1595, 0.1885, 0.024],
            [0.1875, 0.2055, 0.0725],
            [0.1825, 0.2055, 0.1075],
            [0.2105, 0.2285, 0.1915],
            [0.2575, 0.276, 0.1015],
            [0.2525, 0.276, 0.210],
        ],
        dtype=np.float32,
    )

    walls_v = np.array(
        [
            [0.187, 0.208, 0.045],
            [0.1345, 0.177, 0.045],
            [0.143, 0.161, 0.069],
            [0.177, 0.192, 0.069],
            [0.214, 0.231, 0.0685],
            [0.139, 0.158, 0.092],
            [0.177, 0.207, 0.092],
            [0.153, 0.186, 0.116],
            [0.132, 0.152, 0.139],
            [0.169, 0.187, 0.139],
            [0.117, 0.231, 0.162],
            [0.092, 0.137, 0.208],
            [0.175, 0.194, 0.208],
            [0.122, 0.194, 0.231],
            [0.213, 0.231, 0.231],
            [0.166, 0.191, 0.255],
            [0.099, 0.15, 0.255],
            [0.06, 0.102, 0.162],
            [0.07, 0.092, 0.185],
            [0.021, 0.075, 0.208],
            [0.021, 0.038, 0.231],
            [0.053, 0.104, 0.231],
            [0.0, 0.018, 0.255],
            [0.098, 0.121, 0.021],
            [0.098, 0.121, 0.045],
            [0.093, 0.121, 0.069],
            [0.093, 0.121, 0.092],
            [0.021, 0.082, 0.044],
            [0.034, 0.052, 0.069],
            [0.0, 0.014, 0.069],
            [0.021, 0.06, 0.092],
            [0.021, 0.038, 0.116],
            [0.077, 0.132, 0.116],
            [0.021, 0.06, 0.139],
            [0.099, 0.117, 0.139],
        ],
        dtype=np.float32,
    )

    holes = np.array(
        [
            [0.011, 0.216],
            [0.034, 0.185],
            [0.057, 0.185],
            [0.081, 0.185],
            [0.15, 0.191],
            [0.058, 0.128],
            [0.055, 0.128],
            [0.081, 0.157],
            [0.104, 0.143],
            [0.109, 0.141],
            [0.127, 0.146],
            [0.174, 0.216],
            [0.219, 0.216],
            [0.242, 0.216],
            [0.266, 0.198],
            [0.219, 0.178],
            [0.174, 0.17],
            [0.173, 0.143],
            [0.243, 0.143],
            [0.219, 0.132],
            [0.196, 0.12],
            [0.266, 0.088],
            [0.266, 0.043],
            [0.243, 0.058],
            [0.243, 0.013],
            [0.219, 0.058],
            [0.174, 0.013],
            [0.174, 0.038],
            [0.174, 0.071],
            [0.01, 0.107],
            [0.01, 0.05],
            [0.034, 0.067],
            [0.034, 0.025],
            [0.057, 0.088],
            [0.057, 0.037],
            [0.081, 0.107],
            [0.081, 0.07],
            [0.084, 0.014],
            [0.104, 0.071],
            [0.127, 0.031],
            [0.152, 0.111],
            [0.154, 0.109],
            [0.15, 0.058],
        ],
        dtype=np.float32,
    )

    waypoints = np.array(
        [
            [0.150, 0.218],
            [0.080, 0.218],
            [0.080, 0.203],
            [0.056, 0.203],
            [0.056, 0.218],
            [0.034, 0.218],
            [0.034, 0.195],
            [0.009, 0.195],
            [0.009, 0.167],
            [0.033, 0.167],
            [0.033, 0.149],
            [0.007, 0.149],
            [0.007, 0.129],
            [0.033, 0.129],
            [0.033, 0.091],
            [0.007, 0.091],
            [0.007, 0.069],
            [0.029, 0.050],
            [0.017, 0.025],
            [0.017, 0.010],
            [0.050, 0.010],
            [0.080, 0.034],
            [0.080, 0.059],
            [0.062, 0.059],
            [0.062, 0.073],
            [0.075, 0.085],
            [0.104, 0.085],
            [0.104, 0.130],
            [0.075, 0.130],
            [0.057, 0.143],
            [0.057, 0.170],
            [0.104, 0.170],
            [0.104, 0.193],
            [0.128, 0.193],
            [0.128, 0.162],
            [0.150, 0.162],
            [0.150, 0.125],
            [0.127, 0.125],
            [0.127, 0.090],
            [0.150, 0.090],
            [0.150, 0.069],
            [0.127, 0.069],
            [0.127, 0.046],
            [0.104, 0.046],
            [0.104, 0.011],
            [0.150, 0.011],
            [0.150, 0.038],
            [0.171, 0.053],
            [0.197, 0.053],
            [0.197, 0.010],
            [0.220, 0.010],
            [0.220, 0.045],
            [0.247, 0.045],
            [0.261, 0.058],
            [0.243, 0.077],
            [0.243, 0.113],
            [0.219, 0.113],
            [0.219, 0.083],
            [0.197, 0.083],
            [0.197, 0.098],
            [0.173, 0.098],
            [0.173, 0.121],
            [0.199, 0.145],
            [0.219, 0.145],
            [0.219, 0.166],
            [0.195, 0.166],
            [0.195, 0.203],
            [0.244, 0.203],
            [0.244, 0.159],
            [0.266, 0.159],
            [0.266, 0.112],
        ],
        dtype=np.float32,
    )

    return walls_h, walls_v, holes, waypoints


# ============================================================================
# PATH UTILITIES
# ============================================================================


def compute_waypoint_distances(waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute segment lengths and cumulative distances along waypoints."""
    seg_vectors = waypoints[1:] - waypoints[:-1]
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)
    cum_distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return seg_lengths.astype(np.float32), cum_distances.astype(np.float32)


def _project_points_to_path(points: np.ndarray, waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the reference path and return path progress and offset."""
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
    return progress.astype(np.float32), best_offset.astype(np.float32)


def _point_to_segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return minimum Euclidean distance from each point to a set of 2D line segments."""
    seg_vecs = ends - starts
    seg_len_sq = np.maximum(np.sum(seg_vecs**2, axis=1), 1e-10)
    rel = points[:, None, :] - starts[None, :, :]
    t = np.sum(rel * seg_vecs[None, :, :], axis=-1) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[..., None] * seg_vecs[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def _segment_crosses_walls(p: np.ndarray, q: np.ndarray, wall_starts: np.ndarray, wall_ends: np.ndarray) -> bool:
    """Return True if segment p→q crosses any wall segment (2D cross-product test)."""
    r = q - p
    s = wall_ends - wall_starts          # (W, 2)
    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    nonpar = np.abs(rxs) > 1e-10
    qmp = wall_starts - p                # (W, 2)
    t = (qmp[:, 0] * s[:, 1] - qmp[:, 1] * s[:, 0]) / np.where(nonpar, rxs, 1.0)
    u = (qmp[:, 0] * r[1]    - qmp[:, 1] * r[0]   ) / np.where(nonpar, rxs, 1.0)
    return bool((nonpar & (t > 0) & (t < 1) & (u > 0) & (u < 1)).any())


def _segment_distance_matrix(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return full (N_points, N_segments) matrix of distances to 2D line segments."""
    seg_vecs = ends - starts
    seg_len_sq = np.maximum(np.sum(seg_vecs**2, axis=1), 1e-10)
    rel = points[:, None, :] - starts[None, :, :]
    t = np.sum(rel * seg_vecs[None, :, :], axis=-1) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[..., None] * seg_vecs[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


def select_safe_checkpoints(
    waypoints: np.ndarray,
    holes: np.ndarray,
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    reward_every_n_waypoints: int,
    include_corridors: bool = True,
    grid_res: float = 0.002,
) -> np.ndarray:
    """Pick geometrically safe checkpoints where the ball can physically stabilize.

    Two classes qualify:
    - Corner: close to both a horizontal AND a vertical surface (two perpendicular supports).
    - Single-wall: close to exactly one surface family, not sandwiched on the same axis.
      The policy can hold the ball against a single wall with a slight tilt.
    Corridor midpoints (sandwiched on either axis) are excluded. Hole clearance is a
    hard filter with wall-blocked line-of-sight. Result is deduplicated and path-ordered.
    Setting ``include_corridors=False`` keeps only true corner basins.
    """
    corner_slack      = 0.003   # 3 mm: wall must be this close to touching (corner)
    single_wall_slack = 0.0015  # 1.5 mm: must be nearly touching for single-wall support
    corridor_slack    = 0.010   # 10 mm: if walls within this on BOTH sides of same axis → corridor
    touch_wall = WALL_RADIUS + MARBLE_RADIUS
    touch_edge = MARBLE_RADIUS

    xs = np.arange(MARBLE_RADIUS, BOARD_WIDTH - MARBLE_RADIUS + 1e-9, grid_res, dtype=np.float32)
    ys = np.arange(MARBLE_RADIUS, BOARD_HEIGHT - MARBLE_RADIUS + 1e-9, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    cx = candidates[:, 0]  # (N,)
    cy = candidates[:, 1]

    # --- Perpendicular ray-casting ---
    # For each candidate cast 4 axis-aligned rays. A horizontal wall at y=hy only
    # contributes to the up/down ray if the candidate's x falls within [hx0, hx1]
    # (i.e., the wall is directly above/below). Likewise for vertical walls.
    # This avoids the endpoint-distance artefact of _segment_distance_matrix.

    hx0_arr = walls_h[:, 0][None, :]   # (1, Wh)
    hx1_arr = walls_h[:, 1][None, :]
    hy_arr  = walls_h[:, 2][None, :]
    in_x = (cx[:, None] >= hx0_arr - 1e-6) & (cx[:, None] <= hx1_arr + 1e-6)  # (N, Wh)
    dy = hy_arr - cy[:, None]                                                    # (N, Wh)
    ray_up   = np.where(in_x & (dy > 0),  dy, np.inf).min(axis=1)   # (N,)
    ray_down = np.where(in_x & (dy < 0), -dy, np.inf).min(axis=1)

    vy0_arr = walls_v[:, 0][None, :]   # (1, Wv)
    vy1_arr = walls_v[:, 1][None, :]
    vx_arr  = walls_v[:, 2][None, :]
    in_y = (cy[:, None] >= vy0_arr - 1e-6) & (cy[:, None] <= vy1_arr + 1e-6)  # (N, Wv)
    dx = vx_arr - cx[:, None]                                                    # (N, Wv)
    ray_right = np.where(in_y & (dx > 0),  dx, np.inf).min(axis=1)  # (N,)
    ray_left  = np.where(in_y & (dx < 0), -dx, np.inf).min(axis=1)

    # Combined gap to nearest surface (wall or board edge) in each direction.
    # gap = 0 means touching, negative means overlapping.
    gap_up    = np.minimum(ray_up    - touch_wall, BOARD_HEIGHT - cy - touch_edge)
    gap_down  = np.minimum(ray_down  - touch_wall, cy            - touch_edge)
    gap_right = np.minimum(ray_right - touch_wall, BOARD_WIDTH - cx - touch_edge)
    gap_left  = np.minimum(ray_left  - touch_wall, cx            - touch_edge)

    close_up    = gap_up    < corner_slack
    close_down  = gap_down  < corner_slack
    close_right = gap_right < corner_slack
    close_left  = gap_left  < corner_slack

    in_h_corridor = (gap_up < corridor_slack) & (gap_down < corridor_slack)
    in_v_corridor = (gap_left < corridor_slack) & (gap_right < corridor_slack)

    is_corner = (
        (close_up | close_down) & (close_right | close_left)
        & ~in_h_corridor & ~in_v_corridor
    )
    # Single-wall: nearly touching exactly one wall family (H or V), not sandwiched
    sw_up    = gap_up    < single_wall_slack
    sw_down  = gap_down  < single_wall_slack
    sw_right = gap_right < single_wall_slack
    sw_left  = gap_left  < single_wall_slack
    is_single_wall = (
        ((sw_up | sw_down) & ~(close_right | close_left) & ~in_h_corridor)
        | ((sw_right | sw_left) & ~(close_up | close_down) & ~in_v_corridor)
    )
    is_safe = is_corner | (is_single_wall if include_corridors else False)

    # Wall clearance for validity (use _segment_distance_matrix — fine for overlap check)
    h_starts = np.stack([walls_h[:, 0], walls_h[:, 2]], axis=1)
    h_ends   = np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1)
    v_starts = np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1)
    v_ends   = np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1)
    all_wall_dist = _point_to_segment_distance(candidates, np.vstack([h_starts, v_starts]), np.vstack([h_ends, v_ends]))
    wall_clearance = all_wall_dist - touch_wall
    edge_clearance = np.minimum.reduce([cx, BOARD_WIDTH - cx, cy, BOARD_HEIGHT - cy]) - touch_edge

    # --- Hole clearance with wall-blocked line-of-sight ---
    # A hole only threatens a candidate if there is no wall between them.
    # Tight wall-supported corners can safely tolerate a slightly smaller extra
    # restart margin than generic free-space points.
    # Corners: tight margin (two supports). Single-wall: larger margin (less constrained).
    clearance_margin = HOLE_RADIUS + MARBLE_RADIUS + np.where(is_corner, 0.006, 0.010)
    hole_dist_mat = np.linalg.norm(candidates[:, None, :] - holes[None, :, :], axis=-1)  # (N, H)
    # Start with simple distance; candidates that are far enough are trivially safe.
    hole_clearance = hole_dist_mat.min(axis=1) - clearance_margin
    # For candidates where some hole is too close, check if a wall blocks the path.
    at_risk = np.where(hole_clearance < 0)[0]
    if len(at_risk) > 0:
        wall_s = np.vstack([h_starts, v_starts])
        wall_e = np.vstack([h_ends,   v_ends  ])
        for i in at_risk:
            margin_i = clearance_margin[i]
            close_hole_idxs = np.where(hole_dist_mat[i] < margin_i)[0]
            worst = -np.inf
            for j in close_hole_idxs:
                if _segment_crosses_walls(candidates[i], holes[j], wall_s, wall_e):
                    # Wall blocks this hole — not a hazard
                    continue
                # Unblocked: use its clearance
                worst = max(worst, margin_i - hole_dist_mat[i, j])
            # If all close holes were blocked, candidate is safe
            hole_clearance[i] = -worst if worst > -np.inf else margin_i

    progress, path_offset = _project_points_to_path(candidates, waypoints)
    # Keep checkpoints relevant to maze progression, but allow slightly off-path
    # corner basins that are genuinely good stop/restart locations.
    max_path_offset = 0.065

    valid = (
        (hole_clearance > 0.0)
        & (wall_clearance >= -1e-6)
        & (edge_clearance >= -1e-6)
        & (path_offset < max_path_offset)
        & is_safe
    )
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)

    is_corner_v = is_corner[valid]
    candidates = candidates[valid]
    progress = progress[valid]
    hole_clearance = hole_clearance[valid]
    gap_up_v    = gap_up[valid]
    gap_down_v  = gap_down[valid]
    gap_left_v  = gap_left[valid]
    gap_right_v = gap_right[valid]
    touch_score = np.minimum.reduce([gap_up_v, gap_down_v, gap_left_v, gap_right_v])
    quality = 0.5 * hole_clearance - 10.0 * touch_score

    # Wall endpoints: single-wall points must be near one to avoid mid-corridor placement.
    h_ep = np.vstack([
        np.stack([walls_h[:, 0], walls_h[:, 2]], axis=1),
        np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1),
    ])
    v_ep = np.vstack([
        np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1),
        np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1),
    ])
    endpoints = np.vstack([h_ep, v_ep])  # (E, 2)
    dist_to_ep = np.linalg.norm(candidates[:, None, :] - endpoints[None, :, :], axis=-1).min(axis=1)
    max_ep_dist = 0.025  # single-wall must be within 2.5 cm of a wall junction/endpoint

    # Two-pass greedy NMS
    # Pass 1: corners only, original spacing (preserves all previously selected corners)
    min_sep_c  = max(0.010, 0.018 - 0.001 * float(reward_every_n_waypoints))
    # Pass 2: single-wall fills gaps, wider spacing to stay sparse
    min_sep_sw = 0.030

    selected_idx: list[int] = []

    def _nms_add(idx: int, min_sep: float) -> bool:
        cand = candidates[idx]
        if not selected_idx:
            selected_idx.append(idx)
            return True
        sel = candidates[np.asarray(selected_idx)]
        if np.all(np.linalg.norm(sel - cand, axis=1) >= min_sep):
            selected_idx.append(idx)
            return True
        return False

    corner_idxs = np.where(is_corner_v)[0]
    for idx in corner_idxs[np.argsort(-quality[corner_idxs])]:
        _nms_add(int(idx), min_sep_c)

    if include_corridors:
        sw_idxs = np.where(~is_corner_v & (dist_to_ep < max_ep_dist))[0]
        for idx in sw_idxs[np.argsort(-quality[sw_idxs])]:
            _nms_add(int(idx), min_sep_sw)

    selected_idx_arr = np.asarray(selected_idx, dtype=np.int32)
    progress_order = np.argsort(progress[selected_idx_arr])
    return candidates[selected_idx_arr][progress_order].astype(np.float32)


# ============================================================================
# RAYCASTING (simplified - path progress only)
# ============================================================================


def compute_path_progress(
    marble_pos: np.ndarray,
    waypoints: np.ndarray,
    seg_lengths: np.ndarray,
    cum_distances: np.ndarray,
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    holes: np.ndarray,
    num_rays: int = 32,
    hole_radius: float = HOLE_RADIUS,
) -> tuple[float, int, float, np.ndarray]:
    """
    Compute path progress by raycasting to find CLOSEST visible path intersection.
    Fully vectorized implementation with wall/hole occlusion checking.

    Args:
        marble_pos: [2] current marble position
        waypoints: [N, 2] path waypoints
        seg_lengths: [N-1] length of each segment
        cum_distances: [N] cumulative distance to each waypoint
        walls_h: [H, 3] horizontal walls (x_start, x_end, y)
        walls_v: [V, 3] vertical walls (y_start, y_end, x)
        holes: [K, 2] hole positions

    Returns:
        progress: Path progress (distance along path, scaled by 10)
        seg_idx: Index of segment containing the closest visible point
        param: Parameter [0,1] along that segment
        closest_point: [2] position of closest visible point on path
    """
    # === DIRECT PROJECTION FALLBACK ===
    # When marble is very close to path, raycasting fails. Project directly.
    seg_starts = waypoints[:-1]
    seg_ends = waypoints[1:]
    seg_vecs = seg_ends - seg_starts

    # Vector from each segment start to marble
    to_marble = marble_pos - seg_starts  # [S, 2]

    # Project onto each segment: t = (to_marble · seg_vec) / |seg_vec|²
    seg_len_sq = np.sum(seg_vecs**2, axis=1)  # [S]
    t_proj = np.sum(to_marble * seg_vecs, axis=1) / np.maximum(seg_len_sq, 1e-10)  # [S]
    t_proj_clipped = np.clip(t_proj, 0.0, 1.0)  # [S]

    # Closest point on each segment
    closest_on_seg = seg_starts + t_proj_clipped[:, np.newaxis] * seg_vecs  # [S, 2]

    # Distance from marble to closest point on each segment
    dist_to_seg = np.linalg.norm(marble_pos - closest_on_seg, axis=1)  # [S]

    # Find closest segment
    best_seg = np.argmin(dist_to_seg)
    min_dist = dist_to_seg[best_seg]

    # If marble is very close to path (within 2mm), use direct projection
    if min_dist < 0.002:
        best_param = t_proj_clipped[best_seg]
        best_progress = cum_distances[best_seg] + best_param * seg_lengths[best_seg]
        closest_point = closest_on_seg[best_seg]
        return best_progress * 10.0, int(best_seg), float(best_param), closest_point.astype(np.float32)

    # === RAYCASTING ===
    # Generate ray directions: [num_rays, 2]
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # [R, 2]

    # === WALL INTERSECTIONS ===
    # Horizontal walls: [H, 3] -> segments from (x_start, y) to (x_end, y)
    h_seg_starts = np.stack([walls_h[:, 0], walls_h[:, 2]], axis=1)  # [H, 2]
    h_seg_ends = np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1)  # [H, 2]

    # Vertical walls: [V, 3] -> segments from (x, y_start) to (x, y_end)
    v_seg_starts = np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1)  # [V, 2]
    v_seg_ends = np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1)  # [V, 2]

    # Combine all wall segments
    wall_starts = np.vstack([h_seg_starts, v_seg_starts])  # [H+V, 2]
    wall_ends = np.vstack([h_seg_ends, v_seg_ends])  # [H+V, 2]
    wall_vecs = wall_ends - wall_starts  # [H+V, 2]

    # Vectorized ray-wall intersection for all rays × all walls
    ray_dirs_exp = ray_dirs[:, np.newaxis, :]  # [R, 1, 2]
    wall_vecs_exp = wall_vecs[np.newaxis, :, :]  # [1, W, 2]

    denom_w = ray_dirs_exp[:, :, 0] * wall_vecs_exp[:, :, 1] - ray_dirs_exp[:, :, 1] * wall_vecs_exp[:, :, 0]
    parallel_w = np.abs(denom_w) < 1e-8
    safe_denom_w = np.where(parallel_w, 1.0, denom_w)

    diff_w = wall_starts - marble_pos  # [W, 2]
    diff_w_exp = diff_w[np.newaxis, :, :]  # [1, W, 2]

    t_w = (diff_w_exp[:, :, 0] * wall_vecs_exp[:, :, 1] - diff_w_exp[:, :, 1] * wall_vecs_exp[:, :, 0]) / safe_denom_w
    s_w = (diff_w_exp[:, :, 0] * ray_dirs_exp[:, :, 1] - diff_w_exp[:, :, 1] * ray_dirs_exp[:, :, 0]) / safe_denom_w

    valid_w = (~parallel_w) & (t_w > 1e-6) & (s_w >= 0) & (s_w <= 1)
    wall_dists = np.where(valid_w, t_w, np.inf)  # [R, W]
    min_wall_dist = np.min(wall_dists, axis=1)  # [R] - closest wall for each ray

    # === HOLE INTERSECTIONS ===
    oc = marble_pos - holes  # [K, 2]
    oc_exp = oc[np.newaxis, :, :]  # [1, K, 2]
    b = 2.0 * np.sum(oc_exp * ray_dirs_exp, axis=2)  # [R, K]
    c = np.sum(oc * oc, axis=1) - hole_radius**2  # [K]
    c_exp = c[np.newaxis, :]  # [1, K]

    discriminant = b * b - 4 * c_exp  # [R, K]
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    t_hole = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
    t_hole = np.where(discriminant < 0, np.inf, t_hole)  # [R, K]
    min_hole_dist = np.min(t_hole, axis=1)  # [R]

    # === COMBINED OBSTACLE DISTANCE ===
    obstacle_dist = np.minimum(min_wall_dist, min_hole_dist)  # [R]

    # === PATH INTERSECTIONS ===
    # (seg_starts, seg_ends, seg_vecs already computed above)
    num_segs = len(seg_starts)

    seg_vecs_exp = seg_vecs[np.newaxis, :, :]  # [1, S, 2]
    _ = seg_starts[np.newaxis, :, :]  # [1, S, 2] (used implicitly via diff_p_exp)

    denom_p = ray_dirs_exp[:, :, 0] * seg_vecs_exp[:, :, 1] - ray_dirs_exp[:, :, 1] * seg_vecs_exp[:, :, 0]
    parallel_p = np.abs(denom_p) < 1e-8
    safe_denom_p = np.where(parallel_p, 1.0, denom_p)

    diff_p = seg_starts - marble_pos  # [S, 2]
    diff_p_exp = diff_p[np.newaxis, :, :]  # [1, S, 2]

    t_p = (diff_p_exp[:, :, 0] * seg_vecs_exp[:, :, 1] - diff_p_exp[:, :, 1] * seg_vecs_exp[:, :, 0]) / safe_denom_p
    s_p = (diff_p_exp[:, :, 0] * ray_dirs_exp[:, :, 1] - diff_p_exp[:, :, 1] * ray_dirs_exp[:, :, 0]) / safe_denom_p

    # Valid path intersection: not parallel, t > 0, s in [0, 1], AND closer than obstacles
    valid_p = (~parallel_p) & (t_p > 1e-6) & (s_p >= 0) & (s_p <= 1)
    obstacle_dist_exp = obstacle_dist[:, np.newaxis]  # [R, 1]
    valid_p = valid_p & (t_p < obstacle_dist_exp)  # [R, S]

    # For each valid intersection, compute ray distance (t_p)
    path_ray_dists = np.where(valid_p, t_p, np.inf)  # [R, S]
    s_p_clipped = np.clip(s_p, 0.0, 1.0)

    # Compute path progress for ALL valid intersections
    # path_distance[r,s] = cum_distances[s] + s_p[r,s] * seg_lengths[s]
    if not np.any(valid_p):
        return -1.0, 0, 0.0, np.zeros(2, dtype=np.float32)

    # Find the CLOSEST valid intersection (minimum ray distance)
    # First flatten to find global minimum
    flat_ray_dists = path_ray_dists.ravel()
    flat_idx = np.argmin(flat_ray_dists)
    best_ray_idx = flat_idx // num_segs
    best_seg_idx = flat_idx % num_segs

    if flat_ray_dists[flat_idx] == np.inf:
        return -1.0, 0, 0.0, np.zeros(2, dtype=np.float32)

    best_param = s_p_clipped[best_ray_idx, best_seg_idx]
    best_progress = cum_distances[best_seg_idx] + best_param * seg_lengths[best_seg_idx]

    # Compute the actual closest point position
    seg_vec = waypoints[best_seg_idx + 1] - waypoints[best_seg_idx]
    closest_point = waypoints[best_seg_idx] + best_param * seg_vec

    return best_progress * 10.0, int(best_seg_idx), float(best_param), closest_point.astype(np.float32)


# ============================================================================
# MODEL BUILDER
# ============================================================================


def check_endpoint_connected(
    point_x: float,
    point_y: float,
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    is_vertical_wall: bool,
    wall_index: int,
    tol: float = 0.003,
) -> bool:
    """Check if a wall endpoint is connected to another wall or board edge."""
    # Check board edges
    if (
        abs(point_x) < tol
        or abs(point_x - BOARD_WIDTH) < tol
        or abs(point_y) < tol
        or abs(point_y - BOARD_HEIGHT) < tol
    ):
        return True

    if is_vertical_wall:
        # Check connection to horizontal walls
        for h_wall in walls_h:
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol and x_start - tol <= point_x <= x_end + tol:
                return True
        # Check connection to other vertical walls
        for i, v_wall in enumerate(walls_v):
            if i == wall_index:
                continue
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol and (abs(y_start - point_y) < tol or abs(y_end - point_y) < tol):
                return True
    else:
        # Check connection to vertical walls
        for v_wall in walls_v:
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol and y_start - tol <= point_y <= y_end + tol:
                return True
        # Check connection to other horizontal walls
        for i, h_wall in enumerate(walls_h):
            if i == wall_index:
                continue
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol and (abs(x_start - point_x) < tol or abs(x_end - point_x) < tol):
                return True

    return False


def _generate_board_texture(
    holes: np.ndarray,
    waypoints: np.ndarray,
    checkpoint_points: np.ndarray | None = None,
):
    """Generate the board texture used by the simulator.

    Checkpoints are intentionally *not* baked into this texture so vision policies
    do not observe synthetic green markers that do not exist on the real system.
    The ``checkpoint_points`` argument is kept for API compatibility with older
    call sites and offline visualization utilities.
    """
    # Board floor geom spans -0.007 to 0.283 in x, -0.007 to 0.238 in y
    # (centered at [0.138, 0.1155] with half-size [0.145, 0.1225])
    # Maze coordinates start at 0, so offset by 0.007 to place them correctly.
    margin = 0.007
    full_w = BOARD_WIDTH + 2 * margin  # 0.290
    full_h = BOARD_HEIGHT + 2 * margin  # 0.245
    scale = 5000
    w = int(full_w * scale)  # 1450
    h = int(full_h * scale)  # 1225
    img = Image.new("RGB", (w, h), (204, 204, 204))
    draw = ImageDraw.Draw(img)

    # Draw holes as black filled circles (offset by margin)
    hole_r = int(HOLE_RADIUS * scale)
    for hole in holes:
        cx = int((hole[0] + margin) * scale)
        cy = int((hole[1] + margin) * scale)
        draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r], fill=(0, 0, 0))

    # Draw path as dark gray lines between consecutive waypoints
    path_w = max(int(0.002 * scale), 1)
    for i in range(len(waypoints) - 1):
        p1 = (int((waypoints[i][0] + margin) * scale), int((waypoints[i][1] + margin) * scale))
        p2 = (int((waypoints[i + 1][0] + margin) * scale), int((waypoints[i + 1][1] + margin) * scale))
        draw.line([p1, p2], fill=(30, 30, 30), width=path_w)

    # MuJoCo textures have origin at bottom-left
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def build_model(
    walls_h: np.ndarray,
    walls_v: np.ndarray,
    holes: np.ndarray,
    waypoints: np.ndarray,
    checkpoint_points: np.ndarray | None = None,
) -> mujoco.MjModel:
    """Build the MuJoCo model using mjSpec."""
    spec = mujoco.MjSpec()
    spec.modelname = "cyberrunner"
    spec.compiler.autolimits = True
    spec.option.timestep = TIMESTEP

    # Board texture with holes and path baked in
    board_img = _generate_board_texture(holes, waypoints, checkpoint_points)

    tex = spec.add_texture()
    tex.name = "board_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D

    mat = spec.add_material()
    mat.name = "board_mat"
    mat.textures[1] = "board_tex"
    mat.texrepeat = [1, 1]
    mat.emission = 0.5

    world = spec.worldbody

    # Floor (visual only)
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10, 10, 10]
    floor.rgba = [0.8, 0.9, 0.8, 1]
    floor.contype = 0
    floor.conaffinity = 0

    # Light
    light = world.add_light()
    light.pos = [0, 0, 1.3]
    light.dir = [0, 0, -1.3]

    # Base (static)
    base = world.add_body()
    base.name = "base"
    base_geom = base.add_geom()
    base_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    base_geom.size = [0.15, 0.15, 0.01]
    base_geom.pos = [0.138, 0.1155, 0.0]
    base_geom.rgba = [1.0, 1.0, 1.0, 1.0]
    base_geom.contype = 0
    base_geom.conaffinity = 0

    # Camera
    angle_rad = np.radians(25)
    cam = base.add_camera(pos=[0.138, -0.05, 0.4], zaxis=[0, -np.sin(angle_rad), np.cos(angle_rad)])
    cam.name = "board"
    cam.fovy = 50

    # Link body (alpha joint - rotation around Y axis)
    link = world.add_body()
    link.name = "link"
    link.pos = [0, 0, 0]

    link_joint = link.add_joint()
    link_joint.name = "alpha_joint"
    link_joint.type = mujoco.mjtJoint.mjJNT_HINGE
    link_joint.pos = [0.138, 0.1155, 0.0835]
    link_joint.axis = [0, -1, 0]
    link_joint.range = [np.degrees(RANGE_ALPHA[0]), np.degrees(RANGE_ALPHA[1])]
    link_joint.limited = True
    link_joint.damping = DAMPING_ALPHA
    link_joint.frictionloss = FRICTIONLOSS_ALPHA

    link_geom = link.add_geom()
    link_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    link_geom.size = [0.02, 0.02, 0.001]
    link_geom.pos = [0.138, 0.1155, 0.0]
    link_geom.rgba = [0.5, 0.5, 0.5, 1.0]
    link_geom.contype = 0
    link_geom.conaffinity = 0
    link_geom.mass = 0.05

    # Board body (beta joint - rotation around X axis)
    board = link.add_body()
    board.name = "board"
    board.pos = [0, 0, 0]

    board_joint = board.add_joint()
    board_joint.name = "beta_joint"
    board_joint.type = mujoco.mjtJoint.mjJNT_HINGE
    board_joint.pos = [0.138, 0.1155, 0.0835]
    board_joint.axis = [1, 0, 0]
    board_joint.range = [np.degrees(RANGE_BETA[0]), np.degrees(RANGE_BETA[1])]
    board_joint.limited = True
    board_joint.damping = DAMPING_BETA
    board_joint.frictionloss = FRICTIONLOSS_BETA

    # Board inertial properties
    board.ipos = [0.138, 0.1155, 0.0835]
    board.mass = BOARD_MASS
    board.inertia = list(BOARD_INERTIA)

    # Board edges
    _add_board_edges(board)

    # Board floor
    floor_geom = board.add_geom()
    floor_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    floor_geom.size = [0.145, 0.1225, 0.0025]
    floor_geom.pos = [0.138, 0.1155, 0.0705]
    floor_geom.material = "board_mat"
    floor_geom.contype = 2
    floor_geom.conaffinity = 0

    # Maze walls
    _add_maze_walls(board, walls_h, walls_v)

    # Vision camera (attached to board, tracks ball by repositioning before render)
    vision_cam = board.add_camera(pos=[0.138, 0.1155, 0.4], zaxis=[0, 0, 1])
    vision_cam.name = "vision_cam"
    # Narrow FOV to cover ~6cm at camera height (0.4 - 0.0705 = 0.3295m above board)
    vision_cam.fovy = np.degrees(2 * np.arctan(0.03 / 0.3295))

    # Actuators
    _add_actuators(spec)

    # Marble
    _add_marble(world, waypoints[0])

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        board_img.save(f)
        f.flush()
        tex.file = f.name
        return spec.compile()


def _add_board_edges(board):
    """Add collision boxes for board edges."""
    edges = [
        ([0.138, 0.2345, 0.0835], [0.145, 0.0035, 0.0105]),  # Top
        ([0.138, -0.0035, 0.0835], [0.145, 0.0035, 0.0105]),  # Bottom
        ([-0.0035, 0.1155, 0.0835], [0.0035, 0.1155, 0.0105]),  # Left
        ([0.2795, 0.1155, 0.0835], [0.0035, 0.1155, 0.0105]),  # Right
    ]
    for pos, size in edges:
        edge = board.add_geom()
        edge.type = mujoco.mjtGeom.mjGEOM_BOX
        edge.size = size
        edge.pos = pos
        edge.contype = 2
        edge.conaffinity = 0


def _add_maze_walls(board, walls_h: np.ndarray, walls_v: np.ndarray):
    """Add maze walls with rounded edges."""
    wall_r = WALL_RADIUS
    visual_wall_r = max(wall_r, 0.0025)
    wall_height = 0.0075
    z_pos = 0.073 + wall_height
    wall_rgba = [0.9, 0.3, 0.1, 1.0]

    # Vertical walls
    for i, wall in enumerate(walls_v):
        y_start, y_end, x = float(wall[0]), float(wall[1]), float(wall[2])

        needs_cap_start = not check_endpoint_connected(x, y_start, walls_h, walls_v, True, i)
        needs_cap_end = not check_endpoint_connected(x, y_end, walls_h, walls_v, True, i)

        box_y_start = (y_start + visual_wall_r) if needs_cap_start else y_start
        box_y_end = (y_end - visual_wall_r) if needs_cap_end else y_end
        box_length = box_y_end - box_y_start

        if box_length > 0:
            box = board.add_geom()
            box.name = f"vwall{i}"
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.pos = [x, (box_y_start + box_y_end) / 2.0, z_pos]
            box.size = [visual_wall_r, box_length / 2.0, wall_height]
            box.rgba = wall_rgba
            box.contype = 2
            box.conaffinity = 0

        if needs_cap_start:
            cap = board.add_geom()
            cap.name = f"vwall{i}_cap_s"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x, y_start + visual_wall_r, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

        if needs_cap_end:
            cap = board.add_geom()
            cap.name = f"vwall{i}_cap_e"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x, y_end - visual_wall_r, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

    # Horizontal walls
    for i, wall in enumerate(walls_h):
        x_start, x_end, y = float(wall[0]), float(wall[1]), float(wall[2])

        needs_cap_start = not check_endpoint_connected(x_start, y, walls_h, walls_v, False, i)
        needs_cap_end = not check_endpoint_connected(x_end, y, walls_h, walls_v, False, i)

        box_x_start = (x_start + visual_wall_r) if needs_cap_start else x_start
        box_x_end = (x_end - visual_wall_r) if needs_cap_end else x_end
        box_length = box_x_end - box_x_start

        if box_length > 0:
            box = board.add_geom()
            box.name = f"hwall{i}"
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.pos = [(box_x_start + box_x_end) / 2.0, y, z_pos]
            box.size = [box_length / 2.0, visual_wall_r, wall_height]
            box.rgba = wall_rgba
            box.contype = 2
            box.conaffinity = 0

        if needs_cap_start:
            cap = board.add_geom()
            cap.name = f"hwall{i}_cap_s"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x_start + visual_wall_r, y, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0

        if needs_cap_end:
            cap = board.add_geom()
            cap.name = f"hwall{i}_cap_e"
            cap.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [x_end - visual_wall_r, y, z_pos]
            cap.size = [visual_wall_r, wall_height, 0]
            cap.rgba = wall_rgba
            cap.contype = 2
            cap.conaffinity = 0


def _add_actuators(spec):
    """Add motor actuators for the two joints."""
    # Alpha actuator
    act_alpha = spec.add_actuator()
    act_alpha.name = "alpha_motor"
    act_alpha.target = "alpha_joint"
    act_alpha.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act_alpha.gear = [GEAR_ALPHA, 0, 0, 0, 0, 0]
    act_alpha.ctrlrange = [-1.0, 1.0]
    act_alpha.ctrllimited = True
    act_alpha.dyntype = mujoco.mjtDyn.mjDYN_FILTER
    act_alpha.dynprm = [DYNPRM_TAU_ALPHA, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Beta actuator
    act_beta = spec.add_actuator()
    act_beta.name = "beta_motor"
    act_beta.target = "beta_joint"
    act_beta.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act_beta.gear = [GEAR_BETA, 0, 0, 0, 0, 0]
    act_beta.ctrlrange = [-1.0, 1.0]
    act_beta.ctrllimited = True
    act_beta.dyntype = mujoco.mjtDyn.mjDYN_FILTER
    act_beta.dynprm = [DYNPRM_TAU_BETA, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _add_marble(world, start_pos: np.ndarray):
    """Add the marble to the world."""
    marble = world.add_body()
    marble.name = "marble"
    marble.pos = [start_pos[0], start_pos[1], 0.0793]

    # Inertial properties
    marble.ipos = [0, 0, 0]
    marble.mass = MARBLE_MASS
    inertia_val = 2.0 / 5.0 * MARBLE_MASS * MARBLE_RADIUS**2
    marble.inertia = [inertia_val, inertia_val, inertia_val]

    # Free joint
    joint = marble.add_joint()
    joint.type = mujoco.mjtJoint.mjJNT_FREE

    # Sphere geometry
    geom = marble.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [MARBLE_RADIUS, MARBLE_RADIUS, MARBLE_RADIUS]
    geom.rgba = [0.0, 0.4, 1.0, 1.0]
    geom.contype = 4
    geom.conaffinity = 7
    geom.friction = list(MARBLE_FRICTION)
    geom.priority = 1
    geom.solref = list(MARBLE_SOLREF)


# ============================================================================
# GYMNASIUM ENVIRONMENT
# ============================================================================


class CyberRunnerEnv(gym.Env):
    """
    Simplified CyberRunner environment.

    Observation space: [joint_alpha, joint_beta, ball_x, ball_y, rel_path_points...]
        - 4 base values + 20 relative path point values = 24 total
        - All values have noise applied (per-episode bias + per-step noise)

    Action space: [-1, 1]^2 for alpha and beta motor commands

    Reward: Path progress (change in distance along path) + goal bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_length: int = 2000,
        randomize_init_pos: bool = False,
        include_vision: bool = True,
        reward_every_n_waypoints: int = 3,
        hole_penalty: float = 5.0,
        checkpoint_radius: float = 0.010,
        checkpoint_hold_steps: int = 12,
        checkpoint_speed_threshold: float = 0.015,
        checkpoint_arrival_reward: float = 0.2,
        checkpoint_stabilize_reward: float = 1.0,
        checkpoint_hold_reward: float = 0.02,
        safe_hole_margin: float = 0.004,
        checkpoint_speed_ema_alpha: float = 0.8,
        checkpoint_include_corridors: bool = True,
        prior_mode: bool = False,
        prior_task: str = "checkpoint",
        prior_spawn_source: str = "dense_path",
        prior_start_waypoint_window: int = 3,
        prior_init_ball_speed: float = 0.0,
        prior_init_tilt_frac: float = 0.0,
        prior_min_checkpoint_start_dist: float = 0.02,
        prior_max_checkpoint_start_dist: float = 0.12,
        prior_spawn_min_hole_margin: float = 0.02,
        prior_start_point_spacing: float = 0.01,
        prior_spawn_merge_radius: float = 0.0,
        checkpoint_progress_reward_scale: float = 20.0,
        terminate_on_checkpoint_stabilized: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.episode_length = episode_length
        self.randomize_init_pos = randomize_init_pos
        self.include_vision = include_vision
        self.reward_every_n_waypoints = reward_every_n_waypoints
        self.hole_penalty = hole_penalty
        self.checkpoint_radius = checkpoint_radius
        self.checkpoint_hold_steps = checkpoint_hold_steps
        self.checkpoint_speed_threshold = checkpoint_speed_threshold
        self.checkpoint_arrival_reward = checkpoint_arrival_reward
        self.checkpoint_stabilize_reward = checkpoint_stabilize_reward
        self.checkpoint_hold_reward = checkpoint_hold_reward
        self.safe_hole_margin = safe_hole_margin
        self.checkpoint_speed_ema_alpha = checkpoint_speed_ema_alpha
        self.checkpoint_include_corridors = checkpoint_include_corridors
        self.prior_mode = prior_mode
        self.prior_task = prior_task
        self.prior_spawn_source = prior_spawn_source
        self.prior_start_waypoint_window = prior_start_waypoint_window
        self.prior_init_ball_speed = prior_init_ball_speed
        self.prior_init_tilt_frac = prior_init_tilt_frac
        self.prior_min_checkpoint_start_dist = prior_min_checkpoint_start_dist
        self.prior_max_checkpoint_start_dist = prior_max_checkpoint_start_dist
        self.prior_spawn_min_hole_margin = prior_spawn_min_hole_margin
        self.prior_start_point_spacing = prior_start_point_spacing
        self.prior_spawn_merge_radius = prior_spawn_merge_radius
        self.checkpoint_progress_reward_scale = checkpoint_progress_reward_scale
        self.terminate_on_checkpoint_stabilized = terminate_on_checkpoint_stabilized

        # Load maze layout
        self.walls_h, self.walls_v, self.holes, self.waypoints = get_hard_layout()
        self._wall_starts = np.vstack([
            np.stack([self.walls_h[:, 0], self.walls_h[:, 2]], axis=1),
            np.stack([self.walls_v[:, 2], self.walls_v[:, 0]], axis=1),
        ]).astype(np.float32)
        self._wall_ends = np.vstack([
            np.stack([self.walls_h[:, 1], self.walls_h[:, 2]], axis=1),
            np.stack([self.walls_v[:, 2], self.walls_v[:, 1]], axis=1),
        ]).astype(np.float32)
        self.checkpoint_points = select_safe_checkpoints(
            self.waypoints,
            self.holes,
            self.walls_h,
            self.walls_v,
            self.reward_every_n_waypoints,
            include_corridors=self.checkpoint_include_corridors,
        )
        self.prior_start_points = self._build_prior_start_points()

        # Precompute path data
        self.seg_lengths, self.cum_distances = compute_waypoint_distances(self.waypoints)
        self.goal_pos = self.waypoints[-1]

        # Build model
        self.model = build_model(
            self.walls_h,
            self.walls_v,
            self.holes,
            self.waypoints,
            self.checkpoint_points,
        )
        self.data = mujoco.MjData(self.model)

        # Get body IDs
        self.board_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "board")
        self.marble_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "marble")

        # Observation space (always a Dict)
        obs_spaces = {
            "states": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "checkpoint": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        }
        if self.include_vision:
            self.img_size = 64
            obs_spaces["image"] = spaces.Box(0, 255, (self.img_size, self.img_size, 3), np.uint8)
            self.vision_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "vision_cam")
            self.vision_renderer = mujoco.Renderer(self.model, height=self.img_size, width=self.img_size)
            # Disable expensive rendering features for 64x64 observation
            self.vision_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.vision_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self.vision_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            # Disable decoration overlays to reduce update_scene work
            self.vision_vopt = mujoco.MjvOption()
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False
            self.vision_vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        else:
            self.vision_renderer = None
        self.observation_space = spaces.Dict(obs_spaces)

        # Action: alpha and beta motor commands
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Rendering
        self.renderer = None
        self.viewer = None

        # Episode state
        self._step_count = 0
        self._prev_progress = 0.0
        self._seg_idx = 0
        self._closest_point = np.zeros(2, dtype=np.float32)
        self._obs_bias = None
        self._path_detected = False
        self._max_checkpoint_reached = 0
        self._active_checkpoint_idx = 0
        self._stable_steps = 0
        self._in_checkpoint_prev = False
        self._ball_speed = 0.0
        self._ball_speed_true = 0.0
        self._min_hole_distance = np.inf
        self._prev_ball_pos = np.zeros(2, dtype=np.float32)
        self._prev_ball_pos_noisy = np.zeros(2, dtype=np.float32)
        self._prev_checkpoint_dist = np.inf
        self._success = False

    def _build_prior_start_points(self) -> np.ndarray:
        """Recoverable start states for prior-mode resets."""
        if self.prior_spawn_source == "waypoints":
            candidates = np.asarray(self.waypoints, dtype=np.float32)
        else:
            spacing = float(self.prior_start_point_spacing)
            samples = [self.waypoints[0]]
            for i in range(len(self.waypoints) - 1):
                start = self.waypoints[i]
                end = self.waypoints[i + 1]
                seg = end - start
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-8:
                    continue
                num = max(1, int(np.ceil(seg_len / spacing)))
                ts = np.linspace(0.0, 1.0, num + 1, endpoint=False)[1:]
                for t in ts:
                    samples.append((1.0 - t) * start + t * end)
            samples.append(self.waypoints[-1])
            candidates = np.asarray(samples, dtype=np.float32)

        if len(self.checkpoint_points) == 0:
            return candidates

        min_hole_dists = np.linalg.norm(candidates[:, None, :] - self.holes[None, :, :], axis=2).min(axis=1)
        safe_mask = min_hole_dists > (HOLE_RADIUS + self.prior_spawn_min_hole_margin)
        if self.prior_task == "checkpoint":
            dists = np.linalg.norm(candidates[:, None, :] - self.checkpoint_points[None, :, :], axis=2)
            nearest_dists = dists.min(axis=1)
            safe_mask &= (
                (nearest_dists >= self.prior_min_checkpoint_start_dist)
                & (nearest_dists <= self.prior_max_checkpoint_start_dist)
            )
        filtered = candidates[safe_mask]
        if not len(filtered):
            return candidates

        merge_radius = float(self.prior_spawn_merge_radius)
        if merge_radius <= 0.0 or len(filtered) <= 1:
            return filtered

        _, progresses = _project_points_to_path(filtered, self.waypoints)
        order = np.argsort(progresses)
        kept: list[np.ndarray] = []
        merge_radius_sq = merge_radius * merge_radius
        for idx in order:
            point = filtered[idx]
            if any(np.sum((point - prev) ** 2) < merge_radius_sq for prev in kept):
                continue
            kept.append(point)
        return np.asarray(kept, dtype=np.float32)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Set initial marble position
        if self.prior_mode and "spawn_point" in options:
            init_pos = np.asarray(options["spawn_point"], dtype=np.float32)
            dists = np.linalg.norm(self.checkpoint_points - init_pos[None], axis=1)
            self._active_checkpoint_idx = int(np.argmin(dists))
        elif self.prior_mode:
            sampled_points = self.prior_start_points[self.np_random.permutation(len(self.prior_start_points))]
            init_pos = None
            chosen_checkpoint_idx = None

            for candidate_pos in sampled_points:
                dists = np.linalg.norm(self.checkpoint_points - candidate_pos[None], axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest_dist = float(dists[nearest_idx])
                min_hole_dist = self._compute_min_hole_distance(candidate_pos)
                recoverable_from_holes = (
                    min_hole_dist > (HOLE_RADIUS + self.prior_spawn_min_hole_margin)
                )
                if self.prior_task != "checkpoint":
                    init_pos = candidate_pos
                    chosen_checkpoint_idx = nearest_idx
                    break
                if (
                    self.prior_min_checkpoint_start_dist <= nearest_dist <= self.prior_max_checkpoint_start_dist
                    and recoverable_from_holes
                ):
                    init_pos = candidate_pos
                    chosen_checkpoint_idx = nearest_idx
                    break

            if init_pos is None:
                safe_candidates = []
                for candidate_pos in sampled_points:
                    min_hole_dist = self._compute_min_hole_distance(candidate_pos)
                    if min_hole_dist > (HOLE_RADIUS + self.prior_spawn_min_hole_margin):
                        safe_candidates.append(candidate_pos)
                init_pos = safe_candidates[0] if safe_candidates else sampled_points[0]
                dists = np.linalg.norm(self.checkpoint_points - init_pos[None], axis=1)
                chosen_checkpoint_idx = int(np.argmin(dists))

            self._active_checkpoint_idx = int(chosen_checkpoint_idx)
        elif self.randomize_init_pos:
            idx = self.np_random.integers(0, len(self.waypoints))
            init_pos = self.waypoints[idx]
        else:
            init_pos = self.waypoints[0]

        # Set marble qpos (free joint: 3 pos + 4 quat)
        # qpos layout: [alpha, beta, marble_x, marble_y, marble_z, qw, qx, qy, qz]
        self.data.qpos[2] = init_pos[0]
        self.data.qpos[3] = init_pos[1]
        self.data.qpos[4] = 0.0793  # Height above board
        self.data.qpos[5:9] = [1, 0, 0, 0]  # Identity quaternion

        # Prior-mode handoff randomization: initial tilt + marble velocity so the
        # prior is trained on states it will actually see when the main policy
        # hands over control (fast-moving ball, non-zero board tilt).
        if self.prior_mode:
            tilt_frac = float(self.prior_init_tilt_frac)
            if tilt_frac > 0.0:
                self.data.qpos[0] = self.np_random.uniform(
                    RANGE_ALPHA[0] * tilt_frac, RANGE_ALPHA[1] * tilt_frac
                )
                self.data.qpos[1] = self.np_random.uniform(
                    RANGE_BETA[0] * tilt_frac, RANGE_BETA[1] * tilt_frac
                )
            v_max = float(self.prior_init_ball_speed)
            if v_max > 0.0:
                theta = self.np_random.uniform(0.0, 2 * np.pi)
                speed = self.np_random.uniform(0.0, v_max)
                self.data.qvel[2] = speed * np.cos(theta)
                self.data.qvel[3] = speed * np.sin(theta)

        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)

        # Generate per-episode observation bias
        self._obs_bias = {
            "ball": self.np_random.uniform(-BALL_POS_NOISE, BALL_POS_NOISE, size=2),
            "joint": self.np_random.uniform(-JOINT_ANGLE_NOISE, JOINT_ANGLE_NOISE, size=2),
        }

        # Reset episode state
        self._step_count = 0
        self._max_checkpoint_reached = 0
        if not self.prior_mode:
            self._active_checkpoint_idx = 0
        self._stable_steps = 0
        self._in_checkpoint_prev = False
        self._success = False
        ball_pos = self._get_ball_pos_board_frame()
        self._prev_ball_pos = ball_pos.copy()
        ball_noise = self.np_random.uniform(-BALL_POS_NOISE, BALL_POS_NOISE, size=2)
        ball_pos_noisy = ball_pos + self._obs_bias["ball"] + ball_noise
        self._prev_ball_pos_noisy = ball_pos_noisy.astype(np.float32)
        self._ball_speed = 0.0
        self._ball_speed_true = 0.0
        self._min_hole_distance = self._compute_min_hole_distance(ball_pos)
        active_checkpoint = self._get_active_checkpoint_waypoint()
        self._prev_checkpoint_dist = (
            float(np.linalg.norm(ball_pos - active_checkpoint))
            if active_checkpoint is not None
            else np.inf
        )
        self._prev_progress, self._seg_idx, _, self._closest_point = compute_path_progress(
            ball_pos, self.waypoints, self.seg_lengths, self.cum_distances, self.walls_h, self.walls_v, self.holes
        )
        self._path_detected = self._prev_progress >= 0

        obs = self._get_obs()
        info = self._build_info(ball_pos, self._prev_progress)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)

        # Step physics
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get ball state
        ball_pos = self._get_ball_pos_board_frame()
        dt = TIMESTEP * FRAME_SKIP
        self._ball_speed_true = float(np.linalg.norm(ball_pos - self._prev_ball_pos) / max(dt, 1e-8))
        self._prev_ball_pos = ball_pos.copy()
        ball_noise = self.np_random.uniform(-BALL_POS_NOISE, BALL_POS_NOISE, size=2)
        ball_pos_noisy = ball_pos + self._obs_bias["ball"] + ball_noise
        obs_ball_speed = float(np.linalg.norm(ball_pos_noisy - self._prev_ball_pos_noisy) / max(dt, 1e-8))
        self._prev_ball_pos_noisy = ball_pos_noisy.astype(np.float32)
        alpha = self.checkpoint_speed_ema_alpha
        self._ball_speed = float(alpha * self._ball_speed + (1.0 - alpha) * obs_ball_speed)
        self._min_hole_distance = self._compute_min_hole_distance(ball_pos)

        # Compute path progress and get segment info
        curr_progress, self._seg_idx, _, self._closest_point = compute_path_progress(
            ball_pos, self.waypoints, self.seg_lengths, self.cum_distances, self.walls_h, self.walls_v, self.holes
        )
        self._path_detected = curr_progress >= 0

        # Compute reward
        reward = self._compute_reward(ball_pos, curr_progress, self._seg_idx)

        # Check termination
        terminated, truncated, info = self._check_termination(ball_pos)

        # Update state
        if curr_progress >= 0:
            self._prev_progress = curr_progress

        obs = self._get_obs()
        info.update(self._build_info(ball_pos, curr_progress))

        return obs, reward, terminated, truncated, info

    def _compute_min_hole_distance(self, ball_pos: np.ndarray) -> float:
        hole_distances = np.linalg.norm(self.holes - ball_pos, axis=1)
        return float(np.min(hole_distances))

    def _min_wall_distance(self, ball_pos: np.ndarray) -> float:
        """Closest distance from ball center to any wall segment or board edge."""
        wall_d = float(_point_to_segment_distance(ball_pos[None, :], self._wall_starts, self._wall_ends).min())
        edge_d = float(min(ball_pos[0], BOARD_WIDTH - ball_pos[0],
                           ball_pos[1], BOARD_HEIGHT - ball_pos[1])) + WALL_RADIUS
        return min(wall_d, edge_d)

    def _get_active_checkpoint_waypoint(self) -> np.ndarray | None:
        if self._active_checkpoint_idx >= len(self.checkpoint_points):
            return None
        return self.checkpoint_points[self._active_checkpoint_idx]

    def _build_info(self, ball_pos: np.ndarray, curr_progress: float) -> dict[str, Any]:
        active_checkpoint = self._get_active_checkpoint_waypoint()
        checkpoint_dist = (
            float(np.linalg.norm(ball_pos - active_checkpoint))
            if active_checkpoint is not None
            else np.inf
        )
        return {
            "path_progress": float(curr_progress),
            "checkpoint_dist": checkpoint_dist,
            "ball_speed": float(self._ball_speed),
            "ball_speed_true": float(self._ball_speed_true),
            "min_hole_distance": float(self._min_hole_distance),
            "safe_hole_margin": float(self._min_hole_distance - HOLE_RADIUS),
            "active_checkpoint_idx": int(self._active_checkpoint_idx),
            "unlocked_checkpoint_idx": int(self._max_checkpoint_reached),
            "stable_steps": int(self._stable_steps),
            "success": float(self._success),
            "log_path_progress": np.array([curr_progress], dtype=np.float32),
            "log_checkpoint_dist": np.array([checkpoint_dist], dtype=np.float32),
            "log_ball_speed": np.array([self._ball_speed], dtype=np.float32),
            "log_ball_speed_true": np.array([self._ball_speed_true], dtype=np.float32),
            "log_min_hole_distance": np.array([self._min_hole_distance], dtype=np.float32),
            "log_safe_hole_margin": np.array([self._min_hole_distance - HOLE_RADIUS], dtype=np.float32),
            "log_active_checkpoint_idx": np.array([self._active_checkpoint_idx], dtype=np.float32),
            "log_unlocked_checkpoint_idx": np.array([self._max_checkpoint_reached], dtype=np.float32),
            "log_stable_steps": np.array([self._stable_steps], dtype=np.float32),
            "log_success": np.array([float(self._success)], dtype=np.float32),
        }

    def _get_ball_pos_board_frame(self) -> np.ndarray:
        """Get ball position in board frame."""
        # Get world positions
        board_pos = self.data.xpos[self.board_body_id]
        board_mat = self.data.xmat[self.board_body_id].reshape(3, 3)
        marble_pos = self.data.xpos[self.marble_body_id]

        # Transform to board frame
        rel_pos = marble_pos - board_pos
        pos_board = board_mat.T @ rel_pos

        return pos_board[:2].astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get observation with noise.

        Returns dict with:
            "states" (10-dim):
                [0:2] Joint angles (alpha, beta) with noise
                [2:4] Ball position (x, y) with noise
                [4:6] Vector from ball to closest visible path point
                [6:8] Vector from ball to next waypoint
                [8:10] Vector from ball to waypoint after next
            "image" (64x64x3, only if include_vision):
                Cropped board image centered on ball
        """
        # Per-step noise
        ball_noise = self.np_random.uniform(-BALL_POS_NOISE, BALL_POS_NOISE, size=2)
        joint_noise = self.np_random.uniform(-JOINT_ANGLE_NOISE, JOINT_ANGLE_NOISE, size=2)

        # Joint angles with bias + noise
        joint_pos = self.data.qpos[:2] + self._obs_bias["joint"] + joint_noise

        # Ball position with bias + noise
        ball_pos = self._get_ball_pos_board_frame()
        ball_pos_noisy = ball_pos + self._obs_bias["ball"] + ball_noise

        # Vector to closest visible path point
        vec_to_closest = self._closest_point - ball_pos_noisy

        # Vectors to next two waypoints
        num_waypoints = len(self.waypoints)
        next_wp_idx = min(self._seg_idx + 1, num_waypoints - 1)
        next_next_wp_idx = min(self._seg_idx + 2, num_waypoints - 1)

        vec_to_next_wp = self.waypoints[next_wp_idx] - ball_pos_noisy
        vec_to_next_next_wp = self.waypoints[next_next_wp_idx] - ball_pos_noisy

        if not self._path_detected:
            vec_to_closest = np.zeros(2, dtype=np.float32)
            vec_to_next_wp = np.zeros(2, dtype=np.float32)
            vec_to_next_next_wp = np.zeros(2, dtype=np.float32)

        states = np.concatenate(
            [joint_pos, ball_pos_noisy, vec_to_closest, vec_to_next_wp, vec_to_next_next_wp]
        ).astype(np.float32)

        active_checkpoint = self._get_active_checkpoint_waypoint()
        if active_checkpoint is None:
            checkpoint_vec = np.zeros(2, dtype=np.float32)
            checkpoint_dist = 0.0
        elif self.prior_mode and self.prior_task == "stabilize":
            checkpoint_vec = np.zeros(2, dtype=np.float32)
            checkpoint_dist = 0.0
        else:
            checkpoint_vec = active_checkpoint - ball_pos_noisy
            checkpoint_dist = float(np.linalg.norm(checkpoint_vec))

        obs = {
            "states": states,
            "checkpoint": np.array([checkpoint_vec[0], checkpoint_vec[1], checkpoint_dist], dtype=np.float32),
        }

        if self.include_vision:
            # Move vision camera above the ball (in board frame)
            ball_board = self._get_ball_pos_board_frame()
            cam_local = np.array([ball_board[0], ball_board[1], 0.4])
            board_pos = self.data.xpos[self.board_body_id]
            board_mat = self.data.xmat[self.board_body_id].reshape(3, 3)
            self.data.cam_xpos[self.vision_cam_id] = board_pos + board_mat @ cam_local
            # Render directly at 64x64
            self.vision_renderer.update_scene(self.data, camera="vision_cam", scene_option=self.vision_vopt)
            obs["image"] = self.vision_renderer.render()

        return obs

    def _compute_reward(self, ball_pos: np.ndarray, curr_progress: float, seg_idx: int) -> float:
        """Stabilized frontier checkpoints + goal bonus + hole penalty."""
        if self.prior_mode and self.prior_task == "stabilize":
            touching_wall = self._min_wall_distance(ball_pos) < (WALL_RADIUS + MARBLE_RADIUS + 0.002)
            safe_enough = self._min_hole_distance > (HOLE_RADIUS + self.safe_hole_margin)
            stable_here = touching_wall and safe_enough and self._ball_speed < self.checkpoint_speed_threshold
            safe_margin = max(self._min_hole_distance - (HOLE_RADIUS + self.safe_hole_margin), 0.0)

            reward = -0.05 * self._ball_speed + 2.0 * min(safe_margin, 0.02)
            if touching_wall:
                reward += 0.05
            if stable_here:
                self._stable_steps += 1
                reward += self.checkpoint_hold_reward
            else:
                self._stable_steps = 0

            if self._stable_steps >= self.checkpoint_hold_steps:
                reward += self.checkpoint_stabilize_reward
                self._success = True
                self._stable_steps = 0

            hole_reward = -self.hole_penalty if self._min_hole_distance < HOLE_RADIUS else 0.0
            self._in_checkpoint_prev = False
            self._prev_checkpoint_dist = np.inf
            return reward + hole_reward

        checkpoint_reward = 0.0
        active_checkpoint = self._get_active_checkpoint_waypoint()
        in_checkpoint = False

        if active_checkpoint is not None and curr_progress >= 0:
            checkpoint_dist = float(np.linalg.norm(ball_pos - active_checkpoint))
            if self.prior_mode:
                # Asymmetric shaping: reward progress toward, don't penalize moving away
                # (avoids the "stop anywhere" local optimum).
                if np.isfinite(self._prev_checkpoint_dist):
                    progress_delta = self._prev_checkpoint_dist - checkpoint_dist
                    checkpoint_reward += self.checkpoint_progress_reward_scale * max(progress_delta, 0.0)
                # Dense distance penalty: always pay for being far → "stop at checkpoint"
                # is strictly better than "stop anywhere else".
                checkpoint_reward -= 0.1 * checkpoint_dist
            in_checkpoint = checkpoint_dist < self.checkpoint_radius
            # Wall-contact is only relevant once the ball is actually inside the
            # checkpoint basin. Avoid the expensive wall-distance query elsewhere.
            touching_wall = False
            if in_checkpoint:
                wall_contact_dist = self._min_wall_distance(ball_pos)
                touching_wall = wall_contact_dist < (WALL_RADIUS + MARBLE_RADIUS + 0.002)
            stable_here = (
                in_checkpoint
                and touching_wall
                and self._ball_speed < self.checkpoint_speed_threshold
                and self._min_hole_distance > (HOLE_RADIUS + self.safe_hole_margin)
            )
            if in_checkpoint and not self._in_checkpoint_prev and not self.prior_mode:
                checkpoint_reward += self.checkpoint_arrival_reward
            if stable_here:
                self._stable_steps += 1
                # Hold reward scaled by closeness to center (1.0 at center, ~0.5 at edge)
                closeness = 1.0 - 0.5 * (checkpoint_dist / self.checkpoint_radius)
                checkpoint_reward += self.checkpoint_hold_reward * closeness
            else:
                self._stable_steps = 0

            if self._stable_steps >= self.checkpoint_hold_steps:
                checkpoint_reward += self.checkpoint_stabilize_reward
                self._success = True
                if not self.prior_mode:
                    self._max_checkpoint_reached += 1
                    self._active_checkpoint_idx += 1
                self._stable_steps = 0
                in_checkpoint = False
        else:
            self._stable_steps = 0

        self._in_checkpoint_prev = in_checkpoint
        if active_checkpoint is not None:
            self._prev_checkpoint_dist = float(np.linalg.norm(ball_pos - active_checkpoint))

        dist_to_goal = np.linalg.norm(ball_pos - self.goal_pos)
        goal_reward = 0.0 if self.prior_mode else (GOAL_BONUS if dist_to_goal < GOAL_THRESHOLD else 0.0)

        hole_reward = -self.hole_penalty if self._min_hole_distance < HOLE_RADIUS else 0.0

        return checkpoint_reward + goal_reward + hole_reward

    def _check_termination(self, ball_pos: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        """Check termination conditions."""
        info = {}
        terminated = False
        truncated = False

        # Check if in hole
        hole_distances = np.linalg.norm(self.holes - ball_pos, axis=1)
        in_hole = np.any(hole_distances < HOLE_RADIUS)
        if in_hole:
            terminated = True
            info["termination_reason"] = "hole"

        # Check goal reached
        dist_to_goal = np.linalg.norm(ball_pos - self.goal_pos)
        if (not self.prior_mode) and dist_to_goal < GOAL_THRESHOLD:
            terminated = True
            info["termination_reason"] = "goal"

        if self.terminate_on_checkpoint_stabilized and self._success:
            terminated = True
            info["termination_reason"] = (
                "stabilized" if self.prior_mode and self.prior_task == "stabilize" else "checkpoint"
            )

        # Check timeout
        if self._step_count >= self.episode_length:
            truncated = True
            info["termination_reason"] = "timeout"

        return terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception as e:
                # Headless environment - rendering not available
                import warnings

                warnings.warn(f"Rendering not available: {e}")
                return None

        self.renderer.update_scene(self.data, camera="board")

        if self.render_mode == "rgb_array":
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    from mujoco import viewer

                    self.viewer = viewer.launch_passive(self.model, self.data)
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "board")
                except Exception as e:
                    import warnings

                    warnings.warn(f"Viewer not available: {e}")
                    return None
            self.viewer.sync()
        return None

    def close(self):
        if self.vision_renderer is not None:
            self.vision_renderer.close()
            self.vision_renderer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# R2-DREAMER WRAPPER
# ============================================================================


class CyberRunner(gym.Env):
    """Wrapper around CyberRunnerEnv matching R2-Dreamer env interface."""

    def __init__(
        self, name, action_repeat=1, size=(64, 64), seed=0,
        reward_every_n_waypoints=5, hole_penalty=5.0,
        checkpoint_radius=0.010,
        checkpoint_hold_steps=12,
        checkpoint_speed_threshold=0.015,
        checkpoint_arrival_reward=0.2,
        checkpoint_stabilize_reward=1.0,
        checkpoint_hold_reward=0.02,
        safe_hole_margin=0.004,
        checkpoint_speed_ema_alpha=0.8,
        checkpoint_include_corridors=True,
        prior_mode=False,
        prior_task="checkpoint",
        prior_spawn_source="dense_path",
        prior_start_waypoint_window=3,
        prior_init_ball_speed=0.0,
        prior_init_tilt_frac=0.0,
        prior_min_checkpoint_start_dist=0.02,
        prior_max_checkpoint_start_dist=0.12,
        prior_spawn_min_hole_margin=0.02,
        prior_start_point_spacing=0.01,
        prior_spawn_merge_radius=0.0,
        checkpoint_progress_reward_scale=20.0,
        terminate_on_checkpoint_stabilized=False,
    ):
        include_vision = name == "vision"
        self._env = CyberRunnerEnv(
            render_mode="rgb_array",
            episode_length=1_000_000,
            randomize_init_pos=False,
            include_vision=include_vision,
            reward_every_n_waypoints=reward_every_n_waypoints,
            hole_penalty=hole_penalty,
            checkpoint_radius=checkpoint_radius,
            checkpoint_hold_steps=checkpoint_hold_steps,
            checkpoint_speed_threshold=checkpoint_speed_threshold,
            checkpoint_arrival_reward=checkpoint_arrival_reward,
            checkpoint_stabilize_reward=checkpoint_stabilize_reward,
            checkpoint_hold_reward=checkpoint_hold_reward,
            safe_hole_margin=safe_hole_margin,
            checkpoint_speed_ema_alpha=checkpoint_speed_ema_alpha,
            checkpoint_include_corridors=checkpoint_include_corridors,
            prior_mode=prior_mode,
            prior_task=prior_task,
            prior_spawn_source=prior_spawn_source,
            prior_start_waypoint_window=prior_start_waypoint_window,
            prior_init_ball_speed=prior_init_ball_speed,
            prior_init_tilt_frac=prior_init_tilt_frac,
            prior_min_checkpoint_start_dist=prior_min_checkpoint_start_dist,
            prior_max_checkpoint_start_dist=prior_max_checkpoint_start_dist,
            prior_spawn_min_hole_margin=prior_spawn_min_hole_margin,
            prior_start_point_spacing=prior_start_point_spacing,
            prior_spawn_merge_radius=prior_spawn_merge_radius,
            checkpoint_progress_reward_scale=checkpoint_progress_reward_scale,
            terminate_on_checkpoint_stabilized=terminate_on_checkpoint_stabilized,
        )
        self._action_repeat = action_repeat
        self._size = size
        self._include_vision = include_vision
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces_dict = {
            "states": gym.spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "checkpoint": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "log_path_progress": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_checkpoint_dist": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_ball_speed": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_ball_speed_true": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_min_hole_distance": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_safe_hole_margin": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_active_checkpoint_idx": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_unlocked_checkpoint_idx": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_stable_steps": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        if self._include_vision:
            spaces_dict["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces_dict)

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        last_info = None
        for _ in range(self._action_repeat):
            obs, rew, terminated, truncated, info = self._env.step(action)
            reward += rew
            last_info = info
            if terminated or truncated:
                break
        is_last = terminated or truncated
        out = {
            "is_first": False,
            "is_last": is_last,
            "is_terminal": terminated,
            "states": obs["states"],
            "checkpoint": obs["checkpoint"],
            "log_path_progress": last_info["log_path_progress"],
            "log_checkpoint_dist": last_info["log_checkpoint_dist"],
            "log_ball_speed": last_info["log_ball_speed"],
            "log_ball_speed_true": last_info["log_ball_speed_true"],
            "log_min_hole_distance": last_info["log_min_hole_distance"],
            "log_safe_hole_margin": last_info["log_safe_hole_margin"],
            "log_active_checkpoint_idx": last_info["log_active_checkpoint_idx"],
            "log_unlocked_checkpoint_idx": last_info["log_unlocked_checkpoint_idx"],
            "log_stable_steps": last_info["log_stable_steps"],
            "log_success": last_info["log_success"],
        }
        if self._include_vision:
            out["image"] = obs["image"]
        return out, reward, is_last, {}

    def reset(self, **kwargs):
        obs, info = self._env.reset()
        out = {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "states": obs["states"],
            "checkpoint": obs["checkpoint"],
            "log_path_progress": info["log_path_progress"],
            "log_checkpoint_dist": info["log_checkpoint_dist"],
            "log_ball_speed": info["log_ball_speed"],
            "log_ball_speed_true": info["log_ball_speed_true"],
            "log_min_hole_distance": info["log_min_hole_distance"],
            "log_safe_hole_margin": info["log_safe_hole_margin"],
            "log_active_checkpoint_idx": info["log_active_checkpoint_idx"],
            "log_unlocked_checkpoint_idx": info["log_unlocked_checkpoint_idx"],
            "log_stable_steps": info["log_stable_steps"],
            "log_success": info["log_success"],
        }
        if self._include_vision:
            out["image"] = obs["image"]
        return out

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Quick test
    env = CyberRunnerEnv(render_mode="human", randomize_init_pos=True, include_vision=True)

    obs, info = env.reset()
    print(f"States shape: {obs['states'].shape}")
    if "image" in obs:
        print(f"Image shape: {obs['image'].shape}")
    print(f"Initial states: {obs['states'][:4]}")  # joints + ball pos
    print(f"Initial progress: {info['path_progress']:.4f}")

    total_reward = 0
    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(1 / 60)  # Real-time playback at 60 FPS

        if terminated or truncated:
            print(f"Episode ended at step {i}: {info.get('termination_reason', 'unknown')}")
            print(f"Total reward: {total_reward:.4f}")
            break

    env.close()
