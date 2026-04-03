import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Dict, Any
import time
import tempfile


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
RANGE_BETA = (-0.10424974775885551, 0.10424974775885551)   # ~±5.97°
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
FRAME_SKIP = 10           # 60Hz control

# Observation noise
BALL_POS_NOISE = 0.001              # 1mm
JOINT_ANGLE_NOISE = 0.25 * np.pi / 180  # 0.25 degrees

# Reward parameters
PROGRESS_SCALE = 1.0
GOAL_BONUS = 10.0
GOAL_THRESHOLD = 0.004  # 4mm


# ============================================================================
# HARD MAZE LAYOUT
# ============================================================================

def get_hard_layout() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the hard maze layout arrays."""
    walls_h = np.array([
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
        [0.2525, 0.276, 0.210]
    ], dtype=np.float32)

    walls_v = np.array([
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
        [0., 0.018, 0.255],
        [0.098, 0.121, 0.021],
        [0.098, 0.121, 0.045],
        [0.093, 0.121, 0.069],
        [0.093, 0.121, 0.092],
        [0.021, 0.082, 0.044],
        [0.034, 0.052, 0.069],
        [0., 0.014, 0.069],
        [0.021, 0.06, 0.092],
        [0.021, 0.038, 0.116],
        [0.077, 0.132, 0.116],
        [0.021, 0.06, 0.139],
        [0.099, 0.117, 0.139]
    ], dtype=np.float32)

    holes = np.array([
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
    ], dtype=np.float32)

    waypoints = np.array([
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
    ], dtype=np.float32)

    return walls_h, walls_v, holes, waypoints


# ============================================================================
# PATH UTILITIES
# ============================================================================

def compute_waypoint_distances(waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute segment lengths and cumulative distances along waypoints."""
    seg_vectors = waypoints[1:] - waypoints[:-1]
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)
    cum_distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return seg_lengths.astype(np.float32), cum_distances.astype(np.float32)


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
) -> Tuple[float, int, float, np.ndarray]:
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
    seg_len_sq = np.sum(seg_vecs ** 2, axis=1)  # [S]
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
    h_seg_ends = np.stack([walls_h[:, 1], walls_h[:, 2]], axis=1)    # [H, 2]
    
    # Vertical walls: [V, 3] -> segments from (x, y_start) to (x, y_end)
    v_seg_starts = np.stack([walls_v[:, 2], walls_v[:, 0]], axis=1)  # [V, 2]
    v_seg_ends = np.stack([walls_v[:, 2], walls_v[:, 1]], axis=1)    # [V, 2]
    
    # Combine all wall segments
    wall_starts = np.vstack([h_seg_starts, v_seg_starts])  # [H+V, 2]
    wall_ends = np.vstack([h_seg_ends, v_seg_ends])        # [H+V, 2]
    wall_vecs = wall_ends - wall_starts                     # [H+V, 2]
    
    # Vectorized ray-wall intersection for all rays × all walls
    ray_dirs_exp = ray_dirs[:, np.newaxis, :]      # [R, 1, 2]
    wall_vecs_exp = wall_vecs[np.newaxis, :, :]    # [1, W, 2]
    
    denom_w = ray_dirs_exp[:, :, 0] * wall_vecs_exp[:, :, 1] - ray_dirs_exp[:, :, 1] * wall_vecs_exp[:, :, 0]
    parallel_w = np.abs(denom_w) < 1e-8
    safe_denom_w = np.where(parallel_w, 1.0, denom_w)
    
    diff_w = wall_starts - marble_pos  # [W, 2]
    diff_w_exp = diff_w[np.newaxis, :, :]  # [1, W, 2]
    
    t_w = (diff_w_exp[:, :, 0] * wall_vecs_exp[:, :, 1] - diff_w_exp[:, :, 1] * wall_vecs_exp[:, :, 0]) / safe_denom_w
    s_w = (diff_w_exp[:, :, 0] * ray_dirs_exp[:, :, 1] - diff_w_exp[:, :, 1] * ray_dirs_exp[:, :, 0]) / safe_denom_w
    
    valid_w = (~parallel_w) & (t_w > 1e-6) & (s_w >= 0) & (s_w <= 1)
    wall_dists = np.where(valid_w, t_w, np.inf)  # [R, W]
    min_wall_dist = np.min(wall_dists, axis=1)   # [R] - closest wall for each ray
    
    # === HOLE INTERSECTIONS ===
    oc = marble_pos - holes  # [K, 2]
    oc_exp = oc[np.newaxis, :, :]  # [1, K, 2]
    b = 2.0 * np.sum(oc_exp * ray_dirs_exp, axis=2)  # [R, K]
    c = np.sum(oc * oc, axis=1) - hole_radius ** 2   # [K]
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
    
    seg_vecs_exp = seg_vecs[np.newaxis, :, :]      # [1, S, 2]
    seg_starts_exp = seg_starts[np.newaxis, :, :]  # [1, S, 2]
    
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
    path_distances = cum_distances[:-1][np.newaxis, :] + s_p_clipped * seg_lengths[np.newaxis, :]  # [R, S]
    
    # Mask invalid intersections
    path_distances_valid = np.where(valid_p, path_distances, np.inf)
    
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
    tol: float = 0.003
) -> bool:
    """Check if a wall endpoint is connected to another wall or board edge."""
    # Check board edges
    if (abs(point_x) < tol or
        abs(point_x - BOARD_WIDTH) < tol or
        abs(point_y) < tol or
        abs(point_y - BOARD_HEIGHT) < tol):
        return True

    if is_vertical_wall:
        # Check connection to horizontal walls
        for h_wall in walls_h:
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol:
                if x_start - tol <= point_x <= x_end + tol:
                    return True
        # Check connection to other vertical walls
        for i, v_wall in enumerate(walls_v):
            if i == wall_index:
                continue
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol:
                if abs(y_start - point_y) < tol or abs(y_end - point_y) < tol:
                    return True
    else:
        # Check connection to vertical walls
        for v_wall in walls_v:
            y_start, y_end, x = v_wall
            if abs(x - point_x) < tol:
                if y_start - tol <= point_y <= y_end + tol:
                    return True
        # Check connection to other horizontal walls
        for i, h_wall in enumerate(walls_h):
            if i == wall_index:
                continue
            x_start, x_end, y = h_wall
            if abs(y - point_y) < tol:
                if abs(x_start - point_x) < tol or abs(x_end - point_x) < tol:
                    return True

    return False


def _generate_board_texture(holes: np.ndarray, waypoints: np.ndarray):
    """Generate a board texture image with holes and path baked in."""
    # Board floor geom spans -0.007 to 0.283 in x, -0.007 to 0.238 in y
    # (centered at [0.138, 0.1155] with half-size [0.145, 0.1225])
    # Maze coordinates start at 0, so offset by 0.007 to place them correctly.
    margin = 0.007
    full_w = BOARD_WIDTH + 2 * margin   # 0.290
    full_h = BOARD_HEIGHT + 2 * margin  # 0.245
    scale = 5000
    w = int(full_w * scale)  # 1450
    h = int(full_h * scale)  # 1225
    img = Image.new('RGB', (w, h), (204, 204, 204))
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
    waypoints: np.ndarray
) -> mujoco.MjModel:
    """Build the MuJoCo model using mjSpec."""
    spec = mujoco.MjSpec()
    spec.modelname = "cyberrunner"
    spec.compiler.autolimits = True
    spec.option.timestep = TIMESTEP

    # Board texture with holes and path baked in
    board_img = _generate_board_texture(holes, waypoints)

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
    cam = base.add_camera(
        pos=[0.138, -0.05, 0.4],
        zaxis=[0, -np.sin(angle_rad), np.cos(angle_rad)]
    )
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
    vision_cam = board.add_camera(
        pos=[0.138, 0.1155, 0.4],
        zaxis=[0, 0, 1]
    )
    vision_cam.name = "vision_cam"
    # Narrow FOV to cover ~6cm at camera height (0.4 - 0.0705 = 0.3295m above board)
    vision_cam.fovy = np.degrees(2 * np.arctan(0.03 / 0.3295))

    # Actuators
    _add_actuators(spec)

    # Marble
    _add_marble(world, waypoints[0])

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        board_img.save(f)
        f.flush()
        tex.file = f.name
        return spec.compile()


def _add_board_edges(board):
    """Add collision boxes for board edges."""
    edges = [
        ([0.138, 0.2345, 0.0835], [0.145, 0.0035, 0.0105]),   # Top
        ([0.138, -0.0035, 0.0835], [0.145, 0.0035, 0.0105]),  # Bottom
        ([-0.0035, 0.1155, 0.0835], [0.0035, 0.1155, 0.0105]), # Left
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
    inertia_val = 2.0 / 5.0 * MARBLE_MASS * MARBLE_RADIUS ** 2
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
        render_mode: Optional[str] = None,
        episode_length: int = 2000,
        randomize_init_pos: bool = True,
        include_vision: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.episode_length = episode_length
        self.randomize_init_pos = randomize_init_pos
        self.include_vision = include_vision

        # Load maze layout
        self.walls_h, self.walls_v, self.holes, self.waypoints = get_hard_layout()

        # Precompute path data
        self.seg_lengths, self.cum_distances = compute_waypoint_distances(self.waypoints)
        self.goal_pos = self.waypoints[-1]

        # Build model
        self.model = build_model(self.walls_h, self.walls_v, self.holes, self.waypoints)
        self.data = mujoco.MjData(self.model)

        # Get body IDs
        self.board_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "board")
        self.marble_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "marble")

        # Observation space (always a Dict)
        obs_spaces = {
            "states": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        }
        if self.include_vision:
            self.img_size = 64
            obs_spaces["image"] = spaces.Box(0, 255, (self.img_size, self.img_size, 3), np.uint8)
            self.vision_cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "vision_cam"
            )
            self.vision_renderer = mujoco.Renderer(
                self.model, height=self.img_size, width=self.img_size
            )
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
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Rendering
        self.renderer = None
        self.viewer = None

        # Episode state
        self._step_count = 0
        self._prev_progress = 0.0
        self._seg_idx = 0
        self._closest_point = np.zeros(2, dtype=np.float32)
        self._obs_bias = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Set initial marble position
        if self.randomize_init_pos:
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

        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)

        # Generate per-episode observation bias
        self._obs_bias = {
            'ball': self.np_random.uniform(-BALL_POS_NOISE, BALL_POS_NOISE, size=2),
            'joint': self.np_random.uniform(-JOINT_ANGLE_NOISE, JOINT_ANGLE_NOISE, size=2),
        }

        # Reset episode state
        self._step_count = 0
        ball_pos = self._get_ball_pos_board_frame()
        self._prev_progress, self._seg_idx, _, self._closest_point = compute_path_progress(
            ball_pos, self.waypoints, self.seg_lengths, self.cum_distances,
            self.walls_h, self.walls_v, self.holes
        )

        obs = self._get_obs()
        info = {"path_progress": self._prev_progress}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)

        # Step physics
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get ball state
        ball_pos = self._get_ball_pos_board_frame()

        # Compute path progress and get segment info
        curr_progress, self._seg_idx, _, self._closest_point = compute_path_progress(
            ball_pos, self.waypoints, self.seg_lengths, self.cum_distances,
            self.walls_h, self.walls_v, self.holes
        )

        # Compute reward
        reward = self._compute_reward(ball_pos, curr_progress)

        # Check termination
        terminated, truncated, info = self._check_termination(ball_pos)

        # Update state
        self._prev_progress = curr_progress

        obs = self._get_obs()
        info["path_progress"] = curr_progress

        return obs, reward, terminated, truncated, info

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

    def _get_obs(self) -> Dict[str, np.ndarray]:
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
        joint_pos = self.data.qpos[:2] + self._obs_bias['joint'] + joint_noise

        # Ball position with bias + noise
        ball_pos = self._get_ball_pos_board_frame()
        ball_pos_noisy = ball_pos + self._obs_bias['ball'] + ball_noise

        # Vector to closest visible path point
        vec_to_closest = self._closest_point - ball_pos_noisy

        # Vectors to next two waypoints
        num_waypoints = len(self.waypoints)
        next_wp_idx = min(self._seg_idx + 1, num_waypoints - 1)
        next_next_wp_idx = min(self._seg_idx + 2, num_waypoints - 1)

        vec_to_next_wp = self.waypoints[next_wp_idx] - ball_pos_noisy
        vec_to_next_next_wp = self.waypoints[next_next_wp_idx] - ball_pos_noisy

        states = np.concatenate([
            joint_pos,
            ball_pos_noisy,
            vec_to_closest,
            vec_to_next_wp,
            vec_to_next_next_wp
        ]).astype(np.float32)

        obs = {"states": states}

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

    def _compute_reward(self, ball_pos: np.ndarray, curr_progress: float) -> float:
        """Compute reward based on path progress."""
        # Progress reward (only if both valid)
        if curr_progress >= 0 and self._prev_progress >= 0:
            progress_reward = (curr_progress - self._prev_progress) * PROGRESS_SCALE
        else:
            progress_reward = 0.0

        # Goal bonus
        dist_to_goal = np.linalg.norm(ball_pos - self.goal_pos)
        goal_reward = GOAL_BONUS if dist_to_goal < GOAL_THRESHOLD else 0.0

        return progress_reward + goal_reward

    def _check_termination(self, ball_pos: np.ndarray) -> Tuple[bool, bool, Dict[str, Any]]:
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
        if dist_to_goal < GOAL_THRESHOLD:
            terminated = True
            info["termination_reason"] = "goal"

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
        elif self.render_mode == "human":
            if self.viewer is None:
                try:
                    from mujoco import viewer
                    self.viewer = viewer.launch_passive(self.model, self.data)
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_CAMERA, "board"
                    )
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
        time.sleep(1/60)  # Real-time playback at 60 FPS

        if terminated or truncated:
            print(f"Episode ended at step {i}: {info.get('termination_reason', 'unknown')}")
            print(f"Total reward: {total_reward:.4f}")
            break

    env.close()