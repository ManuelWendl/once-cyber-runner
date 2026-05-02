"""Plot the maze geometry with strict-corner safe checkpoints and spawn points.

Static schematic of the safe-prior env's geometry — useful to sanity-check the
corner set and the spawn distribution before / after a training run. No policy,
no MJX, no rendering — pure NumPy + matplotlib so you can run it anywhere.

Usage:
    python viz_safe_prior_setup.py                  # save to safe_prior_setup.png
    python viz_safe_prior_setup.py --show           # interactive window (Mac)
    python viz_safe_prior_setup.py --out foo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

REPO_DIR = Path(__file__).resolve().parent
VENDOR_DIR = REPO_DIR / ".vendor" / "cyberrunner_ppo"
sys.path.insert(0, str(VENDOR_DIR))

from env_mujoco import (  # noqa: E402
    BOARD_HEIGHT,
    BOARD_WIDTH,
    HOLE_RADIUS,
    MARBLE_RADIUS,
    WALL_RADIUS,
    get_hard_layout,
)


# Inlined NumPy helpers (the same code lives in env_mjx.py but importing that
# module would drag in jax/brax — unnecessary for a static schematic).


def _project_points_to_path(points, waypoints):
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


def select_corner_checkpoints(waypoints, holes, walls_h, walls_v, grid_res=0.002):
    """NumPy-only port (verbatim) of env_mjx.select_corner_checkpoints."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="safe_prior_setup.png")
    p.add_argument("--show", action="store_true", help="Open an interactive window")
    p.add_argument(
        "--n-spawns",
        type=int,
        default=69,
        help="Number of spawn points to overlay (waypoints — 69 = all)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def draw_maze(ax, walls_h, walls_v, holes, waypoints) -> None:
    # Board outline.
    ax.add_patch(
        Rectangle(
            (0, 0), BOARD_WIDTH, BOARD_HEIGHT,
            linewidth=1.5, edgecolor="black", facecolor="#f3f0e6",
        )
    )

    # Walls (h: x_start, x_end, y; v: y_start, y_end, x). Each is a thin rect
    # of full height = 2 * WALL_RADIUS centered on the wall axis.
    wall_t = 2 * WALL_RADIUS
    for x0, x1, y in walls_h:
        ax.add_patch(
            Rectangle(
                (x0, y - WALL_RADIUS), x1 - x0, wall_t,
                linewidth=0, facecolor="#3a3a3a",
            )
        )
    for y0, y1, x in walls_v:
        ax.add_patch(
            Rectangle(
                (x - WALL_RADIUS, y0), wall_t, y1 - y0,
                linewidth=0, facecolor="#3a3a3a",
            )
        )

    # Holes.
    for hx, hy in holes:
        ax.add_patch(
            Circle((hx, hy), HOLE_RADIUS, facecolor="#222", edgecolor="none")
        )

    # Path polyline.
    ax.plot(
        waypoints[:, 0], waypoints[:, 1],
        color="#3aa0ff", linewidth=1.0, alpha=0.55,
        zorder=2, label="path",
    )


def main() -> None:
    args = parse_args()
    walls_h, walls_v, holes, waypoints = get_hard_layout()
    corners = select_corner_checkpoints(waypoints, holes, walls_h, walls_v)

    rng = np.random.default_rng(args.seed)
    spawn_indices = rng.choice(
        len(waypoints), size=min(args.n_spawns, len(waypoints)), replace=False
    )
    spawns = waypoints[spawn_indices]

    fig, ax = plt.subplots(figsize=(11, 9))
    draw_maze(ax, walls_h, walls_v, holes, waypoints)

    # Spawn distribution (orange, semi-transparent — visualizes the
    # "uniform over waypoints" sampling that randomize_init_pos=True uses).
    ax.scatter(
        spawns[:, 0], spawns[:, 1],
        s=120, color="#ff8c00", alpha=0.55, edgecolor="black", linewidth=0.6,
        zorder=4, label=f"possible spawns ({len(spawns)})",
    )

    # Strict-corner safe checkpoints (green, fully opaque, on top).
    ax.scatter(
        corners[:, 0], corners[:, 1],
        s=180, color="#22c55e", marker="*", edgecolor="black", linewidth=0.8,
        zorder=5, label=f"safe corners ({len(corners)})",
    )

    # Annotate each corner with its progress order so the backward-target
    # algorithm is easy to read off the plot.
    for i, (x, y) in enumerate(corners):
        ax.annotate(
            str(i), (x, y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#15803d", fontweight="bold",
        )

    # Marble radius hint at one corner so spatial scale is obvious.
    if len(corners):
        ax.add_patch(
            Circle(corners[0], MARBLE_RADIUS, facecolor="none",
                   edgecolor="#22c55e", linewidth=0.8, linestyle="--", zorder=6)
        )

    ax.set_xlim(-0.005, BOARD_WIDTH + 0.005)
    ax.set_ylim(-0.005, BOARD_HEIGHT + 0.005)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Safe-prior setup: {len(corners)} strict-corner targets, "
        f"{len(spawns)} possible spawns (waypoints)"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
