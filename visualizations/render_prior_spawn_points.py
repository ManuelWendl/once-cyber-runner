"""Render the Cyberrunner maze with prior reset spawn points.

Usage:
    python visualizations/render_prior_spawn_points.py
    python visualizations/render_prior_spawn_points.py --checkpoint-mode corners
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def _load_cyberrunner():
    """Load envs/cyberrunner.py directly, bypassing envs/__init__.py."""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "envs" / "cyberrunner.py"
    spec = importlib.util.spec_from_file_location("_cyberrunner_spawn_render", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cyberrunner_spawn_render"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_cr = _load_cyberrunner()
BOARD_HEIGHT = _cr.BOARD_HEIGHT
BOARD_WIDTH = _cr.BOARD_WIDTH
HOLE_RADIUS = _cr.HOLE_RADIUS
get_hard_layout = _cr.get_hard_layout
select_safe_checkpoints = _cr.select_safe_checkpoints
project_points_to_path = _cr._project_points_to_path


def build_prior_start_points(
    waypoints: np.ndarray,
    holes: np.ndarray,
    checkpoint_points: np.ndarray,
    spawn_source: str,
    spacing: float,
    min_checkpoint_dist: float,
    max_checkpoint_dist: float,
    min_hole_margin: float,
    merge_radius: float,
    prior_task: str = "checkpoint",
) -> np.ndarray:
    """Mirror the prior reset filtering logic used in CyberRunnerEnv."""
    if spawn_source == "waypoints":
        candidates = np.asarray(waypoints, dtype=np.float32)
    else:
        samples = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            seg = end - start
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-8:
                continue
            num = max(1, int(np.ceil(seg_len / spacing)))
            ts = np.linspace(0.0, 1.0, num + 1, endpoint=False)[1:]
            for t in ts:
                samples.append((1.0 - t) * start + t * end)
        samples.append(waypoints[-1])
        candidates = np.asarray(samples, dtype=np.float32)
    if len(checkpoint_points) == 0:
        return candidates

    min_hole_dist = np.linalg.norm(candidates[:, None, :] - holes[None, :, :], axis=2).min(axis=1)
    valid = min_hole_dist > (HOLE_RADIUS + min_hole_margin)
    if prior_task == "checkpoint":
        nearest_checkpoint_dist = np.linalg.norm(
            candidates[:, None, :] - checkpoint_points[None, :, :], axis=2
        ).min(axis=1)
        valid &= (
            (nearest_checkpoint_dist >= min_checkpoint_dist)
            & (nearest_checkpoint_dist <= max_checkpoint_dist)
        )
    filtered = candidates[valid]
    if not len(filtered):
        return candidates
    if merge_radius <= 0.0 or len(filtered) <= 1:
        return filtered

    _, progresses = project_points_to_path(filtered, waypoints)
    order = np.argsort(progresses)
    kept = []
    merge_radius_sq = merge_radius * merge_radius
    for idx in order:
        point = filtered[idx]
        if any(np.sum((point - prev) ** 2) < merge_radius_sq for prev in kept):
            continue
        kept.append(point)
    return np.asarray(kept, dtype=np.float32)


def plot_prior_spawn_points(
    reward_every_n_waypoints: int,
    include_corridors: bool,
    prior_task: str,
    prior_spawn_source: str,
    prior_start_point_spacing: float,
    prior_min_checkpoint_start_dist: float,
    prior_max_checkpoint_start_dist: float,
    prior_spawn_min_hole_margin: float,
    prior_spawn_merge_radius: float,
    out_path: str | None = None,
    show: bool = False,
) -> str:
    if reward_every_n_waypoints <= 0:
        raise ValueError(f"reward_every_n_waypoints must be positive, got {reward_every_n_waypoints}")

    mode = "corners_and_corridors" if include_corridors else "corners"
    if out_path is None:
        out_path = f"visualizations/prior_spawn_points_{mode}_n{reward_every_n_waypoints}.png"

    walls_h, walls_v, holes, waypoints = get_hard_layout()
    checkpoints = select_safe_checkpoints(
        waypoints=waypoints,
        holes=holes,
        walls_h=walls_h,
        walls_v=walls_v,
        reward_every_n_waypoints=reward_every_n_waypoints,
        include_corridors=include_corridors,
    )
    spawn_points = build_prior_start_points(
        waypoints=waypoints,
        holes=holes,
        checkpoint_points=checkpoints,
        spawn_source=prior_spawn_source,
        spacing=prior_start_point_spacing,
        min_checkpoint_dist=prior_min_checkpoint_start_dist,
        max_checkpoint_dist=prior_max_checkpoint_start_dist,
        min_hole_margin=prior_spawn_min_hole_margin,
        merge_radius=prior_spawn_merge_radius,
        prior_task=prior_task,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_patch(Rectangle((0, 0), BOARD_WIDTH, BOARD_HEIGHT, facecolor="#cccccc", zorder=0))

    for x_start, x_end, y in walls_h:
        ax.plot([x_start, x_end], [y, y], color="black", lw=2.0, zorder=1)
    for y_start, y_end, x in walls_v:
        ax.plot([x, x], [y_start, y_end], color="black", lw=2.0, zorder=1)

    for hx, hy in holes:
        ax.add_patch(Circle((hx, hy), HOLE_RADIUS, color="black", zorder=2))

    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        color="dimgray",
        lw=1.2,
        alpha=0.7,
        zorder=3,
        label="Reference path",
    )
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=8,
        color="gray",
        alpha=0.6,
        zorder=4,
        label="Waypoints",
    )

    if len(spawn_points):
        ax.scatter(
            spawn_points[:, 0],
            spawn_points[:, 1],
            s=28,
            facecolor="#1e88e5",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.95,
            zorder=5,
            label=f"Prior spawn points ({len(spawn_points)})",
        )

    if len(checkpoints):
        ax.scatter(
            checkpoints[:, 0],
            checkpoints[:, 1],
            s=150,
            facecolor="#43a047",
            edgecolor="white",
            linewidth=1.5,
            zorder=6,
            label=f"Safe checkpoints ({len(checkpoints)})",
        )

    ax.scatter(
        [waypoints[0, 0]],
        [waypoints[0, 1]],
        s=140,
        marker="s",
        color="green",
        edgecolor="black",
        zorder=7,
        label="Start",
    )
    ax.scatter(
        [waypoints[-1, 0]],
        [waypoints[-1, 1]],
        s=220,
        marker="*",
        color="gold",
        edgecolor="black",
        zorder=7,
        label="Goal",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.01, BOARD_WIDTH + 0.01)
    ax.set_ylim(-0.01, BOARD_HEIGHT + 0.01)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Prior spawn points ({mode}, n={reward_every_n_waypoints})"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(out_path)
    print(f"Spawn points: {len(spawn_points)}")
    print(f"Checkpoints: {len(checkpoints)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="reward_every_n_waypoints")
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        default="corners_and_corridors",
        choices=["corners", "corners_and_corridors"],
        help="Whether checkpoints should include only corners or also corridor/single-wall basins.",
    )
    parser.add_argument(
        "--prior-task",
        type=str,
        default="stabilize",
        choices=["checkpoint", "stabilize"],
    )
    parser.add_argument(
        "--prior-spawn-source",
        type=str,
        default="waypoints",
        choices=["waypoints", "dense_path"],
    )
    parser.add_argument("--prior-start-point-spacing", type=float, default=0.01)
    parser.add_argument("--prior-min-checkpoint-start-dist", type=float, default=0.02)
    parser.add_argument("--prior-max-checkpoint-start-dist", type=float, default=0.12)
    parser.add_argument("--prior-spawn-min-hole-margin", type=float, default=0.012)
    parser.add_argument("--prior-spawn-merge-radius", type=float, default=0.02)
    parser.add_argument("--out", type=str, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Also display the figure.")
    args = parser.parse_args()

    plot_prior_spawn_points(
        reward_every_n_waypoints=args.n,
        include_corridors=args.checkpoint_mode == "corners_and_corridors",
        prior_task=args.prior_task,
        prior_spawn_source=args.prior_spawn_source,
        prior_start_point_spacing=args.prior_start_point_spacing,
        prior_min_checkpoint_start_dist=args.prior_min_checkpoint_start_dist,
        prior_max_checkpoint_start_dist=args.prior_max_checkpoint_start_dist,
        prior_spawn_min_hole_margin=args.prior_spawn_min_hole_margin,
        prior_spawn_merge_radius=args.prior_spawn_merge_radius,
        out_path=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
