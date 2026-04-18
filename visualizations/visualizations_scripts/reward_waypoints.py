"""Visualize which waypoints provide reward under the modulo-N sparsification.

Usage:
    python visualizations.reward_waypoints --n 5
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def _load_cyberrunner():
    """Load envs/cyberrunner.py directly, bypassing envs/__init__.py
    (which eagerly imports heavy deps like tensordict that aren't needed here)."""
    # This file lives at <repo>/visualizations/visualizations_scripts/
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "envs", "cyberrunner.py")
    spec = importlib.util.spec_from_file_location("_cyberrunner_std", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cyberrunner_std"] = module
    spec.loader.exec_module(module)
    return module


_cr = _load_cyberrunner()
BOARD_HEIGHT = _cr.BOARD_HEIGHT
BOARD_WIDTH = _cr.BOARD_WIDTH
HOLE_RADIUS = _cr.HOLE_RADIUS
get_hard_layout = _cr.get_hard_layout


def plot_reward_waypoints(
    n: int,
    out_path: str | None = None,
    show: bool = False,
) -> str:
    """Render the maze, highlighting every n-th waypoint as a reward checkpoint.

    Returns the saved file path.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if out_path is None:
        out_path = f"visualizations/reward_waypoints_n{n}.png"

    walls_h, walls_v, holes, waypoints = get_hard_layout()
    total = len(waypoints)
    reward_idxs = [i for i in range(1, total) if i % n == 0]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.add_patch(
        Rectangle((0, 0), BOARD_WIDTH, BOARD_HEIGHT, facecolor="#cccccc", zorder=0)
    )

    for x_start, x_end, y in walls_h:
        ax.plot([x_start, x_end], [y, y], color="black", lw=2.0, zorder=1)
    for y_start, y_end, x in walls_v:
        ax.plot([x, x], [y_start, y_end], color="black", lw=2.0, zorder=1)

    for hx, hy in holes:
        ax.add_patch(Circle((hx, hy), HOLE_RADIUS, color="black", zorder=2))

    ax.plot(
        waypoints[:, 0], waypoints[:, 1],
        color="dimgray", lw=1.5, zorder=3, label="Path",
    )

    ax.scatter(
        waypoints[:, 0], waypoints[:, 1],
        s=12, color="gray", zorder=4, label="Waypoint (no reward)",
    )

    if reward_idxs:
        rw = waypoints[reward_idxs]
        ax.scatter(
            rw[:, 0], rw[:, 1],
            s=90, facecolor="red", edgecolor="white", linewidth=1.2,
            zorder=5, label=f"Reward waypoint (i % {n} == 0)",
        )
        for i in reward_idxs:
            ax.annotate(
                str(i),
                xy=(waypoints[i, 0], waypoints[i, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="darkred",
                zorder=6,
            )

    ax.scatter(
        [waypoints[0, 0]], [waypoints[0, 1]],
        s=140, marker="s", color="green", edgecolor="black",
        zorder=7, label="Start",
    )
    ax.scatter(
        [waypoints[-1, 0]], [waypoints[-1, 1]],
        s=220, marker="*", color="gold", edgecolor="black",
        zorder=7, label="Goal",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.01, BOARD_WIDTH + 0.01)
    ax.set_ylim(-0.01, BOARD_HEIGHT + 0.01)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Reward checkpoints (n={n}): {len(reward_idxs)} of {total} waypoints"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="Reward every n-th waypoint.")
    parser.add_argument("--out", type=str, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Also display the figure.")
    args = parser.parse_args()
    plot_reward_waypoints(args.n, out_path=args.out, show=args.show)


if __name__ == "__main__":
    main()
