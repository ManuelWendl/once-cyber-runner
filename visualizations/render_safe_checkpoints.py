"""Render the Cyberrunner maze with continuous safe checkpoints.

Usage:
    python visualizations/render_safe_checkpoints.py
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
    spec = importlib.util.spec_from_file_location("_cyberrunner_render", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cyberrunner_render"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_cr = _load_cyberrunner()
BOARD_HEIGHT = _cr.BOARD_HEIGHT
BOARD_WIDTH = _cr.BOARD_WIDTH
HOLE_RADIUS = _cr.HOLE_RADIUS
get_hard_layout = _cr.get_hard_layout
select_safe_checkpoints = _cr.select_safe_checkpoints


def plot_safe_checkpoints(
    reward_every_n_waypoints: int,
    out_path: str | None = None,
    show: bool = False,
    show_labels: bool = True,
) -> str:
    if reward_every_n_waypoints <= 0:
        raise ValueError(f"reward_every_n_waypoints must be positive, got {reward_every_n_waypoints}")

    if out_path is None:
        out_path = f"visualizations/safe_checkpoints_continuous_n{reward_every_n_waypoints}.png"

    walls_h, walls_v, holes, waypoints = get_hard_layout()
    checkpoints = select_safe_checkpoints(
        waypoints=waypoints,
        holes=holes,
        walls_h=walls_h,
        walls_v=walls_v,
        reward_every_n_waypoints=reward_every_n_waypoints,
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
        lw=1.5,
        alpha=0.7,
        zorder=3,
        label="Reference path",
    )

    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=10,
        color="gray",
        alpha=0.7,
        zorder=4,
        label="Waypoints",
    )

    if len(checkpoints):
        ax.scatter(
            checkpoints[:, 0],
            checkpoints[:, 1],
            s=180,
            facecolor="#4caf50",
            edgecolor="white",
            linewidth=1.8,
            zorder=6,
            label="Safe checkpoints",
        )
        ax.scatter(
            checkpoints[:, 0],
            checkpoints[:, 1],
            s=70,
            facecolor="#cfcfcf",
            edgecolor="#1b5e20",
            linewidth=1.0,
            zorder=7,
        )
        if show_labels:
            for i, point in enumerate(checkpoints):
                ax.annotate(
                    str(i),
                    xy=(point[0], point[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color="#1b5e20",
                    zorder=8,
                )

    ax.scatter(
        [waypoints[0, 0]],
        [waypoints[0, 1]],
        s=140,
        marker="s",
        color="green",
        edgecolor="black",
        zorder=9,
        label="Start",
    )
    ax.scatter(
        [waypoints[-1, 0]],
        [waypoints[-1, 1]],
        s=220,
        marker="*",
        color="gold",
        edgecolor="black",
        zorder=9,
        label="Goal",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.01, BOARD_WIDTH + 0.01)
    ax.set_ylim(-0.01, BOARD_HEIGHT + 0.01)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Continuous safe checkpoints (n={reward_every_n_waypoints} -> {len(checkpoints)} checkpoints)"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(out_path)
    np.set_printoptions(precision=4, suppress=True)
    print(checkpoints)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Controls the number of extracted checkpoints.")
    parser.add_argument("--out", type=str, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Also display the figure.")
    parser.add_argument("--hide-labels", action="store_true", help="Disable numeric checkpoint labels.")
    args = parser.parse_args()
    plot_safe_checkpoints(
        reward_every_n_waypoints=args.n,
        out_path=args.out,
        show=args.show,
        show_labels=not args.hide_labels,
    )


if __name__ == "__main__":
    main()
