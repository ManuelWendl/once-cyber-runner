import numpy as np
from PIL import Image, ImageDraw

BOARD_WIDTH = 0.276
BOARD_HEIGHT = 0.231

_MAZE_OVERLAY = None
_MAZE_OVERLAY_LAYOUT = None


def _build_maze_overlay(bins, upscale, layout='hard'):
    global _MAZE_OVERLAY, _MAZE_OVERLAY_LAYOUT
    key = (bins, upscale, layout)
    if _MAZE_OVERLAY is not None and _MAZE_OVERLAY_LAYOUT == key:
        return _MAZE_OVERLAY

    import sys, pathlib
    repo_root = str(pathlib.Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from cyberrunner_env_vision import _LAYOUT_LOADERS
    walls_h, walls_v, holes, _ = _LAYOUT_LOADERS[layout]()

    H_px = bins * upscale
    W_px = bins * upscale
    img = Image.new("RGBA", (W_px, H_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def to_px(x, y):
        px = x / BOARD_WIDTH * W_px
        py = (1.0 - y / BOARD_HEIGHT) * H_px
        return px, py

    wall_color = (255, 255, 255, 180)
    hole_color = (255, 255, 255, 120)
    line_width = max(1, upscale // 4)

    for wall in walls_h:
        x_start, x_end, y = wall
        draw.line([to_px(x_start, y), to_px(x_end, y)], fill=wall_color, width=line_width)
    for wall in walls_v:
        y_start, y_end, x = wall
        draw.line([to_px(x, y_start), to_px(x, y_end)], fill=wall_color, width=line_width)

    hole_r = max(2, upscale // 3)
    for hole in holes:
        cx, cy = to_px(hole[0], hole[1])
        draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r], fill=hole_color)

    _MAZE_OVERLAY = np.array(img)
    _MAZE_OVERLAY_LAYOUT = key
    return _MAZE_OVERLAY


def sigma_heatmap(sigma, ball_xy, bins=32, upscale=8, layout='hard'):
    """Spatial sigma heatmap from ball positions and uncertainty values.

    Args:
        sigma: (N,) float — disagreement values
        ball_xy: (N, 2) float — ball (x, y) in board frame
        bins: grid resolution
        upscale: pixel multiplier per bin
        layout: maze layout name

    Returns:
        (H, W, 3) uint8 image. Green=low, Red=high, Grey=unvisited.
    """
    sigma = np.asarray(sigma, dtype=np.float32).ravel()
    ball_xy = np.asarray(ball_xy, dtype=np.float32).reshape(-1, 2)
    n = min(len(sigma), len(ball_xy))
    sigma, ball_xy = sigma[:n], ball_xy[:n]

    xi = np.clip((ball_xy[:, 0] / BOARD_WIDTH * bins).astype(int), 0, bins - 1)
    yi = np.clip((ball_xy[:, 1] / BOARD_HEIGHT * bins).astype(int), 0, bins - 1)
    flat = yi * bins + xi

    sum_g = np.zeros(bins * bins, dtype=np.float32)
    cnt_g = np.zeros(bins * bins, dtype=np.float32)
    np.add.at(sum_g, flat, sigma)
    np.add.at(cnt_g, flat, 1.0)
    mean_g = (sum_g / np.maximum(cnt_g, 1.0)).reshape(bins, bins)

    visited = cnt_g.reshape(bins, bins) > 0
    visited_vals = mean_g[visited]
    if len(visited_vals) > 0:
        vmax = np.percentile(visited_vals, 95) if len(visited_vals) > 1 else visited_vals[0]
        vmax = max(vmax, 1e-8)
    else:
        vmax = 1.0
    nm = np.clip(mean_g / vmax, 0.0, 1.0)

    rr = (255 * nm).astype(np.uint8)
    gg = (255 * (1 - nm)).astype(np.uint8)
    bb = np.zeros_like(rr)
    grey = np.full_like(rr, 60)
    rr = np.where(visited, rr, grey)
    gg = np.where(visited, gg, grey)
    bb = np.where(visited, bb, grey)

    rr, gg, bb = np.flipud(rr), np.flipud(gg), np.flipud(bb)
    rr = np.repeat(np.repeat(rr, upscale, 0), upscale, 1)
    gg = np.repeat(np.repeat(gg, upscale, 0), upscale, 1)
    bb = np.repeat(np.repeat(bb, upscale, 0), upscale, 1)

    overlay = _build_maze_overlay(bins, upscale, layout)
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    inv = 1.0 - alpha
    rr = (rr * inv + overlay[:, :, 0] * alpha).astype(np.uint8)
    gg = (gg * inv + overlay[:, :, 1] * alpha).astype(np.uint8)
    bb = (bb * inv + overlay[:, :, 2] * alpha).astype(np.uint8)

    return np.stack([rr, gg, bb], axis=-1)


def coverage_heatmap(ball_xy, bins=32, upscale=8, layout='hard'):
    """Spatial coverage heatmap from ball positions.

    Returns:
        (H, W, 3) uint8 image. Brighter=more visits, dark=unvisited.
    """
    ball_xy = np.asarray(ball_xy, dtype=np.float32).reshape(-1, 2)
    xi = np.clip((ball_xy[:, 0] / BOARD_WIDTH * bins).astype(int), 0, bins - 1)
    yi = np.clip((ball_xy[:, 1] / BOARD_HEIGHT * bins).astype(int), 0, bins - 1)
    flat = yi * bins + xi

    cnt = np.zeros(bins * bins, dtype=np.float32)
    np.add.at(cnt, flat, 1.0)
    cnt = cnt.reshape(bins, bins)

    visited = cnt > 0
    vmax = max(np.percentile(cnt[visited], 95), 1.0) if visited.any() else 1.0
    nm = np.clip(cnt / vmax, 0.0, 1.0)

    rr = np.where(visited, (50 + 205 * nm).astype(np.uint8), np.uint8(20))
    gg = np.where(visited, (100 + 155 * nm).astype(np.uint8), np.uint8(20))
    bb = np.where(visited, (50 + 205 * nm).astype(np.uint8), np.uint8(20))

    rr, gg, bb = np.flipud(rr), np.flipud(gg), np.flipud(bb)
    rr = np.repeat(np.repeat(rr, upscale, 0), upscale, 1)
    gg = np.repeat(np.repeat(gg, upscale, 0), upscale, 1)
    bb = np.repeat(np.repeat(bb, upscale, 0), upscale, 1)

    overlay = _build_maze_overlay(bins, upscale, layout)
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    inv = 1.0 - alpha
    rr = (rr * inv + overlay[:, :, 0] * alpha).astype(np.uint8)
    gg = (gg * inv + overlay[:, :, 1] * alpha).astype(np.uint8)
    bb = (bb * inv + overlay[:, :, 2] * alpha).astype(np.uint8)

    return np.stack([rr, gg, bb], axis=-1)
