import numpy as np
from PIL import Image

from envs.cyberrunner import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    _generate_board_texture,
    get_hard_layout,
)


_BG_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _maze_background(by: int, bx: int) -> np.ndarray:
    """Low-res maze template (path + holes) at bin-grid resolution, row 0 = top."""
    key = (by, bx)
    if key not in _BG_CACHE:
        _, _, holes, waypoints = get_hard_layout()
        # _generate_board_texture ends in FLIP_TOP_BOTTOM so it matches MuJoCo
        # (origin bottom-left). Flip back so PIL row 0 = top of board (high y).
        img = _generate_board_texture(holes, waypoints).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.resize((bx, by), Image.BILINEAR)
        _BG_CACHE[key] = np.asarray(img, dtype=np.uint8)
    return _BG_CACHE[key]


def sigma_heatmap(sigma, states, bins=32, upscale=8):
    """Build a spatial sigma heatmap from ball positions and uncertainty values.

    sigma:  array (B, T-1) float — per-step prediction uncertainty
    states: array (B, T, >=4) float — indices [2:4] are ball (x, y) in board frame
    Returns (3, H, W) uint8 (channel-first) for TensorBoard add_image.
    Green = low uncertainty, Red = high uncertainty, maze template shows through unvisited cells.
    """
    sigma = np.asarray(sigma, dtype=np.float32)
    states_arr = np.asarray(states, dtype=np.float32)
    # sigma[b, t] is the prediction error for state (b, t+1), so bin it at that target state.
    xy = states_arr[:, 1 : sigma.shape[1] + 1, 2:4].reshape(-1, 2)
    xs, ys = xy[:, 0], xy[:, 1]
    sig_flat = sigma.reshape(-1)

    # Aspect-correct grid over the full board, y increases up (see envs/cyberrunner.py).
    bx = int(bins)
    by = max(1, int(round(bins * BOARD_HEIGHT / BOARD_WIDTH)))
    xi = np.clip((xs / BOARD_WIDTH * bx).astype(int), 0, bx - 1)
    yi = np.clip((ys / BOARD_HEIGHT * by).astype(int), 0, by - 1)
    in_range = (xs >= 0) & (xs <= BOARD_WIDTH) & (ys >= 0) & (ys <= BOARD_HEIGHT)
    flat = (yi * bx + xi)[in_range]
    sig_in = sig_flat[in_range]

    sum_g = np.zeros(by * bx, dtype=np.float32)
    cnt_g = np.zeros(by * bx, dtype=np.float32)
    np.add.at(sum_g, flat, sig_in)
    np.add.at(cnt_g, flat, 1.0)
    mean_g = (sum_g / np.maximum(cnt_g, 1.0)).reshape(by, bx)
    has = (cnt_g.reshape(by, bx) > 0)

    if has.any():
        vals = mean_g[has]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    nm = np.clip((mean_g - lo) / (hi - lo), 0.0, 1.0)

    sigma_rgb = np.stack([
        (255 * nm).astype(np.uint8),
        (255 * (1 - nm)).astype(np.uint8),
        np.zeros_like(nm, dtype=np.uint8),
    ], axis=-1)

    bg = _maze_background(by, bx)
    mp = np.where(has[..., None], sigma_rgb, bg)                # faint maze shows in unvisited cells
    mp = np.flipud(mp)                                          # board y=0 → bottom row
    mp = np.repeat(np.repeat(mp, upscale, axis=0), upscale, axis=1)
    return mp.transpose(2, 0, 1)


def sigma_bar_frames(frames, sigma):
    """Append a 6-pixel green→red uncertainty bar below each video frame.

    frames: (T, H, W, C) uint8
    sigma:  (T,) or (T-1,) float — T-1 case pads first step with 0
    Returns (T, H+6, W, C) uint8.
    """
    T, H, W, C = frames.shape
    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.shape[0] == T - 1:
        sigma = np.concatenate([[0.0], sigma])
    smax = sigma.max() + 1e-8
    sn = np.clip(sigma / smax, 0.0, 1.0)        # (T,)
    r = (255 * sn).astype(np.uint8)
    g = (255 * (1 - sn)).astype(np.uint8)
    z = np.zeros_like(r)
    colors = np.stack([r, g, z], axis=-1)        # (T, 3)
    # broadcast to (T, 6, W, 3) then adapt channels
    bar3 = np.broadcast_to(colors[:, None, None, :], (T, 6, W, 3)).copy()
    if C == 3:
        bar = bar3
    else:
        bar = np.zeros((T, 6, W, C), dtype=np.uint8)
        bar[..., :3] = bar3
    return np.concatenate([frames, bar], axis=1) # (T, H+6, W, C)
