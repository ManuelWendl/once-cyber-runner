import numpy as np


def sigma_heatmap(sigma, states, bins=32, upscale=8):
    """Build a spatial sigma heatmap from ball positions and uncertainty values.

    sigma:  array (N,) or (B, T) float — flattened internally
    states: array (B, T, >=4) float — indices [2:4] are ball (x, y) in board frame
    Returns (3, H, W) uint8 (channel-first) for TensorBoard add_image.
    Green = low uncertainty, Red = high uncertainty, Grey = unvisited.
    """
    sigma = np.asarray(sigma, dtype=np.float32).reshape(-1)
    states_arr = np.asarray(states, dtype=np.float32)
    xy = states_arr.reshape(-1, states_arr.shape[-1])[:, 2:4]
    xs, ys = xy[:, 0], xy[:, 1]

    # Align sigma length with positions (sigma is T-1 per sequence, states is T)
    # Trim states to match sigma length if needed
    if len(sigma) < len(xs):
        xs = xs[: len(sigma)]
        ys = ys[: len(sigma)]
    elif len(sigma) > len(xs):
        sigma = sigma[: len(xs)]

    x_min, x_max = xs.min() - 1e-6, xs.max() + 1e-6
    y_min, y_max = ys.min() - 1e-6, ys.max() + 1e-6

    xi = np.clip(((xs - x_min) / (x_max - x_min) * bins).astype(int), 0, bins - 1)
    yi = np.clip(((ys - y_min) / (y_max - y_min) * bins).astype(int), 0, bins - 1)
    flat = yi * bins + xi

    sum_g = np.zeros(bins * bins, dtype=np.float32)
    cnt_g = np.zeros(bins * bins, dtype=np.float32)
    np.add.at(sum_g, flat, sigma)
    np.add.at(cnt_g, flat, 1.0)
    mean_g = (sum_g / np.maximum(cnt_g, 1.0)).reshape(bins, bins)

    gmax = mean_g.max() + 1e-8
    nm = np.clip(mean_g / gmax, 0.0, 1.0)

    rr = (255 * nm).astype(np.uint8)
    gg = (255 * (1 - nm)).astype(np.uint8)
    bb = np.zeros_like(rr)
    has = cnt_g.reshape(bins, bins) > 0
    grey = np.full_like(rr, 60)
    rr = np.where(has, rr, grey)
    gg = np.where(has, gg, grey)
    bb = np.where(has, bb, grey)

    mp = np.stack([rr, gg, bb], axis=-1)                             # (bins, bins, 3)
    mp = np.flipud(mp.transpose(1, 0, 2))                            # x→right, y→up
    mp = np.repeat(np.repeat(mp, upscale, axis=0), upscale, axis=1) # (H, W, 3)
    return mp.transpose(2, 0, 1)                                     # (3, H, W)


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
