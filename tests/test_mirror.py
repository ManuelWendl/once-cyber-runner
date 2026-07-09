"""Tests for the Dreamer replay-buffer mirror augmentation (arXiv 2312.09906).

Covers:
1. Mirror function properties: involution (double mirror = identity) and
   composition (flip_x then flip_y == flip both).
2. Env-symmetry: stepping a mirrored MuJoCo state with a mirrored action
   yields the mirrored next state (the physical dynamics are symmetric about
   the two vertical planes through the rotation axes, away from walls/holes).
3. Buffer integration: with ``mirror_augment=True`` every transition is
   inserted 4x with distinct episode ids and correctly mirrored contents.
"""

import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from dreamer.buffer import _MIRROR_CONFIGS, Buffer
from envs.cyberrunner import BOARD_HEIGHT, BOARD_WIDTH, CyberRunnerEnv


def _random_batch(B=4):
    return TensorDict(
        {
            "states": torch.rand(B, 10, dtype=torch.float32),
            "action": torch.rand(B, 2, dtype=torch.float32) * 2 - 1,
            "reward": torch.rand(B, 1, dtype=torch.float32),
            "is_first": torch.zeros(B, 1, dtype=torch.bool),
            "is_last": torch.zeros(B, 1, dtype=torch.bool),
            "is_terminal": torch.zeros(B, 1, dtype=torch.bool),
            "episode": torch.arange(B, dtype=torch.int32),
        },
        batch_size=(B,),
    )


# ---------------------------------------------------------------------------
# 1. Mirror function properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("flip_x,flip_y", _MIRROR_CONFIGS)
def test_mirror_is_involution(flip_x, flip_y):
    data = _random_batch()
    twice = Buffer._mirror(Buffer._mirror(data, flip_x, flip_y), flip_x, flip_y)
    torch.testing.assert_close(twice["states"], data["states"])
    torch.testing.assert_close(twice["action"], data["action"])
    # Non-mirrored keys must be untouched.
    torch.testing.assert_close(twice["reward"], data["reward"])
    assert torch.equal(twice["is_terminal"], data["is_terminal"])


@pytest.mark.unit
def test_mirror_composition():
    data = _random_batch()
    both = Buffer._mirror(data, True, True)
    chained = Buffer._mirror(Buffer._mirror(data, True, False), False, True)
    torch.testing.assert_close(both["states"], chained["states"])
    torch.testing.assert_close(both["action"], chained["action"])


@pytest.mark.unit
def test_mirror_index_map():
    """flip_x negates alpha/action_x/vec_x and reflects ball_x; flip_y likewise."""
    data = _random_batch(B=1)
    s, a = data["states"][0], data["action"][0]

    mx = Buffer._mirror(data, True, False)
    sx, ax = mx["states"][0], mx["action"][0]
    torch.testing.assert_close(ax, torch.stack([-a[0], a[1]]))
    torch.testing.assert_close(sx[0], -s[0])                     # alpha
    torch.testing.assert_close(sx[2], BOARD_WIDTH - s[2])        # ball_x
    torch.testing.assert_close(sx[[4, 6, 8]], -s[[4, 6, 8]])     # vec x-components
    torch.testing.assert_close(sx[[1, 3, 5, 7, 9]], s[[1, 3, 5, 7, 9]])  # y untouched

    my = Buffer._mirror(data, False, True)
    sy, ay = my["states"][0], my["action"][0]
    torch.testing.assert_close(ay, torch.stack([a[0], -a[1]]))
    torch.testing.assert_close(sy[1], -s[1])                     # beta
    torch.testing.assert_close(sy[3], BOARD_HEIGHT - s[3])       # ball_y
    torch.testing.assert_close(sy[[5, 7, 9]], -s[[5, 7, 9]])     # vec y-components
    torch.testing.assert_close(sy[[0, 2, 4, 6, 8]], s[[0, 2, 4, 6, 8]])  # x untouched


# ---------------------------------------------------------------------------
# 2. Env dynamics symmetry
# ---------------------------------------------------------------------------


def _clearance(p, walls_h, walls_v, holes):
    """Min distance from point p to any hole center or wall segment."""
    d_holes = np.linalg.norm(holes - p, axis=1).min()

    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        return np.linalg.norm(p - (a + t * ab))

    d_walls = np.inf
    for x0, x1, y in walls_h:  # (x_start, x_end, y)
        d_walls = min(d_walls, seg_dist(p, np.array([x0, y]), np.array([x1, y])))
    for y0, y1, x in walls_v:  # (y_start, y_end, x)
        d_walls = min(d_walls, seg_dist(p, np.array([x, y0]), np.array([x, y1])))
    return min(d_holes, d_walls)


def _find_open_point(env, mirror_fn, margin=0.018):
    """Grid-search the point clearest of walls/holes for both itself and its mirror."""
    best_c, best_p = -1.0, None
    for x in np.linspace(0.03, BOARD_WIDTH - 0.03, 60):
        for y in np.linspace(0.03, BOARD_HEIGHT - 0.03, 50):
            p = np.array([x, y], dtype=np.float64)
            q = mirror_fn(p)
            c = min(
                _clearance(p, env.walls_h, env.walls_v, env.holes),
                _clearance(q, env.walls_h, env.walls_v, env.holes),
            )
            if c > best_c:
                best_c, best_p = c, p
    if best_c < margin:
        pytest.skip(f"no open point found for symmetry test (best clearance {best_c:.4f})")
    return best_p


def _set_ball_state(env, alpha, beta, pos_xy):
    env.data.qpos[:] = 0.0
    env.data.qvel[:] = 0.0
    env.data.qpos[0] = alpha
    env.data.qpos[1] = beta
    env.data.qpos[2] = pos_xy[0]
    env.data.qpos[3] = pos_xy[1]
    env.data.qpos[4] = 0.0793  # height above board
    env.data.qpos[5:9] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.integration
@pytest.mark.parametrize("axis", ["x", "y"])
def test_env_dynamics_mirror_symmetry(axis):
    """Mirrored state + mirrored action ==> mirrored next state (no contacts).

    Uses the easy layout purely for open floor space — the ball/plate
    dynamics being tested are identical across layouts.
    """
    env_a = CyberRunnerEnv(layout="easy")
    env_b = CyberRunnerEnv(layout="easy")
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    if axis == "x":
        mirror_p = lambda p: np.array([BOARD_WIDTH - p[0], p[1]])
        mirror_a = lambda a: np.array([-a[0], a[1]], dtype=np.float32)
        mirror_angles = lambda al, be: (-al, be)
    else:
        mirror_p = lambda p: np.array([p[0], BOARD_HEIGHT - p[1]])
        mirror_a = lambda a: np.array([a[0], -a[1]], dtype=np.float32)
        mirror_angles = lambda al, be: (al, -be)

    p = _find_open_point(env_a, mirror_p)
    alpha, beta = 0.0, 0.0
    _set_ball_state(env_a, alpha, beta, p)
    _set_ball_state(env_b, *mirror_angles(alpha, beta), mirror_p(p))

    action = np.array([0.35, 0.2], dtype=np.float32)
    for _ in range(12):
        env_a.step(action)
        env_b.step(mirror_a(action))

    qa, qb = env_a.data.qpos, env_b.data.qpos
    exp_alpha, exp_beta = mirror_angles(qa[0], qa[1])
    exp_pos = mirror_p(qa[2:4])
    np.testing.assert_allclose(qb[0], exp_alpha, atol=1e-6)
    np.testing.assert_allclose(qb[1], exp_beta, atol=1e-6)
    np.testing.assert_allclose(qb[2:4], exp_pos, atol=1e-6)
    np.testing.assert_allclose(qb[4], qa[4], atol=1e-6)  # height identical

    env_a.close()
    env_b.close()


# ---------------------------------------------------------------------------
# 3. Buffer integration
# ---------------------------------------------------------------------------


def _buffer_cfg():
    return OmegaConf.create(
        {
            "batch_size": 2,
            "batch_length": 8,
            "max_size": 1000,
            "device": "cpu",
            "storage_device": "cpu",
        }
    )


@pytest.mark.unit
def test_buffer_mirror_insertion():
    buf = Buffer(_buffer_cfg(), mirror_augment=True)
    B = 2
    data = _random_batch(B)
    buf.add_transition(data)

    assert buf.count() == 4 * B
    storage = buf._buffer[:]  # (B*4, 1, ...)
    episodes = storage["episode"].flatten().tolist()
    assert sorted(episodes) == list(range(4 * B))

    # Row layout: [original(B), mirror cfg 1 (B), cfg 2 (B), cfg 3 (B)].
    orig = storage[:B, 0]
    for i, (fx, fy) in enumerate(_MIRROR_CONFIGS, start=1):
        block = storage[i * B : (i + 1) * B, 0]
        expected = Buffer._mirror(
            TensorDict({"states": orig["states"], "action": orig["action"]}, batch_size=(B,)),
            fx,
            fy,
        )
        torch.testing.assert_close(block["states"], expected["states"])
        torch.testing.assert_close(block["action"], expected["action"])
        torch.testing.assert_close(block["reward"], orig["reward"])  # invariant


@pytest.mark.unit
def test_buffer_no_mirror():
    buf = Buffer(_buffer_cfg(), mirror_augment=False)
    data = _random_batch(2)
    buf.add_transition(data)
    assert buf.count() == 2
