"""Laptop-runnable tests for the SOOPER gate state machine + obs adapter.

These tests deliberately avoid importing JAX/Brax/Dreamer so they run on
any machine with numpy. They exercise the pure-numpy code paths in
`dreamerv3/dreamerv3/sooper.py`.

Run from the repo root:

    pytest dreamerv3/tests/test_sooper.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make `from dreamerv3.dreamerv3.sooper import ...` resolvable when running
# pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dreamerv3.dreamerv3.sooper import (  # noqa: E402
    PRIOR_OBS_DIM,
    HISTORY_LENGTH,
    FRAME_DIM,
    STATE_SCALES,
    PriorObsAdapter,
    gate_step,
)


# ============================================================================
# Gate state machine
# ============================================================================

def _initial_state(B: int):
    return (np.zeros(B, dtype=bool),
            np.zeros(B, dtype=np.int32),
            np.zeros(B, dtype=np.int32))


GATE_DEFAULTS = dict(
    tau_high=0.3, tau_low=0.1,
    H_min=300, H_max=600, cool_steps=60,
)


def test_idle_no_risk_stays_idle():
    """Risk well below tau_low: gate never triggers."""
    active, hold, cool = _initial_state(B=1)
    risk = np.full(1, 0.05, dtype=np.float32)
    for _ in range(50):
        active, hold, cool, trig, rel, hard = gate_step(
            active, hold, cool, risk, **GATE_DEFAULTS)
    assert not active.any()
    assert (hold == 0).all()
    assert (cool == 0).all()


def test_trigger_fires_when_risk_crosses_tau_high():
    """Single env, risk jumps above tau_high → prior_active=True next step."""
    active, hold, cool = _initial_state(B=1)
    risk_low  = np.full(1, 0.05, dtype=np.float32)
    risk_high = np.full(1, 0.5,  dtype=np.float32)

    # First step at low risk: stays idle.
    active, hold, cool, trig, rel, _ = gate_step(active, hold, cool, risk_low, **GATE_DEFAULTS)
    assert not trig.any() and not active.any()

    # Step with high risk: triggers.
    active, hold, cool, trig, rel, _ = gate_step(active, hold, cool, risk_high, **GATE_DEFAULTS)
    assert trig.any() and active.all()
    assert hold[0] == 1


def test_h_min_blocks_soft_release():
    """Even if risk drops below tau_low, the prior must hold for ≥ H_min steps."""
    active, hold, cool = _initial_state(B=1)
    # Trigger first.
    high = np.full(1, 0.5, dtype=np.float32)
    low  = np.full(1, 0.05, dtype=np.float32)
    active, hold, cool, _, _, _ = gate_step(active, hold, cool, high, **GATE_DEFAULTS)
    assert active.all()

    # Hold for fewer than H_min steps with low risk: should NOT release.
    for k in range(1, GATE_DEFAULTS['H_min']):
        active, hold, cool, _, rel, _ = gate_step(active, hold, cool, low, **GATE_DEFAULTS)
        assert active.all(), f"released early at hold={k}"
        assert not rel.any()

    # On the H_min-th hold step with risk < tau_low, soft-release fires.
    active, hold, cool, _, rel, hard = gate_step(active, hold, cool, low, **GATE_DEFAULTS)
    assert rel.any()
    assert not hard.any()        # soft release, not hard
    assert not active.any()
    assert cool[0] == GATE_DEFAULTS['cool_steps']


def test_hard_release_at_h_max():
    """If risk stays above tau_low, the gate force-releases at H_max."""
    active, hold, cool = _initial_state(B=1)
    high   = np.full(1, 0.5, dtype=np.float32)
    medium = np.full(1, 0.2, dtype=np.float32)   # > tau_low, < tau_high
    active, hold, cool, _, _, _ = gate_step(active, hold, cool, high, **GATE_DEFAULTS)

    # Stay above tau_low forever: no soft-release possible.
    for k in range(1, GATE_DEFAULTS['H_max']):
        active, hold, cool, _, rel, hard = gate_step(active, hold, cool, medium, **GATE_DEFAULTS)
        assert not rel.any(), f"unexpected release at hold={k}"
        assert active.all()

    # The H_max-th step: hard release fires regardless of risk.
    active, hold, cool, _, rel, hard = gate_step(active, hold, cool, medium, **GATE_DEFAULTS)
    assert rel.any() and hard.any()
    assert not active.any()


def test_cooldown_blocks_immediate_retrigger():
    """After release, the gate cannot re-trigger for cool_steps steps."""
    active, hold, cool = _initial_state(B=1)
    high = np.full(1, 0.5, dtype=np.float32)
    low  = np.full(1, 0.05, dtype=np.float32)
    # Trigger, then satisfy H_min, then release.
    active, hold, cool, _, _, _ = gate_step(active, hold, cool, high, **GATE_DEFAULTS)
    for _ in range(GATE_DEFAULTS['H_min']):
        active, hold, cool, _, _, _ = gate_step(active, hold, cool, low, **GATE_DEFAULTS)
    assert not active.any()
    assert cool[0] == GATE_DEFAULTS['cool_steps']

    # During cooldown, even high risk doesn't re-trigger.
    for k in range(GATE_DEFAULTS['cool_steps']):
        active, hold, cool, trig, _, _ = gate_step(active, hold, cool, high, **GATE_DEFAULTS)
        assert not trig.any(), f"triggered during cooldown at k={k}"
        assert not active.any()

    # Cooldown elapsed, high risk now triggers.
    active, hold, cool, trig, _, _ = gate_step(active, hold, cool, high, **GATE_DEFAULTS)
    assert trig.any() and active.all()


def test_independent_envs():
    """B=4 envs evolve their gate state independently."""
    B = 4
    active, hold, cool = _initial_state(B)
    # Env 0 high, others low.
    risk = np.array([0.5, 0.05, 0.05, 0.05], dtype=np.float32)
    active, hold, cool, trig, _, _ = gate_step(active, hold, cool, risk, **GATE_DEFAULTS)
    assert active[0] and not active[1:].any()
    assert hold[0] == 1 and (hold[1:] == 0).all()


# ============================================================================
# PriorObsAdapter
# ============================================================================

def test_adapter_output_shape():
    B = 3
    adapter = PriorObsAdapter(num_envs=B)
    states = np.zeros((B, 10), dtype=np.float32)
    prev_action = np.zeros((B, 2), dtype=np.float32)
    out = adapter.transform({'states': states}, prev_action)
    assert out.shape == (B, PRIOR_OBS_DIM) == (B, HISTORY_LENGTH * FRAME_DIM + 6)


def test_adapter_unscales_states():
    """The adapter multiplies states by STATE_SCALES; a state of all 1s in
    the scaled domain should produce the actual STATE_SCALES values in the
    physical-unit slots of the output (history's joint+ball part)."""
    adapter = PriorObsAdapter(num_envs=1)
    scaled_ones = np.ones((1, 10), dtype=np.float32)
    prev_action = np.zeros((1, 2), dtype=np.float32)
    out = adapter.transform({'states': scaled_ones}, prev_action)
    # Latest history frame is the LAST FRAME_DIM slice of the first 30 dims.
    last_frame = out[0, (HISTORY_LENGTH - 1) * FRAME_DIM : HISTORY_LENGTH * FRAME_DIM]
    np.testing.assert_allclose(last_frame[0], STATE_SCALES[0])  # alpha
    np.testing.assert_allclose(last_frame[1], STATE_SCALES[1])  # beta
    np.testing.assert_allclose(last_frame[2], STATE_SCALES[2])  # ball_x
    np.testing.assert_allclose(last_frame[3], STATE_SCALES[3])  # ball_y
    # Path-relative tail also unscaled (last 6 dims of the obs):
    np.testing.assert_allclose(out[0, -6], STATE_SCALES[4])     # vec_to_closest_x
    np.testing.assert_allclose(out[0, -1], STATE_SCALES[9])     # vec_to_next_next_y


def test_adapter_history_rolls():
    """After 5 distinct steps, the buffer holds the 5 frames in order."""
    adapter = PriorObsAdapter(num_envs=1)
    # Reset (frame ring should already be zeros; reset explicit for clarity).
    adapter.reset_envs(np.array([True]))
    for i, val in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        states = np.full((1, 10), val, dtype=np.float32)
        prev_action = np.full((1, 2), val, dtype=np.float32)
        adapter.transform({'states': states}, prev_action)
    # The 5 history frames should be in increasing alpha order.
    out = adapter.transform({'states': np.full((1, 10), 0.6, dtype=np.float32)},
                            np.full((1, 2), 0.6, dtype=np.float32))
    history = out[0, : HISTORY_LENGTH * FRAME_DIM].reshape(HISTORY_LENGTH, FRAME_DIM)
    # After this 6th step, the buffer holds steps 2..6 (oldest dropped).
    expected_alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    np.testing.assert_allclose(history[:, 0], np.array(expected_alphas) * STATE_SCALES[0])


def test_adapter_reset_clears_history():
    adapter = PriorObsAdapter(num_envs=2)
    states = np.full((2, 10), 0.5, dtype=np.float32)
    prev_action = np.full((2, 2), 0.5, dtype=np.float32)
    for _ in range(5):
        adapter.transform({'states': states}, prev_action)
    adapter.reset_envs(np.array([True, False]))
    out = adapter.transform({'states': np.zeros((2, 10), dtype=np.float32)},
                            np.zeros((2, 2), dtype=np.float32))
    # Env 0: history is [0, 0, 0, 0, latest=0]. Env 1: history retains old data.
    history_0 = out[0, : HISTORY_LENGTH * FRAME_DIM]
    history_1 = out[1, : HISTORY_LENGTH * FRAME_DIM]
    assert (history_0 == 0).all()
    # Env 1 should have non-zero entries in its first 4 frames.
    history_1_old = history_1[: (HISTORY_LENGTH - 1) * FRAME_DIM]
    assert (history_1_old != 0).any()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
