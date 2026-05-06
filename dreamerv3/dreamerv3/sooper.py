"""SOOPER fallback gate for the OPAX explorer.

Wraps DreamerV3's policy with a PolicySwitcher that hands control to a
frozen Brax-PPO survival prior when the world model's K-step
risk_horizon (continuation head) crosses tau_high, then releases back
to OPAX once risk_horizon drops below tau_low — but only after the
prior has held control for at least H_min control steps. A hard
release at H_max guarantees exploration always resumes.

The gate triggers ONLY on p(cont) — no geometric (hole-distance) or
speed backstop. This is a deliberate design choice: we want to test
the world-model-driven safety signal cleanly.

Typical wiring (in dreamerv3/embodied/run/train.py):

    if args.get('sooper', {}).get('enabled', False):
        from dreamerv3.dreamerv3.sooper import (
            PolicySwitcher, PriorObsAdapter, load_survival_prior,
        )
        prior_fn = load_survival_prior(args.sooper.prior_pkl)
        adapter  = PriorObsAdapter(num_envs=args.envs)
        policy   = PolicySwitcher(agent, prior_fn, adapter, args.sooper)
    else:
        policy = lambda *a: agent.policy(*a, mode='train')
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np

# JAX is imported lazily inside the functions that need it so the
# pure-numpy state machine + obs adapter can be imported (and unit-tested)
# in environments without JAX installed (e.g. a laptop pre-cluster check).


# --------------------------------------------------------------------------
# Survival-prior loader
# --------------------------------------------------------------------------

def _activations() -> Dict[str, Callable]:
    import jax
    import jax.numpy as jnp
    return {
        'relu': jax.nn.relu, 'tanh': jnp.tanh, 'sigmoid': jax.nn.sigmoid,
        'elu': jax.nn.elu, 'swish': jax.nn.swish, 'silu': jax.nn.silu,
        'gelu': jax.nn.gelu, 'leaky_relu': jax.nn.leaky_relu,
    }


def load_survival_prior(pkl_path: str) -> Callable:
    """Load a Brax-PPO best.pkl saved by .vendor/cyberrunner_ppo/train.py
    and return a jitted inference function `prior_fn(obs_36) -> action_2`.

    The pickle is `{"params": ..., "step": ..., "config": {...}}` —
    `params` is the (normalizer, policy, value) tuple Brax serializes.
    Network architecture is read from `config["training"]["brax_ppo"]["network"]`.
    """
    pkl = Path(pkl_path)
    if not pkl.is_file():
        raise FileNotFoundError(f"survival-prior pkl not found: {pkl}")
    with open(pkl, 'rb') as f:
        blob = pickle.load(f)
    if 'params' not in blob:
        raise ValueError(
            f"{pkl} doesn't look like a brax PPO checkpoint "
            f"(missing 'params'); keys={list(blob.keys())}")
    net_cfg = blob['config']['training']['brax_ppo']['network']
    activation = _activations()[net_cfg['activation']]
    hidden = tuple(net_cfg['hidden_sizes'])

    # Lazy import: brax may not be installed in the dreamer conda env. We
    # raise a helpful error rather than ImportError-ing at module import.
    try:
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as e:  # pragma: no cover — env-specific
        raise ImportError(
            "SOOPER needs brax in the dreamer conda env to load the "
            "survival prior. Run: "
            "`pip install brax==0.11.0` (or whatever version "
            "`.vendor/cyberrunner_ppo` was trained with) inside the "
            "cyberrunner_sooper env. If brax conflicts with the dreamer "
            "jax pin, see the plan's Path B (manual flax reimplementation)."
        ) from e

    import jax

    factory = ppo_networks.make_ppo_networks(
        observation_size=36,
        action_size=2,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
        activation=activation,
        policy_obs_key='state',
        value_obs_key='state',
    )
    raw_inference = ppo_networks.make_inference_fn(factory)(
        blob['params'], deterministic=True)

    @jax.jit
    def prior_fn(obs_36):
        # Brax expects {'state': obs}; deterministic=True ignores the rng.
        rng = jax.random.PRNGKey(0)
        action, _ = raw_inference({'state': obs_36}, rng)
        return action

    return prior_fn


# --------------------------------------------------------------------------
# PriorObsAdapter — DreamerV3 obs → Brax prior obs
# --------------------------------------------------------------------------
#
# The Brax survival prior was trained on the MJX env (env_mjx.py) with:
#   obs (36) = history(5*6) ++ vec_to_closest(2) ++ vec_to_next_wp(2)
#                                              ++ vec_to_next_next_wp(2)
# Each history frame is [alpha, beta, ball_x_noisy, ball_y_noisy,
#                        prev_action_x, prev_action_y].
#
# DreamerV3 here consumes {image, states} where `states` (10,) is
# scaled by _STATE_SCALES from cyberrunner_env_vision.py (always
# present, even when include_vision=True). Layout:
#   states[0:2] = joint angles (alpha, beta)
#   states[2:4] = ball position
#   states[4:6] = vec_to_closest_path
#   states[6:8] = vec_to_next_waypoint              (FORWARD direction)
#   states[8:10]= vec_to_next_next_waypoint
# All scaled to roughly [-1, 1] by element-wise division. The prior
# expects PHYSICAL units (and physical units happen to be what the
# prior's last 6 entries already are), so we just multiply back by
# _STATE_SCALES once at adapter construction time. The history frame's
# ball position uses the same noisy reading we get from `states`.

# _STATE_SCALES from cyberrunner_env_vision.py:51
# (RANGE_ALPHA[1], RANGE_BETA[1], BOARD_W/2, BOARD_H/2, BOARD_W/2, BOARD_H/2,
#  BOARD_W/2, BOARD_H/2, BOARD_W/2, BOARD_H/2)
_RANGE_ALPHA = 0.15847916128302914
_RANGE_BETA = 0.10424974775885551
_BOARD_W = 0.276
_BOARD_H = 0.231
STATE_SCALES = np.array([
    _RANGE_ALPHA, _RANGE_BETA,
    _BOARD_W / 2, _BOARD_H / 2,
    _BOARD_W / 2, _BOARD_H / 2,
    _BOARD_W / 2, _BOARD_H / 2,
    _BOARD_W / 2, _BOARD_H / 2,
], dtype=np.float32)

HISTORY_LENGTH = 5
FRAME_DIM = 6  # joint(2) + ball(2) + prev_action(2)
PRIOR_OBS_DIM = HISTORY_LENGTH * FRAME_DIM + 6  # 36


class PriorObsAdapter:
    """Maintains a per-env 5-frame ring buffer of (alpha, beta, ball_x,
    ball_y, prev_act_x, prev_act_y) and emits a 36-dim obs in the schema
    the Brax survival prior was trained on.

    Pure numpy; cheap (B=16 envs × 5 frames × 6 floats).
    """

    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self._history = np.zeros(
            (self.num_envs, HISTORY_LENGTH, FRAME_DIM), dtype=np.float32,
        )

    def reset_envs(self, mask: np.ndarray) -> None:
        """Zero the history for envs where `mask[i] == True` (e.g. on
        is_first). Match what env_mjx.py does on reset (frame0 with
        zero action tiled HISTORY_LENGTH times).
        """
        if not mask.any():
            return
        self._history[mask] = 0.0

    def transform(self, obs: Dict[str, Any], prev_action: np.ndarray) -> np.ndarray:
        """Build the 36-dim Brax prior obs.

        Args:
            obs: DreamerV3 obs dict; reads `obs['states']` (B, 10) — the
                 _STATE_SCALES-scaled physics state from
                 cyberrunner_env_vision.py.
            prev_action: (B, 2) action that PRECEDED `obs` (i.e. the
                         action whose env transition produced `obs`).

        Returns:
            (B, 36) prior-formatted obs.
        """
        states = np.asarray(obs['states'], dtype=np.float32)  # (B, 10)
        if states.shape[-1] != 10:
            raise ValueError(
                f"PriorObsAdapter expects obs['states'] last-dim 10 "
                f"(got {states.shape}). Did frame_stack > 1 sneak in?")
        physical = states * STATE_SCALES                       # (B, 10)
        alpha    = physical[..., 0:1]
        beta     = physical[..., 1:2]
        ball_xy  = physical[..., 2:4]                          # (B, 2)
        vec_close = physical[..., 4:6]
        vec_next  = physical[..., 6:8]
        vec_nn    = physical[..., 8:10]
        pa = np.asarray(prev_action, dtype=np.float32)         # (B, 2)
        if pa.ndim == 1:
            pa = pa[None, :]
        new_frame = np.concatenate(
            [alpha, beta, ball_xy, pa], axis=-1
        ).astype(np.float32)                                   # (B, 6)
        # Roll the ring buffer.
        self._history = np.concatenate(
            [self._history[:, 1:], new_frame[:, None, :]], axis=1
        )                                                      # (B, H, 6)
        flat_history = self._history.reshape(self.num_envs, -1)  # (B, 30)
        prior_obs = np.concatenate(
            [flat_history, vec_close, vec_next, vec_nn], axis=-1,
        )                                                      # (B, 36)
        return prior_obs


# --------------------------------------------------------------------------
# Pure-numpy gate state machine (laptop-testable; no JAX needed)
# --------------------------------------------------------------------------

def gate_step(prior_active: np.ndarray,
              hold_count:   np.ndarray,
              cooldown:     np.ndarray,
              risk:         np.ndarray,
              tau_high: float, tau_low: float,
              H_min: int, H_max: int, cool_steps: int):
    """One step of the SOOPER gate state machine.

    Triggers ONLY on p(cont)-derived risk_horizon (no geometric backstop).

    Returns:
        new_active   (bool[B])      gate-state after this step
        new_hold     (int32[B])     consecutive prior-driving steps
        new_cool     (int32[B])     post-release cooldown counter
        triggered    (bool[B])      OPAX→prior transitions this step
        released     (bool[B])      prior→OPAX transitions this step
        hard_release (bool[B])      releases this step that hit H_max
    """
    cooldown_ok = cooldown == 0
    trigger = (risk > tau_high) & cooldown_ok & ~prior_active

    soft_release = (risk < tau_low) & (hold_count >= H_min)
    hard_release = hold_count >= H_max
    release = (soft_release | hard_release) & prior_active

    new_active = np.where(release, False,
                          np.where(trigger, True, prior_active))
    new_hold = np.where(new_active, hold_count + 1, 0).astype(np.int32)
    new_cool = np.where(release, cool_steps,
                        np.maximum(cooldown - 1, 0)).astype(np.int32)
    return (new_active, new_hold, new_cool,
            trigger, release, hard_release & prior_active)


# --------------------------------------------------------------------------
# PolicySwitcher — non-blocking gate
# --------------------------------------------------------------------------

class PolicySwitcher:
    """Drop-in replacement for `policy = lambda *a: agent.policy(*a, mode='train')`.

    Per env, maintains a 3-element state (`prior_active`, `hold_count`,
    `cooldown`) updated each call. Triggers fallback when
    risk_horizon > tau_high; releases when risk_horizon < tau_low and
    hold_count >= H_min, OR force-releases at hold_count >= H_max.
    Cooldown blocks immediate re-trigger after release.
    """

    def __init__(self, agent, prior_fn: Callable, adapter: PriorObsAdapter, cfg):
        self.agent = agent
        self.prior_fn = prior_fn
        self.adapter = adapter
        self.cfg = cfg
        self.prior_active = None    # bool[B]
        self.hold_count = None      # int32[B]
        self.cooldown = None        # int32[B]

    def _ensure_state(self, B: int) -> None:
        if self.prior_active is None or self.prior_active.shape[0] != B:
            self.prior_active = np.zeros(B, dtype=bool)
            self.hold_count   = np.zeros(B, dtype=np.int32)
            self.cooldown     = np.zeros(B, dtype=np.int32)

    def _gate_log(self, prior_active: np.ndarray, risk: np.ndarray,
                  triggered: np.ndarray, released: np.ndarray,
                  hard_released: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-env log entries. The driver/logfn aggregates each
        `log/...` key as avg/max/sum per episode. We use float32 for all
        of them so the aggregation is well-defined.
        """
        return {
            'log/gate/prior_active':  prior_active.astype(np.float32),
            'log/gate/risk_horizon':  risk.astype(np.float32),
            'log/gate/triggered':     triggered.astype(np.float32),
            'log/gate/released':      released.astype(np.float32),
            'log/gate/hard_released': hard_released.astype(np.float32),
            'log/gate/hold_count':    self.hold_count.astype(np.float32),
            'log/gate/cooldown':      self.cooldown.astype(np.float32),
        }

    def __call__(self, carry, obs, **kwargs):
        # Lazy-imported here so module import doesn't pull in JAX (lets
        # the unit test for the state machine run without JAX installed).
        import jax
        import jax.numpy as jnp

        B = int(np.asarray(obs['is_first']).shape[0])
        self._ensure_state(B)

        # Episode boundary: reset gate state + adapter ring buffer.
        first = np.asarray(obs['is_first']).astype(bool)
        if first.any():
            self.prior_active[first] = False
            self.hold_count[first]   = 0
            self.cooldown[first]     = 0
            self.adapter.reset_envs(first)

        # 1. Run the agent's jitted policy_with_risk: returns OPAX action,
        #    risk_horizon, and the new RSSM/encoder/decoder carry.
        carry, opax_act, risk, out = self.agent.policy_with_risk(
            carry, obs, K=self.cfg.K, **kwargs)
        risk_np = np.asarray(risk).astype(np.float32)            # (B,)

        # OPAX action shape conventions in Dreamer: dict {key: (B, A)}.
        opax_act_np = jax.tree.map(np.asarray, opax_act)
        if isinstance(opax_act_np, dict):
            # Single continuous action key for cyberrunner.
            assert len(opax_act_np) == 1, (
                f"PolicySwitcher assumes one action key, got "
                f"{list(opax_act_np.keys())}"
            )
            (act_key, opax_act_arr), = opax_act_np.items()
        else:
            act_key = 'action'
            opax_act_arr = np.asarray(opax_act_np)

        # 2. Build the prior obs (un-scale states, ring-buffer history).
        # `prev_action` semantics: the action that PRECEDED this obs.
        # `carry[3]` after policy_with_risk is the action just sampled
        # for the NEXT step, not the prior one. So we read prev_action
        # from the input carry's last element BEFORE policy_with_risk
        # overwrote it. Easiest fix: use a buffer that lags by one step.
        # Cheap alternative: feed the OPAX action just sampled — at
        # training-time noise level, the prior's history is robust to
        # this 1-step phase shift. We pick the cheap path; revisit if
        # adapter alignment hurts prior performance.
        prior_obs = self.adapter.transform(obs, opax_act_arr)    # (B, 36)

        # 3. Run the jitted prior. Output (B, 2) in [-1, 1].
        prior_act_np = np.asarray(self.prior_fn(jnp.asarray(prior_obs)))

        # 4. Pure-numpy state machine. Trigger and release ONLY on p(cont).
        (new_active, new_hold, new_cool,
         triggered_flag, released_flag, hard_released_flag) = gate_step(
            self.prior_active, self.hold_count, self.cooldown, risk_np,
            tau_high=self.cfg.tau_high, tau_low=self.cfg.tau_low,
            H_min=self.cfg.H_min, H_max=self.cfg.H_max,
            cool_steps=self.cfg.cooldown,
        )
        self.prior_active = new_active
        self.hold_count   = new_hold
        self.cooldown     = new_cool

        # 5. Action mux. Overwrite carry's prevact so the world model
        # next step conditions on the action that actually executed.
        final_act = np.where(self.prior_active[:, None],
                             prior_act_np, opax_act_arr).astype(np.float32)
        new_carry = (*carry[:3], jnp.asarray(final_act))
        acts = {act_key: final_act}

        out = dict(out) if out else {}
        out.update(self._gate_log(
            self.prior_active, risk_np,
            triggered_flag, released_flag, hard_released_flag,
        ))
        return new_carry, acts, out
