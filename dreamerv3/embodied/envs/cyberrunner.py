import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless GPU; fallback: 'osmesa'

import collections
import functools
import sys
import pathlib

import elements
import embodied
import numpy as np


class CyberRunner(embodied.Env):
    """DreamerV3 wrapper for CyberRunnerEnv (gymnasium-style)."""

    def __init__(
        self,
        task='default',
        repo_root=None,
        episode_length=2000,
        randomize_init_pos=True,
        include_vision=True,
        frame_stack=1,
        size=(64, 64),
        layout='hard',
        grayscale=True,
        chain_on_survival=False,
        chain_only_on_prior_hold=False,
        chain_only_on_timeout=False,
    ):
        # Make the top-level cyberrunner_env_vision module importable.
        if repo_root is None:
            repo_root = pathlib.Path(__file__).resolve().parents[3]
        repo_root = str(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        import jax
        devices = jax.devices()
        print(f"[DreamerV3] JAX devices: {devices}", flush=True)

        mujoco_gl = os.environ.get('MUJOCO_GL', 'not set')
        print(f"[DreamerV3] MUJOCO_GL (render backend): {mujoco_gl}", flush=True)

        from cyberrunner_env_vision import CyberRunnerEnv

        self._env = CyberRunnerEnv(
            render_mode=None,
            episode_length=episode_length,
            randomize_init_pos=randomize_init_pos,
            include_vision=include_vision,
            layout=layout,
            grayscale=grayscale,
        )
        self._include_vision = include_vision
        self._frame_stack = frame_stack
        self._state_buffer = None
        self._size = tuple(size)
        self._done = True
        self._info = None
        # Persistent-spawn chaining: when True, an episode that ends by
        # anything other than a hole-fall (timeout or goal) chains — the
        # next episode starts from that episode's final physical state
        # instead of a fresh waypoint spawn. Only a hole resets to origin.
        #
        # `chain_only_on_prior_hold` restricts the chain to ONLY fire when
        # the previous episode ended via PolicySwitcher's max_prior_hold
        # truncation (termination_reason='prior_hold').
        # `chain_only_on_timeout` restricts the chain to ONLY fire when
        # the previous episode ended via the env's episode_length cap
        # (termination_reason='timeout'). Useful when the agent reliably
        # survives long stretches and you want to re-spawn it at the
        # far-frontier state it timed out in, rather than back at origin.
        # The two "only" flags are mutually exclusive in effect:
        # prior_hold takes precedence if both are set.
        self._chain_on_survival = bool(chain_on_survival)
        self._chain_only_on_prior_hold = bool(chain_only_on_prior_hold)
        self._chain_only_on_timeout = bool(chain_only_on_timeout)

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        import gymnasium as gym
        spaces = {}
        for key, space in self._env.observation_space.spaces.items():
            sp = self._convert(space)
            if key == 'states' and self._frame_stack > 1:
                new_shape = (sp.shape[0] * self._frame_stack,)
                low = np.tile(sp.low, self._frame_stack)
                high = np.tile(sp.high, self._frame_stack)
                sp = elements.Space(sp.dtype, new_shape, low, high)
            spaces[key] = sp
        spaces.update(
            reward=elements.Space(np.float32),
            is_first=elements.Space(bool),
            is_last=elements.Space(bool),
            is_terminal=elements.Space(bool),
        )
        spaces['log/path_progress'] = elements.Space(np.float32)
        # Per-step termination-reason flags. Non-zero only on the episode's
        # terminal step (is_last=True). The train loop's logfn auto-aggregates
        # any `log/...` key as avg/max/sum per episode — `.../max` reads as
        # "did this episode end in <reason>" → averaged across episodes gives
        # the per-cause termination rate (the SOOPER baseline-vs-treatment
        # signal).
        spaces['log/hole_terminated']    = elements.Space(np.float32)
        spaces['log/goal_terminated']    = elements.Space(np.float32)
        spaces['log/timeout_terminated'] = elements.Space(np.float32)
        # SOOPER max_prior_hold: episode truncated because the PolicySwitcher
        # held the prior active for too long. Treated as a non-hole
        # termination (chains via chain_on_survival).
        spaces['log/prior_hold_terminated'] = elements.Space(np.float32)
        # SOOPER Mode 2 plumbing: placeholder slots the env always emits as
        # 0.0. The PolicySwitcher overrides them via the policy `outs` dict
        # (the driver merges `outs` after `obs`), so when the gate is active
        # these carry the per-step risk_critic and prior_active flag into
        # replay. Excluded from the encoder/decoder in agent.py (like
        # `reward`); consumed only as loss targets for the distilled risk
        # head and the V^pi~_r intrinsic-value head.
        spaces['prior_risk']   = elements.Space(np.float32)
        spaces['prior_active'] = elements.Space(np.float32)
        # Raw V_prior (the survival prior's value-head output) — the
        # distilled risk head regresses this instead of risk_critic because
        # ~97% of risk_critic targets are 0.0 with v10-calibrated V_norm,
        # which collapsed the head to predicting ~0 everywhere. V_prior has
        # a natural continuous distribution (~40-135).
        spaces['prior_v']      = elements.Space(np.float32)
        return spaces

    @functools.cached_property
    def act_space(self):
        space = self._env.action_space
        return {
            'action': self._convert(space),
            'reset': elements.Space(bool),
            # SOOPER max_prior_hold signal from PolicySwitcher. Declared as
            # bool (discrete) so wrap_env's NormalizeAction/ClipAction skip
            # it — those wrappers promote float32 → float64 via the (x+1)/2
            # arithmetic against Python-float space bounds and trip
            # CheckSpaces. Filtered out of the agent's act_space in main.py
            # (mirrors `reset`). False by default (e.g. plain OPAX) →
            # no truncation.
            '_force_terminate': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            # Persistent-spawn chaining: if the previous episode ended by
            # anything other than a hole-fall, continue from its final
            # physical state. self._env.data still holds the terminal state
            # here (reset() hasn't run yet); self._info holds the terminal
            # step's termination_reason. First reset / hole-fall → None →
            # the underlying env does its normal waypoint spawn.
            options = None
            if self._chain_on_survival and self._info is not None:
                reason = self._info.get('termination_reason', '')
                if self._chain_only_on_prior_hold:
                    should_chain = (reason == 'prior_hold')
                elif self._chain_only_on_timeout:
                    should_chain = (reason == 'timeout')
                else:
                    should_chain = bool(reason) and reason != 'hole'
                if should_chain:
                    options = {'restore_state': {
                        'qpos': self._env.data.qpos.copy(),
                        'qvel': self._env.data.qvel.copy(),
                    }}
            obs, self._info = self._env.reset(options=options)
            obs = self._stack_states(obs, reset=True)
            return self._obs(obs, 0.0, is_first=True)
        act = np.asarray(action['action'], dtype=np.float32)
        obs, reward, terminated, truncated, self._info = self._env.step(act)
        natural_done = bool(terminated or truncated)
        # SOOPER max_prior_hold: PolicySwitcher requests truncation when the
        # prior has been driving for too many consecutive steps. Only override
        # when the env didn't naturally terminate this step — a real hole-fall
        # wins (so chain_on_survival still routes the hole to origin spawn).
        force_term = bool(action.get('_force_terminate', False))
        if force_term and not natural_done:
            self._info = dict(self._info) if self._info is not None else {}
            self._info['termination_reason'] = 'prior_hold'
            self._done = True
        else:
            self._done = natural_done
        obs = self._stack_states(obs, reset=False)
        return self._obs(
            obs,
            reward,
            is_last=self._done,
            is_terminal=bool(terminated),
        )

    def _stack_states(self, obs, reset=False):
        if self._frame_stack <= 1:
            return obs
        if reset or self._state_buffer is None:
            self._state_buffer = collections.deque(
                [obs['states']] * self._frame_stack,
                maxlen=self._frame_stack,
            )
        else:
            self._state_buffer.append(obs['states'])
        obs = dict(obs)
        obs['states'] = np.concatenate(list(self._state_buffer))
        return obs

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        out = {k: np.asarray(v) for k, v in obs.items()}
        info = self._info or {}
        pp = info.get('path_progress', 0.0)
        # Termination-reason flags (only meaningful at episode end). The CPU
        # env writes info["termination_reason"] in {"hole", "goal", "timeout"}.
        reason = info.get('termination_reason', '') if is_last else ''
        out.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        out['log/path_progress']      = np.float32(pp)
        out['log/hole_terminated']    = np.float32(1.0 if reason == 'hole' else 0.0)
        out['log/goal_terminated']    = np.float32(1.0 if reason == 'goal' else 0.0)
        out['log/timeout_terminated'] = np.float32(1.0 if reason == 'timeout' else 0.0)
        out['log/prior_hold_terminated'] = np.float32(1.0 if reason == 'prior_hold' else 0.0)
        # Placeholder SOOPER Mode 2 slots — overridden by PolicySwitcher outs.
        out['prior_risk']   = np.float32(0.0)
        out['prior_active'] = np.float32(0.0)
        out['prior_v']      = np.float32(0.0)
        return out

    def render(self):
        if self._include_vision and 'image' in self._env.observation_space.spaces:
            # Fallback: return zeros if called before any obs is available.
            h, w = self._size
            return np.zeros((h, w, 3), dtype=np.uint8)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _convert(self, space):
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)
