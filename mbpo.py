"""
Model-Based Policy Optimization (MBPO).

World model: probabilistic ensemble of MLPs (PyTorch).
  - Input: (obs, action) — the full (possibly frame-stacked) observation.
  - Output: Gaussian over the *base* obs delta (newest state + path); reward is
    analytic.
  - Residual prediction: base_next = base(obs) + delta. Under frame stacking the
    full next_obs is reconstructed by sliding the window (drop oldest frame,
    prepend the new (predicted-state, action) frame) so the stack stays
    temporally consistent rather than freely predicting the shifted history.

Policy: SB3 SAC trained on synthetic rollouts from the world model.

Training loop:
  1. Collect real transitions → real_buffer.
  2. Every model_train_freq steps: train ensemble on real_buffer.
  3. Generate synthetic rollouts with SAC + model → SAC replay buffer.
  4. SAC gradient steps (utd_ratio per env step).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from omegaconf import DictConfig


# ─────────────────────────────────────────────────────────────────────────────
# Mixed real/model replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class MixedReplayBuffer(ReplayBuffer):
    """SAC replay buffer whose ``sample()`` draws a fixed fraction of every
    batch from an external *real* buffer and the remainder from its own
    (synthetic/model) storage — the canonical MBPO real-ratio mix.

    Adds go only to this buffer's own storage (model rollouts + penalty
    transitions). Real executed transitions live in ``real_buffer`` and are
    pulled in at sample time according to ``real_ratio``. This replaces the
    previous implicit mix (real written into the SAC buffer every step), giving
    explicit control over the real/model batch composition.
    """

    def __init__(self, *args, real_buffer: ReplayBuffer, real_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_buffer = real_buffer
        self.real_ratio = float(real_ratio)

    @staticmethod
    def _concat(parts: list[ReplayBufferSamples]) -> ReplayBufferSamples:
        if len(parts) == 1:
            return parts[0]

        def cat(field):
            vals = [getattr(p, field) for p in parts]
            # Some SB3 versions carry optional fields (e.g. `discounts`) that are
            # None; leave them None rather than trying to concatenate.
            if any(v is None for v in vals):
                return None
            return torch.cat(vals, dim=0)

        return ReplayBufferSamples(*(cat(field) for field in ReplayBufferSamples._fields))

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        n_real = int(round(self.real_ratio * batch_size))
        # Fall back gracefully when either source is empty (e.g. before the
        # first rollout, or before any real data has been collected).
        if self.real_buffer.size() == 0:
            n_real = 0
        elif self.size() == 0:
            n_real = batch_size
        n_model = batch_size - n_real

        parts: list[ReplayBufferSamples] = []
        if n_model > 0:
            parts.append(super().sample(n_model, env=env))
        if n_real > 0:
            parts.append(self.real_buffer.sample(n_real, env=env))
        return self._concat(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble world model
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleLinear(nn.Module):
    """Single batched linear layer for E parallel models."""

    def __init__(self, E: int, in_dim: int, out_dim: int, identity_init: bool = False) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(E, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(E, 1, out_dim))
        # Default init: per-member truncated normal (keeps ensemble members diverse).
        nn.init.trunc_normal_(self.W, std=in_dim ** -0.5)
        # Identity prior: square weight matrices are the identity PLUS the trunc-normal
        # noise above (mean = I, so each member is a near-identity residual map, but
        # members still differ — preserving epistemic diversity). Non-square layers
        # keep the plain trunc-normal init. The decay target μ (see
        # ``EnsembleDynamics.identity_prior_mu``) remains the pure identity.
        if identity_init and in_dim == out_dim:
            with torch.no_grad():
                self.W.add_(torch.eye(in_dim).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, B, in_dim) → (E, B, out_dim)
        return x @ self.W + self.b


class IdentityPriorAdamW(Adam):
    """Adam with decoupled (AdamW-style) weight decay pulling each weight matrix
    toward a per-parameter prior mean ``mu`` instead of toward zero.

    PyTorch port of the reference's ``add_identity_decayed_weights``: square weight
    matrices are pulled to the identity (μ = I), non-square weight matrices toward
    0 (μ = 0), and biases are left untouched (not in ``mu``). The decay is applied
    directly to the parameter, bypassing Adam's first/second moment estimators —
    exactly the decoupled AdamW update ``θ ← θ - lr·wd·(θ - μ)``.
    """

    def __init__(self, params, lr, mu, weight_decay, **kw):
        super().__init__(params, lr=lr, weight_decay=0.0, **kw)
        # dict: Parameter -> prior-mean tensor. Params absent from mu are not decayed.
        self._id_mu = mu
        self._id_wd = float(weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        if self._id_wd > 0:
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    mu = self._id_mu.get(p)
                    if mu is None:
                        continue
                    # Decoupled decay toward μ: θ ← θ - lr·wd·(θ - μ).
                    p.add_(p - mu, alpha=-lr * self._id_wd)
        return super().step(closure)


class SamplingType:
    """Rollout sampling scheme for the probabilistic ensemble — the PyTorch
    counterpart of the reference ``SamplingType`` (``mean`` / ``TS1`` / ``TSInf``
    / ``DS``):

    - ``mean``  : average the ensemble means, sample only aleatoric noise.
    - ``TS1``   : trajectory sampling 1 — pick ONE ensemble member at random for
                  the whole batch on every call (the reference default).
    - ``TSInf`` : trajectory sampling ∞ — keep a fixed ensemble member for the
                  duration of a rollout; the index is set once per rollout via
                  ``EnsembleDynamics.set_sampling_idx``.
    - ``DS``    : distribution sampling — moment-match the ensemble into one
                  Gaussian (aleatoric + epistemic variance) and sample from it.
    """

    NAMES = ("mean", "TS1", "TSInf", "DS")

    def __init__(self, name: str = "TS1") -> None:
        self.set(name)

    def set(self, name: str) -> None:
        assert name in self.NAMES, f"sampling type must be one of {self.NAMES}"
        self.name = name


class EnsembleDynamics(nn.Module):
    """
    Probabilistic ensemble — PyTorch port of the reference Bayesian dynamics
    model's default ``ProbabilisticEnsembleModel`` path.

    Predicts ONLY the *base* observation delta as a Gaussian (reward is supplied
    separately by an analytic reward function, mirroring the reference's external
    ``RewardModel``). Under frame stacking (``n_stack > 1``) the base observation
    is the newest dynamic state + path vectors; the older stacked frames are a
    deterministic shift of known quantities, so they are reconstructed at
    sampling time rather than predicted. Without stacking the base delta is the
    whole ``delta_obs`` (original behaviour). The variance head outputs a
    standard deviation
    soft-clamped to ``[sig_min, sig_max]`` (reference parameterization) rather
    than the learnable-logvar-bounds scheme. A held-out per-dimension
    calibration multiplier (``calib_alpha``) recalibrates the predictive std at
    sampling time, replacing elite selection.

    Operates in the same observation space as the data it's trained on (the
    normalized space when VecNormalize is in use).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...],
        ensemble_size: int,
        sig_min: float = 1e-3,
        sig_max: float = 1e3,
        sampling_type: str = "TS1",
        n_stack: int = 1,
        frame_dim: int = 0,
        state_dim: int = 0,
        path_dim: int = 0,
        identity_prior: bool = False,
    ) -> None:
        super().__init__()
        self.identity_prior = bool(identity_prior)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.sig_min = float(sig_min)
        self.sig_max = float(sig_max)
        self.sampling = SamplingType(sampling_type)
        # TS-∞ ensemble index, set once per rollout (ignored by the other modes).
        self.sampling_idx = 0

        # ── frame-stacking layout ────────────────────────────────────────────
        # With obs_n_stack > 1 the observation is a sliding window:
        #   [ (state, action) × n_stack  (newest first) ] ++ [ path (path_dim) ]
        # Only the NEWEST state (state_dim) and the path (path_dim) are genuinely
        # new at each step; the older frames are an exact shift of known
        # quantities and the newest frame's action IS the model input. So the
        # network predicts only the base observation (state + path =
        # ``pred_dim`` dims) and ``sample()`` reconstructs the full stacked
        # next_obs by shifting in the new (predicted-state, action) frame. This
        # keeps the imagined stacks temporally consistent (correct velocity
        # signal) instead of letting the net hallucinate the shifted history.
        self.n_stack = int(n_stack)
        self.frame_dim = int(frame_dim)
        self.state_dim = int(state_dim)
        self.path_dim = int(path_dim)
        self.stacked = self.n_stack > 1
        # Dimensions the network actually predicts (a delta target):
        #   stacked  → base obs = newest state + path
        #   unstacked → the whole obs (original behaviour)
        self.pred_dim = (self.state_dim + self.path_dim) if self.stacked else obs_dim

        out_dim = self.pred_dim * 2   # mean + std for the predicted delta
        dims = [obs_dim + act_dim] + list(hidden) + [out_dim]
        self.net = nn.ModuleList([
            EnsembleLinear(ensemble_size, dims[i], dims[i + 1], identity_init=self.identity_prior)
            for i in range(len(dims) - 1)
        ])
        # Per-dimension calibration multiplier applied to the predicted std at
        # prediction time (opax-style recalibration; 1.0 until calibrated).
        self.register_buffer("calib_alpha", torch.ones(self.pred_dim))

    def set_sampling_type(self, name: str) -> None:
        self.sampling.set(name)

    def set_sampling_idx(self, idx: int) -> None:
        self.sampling_idx = int(idx) % self.ensemble_size

    def identity_prior_mu(self) -> dict:
        """Prior-mean pytree for the identity prior (reference ``make_identity_mu``),
        as a dict mapping each *weight* Parameter to its μ: identity for square
        ensemble weight stacks (E, d, d), zeros for non-square ones. Biases are
        omitted (not decayed). Consumed by ``IdentityPriorAdamW``."""
        mu = {}
        for layer in self.net:
            W = layer.W                    # (E, in, out)
            E, in_dim, out_dim = W.shape
            if in_dim == out_dim:
                eye = torch.eye(in_dim, device=W.device, dtype=W.dtype)
                mu[W] = eye.unsqueeze(0).expand(E, in_dim, out_dim).clone()
            else:
                mu[W] = torch.zeros_like(W)
        return mu

    def _forward(self, obs: torch.Tensor, act: torch.Tensor):
        """obs/act: (B, d) → mean, std each (E, B, obs_dim).

        The std head is soft-clamped to ``[sig_min, sig_max]`` (reference's
        sig_min/sig_max aleatoric-std bounds)."""
        x = torch.cat([obs, act], -1).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for layer in self.net[:-1]:
            x = F.silu(layer(x))
        x = self.net[-1](x)
        mean, raw_std = x.chunk(2, -1)
        std = F.softplus(raw_std) + self.sig_min
        std = std.clamp(self.sig_min, self.sig_max)
        return mean, std

    def _base_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the base observation (newest state + path) from a (possibly
        stacked) observation, along the last dim. Identity when unstacked."""
        if not self.stacked:
            return obs
        state = obs[..., : self.state_dim]
        path = obs[..., obs.shape[-1] - self.path_dim:]
        return torch.cat([state, path], dim=-1)

    def delta_target(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Residual target the network is trained to predict: the base-obs delta
        (newest state + path) when stacked, else the full obs delta."""
        return self._base_obs(next_obs) - self._base_obs(obs)

    def nll_loss(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL (std parameterization) summed over output dims, averaged
        over batch and ensemble. Matches the reference ``gaussian_log_likelihood``
        (constant term dropped)."""
        mean, std = self._forward(obs, act)
        target = self.delta_target(obs, next_obs)
        tgt = target.unsqueeze(0).expand_as(mean)
        nll = (torch.log(std) + 0.5 * ((tgt - mean) / std).pow(2)).sum(-1).mean()
        return nll

    @torch.no_grad()
    def sample(self, obs: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        One-step prediction for a batch using the configured ``SamplingType``.

        Returns: next_obs (B, obs_dim), uncertainty (B,)
            uncertainty = ‖σ_tot‖₂ — the L2 norm of the DS total predictive std
            combining ALEATORIC (mean of per-member variances) and EPISTEMIC
            (variance of per-member means) uncertainty — used as the optimism /
            exploration bonus on the analytic rollout reward.
        """
        dev = next(self.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(dev)
        act_t = torch.from_numpy(act).float().to(dev)
        mean, std = self._forward(obs_t, act_t)    # (E, B, D)
        std = std * self.calib_alpha               # recalibrated aleatoric std
        E = mean.shape[0]

        # DS moments (used for the optimism bonus regardless of sampling mode).
        al_var = std.pow(2).mean(0)                # (B, D) aleatoric
        ep_var = mean.var(0)                       # (B, D) epistemic
        tot_std = (al_var + ep_var).sqrt()         # (B, D)
        unc = tot_std.norm(dim=1)                  # (B,) ‖σ_tot‖₂

        name = self.sampling.name
        if name == "mean":
            m = mean.mean(0)
            out = m + al_var.sqrt() * torch.randn_like(m)
        elif name == "TS1":
            idx = int(torch.randint(0, E, (1,)).item())
            out = mean[idx] + std[idx] * torch.randn_like(mean[idx])
        elif name == "TSInf":
            idx = int(self.sampling_idx) % E
            out = mean[idx] + std[idx] * torch.randn_like(mean[idx])
        elif name == "DS":
            out = mean.mean(0) + tot_std * torch.randn_like(tot_std)
        else:
            raise ValueError(f"unknown sampling type {name!r}")

        # ``out`` is the predicted base-obs delta (B, pred_dim).
        base_next = self._base_obs(obs_t) + out          # (B, pred_dim)
        next_obs = self._reconstruct_next_obs(obs_t, act_t, base_next)
        return next_obs.cpu().numpy(), unc.cpu().numpy()

    def _reconstruct_next_obs(
        self, obs: torch.Tensor, act: torch.Tensor, base_next: torch.Tensor
    ) -> torch.Tensor:
        """Build the full next observation from the predicted base obs.

        Unstacked: the base obs IS the full obs. Stacked: slide the window —
        drop the oldest frame, prepend the new ``(predicted_state, action)``
        frame, and append the freshly predicted path vectors. This guarantees
        the historical frames in the imagined stack are the exact (known) shift
        of the current stack, so the velocity the policy reads off consecutive
        frames stays physically consistent."""
        if not self.stacked:
            return base_next
        new_state = base_next[..., : self.state_dim]            # (B, state_dim)
        new_path = base_next[..., self.state_dim:]              # (B, path_dim)
        new_frame = torch.cat([new_state, act], dim=-1)         # (B, frame_dim)
        n_frames = self.n_stack * self.frame_dim
        frames = obs[..., :n_frames]                            # (B, n_stack*frame_dim)
        # Keep the newest (n_stack-1) frames; the oldest is dropped.
        kept = frames[..., : (self.n_stack - 1) * self.frame_dim]
        return torch.cat([new_frame, kept, new_path], dim=-1)

    @torch.no_grad()
    def update_calibration(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> float:
        """Recalibrate the predictive std on a held-out batch (PyTorch
        equivalent of the reference ``calculate_calibration_alpha``).

        Sets a per-dimension multiplier ``alpha`` so that the recalibrated
        Gaussian has unit-variance standardized residuals — alpha[d]² = E[((y−μ)/σ)²].
        Returns a scalar calibration-error proxy: |E[z²/alpha²] − 1|."""
        mean, std = self._forward(obs, act)
        target = self.delta_target(obs, next_obs)
        tgt = target.unsqueeze(0).expand_as(mean)
        z2 = ((tgt - mean) / std).pow(2)           # (E, B, D)
        alpha = z2.mean(dim=(0, 1)).clamp_min(1e-8).sqrt()   # (D,)
        self.calib_alpha = alpha
        scaled = z2 / alpha.pow(2)
        return float((scaled.mean() - 1.0).abs())


# ─────────────────────────────────────────────────────────────────────────────
# Analytic reward model (closed-form, replaces a learned reward head)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticCyberRunnerReward:
    """Closed-form maze reward — the PyTorch-side analogue of the reference's
    separate ``RewardModel``, here an exact analytic function rather than a
    learned net.

    It mirrors ``CyberRunnerEnv._compute_reward`` (main task): dense signed
    path-progress shaping + goal bonus + hole penalty. Rollouts run in
    VecNormalize's NORMALIZED observation space, so the reward first un-normalizes
    obs to recover raw board ball positions (dims [2:4]), then evaluates the same
    progress/goal/hole terms the env uses, returning the same RAW reward the env
    produces. Imagined transitions are stored raw and normalized at sample time by
    VecNormalize, exactly like the real ones.
    """

    def __init__(self, vecnorm, raw_env) -> None:
        from envs.cyberrunner import (
            compute_path_progress, GOAL_BONUS, GOAL_THRESHOLD, HOLE_RADIUS,
            PROGRESS_DELTA_CLIP,
        )
        if getattr(raw_env, "prior_mode", False):
            raise NotImplementedError(
                "AnalyticCyberRunnerReward only implements the main-task reward; "
                "prior_mode (recovery) reward depends on ball velocity not present "
                "in a single observation."
            )
        self._vecnorm = vecnorm
        self._progress = compute_path_progress
        self._GOAL_BONUS = float(GOAL_BONUS)
        self._GOAL_THRESHOLD = float(GOAL_THRESHOLD)
        self._HOLE_RADIUS = float(HOLE_RADIUS)
        self._PROGRESS_DELTA_CLIP = float(PROGRESS_DELTA_CLIP)
        self.waypoints = raw_env.waypoints
        self.seg_lengths = raw_env.seg_lengths
        self.cum_distances = raw_env.cum_distances
        self.walls_h = raw_env.walls_h
        self.walls_v = raw_env.walls_v
        self.holes = raw_env.holes
        self.goal_pos = raw_env.goal_pos
        self.scale = float(raw_env.dense_main_progress_scale)
        self.hole_penalty = float(raw_env.hole_penalty)
        # Frame-stacking layout, so ``init_prev_progress`` can scan the stacked
        # history for the most recent on-path ball position (the env's sticky
        # ``_prev_progress`` can reach back through off-path excursions).
        self.n_stack = int(getattr(raw_env, "obs_n_stack", 1))
        # Ball xy sits at offset [2:4] within each stacked (state, action) frame,
        # and at [2:4] of the plain base obs when unstacked.
        self.frame_dim = int(getattr(raw_env, "_frame_dim", 0)) if self.n_stack > 1 else 0

    def _ball_pos(self, obs_norm: np.ndarray) -> np.ndarray:
        """Un-normalize a batch of normalized obs and return raw ball xy (B, 2)."""
        raw = self._vecnorm.unnormalize_obs(np.asarray(obs_norm, dtype=np.float32))
        return np.asarray(raw)[:, 2:4]

    def _progress_of(self, ball: np.ndarray) -> float:
        p, _, _, _ = self._progress(
            ball, self.waypoints, self.seg_lengths, self.cum_distances,
            self.walls_h, self.walls_v, self.holes,
        )
        return float(p)

    def _progress_batch(self, ball: np.ndarray) -> np.ndarray:
        """Per-sample path progress for a batch of ball positions (B, 2) → (B,).
        Negative where the ball is off-path (path not detected)."""
        n = ball.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = self._progress_of(ball[i])
        return out

    def init_prev_progress(self, obs: np.ndarray) -> np.ndarray:
        """Initial sticky ``prev_progress`` (B,) at the start of a rollout.

        The env's ``_prev_progress`` holds the LAST on-path progress and only
        updates when the ball is on-path, so after an off-path excursion it
        reaches back to before the excursion. We reproduce that by scanning the
        stacked frames newest-first and taking the first on-path progress. If no
        frame in the window is on-path we return -1 (off-path); the sticky rule
        then withholds dense reward until the ball rejoins the path, matching the
        env's ``prev_progress >= 0`` gate."""
        raw = np.asarray(
            self._vecnorm.unnormalize_obs(np.asarray(obs, dtype=np.float32))
        )
        n = raw.shape[0]
        out = np.full(n, -1.0, dtype=np.float32)
        for i in range(n):
            for k in range(self.n_stack):
                off = k * self.frame_dim   # 0 for the (unstacked) base obs
                ball = raw[i, off + 2: off + 4]
                p = self._progress_of(ball)
                if p >= 0:
                    out[i] = p
                    break
        return out

    def reward(self, prev_progress: np.ndarray, next_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stateful reward for one imagined step, mirroring
        ``CyberRunnerEnv._compute_reward`` + its sticky ``_prev_progress`` update.

        Given the carried sticky ``prev_progress`` (B,) and the predicted
        ``next_obs``, returns ``(reward, new_prev_progress)`` where:
          - dense = (progress(next) - prev_progress) * scale, gated on BOTH being
            on-path (exactly the env's ``curr>=0 and prev>=0`` condition),
          - goal / hole bonuses as before, and
          - new_prev_progress is ``prev_progress`` updated to progress(next) only
            when next is on-path (the env's sticky update)."""
        ball_next = self._ball_pos(next_obs)
        pn = self._progress_batch(ball_next)
        on_path = pn >= 0
        valid = on_path & (prev_progress >= 0)
        # Clip Δprogress to reject spurious segment-detection jumps (matches the
        # env's PROGRESS_DELTA_CLIP), so imagined dense rewards stay on the same
        # bounded scale as the real ones.
        dprog = np.clip(pn - prev_progress, -self._PROGRESS_DELTA_CLIP, self._PROGRESS_DELTA_CLIP)
        dense = np.where(valid, dprog * self.scale, 0.0)
        goal = np.where(
            np.linalg.norm(ball_next - self.goal_pos, axis=1) < self._GOAL_THRESHOLD,
            self._GOAL_BONUS, 0.0,
        )
        hole = np.where(
            (np.linalg.norm(self.holes[None] - ball_next[:, None], axis=2)
             < self._HOLE_RADIUS).any(axis=1),
            -self.hole_penalty, 0.0,
        )
        rew = (dense + goal + hole).astype(np.float32)
        new_prev = np.where(on_path, pn, prev_progress).astype(np.float32)
        return rew, new_prev

    def terminal(self, next_obs: np.ndarray) -> np.ndarray:
        """Vectorized absorbing-state mask (B,) for imagined transitions.

        Mirrors ``CyberRunnerEnv._check_termination`` (main task): the ball
        entering a hole (failure) or reaching the goal (success) is terminal.
        Timeout truncation is intentionally NOT a terminal here — it shouldn't
        zero the bootstrap, and imagined rollouts are far shorter than the
        episode length anyway. Restoring these terminals makes hole states
        absorbing in imagination (Q = −hole_penalty, no bootstrap), matching the
        real MDP, instead of being treated as survivable recurring costs."""
        ball = self._ball_pos(next_obs)
        in_hole = (np.linalg.norm(
            self.holes[None] - ball[:, None], axis=2
        ) < self._HOLE_RADIUS).any(axis=1)
        at_goal = np.linalg.norm(ball - self.goal_pos, axis=1) < self._GOAL_THRESHOLD
        return (in_hole | at_goal).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# MBPO trainer
# ─────────────────────────────────────────────────────────────────────────────

class MBPOTrainer:

    def __init__(self, env, cfg: DictConfig, device: str = "cpu", seed: int = 0) -> None:
        self.env = env
        self.device = device
        self.seed = int(seed)
        ac = cfg.algo

        # Seed all RNGs (Python, NumPy, Torch) and the env's action sampler so
        # the warmup random exploration and model rollouts are reproducible.
        set_random_seed(self.seed)
        self.env.action_space.seed(self.seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        # Frame-stacking layout (read from the raw env). When n_stack > 1 the
        # dynamics net predicts only the base obs (newest state + path) and
        # reconstructs the rest of the stack by shifting; see EnsembleDynamics.
        raw_env = env.venv.envs[0].unwrapped
        n_stack = int(getattr(raw_env, "obs_n_stack", 1))
        state_dim = int(getattr(raw_env, "_stack_obs_dim", 0))
        path_dim = int(getattr(raw_env, "_path_dim", 0))
        frame_dim = state_dim + act_dim   # (state, action) per stacked frame

        self.dynamics = EnsembleDynamics(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=tuple(ac.hidden_sizes),
            ensemble_size=ac.ensemble_size,
            sig_min=float(ac.get("sig_min", 1e-3)),
            sig_max=float(ac.get("sig_max", 1e3)),
            sampling_type=ac.get("sampling_type", "TS1"),
            n_stack=n_stack,
            frame_dim=frame_dim,
            state_dim=state_dim,
            path_dim=path_dim,
            identity_prior=bool(ac.get("identity_prior", False)),
        ).to(device)

        # Analytic reward function — the closed-form maze reward, replacing the
        # old joint reward head (mirrors the reference's separate RewardModel).
        # Pull the raw CyberRunnerEnv out from under the VecNormalize wrapper so
        # we can read its path/hole geometry; un-normalization uses the wrapper.
        # (``raw_env`` was resolved above to read the frame-stacking layout.)
        self.reward_model = AnalyticCyberRunnerReward(env, raw_env)

        if self.dynamics.identity_prior:
            # Decoupled (AdamW-style) identity prior: square weight matrices decay
            # toward I, non-square toward 0, biases untouched — the prior bypasses
            # Adam's moment estimators (reference add_identity_decayed_weights).
            self.model_opt = IdentityPriorAdamW(
                self.dynamics.parameters(),
                lr=ac.model_lr,
                mu=self.dynamics.identity_prior_mu(),
                weight_decay=float(ac.get("identity_weight_decay", 1e-4)),
            )
        else:
            self.model_opt = Adam(
                self.dynamics.parameters(),
                lr=ac.model_lr,
                weight_decay=ac.model_weight_decay,
            )

        # Real experience buffer (separate from SAC's model buffer). Feeds both
        # dynamics training and the real fraction of each SAC batch.
        self.real_buffer = ReplayBuffer(
            buffer_size=ac.real_buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=1,
            handle_timeout_termination=True,
        )

        # Fraction of every SAC batch drawn from real (vs. model) data. The
        # remainder comes from synthetic rollouts. Replaces the old implicit
        # mix where real transitions were written into the SAC buffer directly.
        self.real_ratio = float(ac.get("real_ratio", 0.1))

        # Optimism / exploration bonus: rollout reward += optimism * ‖σ‖₂, where
        # ‖σ‖₂ is the world model's predicted standard deviation. Rewards visiting
        # state-actions the model is uncertain about (optimism in the face of
        # uncertainty). 0.0 disables it. _unc_mean tracks the last rollout's mean
        # ‖σ‖₂ for tuning the coefficient against the reward scale.
        self.optimism = float(ac.get("optimism", 0.0))
        self._unc_mean = float("nan")
        self._calib_err = float("nan")

        # (Reward conditioning removed: rewards are stored RAW and normalized at
        # sample time by VecNormalize via _vec_normalize_env — see below.)

        # Uncertainty-based rollout truncation (MOPO/M2AC-style). A branch stops
        # being rolled forward once its predictive uncertainty ‖σ‖₂ exceeds this
        # threshold, so imagined rollouts self-limit to the model's reliable
        # region instead of compounding error off-distribution (which destroys
        # the SAC critic — see run earthy-shape-214). None/inf disables it (rely
        # only on max_rollout_length). Tune against train/model_unc: set it a bit
        # above the on-distribution mean so normal steps pass but drift is cut.
        thr = ac.get("rollout_unc_threshold", None)
        self.rollout_unc_threshold = float("inf") if thr is None else float(thr)
        self._rollout_len_eff = float("nan")   # realized mean rollout length

        # SAC operates on synthetic rollouts stored in its replay buffer
        self.sac = SAC(
            "MlpPolicy", env, verbose=0, device=device,
            learning_rate=ac.learning_rate,
            buffer_size=ac.model_buffer_size,
            batch_size=ac.batch_size,
            tau=ac.tau,
            gamma=ac.gamma,
            learning_starts=0,
            ent_coef=ac.ent_coef,
            target_entropy=ac.get("target_entropy", "auto"),
            seed=self.seed,
        )
        # Swap SAC's plain replay buffer for one that mixes in real data at
        # sample time according to real_ratio.
        self.sac.replay_buffer = MixedReplayBuffer(
            ac.model_buffer_size,
            env.observation_space,
            env.action_space,
            device=device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
            real_buffer=self.real_buffer,
            real_ratio=self.real_ratio,
        )
        # SAC.train() requires _logger; silence it — MBPO has its own logging
        self.sac.set_logger(configure_logger(folder=None, format_strings=[]))
        # RAW-storage architecture (mirrors pure SAC exactly). Both buffers store
        # UN-normalized obs and reward; SB3 normalizes obs AND reward at SAMPLE
        # time via _vec_normalize_env (ReplayBuffer._get_samples). This is
        # byte-for-byte the pure-SAC pipeline, so at real_ratio=1 the SAC policy
        # sees identical data. It also gives adaptive reward normalization for
        # free (÷ running return-std, clip ±10) applied uniformly to the real and
        # model halves — replacing the old fixed reward_scale/clip conditioning.
        self.sac._vec_normalize_env = env
        # Keep VecNormalize's reward normalization ON (as pure SAC has it): stats
        # (ret_rms) update from the real env stream during learn(); model rewards
        # are stored raw and normalized with the same stats at sample time.

        # ── Safety shield: recovery/backup policy ─────────────────────────────
        # Loads a trained recovery policy + its OWN VecNormalize. At each step the
        # learning policy's proposed action is judged by the backup Q-function
        # (recoverability); if below threshold the recovery policy acts instead.
        self._rollout_shield_rate = float("nan")   # last rollout's mean trigger rate
        self._load_backup(ac, cfg, device)

        self._ac = ac

    # ── safety shield (backup / recovery policy) ──────────────────────────────

    def _load_backup(self, ac, cfg, device) -> None:
        """Load the recovery/backup policy and ITS normalizer for the safety shield.

        Source priority: a wandb run id (downloads the model artifact — policy
        ``.zip`` + VecNormalize ``.pkl`` + env cfg), else local paths. Both the
        policy AND its VecNormalize are loaded: the shield renormalizes behavioral
        obs into the backup's own obs space before querying its Q-function
        (``_to_backup_obs``). ``backup_wandb_id: null`` disables the shield.
        """
        self.backup_policy = None
        self.backup_vecnorm = None
        self.safety_threshold = float(ac.get("safety_threshold", 0.3))

        bp_path = ac.get("backup_policy_path", None)
        bv_path = ac.get("backup_vecnorm_path", None)
        b_env_cfg_path = None

        wandb_id = ac.get("backup_wandb_id", None)
        if wandb_id:
            import wandb as wb
            project = ac.get("backup_wandb_project", "cyberrunner")
            api = wb.Api()
            run = api.run(f"{project}/{wandb_id}")
            artifact = next((a for a in run.logged_artifacts() if a.type == "model"), None)
            if artifact is None:
                raise RuntimeError(
                    f"[MBPO] No model artifact found for wandb run {project}/{wandb_id}"
                )
            root = Path(artifact.download())
            bp_path = str(next(root.glob("*.zip")))
            bv_path = str(next(root.glob("*.pkl")))
            cfgs = list(root.glob("*_env_cfg.json")) or list(root.glob("*.json"))
            b_env_cfg_path = str(cfgs[0]) if cfgs else None
            print(f"[MBPO] Downloaded backup artifact from {project}/{wandb_id}: {bp_path}", flush=True)

        if not (bp_path and bv_path and os.path.exists(bp_path) and os.path.exists(bv_path)):
            print("[MBPO] No backup policy configured — safety shield DISABLED.", flush=True)
            return

        # PPO backups expose V(s); SAC backups expose Q(s,a). Infer from filename.
        model_cls = PPO if "ppo" in os.path.basename(bp_path).lower() else SAC
        self.backup_policy = model_cls.load(bp_path, device=device)

        # The dummy env only has to reproduce the backup normalizer's obs/action
        # shape, which is fixed by obs_n_stack + layout. Prefer the backup's OWN
        # saved env cfg; fall back to the learner's env config (schema drift-safe).
        from envs.cyberrunner import CyberRunnerEnv
        ecfg = {}
        if b_env_cfg_path and os.path.exists(b_env_cfg_path):
            with open(b_env_cfg_path) as f:
                ecfg = json.load(f)
        obs_n_stack = int(ecfg.get("obs_n_stack", cfg.env.get("obs_n_stack", 1)))
        layout = ecfg.get("layout", cfg.env.get("layout", "hard"))
        dummy = DummyVecEnv([lambda: CyberRunnerEnv(obs_n_stack=obs_n_stack, layout=layout)])
        self.backup_vecnorm = VecNormalize.load(bv_path, dummy)
        self.backup_vecnorm.training = False
        self.backup_vecnorm.norm_reward = False
        print(
            f"[MBPO] Safety shield loaded ({model_cls.__name__}): {bp_path}  "
            f"threshold={self.safety_threshold}",
            flush=True,
        )

    def _to_backup_obs(self, obs_norm: np.ndarray) -> np.ndarray:
        """Convert behavioral-normalized obs → backup-normalized obs.

        The learner and backup share the identical obs layout (same frame stacking
        + path vectors), so this is a pure re-normalization: de-normalize with the
        behavioral VecNormalize stats to physical units, then re-normalize with the
        backup's own stats (and its obs clip)."""
        rms = self.env.obs_rms
        raw = obs_norm * np.sqrt(rms.var + self.env.epsilon) + rms.mean
        b = self.backup_vecnorm
        return np.clip(
            (raw - b.obs_rms.mean) / np.sqrt(b.obs_rms.var + b.epsilon),
            -b.clip_obs, b.clip_obs,
        ).astype(np.float32)

    def _recovery_values(self, backup_obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Recoverability of ``act`` in the given (backup-normalized) states, (B,).

        SAC backup: min(Q1,Q2)(s, a) — the recovery critic's value of taking the
        *proposed* action. PPO backup: V(s) (action-independent)."""
        obs_t = torch.as_tensor(backup_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if isinstance(self.backup_policy, PPO):
                values = self.backup_policy.policy.predict_values(obs_t)
            else:
                act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
                qs = self.backup_policy.policy.critic(obs_t, act_t)
                values = torch.min(qs[0], qs[1])
        return values.cpu().numpy().flatten()

    def _shield(self, obs_norm: np.ndarray, act: np.ndarray):
        """Apply the safety shield to a batch of proposed actions.

        Returns ``(act_exec, unsafe, values)``. Where the backup deems the proposed
        action unrecoverable (recoverability < ``safety_threshold``), the executed
        action is overwritten with the recovery policy's action. ``act`` itself is
        never mutated (a copy is returned when any override happens). If no backup
        is loaded the shield is a no-op."""
        if self.backup_policy is None:
            return act, np.zeros(len(act), dtype=bool), None
        backup_obs = self._to_backup_obs(obs_norm)
        values = self._recovery_values(backup_obs, act)      # recoverability of a_learned
        unsafe = values < self.safety_threshold
        act_exec = act
        if unsafe.any():
            act_exec = act.copy()
            backup_act, _ = self.backup_policy.predict(backup_obs[unsafe], deterministic=True)
            act_exec[unsafe] = backup_act
        return act_exec, unsafe, values

    def predict(self, obs: np.ndarray, prev_obs: np.ndarray | None = None, deterministic: bool = True):
        """Plain (unshielded) SAC predict. ``prev_obs`` is accepted and ignored so
        existing eval callers passing ``(obs, prev_obs)`` keep working."""
        return self.sac.predict(obs, deterministic=deterministic)

    def shielded_predict(self, obs: np.ndarray, prev_obs: np.ndarray | None = None, deterministic: bool = True):
        """SAC predict with the safety shield applied — the executed action (backup
        where the learning action is unrecoverable). This is the deployment/eval
        behavior and mirrors the real-env shielding in ``learn()``. ``prev_obs`` is
        accepted and ignored. No-op when no backup policy is loaded."""
        act, state = self.sac.predict(obs, deterministic=deterministic)
        act, _, _ = self._shield(np.asarray(obs, dtype=np.float32), act)
        return act, state

    # ── world model training ──────────────────────────────────────────────

    def _train_dynamics(self) -> tuple[float, float]:
        ac = self._ac
        if self.real_buffer.size() < ac.batch_size:
            return float("nan"), float("nan")
        self.dynamics.train()
        train_losses = []
        for _ in range(ac.model_train_epochs):
            # real_buffer stores RAW obs; pass env so obs/next_obs come back
            # NORMALIZED — the dynamics model operates in VecNormalize space.
            b = self.real_buffer.sample(min(ac.model_batch_size, self.real_buffer.size()), env=self.env)
            # Target (base-obs delta) is derived from next_obs inside the model;
            # reward is supplied analytically.
            loss = self.dynamics.nll_loss(b.observations, b.actions, b.next_observations)
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.model_opt.step()
            train_losses.append(loss.item())
        # Recalibrate predictive std on a held-out validation batch.
        b = self.real_buffer.sample(min(2048, self.real_buffer.size()), env=self.env)
        self._calib_err = self.dynamics.update_calibration(b.observations, b.actions, b.next_observations)
        with torch.no_grad():
            val_loss = self.dynamics.nll_loss(b.observations, b.actions, b.next_observations).item()
        self.dynamics.eval()
        return float(np.mean(train_losses)), val_loss

    # ── synthetic rollout generation ──────────────────────────────────────

    def _rollout_length(self, step: int) -> int:
        ac = self._ac
        t = max(0.0, (step - ac.warmup_steps) / max(ac.rollout_schedule_steps, 1))
        length = ac.min_rollout_length + t * (ac.max_rollout_length - ac.min_rollout_length)
        return int(min(length, ac.max_rollout_length))

    def _generate_rollouts(self, length: int) -> None:
        ac = self._ac
        # real_buffer stores RAW; pass env so start states come back NORMALIZED —
        # the dynamics model, policy, and analytic reward all operate in
        # VecNormalize space during the rollout. Outputs are un-normalized to raw
        # before storage so the SAC buffer holds raw (normalized again at sample).
        b = self.real_buffer.sample(ac.rollout_batch_size, env=self.env)
        obs = b.observations.cpu().numpy()   # normalized obs space

        # TS-∞: fix one ensemble member for the whole rollout (no-op for the
        # other sampling modes).
        self.dynamics.set_sampling_idx(int(np.random.randint(self.dynamics.ensemble_size)))

        all_obs, all_nobs, all_act, all_rew, all_done = [], [], [], [], []
        # Branches that hit an absorbing state (hole/goal) stop being rolled
        # forward — so a single rollout can't keep stamping penalties past a
        # terminal, and longer rollouts stay on the real MDP's support.
        alive = np.ones(len(obs), dtype=bool)
        n_start = len(obs)
        stored_per_branch = np.zeros(n_start, dtype=np.int64)
        unc_means: list[float] = []
        # Sticky path progress carried across the rollout, replicating the env's
        # stateful ``_prev_progress`` (holds the last on-path value, only updates
        # when on-path). Initialised from the stacked history of the start state.
        prev_prog = self.reward_model.init_prev_progress(obs)
        shield_rates: list[float] = []
        for _ in range(length):
            act, _ = self.sac.predict(obs, deterministic=False)   # a_learned
            # Safety shield: overwrite the executed action with the recovery
            # policy's where a_learned is judged unrecoverable. The dynamics and
            # reward are queried with the EXECUTED action, but the stored action
            # (below) is a_learned — action overwriting on the planning MDP:
            #   transition = (s, a_learned, s'(s, a_exec), r(s, a_exec)).
            act_exec, unsafe, _ = self._shield(obs, act)
            if unsafe is not None:
                shield_rates.append(float(unsafe.mean()))
            nobs, unc = self.dynamics.sample(obs, act_exec)
            # Reward from the analytic model (raw scale), not the dynamics net.
            # Returns the reward and the updated sticky progress for next step.
            rew, prev_prog = self.reward_model.reward(prev_prog, nobs)
            rew = rew + self.optimism * unc   # optimism / UCB exploration bonus
            unc_means.append(float(unc.mean()))
            # Analytic terminals (hole = failure, goal = success), mirroring the
            # real env. Computed on the same (pre-clip) nobs the reward used, so
            # the hole/goal detection can't disagree with the reward's terms.
            # Absorbing states get a zero-bootstrap target, so the critic stops
            # treating holes as survivable recurring costs.
            done = self.reward_model.terminal(nobs)

            # Only keep transitions the model is confident about: an alive branch
            # whose predictive uncertainty is within threshold. High-uncertainty
            # (off-distribution) predictions are dropped, not stored, so they
            # never become bootstrap targets.
            reliable = unc <= self.rollout_unc_threshold
            m = alive & reliable
            all_obs.append(obs[m])
            all_nobs.append(nobs[m])
            all_act.append(act[m])
            all_rew.append(rew[m])
            all_done.append(done[m])
            stored_per_branch[m] += 1

            # Stop rolling a branch if it terminated, went off-distribution, or
            # was already dead.
            alive = alive & reliable & ~done
            if not alive.any():
                break
            obs = nobs

        self._unc_mean = float(np.mean(unc_means)) if unc_means else float("nan")
        self._rollout_len_eff = float(stored_per_branch.mean())
        self._rollout_shield_rate = float(np.mean(shield_rates)) if shield_rates else float("nan")

        # The rollout ran in normalized space; un-normalize obs/next_obs to RAW so
        # the SAC buffer holds raw (VecNormalize re-normalizes at sample time,
        # keeping model transitions on the same footing as real ones). Rewards
        # are analytic raw already.
        obs_raw = self.env.unnormalize_obs(np.concatenate(all_obs))
        nobs_raw = self.env.unnormalize_obs(np.concatenate(all_nobs))
        self._batch_add_to_sac(
            obs_raw, nobs_raw,
            np.concatenate(all_act), np.concatenate(all_rew),
            np.concatenate(all_done),
        )

    def _batch_add_to_sac(self, obs: np.ndarray, nobs: np.ndarray,
                           act: np.ndarray, rew: np.ndarray,
                           dones: np.ndarray | None = None) -> None:
        """Write a batch of transitions directly into SAC's replay buffer arrays."""
        buf = self.sac.replay_buffer
        n = len(obs)
        done = np.zeros((n, 1), dtype=np.float32) if dones is None else dones.reshape(n, 1).astype(np.float32)
        rew2 = rew[:, None]    # (n, 1) to match buf.rewards shape (buffer_size, n_envs)

        def _write(start: int, src_start: int, src_end: int) -> None:
            sl = slice(start, start + (src_end - src_start))
            s, e = src_start, src_end
            buf.observations[sl, 0] = obs[s:e]
            buf.next_observations[sl, 0] = nobs[s:e]
            buf.actions[sl, 0] = act[s:e]
            buf.rewards[sl] = rew2[s:e]
            buf.dones[sl] = done[s:e]
            if hasattr(buf, "timeouts"):
                buf.timeouts[sl] = 0

        pos = buf.pos
        if pos + n <= buf.buffer_size:
            _write(pos, 0, n)
            buf.pos = (pos + n) % buf.buffer_size
            if buf.pos < pos:
                buf.full = True
        else:
            first = buf.buffer_size - pos
            _write(pos, 0, first)
            _write(0, first, n)
            buf.pos = n - first
            buf.full = True

    # ── main loop ─────────────────────────────────────────────────────────

    def learn(self, total_timesteps: int, wandb_run=None) -> None:
        ac = self._ac
        obs = self.env.reset()                    # normalized obs (fed to the policy)
        obs_raw = self.env.get_original_obs()      # raw obs (stored in the buffer)
        n_envs = self.env.num_envs
        step = 0

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        q_values: list[float] = []
        policy_entropies: list[float] = []
        ent_coefs: list[float] = []
        model_train_nlls: list[float] = []
        model_val_nlls: list[float] = []
        backup_triggers: list[float] = []   # per-step real-env shield trigger rate

        while step < total_timesteps:
            # Random exploration during warmup, SAC policy afterwards
            if step < ac.warmup_steps:
                act = np.stack([self.env.action_space.sample() for _ in range(n_envs)])
                act_exec = act
            else:
                act, _ = self.sac.predict(obs, deterministic=False)   # a_learned
                # Safety shield: play the recovery policy where the learning action
                # is judged unrecoverable. The EXECUTED action drives the real env
                # (and is stored for world-model learning below).
                act_exec, unsafe, _ = self._shield(obs, act)
                if unsafe is not None and unsafe.size:
                    backup_triggers.append(float(unsafe.mean()))

            nobs, rew, done, infos = self.env.step(act_exec)
            # RAW views for buffer storage (SB3 stores raw, normalizes at sample).
            nobs_raw = self.env.get_original_obs()
            rew_raw = self.env.get_original_reward()

            # Store each parallel env's transition. Real executed transitions go
            # to real_buffer only — they reach SAC via the real_ratio sampler.
            for i in range(n_envs):
                terminal = bool(done[i]) and not infos[i].get("TimeLimit.truncated", False)
                t = np.array([terminal])
                # On episode end the VecEnv auto-resets, so nobs_raw[i] is the
                # RESET obs; the true final obs is in infos[i]["terminal_observation"]
                # (which VecNormalize stored NORMALIZED — un-normalize it back to
                # raw). Substitute it so the stored next_obs is correct (matters
                # for timeout bootstrapping; SB3 does the same in _store_transition).
                next_o = nobs_raw[i:i+1]
                if bool(done[i]) and "terminal_observation" in infos[i]:
                    term_norm = np.asarray(infos[i]["terminal_observation"], dtype=np.float32)[None]
                    next_o = self.env.unnormalize_obs(term_norm)
                # Store RAW obs + RAW reward; VecNormalize normalizes both at
                # sample time (self.sac._vec_normalize_env = env). The stored action
                # is the EXECUTED one (backup where the shield intervened) so the
                # world model learns the true dynamics s' = f(s, a_exec).
                self.real_buffer.add(
                    obs_raw[i:i+1], next_o, act_exec[i:i+1],
                    rew_raw[i:i+1], t, [infos[i]],
                )
                ep_info = infos[i].get("episode")
                if ep_info is not None:
                    ep_rewards.append(float(ep_info["r"]))
                    ep_lengths.append(int(ep_info["l"]))

            obs = nobs
            obs_raw = nobs_raw
            step += n_envs

            # ``step`` advances by n_envs, so exact ``step % freq == 0`` tests
            # rarely align (they fire at lcm(n_envs, freq)). Use a boundary-
            # crossing test so cadence matches the configured freq for any n_envs.
            def _crossed(freq: int) -> bool:
                return (step // freq) > ((step - n_envs) // freq)

            if step >= ac.warmup_steps and (step - n_envs) < ac.warmup_steps:
                print(f"[MBPO] Warmup done ({step} steps). Starting model training + SAC updates.", flush=True)

            if step >= ac.warmup_steps and _crossed(ac.model_train_freq):
                train_nll, val_nll = self._train_dynamics()
                if not np.isnan(train_nll):
                    model_train_nlls.append(train_nll)
                    model_val_nlls.append(val_nll)

            if step >= ac.warmup_steps and _crossed(ac.rollout_freq):
                self._generate_rollouts(self._rollout_length(step))

            # Gate SAC updates on whichever buffer actually feeds the batch: at
            # real_ratio=1 the model buffer stays empty, so gating on it alone
            # would block all training. MixedReplayBuffer.sample falls back to
            # whichever source is non-empty.
            if step >= ac.warmup_steps and max(
                self.real_buffer.size(), self.sac.replay_buffer.size()
            ) >= ac.batch_size:
                for _ in range(ac.utd_ratio):
                    self.sac.train(gradient_steps=1, batch_size=ac.batch_size)
                    logger = self.sac.logger
                    if hasattr(logger, "name_to_value"):
                        al = logger.name_to_value.get("train/actor_loss")
                        cl = logger.name_to_value.get("train/critic_loss")
                        ec = logger.name_to_value.get("train/ent_coef")
                        if al is not None:
                            actor_losses.append(float(al))
                        if cl is not None:
                            critic_losses.append(float(cl))
                        if ec is not None:
                            ent_coefs.append(float(ec))

            if step >= ac.warmup_steps and _crossed(2_000) and max(
                self.real_buffer.size(), self.sac.replay_buffer.size()
            ) >= ac.batch_size:
                with torch.no_grad():
                    # env → normalized obs (buffer stores raw), matching training.
                    b = self.sac.replay_buffer.sample(512, env=self.env)
                    obs_t = b.observations.to(self.device)
                    act_t, logp_t = self.sac.actor.action_log_prob(obs_t)
                    qs = torch.cat(self.sac.critic(obs_t, act_t), dim=1)
                    q_values.append(float(qs.min(dim=1).values.mean()))
                    # Policy entropy H = E[-logπ]. SAC's target is -dim(A) = -2,
                    # i.e. logπ≈+2, so the soft-Q's -α·logπ term is NEGATIVE and,
                    # when reward is small, dominates Q. Track it to confirm.
                    policy_entropies.append(float((-logp_t).mean()))

            if _crossed(2_000):
                parts = [f"[MBPO] {step:>8}/{total_timesteps}"]
                log = {"train/step": step}
                if ep_rewards:
                    # Smoothed 100-episode rolling mean, matching pure SAC's
                    # WandbCallback (train/ep_rew_mean). A single episode's return
                    # is very noisy under randomize_init_pos, so the last-episode
                    # value alone is not comparable to SAC's smoothed curve.
                    if len(ep_rewards) >= 10:
                        ep_rew_mean = float(np.mean(ep_rewards[-100:]))
                        ep_len_mean = float(np.mean(ep_lengths[-100:]))
                        log["train/ep_rew_mean"] = ep_rew_mean
                        log["train/ep_len_mean"] = ep_len_mean
                        parts.append(f"ep_rew_mean={ep_rew_mean:.3f}")
                        parts.append(f"ep_len_mean={ep_len_mean:.0f}")
                    # Keep the last-episode values too (raw, unsmoothed).
                    log["train/ep_rew"] = ep_rewards[-1]
                    log["train/ep_len"] = ep_lengths[-1]
                if actor_losses:
                    al = np.mean(actor_losses[-100:])
                    cl = np.mean(critic_losses[-100:])
                    parts.append(f"actor_loss={al:.3f}")
                    parts.append(f"critic_loss={cl:.3f}")
                    log["train/actor_loss"] = al
                    log["train/critic_loss"] = cl
                if q_values:
                    qv = np.mean(q_values[-20:])
                    log["train/q_values"] = qv
                    parts.append(f"q={qv:.2f}")
                if policy_entropies:
                    pe = np.mean(policy_entropies[-20:])
                    log["train/policy_entropy"] = pe
                    parts.append(f"entropy={pe:.2f}")
                if ent_coefs:
                    ec = np.mean(ent_coefs[-100:])
                    log["train/ent_coef"] = ec
                    parts.append(f"ent_coef={ec:.3f}")
                if model_train_nlls:
                    log["model/train_nll"] = np.mean(model_train_nlls[-20:])
                    log["model/val_nll"] = np.mean(model_val_nlls[-20:])
                log["train/real_buffer"] = self.real_buffer.size()
                log["train/model_buffer"] = self.sac.replay_buffer.size()
                log["train/real_ratio"] = self.real_ratio
                # ── safety shield trigger rates ───────────────────────────
                if self.backup_policy is not None:
                    if backup_triggers:
                        sr = float(np.mean(backup_triggers[-2000:]))
                        log["shield/real_trigger_rate"] = sr
                        parts.append(f"shield_real={sr:.3f}")
                    if not np.isnan(self._rollout_shield_rate):
                        log["shield/rollout_trigger_rate"] = self._rollout_shield_rate
                        parts.append(f"shield_roll={self._rollout_shield_rate:.3f}")
                log["train/rollout_len"] = self._rollout_length(step)
                # Realized mean rollout length after uncertainty/terminal
                # truncation — if this is far below rollout_len, the model is
                # bailing out early (drift / off-distribution).
                if not np.isnan(self._rollout_len_eff):
                    log["train/rollout_len_eff"] = self._rollout_len_eff
                    parts.append(f"rollout_len_eff={self._rollout_len_eff:.2f}")
                # ── reward diagnostic ─────────────────────────────────────
                # Q ≈ r̄/(1−γ), so the sign of Q tracks the sign of the mean
                # stored (model-buffer) reward.
                buf = self.sac.replay_buffer
                n = buf.buffer_size if buf.full else buf.pos
                if n > 0:
                    mrew = float(buf.rewards[:n].mean())
                    log["train/model_rew_mean"] = mrew
                    parts.append(f"model_rew={mrew:.4f}")
                if not np.isnan(self._unc_mean):
                    log["train/model_unc"] = self._unc_mean           # mean ‖σ‖₂
                    log["train/optimism_bonus"] = self.optimism * self._unc_mean
                    parts.append(f"unc={self._unc_mean:.4f}")
                rn = self.real_buffer.buffer_size if self.real_buffer.full else self.real_buffer.pos
                if rn > 0:
                    rrew = float(self.real_buffer.rewards[:rn].mean())
                    log["train/real_rew_mean"] = rrew
                    parts.append(f"real_rew={rrew:.4f}")
                # Calibration error of the recalibrated predictive std (|E[z²/α²]−1|).
                if not np.isnan(self._calib_err):
                    log["model/calib_err"] = self._calib_err
                    parts.append(f"calib_err={self._calib_err:.3f}")
                parts.append(f"real={self.real_buffer.size()}")
                parts.append(f"model={self.sac.replay_buffer.size()}")
                parts.append(f"rollout_len={self._rollout_length(step)}")
                print("  ".join(parts), flush=True)
                if wandb_run is not None:
                    wandb_run.log(log, step=step)

    def save(self, name: str) -> None:
        self.sac.save(f"{name}_policy")
        torch.save(self.dynamics.state_dict(), f"{name}_dynamics.pt")
        print(f"Saved: {name}_policy.zip  {name}_dynamics.pt")
