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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import set_random_seed
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

    def __init__(self, E: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(E, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(E, 1, out_dim))
        nn.init.trunc_normal_(self.W, std=in_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, B, in_dim) → (E, B, out_dim)
        return x @ self.W + self.b


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
    ) -> None:
        super().__init__()
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
            EnsembleLinear(ensemble_size, dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        # Per-dimension calibration multiplier applied to the predicted std at
        # prediction time (opax-style recalibration; 1.0 until calibrated).
        self.register_buffer("calib_alpha", torch.ones(self.pred_dim))

    def set_sampling_type(self, name: str) -> None:
        self.sampling.set(name)

    def set_sampling_idx(self, idx: int) -> None:
        self.sampling_idx = int(idx) % self.ensemble_size

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
    path-progress shaping + goal bonus + hole penalty. The dynamics model and
    rollouts live in VecNormalize's NORMALIZED observation space, so the reward
    first un-normalizes obs to recover raw board ball positions (dims [2:4]),
    then evaluates the same progress/goal/hole terms the env uses, returning the
    same RAW reward the env stores (norm_reward is disabled for MBPO).
    """

    def __init__(self, vecnorm, raw_env) -> None:
        from envs.cyberrunner import (
            compute_path_progress, GOAL_BONUS, GOAL_THRESHOLD, HOLE_RADIUS,
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
        dense = np.where(valid, (pn - prev_progress) * self.scale, 0.0)
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
        ).to(device)

        # Analytic reward function — the closed-form maze reward, replacing the
        # old joint reward head (mirrors the reference's separate RewardModel).
        # Pull the raw CyberRunnerEnv out from under the VecNormalize wrapper so
        # we can read its path/hole geometry; un-normalization uses the wrapper.
        # (``raw_env`` was resolved above to read the frame-stacking layout.)
        self.reward_model = AnalyticCyberRunnerReward(env, raw_env)

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

        # Cumulative, absolute episode-outcome counters over real (executed)
        # transitions — the baseline safety/performance metric. holes_total is a
        # monotonic count of marble-into-hole failures (NOT a ratio); goals_total
        # counts real goal reaches. Both are read from the env's
        # info["termination_reason"] on terminal steps and logged to W&B.
        self._holes_total = 0
        self._goals_total = 0

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
        # We manage all observations in already-normalized space (VecNormalize output).
        # Nulling this prevents SB3 from double-normalizing obs during predict() and train().
        self.sac._vec_normalize_env = None
        # SB3 stores raw rewards and renormalizes at sample time using current reward_rms.
        # We store already-normalized rewards and sample them as-is (_vec_normalize_env=None),
        # so reward_rms drift over training would make early stored rewards inconsistent with
        # later ones (same raw reward → smaller normalized value as reward_rms.var grows).
        # Disabling norm_reward makes stored rewards raw and consistent across the buffer lifetime.
        if hasattr(env, "norm_reward"):
            env.norm_reward = False

        self._ac = ac

    def predict(self, obs: np.ndarray, prev_obs: np.ndarray | None = None, deterministic: bool = True):
        """Plain SAC predict. ``prev_obs`` is accepted and ignored so existing
        eval callers passing ``(obs, prev_obs)`` keep working."""
        return self.sac.predict(obs, deterministic=deterministic)

    # Backwards-compatible alias (the safety shield has been removed).
    shielded_predict = predict

    # ── world model training ──────────────────────────────────────────────

    def _train_dynamics(self) -> tuple[float, float]:
        ac = self._ac
        if self.real_buffer.size() < ac.batch_size:
            return float("nan"), float("nan")
        self.dynamics.train()
        train_losses = []
        for _ in range(ac.model_train_epochs):
            b = self.real_buffer.sample(min(ac.model_batch_size, self.real_buffer.size()))
            # Target (base-obs delta) is derived from next_obs inside the model;
            # reward is supplied analytically.
            loss = self.dynamics.nll_loss(b.observations, b.actions, b.next_observations)
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.model_opt.step()
            train_losses.append(loss.item())
        # Recalibrate predictive std on a held-out validation batch.
        b = self.real_buffer.sample(min(2048, self.real_buffer.size()))
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
        b = self.real_buffer.sample(ac.rollout_batch_size)
        obs = b.observations.cpu().numpy()   # already in normalized obs space

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
        for _ in range(length):
            act, _ = self.sac.predict(obs, deterministic=False)
            nobs, unc = self.dynamics.sample(obs, act)
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
            nobs = np.clip(nobs, -10.0, 10.0)
            rew = np.clip(rew, -10.0, 10.0)   # guard against model exploitation

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

        self._batch_add_to_sac(
            np.concatenate(all_obs), np.concatenate(all_nobs),
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
        obs = self.env.reset()
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

        while step < total_timesteps:
            # Random exploration during warmup, SAC policy afterwards
            if step < ac.warmup_steps:
                act = np.stack([self.env.action_space.sample() for _ in range(n_envs)])
            else:
                act, _ = self.sac.predict(obs, deterministic=False)

            nobs, rew, done, infos = self.env.step(act)

            # Store each parallel env's transition. Real executed transitions go
            # to real_buffer only — they reach SAC via the real_ratio sampler.
            for i in range(n_envs):
                terminal = bool(done[i]) and not infos[i].get("TimeLimit.truncated", False)
                t = np.array([terminal])
                # Real buffer always gets the executed action (clean ground truth)
                self.real_buffer.add(
                    obs[i:i+1], nobs[i:i+1], act[i:i+1],
                    rew[i:i+1], t, [infos[i]],
                )
                ep_info = infos[i].get("episode")
                if ep_info is not None:
                    ep_rewards.append(float(ep_info["r"]))
                    ep_lengths.append(int(ep_info["l"]))
                # Cumulative absolute outcome counts on real terminal steps.
                if bool(done[i]):
                    reason = infos[i].get("termination_reason")
                    if reason == "hole":
                        self._holes_total += 1
                    elif reason == "goal":
                        self._goals_total += 1

            obs = nobs
            step += n_envs

            if step == ac.warmup_steps:
                print(f"[MBPO] Warmup done ({step} steps). Starting model training + SAC updates.", flush=True)

            if step % ac.model_train_freq == 0 and step >= ac.warmup_steps:
                train_nll, val_nll = self._train_dynamics()
                if not np.isnan(train_nll):
                    model_train_nlls.append(train_nll)
                    model_val_nlls.append(val_nll)

            if step >= ac.warmup_steps and step % ac.rollout_freq == 0:
                self._generate_rollouts(self._rollout_length(step))

            if step >= ac.warmup_steps and self.sac.replay_buffer.size() >= ac.batch_size:
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

            if step >= ac.warmup_steps and step % 2_000 == 0 and self.sac.replay_buffer.size() >= ac.batch_size:
                with torch.no_grad():
                    b = self.sac.replay_buffer.sample(512)
                    obs_t = b.observations.to(self.device)
                    act_t, logp_t = self.sac.actor.action_log_prob(obs_t)
                    qs = torch.cat(self.sac.critic(obs_t, act_t), dim=1)
                    q_values.append(float(qs.min(dim=1).values.mean()))
                    # Policy entropy H = E[-logπ]. SAC's target is -dim(A) = -2,
                    # i.e. logπ≈+2, so the soft-Q's -α·logπ term is NEGATIVE and,
                    # when reward is small, dominates Q. Track it to confirm.
                    policy_entropies.append(float((-logp_t).mean()))

            if step % 2_000 == 0:
                parts = [f"[MBPO] {step:>8}/{total_timesteps}"]
                log = {"train/step": step}
                if ep_rewards:
                    ep_rew = ep_rewards[-1]
                    ep_len = ep_lengths[-1]
                    parts.append(f"ep_rew={ep_rew:.3f}")
                    parts.append(f"ep_len={ep_len:.0f}")
                    log["train/ep_rew"] = ep_rew
                    log["train/ep_len"] = ep_len
                # Cumulative absolute outcome counts (monotonic, not ratios).
                log["train/holes_total"] = self._holes_total
                log["train/goals_total"] = self._goals_total
                parts.append(f"holes_total={self._holes_total}")
                parts.append(f"goals_total={self._goals_total}")
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
