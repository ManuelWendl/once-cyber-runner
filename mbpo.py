"""
Model-Based Policy Optimization (MBPO).

World model: probabilistic ensemble of MLPs (PyTorch).
  - Input: (obs, action)
  - Output: Gaussian over (delta_obs, reward)
  - Residual prediction: next_obs = obs + delta_obs

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

    Predicts ONLY ``delta_obs`` as a Gaussian (reward is supplied separately by
    an analytic reward function, mirroring the reference's external
    ``RewardModel``). The variance head outputs a standard deviation
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
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size
        self.sig_min = float(sig_min)
        self.sig_max = float(sig_max)
        self.sampling = SamplingType(sampling_type)
        # TS-∞ ensemble index, set once per rollout (ignored by the other modes).
        self.sampling_idx = 0

        out_dim = obs_dim * 2   # mean + std for delta_obs (reward is analytic)
        dims = [obs_dim + act_dim] + list(hidden) + [out_dim]
        self.net = nn.ModuleList([
            EnsembleLinear(ensemble_size, dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        # Per-dimension calibration multiplier applied to the predicted std at
        # prediction time (opax-style recalibration; 1.0 until calibrated).
        self.register_buffer("calib_alpha", torch.ones(obs_dim))

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

    def nll_loss(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL (std parameterization) summed over output dims, averaged
        over batch and ensemble. Matches the reference ``gaussian_log_likelihood``
        (constant term dropped)."""
        mean, std = self._forward(obs, act)
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

        delta = out.cpu().numpy()
        return obs + delta, unc.cpu().numpy()

    @torch.no_grad()
    def update_calibration(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor) -> float:
        """Recalibrate the predictive std on a held-out batch (PyTorch
        equivalent of the reference ``calculate_calibration_alpha``).

        Sets a per-dimension multiplier ``alpha`` so that the recalibrated
        Gaussian has unit-variance standardized residuals — alpha[d]² = E[((y−μ)/σ)²].
        Returns a scalar calibration-error proxy: |E[z²/alpha²] − 1|."""
        mean, std = self._forward(obs, act)
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

    def reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """Vectorized over the batch; returns raw reward (B,)."""
        ball_prev = self._ball_pos(obs)
        ball_next = self._ball_pos(next_obs)
        n = ball_next.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            pp = self._progress_of(ball_prev[i])
            pn = self._progress_of(ball_next[i])
            dense = (pn - pp) * self.scale if (pn >= 0 and pp >= 0) else 0.0
            goal = self._GOAL_BONUS if np.linalg.norm(ball_next[i] - self.goal_pos) < self._GOAL_THRESHOLD else 0.0
            hole = -self.hole_penalty if np.any(
                np.linalg.norm(self.holes - ball_next[i], axis=1) < self._HOLE_RADIUS
            ) else 0.0
            out[i] = dense + goal + hole
        return out


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

        self.dynamics = EnsembleDynamics(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=tuple(ac.hidden_sizes),
            ensemble_size=ac.ensemble_size,
            sig_min=float(ac.get("sig_min", 1e-3)),
            sig_max=float(ac.get("sig_max", 1e3)),
            sampling_type=ac.get("sampling_type", "TS1"),
        ).to(device)

        # Analytic reward function — the closed-form maze reward, replacing the
        # old joint reward head (mirrors the reference's separate RewardModel).
        # Pull the raw CyberRunnerEnv out from under the VecNormalize wrapper so
        # we can read its path/hole geometry; un-normalization uses the wrapper.
        raw_env = env.venv.envs[0].unwrapped
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
            # Target is delta_obs only — reward is supplied analytically.
            tgt = b.next_observations - b.observations
            loss = self.dynamics.nll_loss(b.observations, b.actions, tgt)
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.model_opt.step()
            train_losses.append(loss.item())
        # Recalibrate predictive std on a held-out validation batch.
        b = self.real_buffer.sample(min(2048, self.real_buffer.size()))
        tgt = b.next_observations - b.observations
        self._calib_err = self.dynamics.update_calibration(b.observations, b.actions, tgt)
        with torch.no_grad():
            val_loss = self.dynamics.nll_loss(b.observations, b.actions, tgt).item()
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

        all_obs, all_nobs, all_act, all_rew = [], [], [], []
        for _ in range(length):
            act, _ = self.sac.predict(obs, deterministic=False)
            nobs, unc = self.dynamics.sample(obs, act)
            # Reward from the analytic model (raw scale), not the dynamics net.
            rew = self.reward_model.reward(obs, act, nobs)
            rew = rew + self.optimism * unc   # optimism / UCB exploration bonus
            self._unc_mean = float(unc.mean())
            nobs = np.clip(nobs, -10.0, 10.0)
            rew = np.clip(rew, -10.0, 10.0)   # guard against model exploitation

            all_obs.append(obs)
            all_nobs.append(nobs)
            all_act.append(act)
            all_rew.append(rew)
            obs = nobs

        # No terminals in imagination — the model never predicts an absorbing
        # state, so there is no zero-value terminal for the policy to exploit
        # when Q dips negative.
        self._batch_add_to_sac(
            np.concatenate(all_obs), np.concatenate(all_nobs),
            np.concatenate(all_act), np.concatenate(all_rew),
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
