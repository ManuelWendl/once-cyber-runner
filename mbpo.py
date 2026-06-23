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
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.type_aliases import ReplayBufferSamples
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

    def __init__(self, E: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(E, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(E, 1, out_dim))
        nn.init.trunc_normal_(self.W, std=in_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, B, in_dim) → (E, B, out_dim)
        return x @ self.W + self.b


class EnsembleDynamics(nn.Module):
    """
    Probabilistic ensemble.

    Predicts (delta_obs, reward) jointly as a Gaussian.
    Operates in the same observation/reward space as the data it's trained on
    (i.e. the normalized space when VecNormalize is in use).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...],
        ensemble_size: int,
        num_elites: int,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size
        self.num_elites = num_elites
        self.elite_idxs: list[int] = list(range(num_elites))

        out_dim = (obs_dim + 1) * 2   # mean + logvar for [delta_obs, reward]
        dims = [obs_dim + act_dim] + list(hidden) + [out_dim]
        self.net = nn.ModuleList([
            EnsembleLinear(ensemble_size, dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        # Learnable log_var bounds (prevents variance explosion / collapse)
        self.max_logvar = nn.Parameter(0.5 * torch.ones(obs_dim + 1))
        self.min_logvar = nn.Parameter(-10.0 * torch.ones(obs_dim + 1))

    def _forward(self, obs: torch.Tensor, act: torch.Tensor):
        """obs/act: (B, d) → mean, logvar each (E, B, obs_dim+1)."""
        x = torch.cat([obs, act], -1).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for layer in self.net[:-1]:
            x = F.silu(layer(x))
        x = self.net[-1](x)
        mean, logvar = x.chunk(2, -1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll_loss(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL summed over output dims, averaged over batch and ensemble."""
        mean, logvar = self._forward(obs, act)
        tgt = target.unsqueeze(0).expand_as(mean)
        nll = (logvar + (mean - tgt).pow(2) * torch.exp(-logvar)).sum(-1).mean()
        # Keep log_var bounds tight
        reg = 1e-2 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll + reg

    @torch.no_grad()
    def sample(self, obs: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-step prediction for a batch.
        Randomly assigns each sample to one elite model (MBPO-style).
        Returns: next_obs (B, obs_dim), reward (B,), uncertainty (B,)
            uncertainty = ‖σ‖₂, the L2 norm (over the obs+reward output dims) of
            the predicted standard deviation of the sampled elite — used as an
            optimism / exploration bonus on the rollout reward.
        """
        dev = next(self.parameters()).device
        obs_t = torch.from_numpy(obs).float().to(dev)
        act_t = torch.from_numpy(act).float().to(dev)
        mean, logvar = self._forward(obs_t, act_t)   # (E, B, D)
        std = (0.5 * logvar).exp()

        B = obs.shape[0]
        e_idx = torch.from_numpy(np.random.choice(self.elite_idxs, B)).long()
        b_idx = torch.arange(B, device=dev)
        m = mean[e_idx, b_idx]    # (B, obs_dim+1)
        s = std[e_idx, b_idx]
        out = m + s * torch.randn_like(m)

        delta = out[:, :-1].cpu().numpy()
        rew   = out[:, -1].cpu().numpy()
        unc   = s.norm(dim=1).cpu().numpy()   # ‖σ‖₂ over output dims, (B,)
        return obs + delta, rew, unc

    @torch.no_grad()
    def update_elites(self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor) -> None:
        """Rank ensemble members by validation NLL; keep the best num_elites."""
        mean, logvar = self._forward(obs, act)
        tgt = target.unsqueeze(0).expand_as(mean)
        per_model = (logvar + (mean - tgt).pow(2) * torch.exp(-logvar)).mean([1, 2])
        self.elite_idxs = torch.argsort(per_model)[:self.num_elites].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MBPO trainer
# ─────────────────────────────────────────────────────────────────────────────

class MBPOTrainer:

    def __init__(self, env, cfg: DictConfig, device: str = "cpu") -> None:
        self.env = env
        self.device = device
        ac = cfg.algo

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        self.dynamics = EnsembleDynamics(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=tuple(ac.hidden_sizes),
            ensemble_size=ac.ensemble_size,
            num_elites=ac.num_elites,
        ).to(device)

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

        # ── optional backup safety filter ─────────────────────────────────
        # Backup policy guards real env collection: if the world model predicts
        # that the SAC action leads to an unrecoverable next state
        # (V_backup(s') < safety_threshold), the backup policy acts instead.
        self.backup_policy = None
        self.backup_vecnorm = None
        self.safety_threshold = float(ac.get("safety_threshold", 0.3))

        bp_path = ac.get("backup_policy_path", "ppo_backup.zip")
        bv_path = ac.get("backup_vecnorm_path", "ppo_backup_vecnormalize.pkl")

        wandb_id = ac.get("backup_wandb_id", None)
        if wandb_id:
            import wandb as wb
            from pathlib import Path
            project = ac.get("backup_wandb_project", "cyberrunner")
            api = wb.Api()
            run = api.run(f"{project}/{wandb_id}")
            artifact = next((a for a in run.logged_artifacts() if a.type == "model"), None)
            if artifact is None:
                raise RuntimeError(f"[MBPO] No model artifact found for wandb run {project}/{wandb_id}")
            root = Path(artifact.download())
            bp_path = str(next(root.glob("*.zip")))
            bv_path = str(next(root.glob("*.pkl")))
            print(f"[MBPO] Downloaded backup artifact: {bp_path}")

        if os.path.exists(bp_path) and os.path.exists(bv_path):
            from gymnasium.wrappers import FlattenObservation
            from envs.cyberrunner import CyberRunnerEnv
            model_cls = PPO if "ppo" in os.path.basename(bp_path).lower() else SAC
            self.backup_policy = model_cls.load(bp_path, device=device)
            dummy = DummyVecEnv([lambda: FlattenObservation(
                CyberRunnerEnv(include_vision=False, backup_mode=True)
            )])
            self.backup_vecnorm = VecNormalize.load(bv_path, dummy)
            self.backup_vecnorm.training = False
            self.backup_vecnorm.norm_reward = False
            print(f"[MBPO] Backup safety filter loaded ({model_cls.__name__}): {bp_path}  threshold={self.safety_threshold}")
        else:
            print(f"[MBPO] No backup policy found at '{bp_path}' — safety filter disabled.")

    # ── backup obs conversion & safety check ─────────────────────────────

    def _to_backup_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert behavioral obs → backup obs.

        Both policies now share the identical physical layout
        [joint(2), ball_pos(2), ball_vel(2)] (6-d), so this is purely a
        renormalization: de-normalize with the behavioral VecNormalize stats,
        then re-normalize with the backup's. In imagination the velocity is the
        world model's prediction for the state.
        """
        rms = self.env.obs_rms
        raw = obs * np.sqrt(rms.var + 1e-8) + rms.mean   # (B, 6) physical units
        b_rms = self.backup_vecnorm.obs_rms
        return np.clip(
            (raw - b_rms.mean) / np.sqrt(b_rms.var + 1e-8), -10.0, 10.0
        ).astype(np.float32)

    def _recovery_values(self, backup_obs: np.ndarray, act: np.ndarray | None = None) -> np.ndarray:
        """Query backup policy safety value V(s) = P(recover | s). Returns (B,).
        PPO: V(s) directly.
        SAC: min(Q1,Q2)(s, a) where a is the action about to be executed."""
        obs_t = torch.FloatTensor(backup_obs).to(self.device)
        with torch.no_grad():
            if isinstance(self.backup_policy, PPO):
                values = self.backup_policy.policy.predict_values(obs_t)
            else:
                act_t = torch.FloatTensor(act).to(self.device)
                q1, q2 = self.backup_policy.policy.critic(obs_t, act_t)
                values = torch.min(q1, q2)
        return values.cpu().numpy().flatten()

    def shielded_predict(self, obs: np.ndarray, prev_obs: np.ndarray | None = None, deterministic: bool = True):
        """SAC predict with backup safety filter applied (mirrors real-env shielding in learn()).

        ``prev_obs`` is deprecated and ignored — velocity now comes from the
        observation itself. The parameter is retained so existing callers
        (e.g. eval scripts passing ``(obs, prev_obs)``) keep working.
        """
        act, state = self.sac.predict(obs, deterministic=deterministic)
        if self.backup_policy is not None:
            if isinstance(self.backup_policy, SAC):
                backup_obs = self._to_backup_obs(obs)
            else:
                nobs_pred, _, _ = self.dynamics.sample(obs, act)
                nobs_pred = np.clip(nobs_pred, -10.0, 10.0)
                backup_obs = self._to_backup_obs(nobs_pred)
            values = self._recovery_values(backup_obs, act=act)
            unsafe = values < self.safety_threshold
            if unsafe.any():
                cur_backup_obs = self._to_backup_obs(obs)
                backup_act, _ = self.backup_policy.predict(cur_backup_obs[unsafe], deterministic=True)
                act[unsafe] = backup_act
        return act, state

    # ── world model training ──────────────────────────────────────────────

    def _train_dynamics(self) -> tuple[float, float]:
        ac = self._ac
        if self.real_buffer.size() < ac.batch_size:
            return float("nan"), float("nan")
        self.dynamics.train()
        train_losses = []
        for _ in range(ac.model_train_epochs):
            b = self.real_buffer.sample(min(ac.model_batch_size, self.real_buffer.size()))
            tgt = torch.cat([b.next_observations - b.observations, b.rewards], -1)
            loss = self.dynamics.nll_loss(b.observations, b.actions, tgt)
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.model_opt.step()
            train_losses.append(loss.item())
        # Select elites on a held-out validation batch
        b = self.real_buffer.sample(min(2048, self.real_buffer.size()))
        tgt = torch.cat([b.next_observations - b.observations, b.rewards], -1)
        self.dynamics.update_elites(b.observations, b.actions, tgt)
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

        all_obs, all_nobs, all_act, all_rew = [], [], [], []
        for _ in range(length):
            act, _ = self.sac.predict(obs, deterministic=False)
            nobs, rew, unc = self.dynamics.sample(obs, act)
            rew = rew + self.optimism * unc   # optimism / UCB exploration bonus
            self._unc_mean = float(unc.mean())
            nobs = np.clip(nobs, -10.0, 10.0)
            rew = np.clip(rew, -10.0, 10.0)   # guard against model exploitation

            if self.backup_policy is not None:
                # Shield decision, mirroring the real-env filter in learn():
                #   SAC backup → Q_backup(s, a_sac);  PPO backup → V_backup(s'_sac).
                # cur_backup_obs is the current-state backup obs; the SAC path
                # reuses it for both the safety value and the backup action.
                if isinstance(self.backup_policy, SAC):
                    cur_backup_obs = self._to_backup_obs(obs)
                    values = self._recovery_values(cur_backup_obs, act=act)
                else:
                    cur_backup_obs = None
                    values = self._recovery_values(self._to_backup_obs(nobs), act=act)
                unsafe = values < self.safety_threshold
                if unsafe.any():
                    # In reality the unsafe SAC action is NOT executed — the backup
                    # takes over and the ball ends up in the backup's resulting
                    # state with the backup action's reward. Mirror that: store the
                    # SAC action (so the critic learns the consequence of proposing
                    # it), but make the next state AND reward the model's prediction
                    # for the BACKUP action. This keeps the dynamics/reward query
                    # on-distribution — backup actions are what the model was trained
                    # on in unsafe states.
                    if cur_backup_obs is None:
                        cur_backup_obs = self._to_backup_obs(obs)
                    backup_act, _ = self.backup_policy.predict(
                        cur_backup_obs[unsafe], deterministic=True
                    )
                    nobs_backup, rew_backup, unc_backup = self.dynamics.sample(obs[unsafe], backup_act)
                    nobs[unsafe] = np.clip(nobs_backup, -10.0, 10.0)
                    rew[unsafe] = np.clip(rew_backup + self.optimism * unc_backup, -10.0, 10.0)

            all_obs.append(obs)
            all_nobs.append(nobs)
            all_act.append(act)   # stored action is always the SAC action
            all_rew.append(rew)
            obs = nobs

        # No terminals in imagination: the shield redirects would-be-unsafe
        # (e.g. hole-bound) transitions to the backup's recovery state, so the
        # ball never reaches an absorbing state — and with no zero-value terminal
        # there is no "escape" for the policy to exploit when Q dips negative.
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
        backup_triggered: list[float] = []
        recovery_values: list[float] = []
        model_train_nlls: list[float] = []
        model_val_nlls: list[float] = []

        while step < total_timesteps:
            # Random exploration during warmup, SAC policy afterwards
            if step < ac.warmup_steps:
                act = np.stack([self.env.action_space.sample() for _ in range(n_envs)])
            else:
                act, _ = self.sac.predict(obs, deterministic=False)

            # ── backup safety filter on real env actions ───────────────────
            # After warmup, use the world model to predict the next state for
            # each env. If the backup policy's value V(s') < safety_threshold,
            # the next state is predicted to be unrecoverable — replace SAC's
            # action with the backup policy's action for that env.
            unsafe = np.zeros(n_envs, dtype=bool)

            if self.backup_policy is not None and (
                    isinstance(self.backup_policy, SAC)
                    or self.real_buffer.size() >= ac.batch_size):
                if isinstance(self.backup_policy, SAC):
                    backup_obs = self._to_backup_obs(obs)
                else:
                    nobs_pred, _, _ = self.dynamics.sample(obs, act)
                    nobs_pred = np.clip(nobs_pred, -10.0, 10.0)
                    backup_obs = self._to_backup_obs(nobs_pred)
                values = self._recovery_values(backup_obs, act=act)
                unsafe = values < self.safety_threshold
                backup_triggered.append(float(unsafe.mean()))
                recovery_values.append(float(values.mean()))
                if unsafe.any():
                    cur_backup_obs = self._to_backup_obs(obs)
                    backup_act, _ = self.backup_policy.predict(
                        cur_backup_obs[unsafe], deterministic=True
                    )
                    act[unsafe] = backup_act

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
                if backup_triggered:
                    bt = np.mean(backup_triggered[-100:])
                    log["train/backup_triggered"] = bt
                    # Fraction of real steps where the shield overwrote the SAC action.
                    # High → the model trains on shielded behavior but rollouts imagine
                    # raw SAC → train/imagine action mismatch drives model_rew off.
                    parts.append(f"shield={bt:.3f}")
                if recovery_values:
                    log["train/recovery_value"] = np.mean(recovery_values[-100:])
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
                # ── isolate model reward bias from off-distribution drift ──
                # Predict reward on a REAL (obs, act) batch — the model's own
                # training distribution — and compare to the true stored reward.
                # pred ≪ true  → model under-prediction bias (heteroscedastic NLL
                #                on a right-skewed reward). On-distribution problem.
                # pred ≈ true  → the negative rollout reward is off-distribution /
                #                policy drift, not a model fitting bug.
                if self.real_buffer.size() >= self._ac.batch_size:
                    with torch.no_grad():
                        rb = self.real_buffer.sample(512)
                        mean, _ = self.dynamics._forward(rb.observations, rb.actions)
                        pred_rew = float(mean[:, :, -1].mean())   # mean over ensemble & batch
                        true_rew = float(rb.rewards.mean())
                    log["model/pred_rew_on_real"] = pred_rew
                    log["model/true_rew_on_real"] = true_rew
                    log["model/rew_bias"] = pred_rew - true_rew
                    parts.append(f"pred_on_real={pred_rew:.4f}(true={true_rew:.4f})")
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
