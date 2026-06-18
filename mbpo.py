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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from omegaconf import DictConfig


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
    def sample(self, obs: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        One-step prediction for a batch.
        Randomly assigns each sample to one elite model (MBPO-style).
        Returns: next_obs (B, obs_dim), reward (B,)
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
        return obs + delta, rew

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
        )
        # SAC.train() requires _logger; silence it — MBPO has its own logging
        self.sac.set_logger(configure_logger(folder=None, format_strings=[]))

        # Real experience buffer (separate from SAC's model buffer)
        self.real_buffer = ReplayBuffer(
            buffer_size=ac.real_buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=1,
            handle_timeout_termination=True,
        )
        self._ac = ac

        # ── optional backup safety filter ─────────────────────────────────
        # Backup policy (PPO) guards real env collection: if the world model
        # predicts that the SAC action leads to an unrecoverable next state
        # (V_backup(s') < safety_threshold), the backup policy acts instead.
        self.backup_policy = None
        self.backup_vecnorm = None
        self.safety_threshold = float(ac.get("safety_threshold", 0.3))

        bp_path = ac.get("backup_policy_path", "ppo_backup.zip")
        bv_path = ac.get("backup_vecnorm_path", "ppo_backup_vecnormalize.pkl")
        if os.path.exists(bp_path) and os.path.exists(bv_path):
            from gymnasium.wrappers import FlattenObservation
            from envs.cyberrunner import CyberRunnerEnv
            self.backup_policy = PPO.load(bp_path, device=device)
            dummy = DummyVecEnv([lambda: FlattenObservation(
                CyberRunnerEnv(include_vision=False, backup_mode=True)
            )])
            self.backup_vecnorm = VecNormalize.load(bv_path, dummy)
            self.backup_vecnorm.training = False
            self.backup_vecnorm.norm_reward = False
            print(f"[MBPO] Backup safety filter loaded: {bp_path}  threshold={self.safety_threshold}")
        else:
            print(f"[MBPO] No backup policy found at '{bp_path}' — safety filter disabled.")

    # ── backup obs conversion & safety check ─────────────────────────────

    def _to_backup_obs(self, obs: np.ndarray, prev_obs: np.ndarray) -> np.ndarray:
        """
        Convert MBPO-normalised maze obs → backup-policy-normalised obs.

        MBPO obs layout  (13-d): [joint(2), ball_pos(2), vec_closest(2),
                                   vec_next_wp(2), vec_next_next_wp(2), checkpoint(3)]
        Backup obs layout(13-d): [joint(2), ball_pos(2), vec_closest(2),
                                   ball_vel(2), ball_speed(1), 0(1),  checkpoint(3)]
        """
        rms = self.env.obs_rms
        std = np.sqrt(rms.var + 1e-8)

        raw      = obs      * std + rms.mean   # (B, 13)
        prev_raw = prev_obs * std + rms.mean

        dt = 0.00166666 * 10  # TIMESTEP * FRAME_SKIP = 1/60 s

        joint_pos     = raw[:, 0:2]
        ball_pos      = raw[:, 2:4]
        vec_closest   = raw[:, 4:6]
        checkpoint    = raw[:, 10:13]

        prev_ball_pos = prev_raw[:, 2:4]
        ball_vel      = (ball_pos - prev_ball_pos) / dt          # (B, 2)
        ball_speed    = np.linalg.norm(ball_vel, axis=1, keepdims=True)  # (B, 1)
        zeros         = np.zeros((len(obs), 1), dtype=np.float32)

        backup_raw = np.concatenate(
            [joint_pos, ball_pos, vec_closest, ball_vel, ball_speed, zeros, checkpoint], axis=1
        ).astype(np.float32)   # (B, 13)

        b_rms = self.backup_vecnorm.obs_rms
        backup_norm = np.clip(
            (backup_raw - b_rms.mean) / np.sqrt(b_rms.var + 1e-8), -10.0, 10.0
        )
        return backup_norm

    def _recovery_values(self, backup_obs: np.ndarray) -> np.ndarray:
        """Query backup policy value function V(s) = P(recover | s). Returns (B,)."""
        obs_t = torch.FloatTensor(backup_obs).to(self.device)
        with torch.no_grad():
            values = self.backup_policy.policy.predict_values(obs_t)
        return values.cpu().numpy().flatten()

    # ── world model training ──────────────────────────────────────────────

    def _train_dynamics(self) -> None:
        ac = self._ac
        if self.real_buffer.size() < ac.batch_size:
            return
        self.dynamics.train()
        for _ in range(ac.model_train_epochs):
            b = self.real_buffer.sample(min(ac.model_batch_size, self.real_buffer.size()))
            tgt = torch.cat([b.next_observations - b.observations, b.rewards], -1)
            loss = self.dynamics.nll_loss(b.observations, b.actions, tgt)
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
            self.model_opt.step()
        # Select elites on a held-out validation batch
        b = self.real_buffer.sample(min(2048, self.real_buffer.size()))
        tgt = torch.cat([b.next_observations - b.observations, b.rewards], -1)
        self.dynamics.update_elites(b.observations, b.actions, tgt)
        self.dynamics.eval()

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
            nobs, rew = self.dynamics.sample(obs, act)
            nobs = np.clip(nobs, -10.0, 10.0)
            rew = np.clip(rew, -10.0, 10.0)   # guard against model exploitation
            all_obs.append(obs); all_nobs.append(nobs)
            all_act.append(act); all_rew.append(rew)
            obs = nobs

        self._batch_add_to_sac(
            np.concatenate(all_obs), np.concatenate(all_nobs),
            np.concatenate(all_act), np.concatenate(all_rew),
        )

    def _batch_add_to_sac(self, obs: np.ndarray, nobs: np.ndarray,
                           act: np.ndarray, rew: np.ndarray) -> None:
        """Write a batch of transitions directly into SAC's replay buffer arrays."""
        buf = self.sac.replay_buffer
        n = len(obs)
        done = np.zeros((n, 1), dtype=np.float32)
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

    def learn(self, total_timesteps: int) -> None:
        ac = self._ac
        obs = self.env.reset()
        n_envs = self.env.num_envs
        step = 0

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        prev_obs = obs.copy()

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
            if (self.backup_policy is not None
                    and step >= ac.warmup_steps
                    and self.real_buffer.size() >= ac.batch_size):
                nobs_pred, _ = self.dynamics.sample(obs, act)
                nobs_pred = np.clip(nobs_pred, -10.0, 10.0)
                backup_obs = self._to_backup_obs(nobs_pred, obs)
                values = self._recovery_values(backup_obs)
                unsafe = values < self.safety_threshold
                if unsafe.any():
                    cur_backup_obs = self._to_backup_obs(obs, prev_obs)
                    backup_act, _ = self.backup_policy.predict(
                        cur_backup_obs[unsafe], deterministic=True
                    )
                    act[unsafe] = backup_act

            nobs, rew, done, infos = self.env.step(act)

            # Store each parallel env's transition in both buffers
            for i in range(n_envs):
                terminal = bool(done[i]) and not infos[i].get("TimeLimit.truncated", False)
                t = np.array([terminal])
                self.real_buffer.add(
                    obs[i:i+1], nobs[i:i+1], act[i:i+1],
                    rew[i:i+1], t, [infos[i]],
                )
                # Also add real transitions to SAC's buffer to anchor Q-values
                self.sac.replay_buffer.add(
                    obs[i:i+1], nobs[i:i+1], act[i:i+1],
                    rew[i:i+1], t, [infos[i]],
                )
                ep_info = infos[i].get("episode")
                if ep_info is not None:
                    ep_rewards.append(float(ep_info["r"]))
                    ep_lengths.append(int(ep_info["l"]))

            prev_obs = obs.copy()
            obs = nobs
            step += n_envs

            if step == ac.warmup_steps:
                print(f"[MBPO] Warmup done ({step} steps). Starting model training + SAC updates.", flush=True)

            if step % ac.model_train_freq == 0 and step >= ac.warmup_steps:
                self._train_dynamics()

            if step >= ac.warmup_steps and step % ac.rollout_freq == 0:
                self._generate_rollouts(self._rollout_length(step))

            if step >= ac.warmup_steps and self.sac.replay_buffer.size() >= ac.batch_size:
                for _ in range(ac.utd_ratio):
                    self.sac.train(gradient_steps=1, batch_size=ac.batch_size)
                    logger = self.sac.logger
                    if hasattr(logger, "name_to_value"):
                        al = logger.name_to_value.get("train/actor_loss")
                        cl = logger.name_to_value.get("train/critic_loss")
                        if al is not None:
                            actor_losses.append(float(al))
                        if cl is not None:
                            critic_losses.append(float(cl))

            if step % 1_000 == 0:
                parts = [f"[MBPO] {step:>8}/{total_timesteps}"]
                if ep_rewards:
                    parts.append(f"ep_rew={np.mean(ep_rewards[-100:]):.3f}")
                    parts.append(f"ep_len={np.mean(ep_lengths[-100:]):.0f}")
                if actor_losses:
                    parts.append(f"actor_loss={np.mean(actor_losses[-100:]):.3f}")
                    parts.append(f"critic_loss={np.mean(critic_losses[-100:]):.3f}")
                parts.append(f"real={self.real_buffer.size()}")
                parts.append(f"model={self.sac.replay_buffer.size()}")
                parts.append(f"rollout_len={self._rollout_length(step)}")
                print("  ".join(parts), flush=True)

    def save(self, name: str) -> None:
        self.sac.save(f"{name}_policy")
        torch.save(self.dynamics.state_dict(), f"{name}_dynamics.pt")
        print(f"Saved: {name}_policy.zip  {name}_dynamics.pt")
