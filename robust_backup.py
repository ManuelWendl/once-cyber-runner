"""
Pessimistic (robustified) recovery-Q update for a directly-trained SAC backup
policy — a domain-randomization counterpart to
``mbpo.MBPOTrainer._robust_recovery_q_update``.

That MBPO version re-fits an ALREADY-TRAINED backup's critic using a learned
world-model ensemble's epistemic disagreement (``‖σ_epistemic‖₂``) as the
proxy for model discrepancy ``rho_t(s,a)`` in Prop. robust_bound:

    Q_{R_eps,t}^{pi_r}(s,a) = E_{pi_r,mu}[ sum_h  1{s_h in R_eps}
                                - min(2L·(rho_t(s_h,a_h) - m)_+, 1) ]

When training the backup policy ITSELF (train.py, algo=sac + env.prior_mode),
there is no learned dynamics ensemble yet — but ``envs.cyberrunner`` supports
domain-randomizing the marble mass (``randomize_marble_mass``), which gives an
analogous ensemble for free: the TRUE simulator run under several different
(randomly drawn, but held fixed for the run) marble masses. This module
re-fits the backup's own critic while it trains, using the disagreement among
those masses' one-step-ahead physical predictions in place of ``‖σ_epistemic‖₂``.

Everything else mirrors the MBPO version: same Bellman recursion with a
per-step "reward" ``1{recovered} - penalty(rho)``, absorption on
``done`` (prior_mode terminates exactly at recovery/hole/timeout, never
truncates — see ``CyberRunnerEnv._check_termination``), and the same
max-intersection-with-the-critic's-own-belief pessimistic clip (Q is a
recovery PROBABILITY, higher is better, so revision is only ever upward).
It is simpler than the MBPO version in one respect: since the policy being
fit here IS the policy generating the data, there is no cross-normalization
between a "learner" and a separately-normalized "backup" — the SAC model's
own (already VecNormalize-normalized) obs are used directly.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update


_SampledBatch = namedtuple(
    "_SampledBatch", ["obs", "action", "next_obs", "reward", "done", "sim_states"]
)


class SimTransitionBuffer:
    """Fixed-capacity ring buffer of (obs, action, next_obs, reward, done,
    sim_state) transitions, where ``sim_state = (qpos, qvel, act, ctrl)`` is
    the full MuJoCo state snapshot from ``CyberRunnerEnv.get_sim_state()`` —
    enough to replay physics from exactly that state under a different marble
    mass. A plain ring buffer rather than SB3's ``ReplayBuffer`` because it
    needs to carry the raw simulator snapshot alongside the usual transition.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, nq: int, nv: int, na: int, nu: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.qpos = np.zeros((self.capacity, nq), dtype=np.float64)
        self.qvel = np.zeros((self.capacity, nv), dtype=np.float64)
        self.act = np.zeros((self.capacity, na), dtype=np.float64)
        self.ctrl = np.zeros((self.capacity, nu), dtype=np.float64)
        self._ptr = 0
        self.size = 0

    def add(self, obs, action, next_obs, reward: float, done: bool, sim_state) -> None:
        i = self._ptr
        qpos, qvel, act, ctrl = sim_state
        self.obs[i] = obs
        self.action[i] = action
        self.next_obs[i] = next_obs
        self.reward[i] = reward
        self.done[i] = float(done)
        self.qpos[i] = qpos
        self.qvel[i] = qvel
        self.act[i] = act
        self.ctrl[i] = ctrl
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> _SampledBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        sim_states = [(self.qpos[j], self.qvel[j], self.act[j], self.ctrl[j]) for j in idx]
        return _SampledBatch(
            obs=self.obs[idx],
            action=self.action[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[idx],
            done=self.done[idx],
            sim_states=sim_states,
        )


class RobustBackupQCallback(BaseCallback):
    """SB3 callback: periodically re-fits the SAC model's OWN critic (never
    its actor) toward the pessimistic recovery-Q lower bound, using
    domain-randomized-mass simulator disagreement as ``rho_t``.

    Requires the training env to be ``DummyVecEnv``-based (direct, same-
    process access to each sub-env is used to snapshot/replay simulator
    state) — the default for ``make_vec_env`` in this repo's ``train.py``.
    """

    def __init__(
        self,
        env_layout: str,
        marble_mass_low: float,
        marble_mass_high: float,
        ensemble_size: int = 5,
        L: float = 1.0,
        margin: float = 0.0,
        update_freq: int = 2000,
        gradient_steps: int = 50,
        batch_size: int = 256,
        buffer_size: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._layout = env_layout
        self._mass_low = float(marble_mass_low)
        self._mass_high = float(marble_mass_high)
        self._K = max(2, int(ensemble_size))
        self._L = float(L)
        self._margin = float(margin)
        self._update_freq = int(update_freq)
        self._n_grad_steps = int(gradient_steps)
        self._batch_size = int(batch_size)
        self._buffer_size = int(buffer_size)
        self._last_update_step = 0
        self._scratch_env = None
        self._raw_envs = None
        self._pre_state = None
        self._buf: SimTransitionBuffer | None = None
        self._masses = None

    def _on_training_start(self) -> None:
        from envs.cyberrunner import CyberRunnerEnv

        # A single scratch env, reused sequentially for every (mass, batch
        # entry) combination — it is never returned to the caller and never
        # touches the live training envs, so the real rollout is unaffected.
        self._scratch_env = CyberRunnerEnv(layout=self._layout)
        self._masses = np.linspace(self._mass_low, self._mass_high, self._K)

        venv = self.training_env
        while hasattr(venv, "venv"):
            venv = venv.venv
        if not hasattr(venv, "envs"):
            raise RuntimeError(
                "RobustBackupQCallback requires a DummyVecEnv-based training env "
                "(needs direct, same-process access to each sub-env's MuJoCo state)."
            )
        self._raw_envs = [e.unwrapped for e in venv.envs]

        obs_dim = int(np.prod(self.training_env.observation_space.shape))
        act_dim = int(np.prod(self.training_env.action_space.shape))
        m = self._scratch_env.model
        self._buf = SimTransitionBuffer(
            self._buffer_size, obs_dim, act_dim, m.nq, m.nv, m.na, m.nu
        )
        # Seeded from the state right after SB3's initial env.reset() (still
        # current — collect_rollouts hasn't run yet), so it lines up exactly
        # with self.model._last_obs used below.
        self._pre_state = [e.get_sim_state() for e in self._raw_envs]

    def _on_step(self) -> bool:
        # SB3 calls callback.on_step() BEFORE _store_transition() updates
        # self.model._last_obs, so it still holds the PRE-step (normalized)
        # observation here — exactly the (s) half of this step's (s,a,s').
        obs_before = self.model._last_obs
        new_obs = self.locals["new_obs"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for i, env in enumerate(self._raw_envs):
            self._buf.add(
                obs=obs_before[i],
                action=actions[i],
                next_obs=new_obs[i],
                reward=float(rewards[i]),
                done=bool(dones[i]),
                sim_state=self._pre_state[i],
            )
            # Becomes the pre-step state for THIS env's next transition. If
            # this step ended the episode, the vec env already auto-reset the
            # env internally, so this snapshot is that new episode's initial
            # state — exactly the right pre-state for what comes next.
            self._pre_state[i] = env.get_sim_state()

        if (
            self.num_timesteps - self._last_update_step >= self._update_freq
            and self._buf.size >= self._batch_size
        ):
            self._last_update_step = self.num_timesteps
            loss = self._update_critic()
            if not np.isnan(loss):
                self.logger.record("shield/robust_recovery_critic_loss", loss)
        return True

    def _compute_rho(self, sim_states: list, actions: np.ndarray) -> np.ndarray:
        """rho_t(s,a): L2 norm of the per-dim std, across the K domain-
        randomized marble masses, of the TRUE simulator's next kinematic
        state ``[alpha, beta, ball_x, ball_y]`` after one (FRAME_SKIP-step)
        environment step from ``s`` under action ``a``. Replaces the learned
        world-model ensemble's ``‖σ_epistemic‖₂`` used in
        ``mbpo.MBPOTrainer._robust_recovery_q_update``."""
        import mujoco

        from envs.cyberrunner import FRAME_SKIP

        env = self._scratch_env
        B = len(sim_states)
        acts = np.clip(actions, -1.0, 1.0)
        preds = np.empty((self._K, B, 4), dtype=np.float64)
        for k, mass in enumerate(self._masses):
            env._set_marble_mass(float(mass))
            for b in range(B):
                env.set_sim_state(sim_states[b])
                env.data.ctrl[:] = acts[b]
                for _ in range(FRAME_SKIP):
                    mujoco.mj_step(env.model, env.data)
                ball = env._get_ball_pos_board_frame()
                preds[k, b, 0] = env.data.qpos[0]
                preds[k, b, 1] = env.data.qpos[1]
                preds[k, b, 2] = ball[0]
                preds[k, b, 3] = ball[1]
        std = preds.std(axis=0)           # (B, 4) epistemic disagreement per kinematic dim
        return np.linalg.norm(std, axis=1).astype(np.float32)   # (B,)

    def _update_critic(self) -> float:
        model = self.model
        policy = model.policy
        gamma = float(model.gamma)
        tau = float(model.tau)
        target_update_interval = int(getattr(model, "target_update_interval", 1))
        device = model.device

        policy.critic.set_training_mode(True)
        losses = []
        for grad_step in range(self._n_grad_steps):
            batch = self._buf.sample(self._batch_size)
            rho = self._compute_rho(batch.sim_states, batch.action)
            penalty = np.minimum(2.0 * self._L * np.clip(rho - self._margin, 0.0, None), 1.0)
            reward = batch.reward - penalty

            obs_t = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
            next_obs_t = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(batch.action, dtype=torch.float32, device=device)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1)
            absorbed_t = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(-1)

            with torch.no_grad():
                # Plain expectation over the FROZEN recovery policy's actions —
                # no entropy bonus (the proposition's E_{pi_r,mu} is not a soft value).
                next_act, _ = policy.actor.action_log_prob(next_obs_t)
                next_qs = torch.cat(policy.critic_target(next_obs_t, next_act), dim=1)
                next_q, _ = next_qs.min(dim=1, keepdim=True)
                new_target_q = reward_t + (1.0 - absorbed_t) * gamma * next_q

                # Intersect with the critic's own current (pre-update) belief
                # at this exact (s,a) (see mbpo.py's identical trick): Q is a
                # recovery PROBABILITY (higher is better), so pessimism takes
                # the MAX — Q is only ever revised upward, never dragged below
                # what the critic already believes.
                old_qs = torch.cat(policy.critic(obs_t, act_t), dim=1)
                old_target_q, _ = old_qs.min(dim=1, keepdim=True)
                target = torch.maximum(new_target_q, old_target_q)

            current_qs = policy.critic(obs_t, act_t)
            loss = 0.5 * sum(F.mse_loss(q, target) for q in current_qs)

            policy.critic.optimizer.zero_grad()
            loss.backward()
            policy.critic.optimizer.step()
            if grad_step % target_update_interval == 0:
                polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)
            losses.append(float(loss.item()))
        policy.critic.set_training_mode(False)
        return float(np.mean(losses)) if losses else float("nan")
