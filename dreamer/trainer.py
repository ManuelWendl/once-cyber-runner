"""DreamerV3 trainer for the CyberRunner maze env.

Adapted from the source repo's ``OnlineTrainer``: same env-interaction /
update cadence, but shaped like ``MBPOTrainer`` (``learn(total_timesteps,
wandb_run)`` / ``predict`` / ``save``) and logging through the passed
``wandb_run`` instead of a tensorboard logger.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from tensordict import TensorDict

from envs.cyberrunner import CyberRunner, CyberRunnerEnv

from . import tools
from .agent import Dreamer
from .buffer import Buffer
from .parallel import ParallelEnv


def _env_kwargs(env_cfg) -> dict:
    return dict(
        episode_length=env_cfg.episode_length,
        randomize_init_pos=env_cfg.randomize_init_pos,
        reward_every_n_waypoints=env_cfg.reward_every_n_waypoints,
        hole_penalty=env_cfg.hole_penalty,
        dense_main_progress_scale=env_cfg.get("dense_main_progress_scale", 100.0),
        layout=env_cfg.get("layout", "hard"),
        obs_n_stack=env_cfg.get("obs_n_stack", 1),
    )


class DreamerTrainer:
    def __init__(self, cfg, device: str = "cuda", seed: int = 0):
        self.cfg = cfg
        ac = cfg.algo
        self.device = torch.device(device)
        tools.set_seed_everywhere(seed)

        env_kwargs = _env_kwargs(cfg.env)

        def env_constructor(idx):
            return lambda: CyberRunner(seed=seed + idx, action_repeat=ac.action_repeat, **env_kwargs)

        self.envs = ParallelEnv(env_constructor, int(ac.env_num), self.device)
        self.buffer = Buffer(ac.buffer, mirror_augment=bool(ac.mirror_augment))
        self.agent = Dreamer(
            ac.model, self.envs.observation_space, self.envs.action_space
        ).to(self.device)

        self.batch_length = int(ac.batch_length)
        self.action_repeat = int(ac.action_repeat)
        batch_steps = int(ac.batch_size * ac.batch_length)
        # train_ratio is based on replayed data steps rather than env steps.
        self._updates_needed = tools.Every(batch_steps / ac.train_ratio * ac.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(int(ac.update_log_every))
        self._should_save = tools.Every(int(ac.save_every))
        self.pretrain = int(ac.pretrain)

        # Stateful predict() for eval rollouts.
        self._pred_state = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, wandb_run=None) -> None:
        envs, agent = self.envs, self.agent
        step = 0
        update_count = 0
        holes_total = 0.0
        goals_total = 0.0
        ep_rewards: deque[float] = deque(maxlen=100)
        ep_lengths: deque[float] = deque(maxlen=100)
        train_metrics: dict = {}
        t_last, step_last = time.time(), 0

        # (B,) — all envs start with a reset.
        done = torch.ones(envs.env_num, dtype=torch.bool, device=self.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=self.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=self.device)
        # Constant per-env stream ids; episode boundaries are handled by is_first.
        episode_ids = torch.arange(envs.env_num, dtype=torch.int32, device=self.device)
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()

        while step < total_timesteps:
            if self._should_save(step):
                torch.save({"agent_state_dict": agent.state_dict()}, "latest.pt")
            # Record finished episodes.
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        ep_rewards.append(float(returns[i]))
                        ep_lengths.append(float(lengths[i]))
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self.action_repeat  # env-side steps
            lengths += ~done

            # Step environments on CPU; move observations to GPU asynchronously.
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # dict of (B, 1, *)
            trans = trans_cpu.to(self.device, non_blocking=True)
            done = done_cpu.to(self.device)

            # Policy inference (agent resets its state on is_first).
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition; mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # don't lift dim
            self.buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            holes_total += float(trans["log_hole"][:, 0].sum())
            goals_total += float(trans["log_goal"][:, 0].sum())

            # Update models after enough data has accumulated.
            if step // (envs.env_num * self.action_repeat) > self.batch_length + 1:
                update_num = self.pretrain if self._should_pretrain() else self._updates_needed(step)
                for _ in range(update_num):
                    train_metrics = agent.update(self.buffer)
                update_count += update_num

                if self._should_log(step):
                    now = time.time()
                    fps = (step - step_last) / max(now - t_last, 1e-8)
                    t_last, step_last = now, step
                    log = {
                        "train/holes_total": holes_total,
                        "train/goals_total": goals_total,
                        "train/updates": update_count,
                        "train/buffer_count": self.buffer.count(),
                        "train/fps": fps,
                    }
                    if ep_rewards:
                        log["train/ep_rew"] = float(np.mean(ep_rewards))
                        log["train/ep_len"] = float(np.mean(ep_lengths))
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        log[f"train/{name}"] = float(value)
                    if wandb_run is not None:
                        wandb_run.log(log, step=step)
                    print(
                        f"[dreamer] step {step} | ep_rew {log.get('train/ep_rew', float('nan')):.3f} "
                        f"| goals {goals_total:.0f} | holes {holes_total:.0f} "
                        f"| updates {update_count} | fps {fps:.0f}"
                    )

        torch.save({"agent_state_dict": agent.state_dict()}, "latest.pt")

    # ------------------------------------------------------------------
    # Inference / eval
    # ------------------------------------------------------------------

    def reset_predict_state(self) -> None:
        self._pred_state = None

    def _one_step_trans(self, obs_dict) -> TensorDict:
        td = {}
        for key, value in obs_dict.items():
            tensor = torch.as_tensor(np.asarray(value), device=self.device)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            td[key] = tensor.unsqueeze(0)  # add batch dim -> (1, *)
        td["reward"] = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        return TensorDict(td, batch_size=(1,), device=self.device)

    @torch.no_grad()
    def predict(self, obs_dict, deterministic: bool = True):
        """Recurrent one-step policy. ``obs_dict`` is the CyberRunner wrapper
        obs dict for a single env; call reset_predict_state() at episode start."""
        if self._pred_state is None:
            self._pred_state = self.agent.get_initial_state(1)
        trans = self._one_step_trans(obs_dict)
        action, self._pred_state = self.agent.act(trans, self._pred_state, eval=deterministic)
        return tools.to_np(action[0])

    def eval_and_log_video(self, run, fps: int = 30, max_steps: int | None = None) -> None:
        """Roll one episode with the recurrent policy on a raw env, log video."""
        env = CyberRunner(seed=10_000, action_repeat=self.action_repeat, **_env_kwargs(self.cfg.env))
        self.agent.eval()
        self.reset_predict_state()
        obs = env.reset()
        frames, total_reward, steps = [], 0.0, 0
        limit = max_steps or int(self.cfg.env.episode_length)
        done = False
        while not done and steps < limit:
            action = self.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps % 2 == 0:  # 60 Hz control -> 30 fps video
                frames.append(env.render())
        env.close()
        self.agent.train()

        print(f"[dreamer] eval: reward {total_reward:.3f}, length {steps}")
        if run is not None and frames:
            import wandb

            video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W)
            run.log({
                "eval/total_reward": total_reward,
                "eval/ep_length": steps,
                "eval/video": wandb.Video(video, fps=fps, format="mp4"),
            })

    def save(self, name: str) -> None:
        torch.save({"agent_state_dict": self.agent.state_dict()}, f"{name}.pt")

    def close(self) -> None:
        for env in self.envs.envs:
            env.close()
