import pathlib
import sys
import warnings

import hydra
import imageio
import torch
from tensordict import TensorDict

from dreamer import Dreamer
from envs import make_env

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))


def _to_batched_td(obs, device):
    """Convert a numpy obs dict (single env) to a batched TensorDict on device."""
    out = {}
    for k, v in obs.items():
        t = torch.as_tensor(v, device=device).unsqueeze(0)  # add batch dim
        if t.ndim == 1:  # scalar-per-env keys (is_first, is_last, is_terminal)
            t = t.unsqueeze(-1)  # (1,) -> (1, 1) matching lift_dim
        out[k] = t
    return TensorDict(out, batch_size=(1,), device=device)


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    checkpoint = pathlib.Path(config.checkpoint).expanduser()
    video_path = pathlib.Path(config.video_path).expanduser()
    num_episodes = int(config.num_episodes)
    device = torch.device(config.device)

    print(f"Loading checkpoint: {checkpoint}")
    env = make_env(config.env, 0)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    frames = []
    for ep in range(num_episodes):
        obs = env.reset()
        state = agent.get_initial_state(1)
        done = False
        ep_return, ep_length = 0.0, 0
        frames.append(env.render())
        while not done:
            obs_td = _to_batched_td(obs, device)
            action, state = agent.act(obs_td, state, eval=True)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, _ = env.step(action_np)
            ep_return += float(reward)
            ep_length += 1
            frames.append(env.render())
        print(f"Episode {ep}: return={ep_return:.2f}, length={ep_length}")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=60, quality=8)
    print(f"Saved {len(frames)} frames to {video_path}")


if __name__ == "__main__":
    main()
