"""
Test a trained PPO or SAC model with the MuJoCo viewer.

Usage:
    mjpython test_model.py --algo ppo
    mjpython test_model.py --algo sac
    mjpython test_model.py --algo ppo --model-path ./models/ppo_final.zip
    mjpython test_model.py --algo sac --no-render
"""
import argparse
import time
from stable_baselines3 import PPO, SAC
from cyberrunner_env import CyberRunnerEnv

CONTROL_HZ = 60  # matches FRAME_SKIP=10 at 600Hz physics
ALGO_CLS = {"ppo": PPO, "sac": SAC}


def main(args):
    algo = args.algo.lower()
    model_path = args.model_path or f"./models/{algo}_best/best_model.zip"

    model = ALGO_CLS[algo].load(model_path)
    print(f"Loaded {algo.upper()} model from {model_path}")

    render_mode = None if args.no_render else "human"
    env = CyberRunnerEnv(render_mode=render_mode, randomize_init_pos=False)

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            t_start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode:
                env.render()
            total_reward += reward
            step += 1

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - elapsed))

            if terminated or truncated:
                print(f"Episode {episode + 1}: steps={step}, reward={total_reward:.3f}, "
                      f"progress={info['path_progress']:.3f}, "
                      f"reason={info.get('termination_reason', 'timeout')}")
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model zip. Defaults to ./models/<algo>_best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true",
                        help="Disable viewer, just print episode stats")
    args = parser.parse_args()
    main(args)
