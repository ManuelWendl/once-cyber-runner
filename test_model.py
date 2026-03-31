"""
Test a trained PPO or SAC model with the MuJoCo viewer.

Usage:
    mjpython test_model.py --algo ppo --run-name ppo_3stack --frame-stack 3
    mjpython test_model.py --algo ppo --run-name ppo_3stack --no-render --episodes 20
    mjpython test_model.py --algo ppo --run-name ppo_3stack --use-final
"""
import argparse
import os
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from cyberrunner_env import CyberRunnerEnv

CONTROL_HZ = 60
ALGO_CLS = {"ppo": PPO, "sac": SAC}


def main(args):
    algo = args.algo.lower()
    run_name = args.run_name or algo
    model_dir = f"./models/{run_name}"
    vecnorm_path = f"{model_dir}/vecnormalize.pkl"

    if args.use_final:
        model_path = f"{model_dir}/final.zip"
    else:
        model_path = args.model_path or f"{model_dir}/best/best_model.zip"

    model = ALGO_CLS[algo].load(model_path)
    print(f"Loaded {algo.upper()} model from {model_path}")

    render_mode = None if args.no_render else "human"
    env = DummyVecEnv([lambda: CyberRunnerEnv(render_mode=render_mode, randomize_init_pos=False)])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vecnorm_path}")

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        print(f"Frame stacking: {args.frame_stack}")

    for episode in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            t_start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if render_mode:
                env.envs[0].render()
            total_reward += reward[0]
            step += 1

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - elapsed))

            if done[0]:
                print(f"Episode {episode + 1}: steps={step}, reward={total_reward:.3f}, "
                      f"progress={info[0].get('path_progress', 0):.3f}, "
                      f"reason={info[0].get('termination_reason', 'timeout')}")
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name used during training. Loads from ./models/<run-name>/")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (defaults to ./models/<run-name>/best/best_model.zip)")
    parser.add_argument("--use-final", action="store_true",
                        help="Use final model instead of best")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Number of stacked frames (must match training)")
    args = parser.parse_args()
    main(args)
