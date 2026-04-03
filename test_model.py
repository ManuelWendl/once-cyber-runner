"""
Test a trained PPO or SAC model with the MuJoCo viewer.

Usage (interactive — no args):
    python test_model.py

Usage (scripted):
    mjpython test_model.py --algo ppo --run-name ppo_3stack --frame-stack 3
    mjpython test_model.py --algo ppo --run-name ppo_3stack --no-render --episodes 20
    mjpython test_model.py --algo ppo --run-name ppo_3stack --use-final
"""
import argparse
import os
import sys
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from maxinfosac_compat import MaxInfoSAC
from cyberrunner_env import CyberRunnerEnv

CONTROL_HZ = 60
ALGO_CLS = {"ppo": PPO, "sac": SAC, "maxinfosac": MaxInfoSAC}


def _discover_runs(algo: str) -> list[str]:
    """Return run names under ./models/ that match the given algo prefix."""
    if not os.path.isdir("./models"):
        return []
    runs = []
    for name in sorted(os.listdir("./models"), reverse=True):
        if not name.startswith(algo):
            continue
        d = f"./models/{name}"
        if os.path.isdir(d):
            runs.append(name)
    return runs


def _run_has_final(run_name: str) -> bool:
    return os.path.exists(f"./models/{run_name}/final.zip")


def _run_has_vecnorm(run_name: str) -> bool:
    return os.path.exists(f"./models/{run_name}/vecnormalize.pkl")


def interactive_args():
    """Ask the user for test parameters interactively using questionary."""
    import questionary

    # 1. Algorithm
    algo = questionary.select(
        "Algorithm?",
        choices=list(ALGO_CLS.keys()),
    ).ask()
    if algo is None:
        sys.exit(0)

    # 2. Run / model
    runs = _discover_runs(algo)
    if runs:
        choices = runs + ["[enter path manually]"]
        run_choice = questionary.select(
            "Which run?",
            choices=choices,
        ).ask()
        if run_choice is None:
            sys.exit(0)

        if run_choice == "[enter path manually]":
            run_name = None
            model_path = questionary.text("Model path (.zip):").ask()
            if not model_path:
                sys.exit(0)
        else:
            run_name = run_choice
            model_path = None

            # best vs final
            has_final = _run_has_final(run_name)
            if has_final:
                checkpoint = questionary.select(
                    "Checkpoint?",
                    choices=["best", "final"],
                ).ask()
                use_final = checkpoint == "final"
            else:
                use_final = False
    else:
        print(f"No saved runs found for '{algo}' under ./models/")
        run_name = None
        model_path = questionary.text("Model path (.zip):").ask()
        if not model_path:
            sys.exit(0)
        use_final = False

    # 3. Frame stack — only ask if vecnormalize exists (implies it was used in training)
    if run_name and _run_has_vecnorm(run_name):
        frame_stack = questionary.text(
            "Frame stack? (must match training, default 1):",
            default="1",
            validate=lambda v: v.isdigit() and int(v) >= 1,
        ).ask()
        frame_stack = int(frame_stack) if frame_stack else 1
    else:
        frame_stack = 1

    # 4. Episodes
    episodes = questionary.text(
        "Episodes?",
        default="5",
        validate=lambda v: v.isdigit() and int(v) >= 1,
    ).ask()
    episodes = int(episodes) if episodes else 5

    # 5. Render
    render = questionary.confirm("Render?", default=True).ask()

    # 6. Random start position
    rand_start_pos = questionary.confirm("Random start position?", default=False).ask()

    # Print equivalent CLI command
    parts = ["python test_model.py", f"--algo {algo}"]
    if run_name:
        parts.append(f"--run-name {run_name}")
    if use_final:
        parts.append("--use-final")
    if model_path:
        parts.append(f"--model-path {model_path}")
    if frame_stack > 1:
        parts.append(f"--frame-stack {frame_stack}")
    if episodes != 5:
        parts.append(f"--episodes {episodes}")
    if not render:
        parts.append("--no-render")
    if rand_start_pos:
        parts.append("--rand-start-pos")
    print("\n  " + " ".join(parts) + "\n")

    # Build a namespace that matches what main() expects
    return argparse.Namespace(
        algo=algo,
        run_name=run_name,
        model_path=model_path,
        use_final=use_final,
        frame_stack=frame_stack,
        episodes=episodes,
        no_render=not render,
        rand_start_pos=rand_start_pos,
    )


def main(args):
    algo = args.algo.lower()
    frame_stack = args.frame_stack
    use_pre_norm_stack = algo != "ppo" and frame_stack > 1
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
    env = DummyVecEnv([lambda: CyberRunnerEnv(render_mode=render_mode, randomize_init_pos=args.rand_start_pos)])

    if use_pre_norm_stack:
        env = VecFrameStack(env, n_stack=frame_stack)
        print(f"Frame stacking (pre-normalization): {frame_stack}")

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {vecnorm_path}")

    if frame_stack > 1 and not use_pre_norm_stack:
        env = VecFrameStack(env, n_stack=frame_stack)
        print(f"Frame stacking: {frame_stack}")

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
    if len(sys.argv) == 1:
        main(interactive_args())
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "maxinfosac"])
        parser.add_argument("--run-name", type=str, default=None,
                            help="Run name used during training. Loads from ./models/<run-name>/")
        parser.add_argument("--model-path", type=str, default=None,
                            help="Override model path (defaults to ./models/<run-name>/best/best_model.zip)")
        parser.add_argument("--use-final", action="store_true",
                            help="Use final model instead of best")
        parser.add_argument("--episodes", type=int, default=5)
        parser.add_argument("--no-render", action="store_true")
        parser.add_argument("--rand-start-pos", action="store_true",
                            help="Start each episode from a random position")
        parser.add_argument("--frame-stack", type=int, default=1,
                            help="Number of stacked frames (must match training; >1 typically PPO only)")
        main(parser.parse_args())
