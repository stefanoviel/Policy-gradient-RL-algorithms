import argparse
import time

from stable_baselines3 import PPO

from snake import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake.")
    parser.add_argument("--model-path", help="Path to the saved PPO model (.zip). Overrides auto-detection.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to search for saved models.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch.")
    parser.add_argument("--render", action="store_true", help="Render the environment (default).")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering.")
    parser.set_defaults(render=True)
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between steps when rendering.")
    return parser.parse_args()


def main():
    args = parse_args()

    render_mode = "human" if args.render else None
    env = SnakeEnv(render_mode=render_mode)

    model_path = args.model_path or _find_latest_final_checkpoint(args.checkpoint_dir)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if args.render:
                env.render()
                time.sleep(args.sleep)

        print(f"Episode {episode}: score={info.get('score')}, reward={episode_reward}")

    env.close()


def _find_latest_final_checkpoint(directory: str) -> str:
    import glob
    import os

    pattern = os.path.join(directory, "*_final.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No final checkpoints matching '{pattern}'. Run training first or pass --model-path.")
    latest = max(candidates, key=os.path.getmtime)
    return latest


if __name__ == "__main__":
    main()
