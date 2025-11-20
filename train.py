import argparse
import os
from datetime import datetime
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from snake import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on the SnakeEnv.")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total training timesteps.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log-dir", type=str, default="runs/ppo_snake", help="TensorBoard log directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    return parser.parse_args()


def make_env(seed: Optional[int] = None) -> Callable[[], Monitor]:
    def _init():
        env = SnakeEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    env_fns = [make_env(args.seed + i if args.seed is not None else None) for i in range(args.num_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_snake_{timestamp}"

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        n_steps=2048 // args.num_envs * args.num_envs,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, tb_log_name=run_name)

    final_path = os.path.join(args.checkpoint_dir, f"{run_name}_final")
    model.save(final_path)
    vec_env.close()
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
