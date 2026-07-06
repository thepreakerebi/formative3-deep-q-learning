"""Train a DQN agent on Atari Pong (ALE/Pong-v5) with Stable Baselines 3.

All assignment hyperparameters are exposed as CLI flags so each team member
can run their 10 tuning experiments with the same script, e.g.:

    python train.py --run-name elvis-exp1 --lr 1e-4 --gamma 0.99 \
        --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 \
        --epsilon-decay 0.1 --timesteps 200000

Epsilon mapping (assignment -> SB3 DQN):
    epsilon_start -> exploration_initial_eps
    epsilon_end   -> exploration_final_eps
    epsilon_decay -> exploration_fraction (fraction of training over which
                     epsilon is linearly annealed from start to end)
"""

import argparse
import json
from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Pong")
    parser.add_argument("--run-name", default="baseline",
                        help="Name for this experiment run (used for log/model paths)")
    parser.add_argument("--policy", choices=["CnnPolicy", "MlpPolicy"], default="CnnPolicy",
                        help="Policy network architecture to compare (CNN vs MLP)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Experiences sampled from the replay buffer per update")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon for the epsilon-greedy exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="Final epsilon after annealing")
    parser.add_argument("--epsilon-decay", type=float, default=0.1,
                        help="Fraction of total timesteps over which epsilon is annealed")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=50_000,
                        help="Replay buffer size (lower this if RAM is limited)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-as", default="dqn_model.zip",
                        help="Path for the saved model (assignment requires dqn_model.zip)")
    return parser.parse_args()


def make_env(seed: int, monitor_dir: str):
    # ALE/Pong-v5 applies frameskip=4 internally; disable it (frameskip=1) so the
    # standard AtariWrapper used by make_atari_env can do the skipping instead
    # of the two stacking to an effective 16x skip.
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        monitor_dir=monitor_dir,
        env_kwargs={"frameskip": 1},
    )
    # Stack the last 4 frames so the network can perceive ball velocity.
    return VecFrameStack(env, n_stack=4)


def main() -> None:
    args = parse_args()

    log_dir = Path("logs") / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Record the exact configuration next to the logs for the experiment table.
    (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    env = make_env(args.seed, str(log_dir / "monitor"))

    model = DQN(
        args.policy,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay,
        buffer_size=args.buffer_size,
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=1_000,
        optimize_memory_usage=False,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=args.seed,
        verbose=1,
    )

    # Periodic checkpoints so an interrupted run can be resumed or salvaged
    # instead of losing all progress (the final save only happens at the end).
    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="dqn",
    )

    # Reward trends and episode lengths are logged to the monitor CSVs and
    # TensorBoard (rollout/ep_rew_mean, rollout/ep_len_mean).
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=checkpoint)

    model.save(args.save_as)
    print(f"\nModel saved to {args.save_as}")

    if model.ep_info_buffer:
        rewards = [info["r"] for info in model.ep_info_buffer]
        lengths = [info["l"] for info in model.ep_info_buffer]
        print(f"Last {len(rewards)} episodes — "
              f"mean reward: {sum(rewards) / len(rewards):.2f}, "
              f"mean length: {sum(lengths) / len(lengths):.0f} steps")


if __name__ == "__main__":
    main()
