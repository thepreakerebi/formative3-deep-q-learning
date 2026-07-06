"""Load the trained DQN model and play Pong with GUI rendering.

    python play.py --model dqn_model.zip --episodes 3

Evaluation uses the greedy Q policy: model.predict(..., deterministic=True)
always selects the action with the highest Q-value, disabling epsilon-greedy
exploration so the agent plays at its best.
"""

import argparse

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Pong with a trained DQN agent")
    parser.add_argument("--model", default="dqn_model.zip",
                        help="Path to the trained model saved by train.py")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to play")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Same preprocessing as train.py (AtariWrapper + 4-frame stack), with
    # render_mode="human" so the game is displayed on a GUI in real time.
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        env_kwargs={"frameskip": 1, "render_mode": "human"},
    )
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load(args.model)
    print(f"Loaded {args.model} — playing {args.episodes} episode(s) on {ENV_ID}")

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            # Greedy Q policy: pick the argmax-Q action (no exploration).
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
        print(f"Episode {episode}: reward {total_reward:+.0f} over {steps} steps")

    env.close()


if __name__ == "__main__":
    main()
