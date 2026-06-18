"""Train a DQN agent on the Atari *Assault* environment.

The default hyper-parameters reproduce the final model described in the project
report: 1,000,000 timesteps, a 50k replay buffer, a 1e-4 learning rate and a
0.1 exploration fraction.

Examples:
    Train with the default configuration::

        python src/train.py

    Run a short smoke test and watch TensorBoard::

        python src/train.py --timesteps 50000
        tensorboard --logdir logs/tensorboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from env import make_env

DEFAULT_TIMESTEPS: int = 1_000_000
DEFAULT_MODEL_NAME: str = "DQN_assault"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Atari Assault.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total number of training timesteps.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help="File name (without extension) for the saved model.")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory where the trained model is saved.")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for Monitor episode logs.")
    parser.add_argument("--tensorboard-dir", type=str, default="logs/tensorboard",
                        help="Directory for TensorBoard logs.")
    parser.add_argument("--buffer-size", type=int, default=50_000,
                        help="Replay buffer size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size sampled from the replay buffer.")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                        help="Fraction of training over which epsilon is annealed.")
    parser.add_argument("--exploration-final-eps", type=float, default=0.01,
                        help="Final value of the exploration rate epsilon.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    env = make_env(training=True, log_dir=args.log_dir, seed=args.seed)

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=args.tensorboard_dir,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_path = models_dir / args.model_name
    model.save(save_path)
    env.close()
    print(f"Model saved to {save_path}.zip")


if __name__ == "__main__":
    main()
