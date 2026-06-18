"""Evaluate a trained DQN agent on the Atari *Assault* environment.

The agent plays a number of full episodes (default: 20) with a deterministic
(greedy) policy, the raw game score of each episode is recorded, and summary
statistics are printed. Optionally a bar chart of the per-episode scores is
saved.

Examples:
    Evaluate the bundled model over 20 episodes::

        python src/evaluate.py

    Watch the agent play and save the score chart::

        python src/evaluate.py --episodes 10 --render --plot assets/evaluation_scores.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import List

from stable_baselines3 import DQN

from env import make_env


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on Atari Assault.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="models/DQN_assault",
                        help="Path to the saved model (with or without .zip).")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes to play.")
    parser.add_argument("--render", action="store_true",
                        help="Display the game window while playing.")
    parser.add_argument("--plot", type=str, default=None,
                        help="If set, save a bar chart of per-episode scores to this path.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def evaluate(model: DQN, env, episodes: int) -> List[float]:
    """Play ``episodes`` full games and return the score of each one."""
    scores: List[float] = []
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            score += float(rewards[0])
            done = bool(dones[0])
        scores.append(score)
        print(f"Episode {episode:>3}/{episodes} - score: {score:.0f}")
    return scores


def save_score_plot(scores: List[float], path: str) -> None:
    """Save a labelled bar chart of per-episode scores."""
    import matplotlib.pyplot as plt  # imported lazily so it stays optional

    episodes = list(range(1, len(scores) + 1))
    mean = statistics.mean(scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(episodes, scores, color="#7ec8e3", edgecolor="#1f4e79", linewidth=0.8)
    ax.axhline(mean, color="#c0392b", linestyle="--", linewidth=1.3, label=f"Mean = {mean:.0f}")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{score:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"DQN performance over {len(scores)} evaluation episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_xticks(episodes)
    ax.set_ylim(0, max(scores) * 1.12)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Score chart saved to {path}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not (model_path.exists() or model_path.with_suffix(".zip").exists()):
        raise FileNotFoundError(f"Model not found at '{model_path}'.")

    render_mode = "human" if args.render else None
    env = make_env(training=False, render_mode=render_mode, seed=args.seed)
    model = DQN.load(model_path, env=env)

    scores = evaluate(model, env, args.episodes)
    env.close()

    print("\n--- Summary ---")
    print(f"Episodes : {len(scores)}")
    print(f"Mean     : {statistics.mean(scores):.1f}")
    print(f"Median   : {statistics.median(scores):.1f}")
    print(f"Std      : {statistics.pstdev(scores):.1f}")
    print(f"Min      : {min(scores):.0f}")
    print(f"Max      : {max(scores):.0f}")

    if args.plot:
        save_score_plot(scores, args.plot)


if __name__ == "__main__":
    main()
