# Deep Q-Network on Atari *Assault*

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.7.1-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.3-green)

A reinforcement-learning agent that learns to play the Atari 2600 game
**Assault** from raw pixels, using a **Deep Q-Network (DQN)** trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

<p align="center">
  <img src="assets/evaluation_scores.png" alt="Per-episode scores of the trained agent over 20 evaluation games" width="720">
</p>

---

## Overview

In *Assault*, the player controls a ship anchored at the bottom of the screen.
A mother-ship at the top continuously deploys smaller enemies and fires at the
player. The agent must **dodge incoming fire while destroying the deployed
enemies**, which requires anticipating projectile trajectories — a good fit for
a learning-based controller.

The agent observes only the game screen (a stack of pixel frames) and chooses
among the 7 discrete actions of the environment. It is trained end-to-end with
DQN: a convolutional network estimates the action-values *Q(s, a)*, and the
greedy action is taken at evaluation time.

## Why DQN?

The choice of algorithm follows directly from the structure of the problem:

- **Pixel observations.** The state is an image, not a compact vector of
  coordinates. A tabular Q-learning approach is infeasible because the number
  of possible pixel combinations is astronomically large. DQN replaces the
  Q-table with a convolutional neural network that generalises across visually
  similar states.
- **Small discrete action space.** With only 7 actions, the network can output
  one Q-value per action and simply pick the maximum — exactly the setting DQN
  is designed for.
- **Correlated consecutive frames.** Successive frames are nearly identical, so
  training online on consecutive transitions biases the agent and causes it to
  forget. DQN's **experience replay** buffer stores past transitions and trains
  on random mini-batches, breaking the temporal correlation.

## Environment

| Property            | Value                                                        |
|---------------------|--------------------------------------------------------------|
| Environment id      | `ALE/Assault-v5`                                             |
| Observation         | 4 stacked grayscale frames, 84×84 pixels                     |
| Action space        | Discrete (7 actions)                                         |
| Preprocessing       | `AtariWrapper`: no-op reset, fire-on-reset, 4-frame skip + max-pool, grayscale, 84×84 resize |
| Training wrapper     | Reward clipping + episodic life (`terminal_on_life_loss=True`) |
| Evaluation wrapper   | Raw score, full episodes (`clip_reward=False`, `terminal_on_life_loss=False`) |

The preprocessing pipeline is defined once in [`src/env.py`](src/env.py) and
shared by both the training and the evaluation scripts.

## Results

The final agent was trained for **1,000,000 timesteps**. Episode rewards climb
steadily over training, reaching roughly 500 on average with peaks near 800 by
the end:

<p align="center">
  <img src="assets/training_curve.png" alt="Training reward curve over 1,000,000 timesteps" width="720">
</p>

Evaluated over **20 full games** with a deterministic policy, the agent reaches
the following scores:

| Metric  | Value |
|---------|-------|
| Mean    | 840   |
| Median  | 807.5 |
| Std     | 188.3 |
| Min     | 609   |
| Max     | 1437  |

The agent reliably survives and clears enemies, with several games well above
1,200 points. The score chart above (`assets/evaluation_scores.png`) is produced
directly by the evaluation script.

> Reproduce the evaluation chart:
> ```bash
> python src/evaluate.py --plot assets/evaluation_scores.png
> ```

## Project structure

```
RL-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                  # Figures used in this README
│   ├── training_curve.png
│   ├── evaluation_scores.png
│   └── ...                  # hyperparameter-search figures
├── models/
│   └── DQN_assault.zip      # Trained agent (1M timesteps)
└── src/
    ├── env.py               # Shared environment factory
    ├── train.py             # Training entry point
    └── evaluate.py          # Evaluation entry point
```

## Installation

Requires **Python 3.12**.

```bash
# Clone the repository
git clone https://github.com/93kazy/RL-Project.git
cd RL-Project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The Atari ROMs ship with `ale-py`, so no separate ROM download is required.
For GPU training, install the CUDA build of PyTorch that matches your system
from the [official instructions](https://pytorch.org/get-started/locally/).

## Usage

### Evaluate the trained agent

Play 20 games with the bundled model and print summary statistics:

```bash
python src/evaluate.py
```

Watch the agent play in a window:

```bash
python src/evaluate.py --render --episodes 5
```

### Train from scratch

Reproduce the final model with the default hyper-parameters:

```bash
python src/train.py
```

Both scripts expose their options through `--help`. A few common ones:

```bash
python src/train.py --timesteps 500000 --seed 0
python src/evaluate.py --model-path models/DQN_assault --episodes 20 --plot assets/eval.png
```

Training logs are written under `logs/` and can be inspected with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## Hyperparameters

The final configuration (stored in `models/DQN_assault.zip`):

| Hyperparameter           | Value     |
|--------------------------|-----------|
| Policy                   | `CnnPolicy` (Nature CNN) |
| Total timesteps          | 1,000,000 |
| Replay buffer size       | 50,000    |
| Learning rate            | 1e-4      |
| Batch size               | 32        |
| Discount factor (γ)      | 0.99      |
| Exploration fraction     | 0.1       |
| Final exploration ε      | 0.01      |
| Target update interval   | 10,000    |
| Frame stack              | 4         |

## Methodology: hyperparameter search

Candidate values were compared at 500,000 timesteps (a full run is expensive)
before committing to the 1,000,000-timestep final training.

### Learning rate

Three learning rates were compared. `1e-3` and `1e-4` clearly outperformed
`1e-5`; since the two best were close, **`1e-4`** was selected as the more
stable choice.

| `1e-3` | `1e-4` | `1e-5` |
|:------:|:------:|:------:|
| ![lr 1e-3](assets/lr_1e-3.png) | ![lr 1e-4](assets/lr_1e-4.png) | ![lr 1e-5](assets/lr_1e-5.png) |

### Replay buffer size

Buffer sizes between 30,000 and 50,000 produced no meaningful difference, so
the larger **50,000** was kept.

### Exploration fraction

Values of 0.05, 0.1 and 0.2 were compared. **0.1** gave the best learning
behaviour and was selected.

| `0.05` | `0.1` | `0.2` |
|:------:|:-----:|:-----:|
| ![exploration 0.05](assets/exploration_005.png) | ![exploration 0.1](assets/exploration_01.png) | ![exploration 0.2](assets/exploration_02.png) |

## Limitations & future work

- The agent was trained for only 1M timesteps; the reward curve is still rising
  at the end, so more training would likely improve performance further. The
  main constraint was compute time and GPU availability.
- Natural extensions include longer training, **Double DQN**, **Dueling DQN**,
  **Prioritized Experience Replay**, or a distributional / Rainbow variant.
- Running several seeds would give confidence intervals rather than a single
  point estimate.

## References

- Mnih et al., *Human-level control through deep reinforcement learning*,
  Nature, 2015.
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium / Arcade Learning Environment](https://ale.farama.org/)

## Author

**Tomil Shi** — Université Paris Dauphine – PSL, 2025–2026
Reinforcement Learning course project.
