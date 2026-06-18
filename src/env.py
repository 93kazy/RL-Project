"""Environment factory for the Atari *Assault* DQN project.

This module centralises the construction of the training and evaluation
environments so that the exact same preprocessing pipeline is shared between
``train.py`` and ``evaluate.py``.

The pipeline follows the standard "Nature DQN" Atari setup:

* ``AtariWrapper`` applies no-op resets, fire-on-reset, 4-frame skipping with
  max-pooling, grayscale conversion and an 84x84 resize.
* ``VecFrameStack`` stacks the 4 most recent frames so that the agent can infer
  motion (the direction and speed of projectiles).
"""

from __future__ import annotations

from typing import Optional

import ale_py
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack

# Register the Atari Learning Environment ids with Gymnasium.
gym.register_envs(ale_py)

ENV_ID: str = "ALE/Assault-v5"
FRAME_STACK: int = 4


def make_env(
    *,
    training: bool = True,
    log_dir: Optional[str] = None,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> VecEnv:
    """Build the vectorised, frame-stacked *Assault* environment.

    Args:
        training: If ``True``, rewards are clipped to ``{-1, 0, +1}`` and an
            episode ends on every life loss. Both stabilise learning and are
            the conventional choice for Atari DQN training. If ``False``, the
            raw (unclipped) game score is preserved and an episode only ends
            when the game is actually over, which is what we want when
            *measuring* the agent's performance.
        log_dir: Optional directory in which ``Monitor`` records per-episode
            statistics (used during training).
        render_mode: Forwarded to ``gym.make`` (e.g. ``"human"`` to watch the
            agent play in a window).
        seed: Optional seed applied to the vectorised environment for
            reproducibility.

    Returns:
        A :class:`~stable_baselines3.common.vec_env.VecEnv` wrapping a single
        environment with frame stacking applied.
    """

    def _init() -> gym.Env:
        # ``frameskip=1`` because frame skipping is handled by ``AtariWrapper``;
        # this avoids skipping frames twice.
        env: gym.Env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
        if log_dir is not None:
            env = Monitor(env, log_dir)
        if training:
            env = AtariWrapper(env)
        else:
            env = AtariWrapper(env, terminal_on_life_loss=False, clip_reward=False)
        return env

    venv: VecEnv = DummyVecEnv([_init])
    venv = VecFrameStack(venv, n_stack=FRAME_STACK)

    if seed is not None:
        venv.seed(seed)

    return venv
