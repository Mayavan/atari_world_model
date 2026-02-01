from __future__ import annotations

"""Atari environment wrappers for preprocessing and frame stacking."""

from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation


class FloatNormalize(gym.ObservationWrapper):
    """Convert observations to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        """Wrap environment and update observation space to float32."""
        super().__init__(env)
        low = np.zeros(env.observation_space.shape, dtype=np.float32)
        high = np.ones(env.observation_space.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs


class FrameStack(gym.Wrapper):
    """Stack last k frames along axis 0 to produce (k, H, W)."""

    def __init__(self, env: gym.Env, k: int = 4):
        """Initialize a fixed-length frame buffer."""
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque(maxlen=k)
        low = np.repeat(env.observation_space.low[None, ...], k, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        """Reset env and fill the frame buffer with the initial obs."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """Step env and append the newest frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Return stacked frames as a single array."""
        return np.stack(self.frames, axis=0)


def make_atari_env(game: str, seed: Optional[int] = None, render_mode: Optional[str] = None) -> gym.Env:
    """Create a preprocessed Gymnasium Atari environment."""
    import ale_py  # noqa: F401
    gym.register_envs(ale_py)

    env_id = f"ALE/{game}-v5"
    try:
        env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "ROM" in msg or "rom" in msg or "roms" in msg:
            raise RuntimeError(
                "Atari ROMs not found. Install the ROMs and accept the license: \n"
                "1) pip install 'gymnasium[atari,accept-rom-license]'\n"
                "2) Run: python -m gymnasium.utils.install_roms\n"
                "Then retry the command."
            ) from e
        raise

    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FloatNormalize(env)
    env = FrameStack(env, k=4)

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        try:
            env.action_space.seed(seed)
        except Exception:
            pass

    return env
