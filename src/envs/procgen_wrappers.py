from __future__ import annotations

"""Procgen environment wrappers for preprocessing and frame stacking."""

from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation


class StepAPICompatibility(gym.Wrapper):
    """Normalize reset/step outputs to Gymnasium's API."""

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise TypeError("Environment step() must return a tuple.")
        if len(out) == 5:
            return out
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), False, info
        raise ValueError(f"Unexpected step() return length: {len(out)}")


class ExtractRGB(gym.ObservationWrapper):
    """Extract `rgb` frame from dict observations when needed."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "rgb" not in space.spaces:
            raise ValueError("ExtractRGB requires a Dict observation space with key 'rgb'.")
        self.observation_space = space.spaces["rgb"]

    def observation(self, observation):
        if isinstance(observation, dict) and "rgb" in observation:
            return observation["rgb"]
        raise ValueError("Expected dict observation containing key 'rgb'.")


class FloatNormalize(gym.ObservationWrapper):
    """Convert observations to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
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
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque(maxlen=k)
        low = np.repeat(env.observation_space.low[None, ...], k, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


def make_procgen_env(
    game: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    frame_stack: int = 4,
    distribution_mode: str = "easy",
    num_levels: int = 0,
) -> gym.Env:
    """Create a preprocessed Procgen environment."""
    try:
        import procgen  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Procgen is not installed. Run: pip install procgen"
        ) from e

    env_id_candidates = [
        f"procgen-{game.lower()}-v0",
        f"procgen:procgen-{game.lower()}-v0",
    ]
    env = None
    last_error: Exception | None = None
    for env_id in env_id_candidates:
        try:
            env = gym.make(
                env_id,
                render_mode=render_mode,
                distribution_mode=distribution_mode,
                num_levels=num_levels,
                start_level=0 if seed is None else int(seed),
            )
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    if env is None:
        raise RuntimeError(
            f"Unable to create Procgen env for game='{game}'. "
            f"Tried IDs: {env_id_candidates}. Last error: {last_error}"
        )

    env = StepAPICompatibility(env)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = ExtractRGB(env)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FloatNormalize(env)
    env = FrameStack(env, k=frame_stack)

    if seed is not None:
        try:
            env.reset(seed=int(seed))
        except TypeError:
            env.reset()
        try:
            env.action_space.seed(int(seed))
        except Exception:
            pass

    return env
