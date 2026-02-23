from __future__ import annotations

"""Procgen environment wrappers for preprocessing and frame stacking."""

import logging
from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)
_PROCGEN_REGISTERED = False


def _register_procgen_gymnasium_bridge() -> None:
    """Register Procgen Gymnasium env ids via shimmy bridge when needed."""
    global _PROCGEN_REGISTERED
    if _PROCGEN_REGISTERED:
        return

    import procgen.gym_registration as procgen_registration
    from procgen.env import ENV_NAMES
    from shimmy import GymV21CompatibilityV0

    class _ProcgenGymV21CompatibilityV0(GymV21CompatibilityV0):
        """Gym v21 shim without reset(options=...) warnings for reset-mask vector resets."""

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self.gym_env.seed(seed)
            obs = self.gym_env.reset()
            if self.render_mode == "human":
                self.render()
            return obs, {}

    def _make_procgen_env_bridge(**kwargs):
        legacy_env = procgen_registration.make_env(**kwargs)
        # gym3 ToGymEnv.seed() does not accept arguments and only prints a warning.
        # Override it so GymV21CompatibilityV0.reset(seed=...) is compatible.
        legacy_env.seed = lambda seed=None: None
        return _ProcgenGymV21CompatibilityV0(env=legacy_env)

    for env_name in ENV_NAMES:
        env_id = f"procgen-{env_name}-v0"
        if env_id in gym.registry:
            continue
        register(
            id=env_id,
            entry_point=_make_procgen_env_bridge,
            kwargs={"env_name": env_name},
        )

    _PROCGEN_REGISTERED = True


class ExtractRGB(gym.ObservationWrapper):
    """Extract `rgb` frame from dict observations when needed."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not hasattr(space, "spaces") or "rgb" not in space.spaces:
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


class GrayscaleResize(gym.ObservationWrapper):
    """Convert RGB observations to grayscale and resize to (84, 84)."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim != 3 or obs.shape[-1] != 3:
            raise ValueError(f"Unexpected observation shape for GrayscaleResize: {obs.shape}")
        gray = np.asarray(Image.fromarray(obs).convert("L").resize(self.shape, Image.BILINEAR))
        return gray.astype(np.uint8)


class RGBResize(gym.ObservationWrapper):
    """Resize RGB observations to (H, W, 3) uint8."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], 3),
            dtype=np.uint8,
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim != 3 or obs.shape[-1] != 3:
            raise ValueError(f"Unexpected observation shape for RGBResize: {obs.shape}")
        rgb = np.asarray(Image.fromarray(obs).resize(self.shape, Image.BILINEAR))
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Unexpected resized RGB shape: {rgb.shape}")
        return rgb.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack last k frames along axis 0."""

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


def _find_env_id_in_gymnasium(game: str) -> str | None:
    env_ids = (
        f"procgen-{game.lower()}-v0",
        f"procgen:procgen-{game.lower()}-v0",
    )
    registry = gym.registry
    for env_id in env_ids:
        if env_id in registry:
            return env_id
    return None

def make_procgen_env(
    game: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    frame_stack: int = 4,
    obs_mode: str = "grayscale",
    normalize: bool = True,
    distribution_mode: str = "easy",
    num_levels: int = 0,
) -> gym.Env:
    """Create a preprocessed Procgen environment."""
    import procgen  # noqa: F401
    _register_procgen_gymnasium_bridge()
    logger.info("Imported procgen module and ensured Gymnasium Procgen registration")

    env_id = _find_env_id_in_gymnasium(game)
    if env_id is None:
        raise RuntimeError(
            f"Unable to find Gymnasium Procgen env id for game='{game}'. "
            "Legacy Gym/shimmy fallback was removed."
        )
    logger.info("Creating Procgen env via Gymnasium id '%s'", env_id)
    env = gym.make(
        env_id,
        render_mode=render_mode,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=0 if seed is None else int(seed),
    )

    logger.info("Applying Procgen preprocessing wrappers")
    if hasattr(env.observation_space, "spaces") and "rgb" in env.observation_space.spaces:
        logger.info("Extracting 'rgb' key from dict observation space")
        env = ExtractRGB(env)
    if obs_mode == "grayscale":
        env = GrayscaleResize(env, shape=(84, 84))
    elif obs_mode == "rgb":
        env = RGBResize(env, shape=(84, 84))
    else:
        raise ValueError(f"Unsupported obs_mode='{obs_mode}'. Expected 'grayscale' or 'rgb'.")
    if normalize:
        env = FloatNormalize(env)
    env = FrameStack(env, k=frame_stack)

    if seed is not None:
        logger.info("Seeding environment with seed=%d", int(seed))
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    else:
        logger.info("No seed provided")

    obs_space = env.observation_space
    expected_shape = (frame_stack, 84, 84) if obs_mode == "grayscale" else (frame_stack, 84, 84, 3)
    if not isinstance(obs_space, gym.spaces.Box) or obs_space.shape != expected_shape:
        raise RuntimeError(
            "Unexpected observation space after Procgen wrappers. "
            f"Got {obs_space}."
        )

    logger.info("Procgen env ready: obs_shape=%s action_space=%s", obs_space.shape, env.action_space)
    return env
