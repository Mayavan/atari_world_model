import numpy as np
import torch
from gymnasium import spaces

from src.eval import rollout_open_loop
from src.models.world_model import WorldModel


class FakeEnv:
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self._rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        obs = self._rng.random((4, 84, 84, 3), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.random((4, 84, 84, 3), dtype=np.float32)
        reward = 0.0
        terminated = self._step >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_rollout_small_horizon():
    env = FakeEnv()
    model = WorldModel(
        num_actions=4,
        autoencoder_model_cfg={
            "type": "conv_autoencoder",
            "in_channels": 3,
            "image_height": 84,
            "image_width": 84,
            "base_channels": 16,
            "latent_channels": 8,
            "downsample_factor": 8,
            "decoder_type": "upsample_conv",
        },
        n_past_frames=4,
        n_past_actions=0,
        n_future_frames=1,
        sampling_steps=2,
        width_mult=0.5,
    )
    device = torch.device("cpu")
    mse, psnr, frames = rollout_open_loop(
        model,
        env,
        horizon=3,
        device=device,
        sampling_steps=2,
        capture_video=True,
    )
    assert isinstance(mse, float)
    assert isinstance(psnr, float)
    assert len(frames) <= 4
