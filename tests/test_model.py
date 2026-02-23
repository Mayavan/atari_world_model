import torch

from src.models.conv_autoencoder import ConvAutoencoder
from src.models.world_model import WorldModel


def _write_autoencoder_ckpt(tmp_path) -> str:
    model = ConvAutoencoder(in_channels=3, latent_channels=8, base_channels=16)
    ckpt_path = tmp_path / "ae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": {
                "type": "conv_autoencoder",
                "in_channels": model.in_channels,
                "image_height": model.image_height,
                "image_width": model.image_width,
                "base_channels": model.base_channels,
                "latent_channels": model.latent_channels,
                "downsample_factor": model.downsample_factor,
                "decoder_type": model.decoder_type,
            },
        },
        ckpt_path,
    )
    return str(ckpt_path)


def test_model_forward_shape(tmp_path):
    model = WorldModel(
        num_actions=6,
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
        autoencoder_checkpoint=_write_autoencoder_ckpt(tmp_path),
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=2,
        sampling_steps=2,
        width_mult=0.5,
    )
    obs = torch.randn(2, 12, 84, 84)
    past_actions = torch.randint(0, 6, (2, 3))
    future_actions = torch.randint(0, 6, (2, 2))
    out = model(obs, future_actions, past_actions)
    assert out.shape == (2, 6, 84, 84)


def test_model_flow_loss_and_frozen_autoencoder(tmp_path):
    model = WorldModel(
        num_actions=6,
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
        autoencoder_checkpoint=_write_autoencoder_ckpt(tmp_path),
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=1,
        sampling_steps=2,
        width_mult=0.5,
    )
    obs = torch.rand(2, 12, 84, 84)
    next_obs = torch.rand(2, 3, 84, 84)
    past_actions = torch.randint(0, 6, (2, 3))
    future_actions = torch.randint(0, 6, (2, 1))
    losses = model.compute_flow_matching_loss(obs, future_actions, past_actions, next_obs)
    losses["loss"].backward()
    assert losses["loss"].item() >= 0.0
    for param in model.autoencoder.parameters():
        assert param.requires_grad is False
        assert param.grad is None
