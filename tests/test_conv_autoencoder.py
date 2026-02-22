import math

import pytest
import torch

from src.models.conv_autoencoder import ConvAutoencoder


def _downsample_size(size: int, factor: int) -> int:
    steps = int(math.log2(factor))
    out = size
    for _ in range(steps):
        out = ((out + 2 - 4) // 2) + 1
    return out


@pytest.mark.parametrize("downsample_factor", [8, 16])
@pytest.mark.parametrize("decoder_type", ["upsample_conv", "conv_transpose"])
def test_conv_autoencoder_forward_shapes(downsample_factor: int, decoder_type: str) -> None:
    model = ConvAutoencoder(
        in_channels=1,
        image_height=84,
        image_width=84,
        base_channels=32,
        latent_channels=12,
        downsample_factor=downsample_factor,
        decoder_type=decoder_type,
    )
    x = torch.rand(3, 1, 84, 84)
    recon = model(x)

    expected_hw = _downsample_size(84, downsample_factor)
    assert recon.shape == x.shape
    z = model.encode(x)
    assert z.shape == (3, 12, expected_hw, expected_hw)

def test_conv_autoencoder_decode_output_range() -> None:
    model = ConvAutoencoder(
        in_channels=2,
        image_height=84,
        image_width=84,
        base_channels=16,
        latent_channels=8,
        downsample_factor=8,
        decoder_type="upsample_conv",
    )
    x = torch.rand(2, 2, 84, 84)
    z = model.encode(x)
    recon = model.decode(z)
    assert recon.shape == x.shape
    assert float(recon.detach().min()) >= 0.0
    assert float(recon.detach().max()) <= 1.0
