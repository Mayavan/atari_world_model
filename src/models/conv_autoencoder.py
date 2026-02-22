from __future__ import annotations

"""Convolutional autoencoder with spatial latents for image tensors."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = _gn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class _UpBlockTranspose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.norm = _gn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class _UpBlockUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _gn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return F.silu(self.norm(self.conv(x)))


class ConvAutoencoder(nn.Module):
    """Deterministic convolutional autoencoder with spatial latents."""

    def __init__(
        self,
        *,
        in_channels: int,
        image_height: int = 84,
        image_width: int = 84,
        base_channels: int = 64,
        latent_channels: int = 16,
        downsample_factor: Literal[8, 16] = 8,
        decoder_type: Literal["upsample_conv", "conv_transpose"] = "upsample_conv",
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be > 0")
        if downsample_factor not in (8, 16):
            raise ValueError("downsample_factor must be one of: 8, 16")
        if decoder_type not in ("upsample_conv", "conv_transpose"):
            raise ValueError("decoder_type must be 'upsample_conv' or 'conv_transpose'")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_height and image_width must be > 0")

        self.in_channels = int(in_channels)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.decoder_type = str(decoder_type)

        num_down_blocks = int(math.log2(self.downsample_factor))
        channels = [self.base_channels * (2**i) for i in range(num_down_blocks)]

        encoder_blocks: list[nn.Module] = []
        in_ch = self.in_channels
        for out_ch in channels:
            encoder_blocks.append(_DownBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_blocks)

        hidden_channels = channels[-1]
        self.latent_head = nn.Conv2d(hidden_channels, self.latent_channels, kernel_size=1)

        self.decoder_in = nn.Conv2d(self.latent_channels, hidden_channels, kernel_size=1)

        up_block_cls: type[nn.Module]
        if self.decoder_type == "conv_transpose":
            up_block_cls = _UpBlockTranspose
        else:
            up_block_cls = _UpBlockUpsample
        up_channels = list(reversed(channels[:-1])) + [self.base_channels]

        decoder_blocks: list[nn.Module] = []
        dec_in = hidden_channels
        for dec_out in up_channels:
            decoder_blocks.append(up_block_cls(dec_in, dec_out))
            dec_in = dec_out
        self.decoder = nn.Sequential(*decoder_blocks)
        self.output_head = nn.Conv2d(dec_in, self.in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor into a deterministic latent."""
        h = self.encoder(x)
        z = self.latent_head(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back into an image in [0, 1]."""
        h = self.decoder_in(z)
        h = self.decoder(h)
        if h.shape[-2:] != (self.image_height, self.image_width):
            h = F.interpolate(
                h,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(self.output_head(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
