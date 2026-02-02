from __future__ import annotations

"""Baseline action-conditioned world model (encoder + conditioner + decoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN encoder that compresses a 4-frame stack into a latent vector."""
    def __init__(self, latent_dim: int):
        """Initialize convolutional backbone and projection head."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 6 * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode stacked observations into a latent embedding."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    """CNN decoder that predicts the next frame from a fused latent."""
    def __init__(self, input_dim: int):
        """Initialize projection and upsampling stack."""
        super().__init__()
        self.fc = nn.Linear(input_dim, 256 * 6 * 6)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent features into a normalized next frame."""
        x = self.fc(x)
        x = x.view(x.size(0), 256, 6, 6)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, size=(24, 24), mode="bilinear", align_corners=False)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, size=(48, 48), mode="bilinear", align_corners=False)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, size=(84, 84), mode="bilinear", align_corners=False)
        x = self.conv_out(x)
        return torch.sigmoid(x)



class WorldModel(nn.Module):
    """Action-conditioned next-frame predictor with concat conditioning."""
    def __init__(
        self,
        num_actions: int,
        action_embed_dim: int = 32,
        latent_dim: int = 512,
        condition_mode: str = "concat",
    ):
        """Build encoder, action embedding, and decoder blocks."""
        super().__init__()
        self.num_actions = num_actions
        self.condition_mode = condition_mode
        self.encoder = Encoder(latent_dim=latent_dim)
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)

        if condition_mode == "concat":
            fuse_in = latent_dim + action_embed_dim
            self.fuse = nn.Sequential(
                nn.Linear(fuse_in, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256 * 6 * 6),
            )
        elif condition_mode == "film":
            # TODO: implement FiLM conditioning over decoder blocks.
            raise NotImplementedError("FiLM conditioning is TODO")
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        self.decoder = Decoder(input_dim=256 * 6 * 6)

    def forward(self, obs_stack: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next frame given stacked observation and action."""
        z = self.encoder(obs_stack)
        if self.condition_mode == "concat":
            a = self.action_embed(action)
            fused = torch.cat([z, a], dim=1)
            x = self.fuse(fused)
        else:
            raise RuntimeError("Unsupported condition mode")
        return self.decoder(x)
