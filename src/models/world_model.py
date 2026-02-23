from __future__ import annotations

"""Latent flow-matching world model with a frozen convolutional autoencoder."""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_autoencoder import ConvAutoencoder


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Build a GroupNorm layer with the largest valid group count."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _normalize_autoencoder_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize serialized autoencoder config fields."""
    if not isinstance(model_cfg, dict):
        raise ValueError("autoencoder model_cfg must be a mapping.")
    model_type = str(model_cfg.get("type", "conv_autoencoder")).strip().lower()
    if model_type not in {"conv_autoencoder", ""}:
        raise ValueError(
            "Expected autoencoder checkpoint with model_cfg.type='conv_autoencoder', "
            f"got '{model_cfg.get('type')}'."
        )
    in_channels = int(model_cfg.get("in_channels", 0))
    if in_channels <= 0:
        raise ValueError("autoencoder model_cfg.in_channels must be > 0")
    return {
        "type": "conv_autoencoder",
        "in_channels": in_channels,
        "image_height": int(model_cfg.get("image_height", 84)),
        "image_width": int(model_cfg.get("image_width", 84)),
        "base_channels": int(model_cfg.get("base_channels", 64)),
        "latent_channels": int(model_cfg.get("latent_channels", 16)),
        "downsample_factor": int(model_cfg.get("downsample_factor", 8)),
        "decoder_type": str(model_cfg.get("decoder_type", "upsample_conv")),
    }


def _build_autoencoder_from_cfg(model_cfg: dict[str, Any]) -> ConvAutoencoder:
    """Instantiate a ConvAutoencoder from normalized config values."""
    cfg = _normalize_autoencoder_cfg(model_cfg)
    return ConvAutoencoder(
        in_channels=int(cfg["in_channels"]),
        image_height=int(cfg["image_height"]),
        image_width=int(cfg["image_width"]),
        base_channels=int(cfg["base_channels"]),
        latent_channels=int(cfg["latent_channels"]),
        downsample_factor=int(cfg["downsample_factor"]),
        decoder_type=str(cfg["decoder_type"]),
    )


class FiLM(nn.Module):
    """Feature-wise linear modulation block for conditioning feature maps."""

    def __init__(self, cond_dim: int, channels: int, hidden: int = 256):
        """Create the FiLM MLP used to predict gamma/beta modulation."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning to a BCHW feature map."""
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * x + beta


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar flow time values."""

    def __init__(self, dim: int):
        """Initialize embedding dimensionality."""
        super().__init__()
        if dim <= 0:
            raise ValueError("time embedding dim must be > 0")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode a rank-1 batch of times into sinusoidal features."""
        if t.ndim != 1:
            raise ValueError(f"time tensor must be rank-1, got shape={tuple(t.shape)}")
        half = self.dim // 2
        if half == 0:
            return t[:, None]
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class FlowResBlock(nn.Module):
    """Residual conv block with FiLM conditioning in both conv stages."""

    def __init__(self, channels: int, cond_dim: int):
        """Build a residual block operating on latent spatial tensors."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = _gn(channels)
        self.film1 = FiLM(cond_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _gn(channels)
        self.film2 = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict a residual update conditioned on action/time context."""
        residual = x
        x = self.conv1(x)
        x = self.film1(self.norm1(x), cond)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.film2(self.norm2(x), cond)
        x = F.silu(x)
        return x + residual


class FlowPredictor(nn.Module):
    """Backbone network that predicts latent velocity fields."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        width_mult: float,
    ):
        """Construct the latent flow backbone with configurable width."""
        super().__init__()
        hidden = max(64, int(256 * width_mult))
        self.in_proj = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([FlowResBlock(hidden, cond_dim) for _ in range(4)])
        self.out_norm = _gn(hidden)
        self.out_proj = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run the flow backbone to produce per-pixel latent velocity."""
        x = F.silu(self.in_proj(x))
        for block in self.blocks:
            x = block(x, cond)
        x = F.silu(self.out_norm(x))
        return self.out_proj(x)


class WorldModel(nn.Module):
    """Action-conditioned latent flow model with a frozen image autoencoder."""

    def __init__(
        self,
        num_actions: int,
        *,
        autoencoder_model_cfg: dict[str, Any],
        autoencoder_checkpoint: str | Path | None = None,
        n_past_frames: int = 4,
        n_past_actions: int = 0,
        n_future_frames: int = 1,
        frame_channels: int = 3,
        action_embed_dim: int = 64,
        time_embed_dim: int = 128,
        sampling_steps: int = 16,
        width_mult: float = 1.0,
    ):
        """Initialize model components and optionally load pretrained autoencoder weights."""
        super().__init__()
        self.num_actions = int(num_actions)
        self.n_past_frames = int(n_past_frames)
        self.n_past_actions = int(n_past_actions)
        self.n_future_frames = int(n_future_frames)
        self.frame_channels = int(frame_channels)
        self.sampling_steps = int(sampling_steps)

        if self.n_past_frames <= 0:
            raise ValueError("n_past_frames must be > 0")
        if self.n_future_frames <= 0:
            raise ValueError("n_future_frames must be > 0")
        if self.n_past_actions < 0:
            raise ValueError("n_past_actions must be >= 0")
        if self.frame_channels != 3:
            raise ValueError("latent flow world model currently requires RGB frame_channels=3")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be > 0")
        if action_embed_dim <= 0:
            raise ValueError("action_embed_dim must be > 0")
        if time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be > 0")
        if width_mult <= 0.0:
            raise ValueError("width_mult must be > 0")

        self.autoencoder_model_cfg = _normalize_autoencoder_cfg(autoencoder_model_cfg)
        self.autoencoder = _build_autoencoder_from_cfg(self.autoencoder_model_cfg)
        if autoencoder_checkpoint is not None:
            ckpt_path = Path(autoencoder_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Autoencoder checkpoint not found: {ckpt_path}")
            autoencoder_ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" not in autoencoder_ckpt:
                raise KeyError(f"Autoencoder checkpoint missing key 'model': {ckpt_path}")
            self.autoencoder.load_state_dict(autoencoder_ckpt["model"])
            self.autoencoder_checkpoint = str(ckpt_path)
        else:
            self.autoencoder_checkpoint = ""
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)
        self.latent_channels = int(self.autoencoder.latent_channels)

        self.action_embed = nn.Embedding(self.num_actions, action_embed_dim)
        action_cond_dim = action_embed_dim * (self.n_past_actions + self.n_future_frames)
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        cond_dim = action_cond_dim + time_embed_dim

        future_latent_channels = self.n_future_frames * self.latent_channels
        past_latent_channels = self.n_past_frames * self.latent_channels
        self.flow = FlowPredictor(
            in_channels=future_latent_channels + past_latent_channels,
            out_channels=future_latent_channels,
            cond_dim=cond_dim,
            width_mult=width_mult,
        )

    def train(self, mode: bool = True):
        """Toggle train/eval mode while always keeping autoencoder in eval mode."""
        super().train(mode)
        # Keep the pretrained autoencoder deterministic/frozen at all times.
        self.autoencoder.eval()
        return self

    def _to_frame_major(self, x: torch.Tensor, *, n_frames: int) -> torch.Tensor:
        """Reshape channel-packed BCHW tensors into (B,T,C,H,W)."""
        if x.ndim != 4:
            raise ValueError(f"Expected rank-4 BCHW tensor, got shape={tuple(x.shape)}")
        expected_channels = n_frames * self.frame_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels ({n_frames}x{self.frame_channels}), "
                f"got {x.shape[1]}"
            )
        bsz, _, height, width = x.shape
        return x.reshape(bsz, n_frames, self.frame_channels, height, width)

    def _encode_frames(self, x: torch.Tensor, *, n_frames: int) -> torch.Tensor:
        """Encode packed RGB frames into spatial latent tensors per frame."""
        frames = self._to_frame_major(x, n_frames=n_frames)
        bsz, _, _, height, width = frames.shape
        flat = frames.reshape(bsz * n_frames, self.frame_channels, height, width)
        with torch.no_grad():
            z = self.autoencoder.encode(flat)
        return z.reshape(bsz, n_frames, self.latent_channels, z.shape[-2], z.shape[-1])

    def _decode_future_latents(self, z_future: torch.Tensor) -> torch.Tensor:
        """Decode packed future latents back to packed RGB frame tensors."""
        if z_future.ndim != 4:
            raise ValueError(f"Expected rank-4 latent tensor, got shape={tuple(z_future.shape)}")
        expected_channels = self.n_future_frames * self.latent_channels
        if z_future.shape[1] != expected_channels:
            raise ValueError(
                "Unexpected latent channels for future decode: "
                f"expected {expected_channels}, got {z_future.shape[1]}"
            )
        bsz = z_future.shape[0]
        flat = z_future.reshape(
            bsz * self.n_future_frames,
            self.latent_channels,
            z_future.shape[-2],
            z_future.shape[-1],
        )
        with torch.no_grad():
            recon = self.autoencoder.decode(flat)
        recon = recon.reshape(
            bsz,
            self.n_future_frames,
            self.frame_channels,
            recon.shape[-2],
            recon.shape[-1],
        )
        return recon.reshape(
            bsz,
            self.n_future_frames * self.frame_channels,
            recon.shape[-2],
            recon.shape[-1],
        )

    def _build_condition(
        self,
        *,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble action and time conditioning vectors for FiLM blocks."""
        if past_actions is None:
            if self.n_past_actions != 0:
                raise ValueError("past_actions must be provided when n_past_actions > 0")
            past_actions = future_actions[:, :0]

        future_embed = self.action_embed(future_actions)
        past_embed = self.action_embed(past_actions)
        action_cond = torch.cat([past_embed, future_embed], dim=1).reshape(future_embed.shape[0], -1)

        if t.ndim == 0:
            t = t.repeat(future_embed.shape[0])
        elif t.ndim == 1:
            if t.shape[0] == 1:
                t = t.repeat(future_embed.shape[0])
            elif t.shape[0] != future_embed.shape[0]:
                raise ValueError(f"Invalid time batch shape {tuple(t.shape)}")
        elif t.ndim == 4 and t.shape[0] == future_embed.shape[0]:
            t = t.reshape(future_embed.shape[0])
        else:
            raise ValueError(f"Unsupported time tensor shape: {tuple(t.shape)}")
        t_embed = self.time_mlp(self.time_embedding(t))
        return torch.cat([action_cond, t_embed], dim=1)

    def _predict_velocity(
        self,
        *,
        zt: torch.Tensor,
        z_past_ctx: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict latent velocity for the current noised future latent state."""
        cond = self._build_condition(future_actions=future_actions, past_actions=past_actions, t=t)
        flow_in = torch.cat([zt, z_past_ctx], dim=1)
        return self.flow(flow_in, cond)

    def compute_flow_matching_loss(
        self,
        obs_stack: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None,
        next_obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute conditional flow-matching loss in latent space."""
        z_past = self._encode_frames(obs_stack, n_frames=self.n_past_frames)
        z_future = self._encode_frames(next_obs, n_frames=self.n_future_frames)
        z_past_ctx = z_past.reshape(
            z_past.shape[0],
            self.n_past_frames * self.latent_channels,
            z_past.shape[-2],
            z_past.shape[-1],
        )
        x1 = z_future.reshape(
            z_future.shape[0],
            self.n_future_frames * self.latent_channels,
            z_future.shape[-2],
            z_future.shape[-1],
        )
        # CFM target path convention:
        #   z_t = (1 - t) * z_1 + t * z_0, where z_1 is data latent and z_0 is Gaussian noise.
        z0 = torch.randn_like(x1)
        z1 = x1
        t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)
        t_view = t[:, None, None, None]
        zt = (1.0 - t_view) * z1 + t_view * z0
        target_velocity = z0 - z1
        pred_velocity = self._predict_velocity(
            zt=zt,
            z_past_ctx=z_past_ctx,
            future_actions=future_actions,
            past_actions=past_actions,
            t=t,
        )
        flow_loss = F.mse_loss(pred_velocity, target_velocity, reduction="mean")
        return {"loss": flow_loss, "flow_loss": flow_loss}

    def _sample_latent_future(
        self,
        obs_stack: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None = None,
        *,
        sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample future latents with fixed-step Euler integration of predicted velocity."""
        steps = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        if steps <= 0:
            raise ValueError("sampling_steps must be > 0")
        z_past = self._encode_frames(obs_stack, n_frames=self.n_past_frames)
        z_past_ctx = z_past.reshape(
            z_past.shape[0],
            self.n_past_frames * self.latent_channels,
            z_past.shape[-2],
            z_past.shape[-1],
        )
        latent_shape = (
            z_past.shape[0],
            self.n_future_frames * self.latent_channels,
            z_past.shape[-2],
            z_past.shape[-1],
        )
        z = torch.randn(
            latent_shape,
            generator=generator,
            device=obs_stack.device,
            dtype=obs_stack.dtype,
        )
        dt = 1.0 / float(steps)
        # Training uses z_t = (1-t)z1 + tz0, so generation from noise (t=1) to data (t=0)
        # integrates the learned velocity field backward in time.
        for idx in range(steps):
            t_value = 1.0 - (idx + 0.5) * dt
            t = torch.full((latent_shape[0],), t_value, device=z.device, dtype=z.dtype)
            velocity = self._predict_velocity(
                zt=z,
                z_past_ctx=z_past_ctx,
                future_actions=future_actions,
                past_actions=past_actions,
                t=t,
            )
            z = z - dt * velocity
        return z

    def sample_future(
        self,
        obs_stack: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None = None,
        *,
        sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample and decode future RGB frames conditioned on past frames/actions."""
        z_future = self._sample_latent_future(
            obs_stack=obs_stack,
            future_actions=future_actions,
            past_actions=past_actions,
            sampling_steps=sampling_steps,
            generator=generator,
        )
        return self._decode_future_latents(z_future)

    def forward(
        self,
        obs_stack: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default inference path that samples decoded future frames."""
        return self.sample_future(
            obs_stack=obs_stack,
            future_actions=future_actions,
            past_actions=past_actions,
            sampling_steps=self.sampling_steps,
        )
