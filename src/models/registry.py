from __future__ import annotations

"""Model registry and construction helpers."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.world_model import WorldModel

DEFAULT_MODEL_TYPE = "latent_flow_world_model"
_SUPPORTED_MODEL_TYPES = {DEFAULT_MODEL_TYPE}


def resolve_model_type(model_type: str | None) -> str:
    """Normalize configured model type to a canonical registry key."""
    if model_type is None:
        return DEFAULT_MODEL_TYPE
    normalized = str(model_type).strip().lower()
    if normalized not in _SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_MODEL_TYPES))
        raise ValueError(f"Unsupported model.type '{model_type}'. Supported: {supported}")
    return normalized


def _extract_world_model_kwargs(
    model_cfg: dict[str, Any],
    *,
    n_past_frames: int,
    n_past_actions: int,
    n_future_frames: int,
    require_checkpoint: bool,
) -> dict[str, Any]:
    action_embed_dim = int(model_cfg.get("action_embed_dim", 64))
    width_mult = float(model_cfg.get("width_mult", 1.0))
    frame_channels = int(model_cfg.get("frame_channels", 3))
    time_embed_dim = int(model_cfg.get("time_embed_dim", 128))
    sampling_steps = int(model_cfg.get("sampling_steps", 16))
    autoencoder_checkpoint_raw = str(model_cfg.get("autoencoder_checkpoint", "")).strip()
    autoencoder_model_cfg = model_cfg.get("autoencoder_model_cfg")

    if require_checkpoint and not autoencoder_checkpoint_raw:
        raise ValueError(
            "model.autoencoder_checkpoint must be set for latent_flow_world_model training."
        )
    if action_embed_dim <= 0:
        raise ValueError("model.action_embed_dim must be > 0")
    if width_mult <= 0.0:
        raise ValueError("model.width_mult must be > 0")
    if frame_channels != 3:
        raise ValueError("model.frame_channels must be 3 for RGB latent flow training.")
    if time_embed_dim <= 0:
        raise ValueError("model.time_embed_dim must be > 0")
    if sampling_steps <= 0:
        raise ValueError("model.sampling_steps must be > 0")
    if autoencoder_model_cfg is None and not autoencoder_checkpoint_raw:
        raise ValueError(
            "model.autoencoder_model_cfg missing from checkpoint config and no "
            "model.autoencoder_checkpoint provided."
        )

    return {
        "n_past_frames": int(n_past_frames),
        "n_past_actions": int(n_past_actions),
        "n_future_frames": int(n_future_frames),
        "frame_channels": frame_channels,
        "action_embed_dim": action_embed_dim,
        "time_embed_dim": time_embed_dim,
        "sampling_steps": sampling_steps,
        "width_mult": width_mult,
        "autoencoder_checkpoint": autoencoder_checkpoint_raw or None,
        "autoencoder_model_cfg": autoencoder_model_cfg,
    }


def _load_autoencoder_model_cfg(autoencoder_checkpoint: str | Path) -> dict[str, Any]:
    ckpt_path = Path(autoencoder_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ckpt_path}")
    autoencoder_ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = autoencoder_ckpt.get("model_cfg")
    if not isinstance(model_cfg, dict):
        raise ValueError(
            f"Autoencoder checkpoint {ckpt_path} is missing dict model_cfg metadata."
        )
    if str(model_cfg.get("type", "conv_autoencoder")).strip().lower() != "conv_autoencoder":
        raise ValueError(
            "Expected autoencoder checkpoint model_cfg.type='conv_autoencoder', "
            f"got '{model_cfg.get('type')}'."
        )
    return dict(model_cfg)


def build_model_from_config(
    *,
    model_cfg: dict[str, Any],
    num_actions: int,
    n_past_frames: int,
    n_past_actions: int,
    n_future_frames: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build model from config and return model + normalized checkpoint config."""
    model_type = resolve_model_type(model_cfg.get("type"))

    model_kwargs = _extract_world_model_kwargs(
        model_cfg,
        n_past_frames=n_past_frames,
        n_past_actions=n_past_actions,
        n_future_frames=n_future_frames,
        require_checkpoint=True,
    )
    autoencoder_checkpoint = str(model_kwargs["autoencoder_checkpoint"])
    autoencoder_model_cfg = _load_autoencoder_model_cfg(autoencoder_checkpoint)
    model_kwargs["autoencoder_model_cfg"] = autoencoder_model_cfg
    model = WorldModel(num_actions=num_actions, **model_kwargs)

    ckpt_model_cfg = {
        "type": model_type,
        **model_kwargs,
        "autoencoder_checkpoint": autoencoder_checkpoint,
        "autoencoder_model_cfg": model.autoencoder_model_cfg,
    }
    return model, ckpt_model_cfg


def build_model_from_checkpoint_cfg(
    *,
    model_cfg: dict[str, Any],
    num_actions: int,
) -> nn.Module:
    """Build model from model_cfg stored inside a checkpoint."""
    resolve_model_type(model_cfg.get("type"))
    model_kwargs = _extract_world_model_kwargs(
        model_cfg,
        n_past_frames=int(model_cfg.get("n_past_frames", 4)),
        n_past_actions=int(model_cfg.get("n_past_actions", 0)),
        n_future_frames=int(model_cfg.get("n_future_frames", 1)),
        require_checkpoint=False,
    )
    return WorldModel(num_actions=num_actions, **model_kwargs)
