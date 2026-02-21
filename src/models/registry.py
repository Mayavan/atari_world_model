from __future__ import annotations

"""Model registry and construction helpers."""

from typing import Any

import torch.nn as nn

from src.models.world_model import WorldModel

DEFAULT_MODEL_TYPE = "pixel_world_model"
_SUPPORTED_MODEL_TYPES = {DEFAULT_MODEL_TYPE, "world_model"}


def resolve_model_type(model_type: str | None) -> str:
    """Normalize configured model type to a canonical registry key."""
    if model_type is None:
        return DEFAULT_MODEL_TYPE
    normalized = str(model_type).strip().lower()
    if normalized == "world_model":
        return DEFAULT_MODEL_TYPE
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
) -> dict[str, Any]:
    action_embed_dim = int(model_cfg.get("action_embed_dim", 64))
    width_mult = float(model_cfg.get("width_mult", 1.0))
    if action_embed_dim <= 0:
        raise ValueError("model.action_embed_dim must be > 0")
    if width_mult <= 0.0:
        raise ValueError("model.width_mult must be > 0")
    return {
        "n_past_frames": int(n_past_frames),
        "n_past_actions": int(n_past_actions),
        "n_future_frames": int(n_future_frames),
        "action_embed_dim": action_embed_dim,
        "width_mult": width_mult,
    }


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
    if model_type != DEFAULT_MODEL_TYPE:
        raise ValueError(f"Unsupported resolved model type '{model_type}'")

    model_kwargs = _extract_world_model_kwargs(
        model_cfg,
        n_past_frames=n_past_frames,
        n_past_actions=n_past_actions,
        n_future_frames=n_future_frames,
    )
    model = WorldModel(num_actions=num_actions, **model_kwargs)

    ckpt_model_cfg = {
        "type": model_type,
        **model_kwargs,
    }
    return model, ckpt_model_cfg


def build_model_from_checkpoint_cfg(
    *,
    model_cfg: dict[str, Any],
    num_actions: int,
) -> nn.Module:
    """Build model from model_cfg stored inside a checkpoint."""
    model_type = resolve_model_type(model_cfg.get("type"))
    if model_type != DEFAULT_MODEL_TYPE:
        raise ValueError(f"Unsupported resolved model type '{model_type}'")

    model_kwargs = _extract_world_model_kwargs(
        model_cfg,
        n_past_frames=int(model_cfg.get("n_past_frames", 4)),
        n_past_actions=int(model_cfg.get("n_past_actions", 0)),
        n_future_frames=int(model_cfg.get("n_future_frames", 1)),
    )
    return WorldModel(num_actions=num_actions, **model_kwargs)
