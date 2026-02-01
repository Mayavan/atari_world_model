from __future__ import annotations

"""Loss and metric helpers."""

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((pred - target) ** 2)


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss with configurable delta."""
    return torch.nn.functional.huber_loss(pred, target, delta=delta)
