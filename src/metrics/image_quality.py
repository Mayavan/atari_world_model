from __future__ import annotations

"""Image-quality metrics used by supervised and rollout validation."""

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to (B,C,H,W)."""
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected rank-2/3/4 tensor, got shape={tuple(x.shape)}")


def _prepare_image_pair(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Convert inputs to clamped BCHW tensors and validate shape."""
    range_value = float(data_range)
    pred_bchw = _to_bchw(pred).float().clamp(0.0, range_value)
    target_bchw = _to_bchw(target).float().clamp(0.0, range_value)
    if pred_bchw.shape != target_bchw.shape:
        raise ValueError(
            "pred and target must share shape, "
            f"got {tuple(pred_bchw.shape)} vs {tuple(target_bchw.shape)}"
        )
    return pred_bchw, target_bchw, range_value


def compute_image_quality_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute MSE, PSNR, and SSIM for image tensors in [0, data_range]."""
    pred_bchw, target_bchw, range_value = _prepare_image_pair(
        pred=pred,
        target=target,
        data_range=data_range,
    )
    mse = F.mse_loss(pred_bchw, target_bchw, reduction="mean")
    psnr = peak_signal_noise_ratio(pred_bchw, target_bchw, data_range=range_value)
    ssim = structural_similarity_index_measure(pred_bchw, target_bchw, data_range=range_value)
    return {
        "mse": float(mse.item()),
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
    }


def compute_mse_psnr(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> tuple[float, float]:
    """Compute only MSE and PSNR."""
    pred_bchw, target_bchw, range_value = _prepare_image_pair(
        pred=pred,
        target=target,
        data_range=data_range,
    )
    mse = F.mse_loss(pred_bchw, target_bchw, reduction="mean")
    psnr = peak_signal_noise_ratio(pred_bchw, target_bchw, data_range=range_value)
    return float(mse.item()), float(psnr.item())
