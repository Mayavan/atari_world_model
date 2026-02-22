from __future__ import annotations

"""Plot helpers for rollout metrics."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def save_rollout_metric_plot(
    *,
    horizons: Sequence[int],
    mse_values: Sequence[float],
    psnr_values: Sequence[float],
    out_path: Path,
    title: str,
) -> Path:
    """Save a two-panel plot for rollout MSE/PSNR vs horizon."""
    if len(horizons) != len(mse_values) or len(horizons) != len(psnr_values):
        raise ValueError("horizons, mse_values, and psnr_values must have the same length")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(horizons, mse_values, marker="o")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(title)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(horizons, psnr_values, marker="o", color="tab:green")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
