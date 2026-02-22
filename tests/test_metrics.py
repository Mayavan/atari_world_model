from pathlib import Path

import pytest
import torch

from src.metrics.image_quality import compute_image_quality_metrics, compute_mse_psnr
from src.metrics.rollout import save_rollout_metric_plot


def test_compute_image_quality_metrics_identity() -> None:
    image = torch.rand(2, 1, 32, 32)
    metrics = compute_image_quality_metrics(pred=image, target=image)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["psnr"] > 80.0
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-6)


def test_compute_mse_psnr_for_2d_inputs() -> None:
    pred = torch.zeros(16, 16)
    target = torch.ones(16, 16)
    mse, psnr = compute_mse_psnr(pred=pred, target=target)
    assert mse == pytest.approx(1.0, rel=1e-6)
    assert psnr == pytest.approx(0.0, abs=1e-6)


def test_save_rollout_metric_plot(tmp_path: Path) -> None:
    out_path = tmp_path / "rollout_metrics.png"
    result = save_rollout_metric_plot(
        horizons=[1, 2, 3],
        mse_values=[0.2, 0.4, 0.6],
        psnr_values=[20.0, 17.0, 14.0],
        out_path=out_path,
        title="Rollout Metrics",
    )
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
