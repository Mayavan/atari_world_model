import pytest
import torch

from src.train_autoencoder import (
    _build_recon_grid,
    _compute_latent_stats,
    _compute_latent_variance,
    compute_autoencoder_loss,
    compute_autoencoder_losses,
)


def test_compute_autoencoder_loss_huber_small_error() -> None:
    target = torch.zeros(2, 1, 4, 4)
    recon = torch.ones_like(target) * 0.5
    loss = compute_autoencoder_loss(
        recon=recon,
        target=target,
    )
    # delta=1.0 and |error|=0.5 (< delta): 0.5 * error^2 = 0.125
    assert float(loss) == pytest.approx(0.125, abs=1e-6)


def test_compute_autoencoder_loss_huber_large_error() -> None:
    target = torch.zeros(1, 1, 2, 2)
    recon = torch.ones_like(target) * 2.0
    loss = compute_autoencoder_loss(
        recon=recon,
        target=target,
    )
    # delta=1.0 and |error|=2.0 (>= delta): delta*(|e| - 0.5*delta) = 1.5
    assert float(loss) == pytest.approx(1.5, abs=1e-6)


def test_compute_latent_stats() -> None:
    z = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)
    latent_var, latent_mean_norm = _compute_latent_stats(z)
    assert latent_var == pytest.approx(1.25, abs=1e-6)
    assert latent_mean_norm == pytest.approx((14.0) ** 0.5, abs=1e-6)


def test_compute_latent_variance_matches_stats() -> None:
    z = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)
    latent_var = _compute_latent_variance(z)
    assert float(latent_var) == pytest.approx(1.25, abs=1e-6)


def test_compute_autoencoder_losses_with_variance_regularizer() -> None:
    target = torch.zeros(1, 1, 2, 2)
    recon = torch.ones_like(target) * 0.5
    z = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)
    losses = compute_autoencoder_losses(
        recon=recon,
        target=target,
        z=z,
        var_reg_lambda=0.1,
        var_target=1.0,
    )
    assert float(losses["recon_loss"]) == pytest.approx(0.125, abs=1e-6)
    assert float(losses["latent_var"]) == pytest.approx(1.25, abs=1e-6)
    assert float(losses["var_reg_loss"]) == pytest.approx(0.00625, abs=1e-6)
    assert float(losses["loss"]) == pytest.approx(0.13125, abs=1e-6)


def test_build_recon_grid_layout_is_2x2_for_four_samples() -> None:
    target = torch.zeros(4, 3, 2, 2)
    recon = torch.zeros(4, 3, 2, 2)
    for idx in range(4):
        for channel in range(3):
            target[idx, channel] = (idx * 3 + channel) / 100.0
            recon[idx, channel] = (idx * 3 + channel + 12) / 100.0

    grid = _build_recon_grid(target=target, recon=recon)
    assert grid is not None
    assert grid.shape == (4, 8, 3)

    # Sample 0 target tile starts at top-left and recon starts after width=2.
    assert float(grid[0, 0, 0]) == pytest.approx(0.00, abs=1e-6)
    assert float(grid[0, 0, 1]) == pytest.approx(0.01, abs=1e-6)
    assert float(grid[0, 2, 0]) == pytest.approx(0.12, abs=1e-6)
    # Sample 3 target tile is bottom-right quadrant.
    assert float(grid[2, 4, 0]) == pytest.approx(0.09, abs=1e-6)
