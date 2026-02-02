from __future__ import annotations

"""Helpers for training and validation utilities."""

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import wandb

from src.utils.metrics import huber, mse


def parse_train_cli(argv: list[str]) -> tuple[Path, list[str]]:
    """Parse CLI args into (config_path, overrides)."""
    if len(argv) >= 2 and argv[1].endswith((".yaml", ".yml")):
        return Path(argv[1]), argv[2:]
    return Path("config.yaml"), argv[1:]


def run_validation(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    loss_name: str,
    delta: float,
) -> tuple[float, np.ndarray | None]:
    """Evaluate the model on the validation loader and return average loss and viz."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    viz_image = None
    with torch.no_grad():
        for obs, action, next_obs, _ in loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            pred = model(obs, action)
            if loss_name == "huber":
                loss = huber(pred, next_obs, delta=delta)
            else:
                loss = mse(pred, next_obs)
            if viz_image is None:
                viz_image = build_viz_image(obs, pred)
            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    model.train()
    if total_samples == 0:
        return 0.0, viz_image
    return total_loss / total_samples, viz_image


def build_viz_image(obs: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """Create a grayscale strip of input frames plus predicted frame."""
    obs_np = obs.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    frames = [np.clip(f, 0.0, 1.0) for f in obs_np[0]]
    frames.append(np.clip(pred_np[0, 0], 0.0, 1.0))
    return np.concatenate(frames, axis=1)


def log_prediction_image(
    *,
    tag: str,
    step: int,
    obs: torch.Tensor,
    pred: torch.Tensor,
    image_dir: Path,
    wandb_run,
) -> None:
    """Save and optionally log a visualization image."""
    image = build_viz_image(obs, pred)
    save_image(tag=tag, step=step, image=image, image_dir=image_dir, wandb_run=wandb_run)


def save_image(
    *,
    tag: str,
    step: int,
    image: np.ndarray,
    image_dir: Path,
    wandb_run,
) -> None:
    """Write a grayscale PNG and log to W&B if enabled."""
    path = image_dir / f"{tag}_step_{step:08d}.png"
    imageio.imwrite(path, (image * 255.0).astype(np.uint8))
    if wandb_run is not None:
        wandb.log({f"{tag}_viz": wandb.Image(str(path))}, step=step)
