from __future__ import annotations

"""Training loop for a convolutional autoencoder on offline Procgen frames."""

import sys
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.config import apply_overrides, load_config, save_config, validate_data_config
from src.dataset.offline_dataset import create_train_val_loaders
from src.metrics.image_quality import compute_image_quality_metrics
from src.models.conv_autoencoder import ConvAutoencoder
from src.utils.io import append_csv, ensure_dir, init_csv, timestamp_dir
from src.utils.seed import set_seed
from src.utils.train_utils import parse_train_cli


def _extract_autoencoder_input(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Use future-frame tensor as the reconstruction target."""
    if len(batch) != 5:
        raise ValueError(f"Expected batch tuple of size 5, got {len(batch)}")
    target = batch[3]
    if target.ndim != 4:
        raise ValueError(f"Expected rank-4 target tensor, got shape={tuple(target.shape)}")
    if not target.dtype.is_floating_point:
        raise ValueError(f"Expected float target tensor, got dtype={target.dtype}")
    return target


def compute_autoencoder_loss(
    *,
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute reconstruction loss using fixed Huber loss."""
    return F.huber_loss(recon, target, reduction="mean", delta=1.0)


def _compute_latent_variance(z: torch.Tensor) -> torch.Tensor:
    """Compute differentiable mean latent variance across the batch."""
    z_flat = z.view(z.shape[0], -1)
    return z_flat.var(dim=1, unbiased=False).mean()


def compute_autoencoder_losses(
    *,
    recon: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    var_reg_lambda: float,
    var_target: float,
) -> dict[str, torch.Tensor]:
    """Compute full training objective including variance-target regularization."""
    recon_loss = compute_autoencoder_loss(recon=recon, target=target)
    latent_var = _compute_latent_variance(z)
    target_var = float(var_target)
    var_reg_loss = float(var_reg_lambda) * (latent_var - target_var).pow(2)
    loss = recon_loss + var_reg_loss
    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "var_reg_loss": var_reg_loss,
        "latent_var": latent_var,
    }


def _compute_latent_stats(z: torch.Tensor) -> tuple[float, float]:
    """Return (variance, mean L2 norm) for latent tensors."""
    z_flat = z.detach().view(z.shape[0], -1)
    latent_var = _compute_latent_variance(z.detach())
    latent_mean_norm = z_flat.norm(dim=1).mean()
    return float(latent_var.item()), float(latent_mean_norm.item())


def _build_recon_grid(
    *,
    target: torch.Tensor,
    recon: torch.Tensor,
    max_samples: int = 4,
) -> np.ndarray | None:
    """Create a 2x2 grid of sample strips: [target channels | recon channels] per tile."""
    target_np = target.detach().cpu().float().clamp(0.0, 1.0).numpy()
    recon_np = recon.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if target_np.shape != recon_np.shape:
        raise ValueError(
            f"target/recon shape mismatch: {tuple(target_np.shape)} vs {tuple(recon_np.shape)}"
        )
    count = min(int(target_np.shape[0]), int(max_samples), 4)
    if count <= 0:
        return None

    tiles: list[np.ndarray] = []
    for idx in range(count):
        tgt_strip = np.concatenate([target_np[idx, c] for c in range(target_np.shape[1])], axis=1)
        rec_strip = np.concatenate([recon_np[idx, c] for c in range(recon_np.shape[1])], axis=1)
        tiles.append(np.concatenate([tgt_strip, rec_strip], axis=1))

    # Fill to 4 tiles so the layout is always 2x2.
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))

    top_row = np.concatenate([tiles[0], tiles[1]], axis=1)
    bottom_row = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


def _log_recon_grid(
    *,
    tag: str,
    step: int,
    image: np.ndarray,
    image_dir: Path,
    wandb_run,
) -> None:
    path = image_dir / f"{tag}_step_{step:08d}.png"
    imageio.imwrite(path, (image * 255.0).astype(np.uint8))
    if wandb_run is not None:
        wandb.log({tag: wandb.Image(str(path))}, step=step)


def run_validation(
    *,
    model: ConvAutoencoder,
    loader: Iterable,
    device: torch.device,
    var_reg_lambda: float,
    var_target: float,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Evaluate validation reconstruction quality."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_var_reg_loss = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_latent_var = 0.0
    total_latent_mean_norm = 0.0
    total_samples = 0
    grid = None

    with torch.no_grad():
        for batch in loader:
            target = _extract_autoencoder_input(batch).to(device)
            z = model.encode(target)
            recon = model.decode(z)
            losses = compute_autoencoder_losses(
                recon=recon,
                target=target,
                z=z,
                var_reg_lambda=var_reg_lambda,
                var_target=var_target,
            )
            latent_var, latent_mean_norm = _compute_latent_stats(z)
            metrics = compute_image_quality_metrics(pred=recon, target=target, data_range=1.0)
            if grid is None:
                grid = _build_recon_grid(target=target, recon=recon)

            batch_size = int(target.shape[0])
            total_loss += float(losses["loss"].item()) * batch_size
            total_recon_loss += float(losses["recon_loss"].item()) * batch_size
            total_var_reg_loss += float(losses["var_reg_loss"].item()) * batch_size
            total_mse += metrics["mse"] * batch_size
            total_psnr += metrics["psnr"] * batch_size
            total_ssim += metrics["ssim"] * batch_size
            total_latent_var += latent_var * batch_size
            total_latent_mean_norm += latent_mean_norm * batch_size
            total_samples += batch_size

    if was_training:
        model.train()
    if total_samples == 0:
        return {
            "loss": 0.0,
            "recon_loss": 0.0,
            "var_reg_loss": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "latent_var": 0.0,
            "latent_mean_norm": 0.0,
        }, grid
    return {
        "loss": total_loss / total_samples,
        "recon_loss": total_recon_loss / total_samples,
        "var_reg_loss": total_var_reg_loss / total_samples,
        "mse": total_mse / total_samples,
        "psnr": total_psnr / total_samples,
        "ssim": total_ssim / total_samples,
        "latent_var": total_latent_var / total_samples,
        "latent_mean_norm": total_latent_mean_norm / total_samples,
    }, grid


def _resolve_run_dir(*, experiment_cfg: dict, resume_path: Path | None) -> Path:
    run_dir_value = str(experiment_cfg.get("run_dir", ""))
    if run_dir_value:
        return ensure_dir(run_dir_value)
    if resume_path is not None:
        candidate = resume_path.parent
        if candidate.name == "checkpoints":
            candidate = candidate.parent
        return ensure_dir(candidate)
    name = str(experiment_cfg.get("name", "autoencoder")).lower()
    return timestamp_dir("runs", name=name)


def _build_model_from_cfg(
    *,
    model_cfg: dict,
    in_channels: int,
) -> tuple[ConvAutoencoder, dict[str, object]]:
    model = ConvAutoencoder(
        in_channels=in_channels,
        image_height=int(model_cfg.get("image_height", 84)),
        image_width=int(model_cfg.get("image_width", 84)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        downsample_factor=int(model_cfg.get("downsample_factor", 8)),
        decoder_type=str(model_cfg.get("decoder_type", "upsample_conv")),
    )
    ckpt_cfg: dict[str, object] = {
        "type": "conv_autoencoder",
        "in_channels": model.in_channels,
        "image_height": model.image_height,
        "image_width": model.image_width,
        "base_channels": model.base_channels,
        "latent_channels": model.latent_channels,
        "downsample_factor": model.downsample_factor,
        "decoder_type": model.decoder_type,
    }
    return model, ckpt_cfg


def _init_csv_if_needed(*, path: Path, headers: list[str], resume_path: Path | None) -> None:
    if resume_path is not None and path.exists():
        return
    init_csv(path, headers)


def train(cfg: dict) -> None:
    """Run autoencoder training with validation and checkpointing."""
    experiment = cfg["experiment"]
    data_cfg = validate_data_config(cfg["data"])
    model_cfg = cfg.get("model", {})
    train_cfg = cfg["train"]

    set_seed(int(experiment["seed"]))

    resume_raw = str(train_cfg.get("resume", "")).strip()
    resume_path = Path(resume_raw) if resume_raw else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    run_dir = _resolve_run_dir(experiment_cfg=experiment, resume_path=resume_path)
    save_config(cfg, Path(run_dir) / "resolved_config.yaml")

    wandb_run = None
    wandb_cfg = experiment["wandb"]
    if wandb_cfg["mode"] != "disabled":
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity", None),
            name=str(experiment["name"]),
            mode=wandb_cfg["mode"],
            dir=str(run_dir),
        )
        wandb.config.update(cfg, allow_val_change=False)

    force_cpu = bool(train_cfg.get("cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    train_loader, val_loader = create_train_val_loaders(
        data_cfg,
        seed=int(experiment["seed"]),
        drop_last_train=True,
        drop_last_val=False,
    )

    if len(train_loader) == 0:
        raise ValueError("Train loader is empty; cannot infer autoencoder input channels.")
    sample_batch = next(iter(train_loader))
    inferred_target_channels = int(_extract_autoencoder_input(sample_batch).shape[1])

    model_in_channels = int(model_cfg.get("in_channels", inferred_target_channels))
    if model_in_channels != inferred_target_channels:
        raise ValueError(
            "model.in_channels must match inferred dataset target channels for autoencoder training "
            f"(got {model_in_channels} vs {inferred_target_channels})"
        )
    model, ckpt_model_cfg = _build_model_from_cfg(model_cfg=model_cfg, in_channels=model_in_channels)
    model.to(device)

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_enabled = bool(scheduler_cfg.get("enabled", True))
    scheduler_type = str(scheduler_cfg.get("type", "onecycle")).strip().lower()
    if not (scheduler_enabled and scheduler_type == "onecycle"):
        raise ValueError("Scheduler must be enabled with type 'onecycle'.")
    max_lr = float(scheduler_cfg.get("max_lr", train_cfg.get("lr", 3e-4)))
    if max_lr <= 0.0:
        raise ValueError("scheduler.max_lr must be > 0")

    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if weight_decay < 0.0:
        raise ValueError("train.weight_decay must be >= 0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    if grad_clip_norm < 0.0:
        raise ValueError("train.grad_clip_norm must be >= 0")
    var_reg_lambda = float(train_cfg.get("var_reg_lambda", 0.0))
    if var_reg_lambda < 0.0:
        raise ValueError("train.var_reg_lambda must be >= 0")
    var_target = float(train_cfg.get("var_target", 1.0))
    if var_target <= 0.0:
        raise ValueError("train.var_target must be > 0")

    ckpt_dir = ensure_dir(Path(run_dir) / "checkpoints")
    image_dir = ensure_dir(Path(run_dir) / "images")
    metrics_path = Path(run_dir) / "metrics.csv"
    _init_csv_if_needed(
        path=metrics_path,
        headers=["epoch", "step", "loss", "recon_loss", "var_reg_loss", "latent_var", "lr"],
        resume_path=resume_path,
    )
    val_metrics_path = Path(run_dir) / "val_metrics.csv"
    if val_loader is not None and int(train_cfg.get("val_every_steps", 0)) > 0:
        _init_csv_if_needed(
            path=val_metrics_path,
            headers=[
                "step",
                "val_loss",
                "val_recon_loss",
                "val_var_reg_loss",
                "val_mse",
                "val_psnr",
                "val_ssim",
                "val_latent_var",
                "val_latent_mean_norm",
            ],
            resume_path=resume_path,
        )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_val_loss = float(resume_ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from {resume_path} at epoch={start_epoch}, step={global_step}")

    total_steps = len(train_loader) * int(train_cfg["epochs"])
    max_steps = int(train_cfg.get("max_steps", 0))
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    if total_steps <= 0:
        raise ValueError("No train steps configured; check epochs/max_steps/dataset size.")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
        div_factor=float(scheduler_cfg.get("div_factor", 25.0)),
        final_div_factor=float(scheduler_cfg.get("final_div_factor", 10000.0)),
    )
    if resume_ckpt is not None:
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            print("Resume checkpoint missing scheduler state; LR schedule restarted.")

    model.train()
    stop_early = False
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        pbar = tqdm(train_loader, desc=f"ae epoch {epoch}")
        for batch in pbar:
            target = _extract_autoencoder_input(batch).to(device)
            if target.shape[1] != model.in_channels:
                raise ValueError(
                    "Batch channels mismatch with model.in_channels: "
                    f"{target.shape[1]} vs {model.in_channels}"
                )

            optimizer.zero_grad(set_to_none=True)
            z = model.encode(target)
            recon = model.decode(z)
            losses = compute_autoencoder_losses(
                recon=recon,
                target=target,
                z=z,
                var_reg_lambda=var_reg_lambda,
                var_target=var_target,
            )
            losses["loss"].backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % int(train_cfg.get("log_every", 100)) == 0:
                loss_value = float(losses["loss"].item())
                recon_loss_value = float(losses["recon_loss"].item())
                var_reg_loss_value = float(losses["var_reg_loss"].item())
                latent_var_value = float(losses["latent_var"].item())
                lr_value = float(optimizer.param_groups[0]["lr"])
                pbar.set_postfix({"loss": loss_value, "recon": recon_loss_value, "var": latent_var_value})
                append_csv(
                    metrics_path,
                    [
                        epoch,
                        global_step,
                        loss_value,
                        recon_loss_value,
                        var_reg_loss_value,
                        latent_var_value,
                        lr_value,
                    ],
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss": loss_value,
                            "recon_loss": recon_loss_value,
                            "var_reg_loss": var_reg_loss_value,
                            "latent_var": latent_var_value,
                            "lr": lr_value,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if val_loader is not None and int(train_cfg.get("val_every_steps", 0)) > 0:
                if global_step % int(train_cfg["val_every_steps"]) == 0:
                    val_metrics, val_grid = run_validation(
                        model=model,
                        loader=val_loader,
                        device=device,
                        var_reg_lambda=var_reg_lambda,
                        var_target=var_target,
                    )
                    append_csv(
                        val_metrics_path,
                        [
                            global_step,
                            val_metrics["loss"],
                            val_metrics["recon_loss"],
                            val_metrics["var_reg_loss"],
                            val_metrics["mse"],
                            val_metrics["psnr"],
                            val_metrics["ssim"],
                            val_metrics["latent_var"],
                            val_metrics["latent_mean_norm"],
                        ],
                    )
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "val_loss": val_metrics["loss"],
                                "val_recon_loss": val_metrics["recon_loss"],
                                "val_var_reg_loss": val_metrics["var_reg_loss"],
                                "val_mse": val_metrics["mse"],
                                "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"],
                                "val_latent_var": val_metrics["latent_var"],
                                "val_latent_mean_norm": val_metrics["latent_mean_norm"],
                            },
                            step=global_step,
                        )
                    if val_grid is not None:
                        _log_recon_grid(
                            tag="val_recon",
                            step=global_step,
                            image=val_grid,
                            image_dir=image_dir,
                            wandb_run=wandb_run,
                        )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_ckpt = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                            "model_cfg": ckpt_model_cfg,
                        }
                        torch.save(best_ckpt, Path(run_dir) / "best_ckpt.pt")

            if max_steps > 0 and global_step >= max_steps:
                stop_early = True
                break

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model_cfg": ckpt_model_cfg,
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch}.pt")
        torch.save(ckpt, Path(run_dir) / "ckpt.pt")
        if wandb_run is not None:
            wandb.log({"epoch": epoch}, step=global_step)
        if stop_early:
            break

    print(f"Autoencoder training complete. Run dir: {run_dir}")

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    """Entry point for CLI."""
    config_path, overrides = parse_train_cli(sys.argv, default_config="config_autoencoder.yaml")
    cfg = load_config(config_path)
    apply_overrides(cfg, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
