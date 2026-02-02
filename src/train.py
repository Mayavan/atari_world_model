from __future__ import annotations

"""Training loop for the Atari world model with mixed precision support."""

import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from src.config import apply_overrides, load_config, save_config, validate_data_config
from src.dataset.offline_dataset import create_dataloader
from src.envs.atari_wrappers import make_atari_env
from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, init_csv, append_csv, timestamp_dir
from src.utils.metrics import huber, mse
from src.utils.seed import set_seed


def _parse_cli(argv: list[str]) -> tuple[Path, list[str]]:
    if len(argv) >= 2 and argv[1].endswith((".yaml", ".yml")):
        return Path(argv[1]), argv[2:]
    return Path("config.yaml"), argv[1:]


def train(cfg: dict) -> None:
    """Run a simple teacher-forced training loop and save checkpoints."""
    experiment = cfg["experiment"]
    model_cfg = cfg["model"]
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]
    data_cfg = validate_data_config(cfg["data"])
    train_cfg = cfg["train"]

    set_seed(int(experiment["seed"]))

    run_dir_value = str(experiment["run_dir"])
    if run_dir_value:
        run_dir = ensure_dir(run_dir_value)
    else:
        name = str(experiment["name"]).lower()
        run_dir = timestamp_dir("runs", name=name)

    wandb_run = None
    wandb_cfg = experiment["wandb"]
    if wandb_cfg["mode"] != "disabled":
        try:
            wandb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity", None),
                name=str(experiment["name"]),
                mode=wandb_cfg["mode"],
                dir=str(run_dir),
            )
            wandb.config.update(cfg, allow_val_change=False)
        except Exception as e:  # noqa: BLE001
            print(f"W&B init failed, continuing without logging: {e}")
            wandb_run = None

    save_config(cfg, Path(run_dir) / "resolved_config.yaml")

    env = make_atari_env(data_cfg.game, seed=int(experiment["seed"]))
    num_actions = env.action_space.n
    env.close()

    force_cpu = bool(train_cfg.get("cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    use_cuda = device.type == "cuda"
    loader = create_dataloader(data_cfg, shuffle=True, drop_last=True)

    model = WorldModel(num_actions=num_actions, condition_mode=model_cfg["condition"])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(optimizer_cfg["lr"]))
    scaler = GradScaler("cuda" if use_cuda else "cpu", enabled=use_cuda)
    scheduler = None
    if scheduler_cfg["enabled"] and scheduler_cfg["type"] == "onecycle":
        total_steps = len(loader) * int(train_cfg["epochs"])
        max_steps = int(train_cfg["max_steps"])
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
        if total_steps <= 0:
            raise ValueError("OneCycleLR requires at least 1 total step.")
        max_lr = float(scheduler_cfg["max_lr"])
        if max_lr <= 0:
            max_lr = float(optimizer_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=float(scheduler_cfg["pct_start"]),
            div_factor=float(scheduler_cfg["div_factor"]),
            final_div_factor=float(scheduler_cfg["final_div_factor"]),
        )

    ckpt_dir = ensure_dir(Path(run_dir) / "checkpoints")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(metrics_path, ["epoch", "step", "loss"])

    global_step = 0
    model.train()
    stop_early = False
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            obs, action, next_obs, _ = batch
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                pred = model(obs, action)
                if train_cfg["loss"] == "huber":
                    loss = huber(pred, next_obs, delta=float(train_cfg["delta"]))
                else:
                    loss = mse(pred, next_obs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            if global_step % int(train_cfg["log_every"]) == 0:
                pbar.set_postfix({"loss": float(loss.item())})
                append_csv(metrics_path, [epoch, global_step, float(loss.item())])
                print(f"step {global_step} loss {loss.item():.6f}")
                if wandb_run is not None:
                    wandb.log({"loss": float(loss.item()), "epoch": epoch}, step=global_step)
            if int(train_cfg["max_steps"]) > 0 and global_step >= int(train_cfg["max_steps"]):
                stop_early = True
                break

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch}.pt")
        torch.save(ckpt, Path(run_dir) / "ckpt.pt")
        if wandb_run is not None:
            wandb.log({"epoch": epoch}, step=global_step)
        if stop_early:
            break

    print(f"Training complete. Run dir: {run_dir}")
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    """Entry point for CLI."""
    config_path, overrides = _parse_cli(sys.argv)
    cfg = load_config(config_path)
    apply_overrides(cfg, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
