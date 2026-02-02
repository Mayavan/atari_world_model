from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)}")
    return data


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ValueError(f"Invalid override '{override}', empty key")
        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        cursor: Dict[str, Any] = cfg
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            if not isinstance(cursor[part], dict):
                raise ValueError(
                    f"Cannot set '{key}'; '{part}' is not a mapping in config"
                )
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass(frozen=True)
class DataConfig:
    data_dir: str | Path
    game: str
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory: bool
    val_ratio: float


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_data_config(data_cfg: Dict[str, Any]) -> DataConfig:
    required = [
        "data_dir",
        "game",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "val_ratio",
    ]
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise KeyError(f"Missing data config keys: {', '.join(missing)}")

    data_dir = data_cfg["data_dir"]
    if not isinstance(data_dir, (str, Path)):
        raise ValueError("data_dir must be a path or string")

    game = data_cfg["game"]
    if not isinstance(game, str) or not game:
        raise ValueError("game must be a non-empty string")

    batch_size = data_cfg["batch_size"]
    if not _is_int(batch_size) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    num_workers = data_cfg["num_workers"]
    if not _is_int(num_workers) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    pin_memory = data_cfg["pin_memory"]
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a bool")

    persistent_workers = data_cfg["persistent_workers"]
    if not isinstance(persistent_workers, bool):
        raise ValueError("persistent_workers must be a bool")

    prefetch_factor = data_cfg["prefetch_factor"]
    if num_workers == 0:
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor must be null when num_workers=0")
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers>0")
    else:
        if prefetch_factor is None:
            raise ValueError("prefetch_factor must be set when num_workers>0")
        if not _is_int(prefetch_factor) or prefetch_factor < 1:
            raise ValueError("prefetch_factor must be an integer >= 1")

    val_ratio = data_cfg["val_ratio"]
    if not isinstance(val_ratio, (int, float)):
        raise ValueError("val_ratio must be a number")
    val_ratio = float(val_ratio)
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    return DataConfig(
        data_dir=data_dir,
        game=game,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        val_ratio=val_ratio,
    )
