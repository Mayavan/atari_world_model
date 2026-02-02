from __future__ import annotations

"""Dataset and DataLoader utilities for sharded offline Atari transitions."""

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.utils.io import read_json


@dataclass
class Manifest:
    """In-memory representation of the dataset manifest.json."""
    game: str
    steps: int
    shards: List[Dict[str, Any]]
    total: int


class OfflineAtariDataset(Dataset):
    """Read sharded NPY offline dataset via memory-mapped arrays."""

    def __init__(self, data_dir: str | Path):
        """Load the manifest and build shard index."""
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        raw = read_json(manifest_path)
        self.manifest = Manifest(
            game=raw.get("game", ""),
            steps=raw.get("steps", 0),
            shards=raw.get("shards", []),
            total=raw.get("total", 0),
        )

        self.shard_entries = list(self.manifest.shards)
        self.shard_sizes = [int(s["count"]) for s in self.shard_entries]
        self.cum_sizes = np.cumsum(self.shard_sizes).tolist()

        self._cache_index = None
        self._cache = None

    def __len__(self) -> int:
        """Total number of transitions in the dataset."""
        return int(self.manifest.total)

    def _load_shard(self, shard_idx: int):
        """Open a shard with memmap handles so samples stream from disk."""
        if self._cache_index == shard_idx and self._cache is not None:
            return self._cache
        shard = self.shard_entries[shard_idx]
        # np.load(..., mmap_mode="r") keeps arrays on disk; slicing pulls only one sample.
        cached = {
            "obs": np.load(self.data_dir / shard["obs"], mmap_mode="r"),
            "next_obs": np.load(self.data_dir / shard["next_obs"], mmap_mode="r"),
            "action": np.load(self.data_dir / shard["action"], mmap_mode="r"),
            "done": np.load(self.data_dir / shard["done"], mmap_mode="r"),
        }
        # Memmaps are lightweight handles; replacing the cache lets GC close old files naturally.
        self._cache_index = shard_idx
        self._cache = cached
        return cached

    def _get_shard_index(self, index: int) -> Tuple[int, int]:
        """Map a global index to (shard_idx, local_idx)."""
        shard_idx = bisect.bisect_right(self.cum_sizes, index)
        prev = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = index - prev
        return shard_idx, local_idx

    def __getitem__(self, index: int):
        """Return (obs_stack, action, next_obs, done) as torch tensors."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx, local_idx = self._get_shard_index(index)
        data = self._load_shard(shard_idx)
        obs = data["obs"][local_idx]
        next_obs = data["next_obs"][local_idx]
        action = data["action"][local_idx]
        done = data["done"][local_idx]

        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) * np.float32(1.0 / 255.0)
        else:
            obs = obs.astype(np.float32, copy=False)
        if next_obs.dtype == np.uint8:
            next_obs = next_obs.astype(np.float32) * np.float32(1.0 / 255.0)
        else:
            next_obs = next_obs.astype(np.float32, copy=False)

        return (
            torch.from_numpy(obs),
            torch.as_tensor(action, dtype=torch.int64),
            torch.from_numpy(next_obs),
            torch.as_tensor(done, dtype=torch.bool),
        )


def create_dataloader(
    data_cfg: DataConfig,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the offline Atari dataset."""
    dataset = OfflineAtariDataset(data_cfg.data_dir)
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=drop_last,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
    )

