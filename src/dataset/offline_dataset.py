from __future__ import annotations

"""Dataset and DataLoader utilities for sharded offline sequences."""

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import DataConfig
from src.utils.io import read_json


@dataclass
class Manifest:
    """In-memory representation of the dataset manifest.json."""
    game: str
    steps: int
    seq_len: int
    shards: List[Dict[str, Any]]
    total: int


class OfflineDataset(Dataset):
    """Read sharded NPY offline dataset via memory-mapped arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        n_past_frames: int,
        n_past_actions: int,
        n_future_frames: int,
    ):
        """Load the manifest and build shard index."""
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        raw = read_json(manifest_path)
        if "seq_len" not in raw:
            raise KeyError("manifest.json missing required key 'seq_len'")
        self.manifest = Manifest(
            game=raw.get("game", ""),
            steps=raw.get("steps", 0),
            seq_len=int(raw.get("seq_len", 0)),
            shards=raw.get("shards", []),
            total=raw.get("total", 0),
        )
        self.seq_len = int(self.manifest.seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")

        self.n_past_frames = int(n_past_frames)
        self.n_past_actions = int(n_past_actions)
        self.n_future_frames = int(n_future_frames)
        if self.n_past_frames <= 0:
            raise ValueError("n_past_frames must be > 0")
        if self.n_future_frames <= 0:
            raise ValueError("n_future_frames must be > 0")
        if self.n_past_actions < 0:
            raise ValueError("n_past_actions must be >= 0")
        if self.n_past_actions > self.n_past_frames - 1:
            raise ValueError("n_past_actions must be <= n_past_frames - 1")
        if self.n_past_frames + self.n_future_frames > self.seq_len:
            raise ValueError(
                "n_past_frames + n_future_frames must be <= seq_len "
                f"(got {self.n_past_frames + self.n_future_frames} > {self.seq_len})"
            )

        self.shard_entries = list(self.manifest.shards)
        self.shard_sizes = [int(s["count"]) for s in self.shard_entries]
        self.cum_sizes = np.cumsum(self.shard_sizes).tolist()

        self._cache_index = None
        self._cache = None

    def __len__(self) -> int:
        """Total number of sequences in the dataset."""
        return int(self.manifest.total)

    def _load_shard(self, shard_idx: int):
        """Open a shard with memmap handles so samples stream from disk."""
        if self._cache_index == shard_idx and self._cache is not None:
            return self._cache
        shard = self.shard_entries[shard_idx]
        # np.load(..., mmap_mode="r") keeps arrays on disk; slicing pulls only one sample.
        if "frames" not in shard or "actions" not in shard:
            raise KeyError("Shard is missing 'frames' or 'actions' entries")
        cached = {
            "frames": np.load(self.data_dir / shard["frames"], mmap_mode="r"),
            "actions": np.load(self.data_dir / shard["actions"], mmap_mode="r"),
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
        """Return (past_frames, past_actions, future_actions, future_frames, done) tensors."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx, local_idx = self._get_shard_index(index)
        data = self._load_shard(shard_idx)
        frames = data["frames"][local_idx]
        actions = data["actions"][local_idx]
        done = data["done"][local_idx]

        channels_per_frame = 1
        if frames.ndim == 4 and frames.shape[-1] == 3:
            channels_per_frame = 3
            if frames.dtype == np.uint8:
                frames = frames.astype(np.float32) * np.float32(1.0 / 255.0)
            else:
                frames = frames.astype(np.float32, copy=False)
            frames = np.transpose(frames, (0, 3, 1, 2)).reshape(
                frames.shape[0] * channels_per_frame,
                frames.shape[1],
                frames.shape[2],
            )
        elif frames.ndim == 3:
            if frames.dtype == np.uint8:
                frames = frames.astype(np.float32) * np.float32(1.0 / 255.0)
            else:
                frames = frames.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported frame tensor shape in shard sample: {frames.shape}")

        past_channels = self.n_past_frames * channels_per_frame
        future_channels = self.n_future_frames * channels_per_frame
        past_frames = frames[:past_channels]
        future_frames = frames[past_channels : past_channels + future_channels]
        pivot = self.n_past_frames - 1
        # actions[i] corresponds to the transition from frames[i] -> frames[i+1]
        if self.n_past_actions == 0:
            past_actions = actions[:0]
        else:
            past_actions = actions[pivot - self.n_past_actions : pivot]
        future_actions = actions[pivot : pivot + self.n_future_frames]

        return (
            torch.from_numpy(past_frames),
            torch.as_tensor(past_actions, dtype=torch.int64),
            torch.as_tensor(future_actions, dtype=torch.int64),
            torch.from_numpy(future_frames),
            torch.as_tensor(done, dtype=torch.bool),
        )


def create_dataloader(
    data_cfg: DataConfig,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the offline dataset."""
    return _build_loader(
        dataset=OfflineDataset(
            data_cfg.data_dir,
            n_past_frames=data_cfg.n_past_frames,
            n_past_actions=data_cfg.n_past_actions,
            n_future_frames=data_cfg.n_future_frames,
        ),
        data_cfg=data_cfg,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def create_train_val_loaders(
    data_cfg: DataConfig,
    *,
    seed: int,
    drop_last_train: bool = True,
    drop_last_val: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train/val DataLoaders from a single dataset split."""
    dataset = OfflineDataset(
        data_cfg.data_dir,
        n_past_frames=data_cfg.n_past_frames,
        n_past_actions=data_cfg.n_past_actions,
        n_future_frames=data_cfg.n_future_frames,
    )
    if data_cfg.val_ratio <= 0.0:
        return _build_loader(dataset, data_cfg, shuffle=True, drop_last=drop_last_train), None

    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset is empty")
    num_val = int(num_samples * data_cfg.val_ratio)
    if data_cfg.val_ratio > 0.0 and num_val == 0:
        num_val = 1
    if num_val >= num_samples:
        raise ValueError("val_ratio leaves no samples for training")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    val_indices = perm[:num_val].tolist()
    train_indices = perm[num_val:].tolist()

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = _build_loader(
        dataset=train_set,
        data_cfg=data_cfg,
        shuffle=True,
        drop_last=drop_last_train,
    )
    val_loader = _build_loader(
        dataset=val_set,
        data_cfg=data_cfg,
        shuffle=False,
        drop_last=drop_last_val,
    )
    return train_loader, val_loader


def _build_loader(
    dataset: Dataset,
    data_cfg: DataConfig,
    *,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Create a DataLoader for an existing dataset."""
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
