import json
from pathlib import Path

import numpy as np
import torch

from src.dataset.offline_dataset import OfflineAtariDataset


def test_dataset_shapes_and_dtypes(tmp_path: Path):
    obs = np.random.randint(0, 256, size=(10, 4, 84, 84), dtype=np.uint8)
    action = np.random.randint(0, 4, size=(10,), dtype=np.int64)
    next_obs = np.random.randint(0, 256, size=(10, 1, 84, 84), dtype=np.uint8)
    done = np.random.rand(10) > 0.5

    obs_path = tmp_path / "shard_000000_obs.npy"
    next_path = tmp_path / "shard_000000_next_obs.npy"
    action_path = tmp_path / "shard_000000_action.npy"
    done_path = tmp_path / "shard_000000_done.npy"
    np.save(obs_path, obs, allow_pickle=False)
    np.save(next_path, next_obs, allow_pickle=False)
    np.save(action_path, action, allow_pickle=False)
    np.save(done_path, done, allow_pickle=False)

    manifest = {
        "game": "Pong",
        "steps": 10,
        "shards": [
            {
                "id": 0,
                "count": 10,
                "obs": obs_path.name,
                "next_obs": next_path.name,
                "action": action_path.name,
                "done": done_path.name,
            }
        ],
        "total": 10,
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    ds = OfflineAtariDataset(tmp_path)
    o, a, n, d = ds[0]

    assert o.shape == (4, 84, 84)
    assert n.shape == (1, 84, 84)
    assert o.dtype == torch.float32
    assert n.dtype == torch.float32
    assert a.dtype == torch.int64
    assert d.dtype == torch.bool

    scale = np.float32(1.0 / 255.0)
    expected_obs = torch.from_numpy(obs[0].astype(np.float32) * scale)
    expected_next = torch.from_numpy(next_obs[0].astype(np.float32) * scale)
    assert torch.allclose(o, expected_obs)
    assert torch.allclose(n, expected_next)
