import json
from pathlib import Path

import numpy as np
import torch

from src.dataset.offline_dataset import OfflineDataset


def test_dataset_shapes_and_dtypes(tmp_path: Path):
    seq_len = 8
    frames = np.zeros((10, seq_len, 84, 84), dtype=np.uint8)
    for i in range(seq_len):
        frames[0, i] = i
    actions = np.tile(np.arange(seq_len, dtype=np.int64), (10, 1))
    done = np.random.rand(10, seq_len) > 0.5

    frames_path = tmp_path / "shard_000000_frames.npy"
    actions_path = tmp_path / "shard_000000_actions.npy"
    done_path = tmp_path / "shard_000000_done.npy"
    np.save(frames_path, frames, allow_pickle=False)
    np.save(actions_path, actions, allow_pickle=False)
    np.save(done_path, done, allow_pickle=False)

    manifest = {
        "game": "Pong",
        "steps": 10,
        "seq_len": seq_len,
        "shards": [
            {
                "id": 0,
                "count": 10,
                "frames": frames_path.name,
                "actions": actions_path.name,
                "done": done_path.name,
            }
        ],
        "total": 10,
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    ds = OfflineDataset(
        tmp_path,
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=2,
    )
    o, past_a, future_a, n, d = ds[0]

    assert o.shape == (4, 84, 84)
    assert n.shape == (2, 84, 84)
    assert o.dtype == torch.float32
    assert n.dtype == torch.float32
    assert past_a.dtype == torch.int64
    assert future_a.dtype == torch.int64
    assert d.dtype == torch.bool

    scale = np.float32(1.0 / 255.0)
    expected_obs = torch.from_numpy(frames[0, :4].astype(np.float32) * scale)
    expected_next = torch.from_numpy(frames[0, 4:6].astype(np.float32) * scale)
    assert torch.allclose(o, expected_obs)
    assert torch.allclose(n, expected_next)
    assert torch.equal(past_a, torch.tensor([0, 1, 2]))
    assert torch.equal(future_a, torch.tensor([3, 4]))
