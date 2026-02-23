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


def test_dataset_rgb_frames_decode_to_flattened_channels(tmp_path: Path):
    seq_len = 5
    height = 4
    width = 4
    frames = np.zeros((2, seq_len, height, width, 3), dtype=np.uint8)
    for t in range(seq_len):
        for c in range(3):
            frames[0, t, :, :, c] = t * 10 + c
    actions = np.tile(np.arange(seq_len, dtype=np.int64), (2, 1))
    done = np.zeros((2, seq_len), dtype=np.bool_)

    frames_path = tmp_path / "shard_000000_frames.npy"
    actions_path = tmp_path / "shard_000000_actions.npy"
    done_path = tmp_path / "shard_000000_done.npy"
    np.save(frames_path, frames, allow_pickle=False)
    np.save(actions_path, actions, allow_pickle=False)
    np.save(done_path, done, allow_pickle=False)

    manifest = {
        "game": "coinrun",
        "steps": 10,
        "seq_len": seq_len,
        "channel_count": 3,
        "frame_dtype": "uint8",
        "frame_layout": "THWC",
        "shards": [
            {
                "id": 0,
                "count": 2,
                "frames": frames_path.name,
                "actions": actions_path.name,
                "done": done_path.name,
            }
        ],
        "total": 2,
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    ds = OfflineDataset(
        tmp_path,
        n_past_frames=2,
        n_past_actions=1,
        n_future_frames=1,
    )
    obs, past_a, future_a, nxt, done_t = ds[0]

    assert obs.shape == (6, height, width)
    assert nxt.shape == (3, height, width)
    assert obs.dtype == torch.float32
    assert nxt.dtype == torch.float32
    assert past_a.dtype == torch.int64
    assert future_a.dtype == torch.int64
    assert done_t.dtype == torch.bool

    scale = np.float32(1.0 / 255.0)
    assert torch.allclose(obs[0], torch.full((height, width), 0.0 * scale))
    assert torch.allclose(obs[1], torch.full((height, width), 1.0 * scale))
    assert torch.allclose(obs[2], torch.full((height, width), 2.0 * scale))
    assert torch.allclose(obs[3], torch.full((height, width), 10.0 * scale))
    assert torch.allclose(obs[4], torch.full((height, width), 11.0 * scale))
    assert torch.allclose(obs[5], torch.full((height, width), 12.0 * scale))
    assert torch.allclose(nxt[0], torch.full((height, width), 20.0 * scale))
    assert torch.allclose(nxt[1], torch.full((height, width), 21.0 * scale))
    assert torch.allclose(nxt[2], torch.full((height, width), 22.0 * scale))
