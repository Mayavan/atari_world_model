from __future__ import annotations

"""CLI for generating offline Atari sequence shards from a random policy."""

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.envs.atari_wrappers import make_atari_env
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def _should_uint8(arr: np.ndarray) -> bool:
    """Return True if normalized float observations can be stored as uint8."""
    if np.issubdtype(arr.dtype, np.floating):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        return min_val >= -1e-6 and max_val <= 1.0 + 1e-6
    return False


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert normalized float observations to uint8 for compact on-disk storage."""
    scaled = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
    return scaled


def _open_shard_memmaps(
    *,
    out_dir: Path,
    shard_idx: int,
    shard_size: int,
    seq_len: int,
    frame_shape: Tuple[int, int],
    frame_dtype: np.dtype,
) -> Tuple[Dict[str, str], np.memmap, np.memmap, np.memmap]:
    """Open memmaps for a shard so samples can stream to disk."""
    shard_base = f"shard_{shard_idx:06d}"
    paths = {
        "frames": f"{shard_base}_frames.npy",
        "actions": f"{shard_base}_actions.npy",
        "done": f"{shard_base}_done.npy",
    }
    frames = np.lib.format.open_memmap(
        out_dir / paths["frames"],
        mode="w+",
        dtype=frame_dtype,
        shape=(shard_size, seq_len, *frame_shape),
    )
    actions = np.lib.format.open_memmap(
        out_dir / paths["actions"],
        mode="w+",
        dtype=np.int64,
        shape=(shard_size, seq_len),
    )
    done = np.lib.format.open_memmap(
        out_dir / paths["done"],
        mode="w+",
        dtype=np.bool_,
        shape=(shard_size, seq_len),
    )
    return paths, frames, actions, done


def generate_dataset(args: argparse.Namespace) -> None:
    """Collect sequences and write sharded NPY dataset plus manifest."""
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    env = make_atari_env(args.game, seed=args.seed, frame_stack=1)

    obs, _ = env.reset(seed=args.seed)
    if obs.ndim == 3:
        frame = obs[-1]
    else:
        frame = obs

    seq_len = int(args.seq_len)
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    shard_size = int(args.shard_size)
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")

    frame_hist: deque[np.ndarray] = deque(maxlen=seq_len + 1)
    action_hist: deque[int] = deque(maxlen=seq_len)
    done_hist: deque[bool] = deque(maxlen=seq_len)
    frame_hist.append(frame)
    frame_shape = tuple(int(v) for v in frame.shape)
    use_uint8 = _should_uint8(frame)
    frame_dtype = np.uint8 if use_uint8 else frame.dtype

    shard_paths: Optional[Dict[str, str]] = None
    shard_frames: Optional[np.memmap] = None
    shard_actions: Optional[np.memmap] = None
    shard_done: Optional[np.memmap] = None
    shard_count = 0

    shards = []
    total = 0
    shard_idx = 0

    for _ in tqdm(range(args.steps), desc="collect"):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        if next_obs.ndim == 3:
            next_frame = next_obs[-1]
        else:
            next_frame = next_obs

        action_hist.append(int(action))
        done_hist.append(done)
        frame_hist.append(next_frame)

        if len(action_hist) == seq_len and len(frame_hist) == seq_len + 1:
            # frames_seq[0]..frames_seq[-1] are consecutive frames; actions_seq[i]
            # corresponds to the action taken after frames_seq[i] (leading to frames_seq[i+1]).
            frames_seq = np.ascontiguousarray(np.stack(list(frame_hist)[:-1], axis=0))
            if use_uint8:
                frames_seq = _to_uint8(frames_seq)
            actions_seq = np.asarray(action_hist, dtype=np.int64)
            done_seq = np.asarray(done_hist, dtype=np.bool_)

            if shard_frames is None:
                shard_paths, shard_frames, shard_actions, shard_done = _open_shard_memmaps(
                    out_dir=out_dir,
                    shard_idx=shard_idx,
                    shard_size=shard_size,
                    seq_len=seq_len,
                    frame_shape=frame_shape,
                    frame_dtype=frame_dtype,
                )

            shard_frames[shard_count] = frames_seq
            shard_actions[shard_count] = actions_seq
            shard_done[shard_count] = done_seq
            shard_count += 1
            total += 1

            if shard_count >= shard_size:
                shard_frames.flush()
                shard_actions.flush()
                shard_done.flush()
                shards.append(
                    {
                        "id": shard_idx,
                        "count": shard_count,
                        "frames": shard_paths["frames"],
                        "actions": shard_paths["actions"],
                        "done": shard_paths["done"],
                    }
                )
                shard_idx += 1
                shard_frames = None
                shard_actions = None
                shard_done = None
                shard_paths = None
                shard_count = 0

        if done:
            obs, _ = env.reset()
            if obs.ndim == 3:
                frame = obs[-1]
            else:
                frame = obs
            frame_hist.clear()
            action_hist.clear()
            done_hist.clear()
            frame_hist.append(frame)
        else:
            obs = next_obs

    if shard_frames is not None and shard_count > 0:
        shard_frames.flush()
        shard_actions.flush()
        shard_done.flush()
        shards.append(
            {
                "id": shard_idx,
                "count": shard_count,
                "frames": shard_paths["frames"],
                "actions": shard_paths["actions"],
                "done": shard_paths["done"],
            }
        )

    manifest = {
        "game": args.game,
        "steps": args.steps,
        "seq_len": seq_len,
        "shards": shards,
        "total": total,
    }
    write_json(out_dir / "manifest.json", manifest)
    env.close()
    print(f"Saved dataset to {out_dir} with {total} sequences and {len(shards)} shards")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for dataset generation."""
    p = argparse.ArgumentParser(description="Generate offline Atari dataset from random policy")
    p.add_argument("--game", type=str, required=True, help="Atari game name (e.g., Pong)")
    p.add_argument("--steps", type=int, default=300_000, help="Number of transitions to collect")
    p.add_argument("--out_dir", type=str, default="data", help="Output directory")
    p.add_argument("--shard_size", type=int, default=50_000, help="Sequences per shard")
    p.add_argument("--seq_len", type=int, default=8, help="Frames/actions per sequence")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    """Entry point for CLI."""
    args = build_parser().parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
