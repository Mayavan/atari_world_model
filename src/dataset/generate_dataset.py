from __future__ import annotations

"""CLI for generating offline Atari transition shards from a random policy."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.envs.atari_wrappers import make_atari_env
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def _maybe_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert normalized float observations to uint8 for compact on-disk storage."""
    if np.issubdtype(arr.dtype, np.floating):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if min_val >= -1e-6 and max_val <= 1.0 + 1e-6:
            scaled = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
            return scaled
    return arr


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    obs_buf: List[np.ndarray],
    action_buf: List[int],
    next_buf: List[np.ndarray],
    done_buf: List[bool],
) -> Optional[Tuple[Dict[str, str], int]]:
    """Write a single shard to disk and return its metadata."""
    if not obs_buf:
        return None

    obs_arr = np.ascontiguousarray(np.stack(obs_buf, axis=0))
    next_arr = np.ascontiguousarray(np.stack(next_buf, axis=0))
    obs_arr = _maybe_uint8(obs_arr)
    next_arr = _maybe_uint8(next_arr)
    action_arr = np.asarray(action_buf, dtype=np.int64)
    done_arr = np.asarray(done_buf, dtype=np.bool_)

    shard_base = f"shard_{shard_idx:06d}"
    paths = {
        "obs": f"{shard_base}_obs.npy",
        "next_obs": f"{shard_base}_next_obs.npy",
        "action": f"{shard_base}_action.npy",
        "done": f"{shard_base}_done.npy",
    }
    np.save(out_dir / paths["obs"], obs_arr, allow_pickle=False)
    np.save(out_dir / paths["next_obs"], next_arr, allow_pickle=False)
    np.save(out_dir / paths["action"], action_arr, allow_pickle=False)
    np.save(out_dir / paths["done"], done_arr, allow_pickle=False)
    return paths, len(obs_buf)


def generate_dataset(args: argparse.Namespace) -> None:
    """Collect transitions and write sharded NPY dataset plus manifest."""
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    env = make_atari_env(args.game, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)

    obs_buf: List[np.ndarray] = []
    action_buf: List[int] = []
    next_buf: List[np.ndarray] = []
    done_buf: List[bool] = []

    shards = []
    total = 0
    shard_idx = 0

    for _ in tqdm(range(args.steps), desc="collect"):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        obs_buf.append(obs)
        action_buf.append(int(action))
        next_buf.append(next_obs[-1:])
        done_buf.append(done)

        total += 1

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        if len(obs_buf) >= args.shard_size:
            result = _flush_shard(out_dir, shard_idx, obs_buf, action_buf, next_buf, done_buf)
            if result is not None:
                paths, count = result
                shards.append(
                    {
                        "id": shard_idx,
                        "count": count,
                        "obs": paths["obs"],
                        "next_obs": paths["next_obs"],
                        "action": paths["action"],
                        "done": paths["done"],
                    }
                )
                shard_idx += 1
            obs_buf, action_buf, next_buf, done_buf = [], [], [], []

    result = _flush_shard(out_dir, shard_idx, obs_buf, action_buf, next_buf, done_buf)
    if result is not None:
        paths, count = result
        shards.append(
            {
                "id": shard_idx,
                "count": count,
                "obs": paths["obs"],
                "next_obs": paths["next_obs"],
                "action": paths["action"],
                "done": paths["done"],
            }
        )

    manifest = {
        "game": args.game,
        "steps": args.steps,
        "shards": shards,
        "total": total,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Saved dataset to {out_dir} with {total} transitions and {len(shards)} shards")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for dataset generation."""
    p = argparse.ArgumentParser(description="Generate offline Atari dataset from random policy")
    p.add_argument("--game", type=str, required=True, help="Atari game name (e.g., Pong)")
    p.add_argument("--steps", type=int, default=300_000, help="Number of transitions to collect")
    p.add_argument("--out_dir", type=str, default="data", help="Output directory")
    p.add_argument("--shard_size", type=int, default=50_000, help="Transitions per shard")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    """Entry point for CLI."""
    args = build_parser().parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
