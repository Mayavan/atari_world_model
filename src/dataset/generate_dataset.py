from __future__ import annotations

"""CLI for generating offline sequence shards from vectorized Procgen rollouts."""

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.envs.procgen_wrappers import make_procgen_env
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed

EPSILON = 0.3
STUCK_K = 15
FORCED_RANDOM_STEPS = 5
CUTOFF_MIN = 64
CUTOFF_MAX = 256
ENV_SEED_STRIDE = 10_000
WORKER_SEED_OFFSET = 1_000_000
HASH_DOWNSAMPLE_SIZE = (32, 32)
DIVERSITY_LOG_INTERVAL = 10_000
DIVERSITY_STATS_FILE = "diversity_stats.jsonl"


def _open_shard_memmaps(
    *,
    out_dir: Path,
    shard_idx: int,
    shard_size: int,
    seq_len: int,
    frame_shape: Tuple[int, ...],
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


def _resolve_base_seed(base_seed_arg: int | None) -> int:
    if base_seed_arg is None:
        raise ValueError("--base_seed must be explicitly configured.")
    return int(base_seed_arg)


def _make_diversity_hash(frame: np.ndarray) -> str:
    gray = np.asarray(
        Image.fromarray(frame).convert("L").resize(HASH_DOWNSAMPLE_SIZE, Image.BILINEAR),
        dtype=np.uint8,
    )
    return hashlib.sha1(gray.tobytes()).hexdigest()


def _append_diversity_stats(
    *,
    path: Path,
    episodes_seen: int,
    unique_hashes: int,
    unique_rate: float,
    tag: str,
) -> None:
    payload = {
        "tag": tag,
        "episodes_seen": int(episodes_seen),
        "unique_hashes": int(unique_hashes),
        "unique_rate": float(unique_rate),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def _extract_frame_batch(obs_batch: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_batch)
    if obs.ndim != 5:
        raise ValueError(f"Expected vector observation rank 5 (N,1,H,W,C), got {obs.shape}")
    if obs.shape[1] != 1:
        raise ValueError(f"Expected frame_stack=1 in observations, got shape {obs.shape}")
    if obs.shape[-1] != 3:
        raise ValueError(f"Expected RGB observations (C=3), got shape {obs.shape}")
    return obs[:, 0]


def _sample_cutoff_lengths(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(CUTOFF_MIN, CUTOFF_MAX + 1, size=count, dtype=np.int32)


def _make_env_fn(
    *,
    game: str,
    seed: int,
):
    def _thunk():
        return make_procgen_env(
            game,
            seed=seed,
            frame_stack=1,
            obs_mode="rgb",
            normalize=False,
        )

    return _thunk


def generate_dataset(args: argparse.Namespace) -> None:
    """Collect sequences and write sharded NPY dataset plus manifest."""
    worker_id = int(args.worker_id)
    base_seed = _resolve_base_seed(args.base_seed)
    set_seed(base_seed)
    rng = np.random.default_rng(base_seed + worker_id * WORKER_SEED_OFFSET + 1337)

    out_dir = ensure_dir(args.out_dir)
    stats_path = out_dir / DIVERSITY_STATS_FILE
    if stats_path.exists():
        stats_path.unlink()

    steps = 1000 if bool(args.dry_run) else int(args.steps)
    if steps <= 0:
        raise ValueError("steps must be > 0")
    seq_len = int(args.seq_len)
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    shard_size = int(args.shard_size)
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    num_envs = int(args.num_envs)
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")

    assert hasattr(gym.vector, "AutoresetMode"), (
        "Dataset generation requires Gymnasium >= 1.1 with vector reset_mask support."
    )

    print(f"Using base_seed={base_seed} worker_id={worker_id} num_envs={num_envs}")
    env_fns = []
    for env_idx in range(num_envs):
        env_seed = base_seed + worker_id * WORKER_SEED_OFFSET + env_idx * ENV_SEED_STRIDE
        env_fns.append(_make_env_fn(game=args.game, seed=env_seed))
    env = gym.vector.SyncVectorEnv(
        env_fns=env_fns,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    if not isinstance(env.single_action_space, gym.spaces.Discrete):
        raise TypeError(
            "Only Discrete action spaces are supported for dataset generation, "
            f"got {env.single_action_space}"
        )
    num_actions = int(env.single_action_space.n)

    obs_batch, _ = env.reset()
    frame_batch = _extract_frame_batch(obs_batch)
    frame_shape = tuple(int(v) for v in frame_batch[0].shape)
    if frame_batch.dtype != np.uint8:
        raise TypeError(f"Expected uint8 RGB observations, got dtype={frame_batch.dtype}")
    if frame_shape != (84, 84, 3):
        raise ValueError(f"Expected frame shape (84, 84, 3), got {frame_shape}")
    frame_dtype = np.uint8
    channel_count = 3
    frame_layout = "THWC"

    frame_hists = [deque(maxlen=seq_len + 1) for _ in range(num_envs)]
    action_hists = [deque(maxlen=seq_len) for _ in range(num_envs)]
    done_hists = [deque(maxlen=seq_len) for _ in range(num_envs)]
    for env_idx in range(num_envs):
        frame_hists[env_idx].append(frame_batch[env_idx])

    unique_hashes: set[str] = set()
    episodes_seen = 0
    next_diversity_log = DIVERSITY_LOG_INTERVAL
    for env_idx in range(num_envs):
        unique_hashes.add(_make_diversity_hash(frame_batch[env_idx]))
        episodes_seen += 1

    last_action = rng.integers(num_actions, size=num_envs, dtype=np.int64)
    prev_reward = np.full(num_envs, np.nan, dtype=np.float32)
    stuck_counter = np.zeros(num_envs, dtype=np.int32)
    forced_random_remaining = np.zeros(num_envs, dtype=np.int32)
    t_in_episode = np.zeros(num_envs, dtype=np.int32)
    cutoff_lengths = _sample_cutoff_lengths(rng, num_envs)
    action_histogram = np.zeros(num_actions, dtype=np.int64)

    shard_paths: Optional[Dict[str, str]] = None
    shard_frames: Optional[np.memmap] = None
    shard_actions: Optional[np.memmap] = None
    shard_done: Optional[np.memmap] = None
    shard_count = 0

    shards = []
    total = 0
    shard_idx = 0
    transitions_collected = 0

    progress = tqdm(total=steps, desc="collect")
    while transitions_collected < steps:
        random_actions = rng.integers(num_actions, size=num_envs, dtype=np.int64)
        epsilon_random = rng.random(num_envs) < EPSILON
        forced_random = forced_random_remaining > 0
        random_mask = epsilon_random | forced_random
        actions = np.where(random_mask, random_actions, last_action).astype(np.int64)
        last_action = actions
        forced_random_remaining = np.maximum(forced_random_remaining - forced_random.astype(np.int32), 0)

        next_obs_batch, rewards, terminated, truncated, _ = env.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        next_frame_batch = _extract_frame_batch(next_obs_batch)

        reward_unchanged = np.isclose(rewards, prev_reward)
        stuck_counter = np.where(reward_unchanged, stuck_counter + 1, 0).astype(np.int32)
        stuck_trigger = (stuck_counter >= STUCK_K) & (forced_random_remaining == 0)
        forced_random_remaining = np.where(
            stuck_trigger,
            np.full(num_envs, FORCED_RANDOM_STEPS, dtype=np.int32),
            forced_random_remaining,
        ).astype(np.int32)
        prev_reward = rewards

        t_in_episode = t_in_episode + 1
        cutoff_done = t_in_episode >= cutoff_lengths
        done = terminated | truncated | cutoff_done

        remaining = steps - transitions_collected
        active_count = min(num_envs, remaining)
        active_mask = np.zeros(num_envs, dtype=bool)
        active_mask[:active_count] = True
        done = done & active_mask

        if active_count > 0:
            np.add.at(action_histogram, actions[:active_count], 1)

        reset_frame_batch = None
        if np.any(done):
            reset_obs_batch, _ = env.reset(options={"reset_mask": done})
            reset_frame_batch = _extract_frame_batch(reset_obs_batch)
            reset_indices = np.flatnonzero(done)
            for env_idx in reset_indices:
                unique_hashes.add(_make_diversity_hash(reset_frame_batch[env_idx]))
                episodes_seen += 1
            while episodes_seen >= next_diversity_log:
                unique_count = len(unique_hashes)
                unique_rate = unique_count / float(episodes_seen)
                print(
                    "[diversity] "
                    f"episodes_seen={episodes_seen} "
                    f"unique_hashes={unique_count} "
                    f"unique_rate={unique_rate:.6f}"
                )
                _append_diversity_stats(
                    path=stats_path,
                    episodes_seen=episodes_seen,
                    unique_hashes=unique_count,
                    unique_rate=unique_rate,
                    tag="interval",
                )
                next_diversity_log += DIVERSITY_LOG_INTERVAL

            reset_size = int(reset_indices.size)
            last_action[reset_indices] = rng.integers(num_actions, size=reset_size, dtype=np.int64)
            prev_reward[reset_indices] = np.nan
            stuck_counter[reset_indices] = 0
            forced_random_remaining[reset_indices] = 0
            t_in_episode[reset_indices] = 0
            cutoff_lengths[reset_indices] = _sample_cutoff_lengths(rng, reset_size)

        for env_idx in range(active_count):
            action_hists[env_idx].append(int(actions[env_idx]))
            done_hists[env_idx].append(bool(done[env_idx]))
            frame_hists[env_idx].append(next_frame_batch[env_idx])

            if len(action_hists[env_idx]) == seq_len and len(frame_hists[env_idx]) == seq_len + 1:
                # frames_seq[0]..frames_seq[-1] are consecutive frames; actions_seq[i]
                # corresponds to the action taken after frames_seq[i] (leading to frames_seq[i+1]).
                frames_seq = np.ascontiguousarray(np.stack(list(frame_hists[env_idx])[:-1], axis=0))
                actions_seq = np.asarray(action_hists[env_idx], dtype=np.int64)
                done_seq = np.asarray(done_hists[env_idx], dtype=np.bool_)

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

            if done[env_idx]:
                if reset_frame_batch is None:
                    raise RuntimeError("Missing reset observations for done environments.")
                frame_hists[env_idx].clear()
                action_hists[env_idx].clear()
                done_hists[env_idx].clear()
                frame_hists[env_idx].append(reset_frame_batch[env_idx])

        transitions_collected += active_count
        progress.update(active_count)

    progress.close()

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

    unique_count = len(unique_hashes)
    unique_rate = unique_count / float(episodes_seen)
    _append_diversity_stats(
        path=stats_path,
        episodes_seen=episodes_seen,
        unique_hashes=unique_count,
        unique_rate=unique_rate,
        tag="final",
    )

    manifest = {
        "game": args.game,
        "steps": steps,
        "seq_len": seq_len,
        "shards": shards,
        "total": total,
        "channel_count": channel_count,
        "frame_dtype": str(np.dtype(frame_dtype)),
        "frame_layout": frame_layout,
        "base_seed": int(base_seed),
        "worker_id": int(worker_id),
        "num_envs": int(num_envs),
        "diversity_stats_file": DIVERSITY_STATS_FILE,
    }
    write_json(out_dir / "manifest.json", manifest)
    env.close()
    print(f"Saved dataset to {out_dir} with {total} sequences and {len(shards)} shards")
    print(f"obs shape: {frame_shape}")
    print(f"dtype: {np.dtype(frame_dtype)}")
    print(f"action histogram: {action_histogram.tolist()}")
    print(f"unique_rate: {unique_rate:.6f} ({unique_count}/{episodes_seen})")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for dataset generation."""
    p = argparse.ArgumentParser(description="Generate offline Procgen dataset")
    p.add_argument("--game", type=str, required=True, help="Procgen game name (e.g., coinrun)")
    p.add_argument("--steps", type=int, default=300_000, help="Number of transitions to collect")
    p.add_argument("--out_dir", type=str, default="data", help="Output directory")
    p.add_argument("--shard_size", type=int, default=50_000, help="Sequences per shard")
    p.add_argument("--seq_len", type=int, default=8, help="Frames/actions per sequence")
    p.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
    p.add_argument("--base_seed", type=int, default=0, help="Base seed for per-env seed derivation")
    p.add_argument("--worker_id", type=int, default=0, help="Worker/rank id for seed offset")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Collect 1,000 transitions and print collection diagnostics",
    )
    return p


def main() -> None:
    """Entry point for CLI."""
    args = build_parser().parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
