import pytest

from src.models.registry import (
    DEFAULT_MODEL_TYPE,
    build_model_from_checkpoint_cfg,
    build_model_from_config,
    resolve_model_type,
)


def test_resolve_model_type_aliases() -> None:
    assert resolve_model_type(None) == DEFAULT_MODEL_TYPE
    assert resolve_model_type("world_model") == DEFAULT_MODEL_TYPE
    assert resolve_model_type("pixel_world_model") == DEFAULT_MODEL_TYPE


def test_resolve_model_type_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_model_type("unknown")


def test_build_model_from_config_includes_type_in_ckpt_cfg() -> None:
    model, ckpt_cfg = build_model_from_config(
        model_cfg={"type": "pixel_world_model", "action_embed_dim": 32, "width_mult": 1.0},
        num_actions=7,
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=2,
    )
    assert model is not None
    assert ckpt_cfg["type"] == DEFAULT_MODEL_TYPE
    assert ckpt_cfg["n_past_frames"] == 4
    assert ckpt_cfg["n_past_actions"] == 3
    assert ckpt_cfg["n_future_frames"] == 2


def test_build_model_from_checkpoint_cfg_backcompat_without_type() -> None:
    model = build_model_from_checkpoint_cfg(
        model_cfg={"n_past_frames": 4, "n_past_actions": 3, "n_future_frames": 1},
        num_actions=5,
    )
    assert model is not None
