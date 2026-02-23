import pytest
import torch

from src.models.conv_autoencoder import ConvAutoencoder
from src.models.registry import (
    DEFAULT_MODEL_TYPE,
    build_model_from_checkpoint_cfg,
    build_model_from_config,
    resolve_model_type,
)


def _write_autoencoder_ckpt(tmp_path) -> str:
    model = ConvAutoencoder(in_channels=3, latent_channels=8, base_channels=16)
    ckpt_path = tmp_path / "ae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": {
                "type": "conv_autoencoder",
                "in_channels": model.in_channels,
                "image_height": model.image_height,
                "image_width": model.image_width,
                "base_channels": model.base_channels,
                "latent_channels": model.latent_channels,
                "downsample_factor": model.downsample_factor,
                "decoder_type": model.decoder_type,
            },
        },
        ckpt_path,
    )
    return str(ckpt_path)


def test_resolve_model_type_default() -> None:
    assert resolve_model_type(None) == DEFAULT_MODEL_TYPE
    assert resolve_model_type("latent_flow_world_model") == DEFAULT_MODEL_TYPE


def test_resolve_model_type_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_model_type("unknown")


def test_build_model_from_config_requires_autoencoder_checkpoint() -> None:
    with pytest.raises(ValueError):
        build_model_from_config(
            model_cfg={"type": "latent_flow_world_model"},
            num_actions=7,
            n_past_frames=4,
            n_past_actions=3,
            n_future_frames=2,
        )


def test_build_model_from_config_includes_type_in_ckpt_cfg(tmp_path) -> None:
    ae_path = _write_autoencoder_ckpt(tmp_path)
    model, ckpt_cfg = build_model_from_config(
        model_cfg={
            "type": "latent_flow_world_model",
            "action_embed_dim": 32,
            "width_mult": 1.0,
            "sampling_steps": 4,
            "autoencoder_checkpoint": ae_path,
        },
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
    assert ckpt_cfg["autoencoder_checkpoint"] == ae_path
    assert ckpt_cfg["autoencoder_model_cfg"]["type"] == "conv_autoencoder"


def test_build_model_from_checkpoint_cfg() -> None:
    model = build_model_from_checkpoint_cfg(
        model_cfg={
            "type": "latent_flow_world_model",
            "n_past_frames": 4,
            "n_past_actions": 3,
            "n_future_frames": 1,
            "sampling_steps": 4,
            "autoencoder_model_cfg": {
                "type": "conv_autoencoder",
                "in_channels": 3,
                "image_height": 84,
                "image_width": 84,
                "base_channels": 16,
                "latent_channels": 8,
                "downsample_factor": 8,
                "decoder_type": "upsample_conv",
            },
        },
        num_actions=5,
    )
    assert model is not None
