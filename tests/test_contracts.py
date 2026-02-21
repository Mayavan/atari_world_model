import pytest
import torch

from src.utils.contracts import (
    validate_model_output,
    validate_rollout_prediction,
    validate_supervised_batch,
)


def test_validate_supervised_batch_ok() -> None:
    obs = torch.rand(2, 4, 84, 84)
    past_actions = torch.zeros(2, 3, dtype=torch.int64)
    future_actions = torch.zeros(2, 2, dtype=torch.int64)
    next_obs = torch.rand(2, 2, 84, 84)
    validate_supervised_batch(
        obs=obs,
        past_actions=past_actions,
        future_actions=future_actions,
        next_obs=next_obs,
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=2,
    )


def test_validate_supervised_batch_shape_error() -> None:
    obs = torch.rand(2, 4, 84, 84)
    past_actions = torch.zeros(2, 2, dtype=torch.int64)
    future_actions = torch.zeros(2, 2, dtype=torch.int64)
    next_obs = torch.rand(2, 2, 84, 84)
    with pytest.raises(ValueError):
        validate_supervised_batch(
            obs=obs,
            past_actions=past_actions,
            future_actions=future_actions,
            next_obs=next_obs,
            n_past_frames=4,
            n_past_actions=3,
            n_future_frames=2,
        )


def test_validate_model_output_shape_error() -> None:
    with pytest.raises(ValueError):
        validate_model_output(logits=torch.rand(1, 1, 84, 84), next_obs=torch.rand(1, 2, 84, 84))


def test_validate_rollout_prediction_ok() -> None:
    validate_rollout_prediction(
        logits=torch.rand(1, 4, 84, 84),
        n_future_frames=4,
        height=84,
        width=84,
    )
