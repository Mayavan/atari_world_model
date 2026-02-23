from __future__ import annotations

"""Runtime interface checks for tensors passed between data/model/eval code."""

import torch


def _shape(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in t.shape)


def validate_supervised_batch(
    *,
    obs: torch.Tensor,
    past_actions: torch.Tensor,
    future_actions: torch.Tensor,
    next_obs: torch.Tensor,
    n_past_frames: int,
    n_past_actions: int,
    n_future_frames: int,
    frame_channels: int = 1,
) -> None:
    """Validate training/validation batch contract before model forward."""
    if obs.ndim != 4:
        raise ValueError(f"obs must be rank-4 (B,C,H,W), got shape={_shape(obs)}")
    if next_obs.ndim != 4:
        raise ValueError(f"next_obs must be rank-4 (B,C,H,W), got shape={_shape(next_obs)}")
    if past_actions.ndim != 2:
        raise ValueError(
            f"past_actions must be rank-2 (B,Tpast), got shape={_shape(past_actions)}"
        )
    if future_actions.ndim != 2:
        raise ValueError(
            f"future_actions must be rank-2 (B,Tfuture), got shape={_shape(future_actions)}"
        )

    b_obs, c_obs, h_obs, w_obs = obs.shape
    b_next, c_next, h_next, w_next = next_obs.shape
    b_past, t_past = past_actions.shape
    b_future, t_future = future_actions.shape

    if b_obs != b_next or b_obs != b_past or b_obs != b_future:
        raise ValueError(
            "Batch size mismatch: "
            f"obs={b_obs}, next_obs={b_next}, past_actions={b_past}, future_actions={b_future}"
        )
    expected_obs_channels = int(n_past_frames) * int(frame_channels)
    expected_next_channels = int(n_future_frames) * int(frame_channels)
    if c_obs != expected_obs_channels:
        raise ValueError(
            "obs channels must equal "
            f"n_past_frames*frame_channels={expected_obs_channels}, got {c_obs}"
        )
    if c_next != expected_next_channels:
        raise ValueError(
            "next_obs channels must equal "
            f"n_future_frames*frame_channels={expected_next_channels}, got {c_next}"
        )
    if t_past != int(n_past_actions):
        raise ValueError(
            f"past_actions length must equal n_past_actions={n_past_actions}, got {t_past}"
        )
    if t_future != int(n_future_frames):
        raise ValueError(
            f"future_actions length must equal n_future_frames={n_future_frames}, got {t_future}"
        )
    if h_obs != h_next or w_obs != w_next:
        raise ValueError(
            f"obs and next_obs spatial shapes must match, got {_shape(obs)} vs {_shape(next_obs)}"
        )

    if not obs.dtype.is_floating_point:
        raise ValueError(f"obs must be floating-point, got dtype={obs.dtype}")
    if not next_obs.dtype.is_floating_point:
        raise ValueError(f"next_obs must be floating-point, got dtype={next_obs.dtype}")
    if past_actions.dtype != torch.int64:
        raise ValueError(f"past_actions must be int64, got dtype={past_actions.dtype}")
    if future_actions.dtype != torch.int64:
        raise ValueError(f"future_actions must be int64, got dtype={future_actions.dtype}")


def validate_model_output(*, pred: torch.Tensor, next_obs: torch.Tensor) -> None:
    """Validate model output tensor against supervised target shape contract."""
    if pred.shape != next_obs.shape:
        raise ValueError(
            "model output shape mismatch: "
            f"pred={_shape(pred)} vs next_obs={_shape(next_obs)}"
        )


def validate_rollout_prediction(
    *,
    pred: torch.Tensor,
    expected_channels: int,
    height: int,
    width: int,
) -> None:
    """Validate model output shape in open-loop rollout where target is unavailable."""
    if pred.ndim != 4:
        raise ValueError(f"pred must be rank-4 (B,C,H,W), got shape={_shape(pred)}")
    if pred.shape[0] != 1:
        raise ValueError(f"rollout expects batch size 1 pred tensor, got {pred.shape[0]}")
    if pred.shape[1] != int(expected_channels):
        raise ValueError(
            "pred channels must match expected channels; "
            f"expected {expected_channels}, got {pred.shape[1]}"
        )
    if pred.shape[2] != int(height) or pred.shape[3] != int(width):
        raise ValueError(
            f"pred spatial shape must be ({height}, {width}), got {pred.shape[2:]}"
        )


def validate_rollout_stack(*, pred_stack, n_past_frames: int, frame_channels: int = 1) -> None:
    """Validate open-loop rollout frame stack contract."""
    if pred_stack is None:
        raise ValueError("pred_stack must not be None")
    if not hasattr(pred_stack, "ndim"):
        raise ValueError("pred_stack must be an array-like tensor with shape (T,H,W)")
    if int(frame_channels) == 3:
        if pred_stack.ndim != 4 or pred_stack.shape[-1] != 3:
            raise ValueError(
                "RGB rollout stack must be rank-4 (T,H,W,3), "
                f"got shape={pred_stack.shape}"
            )
    elif int(frame_channels) == 1:
        if pred_stack.ndim != 3:
            raise ValueError(f"pred_stack must be rank-3 (T,H,W), got shape={pred_stack.shape}")
    else:
        raise ValueError(f"Unsupported frame_channels={frame_channels}")
    if pred_stack.shape[0] != int(n_past_frames):
        raise ValueError(
            f"pred_stack time dimension must equal n_past_frames={n_past_frames}, "
            f"got {pred_stack.shape[0]}"
        )
