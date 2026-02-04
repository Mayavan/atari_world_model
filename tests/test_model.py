import torch

from src.models.world_model import WorldModel


def test_model_forward_shape():
    model = WorldModel(num_actions=6, n_past_frames=4, n_past_actions=3, n_future_frames=2)
    obs = torch.randn(2, 4, 84, 84)
    past_actions = torch.randint(0, 6, (2, 3))
    future_actions = torch.randint(0, 6, (2, 2))
    out = model(obs, future_actions, past_actions)
    assert out.shape == (2, 2, 84, 84)
