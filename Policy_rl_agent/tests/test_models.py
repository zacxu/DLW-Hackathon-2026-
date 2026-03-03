from __future__ import annotations

import torch

from src.models.policy_net import PolicyNet
from src.models.value_net import ValueNet


def test_value_net_shape(toy_env) -> None:
    obs, _ = toy_env.reset(seed=3)
    state = torch.from_numpy(obs["state_vec"]).float()
    model = ValueNet(state_dim=state.shape[0], hidden_dims=(32, 32))
    out = model(state)
    assert out.shape == (1,)


def test_policy_net_shape_and_masking(toy_env) -> None:
    obs, _ = toy_env.reset(seed=5)
    state = torch.from_numpy(obs["state_vec"]).float()
    edge = torch.from_numpy(obs["edge_feats"]).float()
    mask = torch.from_numpy(obs["action_mask"])

    model = PolicyNet(
        state_dim=state.shape[0],
        edge_feat_dim=edge.shape[1],
        hidden_dim=32,
    )
    logits = model(state, edge, mask)
    assert logits.shape == (1, toy_env.max_actions)

    invalid_idx = (mask == 0).nonzero(as_tuple=True)[0]
    if len(invalid_idx) > 0:
        assert torch.all(logits[0, invalid_idx] < -1e8)
