from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Iterable[int] = (128, 128)):
        super().__init__()
        dims = [state_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        out = self.net(state_vec)
        return out.squeeze(-1)
