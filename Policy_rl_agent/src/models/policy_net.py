from __future__ import annotations

import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, edge_feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_vec: torch.Tensor,
        edge_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        if edge_feats.dim() == 2:
            edge_feats = edge_feats.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        state_h = self.state_encoder(state_vec)
        edge_h = self.edge_encoder(edge_feats)
        bsz, a_count, _ = edge_h.shape

        state_h = state_h.unsqueeze(1).expand(bsz, a_count, state_h.shape[-1])
        joined = torch.cat([state_h, edge_h], dim=-1)
        logits = self.scorer(joined).squeeze(-1)

        mask = action_mask.to(torch.bool)
        logits = logits.masked_fill(~mask, -1e9)
        return logits
