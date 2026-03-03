from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_policy_targets(
    trajectory: dict[str, Any],
    value_net: nn.Module,
    gamma: float,
    device: str = "cpu",
    teacher_value_weight: float = 0.3,
    teacher_goal_cost_weight: float = 1.2,
) -> dict[str, np.ndarray]:
    transitions = trajectory["transitions"]
    states = np.stack([t["state_vec"] for t in transitions]).astype(np.float32)
    edge_feats = np.stack([t["edge_feats"] for t in transitions]).astype(np.float32)
    action_masks = np.stack([t["action_mask"] for t in transitions]).astype(np.int8)
    lookahead_rewards = np.stack([t["lookahead_rewards"] for t in transitions]).astype(np.float32)
    lookahead_dones = np.stack([t["lookahead_dones"] for t in transitions]).astype(np.float32)
    lookahead_next_states = np.stack(
        [t["lookahead_next_state_vecs"] for t in transitions]
    ).astype(np.float32)
    lookahead_next_total_goal_costs = np.stack(
        [t["lookahead_next_total_goal_costs"] for t in transitions]
    ).astype(np.float32)

    t_count, a_count, s_dim = lookahead_next_states.shape

    flat_next = torch.from_numpy(lookahead_next_states.reshape(-1, s_dim)).to(device)
    with torch.no_grad():
        next_values = value_net(flat_next).reshape(t_count, a_count).cpu().numpy()

    td_component = lookahead_rewards + gamma * (1.0 - lookahead_dones) * next_values
    q_vals = (
        teacher_value_weight * td_component
        - teacher_goal_cost_weight * lookahead_next_total_goal_costs
    )
    q_vals[action_masks == 0] = -1e12
    teacher_actions = q_vals.argmax(axis=1).astype(np.int64)

    return {
        "states": states,
        "edge_feats": edge_feats,
        "action_masks": action_masks,
        "teacher_actions": teacher_actions,
    }


class PolicyImitationTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def update_policy(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        states = torch.from_numpy(batch["states"]).to(self.device)
        edge_feats = torch.from_numpy(batch["edge_feats"]).to(self.device)
        masks = torch.from_numpy(batch["action_masks"]).to(self.device)
        teachers = torch.from_numpy(batch["teacher_actions"]).to(self.device)

        idx = torch.randperm(states.shape[0], device=self.device)
        states = states[idx]
        edge_feats = edge_feats[idx]
        masks = masks[idx]
        teachers = teachers[idx]

        total_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        for start in range(0, states.shape[0], self.batch_size):
            end = start + self.batch_size
            s = states[start:end]
            e = edge_feats[start:end]
            m = masks[start:end]
            t = teachers[start:end]

            logits = self.model(s, e, m)
            loss = F.cross_entropy(logits, t)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

            total_loss += float(loss.item())
            total_entropy += float(entropy.item())
            n_batches += 1

        return {
            "policy_loss": total_loss / max(n_batches, 1),
            "policy_entropy": total_entropy / max(n_batches, 1),
        }
