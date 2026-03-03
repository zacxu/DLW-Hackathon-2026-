from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_td_batch_from_trajectory(trajectory: dict[str, Any]) -> dict[str, np.ndarray]:
    transitions = trajectory["transitions"]
    states = np.stack([t["state_vec"] for t in transitions]).astype(np.float32)
    next_states = np.stack([t["next_state_vec"] for t in transitions]).astype(np.float32)
    rewards = np.array([t["reward"] for t in transitions], dtype=np.float32)
    dones = np.array([1.0 if t["done"] else 0.0 for t in transitions], dtype=np.float32)
    return {
        "states": states,
        "next_states": next_states,
        "rewards": rewards,
        "dones": dones,
    }


class ValueTrainer:
    def __init__(
        self,
        model: nn.Module,
        gamma: float,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def update_value(self, td_batch: dict[str, np.ndarray]) -> dict[str, float]:
        states = torch.from_numpy(td_batch["states"]).to(self.device)
        next_states = torch.from_numpy(td_batch["next_states"]).to(self.device)
        rewards = torch.from_numpy(td_batch["rewards"]).to(self.device)
        dones = torch.from_numpy(td_batch["dones"]).to(self.device)

        idx = torch.randperm(states.shape[0], device=self.device)
        states = states[idx]
        next_states = next_states[idx]
        rewards = rewards[idx]
        dones = dones[idx]

        total_loss = 0.0
        n_batches = 0
        for start in range(0, states.shape[0], self.batch_size):
            end = start + self.batch_size
            s = states[start:end]
            ns = next_states[start:end]
            r = rewards[start:end]
            d = dones[start:end]

            with torch.no_grad():
                next_v = self.model(ns)
                target = r + self.gamma * (1.0 - d) * next_v

            pred = self.model(s)
            loss = F.smooth_l1_loss(pred, target)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        with torch.no_grad():
            mean_v = float(self.model(states).mean().item())

        return {
            "value_loss": total_loss / max(n_batches, 1),
            "value_mean": mean_v,
        }
