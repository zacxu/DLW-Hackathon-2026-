from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def evaluate(
    policy: torch.nn.Module,
    env_factory: Callable[[int], Any],
    n_episodes: int,
    device: str = "cpu",
    deterministic: bool = True,
    reset_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    returns = []
    lengths = []
    successes = []
    goals_covered = []

    for i in range(n_episodes):
        env = env_factory(i)
        obs, info = env.reset(seed=env.cfg.seed + i, options=reset_options)
        total_return = 0.0
        done = False
        truncated = False
        step_info = {"remaining_goal_count": len(info.get("goal_nodes", []))}

        while not done and not truncated:
            with torch.no_grad():
                s = torch.from_numpy(obs["state_vec"]).to(device)
                e = torch.from_numpy(obs["edge_feats"]).to(device)
                m = torch.from_numpy(obs["action_mask"]).to(device)
                logits = policy(s, e, m).squeeze(0)

            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                action = int(env.rng.choice(np.arange(len(probs)), p=probs))

            obs, reward, done, truncated, step_info = env.step(action)
            total_return += float(reward)

        total_goals = max(len(info.get("goal_nodes", [])), 1)
        remaining = int(step_info.get("remaining_goal_count", total_goals))
        returns.append(total_return)
        lengths.append(env.step_count)
        successes.append(1.0 if done else 0.0)
        goals_covered.append((total_goals - remaining) / total_goals)

    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_goals_covered_ratio": float(np.mean(goals_covered)),
    }
