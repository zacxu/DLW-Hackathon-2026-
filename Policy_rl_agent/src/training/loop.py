from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import trange

from src.config import TrainConfig
from src.eval.baselines import evaluate_random_policy
from src.eval.evaluate import evaluate
from src.training.policy_imitation_trainer import build_policy_targets
from src.training.value_trainer import build_td_batch_from_trajectory


def _masked_probs(
    logits: torch.Tensor,
    action_mask: np.ndarray,
    temperature: float,
    epsilon: float,
) -> np.ndarray:
    logits = logits.detach().cpu().numpy()
    valid = action_mask.astype(bool)
    if valid.sum() == 0:
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[0] = 1.0
        return probs

    scaled = logits.copy()
    scaled[~valid] = -1e12
    scaled = scaled / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled[valid])
    exp = np.exp(scaled, where=valid, out=np.zeros_like(scaled, dtype=np.float64))
    probs = exp / max(exp.sum(), 1e-12)

    uniform = np.zeros_like(probs, dtype=np.float64)
    uniform[valid] = 1.0 / valid.sum()
    probs = (1.0 - epsilon) * probs + epsilon * uniform
    probs = probs / probs.sum()
    return probs


def _expert_action_from_lookahead(lookahead: dict[str, np.ndarray], action_mask: np.ndarray) -> int:
    valid = action_mask.astype(bool)
    if valid.sum() == 0:
        return 0
    costs = lookahead["next_total_goal_costs"].astype(np.float64).copy()
    costs[~valid] = np.inf
    return int(np.argmin(costs))


def _apply_backtrack_penalty(
    probs: np.ndarray,
    lookahead: dict[str, np.ndarray],
    previous_node: int | None,
    strength: float,
) -> np.ndarray:
    if previous_node is None or strength <= 0.0:
        return probs
    next_nodes = lookahead.get("next_nodes")
    if next_nodes is None:
        return probs
    backtrack = np.where(next_nodes == previous_node)[0]
    if len(backtrack) == 0:
        return probs
    if np.count_nonzero(probs > 0) <= 1:
        return probs

    adjusted = probs.copy()
    adjusted[backtrack] *= max(1.0 - strength, 1e-6)
    total = adjusted.sum()
    if total <= 0:
        return probs
    return adjusted / total


def collect_episode(
    env: Any,
    policy: torch.nn.Module,
    mode: str = "train",
    temperature: float = 1.0,
    epsilon: float = 0.05,
    expert_action_prob: float = 0.0,
    backtrack_penalty_strength: float = 0.0,
    device: str = "cpu",
    reset_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs, info = env.reset(options=reset_options)
    transitions: list[dict[str, Any]] = []
    done = False
    episode_return = 0.0
    truncated = False
    step_info: dict[str, Any] = {"remaining_goal_count": len(info.get("goal_nodes", []))}
    path_nodes: list[int] = [int(info["start_node"])] if "start_node" in info else []
    previous_node: int | None = None

    while not done and not truncated:
        lookahead = env.get_lookahead()
        from_node = int(getattr(env, "current_node", -1))

        if mode == "random":
            valid_idx = np.where(obs["action_mask"].astype(bool))[0]
            action = int(env.rng.choice(valid_idx)) if len(valid_idx) else 0
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(obs["state_vec"]).to(device)
                edge_t = torch.from_numpy(obs["edge_feats"]).to(device)
                mask_t = torch.from_numpy(obs["action_mask"]).to(device)
                logits = policy(state_t, edge_t, mask_t).squeeze(0)

            if mode == "eval":
                action = int(torch.argmax(logits).item())
            else:
                use_expert = expert_action_prob > 0.0 and env.rng.random() < expert_action_prob
                if use_expert:
                    action = _expert_action_from_lookahead(lookahead, obs["action_mask"])
                else:
                    probs = _masked_probs(logits, obs["action_mask"], temperature, epsilon)
                    probs = _apply_backtrack_penalty(
                        probs=probs,
                        lookahead=lookahead,
                        previous_node=previous_node,
                        strength=backtrack_penalty_strength,
                    )
                    action = int(env.rng.choice(np.arange(len(probs)), p=probs))

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        transitions.append(
            {
                "state_vec": obs["state_vec"].copy(),
                "edge_feats": obs["edge_feats"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "action": int(action),
                "from_node": from_node,
                "to_node": int(getattr(env, "current_node", -1)),
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "next_state_vec": next_obs["state_vec"].copy(),
                "lookahead_rewards": lookahead["rewards"].copy(),
                "lookahead_dones": lookahead["dones"].copy(),
                "lookahead_next_state_vecs": lookahead["next_state_vecs"].copy(),
                "lookahead_next_goal_costs": lookahead["next_goal_costs"].copy(),
                "lookahead_next_total_goal_costs": lookahead["next_total_goal_costs"].copy(),
            }
        )
        if transitions[-1]["to_node"] >= 0:
            path_nodes.append(transitions[-1]["to_node"])

        episode_return += float(reward)
        done = terminated
        previous_node = from_node if from_node >= 0 else None
        obs = next_obs

    total_goals = max(len(info.get("goal_nodes", [])), 1)
    remaining = int(step_info.get("remaining_goal_count", total_goals))
    covered_ratio = (total_goals - remaining) / total_goals

    return {
        "transitions": transitions,
        "episode_return": episode_return,
        "episode_length": len(transitions),
        "success": bool(done),
        "goals_covered_ratio": float(covered_ratio),
        "start_node": info.get("start_node"),
        "goal_nodes": list(info.get("goal_nodes", [])),
        "path_nodes": path_nodes,
    }


def _save_checkpoint(
    out_dir: Path,
    name: str,
    episode: int,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{name}.pt"
    payload = {
        "episode": episode,
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def run_training_loop(
    train_env: Any,
    eval_env_factory: Callable[[int], Any],
    policy_net: torch.nn.Module,
    value_trainer: Any,
    policy_trainer: Any,
    cfg: TrainConfig,
    output_dir: str | Path,
    device: str = "cpu",
    train_reset_options: dict[str, Any] | None = None,
    eval_reset_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    best_eval_return = -float("inf")
    best_eval_success = -1.0
    best_eval_goal_coverage = -1.0
    best_eval_improvement = -float("inf")
    best_eval: dict[str, Any] = {}
    best_ckpt = None
    last_eval = {}

    for ep in trange(1, cfg.episodes + 1, desc="train", leave=False):
        frac = (ep - 1) / max(cfg.episodes - 1, 1)
        temp = cfg.temperature_start + frac * (cfg.temperature_end - cfg.temperature_start)
        expert_action_prob = float(
            np.clip(
                cfg.expert_action_prob_start
                + frac * (cfg.expert_action_prob_end - cfg.expert_action_prob_start),
                0.0,
                1.0,
            )
        )

        trajectory = collect_episode(
            env=train_env,
            policy=policy_net,
            mode="train",
            temperature=temp,
            epsilon=cfg.epsilon_explore,
            expert_action_prob=expert_action_prob,
            backtrack_penalty_strength=cfg.backtrack_penalty_strength,
            device=device,
            reset_options=train_reset_options,
        )

        td_batch = build_td_batch_from_trajectory(trajectory)
        value_metrics = value_trainer.update_value(td_batch)
        imitation_batch = build_policy_targets(
            trajectory=trajectory,
            value_net=value_trainer.model,
            gamma=cfg.gamma,
            device=device,
            teacher_value_weight=cfg.teacher_value_weight,
            teacher_goal_cost_weight=cfg.teacher_goal_cost_weight,
        )
        policy_metrics = policy_trainer.update_policy(imitation_batch)

        row = {
            "episode": ep,
            "temperature": temp,
            "expert_action_prob": expert_action_prob,
            "train_return": trajectory["episode_return"],
            "train_success": trajectory["success"],
            "train_goals_covered_ratio": trajectory["goals_covered_ratio"],
            "train_length": trajectory["episode_length"],
            **value_metrics,
            **policy_metrics,
        }

        if ep % cfg.eval_every == 0:
            eval_metrics = evaluate(
                policy=policy_net,
                env_factory=eval_env_factory,
                n_episodes=cfg.eval_episodes,
                device=device,
                deterministic=True,
                reset_options=eval_reset_options,
            )
            rand_metrics = evaluate_random_policy(
                env_factory=eval_env_factory,
                n_episodes=cfg.eval_episodes,
                reset_options=eval_reset_options,
            )
            denom = abs(rand_metrics["mean_return"]) + 1e-6
            improvement = (eval_metrics["mean_return"] - rand_metrics["mean_return"]) / denom

            row.update(
                {
                    "eval_return": eval_metrics["mean_return"],
                    "eval_success_rate": eval_metrics["success_rate"],
                    "eval_goals_covered_ratio": eval_metrics["mean_goals_covered_ratio"],
                    "random_return": rand_metrics["mean_return"],
                    "improvement_vs_random": improvement,
                }
            )
            last_eval = {
                "policy": eval_metrics,
                "random": rand_metrics,
                "improvement_vs_random": improvement,
            }

            better_coverage = eval_metrics["mean_goals_covered_ratio"] > best_eval_goal_coverage
            same_coverage = np.isclose(
                eval_metrics["mean_goals_covered_ratio"], best_eval_goal_coverage
            )
            better_success = eval_metrics["success_rate"] > best_eval_success
            same_success = np.isclose(eval_metrics["success_rate"], best_eval_success)
            better_return = eval_metrics["mean_return"] > best_eval_return

            if better_coverage or (same_coverage and (better_success or (same_success and better_return))):
                best_eval_goal_coverage = eval_metrics["mean_goals_covered_ratio"]
                best_eval_success = eval_metrics["success_rate"]
                best_eval_return = eval_metrics["mean_return"]
                best_eval_improvement = improvement
                best_eval = {
                    "policy": eval_metrics,
                    "random": rand_metrics,
                    "improvement_vs_random": improvement,
                }
                best_ckpt = _save_checkpoint(
                    out_dir=out_dir,
                    name="best",
                    episode=ep,
                    policy_net=policy_net,
                    value_net=value_trainer.model,
                    extra=last_eval,
                )

        if ep % cfg.checkpoint_every == 0:
            _save_checkpoint(
                out_dir=out_dir,
                name="latest",
                episode=ep,
                policy_net=policy_net,
                value_net=value_trainer.model,
                extra=last_eval,
            )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    _save_checkpoint(
        out_dir=out_dir,
        name="final",
        episode=cfg.episodes,
        policy_net=policy_net,
        value_net=value_trainer.model,
        extra=last_eval,
    )

    return {
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "best_eval": best_eval,
        "best_eval_return": best_eval_return,
        "best_eval_success": best_eval_success,
        "best_eval_goal_coverage": best_eval_goal_coverage,
        "best_eval_improvement": best_eval_improvement,
        "last_eval": last_eval,
        "metrics_path": str(metrics_path),
    }
