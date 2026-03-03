from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PlannerEngine:
    config_path: Path
    checkpoint_path: Path
    graph: Any
    cfg: Any
    policy: Any


_ENGINE: PlannerEngine | None = None
_ENGINE_LOCK = threading.Lock()
_ROUTE_LOCK = threading.Lock()


def _ensure_policy_import_path(policy_root: Path) -> None:
    policy_root_str = str(policy_root)
    if policy_root_str not in sys.path:
        sys.path.insert(0, policy_root_str)


def _resolve_checkpoint_path(policy_root: Path, cfg: Any) -> Path:
    override = os.getenv("POLICY_AGENT_CHECKPOINT", "").strip()
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = policy_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"POLICY_AGENT_CHECKPOINT not found: {candidate}")
        return candidate

    run_dir = Path(cfg.output.run_dir)
    if not run_dir.is_absolute():
        run_dir = policy_root / run_dir
    candidates = [run_dir / "best.pt", run_dir / "latest.pt", run_dir / "final.pt"]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No policy checkpoint found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _build_engine(project_root: Path) -> PlannerEngine:
    policy_root = project_root / "Policy_rl_agent"
    if not policy_root.exists():
        raise FileNotFoundError(f"Policy agent folder not found: {policy_root}")

    _ensure_policy_import_path(policy_root)

    from src.config import load_config
    from src.env.road_multi_goal_env import RoadMultiGoalEnv
    from src.graph_loader import load_singapore_graph
    from src.models.policy_net import PolicyNet

    config_path = os.getenv("POLICY_AGENT_CONFIG", "").strip()
    if config_path:
        resolved_config = Path(config_path)
        if not resolved_config.is_absolute():
            resolved_config = policy_root / resolved_config
    else:
        resolved_config = policy_root / "configs" / "default.yaml"

    if not resolved_config.exists():
        raise FileNotFoundError(f"Policy config not found: {resolved_config}")

    cfg = load_config(resolved_config)
    graph = load_singapore_graph(cfg.graph)
    checkpoint_path = _resolve_checkpoint_path(policy_root=policy_root, cfg=cfg)

    probe_env = RoadMultiGoalEnv(graph, cfg.env)
    probe_obs, _ = probe_env.reset(seed=cfg.env.seed)
    state_dim = int(probe_obs["state_vec"].shape[0])
    edge_feat_dim = int(probe_obs["edge_feats"].shape[1])

    policy = PolicyNet(
        state_dim=state_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=cfg.model.policy_hidden_dim,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    return PlannerEngine(
        config_path=resolved_config,
        checkpoint_path=checkpoint_path,
        graph=graph,
        cfg=cfg,
        policy=policy,
    )


def _get_engine(project_root: Path) -> PlannerEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine(project_root=project_root)
    return _ENGINE


def _build_path_coords(graph: Any, nodes: list[int]) -> list[dict[str, float | int]]:
    path_coords: list[dict[str, float | int]] = []
    for idx, node in enumerate(nodes):
        attrs = graph.nodes[int(node)]
        path_coords.append(
            {
                "seq": idx,
                "node": int(node),
                "lat": float(attrs["y"]),
                "lon": float(attrs["x"]),
            }
        )
    return path_coords


def _estimate_path_travel_time_sec(graph: Any, nodes: list[int]) -> float:
    total = 0.0
    for i in range(len(nodes) - 1):
        u = int(nodes[i])
        v = int(nodes[i + 1])
        edge_data = graph.get_edge_data(u, v, default={})
        if not edge_data:
            continue
        edge_times = [
            float(data.get("travel_time", 0.0))
            for data in edge_data.values()
            if float(data.get("travel_time", 0.0)) > 0.0
        ]
        if edge_times:
            total += min(edge_times)
    return total


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "on"}:
            return True
        if norm in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def build_policy_route(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    engine = _get_engine(project_root=project_root)

    from src.env.road_multi_goal_env import RoadMultiGoalEnv
    from src.training.loop import collect_episode
    from src.visualization.plot_graph import (
        plot_graph_with_overlays,
        select_start_and_goals_on_graph,
    )

    goal_count_raw = payload.get("goal_count")
    if goal_count_raw is None or str(goal_count_raw).strip() == "":
        goal_count = int(engine.cfg.env.goal_count)
    else:
        goal_count = int(goal_count_raw)
    if goal_count <= 0:
        raise ValueError("goal_count must be >= 1.")
    show_route_plot = _parse_bool(payload.get("show_route_plot"), default=True)

    policy_root = project_root / "Policy_rl_agent"
    route_plot_path = policy_root / "artifacts" / "web_route_latest.png"

    with _ROUTE_LOCK:
        resolved_start, resolved_goals = select_start_and_goals_on_graph(
            graph=engine.graph,
            goal_count=goal_count,
            title=f"Web planner: click 1 start + {goal_count} goals",
        )

    if not resolved_goals:
        raise ValueError("No goals selected.")

    reset_options = {"start_node": int(resolved_start), "goal_nodes": [int(g) for g in resolved_goals]}
    env = RoadMultiGoalEnv(engine.graph, engine.cfg.env)
    episode = collect_episode(
        env=env,
        policy=engine.policy,
        mode="eval",
        temperature=0.1,
        epsilon=0.0,
        device="cpu",
        reset_options=reset_options,
    )
    plot_graph_with_overlays(
        graph=engine.graph,
        title="Web Policy Route",
        start_node=int(resolved_start),
        goal_nodes=[int(g) for g in resolved_goals],
        path_nodes=[int(n) for n in episode["path_nodes"]],
        route_edges=[
            (int(t["from_node"]), int(t["to_node"])) for t in episode["transitions"]
        ],
        end_node=int(episode["path_nodes"][-1]) if episode["path_nodes"] else None,
        output_path=route_plot_path,
        show=show_route_plot,
    )

    path_nodes = [int(n) for n in episode["path_nodes"]]
    path_coords = _build_path_coords(engine.graph, path_nodes)
    goals_set = {int(g) for g in resolved_goals}
    visited_goal_nodes = sorted(goals_set.intersection(path_nodes))
    travel_time_sec = _estimate_path_travel_time_sec(engine.graph, path_nodes)

    return {
        "config_path": str(engine.config_path),
        "checkpoint_path": str(engine.checkpoint_path),
        "start_node": int(resolved_start),
        "goal_nodes": [int(g) for g in resolved_goals],
        "visited_goal_nodes": visited_goal_nodes,
        "completed_all_goals": bool(episode["success"]),
        "goals_covered_ratio": float(episode["goals_covered_ratio"]),
        "episode_length": int(episode["episode_length"]),
        "episode_return": float(episode["episode_return"]),
        "max_steps": int(env.max_steps),
        "estimated_travel_time_sec": float(travel_time_sec),
        "route_plot_path": str(route_plot_path),
        "route_plot_shown": bool(show_route_plot),
        "path_nodes": path_nodes,
        "path_coords": path_coords,
    }
