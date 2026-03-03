from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from src.cli_inputs import (
    parse_latlon_pairs,
    parse_node_list,
    parse_single_latlon,
    resolve_start_and_goals,
    validate_selection_mode_conflicts,
)
from src.config import load_config
from src.env.road_multi_goal_env import RoadMultiGoalEnv
from src.eval.baselines import evaluate_random_policy
from src.eval.evaluate import evaluate
from src.graph_loader import load_singapore_graph
from src.models.policy_net import PolicyNet
from src.training.loop import collect_episode
from src.visualization.plot_graph import (
    plot_graph_with_overlays,
    select_start_and_goals_on_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policy checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--goal-count", type=int, default=None)
    parser.add_argument("--start-node", type=int, default=None)
    parser.add_argument("--goal-nodes", type=str, default=None)
    parser.add_argument("--start-coord", type=str, default=None)
    parser.add_argument("--goal-coords", type=str, default=None)
    parser.add_argument(
        "--select-on-graph",
        action="store_true",
        help="Interactively click start and goals on the graph (1 start + goal-count goals).",
    )
    parser.add_argument(
        "--show-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show graph window for the final evaluated route.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Optional output path for saved route plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.select_on_graph and not args.show_graph:
        raise ValueError("--select-on-graph requires --show-graph.")
    cfg = load_config(args.config)
    if args.goal_count is not None:
        cfg.env.goal_count = args.goal_count

    print("Loading graph...")
    graph = load_singapore_graph(cfg.graph)
    cli_goal_nodes = parse_node_list(args.goal_nodes)
    cli_goal_coords = parse_latlon_pairs(args.goal_coords)
    cli_start_coord = parse_single_latlon(args.start_coord)
    validate_selection_mode_conflicts(
        select_on_graph=args.select_on_graph,
        start_node=args.start_node,
        start_coord=cli_start_coord,
        goal_nodes=cli_goal_nodes,
        goal_coords=cli_goal_coords,
    )
    if args.select_on_graph:
        print(
            f"Interactive selection mode: click 1 start + {cfg.env.goal_count} goals."
        )
        fixed_start_node, fixed_goal_nodes = select_start_and_goals_on_graph(
            graph=graph,
            goal_count=cfg.env.goal_count,
            title="Evaluation: click start, then goals",
        )
    else:
        fixed_start_node, fixed_goal_nodes = resolve_start_and_goals(
            graph=graph,
            start_node=args.start_node,
            start_coord=cli_start_coord,
            goal_nodes=cli_goal_nodes,
            goal_coords=cli_goal_coords,
        )
    if fixed_goal_nodes is not None:
        cfg.env.goal_count = len(fixed_goal_nodes)

    reset_options = {}
    if fixed_start_node is not None:
        reset_options["start_node"] = fixed_start_node
    if fixed_goal_nodes is not None:
        reset_options["goal_nodes"] = fixed_goal_nodes
    reset_options = reset_options or None

    probe_env = RoadMultiGoalEnv(graph, cfg.env)
    probe_obs, _ = probe_env.reset(seed=cfg.env.seed, options=reset_options)
    state_dim = int(probe_obs["state_vec"].shape[0])
    edge_feat_dim = int(probe_obs["edge_feats"].shape[1])

    policy = PolicyNet(
        state_dim=state_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=cfg.model.policy_hidden_dim,
    )
    payload = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    def env_factory(seed_offset: int) -> RoadMultiGoalEnv:
        eval_cfg = replace(cfg.env, seed=cfg.env.seed + 20_000 + seed_offset)
        return RoadMultiGoalEnv(graph, eval_cfg)

    policy_metrics = evaluate(
        policy=policy,
        env_factory=env_factory,
        n_episodes=args.episodes,
        device="cpu",
        deterministic=True,
        reset_options=reset_options,
    )
    random_metrics = evaluate_random_policy(
        env_factory=env_factory,
        n_episodes=args.episodes,
        reset_options=reset_options,
    )
    denom = abs(random_metrics["mean_return"]) + 1e-6
    improvement = (policy_metrics["mean_return"] - random_metrics["mean_return"]) / denom

    demo_episode = collect_episode(
        env=probe_env,
        policy=policy,
        mode="eval",
        temperature=0.1,
        epsilon=0.0,
        device="cpu",
        reset_options=reset_options,
    )
    plot_path = Path(args.plot_path) if args.plot_path else Path("artifacts/eval_route.png")
    plot_graph_with_overlays(
        graph=graph,
        title="Evaluated Policy Route",
        start_node=int(demo_episode["start_node"]),
        goal_nodes=[int(g) for g in demo_episode["goal_nodes"]],
        path_nodes=[int(n) for n in demo_episode["path_nodes"]],
        route_edges=[
            (int(t["from_node"]), int(t["to_node"])) for t in demo_episode["transitions"]
        ],
        end_node=int(demo_episode["path_nodes"][-1]) if demo_episode["path_nodes"] else None,
        output_path=plot_path,
        show=args.show_graph,
    )

    print("Policy metrics:", policy_metrics)
    print("Random metrics:", random_metrics)
    print(
        "Demo episode:",
        {
            "length": demo_episode["episode_length"],
            "success": demo_episode["success"],
            "goals_covered_ratio": demo_episode["goals_covered_ratio"],
            "start_node": demo_episode["start_node"],
            "goal_nodes": demo_episode["goal_nodes"],
        },
    )
    print(f"Improvement vs random: {improvement:.3f}")
    print(f"Route plot: {plot_path}")


if __name__ == "__main__":
    main()
