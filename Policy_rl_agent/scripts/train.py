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
from src.graph_loader import load_singapore_graph
from src.models.policy_net import PolicyNet
from src.models.value_net import ValueNet
from src.training.loop import collect_episode, run_training_loop
from src.training.policy_imitation_trainer import PolicyImitationTrainer
from src.training.value_trainer import ValueTrainer
from src.visualization.plot_graph import (
    plot_graph_with_overlays,
    select_start_and_goals_on_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-goal road graph RL model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--goal-count", type=int, default=None, help="Override goal count.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--start-node",
        type=int,
        default=None,
        help="Fixed start node id for episodes.",
    )
    parser.add_argument(
        "--goal-nodes",
        type=str,
        default=None,
        help="Comma-separated goal node ids, e.g. '123,456,789'.",
    )
    parser.add_argument(
        "--start-coord",
        type=str,
        default=None,
        help="Start point as 'lat,lon'. Nearest graph node is used.",
    )
    parser.add_argument(
        "--goal-coords",
        type=str,
        default=None,
        help="Goal points as 'lat,lon;lat,lon;...'. Nearest graph nodes are used.",
    )
    parser.add_argument(
        "--show-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show graph windows (overview and final route).",
    )
    parser.add_argument(
        "--select-on-graph",
        action="store_true",
        help="Interactively click start and goals on the graph (1 start + goal-count goals).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.select_on_graph and not args.show_graph:
        raise ValueError("--select-on-graph requires --show-graph.")
    cfg = load_config(args.config)
    if args.goal_count is not None:
        cfg.env.goal_count = args.goal_count
    output_dir = Path(args.output_dir or cfg.output.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Singapore road graph...")
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
            title="Training: click start, then goals",
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

    print(
        "Graph ready: "
        f"nodes={graph.number_of_nodes()} "
        f"edges={graph.number_of_edges()} "
        f"goal_count={cfg.env.goal_count}"
    )
    if reset_options:
        print(f"Using fixed episode nodes: {reset_options}")
    else:
        print("Using sampled start/goals each episode.")

    train_env = RoadMultiGoalEnv(graph, cfg.env)
    probe_obs, probe_info = train_env.reset(seed=cfg.env.seed, options=reset_options)
    state_dim = int(probe_obs["state_vec"].shape[0])
    edge_feat_dim = int(probe_obs["edge_feats"].shape[1])

    overview_path = output_dir / "graph_overview.png"
    plot_graph_with_overlays(
        graph=graph,
        title="Singapore Subgraph (Training)",
        start_node=int(probe_info["start_node"]),
        goal_nodes=[int(g) for g in probe_info["goal_nodes"]],
        output_path=overview_path,
        show=args.show_graph,
    )
    print(f"Saved graph overview: {overview_path}")

    value_net = ValueNet(
        state_dim=state_dim,
        hidden_dims=(cfg.model.value_hidden_dim, cfg.model.value_hidden_dim),
    )
    policy_net = PolicyNet(
        state_dim=state_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=cfg.model.policy_hidden_dim,
    )

    device = "cpu"
    value_trainer = ValueTrainer(
        model=value_net,
        gamma=cfg.train.gamma,
        lr=cfg.train.value_lr,
        batch_size=cfg.train.batch_size,
        device=device,
    )
    policy_trainer = PolicyImitationTrainer(
        model=policy_net,
        lr=cfg.train.policy_lr,
        batch_size=cfg.train.batch_size,
        device=device,
    )

    def eval_env_factory(seed_offset: int) -> RoadMultiGoalEnv:
        eval_cfg = replace(cfg.env, seed=cfg.env.seed + 10_000 + seed_offset)
        return RoadMultiGoalEnv(graph, eval_cfg)

    summary = run_training_loop(
        train_env=train_env,
        eval_env_factory=eval_env_factory,
        policy_net=policy_net,
        value_trainer=value_trainer,
        policy_trainer=policy_trainer,
        cfg=cfg.train,
        output_dir=output_dir,
        device=device,
        train_reset_options=reset_options,
        eval_reset_options=reset_options,
    )

    final_episode = collect_episode(
        env=train_env,
        policy=policy_net,
        mode="eval",
        temperature=0.1,
        epsilon=0.0,
        device=device,
        reset_options=reset_options,
    )
    final_route_path = output_dir / "final_route.png"
    plot_graph_with_overlays(
        graph=graph,
        title="Final Policy Route",
        start_node=int(final_episode["start_node"]),
        goal_nodes=[int(g) for g in final_episode["goal_nodes"]],
        path_nodes=[int(n) for n in final_episode["path_nodes"]],
        route_edges=[
            (int(t["from_node"]), int(t["to_node"])) for t in final_episode["transitions"]
        ],
        end_node=int(final_episode["path_nodes"][-1]) if final_episode["path_nodes"] else None,
        output_path=final_route_path,
        show=args.show_graph,
    )

    print("Training complete.")
    print(f"Metrics: {summary['metrics_path']}")
    print(f"Best checkpoint: {summary['best_checkpoint']}")
    print(f"Final route plot: {final_route_path}")
    print(
        "Final episode:",
        {
            "length": final_episode["episode_length"],
            "success": final_episode["success"],
            "goals_covered_ratio": final_episode["goals_covered_ratio"],
            "start_node": final_episode["start_node"],
            "goal_nodes": final_episode["goal_nodes"],
        },
    )
    if summary["last_eval"]:
        last_imp = summary["last_eval"]["improvement_vs_random"]
        print(f"Last eval improvement vs random: {last_imp:.3f}")
    if summary.get("best_eval"):
        best_imp = summary["best_eval"]["improvement_vs_random"]
        best_success = summary["best_eval"]["policy"]["success_rate"]
        best_coverage = summary["best_eval"]["policy"]["mean_goals_covered_ratio"]
        print(
            "Best eval:",
            {
                "improvement_vs_random": round(best_imp, 3),
                "success_rate": round(best_success, 3),
                "goals_covered_ratio": round(best_coverage, 3),
            },
        )
        if best_imp >= 0.25:
            print("Milestone target reached (>=25% better than random).")
        else:
            print("Milestone target not reached yet.")


if __name__ == "__main__":
    main()
