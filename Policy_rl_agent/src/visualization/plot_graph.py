from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from matplotlib.collections import LineCollection

from src.cli_inputs import resolve_nearest_node


def _node_xy(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    attrs = graph.nodes[node]
    return float(attrs["x"]), float(attrs["y"])


def _ensure_graph_crs(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if "crs" in graph.graph:
        return graph
    g = graph.copy()
    g.graph["crs"] = "EPSG:4326"
    return g


def _sanitize_route_for_plot(
    graph: nx.MultiDiGraph, path_nodes: Sequence[int] | None
) -> list[int]:
    if not path_nodes:
        return []

    route: list[int] = []
    for node in path_nodes:
        node_i = int(node)
        if node_i not in graph:
            continue
        if not route:
            route.append(node_i)
            continue

        prev = route[-1]
        if node_i == prev:
            continue
        if graph.has_edge(prev, node_i):
            route.append(node_i)

    return route


def _edge_points_for_pair(graph: nx.MultiDiGraph, u: int, v: int) -> list[tuple[float, float]]:
    if not graph.has_edge(u, v):
        return []
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        return []

    # Pick the fastest parallel edge if several exist.
    best = min(
        edge_data.values(),
        key=lambda d: float(d.get("travel_time", float("inf"))),
    )
    geom = best.get("geometry")
    if geom is not None:
        return [(float(x), float(y)) for x, y in geom.coords]

    ux, uy = _node_xy(graph, u)
    vx, vy = _node_xy(graph, v)
    return [(ux, uy), (vx, vy)]


def _draw_route_edges(
    graph: nx.MultiDiGraph,
    ax: plt.Axes,
    route_edges: Sequence[tuple[int, int]],
    color: str = "#e53935",
    linewidth: float = 2.6,
) -> bool:
    segments: list[list[tuple[float, float]]] = []
    for u, v in route_edges:
        pts = _edge_points_for_pair(graph, int(u), int(v))
        if len(pts) >= 2:
            segments.append(pts)

    if not segments:
        return False

    lc = LineCollection(segments, colors=color, linewidths=linewidth, zorder=6)
    ax.add_collection(lc)
    ax.plot([], [], color=color, linewidth=linewidth, label="Policy route")
    return True


def plot_graph_with_overlays(
    graph: nx.MultiDiGraph,
    title: str,
    start_node: int | None = None,
    goal_nodes: Sequence[int] | None = None,
    path_nodes: Sequence[int] | None = None,
    route_edges: Sequence[tuple[int, int]] | None = None,
    end_node: int | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    graph_plot = _ensure_graph_crs(graph)
    fig, ax = ox.plot_graph(
        graph_plot,
        node_size=4,
        edge_linewidth=0.5,
        edge_color="#9aa0a6",
        bgcolor="white",
        show=False,
        close=False,
    )

    drew_edges = False
    if route_edges:
        drew_edges = _draw_route_edges(
            graph=graph_plot,
            ax=ax,
            route_edges=route_edges,
            color="#e53935",
            linewidth=2.6,
        )

    if not drew_edges:
        route = _sanitize_route_for_plot(graph_plot, path_nodes)
        if len(route) >= 2:
            fig, ax = ox.plot_graph_route(
                graph_plot,
                route=route,
                route_color="#e53935",
                route_linewidth=2.6,
                orig_dest_size=0,
                ax=ax,
                show=False,
                close=False,
            )
            # Explicit legend handle for route.
            ax.plot([], [], color="#e53935", linewidth=2.6, label="Policy route")

    if path_nodes:
        unique_route_nodes = [int(n) for n in dict.fromkeys(path_nodes) if int(n) in graph_plot]
        if unique_route_nodes:
            rx = []
            ry = []
            for n in unique_route_nodes:
                x, y = _node_xy(graph_plot, n)
                rx.append(x)
                ry.append(y)
            ax.scatter(
                rx,
                ry,
                c="#e53935",
                s=14,
                alpha=0.85,
                marker="o",
                zorder=6.5,
                label="Visited nodes",
            )

    if goal_nodes:
        gx = []
        gy = []
        for g in goal_nodes:
            x, y = _node_xy(graph_plot, g)
            gx.append(x)
            gy.append(y)
        ax.scatter(
            gx,
            gy,
            c="#2e7d32",
            s=80,
            marker="*",
            zorder=7,
            label="Goals",
        )

    if start_node is not None:
        sx, sy = _node_xy(graph_plot, start_node)
        ax.scatter([sx], [sy], c="#1565c0", s=58, marker="o", zorder=8, label="Start")

    if end_node is not None and end_node in graph_plot:
        ex, ey = _node_xy(graph_plot, int(end_node))
        ax.scatter(
            [ex],
            [ey],
            c="#fb8c00",
            s=64,
            marker="X",
            zorder=8,
            label="End",
        )

    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def select_start_and_goals_on_graph(
    graph: nx.MultiDiGraph,
    goal_count: int,
    title: str = "Click start, then each goal (press Enter when done)",
) -> tuple[int, list[int]]:
    if goal_count <= 0:
        raise ValueError("goal_count must be >= 1 for interactive selection.")

    graph_plot = _ensure_graph_crs(graph)
    fig, ax = ox.plot_graph(
        graph_plot,
        node_size=4,
        edge_linewidth=0.5,
        edge_color="#9aa0a6",
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.set_title(title)
    ax.text(
        0.01,
        0.99,
        f"Select {goal_count + 1} points: 1 start + {goal_count} goals",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
    )

    try:
        plt.show(block=False)
        plt.pause(0.1)
        clicks = plt.ginput(n=goal_count + 1, timeout=0, show_clicks=True)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Interactive selection failed. Use a GUI-enabled environment or pass "
            "--start-node/--goal-nodes or --start-coord/--goal-coords."
        ) from exc

    if len(clicks) != goal_count + 1:
        plt.close(fig)
        raise ValueError(
            f"Expected {goal_count + 1} clicks (1 start + {goal_count} goals), got {len(clicks)}."
        )

    selected_nodes: list[int] = []
    for x, y in clicks:
        selected_nodes.append(resolve_nearest_node(graph=graph_plot, lat=y, lon=x))

    start_node = selected_nodes[0]
    goal_nodes = selected_nodes[1:]

    if len(set(goal_nodes)) != len(goal_nodes):
        plt.close(fig)
        raise ValueError("Duplicate goal nodes selected. Please try again.")
    if start_node in goal_nodes:
        plt.close(fig)
        raise ValueError("Start node overlaps with a goal node. Please try again.")

    sx, sy = _node_xy(graph_plot, start_node)
    gx = []
    gy = []
    for g in goal_nodes:
        x, y = _node_xy(graph_plot, g)
        gx.append(x)
        gy.append(y)
    ax.scatter([sx], [sy], c="#1565c0", s=70, marker="o", zorder=7, label="Start")
    ax.scatter(gx, gy, c="#2e7d32", s=80, marker="*", zorder=7, label="Goals")
    ax.legend(loc="upper right")
    plt.show(block=True)
    plt.close(fig)

    return start_node, goal_nodes
