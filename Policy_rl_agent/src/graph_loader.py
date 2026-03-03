from __future__ import annotations

import warnings
from typing import Iterable

import networkx as nx
import osmnx as ox

from src.config import GraphConfig


def extract_largest_component(
    graph: nx.MultiDiGraph, network_type: str = "drive"
) -> nx.MultiDiGraph:
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty; cannot extract a connected component.")

    if network_type == "drive":
        components = list(nx.strongly_connected_components(graph))
    else:
        components = list(nx.weakly_connected_components(graph))

    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def validate_graph_for_env(graph: nx.MultiDiGraph) -> None:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Graph must be non-empty.")
    if not graph.is_directed():
        raise ValueError("Graph must be directed for this environment.")

    missing: list[tuple[int, int, int]] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        tt = data.get("travel_time")
        if tt is None or tt <= 0:
            missing.append((u, v, k))
            if len(missing) >= 5:
                break
    if missing:
        raise ValueError(
            f"Graph edges missing positive travel_time. Sample edges: {missing}"
        )


def _ensure_node_coordinates(graph: nx.MultiDiGraph, nodes: Iterable[int]) -> None:
    for node in nodes:
        attrs = graph.nodes[node]
        if "x" not in attrs or "y" not in attrs:
            raise ValueError(f"Node {node} is missing x/y coordinates.")


def load_singapore_graph(cfg: GraphConfig) -> nx.MultiDiGraph:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The expected order of coordinates in `bbox` will change in the v2.0.0 release",
            category=FutureWarning,
        )
        graph = ox.graph_from_point(
            (cfg.center_lat, cfg.center_lon),
            dist=cfg.dist_m,
            network_type=cfg.network_type,
            simplify=cfg.simplify,
        )
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    graph = extract_largest_component(graph, cfg.network_type)
    _ensure_node_coordinates(graph, graph.nodes())
    validate_graph_for_env(graph)
    return graph
