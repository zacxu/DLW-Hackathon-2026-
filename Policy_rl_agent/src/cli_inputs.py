from __future__ import annotations

from typing import Sequence

import networkx as nx
import osmnx as ox


def parse_node_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens:
        raise ValueError("Node list is empty.")
    return [int(t) for t in tokens]


def parse_latlon_pairs(raw: str | None) -> list[tuple[float, float]] | None:
    if raw is None:
        return None
    chunks = [x.strip() for x in raw.split(";") if x.strip()]
    if not chunks:
        raise ValueError("Coordinate list is empty.")

    pairs: list[tuple[float, float]] = []
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                "Invalid coordinate format. Use 'lat,lon;lat,lon' for multiple points."
            )
        lat = float(parts[0])
        lon = float(parts[1])
        pairs.append((lat, lon))
    return pairs


def parse_single_latlon(raw: str | None) -> tuple[float, float] | None:
    pairs = parse_latlon_pairs(raw)
    if pairs is None:
        return None
    if len(pairs) != 1:
        raise ValueError("Expected one coordinate in --start-coord as 'lat,lon'.")
    return pairs[0]


def _nearest_node_bruteforce(graph: nx.MultiDiGraph, x: float, y: float) -> int:
    best_node: int | None = None
    best_dist2 = float("inf")
    for node, attrs in graph.nodes(data=True):
        nx_ = float(attrs["x"])
        ny_ = float(attrs["y"])
        dx = nx_ - x
        dy = ny_ - y
        dist2 = dx * dx + dy * dy
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_node = int(node)

    if best_node is None:
        raise ValueError("Graph has no nodes.")
    return best_node


def resolve_nearest_node(
    graph: nx.MultiDiGraph, lat: float, lon: float
) -> int:
    # OSMnx nearest_nodes on unprojected graphs requires scikit-learn.
    # Fall back to a deterministic brute-force nearest node if sklearn is absent.
    try:
        node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
        return int(node)
    except ImportError:
        return _nearest_node_bruteforce(graph, x=lon, y=lat)


def resolve_start_and_goals(
    graph: nx.MultiDiGraph,
    start_node: int | None = None,
    start_coord: tuple[float, float] | None = None,
    goal_nodes: Sequence[int] | None = None,
    goal_coords: Sequence[tuple[float, float]] | None = None,
) -> tuple[int | None, list[int] | None]:
    if start_node is not None and start_coord is not None:
        raise ValueError("Use either --start-node or --start-coord, not both.")
    if goal_nodes is not None and goal_coords is not None:
        raise ValueError("Use either --goal-nodes or --goal-coords, not both.")

    resolved_start = start_node
    if start_coord is not None:
        resolved_start = resolve_nearest_node(graph, start_coord[0], start_coord[1])

    resolved_goals = None
    if goal_nodes is not None:
        resolved_goals = [int(g) for g in goal_nodes]
    elif goal_coords is not None:
        resolved_goals = [
            resolve_nearest_node(graph, lat=lat, lon=lon) for lat, lon in goal_coords
        ]

    return resolved_start, resolved_goals


def validate_selection_mode_conflicts(
    select_on_graph: bool,
    start_node: int | None,
    start_coord: tuple[float, float] | None,
    goal_nodes: Sequence[int] | None,
    goal_coords: Sequence[tuple[float, float]] | None,
) -> None:
    if not select_on_graph:
        return

    if any(
        x is not None
        for x in [start_node, start_coord, goal_nodes, goal_coords]
    ):
        raise ValueError(
            "--select-on-graph cannot be combined with --start-node/--start-coord/"
            "--goal-nodes/--goal-coords."
        )
