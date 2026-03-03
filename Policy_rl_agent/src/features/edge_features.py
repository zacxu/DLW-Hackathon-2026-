from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi
from typing import Sequence

import networkx as nx
import numpy as np

from src.features.state_features import GraphStats


@dataclass
class EdgeAction:
    u: int
    v: int
    key: int
    travel_time: float
    length: float
    speed_kph: float


def node_xy(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    attrs = graph.nodes[node]
    return float(attrs["x"]), float(attrs["y"])


def get_outgoing_actions(graph: nx.MultiDiGraph, node: int) -> list[EdgeAction]:
    actions: list[EdgeAction] = []
    for u, v, k, data in graph.out_edges(node, keys=True, data=True):
        actions.append(
            EdgeAction(
                u=u,
                v=v,
                key=k,
                travel_time=float(data.get("travel_time", 1.0)),
                length=float(data.get("length", 1.0)),
                speed_kph=float(data.get("speed_kph", 30.0)),
            )
        )
    return actions


def _nearest_goal(graph: nx.MultiDiGraph, node: int, goals: Sequence[int]) -> int | None:
    if not goals:
        return None
    x0, y0 = node_xy(graph, node)
    best = None
    best_d = float("inf")
    for g in goals:
        xg, yg = node_xy(graph, g)
        d = hypot(xg - x0, yg - y0)
        if d < best_d:
            best_d = d
            best = g
    return best


def build_edge_features(
    graph: nx.MultiDiGraph,
    node: int,
    actions: Sequence[EdgeAction],
    remaining_goals: Sequence[int],
    stats: GraphStats,
) -> np.ndarray:
    feats = np.zeros((len(actions), 5), dtype=np.float32)
    goal = _nearest_goal(graph, node, remaining_goals)

    x0, y0 = node_xy(graph, node)
    if goal is not None:
        xg, yg = node_xy(graph, goal)
        goal_dist = hypot(xg - x0, yg - y0)
        goal_bearing = atan2(yg - y0, xg - x0)
    else:
        goal_dist = 0.0
        goal_bearing = 0.0

    for i, action in enumerate(actions):
        x1, y1 = node_xy(graph, action.v)
        edge_bearing = atan2(y1 - y0, x1 - x0)
        if goal is not None:
            next_goal_dist = hypot(xg - x1, yg - y1)
            improvement = (goal_dist - next_goal_dist) / max(stats.bbox_diag, 1e-6)
            angle_delta = edge_bearing - goal_bearing
            while angle_delta > pi:
                angle_delta -= 2 * pi
            while angle_delta < -pi:
                angle_delta += 2 * pi
            heading_delta = angle_delta / pi
        else:
            improvement = 0.0
            heading_delta = 0.0

        feats[i, 0] = action.travel_time / max(stats.p95_travel_time, 1e-6)
        feats[i, 1] = action.length / max(stats.p95_length, 1e-6)
        feats[i, 2] = action.speed_kph / max(stats.p95_speed, 1e-6)
        feats[i, 3] = heading_delta
        feats[i, 4] = improvement

    return feats
