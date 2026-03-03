from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Iterable, Sequence

import networkx as nx
import numpy as np


@dataclass
class GraphStats:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bbox_diag: float
    max_out_degree: int
    median_edge_travel_time: float
    p95_travel_time: float
    p95_length: float
    p95_speed: float


def compute_graph_stats(graph: nx.MultiDiGraph) -> GraphStats:
    xs = np.array([float(graph.nodes[n]["x"]) for n in graph.nodes()], dtype=np.float64)
    ys = np.array([float(graph.nodes[n]["y"]) for n in graph.nodes()], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox_diag = max(hypot(x_max - x_min, y_max - y_min), 1e-6)

    out_degrees = [graph.out_degree(n) for n in graph.nodes()]
    max_out_degree = max(max(out_degrees), 1)

    travel_times = []
    lengths = []
    speeds = []
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        travel_times.append(float(data.get("travel_time", 0.0)))
        lengths.append(float(data.get("length", 0.0)))
        speeds.append(float(data.get("speed_kph", 0.0)))

    tt_arr = np.array(travel_times, dtype=np.float64)
    len_arr = np.array(lengths, dtype=np.float64)
    spd_arr = np.array(speeds, dtype=np.float64)

    return GraphStats(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        bbox_diag=bbox_diag,
        max_out_degree=max_out_degree,
        median_edge_travel_time=float(np.median(tt_arr)) if tt_arr.size else 1.0,
        p95_travel_time=float(np.percentile(tt_arr, 95)) if tt_arr.size else 1.0,
        p95_length=float(np.percentile(len_arr, 95)) if len_arr.size else 1.0,
        p95_speed=float(np.percentile(spd_arr, 95)) if spd_arr.size else 1.0,
    )


def _node_xy(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    attrs = graph.nodes[node]
    return float(attrs["x"]), float(attrs["y"])


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (v - lo) / (hi - lo)


def _nearest_goal_stats(
    graph: nx.MultiDiGraph,
    node: int,
    remaining_goals: Sequence[int],
    stats: GraphStats,
) -> tuple[float, float, float, float, float]:
    if not remaining_goals:
        return 0.0, 0.0, 0.0, 0.0, 1.0

    x0, y0 = _node_xy(graph, node)
    dists = []
    goal_angles = []
    for g in remaining_goals:
        xg, yg = _node_xy(graph, g)
        dx = xg - x0
        dy = yg - y0
        d = hypot(dx, dy)
        dists.append(d / stats.bbox_diag)
        goal_angles.append(atan2(dy, dx))

    d_arr = np.array(dists, dtype=np.float64)
    nearest_i = int(d_arr.argmin())
    nearest_angle = goal_angles[nearest_i]
    return (
        float(d_arr.min()),
        float(d_arr.mean()),
        float(d_arr.max()),
        float(sin(nearest_angle)),
        float(cos(nearest_angle)),
    )


def encode_state_features(
    graph: nx.MultiDiGraph,
    node: int,
    remaining_goals: Iterable[int],
    total_goal_count: int,
    step_count: int,
    max_steps: int,
    stats: GraphStats,
) -> np.ndarray:
    rem = list(remaining_goals)
    x, y = _node_xy(graph, node)
    x_norm = _normalize(x, stats.x_min, stats.x_max)
    y_norm = _normalize(y, stats.y_min, stats.y_max)
    out_degree_norm = graph.out_degree(node) / max(stats.max_out_degree, 1)
    remaining_ratio = len(rem) / max(total_goal_count, 1)
    min_d, mean_d, max_d, bearing_sin, bearing_cos = _nearest_goal_stats(
        graph, node, rem, stats
    )
    step_ratio = step_count / max(max_steps, 1)
    is_current_goal = 1.0 if node in rem else 0.0

    return np.array(
        [
            x_norm,
            y_norm,
            out_degree_norm,
            remaining_ratio,
            min_d,
            mean_d,
            max_d,
            bearing_sin,
            bearing_cos,
            step_ratio,
            is_current_goal,
        ],
        dtype=np.float32,
    )
