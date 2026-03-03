from __future__ import annotations

import networkx as nx
import pytest

from src.config import EnvConfig
from src.env.road_multi_goal_env import RoadMultiGoalEnv


@pytest.fixture
def toy_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (0.0, 1.0),
        4: (1.0, 1.0),
        5: (2.0, 1.0),
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)

    def add(u: int, v: int, tt: float) -> None:
        g.add_edge(
            u,
            v,
            travel_time=tt,
            length=tt * 30.0,
            speed_kph=40.0,
        )

    # Fast corridor.
    add(0, 1, 4.0)
    add(1, 2, 4.0)
    add(2, 5, 4.0)
    add(5, 4, 4.0)
    add(4, 3, 4.0)
    add(3, 0, 4.0)

    # Reverse corridor.
    add(1, 0, 4.5)
    add(2, 1, 4.5)
    add(5, 2, 4.5)
    add(4, 5, 4.5)
    add(3, 4, 4.5)
    add(0, 3, 4.5)

    # Slower cross-links.
    add(0, 4, 10.0)
    add(4, 0, 10.0)
    add(1, 4, 8.0)
    add(4, 1, 8.0)
    add(2, 4, 7.0)
    add(3, 1, 9.0)
    add(5, 1, 10.0)

    return g


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(
        goal_count=2,
        reward_time_scale_sec=10.0,
        completion_bonus=5.0,
        max_steps_factor=2.0,
        seed=13,
    )


@pytest.fixture
def toy_env(toy_graph: nx.MultiDiGraph, env_config: EnvConfig) -> RoadMultiGoalEnv:
    return RoadMultiGoalEnv(toy_graph, env_config)
