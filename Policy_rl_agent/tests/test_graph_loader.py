from __future__ import annotations

import networkx as nx

from src.graph_loader import extract_largest_component, validate_graph_for_env


def test_graph_validation_accepts_toy_graph(toy_graph: nx.MultiDiGraph) -> None:
    validate_graph_for_env(toy_graph)


def test_extract_largest_component_for_drive() -> None:
    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=1.0, y=0.0)
    g.add_node(2, x=2.0, y=0.0)
    g.add_node(3, x=100.0, y=100.0)

    g.add_edge(0, 1, travel_time=1.0, length=10.0, speed_kph=40.0)
    g.add_edge(1, 0, travel_time=1.0, length=10.0, speed_kph=40.0)
    g.add_edge(1, 2, travel_time=1.0, length=10.0, speed_kph=40.0)
    g.add_edge(2, 1, travel_time=1.0, length=10.0, speed_kph=40.0)
    g.add_edge(3, 3, travel_time=1.0, length=10.0, speed_kph=40.0)

    largest = extract_largest_component(g, network_type="drive")
    assert set(largest.nodes()) == {0, 1, 2}
    validate_graph_for_env(largest)
