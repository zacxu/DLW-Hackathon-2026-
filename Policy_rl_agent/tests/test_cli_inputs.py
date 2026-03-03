from __future__ import annotations

import pytest

from src.cli_inputs import (
    parse_latlon_pairs,
    parse_node_list,
    parse_single_latlon,
    resolve_start_and_goals,
    validate_selection_mode_conflicts,
)


def test_parse_node_list() -> None:
    assert parse_node_list("1,2,3") == [1, 2, 3]


def test_parse_latlon_pairs() -> None:
    pairs = parse_latlon_pairs("1.0,103.0;1.1,103.1")
    assert pairs == [(1.0, 103.0), (1.1, 103.1)]


def test_parse_single_latlon() -> None:
    assert parse_single_latlon("1.2,103.2") == (1.2, 103.2)


def test_resolve_start_and_goals_nodes(toy_graph) -> None:
    start, goals = resolve_start_and_goals(
        graph=toy_graph,
        start_node=0,
        goal_nodes=[2, 5],
    )
    assert start == 0
    assert goals == [2, 5]


def test_resolve_nearest_node_fallback_without_sklearn(monkeypatch, toy_graph) -> None:
    import osmnx as ox
    from src.cli_inputs import resolve_nearest_node

    def _raise_import_error(*args, **kwargs):
        raise ImportError("scikit-learn must be installed to search an unprojected graph")

    monkeypatch.setattr(ox.distance, "nearest_nodes", _raise_import_error)
    node = resolve_nearest_node(toy_graph, lat=0.02, lon=0.01)
    assert node == 0


def test_validate_selection_mode_conflicts_ok() -> None:
    validate_selection_mode_conflicts(
        select_on_graph=False,
        start_node=1,
        start_coord=None,
        goal_nodes=[2],
        goal_coords=None,
    )


def test_validate_selection_mode_conflicts_raises() -> None:
    with pytest.raises(ValueError):
        validate_selection_mode_conflicts(
            select_on_graph=True,
            start_node=1,
            start_coord=None,
            goal_nodes=None,
            goal_coords=None,
        )
