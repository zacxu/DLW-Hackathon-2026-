from __future__ import annotations

from src.visualization.plot_graph import _sanitize_route_for_plot, plot_graph_with_overlays


def test_sanitize_route_keeps_valid_edges(toy_graph) -> None:
    # Includes duplicate and invalid node; expected route keeps valid contiguous edges.
    raw = [0, 0, 1, 999999, 2, 4]
    route = _sanitize_route_for_plot(toy_graph, raw)
    assert route == [0, 1, 2, 4]


def test_plot_graph_with_route_saves_image(toy_graph, tmp_path) -> None:
    out = tmp_path / "route.png"
    plot_graph_with_overlays(
        graph=toy_graph,
        title="route",
        start_node=0,
        goal_nodes=[2, 5],
        path_nodes=[0, 1, 2, 5],
        route_edges=[(0, 1), (1, 2), (2, 5)],
        end_node=5,
        output_path=out,
        show=False,
    )
    assert out.exists()
