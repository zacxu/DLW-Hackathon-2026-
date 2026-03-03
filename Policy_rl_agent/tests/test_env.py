from __future__ import annotations

import numpy as np

from src.features.edge_features import get_outgoing_actions


def test_env_reset_observation_schema(toy_env) -> None:
    obs, info = toy_env.reset(seed=123)
    assert obs["state_vec"].shape == (toy_env.state_dim,)
    assert obs["edge_feats"].shape == (toy_env.max_actions, toy_env.edge_feat_dim)
    assert obs["action_mask"].shape == (toy_env.max_actions,)
    assert int(obs["valid_action_count"]) >= 1
    assert "goal_nodes" in info


def test_invalid_action_is_deterministic(toy_env) -> None:
    toy_env.reset(seed=7, options={"start_node": 0, "goal_nodes": [2, 5]})
    actions = get_outgoing_actions(toy_env.graph, toy_env.current_node)
    expected_next = actions[0].v

    _, _, _, _, info = toy_env.step(999)
    assert info["invalid_action"] is True
    assert toy_env.current_node == expected_next


def test_goal_progression_and_termination(toy_env) -> None:
    toy_env.reset(seed=11, options={"start_node": 0, "goal_nodes": [1]})
    actions = get_outgoing_actions(toy_env.graph, toy_env.current_node)
    idx_to_goal = next(i for i, a in enumerate(actions) if a.v == 1)

    _, reward, terminated, truncated, _ = toy_env.step(idx_to_goal)
    assert truncated is False
    assert terminated is True
    assert reward > 0.0  # completion bonus applied
    assert len(toy_env.remaining_goals) == 0


def test_step_cap_is_deterministic_and_scales(toy_env) -> None:
    cap_one = toy_env._estimate_step_cap(start=0, goals=[2])
    cap_two = toy_env._estimate_step_cap(start=0, goals=[2, 5])
    cap_one_repeat = toy_env._estimate_step_cap(start=0, goals=[2])

    assert cap_one > 0
    assert cap_two >= cap_one
    assert cap_one == cap_one_repeat


def test_reset_with_only_goal_nodes(toy_env) -> None:
    obs, info = toy_env.reset(seed=42, options={"goal_nodes": [2, 5]})
    assert info["start_node"] not in info["goal_nodes"]
    assert sorted(info["goal_nodes"]) == [2, 5]
    assert int(obs["valid_action_count"]) >= 1


def test_reset_with_only_start_node(toy_env) -> None:
    _obs, info = toy_env.reset(seed=42, options={"start_node": 0})
    assert info["start_node"] == 0
    assert len(info["goal_nodes"]) == toy_env.cfg.goal_count
    assert 0 not in info["goal_nodes"]
