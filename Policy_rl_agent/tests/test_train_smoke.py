from __future__ import annotations

from src.models.policy_net import PolicyNet
from src.models.value_net import ValueNet
from src.training.loop import collect_episode
from src.training.policy_imitation_trainer import (
    PolicyImitationTrainer,
    build_policy_targets,
)
from src.training.value_trainer import ValueTrainer, build_td_batch_from_trajectory


def test_training_smoke(toy_env) -> None:
    obs, _ = toy_env.reset(seed=17)
    state_dim = obs["state_vec"].shape[0]
    edge_feat_dim = obs["edge_feats"].shape[1]

    value = ValueNet(state_dim=state_dim, hidden_dims=(32, 32))
    policy = PolicyNet(state_dim=state_dim, edge_feat_dim=edge_feat_dim, hidden_dim=32)
    vt = ValueTrainer(model=value, gamma=0.95, lr=1e-3, batch_size=16, device="cpu")
    pt = PolicyImitationTrainer(model=policy, lr=1e-3, batch_size=16, device="cpu")

    traj = collect_episode(
        env=toy_env,
        policy=policy,
        mode="train",
        temperature=0.8,
        epsilon=0.1,
        device="cpu",
    )
    assert len(traj["transitions"]) > 0
    assert len(traj["path_nodes"]) >= 2
    assert traj["start_node"] is not None
    assert len(traj["goal_nodes"]) >= 1

    td_batch = build_td_batch_from_trajectory(traj)
    v_metrics = vt.update_value(td_batch)
    assert "value_loss" in v_metrics

    imitation_batch = build_policy_targets(traj, value, gamma=0.95, device="cpu")
    p_metrics = pt.update_policy(imitation_batch)
    assert "policy_loss" in p_metrics
