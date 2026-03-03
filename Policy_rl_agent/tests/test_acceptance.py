from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import TrainConfig
from src.env.road_multi_goal_env import RoadMultiGoalEnv
from src.models.policy_net import PolicyNet
from src.models.value_net import ValueNet
from src.training.loop import run_training_loop
from src.training.policy_imitation_trainer import PolicyImitationTrainer
from src.training.value_trainer import ValueTrainer


@pytest.mark.slow
def test_acceptance_beats_random_by_25_percent(tmp_path, toy_graph, env_config) -> None:
    train_env_cfg = replace(env_config, goal_count=1, seed=23)
    train_env = RoadMultiGoalEnv(toy_graph, train_env_cfg)
    probe_obs, _ = train_env.reset(seed=train_env_cfg.seed)
    state_dim = probe_obs["state_vec"].shape[0]
    edge_feat_dim = probe_obs["edge_feats"].shape[1]

    value_net = ValueNet(state_dim=state_dim, hidden_dims=(64, 64))
    policy_net = PolicyNet(state_dim=state_dim, edge_feat_dim=edge_feat_dim, hidden_dim=64)
    value_trainer = ValueTrainer(
        model=value_net,
        gamma=0.95,
        lr=1e-3,
        batch_size=32,
        device="cpu",
    )
    policy_trainer = PolicyImitationTrainer(
        model=policy_net,
        lr=1e-3,
        batch_size=32,
        device="cpu",
    )

    cfg = TrainConfig(
        episodes=80,
        gamma=0.95,
        value_lr=1e-3,
        policy_lr=1e-3,
        batch_size=32,
        eval_every=20,
        checkpoint_every=40,
        temperature_start=1.0,
        temperature_end=0.2,
        epsilon_explore=0.05,
        eval_episodes=20,
    )

    def eval_env_factory(seed_offset: int) -> RoadMultiGoalEnv:
        eval_cfg = replace(train_env_cfg, seed=train_env_cfg.seed + 5000 + seed_offset)
        return RoadMultiGoalEnv(toy_graph, eval_cfg)

    summary = run_training_loop(
        train_env=train_env,
        eval_env_factory=eval_env_factory,
        policy_net=policy_net,
        value_trainer=value_trainer,
        policy_trainer=policy_trainer,
        cfg=cfg,
        output_dir=tmp_path,
        device="cpu",
    )

    assert summary["last_eval"], "No evaluation results were recorded."
    assert summary["last_eval"]["improvement_vs_random"] >= 0.25
