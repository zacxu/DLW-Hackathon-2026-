from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class GraphConfig:
    place: str = "Singapore"
    network_type: str = "drive"
    center_lat: float = 1.3048
    center_lon: float = 103.8318
    dist_m: int = 2500
    simplify: bool = True


@dataclass
class EnvConfig:
    goal_count: int = 3
    reward_time_scale_sec: float = 30.0
    completion_bonus: float = 80.0
    goal_reached_bonus: float = 25.0
    progress_reward_scale: float = 10.0
    revisit_penalty: float = 0.35
    revisit_penalty_cap: int = 8
    stagnation_penalty: float = 0.03
    unfinished_goal_penalty: float = 20.0
    max_steps_factor: float = 1.2
    max_steps_hard_cap: int = 140
    seed: int = 7


@dataclass
class TrainConfig:
    episodes: int = 300
    gamma: float = 0.98
    value_lr: float = 1e-3
    policy_lr: float = 1e-3
    batch_size: int = 128
    eval_every: int = 25
    checkpoint_every: int = 50
    temperature_start: float = 1.0
    temperature_end: float = 0.2
    epsilon_explore: float = 0.05
    eval_episodes: int = 30
    teacher_value_weight: float = 0.1
    teacher_goal_cost_weight: float = 2.5
    expert_action_prob_start: float = 0.9
    expert_action_prob_end: float = 0.1
    backtrack_penalty_strength: float = 0.8


@dataclass
class ModelConfig:
    value_hidden_dim: int = 128
    policy_hidden_dim: int = 128


@dataclass
class OutputConfig:
    run_dir: str = "artifacts/run_default"


@dataclass
class ExperimentConfig:
    graph: GraphConfig
    env: EnvConfig
    train: TrainConfig
    model: ModelConfig
    output: OutputConfig


def _merge_dataclass(cls: Any, raw: Dict[str, Any] | None) -> Any:
    raw = raw or {}
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        graph=_merge_dataclass(GraphConfig, raw.get("graph")),
        env=_merge_dataclass(EnvConfig, raw.get("env")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        output=_merge_dataclass(OutputConfig, raw.get("output")),
    )
