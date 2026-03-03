# Multi-Goal RL Pathfinding on Singapore Road Graph

This project implements a modular reinforcement learning system for weighted directed graph pathfinding with multiple goals on a Singapore road subgraph (OSMnx `drive` network).

The system trains two neural networks:

- `ValueNet`: estimates state value `V(s)` with TD(0).
- `PolicyNet`: predicts logits over valid outgoing edges and is trained by imitation of a value/goal-cost-based teacher.

## Executive Summary

- Environment: Gymnasium-compatible, weighted directed road graph, multiple goals visited in any order.
- Core objective: maximize return while covering all goals before truncation.
- Current training approach: TD value learning + teacher imitation + exploration shaping.
- Key anti-loop upgrades:
- shortest-path goal-cost features included in per-edge policy inputs
- expert-action curriculum during rollout collection
- backtrack probability penalty during sampling
- Milestone criterion: policy mean return at least `25%` better than random baseline.

## Methodology

### 1) Graph Construction Pipeline

The graph pipeline in `src/graph_loader.py` does:

1. Download subgraph around configured center point using OSMnx (`graph_from_point`).
2. Add edge speeds and travel times (`add_edge_speeds`, `add_edge_travel_times`).
3. Keep largest strongly connected component for `drive` graphs.
4. Validate graph integrity:
- directed graph
- non-empty graph
- each edge has positive `travel_time`
- all nodes have `x/y` coordinates

Default area is configured in `configs/default.yaml`:

- `center_lat: 1.3048`
- `center_lon: 103.8318`
- `dist_m: 1800`
- `network_type: drive`

### 2) MDP / Environment Design

Environment: `src/env/road_multi_goal_env.py`

State:

- current node
- remaining goal set
- step count
- visit counts (for anti-loop penalties)

Episode initialization:

- sample start + goals, or accept fixed `start_node`/`goal_nodes`
- terminate when all goals are visited
- truncate when dynamic step cap is hit

Step cap estimator:

- uses shortest travel-time estimates over `{start + goals}`
- nearest-goal greedy chain estimate
- scaled by `max_steps_factor`
- clipped by `max_steps_hard_cap`

### 3) Feature Engineering

State feature vector (`state_dim=11`, `src/features/state_features.py`):

- normalized node `x/y`
- normalized out-degree
- remaining-goal ratio
- min/mean/max straight-line distance to remaining goals
- nearest-goal bearing `sin/cos`
- normalized step ratio
- current-node-is-goal flag

Action feature matrix (`edge_feat_dim=7`, `src/features/edge_features.py` + env augmentation):

- base edge features:
- normalized `travel_time`
- normalized `length`
- normalized `speed_kph`
- heading delta vs nearest-goal direction
- straight-line distance improvement to nearest goal
- added shortest-path cost features:
- normalized shortest travel-time from next node to remaining goals
- normalized `edge_travel_time + next_goal_cost`

The final two features are critical to non-myopic decisions and loop reduction.

### 4) Reward Function

Per transition reward in `RoadMultiGoalEnv._compute_transition_reward`:

`r = -(travel_time / reward_time_scale)`

plus shaping:

- shortest-path progress reward:
`+ progress_reward_scale * ((d_before - d_after) / reward_time_scale)`
- `+ goal_reached_bonus` when a new goal is visited
- `- revisit_penalty * revisit_count` (capped) for repeated nodes
- `- stagnation_penalty` when no progress and no goal reached
- `+ completion_bonus` when all goals finished
- `- unfinished_goal_penalty * remaining_goals` on truncation

### 5) Model Architecture

`ValueNet` (`src/models/value_net.py`):

- MLP with ReLU hidden layers
- scalar output `V(s)`

`PolicyNet` (`src/models/policy_net.py`):

- state encoder MLP
- edge encoder MLP
- concatenation and scoring MLP per action
- invalid actions masked to large negative logits

### 6) Training Approach

Training loop: `src/training/loop.py`

Per episode:

1. Collect trajectory with stochastic sampling.
2. Update `ValueNet` with TD(0), smooth L1 loss.
3. Build teacher actions for policy imitation.
4. Update `PolicyNet` with masked cross-entropy.

Teacher action scoring (`src/training/policy_imitation_trainer.py`):

`q_teacher(s,a) = w_v * [r + gamma * (1-done) * V(s')] - w_c * total_goal_cost(s,a)`

where:

- `total_goal_cost = edge_travel_time + shortest_cost(next_node -> remaining_goals)`
- `w_v = teacher_value_weight`
- `w_c = teacher_goal_cost_weight`

Exploration + anti-loop controls:

- softmax temperature schedule
- epsilon mixing with uniform valid-action distribution
- expert-action curriculum:
- probability decays from `expert_action_prob_start` to `expert_action_prob_end`
- expert action is `argmin(next_total_goal_costs)` from lookahead
- backtrack penalty:
- downweights probability of immediately returning to previous node

Checkpointing and model selection:

- periodic `latest.pt`
- `best.pt` selected by:
- higher goal coverage
- then higher success rate
- then higher return

Milestone reporting:

- final status now reports using `best_eval` (not only `last_eval`) to avoid false negatives after late-run regression.

## User Interaction Workflow

Manual node selection is supported for both training and evaluation:

- `--select-on-graph` opens a matplotlib GUI.
- Click exactly `1 start + K goals`.
- Alternative: pass `--start-node/--goal-nodes` or `--start-coord/--goal-coords`.

Nearest-node resolution:

- uses OSMnx nearest-node when available
- deterministic brute-force fallback when `scikit-learn` is unavailable for unprojected graphs

## Results and Observations (Artifact Snapshot)

The table below summarizes metrics extracted from saved `metrics.jsonl` files in `artifacts/` as of **March 3, 2026**.

| Run directory | Best episode | Best return | Best success | Best coverage | Best improvement vs random | Last improvement |
|---|---:|---:|---:|---:|---:|---:|
| `run_tuned_quick` | 120 | -261.9610 | 0.0000 | 0.0333 | -1.5734 | -1.5734 |
| `run_tuned_quick_v2` | 60 | -242.9511 | 0.0667 | 0.1333 | -1.1113 | -1.3616 |
| `run_tuned_quick_v3` | 80 | -273.9973 | 0.0000 | 0.0333 | -1.3812 | -1.3858 |
| `run_tuned_goalfix_quick` | 60 | 144.0795 | 0.9500 | 0.9500 | 2.2111 | 1.4256 |
| `run_tuned_goalfix_quick2` | 80 | 166.7325 | 1.0000 | 1.0000 | 2.4015 | 2.4014 |
| `run_default` | 380 | 226.5100 | 1.0000 | 1.0000 | 2.4974 | -0.4055 |
| `run_milestone_check` | 20 | 173.7978 | 1.0000 | 1.0000 | 2.3486 | 2.3486 |

Interpretation:

- Early tuned variants (`run_tuned_quick*`) were loop-prone and failed the milestone.
- After goal-cost features + expert curriculum + backtrack penalty, short runs (`run_tuned_goalfix_quick2`) reached full success/coverage.
- `run_default` demonstrates why relying only on `last_eval` can be misleading: best metrics are excellent, but final eval regressed.

## Testing Procedures

Command:

```powershell
myenv\Scripts\python -m pytest -q
```

Current result:

- `21 passed`

Test coverage highlights:

- graph construction and validation (`tests/test_graph_loader.py`)
- environment reset/step behavior and termination logic (`tests/test_env.py`)
- model shape and action-mask correctness (`tests/test_models.py`)
- CLI parsing and selection-mode conflict validation (`tests/test_cli_inputs.py`)
- nearest-node fallback path without scikit-learn (`tests/test_cli_inputs.py`)
- route plotting and sanitization (`tests/test_plot_graph.py`)
- end-to-end training smoke test (`tests/test_train_smoke.py`)
- acceptance test for `>=25%` improvement vs random (`tests/test_acceptance.py`)

## Key Findings

1. Local geometric features alone are insufficient for reliable multi-goal road routing.
2. Adding shortest-path goal-cost action features significantly improves policy reliability.
3. Expert-guided exploration is effective for breaking early policy-collapse loops.
4. Backtrack suppression improves trajectory quality during data collection.
5. Best-checkpoint metrics are a better operational KPI than final-epoch metrics for this setup.

## Limitations

- This is still an engineered-feature approach, not a graph neural network.
- Policy teacher uses one-step lookahead, not full planning.
- Results can vary by selected start/goal nodes and graph radius.
- `metrics.jsonl` is append-only; stale history can mix multiple experiments in one file.

## Reproducibility and Recommended Runs

### Environment setup

```powershell
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

### Train (interactive graph selection)

```powershell
myenv\Scripts\python -m scripts.train --config configs/default.yaml --goal-count 3 --select-on-graph
```

Outputs:

- `artifacts/run_default/graph_overview.png`
- `artifacts/run_default/final_route.png`
- `artifacts/run_default/metrics.jsonl`
- `artifacts/run_default/best.pt`, `latest.pt`, `final.pt`

### Evaluate best checkpoint

```powershell
myenv\Scripts\python -m scripts.evaluate --config configs/default.yaml --checkpoint artifacts/run_default/best.pt --goal-count 3 --select-on-graph --plot-path artifacts/eval_route.png
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Cause:

- using system Python instead of project venv.

Fix:

- run via `myenv\Scripts\python ...` or activate the venv first.

### GUI selection does not respond

Cause:

- headless terminal/session, or GUI backend unavailable.

Fix:

- run in a desktop Python session with matplotlib GUI backend, or pass explicit nodes/coords.

### Checkpoint load shape mismatch

Cause:

- policy action feature dimension changed from 5 to 7 after goal-cost feature upgrade.

Fix:

- retrain and evaluate with checkpoints produced by the current code.

## Project Layout

- `src/graph_loader.py`: OSMnx graph download and validation.
- `src/env/road_multi_goal_env.py`: multi-goal weighted-graph environment.
- `src/features/state_features.py`: state feature engineering.
- `src/features/edge_features.py`: base edge feature engineering.
- `src/models/value_net.py`: value network.
- `src/models/policy_net.py`: policy network.
- `src/training/value_trainer.py`: TD value training.
- `src/training/policy_imitation_trainer.py`: policy imitation training.
- `src/training/loop.py`: rollout, update, eval, checkpoint loop.
- `src/eval/evaluate.py`: deterministic/stochastic policy evaluation.
- `src/eval/baselines.py`: random baseline evaluation.
- `src/visualization/plot_graph.py`: graph plotting and interactive node selection.
- `scripts/train.py`: training CLI entrypoint.
- `scripts/evaluate.py`: evaluation CLI entrypoint.
- `tests/`: automated tests.
