# Aegis Incident Console: An Integrated Framework for Multi-Model Incident Detection, Emergency Escalation, and RL-Based Route Planning

## Abstract

This report presents **Aegis Incident Console**, an integrated operational framework that unifies three traditionally disjoint functions in emergency-response pipelines: (i) media-based anomaly detection, (ii) structured escalation support, and (iii) multi-goal route planning via reinforcement learning (RL). The system combines two inference models (violence classification and fire detection), converts model outputs into responder-ready emergency summaries, and supports interactive policy-guided route generation on a Singapore road graph. The central design thesis is that operational impact in incident systems depends not only on model accuracy but on **workflow continuity** from detection to action. Empirical evidence from stored RL artifacts indicates substantial gains after anti-loop and goal-cost feature improvements, with best-run improvements above random baseline by more than 2.3x. Local endpoint smoke tests executed on March 3, 2026 confirmed reliable operation of core APIs (`/api/infer/`, `/api/emergency/`, `/api/emergency/standalone/`, `/api/contact/`). The report documents methodology, implementation choices, evaluation outcomes, and limitations relevant to practical deployment.

## Keywords

Incident response, anomaly detection, reinforcement learning, route planning, operational AI, emergency escalation, decision support.

## 1. Introduction

Real-world emergency operations frequently suffer from system fragmentation: detection outputs are produced in one tool, escalation records in another, and navigation planning in yet another. This fragmentation increases response latency, cognitive switching cost, and risk of procedural inconsistency during high-pressure events. The present work addresses this gap through an integrated console that supports **end-to-end incident handling**.

The primary objective is not to introduce a single novel model architecture, but to establish a practical and reproducible framework that links machine inference to operational action. Specifically, this project seeks to:

1. Provide unified, single-request media inference for multiple anomaly signals.
2. Convert model outputs into structured emergency communication artifacts.
3. Support configurable multi-goal policy route planning through an interactive interface.
4. Enable reproducible testing through explicit setup and testbench assets.

## 2. System Overview and Contributions

The system is implemented as a modular stack with three operational layers.

First, an anomaly inference layer executes two models over uploaded media and emits a normalized response schema. Second, an emergency orchestration layer consumes inference output (or manual input) to prepare escalation summaries and call metadata. Third, an RL planner layer supports map-based goal selection and policy route generation over a travel-time-weighted road graph.

The principal contributions are:

1. **Workflow integration** across detection, escalation, and routing.
2. **Dual escalation design** (inference-driven and standalone fallback) for continuity under partial system availability.
3. **Interactive RL planning UX** with user-specified goal count and route visualization.
4. **Reproducibility packaging** via `README` and `testbench/` artifacts.

## 3. Methodology

## 3.1 Anomaly Inference Pipeline

The detection subsystem jointly evaluates violence and fire risk from a single input file (image or video).

For violence scoring, the Keras model input shape is inspected at runtime, and media are transformed into the required temporal tensor representation. Video inputs are processed via evenly sampled frame extraction; image inputs are temporally tiled to match sequence requirements.

For fire scoring, a YOLO detector is applied either to the input image or to a representative frame extracted from video input. Detection parsing then derives a consolidated confidence value for fire-related classes.

Both outputs are returned in a normalized JSON structure (`detected_anomaly`, `model_used`, `confidence`) to simplify downstream orchestration.

## 3.2 Emergency Escalation Design

Two escalation pathways are provided.

The first pathway is inference-driven: model outputs are ranked by confidence, transformed into a structured summary, and packaged with incident metadata (location, notes, timestamp, incident ID). The second pathway is standalone: operators can generate escalation summaries without waiting for inference completion. This design reduces single-point dependency on model execution and supports immediate call preparation.

## 3.3 RL Route Planning Methodology

The route planning subsystem is based on the `Policy_rl_agent` module and a directed Singapore road graph derived via OSMnx. The environment encodes a multi-goal objective where a policy must traverse weighted road edges (travel time) while minimizing loop behavior and maximizing goal completion.

The RL stack includes:

1. **ValueNet** for state-value estimation with TD(0).
2. **PolicyNet** trained by imitation from a value/goal-cost-informed teacher.

Loop suppression and policy stabilization are supported by:

1. shortest-path goal-cost action features,
2. expert-action curriculum during rollout collection, and
3. backtrack probability penalty.

In web operation, users click one start node and K goals on an interactive matplotlib graph; the policy is then evaluated under those constraints, and the resulting path is returned and visualized.

## 3.4 Integration and Reliability Strategy

Several engineering controls were introduced to improve operational robustness:

1. upload-size guardrails and explicit 413 JSON responses,
2. safer frontend response parsing for non-JSON failure cases,
3. route planner locking to avoid overlapping interactive sessions,
4. cached planner engine loading to reduce repeated initialization overhead.

## 4. Experimental Setup

## 4.1 Environment and Artifacts

Experiments and execution were conducted in a Python/Django environment with TensorFlow, PyTorch, OSMnx, and Ultralytics dependencies defined in `requirements.txt`. RL evaluation references artifact directories produced in `Policy_rl_agent/artifacts/`.

## 4.2 Evaluation Scope

Evaluation was conducted at two levels:

1. **Model/agent outcomes** from previously generated RL artifact metrics.
2. **System-level functionality** via endpoint smoke testing and Django integrity checks.

## 4.3 Verification Protocol

The following procedures were used:

1. Django configuration integrity check (`manage.py check`).
2. Local endpoint smoke test (`testbench/smoke_test_local.py`) covering inference, emergency, standalone emergency, and contact APIs.
3. Manual RL planner validation (interactive map selection and route plot generation).

## 5. Results

## 5.1 Functional Results

The integrated console successfully supports:

1. single-upload dual-model inference,
2. structured emergency summary generation from inference outputs,
3. standalone emergency reporting without model dependency,
4. interactive RL route planning with configurable goal count,
5. route visualization and machine-readable route serialization.

## 5.2 RL Performance Outcomes (Artifact Snapshot, March 3, 2026)

| Run | Best Return | Best Success | Best Coverage | Best Improvement vs Random |
|---|---:|---:|---:|---:|
| `run_tuned_quick` | -261.9610 | 0.0000 | 0.0333 | -1.5734 |
| `run_tuned_quick_v2` | -242.9511 | 0.0667 | 0.1333 | -1.1113 |
| `run_tuned_quick_v3` | -273.9973 | 0.0000 | 0.0333 | -1.3812 |
| `run_tuned_goalfix_quick` | 144.0795 | 0.9500 | 0.9500 | 2.2111 |
| `run_tuned_goalfix_quick2` | 166.7325 | 1.0000 | 1.0000 | 2.4015 |
| `run_default` | 226.5100 | 1.0000 | 1.0000 | 2.4974 |
| `run_milestone_check` | 173.7978 | 1.0000 | 1.0000 | 2.3486 |

These results indicate a marked transition from early loop-prone behavior to near-complete success after feature and curriculum refinements.

## 5.3 System Smoke Test Results (March 3, 2026)

The local smoke test completed successfully:

1. `/api/infer/` passed,
2. `/api/emergency/` passed,
3. `/api/emergency/standalone/` passed,
4. `/api/contact/` passed.

RL route planning is intentionally excluded from automated smoke due to mandatory interactive map input.

## 6. Observations

Three practical observations emerged during development and testing.

First, operational utility increased most when model outputs were immediately connected to escalation actions rather than presented as raw scores. Second, route-planning reliability was highly sensitive to loop-control mechanisms and goal-cost-aware features. Third, replacing manual coordinate entry with interactive map selection improved usability and reduced input ambiguity.

## 7. Key Findings

1. **Operational integration is the dominant innovation**: practical value derives from the continuity of detect -> escalate -> route.
2. **Goal-cost-informed policy features are decisive** for stable multi-goal navigation on road graphs.
3. **Fallback escalation pathways are necessary** for resilient incident operations.
4. **Evaluation readiness requires packaging**, and the `testbench/` structure materially improves reproducibility for external reviewers.

## 8. Limitations and Threats to Validity

The system has current limitations.

1. RL route planning depends on a GUI-capable session for matplotlib interaction.
2. Road-graph acquisition depends on network availability and OSM service accessibility.
3. The current implementation emphasizes demonstrability and reproducibility over production hardening.
4. RL performance can vary with selected start/goal configurations and graph region parameters.

## 9. Future Work

Priority extensions include:

1. browser-native interactive route selection (to remove local GUI dependency),
2. asynchronous task execution for long-running inference,
3. broader automated validation for route planning with deterministic non-GUI fixtures,
4. persistent incident and route history with audit-grade export.

## 10. Conclusion

Aegis Incident Console demonstrates that incident AI systems benefit most from integrated decision workflows rather than isolated model endpoints. By combining anomaly detection, emergency summary generation, and RL route planning in one coherent interface, the project establishes a practical template for operational AI in safety-critical contexts. The reported results and reproducibility artifacts indicate a system that is both functionally complete for demonstration and extensible for future deployment-oriented research.
