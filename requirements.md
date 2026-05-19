# Project Requirements: Freshman (HITL RL) - Ground Truth Edition

This document defines the absolute requirements for the Freshman project, derived from `project_goal.md`.

## 1. Core RL Loop (Stage 1)
- **R1.1**: Support PPO and CQL.
- **R1.2**: Explicit Terminal Logic: Bootstrap on `truncated`, zero on `terminated`.
- **R1.3**: Buffers: `global_rl_buffer` (offline replay for CQL, rollout for PPO).

## 2. Human Review & Correction (Stage 2)
- **R2.1**: Interactive Replay: Skim with arrows, branch with spacebar.
- **R2.2**: Anti-BC: Branching marks discarded future as "bad behavior" in `anti_example` buffer.
- **R2.3**: Accept/Reject: Decision-based trajectory commitment.
- **R2.4**: Resume Logic: Deterministic replay using historical actions to branch from any frame.
- **R2.5**: Context-Aware Annotation: Capture exact observation magnitudes and human-readable formatting.
- **R2.6**: BC: Human takes control with spacebar, those actions are behavior cloned to shortcut learning

## 3. Curriculum Learning (Stage 3)
- **R3.1**: Annotation Classification: `Goal`, `Heuristic`, `Generic`.
- **R3.2**: Goal Processing: Dynamic generation of auxiliary reward functions.
- **R3.3**: Reward Isolation: Auxiliary rewards MUST be kept in a local buffer; global buffer only receives environment rewards.
- **R3.4**: Targeted Replay: Localized training from a specific start state to overcome sparse rewards.

## 4. Semi-Supervised Learning (Stage 3)
- **R4.1**: Rule-of-Thumb Extraction: Heuristics pinpoint key features and thresholds.
- **R4.2**: Buffer Mining: Search `global_rl_buffer` and `example_buffer` for all states matching the rule.
- **R4.3**: Consistency Training: Add noise to unspecified features and ensure consistent action prediction (FixMatch style).

## 6. Noisy Human Trajectories (Stage 3)
- **R6.1**: Feature Isolation: Identify unimportant features from human comments.
- **R6.2**: Data Augmentation: Generate noisy variations of human demonstrations by perturbing unimportant features to improve generalization and focus.

## 5. Unified Update & Telemetry
- **R5.1**: Multi-Faceted Update: Incorporate curriculum, supervised, and SSL data sequentially for multiple epochs.
- **R5.2**: Telemetry Tracker: Record detailed human vs compute time, including `reviewing` and `annotating` time.
