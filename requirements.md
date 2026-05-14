# Project Requirements: Freshman (HITL RL)

This document outlines the concrete requirements for the Freshman project, grounded in the existing implementation as of May 2026.

## 1. Core Agent Architecture
**Goal:** Modular agents capable of Reinforcement Learning, Behavior Cloning, and Semi-Supervised Learning.

*   **R1.1: Multi-Algorithm Support**: Support for on-policy (PPO) and off-policy (CQL) algorithms.
    *   *Implementation*: `PPOAgent` and `CQLAgent` classes in `PPO.py` and `CQL.py`.
*   **R1.2: Standardized Agent API**: All agents must implement the `Agent` abstract base class.
    *   *Implementation*: `Agent` in `Agent.py` with methods `act`, `predict`, `store_transition`, `rl_update`, `supervised_update`, and `ssl_update`.
*   **R1.3: Terminal Logic Integrity**: Explicit handling of `terminated` vs. `truncated` states.
    *   *Requirement*: Bootstrapping on `truncated`, zeroing on `terminated`.
    *   *Implementation*: `PPOAgent._calculate_gae` and `CQLAgent.rl_update` (TD-target logic).

## 2. Interactive Data Pipeline
**Goal:** A human-in-the-loop interface for real-time control, review, and annotation.

*   **R2.1: Deterministic Replay**: Capability to perfectly recreate a trajectory using an initial seed and historical actions.
    *   *Implementation*: `InteractiveGymWrapper._restore_state` and `main.py` curriculum loop using `task['historical_actions']`.
*   **R2.2: Accept/Reject Branching**: Human can override agent actions and choose to accept the new trajectory or roll back.
    *   *Implementation*: `InteractiveGymWrapper._branch_timeline` and `_handle_decision`.
*   **R2.3: Data Segregation**: Separated buffers for different learning signals.
    *   *Implementation*: `buffers.py` containing `ReplayBuffer`, `LLMBuffer`, `CurriculumBuffer`, and `SemiSupervisedBuffer`.

## 3. LLM-Assisted Routing & Logic
**Goal:** Translate natural language into training signals (Rewards and Feature Masks).

*   **R3.1: Natural Language Classification**: Classify human notes into `GENERIC`, `GOAL`, or `HEURISTIC`.
    *   *Implementation*: `LLMRouter._mock_llm_classify` in `llm_router.py`.
*   **R3.2: Reward Function Generation**: Dynamic generation and execution of Python reward functions.
    *   *Implementation*: `LLMRouter._create_reward_fn` and `main.py` curriculum loop applying `task['reward_fn']`.
*   **R3.3: Feature Masking**: Isolation of specific state features for consistency training.
    *   *Implementation*: `LLMRouter` returning `feature_mask` for `HEURISTIC` types; used in `agent.ssl_update`.

## 4. Training & Memory Management
**Goal:** Balanced updates that prevent catastrophic forgetting or reward hacking.

*   **R4.1: Supervised Learning (BC/Anti-BC)**:
    *   *Implementation*: `agent.supervised_update` using `buffers['example']` (BC) and `buffers['anti_example']` (Anti-BC).
*   **R4.2: Isolated Curriculum Learning**: Auxiliary rewards must NOT pollute the global value function.
    *   *Requirement*: Auxiliary rewards go to a local buffer; global buffer maintains environment rewards.
    *   *Implementation*: `agent.store_local_transition` and `agent.rl_update(local=True)`.
*   **R4.3: Semi-Supervised Consistency**: Training on augmented observations with focused masks.
    *   *Implementation*: `agent.ssl_update` utilizing noise injection on non-masked features.
