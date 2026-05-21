# Technical Debt Audit: Freshman (HITL RL)

This document identifies areas of the codebase that have become monolithic, brittle, or difficult to maintain due to incremental feature addition.

## 1. `main.py`

### Monolithic `main()` function
The `main()` function has been significantly simplified by moving the heuristic verification logic to `VerificationManager`.
- **Status**: Partially refactored. The unified update loop and curriculum orchestration still reside in `main()`.
- **Recommendation**: Continue decoupling sub-phases.

### Deeply Nested Interaction Logic
While cleaner, the verification manager still relies on importing `process_events` locally and manual state management.
- **Recommendation**: Standardize the event loop interface across all interactive components.

---

## 2. `verification_manager.py`

### Manual Event Consumption
`VerificationManager` currently manually consumes Pygame events before passing them to `process_events`.
- **Issue**: This is brittle and can lead to missed events if the order of checks changes.
- **Recommendation**: Refactor `process_events` to return raw key states or use a more robust Event Dispatcher.

---

## 3. `llm_router.py`
...

### Long `if/elif` chain in `_mock_llm_classify`
While heuristics have been moved to `LunarLander_v3_heuristics.py`, the `GOAL` definitions (reward function code strings) are still hardcoded in a long conditional block.
- **Issue**: Adding support for new environments will require modifying this core file.
- **Recommendation**: Move Goal templates to environment-specific configuration files or a dedicated library similar to `HEURISTICS`.

### Reward Function Evaluation (`_create_reward_fn`)
The use of `exec()` for dynamic reward functions is inherently brittle and poses a security risk.
- **Issue**: Debugging a malformed reward function string is difficult as it happens at runtime.
- **Recommendation**: Use a more structured approach for auxiliary rewards (e.g., a registry of parameterized reward components).

---

## 3. `wrapper.py`

### Mode Management
The `InteractiveGymWrapper` uses a string-based `self.mode` with logic scattered across `process_events` and `run()`.
- **Issue**: Adding new modes (like the recently added `verification`) requires updates in multiple files and methods, making it prone to state-desync bugs.
- **Recommendation**: Implement a formal State Pattern for UI modes.

---

## 5. Final Refactoring Plan: Loss Signal & Data Source Decoupling

To support the rigorous comparative analysis in `Usage.md`, the codebase must decouple **Loss Signals** (the mathematical objective) from **Data Sources** (where transitions come from).

### A. Modular Loss Signals (Agent class)
The standalone `ssl_update()` will be removed. Invariance training (feature noise) will be integrated directly into the core loss functions via a shared **`Agent.ssl_augment(batch, masks)`** utility.
- **`Agent.ssl_augment(batch, masks)`**: 
    - Handles complex noise logic: 
        - **Gaussian**: Default centered on $s_t$ with $\sigma=0.1$ (or user-specified).
        - **Uniform**: Samples from user-defined ranges (e.g., "above height" $\rightarrow [y_t, y_{max}]$).
    - Performs batch-wise augmentation using Torch for efficiency.
- **`Agent.update_td(batch, ssl=False, masks=None)`**: 
    - If `ssl=True`, calls `ssl_augment()` before calculating CQL/DQN loss.
- **`Agent.update_supervised(batch, ssl=False, masks=None, advantages=None)`**:
    - If `ssl=True`, calls `ssl_augment()` before calculating Cross-Entropy.
    - If `advantages` is provided, applies AWBC.


### B. Explicit Data Source Mapping
CLI arguments will map directly to specific buffer-source pairs during the `unified_train_step`:
- **`--online_rl`**: Samples from `agent.replay_buffer` (data collected via autonomous exploration) $\rightarrow$ `update_td()`.
- **`--offline_rl`**: Samples from `buffers['example']` (data from pre-load or human intervention) $\rightarrow$ `update_td()`.
- **`--bc`**: Samples from `buffers['example']` $\rightarrow$ `update_supervised(advantages=None)`.
- **`--awbc`**: Samples from `buffers['example']` $\rightarrow$ `update_supervised(advantages=calculated_adv)`.
- **`--ssl`**: When active, sets `ssl=True` in the above update calls for transitions that have an associated `feature_mask`.

### C. Orchestration (`main.py`)
`main()` will be refactored into a high-level manager:
1. **Acquisition**: `run_rl_collection` (if `--num_rl_frames > 0`).
2. **Human Interface**: `InteractiveGymWrapper` (if `--intervention` active and episodes collected).
3. **Training**: `unified_train_step` sampling from the correct buffers based on the flags above.
4. **Plotting**: Updated to support the 6 required graph types, including "Human Likeness" (entropy/divergence metrics).
