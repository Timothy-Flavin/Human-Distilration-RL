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

## 4. Agents (`CQL.py`, `PPO.py`)

### Duplicated SSL/BC Logic
`CQLAgent` and `PPOAgent` have very similar `ssl_update` and `supervised_update` methods, with only the network call changing.
- **Status**: Partially improved. Shared logic still exists but has been stabilized.
- **Recommendation**: Pull shared training logic into the `Agent` base class or a specialized `Trainer` utility.

### Previous Issue: Destructive SSL Noise
The previous hardcoded `noise_scale=0.5` was too aggressive for normalized state features, leading to performance destruction.
- **Fix**: Reduced to `0.05`.

### Previous Issue: Per-Item Optimizer Steps
`ssl_update` was previously calling `optimizer.step()` for every item in the batch.
- **Fix**: Refactored to accumulate loss across the entire batch for stable gradient updates.
