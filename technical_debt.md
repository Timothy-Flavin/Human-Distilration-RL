# Technical Debt Audit: Freshman (HITL RL)

This document identifies areas of the codebase that have become monolithic, brittle, or difficult to maintain due to incremental feature addition.

## 1. `main.py`

### Monolithic `main()` function
The `main()` function has grown to encompass environment setup, buffer initialization, the primary iteration loop, and several complex sub-phases (Interactive Review, Routing/Verification, Curriculum, Unified Update).
- **Issue**: Deep nesting and high cyclomatic complexity make it difficult to unit test the orchestration logic.
- **Recommendation**: Refactor sub-phases into dedicated classes or functions (e.g., `UpdateOrchestrator`, `InteractionManager`).

### Nested Verification Loop in Step 3
The logic for handling human rephrasing during heuristic verification is deeply nested within the iteration loop.
- **Issue**: Brittle flow control; adding more interaction types (e.g., rephrasing a Goal) will make this unreadable.
- **Recommendation**: Move the verification state machine into `InteractiveGymWrapper` or a dedicated `VerificationManager`.

### Iteration Loop Bloat
The `for iteration in range(TOTAL_ITERATIONS)` loop contains too much low-level code (saving files, printing metrics, evaluation).
- **Issue**: It's hard to see the high-level "Stage 1 -> Stage 2 -> Stage 3" flow described in `project_goal.md`.

---

## 2. `llm_router.py`

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
- **Issue**: Bug fixes or improvements (like adding termination rule support) must be applied twice.
- **Recommendation**: Pull shared training logic into the `Agent` base class or a specialized `Trainer` utility.
