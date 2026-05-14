# Project TODO: Ground Truth Alignment

The following gaps remain to achieve 100% alignment with `project_goal.md`.

## 1. Metrics & Telemetry (R5.2)
- [ ] **Task**: Add `time_human_reviewing` to `MetricsLogger`.
- [ ] **Task**: Update `InteractiveGymWrapper` to start/stop the `human_reviewing` timer when arrows are pressed vs idle.
- [ ] **Task**: Add `frames_generated_ssl` and `frames_generated_curriculum` to frame counters.

## 2. SSL Buffer Mining (R4.2)
- [ ] **Task**: Update `llm_router.py` to provide a `rule_lambda` or similar threshold-based check for heuristics.
- [ ] **Task**: Modify `agent.ssl_update` or `main.py` to search the existing buffers for all transitions matching the heuristic rule, rather than just using the single annotated frame.

## 3. Unified Update Epochs (R5.1)
- [ ] **Task**: Refactor the update section in `main.py` to perform a unified update loop where all buffers are sampled and updated over $N$ epochs, ensuring consistent convergence.

## 4. LunarLander Formatter Enhancement (R2.5)
- [ ] **Task**: Ensure the `InteractiveGymWrapper._format_obs` provides a full, pretty-printed JSON-like context for the LLM as requested in the ground truth.
