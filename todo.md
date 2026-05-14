# Project TODO: Remaining Implementation Gaps

Based on the `requirements.md` and a review of the current codebase, the following items are needed to fully meet the Project Goal.

## 1. Local Memory Management (Critical)
**Requirement R4.2**: Decouple curriculum training from the global memory buffer.
- [ ] **Task**: Implement explicit `local_buffer` management in `PPOAgent` and `CQLAgent` to ensure that when `local=True`, the agent *only* trains on the recent auxiliary-rewarded experience.
- [ ] **Potential Solution**: Ensure `store_local_transition` is fully implemented and that `rl_update(local=True)` drains/samples only from that local buffer before clearing it.
- [ ] **Refinement**: Currently `PPOAgent.rl_update` clears `self.buffer`. It should only clear `self.local_buffer` when `local=True`.

## 2. Interactive Wrapper State Capture
**Requirement R2.1**: Perfect Deterministic Replay.
- [ ] **Task**: The `InteractiveGymWrapper` currently relies on `process_events` to provide `branch_timeline`. Ensure that when a human overrides, the *original* RL actions from that point forward are correctly identified and moved to the `anti_example_buffer`.
- [ ] **Task**: Ensure `historical_actions` in the `CurriculumBuffer` always includes the full path from `reset(seed)` to `start_frame`.

## 3. LLM Router Integration (Future-Proofing)
**Requirement R3.1**: Natural Language Classification.
- [ ] **Task**: Replace `_mock_llm_classify` with a real LLM API call (e.g., OpenAI/Gemini) to handle unstructured human feedback.
- [ ] **Task**: Enhance the code generation prompt in `_create_reward_fn` to handle more complex LunarLander state logic safely.

## 4. Evaluation & Metrics
**Requirement R1.3**: Terminal Logic Verification.
- [ ] **Task**: Add unit tests or assertions in `PPOAgent._calculate_gae` to verify that GAE correctly bootstraps on `truncated` while stopping on `terminated`.
- [ ] **Task**: Update `MetricsLogger` to track the separate performance of the curriculum sub-policies to see if they are actually improving localized behavior.

## 5. Scripting & Usability
- [ ] **Task**: Update `main.py` argparse to allow setting `num_local_epochs` and `trajectory_length` for curriculum tasks via command line.
- [ ] **Task**: Create a unified `run_experiment.sh` that automates the creation of results directories and calls `plot_results.py` after completion.
