# Roadmap: Generalizing Freshman for Multiple Environments

This document outlines the steps required to add support for a new environment, such as `highway-env`, while maintaining the interactive LLM-assisted pipeline.

## 1. Environment-Specific Assets (The "Content" Tier)
These files provide the context and "expert knowledge" for the LLM and the human interaction.
- [ ] **State Descriptor (`NewEnv-v0.md`)**: Create a markdown file defining qualitative thresholds for the new environment's observation vector (e.g., "fast" for a car vs "fast" for a lander).
- [ ] **Heuristics Library (`NewEnv_heuristics.py`)**: Define rule-of-thumb policies (trigger/action/termination) specific to the new task (e.g., `EMERGENCY_BRAKE`, `LANE_CHANGE`).
- [ ] **Usage Examples (`usage_examples_new.md`)**: Create baseline annotation examples for the new state space.

## 2. Interactive Components (The "UI/UX" Tier)
These files need to be generalized to handle different observation formats and control schemes.
- [ ] **`input_handler.py`**:
    - Refactor `get_realtime_action` to be environment-agnostic or take a mapping dictionary. Currently hardcoded for 4-action discrete spaces.
    - Need to support different keybindings for driving (e.g., `Up` for acceleration, `Left/Right` for steering).
- [ ] **`wrapper.py` (InteractiveGymWrapper)**:
    - Generalize `_format_obs`. Currently it has a hardcoded check for `len(obs) == 8` for LunarLander.
    - Implement an environment-specific `ObsFormatter` class that can be injected.
    - Ensure `env_state` capture (`get_state`/`set_state`) works for the new environment (Pharma's `highway-env` may require special handling for its kinematic state).

## 3. Core Logic & Routing (The "Backend" Tier)
- [ ] **`llm_router.py`**:
    - **`_mock_llm_classify`**: This contains a massive `if/elif` block hardcoded for LunarLander features (`x_pos`, `y_vel`, etc.).
    - **Action Plan**: Move these mocks into an `EnvironmentRegistry` or separate config files per environment.
    - **Feature Mapping**: The `NOISY_HUMAN` feature map (`feat_map`) is currently hardcoded for the 8-dim LunarLander vector. This must be dynamically loaded based on the environment name.
- [ ] **`LunarLander_v3_heuristics.py`**:
    - This logic is highly coupled with the 8-dim vector. A new environment needs its own library that exports a standardized `HEURISTICS` dictionary.

## 4. Main Integration (`main.py`)
- [ ] **Algorithm selection**: Ensure `obs_dim` and `action_dim` are correctly inferred (already largely handled).
- [ ] **Router Initialization**: Pass the environment-specific heuristics and feature maps to the `LLMRouter`.

## Implementation Priority for `highway-env`:
1. **Kinematic Formatter**: `highway-env` often uses a $(N, F)$ matrix for nearby vehicles. `InteractiveGymWrapper._format_obs` must be updated to handle multi-dimensional or flattened matrices.
2. **Control Mapping**: Update `input_handler.py` to support typical driving controls (Discrete actions: 0:LANE_LEFT, 1:IDLE, 2:LANE_RIGHT, 3:FASTER, 4:SLOWER).
3. **Registry Pattern**: Refactor `llm_router.py` to use a registry: `ROUTERS = {"LunarLander-v3": LunarLanderRouter, "highway-v0": HighwayRouter}`.
