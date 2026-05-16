# Data Artifacts Documentation: Freshman (HITL RL)

This document describes the structure, purpose, and lifecycle of the various buffer and metrics files generated during training and interaction.

## 1. Buffers (`.pt` files)

The buffers are serialized using `torch.save` and contain experience collected from various sources.

### `example_buffer_{iteration}.pt`
- **Purpose**: Stores positive demonstrations provided by the human coach during interactive overrides.
- **Content**: A list of `(obs, action)` tuples.
- **Usage**: Used for Behavior Cloning (BC) or Advantage Weighted Behavior Cloning (AWBC) in the unified update pipeline.
- **Save/Load**: Saved in `main.py` using `buffers['example'].save()`. Loaded via `ReplayBuffer.load()` in `buffers.py`.

### `anti_example_buffer_{iteration}.pt`
- **Purpose**: Stores "bad" behavior trajectories—specifically the future frames that were discarded when a human chose to branch the timeline.
- **Content**: A list of `(obs, action)` tuples.
- **Usage**: Used for Anti-Behavior Cloning (Anti-BC) to minimize the probability of the agent taking actions that led to undesirable states.
- **Save/Load**: Saved in `main.py` using `buffers['anti_example'].save()`. Loaded via `ReplayBuffer.load()` in `buffers.py`.

### `annotations_{iteration}.json`
- **Purpose**: Records the raw human feedback for each iteration to ensure reproducability and traceability of the "Golden Path".
- **Content**: A list of dictionaries containing the frame index, the raw text note, and the pretty-printed observation context at the time of annotation.
- **Usage**: Diagnostic tool for understanding why specific curriculum tasks or SSL mined datasets were created.
- **Save/Load**: Saved in `main.py` at the end of the interactive review phase.

### Agent Checkpoints (e.g., `CQL_agent_update_{iteration}.pt`)
- **Purpose**: Snapshots of the neural network weights at specific points in the training iteration.
- **Variants**:
    - `rl_collection`: After standard RL experience gathering.
    - `interactive_review`: After human interaction and timeline branching.
    - `agent_update`: After the multi-faceted unified update loop.
- **Content**: State dictionaries (`state_dict`) of the Actor/Critic or Q-networks.
- **Save/Load**: Handled by `Agent.checkpoint_model()` and `Agent.load_model()`.

---

## 2. Telemetry & Metrics (`.json` files)

Metrics are stored as human-readable JSON files, providing a detailed breakdown of time and performance.

### `metrics_{iteration}.json` and `metrics_latest.json`
- **Structure**:
    - `timers`: Wall-clock time (seconds) spent on various tasks:
        - `rl_experience`: Standard RL collection.
        - `human_overriding`: Time spent by the human taking control.
        - `human_reviewing`: Time spent skimming through replays.
        - `human_annotating`: Time spent writing notes.
        - `llm_processing`: Time for the LLM Router to classify notes.
        - `agent_updating_...`: Time spent on specific training phases (BC, SSL, Curriculum).
    - `frames`: Cumulative frame counters for different experience sources:
        - `rl`: Standard agent-generated frames.
        - `human`: Human-controlled frames.
        - `curriculum`: Frames generated during localized curriculum learning.
        - `ssl`: Frames processed via semi-supervised mining.
    - `evaluations`: A list of dictionaries tracking performance over time:
        - `iteration`: The iteration index.
        - `return_mean` / `return_std`: Performance in the environment.
        - `bc_loss` / `anti_bc_loss`: Loss values for the supervised components.

- **Lifecycle**: Managed by the `MetricsLogger` class in `metrics.py`. Saved at the end of every iteration in `main.py`.

---

## 3. Buffer Mining & Routing (Internal)

While not always saved to disk, these buffers are critical to the pipeline:
- **LLM Buffer**: A transient queue (`LLMBuffer`) for human notes waiting to be processed by the `LLMRouter`.
- **Curriculum Buffer**: Stores tasks (`seed`, `reward_fn`, `historical_actions`) for targeted curriculum training.
- **SSL Buffer**: Stores mined transitions from the global and example buffers that match human heuristics for consistency training.
- **KL Target Buffer**: Specifically used in the `kl` curriculum strategy to store observations for targeted policy regularization.
