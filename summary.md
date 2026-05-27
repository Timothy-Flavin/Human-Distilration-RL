# Comprehensive Human Distillation: Summary Report

## Project Goal: Comprehensive Human Distillation
The "Comprehensive Human Distillation" project aims to discover the most effective methods for distilling human knowledge into autonomous agents through active learning. By leveraging a human-in-the-loop (HIL) pipeline, the project explores how human corrections, annotations, and heuristics can accelerate reinforcement learning (RL) and align agents with complex, often "invisible" human preferences. The goal is to move beyond simple behavior cloning to a multi-modal interactive system that combines static data, real-time interventions, and natural language feedback.

---

## Environments

### 1. Lunar Lander (v3)
A classic 2D flight control task. The agent must land a spacecraft on a designated pad.
- **State Space**: 8D vector including position, velocity, angle, angular velocity, and leg contact sensors.
- **Actions**: Discrete (Do nothing, fire left engine, fire main engine, fire right engine).
- **HIL Focus**: Precision paths, stable hovering, and recovering from unrecoverable spins.

### 2. Highway (v0)
A tactical driving environment where the agent navigates a multi-lane highway with other vehicles.
- **State Space**: $V \times F$ matrix (typically $5 \times 5$) representing the ego-vehicle and its 4 nearest neighbors with features like position and velocity.
- **Actions**: Lane changes (left/right) and speed control (faster/slower).
- **HIL Focus**: Enforcing safety constraints like following distance that are not captured by the default "maximize speed" reward function.

### 3. Crafter
A survival and crafting environment inspired by Minecraft, used for testing recurrent agents in more complex, long-horizon tasks.
- **State Space**: Pixel-based observations (processed via recurrent architectures).
- **Goal**: Survival and achievement unlocking (e.g., crafting tools, mining resources).
- **Metrics**: Success is measured by the geometric mean of achievement completion rates.

---

## Training Protocol & Methodology

The project follows a 4-stage interactive loop designed to refine the agent iteratively:

### Stage 1: Initial RL Collection
Agents (PPO or CQL) collect a baseline of environment trajectories. This provides a "starting point" for human review.

### Stage 2: Human Review & Coaching (State Machine)
A human coach reviews the agent's performance using an interactive playback tool:
- **Skimming**: Using left/right arrows to scrub through replay frames.
- **HG-Dagger Interventions**: Pressing the `Spacebar` to take manual control. This creates a branched timeline. The coach's actions are saved to an `example_buffer`, while the agent's original (overridden) actions are moved to an `anti_example_buffer`.
- **Annotation**: Pressing `Enter` to leave a natural language note at a specific frame. Notes are classified by an LLM into:
    - **Goals**: Subtasks with auxiliary rewards (e.g., "Halt movement").
    - **Heuristics**: Rule-of-thumb policies (e.g., "If spinning, fire side engine"). *Note: Direct heuristic controllers were tested but found to be less effective than other methods.*
    - **Feature Importance**: Identifying which features are relevant to a behavior (e.g., "Ignore altitude, focus on horizontal drift").

### Stage 3: Multi-Source Update
The model is updated sequentially or jointly using:
- **Behavior Cloning (BC/AWBC)**: Learning from human examples.
- **Conservative Q-Learning (CQL)**: Learning from both agent and human transitions.
- **Curriculum RL**: Learning from subtasks defined by human goals.
- **Semi-Supervised Learning (SSL)**: Applying noise masks to "unimportant" features identified by the human to improve generalization.

### Stage 4: Evaluation & Ablation
The pipeline is evaluated against 6 key metrics, including sample efficiency, human-likeness (entropy), and active human time vs. performance.

---

## Tracked Metrics & Telemetry
To justify the human-in-the-loop approach, the `MetricsLogger` tracks:
- **Compute Time**: Time spent in RL collection and agent updates.
- **Human Time**: Wall-clock time spent reviewing, overriding, and annotating.
- **Sample Efficiency**: Evaluation score vs. total environment interactions.
- **Human Likeness**: How closely the agent's policy matches the human's expert distribution.

---

## Data Structure of Saved Expert Episodes
Expert data is stored in `.pkl` files as a list of episodes. Each episode contains:
- **Transitions**: A list of dictionaries:
  ```python
  {
      'obs': [...],         # State vector
      'action': int,        # Action taken
      'reward': float,      # Reward received
      'next_obs': [...],    # Resulting state
      'terminated': bool,   # Episode end
      'truncated': bool,    # Time limit
      'info': {...},        # Metadata (source: human/agent, mask: feature importance)
  }
  ```
- **Metadata**: Episode duration and total reward.

---

## Summary of Experiments

The project conducts a series of 17 experiments to ablate the Freshman pipeline:
1.  **Static Baselines (1-6)**: Pure BC, AWBC, and CQL on fixed expert datasets with optional online RL collection.
2.  **Interactive Override Ablations (7-11)**: Testing the impact of HG-Dagger interventions (Correction Only) with and without "Hot Start" pre-loading of large expert datasets.
3.  **Annotation Pipeline Ablations (12-13)**: Comparing methods for incorporating LLM-generated curriculum rewards (Main, Separate, KL) and SSL feature noise.
4.  **Full "Freshman" Pipeline (14-15)**: Combining the best-performing components into a unified system.
5.  **HIL Necessity Demonstrations (16-17)**: Showing how HIL solves "invisible" constraints like safety buffers in Highway or precision descents in Lunar Lander.

---

## Visual Examples
*(Space reserved for screencaps of the interactive coaching interface and result graphs)*

[IMAGE: Interactive Playback and Branching Timeline]

[IMAGE: Training Performance Graphs - Sample Efficiency vs. Human Time]

[IMAGE: Feature Importance Heatmaps / SSL Noise Masks]
