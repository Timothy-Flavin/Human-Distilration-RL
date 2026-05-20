# Freshman: Experimental Suite & Reproducibility Guide

This document contains the exact commands and configurations used to generate the results for the Freshman paper.

## 1. Static Baselines (Offline Learning)
These experiments use the `expert_demonstrations_LunarLander-v3.pkl` dataset to compare supervised learning against offline reinforcement learning.

### Exp 1: Pure Behavior Cloning (BC)
*   **Description**: Trains a policy using only the expert (obs, action) pairs with Cross-Entropy loss.
*   **Command**:
    ```bash
    # Run 3 seeds to show consistency
    python main.py --env LunarLander-v3 --algo cql --bc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_bc" --seed 42
    python main.py --env LunarLander-v3 --algo cql --bc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_bc" --seed 43
    python main.py --env LunarLander-v3 --algo cql --bc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_bc" --seed 44
    ```

### Exp 2: Pure Offline CQL
*   **Description**: Trains using the Conservative Q-Learning objective on the expert transitions.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --rl --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_cql" --seed 42
    ```

### Exp 3: Advantage-Weighted CQL (AW-CQL)
*   **Description**: Combines BC and CQL, weighting the supervised loss by the estimated advantage.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --bc --rl --awbc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_awcql" --seed 42
    ```

---

## 2. Interactive Pipeline Ablations
These experiments measure the impact of each interactive component on sample efficiency and human effort.

### Exp 4: Interactive BC (Correction Only)
*   **Description**: Only uses human overrides (Spacebar) to push data to the `example_buffer`. No LLM involvement.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --bc --rl --num_rl_frames 2000 --experiment_name "interactive_bc" --seed 42
    ```

### Exp 5: BC + Anti-BC (Penalty for Failure)
*   **Description**: Adds the Anti-BC penalty for trajectories the human discarded during overrides.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --bc --anti_bc --rl --num_rl_frames 2000 --experiment_name "interactive_bc_anti" --seed 42
    ```

### Exp 6: Heuristic Heuristics (Legacy SSL - Negative Result)
*   **Description**: Tests the "Expert System" style heuristic controllers. This is the baseline negative result.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --rl --ssl --num_rl_frames 2000 --experiment_name "legacy_heuristics" --seed 42
    ```

### Exp 7: LLM Curriculum Method Comparison
*   **Description**: Compares the three different ways to incorporate auxiliary LLM rewards. Run all three to select the "Greedy Best" for the final pipeline.
*   **Commands**:
    ```bash
    # Method 1: Main (Direct update to primary Q-net)
    python main.py --env LunarLander-v3 --rl --curriculum --curriculum_method main --num_rl_frames 2000 --experiment_name "curriculum_main"

    # Method 2: Separate (Train aux agent, then main learns from its transitions)
    python main.py --env LunarLander-v3 --rl --curriculum --curriculum_method separate --num_rl_frames 2000 --experiment_name "curriculum_separate"

    # Method 3: KL (Train aux agent, then main agent pulled via KL-Divergence)
    python main.py --env LunarLander-v3 --rl --curriculum --curriculum_method kl --num_rl_frames 2000 --experiment_name "curriculum_kl"
    ```

---

## 3. The "Greedy Best" Pipeline
After selecting the best Curriculum method (e.g., KL) and verifying the Noisy Trajectory method, combine them into the final proposed system.

### Exp 9: Full "Freshman" Pipeline (LunarLander)
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --bc --rl --awbc --ssl --curriculum --curriculum_method kl --num_noisy_samples 10 --experiment_name "freshman_final_lander"
    ```

### Exp 10: Full "Freshman" Pipeline (Highway)
*   **Command**:
    ```bash
    python main.py --env highway-v0 --algo cql --bc --rl --awbc --ssl --curriculum --curriculum_method kl --num_noisy_samples 10 --experiment_name "freshman_final_highway"
    ```

---

## 4. Demonstrating HIL Necessity: Hidden Desired Behaviors
These experiments prove that HIL is required for policies that have strict but "invisible" constraints not captured by the environment's base reward function.

### Exp 11: Highway "Safety First" (Following Distance)
*   **Goal**: Standard RL will tailgate to maintain high speed. HIL must enforce a 2-second following distance.
*   **Interaction**: During review, take control when the agent is tailgating. Leave a note: `"ignore x_pos, maintain safe following distance below 0.3 speed difference"`.
*   **Command**:
    ```bash
    python main.py --env highway-v0 --algo cql --rl --bc --ssl --experiment_name "hidden_safety"
    ```

### Exp 12: Lander "Precision Path" (Hover-then-Descend)
*   **Goal**: Environment rewards any landing, but the "Expert" requires a specific vertical descent from the center-line only.
*   **Interaction**: Use `NOISY_HUMAN` to demonstrate the vertical drop from a hover, ignoring altitude variations.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --rl --bc --ssl --num_noisy_samples 15 --experiment_name "hidden_precision_path"
    ```

---

## 5. Post-hoc Compliance Evaluation
These scripts measure how well an agent (or a human dataset) adheres to "hidden" behavioral requirements.

### Eval 1: Analyze Expert Dataset Compliance
*   **Description**: Calculates the Arbitrary Behavior Score for a recorded `.pkl` dataset.
*   **Command**:
    ```bash
    python analyze_expert_data.py --env LunarLander-v3
    ```

### Eval 2: Evaluate Model Checkpoint Compliance
*   **Description**: Loads a trained model and runs evaluation episodes to measure its compliance score.
*   **Command**:
    ```bash
    python eval_compliance.py --env LunarLander-v3 --algo cql --model_path "results/LunarLander-v3/baseline_cql/cql_rl1_bc0_anti0_ssl0_cur0_seed42/CQL_latest.pt"
    ```

## 6. Project Directories & Metrics

After running the experiments, use the following to generate statistics and aggregated plots:

1.  **Aggregated Visualization**:
    This script automatically finds all seeds under an experiment name and plots the mean (bold) with individual runs (transparent).
    ```bash
    python plot_results_aggregate.py --env LunarLander-v3 --experiment_name "baseline_bc"
    ```
2.  **Dataset Stats**:
    ```bash
    python analyze_expert_data.py --env LunarLander-v3
    ```
3.  **Directory Structure**:
    Results are saved as: `results/{env}/{experiment_name}/{algo}_rl_bc_anti_ssl_cur_seed{X}/`
