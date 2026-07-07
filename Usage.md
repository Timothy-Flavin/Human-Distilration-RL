# Freshman: Experimental Suite Description

This document contains the experiments needed to validate the Freshman HIL pipeline. For each experiment, up to 6 graphs will be produced. 
### 1: Total env interactions X axis vs Eval Score Y axis (measure pipeline sample efficiency)
### 2: Total wall clock time X axis vs Eval Score Y axis (measure real-time to deliver results)
### 3: Human likeness (for methods that include human data cross measure entropy at the end of each iteration)
### 4: Active Human Time X axis vs Eval Score Y axis (not used for hands-free methods, measures human effort efficiency)
### 5: Bar-graph of frames in each category
### 6: Bar-graph of time in each category (including static expert dataset)

The following experiments will be used to identify the tradeoffs for each strategy. All human data goes into the human expert buffer. Agent collected data may go in the global rl buffer or individual curriculum rl buffers. Human data that got annotated for noisy state augmentation goes into the ssl buffer with saved noise masks. 

## 1. Static Baselines (Hands-Free Offline Learning)
These experiments use the `expert_demonstrations_LunarLander-v3.pkl` dataset to compare supervised learning against offline reinforcement learning.

### Exp 1: Pure Behavior Cloning (BC)
*   **Description**: Trains a policy using only the expert (obs, action) pairs with Cross-Entropy loss.
*   **Command**:
    ```bash python main.py --env LunarLander-v3 --algo cql --bc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_bc" --seed 42
    ```

### Exp 2: Advantage Weighted Behavior Cloning (AWBC)
*   **Description**: Trains a policy using only the expert (obs, action) pairs with Advantage weighted Cross-Entropy loss.
*   **Command**:
    ```bash python main.py --env LunarLander-v3 --algo cql --awbc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_bc" --seed 42
    ```

### Exp 3: Pure Offline CQL
*   **Description**: Trains using the Conservative Q-Learning objective on the expert transitions. The arg `--offline_rl` means CQL will update using offline RL loss from the human example buffer
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_cql" --seed 42
    ```

### Exp 4: Advantage-Weighted BC + CQL (AWBC-CQL)
*   **Description**: Combines AWBC and CQL, weighting the supervised loss by the estimated advantage.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --awbc --num_rl_frames 0 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_awcql" --seed 42
    ```

### Exp 5: Online RL (no conservative loss) + Offline CQL
*   **Description**: Learns from CQL loss on both the offline expert data and the online collected rl data
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --online_rl --num_rl_frames 2000 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_awcql" --seed 42
    ```

### Exp 6: Online RL (no conservative loss) + Offline CQL + AWBC
*   **Description**: Learns from CQL loss on both the offline expert data and the online collected rl data and advantage weighted cross entropy loss from the human expert data
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --awbc --offline_rl --online_rl --num_rl_frames 2000 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "baseline_awcql" --seed 42
    ```

---

## 2. Interactive Override Pipeline Ablations
These experiments measure the impact of each form of override loss component on sample efficiency and human effort. 

### Exp 7: Interactive AWBC (Correction Only)
*   **Description**: Only uses human overrides (Spacebar) to push data to the `example_buffer`. No LLM involvement.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --awbc --online_rl --intervention --num_rl_frames 2000 --experiment_name "interactive_bc" --seed 42 --num_unified_epochs 200 
    ```
### Exp 8: Interactive OfflineRL (Correction Only)
*   **Description**: Only uses human overrides (Spacebar) to push data to the `example_buffer`. No LLM involvement.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --online_rl --intervention --num_rl_frames 2000 --experiment_name "interactive_rl" --seed 42 --num_unified_epochs 200 
    ```
  
### Exp 9: Hot Start + Interactive AWBC (Expert dataset + online corrections)
*   **Description**: Combines pre-loaded expert data with live interactive overrides.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --awbc --online_rl --intervention --num_rl_frames 2000 --experiment_name "hotstart_bc" --seed 42 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --num_unified_epochs 200 
    ```

### Exp 10: Hot Start + Interactive OfflineRL (Expert dataset + online corrections)
*   **Description**: Combines pre-loaded expert data with live interactive overrides, learning via Offline RL.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --online_rl --intervention --num_rl_frames 2000 --experiment_name "hotstart_rl" --seed 42 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --num_unified_epochs 200 
    ```
  
### Exp 11: Hot Start + Interactive OfflineRL + AWBC (Expert dataset + online corrections)
*   **Description**: Full override-based learning using both Supervised and TD signals.
*   **Command**:
    ```bash
    python main.py --env LunarLander-v3 --algo cql --offline_rl --online_rl --awbc --intervention --num_rl_frames 2000 --experiment_name "hotstart_combined" --seed 42 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --num_unified_epochs 200 
    ```

---

## 3. Interactive Annotation Pipeline Ablations

### Exp 12: LLM Curriculum Method Comparison
*   **Description**: Compares the three different ways to incorporate auxiliary LLM rewards.
*   **Commands**:
    ```bash
    # Method 1: Main (Direct update to primary Q-net)
    python main.py --env LunarLander-v3 --online_rl --intervention --curriculum --curriculum_method main --num_rl_frames 2000 --experiment_name "curriculum_main" --num_unified_epochs 200 

    # Method 2: Separate
    python main.py --env LunarLander-v3 --online_rl --intervention --curriculum --curriculum_method separate --num_rl_frames 2000 --experiment_name "curriculum_separate" --num_unified_epochs 200 

    # Method 3: KL
    python main.py --env LunarLander-v3 --online_rl --intervention --curriculum --curriculum_method kl --num_rl_frames 2000 --experiment_name "curriculum_kl" --num_unified_epochs 200 
    ```

### Exp 13: LLM SSL User gives example behavior with feature noise constraints
*   **Description**: Compares the three different ways to incorporate auxiliary LLM rewards. Run all three which will share the same plot to find out which curriculum method is most effective.
*   **Commands**:
    ```bash
    # Method 1: Main (Noise added to human trajectories that have a noise map during offline-rl update)
    python main.py --env LunarLander-v3 --online_rl --offline_rl --ssl --num_rl_frames 2000 --experiment_name "curriculum_main" --num_unified_epochs 200  
    # Method 1: Main (Noise added to human trajectories that have a noise map during awbc update)
    python main.py --env LunarLander-v3 --online_rl --awbc --ssl --num_rl_frames 2000 --experiment_name "curriculum_main" --num_unified_epochs 200 
    ```
---

## 3. The "Greedy Best" Pipeline (Exact arguments TBD by experimentation)
After selecting the best Curriculum method (e.g., KL) and verifying the Noisy Trajectory method, combine them into the final proposed system.

### Exp 14: Full "Freshman" Pipeline (LunarLander)
*   **Command**:
    ```bash
    # Commands TBD from earlier experiment results
    # Choose best SSL, Annotation, Intervention, and Hands-Free combination for full pipeline
    python main.py --algo cql --online_rl --offline_rl --awbc --intervention --curriculum --num_rl_frames 2000 --num_unified_epochs 500 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "full" --seed 1
    ```

### Exp 15: Full "Freshman" Pipeline (Highway)
*   **Command**:
    ```bash
    # Commands TBD from earlier experiment results
    # RL Only
    # Best hands-Free pipeline
    # Choose best SSL, Annotation, Intervention, and Hands-Free combination for full pipeline
    ```

---

## 4. Demonstrating HIL Necessity: Hidden Desired Behaviors
These experiments prove that HIL is required for policies that have strict but "invisible" constraints not captured by the environment's base reward function.

### Exp 16: Highway "Safety First" (Following Distance)
*   **Goal**: Standard RL will tailgate to maintain high speed. HIL must enforce a 2-second following distance.
*   **Interaction**: During review, take control when the agent is tailgating. Leave a note: `"ignore x_pos, maintain safe following distance below 0.3 speed difference"`.
*   **Command**:
    ```bash
    # RL Only
    # Best hands-Free pipeline
    # Choose best SSL, Annotation, Intervention, and Hands-Free combination for full pipeline
    ```

### Exp 17: Lander "Precision Path" (Hover-then-Descend)
*   **Goal**: Environment rewards any landing, but the "Expert" requires a specific vertical descent from the center-line only.
*   **Interaction**: Use `NOISY_HUMAN` to demonstrate the vertical drop from a hover, ignoring altitude variations.
*   **Command**:
    ```bash
    # Commands TBD from earlier experiment results
    ```

---

## 5. Analysis & Evaluation

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
    Results are saved as: `results/{env}/{experiment_name}/{algo}_{online_rl}_{offline_rl}_{bc}_{awbc}_{ssl}_seed{X}/`
