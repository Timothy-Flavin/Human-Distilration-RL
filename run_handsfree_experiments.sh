#!/bin/bash

# Freshman: Hands-Free Experimental Suite
# This script runs all experiments that do not require human interaction.
# It iterates through 5 seeds each to provide statistical consistency.

# 1. Path to the virtual environment (adjust if necessary)
VENV_PATH="~/../../opt/pytorch-build/venv/bin/activate"
source $VENV_PATH

# 2. Configuration
ENV="LunarLander-v3"
EXPERT_DATA="expert_demonstrations_LunarLander-v3.pkl"
ITERATIONS=20
UNIFIED_EPOCHS_OFFLINE=400
UNIFIED_EPOCHS_ONLINE=400
RL_FRAMES=2000

echo "=========================================================="
echo "Freshman: Starting Hands-Free Experimental Suite"
echo "Environment: $ENV"
echo "Iterations:  $ITERATIONS"
echo "Seeds:       1 to 5"
echo "=========================================================="

for seed in {1..5}
do
    echo ""
    echo ">>> STARTING SEED: $seed"
    echo ""

    # # --- EXPERIMENT 1: Pure Behavior Cloning (Offline) ---
    # # Only BC on static expert data. No RL, no collection.
    # echo "[Exp 1] Pure BC (Seed $seed)"
    # python main.py --env $ENV --algo cql --bc --num_rl_frames 0 \
    #     --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
    #     --preload_expert_data $EXPERT_DATA \
    #     --experiment_name "baseline_bc" --seed $seed

    # # --- EXPERIMENT 2: Pure Offline CQL (Offline RL) ---
    # # Only CQL updates on static expert transitions. No BC, no collection.
    # echo "[Exp 2] Pure Offline CQL (Seed $seed)"
    # python main.py --env $ENV --algo cql --rl --num_rl_frames 0 \
    #     --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
    #     --preload_expert_data $EXPERT_DATA \
    #     --experiment_name "baseline_cql" --seed $seed

    # # --- EXPERIMENT 3: Advantage-Weighted CQL (Offline) ---
    # # Hybrid BC + CQL on static expert data. No collection.
    # echo "[Exp 3] AW-CQL (Seed $seed)"
    # python main.py --env $ENV --algo cql --bc --rl --awbc --num_rl_frames 0 \
    #     --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
    #     --preload_expert_data $EXPERT_DATA \
    #     --experiment_name "baseline_awcql" --seed $seed

    # --- EXPERIMENT 4: Hands-Free Online RL + BC ---
    # The model collects new RL data but still clones the static expert.
    # NO human interventions/reviews.
    echo "[Exp 4] Hands-Free Online RL + Expert BC (Seed $seed)"
    python main.py --env $ENV --algo cql --bc --rl --awbc \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $UNIFIED_EPOCHS_ONLINE \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "online_awbc_handsfree" --seed $seed

done

echo ""
echo "=========================================================="
echo "All hands-free experiments completed successfully."
echo "Results saved in ./results/$ENV/"
echo "Use 'python plot_results_aggregate.py' to visualize."
echo "=========================================================="


