#!/bin/bash

# Freshman: Recurrent Hands-Free Experimental Suite (Crafter)
# This script runs all experiments for the Crafter environment using the RCQL agent.
# It includes the drop_bottom 0.1 cleaning step as a pre-process.

# 1. Path to the local virtual environment
VENV_PATH="./venv/bin/activate"
source $VENV_PATH

# 2. Configuration
ENV="crafter"
RAW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"
ITERATIONS=50
UNIFIED_EPOCHS_OFFLINE=100
UNIFIED_EPOCHS_ONLINE=100
RL_FRAMES=2000

echo "=========================================================="
echo "Freshman: Starting Recurrent Hands-Free Suite (Crafter)"
echo "Applying drop_bottom 0.1 to expert data..."
python3 analyze_expert_data.py --path $RAW_EXPERT_DATA --drop_bottom 0.1
echo "=========================================================="

for seed in {1..2}
do
    echo ""
    echo ">>> STARTING SEED: $seed"
    echo ""

    
    # #--- EXPERIMENT 1: Pure Behavior Cloning (Offline) ---
    # echo "[Exp 1] Recurrent BC (Seed $seed)"
    # python3 recurrent_main.py --env $ENV --bc --num_rl_frames 0 \
    #     --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
    #     --preload_expert_data $CLEANED_EXPERT_DATA \
    #     --experiment_name "baseline_bc" --seed $seed

    # --- EXPERIMENT 2: Pure Offline RCQL (Offline RL) ---
    echo "[Exp 2] Pure Offline RCQL (Seed $seed)"
    python3 recurrent_main.py --env $ENV --offline_rl --num_rl_frames 0 \
        --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "baseline_rcql" --seed $seed

    # # --- EXPERIMENT 3: Advantage-Weighted RCQL (Offline) ---
    # echo "[Exp 3] AW-RCQL (Seed $seed)"
    # python3 recurrent_main.py --env $ENV --bc --offline_rl --awbc --num_rl_frames 0 \
    #     --num_unified_epochs $UNIFIED_EPOCHS_OFFLINE \
    #     --preload_expert_data $CLEANED_EXPERT_DATA \
    #     --experiment_name "baseline_awrcql" --seed $seed
    # # --- EXPERIMENT 4: Hands-Free Online RL + BC ---
    # echo "[Exp 4] Hands-Free Online RL + Expert BC (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --offline_rl --awbc\
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $UNIFIED_EPOCHS_ONLINE \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "online_offline_awbc" --seed $seed
done
echo ""
echo "=========================================================="
echo "All recurrent hands-free experiments completed successfully."
echo "Results saved in ./results/crafter/"
echo "=========================================================="
