#!/bin/bash

# Freshman: Recurrent Hands-Free Experimental Suite (Crafter)
# This script runs all experiments for the Crafter environment using the RCQL agent.
# It includes the drop_bottom 0.1 cleaning step as a pre-process.

# 1. Path to the local virtual environment
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

# 2. Configuration
ENV="crafter"
RAW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"

# BC is supervised (no bootstrapping), so high data reuse is safe: fewer, larger
# training rounds. Online TD is bootstrapped, so keep the replay ratio low:
# 3 epochs x 64 seqs x 48 steps ~= 9.2k samples per 2k collected frames (~4.6x).
ITERATIONS_BC=50
EPOCHS_BC=100
ITERATIONS_ONLINE=500
EPOCHS_ONLINE=3
RL_FRAMES=2000

echo "=========================================================="
echo "Freshman: Starting Recurrent Hands-Free Suite (Crafter)"
echo "Applying drop_bottom 0.1 to expert data..."
python3 analyze_expert_data.py --path $RAW_EXPERT_DATA --drop_bottom 0.1
echo "=========================================================="

# --- EXPERIMENT 1: Pure Behavior Cloning (Offline, stored-state burn-in) ---
# for seed in {99..100}
# do
#     echo ""
#     echo "[Exp 1] Recurrent BC (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --bc --num_rl_frames 0 \
#         --num_unified_epochs $EPOCHS_BC \
#         --total_iterations $ITERATIONS_BC \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "baseline_bc" --seed $seed
# done

# # --- EXPERIMENT 5: Online DQN without BC (double DQN, stored-state burn-in) ---
# for seed in {99..100}
# do
#     echo ""
#     echo "[Exp 5] Online Recurrent DQN, no BC (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --online_rl \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data "" \
#         --experiment_name "online_dqn" --seed $seed
# done

# --- EXPERIMENT 5: Online DQN + awbc ---
for seed in {99..100}
do
    echo ""
    echo "[Exp 5] Online Recurrent DQN, no BC (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --awbc \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS_ONLINE \
        --total_iterations $ITERATIONS_ONLINE \
        --num_envs 8 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "online_awbc" --seed $seed
done


# # --- EXPERIMENT 2: Pure Offline RCQL (Offline RL) ---
# python3 recurrent_main.py --env $ENV --offline_rl --num_rl_frames 0 \
#     --num_unified_epochs $EPOCHS_BC --total_iterations $ITERATIONS_BC \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "baseline_rcql" --seed $seed

# # --- EXPERIMENT 3: Advantage-Weighted RCQL (Offline) ---
# python3 recurrent_main.py --env $ENV --bc --offline_rl --awbc --num_rl_frames 0 \
#     --num_unified_epochs $EPOCHS_BC --total_iterations $ITERATIONS_BC \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "baseline_awrcql" --seed $seed

# # --- EXPERIMENT 4: Hands-Free Online RL + BC ---
# python3 recurrent_main.py --env $ENV --online_rl --offline_rl --awbc \
#     --num_rl_frames $RL_FRAMES \
#     --num_unified_epochs $EPOCHS_ONLINE --total_iterations $ITERATIONS_ONLINE \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "online_offline_awbc" --seed $seed

echo ""
echo "=========================================================="
echo "All recurrent hands-free experiments completed successfully."
echo "Results saved in ./results/crafter/"
echo "=========================================================="
