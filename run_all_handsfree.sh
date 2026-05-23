#!/bin/bash

# Freshman: Final Hands-Free Experimental Suite
# Automates all non-interactive experiments with 5 seeds each.

VENV_PATH="/opt/pytorch-build/venv/bin/activate"
source $VENV_PATH

ENV="LunarLander-v3"
EXPERT_DATA="expert_demonstrations_LunarLander-v3.pkl"
SEEDS=(42 43 44 45 46)

# 1. Static Baselines (Offline-only)
for seed in "${SEEDS[@]}"
do
    echo ">>> Running Static Baselines (Seed $seed)"
    
    # Exp 1: Pure BC
    python main.py --env $ENV --algo cql --bc --num_rl_frames 0 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "baseline_bc" --seed $seed

    # Exp 2: AWBC (Independent Value function test)
    python main.py --env $ENV --algo cql --awbc --num_rl_frames 0 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "baseline_awbc" --seed $seed

    # Exp 3: Pure Offline CQL
    python main.py --env $ENV --algo cql --offline_rl --num_rl_frames 0 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "baseline_cql" --seed $seed

    # Exp 4: AWBC-CQL (Offline Hybrid)
    python main.py --env $ENV --algo cql --offline_rl --awbc --num_rl_frames 0 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "baseline_awcql" --seed $seed
done

# 2. Hands-Free Online RL (Mixed Exploration + Expert Data)
for seed in "${SEEDS[@]}"
do
    echo ">>> Running Hands-Free Online Mixed (Seed $seed)"

    # Exp 5: Online Exploration + Expert BC
    python main.py --env $ENV --algo cql --online_rl --bc --num_rl_frames 2000 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "online_bc_handsfree" --seed $seed

    # Exp 6: Online Exploration + Expert AWBC
    python main.py --env $ENV --algo cql --online_rl --awbc --num_rl_frames 2000 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "online_awbc_handsfree" --seed $seed

    # Exp 7: Online Exploration + Offline CQL
    python main.py --env $ENV --algo cql --online_rl --offline_rl --num_rl_frames 2000 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "online_offline_cql" --seed $seed

    # Exp 8: Full Hands-Free Mixed (Online RL + Offline RL + AWBC)
    python main.py --env $ENV --algo cql --online_rl --offline_rl --awbc --num_rl_frames 2000 \
        --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
        --experiment_name "online_offline_awbc_cql" --seed $seed
done

echo "=========================================================="
echo "Final hands-free experimental suite completed."
echo "Use 'python plot_results_aggregate.py' to generate paper graphs."
echo "=========================================================="
