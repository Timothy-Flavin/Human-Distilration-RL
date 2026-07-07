#!/bin/bash

# Freshman: Final Hands-Free Experimental Suite
# Automates all non-interactive experiments with 5 seeds each for both environments.

VENV_PATH="/opt/pytorch-build/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

SEEDS=(42 43 44 45 46)
ENVS=("LunarLander-v3" "highway-v0")
DATASETS=("expert_demonstrations_LunarLander-v3.pkl" "expert_demonstrations_highway-v0.pkl")

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    EXPERT_DATA="${DATASETS[$i]}"
    
    echo "=========================================================="
    echo "Running experiments for $ENV"
    echo "=========================================================="

    for seed in "${SEEDS[@]}"
    do
        echo ">>> Running Static Baselines (Seed $seed) for $ENV"
        
        # Exp 1: Pure BC
        python main.py --env $ENV --algo cql --bc --num_rl_frames 0 \
            --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
            --experiment_name "baseline_bc" --seed $seed

        # Exp 2: AWBC
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

        echo ">>> Running Hands-Free Online Mixed (Seed $seed) for $ENV"

        # Exp 5: Online RL (no conservative loss) + Offline CQL
        python main.py --env $ENV --algo cql --offline_rl --online_rl --num_rl_frames 2000 \
            --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
            --experiment_name "online_cql" --seed $seed

        # Exp 6: Online RL (no conservative loss) + Offline CQL + AWBC
        python main.py --env $ENV --algo cql --awbc --offline_rl --online_rl --num_rl_frames 2000 \
            --num_unified_epochs 200 --preload_expert_data $EXPERT_DATA \
            --experiment_name "online_awcql" --seed $seed
    done
done

echo "=========================================================="
echo "Final hands-free experimental suite completed."
echo "Use 'python plot_results_aggregate.py' to generate paper graphs."
echo "=========================================================="
