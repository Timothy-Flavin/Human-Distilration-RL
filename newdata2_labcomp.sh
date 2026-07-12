#!/bin/bash

# =====================================================================
# New-data suite phase 2 — LAB COMPUTER. Attribution control.
#
# newdata_dqfd hit 9.29/19.5% @800 (vs old-data best 8.43/13.6%), but
# DQfD x nature + fixed PER was never run on the OLD dataset (phase-4
# abl_dqfd_nat was superseded by the new recording). This run closes
# that loop: same config, OLD cleaned data. If it plateaus ~7.5 like
# dqfd x elu did, the jump is attributable to the DATASET (short,
# iron-rich demos), not to dqfd x nature being latently strong.
# Old-code dqfd_lite (~7.4 flattening) already suggests this; one
# matched-code run makes it rigorous.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
OLD_EXPERT_DATA="expert_demonstrations_crafter_cleaned_old.pkl"
EPOCHS=30
RL_FRAMES=2000

for seed in 42 43
do
    echo ""
    echo "[F1] DQfD-lite x nature + fixed PER, OLD data (Seed $seed), 800 iters"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 800 \
        --num_envs 8 \
        --preload_expert_data $OLD_EXPERT_DATA \
        --experiment_name "olddata_dqfd_nat" --seed $seed
done

echo ""
echo "Old-data attribution control complete."
