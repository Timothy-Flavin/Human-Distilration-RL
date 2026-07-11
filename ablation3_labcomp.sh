#!/bin/bash

# =====================================================================
# Ablation phase 3 — LAB COMPUTER. One resume (see ablation3_main.sh
# header for the phase-2 verdict). Must run in the results tree where
# abl_dqfd_elu seed 43 lives on this machine.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"
EPOCHS=30
RL_FRAMES=2000

echo ""
echo "[E2] RESUME DQfD-lite x impala_elu (Seed 43) -> 800"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder impala_elu \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_elu" --seed 43 \
    --resume

echo ""
echo "Phase-3 lab-comp resume complete."
