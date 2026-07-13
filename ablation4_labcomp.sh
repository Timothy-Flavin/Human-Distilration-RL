#!/bin/bash

# =====================================================================
# Ablation phase 4 — LAB COMPUTER. One run: seed 43 of abl_dqfd_nat
# (see ablation4_main.sh header for rationale).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"
EPOCHS=30
RL_FRAMES=2000

echo ""
echo "[F1] DQfD-lite x nature + fixed PER, 800 iters (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder nature \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_nat" --seed 43

echo ""
echo "Phase-4 lab-comp run complete."
