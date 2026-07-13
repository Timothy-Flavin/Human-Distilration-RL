#!/bin/bash

# =====================================================================
# Ablation phase 4 — MAIN PC. One run. Companion: ablation4_labcomp.sh
# (same config, seed 43).
#
# Phase-3 verdict @800 iters / 2.4M frames (pooled 701-800, 2 seeds):
#   nature online+PER   8.43 / 13.6%   <- winner on both metrics
#   elu online+PER      7.52 / 10.7%   (matched nature @400, caps lower;
#                                       ELU fixes ReLU-impala but the
#                                       deeper net still plateaus below)
#   dqfd_elu            7.25 / 12.0%   (score lead @400 gone by 800)
#
# One confound remains before the paper can claim "demos buy sample
# efficiency but do not raise the ceiling": dqfd's plateau deficit was
# measured on the ELU-IMPALA encoder, which itself caps ~0.9 below
# nature. The old nature dqfd_lite (7.3-7.4 flattening) is suggestive
# but ran on the old code without PER.
#
#   F1 abl_dqfd_nat: DQfD-lite x NATURE + fixed PER, 800 iters from
#      scratch — the ceiling comparison against nature online 8.43/13.6
#      with the encoder held fixed. Last cell of the grid.
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
echo "[F1] DQfD-lite x nature + fixed PER, 800 iters (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder nature \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_nat" --seed 42

echo ""
echo "Phase-4 main-PC run complete."
