#!/bin/bash

# =====================================================================
# Ablation phase 3 — MAIN PC (9950X + 4080). Resumes only.
# Companion: ablation3_labcomp.sh (one resume).
#
# Phase-2 verdict (return / crafter score, 2 seeds, paper formula):
#   nature online+PER  @400: 7.11 /  9.6   @800: 8.43 / 13.6  PLATEAUED
#     -> new project-best on BOTH metrics; beats every demo arm to date
#   elu online+PER     @400: 7.34 /  9.9   still rising +0.6/100it
#   dqfd_elu           @400: 6.86 / 12.2   still rising +0.6-0.8/100it
#     -> matches old best score (dqfd_lite 12.1) at HALF the frames:
#        demos buy sample-efficiency on the stone tier, not (yet) ceiling
#   elu OD noPER       @800: 7.20 / 10.3 (s42) PLATEAUED
#
# Remaining question: do the two risers plateau above or below the
# nature online 8.43/13.6 reference at matched 2.4M frames?
#   E1 resume abl_impelu_online_per s42,s43 -> 800  (encoder decision
#      at plateau; @400 elu vs nature is a coin flip, 7.34 vs 7.11)
#   E2 resume abl_dqfd_elu s42 -> 800  (do demos still matter at the
#      plateau, or does pure online catch up on score too?)
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
echo "[E2] RESUME DQfD-lite x impala_elu (Seed 42) -> 800"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder impala_elu \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_elu" --seed 42 \
    --resume

echo ""
echo "[E1] RESUME impala_elu online-only + PER (Seed 42) -> 800"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_impelu_online_per" --seed 42 \
    --resume

echo ""
echo "[E1] RESUME impala_elu online-only + PER (Seed 43) -> 800"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_impelu_online_per" --seed 43 \
    --resume

echo ""
echo "Phase-3 main-PC resumes complete."
