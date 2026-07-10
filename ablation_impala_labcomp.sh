#!/bin/bash

# =====================================================================
# IMPALA ablation — LAB COMPUTER part (EPYC + RTX Ada 6000), ~40%
# Companion file: ablation_impala_main.sh (9950X + 4080, ~60%)
#
# All-nature runs so this machine only needs the buffers.py PER fix
# (per_burn_in episode-head pinning fix) — REQUIRED, do not run on the
# pre-fix checkout, or PER results will reproduce the pinning bug.
# New code path (unified loss), 30 epochs, 400 iterations.
#
# This file:
#   B1 abl_nat_per        nature + fixed PER, online + demo starts
#                         (seeds 42,43)
#        vs abl_nat_noper (main PC) and old-code 6.32 -> does fixed PER
#        help the nature encoder? Completes the encoder x PER 2x2 with
#        A1/A2 on the main PC.
#   B2 abl_nat_online_per nature + fixed PER, ONLINE-ONLY, no demo data
#                         (seeds 42,43)
#        vs old R2D2 online_dqn 6.5-7.1 (e30, lab-comp_results/) -> the
#        new baseline row: nature CNN + new loss + fixed prioritized
#        replay, nothing else. No expert preload, no demo starts.
#
# 4 nature runs, ~1 h each on the 4080; likely faster here.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"
ITERATIONS=400
EPOCHS=30
RL_FRAMES=2000

echo "=========================================================="
echo "IMPALA ablation (lab comp): B1 nature+fixedPER x2,"
echo "B2 nature+fixedPER online-only baseline x2 — e30, 400 iters"
echo "=========================================================="

# --- B1 seed 42: nature + fixed PER, demo starts ---
echo ""
echo "[B1] nature + fixed PER, online + demo starts (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_nat_per" --seed 42

# --- B2 seed 42: new baseline — nature + fixed PER, online only ---
echo ""
echo "[B2] nature + fixed PER, online-only baseline (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_nat_online_per" --seed 42

# --- B1 seed 43 ---
echo ""
echo "[B1] nature + fixed PER, online + demo starts (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_nat_per" --seed 43

# --- B2 seed 43 ---
echo ""
echo "[B2] nature + fixed PER, online-only baseline (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_nat_online_per" --seed 43

echo ""
echo "=========================================================="
echo "Lab-comp ablation runs complete. Results in ./results/crafter/"
echo "abl_nat_per, abl_nat_online_per"
echo "=========================================================="
