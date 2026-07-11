#!/bin/bash

# =====================================================================
# Ablation phase 2 — MAIN PC (9950X + 4080), ~60%
# Companion: ablation2_labcomp.sh (EPYC + Ada 6000, ~40%)
#
# Phase-1 verdict (pooled evals 301-400, e30, 400 iters, unified loss):
#   unified refactor: innocent (nature noPER new 6.08 vs old 6.32)
#   fixed PER:        neutral both encoders (nature 6.10 vs 6.08;
#                     impala 4.41 vs 4.58)
#   activation is the whole encoder story: impala-ReLU 4.4-4.6,
#     impala-ELU 6.60 (>= nature, steepest slope +1.0/100it)
#   new best baseline: nature online-only + PER + unified 7.07/7.16,
#     still rising +0.7-1.1/100it
#   demo starts cost ~1 return on pure-online arms (7.1 vs 6.1)
#
# Phase 2: crown a final config.
#   C1 abl_impelu_online_per  impala_elu, ONLINE-ONLY, fixed PER
#        (s42,43) — head-to-head vs nature 7.07/7.16 on the best arm;
#        decides the paper encoder.
#   C2 abl_dqfd_elu           DQfD-lite x impala_elu, fixed PER (s42)
#        — best demo arm (nature dqfd_lite 7.41) x best encoder;
#        candidate new best overall. ELU's stronger combat/crafting
#        (zombie 0.64, stone 0.13 on OD) may compound with BC there.
#   C3 abl_impelu_noper seed 43 — 2nd seed for the phase-1 headline
#        (6.60 is single-seed; impala seed spread was ~1.0).
#   C4 resume abl_impelu_noper s42 -> 800 iters (was rising +1.0/100it;
#        plateau needed for the final table).
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

# --- C1 seed 42: impala_elu online-only baseline (decision-critical) ---
echo ""
echo "[C1] impala_elu + fixed PER, online-only (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_impelu_online_per" --seed 42

# --- C2 seed 42: DQfD-lite x impala_elu ---
echo ""
echo "[C2] DQfD-lite + impala_elu + fixed PER (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder impala_elu \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_elu" --seed 42

# --- C3: second seed for the phase-1 impala_elu headline ---
echo ""
echo "[C3] impala_elu + no PER, online + demo starts (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_impelu_noper" --seed 43

# --- C1 seed 43 ---
echo ""
echo "[C1] impala_elu + fixed PER, online-only (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_impelu_online_per" --seed 43

# --- C4: extend the phase-1 impala_elu run (still rising at 400) ---
echo ""
echo "[C4] RESUME impala_elu + no PER OD (Seed 42) -> 800 iters"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_impelu_noper" --seed 42 \
    --resume

echo ""
echo "Phase-2 main-PC runs complete."
