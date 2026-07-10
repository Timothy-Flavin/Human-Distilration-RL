#!/bin/bash

# =====================================================================
# IMPALA ablation — MAIN PC part (9950X + RTX 4080), ~60% of the grid
# Companion file: ablation_impala_labcomp.sh (EPYC + Ada 6000, ~40%)
#
# Goal: attribute the IMPALA-run regression (paper "I -" rows) to its
# parts. Everything here runs the NEW code path (unified single-backward
# loss) at the ORIGINAL replay ratio (30 epochs), 400 iterations, on the
# online + demo-starts arm unless stated. "fixed PER" = prioritized
# replay AFTER the episode-head pinning fix in buffers.py (per_burn_in).
#
# REQUIRES the buffers.py PER fix + the impala_elu encoder option
# (RCQL.py / recurrent_main.py) — do not run on pre-fix checkouts.
#
# Anchor cells already measured (no rerun needed):
#   nature  / no PER / old code / e30 : 6.32  (online_demostart s10)
#   impala  / buggy PER        / e60 : 3.45-3.90 (s42 x2, plateaued)
#   impala  / no PER / new code/ e30 : 4.58  (online_demostart_noper_e30
#                                             s42, still rising at 400)
#
# This file:
#   A1 abl_imp_per     impala + fixed PER   (seeds 42,43)
#        vs 4.58 no-PER -> does PER, once fixed, help or hurt impala?
#   A2 abl_nat_noper   nature + no PER, new code (seed 42)
#        vs 6.32 old code -> parity check: did the unified-loss refactor
#        itself regress nature? If this lands ~6.3 the refactor is clean
#        and the encoder explains the rest.
#   A3 abl_impelu_noper impala_elu + no PER (seed 42)
#        vs 4.58 relu-impala and 6.32 nature -> is the encoder gap the
#        ReLU dead-unit issue (nature was moved to ELU for exactly that)?
#
# ~10 s/iter impala, slightly less nature -> roughly 5-6 h total.
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
echo "IMPALA ablation (main PC): A1 impala+fixedPER x2, A2 nature"
echo "parity x1, A3 impala_elu x1 — all e30, 400 iters"
echo "=========================================================="

# --- A1 seed 42: impala + fixed PER (most informative first) ---
echo ""
echo "[A1] impala + fixed PER, online + demo starts (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_imp_per" --seed 42

# --- A2: nature + no PER on the NEW code (refactor parity check) ---
echo ""
echo "[A2] nature + no PER, new code, online + demo starts (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_nat_noper" --seed 42

# --- A1 seed 43 ---
echo ""
echo "[A1] impala + fixed PER, online + demo starts (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_imp_per" --seed 43

# --- A3: impala with ELU activations, no PER ---
echo ""
echo "[A3] impala_elu + no PER, online + demo starts (Seed 42)"
python3 recurrent_main.py --env $ENV --online_rl --encoder impala_elu --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_impelu_noper" --seed 42

echo ""
echo "=========================================================="
echo "Main-PC ablation runs complete. Results in ./results/crafter/"
echo "abl_imp_per, abl_nat_noper, abl_impelu_noper"
echo "=========================================================="
