#!/bin/bash

# =====================================================================
# Ablation phase 2 — LAB COMPUTER (EPYC + Ada 6000), ~40%
# Companion: ablation2_main.sh (see its header for the phase-1 verdict).
# Requires the ablation commit (4b64098: PER fix + impala_elu encoder).
#
#   D1 abl_dqfd_elu seed 43 — 2nd seed of DQfD-lite x impala_elu
#        (candidate new best overall; seed 42 runs on the main PC).
#   D2/D3 resume abl_nat_online_per s42/43 -> 800 iters — the new
#        nature online-only baselines were still rising +0.7-1.1/100it
#        at 400; their plateau anchors the final table.
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

# --- D1: DQfD-lite x impala_elu, seed 43 ---
echo ""
echo "[D1] DQfD-lite + impala_elu + fixed PER (Seed 43)"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder impala_elu \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations $ITERATIONS \
    --num_envs 8 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "abl_dqfd_elu" --seed 43

# --- D2/D3: extend the new online-only baselines (still rising at 400) ---
# NOTE: must run in the same results tree the B2 runs live in on this
# machine (results/crafter/abl_nat_online_per/...), same flags/seed.
echo ""
echo "[D2] RESUME nature + fixed PER online-only (Seed 42) -> 800 iters"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_nat_online_per" --seed 42 \
    --resume

echo ""
echo "[D3] RESUME nature + fixed PER online-only (Seed 43) -> 800 iters"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data "" \
    --experiment_name "abl_nat_online_per" --seed 43 \
    --resume

echo ""
echo "Phase-2 lab-comp runs complete."
