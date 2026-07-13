#!/bin/bash

# =====================================================================
# FINAL fill-in suite — LAB COMPUTER. Completes the paper grid:
# every unified-update cell gets >=2 seeds (3 for the headline arms).
# Companion: final_fillin_main.sh (G1 + one checkpoint-bound resume).
#
# All-Nature runs here: the lab EPYC's slow single-thread CPU stretches
# the collection/eval phases, so the GPU-heavy IMPALA cell (G1) runs on
# the 4080 instead (see final_fillin_main.sh).
#
# Ordered by importance:
#   G2 olddata_dqfd_nat: finish s42 (720 -> 800) and run s43 — second
#      seed for the attribution control (currently 8.04 / 11.6%, s42
#      only). --resume makes both idempotent: skip/continue if already
#      done or running.
#   G3 third seeds (44) for the two headline arms, @1200:
#      newdata_dqfd (10.29 / 23.4%) and newdata_awbc (9.39 / 21.5%).
#   G4 abl_nat_noper s43 @400 — second seed for the refactor-parity
#      cell (6.08 / 7.3% vs old-code 6.32 / 7.2%).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
OLD_EXPERT_DATA="expert_demonstrations_crafter_cleaned_old.pkl"
EPOCHS=30
RL_FRAMES=2000

# --- G2: attribution control, finish s42 + second seed ---
for seed in 42 43
do
    echo ""
    echo "[G2] DQfD-lite x nature + fixed PER, OLD data (Seed $seed) -> 800"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 800 \
        --num_envs 8 \
        --preload_expert_data $OLD_EXPERT_DATA \
        --experiment_name "olddata_dqfd_nat" --seed $seed \
        --resume
done

# --- G3: third seeds for the headline arms ---
echo ""
echo "[G3] DQfD-lite, new data (Seed 44) -> 1200"
python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
    --encoder nature \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 1200 \
    --num_envs 8 \
    --preload_expert_data $NEW_EXPERT_DATA \
    --experiment_name "newdata_dqfd" --seed 44 \
    --resume

echo ""
echo "[G3] AWBC + online TD, new data (Seed 44) -> 1200"
python3 recurrent_main.py --env $ENV --online_rl --awbc \
    --encoder nature \
    --bc_epsilon 0.02 \
    --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 1200 \
    --num_envs 8 \
    --preload_expert_data $NEW_EXPERT_DATA \
    --experiment_name "newdata_awbc" --seed 44 \
    --resume

# --- G4: second seed for the refactor-parity cell ---
echo ""
echo "[G4] nature + no PER, new code, online + demo starts (Seed 43) -> 400"
python3 recurrent_main.py --env $ENV --online_rl --encoder nature --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames $RL_FRAMES \
    --num_unified_epochs $EPOCHS \
    --total_iterations 400 \
    --num_envs 8 \
    --preload_expert_data $OLD_EXPERT_DATA \
    --experiment_name "abl_nat_noper" --seed 43 \
    --resume

echo ""
echo "Final fill-in (lab) complete."
