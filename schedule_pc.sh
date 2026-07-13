#!/bin/bash

# =====================================================================
# Final schedule — THIS PC (4080). ~11 h total; budget 14 h.
# Launch AFTER the live H1 run (abl_impelu_noper s43 -> 800, ~0.9 h
# left) finishes — do not run concurrently with it.
# Companion: schedule_lab.sh (~13.5 h; budget 24 h; lab is idle, can
# start immediately).
#
# Measured rates (metrics_N.json mtimes): PC impala_elu dqfd
# 15.4 s/iter.
#   S1 newdata_impelu_dqfd s42 @1200  ~5.1 h
#   S2 newdata_impelu_dqfd s43 @1200  ~5.1 h
# (abl_nat_noper s43 moved to the lab's MPS lane B.)
# All --resume: idempotent, safe to rerun after interruption.
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

for seed in 42 43
do
    echo ""
    echo "[S1/S2] DQfD-lite x impala_elu + fixed PER, NEW data (Seed $seed) -> 1200"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder impala_elu \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 1200 \
        --num_envs 8 \
        --preload_expert_data $NEW_EXPERT_DATA \
        --experiment_name "newdata_impelu_dqfd" --seed $seed \
        --resume
done

echo ""
echo "PC schedule complete."
