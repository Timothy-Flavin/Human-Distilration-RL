#!/bin/bash

# =====================================================================
# New-data suite phase 2 — MAIN PC. Resumes only.
# Companion: newdata2_labcomp.sh (old-data attribution control).
#
# New-data results @800 (2 seeds, pooled 701-800; ref: best old-data
# config = nature online-only + PER, PLATEAUED at 8.43 / 13.6%):
#   newdata_awbc     9.32 / 20.4%  still rising +0.3-0.6/100it
#   newdata_dqfd     9.29 / 19.5%  still rising +0.5-0.8/100it
#   newdata_dqfd_ds  8.33 / 13.9%  flattening — demo starts hurt again
# Iron tier open: collect_iron 1.2-1.3%, stone_pickaxe 18-27%,
# furnace 8-12% (online-only: 0 / 1.3 / 0.4).
#
# Both winners are still climbing at 800 -> extend to 1200 to find the
# plateaus. Demo-start arm left at 800 (consistent negative, done).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
EXPERT_DATA="expert_demonstrations_crafter.pkl"
EPOCHS=30
RL_FRAMES=2000

for seed in 42 43
do
    echo ""
    echo "[R] RESUME AWBC + online TD, new data (Seed $seed) -> 1200"
    python3 recurrent_main.py --env $ENV --online_rl --awbc \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 1200 \
        --num_envs 8 \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "newdata_awbc" --seed $seed \
        --resume

    echo ""
    echo "[R] RESUME DQfD-lite, new data (Seed $seed) -> 1200"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 1200 \
        --num_envs 8 \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "newdata_dqfd" --seed $seed \
        --resume
done

echo ""
echo "New-data phase-2 resumes complete."
