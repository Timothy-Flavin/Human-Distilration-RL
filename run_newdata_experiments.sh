#!/bin/bash

# =====================================================================
# New expert dataset (recorded 2026-07-11) — demo-integration suite
# Nature CNN (ablation winner), new unified code, fixed PER, 30 epochs.
#
# Uses the FULL raw recording — no drop_bottom cleaning, every episode
# kept (unlike the old-dataset experiments, which dropped the bottom
# 10% by return). Per seed, runs all three arms:
#   1 newdata_dqfd     DQfD-lite: per-epoch BC + 1/16 5-step demo TD
#   2 newdata_dqfd_ds  DQfD-lite + demo-state starts (2 of 8 envs)
#   3 newdata_awbc     advantage-weighted BC (--awbc) + online TD;
#                      imitation-term ablation of arm 1 — no demo TD
#                      channel, advantages from the Q-net's own values
# Imitation anneal matched across all three: bc_weight 1.0 -> 0.1 over
# 500k frames, bc_epsilon 0.02.
#
# Reference plateaus @800 on the OLD dataset: nature online-only + PER
# 8.43 / 13.6%; dqfd x elu 7.25 / ~11.9%. 800 iters x 6 runs ~= 11 h
# on the 4080 (nature ~8 s/iter); halve ITERATIONS for a quick pass.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
# Full dataset, all episodes kept
EXPERT_DATA="expert_demonstrations_crafter.pkl"
ITERATIONS=800
EPOCHS=30
RL_FRAMES=2000

for seed in 42 43
do
    echo ""
    echo "[1] DQfD-lite, new data (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations $ITERATIONS \
        --num_envs 8 \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "newdata_dqfd" --seed $seed

    echo ""
    echo "[2] DQfD-lite + demo starts, new data (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations $ITERATIONS \
        --num_envs 8 \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "newdata_dqfd_ds" --seed $seed

    echo ""
    echo "[3] AWBC + online TD, new data (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --awbc \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations $ITERATIONS \
        --num_envs 8 \
        --preload_expert_data $EXPERT_DATA \
        --experiment_name "newdata_awbc" --seed $seed
done

echo ""
echo "=========================================================="
echo "New-data suite complete: newdata_dqfd, newdata_dqfd_ds,"
echo "newdata_awbc (seeds 42,43) in ./results/crafter/"
echo "=========================================================="
