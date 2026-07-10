#!/bin/bash

# ============================================================================
# Extend runs that were still improving at their 400-iteration cutoff.
#
# --resume loads RCQL_latest.pt (net + target + optimizer) and the metrics
# history from each run's results dir and continues from the last logged
# iteration up to --total_iterations. The online replay buffer cannot be
# restored, so before any updates the runner collects
# --resume_warmup_frames (default 20000) of update-free on-policy frames
# with the loaded weights — resuming straight into 30 epochs on a
# near-empty buffer would overfit a few episodes and could destroy the
# policy.
#
# Schedules survive the resume: epsilon sits at its 0.05 floor and the
# annealed CQL alpha stays at 0 (both derive from cumulative progress).
#
# NOTE: run this only after run_recurrent_handsfree.sh has finished — it
# targets the same results dirs.
# ============================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"
EPOCHS_ONLINE=30
RL_FRAMES=2000
EXTENDED_ITERATIONS=800   # 400 done + ~800k more frames

# --- online_offline_anneal: 4.5 and climbing ~linearly at cutoff ---
# for seed in {10..12}
# do
#     echo ""
#     echo "[Resume] online_offline_anneal (Seed $seed) -> $EXTENDED_ITERATIONS iterations"
#     python3 recurrent_main.py --env $ENV --online_rl --offline_rl \
#         --cql_alpha 1.0 --cql_alpha_end 0.0 --cql_alpha_decay_frames 500000 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $EXTENDED_ITERATIONS \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "online_offline_anneal" --seed $seed \
#         --resume
# done

# --- r2d3_ne5 [DONE: plateaued 6.8-7.0 by ~500 iters] ---
# for seed in {10..12}
# do
#     echo ""
#     echo "[Resume] r2d3_ne5 (Seed $seed) -> $EXTENDED_ITERATIONS iterations"
#     python3 recurrent_main.py --env $ENV --online_rl --r2d3 --n_step_expert 5 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $EXTENDED_ITERATIONS \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "r2d3_ne5" --seed $seed \
#         --resume
# done

# --- Exp 9 resumes: run only AFTER run_recurrent_handsfree.sh finishes. ---
# dqfd_demostart s10 @400: slope +1.4/100 iters (vs +0.9 for flattening
# dqfd_lite), hard-tier rates all rising in the last 50 iters (stone 24->32%,
# coal 4->7.6%, furnace trending); the late return dip is composition shift
# (sapling/plant farming down, mining up), not regression. CE ~0.6, |Q| ~3.
# bc_weight sits at its 0.1 floor and demo-start priorities recompute each
# iteration, so both schedules survive the resume; warmup collection includes
# the 2 demo-start envs. online_demostart resumes too so the 9a baseline
# stays matched-length.
for seed in {10..12}
do
    
    echo "[Resume] dqfd_demostart (Seed $seed) -> $EXTENDED_ITERATIONS iterations"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS_ONLINE \
        --total_iterations $EXTENDED_ITERATIONS \
        --num_envs 8 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "dqfd_demostart" --seed $seed \
        --resume

    echo ""
    echo "[Resume] online_demostart (Seed $seed) -> $EXTENDED_ITERATIONS iterations"
    python3 recurrent_main.py --env $ENV --online_rl \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS_ONLINE \
        --total_iterations $EXTENDED_ITERATIONS \
        --num_envs 8 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "online_demostart" --seed $seed \
        --resume

    echo ""
done

echo ""
echo "All resumes complete."
