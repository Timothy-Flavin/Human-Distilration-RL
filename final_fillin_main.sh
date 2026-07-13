#!/bin/bash

# =====================================================================
# FINAL fill-in — MAIN PC (4080).
#   G1 newdata_impelu_dqfd s42,43 @1200 — the missing cell: DQfD-lite
#      x IMPALA-ELU on the NEW data. Tests whether the "Nature >
#      ELU-IMPALA" ranking (established on old data at its ~8.4
#      ceiling) still holds at the new ~10.3 ceiling. Compare vs
#      newdata_dqfd 10.29 / 23.4% @1200. Runs HERE because the lab
#      EPYC's slow CPU phases would stretch these two IMPALA runs to
#      ~2 days; the 4080 box has the fast CPU.
#   H1 abl_impelu_noper s43: 400 -> 800 resume — must run here (the
#      seed-43 checkpoint lives in this machine's results tree), so
#      the IMPALA-ELU (no PER, online + DS) cell has 2 seeds at the
#      800-iter window (currently 7.20 / 10.3% from s42 alone).
# Everything else is in final_fillin_labcomp.sh (all-Nature runs).
# Old-data runs use the OLD dataset under its post-rename path
# (identical contents to what those runs originally preloaded).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- G1: DQfD-lite x IMPALA-ELU on the NEW data ---
for seed in 42 43
do
    echo ""
    echo "[G1] DQfD-lite x impala_elu + fixed PER, NEW data (Seed $seed) -> 1200"
    python3 recurrent_main.py --env crafter --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder impala_elu \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames 2000 \
        --num_unified_epochs 30 \
        --total_iterations 1200 \
        --num_envs 8 \
        --preload_expert_data expert_demonstrations_crafter.pkl \
        --experiment_name "newdata_impelu_dqfd" --seed $seed \
        --resume
done

echo ""
echo "[H1] RESUME impala_elu + no PER, online + demo starts (Seed 43) -> 800"
python3 recurrent_main.py --env crafter --online_rl --encoder impala_elu --no_per \
    --demo_start_envs 2 --demo_start_priority 0.6 \
    --num_rl_frames 2000 \
    --num_unified_epochs 30 \
    --total_iterations 800 \
    --num_envs 8 \
    --preload_expert_data expert_demonstrations_crafter_cleaned_old.pkl \
    --experiment_name "abl_impelu_noper" --seed 43 \
    --resume

echo ""
echo "Final fill-in (main PC) complete."
