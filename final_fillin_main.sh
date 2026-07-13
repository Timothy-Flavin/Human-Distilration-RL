#!/bin/bash

# =====================================================================
# FINAL fill-in — MAIN PC. One resume that must run here because the
# seed-43 checkpoint lives in this machine's results tree:
#   abl_impelu_noper s43: 400 -> 800, so the IMPALA-ELU (no PER,
#   online + demo starts) cell has 2 seeds at the 800-iter window
#   (currently 7.20 / 10.3% from s42 alone).
# Everything else is in final_fillin_labcomp.sh.
# Uses the OLD dataset under its post-rename path (identical contents
# to what the run originally preloaded).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
