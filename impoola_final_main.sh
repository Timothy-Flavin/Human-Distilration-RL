#!/bin/bash

# =====================================================================
# Impoola AWBC — MAIN PC (4080). The "1.5 seeds": s43 from scratch +
# s42 resumed from its existing 400-iter checkpoint, both to 800.
# Companion: impoola_final_labcomp.sh (2 seeds impoola DQfD, MPS lanes).
#
# Target 1200 on both machines (Tim: lab's ~13 h is close enough).
# PC: s43 1200 x 19.3 s + s42 800 x 19.3 s ~= 10.8 h.
# --resume everywhere: reruns extend/skip cleanly.
#
# s42's round-1 run used --amp (confirmed by Tim), so amp stays on for
# every impoola run to keep seeds poolable.
# Round-1 reference: impoola_awbc s42 @1M official score 11.8%,
# @400 return 9.04 — project best @1M, pending this confirmation.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
ITERS=1200
EPOCHS=30
RL_FRAMES=2000

for seed in 43 42   # s43 first: the fresh confirmation seed
do
    echo ""
    echo "[AWBC x impoola_elu tau=2 + amp] NEW data (Seed $seed) -> $ITERS"
    python3 recurrent_main.py --env $ENV --online_rl --awbc \
        --encoder impoola_elu --encoder_width 2 --amp \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations $ITERS \
        --num_envs 8 \
        --preload_expert_data $NEW_EXPERT_DATA \
        --experiment_name "newdata_impoola_awbc" --seed $seed \
        --resume
done

echo ""
echo "Impoola AWBC (main PC) complete."
