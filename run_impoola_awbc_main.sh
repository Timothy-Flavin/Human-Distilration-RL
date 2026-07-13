#!/bin/bash

# =====================================================================
# newdata_impoola_awbc — MAIN PC (4080), seeds sequential.
#
# Companion cell to run_impelu_awbc_lab.sh (newdata_impelu_awbc): same
# AWBC x Dataset B config, but with the new "impoola" encoder — the
# ImpalaCNN with the flatten replaced by global average pooling
# (Impoola architecture) plus channelwise LayerNorm after each conv
# block. ELU variant, so the delta vs impelu_awbc is GAP + LN + width
# (activation held fixed at the known-good choice). encoder_width 2 =
# the Impoola paper's recommended tau: depths [32,64,64], pooled
# feature 64-d (tau=1's 32-d bottleneck is the paper's weakest config).
#
# VRAM/speed: unified_update backwards each loss term as it is built
# (grad accumulation, one clip/step — same math), so only one IMPALA
# graph is live at a time; the AWBC arm's expert+online double graph
# OOM'd the 16GB card at tau=1 before this. --amp = bf16 autocast on
# the update forwards: tau=2 peaks 8.3GB (fits beside the ~2.9GB
# buffers) at 419 ms/epoch vs fp32's 13.4GB/789ms — grads within
# ~0.5% of fp32. If bf16 ever misbehaves, swap --amp for
# --encoder_chunks 4 (bit-exact fp32, 3.5GB, ~2x slower updates).
# AWBC advantages now reuse the imitation forward's own V estimates
# (no per-epoch update_value rebuild; numerically identical).
#
# 400 iterations, official-protocol score budget = iters 1-333.
# --resume makes reruns safe.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
EPOCHS=30
RL_FRAMES=2000

for seed in 42 43
do
    echo ""
    echo "[AWBC x impoola_elu] NEW data (Seed $seed) -> 400"
    python3 recurrent_main.py --env $ENV --online_rl \
        --encoder impoola_elu --encoder_width 2 --amp \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 400 \
        --num_envs 8 \
        --preload_expert_data $NEW_EXPERT_DATA \
        --experiment_name "impoola_online" --seed $seed \
        --resume
done

# for seed in 42 43
# do
#     echo ""
#     echo "[AWBC x impoola_elu] NEW data (Seed $seed) -> 400"
#     python3 recurrent_main.py --env $ENV --online_rl --awbc \
#         --encoder impoola_elu --encoder_width 2 --amp \
#         --bc_epsilon 0.02 \
#         --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS \
#         --total_iterations 400 \
#         --num_envs 8 \
#         --preload_expert_data $NEW_EXPERT_DATA \
#         --experiment_name "newdata_impoola_awbc" --seed $seed \
#         --resume
# done

echo ""
echo "newdata_impoola_awbc complete (both seeds)."
