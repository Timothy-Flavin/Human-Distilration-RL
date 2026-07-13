#!/bin/bash

# =====================================================================
# newdata_impelu_awbc — LAB COMPUTER, both seeds concurrent via MPS.
#
# Rationale: under the official leaderboard protocol @1M, IMPALA-ELU x
# DQfD scored 9.2±0.3 (best) via its fast start, and AWBC is the best
# imitation term @1M on Nature (9.1±0.4). This cell combines them:
# IMPALA-ELU x AWBC on Dataset B — the candidate best @1M config.
# 400 iterations is all the question needs (score budget = iters 1-333).
#
# Dual-lane MPS as in schedule_lab.sh: one seed per NUMA node,
# ~2x15GB VRAM on the 48GB card, ~3.5-4 h total at lab impala rates.
# --resume makes reruns safe.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
EPOCHS=30
RL_FRAMES=2000

# --- CUDA MPS ---
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null || true
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
trap stop_mps EXIT

if command -v numactl >/dev/null; then
    PIN_A="numactl --cpunodebind=1 --preferred=1"   # GPU-local node
    PIN_B="numactl --cpunodebind=0 --preferred=0"
else
    PIN_A="taskset -c 16-31,48-63"
    PIN_B="taskset -c 0-15,32-47"
fi

run_seed() {
    local pin="$1" seed="$2"
    echo "[AWBC x impala_elu] NEW data (Seed $seed) -> 400"
    $pin python3 recurrent_main.py --env $ENV --online_rl --awbc \
        --encoder impala_elu \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 400 \
        --num_envs 8 \
        --preload_expert_data $NEW_EXPERT_DATA \
        --experiment_name "newdata_impelu_awbc" --seed $seed \
        --resume
}

run_seed "$PIN_A" 42 > impelu_awbc_s42.log 2>&1 &
PID_A=$!
run_seed "$PIN_B" 43 > impelu_awbc_s43.log 2>&1 &
PID_B=$!
echo "Seed 42 pid $PID_A -> impelu_awbc_s42.log | Seed 43 pid $PID_B -> impelu_awbc_s43.log"
wait $PID_A $PID_B

echo ""
echo "newdata_impelu_awbc complete (both seeds)."
