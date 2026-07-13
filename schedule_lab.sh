#!/bin/bash

# =====================================================================
# Final schedule — LAB COMPUTER, dual-lane via CUDA MPS.
# RTX 6000 Ada (48GB) + 2x EPYC 7282; GPU is on NUMA node 1
# (CPUs 16-31,48-63). Two experiments run concurrently: MPS lets their
# kernels share the GPU instead of time-slicing, and each lane is
# pinned to its own NUMA node so the 2x9 processes (main + 8 env
# workers each) don't fight over cores. VRAM: 2 nature jobs ~5GB each.
#
#   Lane A (node 1, GPU-local):
#     L1 olddata_dqfd_nat s43 resume 653 -> 800   ~0.9 h
#     L2 newdata_dqfd s44 @1200                   ~6.3 h
#   Lane B (node 0):
#     L3 newdata_awbc s44 @1200                   ~6.3 h
#     L4 abl_nat_noper s43 @400                   ~2.1 h
#
# Rates measured solo at 18.8 s/iter (nature); expect ~10% concurrency
# overhead -> makespan ~9-10 h (budget 24 h). All --resume/idempotent:
# safe to rerun this script from the top after any interruption (MPS
# daemon start is also idempotent). Companion: schedule_pc.sh (IMPALA
# pair only).
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
OLD_EXPERT_DATA="expert_demonstrations_crafter_cleaned_old.pkl"
EPOCHS=30
RL_FRAMES=2000

# --- CUDA MPS ---
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d 2>/dev/null || true   # idempotent
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
trap stop_mps EXIT

# --- NUMA pinning (fall back to taskset if numactl is missing) ---
if command -v numactl >/dev/null; then
    PIN_A="numactl --cpunodebind=1 --preferred=1"   # GPU-local node
    PIN_B="numactl --cpunodebind=0 --preferred=0"
else
    PIN_A="taskset -c 16-31,48-63"
    PIN_B="taskset -c 0-15,32-47"
fi

lane_a() {
    echo "[L1] RESUME DQfD-lite x nature + fixed PER, OLD data (Seed 43) -> 800"
    $PIN_A python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder nature \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 800 \
        --num_envs 8 \
        --preload_expert_data $OLD_EXPERT_DATA \
        --experiment_name "olddata_dqfd_nat" --seed 43 \
        --resume

    echo "[L2] DQfD-lite, new data (Seed 44) -> 1200"
    $PIN_A python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
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
}

lane_b() {
    echo "[L3] AWBC + online TD, new data (Seed 44) -> 1200"
    $PIN_B python3 recurrent_main.py --env $ENV --online_rl --awbc \
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

    echo "[L4] nature + no PER, new code, online + demo starts (Seed 43) -> 400"
    $PIN_B python3 recurrent_main.py --env $ENV --online_rl --encoder nature --no_per \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations 400 \
        --num_envs 8 \
        --preload_expert_data $OLD_EXPERT_DATA \
        --experiment_name "abl_nat_noper" --seed 43 \
        --resume
}

lane_a > lane_a.log 2>&1 &
PID_A=$!
lane_b > lane_b.log 2>&1 &
PID_B=$!
echo "Lane A (L1,L2) pid $PID_A -> lane_a.log | Lane B (L3,L4) pid $PID_B -> lane_b.log"
wait $PID_A $PID_B

echo ""
echo "Lab schedule complete (both lanes)."
