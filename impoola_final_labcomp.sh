#!/bin/bash

# =====================================================================
# Impoola DQfD — LAB COMPUTER, both seeds concurrent via MPS.
# Companion: impoola_final_main.sh (1.5 AWBC seeds on the 4080).
#
# newdata_impoola_dqfd s42 + s43 @1200, one seed per NUMA node.
# Projected ~37-40 s/iter/lane (impoola tau=2 + amp; 4080 measured
# 19.3, lab impala ratio ~1.9x) -> ~13 h ("close enough" per Tim).
# VRAM ~2 x 14GB on the 48GB card. --resume: reruns extend/skip.
# amp stays on to match every other impoola run.
# =====================================================================

if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

ENV="crafter"
NEW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
ITERS=1200
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
    echo "[DQfD-lite x impoola_elu tau=2 + amp] NEW data (Seed $seed) -> $ITERS"
    $pin python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --encoder impoola_elu --encoder_width 2 --amp \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS \
        --total_iterations $ITERS \
        --num_envs 8 \
        --preload_expert_data $NEW_EXPERT_DATA \
        --experiment_name "newdata_impoola_dqfd" --seed $seed \
        --resume
}

run_seed "$PIN_A" 42 > impoola_dqfd_s42.log 2>&1 &
PID_A=$!
run_seed "$PIN_B" 43 > impoola_dqfd_s43.log 2>&1 &
PID_B=$!
echo "Seed 42 pid $PID_A -> impoola_dqfd_s42.log | Seed 43 pid $PID_B -> impoola_dqfd_s43.log"
wait $PID_A $PID_B

echo ""
echo "Impoola DQfD (lab) complete (both seeds)."
