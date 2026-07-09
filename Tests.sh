#!/bin/bash

# ============================================================================
# DIAGNOSTIC SUITE (Crafter / RCQL)
#
# Symptom being narrowed down: expert agreement is now high (validation
# bc_loss 0.2-0.4, ~97% top-1 on expert states) but eval return sits at the
# ~2.1 random baseline. The losses are learning; deployment is failing.
# Each phase below isolates ONE hypothesis. Run phases in order — Phase 0
# costs minutes and may answer the question outright.
#
# DECISION GUIDE (what each outcome means):
#   P0a  eps-greedy eval >> deterministic eval
#          -> argmax policy gets stuck in action loops; the ceiling is an
#             eval-time pathology, not a learning failure. Cheap fix.
#   P0b  hidden-reset eval >> normal eval
#          -> LSTM state drifts/saturates over long horizons; training only
#             ever BPTTs 64 steps. Consider periodic state reset or longer seqs.
#   P0c  achievement rates: only easy ones (collect_wood/sapling/drink), short
#        episodes
#          -> compounding covariate shift: the policy falls off the expert
#             manifold and dies. Points to data/corrections, not hyperparams.
#   P1   return scales with expert_fraction (0.25 < 0.5 < 1.0)
#          -> data-limited: more demos should raise the ceiling.
#        return flat across fractions
#          -> NOT data-limited at this scale; more demos won't help.
#   P2   zero_state ~= stored-state baseline
#          -> R2D2 stored-state/refresh machinery is irrelevant to the ceiling.
#   P3a  eval |Q| grows toward ~2-6 and return trends up
#          -> TD is healthy now; it just needs frames. Keep training.
#        eval |Q| stays near 0 / policy entropy collapses
#          -> TD still broken (look at lr, tau, grad clip saturation).
#   P3b  decoupled bc_epsilon >> coupled
#          -> exploration epsilon (0.25 early) was blurring the BC target.
#
# All eval logs now include: episode length, achievement rates, action
# distribution, and mean |Q| at deployment (metrics_*.json "evaluations").
# ============================================================================

# 1. Path to the local virtual environment
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
fi

# 2. Configuration
ENV="crafter"
RAW_EXPERT_DATA="expert_demonstrations_crafter.pkl"
CLEANED_EXPERT_DATA="expert_demonstrations_crafter_cleaned.pkl"

# Existing checkpoints probed by Phase 0 (best likeness examples so far)
BC_CKPT="results/crafter/baseline_bc/rcql_on0_off0_bc1_aw0_seed99/RCQL_latest.pt"
OO_CKPT="results/crafter/online_offline/rcql_on1_off1_bc0_aw0_seed99/RCQL_latest.pt"

echo "=========================================================="
echo "RCQL Diagnostic Suite (Crafter)"
echo "Applying drop_bottom 0.1 to expert data..."
python3 analyze_expert_data.py --path $RAW_EXPERT_DATA --drop_bottom 0.1
echo "=========================================================="

# ----------------------------------------------------------------------------
# PHASE 0: Eval-time probes on EXISTING checkpoints (no training, ~minutes).
# 30 episodes per probe gives +/-0.2 on the mean instead of +/-1.0 at 5.
# ----------------------------------------------------------------------------
for CKPT_SPEC in "bc:$BC_CKPT" "oo:$OO_CKPT"; do
    TAG="${CKPT_SPEC%%:*}"
    CKPT="${CKPT_SPEC#*:}"
    if [ ! -f "$CKPT" ]; then
        echo "[Phase 0] Skipping $TAG probe, checkpoint missing: $CKPT"
        continue
    fi

    # P0a: deterministic vs epsilon-greedy eval (stuck-in-a-loop probe)
    for eps in 0.0 0.01 0.05; do
        echo ""
        echo "[Phase 0a] $TAG checkpoint, eval_epsilon=$eps"
        python3 recurrent_main.py --env $ENV --eval_only \
            --load_checkpoint "$CKPT" \
            --eval_episodes 30 --eval_epsilon $eps \
            --experiment_name "probe_eval_$TAG" --seed 42
    done

    # P0b: periodic LSTM state reset (recurrent-drift probe)
    for hr in 64 128; do
        echo ""
        echo "[Phase 0b] $TAG checkpoint, eval_hidden_reset=$hr"
        python3 recurrent_main.py --env $ENV --eval_only \
            --load_checkpoint "$CKPT" \
            --eval_episodes 30 --eval_hidden_reset $hr \
            --experiment_name "probe_eval_$TAG" --seed 42
    done
done
# P0c needs no extra runs: compare achievement_rates in the probe JSONs
# (results/crafter/probe_eval_*/rcql_*/eval_probe_*.json) against the expert's
# rates from: python3 analyze_expert_data.py --path $CLEANED_EXPERT_DATA

# ----------------------------------------------------------------------------
# PHASE 1: Data-scaling law for BC (is the ceiling data-limited?).
# Short BC runs; 20 eval episodes for signal. ~3 x 45 min.
# ----------------------------------------------------------------------------
for frac in 0.25 0.5 1.0; do
    echo ""
    echo "[Phase 1] BC data scaling, expert_fraction=$frac"
    python3 recurrent_main.py --env $ENV --bc --num_rl_frames 0 \
        --num_unified_epochs 100 \
        --total_iterations 30 \
        --expert_fraction $frac \
        --eval_episodes 20 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "probe_datascale_$frac" --seed 10
done

# ----------------------------------------------------------------------------
# PHASE 2: Stored-state ablation (does the R2D2 machinery matter?).
# Same budget as a Phase 1 full-data run; differs ONLY in --zero_state.
# ----------------------------------------------------------------------------
echo ""
echo "[Phase 2] BC with zero stored state (vs probe_datascale_1.0)"
python3 recurrent_main.py --env $ENV --bc --num_rl_frames 0 \
    --num_unified_epochs 100 \
    --total_iterations 30 \
    --zero_state \
    --eval_episodes 20 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "probe_zerostate" --seed 10

# ----------------------------------------------------------------------------
# PHASE 3a: Online TD health check (does Q leave initialization now?).
# Watch "Eval |Q|" in the logs: healthy Q should approach the return scale
# (~2-6); flat-at-zero means TD optimization is still the blocker.
# ----------------------------------------------------------------------------
echo ""
echo "[Phase 3a] Online DQN TD health check (200k frames)"
python3 recurrent_main.py --env $ENV --online_rl \
    --num_rl_frames 2000 \
    --num_unified_epochs 30 \
    --total_iterations 100 \
    --num_envs 8 \
    --eval_episodes 10 \
    --preload_expert_data "" \
    --experiment_name "probe_td_health" --seed 10

# ----------------------------------------------------------------------------
# PHASE 3b: BC-target decoupling (does the exploration schedule blur BC?).
# Identical pair; only --bc_epsilon differs. Compare bc_loss AND return.
# ----------------------------------------------------------------------------
echo ""
echo "[Phase 3b] online+offline+awbc, coupled bc target (control)"
python3 recurrent_main.py --env $ENV --online_rl --offline_rl --awbc \
    --num_rl_frames 2000 \
    --num_unified_epochs 30 \
    --total_iterations 100 \
    --num_envs 8 \
    --eval_episodes 10 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "probe_bceps_coupled" --seed 10

echo ""
echo "[Phase 3b] online+offline+awbc, decoupled bc_epsilon=0.02"
python3 recurrent_main.py --env $ENV --online_rl --offline_rl --awbc \
    --num_rl_frames 2000 \
    --num_unified_epochs 30 \
    --total_iterations 100 \
    --num_envs 8 \
    --eval_episodes 10 \
    --bc_epsilon 0.02 \
    --preload_expert_data $CLEANED_EXPERT_DATA \
    --experiment_name "probe_bceps_decoupled" --seed 10

echo ""
echo "=========================================================="
echo "Diagnostic suite complete. Compare:"
echo "  Phase 0: results/crafter/probe_eval_*/rcql_*/eval_probe_*.json"
echo "  Phase 1-3: 'evaluations' in results/crafter/probe_*/rcql_*/metrics_latest.json"
echo "See the DECISION GUIDE at the top of this script."
echo "=========================================================="
