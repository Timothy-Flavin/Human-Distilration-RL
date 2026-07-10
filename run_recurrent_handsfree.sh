#!/bin/bash

# Freshman: Recurrent Hands-Free Experimental Suite (Crafter)
# This script runs all experiments for the Crafter environment using the RCQL agent.
# It includes the drop_bottom 0.1 cleaning step as a pre-process.

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

# BC is supervised (no bootstrapping), so high data reuse is safe: fewer, larger
# training rounds. Online TD is bootstrapped, so keep the replay ratio low:
# 3 epochs x 64 seqs x 48 steps ~= 9.2k samples per 2k collected frames (~4.6x).
ITERATIONS_BC=15
EPOCHS_BC=100
ITERATIONS_ONLINE=400
EPOCHS_ONLINE=30
RL_FRAMES=2000

echo "=========================================================="
echo "Freshman: Starting Recurrent Hands-Free Suite (Crafter)"
echo "Applying drop_bottom 0.1 to expert data..."
#python3 analyze_expert_data.py --path $RAW_EXPERT_DATA --drop_bottom 0.1
echo "=========================================================="

# --- EXPERIMENT 1: Pure Behavior Cloning (Offline, stored-state burn-in) ---
# for seed in {1..3}
# do
#     echo ""
#     echo "[Exp 1] Recurrent BC (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --bc --num_rl_frames 0 \
#         --num_unified_epochs $EPOCHS_BC \
#         --total_iterations $ITERATIONS_BC \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "baseline_bc" --seed $seed
# done

# for seed in {1..3}
# do
#     echo ""
#     echo "[Exp 1] CQL (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --offline_rl --num_rl_frames 0 \
#         --num_unified_epochs $EPOCHS_BC \
#         --total_iterations $ITERATIONS_BC \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "baseline_rcql" --seed $seed
# done

# --- EXPERIMENT 5: Online DQN without BC (double DQN, stored-state burn-in) ---
# for seed in {50..52}
# do
#     echo ""
#     echo "[Exp 5] Online Recurrent DQN, no BC (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --online_rl \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data "" \
#         --experiment_name "online_dqn" --seed $seed


#--- EXPERIMENT 5: Online DQN + awbc ---

#     echo ""
#     echo "[Exp 5] Online AWBC DQN(Seed $seed)"
#     python3 recurrent_main.py --env $ENV --online_rl --awbc \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "online_offline_awbc" --seed $seed
# done

# for seed in {99..100}
# do
#     echo ""
#     echo "[Exp 5] Online Offline DQN, no BC (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --online_rl --offline_rl \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "online_offline" --seed $seed
# done

# --- EXPERIMENT 6: R2D3 (demos as plain TD replay, 1/16 demo ratio, no CQL) ---
# Human data grounds through observed returns instead of an imitation anchor:
# it can jumpstart learning but never cap the policy. Compare against
# online_dqn (free policy) and online_offline (CQL anchor).
# for seed in {10..12}
# do
#     echo ""
#     echo "[Exp 6] R2D3 demo-ratio replay (Seed $seed)"
#     python3 recurrent_main.py --env $ENV --online_rl --r2d3 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "r2d3" --seed $seed
# done

# --- EXPERIMENT 7: n-step / annealing ablations ---
# 7a RETIRED: online n_step=5 sped up Q1 (2.7 vs ~2.2) then plateaued at ~4.2
#    vs 7.1 for 1-step — uncorrected n-step is poisoned by eps-greedy/stale
#    replay actions inside the 5-step window. Do not re-run.
# 7b R2D3 with n-step on the DEMO channel only (--n_step_expert 5, online
#    stays 1-step): expert returns carry no exploration noise, so deep
#    backups there are clean; this is the DQfD-style optimistic prior.
# 7c anneals the CQL anchor 1.0 -> 0 over the first 500k frames: imitation
#    jumpstart + likeness early, free policy late. Matched 1-step so it
#    compares directly against the finished online_offline / online_dqn runs.
# for seed in {10..12}
# do
#     # echo ""
#     # echo "[Exp 7a] Online DQN + 5-step returns (Seed $seed)  [RETIRED]"
#     # python3 recurrent_main.py --env $ENV --online_rl --n_step 5 \
#     #     --num_rl_frames $RL_FRAMES \
#     #     --num_unified_epochs $EPOCHS_ONLINE \
#     #     --total_iterations $ITERATIONS_ONLINE \
#     #     --num_envs 8 \
#     #     --preload_expert_data "" \
#     #     --experiment_name "online_dqn_n5" --seed $seed

#     echo ""
#     echo "[Exp 7b] R2D3 + 5-step returns on demos only (Seed $seed)  [DONE: ~6.9, = online_dqn, CE at chance]"
#     python3 recurrent_main.py --env $ENV --online_rl --r2d3 --n_step_expert 5 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "r2d3_ne5" --seed $seed

#     echo ""
#     echo "[Exp 7c] Online+Offline, annealed CQL anchor (Seed $seed)  [DONE: plateaus 5.4-5.5 even at 2.3M frames]"
#     python3 recurrent_main.py --env $ENV --online_rl --offline_rl \
#         --cql_alpha 1.0 --cql_alpha_end 0.0 --cql_alpha_decay_frames 500000 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "online_offline_anneal" --seed $seed
# done

# --- EXPERIMENT 8: imitation term on demos + free online TD ---
# Gap analysis (2026-07-09): agents stall at wood tier — collect_stone ~0.48
# but make_stone_pickaxe <=0.01 for 1.5M frames (expert: 1.00/1.00), and die
# ~190 steps vs expert median 559. Q-only demo channels transfer nothing
# (r2d3 CE = chance); a global CQL anchor transfers likeness but caps return.
# Untested quadrant: supervised imitation on demos EVERY epoch + unconstrained
# online TD, imitation annealed 1.0 -> 0.1 floor over 500k frames (floor pins
# likeness; --bc_epsilon 0.02 keeps BC targets sharp, decoupled from the
# exploration eps 0.05 floor).
# 8a: BC + online TD only — no demo TD anywhere (cleanest test of the term).
# 8b: DQfD-lite — adds the 1/16 5-step demo TD channel from 7b on top; with
#     --bc the demo batch now samples every epoch and only demo TD is gated.
# for seed in {10..12}
# do
#     echo ""
#     echo "[Exp 8a] Online DQN + annealed demo BC (Seed $seed)  [DONE: 7.1 = online-only; stone pickaxes n.s.]"
#     python3 recurrent_main.py --env $ENV --online_rl --bc \
#         --bc_epsilon 0.02 \
#         --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "online_bc_anneal" --seed $seed

#     echo ""
#     echo "[Exp 8b] DQfD-lite (Seed $seed)  [DONE: ~8, 18 stone pickaxes — best arm, reward gain still small]"
#     python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
#         --bc_epsilon 0.02 \
#         --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
#         --num_rl_frames $RL_FRAMES \
#         --num_unified_epochs $EPOCHS_ONLINE \
#         --total_iterations $ITERATIONS_ONLINE \
#         --num_envs 8 \
#         --preload_expert_data $CLEANED_EXPERT_DATA \
#         --experiment_name "dqfd_lite" --seed $seed
# done

# --- EXPERIMENT 9: demo-state episode starts (Backplay) ---
# Change the START DISTRIBUTION instead of the loss: the first 2 of 8 envs
# (25%) reset from mid-demo states (world rebuilt from info['semantic'] +
# inventory/achievements/player_pos; only NEW achievements pay reward), so
# online TD learns at stone/iron-tier states on the agent's own policy.
# Restart points are re-scored each iteration against the current nets and
# sampled with priority toward states just BEFORE large TD error on the demo
# (--demo_start_priority 0.6, lookahead 50, 20% uniform floor).
# 9a: online-only from the new start distribution (baseline for 9b).
# 9b: dqfd_lite (Exp 8b) + demo starts.
for seed in {10..12}
do
    echo ""
    echo "[Exp 9a] Online DQN + demo-state starts (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS_ONLINE \
        --total_iterations $ITERATIONS_ONLINE \
        --num_envs 8 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "online_demostart" --seed $seed

    echo ""
    echo "[Exp 9b] DQfD-lite + demo-state starts (Seed $seed)"
    python3 recurrent_main.py --env $ENV --online_rl --bc --r2d3 --n_step_expert 5 \
        --bc_epsilon 0.02 \
        --bc_weight 1.0 --bc_weight_end 0.1 --bc_weight_decay_frames 500000 \
        --demo_start_envs 2 --demo_start_priority 0.6 \
        --num_rl_frames $RL_FRAMES \
        --num_unified_epochs $EPOCHS_ONLINE \
        --total_iterations $ITERATIONS_ONLINE \
        --num_envs 8 \
        --preload_expert_data $CLEANED_EXPERT_DATA \
        --experiment_name "dqfd_demostart" --seed $seed
done

# --- RESUME EXAMPLE: continue a finished run that is still trending up ---
# Same flags/experiment_name/seed as the original run select the results dir;
# --resume picks up RCQL_latest.pt (net+target+optimizer) and metrics, and
# continues from the last logged iteration up to the new --total_iterations.
# python3 recurrent_main.py --env $ENV --online_rl --offline_rl \
#     --num_rl_frames $RL_FRAMES \
#     --num_unified_epochs $EPOCHS_ONLINE \
#     --total_iterations 800 \
#     --num_envs 8 \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "online_offline" --seed 99 \
#     --resume

# # --- EXPERIMENT 2: Pure Offline RCQL (Offline RL) ---
# python3 recurrent_main.py --env $ENV --offline_rl --num_rl_frames 0 \
#     --num_unified_epochs $EPOCHS_BC --total_iterations $ITERATIONS_BC \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "baseline_rcql" --seed $seed

# # --- EXPERIMENT 3: Advantage-Weighted RCQL (Offline) ---
# python3 recurrent_main.py --env $ENV --bc --offline_rl --awbc --num_rl_frames 0 \
#     --num_unified_epochs $EPOCHS_BC --total_iterations $ITERATIONS_BC \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "baseline_awrcql" --seed $seed

# # --- EXPERIMENT 4: Hands-Free Online RL + BC ---
# python3 recurrent_main.py --env $ENV --online_rl --offline_rl --awbc \
#     --num_rl_frames $RL_FRAMES \
#     --num_unified_epochs $EPOCHS_ONLINE --total_iterations $ITERATIONS_ONLINE \
#     --preload_expert_data $CLEANED_EXPERT_DATA \
#     --experiment_name "online_offline_awbc" --seed $seed

echo ""
echo "=========================================================="
echo "All recurrent hands-free experiments completed successfully."
echo "Results saved in ./results/crafter/"
echo "=========================================================="
