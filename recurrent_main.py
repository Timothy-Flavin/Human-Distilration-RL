import gymnasium as gym
import torch
import numpy as np
import random
import os
import json
import torch.nn.functional as F
import time
import collections
import pickle

from Agent import Agent
from RCQL import RCQLAgent
from wrapper import InteractiveGymWrapper
from buffers import (
    ReplayBuffer,
    LLMBuffer,
    CurriculumBuffer,
    SemiSupervisedBuffer,
    ObservationBuffer,
    FastGPUEpisodicBuffer,
)
from metrics import MetricsLogger
from llm_router import LLMRouter
from eval_agent import evaluate_return, calculate_cross_entropy
from verification_manager import VerificationManager

torch.set_num_threads(4)


# --- Crafter Environment Wrapper ---
class CrafterGymnasiumWrapper:
    def __init__(self):
        import crafter

        self._env = crafter.Env()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        # if seed is not None:
        #    self._env.seed(seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info

    def render(self):
        return self._env.render(size=(512, 512))

    def close(self):
        pass


# --- Sub-functions ---
def pre_load_episodic_data(args, buffers, metrics):
    """Loads expert episodic demonstrations straight into VRAM."""
    if args.preload_expert_data and os.path.exists(args.preload_expert_data):
        with open(args.preload_expert_data, "rb") as f:
            expert_dataset = pickle.load(f)

        # Data-scaling probe: train on a random subset of the expert episodes
        if getattr(args, "expert_fraction", 1.0) < 1.0:
            rng = random.Random(args.seed)
            keep = max(1, int(len(expert_dataset) * args.expert_fraction))
            expert_dataset = rng.sample(expert_dataset, keep)
            print(
                f"[Preload] Subsampled expert data to {keep} episodes (fraction {args.expert_fraction})."
            )

        loaded_count = 0
        total_duration = 0.0

        for item in expert_dataset:
            # Handle both [transitions, ...] and [{'transitions': transitions}, ...]
            if isinstance(item, dict) and "transitions" in item:
                transitions = item["transitions"]
                duration = item.get("duration", len(transitions) / 30.0)
            else:
                transitions = item
                duration = len(transitions) / 30.0

            for t in transitions:
                t["reward"] = t.get("reward", 0.0) #* 10.0

            buffers["expert"].add_episode(transitions)
            loaded_count += len(transitions)
            total_duration += duration

        print(
            f"[Preload] Successfully loaded {len(expert_dataset)} episodes ({loaded_count} transitions) into GPU buffer."
        )
        metrics.log_frames(loaded_count, source="expert_preload")
        metrics.timers["expert_preload_effort"] = total_duration
    elif args.preload_expert_data:
        print(
            f"[Preload] Warning: Expert data file not found at {args.preload_expert_data}"
        )


def run_rl_collection(agent, env, buffers, num_frames, metrics, min_episodes=0):
    """Collects episodic experience into the fast online buffer."""
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0

    current_episodes = buffers["online"].current_size
    effective_min = min_episodes if current_episodes < min_episodes else 0

    if effective_min > 0:
        print(
            f"Collecting RL experience (Target: {num_frames} frames OR {effective_min} new episodes)..."
        )
    else:
        print(f"Collecting {num_frames} RL frames...")

    while total_frames < num_frames or len(episodes) < effective_min:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        agent.reset_hidden()
        terminated = False
        truncated = False
        episode_transitions = []
        trajectory_lite = []
        total_reward = 0

        # Initial dummy state for trajectory tracking
        trajectory_lite.append(
            {
                "obs": obs,
                "action": 0,
                "reward": 0,
                "next_obs": obs,
                "frame_image": None,
                "terminated": False,
                "truncated": False,
                "env_state": None,
                "source": "rl",
            }
        )

        while not (terminated or truncated):
            # Snapshot the actor's recurrent state BEFORE it consumes obs (R2D2 stored-state)
            hidden_snapshot = agent.get_hidden_snapshot()
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_transitions.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward, # * 10.0,
                    "next_obs": next_obs,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                    "hidden": hidden_snapshot,
                }
            )

            trajectory_lite.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward, # * 10.0,
                    "next_obs": next_obs,
                    "frame_image": None,
                    "terminated": terminated,
                    "truncated": truncated,
                    "env_state": None,
                    "source": "rl",
                }
            )
            total_reward += reward
            obs = next_obs
            metrics.log_frames(1, source="rl")
            total_frames += 1
            # We don't break early here to ensure full episodes are always recorded
            # if total_frames >= num_frames: break

        # Push to the fast GPU buffer
        if len(episode_transitions) > 0:
            buffers["online"].add_episode(episode_transitions)
            episodes.append(
                {
                    "seed": seed,
                    "total_reward": total_reward,
                    "trajectory": trajectory_lite,
                }
            )

        if total_frames >= num_frames and len(episodes) >= effective_min:
            break

    metrics.stop_timer("rl_experience")
    return episodes


def run_rl_collection_parallel(agent, penvs, buffers, num_frames, metrics, min_episodes=0):
    """Parallel version of run_rl_collection: envs step in worker processes,
    one batched GPU forward drives all of them. Stores only whole episodes.
    Once the frame budget is met, in-flight episodes run to completion but
    finished envs are not reset."""
    metrics.start_timer("rl_experience")
    n = penvs.num_envs
    episodes = []
    total_frames = 0

    current_episodes = buffers["online"].current_size
    effective_min = min_episodes if current_episodes < min_episodes else 0
    print(f"Collecting {num_frames} RL frames across {n} parallel envs...")

    obs_list = penvs.reset_all()
    agent.reset_hidden()
    ep_transitions = [[] for _ in range(n)]
    ep_rewards = [0.0] * n
    running = set(range(n))    # envs mid-episode
    resetting = set()          # envs regenerating a world (async, ~100ms+)

    while running or resetting:
        # Absorb any finished async resets before snapshotting hiddens
        for i in list(resetting):
            new_obs = penvs.poll_reset(i)
            if new_obs is not None:
                obs_list[i] = new_obs
                agent.reset_hidden_index(i)
                resetting.discard(i)
                running.add(i)

        if not running:
            time.sleep(0.001)
            continue

        snaps = {i: agent.get_hidden_snapshot(idx=i) for i in running}
        # Forward the full batch (cheap); actions of idle envs are ignored.
        actions = agent.predict(np.stack(obs_list), deterministic=False)
        results = penvs.step({i: actions[i] for i in running})

        for i, (next_obs, reward, done, info) in results.items():
            ep_transitions[i].append(
                {
                    "obs": obs_list[i],
                    "action": int(actions[i]),
                    "reward": reward, # * 10.0,
                    "next_obs": next_obs,
                    "terminated": done,
                    "truncated": False,
                    "info": info,
                    "hidden": snaps[i],
                }
            )
            ep_rewards[i] += reward
            obs_list[i] = next_obs
            metrics.log_frames(1, source="rl")
            total_frames += 1

            if done:
                buffers["online"].add_episode(ep_transitions[i])
                episodes.append(
                    {"seed": None, "total_reward": ep_rewards[i], "trajectory": None}
                )
                ep_transitions[i] = []
                ep_rewards[i] = 0.0
                running.discard(i)
                if total_frames < num_frames or len(episodes) < effective_min:
                    penvs.reset_async(i)
                    resetting.add(i)

    metrics.stop_timer("rl_experience")
    return episodes


def hydrate_trajectory(env, seed, trajectory_lite):
    print(f"[Hydration] Re-simulating episode (Seed: {seed})...")
    env.reset(seed=seed)
    trajectory_lite[0]["frame_image"] = env.render()
    for i in range(1, len(trajectory_lite)):
        action = trajectory_lite[i]["action"]
        env.step(action)
        trajectory_lite[i]["frame_image"] = env.render()
    return trajectory_lite


def unified_train_step(args, agent, buffers, metrics):
    batch_size = 64
    seq_len = 64
    burn_in = 16

    has_online = args.online_rl and buffers["online"].current_size >= 8
    has_offline = (args.offline_rl or args.bc or args.awbc or args.r2d3) and buffers[
        "expert"
    ].current_size > 0

    if not (has_online or has_offline):
        print(
            f"[Training] Skipping updates: Not enough episodes (Online: {buffers['online'].current_size}, Offline: {buffers['expert'].current_size})"
        )
        return

    print(f"Updating Recurrent Agent ({args.num_unified_epochs} epochs)...")

    # Expert demos carry no actor hidden states, and any stored states go stale as
    # weights move: replay episodes through the current network to refresh them.
    # --zero_state skips this (ablation: demo windows then train from zero hiddens).
    if has_offline and not args.zero_state:
        metrics.start_timer("hidden_refresh")
        buffers["expert"].refresh_hidden_states(agent.q_net)
        metrics.stop_timer("hidden_refresh")

    start_time = time.time()
    total_frames_processed = 0

    for epoch in range(args.num_unified_epochs):

        exp_batch = None
        if has_offline:
            # R2D3-style demo ratio: demos are plain replay sampled once every
            # 16 epochs instead of a per-epoch CQL anchor.
            if not args.r2d3 or epoch % 16 == 0:
                exp_batch = buffers["expert"].sample_batch(batch_size, seq_len=seq_len)
                total_frames_processed += batch_size * (seq_len + 1)

        on_batch = None
        if has_online:
            on_batch = buffers["online"].sample_batch(batch_size, seq_len=seq_len)
            total_frames_processed += batch_size * (seq_len + 1)

        # 1. TD / POLICY UPDATES (CQL or Online RL)
        cached_h_q = None
        cached_v = None
        if (args.offline_rl or args.r2d3) and exp_batch:
            metrics.start_timer("agent_updating_rl")
            expert_n = (
                args.n_step_expert if args.n_step_expert is not None else args.n_step
            )
            td_results = agent.update_td(
                *exp_batch, burn_in=burn_in, use_cql=not args.r2d3, n_step=expert_n
            )
            cached_h_q = td_results.get("h_q")
            cached_v = (td_results["current_v"], td_results["next_v"])
            metrics.stop_timer("agent_updating_rl")

        if args.online_rl and on_batch:
            metrics.start_timer("agent_updating_rl")
            agent.update_td(
                *on_batch, burn_in=burn_in, use_cql=False, n_step=args.n_step
            )
            metrics.stop_timer("agent_updating_rl")

        # 2. VALUE FUNCTION UPDATES (for AWBC)
        if args.awbc:
            train_v = not (args.online_rl or args.offline_rl)

            if exp_batch and (cached_v is None or train_v):
                metrics.start_timer("agent_updating_value")
                v_results = agent.update_value(
                    *exp_batch, burn_in=burn_in, train=train_v
                )
                cached_v = (v_results["current_v"], v_results["next_v"])
                metrics.stop_timer("agent_updating_value")

            # Update value using online data (if available and training V)
            if on_batch and train_v:
                metrics.start_timer("agent_updating_value")
                agent.update_value(*on_batch, burn_in=burn_in, train=True)
                metrics.stop_timer("agent_updating_value")

        # 3. SUPERVISED / BC UPDATES
        if (args.awbc or args.bc) and exp_batch:
            metrics.start_timer("agent_updating_bc")
            advantages = None
            if args.awbc and cached_v is not None:
                # Calculate Advantage using Value Network: A = r + gamma*V(s') - V(s)
                v_s, v_ns = cached_v
                r_active = exp_batch[2][:, burn_in:]  # rewards
                d_active = exp_batch[3][:, burn_in:]  # dones
                td_error = r_active + (1.0 - d_active) * agent.gamma * v_ns - v_s
                advantages = F.relu(td_error + 1.0)

            agent.update_supervised(
                exp_batch[0],
                exp_batch[1],
                exp_batch[4],
                init_hidden=exp_batch[5],
                burn_in=burn_in,
                advantages=advantages,
                h_q=cached_h_q,
                bc_epsilon=args.bc_epsilon if args.bc_epsilon >= 0 else None,
            )
            metrics.stop_timer("agent_updating_bc")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            fps = total_frames_processed / (elapsed + 1e-6)
            print(
                f"    Epoch {epoch+1}/{args.num_unified_epochs} complete... ({fps:.2f} FPS)"
            )

    total_elapsed = time.time() - start_time
    total_fps = total_frames_processed / (total_elapsed + 1e-6)
    print(f"[Benchmark] Training step complete. Total FPS: {total_fps:.2f}")
    metrics.timers["training_throughput_fps"] = total_fps


def run_eval(agent, env, num_episodes=5, epsilon=0.0, hidden_reset=0):
    """Evaluation with diagnostics. Two probe knobs isolate deployment failures:
    epsilon > 0 uses epsilon-greedy instead of argmax (stuck-in-a-loop probe);
    hidden_reset = N zeroes the LSTM state every N steps (recurrent-drift probe).
    Returns per-episode returns/lengths, achievement rates, action distribution,
    and mean |Q| at deployment."""
    rewards = []
    lengths = []
    action_counts = collections.defaultdict(int)
    achievement_counts = collections.defaultdict(int)
    q_mag_sum, q_mag_n = 0.0, 0

    old_epsilon = agent.epsilon
    if epsilon > 0:
        agent.epsilon = epsilon
    for _ in range(num_episodes):
        e_obs, _ = env.reset()
        agent.reset_hidden()
        e_term = False
        e_trunc = False
        e_total = 0.0
        steps = 0
        last_info = {}
        while not (e_term or e_trunc):
            e_act = agent.predict(e_obs, deterministic=(epsilon <= 0))
            action_counts[int(e_act)] += 1
            if agent.last_q is not None:
                q_mag_sum += agent.last_q.abs().mean().item()
                q_mag_n += 1
            e_obs, e_rew, e_term, e_trunc, e_info = env.step(e_act)
            e_total += e_rew
            steps += 1
            if isinstance(e_info, dict):
                last_info = e_info
            if hidden_reset > 0 and steps % hidden_reset == 0:
                agent.reset_hidden()
        rewards.append(float(e_total))
        lengths.append(steps)
        # Crafter reports cumulative achievement counts in info
        for ach, val in (last_info.get("achievements") or {}).items():
            if val > 0:
                achievement_counts[ach] += 1
    agent.epsilon = old_epsilon

    return {
        "num_episodes": num_episodes,
        "eval_epsilon": epsilon,
        "hidden_reset": hidden_reset,
        "return_mean": float(np.mean(rewards)),
        "return_std": float(np.std(rewards)),
        "returns": rewards,
        "length_mean": float(np.mean(lengths)),
        "q_abs_mean": q_mag_sum / max(q_mag_n, 1),
        "action_dist": {int(a): c for a, c in sorted(action_counts.items())},
        "achievement_rates": {
            a: achievement_counts[a] / num_episodes
            for a in sorted(achievement_counts)
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="crafter")
    parser.add_argument("--experiment_name", type=str, default="recurrent_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online_rl", action="store_true")
    parser.add_argument("--offline_rl", action="store_true")
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--awbc", action="store_true")
    parser.add_argument("--intervention", action="store_true")
    parser.add_argument("--num_rl_frames", type=int, default=2000)
    parser.add_argument("--num_unified_epochs", type=int, default=20)
    parser.add_argument("--total_iterations", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Parallel env workers for RL collection (crafter only, ignored with --intervention)")
    parser.add_argument(
        "--preload_expert_data", type=str, default="expert_demonstrations_crafter.pkl"
    )
    # --- Diagnostic probes (see run_recurrent_handsfree.sh decision guide) ---
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Evaluation episodes per iteration")
    parser.add_argument("--eval_epsilon", type=float, default=0.0,
                        help="Eval-time epsilon-greedy noise; 0 = deterministic argmax")
    parser.add_argument("--eval_hidden_reset", type=int, default=0,
                        help="Zero the LSTM state every N eval steps; 0 = never")
    parser.add_argument("--expert_fraction", type=float, default=1.0,
                        help="Train on a random fraction of expert episodes")
    parser.add_argument("--zero_state", action="store_true",
                        help="Skip refresh_hidden_states: train demo windows from zero hiddens")
    parser.add_argument("--bc_epsilon", type=float, default=-1.0,
                        help="Fixed BC target entropy epsilon; <0 = coupled to exploration epsilon")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load a checkpoint, run eval diagnostics, and exit (no training)")
    parser.add_argument("--load_checkpoint", type=str, default="",
                        help="Checkpoint path for --eval_only, or to warm-start training weights")
    parser.add_argument("--r2d3", action="store_true",
                        help="R2D3 mode: expert demos are plain TD replay (no CQL) sampled "
                             "1/16 epochs; overrides CQL even if --offline_rl is set")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from RCQL_latest.pt + metrics_latest.json in this run's "
                             "results dir; raise --total_iterations to train further "
                             "(online replay buffer restarts empty)")
    parser.add_argument("--n_step", type=int, default=1,
                        help="n-step TD returns for ONLINE batches. Uncorrected n-step is "
                             "biased by epsilon-greedy/stale replay; >1 sped up early "
                             "learning but plateaued low in crafter")
    parser.add_argument("--n_step_expert", type=int, default=None,
                        help="n-step TD returns for EXPERT batches (default: same as "
                             "--n_step). Demo returns carry no exploration noise, so "
                             "large n is safe and propagates expert reward fast")
    parser.add_argument("--cql_alpha", type=float, default=1.0,
                        help="CQL anchor weight at the start of training")
    parser.add_argument("--cql_alpha_end", type=float, default=None,
                        help="CQL weight after annealing; default = no annealing")
    parser.add_argument("--cql_alpha_decay_frames", type=int, default=500000,
                        help="RL frames over which cql_alpha anneals to cql_alpha_end")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use awbc in folder name if passed, as it implies a different training mode
    hparam_str = f"rcql_on{int(args.online_rl)}_off{int(args.offline_rl)}_bc{int(args.bc or args.awbc)}_aw{int(args.awbc)}{'_r2d3' if args.r2d3 else ''}_seed{args.seed}"
    results_base_dir = os.path.join(
        "results", args.env, args.experiment_name, hparam_str
    )
    os.makedirs(results_base_dir, exist_ok=True)

    # 1. Setup Environment
    if args.env == "crafter":
        env = CrafterGymnasiumWrapper()
        obs_dim = (3, 64, 64)
        action_dim = 17
    else:
        env = gym.make(args.env, render_mode="rgb_array")
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n

    agent = RCQLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        name="RCQL",
        save_dir=results_base_dir,
        device_name=device,
    )

    # Eval-only probe: no buffers, no training — evaluate a checkpoint and exit.
    if args.eval_only:
        if args.load_checkpoint:
            agent.load_model(args.load_checkpoint)
            print(f"[EvalProbe] Loaded checkpoint: {args.load_checkpoint}")
        stats = run_eval(
            agent,
            env,
            num_episodes=args.eval_episodes,
            epsilon=args.eval_epsilon,
            hidden_reset=args.eval_hidden_reset,
        )
        print(f"\n[EvalProbe] eps={args.eval_epsilon} hidden_reset={args.eval_hidden_reset} "
              f"({args.eval_episodes} episodes)")
        print(f"    Return: {stats['return_mean']:.2f} +/- {stats['return_std']:.2f}")
        print(f"    Ep Length: {stats['length_mean']:.0f} | Eval |Q|: {stats['q_abs_mean']:.3f}")
        print(f"    Action Distribution: {stats['action_dist']}")
        print(f"    Achievement Rates: {stats['achievement_rates']}")
        probe_name = f"eval_probe_eps{args.eval_epsilon}_hr{args.eval_hidden_reset}.json"
        with open(os.path.join(results_base_dir, probe_name), "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[EvalProbe] Saved to {os.path.join(results_base_dir, probe_name)}")
        env.close()
        return

    # 2. Setup Fast GPU Buffers
    # Total transitions: 200,000 (~2.4 GB)
    buffers = {
        "expert": FastGPUEpisodicBuffer(
            max_total_transitions=50000, device=device, obs_shape=obs_dim
        ),
        "online": FastGPUEpisodicBuffer(
            max_total_transitions=150000, device=device, obs_shape=obs_dim
        ),
        "llm": LLMBuffer(),
        "curriculum": CurriculumBuffer(),
        "ssl": SemiSupervisedBuffer(capacity=5000),
        "kl_target": ObservationBuffer(capacity=10000),
    }

    metrics = MetricsLogger()

    start_iteration = 0
    if args.resume:
        ckpt_path = os.path.join(results_base_dir, "RCQL_latest.pt")
        metrics_path = os.path.join(results_base_dir, "metrics_latest.json")
        if os.path.exists(ckpt_path):
            agent.load_model(ckpt_path)
            if metrics.load_from_json(metrics_path) and metrics.evaluations:
                start_iteration = metrics.evaluations[-1]["iteration"] + 1
                # pre_load re-logs these below; don't double count across resumes
                metrics.frames["expert_preload"] = 0
            print(
                f"[Resume] Loaded {ckpt_path}; continuing at iteration {start_iteration} "
                f"of {args.total_iterations} (online buffer restarts empty)."
            )
        else:
            print(f"[Resume] No checkpoint at {ckpt_path}; starting fresh.")
    elif args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
        print(f"[Init] Warm-started weights from {args.load_checkpoint}")

    if start_iteration >= args.total_iterations:
        print(
            f"[Resume] Nothing to do: iteration {start_iteration} >= "
            f"--total_iterations {args.total_iterations}."
        )
        env.close()
        return

    pre_load_episodic_data(args, buffers, metrics)

    router = LLMRouter(buffers["curriculum"], buffers["ssl"], env_name=args.env)

    # Parallel collection workers (env.step dominates collection time; see parallel_envs.py).
    # Intervention mode needs the serial path for trajectory replay.
    penvs = None
    if args.env == "crafter" and args.num_envs > 1 and not args.intervention and args.num_rl_frames > 0:
        from parallel_envs import ParallelCrafterEnvs
        penvs = ParallelCrafterEnvs(args.num_envs)

    for iteration in range(start_iteration, args.total_iterations):
        print(f"\n=== Iteration {iteration} ===")

        # Epsilon gate: configs that collect online experience need exploration and
        # use the decay schedule (0.25 -> 0.05 over the first 50k RL frames).
        # Offline-only configs (BC / offline RL) never explore, so epsilon only sets
        # the BC target entropy: keep it fixed and sharp.
        if args.online_rl and args.num_rl_frames > 0:
            total_rl_frames_collected = iteration * args.num_rl_frames
            decay_fraction = min(1.0, total_rl_frames_collected / 50000.0)
            agent.epsilon = 0.25 - decay_fraction * (0.25 - 0.05)
        else:
            agent.epsilon = 0.02
        print(f"    Current Epsilon: {agent.epsilon:.4f}")

        # CQL anchor annealing: strong imitation prior early, free policy late.
        # Driven by cumulative RL frames so it survives --resume.
        if args.cql_alpha_end is not None:
            alpha_frac = min(
                1.0, metrics.frames["rl"] / max(args.cql_alpha_decay_frames, 1)
            )
            agent.cql_alpha = args.cql_alpha + alpha_frac * (
                args.cql_alpha_end - args.cql_alpha
            )
        else:
            agent.cql_alpha = args.cql_alpha
        if args.offline_rl:
            print(f"    Current CQL alpha: {agent.cql_alpha:.4f}")

        episodes = []
        if args.num_rl_frames > 0:
            # If online_rl is enabled and we don't have enough episodes yet,
            # ensure we collect at least 8 to start training (also refills the
            # empty online buffer on the first iteration after a resume).
            min_episodes = 8 if args.online_rl and iteration == start_iteration else 0
            if penvs is not None:
                episodes = run_rl_collection_parallel(
                    agent,
                    penvs,
                    buffers,
                    num_frames=args.num_rl_frames,
                    metrics=metrics,
                    min_episodes=min_episodes,
                )
            else:
                episodes = run_rl_collection(
                    agent,
                    env,
                    buffers,
                    num_frames=args.num_rl_frames,
                    metrics=metrics,
                    min_episodes=min_episodes,
                )

        if args.intervention and len(episodes) > 0:
            print("Starting Interactive Review...")
            summary_ep = min(episodes, key=lambda x: x["total_reward"])
            hydrated = hydrate_trajectory(
                env, summary_ep["seed"], summary_ep["trajectory"]
            )

            wrapper = InteractiveGymWrapper(
                env,
                agent=agent,
                buffers=buffers,
                metrics=metrics,
                initial_trajectory=hydrated,
                initial_seed=summary_ep["seed"],
                env_name=args.env,
            )
            corrected_trajectory, annotations, _ = wrapper.run()

            human_episode = []
            for i in range(len(corrected_trajectory) - 1):
                s, ns = corrected_trajectory[i], corrected_trajectory[i + 1]
                if ns.get("action") is not None:
                    human_episode.append(
                        {
                            "obs": s["obs"],
                            "action": ns["action"],
                            "reward": ns.get("reward", 0.0), #* 10.0,
                            "next_obs": ns["obs"],
                            "terminated": ns.get("terminated", False),
                            "truncated": ns.get("truncated", False),
                        }
                    )
            if human_episode:
                buffers["expert"].add_episode(human_episode)

        # Train
        unified_train_step(args, agent, buffers, metrics)

        # Eval
        print("Evaluating...")
        eval_stats = run_eval(
            agent,
            env,
            num_episodes=args.eval_episodes,
            epsilon=args.eval_epsilon,
            hidden_reset=args.eval_hidden_reset,
        )
        mean_ret = eval_stats["return_mean"]
        std_ret = eval_stats["return_std"]

        bc_loss = 0.0
        if buffers["expert"].current_size > 0:
            v_obs, v_acts, _, _, v_masks, v_hidden = buffers["expert"].sample_batch(
                min(buffers["expert"].current_size, 16), seq_len=48
            )
            bc_loss = agent.get_bc_loss(v_obs, v_acts, v_masks, init_hidden=v_hidden, burn_in=16)

        print(f"    Eval Return: {mean_ret:.2f}")
        print(f"    Validation BC Loss: {bc_loss:.4f}")
        print(f"    Ep Length: {eval_stats['length_mean']:.0f} | Eval |Q|: {eval_stats['q_abs_mean']:.3f}")
        print(f"    Action Distribution: {eval_stats['action_dist']}")
        print(f"    Achievement Rates: {eval_stats['achievement_rates']}")

        metrics.log_evaluation(
            iteration,
            mean_ret,
            std_ret,
            bc_loss,
            length_mean=eval_stats["length_mean"],
            q_abs_mean=eval_stats["q_abs_mean"],
            action_dist=eval_stats["action_dist"],
            achievement_rates=eval_stats["achievement_rates"],
        )
        metrics.log_iteration()
        metrics.save_to_json(os.path.join(results_base_dir, "metrics_latest.json"))
        metrics.save_to_json(
            os.path.join(results_base_dir, f"metrics_{iteration}.json")
        )
        agent.checkpoint_model()

    if penvs is not None:
        penvs.close()
    env.close()


if __name__ == "__main__":
    main()
