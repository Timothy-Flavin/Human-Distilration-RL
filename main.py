import gymnasium as gym
import torch
import numpy as np
import random
import os
import json 
import torch.nn.functional as F
import time
import collections

from Agent import Agent
from CQL import CQLAgent
from wrapper import InteractiveGymWrapper
from buffers import ReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer, ObservationBuffer, DenseTorchBuffer
from metrics import MetricsLogger
from llm_router import LLMRouter
from eval_agent import evaluate_return, calculate_cross_entropy
from verification_manager import VerificationManager

torch.set_num_threads(4)

# --- Sub-functions for modularity ---

def pre_load_data(args, agent, buffers, metrics):
    """Loads expert demonstrations and populates the unified example buffer."""
    if args.preload_expert_data and os.path.exists(args.preload_expert_data):
        import pickle
        with open(args.preload_expert_data, 'rb') as f:
            expert_dataset = pickle.load(f)
        
        loaded_count = 0
        total_duration = 0.0
        all_transitions = []
        
        for item in expert_dataset:
            if isinstance(item, dict):
                transitions = item['transitions']
                total_duration += item.get('duration', len(transitions) / 30.0)
            else:
                transitions = item
                total_duration += len(transitions) / 30.0
                
            all_transitions.extend(transitions)
            for t in transitions:
                # Push full transitions to example buffer (Expert Source)
                buffers['example'].push(
                    t['obs'], t['action'], reward=t['reward'], 
                    next_obs=t['next_obs'], terminated=t['terminated'], 
                    truncated=t['truncated'], mask=None
                )
                loaded_count += 1
        
        if 'dense_example' in buffers:
            print(f"[Preload] Flushing {len(all_transitions)} transitions to DenseTorchBuffer...")
            buffers['dense_example'].add_transitions(all_transitions)
            # Ensure sync_dense_buffers doesn't re-sync these
            if not hasattr(sync_dense_buffers, "synced_counts"):
                sync_dense_buffers.synced_counts = collections.defaultdict(int)
            sync_dense_buffers.synced_counts['example'] = len(buffers['example'])
            
        print(f"[Preload] Successfully loaded {loaded_count} transitions into example_buffer.")
        metrics.log_frames(loaded_count, source="expert_preload")
        metrics.timers["expert_preload_effort"] = total_duration
    elif args.preload_expert_data:
        print(f"[Preload] Warning: Expert data file not found at {args.preload_expert_data}")

def run_rl_collection(agent, collection_agent, env, cpu_buffer, num_frames, metrics, update=False):
    """
    Collects experience using a lightweight CPU agent.
    Writes directly to a CPU DenseTorchBuffer for fast bulk GPU transfer later.
    """
    metrics.start_timer("rl_experience")
    
    # Sync weights from GPU to CPU (Param Copy Only)
    if collection_agent is not agent:
        collection_agent.sync_from(agent)

    cpu_buffer.ptr = 0 # Reset CPU buffer for this collection phase
    cpu_buffer.size = 0

    episodes = []
    total_frames = 0
    
    all_new_transitions = []

    while total_frames < num_frames:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        terminated = False; truncated = False
        trajectory_lite = []
        total_reward = 0
        
        while not (terminated or truncated):
            action = collection_agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect in list for bulk addition
            all_new_transitions.append({
                'obs': obs, 'action': action, 'reward': reward, 
                'next_obs': next_obs, 'terminated': terminated, 'truncated': truncated
            })
            
            trajectory_lite.append({
                "obs": obs, "action": action, "reward": reward, "next_obs": next_obs,
                "frame_image": None, "terminated": terminated, "truncated": truncated, 
                "env_state": None, "source": "rl"
            })
            total_reward += reward
            obs = next_obs
            metrics.log_frames(1, source="rl")
            total_frames += 1
            if total_frames >= num_frames: break
        
        # Add the final state for trajectory visualization
        trajectory_lite.append({
            "obs": obs, "action": 0, "reward": 0, "next_obs": None,
            "frame_image": None, "terminated": terminated, "truncated": truncated, 
            "env_state": None, "source": "rl"
        })
            
        episodes.append({"seed": seed, "total_reward": total_reward, "trajectory": trajectory_lite})

    # Bulk add to CPU buffer
    if all_new_transitions:
        cpu_buffer.add_transitions(all_new_transitions)

    metrics.stop_timer("rl_experience")
    return episodes

def hydrate_trajectory(env, seed, trajectory_lite):
    """Re-simulates an episode to generate frames for review."""
    print(f"[Hydration] Re-simulating episode for review (Seed: {seed})...")
    env.reset(seed=seed)
    for i in range(len(trajectory_lite)):
        trajectory_lite[i]["frame_image"] = env.render()
        if i < len(trajectory_lite) - 1:
            action = trajectory_lite[i]["action"]
            env.step(action)
    return trajectory_lite

def unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy):
    """Refactored core training loop: Decouples Loss Signals from Data Sources."""
    print(f"Updating Agent (Unified Pipeline, {args.num_unified_epochs} epochs)...")
    
    # Use DenseTorchBuffers if available
    dense_online = buffers.get('dense_online')
    dense_example = buffers.get('dense_example')
    dense_anti = buffers.get('dense_anti_example')

    # Optimization: On GPU, larger batches are significantly more efficient
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 256 if agent.device_name == "cuda" else 32
    print(f"[Train] Using batch size {batch_size} for {agent.device_name} updates.")

    profiler = collections.defaultdict(float)

    for epoch in range(args.num_unified_epochs):
        t0 = time.perf_counter()
        
        # 1. Unified Sampling (Fetch all data needed for this epoch at once)
        online_batch = None
        if dense_online and len(dense_online) >= batch_size:
            online_batch = dense_online.sample(batch_size)
        
        expert_batch = None
        if dense_example and len(dense_example) >= batch_size:
            expert_batch = dense_example.sample(batch_size)
            
        if online_batch is None and expert_batch is None:
            continue
            
        # Fallback to avoid None types in the agent method
        if online_batch is None: online_batch = expert_batch
        if expert_batch is None: expert_batch = online_batch
        
        if agent.device_name == "cuda": torch.cuda.synchronize()
        profiler['sampling'] += time.perf_counter() - t0

        # 2. Unified Training Step (Structural Parallelization)
        t1 = time.perf_counter()
        metrics.start_timer("agent_updating_unified")
        
        # Combined forward/backward for all active signals
        agent.train_iteration(
            online_batch, expert_batch,
            awbc=args.awbc, bc=args.bc,
            online_rl=args.online_rl, offline_rl=args.offline_rl,
            anti_bc=args.anti_bc, ssl=args.ssl
        )
        
        metrics.stop_timer("agent_updating_unified")
        if agent.device_name == "cuda": torch.cuda.synchronize()
        profiler['update_unified'] += time.perf_counter() - t1

    print(f"[Profiling] Unified Train Step breakdown (Device: {agent.device_name}):")
    total_time = sum(profiler.values())
    if total_time > 0:
        for k, v in profiler.items():
            print(f"  - {k}: {v:.4f}s ({(v/total_time*100):.1f}%)")



        # --- 3. LOSS SIGNAL: ANTI-BC ---
        if args.anti_bc and dense_anti and len(dense_anti) >= batch_size:
            metrics.start_timer("agent_updating_anti_bc")
            obs, actions, rewards, next_obs, dones = dense_anti.sample(batch_size)
            agent.update_supervised(obs, actions, anti=True)
            metrics.stop_timer("agent_updating_anti_bc")

        # --- 4. LOSS SIGNAL: CURRICULUM/KL ---
        if args.curriculum and args.curriculum_method == 'kl' and len(buffers['kl_target']) >= batch_size:
            kl_obs = buffers['kl_target'].sample(batch_size)
            agent.kl_update(kl_obs, aux_agent)

def sync_dense_buffers(buffers):
    """Syncs Pythonic list-of-dicts buffers to DenseTorchBuffers for training."""
    # Use static variables to track synced counts across calls
    if not hasattr(sync_dense_buffers, "synced_counts"):
        sync_dense_buffers.synced_counts = collections.defaultdict(int)
    
    for key, dense_key in [('example', 'dense_example'), ('anti_example', 'dense_anti_example')]:
        if dense_key in buffers and len(buffers[key]) > sync_dense_buffers.synced_counts[key]:
            start_idx = sync_dense_buffers.synced_counts[key]
            import itertools
            new_transitions = list(itertools.islice(buffers[key].buffer, start_idx, len(buffers[key])))
            
            if len(new_transitions) > 0:
                print(f"[Sync] Adding {len(new_transitions)} new transitions to {dense_key}...")
                # Still using add_transitions for these as they come from Deques
                buffers[dense_key].add_transitions(new_transitions)
                sync_dense_buffers.synced_counts[key] = len(buffers[key])

def main():
    global args
    # 1. Setup Environment
    env_name = args.env
    if env_name == "highway":
        env_name = "highway-v0"
    elif env_name == "football":
        env_name = "gfootball"

    hparam_str = f"{args.algo}_on{int(args.online_rl)}_off{int(args.offline_rl)}_bc{int(args.bc)}_aw{int(args.awbc)}_ssl{int(args.ssl)}_seed{args.seed}"
    results_base_dir = os.path.join("results", env_name, args.experiment_name, hparam_str)
    os.makedirs(results_base_dir, exist_ok=True)

    if "highway" in env_name:
        import highway_env
    elif "football" in env_name or "gfootball" in env_name:
        import gfootball

    if "football" in env_name or "gfootball" in env_name:
        from dizoo.gfootball.envs.gfootball_env import GfootballEnv
        from easydict import EasyDict
        import gfootball.env as football_env

        class CustomGfootballEnv(GfootballEnv):
            def _launch_env(self, gui=False):
                self._env = football_env.create_environment(
                    env_name=self._cfg.env_name,
                    stacked=False,
                    representation='raw',
                    number_of_left_players_agent_controls=0,
                    number_of_right_players_agent_controls=1,
                    logdir='./tmp/football',
                    write_goal_dumps=False,
                    write_full_episode_dumps=self.save_replay,
                    write_video=self.save_replay,
                    render=False
                )
                self._launch_env_flag = True

            def step(self, action):
                timestep = super().step(action)
                obs = timestep.obs['processed_obs']
                if isinstance(obs, list): obs = np.array(obs)
                if len(obs.shape) > 1: obs = obs[0]
                return obs, timestep.reward, timestep.done, False, timestep.info

            def reset(self, seed=None):
                if seed is not None: self.seed(seed)
                obs_dict = super().reset()
                obs = obs_dict['processed_obs']
                if isinstance(obs, list): obs = np.array(obs)
                if len(obs.shape) > 1: obs = obs[0]
                return obs, {}
            
            def render(self):
                return self._env.render(mode='rgb_array')

            def get_state(self):
                return self._env.get_state()
            
            def set_state(self, state):
                self._env.set_state(state)

        cfg = EasyDict({
            'env_name': '11_vs_11_stochastic',
            'save_replay': False,
            'save_replay_gif': False,
        })
        env = CustomGfootballEnv(cfg)
        obs_dim = 115 # Standard for simple115 or raw-processed
        # We try to detect it if possible
        try:
            test_obs, _ = env.reset()
            obs_dim = test_obs.shape[0]
        except: pass
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        if "highway" in env_name:
            env = gym.wrappers.FlattenObservation(env)
        obs_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.nvec[0]
    if "football" in env_name or "gfootball" in env_name:
        action_dim = 19 # Standard gfootball action set
    
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Main] Using device: {device}")
    agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL", save_dir=results_base_dir, device_name=device)

    # Optimization: Dedicated CPU Agent for collection to avoid GPU kernel overhead
    collection_agent = agent
    if device == "cuda":
        print("[Setup] Initializing dedicated CPU collection agent...")
        collection_agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL_CPU", save_dir=results_base_dir, device_name="cpu")

    aux_agent = None
    if args.curriculum and args.curriculum_method in ['separate', 'kl']:
        aux_agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL_Aux", save_dir=results_base_dir, device_name=device)

    # 2. Setup Buffers & Telemetry
    buffers = {
        'example': ReplayBuffer(capacity=50000), 
        'anti_example': ReplayBuffer(capacity=10000),
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000),
        'kl_target': ObservationBuffer(capacity=10000)
    }

    # Initialize Dense Tensors for training speed
    buffers['dense_online'] = DenseTorchBuffer(capacity=100000, obs_dim=obs_dim, device=agent.device_name)
    buffers['dense_online_cpu'] = DenseTorchBuffer(capacity=max(args.num_rl_frames, 10000), obs_dim=obs_dim, device="cpu")
    buffers['dense_example'] = DenseTorchBuffer(capacity=100000, obs_dim=obs_dim, device=agent.device_name)
    buffers['dense_anti_example'] = DenseTorchBuffer(capacity=10000, obs_dim=obs_dim, device=agent.device_name)

    metrics = MetricsLogger()
    pre_load_data(args, agent, buffers, metrics)
    
    global_buffer_proxy = MagicReplayProxy(agent) 
    router = LLMRouter(
        buffers['curriculum'], buffers['ssl'], 
        global_buffer=global_buffer_proxy, example_buffer=buffers['example'], 
        metrics=metrics, noise_scale=args.noise_scale, num_noisy_samples=args.num_noisy_samples,
        env_name=env_name
    )
    
    TOTAL_ITERATIONS = 10
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        active_heuristics = []
        
        # PHASE 1: ACQUISITION
        episodes = []
        if args.num_rl_frames > 0:
            print(f"Collecting {args.num_rl_frames} RL frames (via CPU agent)...")
            episodes = run_rl_collection(agent, collection_agent, env, buffers['dense_online_cpu'], num_frames=args.num_rl_frames, metrics=metrics, update=args.online_rl)
            
            # Vectorized transfer: Sync entire CPU collection buffer to GPU at once
            print(f"[Sync] Transferring {buffers['dense_online_cpu'].size} transitions to GPU...")
            buffers['dense_online'].add_from_buffer(buffers['dense_online_cpu'])
        
        # PHASE 2: INTERACTION
        annotations = []
        if args.intervention and len(episodes) > 0:
            print("Starting Human Interactive Review...")
            summary_ep = min(episodes, key=lambda x: x['total_reward'])
            hydrated_trajectory = hydrate_trajectory(env, summary_ep['seed'], summary_ep['trajectory'])
            
            wrapper = InteractiveGymWrapper(
                env, agent=agent, buffers=buffers, metrics=metrics,
                initial_trajectory=hydrated_trajectory, initial_seed=summary_ep['seed'], 
                env_name=env_name, is_curriculum=args.curriculum
            )
            corrected_trajectory, annotations, final_seed = wrapper.run()

            # Intent Processing
            print(f"Processing {len(buffers['llm'])} annotations from LLM Buffer...")
            metrics.start_timer("llm_processing")
            v_manager = VerificationManager(env, agent, buffers, metrics)
            processed_count = 0
            while not buffers['llm'].is_empty():
                item = buffers['llm'].pop()
                classification = router.classify(item)
                if classification['type'] == 'HEURISTIC':
                    v_manager.verify_heuristic(item, classification, router)
                    active_heuristics.append(classification)
                else:
                    router.commit(item, classification)
                processed_count += 1
            print(f"Processed {processed_count} annotations.")
            metrics.stop_timer("llm_processing")
            
        # PHASE 3: CURRICULUM
        if len(buffers['curriculum']) > 0:
            if args.curriculum:
                metrics.start_timer("agent_updating_local_rl")
                print(f"[Curriculum] Replaying {len(buffers['curriculum'])} tasks (Method: {args.curriculum_method})...")
                while not buffers['curriculum'].is_empty():
                    task = buffers['curriculum'].pop()
                    # ... replaying task logic ...
                    if args.curriculum_method in ['separate', 'kl']:
                        aux_agent.q_net.load_state_dict(agent.q_net.state_dict())
                        aux_agent.q_target.load_state_dict(agent.q_net.state_dict())
                    train_agent = aux_agent if args.curriculum_method in ['separate', 'kl'] else agent
                    for local_epoch in range(args.num_local_epochs):
                        obs, info = env.reset(seed=task['seed'])
                        if task['historical_actions']:
                            for action in task['historical_actions']:
                                obs, _, term, trunc, _ = env.step(action); 
                                if term or trunc: break
                        n_frames = 0
                        traj_len = args.curriculum_traj_len if args.curriculum_traj_len > 0 else task.get('trajectory_length', 100)
                        for _ in range(traj_len):
                            action = train_agent.predict(obs, deterministic=False)
                            next_obs, reward, term, trunc, info = env.step(action)
                            local_reward = task['reward_fn'](obs, next_obs, reward) if task.get('reward_fn') else reward
                            if args.curriculum_method == 'main':
                                agent.update_td([(obs, action, reward, next_obs, term, trunc)])
                                agent.update_td([(obs, action, local_reward, next_obs, term, trunc)])
                            elif args.curriculum_method in ['separate', 'kl']:
                                train_agent.update_td([(obs, action, local_reward, next_obs, term, trunc)])
                                if args.curriculum_method == 'kl': buffers['kl_target'].push(obs)
                                else: buffers['example'].push(obs, action, reward=reward, next_obs=next_obs, terminated=term, truncated=trunc)
                            n_frames += 1
                            if term or trunc: break
                            obs = next_obs
                        metrics.log_frames(n_frames, source="curriculum")
                metrics.stop_timer("agent_updating_local_rl")
            else:
                print(f"[Curriculum] Warning: {len(buffers['curriculum'])} tasks in buffer but --curriculum flag is not set. Skipping.")
                # We don't pop them so they stay for when the user might enable it? 
                # Actually, in this loop, it's better to clear them or leave them. 
                # If we leave them, they'll accumulate. Let's just warn for now.

        # PHASE 4: UNIFIED UPDATE
        sync_dense_buffers(buffers)
        unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy)

        # --- PHASE 5: EVALUATION ---
        print("Evaluating Agent...")
        mean_ret, std_ret, compliance_score = evaluate_return(agent, env_name, num_episodes=5)
        # Always calculate BC loss for Human Likeness tracking if buffer is not empty
        bc_loss = calculate_cross_entropy(agent, buffers['example'], anti=False) if len(buffers['example']) > 0 else 0.0
        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss, compliance_score=compliance_score)

        metrics.log_iteration()
        metrics.save_to_json(os.path.join(results_base_dir, "metrics_latest.json"))
        metrics.save_to_json(os.path.join(results_base_dir, f"metrics_{iteration}.json"))
        agent.checkpoint_model()
        
    env.close()

class MagicReplayProxy:
    def __init__(self, agent):
        self.agent = agent
    @property
    def buffer(self):
        return [(step[0], step[1]) for step in self.agent.replay_buffer]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", type=str, default="cql")
    parser.add_argument("--online_rl", action="store_true")
    parser.add_argument("--offline_rl", action="store_true")
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--awbc", action="store_true")
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--intervention", action="store_true")
    parser.add_argument("--anti_bc", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_method", type=str, default="main", choices=["main", "separate", "kl"])
    parser.add_argument("--num_rl_frames", type=int, default=2000)
    parser.add_argument("--num_unified_epochs", type=int, default=50)
    parser.add_argument("--num_local_epochs", type=int, default=5)
    parser.add_argument("--curriculum_traj_len", type=int, default=0)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--num_noisy_samples", type=int, default=5)
    parser.add_argument("--preload_expert_data", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    main()
