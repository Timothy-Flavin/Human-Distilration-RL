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
from buffers import ReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer, ObservationBuffer
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
        
        for item in expert_dataset:
            if isinstance(item, dict):
                transitions = item['transitions']
                total_duration += item.get('duration', len(transitions) / 30.0)
            else:
                transitions = item
                total_duration += len(transitions) / 30.0
                
            for t in transitions:
                # Push full transitions to example buffer (Expert Source)
                buffers['example'].push(
                    t['obs'], t['action'], reward=t['reward'], 
                    next_obs=t['next_obs'], terminated=t['terminated'], 
                    truncated=t['truncated'], mask=None
                )
                loaded_count += 1
        
        print(f"[Preload] Successfully loaded {loaded_count} transitions into example_buffer.")
        metrics.log_frames(loaded_count, source="expert_preload")
        metrics.timers["expert_preload_effort"] = total_duration
    elif args.preload_expert_data:
        print(f"[Preload] Warning: Expert data file not found at {args.preload_expert_data}")

def run_rl_collection(agent, env, num_frames, metrics, update=False):
    """Collects experience using the current policy. Lightweight (no frames)."""
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0
    while total_frames < num_frames:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        terminated = False; truncated = False
        trajectory_lite = []
        total_reward = 0
        
        trajectory_lite.append({
            "obs": obs, "action": 0, "reward": 0, "next_obs": obs,
            "frame_image": None, "terminated": False, "truncated": False, 
            "env_state": None, "source": "rl"
        })

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store in agent's internal buffer for Online RL (exploration source)
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)
            
            if update:
                metrics.start_timer("agent_updating_rl")
                agent.update_td([(obs, action, reward, next_obs, terminated, truncated)])
                metrics.stop_timer("agent_updating_rl")
            
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
            
        episodes.append({"seed": seed, "total_reward": total_reward, "trajectory": trajectory_lite})
    metrics.stop_timer("rl_experience")
    return episodes

def hydrate_trajectory(env, seed, trajectory_lite):
    """Re-simulates an episode to generate frames for review."""
    print(f"[Hydration] Re-simulating episode for review (Seed: {seed})...")
    env.reset(seed=seed)
    trajectory_lite[0]["frame_image"] = env.render()
    for i in range(1, len(trajectory_lite)):
        action = trajectory_lite[i]["action"]
        env.step(action)
        trajectory_lite[i]["frame_image"] = env.render()
    return trajectory_lite

def unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy):
    """Refactored core training loop: Decouples Loss Signals from Data Sources."""
    print(f"Updating Agent (Unified Pipeline, {args.num_unified_epochs} epochs)...")
    
    for epoch in range(args.num_unified_epochs):
        # --- 0. VALUE FUNCTION UPDATE (Independent Signal) ---
        # Train V on both buffers to get a representative baseline
        all_batch = []
        if len(agent.replay_buffer) >= 16:
            all_batch.extend(random.sample(agent.replay_buffer, 16))
        if len(buffers['example']) >= 16:
            # We sample transitions from the expert buffer too
            batch_items = buffers['example'].sample(16)
            all_batch.extend([(item['obs'], item['action'], item['reward'], item['next_obs'], item['terminated'], item['truncated']) for item in batch_items])
            
        if len(all_batch) >= 16:
            metrics.start_timer("agent_updating_rl")
            agent.update_value(all_batch)
            metrics.stop_timer("agent_updating_rl")

        # --- 1. LOSS SIGNAL: TEMPORAL DIFFERENCE (RL) ---
        
        # Source: Online RL (Exploration Data)
        if args.online_rl and len(agent.replay_buffer) >= 32:
            metrics.start_timer("agent_updating_rl")
            batch = random.sample(agent.replay_buffer, 32)
            agent.update_td(batch, ssl=False) 
            metrics.stop_timer("agent_updating_rl")

        # Source: Offline RL (Human/Expert Data)
        if args.offline_rl and len(buffers['example']) >= 32:
            metrics.start_timer("agent_updating_rl")
            batch_items = buffers['example'].sample(32)
            batch = [(item['obs'], item['action'], item['reward'], item['next_obs'], item['terminated'], item['truncated']) for item in batch_items]
            masks = [item.get('mask') for item in batch_items] if args.ssl else None
            agent.update_td(batch, ssl=args.ssl, masks=masks)
            metrics.stop_timer("agent_updating_rl")

        # --- 2. LOSS SIGNAL: SUPERVISED (BC) ---

        if (args.bc or args.awbc) and len(buffers['example']) >= 32:
            metrics.start_timer("agent_updating_bc")
            batch_items = buffers['example'].sample(32)
            batch = [(item['obs'], item['action']) for item in batch_items]
            masks = [item.get('mask') for item in batch_items] if args.ssl else None
            
            advantages = None
            if args.awbc:
                # Calculate Advantage using TD Error: A = r + gamma*V(s') - V(s)
                obs_t = torch.stack([item['obs'] for item in batch_items]).to(agent.device_name)
                next_obs_t = torch.stack([item['next_obs'] for item in batch_items]).to(agent.device_name)
                rewards_t = torch.tensor([item['reward'] for item in batch_items], dtype=torch.float32).to(agent.device_name)
                dones_t = torch.tensor([item['terminated'] or item['truncated'] for item in batch_items], dtype=torch.float32).to(agent.device_name)
                
                with torch.no_grad():
                    v_s = agent.get_value(obs_t)
                    v_ns = agent.get_value(next_obs_t)
                    td_error = rewards_t + (1 - dones_t) * agent.gamma * v_ns - v_s
                    # Weight w = exp(Adv / temp). Using relu as a stable proxy.
                    advantages = F.relu(td_error + 1.0)
            
            agent.update_supervised(batch, ssl=args.ssl, masks=masks, advantages=advantages)
            metrics.stop_timer("agent_updating_bc")

        # --- 3. LOSS SIGNAL: ANTI-BC ---
        if args.anti_bc and len(buffers['anti_example']) >= 32:
            metrics.start_timer("agent_updating_anti_bc")
            batch_items = buffers['anti_example'].sample(32)
            batch = [(item['obs'], item['action']) for item in batch_items]
            agent.update_supervised(batch, anti=True)
            metrics.stop_timer("agent_updating_anti_bc")

        # --- 4. LOSS SIGNAL: CURRICULUM/KL ---
        if args.curriculum and args.curriculum_method == 'kl' and len(buffers['kl_target']) >= 32:
            kl_obs = buffers['kl_target'].sample(32)
            agent.kl_update(kl_obs, aux_agent)

def main():
    global args
    # 1. Setup Environment
    env_name = args.env
    hparam_str = f"{args.algo}_on{int(args.online_rl)}_off{int(args.offline_rl)}_bc{int(args.bc)}_aw{int(args.awbc)}_ssl{int(args.ssl)}_seed{args.seed}"
    results_base_dir = os.path.join("results", env_name, args.experiment_name, hparam_str)
    os.makedirs(results_base_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    if "highway" in env_name:
        import highway_env
        env = gym.wrappers.FlattenObservation(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL", save_dir=results_base_dir, device_name="cpu")

    aux_agent = None
    if args.curriculum and args.curriculum_method in ['separate', 'kl']:
        aux_agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL_Aux", save_dir=results_base_dir, device_name="cpu")

    # 2. Setup Buffers & Telemetry
    buffers = {
        'example': ReplayBuffer(capacity=50000), 
        'anti_example': ReplayBuffer(capacity=10000),
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000),
        'kl_target': ObservationBuffer(capacity=10000)
    }
    
    metrics = MetricsLogger()
    pre_load_data(args, agent, buffers, metrics)
    
    global_buffer_proxy = MagicReplayProxy(agent) 
    router = LLMRouter(
        buffers['curriculum'], buffers['ssl'], 
        global_buffer=global_buffer_proxy, example_buffer=buffers['example'], 
        metrics=metrics, noise_scale=args.noise_scale, num_noisy_samples=args.num_noisy_samples,
        env_name=env_name
    )
    
    TOTAL_ITERATIONS = 20
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        active_heuristics = []
        
        # PHASE 1: ACQUISITION
        episodes = []
        if args.num_rl_frames > 0:
            print(f"Collecting {args.num_rl_frames} RL frames...")
            episodes = run_rl_collection(agent, env, num_frames=args.num_rl_frames, metrics=metrics, update=args.online_rl)
        
        # PHASE 2: INTERACTION
        annotations = []
        if args.intervention and len(episodes) > 0:
            print("Starting Human Interactive Review...")
            summary_ep = min(episodes, key=lambda x: x['total_reward'])
            hydrated_trajectory = hydrate_trajectory(env, summary_ep['seed'], summary_ep['trajectory'])
            
            wrapper = InteractiveGymWrapper(
                env, agent=agent, buffers=buffers, metrics=metrics,
                initial_trajectory=hydrated_trajectory, initial_seed=summary_ep['seed'], env_name=env_name
            )
            corrected_trajectory, annotations, final_seed = wrapper.run()
            
            # Commit human frames to example buffer (Source for offline RL and BC)
            for i in range(len(corrected_trajectory) - 1):
                s, ns = corrected_trajectory[i], corrected_trajectory[i+1]
                buffers['example'].push(
                    s['obs'], ns['action'], reward=ns['reward'], 
                    next_obs=ns['obs'], terminated=ns['terminated'], 
                    truncated=ns['truncated'], mask=None
                )

            # Intent Processing
            print("Processing LLM Buffer...")
            metrics.start_timer("llm_processing")
            v_manager = VerificationManager(env, agent, buffers, metrics)
            while not buffers['llm'].is_empty():
                item = buffers['llm'].pop()
                classification = router.classify(item)
                if classification['type'] == 'HEURISTIC':
                    v_manager.verify_heuristic(item, classification, router)
                    active_heuristics.append(classification)
                else:
                    router.commit(item, classification)
            metrics.stop_timer("llm_processing")
            
        # PHASE 3: CURRICULUM
        if len(buffers['curriculum']) > 0 and args.curriculum:
            metrics.start_timer("agent_updating_local_rl")
            print(f"[Curriculum] Replaying tasks (Method: {args.curriculum_method})...")
            while not buffers['curriculum'].is_empty():
                task = buffers['curriculum'].pop()
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

        # PHASE 4: UNIFIED UPDATE
        unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy)

        # PHASE 5: EVALUATION
        print("Evaluating Agent...")
        mean_ret, std_ret = evaluate_return(agent, env_name, num_episodes=5)
        bc_loss = calculate_cross_entropy(agent, buffers['example'], anti=False) if (args.bc or args.awbc) else None
        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss)
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
    args = parser.parse_args()
    main()
