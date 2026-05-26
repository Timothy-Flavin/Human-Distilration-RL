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
from buffers import ReplayBuffer, EpisodicReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer, ObservationBuffer
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
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self):
        # Upscale for better visibility if needed (e.g., in interactive mode)
        # But for hydration/storage, standard 64x64 is fine.
        # wrapper.py will call this.
        return self._env.render(size=(512, 512))

    def close(self):
        pass

# --- Sub-functions ---

def pre_load_episodic_data(args, agent, buffers, metrics):
    """Loads expert episodic demonstrations."""
    if args.preload_expert_data and os.path.exists(args.preload_expert_data):
        with open(args.preload_expert_data, 'rb') as f:
            expert_dataset = pickle.load(f)
        
        loaded_count = 0
        total_duration = 0.0
        
        for item in expert_dataset:
            # item is already a dict with 'transitions' and 'duration'
            buffers['example'].push(item)
            loaded_count += len(item['transitions'])
            total_duration += item.get('duration', len(item['transitions']) / 30.0)
        
        print(f"[Preload] Successfully loaded {len(expert_dataset)} episodes ({loaded_count} transitions) into example_buffer.")
        metrics.log_frames(loaded_count, source="expert_preload")
        metrics.timers["expert_preload_effort"] = total_duration

def run_rl_collection(agent, env, num_frames, metrics):
    """Collects episodic experience using the RCQL agent."""
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0
    while total_frames < num_frames:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        agent.reset_hidden()
        terminated = False; truncated = False
        episode_transitions = []
        trajectory_lite = []
        total_reward = 0
        
        # Initial trajectory step
        trajectory_lite.append({
            "obs": obs, "action": 0, "reward": 0, "next_obs": obs,
            "frame_image": None, "terminated": False, "truncated": False, 
            "env_state": None, "source": "rl"
        })

        while not (terminated or truncated):
            # RCQL predict handles tensor conversion and hidden states
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Transition for RCQL Episodic Replay
            episode_transitions.append({
                'obs': obs, 'action': action, 'reward': reward,
                'next_obs': next_obs, 'terminated': terminated, 'truncated': truncated,
                'info': info # Crucial for achievement tracking
            })
            
            # For Interactive Review
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
            
        # Store full episode in agent's buffer
        agent.store_episode({'transitions': episode_transitions})
        episodes.append({"seed": seed, "total_reward": total_reward, "trajectory": trajectory_lite})
        
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

def unified_train_step(args, agent, aux_agent, buffers, metrics):
    """Episodic RCQL training loop."""
    # Use a dynamic batch size based on what's available
    batch_size = 8
    
    # Check if we have enough data to train at all
    has_online = args.online_rl and len(agent.replay_buffer) >= batch_size
    has_offline = (args.offline_rl or args.bc or args.awbc) and len(buffers['example']) >= batch_size
    
    if not (has_online or has_offline):
        print(f"[Training] Skipping updates: Not enough episodes (Online: {len(agent.replay_buffer)}, Offline: {len(buffers['example'])})")
        return

    print(f"Updating Recurrent Agent ({args.num_unified_epochs} epochs)...")
    
    for epoch in range(args.num_unified_epochs):
        # 0. Value Update (AWBC)
        if args.awbc:
            batch = []
            if len(agent.replay_buffer) >= batch_size:
                batch.extend(random.sample(list(agent.replay_buffer), batch_size))
            if len(buffers['example']) >= batch_size:
                batch.extend(buffers['example'].sample(batch_size))
            if len(batch) >= batch_size:
                metrics.start_timer("agent_updating_value")
                agent.update_value(batch)
                metrics.stop_timer("agent_updating_value")

        # 1. TD Updates (RL)
        if args.online_rl and len(agent.replay_buffer) >= batch_size:
            metrics.start_timer("agent_updating_rl")
            batch = random.sample(list(agent.replay_buffer), batch_size)
            agent.update_td(batch)
            metrics.stop_timer("agent_updating_rl")

        if args.offline_rl and len(buffers['example']) >= batch_size:
            metrics.start_timer("agent_updating_rl")
            batch = buffers['example'].sample(batch_size)
            agent.update_td(batch)
            metrics.stop_timer("agent_updating_rl")

        # 2. Supervised Updates (BC)
        if (args.bc or args.awbc) and len(buffers['example']) >= batch_size:
            metrics.start_timer("agent_updating_bc")
            batch = buffers['example'].sample(batch_size)
            agent.update_supervised(batch)
            metrics.stop_timer("agent_updating_bc")
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{args.num_unified_epochs} complete...")


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
    parser.add_argument("--preload_expert_data", type=str, default="expert_demonstrations_crafter.pkl")
    args = parser.parse_args()

    hparam_str = f"rcql_on{int(args.online_rl)}_off{int(args.offline_rl)}_bc{int(args.bc)}_seed{args.seed}"
    results_base_dir = os.path.join("results", args.env, args.experiment_name, hparam_str)
    os.makedirs(results_base_dir, exist_ok=True)

    # 1. Setup Environment
    if args.env == "crafter":
        env = CrafterGymnasiumWrapper()
        obs_dim = (3, 64, 64)
        action_dim = 17
    else:
        # Fallback for simple tests
        env = gym.make(args.env, render_mode="rgb_array")
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n

    agent = RCQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="RCQL", save_dir=results_base_dir, device_name="cpu")

    # 2. Setup Episodic Buffers
    buffers = {
        'example': EpisodicReplayBuffer(capacity=1000), 
        'anti_example': EpisodicReplayBuffer(capacity=1000), # Standard is fine for anti-examples (pointwise rejection)
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000),
        'kl_target': ObservationBuffer(capacity=10000)
    }
    
    metrics = MetricsLogger()
    pre_load_episodic_data(args, agent, buffers, metrics)
    
    # Router (Legacy support, though Crafter might need new heuristics)
    router = LLMRouter(buffers['curriculum'], buffers['ssl'], env_name=args.env)
    
    TOTAL_ITERATIONS = 20
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Iteration {iteration} ===")
        
        # PHASE 1: ACQUISITION
        episodes = []
        if args.num_rl_frames > 0:
            episodes = run_rl_collection(agent, env, num_frames=args.num_rl_frames, metrics=metrics)
        
        # PHASE 2: INTERACTION
        if args.intervention and len(episodes) > 0:
            print("Starting Interactive Review...")
            summary_ep = min(episodes, key=lambda x: x['total_reward'])
            hydrated = hydrate_trajectory(env, summary_ep['seed'], summary_ep['trajectory'])
            
            wrapper = InteractiveGymWrapper(
                env, agent=agent, buffers=buffers, metrics=metrics,
                initial_trajectory=hydrated, initial_seed=summary_ep['seed'], env_name=args.env
            )
            corrected_trajectory, annotations, _ = wrapper.run()
            
            # Convert trajectory to episode
            human_episode = {'transitions': []}
            for i in range(len(corrected_trajectory) - 1):
                s, ns = corrected_trajectory[i], corrected_trajectory[i+1]
                if ns.get('action') is not None:
                    human_episode['transitions'].append({
                        'obs': s['obs'], 'action': ns['action'], 'reward': ns.get('reward', 0.0),
                        'next_obs': ns['obs'], 'terminated': ns.get('terminated', False),
                        'truncated': ns.get('truncated', False)
                    })
            if human_episode['transitions']:
                buffers['example'].push(human_episode)

        # PHASE 3: UPDATES
        unified_train_step(args, agent, None, buffers, metrics)

        # PHASE 4: EVALUATION
        print("Evaluating...")
        # Custom evaluation loop to track action distribution
        eval_rewards = []
        action_counts = collections.defaultdict(int)
        for _ in range(5):
            e_obs, _ = env.reset()
            agent.reset_hidden()
            e_term = False; e_trunc = False; e_total = 0
            while not (e_term or e_trunc):
                e_act = agent.predict(e_obs, deterministic=True)
                action_counts[int(e_act)] += 1
                e_obs, e_rew, e_term, e_trunc, _ = env.step(e_act)
                e_total += e_rew
            eval_rewards.append(e_total)
        
        mean_ret = np.mean(eval_rewards)
        std_ret = np.std(eval_rewards)
        
        # Calculate BC validation loss
        bc_loss = 0.0
        if len(buffers['example']) > 0:
            val_batch = buffers['example'].sample(min(len(buffers['example']), 16))
            bc_loss = agent.get_bc_loss(val_batch)

        print(f"    Eval Return: {mean_ret:.2f}")
        print(f"    Validation BC Loss: {bc_loss:.4f}")
        print(f"    Action Distribution: {dict(action_counts)}")

        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss)
        metrics.log_iteration()
        metrics.save_to_json(os.path.join(results_base_dir, "metrics_latest.json"))
        metrics.save_to_json(os.path.join(results_base_dir, f"metrics_{iteration}.json"))
        agent.checkpoint_model()
        
    env.close()

if __name__ == "__main__":
    main()
