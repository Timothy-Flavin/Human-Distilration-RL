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
from buffers import ReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer, ObservationBuffer, FastGPUEpisodicBuffer
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
        #if seed is not None:
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
        with open(args.preload_expert_data, 'rb') as f:
            expert_dataset = pickle.load(f)
        
        loaded_count = 0
        total_duration = 0.0
        
        for item in expert_dataset:
            # Handle both [transitions, ...] and [{'transitions': transitions}, ...]
            if isinstance(item, dict) and 'transitions' in item:
                transitions = item['transitions']
                duration = item.get('duration', len(transitions) / 30.0)
            else:
                transitions = item
                duration = len(transitions) / 30.0
                
            buffers['expert'].add_episode(transitions)
            loaded_count += len(transitions)
            total_duration += duration
        
        print(f"[Preload] Successfully loaded {len(expert_dataset)} episodes ({loaded_count} transitions) into GPU buffer.")
        metrics.log_frames(loaded_count, source="expert_preload")
        metrics.timers["expert_preload_effort"] = total_duration
    elif args.preload_expert_data:
        print(f"[Preload] Warning: Expert data file not found at {args.preload_expert_data}")

def run_rl_collection(agent, env, buffers, num_frames, metrics, min_episodes=0):
    """Collects episodic experience into the fast online buffer."""
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0
    
    current_episodes = buffers['online'].current_size
    effective_min = min_episodes if current_episodes < min_episodes else 0
    
    if effective_min > 0:
        print(f"Collecting RL experience (Target: {num_frames} frames OR {effective_min} new episodes)...")
    else:
        print(f"Collecting {num_frames} RL frames...")

    while total_frames < num_frames or len(episodes) < effective_min:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        agent.reset_hidden()
        terminated = False; truncated = False
        episode_transitions = []
        trajectory_lite = []
        total_reward = 0
        
        # Initial dummy state for trajectory tracking
        trajectory_lite.append({
            "obs": obs, "action": 0, "reward": 0, "next_obs": obs,
            "frame_image": None, "terminated": False, "truncated": False, 
            "env_state": None, "source": "rl"
        })

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_transitions.append({
                'obs': obs, 'action': action, 'reward': reward,
                'next_obs': next_obs, 'terminated': terminated, 'truncated': truncated,
                'info': info 
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
            # We don't break early here to ensure full episodes are always recorded
            # if total_frames >= num_frames: break 
            
        # Push to the fast GPU buffer
        if len(episode_transitions) > 0:
            buffers['online'].add_episode(episode_transitions)
            episodes.append({"seed": seed, "total_reward": total_reward, "trajectory": trajectory_lite})
        
        if total_frames >= num_frames and len(episodes) >= effective_min:
            break
        
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
    batch_size = 8
    seq_len = 48
    burn_in = 16
    
    has_online = args.online_rl and buffers['online'].current_size >= batch_size
    has_offline = (args.offline_rl or args.bc or args.awbc) and buffers['expert'].current_size >= batch_size
    
    if not (has_online or has_offline):
        print(f"[Training] Skipping updates: Not enough episodes (Online: {buffers['online'].current_size}, Offline: {buffers['expert'].current_size})")
        return

    print(f"Updating Recurrent Agent ({args.num_unified_epochs} epochs)...")
    
    for epoch in range(args.num_unified_epochs):
        
        # --- OFFLINE / BEHAVIOR CLONING UPDATES ---
        if has_offline:
            exp_obs, exp_acts, exp_rews, exp_dones, exp_masks = buffers['expert'].sample_batch(batch_size, seq_len=seq_len)
            
            if args.awbc:
                metrics.start_timer("agent_updating_value")
                agent.update_value(exp_obs, exp_acts, exp_rews, exp_dones, exp_masks, burn_in=burn_in)
                metrics.stop_timer("agent_updating_value")
                
                metrics.start_timer("agent_updating_bc")
                # Calculate Advantage using Value Network: A = r + gamma*V(s') - V(s)
                with torch.no_grad():
                    v_s_full, _ = agent.v_net(exp_obs[:, burn_in:, :])
                    v_s = v_s_full[:, :-1, :].squeeze(-1)
                    v_ns = v_s_full[:, 1:, :].squeeze(-1)
                    
                    r_active = exp_rews[:, burn_in:]
                    d_active = exp_dones[:, burn_in:]
                    td_error = r_active + (1.0 - d_active) * agent.gamma * v_ns - v_s
                    advantages = F.relu(td_error + 1.0)
                
                agent.update_supervised(exp_obs, exp_acts, exp_masks, burn_in=burn_in, advantages=advantages)
                metrics.stop_timer("agent_updating_bc")
            
            elif args.bc:
                metrics.start_timer("agent_updating_bc")
                agent.update_supervised(exp_obs, exp_acts, exp_masks, burn_in=burn_in)
                metrics.stop_timer("agent_updating_bc")
                
            if args.offline_rl:
                metrics.start_timer("agent_updating_rl")
                agent.update_td(exp_obs, exp_acts, exp_rews, exp_dones, exp_masks, burn_in=burn_in)
                metrics.stop_timer("agent_updating_rl")

        # --- ONLINE RL UPDATES ---
        if has_online:
            on_obs, on_acts, on_rews, on_dones, on_masks = buffers['online'].sample_batch(batch_size, seq_len=seq_len)
            
            metrics.start_timer("agent_updating_rl")
            agent.update_td(on_obs, on_acts, on_rews, on_dones, on_masks, burn_in=burn_in)
            metrics.stop_timer("agent_updating_rl")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparam_str = f"rcql_on{int(args.online_rl)}_off{int(args.offline_rl)}_bc{int(args.bc)}_seed{args.seed}"
    results_base_dir = os.path.join("results", args.env, args.experiment_name, hparam_str)
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

    agent = RCQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="RCQL", save_dir=results_base_dir, device_name=device)

    # 2. Setup Fast GPU Buffers
    # Crafter maximum episode limit is typically 2000-2500 steps. 2600 is safe.
    buffers = {
        'expert': FastGPUEpisodicBuffer(max_episodes=50, max_ep_len=2600, device=device, obs_shape=obs_dim), 
        'online': FastGPUEpisodicBuffer(max_episodes=200, max_ep_len=2600, device=device, obs_shape=obs_dim),
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000),
        'kl_target': ObservationBuffer(capacity=10000)
    }
    
    metrics = MetricsLogger()
    pre_load_episodic_data(args, buffers, metrics)
    
    router = LLMRouter(buffers['curriculum'], buffers['ssl'], env_name=args.env)
    
    TOTAL_ITERATIONS = 20
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Iteration {iteration} ===")
        
        episodes = []
        if args.num_rl_frames > 0:
            # If online_rl is enabled and we don't have enough episodes yet, 
            # ensure we collect at least 8 to start training.
            min_episodes = 8 if args.online_rl and iteration == 0 else 0
            episodes = run_rl_collection(agent, env, buffers, num_frames=args.num_rl_frames, metrics=metrics, min_episodes=min_episodes)
        
        if args.intervention and len(episodes) > 0:
            print("Starting Interactive Review...")
            summary_ep = min(episodes, key=lambda x: x['total_reward'])
            hydrated = hydrate_trajectory(env, summary_ep['seed'], summary_ep['trajectory'])
            
            wrapper = InteractiveGymWrapper(
                env, agent=agent, buffers=buffers, metrics=metrics,
                initial_trajectory=hydrated, initial_seed=summary_ep['seed'], env_name=args.env
            )
            corrected_trajectory, annotations, _ = wrapper.run()
            
            human_episode = []
            for i in range(len(corrected_trajectory) - 1):
                s, ns = corrected_trajectory[i], corrected_trajectory[i+1]
                if ns.get('action') is not None:
                    human_episode.append({
                        'obs': s['obs'], 'action': ns['action'], 'reward': ns.get('reward', 0.0),
                        'next_obs': ns['obs'], 'terminated': ns.get('terminated', False),
                        'truncated': ns.get('truncated', False)
                    })
            if human_episode:
                buffers['expert'].add_episode(human_episode)

        # Train 
        unified_train_step(args, agent, buffers, metrics)

        # Eval
        print("Evaluating...")
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
        
        bc_loss = 0.0
        if buffers['expert'].current_size > 0:
            v_obs, v_acts, _, _, v_masks = buffers['expert'].sample_batch(min(buffers['expert'].current_size, 16), seq_len=48)
            bc_loss = agent.get_bc_loss(v_obs, v_acts, v_masks, burn_in=16)

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