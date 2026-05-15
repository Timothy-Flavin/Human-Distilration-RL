import gymnasium as gym
import torch
import numpy as np
import random
import os

from Agent import Agent
from CQL import CQLAgent
from PPO import PPOAgent
from wrapper import InteractiveGymWrapper
from buffers import ReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer
from metrics import MetricsLogger
from llm_router import LLMRouter
from eval_agent import evaluate_return, calculate_cross_entropy

def run_rl_collection(agent, env, num_episodes, metrics, update=False):
    metrics.start_timer("rl_experience")
    episodes = []
    for _ in range(num_episodes):
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        trajectory = []
        
        # Record initial state
        frame = env.render()
        trajectory.append({
            "obs": obs,
            "action": 0,
            "reward": 0,
            "frame_image": frame,
            "terminated": False,
            "truncated": False,
            "env_state": None
        })

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            
            # Store for standard RL
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)
            
            # Perform standard RL update
            if update:
                agent.rl_update()
            
            trajectory.append({
                "obs": next_obs,
                "action": action,
                "reward": reward,
                "frame_image": frame,
                "terminated": terminated,
                "truncated": truncated,
                "env_state": None
            })
            obs = next_obs
            metrics.log_frames(1, source="rl")
        episodes.append({"trajectory": trajectory, "seed": seed})
    metrics.stop_timer("rl_experience")
    return episodes

def main():
    global args
    # 1. Setup Environment & Agent
    env_name = "LunarLander-v3"
    
    # Organized result structure: ./results/{algorithm}/{environment}/{experiment_name}/
    results_base_dir = os.path.join("results", args.algo, env_name, args.experiment_name)
    os.makedirs(results_base_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if args.algo == "cql":
        agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL", save_dir=results_base_dir, device_name="cpu")
    elif args.algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, name="PPO", save_dir=results_base_dir, device_name="cpu")
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # 2. Setup Buffers & Router
    buffers = {
        'example': ReplayBuffer(capacity=10000),
        'anti_example': ReplayBuffer(capacity=10000),
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000)
    }
    
    metrics = MetricsLogger()
    # R4.2: Router needs access to global buffers for mining
    # For CQL, we use its internal replay_buffer. For PPO, we'd need to expose its buffer or use the episodes.
    global_buffer_proxy = MagicReplayProxy(agent) 
    router = LLMRouter(buffers['curriculum'], buffers['ssl'], global_buffer=global_buffer_proxy, example_buffer=buffers['example'])
    
    TOTAL_ITERATIONS = 20
    
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        
        # Step 1: Base RL Collection
        print("Collecting RL experience...")
        nep = 20 if args.algo == "ppo" else 5
        episodes = run_rl_collection(agent, env, num_episodes=nep, metrics=metrics, update=args.rl)
        agent.checkpoint_model(specific_name=f"rl_collection_{iteration}")
        
        # Step 2: Human Interactive Review
        if args.bc or args.anti_bc or args.ssl or args.curriculum:
            print("Starting Human Interactive Review...")
            
            # Sample the episode with the lowest return for review
            best_ep = min(episodes, key=lambda x: sum(step['reward'] for step in x['trajectory']))
            
            wrapper = InteractiveGymWrapper(
                env, 
                agent=agent, 
                buffers=buffers, 
                metrics=metrics,
                initial_trajectory=best_ep['trajectory'],
                initial_seed=best_ep['seed']
            )
            corrected_trajectory, annotations, final_seed = wrapper.run()
            totr = 0
            #print(corrected_trajectory)
            for ti in range(len(corrected_trajectory)):
                totr+=corrected_trajectory[ti]['reward']
            print(f"Total reward for corrected trajectory: {totr}")
            # Push corrected trajectory to RL buffer
            if args.algo != "ppo":
                for i in range(len(corrected_trajectory) - 1):
                    step = corrected_trajectory[i]
                    next_step = corrected_trajectory[i+1]
                    agent.store_transition(
                        step['obs'], next_step['action'], next_step['reward'], 
                        next_step['obs'], next_step['terminated'], next_step['truncated']
                    )

            agent.checkpoint_model(specific_name=f"interactive_review_{iteration}")
            
            # Step 3: LLM Routing
            print("Processing LLM Buffer...")
            metrics.start_timer("llm_processing")
            while not buffers['llm'].is_empty():
                item = buffers['llm'].pop()
                router.process(item)
            metrics.stop_timer("llm_processing")
            
            # Step 4: Multi-Faceted Update (R5.1: Unified Update Epochs)
            print("Updating Agent (Unified Pipeline)...")
            num_unified_epochs = 5
            for epoch in range(num_unified_epochs):
                # Supervised BC
                if len(buffers['example']) >= 32 and args.bc:
                    metrics.start_timer("agent_updating_bc")
                    obs, labels = buffers['example'].sample(32)
                    agent.supervised_update(obs, labels, anti=False)
                    metrics.stop_timer("agent_updating_bc")
                    
                # Supervised Anti-BC
                if len(buffers['anti_example']) >= 32 and args.anti_bc:
                    metrics.start_timer("agent_updating_anti_bc")
                    obs, labels = buffers['anti_example'].sample(32)
                    agent.supervised_update(obs, labels, anti=True)
                    metrics.stop_timer("agent_updating_anti_bc")
                
                # SSL Updates
                if len(buffers['ssl']) >= 8 and args.ssl:
                    metrics.start_timer("agent_updating_ssl")
                    batch = buffers['ssl'].sample(8)
                    agent.ssl_update(batch)
                    metrics.log_frames(len(batch), source="ssl")
                    metrics.stop_timer("agent_updating_ssl")

            # Curriculum Updates (Localized RL)
            # Curriculum is special because it involves environment interaction
            if len(buffers['curriculum']) > 0 and args.curriculum:
                metrics.start_timer("agent_updating_local_rl")
                print(f"[Curriculum] Replaying tasks...")
                # We consume curriculum tasks once per iteration but can do multiple local epochs
                while not buffers['curriculum'].is_empty():
                    task = buffers['curriculum'].pop()
                    for local_epoch in range(args.num_local_epochs):
                        obs, info = env.reset(seed=task['seed'])
                        if task['historical_actions']:
                            for action in task['historical_actions']:
                                obs, _, term, trunc, _ = env.step(action)
                                if term or trunc: break
                        
                        n_frames = 0
                        traj_len = args.curriculum_traj_len if args.curriculum_traj_len > 0 else task.get('trajectory_length', 100)
                        for _ in range(traj_len):
                            action = agent.predict(obs, deterministic=False)
                            next_obs, reward, term, trunc, info = env.step(action)
                            agent.store_transition(obs, action, reward, next_obs, term, trunc)
                            
                            local_reward = task['reward_fn'](obs, next_obs, reward) if task.get('reward_fn') else reward
                            if hasattr(agent, 'store_local_transition'):
                                agent.store_local_transition(obs, action, local_reward, next_obs, term, trunc)
                            
                            env.render()
                            agent.rl_update(local=True)
                            n_frames += 1
                            if term or trunc: break
                            obs = next_obs
                        metrics.log_frames(n_frames, source="curriculum")
                metrics.stop_timer("agent_updating_local_rl")

            # Save buffers
            buffers['example'].save(os.path.join(results_base_dir, f"example_buffer_{iteration}.pt"))
            buffers['anti_example'].save(os.path.join(results_base_dir, f"anti_example_buffer_{iteration}.pt"))
                
        agent.checkpoint_model(specific_name=f"agent_update_{iteration}")
        
        # --- Evaluation ---
        print("Evaluating Agent...")
        mean_ret, std_ret = evaluate_return(agent, env_name, num_episodes=5)
        bc_loss = calculate_cross_entropy(agent, buffers['example'], anti=False) if args.bc else None
        anti_bc_loss = calculate_cross_entropy(agent, buffers['anti_example'], anti=True) if args.anti_bc else None
        
        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss, anti_bc_loss)
        
        # Step 5: Log Telemetry
        metrics.log_iteration()
        metrics.save_to_json(os.path.join(results_base_dir, f"metrics_{iteration}.json"))
        metrics.save_to_json(os.path.join(results_base_dir, "metrics_latest.json"))
        
    env.close()

class MagicReplayProxy:
    """Helper to expose agent's internal buffer to the router for mining."""
    def __init__(self, agent):
        self.agent = agent
    @property
    def buffer(self):
        if hasattr(self.agent, 'replay_buffer'):
            return [(step[0], step[1]) for step in self.agent.replay_buffer]
        elif hasattr(self.agent, 'buffer'):
            return [(step['obs'], step['action']) for step in self.agent.buffer]
        return []

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--anti_bc", action="store_true")
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    parser.add_argument("--num_local_epochs", type=int, default=5)
    parser.add_argument("--curriculum_traj_len", type=int, default=0)
    args = parser.parse_args()
    main()
