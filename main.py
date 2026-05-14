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
    router = LLMRouter(buffers['curriculum'], buffers['ssl'])
    
    TOTAL_ITERATIONS = 20
    
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        
        # Step 1: Base RL Collection
        print("Collecting RL experience...")
        if args.algo == "ppo":
            nep = 20
        else:
            nep = 5
        episodes = run_rl_collection(agent, env, num_episodes=nep, metrics=metrics, update=args.rl)
        agent.checkpoint_model(specific_name=f"rl_collection_{iteration}")
        
        # Step 2: Human Interactive Review
        if args.bc or args.anti_bc or args.semi_supervised or args.curriculum:
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
            
            # --- NEW: Push the FULL corrected trajectory to the RL Replay Buffer ---
            # Note: RL agent actions from interactive window go to RL buffer here, 
            # while human actions only went to BC buffer during the interactive session.
            if args.algo != "ppo":
                for i in range(len(corrected_trajectory) - 1):
                    step = corrected_trajectory[i]
                    next_step = corrected_trajectory[i+1]
                    agent.store_transition(
                        step['obs'], 
                        next_step['action'], 
                        next_step['reward'], 
                        next_step['obs'], 
                        next_step['terminated'],
                        next_step['truncated']
                    )

            agent.checkpoint_model(specific_name=f"interactive_review_{iteration}")
            
            # Step 3: LLM Routing
            print("Processing LLM Buffer...")
            metrics.start_timer("llm_processing")
            while not buffers['llm'].is_empty():
                item = buffers['llm'].pop()
                router.process(item)
            metrics.stop_timer("llm_processing")
            agent.checkpoint_model(specific_name=f"llm_routing_{iteration}")
            
            # Step 4: Multi-Faceted Agent Update
            print("Updating Agent...")
            
            # Supervised Updates
            if len(buffers['example']) >= 32 and args.bc:
                metrics.start_timer("agent_updating_bc")
                for si in range(10):
                    obs, labels = buffers['example'].sample(32)
                    agent.supervised_update(obs, labels, anti=False)
                metrics.stop_timer("agent_updating_bc")
                
            if len(buffers['anti_example']) >= 32 and args.anti_bc:
                metrics.start_timer("agent_updating_anti_bc")
                for si in range(10):
                    obs, labels = buffers['anti_example'].sample(32)
                    agent.supervised_update(obs, labels, anti=True)
                metrics.stop_timer("agent_updating_anti_bc")
                
            # Curriculum Updates (Local RL)
            if len(buffers['curriculum']) > 0 and args.curriculum:
                metrics.start_timer("agent_updating_local_rl")
                print(f"\n[Curriculum] Processing tasks...")
                
                while not buffers['curriculum'].is_empty():
                    task = buffers['curriculum'].pop()
                    print(f" > Task: Seed {task['seed']}, Start Frame {task['start_frame']}")
                    
                    # Training Loop: Replay the targeted segment multiple times
                    num_local_epochs = 5
                    for epoch in range(num_local_epochs):
                        # 1. Restore Environment to the starting frame of the task
                        # Use historical actions for perfect reconstruction
                        obs, info = env.reset(seed=task['seed'])
                        if task['historical_actions']:
                            for action in task['historical_actions']:
                                obs, _, terminated, truncated, _ = env.step(action)
                                if terminated or truncated: break
                        
                        # 2. Collect Experience from this point and update
                        n_frames = 0
                        for _ in range(task.get('trajectory_length', 100)):
                            action = agent.predict(obs, deterministic=False)
                            next_obs, reward, terminated, truncated, info = env.step(action)
                            
                            # Standard reward for the GLOBAL buffer (so it learns standard env physics)
                            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)
                            
                            # Custom reward for the LOCAL buffer (so it learns the curriculum task)
                            local_reward = reward
                            if task.get('reward_fn'):
                                local_reward = task['reward_fn'](obs, next_obs, reward)
                            
                            if hasattr(agent, 'store_local_transition'):
                                agent.store_local_transition(obs, action, local_reward, next_obs, terminated, truncated)
                            else:
                                # Fallback if agent doesn't support local buffers yet
                                agent.store_transition(obs, action, local_reward, next_obs, terminated, truncated)
                            
                            env.render() # Visual verification
                            
                            # 3. Perform a localized RL update PER FRAME
                            agent.rl_update(local=True)
                            
                            n_frames += 1
                            if terminated or truncated: break
                            obs = next_obs
                        
                        metrics.log_frames(n_frames, source="curriculum")
                        
                metrics.stop_timer("agent_updating_local_rl")
                
            # SSL Updates
            if len(buffers['ssl']) >= 8 and args.semi_supervised:
                metrics.start_timer("agent_updating_ssl")
                batch = buffers['ssl'].sample(8)
                agent.ssl_update(batch)
                metrics.stop_timer("agent_updating_ssl")

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Boolean flags (default = False)
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--anti_bc", action="store_true")
    parser.add_argument("--semi_supervised", action="store_true")
    parser.add_argument("--curriculum", action="store_true")

    # String argument
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])

    args = parser.parse_args()

    print(args)
    main()
