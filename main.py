import gymnasium as gym
import torch
import numpy as np
import random

from Agent import Agent
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
            action = agent.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            
            # Store for standard RL
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            
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
    env = gym.make(env_name, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = Agent(obs_dim=obs_dim, action_dim=action_dim, name="Agent", save_dir=f"./{env_name}", device_name="cpu")
    
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
    
    TOTAL_ITERATIONS = 10
    
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        
        # Step 1: Base RL Collection
        print("Collecting RL experience...")
        episodes = run_rl_collection(agent, env, num_episodes=5, metrics=metrics, update=args.rl)
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
            # This follows the requirement: "a full episode ... to add to the RL buffer as a new episode"
            for i in range(len(corrected_trajectory) - 1):
                step = corrected_trajectory[i]
                next_step = corrected_trajectory[i+1]
                agent.store_transition(
                    step['obs'], 
                    next_step['action'], 
                    next_step['reward'], 
                    next_step['obs'], 
                    next_step['terminated'] or next_step['truncated']
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
                obs, labels = buffers['example'].sample(32)
                agent.supervised_update(obs, labels, anti=False)
                metrics.stop_timer("agent_updating_bc")
                
            if len(buffers['anti_example']) >= 32 and args.anti_bc:
                metrics.start_timer("agent_updating_anti_bc")
                obs, labels = buffers['anti_example'].sample(32)
                agent.supervised_update(obs, labels, anti=True)
                metrics.stop_timer("agent_updating_anti_bc")
                
            # Curriculum Updates (Local RL)
            if len(buffers['curriculum']) > 0 and args.curriculum:
                metrics.start_timer("agent_updating_local_rl")
                for task in buffers['curriculum']:
                    metrics.log_frames(task.get('trajectory_length', 50), source="curriculum")
                    agent.rl_update(local=True)
                metrics.stop_timer("agent_updating_local_rl")
                
            # SSL Updates
            if len(buffers['ssl']) >= 8 and args.semi_supervised:
                metrics.start_timer("agent_updating_ssl")
                batch = buffers['ssl'].sample(8)
                agent.ssl_update(batch)
                metrics.stop_timer("agent_updating_ssl")

            # Save buffers for evaluation
            buffers['example'].save(f"./{env_name}/example_buffer_{iteration}.pt")
            buffers['anti_example'].save(f"./{env_name}/anti_example_buffer_{iteration}.pt")
                
        agent.checkpoint_model(specific_name=f"agent_update_{iteration}")
        
        # --- Evaluation ---
        print("Evaluating Agent...")
        mean_ret, std_ret = evaluate_return(agent, env_name, num_episodes=5)
        bc_loss = calculate_cross_entropy(agent, buffers['example'], anti=False) if args.bc else None
        anti_bc_loss = calculate_cross_entropy(agent, buffers['anti_example'], anti=True) if args.anti_bc else None
        
        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss, anti_bc_loss)
        
        # Step 5: Log Telemetry
        metrics.log_iteration()
        metrics.save_to_json(f"./{env_name}/{args.file_name}_metrics_{iteration}.json")
        metrics.save_to_json(f"./{env_name}/{args.file_name}_metrics_latest.json")
        
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
    parser.add_argument("--file_name", type=str, required=True)

    args = parser.parse_args()

    print(args)
    main()
