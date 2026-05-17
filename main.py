import gymnasium as gym
import torch
import numpy as np
import random
import os
import json 

from Agent import Agent
from CQL import CQLAgent
from PPO import PPOAgent
from wrapper import InteractiveGymWrapper
from buffers import ReplayBuffer, LLMBuffer, CurriculumBuffer, SemiSupervisedBuffer, ObservationBuffer
from metrics import MetricsLogger
from llm_router import LLMRouter
from eval_agent import evaluate_return, calculate_cross_entropy
from verification_manager import VerificationManager

torch.set_num_threads(4)

def run_rl_collection(agent, env, num_frames, metrics, update=False):
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0
    while total_frames < num_frames:
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
            total_frames += 1
            if total_frames >= num_frames:
                break
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
    
    # For curriculum method 'separate' or 'kl'
    aux_agent = None
    if args.curriculum and args.curriculum_method in ['separate', 'kl']:
        if args.algo == "cql":
            aux_agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL_Aux", save_dir=results_base_dir, device_name="cpu")
        elif args.algo == "ppo":
            aux_agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, name="PPO_Aux", save_dir=results_base_dir, device_name="cpu")

    # 2. Setup Buffers & Router
    buffers = {
        'example': ReplayBuffer(capacity=10000),
        'anti_example': ReplayBuffer(capacity=10000),
        'llm': LLMBuffer(),
        'curriculum': CurriculumBuffer(),
        'ssl': SemiSupervisedBuffer(capacity=5000),
        'kl_target': ObservationBuffer(capacity=10000)
    }
    
    metrics = MetricsLogger()
    # R4.2: Router needs access to global buffers for mining
    global_buffer_proxy = MagicReplayProxy(agent) 
    router = LLMRouter(buffers['curriculum'], buffers['ssl'], global_buffer=global_buffer_proxy, example_buffer=buffers['example'], metrics=metrics)
    
    TOTAL_ITERATIONS = 20
    
    for iteration in range(TOTAL_ITERATIONS):
        print(f"\n=== Starting Iteration {iteration} ===")
        
        # Track heuristics for dynamic mining
        active_heuristics = []
        
        # Step 1: Base RL Collection
        print("Collecting RL experience...")
        n_frames = 5000 if args.algo == "ppo" else 2000
        episodes = run_rl_collection(agent, env, num_frames=n_frames, metrics=metrics, update=args.rl)
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
            
            # Step 3: LLM Routing & Verification
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
                    # Goals and Generics are committed immediately
                    router.commit(item, classification)
            metrics.stop_timer("llm_processing")
            
            # Step 4: Curriculum Updates (Localized RL)
            # Must happen before Unified Update to prepare aux_agent and kl_target buffer
            if len(buffers['curriculum']) > 0 and args.curriculum:
                metrics.start_timer("agent_updating_local_rl")
                print(f"[Curriculum] Replaying tasks (Method: {args.curriculum_method})...")
                
                while not buffers['curriculum'].is_empty():
                    task = buffers['curriculum'].pop()
                    
                    if args.curriculum_method in ['separate', 'kl']:
                        # Copy main agent to aux agent
                        if args.algo == "ppo":
                            aux_agent.actor.load_state_dict(agent.actor.state_dict())
                            aux_agent.critic.load_state_dict(agent.critic.state_dict())
                            aux_agent.actor_old.load_state_dict(agent.actor.state_dict())
                        else:
                            aux_agent.q_net.load_state_dict(agent.q_net.state_dict())
                            aux_agent.q_target.load_state_dict(agent.q_net.state_dict())

                    train_agent = aux_agent if args.curriculum_method in ['separate', 'kl'] else agent
                    
                    for local_epoch in range(args.num_local_epochs):
                        obs, info = env.reset(seed=task['seed'])
                        if task['historical_actions']:
                            for action in task['historical_actions']:
                                obs, _, term, trunc, _ = env.step(action)
                                if term or trunc: break
                        
                        n_frames = 0
                        traj_len = args.curriculum_traj_len if args.curriculum_traj_len > 0 else task.get('trajectory_length', 100)
                        for _ in range(traj_len):
                            action = train_agent.predict(obs, deterministic=False)
                            next_obs, reward, term, trunc, info = env.step(action)
                            
                            local_reward = task['reward_fn'](obs, next_obs, reward) if task.get('reward_fn') else reward
                            
                            if args.curriculum_method == 'main':
                                # Train main agent on both
                                agent.store_transition(obs, action, reward, next_obs, term, trunc) # True rewards
                                agent.store_local_transition(obs, action, local_reward, next_obs, term, trunc) # Aux rewards
                                agent.rl_update(local=True)
                                agent.rl_update(local=False)
                            elif args.curriculum_method == 'separate':
                                # Train aux agent on local
                                aux_agent.store_transition(obs, action, local_reward, next_obs, term, trunc)
                                aux_agent.rl_update()
                                # Expose behavior to main agent with true rewards
                                agent.store_transition(obs, action, reward, next_obs, term, trunc)
                            elif args.curriculum_method == 'kl':
                                # Train aux agent on local
                                aux_agent.store_transition(obs, action, local_reward, next_obs, term, trunc)
                                aux_agent.rl_update()
                                # Store observations for targeted KL
                                buffers['kl_target'].push(obs)
                            
                            env.render()
                            n_frames += 1
                            if term or trunc: break
                            obs = next_obs
                        metrics.log_frames(n_frames, source="curriculum")
                
                metrics.stop_timer("agent_updating_local_rl")

            # Step 5: Multi-Faceted Update (R5.1: Unified Update Epochs)
            print("Updating Agent (Unified Pipeline)...")
            num_unified_epochs = 5
            for epoch in range(num_unified_epochs):
                # Standard RL update (global buffer)
                if args.rl:
                    agent.rl_update()

                # Targeted KL penalty (only on curriculum observations)
                if args.curriculum and args.curriculum_method == 'kl' and len(buffers['kl_target']) >= 32:
                    kl_obs = buffers['kl_target'].sample(32)
                    agent.kl_update(kl_obs, aux_agent)

                # Supervised BC
                if len(buffers['example']) >= 32 and args.bc:
                    metrics.start_timer("agent_updating_bc")
                    obs, labels = buffers['example'].sample(32)
                    
                    advantages = None
                    if args.awbc:
                        # Estimate advantage using current critic/Q-net
                        with torch.no_grad():
                            if args.algo == "ppo":
                                values = agent.critic(obs.to(agent.device_name)).squeeze()
                                advantages = torch.ones_like(labels, dtype=torch.float32)
                            else:
                                q_values = agent.q_net(obs.to(agent.device_name))
                                v_values = q_values.max(1)[0]
                                q_selected = q_values.gather(1, labels.to(agent.device_name).unsqueeze(1)).squeeze()
                                advantages = F.relu(q_selected - v_values + 1.0)
                    
                    agent.supervised_update(obs, labels, anti=False, advantages=advantages)
                    metrics.stop_timer("agent_updating_bc")
                    
                # Supervised Anti-BC
                if len(buffers['anti_example']) >= 32 and args.anti_bc:
                    metrics.start_timer("agent_updating_anti_bc")
                    obs, labels = buffers['anti_example'].sample(32)
                    agent.supervised_update(obs, labels, anti=True)
                    metrics.stop_timer("agent_updating_anti_bc")
                
                # --- Dynamic SSL Mining (Temporary "borrow" frames) ---
                if args.ssl and active_heuristics:
                    metrics.start_timer("agent_updating_ssl")
                    mining_batch = []
                    
                    # 1. Start with pristine verified data from the buffer
                    if len(buffers['ssl']) > 0:
                        mining_batch.extend(buffers['ssl'].sample(8))
                        
                    # 2. Temporarily borrow matching frames from the global buffer
                    # We mine a small amount per epoch to keep updates fast and diverse
                    for h in active_heuristics:
                        rule = h.get('rule')
                        if rule and global_buffer_proxy.buffer:
                            # Sample some candidates from global buffer
                            candidates = random.sample(global_buffer_proxy.buffer, min(len(global_buffer_proxy.buffer), 100))
                            for obs, _ in candidates:
                                obs_np = obs.numpy() if hasattr(obs, 'numpy') else obs
                                if rule(obs_np):
                                    target_action = h['action_fn'](obs_np) if h.get('action_fn') else h['action']
                                    if target_action is not None:
                                        mining_batch.append({
                                            "obs": obs,
                                            "action": target_action,
                                            "feature_mask": h['feature_mask'],
                                            "termination_rule": h.get('termination_rule')
                                        })
                                if len(mining_batch) >= 16: break
                        if len(mining_batch) >= 16: break
                    
                    if len(mining_batch) >= 8:
                        agent.ssl_update(mining_batch[:16])
                    metrics.stop_timer("agent_updating_ssl")

            # Save buffers and annotations
            buffers['example'].save(os.path.join(results_base_dir, f"example_buffer_{iteration}.pt"))
            buffers['anti_example'].save(os.path.join(results_base_dir, f"anti_example_buffer_{iteration}.pt"))
            
            # Save raw annotations for traceability
            with open(os.path.join(results_base_dir, f"annotations_{iteration}.json"), "w") as f:
                json.dump(annotations, f, indent=4)
                
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
        # CQL uses a list of tuples: (obs, action, reward, next_obs, terminated, truncated)
        if hasattr(self.agent, 'replay_buffer'):
            return [(step[0], step[1]) for step in self.agent.replay_buffer]
        # PPO uses a list of dicts: {'obs': ..., 'action': ..., ...}
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
    parser.add_argument("--awbc", action="store_true", help="Use Advantage Weighted Behavior Cloning")
    parser.add_argument("--curriculum_method", type=str, default="main", choices=["main", "separate", "kl"])
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    parser.add_argument("--num_local_epochs", type=int, default=5)
    parser.add_argument("--curriculum_traj_len", type=int, default=0)
    args = parser.parse_args()
    main()
