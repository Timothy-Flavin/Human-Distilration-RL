import gymnasium as gym
import torch
import numpy as np
import random
import os
import json 
import torch.nn.functional as F

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

def pre_load_data(args, agent, buffers, metrics):
    if args.preload_expert_data and os.path.exists(args.preload_expert_data):
        import pickle
        with open(args.preload_expert_data, 'rb') as f:
            expert_dataset = pickle.load(f)
        loaded_count = 0
        total_duration = 0.0
        
        for item in expert_dataset:
            # Handle both formats (lite list or dict with transitions/duration)
            if isinstance(item, dict):
                transitions = item['transitions']
                # Use the real 30 FPS from record_expert_data.py for legacy data
                total_duration += item.get('duration', len(transitions) / 30.0)
            else:
                transitions = item
                total_duration += len(transitions) / 30.0
                
            for transition in transitions:
                buffers['example'].push(transition['obs'], transition['action'])
                agent.store_transition(
                    transition['obs'], transition['action'], transition['reward'],
                    transition['next_obs'], transition['terminated'], transition['truncated']
                )
                loaded_count += 1
        
        print(f"[Preload] Successfully loaded {loaded_count} transitions into example_buffer AND agent.replay_buffer.")
        metrics.log_frames(loaded_count, source="expert_preload")
        # Use recorded duration if available, otherwise 15 FPS estimate
        metrics.timers["expert_preload_effort"] = total_duration
    elif args.preload_expert_data:
        print(f"[Preload] Warning: Expert data file not found at {args.preload_expert_data}")

def run_rl_collection(agent, env, num_frames, metrics, update=False):
    metrics.start_timer("rl_experience")
    episodes = []
    total_frames = 0
    while total_frames < num_frames:
        seed = np.random.randint(0, 1000000)
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        
        trajectory_lite = []
        total_reward = 0
        
        # Initial step
        trajectory_lite.append({
            "obs": obs, "action": 0, "reward": 0, "next_obs": obs,
            "frame_image": None, "terminated": False, "truncated": False, 
            "env_state": None, "source": "rl"
        })

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store for standard RL
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)
            
            if update:
                metrics.start_timer("agent_updating_rl")
                agent.rl_update()
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
            if total_frames >= num_frames:
                break
        
        episodes.append({
            "seed": seed,
            "total_reward": total_reward,
            "trajectory": trajectory_lite
        })
    metrics.stop_timer("rl_experience")
    return episodes

def hydrate_trajectory(env, seed, trajectory_lite):
    """Re-simulates an episode to add frames to an existing lite trajectory."""
    print(f"[Hydration] Re-simulating episode to generate frames (Seed: {seed})...")
    env.reset(seed=seed)
    
    # 1. Hydrate initial state
    trajectory_lite[0]["frame_image"] = env.render()
    
    # 2. Hydrate subsequent steps
    # We use the actions already stored in the lite trajectory
    for i in range(1, len(trajectory_lite)):
        action = trajectory_lite[i]["action"]
        env.step(action)
        trajectory_lite[i]["frame_image"] = env.render()
        
    return trajectory_lite

def unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy):
    print(f"Updating Agent (Unified Pipeline, {args.num_unified_epochs} epochs)...")
    
    for epoch in range(args.num_unified_epochs):
        # 1. Standard RL update (global buffer)
        if args.rl:
            metrics.start_timer("agent_updating_rl")
            agent.rl_update()
            metrics.stop_timer("agent_updating_rl")

        # 2. Targeted KL penalty (only on curriculum observations)
        if args.curriculum and args.curriculum_method == 'kl' and len(buffers['kl_target']) >= 32:
            kl_obs = buffers['kl_target'].sample(32)
            agent.kl_update(kl_obs, aux_agent)

        # 3. Supervised BC (Expert or Intervention data)
        if len(buffers['example']) >= 32 and args.bc:
            metrics.start_timer("agent_updating_bc")
            obs, labels = buffers['example'].sample(32)
            
            advantages = None
            if args.awbc:
                with torch.no_grad():
                    if args.algo == "ppo":
                        advantages = torch.ones_like(labels, dtype=torch.float32)
                    else:
                        q_values = agent.q_net(obs.to(agent.device_name))
                        v_values = q_values.max(1)[0]
                        q_selected = q_values.gather(1, labels.to(agent.device_name).unsqueeze(1)).squeeze()
                        advantages = F.relu(q_selected - v_values + 1.0)
            
            agent.supervised_update(obs, labels, anti=False, advantages=advantages)
            metrics.stop_timer("agent_updating_bc")
                
        # 4. Supervised Anti-BC
        if len(buffers['anti_example']) >= 32 and args.anti_bc:
            metrics.start_timer("agent_updating_anti_bc")
            obs, labels = buffers['anti_example'].sample(32)
            agent.supervised_update(obs, labels, anti=True)
            metrics.stop_timer("agent_updating_anti_bc")
        
        # 5. Dynamic SSL Mining
        if args.ssl and active_heuristics:
            metrics.start_timer("agent_updating_ssl")
            mining_batch = []
            if len(buffers['ssl']) > 0:
                mining_batch.extend(buffers['ssl'].sample(min(len(buffers['ssl']), 8)))
                
            for h in active_heuristics:
                rule = h.get('rule')
                if rule and global_buffer_proxy.buffer:
                    candidates = random.sample(global_buffer_proxy.buffer, min(len(global_buffer_proxy.buffer), 100))
                    for obs, _ in candidates:
                        obs_np = obs.numpy() if hasattr(obs, 'numpy') else obs
                        if rule(obs_np):
                            target_action = h['action_fn'](obs_np) if h.get('action_fn') else h['action']
                            if target_action is not None:
                                mining_batch.append({
                                    "obs": obs, "action": target_action,
                                    "feature_mask": h['feature_mask'],
                                    "termination_rule": h.get('termination_rule')
                                })
                        if len(mining_batch) >= 16: break
                if len(mining_batch) >= 16: break
            
            if len(mining_batch) >= 8:
                agent.ssl_update(mining_batch[:16])
            metrics.stop_timer("agent_updating_ssl")

def main():
    global args
    # 1. Setup Environment & Agent
    env_name = args.env
    hparam_str = f"{args.algo}_rl{int(args.rl)}_bc{int(args.bc)}_int{int(args.intervention)}_seed{args.seed}"
    results_base_dir = os.path.join("results", env_name, args.experiment_name, hparam_str)
    os.makedirs(results_base_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    if "highway" in env_name:
        import highway_env
        env = gym.wrappers.FlattenObservation(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if args.algo == "cql":
        agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="CQL", save_dir=results_base_dir, device_name="cpu")
    elif args.algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, name="PPO", save_dir=results_base_dir, device_name="cpu")

    aux_agent = None
    if args.curriculum and args.curriculum_method in ['separate', 'kl']:
        AgentClass = CQLAgent if args.algo == "cql" else PPOAgent
        aux_agent = AgentClass(obs_dim=obs_dim, action_dim=action_dim, name=f"{args.algo.upper()}_Aux", save_dir=results_base_dir, device_name="cpu")

    # 2. Setup Buffers, Metrics & Router
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
        
        # Step 1: Base RL Collection
        episodes = []
        if args.num_rl_frames > 0:
            print(f"Collecting {args.num_rl_frames} RL frames...")
            # Optimization: Only render/store if intervention is active
            episodes = run_rl_collection(agent, env, num_frames=args.num_rl_frames, metrics=metrics, update=args.rl, do_render=args.intervention)
        
        agent.checkpoint_model(specific_name=f"rl_collection_{iteration}")
        
        # Step 2: Interactive Review
        annotations = []
        if args.intervention and len(episodes) > 0:
            print("Starting Human Interactive Review...")
            # Pick the episode with the LOWEST total reward to review
            summary_ep = min(episodes, key=lambda x: x['total_reward'])
            
            # Hydrate the summary into a full trajectory with images for the UI
            hydrated_trajectory = hydrate_trajectory(env, summary_ep['seed'], summary_ep['trajectory'])
            
            wrapper = InteractiveGymWrapper(
                env, agent=agent, buffers=buffers, metrics=metrics,
                initial_trajectory=hydrated_trajectory, initial_seed=summary_ep['seed'], env_name=env_name
            )
            corrected_trajectory, annotations, final_seed = wrapper.run()
            
            if args.algo != "ppo":
                for i in range(len(corrected_trajectory) - 1):
                    s, ns = corrected_trajectory[i], corrected_trajectory[i+1]
                    agent.store_transition(s['obs'], ns['action'], ns['reward'], ns['obs'], ns['terminated'], ns['truncated'])

            agent.checkpoint_model(specific_name=f"interactive_review_{iteration}")
            
            # Step 3: LLM Routing
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
            
        # Step 4: Curriculum (Localized RL)
        if len(buffers['curriculum']) > 0 and args.curriculum:
            metrics.start_timer("agent_updating_local_rl")
            while not buffers['curriculum'].is_empty():
                task = buffers['curriculum'].pop()
                if args.curriculum_method in ['separate', 'kl']:
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
                            agent.store_transition(obs, action, reward, next_obs, term, trunc)
                            agent.store_local_transition(obs, action, local_reward, next_obs, term, trunc)
                            agent.rl_update(local=True); agent.rl_update(local=False)
                        elif args.curriculum_method == 'separate':
                            aux_agent.store_transition(obs, action, local_reward, next_obs, term, trunc)
                            aux_agent.rl_update()
                            agent.store_transition(obs, action, reward, next_obs, term, trunc)
                        elif args.curriculum_method == 'kl':
                            aux_agent.store_transition(obs, action, local_reward, next_obs, term, trunc)
                            aux_agent.rl_update()
                            buffers['kl_target'].push(obs)
                        n_frames += 1
                        if term or trunc: break
                        obs = next_obs
                    metrics.log_frames(n_frames, source="curriculum")
            metrics.stop_timer("agent_updating_local_rl")

        # Step 5: Unified Update
        unified_train_step(args, agent, aux_agent, buffers, metrics, active_heuristics, global_buffer_proxy)

        # Iteration cleanup & checkpoint
        buffers['example'].save(os.path.join(results_base_dir, f"example_buffer_{iteration}.pt"))
        buffers['anti_example'].save(os.path.join(results_base_dir, f"anti_example_buffer_{iteration}.pt"))
        with open(os.path.join(results_base_dir, f"annotations_{iteration}.json"), "w") as f:
            json.dump(annotations, f, indent=4)
        agent.checkpoint_model(specific_name=f"agent_update_{iteration}")
        
        # Evaluation
        print("Evaluating Agent...")
        mean_ret, std_ret = evaluate_return(agent, env_name, num_episodes=5)
        bc_loss = calculate_cross_entropy(agent, buffers['example'], anti=False) if args.bc else None
        anti_bc_loss = calculate_cross_entropy(agent, buffers['anti_example'], anti=True) if args.anti_bc else None
        metrics.log_evaluation(iteration, mean_ret, std_ret, bc_loss, anti_bc_loss)
        metrics.log_iteration()
        metrics.save_to_json(os.path.join(results_base_dir, f"metrics_{iteration}.json"))
        metrics.save_to_json(os.path.join(results_base_dir, "metrics_latest.json"))
        
    env.close()

class MagicReplayProxy:
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
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Environment to run (LunarLander-v3 or highway-v0)")
    parser.add_argument("--rl", action="store_true", help="Enable RL/Offline Q-Learning updates")
    parser.add_argument("--bc", action="store_true", help="Enable Behavior Cloning updates")
    parser.add_argument("--intervention", action="store_true", help="Enable human interactive review phase")
    parser.add_argument("--anti_bc", action="store_true", help="Enable Anti-BC updates (requires intervention)")
    parser.add_argument("--ssl", action="store_true", help="Enable Semi-Supervised Learning / Noisy Trajectories")
    parser.add_argument("--curriculum", action="store_true", help="Enable Curriculum Learning / Auxiliary Rewards")
    parser.add_argument("--awbc", action="store_true", help="Use Advantage Weighted Behavior Cloning")
    parser.add_argument("--curriculum_method", type=str, default="main", choices=["main", "separate", "kl"])
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the experiment")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    parser.add_argument("--num_local_epochs", type=int, default=5)
    parser.add_argument("--curriculum_traj_len", type=int, default=0)
    parser.add_argument("--num_rl_frames", type=int, default=2000, help="Number of RL frames to collect per iteration. Set to 0 for offline-only.")
    parser.add_argument("--num_unified_epochs", type=int, default=5, help="Number of training epochs per iteration in the unified pipeline.")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Scale of Gaussian noise for NOISY_HUMAN augmentation")
    parser.add_argument("--num_noisy_samples", type=int, default=5, help="Number of noisy samples per human frame")
    parser.add_argument("--preload_expert_data", type=str, default=None, help="Path to a .pkl expert dataset to pre-populate the example_buffer")
    args = parser.parse_args()
    main()
