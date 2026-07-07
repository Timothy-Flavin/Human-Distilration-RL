import os
import json
import torch
import gymnasium as gym
import numpy as np
import argparse
from Agent import Agent
from CQL import CQLAgent
from PPO import PPOAgent
from buffers import ReplayBuffer

from compliance_metrics import get_compliance_score

def evaluate_return(agent, env, env_name, num_episodes=25):
    # Optimization: For vector-based environments, step-by-step GPU inference is slow.
    original_device = agent.device_name
    use_cpu = (original_device == "cuda" and isinstance(agent.obs_dim, int) and agent.obs_dim < 200)
    if use_cpu:
        agent.to("cpu")
            
    total_returns = []
    compliance_scores = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        terminated = False
        truncated = False
        episode_return = 0
        episode_obs = [obs]
        while not (terminated or truncated):
            action = agent.predict(obs) # predict uses deterministic=True
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_obs.append(obs)
        total_returns.append(episode_return)
        compliance_scores.append(get_compliance_score(env_name, episode_obs))
    
    if use_cpu:
        agent.to(original_device)
        
    return np.mean(total_returns), np.std(total_returns), np.mean(compliance_scores)


def calculate_cross_entropy(agent, buffer, anti=False):
    if len(buffer) == 0:
        return None
    
    # Use the whole buffer or a large sample for evaluation
    batch_size = min(len(buffer), 1024)
    batch_items = buffer.sample(batch_size)
    
    # Extract and convert to tensors robustly
    obs_list = []
    labels_list = []
    for item in batch_items:
        o = item['obs']
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(o, dtype=torch.float32)
        obs_list.append(o)
        
        a = item['action']
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.long)
        labels_list.append(a)

    obs = torch.stack(obs_list).to(agent.device_name)
    labels = torch.stack(labels_list).to(agent.device_name)
    
    # We assume the agent has a way to set eval mode or it doesn't matter for get_logits
    with torch.no_grad():
        logits = agent.get_logits(obs)
        
        if not anti:
            # Standard CrossEntropy
            loss = torch.nn.functional.cross_entropy(logits, labels)
        else:
            # Anti-BC loss
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_rejected = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = -torch.log(1.0 - prob_rejected + 1e-6).mean()
            
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", type=str, required=True, help="Path to the agent .pt checkpoint")
    parser.add_argument("--env_name", type=str, default="LunarLander-v3")
    parser.add_argument("--example_buffer", type=str, help="Path to serialized example buffer (optional)")
    parser.add_argument("--anti_example_buffer", type=str, help="Path to serialized anti-example buffer (optional)")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    args = parser.parse_args()
    
    # 1. Initialize Agent
    # We need to know obs/action dim. For LunarLander-v3 it's 8/4.
    # In a real script, we'd detect this from the env.
    env_name = args.env_name
    if env_name == "highway":
        env_name = "highway-v0"
    if "highway" in env_name:
        import highway_env
    
    temp_env = gym.make(env_name)
    if "highway" in env_name:
        from gymnasium.wrappers import FlattenObservation
        temp_env = FlattenObservation(temp_env)
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()
    
    if args.algo == "cql":
        agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim, device_name="cpu")
    elif args.algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device_name="cpu")
    
    agent.load_model(args.agent_path)
    
    # 2. Measure Return
    print(f"Evaluating return for {args.agent_path}...")
    mean_ret, std_ret, compliance_score = evaluate_return(agent, temp_env, args.env_name)
    temp_env.close()
    
    # 3. Measure Cross-Entropy
    bc_loss = None
    anti_bc_loss = None
    
    if args.example_buffer and os.path.exists(args.example_buffer):
        ex_buf = ReplayBuffer(capacity=10000)
        ex_buf.load(args.example_buffer)
        bc_loss = calculate_cross_entropy(agent, ex_buf, anti=False)
        
    if args.anti_example_buffer and os.path.exists(args.anti_example_buffer):
        anti_buf = ReplayBuffer(capacity=10000)
        anti_buf.load(args.anti_example_buffer)
        anti_bc_loss = calculate_cross_entropy(agent, anti_buf, anti=True)
    
    # 4. Save Results
    perf_data = {
        "agent_path": args.agent_path,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "compliance_score": compliance_score,
        "bc_cross_entropy": bc_loss,
        "anti_bc_cross_entropy": anti_bc_loss
    }
    
    output_path = args.agent_path.replace(".pt", "_perf.json")
    with open(output_path, "w") as f:
        json.dump(perf_data, f, indent=4)
    
    print(f"Performance results saved to {output_path}")
    print(f"Mean Return: {mean_ret:.2f} +/- {std_ret:.2f}")

if __name__ == "__main__":
    main()
