import gymnasium as gym
import torch
import numpy as np
import os
import argparse
from CQL import CQLAgent
from PPO import PPOAgent
from compliance_metrics import get_compliance_score

def evaluate_model_compliance(agent, env_name, num_episodes=10):
    if env_name == "highway":
        env_name = "highway-v0"
    if "highway" in env_name:
        import highway_env

    env = gym.make(env_name)
    if "highway" in env_name:
        env = gym.wrappers.FlattenObservation(env)
    
    compliance_scores = []
    returns = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_obs = [obs]
        done = False
        ep_ret = 0
        
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(obs)
            ep_ret += reward
            done = terminated or truncated
            
        returns.append(ep_ret)
        compliance_scores.append(get_compliance_score(env_name, episode_obs))
    
    env.close()
    return np.mean(returns), np.std(returns), np.mean(compliance_scores), np.std(compliance_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # Initialize Agent
    env_name = args.env
    if env_name == "highway":
        env_name = "highway-v0"
    if "highway" in env_name:
        import highway_env
        
    env = gym.make(env_name)
    if "highway" in env_name:
        env = gym.wrappers.FlattenObservation(env)
        
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    if args.algo == "cql":
        agent = CQLAgent(obs_dim=obs_dim, action_dim=action_dim)
    else:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
    
    agent.load_model(args.model_path)
    
    print(f"[*] Evaluating compliance for model: {args.model_path}")
    mean_ret, std_ret, mean_comp, std_comp = evaluate_model_compliance(agent, args.env, args.episodes)
    
    print(f"\n=== Compliance Report ===")
    print(f"Environment:      {args.env}")
    print(f"Mean Return:      {mean_ret:.2f} +/- {std_ret:.2f}")
    print(f"Compliance Score: {mean_comp:.4f} +/- {std_comp:.4f}")
    print("-" * 25)

if __name__ == "__main__":
    main()
