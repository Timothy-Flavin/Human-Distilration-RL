import torch
import numpy as np
import matplotlib.pyplot as plt
from rcql_test_env import MemorySanityEnv
from RCQL import RCQLAgent
import random
import os
from buffers import FastGPUEpisodicBuffer # Assuming you saved it here

# Suppress hardware warnings globally for manual runs
os.environ['MKLDNN_VERBOSE'] = '0'
os.environ['MKL_VERBOSE'] = '0'
os.environ['NNPACK_VERBOSE'] = '0'
torch.backends.nnpack.enabled = False

def get_final_step_qs(agent, env_mode, device):
    """
    Simulates a full 8-step episode and returns Q-values at the terminal decision point.
    """
    env = MemorySanityEnv(mode=env_mode, img_size=16)
    obs, _ = env.reset()
    agent.reset_hidden()
    
    obs_seq = [obs]
    for _ in range(7):
        noise_obs = env._get_noise()
        obs_seq.append(noise_obs)
    
    # (1, 8, 3, 16, 16)
    obs_tensor = torch.tensor(np.array(obs_seq), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = agent.get_logits(obs_tensor) # (1, 8, action_dim)
        final_qs = logits[0, -1].cpu().numpy()
    return final_qs

def run_test(mode="always_blue", num_episodes=512, test_name="Sanity Test"):
    print(f"\n>>> Starting {test_name} ({mode})")
    env = MemorySanityEnv(mode=mode, img_size=16)
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # standard hyperparameters
    agent = RCQLAgent(obs_dim, action_dim, name=f"RCQL_{mode}", device_name=device, lr=3e-4, epsilon=0.2)
    agent.cql_alpha = 0.0 # pure Q-learning verification
    fast_buffer = FastGPUEpisodicBuffer(
        max_episodes=num_episodes, 
        max_ep_len=10, 
        device=device, 
        obs_shape=(3, 16, 16)  # <--- Add this
    )
    
    eval_intervals = 32
    eval_episodes = 16
    
    eval_rewards_mean = []
    eval_rewards_std = []
    eval_q_values = {a: [] for a in range(action_dim)}
    eval_steps = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset_hidden()
        episode_transitions = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            action = agent.act(obs_t).item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_transitions.append({
                'obs': obs, 'action': action, 'reward': reward,
                'next_obs': next_obs, 'terminated': terminated, 'truncated': truncated
            })
            obs = next_obs
            
        agent.store_episode({'transitions': episode_transitions}) # Keep for compatibility if needed
        fast_buffer.add_episode(episode_transitions)
        if fast_buffer.current_size >= 16:
            for _ in range(4):
                # Sample with seq_len=8 (covers the entire 8-step episode)
                obs_b, act_b, rew_b, done_b, mask_b = fast_buffer.sample_batch(8, seq_len=8)
                # Use a small burn_in for the tiny test env
                agent.update_td(obs_b, act_b, rew_b, done_b, mask_b, burn_in=0)
                # agent.update_value(obs_b, act_b, rew_b, done_b, mask_b, burn_in=2)
            
        if (ep + 1) % 20 == 0:
            print(f"[*] Episode {ep+1}/{num_episodes}...")

        if (ep + 1) % eval_intervals == 0:
            rewards = []
            for _ in range(eval_episodes):
                e_obs, _ = env.reset()
                agent.reset_hidden()
                e_term = False; e_trunc = False
                total_reward = 0
                while not (e_term or e_trunc):
                    e_obs_t = torch.tensor(e_obs, dtype=torch.float32).to(device)
                    e_action = agent.act(e_obs_t, deterministic=True).item()
                    e_obs, e_reward, e_term, e_trunc, _ = env.step(e_action)
                    total_reward += e_reward
                rewards.append(total_reward)
            
            # Decision Point Q-values
            if mode == "always_blue":
                final_qs = get_final_step_qs(agent, "always_blue", device)
            else:
                qs_list = [get_final_step_qs(agent, "stochastic", device) for _ in range(5)]
                final_qs = np.mean(qs_list, axis=0)

            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            eval_rewards_mean.append(mean_r)
            eval_rewards_std.append(std_r)
            eval_steps.append(ep + 1)
            for a in range(action_dim):
                eval_q_values[a].append(final_qs[a])
                
            print(f"    Eval {ep+1}: Reward {mean_r:.2f} | Final-Step Qs: {final_qs}")

    return eval_steps, eval_rewards_mean, eval_rewards_std, eval_q_values

def main():
    steps1, mean1, std1, q1 = run_test(mode="always_blue", num_episodes=256, test_name="Sanity Test")
    steps2, mean2, std2, q2 = run_test(mode="stochastic", num_episodes=512, test_name="Memorization Test")
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1); plt.errorbar(steps1, mean1, yerr=std1, fmt='-o', capsize=5)
    plt.title("Sanity Test: Rewards (Always Blue)"); plt.ylabel("Mean Reward"); plt.grid(True)
    plt.subplot(2, 2, 2)
    for a in range(len(q1)): plt.plot(steps1, q1[a], label=f"Action {a}")
    plt.title("Sanity Test: Final-Step Q-Values"); plt.ylabel("Q-Value"); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 3); plt.errorbar(steps2, mean2, yerr=std2, fmt='-o', capsize=5)
    plt.title("Memorization Test: Rewards (Stochastic)"); plt.ylabel("Mean Reward"); plt.grid(True)
    plt.subplot(2, 2, 4)
    for a in range(len(q2)): plt.plot(steps2, q2[a], label=f"Action {a}")
    plt.title("Memorization Test: Final-Step Q-Values"); plt.ylabel("Q-Value"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("rcql_test_results.png")
    print(f"\n[*] Training complete. Results saved to rcql_test_results.png")

if __name__ == "__main__":
    main()
