import torch
import numpy as np
import matplotlib.pyplot as plt
from RCQL import RCQLAgent
from rcql_test_env import FlickeringCatchEnv
from buffers import FastGPUEpisodicBuffer

env = FlickeringCatchEnv(size=16, flicker_steps=4)
obs_dim = env.observation_space.shape
action_dim = env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = RCQLAgent(obs_dim, action_dim, device_name=device, lr=1e-3, epsilon=1.0)
agent.cql_alpha = 0.0

fast_buffer = FastGPUEpisodicBuffer(
    max_total_transitions=20000, 
    device=device, 
    obs_shape=(3, 16, 16)
)

eval_intervals = 50
eval_episodes = 20
history = []

for ep in range(1000):
    # Epsilon decay
    agent.epsilon = max(0.05, 1.0 - ep / 500)
    
    obs, _ = env.reset()
    agent.reset_hidden()
    episode = []
    term = False
    while not term:
        obs_t = torch.tensor(obs).to(device)
        action = agent.act(obs_t).item()
        next_obs, reward, term, trunc, _ = env.step(action)
        episode.append({'obs': obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'terminated': term, 'truncated': trunc})
        obs = next_obs
    
    fast_buffer.add_episode(episode)
    if fast_buffer.current_size >= 16:
        for _ in range(4):
            obs_b, act_b, rew_b, done_b, mask_b = fast_buffer.sample_batch(8, seq_len=15)
            agent.update_td(obs_b, act_b, rew_b, done_b, mask_b, burn_in=4)
    
    if (ep + 1) % eval_intervals == 0:
        eval_rewards = []
        for _ in range(eval_episodes):
            e_obs, _ = env.reset()
            agent.reset_hidden()
            e_term = False; e_total = 0
            while not e_term:
                e_obs_t = torch.tensor(e_obs).to(device)
                e_act = agent.act(e_obs_t, deterministic=True).item()
                e_obs, e_rew, e_term, _, _ = env.step(e_act)
                e_total += e_rew
            eval_rewards.append(e_total)
        
        mean_r = np.mean(eval_rewards)
        history.append(mean_r)
        print(f"Episode {ep+1}: Eval Reward = {mean_r:.2f}, Epsilon = {agent.epsilon:.2f}")

