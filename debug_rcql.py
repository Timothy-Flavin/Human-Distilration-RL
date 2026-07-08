import torch
from RCQL import RCQLAgent
from rcql_test_env import FlickeringCatchEnv
from buffers import FastGPUEpisodicBuffer
import numpy as np

env = FlickeringCatchEnv(size=16, flicker_steps=4)
obs_dim = env.observation_space.shape
action_dim = env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = RCQLAgent(obs_dim, action_dim, device_name=device, lr=1e-3, epsilon=0.2)
agent.cql_alpha = 0.0

fast_buffer = FastGPUEpisodicBuffer(
    max_total_transitions=1000, 
    device=device, 
    obs_shape=(3, 16, 16)
)

eval_rewards = []
for ep in range(150):
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
            res = agent.update_td(obs_b, act_b, rew_b, done_b, mask_b, burn_in=4)
            if ep == 149 and _ == 3:
                print("TD Loss:", res["loss_td"])
                print("Q Loss:", res["q_loss"])
                print("Q Mean:", agent.q_net(obs_b[:, 4:, :])[0].mean().item())

    if (ep + 1) % 50 == 0:
        eval_reward = 0
        for _ in range(20):
            e_obs, _ = env.reset()
            agent.reset_hidden()
            e_term = False
            e_total = 0
            while not e_term:
                e_obs_t = torch.tensor(e_obs).to(device)
                e_act = agent.act(e_obs_t, deterministic=True).item()
                e_obs, e_rew, e_term, _, _ = env.step(e_act)
                e_total += e_rew
            eval_reward += e_total
        print(f"Ep {ep+1} Eval:", eval_reward / 20)
