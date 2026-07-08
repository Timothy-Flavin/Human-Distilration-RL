import torch
import torch.nn as nn
import torch.nn.functional as F
from RCQL import RCQLAgent
from rcql_test_env import FlickeringCatchEnv
from buffers import FastGPUEpisodicBuffer

class OldQNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=3, img_size=16, hidden_dim=512):
        super().__init__()
        from RCQL import RecurrentCNNEncoder
        self.encoder = RecurrentCNNEncoder(in_channels, img_size)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    def forward(self, x, hidden=None, features=None):
        batch_size, seq_len, c, h, w = x.size()
        if features is None:
            x_flat = x.reshape(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.reshape(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(features, hidden)
        q = self.fc(lstm_out)
        return q, q, q, hidden

agent = RCQLAgent((3, 16), 3, device_name="cuda", lr=1e-3, epsilon=0.2)
agent.q_net = OldQNetwork(3, 3, 16, 512).to("cuda")
agent.q_target = OldQNetwork(3, 3, 16, 512).to("cuda")
agent.q_target.load_state_dict(agent.q_net.state_dict())
agent.q_optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=1e-3)
agent.cql_alpha = 0.0

env = FlickeringCatchEnv(size=16, flicker_steps=4)
fast_buffer = FastGPUEpisodicBuffer(max_total_transitions=1000, device="cuda", obs_shape=(3, 16, 16))

for ep in range(1000):
    obs, _ = env.reset()
    agent.reset_hidden()
    episode = []
    term = False
    while not term:
        obs_t = torch.tensor(obs).float().unsqueeze(0).unsqueeze(0).to("cuda") / 255.0
        q, _, _, agent.q_hidden = agent.q_net(obs_t, agent.q_hidden)
        action = q.squeeze(1).argmax(dim=1).item() if torch.rand(1).item() > 0.8 else torch.randint(0, 3, (1,)).item()
        next_obs, reward, term, trunc, _ = env.step(action)
        episode.append({'obs': obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'terminated': term, 'truncated': trunc})
        obs = next_obs
    
    fast_buffer.add_episode(episode)
    if fast_buffer.current_size >= 16:
        for _ in range(4):
            obs_b, act_b, rew_b, done_b, mask_b = fast_buffer.sample_batch(8, seq_len=15)
            agent.update_td(obs_b, act_b, rew_b, done_b, mask_b, burn_in=4)

    if (ep + 1) % 50 == 0:
        eval_reward = 0
        for _ in range(20):
            e_obs, _ = env.reset()
            agent.reset_hidden()
            e_term = False
            e_total = 0
            while not e_term:
                e_obs_t = torch.tensor(e_obs).float().unsqueeze(0).unsqueeze(0).to("cuda") / 255.0
                q, _, _, agent.q_hidden = agent.q_net(e_obs_t, agent.q_hidden)
                e_act = q.squeeze(1).argmax(dim=1).item()
                e_obs, e_rew, e_term, _, _ = env.step(e_act)
                e_total += e_rew
            eval_reward += e_total
        print(f"Ep {ep+1} Eval:", eval_reward / 20)
