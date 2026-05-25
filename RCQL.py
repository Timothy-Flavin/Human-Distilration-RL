import os
import warnings

# 1. Suppress all Python warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# 2. Suppress hardware-specific and logging-heavy backends
os.environ['MKLDNN_VERBOSE'] = '0'
os.environ['MKL_VERBOSE'] = '0'
os.environ['NNPACK_VERBOSE'] = '0'

# 3. Import torch and immediately configure backends
import torch
try:
    torch.backends.nnpack.enabled = False
    torch.backends.cudnn.enabled = False
except Exception:
    pass

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
from Agent import Agent

class RecurrentCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=64):
        super(RecurrentCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Calculate flatten size
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            dummy_out = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
            flatten_size = dummy_out.numel()
            
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flatten_size, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

class RecurrentQNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=3, img_size=64, hidden_dim=512):
        super(RecurrentQNetwork, self).__init__()
        self.encoder = RecurrentCNNEncoder(in_channels, img_size)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None, features=None):
        # x shape: (Batch, Time, Channels, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        if features is None:
            # Flatten Batch and Time dimensions for the CNN
            x_flat = x.view(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.view(batch_size, seq_len, -1)
        
        lstm_out, hidden = self.lstm(features, hidden)
        q_values = self.fc(lstm_out)
        return q_values, hidden

class RecurrentValueNetwork(nn.Module):
    def __init__(self, in_channels=3, img_size=64, hidden_dim=512):
        super(RecurrentValueNetwork, self).__init__()
        self.encoder = RecurrentCNNEncoder(in_channels, img_size)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None, features=None):
        batch_size, seq_len, c, h, w = x.size()
        if features is None:
            x_flat = x.view(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.view(batch_size, seq_len, -1)
        
        lstm_out, hidden = self.lstm(features, hidden)
        values = self.fc(lstm_out)
        return values, hidden


class RCQLAgent(Agent):
    def __init__(self, obs_dim, action_dim, name="RCQL", save_dir="results", device_name="cpu", hidden_dim=512, lr=3e-4, epsilon=0.1):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        
        in_channels = obs_dim[0]
        img_size = obs_dim[1]
        
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        
        self.q_net = RecurrentQNetwork(action_dim, in_channels, img_size, hidden_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.v_net = RecurrentValueNetwork(in_channels, img_size, hidden_dim).to(self.device_name)
        self.v_target = copy.deepcopy(self.v_net)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0 
        
        self.q_hidden = None
        self.v_hidden = None
        self.replay_buffer = collections.deque(maxlen=1000)

    def reset_hidden(self):
        self.q_hidden = None
        self.v_hidden = None

    def _normalize(self, obs):
        """Force consistent normalization to [0, 1]."""
        if obs.dtype == torch.uint8 or obs.max() > 1.0:
            return obs.float() / 255.0
        return obs.float()

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        self.q_net.eval()
        with torch.no_grad():
            if observations.ndim == 3:
                obs = observations.unsqueeze(0).unsqueeze(0)
                batch_size = 1
            else:
                obs = observations.unsqueeze(1)
                batch_size = observations.shape[0]
            
            obs = self._normalize(obs)
            q_values, self.q_hidden = self.q_net(obs, self.q_hidden)
            q_values = q_values.squeeze(1) 
            
            if deterministic:
                action = q_values.argmax(dim=1)
            else:
                if random.random() < self.epsilon:
                    action = torch.randint(0, self.action_dim, (batch_size,)).to(self.device_name)
                else:
                    action = q_values.argmax(dim=1)
        self.q_net.train()
        return action

    def store_episode(self, episode):
        self.replay_buffer.append(episode)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    def _prepare_batch(self, batch_episodes, seq_len=32):
        """Samples sequences of length seq_len from anywhere in the episode."""
        obs_seqs, action_seqs, reward_seqs, done_seqs = [], [], [], []
        
        for ep in batch_episodes:
            transitions = ep['transitions']
            L = len(transitions)
            
            if L <= seq_len:
                start_idx = 0
                actual_len = L
            else:
                # Random window sampling
                start_idx = random.randint(0, L - seq_len)
                actual_len = seq_len
                
            sub_seq = transitions[start_idx : start_idx + actual_len]
            
            # obs includes s_t ... s_{t+actual_len}
            obs = [t['obs'] for t in sub_seq]
            # s_{t+actual_len+1} is the next_obs of the last transition in window
            obs.append(sub_seq[-1]['next_obs'])
            
            actions = [t['action'] for t in sub_seq]
            rewards = [t['reward'] for t in sub_seq]
            dones = [float(t['terminated'] or t['truncated']) for t in sub_seq]
            
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                obs += [np.zeros_like(obs[0])] * pad_len
                actions += [0] * pad_len
                rewards += [0.0] * pad_len
                dones += [1.0] * pad_len
                
            obs_seqs.append(obs)
            action_seqs.append(actions)
            reward_seqs.append(rewards)
            done_seqs.append(dones)

        obs_tensor = self._normalize(torch.tensor(np.array(obs_seqs), dtype=torch.float32).to(self.device_name))
        action_tensor = torch.tensor(np.array(action_seqs), dtype=torch.long).to(self.device_name)
        reward_tensor = torch.tensor(np.array(reward_seqs), dtype=torch.float32).to(self.device_name)
        done_tensor = torch.tensor(np.array(done_seqs), dtype=torch.float32).to(self.device_name)
        
        return obs_tensor, action_tensor, reward_tensor, done_tensor


    def update_value(self, batch_episodes) -> dict:
        joint_obs, _, rewards, dones = self._prepare_batch(batch_episodes)
        
        # joint_obs has length seq_len + 1
        v_full, _ = self.v_net(joint_obs)
        current_v = v_full[:, :-1, :].squeeze(-1)
        
        with torch.no_grad():
            next_v = v_full[:, 1:, :].squeeze(-1)
            target_v = rewards + (1.0 - dones) * self.gamma * next_v

        v_loss = F.mse_loss(current_v, target_v)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        for param, target_param in zip(self.v_net.parameters(), self.v_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return {"loss_v": v_loss.item(), "v_mean": current_v.mean().item()}

    def update_td(self, batch_episodes, ssl: bool = False, masks: list = None) -> dict:
        joint_obs, actions, rewards, dones = self._prepare_batch(batch_episodes)
        
        q_logits_full, _ = self.q_net(joint_obs)
        q_logits = q_logits_full[:, :-1, :]
        current_q = q_logits.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            q_target_full, _ = self.q_target(joint_obs)
            next_q = q_target_full[:, 1:, :].max(2)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        q_loss = F.mse_loss(current_q, target_q)
        cql_loss = torch.logsumexp(q_logits, dim=2).mean() - current_q.mean()
        total_loss = q_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"loss_td": total_loss.item(), "q_loss": q_loss.item(), "cql_loss": cql_loss.item(), "q_mean": current_q.mean().item()}

    def update_supervised(self, batch_episodes, ssl: bool = False, masks: list = None, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        obs, actions, _, _ = self._prepare_batch(batch_episodes)
        obs_trimmed = obs[:, :-1, :] # BC only needs s0..sT
        
        logits, _ = self.q_net(obs_trimmed)
        logits = logits.view(-1, self.action_dim)
        labels = actions.view(-1)

        if anti:
            probs = F.softmax(logits, dim=1)
            bad_action_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
            loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
        else:
            if advantages is not None:
                adv = advantages.to(self.device_name).view(-1)
                log_probs = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
                loss = -(adv * selected_log_probs).mean()
            else:
                loss = F.cross_entropy(logits, labels)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return {"loss_supervised": loss.item()}

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        logits, _ = self.q_net(obs)
        return logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        values, _ = self.v_net(obs)
        return values.squeeze(-1)

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        current_logits, _ = self.q_net(obs)
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        kl_loss = F.kl_div(current_log_probs.view(-1, self.action_dim), 
                           target_probs.view(-1, self.action_dim), 
                           reduction='batchmean')
        self.q_optimizer.zero_grad()
        kl_loss.backward()
        self.q_optimizer.step()
        return {"loss_kl": kl_loss.item()}

    def _save_checkpoint(self, path):
        state = {'q_net': self.q_net.state_dict(), 'v_net': self.v_net.state_dict()}
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.v_net.load_state_dict(checkpoint['v_net'])
        self.q_target = copy.deepcopy(self.q_net)
        self.v_target = copy.deepcopy(self.v_net)
