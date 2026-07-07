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
        self.fc = nn.Linear(hidden_dim, action_dim+1)

    def forward(self, x, hidden=None, features=None):
        # x shape: (Batch, Time, Channels, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        if features is None:
            # Flatten Batch and Time dimensions for the CNN
            x_flat = x.reshape(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.reshape(batch_size, seq_len, -1)
        
        lstm_out, hidden = self.lstm(features, hidden)
        logits = self.fc(lstm_out)
        advantages = logits[:, :, :-1]  # All but last dimension are Q-values
        advantages -= advantages.mean(dim=-1, keepdim=True)  # Normalize Q-values
        v = logits[:, :, -1:]  # Last dimension is the value function
        return v, advantages, hidden

class RecurrentValueNetwork(nn.Module):
    def __init__(self, in_channels=3, img_size=64, hidden_dim=512):
        super(RecurrentValueNetwork, self).__init__()
        self.encoder = RecurrentCNNEncoder(in_channels, img_size)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None, features=None):
        batch_size, seq_len, c, h, w = x.size()
        if features is None:
            x_flat = x.reshape(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.reshape(batch_size, seq_len, -1)
        
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

    def _ensure_channel_first(self, x):
        """Detects and converts (..., H, W, C) to (..., C, H, W) if needed."""
        # If last dim is 1 or 3 and it's not the channel dim already, permute.
        # We assume H, W are larger than 3 (usually 16, 64, etc.)
        if x.shape[-1] in [1, 3] and x.shape[-3] not in [1, 3]:
            ndim = x.ndim
            if ndim == 3: return x.permute(2, 0, 1)
            if ndim == 4: return x.permute(0, 3, 1, 2)
            if ndim == 5: return x.permute(0, 1, 4, 2, 3)
        return x

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        self.q_net.eval()
        with torch.no_grad():
            if observations.ndim == 3:
                obs = observations.unsqueeze(0).unsqueeze(0)
                batch_size = 1
            else:
                obs = observations.unsqueeze(1)
                batch_size = observations.shape[0]
            
            obs = self._ensure_channel_first(obs)
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

    def predict(self, observations, deterministic: bool = True):
        # observations might be numpy array or torch tensor
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device_name)
        
        # Ensure it has at least (C, H, W)
        action = self.act(observations, deterministic=deterministic)
        return action.item() if action.numel() == 1 else action.cpu().numpy()

    def store_episode(self, episode):
        self.replay_buffer.append(episode)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    # def _prepare_batch(self, batch_episodes, seq_len=32):
    #     """Samples sequences of length seq_len from anywhere in the episode."""
    #     obs_seqs, action_seqs, reward_seqs, done_seqs = [], [], [], []
        
    #     for ep in batch_episodes:
    #         transitions = ep['transitions']
    #         L = len(transitions)
            
    #         if L <= seq_len:
    #             start_idx = 0
    #             actual_len = L
    #         else:
    #             # Random window sampling
    #             start_idx = random.randint(0, L - seq_len)
    #             actual_len = seq_len
                
    #         sub_seq = transitions[start_idx : start_idx + actual_len]
            
    #         # obs includes s_t ... s_{t+actual_len}
    #         obs = [t['obs'] for t in sub_seq]
    #         # s_{t+actual_len+1} is the next_obs of the last transition in window
    #         obs.append(sub_seq[-1]['next_obs'])
            
    #         actions = [t['action'] for t in sub_seq]
    #         rewards = [t['reward'] for t in sub_seq]
    #         dones = [float(t['terminated'] or t['truncated']) for t in sub_seq]
            
    #         if actual_len < seq_len:
    #             pad_len = seq_len - actual_len
    #             obs += [np.zeros_like(obs[0])] * pad_len
    #             actions += [0] * pad_len
    #             rewards += [0.0] * pad_len
    #             dones += [1.0] * pad_len
                
    #         obs_seqs.append(obs)
    #         action_seqs.append(actions)
    #         reward_seqs.append(rewards)
    #         done_seqs.append(dones)

    #     obs_tensor = torch.tensor(np.array(obs_seqs), dtype=torch.float32).to(self.device_name)
    #     obs_tensor = self._ensure_channel_first(obs_tensor)
    #     obs_tensor = self._normalize(obs_tensor)
        
    #     action_tensor = torch.tensor(np.array(action_seqs), dtype=torch.long).to(self.device_name)
    #     reward_tensor = torch.tensor(np.array(reward_seqs), dtype=torch.float32).to(self.device_name)
    #     done_tensor = torch.tensor(np.array(done_seqs), dtype=torch.float32).to(self.device_name)
        
    #     return obs_tensor, action_tensor, reward_tensor, done_tensor


    def update_value(self, obs, actions, rewards, dones, masks, burn_in=16) -> dict:
        # 1. Burn-in Phase (No Gradients)
        with torch.no_grad():
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, h_v = self.v_net(burn_obs)
                _, h_v_target = self.v_target(burn_obs)
            else:
                h_v, h_v_target = None, None

        # 2. Main Sequence Processing
        obs_active = obs[:, burn_in:, :] # Length: update_len + 1
        r_active = rewards[:, burn_in:]
        d_active = dones[:, burn_in:]
        m_active = masks[:, burn_in:]

        v_full, _ = self.v_net(obs_active, hidden=h_v)
        current_v = v_full[:, :-1, :].squeeze(-1)

        with torch.no_grad():
            v_target_full, _ = self.v_target(obs_active, hidden=h_v_target)
            next_v = v_target_full[:, 1:, :].squeeze(-1)
            target_v = r_active + (1.0 - d_active) * self.gamma * next_v

        # 3. Masked Loss
        v_td_loss_unmasked = F.mse_loss(current_v, target_v, reduction='none')
        valid_steps = m_active.sum() + 1e-8
        v_loss = (v_td_loss_unmasked * m_active).sum() / valid_steps

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=1.0)
        self.v_optimizer.step()

        for param, target_param in zip(self.v_net.parameters(), self.v_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            "loss_v": v_loss.item(), 
            "v_mean": (current_v * m_active).sum().item() / valid_steps.item(),
            "current_v": current_v.detach(),
            "next_v": next_v.detach()
        }

    def update_td(self, obs, actions, rewards, dones, masks, burn_in=16, use_cql=True) -> dict:
        # 1. Burn-in Phase
        with torch.no_grad():
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, h_q = self.q_net(burn_obs)
                _, h_q_target = self.q_target(burn_obs)
            else:
                h_q, h_q_target = None, None

        # 2. Main Sequence
        obs_active = obs[:, burn_in:, :]
        a_active = actions[:, burn_in:]
        r_active = rewards[:, burn_in:]
        d_active = dones[:, burn_in:]
        m_active = masks[:, burn_in:]

        q_logits_full, _ = self.q_net(obs_active, hidden=h_q)
        q_logits = q_logits_full[:, :-1, :]
        current_q = q_logits.gather(2, a_active.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_target_full, _ = self.q_target(obs_active, hidden=h_q_target)
            next_q = q_target_full[:, 1:, :].max(2)[0]
            target_q = r_active + (1.0 - d_active) * self.gamma * next_q

        # 3. Masked Loss
        q_td_loss_unmasked = F.mse_loss(current_q, target_q, reduction='none')
        
        valid_steps = m_active.sum() + 1e-8
        q_loss = (q_td_loss_unmasked * m_active).sum() / valid_steps
        
        cql_loss = torch.tensor(0.0).to(self.device_name)
        if use_cql:
            cql_loss_unmasked = torch.logsumexp(q_logits, dim=2) - current_q
            cql_loss = (cql_loss_unmasked * m_active).sum() / valid_steps
            total_loss = q_loss + self.cql_alpha * cql_loss
        else:
            total_loss = q_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "loss_td": total_loss.item(), 
            "q_loss": q_loss.item(), 
            "cql_loss": cql_loss.item(),
            "h_q": h_q # Return cached h_q for supervised update
        }

    def update_supervised(self, obs, actions, masks, burn_in=16, anti=False, advantages=None, h_q=None) -> dict:
        # 1. Burn-in Phase (only if h_q not provided)
        if h_q is None:
            with torch.no_grad():
                if burn_in > 0:
                    burn_obs = obs[:, :burn_in, :]
                    _, h_q = self.q_net(burn_obs)
                else:
                    h_q = None

        # 2. Main Sequence
        obs_active = obs[:, burn_in:-1, :] # s_burn..s_T
        a_active = actions[:, burn_in:]
        m_active = masks[:, burn_in:]

        logits, _ = self.q_net(obs_active, hidden=h_q)

        # 3. Masked Supervised Loss
        if anti:
            probs = F.softmax(logits, dim=2)
            bad_action_probs = probs.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
            loss_unmasked = -torch.log(1 - bad_action_probs + 1e-8)
        else:
            if advantages is not None:
                adv = advantages.to(self.device_name)
                log_probs = F.log_softmax(logits, dim=2)
                selected_log_probs = log_probs.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
                loss_unmasked = -(adv * selected_log_probs)
            else:
                logits_flat = logits.reshape(-1, self.action_dim)
                a_flat = a_active.reshape(-1)
                ce_unmasked = F.cross_entropy(logits_flat, a_flat, reduction='none')
                loss_unmasked = ce_unmasked.reshape(a_active.shape)

        valid_steps = m_active.sum() + 1e-8
        loss = (loss_unmasked * m_active).sum() / valid_steps

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        
        return {"loss_supervised": loss.item()}
    
    def get_bc_loss(self, obs, actions, masks, burn_in=16) -> float:
        self.q_net.eval()
        with torch.no_grad():
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, h_q = self.q_net(burn_obs)
            else:
                h_q = None

            obs_active = obs[:, burn_in:-1, :]
            a_active = actions[:, burn_in:]
            m_active = masks[:, burn_in:]

            logits, _ = self.q_net(obs_active, hidden=h_q)
            logits_flat = logits.reshape(-1, self.action_dim)
            a_flat = a_active.reshape(-1)
            
            ce_unmasked = F.cross_entropy(logits_flat, a_flat, reduction='none')
            loss_unmasked = ce_unmasked.reshape(a_active.shape)
            
            valid_steps = m_active.sum() + 1e-8
            loss = (loss_unmasked * m_active).sum() / valid_steps
            
        self.q_net.train()
        return loss.item()
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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        return {"loss_kl": kl_loss.item()}

    def to(self, device_name):
        self.device_name = device_name
        self.q_net.to(device_name)
        self.q_target.to(device_name)
        self.v_net.to(device_name)
        self.v_target.to(device_name)
        # Move optimizer states
        for state in self.q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)
        for state in self.v_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)

    def sync_from(self, source_agent):
        with torch.no_grad():
            for p, src_p in zip(self.q_net.parameters(), source_agent.q_net.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.v_net.parameters(), source_agent.v_net.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.q_target.parameters(), source_agent.q_target.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.v_target.parameters(), source_agent.v_target.parameters()):
                p.data.copy_(src_p.data)

    def _save_checkpoint(self, path):
        state = {'q_net': self.q_net.state_dict(), 'v_net': self.v_net.state_dict()}
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.v_net.load_state_dict(checkpoint['v_net'])
        self.q_target = copy.deepcopy(self.q_net)
        self.v_target = copy.deepcopy(self.v_net)
