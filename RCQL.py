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
from ValueBC import temperature_scaled_bc_loss

class NatureCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=64):
        super(NatureCNNEncoder, self).__init__()
        # Standard Nature CNN convolutional stack
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flatten size dynamically (1024 for 64x64 inputs)
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        with torch.no_grad():
            dummy_out = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_size = dummy_out.numel()
            
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flatten_size, 512)

    def forward(self, x):
        # Swapped from ReLU to ELU to prevent dead units during BPTT
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.flatten(x)
        x = F.elu(self.fc(x))
        return x

class RecurrentQNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=3, img_size=64, hidden_dim=512):
        super(RecurrentQNetwork, self).__init__()
        # Injecting the Nature CNN encoder
        self.encoder = NatureCNNEncoder(in_channels, img_size)
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim + 1)

    def forward(self, x, hidden=None, features=None):
        batch_size, seq_len, c, h, w = x.size()
        
        if features is None:
            # Flatten batch and sequence for the CNN forward pass
            x_flat = x.reshape(batch_size * seq_len, c, h, w)
            features = self.encoder(x_flat)
            features = features.reshape(batch_size, seq_len, -1)
        
        lstm_out, hidden = self.lstm(features, hidden)
        out = self.fc(lstm_out)
        
        # Dueling Q-Network stream extraction
        adv = out[:, :, :-1]
        adv = adv - adv.mean(dim=-1, keepdim=True)
        v = out[:, :, -1:]
        q = v + adv
        
        return q, v, adv, hidden


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
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 0.5 
        
        self.q_hidden = None
        self.replay_buffer = collections.deque(maxlen=1000)

    def reset_hidden(self):
        self.q_hidden = None

    def _normalize(self, obs):
        if obs.dtype == torch.uint8 or obs.max() > 1.0:
            return obs.float() / 255.0
        return obs.float()

    def _ensure_channel_first(self, x):
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
            q_values, _, _, self.q_hidden = self.q_net(obs, self.q_hidden)

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
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device_name)
        
        action = self.act(observations, deterministic=deterministic)
        return action.item() if action.numel() == 1 else action.cpu().numpy()

    def store_episode(self, episode):
        self.replay_buffer.append(episode)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    def update_value(self, obs, actions, rewards, dones, masks, burn_in=16, train=False) -> dict:
        if burn_in > 0:
            with torch.no_grad():
                burn_obs = obs[:, :burn_in, :]
                _, _, _, h_q = self.q_net(burn_obs)
                _, _, _, h_q_target = self.q_target(burn_obs)
        else:
            h_q, h_q_target = None, None

        obs_active = obs[:, burn_in:, :]
        
        if train:
            _, v_full, _, _ = self.q_net(obs_active, hidden=h_q)
            current_v = v_full[:, :-1, :].squeeze(-1)
            
            with torch.no_grad():
                _, v_target_full, _, _ = self.q_target(obs_active, hidden=h_q_target)
                next_v = v_target_full[:, 1:, :].squeeze(-1)
            
            r_active = rewards[:, burn_in:]
            d_active = dones[:, burn_in:]
            m_active = masks[:, burn_in:]
            
            target_v = r_active + (1.0 - d_active) * self.gamma * next_v
            loss_v_unmasked = F.mse_loss(current_v, target_v, reduction='none')
            
            valid_steps = m_active.sum() + 1e-8
            loss_v = (loss_v_unmasked * m_active).sum() / valid_steps
            
            self.q_optimizer.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.q_optimizer.step()
            
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            loss_v_val = loss_v.item()
            self.epsilon = 0.01
        else:
            with torch.no_grad():
                _, v_full, _, _ = self.q_net(obs_active, hidden=h_q)
                current_v = v_full[:, :-1, :].squeeze(-1)
                
                _, v_target_full, _, _ = self.q_target(obs_active, hidden=h_q_target)
                next_v = v_target_full[:, 1:, :].squeeze(-1)
            loss_v_val = 0.0

        return {
            "loss_v": loss_v_val, 
            "v_mean": current_v.mean().item(),
            "current_v": current_v.detach(),
            "next_v": next_v.detach()
        }

    def update_td(self, obs, actions, rewards, dones, masks, burn_in=16, use_cql=True, td_scale=1.0) -> dict:
        with torch.no_grad():
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, _, _, h_q = self.q_net(burn_obs)
                _, _, _, h_q_target = self.q_target(burn_obs)
            else:
                h_q, h_q_target = None, None

        obs_active = obs[:, burn_in:, :]
        a_active = actions[:, burn_in:]
        r_active = rewards[:, burn_in:]
        d_active = dones[:, burn_in:]
        m_active = masks[:, burn_in:]

        q_logits_full, v_full, _, _ = self.q_net(obs_active, hidden=h_q)
        q_logits = q_logits_full[:, :-1, :]
        current_q = q_logits.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
        current_v = v_full[:, :-1, :].squeeze(-1)

        with torch.no_grad():
            q_target_full, v_target_full, _, _ = self.q_target(obs_active, hidden=h_q_target)
            next_q = q_target_full[:, 1:, :].max(2)[0]
            next_v = v_target_full[:, 1:, :].squeeze(-1)
            target_q = r_active + (1.0 - d_active) * self.gamma * next_q

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
        (total_loss / (td_scale ** 2)).backward()
        
        rl_grad_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                rl_grad_norm += param_norm.item() ** 2
        rl_grad_norm = rl_grad_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "loss_td": total_loss.item(), 
            "q_loss": q_loss.item(), 
            "cql_loss": cql_loss.item(),
            "h_q": h_q,
            "current_v": current_v.detach(),
            "next_v": next_v.detach()
        }

    def update_supervised(self, obs, actions, masks, burn_in=16, anti=False, advantages=None, h_q=None, naive=False) -> dict:
        if h_q is None:
            with torch.no_grad():
                if burn_in > 0:
                    burn_obs = obs[:, :burn_in, :]
                    _, _, _, h_q = self.q_net(burn_obs)
                else:
                    h_q = None

        obs_active = obs[:, burn_in:-1, :]
        a_active = actions[:, burn_in:]
        m_active = masks[:, burn_in:]

        q_logits, _, adv_active, _ = self.q_net(obs_active, hidden=h_q)

        if anti:
            probs = F.softmax(q_logits, dim=2)
            bad_action_probs = probs.gather(2, a_active.unsqueeze(-1)).squeeze(-1)
            loss_unmasked = -torch.log(1 - bad_action_probs + 1e-8)
            
            valid_steps = m_active.sum() + 1e-8
            loss = (loss_unmasked * m_active).sum() / valid_steps

            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.q_optimizer.step()
            
            return {"loss_supervised": loss.item()}
        else:
            if advantages is not None:
                adv = advantages.to(self.device_name)
                batch_size, seq_len, act_dim = adv_active.shape
                adv_flat = adv_active.reshape(-1, act_dim)
                a_flat = a_active.reshape(-1)
                m_flat = m_active.reshape(-1)
                adv_weight_flat = adv.reshape(-1)
                
                valid_indices = torch.nonzero(m_flat).squeeze(-1)
                if valid_indices.numel() > 0:
                    valid_act = a_flat[valid_indices]
                    if naive:
                        q_logits_flat = q_logits.reshape(-1, act_dim)
                        valid_q = q_logits_flat[valid_indices]
                        loss = F.cross_entropy(valid_q, valid_act)
                    else:
                        valid_adv = adv_flat[valid_indices]
                        valid_weights = adv_weight_flat[valid_indices]
                        loss = temperature_scaled_bc_loss(valid_adv, valid_act, epsilon=self.epsilon, weights=valid_weights)
                else:
                    loss = torch.tensor(0.0).to(self.device_name)

                self.q_optimizer.zero_grad()
                grad_norm = 0.0
                if loss.requires_grad:
                    loss.backward()
                    for p in self.q_net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                    self.q_optimizer.step()
                
                return {"loss_supervised": loss.item() if hasattr(loss, 'item') else 0.0, "grad_norm": grad_norm}
            else:
                batch_size, seq_len, act_dim = adv_active.shape
                adv_flat = adv_active.reshape(-1, act_dim)
                a_flat = a_active.reshape(-1)
                m_flat = m_active.reshape(-1)
                
                valid_indices = torch.nonzero(m_flat).squeeze(-1)
                if valid_indices.numel() > 0:
                    valid_act = a_flat[valid_indices]
                    if naive:
                        q_logits_flat = q_logits.reshape(-1, act_dim)
                        valid_q = q_logits_flat[valid_indices]
                        loss = F.cross_entropy(valid_q, valid_act)
                    else:
                        valid_adv = adv_flat[valid_indices]
                        loss = temperature_scaled_bc_loss(valid_adv, valid_act, epsilon=self.epsilon)
                else:
                    loss = torch.tensor(0.0).to(self.device_name)
                
                self.q_optimizer.zero_grad()
                grad_norm = 0.0
                if loss.requires_grad:
                    loss.backward()
                    for p in self.q_net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                    self.q_optimizer.step()
                
                return {"loss_supervised": loss.item() if hasattr(loss, 'item') else 0.0, "grad_norm": grad_norm}
    
    def get_bc_loss(self, obs, actions, masks, burn_in=16) -> float:
        self.q_net.eval()
        with torch.no_grad():
            if burn_in > 0:
                burn_obs = obs[:, :burn_in, :]
                _, _, _, h_q = self.q_net(burn_obs)
            else:
                h_q = None

            obs_active = obs[:, burn_in:-1, :]
            a_active = actions[:, burn_in:]
            m_active = masks[:, burn_in:]

            _, _, adv_active, _ = self.q_net(obs_active, hidden=h_q)
            
            batch_size, seq_len, act_dim = adv_active.shape
            adv_flat = adv_active.reshape(-1, act_dim)
            a_flat = a_active.reshape(-1)
            m_flat = m_active.reshape(-1)
            
            valid_indices = torch.nonzero(m_flat).squeeze(-1)
            if valid_indices.numel() > 0:
                valid_adv = adv_flat[valid_indices]
                valid_act = a_flat[valid_indices]
                loss = F.cross_entropy(valid_adv, valid_act)
            else:
                loss = torch.tensor(0.0).to(self.device_name)
            
        self.q_net.train()
        return loss.item() if hasattr(loss, 'item') else 0.0

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        logits, _, _, _ = self.q_net(obs)
        return logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        _, v, _, _ = self.q_net(obs)
        return v.squeeze(-1)

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
        obs = self._normalize(obs)
        current_logits, _, _, _ = self.q_net(obs)
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
        for state in self.q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)

    def sync_from(self, source_agent):
        with torch.no_grad():
            for p, src_p in zip(self.q_net.parameters(), source_agent.q_net.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.q_target.parameters(), source_agent.q_target.parameters()):
                p.data.copy_(src_p.data)

    def _save_checkpoint(self, path):
        state = {'q_net': self.q_net.state_dict()}
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        if isinstance(checkpoint, dict) and 'q_net' in checkpoint:
            self.q_net.load_state_dict(checkpoint['q_net'])
        else:
            self.q_net.load_state_dict(checkpoint)
        self.q_target = copy.deepcopy(self.q_net)
