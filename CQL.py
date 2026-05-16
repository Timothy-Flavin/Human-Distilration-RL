import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random
import os
from Agent import Agent

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class CQLAgent(Agent):
    def __init__(self, obs_dim=8, action_dim=4, name="CQL", save_dir="./default_environment", device_name="cpu", lr=1e-3, gamma=0.99, tau=0.005, cql_weight=1.0):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        self.gamma = gamma
        self.tau = tau
        self.cql_weight = cql_weight

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net).to(self.device_name)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Internal replay buffer for standard RL
        self.replay_buffer = [] 
        self.local_replay_buffer = []
        self.max_buffer_size = 50000

    def act(self, observations: torch.Tensor, deterministic: bool = False, epsilon=0.1):
        if not deterministic and np.random.random() < epsilon:
            return torch.randint(0, self.action_dim, (observations.shape[0],), device=self.device_name)
        
        with torch.no_grad():
            q_values = self.q_net(observations)
            return torch.argmax(q_values, dim=-1)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.replay_buffer.append((obs, action, reward, next_obs, terminated, truncated))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def store_local_transition(self, obs, action, reward, next_obs, terminated, truncated):
        """Stores transition in a separate buffer for localized curriculum training."""
        self.local_replay_buffer.append((obs, action, reward, next_obs, terminated, truncated))
        if len(self.local_replay_buffer) > self.max_buffer_size:
            self.local_replay_buffer.pop(0)

    def rl_update(self, batch_size=64, local: bool = False) -> dict:
        target_buffer = self.local_replay_buffer if local else self.replay_buffer
        if len(target_buffer) < batch_size:
            return {}

        batch = random.sample(target_buffer, batch_size)
        obs, action, reward, next_obs, terminated, truncated = zip(*batch)
        
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device_name)
        action = torch.tensor(action, dtype=torch.long).to(self.device_name).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device_name).unsqueeze(1)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device_name)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device_name).unsqueeze(1)
        truncated = torch.tensor(truncated, dtype=torch.float32).to(self.device_name).unsqueeze(1)

        # Standard DQN Update
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1, keepdim=True)[0]
            target_q = reward + (1 - terminated) * self.gamma * next_q

        current_q = self.q_net(obs).gather(1, action)
        td_loss = F.mse_loss(current_q, target_q)

        # CQL Penalty
        q_logits = self.q_net(obs)
        cql_loss = (torch.logsumexp(q_logits, dim=1) - current_q.squeeze()).mean()

        loss = td_loss + self.cql_weight * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"td_loss": td_loss.item(), "cql_loss": cql_loss.item(), "total_loss": loss.item()}

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        self.q_net.train()
        obs = obs.to(self.device_name)
        q_logits = self.q_net(obs)
        
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=-1)
        
        current_log_probs = F.log_softmax(q_logits, dim=-1)
        kl_loss = F.kl_div(current_log_probs, target_probs, reduction='batchmean')
        
        self.optimizer.zero_grad()
        kl_loss.backward()
        self.optimizer.step()
        
        return {"kl_loss": kl_loss.item()}

    def supervised_update(self, obs: torch.Tensor, labels: torch.Tensor, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        self.q_net.train()
        logits = self.q_net(obs.to(self.device_name))
        labels = labels.to(self.device_name)

        if not anti:
            if advantages is not None:
                advantages = advantages.to(self.device_name)
                # Weighted cross entropy
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(log_probs.gather(1, labels.unsqueeze(1)).squeeze() * advantages).mean()
            else:
                loss = self.criterion(logits, labels)
        else:
            probs = F.softmax(logits, dim=-1)
            prob_rejected = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = -torch.log(1.0 - prob_rejected + 1e-6).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"bc_loss" if not anti else "anti_bc_loss": loss.item()}

    def ssl_update(self, batch) -> dict:
        self.q_net.train()
        total_loss = 0
        for item in batch:
            obs = item['obs'].to(self.device_name)
            action = item['action'].to(self.device_name)
            mask = item['feature_mask']
            noise_scale = 0.5
            N = 5
            aug_losses = []
            for _ in range(N):
                noisy_obs = obs.clone()
                unimportant_indices = [i for i in range(self.obs_dim) if i not in mask]
                noisy_obs[unimportant_indices] += torch.randn(len(unimportant_indices), device=self.device_name) * noise_scale
                aug_logits = self.q_net(noisy_obs.unsqueeze(0))
                aug_losses.append(self.criterion(aug_logits, action.unsqueeze(0)))
            loss = torch.stack(aug_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {"ssl_loss": total_loss / len(batch)}

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def _save_checkpoint(self, path):
        torch.save(self.q_net.state_dict(), path)

    def checkpoint_model(self, specific_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{self.name}_{specific_name if specific_name else 'latest'}.pt"
        path = os.path.join(self.save_dir, filename)
        self._save_checkpoint(path)
        print(f"[Checkpoint] Saved CQL model to {path}")

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device_name))
        self.q_target = copy.deepcopy(self.q_net)
