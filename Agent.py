import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random

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

class Agent():
    def __init__(self, obs_dim=8, action_dim=4, name="CQL", save_dir="./default_environment", device_name="cpu", lr=1e-3, gamma=0.99, tau=0.005, cql_weight=1.0):
        self.name = name
        self.save_dir = save_dir
        self.device_name = device_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_weight = cql_weight

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net).to(self.device_name)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # For supervised/SSL updates, we can use the same Q-network as an "actor" 
        # by treating Q-values as logits (or having a separate policy, but let's keep it simple)
        self.criterion = nn.CrossEntropyLoss()
        
        # Internal replay buffer for standard RL
        self.replay_buffer = [] 
        self.max_buffer_size = 50000

    def act(self, observations:torch.Tensor, deterministic:bool=False, epsilon=0.1):
        if not deterministic and np.random.random() < epsilon:
            return torch.randint(0, self.action_dim, (observations.shape[0],), device=self.device_name)
        
        with torch.no_grad():
            q_values = self.q_net(observations)
            return torch.argmax(q_values, dim=-1)

    def predict(self, observations):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device_name)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        action = self.act(observations, deterministic=True)
        return action.cpu().item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def rl_update(self, batch_size=64, local:bool=False)->dict:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device_name)
        action = torch.tensor(action, dtype=torch.long).to(self.device_name).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device_name).unsqueeze(1)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device_name)
        done = torch.tensor(done, dtype=torch.float32).to(self.device_name).unsqueeze(1)

        # Standard DQN Update
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1, keepdim=True)[0]
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q = self.q_net(obs).gather(1, action)
        td_loss = F.mse_loss(current_q, target_q)

        # CQL Penalty: logsumexp(Q) - Q(s, a)
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

    def supervised_update(self, obs:torch.Tensor, labels:torch.Tensor, anti:bool=False):
        self.q_net.train()
        logits = self.q_net(obs.to(self.device_name))
        labels = labels.to(self.device_name)

        if not anti:
            loss = self.criterion(logits, labels)
        else:
            probs = F.softmax(logits, dim=-1)
            prob_rejected = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = -torch.log(1.0 - prob_rejected + 1e-6).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"bc_loss" if not anti else "anti_bc_loss": loss.item()}

    def ssl_update(self, batch):
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

    def checkpoint_model(self, specific_name=None):
        import os
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = f"Agent_{specific_name if specific_name else 'latest'}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save(self.q_net.state_dict(), path)
        print(f"[Checkpoint] Saved model to {path}")

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device_name))
        self.q_target = copy.deepcopy(self.q_net)

