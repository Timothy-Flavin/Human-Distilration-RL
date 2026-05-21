import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
import os
from Agent import Agent

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CQLAgent(Agent):
    def __init__(self, obs_dim, action_dim, name="CQL", save_dir="results", device_name="cpu"):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0 # CQL regularization weight
        
        # Standard Replay Buffer for Offline/Online transitions
        self.replay_buffer = collections.deque(maxlen=100000)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            q_values = self.q_net(observations)
            if deterministic:
                return q_values.argmax(dim=1)
            else:
                # Simple epsilon-greedy for online exploration
                if random.random() < 0.05:
                    return torch.tensor([random.randint(0, self.action_dim-1)] * observations.shape[0]).to(self.device_name)
                return q_values.argmax(dim=1)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.replay_buffer.append((obs, action, reward, next_obs, terminated, truncated))

    def update_td(self, batch, ssl: bool = False, masks: list = None) -> dict:
        """
        Conservative Q-Learning TD Update.
        batch: List of tuples (obs, action, reward, next_obs, terminated, truncated)
        """
        obs, actions, rewards, next_obs, terminated, truncated = zip(*batch)
        
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device_name)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device_name)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device_name)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device_name)
        dones = torch.tensor(np.array(terminated) | np.array(truncated), dtype=torch.float32).to(self.device_name)

        if ssl and masks:
            obs = self.ssl_augment(obs, masks)

        # 1. Standard Q-Learning Loss
        current_q = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1)[0]
            # Boostrap on truncated, zero on terminated (handled by main.py but double checked here)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        q_loss = F.mse_loss(current_q, target_q)

        # 2. CQL Regularization (Push down Q-values of all actions, push up Q-values of dataset actions)
        q_logits = self.q_net(obs)
        dataset_q = q_logits.gather(1, actions.unsqueeze(1)).squeeze()
        
        # logsumexp(Q(s, a)) - Q(s, a_dataset)
        cql_loss = torch.logsumexp(q_logits, dim=1).mean() - dataset_q.mean()

        total_loss = q_loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "loss_td": total_loss.item(),
            "q_loss": q_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": dataset_q.mean().item()
        }

    def update_supervised(self, batch, ssl: bool = False, masks: list = None, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        """
        Behavior Cloning (Cross-Entropy) Loss.
        batch: List of tuples (obs, action)
        """
        obs, labels = zip(*batch)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device_name)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device_name)

        if ssl and masks:
            obs = self.ssl_augment(obs, masks)

        logits = self.q_net(obs)
        
        if anti:
            # For anti-BC, we want to MINIMIZE the probability of the discarded action
            # Loss = - log(1 - prob(bad_action)) = - log(sum(probs_of_other_actions))
            probs = F.softmax(logits, dim=1)
            bad_action_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
            loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
        else:
            if advantages is not None:
                # AWBC: weight cross entropy by advantages
                adv = advantages.to(self.device_name)
                # Ensure labels are indices
                log_probs = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
                loss = -(adv * selected_log_probs).mean()
            else:
                loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss_supervised": loss.item()}

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        """Pulls this agent's policy towards the target_agent's policy using KL."""
        current_logits = self.q_net(obs)
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=1)
        
        current_log_probs = F.log_softmax(current_logits, dim=1)
        # KL(Target || Current)
        kl_loss = F.kl_div(current_log_probs, target_probs, reduction='batchmean')
        
        self.optimizer.zero_grad()
        kl_loss.backward()
        self.optimizer.step()
        
        return {"loss_kl": kl_loss.item()}

    def _save_checkpoint(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device_name))
        self.q_target = copy.deepcopy(self.q_net)
