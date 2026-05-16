import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from Agent import Agent

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class PPOAgent(Agent):
    def __init__(self, obs_dim=8, action_dim=4, name="PPO", save_dir="./default_environment", device_name="cpu", lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, entropy_coef=0.05, gae_lambda=0.95):

        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef

        self.actor = Actor(obs_dim, action_dim).to(self.device_name)
        self.critic = Critic(obs_dim).to(self.device_name)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

        self.actor_old = Actor(obs_dim, action_dim).to(self.device_name)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Online buffer for PPO
        self.buffer = []
        self.local_buffer = []

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            logits = self.actor_old(observations)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            return action

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'terminated': terminated,
            'truncated': truncated
        })

    def store_local_transition(self, obs, action, reward, next_obs, terminated, truncated):
        """Stores transition in a separate buffer for localized curriculum training."""
        self.local_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'terminated': terminated,
            'truncated': truncated
        })

    def _calculate_gae(self, obs, rewards, terminateds, truncateds, next_obs):
        with torch.no_grad():
            values = self.critic(obs).squeeze(-1)
            next_values = self.critic(next_obs).squeeze(-1)
            
        # Ensure 1D tensors even if batch size is 1
        if values.dim() == 0:
            values = values.unsqueeze(0)
            next_values = next_values.unsqueeze(0)

        advantages = torch.zeros_like(rewards).to(self.device_name)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            # Terminated (done) vs Truncated (bootstrap)
            # If terminated, next value is 0. If truncated, we bootstrap with next_values[t]
            mask = 1.0 - terminateds[t]
            # Truncation doesn't zero out the future return, but it does end the current GAE chain calculation
            # for that segment if we consider it a boundary. However, in GAE, we usually just want to know
            # if we should bootstrap.
            delta = rewards[t] + self.gamma * next_values[t] * (1 - terminateds[t]) - values[t]
            # If truncated, we don't zero out next_values, but we might reset the GAE chain if it's a hard boundary.
            # Usually, truncated means the episode ended for time, so we DO bootstrap.
            # Terminated means the task ended, so we DO NOT bootstrap.
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - terminateds[t]) * (1 - truncateds[t]) * last_gae_lam
            
        returns = advantages + values
        return advantages, returns

    def rl_update(self, batch_size=None, local: bool = False) -> dict:
        target_buffer = self.local_buffer if local else self.buffer
        
        min_batch = batch_size if batch_size else (32 if local else 256)
        if len(target_buffer) < min_batch:
            return {}

        # Convert buffer to tensors
        obs = torch.tensor(np.array([t['obs'] for t in target_buffer]), dtype=torch.float32).to(self.device_name)
        actions = torch.tensor(np.array([t['action'] for t in target_buffer]), dtype=torch.long).to(self.device_name)
        rewards = torch.tensor(np.array([t['reward'] for t in target_buffer]), dtype=torch.float32).to(self.device_name)
        terminateds = torch.tensor(np.array([t['terminated'] for t in target_buffer]), dtype=torch.float32).to(self.device_name)
        truncateds = torch.tensor(np.array([t['truncated'] for t in target_buffer]), dtype=torch.float32).to(self.device_name)
        next_obs = torch.tensor(np.array([t['next_obs'] for t in target_buffer]), dtype=torch.float32).to(self.device_name)

        # Calculate Advantages and Returns using GAE
        advantages, returns = self._calculate_gae(obs, rewards, terminateds, truncateds, next_obs)
        
        # Normalize advantages safely
        if advantages.numel() > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-7)
            else:
                advantages = advantages - advantages.mean()

        # Optimize policy for K epochs
        total_loss = 0
        for _ in range(self.K_epochs):
            logits = self.actor(obs)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(obs).squeeze(-1)
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            with torch.no_grad():
                old_logits = self.actor_old(obs)
                old_probs = F.softmax(old_logits, dim=-1)
                old_dist = Categorical(old_probs)
                old_logprobs = old_dist.log_prob(actions)
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, returns) - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.mean().item()

        self.actor_old.load_state_dict(self.actor.state_dict())

        # Clear the specific buffer
        if local:
            self.local_buffer = []
        else:
            self.buffer = []

        return {"ppo_loss": total_loss / self.K_epochs}

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        self.actor.train()
        obs = obs.to(self.device_name)
        logits = self.actor(obs)
        
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=-1)
        
        current_log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(current_log_probs, target_probs, reduction='batchmean')
        
        self.optimizer.zero_grad()
        kl_loss.backward()
        self.optimizer.step()
        
        self.actor_old.load_state_dict(self.actor.state_dict())
        return {"kl_loss": kl_loss.item()}

    def supervised_update(self, obs: torch.Tensor, labels: torch.Tensor, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        self.actor.train()
        logits = self.actor(obs.to(self.device_name))
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
            # Anti-BC: minimize probability of these actions
            probs = F.softmax(logits, dim=-1)
            prob_rejected = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = -torch.log(1.0 - prob_rejected + 1e-6).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update old actor to match
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        return {"bc_loss" if not anti else "anti_bc_loss": loss.item()}

    def ssl_update(self, batch) -> dict:
        self.actor.train()
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
                aug_logits = self.actor(noisy_obs.unsqueeze(0))
                aug_losses.append(self.criterion(aug_logits, action.unsqueeze(0)))
            loss = torch.stack(aug_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        self.actor_old.load_state_dict(self.actor.state_dict())
        return {"ssl_loss": total_loss / len(batch)}

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def _save_checkpoint(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def checkpoint_model(self, specific_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{self.name}_{specific_name if specific_name else 'latest'}.pt"
        path = os.path.join(self.save_dir, filename)
        self._save_checkpoint(path)
        print(f"[Checkpoint] Saved PPO model to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.actor_old.load_state_dict(self.actor.state_dict())
