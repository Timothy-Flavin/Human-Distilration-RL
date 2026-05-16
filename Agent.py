from abc import ABC, abstractmethod
import torch
import numpy as np
import os

class Agent(ABC):
    def __init__(self, obs_dim, action_dim, name, save_dir, device_name):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name
        self.save_dir = save_dir
        self.device_name = device_name

    @abstractmethod
    def act(self, observations: torch.Tensor, deterministic: bool = False):
        """Returns actions for the given observations."""
        pass

    def predict(self, observations, deterministic: bool = True):
        """Convenience method for single observation prediction."""
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device_name)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        action = self.act(observations, deterministic=deterministic)
        return action.cpu().item()

    @abstractmethod
    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        """Stores a transition in the agent's internal memory (if applicable)."""
        pass

    @abstractmethod
    def rl_update(self, batch_size=64, local: bool = False, target_agent=None) -> dict:
        """Performs a Reinforcement Learning update."""
        pass

    @abstractmethod
    def supervised_update(self, obs: torch.Tensor, labels: torch.Tensor, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        """Performs a Supervised Learning (Behavior Cloning) update."""
        pass

    @abstractmethod
    def ssl_update(self, batch) -> dict:
        """Performs a Semi-Supervised Learning update."""
        pass

    @abstractmethod
    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the raw logits for the given observations."""
        pass

    @abstractmethod
    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        """Performs a KL-Divergence update towards a target agent's policy."""
        pass

    def checkpoint_model(self, specific_name=None):
        """Saves the model weights."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{self.name}_{specific_name if specific_name else 'latest'}.pt"
        path = os.path.join(self.save_dir, filename)
        self._save_checkpoint(path)
        print(f"[Checkpoint] Saved {self.name} model to {path}")

    @abstractmethod
    def _save_checkpoint(self, path):
        """Internal method to save weights to a specific path."""
        pass

    @abstractmethod
    def load_model(self, path):
        """Loads the model weights."""
        pass
