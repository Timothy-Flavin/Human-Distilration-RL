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
    def update_td(self, batch, ssl: bool = False, masks: list = None) -> dict:
        """Strictly applies Temporal Difference (e.g. CQL/DQN) loss."""
        pass

    @abstractmethod
    def update_supervised(self, batch, ssl: bool = False, masks: list = None, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        """Strictly applies Supervised Learning (Behavior Cloning) loss."""
        pass

    @abstractmethod
    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the raw logits for the given observations."""
        pass

    @abstractmethod
    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        """Performs a KL-Divergence update towards a target agent's policy."""
        pass

    def ssl_augment(self, obs_batch: torch.Tensor, masks: list) -> torch.Tensor:
        """
        Augments observations with feature-specific noise.
        masks: A list of dicts, one per batch item. 
               Each dict: {feature_idx: {'dist': 'gaussian'|'uniform', 'scale': 0.1, 'low': x, 'high': y}}
        """
        augmented_obs = obs_batch.clone()
        for i, mask in enumerate(masks):
            if not mask:
                continue
            
            for feat_idx, spec in mask.items():
                if spec['dist'] == 'gaussian':
                    scale = spec.get('scale', 0.1)
                    # Use size=() for scalar noise
                    noise = torch.normal(0, scale, size=()).to(self.device_name)
                    augmented_obs[i, feat_idx] += noise
                elif spec['dist'] == 'uniform':
                    low = spec.get('low', -1.0)
                    high = spec.get('high', 1.0)
                    # Sample uniformly in the range [low, high]
                    val = (high - low) * torch.rand(size=()).to(self.device_name) + low
                    augmented_obs[i, feat_idx] = val
        return augmented_obs

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
    def to(self, device_name):
        """Moves the agent's models to the specified device."""
        pass

    @abstractmethod
    def sync_from(self, source_agent):
        """Copies parameters from source_agent to this agent efficiently."""
        pass

    @abstractmethod
    def load_model(self, path):
        """Loads the model weights."""
        pass
