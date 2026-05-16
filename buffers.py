import torch
import collections
import random
import numpy as np

class ReplayBuffer:
    """Standard FIFO replay buffer for (obs, action) pairs."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action):
        # Ensure they are tensors or convert them
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        self.buffer.append((obs, action))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        obs, action = zip(*batch)
        return torch.stack(obs), torch.stack(action)

    def save(self, path):
        """Serializes the buffer to a file."""
        torch.save(list(self.buffer), path)
        print(f"[Buffer] Saved to {path}")

    def load(self, path):
        """Loads a serialized buffer from a file."""
        data = torch.load(path)
        self.buffer = collections.deque(data, maxlen=self.buffer.maxlen)
        print(f"[Buffer] Loaded {len(self.buffer)} items from {path}")

    def __len__(self):
        return len(self.buffer)

class LLMBuffer:
    """Queue for human notes and context to be processed by LLM."""
    def __init__(self):
        self.buffer = collections.deque()

    def push(self, episode_trajectory, seed, note_text, note_frame_idx, current_obs_dict):
        self.buffer.append({
            "episode_trajectory": episode_trajectory,
            "seed": seed,
            "note_text": note_text,
            "note_frame_idx": note_frame_idx,
            "current_obs_dict": current_obs_dict
        })

    def pop(self):
        return self.buffer.popleft() if self.buffer else None

    def __len__(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

class CurriculumBuffer:
    """Buffer for localized RL tasks with custom rewards."""
    def __init__(self):
        self.tasks = collections.deque()

    def push(self, seed, start_frame, trajectory_length, reward_function_callable, historical_actions=None):
        self.tasks.append({
            "seed": seed,
            "start_frame": start_frame,
            "trajectory_length": trajectory_length,
            "reward_fn": reward_function_callable,
            "historical_actions": historical_actions # Actions taken to reach start_frame
        })

    def pop(self):
        return self.tasks.popleft() if self.tasks else None

    def is_empty(self):
        return len(self.tasks) == 0

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)

class SemiSupervisedBuffer:
    """Buffer for SSL with feature masks."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, feature_mask):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        self.buffer.append({
            "obs": obs,
            "action": action,
            "feature_mask": feature_mask
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch

    def __len__(self):
        return len(self.buffer)

class ObservationBuffer:
    """Buffer for storing observations (e.g., for KL-divergence targets)."""
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        self.buffer.append(obs)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return torch.stack(batch)

    def __len__(self):
        return len(self.buffer)
