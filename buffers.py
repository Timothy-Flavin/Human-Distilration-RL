import torch
import collections
import random
import numpy as np

class ReplayBuffer:
    """Unified replay buffer for (obs, action, reward, next_obs, done, mask) transitions."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, reward=0.0, next_obs=None, terminated=False, truncated=False, mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        if next_obs is None:
            next_obs = obs # Fallback for pure BC data
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            
        self.buffer.append({
            "obs": obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "mask": mask # Optional dict of feature noise specs
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch

    def save(self, path):
        torch.save(list(self.buffer), path)
        print(f"[Buffer] Saved to {path}")

    def load(self, path):
        data = torch.load(path)
        self.buffer = collections.deque(data, maxlen=self.buffer.maxlen)
        print(f"[Buffer] Loaded {len(self.buffer)} items from {path}")

    def __len__(self):
        return len(self.buffer)

class EpisodicReplayBuffer:
    """Buffer to store and sample full episodes (sequences of transitions)."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, episode):
        """Episode should be a dict: {'transitions': [...], 'duration': ...}"""
        self.buffer.append(episode)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch

    def __len__(self):
        return len(self.buffer)

class LLMBuffer:
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

    def is_empty(self):
        return len(self.buffer) == 0

    def __len__(self):
        return len(self.buffer)

class CurriculumBuffer:
    def __init__(self):
        self.tasks = collections.deque()

    def push(self, seed, start_frame, trajectory_length, reward_function_callable, historical_actions=None):
        self.tasks.append({
            "seed": seed,
            "start_frame": start_frame,
            "trajectory_length": trajectory_length,
            "reward_fn": reward_function_callable,
            "historical_actions": historical_actions
        })

    def pop(self):
        return self.tasks.popleft() if self.tasks else None

    def is_empty(self):
        return len(self.tasks) == 0

    def __len__(self):
        return len(self.tasks)

class SemiSupervisedBuffer:
    """Buffer for SSL with feature masks and termination conditions."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, feature_mask, termination_rule=None):
        # We store as a full transition for integrated SSL
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        self.buffer.append({
            "obs": obs,
            "action": action,
            "mask": feature_mask, # Unified name 'mask'
            "reward": 0.0,
            "next_obs": obs,
            "terminated": False,
            "truncated": False,
            "termination_rule": termination_rule
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch

    def __len__(self):
        return len(self.buffer)

class ObservationBuffer:
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
