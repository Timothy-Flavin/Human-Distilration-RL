import torch
import collections
import random
import numpy as np
import random

import torch
import numpy as np
import random

class FastGPUEpisodicBuffer:
    def __init__(self, max_episodes=500, max_ep_len=2600, device="cuda", obs_shape=(3, 64, 64)):
        self.max_episodes = max_episodes
        self.max_ep_len = max_ep_len
        self.device = device
        self.current_size = 0
        self.ptr = 0
        
        print(f"[Buffer] Allocating GPU memory for {max_episodes} episodes with obs shape {obs_shape}...")
        
        # Use the dynamic *obs_shape unpacking
        self.obs = torch.zeros((max_episodes, max_ep_len, *obs_shape), dtype=torch.uint8, device=device)
        
        self.actions = torch.zeros((max_episodes, max_ep_len), dtype=torch.long, device=device)
        self.rewards = torch.zeros((max_episodes, max_ep_len), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_episodes, max_ep_len), dtype=torch.float32, device=device)
        self.masks = torch.zeros((max_episodes, max_ep_len), dtype=torch.float32, device=device)
        
        self.ep_lengths = torch.zeros(max_episodes, dtype=torch.long, device=device)

    def load_expert_data(self, expert_dataset):
        """The expensive upfront operation."""
        print(f"[Buffer] Converting {len(expert_dataset)} expert episodes to GPU tensors...")
        for item in expert_dataset:
            self.add_episode(item['transitions'])
            
    def add_episode(self, transitions):
        """
        The moderately expensive interval operation (runs once every 20 mins).
        Converts a list of dicts into tensor blocks and writes directly to VRAM.
        """
        ep_len = len(transitions)
        if ep_len > self.max_ep_len - 1:
            # We need space for the final next_obs, so we truncate if strictly necessary
            ep_len = self.max_ep_len - 1
            transitions = transitions[:ep_len]

        # Extract to numpy first for fast bulk conversion
        # Ensure channel-first format (C, H, W) for PyTorch
        obs_np = np.array([t['obs'] for t in transitions])
        if obs_np.shape[-1] == 3: 
            obs_np = np.transpose(obs_np, (0, 3, 1, 2))
            
        next_obs_final = transitions[-1]['next_obs']
        if next_obs_final.shape[-1] == 3:
            next_obs_final = np.transpose(next_obs_final, (2, 0, 1))

        act_np = np.array([t['action'] for t in transitions], dtype=np.int64)
        rew_np = np.array([t['reward'] for t in transitions], dtype=np.float32)
        done_np = np.array([float(t['terminated'] or t['truncated']) for t in transitions], dtype=np.float32)

        idx = self.ptr

        # Write to GPU memory
        self.obs[idx, :ep_len] = torch.tensor(obs_np, dtype=torch.uint8, device=self.device)
        self.obs[idx, ep_len] = torch.tensor(next_obs_final, dtype=torch.uint8, device=self.device) # Store final next_obs
        
        self.actions[idx, :ep_len] = torch.tensor(act_np, device=self.device)
        self.rewards[idx, :ep_len] = torch.tensor(rew_np, device=self.device)
        self.dones[idx, :ep_len] = torch.tensor(done_np, device=self.device)
        
        # Write masks (1.0 for valid transitions, 0.0 for pre-allocated zeros)
        self.masks[idx, :ep_len] = 1.0
        # Ensure cleanup of old data if buffer wraps around
        self.masks[idx, ep_len:] = 0.0 
        self.ep_lengths[idx] = ep_len

        self.ptr = (self.ptr + 1) % self.max_episodes
        self.current_size = min(self.current_size + 1, self.max_episodes)

    def sample_batch(self, batch_size, seq_len=48):
        """
        The blazing-fast backprop operation.
        Returns tensors directly sliced from GPU memory.
        """
        # Randomly select episode indices
        ep_indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        lengths = self.ep_lengths[ep_indices]

        start_indices = []
        for L in lengths:
            valid_len = L.item()
            if valid_len <= seq_len:
                start_indices.append(0)
            else:
                start_indices.append(random.randint(0, valid_len - seq_len))
                
        # Because sequence lengths might cross the max_ep_len boundary during padded slicing,
        # we construct a grid of indices to extract exactly what we need in one vectorized gather.
        
        # Shape: (batch_size, seq_len)
        step_offsets = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
        starts_tensor = torch.tensor(start_indices, device=self.device).unsqueeze(1)
        gather_indices = starts_tensor + step_offsets
        
        # We need seq_len + 1 observations to cover s_t and s_{t+1} for the final step target Q-value
        obs_offsets = torch.arange(seq_len + 1, device=self.device).unsqueeze(0).expand(batch_size, seq_len + 1)
        obs_gather_indices = starts_tensor + obs_offsets

        # Vectorized slicing using advanced indexing
        batch_obs_uint8 = self.obs[ep_indices.unsqueeze(1), obs_gather_indices]
        batch_actions = self.actions[ep_indices.unsqueeze(1), gather_indices]
        batch_rewards = self.rewards[ep_indices.unsqueeze(1), gather_indices]
        batch_dones = self.dones[ep_indices.unsqueeze(1), gather_indices]
        batch_masks = self.masks[ep_indices.unsqueeze(1), gather_indices]

        # Convert images to float32 [0, 1] ONLY at the final moment before returning
        batch_obs = batch_obs_uint8.float() / 255.0

        return batch_obs, batch_actions, batch_rewards, batch_dones, batch_masks

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
