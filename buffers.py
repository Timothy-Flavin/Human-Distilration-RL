import torch
import collections
import random
import numpy as np

class FastGPUEpisodicBuffer:
    """
    Optimized Flat Buffer: Stores transitions consecutively to save VRAM.
    Eliminates the (max_episodes, max_ep_len) zero-padding.
    """
    def __init__(self, max_total_transitions=200000, device="cuda", obs_shape=(3, 64, 64)):
        self.max_transitions = max_total_transitions
        self.device = device
        self.obs_shape = obs_shape
        
        print(f"[Buffer] Allocating Flat GPU memory for {max_total_transitions} transitions...")
        
        # 350,000 * 3 * 64 * 64 * 1 byte = ~4.3 GB
        self.obs = torch.zeros((max_total_transitions, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(max_total_transitions, dtype=torch.long, device=device)
        self.rewards = torch.zeros(max_total_transitions, dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_total_transitions, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.full = False
        
        # Episode management: stores (start_idx, length)
        self.episode_metadata = []
        self.current_size_episodes = 0

    @property
    def current_size(self):
        # Compatibility with existing code expecting current_size to be episode count
        return len(self.episode_metadata)

    def add_episode(self, transitions):
        ep_len = len(transitions)
        if ep_len == 0: return
        
        # Ensure we have enough space (circular buffer logic)
        # For simplicity, if we hit the end, we wrap around and clear old episode metadata that we overwrite
        start_ptr = self.ptr
        end_ptr = self.ptr + ep_len
        
        # If this episode would exceed the buffer, we wrap around to 0
        if end_ptr > self.max_transitions:
            self.ptr = 0
            self.full = True
            start_ptr = 0
            end_ptr = ep_len
            
        if ep_len > self.max_transitions:
            # Episode is literally too big for the whole buffer (should not happen)
            transitions = transitions[:self.max_transitions-1]
            ep_len = len(transitions)
            end_ptr = ep_len

        # Remove any metadata entries that overlap with our new range
        self.episode_metadata = [m for m in self.episode_metadata if not (m[0] < end_ptr and (m[0] + m[1]) > start_ptr)]

        # Extract numpy data
        obs_np = np.array([t['obs'] for t in transitions])
        if obs_np.shape[-1] == 3: 
            obs_np = np.transpose(obs_np, (0, 3, 1, 2))
            
        next_obs_final = transitions[-1]['next_obs']
        if next_obs_final.shape[-1] == 3:
            next_obs_final = np.transpose(next_obs_final, (2, 0, 1))

        act_np = np.array([t['action'] for t in transitions], dtype=np.int64)
        rew_np = np.array([t['reward'] for t in transitions], dtype=np.float32)
        done_np = np.array([float(t['terminated'] or t['truncated']) for t in transitions], dtype=np.float32)

        # Write to GPU
        idx_range = torch.arange(start_ptr, end_ptr, device=self.device)
        self.obs[start_ptr:end_ptr] = torch.tensor(obs_np, dtype=torch.uint8, device=self.device)
        
        # We need a place for next_obs of the last transition. 
        # In a flat buffer, we can either store it in the next slot or have a separate next_obs buffer.
        # To keep it simple and compatible with seq_len sampling, we'll ensure we always have one extra slot.
        # But wait, next_obs of t is just obs of t+1. 
        # The ONLY problem is the very last transition of the episode.
        
        self.actions[start_ptr:end_ptr] = torch.tensor(act_np, device=self.device)
        self.rewards[start_ptr:end_ptr] = torch.tensor(rew_np, device=self.device)
        self.dones[start_ptr:end_ptr] = torch.tensor(done_np, device=self.device)
        
        # Store metadata
        self.episode_metadata.append((start_ptr, ep_len, next_obs_final))
        self.ptr = end_ptr

    def sample_batch(self, batch_size, seq_len=48):
        if len(self.episode_metadata) < batch_size:
            # Not enough data yet
            ep_indices = random.choices(range(len(self.episode_metadata)), k=batch_size)
        else:
            ep_indices = random.sample(range(len(self.episode_metadata)), batch_size)
            
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_dones = []
        batch_masks = []
        
        for idx in ep_indices:
            start_ptr, ep_len, next_obs_final = self.episode_metadata[idx]
            
            if ep_len <= seq_len:
                # Pad if episode is shorter than seq_len
                sample_start = start_ptr
                actual_len = ep_len
                pad_len = seq_len - ep_len
            else:
                sample_start = start_ptr + random.randint(0, ep_len - seq_len)
                actual_len = seq_len
                pad_len = 0
            
            # Observations (need actual_len + 1)
            # If we are at the end of the episode, the last next_obs is next_obs_final
            if sample_start + actual_len == start_ptr + ep_len:
                # We reached the end
                obs = self.obs[sample_start : sample_start + actual_len]
                # Append next_obs_final
                next_obs_t = torch.tensor(next_obs_final, dtype=torch.uint8, device=self.device).unsqueeze(0)
                obs = torch.cat([obs, next_obs_t], dim=0)
            else:
                obs = self.obs[sample_start : sample_start + actual_len + 1]
                
            acts = self.actions[sample_start : sample_start + actual_len]
            rews = self.rewards[sample_start : sample_start + actual_len]
            dones = self.dones[sample_start : sample_start + actual_len]
            masks = torch.ones(actual_len, device=self.device)
            
            if pad_len > 0:
                # Padding
                obs_pad = torch.zeros((pad_len, *self.obs_shape), dtype=torch.uint8, device=self.device)
                obs = torch.cat([obs, obs_pad], dim=0)
                
                acts_pad = torch.zeros(pad_len, dtype=torch.long, device=self.device)
                acts = torch.cat([acts, acts_pad], dim=0)
                
                rews_pad = torch.zeros(pad_len, dtype=torch.float32, device=self.device)
                rews = torch.cat([rews, rews_pad], dim=0)
                
                dones_pad = torch.ones(pad_len, dtype=torch.float32, device=self.device) # Pad with 'done'
                dones = torch.cat([dones, dones_pad], dim=0)
                
                masks_pad = torch.zeros(pad_len, dtype=torch.float32, device=self.device)
                masks = torch.cat([masks, masks_pad], dim=0)
                
            batch_obs.append(obs)
            batch_acts.append(acts)
            batch_rews.append(rews)
            batch_dones.append(dones)
            batch_masks.append(masks)
            
        # Stack into tensors
        obs_tensor = torch.stack(batch_obs).float() / 255.0
        acts_tensor = torch.stack(batch_acts)
        rews_tensor = torch.stack(batch_rews)
        dones_tensor = torch.stack(batch_dones)
        masks_tensor = torch.stack(batch_masks)
        
        return obs_tensor, acts_tensor, rews_tensor, dones_tensor, masks_tensor

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, obs, action, reward=0.0, next_obs=None, terminated=False, truncated=False, mask=None):
        self.buffer.append({"obs": obs, "action": action, "reward": float(reward), "next_obs": next_obs, "terminated": bool(terminated), "truncated": bool(truncated), "mask": mask})
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    def __len__(self):
        return len(self.buffer)

class LLMBuffer:
    def __init__(self):
        self.buffer = collections.deque()
    def push(self, episode_trajectory, seed, note_text, note_frame_idx, current_obs_dict):
        self.buffer.append({"episode_trajectory": episode_trajectory, "seed": seed, "note_text": note_text, "note_frame_idx": note_frame_idx, "current_obs_dict": current_obs_dict})
    def pop(self): return self.buffer.popleft() if self.buffer else None
    def is_empty(self): return len(self.buffer) == 0
    def __len__(self): return len(self.buffer)

class CurriculumBuffer:
    def __init__(self):
        self.tasks = collections.deque()
    def push(self, seed, start_frame, trajectory_length, reward_function_callable, historical_actions=None):
        self.tasks.append({"seed": seed, "start_frame": start_frame, "trajectory_length": trajectory_length, "reward_fn": reward_function_callable, "historical_actions": historical_actions})
    def pop(self): return self.tasks.popleft() if self.tasks else None
    def is_empty(self): return len(self.tasks) == 0
    def __len__(self): return len(self.tasks)

class SemiSupervisedBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, obs, action, feature_mask, termination_rule=None):
        self.buffer.append({"obs": obs, "action": action, "mask": feature_mask, "reward": 0.0, "next_obs": obs, "terminated": False, "truncated": False, "termination_rule": termination_rule})
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    def __len__(self): return len(self.buffer)

class ObservationBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, obs):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        self.buffer.append(obs)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return torch.stack(batch)
    def __len__(self): return len(self.buffer)

class DenseTorchBuffer:
    """
    Optimized Flat Buffer for non-recurrent (vector) observations.
    Pre-allocates memory for fast vectorized sampling.
    """
    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        
        print(f"[DenseBuffer] Pre-allocating {capacity} transitions (Obs Dim: {obs_dim}) on {device}...")
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0

    def add_from_buffer(self, other_buffer, num_items=None):
        """
        Efficiently copies data from another DenseTorchBuffer (e.g., CPU -> GPU).
        Uses vectorized torch slices for speed.
        """
        if num_items is None:
            num_items = other_buffer.size
        if num_items == 0: return

        # Target indices (this buffer)
        indices = (torch.arange(self.ptr, self.ptr + num_items)) % self.capacity
        # Source indices (other buffer)
        # Note: This assumes the other buffer was filled from 0 or we want its current content
        # For simplicity in collection, we'll assume we take the first num_items of 'other'
        src_indices = torch.arange(0, num_items) % other_buffer.capacity
        
        self.obs[indices] = other_buffer.obs[src_indices].to(self.device)
        self.next_obs[indices] = other_buffer.next_obs[src_indices].to(self.device)
        self.actions[indices] = other_buffer.actions[src_indices].to(self.device)
        self.rewards[indices] = other_buffer.rewards[src_indices].to(self.device)
        self.dones[indices] = other_buffer.dones[src_indices].to(self.device)
        
        self.ptr = (self.ptr + num_items) % self.capacity
        self.size = min(self.size + num_items, self.capacity)

    def add_transitions(self, transitions):
        """
        Ingests a list of transition dictionaries.
        """
        num_new = len(transitions)
        if num_new == 0: return

        # Extract and convert to numpy for bulk transfer
        obs_np = np.array([t['obs'] for t in transitions], dtype=np.float32)
        next_obs_np = np.array([t['next_obs'] for t in transitions], dtype=np.float32)
        act_np = np.array([t['action'] for t in transitions], dtype=np.int64)
        rew_np = np.array([t['reward'] for t in transitions], dtype=np.float32)
        done_np = np.array([float(t['terminated'] or t['truncated']) for t in transitions], dtype=np.float32)

        # Vectorized write to pre-allocated tensor slices
        indices = (torch.arange(self.ptr, self.ptr + num_new)) % self.capacity
        
        self.obs[indices] = torch.from_numpy(obs_np).to(self.device)
        self.next_obs[indices] = torch.from_numpy(next_obs_np).to(self.device)
        self.actions[indices] = torch.from_numpy(act_np).to(self.device)
        self.rewards[indices] = torch.from_numpy(rew_np).to(self.device)
        self.dones[indices] = torch.from_numpy(done_np).to(self.device)
        
        self.ptr = (self.ptr + num_new) % self.capacity
        self.size = min(self.size + num_new, self.capacity)

    def sample(self, batch_size):
        if self.size == 0: return None
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size
