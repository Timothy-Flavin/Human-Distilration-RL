import torch
import collections
import random
import numpy as np

class FastGPUEpisodicBuffer:
    """
    Optimized Flat Buffer: Stores transitions consecutively to save VRAM.
    Eliminates the (max_episodes, max_ep_len) zero-padding.
    """
    def __init__(self, max_total_transitions=200000, device="cuda", obs_shape=(3, 64, 64), hidden_dim=512,
                 prioritized=False, per_alpha=0.6, per_eps=1e-3, per_burn_in=16, track_td=False):
        self.max_transitions = max_total_transitions
        self.device = device
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        print(f"[Buffer] Allocating Flat GPU memory for {max_total_transitions} transitions...")

        # 350,000 * 3 * 64 * 64 * 1 byte = ~4.3 GB
        self.obs = torch.zeros((max_total_transitions, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(max_total_transitions, dtype=torch.long, device=device)
        self.rewards = torch.zeros(max_total_transitions, dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_total_transitions, dtype=torch.float32, device=device)
        # R2D2-style stored recurrent state: (h, c) BEFORE processing obs[i].
        # fp16 halves the footprint (~200MB at 200k transitions for hidden_dim=512).
        self.hiddens = torch.zeros((max_total_transitions, 2, hidden_dim), dtype=torch.float16, device=device)

        self.ptr = 0
        self.full = False

        # Episode management: stores (start_idx, length)
        self.episode_metadata = []
        self.current_size_episodes = 0

        # --- Prioritized replay (per-transition) ---
        # priorities holds raw |TD error| + eps; 0 marks slots that must never
        # anchor a sample: dead slots (no live episode) AND the first
        # per_burn_in steps of every episode. Windows can't start before the
        # episode head, so those head steps are only ever burn-in context —
        # they can never sit in a window's active loss region, so their
        # priority would never be refreshed by update_priorities and they'd
        # stay pinned at insertion max_priority forever, starving the rest of
        # the buffer of anchors. New episodes' trainable steps enter at
        # max_priority so they are seen at least once before their real TD
        # error takes over.
        self.prioritized = prioritized
        self.per_alpha = per_alpha
        self.per_eps = per_eps
        self.per_burn_in = per_burn_in
        self.max_priority = 1.0
        if prioritized:
            self.priorities = torch.zeros(max_total_transitions, dtype=torch.float32, device=device)
            # per-transition episode bounds, for clamping sampled windows
            self.ep_start_map = torch.full((max_total_transitions,), -1, dtype=torch.long, device=device)
            self.ep_len_map = torch.zeros(max_total_transitions, dtype=torch.long, device=device)
            self._meta_by_start = {}  # start_ptr -> (ep_len, next_obs_final)

        # --- Passive per-transition TD-error store (track_td=True) ---
        # Unlike PER this never changes sampling; it just remembers the last
        # |TD error| seen for each transition during training updates, so
        # consumers (demo-start priorities) don't need a full sweep. -1 marks
        # "never measured".
        self.track_td = track_td
        if track_td:
            self.td_store = torch.full(
                (max_total_transitions,), -1.0, dtype=torch.float32, device=device
            )

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
        if self.prioritized or self.track_td:
            # Clear the FULL range of every overwritten episode so its
            # leftover transitions can never be sampled or read again.
            for m in self.episode_metadata:
                if m[0] < end_ptr and (m[0] + m[1]) > start_ptr:
                    if self.prioritized:
                        self.priorities[m[0]:m[0] + m[1]] = 0.0
                        self.ep_start_map[m[0]:m[0] + m[1]] = -1
                        self.ep_len_map[m[0]:m[0] + m[1]] = 0
                        self._meta_by_start.pop(m[0], None)
                    if self.track_td:
                        self.td_store[m[0]:m[0] + m[1]] = -1.0
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

        # Actor hidden states captured at collection time (state BEFORE processing obs).
        # Episodes without them (e.g. expert demos) get zeros; refresh_hidden_states fills them in.
        hid_stack = torch.zeros((ep_len, 2, self.hidden_dim), dtype=torch.float16, device=self.device)
        for i, t in enumerate(transitions):
            hid = t.get('hidden') if isinstance(t, dict) else None
            if hid is not None:
                hid_stack[i] = hid.to(self.device, dtype=torch.float16)
        self.hiddens[start_ptr:end_ptr] = hid_stack
        
        # Store metadata
        self.episode_metadata.append((start_ptr, ep_len, next_obs_final))
        if self.prioritized:
            self.priorities[start_ptr:end_ptr] = self.max_priority
            # Head steps can never be trained on (see __init__): keep them at
            # 0 so they never anchor. They still serve as burn-in context.
            self.priorities[start_ptr:start_ptr + min(self.per_burn_in, ep_len)] = 0.0
            self.ep_start_map[start_ptr:end_ptr] = start_ptr
            self.ep_len_map[start_ptr:end_ptr] = ep_len
            self._meta_by_start[start_ptr] = (ep_len, next_obs_final)
        if self.track_td:
            self.td_store[start_ptr:end_ptr] = -1.0
        self.ptr = end_ptr

    def _gather_windows(self, window_specs, seq_len, return_flat_idx=False):
        """window_specs: list of (start_ptr, ep_len, next_obs_final, sample_start).
        Assembles the padded (B, seq_len) batch tensors; optionally also the
        flat buffer index of every window step (-1 where padded), so callers
        can write per-transition priorities back."""
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_dones = []
        batch_masks = []
        batch_hiddens = []
        batch_flat_idx = []

        for start_ptr, ep_len, next_obs_final, sample_start in window_specs:
            actual_len = min(seq_len, ep_len)
            pad_len = seq_len - actual_len

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
            flat_idx = torch.arange(sample_start, sample_start + actual_len, device=self.device)

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

                flat_idx = torch.cat([
                    flat_idx,
                    torch.full((pad_len,), -1, dtype=torch.long, device=self.device),
                ], dim=0)

            batch_obs.append(obs)
            batch_acts.append(acts)
            batch_rews.append(rews)
            batch_dones.append(dones)
            batch_masks.append(masks)
            batch_hiddens.append(self.hiddens[sample_start])
            batch_flat_idx.append(flat_idx)

        # Stack into tensors
        obs_tensor = torch.stack(batch_obs).float() / 255.0
        acts_tensor = torch.stack(batch_acts)
        rews_tensor = torch.stack(batch_rews)
        dones_tensor = torch.stack(batch_dones)
        masks_tensor = torch.stack(batch_masks)
        hiddens_tensor = torch.stack(batch_hiddens).float()  # (B, 2, hidden_dim)

        batch = (obs_tensor, acts_tensor, rews_tensor, dones_tensor, masks_tensor, hiddens_tensor)
        if return_flat_idx:
            return batch, torch.stack(batch_flat_idx)
        return batch

    def sample_batch(self, batch_size, seq_len=48, return_info=False, per_beta=0.4, burn_in=16):
        """Uniform episode/window sampling (default), or — on a prioritized
        buffer with return_info=True — PER anchor sampling: transitions are
        drawn proportional to priority^alpha, each window is placed so its
        anchor lands in the post-burn-in region, and importance-sampling
        weights (N * P)^-beta (normalized by the batch max) correct the bias.

        Returns the usual 6-tuple, or with return_info=True:
        (*6-tuple, is_weights (B,), flat_idx (B, seq_len) with -1 at padding).
        """
        if self.prioritized and return_info:
            return self._sample_batch_per(batch_size, seq_len, per_beta, burn_in)

        if len(self.episode_metadata) < batch_size:
            # Not enough data yet
            ep_indices = random.choices(range(len(self.episode_metadata)), k=batch_size)
        else:
            ep_indices = random.sample(range(len(self.episode_metadata)), batch_size)

        specs = []
        for idx in ep_indices:
            start_ptr, ep_len, next_obs_final = self.episode_metadata[idx]
            if ep_len <= seq_len:
                sample_start = start_ptr
            else:
                sample_start = start_ptr + random.randint(0, ep_len - seq_len)
            specs.append((start_ptr, ep_len, next_obs_final, sample_start))

        if return_info:
            batch, flat_idx = self._gather_windows(specs, seq_len, return_flat_idx=True)
            weights = torch.ones(batch_size, device=self.device)
            return (*batch, weights, flat_idx)
        return self._gather_windows(specs, seq_len)

    def _sample_batch_per(self, batch_size, seq_len, per_beta, burn_in):
        # 0-priority slots hold no live episode; keep them at exactly 0 even
        # for per_alpha=0 (0**0 == 1 would resurrect them).
        probs_raw = torch.where(
            self.priorities > 0,
            self.priorities ** self.per_alpha,
            torch.zeros((), device=self.device),
        )
        if probs_raw.sum() <= 0:
            # Degenerate case (e.g. only episodes shorter than per_burn_in):
            # anchor uniformly over live slots instead.
            probs_raw = (self.ep_start_map >= 0).float()
        probs = probs_raw / probs_raw.sum()
        anchors = torch.multinomial(probs, batch_size, replacement=True)

        specs = []
        for a in anchors.tolist():
            start_ptr = int(self.ep_start_map[a].item())
            ep_len = int(self.ep_len_map[a].item())
            _, next_obs_final = self._meta_by_start[start_ptr]
            if ep_len <= seq_len:
                sample_start = start_ptr
            else:
                # Place the window so the anchor falls inside it, preferring
                # the post-burn-in region so its priority gets refreshed by
                # the very update it was sampled for.
                lo = max(start_ptr, a - seq_len + 1)
                hi = min(a - burn_in, start_ptr + ep_len - seq_len)
                if hi < lo:
                    hi = lo  # anchor too close to the episode head
                sample_start = random.randint(lo, hi)
            specs.append((start_ptr, ep_len, next_obs_final, sample_start))

        batch, flat_idx = self._gather_windows(specs, seq_len, return_flat_idx=True)

        # Importance-sampling correction for the sampling bias.
        n_valid = (self.priorities > 0).sum().clamp(min=1)
        weights = (n_valid.float() * probs[anchors]) ** (-per_beta)
        weights = weights / weights.max().clamp(min=1e-8)
        return (*batch, weights, flat_idx)

    def update_priorities(self, flat_idx, td_abs, masks=None):
        """Write |TD error| priorities for the transitions just trained on.
        flat_idx / td_abs: (B, L) aligned tensors (pass the post-burn-in
        slices); masks zeroes out padded steps. Call after each update — this
        replaces any full-sweep recomputation."""
        if not self.prioritized:
            return
        valid = flat_idx >= 0
        if masks is not None:
            valid = valid & (masks > 0)
        idx = flat_idx[valid]
        if idx.numel() == 0:
            return
        pri = td_abs[valid].float() + self.per_eps
        self.priorities[idx] = pri
        self.max_priority = max(self.max_priority, float(pri.max().item()))

    def refresh_priorities(self, q_net, q_target, gamma=0.99):
        """Optional full sweep: recompute every stored transition's priority
        via compute_td_errors (e.g. every N iterations to wash out staleness)."""
        if not self.prioritized:
            return
        deltas = self.compute_td_errors(q_net, q_target, gamma=gamma)
        for (start_ptr, ep_len, _), d in zip(self.episode_metadata, deltas):
            self.priorities[start_ptr:start_ptr + ep_len] = (
                torch.as_tensor(d, device=self.device) + self.per_eps
            )
            # Untrainable head steps stay non-anchoring (see __init__)
            self.priorities[start_ptr:start_ptr + min(self.per_burn_in, ep_len)] = 0.0

    def record_td(self, flat_idx, td_abs, masks=None):
        """Remember |TD error| for the transitions a training update just
        touched (track_td store; does not affect sampling). Same alignment
        contract as update_priorities: pass the post-burn-in slices."""
        if not self.track_td:
            return
        valid = flat_idx >= 0
        if masks is not None:
            valid = valid & (masks > 0)
        idx = flat_idx[valid]
        if idx.numel() == 0:
            return
        self.td_store[idx] = td_abs[valid].float()

    def store_td_sweep(self, deltas):
        """Seed the track_td store from a compute_td_errors sweep result."""
        if not self.track_td:
            return
        for (start_ptr, ep_len, _), d in zip(self.episode_metadata, deltas):
            self.td_store[start_ptr:start_ptr + ep_len] = torch.as_tensor(
                d, device=self.device
            )

    def td_errors_per_episode(self, fill=None):
        """Read the track_td store as per-episode |TD error| arrays (insertion
        order, matching compute_td_errors). Never-measured transitions (-1)
        are replaced with `fill`, defaulting to the mean of measured values
        (neutral weight), or 1.0 if nothing has been measured yet."""
        if not self.track_td:
            return []
        if fill is None:
            known = self.td_store[self.td_store >= 0]
            fill = float(known.mean().item()) if known.numel() > 0 else 1.0
        result = []
        for start_ptr, ep_len, _ in self.episode_metadata:
            d = self.td_store[start_ptr:start_ptr + ep_len].clone()
            d[d < 0] = fill
            result.append(d.cpu().numpy())
        return result

    @torch.no_grad()
    def refresh_hidden_states(self, q_net, cnn_chunk=2048):
        """Recomputes stored (h, c) for every transition by replaying episodes
        through the current network. Used for episodes that have no actor states
        (expert demos) or to wash out staleness. hiddens[i] = state BEFORE obs[i]."""
        if not self.episode_metadata:
            return
        was_training = q_net.training
        q_net.eval()

        metas = self.episode_metadata
        n = len(metas)
        max_len = max(m[1] for m in metas)
        lens = torch.tensor([m[1] for m in metas], device=self.device)
        starts = torch.tensor([m[0] for m in metas], device=self.device)

        # 1. Encode all frames (chunked CNN forward), padded to (n, max_len, 512)
        feats = torch.zeros((n, max_len, 512), device=self.device)
        for i, (start, ep_len, _) in enumerate(metas):
            obs = self.obs[start:start + ep_len].float() / 255.0
            for j in range(0, ep_len, cnn_chunk):
                end = min(j + cnn_chunk, ep_len)
                feats[i, j:end] = q_net.encoder(obs[j:end])

        # 2. Step the LSTM across all episodes in parallel, writing pre-step states
        h = torch.zeros((1, n, self.hidden_dim), device=self.device)
        c = torch.zeros((1, n, self.hidden_dim), device=self.device)
        for t in range(max_len):
            active = lens > t
            if not active.any():
                break
            idxs = (starts + t)[active]
            self.hiddens[idxs, 0] = h[0, active].half()
            self.hiddens[idxs, 1] = c[0, active].half()
            _, (h, c) = q_net.lstm(feats[:, t:t + 1, :], (h, c))

        if was_training:
            q_net.train()

    @torch.no_grad()
    def compute_td_errors(self, q_net, q_target, gamma=0.99, cnn_chunk=2048):
        """|1-step double-DQN TD error| for every stored transition, replayed
        from zero hiddens (like refresh_hidden_states). Returns a list of 1-D
        float32 numpy arrays, one per episode in insertion order. Used to
        prioritize demo restart points."""
        if not self.episode_metadata:
            return []
        was_training = q_net.training
        q_net.eval()
        q_target.eval()

        metas = self.episode_metadata
        n = len(metas)
        # +1 column for the final next_obs of each episode
        max_len = max(m[1] for m in metas) + 1
        lens = torch.tensor([m[1] for m in metas], device=self.device)
        action_dim = q_net.fc.out_features - 1

        q_all = {}
        for name, net in (("online", q_net), ("target", q_target)):
            feats = torch.zeros((n, max_len, 512), device=self.device)
            for i, (start, ep_len, next_obs_final) in enumerate(metas):
                obs = self.obs[start:start + ep_len].float() / 255.0
                for j in range(0, ep_len, cnn_chunk):
                    end = min(j + cnn_chunk, ep_len)
                    feats[i, j:end] = net.encoder(obs[j:end])
                nof = torch.as_tensor(
                    np.ascontiguousarray(next_obs_final),
                    dtype=torch.float32, device=self.device) / 255.0
                feats[i, ep_len] = net.encoder(nof.unsqueeze(0))[0]

            q = torch.zeros((n, max_len, action_dim), device=self.device)
            h = torch.zeros((1, n, self.hidden_dim), device=self.device)
            c = torch.zeros((1, n, self.hidden_dim), device=self.device)
            for t in range(max_len):
                if not (lens + 1 > t).any():
                    break
                out, (h, c) = net.lstm(feats[:, t:t + 1, :], (h, c))
                head = net.fc(out[:, 0])
                adv = head[:, :-1]
                adv = adv - adv.mean(dim=-1, keepdim=True)
                q[:, t] = head[:, -1:] + adv
            q_all[name] = q
            del feats

        result = []
        for i, (start, ep_len, _) in enumerate(metas):
            a = self.actions[start:start + ep_len]
            r = self.rewards[start:start + ep_len]
            d = self.dones[start:start + ep_len]
            q_sa = q_all["online"][i, :ep_len].gather(1, a.unsqueeze(1)).squeeze(1)
            a_star = q_all["online"][i, 1:ep_len + 1].argmax(dim=-1)
            q_next = q_all["target"][i, 1:ep_len + 1].gather(
                1, a_star.unsqueeze(1)).squeeze(1)
            delta = r + gamma * (1.0 - d) * q_next - q_sa
            result.append(delta.abs().float().cpu().numpy())

        if was_training:
            q_net.train()
        return result

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
