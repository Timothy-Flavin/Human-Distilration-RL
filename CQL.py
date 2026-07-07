import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
import os
import time
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

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CQLAgent(Agent):
    def __init__(self, obs_dim, action_dim, name="CQL", save_dir="results", device_name="cpu"):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        
        # 1. Q-Networks (for CQL/TD Loss)
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        
        # 2. Value-Network (Independent signal for Advantage Weighting)
        # Trains only on Bellman loss (no CQL regularization)
        self.v_net = ValueNetwork(obs_dim).to(self.device_name)
        self.v_target = copy.deepcopy(self.v_net)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0 
        
        self.replay_buffer = collections.deque(maxlen=100000)

        # Experimental: Compile the unified iteration for kernel fusion
        if self.device_name == "cuda":
            try:
                print("[CQL] Attempting torch.compile for structural fusion...")
                # We compile the networks to fuse their internal layers
                self.q_net = torch.compile(self.q_net)
                self.v_net = torch.compile(self.v_net)
            except Exception as e:
                print(f"[CQL] torch.compile failed or not supported: {e}")

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            q_values = self.q_net(observations)
            if deterministic:
                return q_values.argmax(dim=1)
            else:
                if random.random() < 0.05:
                    return torch.tensor([random.randint(0, self.action_dim-1)] * observations.shape[0]).to(self.device_name)
                return q_values.argmax(dim=1)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.replay_buffer.append((obs, action, reward, next_obs, terminated, truncated))

    def sync_from(self, source_agent):
        """Vectorized parameter copy to avoid heavy state_dict moves."""
        with torch.no_grad():
            for p, src_p in zip(self.q_net.parameters(), source_agent.q_net.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.v_net.parameters(), source_agent.v_net.parameters()):
                p.data.copy_(src_p.data)
            # We don't necessarily need to sync targets for collection agents, 
            # but we'll do it for completeness if they are used elsewhere.
            for p, src_p in zip(self.q_target.parameters(), source_agent.q_target.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.v_target.parameters(), source_agent.v_target.parameters()):
                p.data.copy_(src_p.data)

    def train_iteration(self, online_batch, expert_batch, awbc=False, bc=False, online_rl=False, offline_rl=False, anti_bc=False, ssl=False):
        """
        Structural Optimization: Batches multiple RL signals into unified forward/backward passes.
        Minimizes kernel launch overhead and intermediate CPU/GPU synchronization.
        """
        # 1. Prepare Data
        # online_batch: (obs, acts, rews, next_obs, dones)
        # expert_batch: (obs, acts, rews, next_obs, dones)
        
        o_obs, o_act, o_rew, o_nobs, o_done = online_batch
        e_obs, e_act, e_rew, e_nobs, e_done = expert_batch
        
        # 2. Combined Forward Pass: Q-Network
        # We need q(s,a) for TD, q(s) for CQL, q(s) for BC
        # Combine all unique observations that need Q-logits
        q_obs_list = []
        if online_rl: q_obs_list.append(o_obs)
        if offline_rl or bc or awbc or anti_bc: q_obs_list.append(e_obs)
        
        if not q_obs_list: return {}
        
        q_obs_all = torch.cat(q_obs_list)
        q_logits_all = self.q_net(q_obs_all)
        
        # Split logits back to original signals
        ptr = 0
        q_logits_o = None; q_logits_e = None
        if online_rl:
            q_logits_o = q_logits_all[ptr:ptr+o_obs.shape[0]]
            ptr += o_obs.shape[0]
        if offline_rl or bc or awbc or anti_bc:
            q_logits_e = q_logits_all[ptr:ptr+e_obs.shape[0]]

        # 3. Combined Forward Pass: Target Q-Network
        qn_obs_list = []
        if online_rl: qn_obs_list.append(o_nobs)
        if offline_rl: qn_obs_list.append(e_nobs)
        
        if qn_obs_list:
            qn_obs_all = torch.cat(qn_obs_list)
            with torch.no_grad():
                qn_logits_all = self.q_target(qn_obs_all)
            
            ptr = 0
            qn_logits_o = None; qn_logits_e = None
            if online_rl:
                qn_logits_o = qn_logits_all[ptr:ptr+o_nobs.shape[0]]
                ptr += o_nobs.shape[0]
            if offline_rl:
                qn_logits_e = qn_logits_all[ptr:ptr+e_nobs.shape[0]]

        # 4. Calculate Q-Losses
        total_q_loss = torch.tensor(0.0, device=self.device_name)
        metrics = {}

        if online_rl:
            current_q = q_logits_o.gather(1, o_act.unsqueeze(1)).squeeze(1)
            next_q = qn_logits_o.max(1)[0]
            target_q = o_rew + (1 - o_done) * self.gamma * next_q
            td_loss_o = F.mse_loss(current_q, target_q)
            cql_loss_o = torch.logsumexp(q_logits_o, dim=1).mean() - current_q.mean()
            total_q_loss += td_loss_o + self.cql_alpha * cql_loss_o
            metrics["td_o"] = td_loss_o

        if offline_rl:
            current_q = q_logits_e.gather(1, e_act.unsqueeze(1)).squeeze(1)
            next_q = qn_logits_e.max(1)[0]
            target_q = e_rew + (1 - e_done) * self.gamma * next_q
            td_loss_e = F.mse_loss(current_q, target_q)
            cql_loss_e = torch.logsumexp(q_logits_e, dim=1).mean() - current_q.mean()
            total_q_loss += td_loss_e + self.cql_alpha * cql_loss_e
            metrics["td_e"] = td_loss_e

        if bc or awbc:
            advantages = None
            if awbc:
                # We need Value for advantage. We'll do a separate V-pass or batch it too.
                # To keep it simple for now, we'll do value separately or in V-block below.
                # For AWBC Advantage: A = r + gamma*V(s') - V(s)
                # This needs V(e_obs) and V(e_nobs)
                with torch.no_grad():
                    v_s = self.v_net(e_obs).squeeze()
                    v_ns = self.v_target(e_nobs).squeeze()
                    td_error = e_rew + (1 - e_done) * self.gamma * v_ns - v_s
                    advantages = F.relu(td_error + 1.0)
            
            if advantages is not None:
                log_probs = F.log_softmax(q_logits_e, dim=1)
                selected_log_probs = log_probs.gather(1, e_act.unsqueeze(1)).squeeze()
                bc_loss = -(advantages * selected_log_probs).mean()
            else:
                bc_loss = F.cross_entropy(q_logits_e, e_act)
            total_q_loss += bc_loss
            metrics["bc"] = bc_loss

        if anti_bc:
            # We assume anti_bc data might be different, but for now we use expert_batch
            # or caller should provide a dedicated anti_expert_batch.
            # Assuming e_act are 'bad' actions for anti_bc.
            probs = F.softmax(q_logits_e, dim=1)
            bad_action_probs = probs.gather(1, e_act.unsqueeze(1)).squeeze()
            anti_loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
            total_q_loss += anti_loss
            metrics["anti"] = anti_loss

        # 5. Q-Optimizer Step
        self.q_optimizer.zero_grad(set_to_none=True)
        total_q_loss.backward()
        self.q_optimizer.step()

        # 6. Value Update
        total_v_loss = None
        if awbc:
            # Value trains on both online and expert data for robustness
            v_obs = torch.cat([o_obs, e_obs])
            v_rews = torch.cat([o_rew, e_rew])
            v_nobs = torch.cat([o_nobs, e_nobs])
            v_dones = torch.cat([o_done, e_done])
            
            current_v = self.v_net(v_obs).squeeze()
            with torch.no_grad():
                next_v = self.v_target(v_nobs).squeeze()
                target_v = v_rews + (1 - v_dones) * self.gamma * next_v
            
            total_v_loss = F.mse_loss(current_v, target_v)
            self.v_optimizer.zero_grad(set_to_none=True)
            total_v_loss.backward()
            self.v_optimizer.step()
            metrics["v"] = total_v_loss

        # 7. Soft Sync (Single pass using lerp_)
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.lerp_(p.data, self.tau)
            if total_v_loss is not None:
                for p, tp in zip(self.v_net.parameters(), self.v_target.parameters()):
                    tp.data.lerp_(p.data, self.tau)

        return metrics

    def update_value(self, obs, actions=None, rewards=None, next_obs=None, dones=None) -> dict:
        """
        Independent Value function update using standard Bellman MSE.
        Used as the baseline for advantage calculation.
        """
        if isinstance(obs, (list, collections.deque)):
            batch = obs
            obs_np, _, rewards_np, next_obs_np, terminated, truncated = zip(*batch)
            obs = torch.tensor(np.array(obs_np), dtype=torch.float32).to(self.device_name)
            rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device_name)
            next_obs = torch.tensor(np.array(next_obs_np), dtype=torch.float32).to(self.device_name)
            dones = torch.tensor(np.array(terminated) | np.array(truncated), dtype=torch.float32).to(self.device_name)

        # Standard Bellman V-learning
        current_v = self.v_net(obs).squeeze()
        with torch.no_grad():
            next_v = self.v_target(next_obs).squeeze()
            target_v = rewards + (1 - dones) * self.gamma * next_v

        v_loss = F.mse_loss(current_v, target_v)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Soft update V-target
        for param, target_param in zip(self.v_net.parameters(), self.v_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"loss_v": v_loss.item(), "v_mean": current_v.mean().item()}

    def update_td(self, obs, actions=None, rewards=None, next_obs=None, dones=None, ssl: bool = False, masks: list = None) -> dict:
        """Conservative Q-Learning TD Update."""
        if isinstance(obs, (list, collections.deque)):
            batch = obs
            obs_np, actions_np, rewards_np, next_obs_np, terminated, truncated = zip(*batch)
            obs = torch.tensor(np.array(obs_np), dtype=torch.float32).to(self.device_name)
            actions = torch.tensor(actions_np, dtype=torch.long).to(self.device_name)
            rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device_name)
            next_obs = torch.tensor(np.array(next_obs_np), dtype=torch.float32).to(self.device_name)
            dones = torch.tensor(np.array(terminated) | np.array(truncated), dtype=torch.float32).to(self.device_name)

        if ssl and masks:
            obs = self.ssl_augment(obs, masks)

        # 1. Forward Pass (Unified)
        q_logits = self.q_net(obs)
        current_q = q_logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Standard Q-Learning Loss
        q_loss = F.mse_loss(current_q, target_q)

        # 2. CQL Regularization (Uses same q_logits)
        cql_loss = torch.logsumexp(q_logits, dim=1).mean() - current_q.mean()

        total_loss = q_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        # 3. Optimized Soft Update (Using lerp_ is faster and uses fewer kernels)
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.lerp_(param.data, self.tau)

        # V-RAM Efficiency: Return tensors instead of .item() to avoid synchronization
        # The caller can .item() them later in bulk if needed.
        return {
            "loss_td": total_loss,
            "q_loss": q_loss,
            "cql_loss": cql_loss,
            "q_mean": current_q.mean()
        }

    def update_supervised(self, obs, labels=None, ssl: bool = False, masks: list = None, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        """Behavior Cloning (Cross-Entropy) Loss."""
        if isinstance(obs, (list, collections.deque)):
            batch = obs
            obs_np, labels_np = zip(*batch)
            obs = torch.tensor(np.array(obs_np), dtype=torch.float32).to(self.device_name)
            labels = torch.tensor(labels_np, dtype=torch.long).to(self.device_name)

        if ssl and masks:
            obs = self.ssl_augment(obs, masks)

        logits = self.q_net(obs)
        
        if anti:
            probs = F.softmax(logits, dim=1)
            bad_action_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
            loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
        else:
            if advantages is not None:
                # AWBC: weight cross entropy by provided advantages
                adv = advantages.to(self.device_name)
                log_probs = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
                loss = -(adv * selected_log_probs).mean()
            else:
                loss = F.cross_entropy(logits, labels)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return {"loss_supervised": loss}


    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar values from the independent V-network."""
        return self.v_net(obs).squeeze()

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        current_logits = self.q_net(obs)
        with torch.no_grad():
            target_logits = target_agent.get_logits(obs)
            target_probs = F.softmax(target_logits, dim=1)
        current_log_probs = F.log_softmax(current_logits, dim=1)
        kl_loss = F.kl_div(current_log_probs, target_probs, reduction='batchmean')
        self.q_optimizer.zero_grad()
        kl_loss.backward()
        self.q_optimizer.step()
        return {"loss_kl": kl_loss.item()}

    def to(self, device_name):
        self.device_name = device_name
        self.q_net.to(device_name)
        self.q_target.to(device_name)
        self.v_net.to(device_name)
        self.v_target.to(device_name)
        # Move optimizer states as well
        for state in self.q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)
        for state in self.v_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)

    def _save_checkpoint(self, path):
        # Save both Q and V nets
        state = {
            'q_net': self.q_net.state_dict(),
            'v_net': self.v_net.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        if isinstance(checkpoint, dict) and 'q_net' in checkpoint:
            self.q_net.load_state_dict(checkpoint['q_net'])
            if 'v_net' in checkpoint:
                self.v_net.load_state_dict(checkpoint['v_net'])
        else:
            # Legacy support
            self.q_net.load_state_dict(checkpoint)
        self.q_target = copy.deepcopy(self.q_net)
        self.v_target = copy.deepcopy(self.v_net)
