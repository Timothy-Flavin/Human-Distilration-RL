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
from ValueBC import temperature_scaled_bc_loss

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim + 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        adv = out[:, :-1]
        adv = adv - adv.mean(dim=1, keepdim=True)
        v = out[:, -1:]
        q = v + adv
        return q, v, adv

class CQLAgent(Agent):
    def __init__(self, obs_dim, action_dim, name="CQL", save_dir="results", device_name="cpu"):
        super().__init__(obs_dim, action_dim, name, save_dir, device_name)
        
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device_name)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0 
        self.epsilon = 0.05
        
        self.replay_buffer = collections.deque(maxlen=100000)

        if self.device_name == "cuda":
            try:
                print("[CQL] Attempting torch.compile for structural fusion...")
                self.q_net = torch.compile(self.q_net)
            except Exception as e:
                print(f"[CQL] torch.compile failed or not supported: {e}")

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            q_values, _, _ = self.q_net(observations)
            if deterministic:
                return q_values.argmax(dim=1)
            else:
                if random.random() < self.epsilon:
                    return torch.tensor([random.randint(0, self.action_dim-1)] * observations.shape[0]).to(self.device_name)
                return q_values.argmax(dim=1)

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.replay_buffer.append((obs, action, reward, next_obs, terminated, truncated))

    def sync_from(self, source_agent):
        with torch.no_grad():
            for p, src_p in zip(self.q_net.parameters(), source_agent.q_net.parameters()):
                p.data.copy_(src_p.data)
            for p, src_p in zip(self.q_target.parameters(), source_agent.q_target.parameters()):
                p.data.copy_(src_p.data)

    def train_iteration(self, online_batch, expert_batch, awbc=False, bc=False, online_rl=False, offline_rl=False, anti_bc=False, ssl=False):
        o_obs, o_act, o_rew, o_nobs, o_done = online_batch
        e_obs, e_act, e_rew, e_nobs, e_done = expert_batch
        
        q_obs_list = []
        if online_rl: q_obs_list.append(o_obs)
        if offline_rl or bc or awbc or anti_bc: q_obs_list.append(e_obs)
        
        if not q_obs_list: return {}
        
        q_obs_all = torch.cat(q_obs_list)
        q_logits_all, v_all, adv_all = self.q_net(q_obs_all)
        
        ptr = 0
        q_logits_o = None; q_logits_e = None
        v_e = None; adv_e = None
        if online_rl:
            q_logits_o = q_logits_all[ptr:ptr+o_obs.shape[0]]
            ptr += o_obs.shape[0]
        if offline_rl or bc or awbc or anti_bc:
            q_logits_e = q_logits_all[ptr:ptr+e_obs.shape[0]]
            v_e = v_all[ptr:ptr+e_obs.shape[0]]
            adv_e = adv_all[ptr:ptr+e_obs.shape[0]]

        qn_obs_list = []
        if online_rl: qn_obs_list.append(o_nobs)
        if offline_rl or awbc: qn_obs_list.append(e_nobs)
        
        if qn_obs_list:
            qn_obs_all = torch.cat(qn_obs_list)
            with torch.no_grad():
                qn_logits_all, vn_all, _ = self.q_target(qn_obs_all)
            
            ptr = 0
            qn_logits_o = None; qn_logits_e = None
            vn_e = None
            if online_rl:
                qn_logits_o = qn_logits_all[ptr:ptr+o_nobs.shape[0]]
                ptr += o_nobs.shape[0]
            if offline_rl or awbc:
                qn_logits_e = qn_logits_all[ptr:ptr+e_nobs.shape[0]]
                vn_e = vn_all[ptr:ptr+e_nobs.shape[0]]

        total_q_loss = torch.tensor(0.0, device=self.device_name)
        metrics = {}

        if online_rl:
            current_q = q_logits_o.gather(1, o_act.unsqueeze(1)).squeeze(1)
            next_q = qn_logits_o.max(1)[0]
            target_q = o_rew + (1 - o_done) * self.gamma * next_q
            td_loss_o = F.mse_loss(current_q, target_q)
            total_q_loss += td_loss_o
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
            if awbc:
                with torch.no_grad():
                    v_s = v_e.squeeze()
                    v_ns = vn_e.squeeze()
                    td_error = e_rew + (1 - e_done) * self.gamma * v_ns - v_s
                    awbc_adv = F.relu(td_error + 1.0)
                
                bc_loss = temperature_scaled_bc_loss(adv_e, e_act, epsilon=self.epsilon, weights=awbc_adv)
                
                if not online_rl and not offline_rl:
                    # Train the value function v head directly to avoid interference with the adv policy head
                    current_v = v_e.squeeze(1)
                    target_v = e_rew + (1 - e_done) * self.gamma * vn_e.squeeze(1)
                    td_loss_v = F.mse_loss(current_v, target_v)
                    total_q_loss += td_loss_v
                    metrics["td_v_fallback"] = td_loss_v
                    self.epsilon = 0.01
            else:
                bc_loss = temperature_scaled_bc_loss(adv_e, e_act, epsilon=self.epsilon)
                
            total_q_loss += bc_loss
            metrics["bc"] = bc_loss

        if anti_bc:
            probs = F.softmax(q_logits_e, dim=1)
            bad_action_probs = probs.gather(1, e_act.unsqueeze(1)).squeeze()
            anti_loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
            total_q_loss += anti_loss
            metrics["anti"] = anti_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        total_q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return metrics

    def update_value(self, obs, actions=None, rewards=None, next_obs=None, dones=None) -> dict:
        return {"loss_v": 0.0, "v_mean": 0.0}

    def update_td(self, obs, actions=None, rewards=None, next_obs=None, dones=None, ssl: bool = False, masks: list = None, use_cql=True) -> dict:
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

        q_logits, _, _ = self.q_net(obs)
        current_q = q_logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q, _, _ = self.q_target(next_obs)
            next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        q_loss = F.mse_loss(current_q, target_q)

        cql_loss = torch.tensor(0.0).to(self.device_name)
        if use_cql:
            cql_loss = torch.logsumexp(q_logits, dim=1).mean() - current_q.mean()

        total_loss = q_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.lerp_(param.data, self.tau)

        return {
            "loss_td": total_loss,
            "q_loss": q_loss,
            "cql_loss": cql_loss,
            "q_mean": current_q.mean()
        }

    def update_supervised(self, obs, labels=None, ssl: bool = False, masks: list = None, anti: bool = False, advantages: torch.Tensor = None) -> dict:
        if isinstance(obs, (list, collections.deque)):
            batch = obs
            obs_np, labels_np = zip(*batch)
            obs = torch.tensor(np.array(obs_np), dtype=torch.float32).to(self.device_name)
            labels = torch.tensor(labels_np, dtype=torch.long).to(self.device_name)

        if ssl and masks:
            obs = self.ssl_augment(obs, masks)

        q_logits, _, adv = self.q_net(obs)
        
        if anti:
            probs = F.softmax(q_logits, dim=1)
            bad_action_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
            loss = -torch.log(1 - bad_action_probs + 1e-8).mean()
        else:
            if advantages is not None:
                adv_tensor = advantages.to(self.device_name)
                log_probs = F.log_softmax(q_logits, dim=1)
                selected_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
                loss = -(adv_tensor * selected_log_probs).mean()
            else:
                loss = temperature_scaled_bc_loss(adv, labels, epsilon=self.epsilon)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return {"loss_supervised": loss}

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        q, _, _ = self.q_net(obs)
        return q

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, v, _ = self.q_net(obs)
        return v.squeeze()

    def kl_update(self, obs: torch.Tensor, target_agent) -> dict:
        current_logits, _, _ = self.q_net(obs)
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
        for state in self.q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_name)

    def _save_checkpoint(self, path):
        state = {'q_net': self.q_net.state_dict()}
        torch.save(state, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device_name)
        if isinstance(checkpoint, dict) and 'q_net' in checkpoint:
            self.q_net.load_state_dict(checkpoint['q_net'])
        else:
            self.q_net.load_state_dict(checkpoint)
        self.q_target = copy.deepcopy(self.q_net)
