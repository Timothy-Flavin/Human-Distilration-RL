import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import collections
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from CQL import CQLAgent
from RCQL import RCQLAgent
from buffers import FastGPUEpisodicBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"

class MinimalCQLEnv(gym.Env):
    def __init__(self, reward_scale=1.0):
        super().__init__()
        self.reward_scale = reward_scale
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.state_type = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_type = self.np_random.integers(0, 2)
        if self.state_type == 0:
            obs = np.array([1.0, 0.0], dtype=np.float32)
        else:
            obs = np.array([0.0, 1.0], dtype=np.float32)
        return obs, {}
        
    def step(self, action):
        reward = 0.0
        if self.state_type == 0:
            if action in [0, 1]: reward = self.reward_scale
        else:
            if action in [1, 2]: reward = self.reward_scale
        return np.zeros(2, dtype=np.float32), reward, True, False, {}

class MinimalRCQLEnv(gym.Env):
    def __init__(self, reward_scale=1.0):
        super().__init__()
        self.reward_scale = reward_scale
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 64,64), dtype=np.uint8)
        self.max_steps = 4
        self.current_step = 0
        self.state_type = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state_type = self.np_random.integers(0, 2)
        obs = np.zeros((3, 64, 64), dtype=np.uint8)
        if self.state_type == 0:
            obs[0, :, :].fill(255) # Red
        else:
            obs[2, :, :].fill(255) # Blue
        return obs, {}
        
    def step(self, action):
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        reward = 0.0
        
        if terminated:
            if self.state_type == 0:
                if action in [0, 1]: reward = self.reward_scale
            else:
                if action in [1, 2]: reward = self.reward_scale
                
        obs = np.zeros((3, 64, 64), dtype=np.uint8)
        return obs, reward, terminated, False, {}

class SimpleBuffer:
    def __init__(self):
        self.data = []
    def add(self, transition):
        self.data.append(transition)
    def sample_batch(self, batch_size):
        if len(self.data) < batch_size: return None
        batch = random.sample(self.data, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32).to(device),
            torch.tensor(np.array(act), dtype=torch.long).to(device),
            torch.tensor(np.array(rew), dtype=torch.float32).to(device),
            torch.tensor(np.array(nobs), dtype=torch.float32).to(device),
            torch.tensor(np.array(done), dtype=torch.float32).to(device)
        )
    @property
    def current_size(self):
        return len(self.data)

def generate_cql_expert(num_episodes=200, reward_scale=1.0):
    env = MinimalCQLEnv(reward_scale)
    buffer = SimpleBuffer()
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action = 0 if env.state_type == 0 else 2
        next_obs, reward, term, trunc, _ = env.step(action)
        buffer.add((obs, action, reward, next_obs, term or trunc))
    return buffer

def generate_rcql_expert(num_episodes=200, reward_scale=1.0):
    env = MinimalRCQLEnv(reward_scale)
    buffer = FastGPUEpisodicBuffer(max_total_transitions=2000, device=device, obs_shape=(3, 64, 64))
    for _ in range(num_episodes):
        obs, _ = env.reset()
        term = False
        ep = []
        while not term:
            if env.current_step == env.max_steps - 1:
                action = 0 if env.state_type == 0 else 2
            else:
                action = 1
            next_obs, reward, term, trunc, _ = env.step(action)
            ep.append({'obs': obs, 'action': action, 'reward': reward, 
                       'next_obs': next_obs, 'terminated': term, 'truncated': trunc})
            obs = next_obs
        buffer.add_episode(ep)
    return buffer

def eval_cql_kl(agent):
    s0 = torch.tensor([1.0, 0.0], dtype=torch.float32).to(device).unsqueeze(0)
    s1 = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device).unsqueeze(0)
    obs = torch.cat([s0, s1], dim=0)
    with torch.no_grad():
        logits = agent.get_logits(obs)
        log_probs = F.log_softmax(logits, dim=-1)
    
    target_probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32).to(device)
    kl = F.kl_div(log_probs, target_probs, reduction='batchmean')
    return kl.item()

def eval_rcql_kl(agent):
    obs0 = np.zeros((4, 3, 64, 64), dtype=np.uint8)
    obs0[0, 0, :, :].fill(255)
    
    obs1 = np.zeros((4, 3, 64, 64), dtype=np.uint8)
    obs1[0, 2, :, :].fill(255)
    
    obs_batch = torch.tensor(np.stack([obs0, obs1]), dtype=torch.float32).to(device)
    with torch.no_grad():
        agent.reset_hidden()
        logits = agent.get_logits(obs_batch)
        final_logits = logits[:, -1, :]
        log_probs = F.log_softmax(final_logits, dim=-1)
        
    target_probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32).to(device)
    kl = F.kl_div(log_probs, target_probs, reduction='batchmean')
    return kl.item()

def run_cql_experiment(mode, seeds=10, iters=300, reward_scale=1.0):
    results = {'kl': [], 'td_loss': [], 'bc_loss': [], 'grad_norm': [], 'rl_grad_norm': []}
    for seed in range(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        env = MinimalCQLEnv(reward_scale)
        agent = CQLAgent(obs_dim=2, action_dim=3, name=f"CQL_{mode}", device_name=device)
        agent.cql_alpha = 0.0
        agent.epsilon = 0.05
        
        expert_buffer = generate_cql_expert(reward_scale=reward_scale)
        online_buffer = SimpleBuffer()
        
        online_rl = mode in ["RL", "RL_BC", "RL_Naive_BC"]
        bc = mode in ["BC", "RL_BC", "RL_Naive_BC"]
        naive_bc = mode == "RL_Naive_BC"
        
        kl_hist, td_hist, bc_hist, grad_hist, rl_grad_hist = [], [], [], [], []
        
        for i in range(iters):
            if online_rl:
                obs, _ = env.reset()
                action = agent.predict(obs, deterministic=False)
                next_obs, reward, term, trunc, _ = env.step(action)
                online_buffer.add((obs, action, reward, next_obs, term or trunc))
            
            o_batch = online_buffer.sample_batch(32) if online_rl and online_buffer.current_size >= 32 else None
            e_batch = expert_buffer.sample_batch(32) if bc and (o_batch is not None or not online_rl) else None
            
            metrics = {}
            if o_batch is not None:
                metrics1 = agent.update_td(obs=o_batch[0], actions=o_batch[1], rewards=o_batch[2], next_obs=o_batch[3], dones=o_batch[4], use_cql=False, td_scale=reward_scale)
                metrics.update(metrics1)
            if e_batch is not None:
                metrics2 = agent.update_supervised(obs=e_batch[0], labels=e_batch[1], naive=naive_bc)
                metrics.update(metrics2)

            td_loss = metrics.get('td_o', metrics.get('loss_td', 0.0))
            if isinstance(td_loss, torch.Tensor): td_loss = td_loss.item()
            bc_loss = metrics.get('bc', metrics.get('loss_supervised', 0.0))
            if isinstance(bc_loss, torch.Tensor): bc_loss = bc_loss.item()
            grad_norm = metrics.get('grad_norm', 0.0)
            if isinstance(grad_norm, torch.Tensor): grad_norm = grad_norm.item()
            rl_grad_norm = metrics.get('rl_grad_norm', 0.0)
            if isinstance(rl_grad_norm, torch.Tensor): rl_grad_norm = rl_grad_norm.item()
            
            td_hist.append(td_loss)
            bc_hist.append(bc_loss)
            grad_hist.append(grad_norm)
            rl_grad_hist.append(rl_grad_norm)
            kl_hist.append(eval_cql_kl(agent))
            
        results['kl'].append(kl_hist)
        results['td_loss'].append(td_hist)
        results['bc_loss'].append(bc_hist)
        results['grad_norm'].append(grad_hist)
        results['rl_grad_norm'].append(rl_grad_hist)
    return results

def run_rcql_experiment(mode, seeds=10, iters=300, reward_scale=1.0):
    results = {'kl': [], 'td_loss': [], 'bc_loss': [], 'grad_norm': [], 'rl_grad_norm': []}
    for seed in range(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        env = MinimalRCQLEnv(reward_scale)
        agent = RCQLAgent(obs_dim=(3, 64, 64), action_dim=3, name=f"RCQL_{mode}", device_name=device, hidden_dim=64)
        agent.cql_alpha = 0.0
        agent.epsilon = 0.05
        
        expert_buffer = generate_rcql_expert(reward_scale=reward_scale)
        online_buffer = FastGPUEpisodicBuffer(max_total_transitions=2000, device=device, obs_shape=(3, 64, 64))
        
        online_rl = mode in ["RL", "RL_BC", "RL_Naive_BC"]
        bc = mode in ["BC", "RL_BC", "RL_Naive_BC"]
        naive_bc = mode == "RL_Naive_BC"
        
        kl_hist, td_hist, bc_hist, grad_hist, rl_grad_hist = [], [], [], [], []
        
        for i in range(iters):
            if online_rl:
                obs, _ = env.reset()
                term = False
                ep = []
                agent.reset_hidden()
                while not term:
                    action = agent.predict(obs, deterministic=False)
                    next_obs, reward, term, trunc, _ = env.step(action)
                    ep.append({'obs': obs, 'action': action, 'reward': reward, 
                               'next_obs': next_obs, 'terminated': term, 'truncated': trunc})
                    obs = next_obs
                online_buffer.add_episode(ep)
                
            o_batch = online_buffer.sample_batch(16, seq_len=4) if online_rl and online_buffer.current_size > 16 else None
            e_batch = expert_buffer.sample_batch(16, seq_len=4) if bc and expert_buffer.current_size > 16 and (o_batch is not None or not online_rl) else None
            
            metrics = {}
            if o_batch is not None:
                metrics1 = agent.update_td(o_batch[0], o_batch[1], o_batch[2], o_batch[3], o_batch[4], burn_in=0, use_cql=False, td_scale=reward_scale)
                metrics.update(metrics1)
            if e_batch is not None:
                metrics2 = agent.update_supervised(e_batch[0], e_batch[1], e_batch[4], burn_in=0, naive=naive_bc)
                metrics.update(metrics2)

            td_loss = metrics.get('loss_td', 0.0)
            if isinstance(td_loss, torch.Tensor): td_loss = td_loss.item()
            bc_loss = metrics.get('loss_supervised', 0.0)
            if isinstance(bc_loss, torch.Tensor): bc_loss = bc_loss.item()
            grad_norm = metrics.get('grad_norm', 0.0)
            if isinstance(grad_norm, torch.Tensor): grad_norm = grad_norm.item()
            rl_grad_norm = metrics.get('rl_grad_norm', 0.0)
            if isinstance(rl_grad_norm, torch.Tensor): rl_grad_norm = rl_grad_norm.item()
            
            td_hist.append(td_loss)
            bc_hist.append(bc_loss)
            grad_hist.append(grad_norm)
            rl_grad_hist.append(rl_grad_norm)
            kl_hist.append(eval_rcql_kl(agent))
            
        results['kl'].append(kl_hist)
        results['td_loss'].append(td_hist)
        results['bc_loss'].append(bc_hist)
        results['grad_norm'].append(grad_hist)
        results['rl_grad_norm'].append(rl_grad_hist)
    return results

if __name__ == "__main__":
    modes = ["RL", "BC", "RL_BC", "RL_Naive_BC"]
    os.makedirs("test_results", exist_ok=True)
    
    scales = [1.0, 100.0]
    
    for scale in scales:
        print(f"\n--- Running experiments for reward scale: {scale} ---")
        print("Running CQL tests...")
        cql_res = {}
        for mode in modes:
            print(f"CQL Mode: {mode}")
            res = run_cql_experiment(mode, seeds=10, iters=300, reward_scale=scale)
            cql_res[mode] = res
            np.save(f"test_results/cql_{mode}_scale_{scale}_kl.npy", np.array(res['kl']))
            np.save(f"test_results/cql_{mode}_scale_{scale}_td.npy", np.array(res['td_loss']))
            np.save(f"test_results/cql_{mode}_scale_{scale}_bc.npy", np.array(res['bc_loss']))
            np.save(f"test_results/cql_{mode}_scale_{scale}_grad.npy", np.array(res['grad_norm']))
            np.save(f"test_results/cql_{mode}_scale_{scale}_rl_grad.npy", np.array(res['rl_grad_norm']))
            
        print("Running RCQL tests...")
        rcql_res = {}
        for mode in modes:
            print(f"RCQL Mode: {mode}")
            res = run_rcql_experiment(mode, seeds=10, iters=300, reward_scale=scale)
            rcql_res[mode] = res
            np.save(f"test_results/rcql_{mode}_scale_{scale}_kl.npy", np.array(res['kl']))
            np.save(f"test_results/rcql_{mode}_scale_{scale}_td.npy", np.array(res['td_loss']))
            np.save(f"test_results/rcql_{mode}_scale_{scale}_bc.npy", np.array(res['bc_loss']))
            np.save(f"test_results/rcql_{mode}_scale_{scale}_grad.npy", np.array(res['grad_norm']))
            np.save(f"test_results/rcql_{mode}_scale_{scale}_rl_grad.npy", np.array(res['rl_grad_norm']))
            
        fig, axs = plt.subplots(2, 5, figsize=(25, 8))
        for i, arch in enumerate(["cql", "rcql"]):
            res_dict = cql_res if arch == "cql" else rcql_res
            for mode in modes:
                kl_mean = np.mean(res_dict[mode]['kl'], axis=0)
                kl_std = np.std(res_dict[mode]['kl'], axis=0)
                axs[i, 0].plot(kl_mean, label=mode)
                axs[i, 0].fill_between(range(len(kl_mean)), kl_mean-kl_std, kl_mean+kl_std, alpha=0.2)
                
                td_mean = np.mean(res_dict[mode]['td_loss'], axis=0)
                axs[i, 1].plot(td_mean, label=mode)
                
                bc_mean = np.mean(res_dict[mode]['bc_loss'], axis=0)
                axs[i, 2].plot(bc_mean, label=mode)
                
                grad_mean = np.mean(res_dict[mode]['grad_norm'], axis=0)
                axs[i, 3].plot(grad_mean, label=mode)
                
                rl_grad_mean = np.mean(res_dict[mode]['rl_grad_norm'], axis=0)
                axs[i, 4].plot(rl_grad_mean, label=mode)
                
            axs[i, 0].set_title(f"{arch.upper()} KL Divergence (Scale: {scale})")
            axs[i, 0].set_ylabel("KL Div")
            axs[i, 0].set_xlabel("Iterations")
            axs[i, 0].legend()
            
            axs[i, 1].set_title(f"{arch.upper()} TD Loss (Scale: {scale})")
            axs[i, 1].set_xlabel("Iterations")
            
            axs[i, 2].set_title(f"{arch.upper()} BC Loss (Scale: {scale})")
            axs[i, 2].set_xlabel("Iterations")
            
            axs[i, 3].set_title(f"{arch.upper()} BC Grad Norm (Scale: {scale})")
            axs[i, 3].set_xlabel("Iterations")
            axs[i, 3].set_yscale('log')
            
            axs[i, 4].set_title(f"{arch.upper()} RL Grad Norm (Scale: {scale})")
            axs[i, 4].set_xlabel("Iterations")
            axs[i, 4].set_yscale('log')
            
        plt.tight_layout()
        plt.savefig(f"test_results/integration_results_scale_{scale}.png")
        print(f"Saved plots to test_results/integration_results_scale_{scale}.png")
