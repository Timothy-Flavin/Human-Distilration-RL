import torch
import collections
import pickle
import numpy as np

from RCQL import RCQLAgent
from buffers import FastGPUEpisodicBuffer

# Dummy env shape for crafter
obs_shape = (3, 64, 64)
action_dim = 17

agent = RCQLAgent(obs_dim=obs_shape, action_dim=action_dim, device_name="cuda")

# load expert data
with open("expert_demonstrations_crafter.pkl", "rb") as f:
    expert_dataset = pickle.load(f)

buffer = FastGPUEpisodicBuffer(max_total_transitions=100000, obs_shape=obs_shape)

for item in expert_dataset[:5]: # just 5 episodes
    transitions = item['transitions'] if isinstance(item, dict) and 'transitions' in item else item
    buffer.add_episode(transitions)

v_obs, v_acts, _, _, v_masks = buffer.sample_batch(5, seq_len=64)

agent.q_net.eval()
with torch.no_grad():
    q_logits, _, adv_active, _ = agent.q_net(v_obs[:, 16:-1])
    a_active = v_acts[:, 16:]
    m_active = v_masks[:, 16:]
    
    # Calculate true cross entropy
    batch_size, seq_len, act_dim = adv_active.shape
    adv_flat = adv_active.reshape(-1, act_dim)
    a_flat = a_active.reshape(-1)
    m_flat = m_active.reshape(-1)
    
    valid_indices = torch.nonzero(m_flat).squeeze(-1)
    valid_adv = adv_flat[valid_indices]
    valid_act = a_flat[valid_indices]
    
    ce_loss = torch.nn.functional.cross_entropy(valid_adv, valid_act)
    print("True Cross Entropy Loss (Random Init):", ce_loss.item())
    print("Expected CE Loss:", np.log(17))

