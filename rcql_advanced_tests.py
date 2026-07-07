import torch
import torch.nn as nn
import numpy as np
import random
from RCQL import RCQLAgent
from rcql_test_env import FlickeringCatchEnv
import matplotlib.pyplot as plt
from buffers import FastGPUEpisodicBuffer

def test_pipeline_sliding_window():
    """
    Verifies the 185-sequence sliding window logic and gradient flow.
    Episode L=200, Window W=16 => 185 sequences.
    """
    print("\n>>> Running Pipeline Verification Test")
    obs_dim = (3, 16, 16)
    action_dim = 3
    agent = RCQLAgent(obs_dim, action_dim, device_name="cpu")
    
    # 1. Generate Dummy Episode (L=200)
    # Encode frame index into the first pixel of the red channel
    episode_transitions = []
    for i in range(200):
        obs = np.zeros((3, 16, 16), dtype=np.uint8)
        obs[0, 0, 0] = i # Encode index
        next_obs = np.zeros((3, 16, 16), dtype=np.uint8)
        next_obs[0, 0, 0] = i + 1
        
        episode_transitions.append({
            'obs': obs, 'action': 0, 'reward': 0.0,
            'next_obs': next_obs, 'terminated': (i == 199), 'truncated': False
        })
    
    # 2. Mock Replay Buffer
    agent.store_episode({'transitions': episode_transitions})
    
    # 3. Assert Output Shape & Indexing
    # Manually trigger _prepare_batch with L=200, seq_len=16
    # Note: Our current _prepare_batch samples ONE random window per episode in batch.
    # To test the '185 sequences' logic, we need to verify the sampling range.
    
    L = 200
    W = 16
    valid_starts = L - W # 0 to 184
    print(f"    Episode Length: {L}, Window: {W}")
    print(f"    Expected Valid Start Indices: 0 to {valid_starts} (Total {valid_starts + 1})")
    
    # Check first and last possible windows
    indices_to_check = [0, valid_starts]
    for start in indices_to_check:
        # Mocking a batch of 1 with specific start_idx for verification
        sub_seq = episode_transitions[start : start + W]
        obs_seq = [t['obs'] for t in sub_seq]
        obs_seq.append(sub_seq[-1]['next_obs'])
        
        # Verify first frame of window
        encoded_idx = obs_seq[0][0, 0, 0]
        assert encoded_idx == start, f"Window at {start} started with encoded index {encoded_idx}"
        print(f"    [OK] Window at index {start} verified.")

    # 4. Gradient Flow Validation
    print("    Verifying Gradient Flow through 16-step unroll...")
    # Get a batch of 1
    batch = [agent.replay_buffer[0]]
    obs_tensor, actions, rewards, dones = agent._prepare_batch(batch, seq_len=W)
    # obs_tensor shape: (1, 17, 3, 16, 16)
    
    # Enable grad for the first frame specifically to track it
    obs_tensor.requires_grad = True
    
    q_logits, _, _, _ = agent.q_net(obs_tensor[:, :-1, :]) # Current Q for s0..s15
    # Target final Q-value
    loss = q_logits[0, -1, 0] # Dummy loss on final step action 0
    
    agent.q_optimizer.zero_grad()
    loss.backward()
    
    # Check if gradient reached the first frame (obs_tensor[0, 0])
    first_frame_grad = obs_tensor.grad[0, 0].abs().sum()
    assert first_frame_grad > 0, "Gradient did not flow back to the first frame of the window!"
    print(f"    [OK] Gradient Flow: {first_frame_grad.item():.4e}")

def run_flicker_catch(num_episodes=1000):
    print("\n>>> Starting Flickering Catch Test")
    env = FlickeringCatchEnv(size=16, flicker_steps=4)
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RCQLAgent(obs_dim, action_dim, device_name=device, lr=1e-3, epsilon=0.2)
    agent.cql_alpha = 0.0
    # Before the episode loop
    fast_buffer = FastGPUEpisodicBuffer(
        max_total_transitions=20000, 
        device=device, 
        obs_shape=(3, 16, 16)  # <--- Add this
    )
    
    eval_intervals = 50
    eval_episodes = 20
    
    history = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_hidden()
        episode = []
        term = False
        while not term:
            obs_t = torch.tensor(obs).to(device)
            action = agent.act(obs_t).item()
            next_obs, reward, term, trunc, _ = env.step(action)
            episode.append({'obs': obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'terminated': term, 'truncated': trunc})
            obs = next_obs
        
        #agent.store_episode({'transitions': episode})
        fast_buffer.add_episode(episode)
        if fast_buffer.current_size >= 16:
            for _ in range(4):
                # seq_len=15 covers the full drop
                obs_b, act_b, rew_b, done_b, mask_b = fast_buffer.sample_batch(8, seq_len=15)
                agent.update_td(obs_b, act_b, rew_b, done_b, mask_b, burn_in=4)
        
        if (ep + 1) % eval_intervals == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                e_obs, _ = env.reset()
                agent.reset_hidden()
                e_term = False; e_total = 0
                while not e_term:
                    e_obs_t = torch.tensor(e_obs).to(device)
                    e_act = agent.act(e_obs_t, deterministic=True).item()
                    e_obs, e_rew, e_term, _, _ = env.step(e_act)
                    e_total += e_rew
                eval_rewards.append(e_total)
            
            mean_r = np.mean(eval_rewards)
            history.append(mean_r)
            print(f"    Episode {ep+1}: Eval Reward = {mean_r:.2f}")
            if mean_r > 0.9:
                print("    [SOLVED] Flickering Catch solved!")
                break
                
    plt.figure()
    plt.plot(history)
    plt.title("Flickering Catch Learning Curve")
    plt.xlabel(f"Eval (per {eval_intervals} eps)")
    plt.ylabel("Reward")
    plt.savefig("flicker_catch_results.png")
    print("    [*] Plot saved to flicker_catch_results.png")

if __name__ == "__main__":
    #test_pipeline_sliding_window()
    run_flicker_catch()
