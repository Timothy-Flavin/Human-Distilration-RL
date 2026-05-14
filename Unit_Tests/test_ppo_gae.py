import unittest
import torch
import numpy as np
from PPO import PPOAgent

class TestPPOGAE(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 8
        self.action_dim = 4
        self.agent = PPOAgent(obs_dim=self.obs_dim, action_dim=self.action_dim, gamma=0.9)

    def test_gae_terminated_vs_truncated(self):
        # Create a simple sequence of 3 steps
        # Step 0: normal
        # Step 1: normal
        # Step 2: ends
        
        obs = torch.randn(3, self.obs_dim)
        next_obs = torch.randn(3, self.obs_dim)
        rewards = torch.tensor([1.0, 1.0, 1.0])
        
        # Mock critic values: v(s) = 0 for all s
        # This means delta_t = r_t + gamma * v(s_{t+1}) - v(s_t) = r_t
        self.agent.critic = MagicMock()
        self.agent.critic.side_effect = lambda x: torch.zeros(x.shape[0], 1)

        # Case 1: Terminated at step 2
        terminateds = torch.tensor([0.0, 0.0, 1.0])
        truncateds = torch.tensor([0.0, 0.0, 0.0])
        
        adv, ret = self.agent._calculate_gae(obs, rewards, terminateds, truncateds, next_obs)
        
        # For step 2 (terminated): 
        # delta_2 = r_2 + gamma * v(next_s_2) * (1-1) - v(s_2) = 1 + 0 - 0 = 1
        # adv_2 = delta_2 = 1
        # ret_2 = adv_2 + v(s_2) = 1
        self.assertEqual(adv[2], 1.0)
        self.assertEqual(ret[2], 1.0)

        # Case 2: Truncated at step 2
        terminateds = torch.tensor([0.0, 0.0, 0.0])
        truncateds = torch.tensor([0.0, 0.0, 1.0])
        
        # Now mock critic values to be non-zero to see bootstrapping
        # v(s) = 10
        self.agent.critic.side_effect = lambda x: torch.ones(x.shape[0], 1) * 10.0
        
        adv, ret = self.agent._calculate_gae(obs, rewards, terminateds, truncateds, next_obs)
        
        # For step 2 (truncated):
        # delta_2 = r_2 + gamma * v(next_s_2) * (1-0) - v(s_2) = 1 + 0.9 * 10 - 10 = 1 + 9 - 10 = 0
        # adv_2 = delta_2 = 0
        # ret_2 = adv_2 + v(s_2) = 0 + 10 = 10
        self.assertEqual(adv[2], 0.0)
        self.assertEqual(ret[2], 10.0)
        
        # If it were terminated with v=10:
        # delta_2 = r_2 + gamma * v(next_s_2) * (1-1) - v(s_2) = 1 + 0 - 10 = -9
        # adv_2 = -9
        # ret_2 = -9 + 10 = 1
        terminateds = torch.tensor([0.0, 0.0, 1.0])
        truncateds = torch.tensor([0.0, 0.0, 0.0])
        adv, ret = self.agent._calculate_gae(obs, rewards, terminateds, truncateds, next_obs)
        self.assertEqual(adv[2], -9.0)
        self.assertEqual(ret[2], 1.0)

from unittest.mock import MagicMock

if __name__ == '__main__':
    unittest.main()
