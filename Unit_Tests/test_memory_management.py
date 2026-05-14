import unittest
import torch
import numpy as np
from PPO import PPOAgent
from CQL import CQLAgent

class TestMemoryManagement(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 8
        self.action_dim = 4
        self.ppo_agent = PPOAgent(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.cql_agent = CQLAgent(obs_dim=self.obs_dim, action_dim=self.action_dim)

    def test_ppo_buffer_isolation(self):
        # 1. Store in global buffer
        obs = np.random.rand(self.obs_dim).astype(np.float32)
        next_obs = np.random.rand(self.obs_dim).astype(np.float32)
        self.ppo_agent.store_transition(obs, 1, 0.5, next_obs, False, False)
        self.assertEqual(len(self.ppo_agent.buffer), 1)
        self.assertEqual(len(self.ppo_agent.local_buffer), 0)

        # 2. Store in local buffer
        self.ppo_agent.store_local_transition(obs, 2, 1.0, next_obs, False, False)
        self.assertEqual(len(self.ppo_agent.buffer), 1)
        self.assertEqual(len(self.ppo_agent.local_buffer), 1)

        # 3. Update local (should clear local, keep global)
        # Note: PPO needs a min batch for update, but we can check if it attempts to use the right buffer
        # For PPO, rl_update with local=True uses local_buffer
        self.ppo_agent.rl_update(batch_size=1, local=True)
        self.assertEqual(len(self.ppo_agent.local_buffer), 0)
        self.assertEqual(len(self.ppo_agent.buffer), 1)

        # 4. Update global (should clear global)
        self.ppo_agent.rl_update(batch_size=1, local=False)
        self.assertEqual(len(self.ppo_agent.buffer), 0)

    def test_cql_buffer_isolation(self):
        # 1. Store in global buffer
        obs = np.random.rand(self.obs_dim).astype(np.float32)
        next_obs = np.random.rand(self.obs_dim).astype(np.float32)
        self.cql_agent.store_transition(obs, 1, 0.5, next_obs, False, False)
        self.assertEqual(len(self.cql_agent.replay_buffer), 1)
        self.assertEqual(len(self.cql_agent.local_replay_buffer), 0)

        # 2. Store in local buffer
        self.cql_agent.store_local_transition(obs, 2, 1.0, next_obs, False, False)
        self.assertEqual(len(self.cql_agent.replay_buffer), 1)
        self.assertEqual(len(self.cql_agent.local_replay_buffer), 1)

        # 3. Update local
        self.cql_agent.rl_update(batch_size=1, local=True)
        # Note: CQL sample doesn't clear the buffer, but we can verify it samples from local
        # Actually, standard DQN/CQL doesn't clear the buffer.
        # However, for curriculum, we might WANT to clear it or manage it differently.
        # But for now, the requirement is isolation.
        self.assertEqual(self.cql_replay_buffer_len(local=True), 1)
        self.assertEqual(len(self.cql_agent.replay_buffer), 1)

    def cql_replay_buffer_len(self, local=False):
        return len(self.cql_agent.local_replay_buffer if local else self.cql_agent.replay_buffer)

if __name__ == '__main__':
    unittest.main()
