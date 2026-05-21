import torch
import numpy as np
import unittest
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CQL import CQLAgent

class TestRefactoredPipeline(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 8
        self.action_dim = 4
        self.agent = CQLAgent(self.obs_dim, self.action_dim, name="TestAgent", save_dir="test_results", device_name="cpu")

    def test_ssl_augment_gaussian(self):
        """Verify that Gaussian noise is correctly applied."""
        obs = torch.ones((2, self.obs_dim))
        masks = [
            {0: {'dist': 'gaussian', 'scale': 0.1}, 1: {'dist': 'gaussian', 'scale': 0.5}}, # Item 0
            {4: {'dist': 'gaussian', 'scale': 0.0}} # Item 1 (no noise)
        ]
        
        # We set random seed for reproducibility
        torch.manual_seed(42)
        augmented = self.agent.ssl_augment(obs, masks)
        
        # Item 0: features 0 and 1 should have changed
        self.assertNotEqual(augmented[0, 0].item(), 1.0)
        self.assertNotEqual(augmented[0, 1].item(), 1.0)
        # Other features should remain 1.0
        self.assertEqual(augmented[0, 2].item(), 1.0)
        
        # Item 1: feature 4 had scale 0.0, so no change
        self.assertEqual(augmented[1, 4].item(), 1.0)
        self.assertTrue(torch.all(augmented[1] == 1.0))

    def test_ssl_augment_uniform(self):
        """Verify that Uniform range sampling is correctly applied."""
        obs = torch.zeros((2, self.obs_dim))
        masks = [
            {1: {'dist': 'uniform', 'low': 0.5, 'high': 1.0}}, # Item 0
            {0: {'dist': 'uniform', 'low': -1.0, 'high': -0.5}} # Item 1
        ]
        
        augmented = self.agent.ssl_augment(obs, masks)
        
        # Item 0: feature 1 should be in [0.5, 1.0]
        self.assertGreaterEqual(augmented[0, 1].item(), 0.5)
        self.assertLessEqual(augmented[0, 1].item(), 1.0)
        
        # Item 1: feature 0 should be in [-1.0, -0.5]
        self.assertGreaterEqual(augmented[1, 0].item(), -1.0)
        self.assertLessEqual(augmented[1, 0].item(), -0.5)

    def test_update_td_with_ssl(self):
        """Test that update_td accepts ssl=True and masks."""
        batch = [
            (np.ones(self.obs_dim), 1, 1.0, np.ones(self.obs_dim), False, False),
            (np.zeros(self.obs_dim), 0, 0.0, np.zeros(self.obs_dim), True, False)
        ]
        masks = [
            {0: {'dist': 'gaussian', 'scale': 0.1}},
            {1: {'dist': 'uniform', 'low': 0.1, 'high': 0.2}}
        ]
        
        # This should run without crashing
        metrics = self.agent.update_td(batch, ssl=True, masks=masks)
        self.assertIn("loss_td", metrics)

    def test_update_supervised_awbc(self):
        """Test AWBC integration."""
        batch = [
            (np.ones(self.obs_dim), 1),
            (np.zeros(self.obs_dim), 0)
        ]
        advantages = torch.tensor([5.0, 0.1], dtype=torch.float32)
        
        # This should run without crashing
        metrics = self.agent.update_supervised(batch, advantages=advantages)
        self.assertIn("loss_supervised", metrics)

if __name__ == "__main__":
    unittest.main()
