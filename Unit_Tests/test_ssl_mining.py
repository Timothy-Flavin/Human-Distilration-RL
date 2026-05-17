import unittest
import torch
import numpy as np
from llm_router import LLMRouter
from buffers import SemiSupervisedBuffer, CurriculumBuffer
from unittest.mock import MagicMock, patch

class TestSSLMining(unittest.TestCase):
    def test_buffer_mining(self):
        # NOTE: Dynamic mining moved to main.py unified update loop for stability.
        # This test now verifies that verified verification frames are committed,
        # but historical mining happens externally.
        
        cur_buf = CurriculumBuffer()
        ssl_buf = SemiSupervisedBuffer(capacity=100)
        
        router = LLMRouter(cur_buf, ssl_buf)
        
        # Mock a heuristic item
        item = {
            'note_text': "catch drift",
            'current_obs_dict': {},
            'episode_trajectory': [
                {'obs': np.zeros(8), 'action': 0}, 
                {'obs': np.zeros(8), 'action': 1}
            ],
            'seed': 42,
            'note_frame_idx': 0
        }
        
        # Mock verification trajectory (1 verified frame)
        verif_traj = [
            {'obs': np.array([0.5, 0, 0, 0, 0, 0, 0, 0]), 'action': 3, 'source': 'heuristic'}
        ]
        
        # Manually trigger a heuristic with a rule for testing
        with patch.object(router, '_mock_llm_classify') as mock_classify:
            mock_classify.return_value = {
                'type': 'HEURISTIC',
                'action': 3,
                'feature_mask': [0],
                'rule': lambda o: o[0] > 0.4,
                'phrase': 'catch drift'
            }
            router.commit(item, mock_classify.return_value, verification_trajectory=verif_traj)
            
        # Should have found 1 matching verified frame
        self.assertEqual(len(ssl_buf), 1)
        self.assertEqual(ssl_buf.buffer[0]['action'], 3)

if __name__ == '__main__':
    unittest.main()
