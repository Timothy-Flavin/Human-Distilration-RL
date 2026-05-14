import unittest
import torch
import numpy as np
from llm_router import LLMRouter
from buffers import SemiSupervisedBuffer, CurriculumBuffer

class TestSSLMining(unittest.TestCase):
    def test_buffer_mining(self):
        cur_buf = CurriculumBuffer()
        ssl_buf = SemiSupervisedBuffer(capacity=100)
        
        # Mock global buffer with some states
        # [0] is x_pos. Rule: x_pos > 0.5
        global_buffer = MagicMock()
        global_buffer.buffer = [
            (torch.tensor([0.1, 0, 0, 0, 0, 0, 0, 0]), 0),
            (torch.tensor([0.6, 0, 0, 0, 0, 0, 0, 0]), 1),
            (torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0]), 2),
        ]
        
        router = LLMRouter(cur_buf, ssl_buf, global_buffer=global_buffer)
        
        # Mock a heuristic item
        item = {
            'note_text': "center lander", # This triggers a heuristic with a rule in my mock logic
            'current_obs_dict': {},
            'episode_trajectory': [{'action': 0}, {'obs': np.zeros(8), 'action': 1}],
            'seed': 42,
            'note_frame_idx': 1
        }
        
        # Manually trigger a heuristic with a rule for testing
        with patch.object(router, '_mock_llm_classify') as mock_classify:
            mock_classify.return_value = {
                'type': 'HEURISTIC',
                'action': 3,
                'feature_mask': [0],
                'rule': lambda o: o[0] > 0.5
            }
            router.process(item)
            
        # Should have found 2 matching states (0.6 and 0.8)
        self.assertEqual(len(ssl_buf), 2)
        self.assertEqual(ssl_buf.buffer[0]['action'], 3)

from unittest.mock import MagicMock, patch

if __name__ == '__main__':
    unittest.main()
