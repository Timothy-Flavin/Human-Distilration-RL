import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import gymnasium as gym
from wrapper import InteractiveGymWrapper

class TestInteractiveWrapper(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(spec=gym.Env)
        self.env.unwrapped = MagicMock()
        self.env.unwrapped.get_state.return_value = None # Force O(N) replay
        self.env.render.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        self.env.reset.return_value = (np.zeros(8), {})
        self.env.step.return_value = (np.zeros(8), 0.0, False, False, {})
        
        # Mock pygame to avoid window creation
        self.pygame_patcher = patch('wrapper.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        self.wrapper = InteractiveGymWrapper(self.env)

    def tearDown(self):
        self.pygame_patcher.stop()

    def test_record_step(self):
        self.wrapper.reset_env()
        self.assertEqual(len(self.wrapper.trajectory), 1)
        self.assertEqual(self.wrapper.current_frame_idx, 0)

        self.wrapper.step_forward(action=2)
        self.assertEqual(len(self.wrapper.trajectory), 2)
        self.assertEqual(self.wrapper.current_frame_idx, 1)
        self.assertEqual(self.wrapper.trajectory[1]['action'], 2)

    def test_restore_state_deterministic(self):
        # Setup a trajectory
        self.wrapper.reset_env() # idx 0, obs_0
        self.env.step.return_value = (np.ones(8), 1.0, False, False, {})
        self.wrapper.step_forward(action=1) # idx 1, obs_1, action 1
        self.env.step.return_value = (np.ones(8)*2, 2.0, False, False, {})
        self.wrapper.step_forward(action=2) # idx 2, obs_2, action 2

        # Reset call count
        self.env.reset.reset_mock()
        self.env.step.reset_mock()

        # Restore to frame 1
        self.wrapper._restore_state(1)
        
        # Should call reset once, then step(1) once to reach frame 1
        self.env.reset.assert_called_once()
        self.env.step.assert_called_once_with(1)

    def test_branching_and_anti_bc(self):
        self.wrapper.buffers = {
            'example': MagicMock(),
            'anti_example': MagicMock()
        }
        
        # Setup trajectory: frame 0, 1, 2
        self.wrapper.reset_env()
        self.wrapper.step_forward(1)
        self.wrapper.step_forward(2)
        
        # Move back to frame 1 and branch
        self.wrapper.current_frame_idx = 1
        self.wrapper._branch_timeline(source="realtime")
        
        # Verify discarded trajectory has the action from frame 2
        self.assertEqual(len(self.wrapper.discarded_trajectory), 1)
        self.assertEqual(self.wrapper.discarded_trajectory[0]['action'], 2)
        
        # Accept decision
        self.wrapper._handle_decision("accept")
        
        # Verify anti_example buffer received the rejected action
        # The rejected action was at idx 2, which led from obs at idx 1.
        self.wrapper.buffers['anti_example'].push.assert_called()

if __name__ == '__main__':
    unittest.main()
