import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MemorySanityEnv(gym.Env):
    """
    A sanity-check environment to test CNN + LSTM architectures.
    Episode length is strictly 8 steps. 
    Reward is only given at the final step based on the initial observation.
    Now uses RGB images (3, 16, 16) with values in [0, 255].
    """
    metadata = {"render_modes": []}

    def __init__(self, mode="stochastic", img_size=16):
        super().__init__()
        # modes: "always_blue" or "stochastic"
        self.mode = mode 
        self.max_steps = 8
        self.current_step = 0
        self.target_action = None
        self.img_size = img_size

        # 3 Actions: 0 (if Red), 1 (if Blue), 2 (if Noise)
        self.action_space = spaces.Discrete(3)

        # Observation: (Channels, Height, Width)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(3, self.img_size, self.img_size), 
            dtype=np.uint8
        )

    def _get_noise(self):
        return self.np_random.integers(
            0, 256, (3, self.img_size, self.img_size), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.mode == "always_blue":
            self.target_action = 1
            obs = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
            obs[2, :, :].fill(255) # Blue channel
        else: # stochastic mode
            rand_val = self.np_random.random()
            if rand_val < 0.333:
                self.target_action = 0  # Red
                obs = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
                obs[0, :, :].fill(255) # Red channel
            elif rand_val < 0.666:
                self.target_action = 1  # Blue
                obs = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
                obs[2, :, :].fill(255) # Blue channel
            else:
                self.target_action = 2  # Noise
                obs = self._get_noise()

        return obs, {}

    def step(self, action):
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        reward = 0.0

        # Sparse reward at the end of the episode
        if terminated:
            if action == self.target_action:
                reward = 1.0
            else:
                reward = -1.0

        # All intermediate states are purely random noise, 
        # forcing the LSTM to remember step 0.
        obs = self._get_noise()

        return obs, reward, terminated, truncated, {}
