import gymnasium as gym
from wrapper import InteractiveGymWrapper

class DummyAgent:
    """A minimal mock agent for demonstration."""
    def predict(self, obs):
        # Always forces 'MAIN ENGINE' thrust (2) in LunarLander as an example
        return 2

def main():
    # Initialize your Gymnasium environment
    # Using LunarLander as it fits the WASD / Arrows discrete control scheme cleanly
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Initialize agent
    agent = DummyAgent()

    # Wrap the environment
    wrapper = InteractiveGymWrapper(env, agent=agent, fps=30)
    
    # Start the interactive loop
    print("Controls:")
    print("  Arrows : Step backward/forward (in 'step' mode)")
    print("  Space  : Toggle continuous realtime control")
    print("  Enter  : Open/Submit note editor")
    print("  Escape : Cancel note editor")
    print("  Tab    : Toggle agent control")
    print("  R      : Reset environment")
    
    trajectory, annotations = wrapper.run()

    # Once closed, you have full access to the saved trajectory and your mapped notes
    print(f"\nSession Ended. Saved {len(trajectory)} frames and {len(annotations)} notes.")
    env.close()

if __name__ == "__main__":
    main()