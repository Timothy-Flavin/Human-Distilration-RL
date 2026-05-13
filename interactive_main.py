import gymnasium as gym
from wrapper import InteractiveGymWrapper
from Agent import Agent
from buffers import ReplayBuffer, LLMBuffer
from metrics import MetricsLogger
import torch

class DummyAgent(Agent):
    """A minimal mock agent for demonstration."""
    def act(self, observations: torch.Tensor, deterministic: bool = False):
        # Always forces 'MAIN ENGINE' thrust (2) in LunarLander as an example
        # Returns a tensor of shape (batch_size,)
        return torch.ones(observations.shape[0], dtype=torch.long) * 2

def main():
    # Initialize your Gymnasium environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Initialize buffers
    buffers = {
        'example': ReplayBuffer(capacity=10000),
        'anti_example': ReplayBuffer(capacity=10000),
        'llm': LLMBuffer()
    }

    # Initialize metrics
    metrics = MetricsLogger()

    # Initialize agent
    agent = DummyAgent(name="Dummy", device_name="cpu")

    # Wrap the environment
    wrapper = InteractiveGymWrapper(env, agent=agent, fps=30, buffers=buffers, metrics=metrics)

    # Start the interactive loop
    print("Controls:")
    print("  Arrows : Step backward/forward (in 'step' mode)")
    print("  Space  : Toggle continuous realtime control")
    print("  Enter  : Open/Submit note editor")
    print("  Escape : Cancel note editor")
    print("  Tab    : Toggle agent control")
    print("  R      : Reset environment")
    print("\nDecision Mode (after Space/Tab):")
    print("  A      : Accept Override")
    print("  R      : Reject Override")

    trajectory, annotations = wrapper.run()

    # Once closed, you have full access to the saved trajectory and your mapped notes
    print(f"\nSession Ended. Saved {len(trajectory)} frames and {len(annotations)} notes.")

    # Print metrics
    metrics.log_iteration()

    # Print buffer sizes
    print(f"Example Buffer: {len(buffers['example'])} samples")
    print(f"Anti-Example Buffer: {len(buffers['anti_example'])} samples")
    print(f"LLM Buffer: {len(buffers['llm'])} notes")

    env.close()

if __name__ == "__main__":
    main()