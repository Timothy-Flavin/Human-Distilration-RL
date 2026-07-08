import torch
import os
import cv2
import glob
from recurrent_main import CrafterGymnasiumWrapper
from RCQL import RCQLAgent

def play_latest_crafter_episode():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = CrafterGymnasiumWrapper()
    action_dim = 17
    obs_dim = (3, 64, 64)
    
    # Initialize the agent
    agent = RCQLAgent(obs_dim=obs_dim, action_dim=action_dim, name="RCQL", device_name=device)
    
    # Find the most recently saved .pt model in the crafter results directory
    search_path = os.path.join("results", "crafter", "**", "*.pt")
    models = glob.glob(search_path, recursive=True)
    if not models:
        print("No models found in results/crafter/")
        return
        
    latest_model = max(models, key=os.path.getmtime)
    print(f"Loading {latest_model}...")
    agent.load_model(latest_model)
    
    obs, _ = env.reset()
    agent.reset_hidden()
    term = False
    trunc = False
    total_reward = 0.0
    
    print("Starting episode (Press ESC to quit)...")
    while not (term or trunc):
        # The environment uses (3, 64, 64) for the agent, but render gives (512, 512, 3)
        action = agent.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        
        # Display the frame using OpenCV
        frame = env.render()
        
        # cv2 expects BGR format instead of RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Crafter RL Agent", frame_bgr)
        
        # 50ms delay translates to ~20 FPS. Break if ESC is pressed.
        if cv2.waitKey(50) & 0xFF == 27:
            break
            
    print(f"Episode finished. Total Reward: {total_reward}")
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    play_latest_crafter_episode()
