import gymnasium as gym
import pygame
import pickle
import os
import numpy as np

DATASET_PATH = "expert_demonstrations.pkl"

def get_action(keys):
    """Map WASD or Arrow Keys to LunarLander actions."""
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 1
    elif keys[pygame.K_UP] or keys[pygame.K_w]:
        return 2
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 3
    return 0

def main():
    print(f"=== Freshman Expert Data Recorder ===")
    print(f"Saving data to: {DATASET_PATH}")
    
    # 1. Load existing dataset if it exists
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[*] Loaded existing dataset containing {len(dataset)} episodes.")
    else:
        dataset = []
        print("[*] Starting a fresh dataset.")

    print("\nControls:")
    print(" - Arrow Keys or WASD to fly the Lander.")
    print(" - Press 'Q' or close the window to safely save and quit.")
    print("--------------------------------------------------")

    # 2. Setup environment and Pygame
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    fps = 30
    
    running = True
    
    # 3. Main Collection Loop
    while running:
        obs, info = env.reset()
        episode_transitions = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            frame = env.render()
            
            # Initialize screen on first frame
            if screen is None and frame is not None:
                height, width, _ = frame.shape
                screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Expert Data Collection - Press 'Q' to quit")
            
            # Draw frame
            if frame is not None and screen is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            
            # Handle Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            if not running:
                break
            
            # Get Action
            keys = pygame.key.get_pressed()
            action = get_action(keys)
            
            # Step Environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Save Transition
            # (Note: including 'action' as it is critical for BC / TD3-BC)
            episode_transitions.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            obs = next_obs
            clock.tick(fps)
            
        # 4. Save episode when finished
        # If we quit mid-episode, we still save the partial episode data to avoid wasting time
        if episode_transitions:
            dataset.append(episode_transitions)
            
            # Append/Save to disk immediately after each episode
            with open(DATASET_PATH, 'wb') as f:
                pickle.dump(dataset, f)
                
            print(f"[+] Episode finished. Appended {len(episode_transitions)} transitions. Total Episodes: {len(dataset)}")
            
    env.close()
    pygame.quit()
    print(f"\n=== Shutdown Complete ===")
    print(f"Total dataset size: {len(dataset)} episodes saved to {DATASET_PATH}.")

if __name__ == "__main__":
    main()
