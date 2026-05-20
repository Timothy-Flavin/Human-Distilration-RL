import gymnasium as gym
import pygame
import pickle
import os
import numpy as np

def get_action(keys, env_name="LunarLander-v3"):
    """Map WASD or Arrow Keys to environment actions."""
    if "LunarLander" in env_name:
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 1
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            return 2
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 3
        return 0
    elif "highway" in env_name:
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 2
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            return 3
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return 4
        return 1 # IDLE
    return 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Environment to record")
    args = parser.parse_args()
    
    env_name = args.env
    dataset_path = f"expert_demonstrations_{env_name}.pkl"
    
    print(f"=== Freshman Expert Data Recorder ===")
    print(f"Environment: {env_name}")
    print(f"Saving data to: {dataset_path}")
    
    # 1. Load existing dataset if it exists
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[*] Loaded existing dataset containing {len(dataset)} episodes.")
    else:
        dataset = []
        print("[*] Starting a fresh dataset.")

    print("\nControls:")
    if "LunarLander" in env_name:
        print(" - Arrow Keys or WASD (Left, Up, Right) to fly.")
    elif "highway" in env_name:
        print(" - Arrow Keys or WASD (Left, Right, Up, Down) to drive.")
    print(" - Press 'Q' or close the window to safely save and quit.")
    print("--------------------------------------------------")

    # 2. Setup environment and Pygame
    env = gym.make(env_name, render_mode="rgb_array")
    if "highway" in env_name:
        import highway_env
        env = gym.wrappers.FlattenObservation(env)
        
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    fps = 30
    
    running = True
    
    while running:
        obs, info = env.reset()
        episode_transitions = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            frame = env.render()
            
            if screen is None and frame is not None:
                height, width, _ = frame.shape
                screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption(f"Expert Recording: {env_name}")
            
            if frame is not None and screen is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            
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
            action = get_action(keys, env_name=env_name)
            
            # Step Environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Save Transition: Pair CURRENT state with the action taken in it
            episode_transitions.append({
                'obs': obs, # Correct: current state
                'action': action, # Correct: action taken in current state
                'reward': reward,
                'next_obs': next_obs,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            obs = next_obs
            clock.tick(fps)
            
        if episode_transitions:
            dataset.append(episode_transitions)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"[+] Episode finished. Total Episodes: {len(dataset)}")
            
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
