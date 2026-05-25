import gymnasium as gym
import pygame
import pickle
import os
import numpy as np
import time

class FootballUtils:
    @staticmethod
    def get_action(keys):
        """Map WASD/Arrows and action keys to gfootball actions."""
        # Movement (0: idle, 1: left, 2: TL, 3: T, 4: TR, 5: R, 6: BR, 7: B, 8: BL)
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        
        move_action = 0
        if left and up: move_action = 2
        elif left and down: move_action = 8
        elif right and up: move_action = 4
        elif right and down: move_action = 6
        elif left: move_action = 1
        elif right: move_action = 5
        elif up: move_action = 3
        elif down: move_action = 7
        
        if move_action != 0: 
            return move_action
        
        # Action Set (Standard)
        if keys[pygame.K_j]: return 9  # Long Pass
        if keys[pygame.K_i]: return 10 # High Pass
        if keys[pygame.K_k]: return 11 # Short Pass
        if keys[pygame.K_l]: return 12 # Shot
        if keys[pygame.K_LSHIFT]: return 13 # Sprint
        if keys[pygame.K_SEMICOLON]: return 14 # Release Direction
        if keys[pygame.K_SPACE]: return 17 # Dribble
        if keys[pygame.K_n]: return 18 # Release Sprint
        
        return 0 # Idle

    @staticmethod
    def get_controls_text():
        return " - WASD/Arrows: Move, K: Short Pass, J: Long Pass, L: Shot, I: High Pass, LSHIFT: Sprint"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="11_vs_11_stochastic", help="Scenario to play")
    args = parser.parse_args()
    
    print(f"=== Single-Agent Distillation Recorder ===")
    print(f"Environment: {args.env} | Auto-switching: ENABLED")
    
    from gfootball.env import create_environment

    # Create the standard single-agent environment
    env = create_environment(
        env_name=args.env,
        stacked=False,
        representation='simple115v2', # Natively outputs the flat 115D ML vector
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0, # Built-in bots handle the right team
        render=False # We render manually via pygame
    )

    dataset_path = f"expert_demonstrations_{args.env}.pkl"
    dataset = []
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f: dataset = pickle.load(f)
        print(f"[*] Loaded existing dataset containing {len(dataset)} episodes.")
    else:
        print("[*] Starting a fresh dataset.")

    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    fps = 30
    running = True
    
    while running:
        obs = env.reset() 
        episode_transitions = []
        terminated = False
        start_time = time.time()
        
        while not terminated:
            # Render the frame natively for the human to see
            frame = env.unwrapped.render(mode='rgb_array')
            
            if screen is None and frame is not None:
                height, width, _ = frame.shape
                screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption(f"Recording Distillation Data")
            
            if frame is not None and screen is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
            if not running: break
            
            # Get human input
            keys = pygame.key.get_pressed()
            action = FootballUtils.get_action(keys)
            
            # Step the environment
            next_obs, reward, done, info = env.step(action)
            
            # Record the 115D vector, action, and reward
            episode_transitions.append({
                'obs': obs, 
                'action': action, 
                'reward': reward,
                'next_obs': next_obs, 
                'done': done, 
            })
            
            obs = next_obs
            clock.tick(fps)
            
        if episode_transitions:
            duration = time.time() - start_time
            dataset.append({"transitions": episode_transitions, "duration": duration})
            with open(dataset_path, 'wb') as f: pickle.dump(dataset, f)
            print(f"[+] Episode finished ({duration:.1f}s). Total Episodes: {len(dataset)}")
            
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()