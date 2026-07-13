import gymnasium as gym
import pygame
import pickle
import os
import numpy as np
import time

class LanderUtils:
    @staticmethod
    def get_action(keys):
        """Map WASD or Arrow Keys to LunarLander actions."""
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 1 # Rotate Left
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            return 2 # Main Engine
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 3 # Rotate Right
        return 0 # NOOP

    @staticmethod
    def get_controls_text():
        return " - Arrow Keys or WASD (Left, Up, Right) to fly."
    
    @staticmethod
    def setup_env(env):
        return env

class HighwayUtils:
    @staticmethod
    def get_action(keys):
        """Map WASD or Arrow Keys to Highway actions."""
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 0 # LANE_LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 2 # LANE_RIGHT
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            return 3 # FASTER
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return 4 # SLOWER
        return 1 # IDLE

    @staticmethod
    def get_controls_text():
        return " - Arrow Keys or WASD (Left, Right, Up, Down) to drive."
    
    @staticmethod
    def setup_env(env):
        from gymnasium.wrappers import FlattenObservation
        return FlattenObservation(env)

class CrafterUtils:
    is_turn_based = True

    @staticmethod
    def get_action(keys):
        """Map keys to Crafter actions (for continuous compatibility if needed)."""
        if keys[pygame.K_a]: return 1     # move_left
        if keys[pygame.K_d]: return 2     # move_right
        if keys[pygame.K_w]: return 3     # move_up
        if keys[pygame.K_s]: return 4     # move_down
        if keys[pygame.K_SPACE]: return 5 # do
        if keys[pygame.K_TAB]: return 6   # sleep
        if keys[pygame.K_r]: return 7     # place_stone
        if keys[pygame.K_t]: return 8     # place_table
        if keys[pygame.K_f]: return 9     # place_furnace
        if keys[pygame.K_p]: return 10    # place_plant
        if keys[pygame.K_1]: return 11    # make_wood_pickaxe
        if keys[pygame.K_2]: return 12    # make_stone_pickaxe
        if keys[pygame.K_3]: return 13    # make_iron_pickaxe
        if keys[pygame.K_4]: return 14    # make_wood_sword
        if keys[pygame.K_5]: return 15    # make_stone_sword
        if keys[pygame.K_6]: return 16    # make_iron_sword
        return 0

    @staticmethod
    def get_action_from_key(key):
        """Map a single key to a Crafter action."""
        mapping = {
            pygame.K_a: 1,     # move_left
            pygame.K_d: 2,     # move_right
            pygame.K_w: 3,     # move_up
            pygame.K_s: 4,     # move_down
            pygame.K_SPACE: 5, # do
            pygame.K_TAB: 6,   # sleep
            pygame.K_r: 7,     # place_stone
            pygame.K_t: 8,     # place_table
            pygame.K_f: 9,     # place_furnace
            pygame.K_p: 10,    # place_plant
            pygame.K_1: 11,    # make_wood_pickaxe
            pygame.K_2: 12,    # make_stone_pickaxe
            pygame.K_3: 13,    # make_iron_pickaxe
            pygame.K_4: 14,    # make_wood_sword
            pygame.K_5: 15,    # make_stone_sword
            pygame.K_6: 16     # make_iron_sword
        }
        return mapping.get(key, None)

    @staticmethod
    def get_controls_text():
        return (
            " - WASD: Move\n"
            " - Space: Do, Tab: Sleep\n"
            " - R: Stone, T: Table, F: Furnace, P: Plant (Place)\n"
            " - 1-3: Pickaxe (Wood, Stone, Iron)\n"
            " - 4-6: Sword (Wood, Stone, Iron)\n"
            " - Turn-based: One action per key press."
        )
    
    @staticmethod
    def make_env(env_name):
        try:
            import crafter
        except ImportError:
            print("[!] Error: crafter not found. Please install it.")
            raise

        class CrafterGymnasiumWrapper:
            def __init__(self):
                self._env = crafter.Env()
                self.observation_space = self._env.observation_space
                self.action_space = self._env.action_space
                self.render_mode = "rgb_array"

            def reset(self, seed=None, options=None):
                if seed is not None:
                    self._env.seed(seed)
                obs = self._env.reset()
                return obs, {}

            def step(self, action):
                obs, reward, done, info = self._env.step(action)
                # Gymnasium expects (obs, reward, terminated, truncated, info)
                return obs, reward, done, False, info

            def render(self):
                # Upscale for better visibility (64x64 -> 576x576)
                # Crafter's render(size) handles this nicely
                return self._env.render(size=(576, 576))

            def close(self):
                pass

        return CrafterGymnasiumWrapper()

def get_utils(env_name):
    if "LunarLander" in env_name:
        return LanderUtils
    elif "highway" in env_name:
        return HighwayUtils
    elif "crafter" in env_name:
        return CrafterUtils
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Environment to record")
    args = parser.parse_args()
    print("yooo")
    
    env_name = args.env
    if env_name == "highway":
        env_name = "highway-v0"
    elif env_name == "crafter":
        env_name = "crafter"
    print("yooo")
        
    utils = get_utils(env_name)
    
    print("yooo")

    if utils is None:
        print(f"[!] Error: Unsupported environment '{env_name}'")
        return

    # Ensure environment-specific imports are handled
    if "highway" in env_name:
        try: import highway_env
        except ImportError: 
            print("no highway")
    elif "crafter" in env_name:
        try: import crafter
        except ImportError:
            print("[!] Error: crafter not found. Please install it.")
    print("yooo")

    dataset_path = f"expert_demonstrations_{env_name}.pkl"
    
    print(f"=== Freshman Expert Data Recorder ===")
    print(f"Environment: {env_name}")
    print(f"Saving data to: {dataset_path}")
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"[*] Loaded existing dataset containing {len(dataset)} episodes.")
    else:
        dataset = []
        print("[*] Starting a fresh dataset.")

    print("\nControls:")
    print(utils.get_controls_text())
    print(" - Press 'Q' or close the window to safely save and quit.")
    print("--------------------------------------------------")

    if hasattr(utils, 'make_env'):
        env = utils.make_env(env_name)
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        env = utils.setup_env(env)
        
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    fps = 30
    
    running = True
    is_turn_based = getattr(utils, 'is_turn_based', False)
    
    while running:
        obs, info = env.reset()
        episode_transitions = []
        terminated = False
        truncated = False
        
        start_time = time.time()
        
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
            
            action = None
            if is_turn_based:
                # Wait for a valid action key press
                while action is None and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                running = False
                            else:
                                action = utils.get_action_from_key(event.key)
                    if not running: break
                    pygame.time.wait(10) # Prevent 100% CPU usage while waiting
                if not running: break
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q: running = False
                
                if not running: break
                
                keys = pygame.key.get_pressed()
                action = utils.get_action(keys)
                clock.tick(fps)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_transitions.append({
                'obs': obs, 'action': action, 'reward': reward,
                'next_obs': next_obs, 'terminated': terminated,
                'truncated': truncated, 'info': info
            })
            obs = next_obs
            
        if episode_transitions:
            # Quitting ('Q' / window close) breaks out before env.step, so a
            # quit episode's last transition has neither flag set: mark it
            # truncated (cut off at a live state; consumers bootstrap through
            # it, unlike terminated).
            last = episode_transitions[-1]
            if not last['terminated'] and not last['truncated']:
                last['truncated'] = True
            duration = time.time() - start_time
            dataset.append({
                "transitions": episode_transitions,
                "duration": duration
            })
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"[+] Episode finished ({duration:.1f}s). Total Episodes: {len(dataset)}")
            
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
