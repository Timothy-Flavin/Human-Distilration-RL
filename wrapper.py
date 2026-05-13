import pygame
import numpy as np
import gymnasium as gym
from input_handler import process_events, get_realtime_action

class InteractiveGymWrapper:
    def __init__(self, env: gym.Env, agent=None, fps=30):
        self.env = env
        self.agent = agent
        self.fps = fps
        self.env.unwrapped.render_mode = "rgb_array" # Enforce rgb_array for Pygame
        
        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()
        
        # State Machine
        self.mode = "step"  # "step", "realtime", "note", "agent"
        
        # UI & Input
        self.text_buffer = ""
        self.font = pygame.font.SysFont("Courier", 24)
        self.small_font = pygame.font.SysFont("Courier", 18)
        
        # Data Buffers & Seed State
        self.trajectory = []
        self.notes = []
        self.current_frame_idx = -1
        self.current_obs = None
        self.current_seed = None  # Crucial for deterministic replay
        self.running = False

    def reset_env(self):
        # 1. Generate and save a deterministic seed for this episode
        self.current_seed = np.random.randint(0, 1000000)
        obs, info = self.env.reset(seed=self.current_seed)
        
        self.trajectory = []
        self.notes = []
        self.current_frame_idx = -1
        self.current_obs = obs
        self._record_step(obs, 0, 0, False, False, info)
        
    def _record_step(self, obs, action, reward, terminated, truncated, info):
        frame = self.env.render()
        
        # Attempt to capture internal state for O(1) branching
        try:
            env_state = self.env.unwrapped.get_state()
        except AttributeError:
            env_state = None  # Safe failure, will trigger rollback later

        # Initialize Pygame surface on the first render
        if self.screen is None and frame is not None:
            height, width, _ = frame.shape
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Interactive Replay Review")

        step_data = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "frame_image": frame,
            "env_state": env_state,
            "terminated": terminated,
            "truncated": truncated
        }
        self.trajectory.append(step_data)
        self.current_frame_idx += 1

    def _verify_observations(self, obs1, obs2, atol=1e-5):
        """Recursively checks if two observations match."""
        try:
            if isinstance(obs1, dict):
                if obs1.keys() != obs2.keys():
                    return False
                return all(self._verify_observations(obs1[k], obs2[k], atol) for k in obs1)
            elif isinstance(obs1, (tuple, list)):
                if len(obs1) != len(obs2):
                    return False
                return all(self._verify_observations(o1, o2, atol) for o1, o2 in zip(obs1, obs2))
            elif isinstance(obs1, np.ndarray) or isinstance(obs2, np.ndarray):
                # Use allclose for floats to avoid strict equality failure on tiny drifts
                return np.allclose(obs1, obs2, atol=atol)
            else:
                return obs1 == obs2
        except Exception:
            # If shapes mismatch or types don't align, it's a desync
            return False

    def _branch_timeline(self):
        """Truncates future history and restores environment state via hybrid approach."""
        if self.current_frame_idx >= len(self.trajectory) - 1:
            return # Already at the present, no branching needed

        print(f"\n[Timeline] Branching universe at frame {self.current_frame_idx}...")
        
        # 1. Truncate future trajectories and notes
        self.trajectory = self.trajectory[:self.current_frame_idx + 1]
        self.notes = [n for n in self.notes if n["frame"] <= self.current_frame_idx]
        
        # 2. Try O(1) state restoration first
        saved_state = self.trajectory[self.current_frame_idx]["env_state"]
        state_restored = False
        
        if saved_state is not None:
            try:
                self.env.unwrapped.set_state(saved_state)
                state_restored = True
                print("[Timeline] O(1) State restoration successful.")
            except AttributeError:
                pass # Environment lies about having a valid set_state, fall through
                
        # 3. Fallback to O(N) Deterministic Replay with Verification
        if not state_restored:
            print("[Timeline] Missing set_state(). Triggering O(N) Deterministic Replay...")
            
            # Re-seed to the exact start of this episode
            self.env.reset(seed=self.current_seed)
            desync_detected = False
            
            # Fast-forward by silently feeding past actions
            for i in range(1, self.current_frame_idx + 1):
                past_action = self.trajectory[i]["action"]
                obs, _, _, _, _ = self.env.step(past_action)
                
                # --- VERIFICATION STEP ---
                original_obs = self.trajectory[i]["obs"]
                if not self._verify_observations(obs, original_obs):
                    print(f"⚠️ [WARNING] DESYNC DETECTED at frame {i}!")
                    print("The environment broke the deterministic RNG contract.")
                    desync_detected = True
                    break # Stop verifying to avoid console spam, but continue the loop
                
            if desync_detected:
                print("❌ [Timeline] Rebuild is corrupted. Future states will diverge from history.")
            else:
                print("✅ [Timeline] Fast-forward complete and perfectly verified.")
                
    def step_forward(self, action):
        """Advances the environment if at the end of the buffer, or steps forward in history."""
        if self.current_frame_idx < len(self.trajectory) - 1:
            self.current_frame_idx += 1
            return
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_step(obs, action, reward, terminated, truncated, info)
        self.current_obs = obs

        if terminated or truncated:
            self.mode = "step"

    def step_backward(self):
        """Steps backward through the saved trajectory history."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1

    def draw_overlay(self):
        """Draws UI elements (mode, frame index, text buffer, existing notes)."""
        if self.screen is None:
            return

        frame = self.trajectory[self.current_frame_idx]["frame_image"]
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (0, 0))

        mode_surf = self.font.render(f"MODE: {self.mode.upper()}", True, (255, 255, 0))
        self.screen.blit(mode_surf, (10, 10))

        frame_surf = self.small_font.render(
            f"Frame: {self.current_frame_idx}/{len(self.trajectory)-1}", 
            True, (200, 200, 200)
        )
        self.screen.blit(frame_surf, (10, 40))

        if self.mode == "note":
            bg_rect = pygame.Rect(0, self.screen.get_height() - 50, self.screen.get_width(), 50)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            note_surf = self.font.render(f"> {self.text_buffer}_", True, (0, 255, 0))
            self.screen.blit(note_surf, (10, self.screen.get_height() - 40))

        current_note = next((n["text"] for n in self.notes if n["frame"] == self.current_frame_idx), None)
        if current_note:
            note_display = self.small_font.render(f"NOTE: {current_note}", True, (255, 100, 100))
            self.screen.blit(note_display, (10, 70))

        pygame.display.flip()

    def run(self):
        self.running = True
        self.reset_env()

        while self.running:
            events = pygame.event.get()
            
            # Use the input handler that includes the `branch_timeline` boolean return
            new_mode, self.text_buffer, submitted_note, step_dir, reset, branch_timeline = process_events(
                events, self.mode, self.text_buffer
            )
            
            self.mode = new_mode
            if self.mode == "quit":
                self.running = False
                break

            # Timeline Branch Check
            if branch_timeline:
                self._branch_timeline()

            if reset:
                self.reset_env()
            
            if submitted_note:
                self.notes.append({
                    "frame": self.current_frame_idx,
                    "text": submitted_note
                })

            if self.mode == "step":
                if step_dir == 1:
                    self.step_forward(action=0)
                elif step_dir == -1:
                    self.step_backward()

            elif self.mode == "realtime":
                keys = pygame.key.get_pressed()
                action = get_realtime_action(keys)
                self.step_forward(action)

            elif self.mode == "agent":
                if self.agent is not None:
                    action = self.agent.predict(self.current_obs) 
                else:
                    action = self.env.action_space.sample()
                self.step_forward(action)

            self.draw_overlay()
            self.clock.tick(self.fps)

        pygame.quit()
        return self.trajectory, self.notes