import pygame
import numpy as np
import gymnasium as gym
from input_handler import process_events, get_realtime_action

class InteractiveGymWrapper:
    def __init__(self, env: gym.Env, agent=None, fps=30, buffers=None, metrics=None, initial_trajectory=None, initial_seed=None):
        self.env = env
        self.agent = agent
        self.fps = fps
        self.buffers = buffers # Expected: dict with 'example', 'anti_example', 'llm'
        self.metrics = metrics
        self.env.unwrapped.render_mode = "rgb_array" # Enforce rgb_array for Pygame

        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

        # State Machine
        self.mode = "step"  # "step", "realtime", "note", "agent", "decision"

        # UI & Input
        self.text_buffer = ""
        self.font = pygame.font.SysFont("Courier", 24)
        self.small_font = pygame.font.SysFont("Courier", 18)

        # Data Buffers & Seed State
        self.trajectory = initial_trajectory if initial_trajectory is not None else []
        self.notes = []
        self.current_frame_idx = len(self.trajectory) - 1 if self.trajectory else -1
        self.current_obs = self.trajectory[-1]["obs"] if self.trajectory else None
        self.current_seed = initial_seed  # Crucial for deterministic replay
        self.running = False

        # Override state
        self.override_start_frame = -1
        self.discarded_trajectory = []

    def load_trajectory(self, trajectory, seed):
        """Loads a pre-recorded trajectory into the wrapper for review."""
        self.trajectory = trajectory
        self.current_seed = seed
        self.current_frame_idx = 0
        self.current_obs = self.trajectory[0]["obs"]
        self.notes = []
        self._restore_state(0)

    def _format_obs(self, obs):

        """Formats observations for human readability (LunarLander specific)."""
        if isinstance(obs, np.ndarray) and len(obs) == 8:
            return {
                "x_pos": float(obs[0]), "y_pos": float(obs[1]),
                "x_vel": float(obs[2]), "y_vel": float(obs[3]),
                "angle": float(obs[4]), "angular_vel": float(obs[5]),
                "leg1_contact": bool(obs[6]), "leg2_contact": bool(obs[7])
            }
        return str(obs)

    def reset_env(self):
        # 1. Generate and save a deterministic seed for this episode
        if self.current_seed is None:
            self.current_seed = np.random.randint(0, 1000000)

        obs, info = self.env.reset(seed=self.current_seed)

        self.trajectory = []
        self.notes = []
        self.current_frame_idx = -1
        self.current_obs = obs
        self._record_step(obs, 0, 0, False, False, info)

    def _record_step(self, obs, action, reward, terminated, truncated, info):
        frame = self.env.render()
        if frame is None:
            return # Avoid recording if render fails or returns None

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

    def _restore_state(self, target_frame_idx):
        """Restores environment state to target_frame_idx."""
        # 1. Try O(1) state restoration first
        saved_state = self.trajectory[target_frame_idx]["env_state"]
        state_restored = False

        if saved_state is not None:
            try:
                self.env.unwrapped.set_state(saved_state)
                state_restored = True
                print("[Timeline] O(1) State restoration successful.")
            except AttributeError:
                pass 

        # 2. Fallback to O(N) Deterministic Replay
        if not state_restored:
            print("[Timeline] Triggering O(N) Deterministic Replay...")
            self.env.reset(seed=self.current_seed)
            for i in range(1, target_frame_idx + 1):
                past_action = self.trajectory[i]["action"]
                self.env.step(past_action)
            print("✅ [Timeline] Fast-forward complete.")

    def _branch_timeline(self):
        """Prepares for branching by saving the future trajectory."""
        if self.current_frame_idx >= len(self.trajectory) - 1:
            self.override_start_frame = self.current_frame_idx
            self.discarded_trajectory = []
            return 

        print(f"\n[Timeline] Preparing to branch at frame {self.current_frame_idx}...")

        self.override_start_frame = self.current_frame_idx
        self.discarded_trajectory = self.trajectory[self.current_frame_idx + 1:]

        # Truncate
        self.trajectory = self.trajectory[:self.current_frame_idx + 1]
        self.notes = [n for n in self.notes if n["frame"] <= self.current_frame_idx]

        self._restore_state(self.current_frame_idx)

    def _handle_decision(self, decision):
        """Processes Accept/Reject decision for an override."""
        if decision == "accept":
            print("✅ [Override] Accepted.")
            if self.buffers:
                # 1. Push up to 100 steps of the rejected (RL) segment to anti-example buffer.
                # We cap this because there is no guarantee the entire future trajectory was "bad".
                # We use the observation from the frame where the action was actually taken.
                max_anti_frames = 100
                prev_obs = self.trajectory[self.override_start_frame]['obs']
                for step in self.discarded_trajectory[:max_anti_frames]:
                    self.buffers['anti_example'].push(prev_obs, step['action'])
                    prev_obs = step['obs']

                # 2. Push EVERY step of the new (Human) segment to example buffer
                # Again, ensuring we pair the action with the observation it was taken from.
                prev_obs = self.trajectory[self.override_start_frame]['obs']
                new_part = self.trajectory[self.override_start_frame + 1:]
                for step in new_part:
                    self.buffers['example'].push(prev_obs, step['action'])
                    prev_obs = step['obs']

            self.discarded_trajectory = []

        elif decision == "reject":
            print("❌ [Override] Rejected. Rolling back to branching point...")
            # Restore trajectory to original state before override
            self.trajectory = self.trajectory[:self.override_start_frame + 1] + self.discarded_trajectory
            self.discarded_trajectory = []
            
            # Reset current frame to the branching point
            self.current_frame_idx = self.override_start_frame
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]
            
            # Restore env state to the branching point
            self._restore_state(self.current_frame_idx)

    def step_forward(self, action):
        """Advances the environment if at the end of the buffer, or steps forward in history."""
        if self.current_frame_idx < len(self.trajectory) - 1:
            self.current_frame_idx += 1
            return

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_step(obs, action, reward, terminated, truncated, info)
        self.current_obs = obs

        if terminated or truncated:
            self.mode = "decision" # Force decision at end of episode if we were overriding

    def step_backward(self):
        """Steps backward through the saved trajectory history."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1

    def draw_overlay(self):
        """Draws UI elements (mode, frame index, text buffer, existing notes)."""
        if self.screen is None or not self.trajectory:
            return

        frame = self.trajectory[self.current_frame_idx].get("frame_image")
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (0, 0))
        else:
            # Fallback if frame_image is missing (e.g. trajectory from run_rl_collection)
            self.screen.fill((50, 50, 50))
            msg = self.font.render("No Image Frame Saved", True, (255, 255, 255))
            self.screen.blit(msg, (self.screen.get_width() // 2 - msg.get_width() // 2, self.screen.get_height() // 2))

        # Mode display
        color = (255, 255, 0)
        if self.mode == "decision": color = (255, 0, 0)
        mode_surf = self.font.render(f"MODE: {self.mode.upper()}", True, color)
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

        if self.mode == "decision":
            overlay = pygame.Surface((self.screen.get_width(), 100))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, self.screen.get_height() // 2 - 50))

            msg = self.font.render("[A]ccept Override or [R]eject Override?", True, (255, 255, 255))
            self.screen.blit(msg, (self.screen.get_width() // 2 - msg.get_width() // 2, self.screen.get_height() // 2 - 20))

        current_note = next((n["text"] for n in self.notes if n["frame"] == self.current_frame_idx), None)
        if current_note:
            note_display = self.small_font.render(f"NOTE: {current_note}", True, (255, 100, 100))
            self.screen.blit(note_display, (10, 70))

        pygame.display.flip()

    def run(self):
        self.running = True
        if not self.trajectory:
            self.reset_env()
        else:
            # If we have a trajectory but no screen yet, try to initialize it
            if self.screen is None:
                # Default size if no frames are present
                self.screen = pygame.display.set_mode((600, 400))
                pygame.display.set_caption("Interactive Replay Review")

        step_counter = 0

        while self.running:
            events = pygame.event.get()

            # Use the input handler that includes the decision and branch_timeline returns
            new_mode, self.text_buffer, submitted_note, step_dir, reset, branch_timeline, decision = process_events(
                events, self.mode, self.text_buffer
            )

            if self.mode in ["realtime", "agent"] and new_mode == "decision":
                if self.metrics: self.metrics.stop_timer("human_overriding")

            self.mode = new_mode
            if self.mode in ["quit", "finish"]:
                self.running = False
                break

            if self.mode == "note" and self.metrics:
                self.metrics.start_timer("human_annotating")

            # Timeline Branch Check
            if branch_timeline:
                self._branch_timeline()
                if self.metrics: self.metrics.start_timer("human_overriding")

            if decision:
                self._handle_decision(decision)

            if reset:
                self.reset_env()

            if submitted_note:
                if self.metrics: self.metrics.stop_timer("human_annotating")
                note_data = {
                    "frame": self.current_frame_idx,
                    "text": submitted_note,
                    "obs_context": self._format_obs(self.trajectory[self.current_frame_idx]["obs"])
                }
                self.notes.append(note_data)
                if self.buffers:
                    self.buffers['llm'].push(
                        episode_trajectory=self.trajectory,
                        seed=self.current_seed,
                        note_text=submitted_note,
                        note_frame_idx=self.current_frame_idx,
                        current_obs_dict=note_data["obs_context"]
                    )

            if self.mode == "step":
                # Only step every N frames of the loop to keep it controllable
                if step_dir != 0:
                    step_counter += 1
                    if step_counter % 2 == 0: # Step every 2 ticks (approx 15 fps)
                        if step_dir == 1:
                            self.step_forward(action=0)
                        elif step_dir == -1:
                            self.step_backward()
                else:
                    step_counter = 0

            elif self.mode == "realtime":

                keys = pygame.key.get_pressed()
                action = get_realtime_action(keys)
                self.step_forward(action)
                if self.metrics: self.metrics.log_frames(1, source="human")

            elif self.mode == "agent":
                if self.agent is not None:
                    action = self.agent.predict(self.trajectory[self.current_frame_idx]["obs"]) 
                else:
                    action = self.env.action_space.sample()
                self.step_forward(action)
                if self.metrics: self.metrics.log_frames(1, source="rl")

            self.draw_overlay()
            self.clock.tick(self.fps)

        pygame.quit()
        return self.trajectory, self.notes