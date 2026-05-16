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
        # Ensure initial trajectory has sources
        for step in self.trajectory:
            if "source" not in step: step["source"] = "rl"

        self.notes = []
        self.current_frame_idx = len(self.trajectory) - 1 if self.trajectory else -1
        self.current_obs = self.trajectory[-1]["obs"] if self.trajectory else None
        self.current_seed = initial_seed  # Crucial for deterministic replay
        self.running = False

        # Override state
        self.override_start_frame = -1
        self.discarded_trajectory = []
        self.override_source = None # "realtime" or "agent"

    def load_trajectory(self, trajectory, seed):
        """Loads a pre-recorded trajectory into the wrapper for review."""
        self.trajectory = trajectory
        for step in self.trajectory:
            if "source" not in step: step["source"] = "rl"

        self.current_seed = seed
        self.current_frame_idx = 0
        self.current_obs = self.trajectory[0]["obs"]
        self.notes = []
        self._restore_state(0)

    def _format_obs(self, obs):
        """Formats observations for human readability (LunarLander specific)."""
        if isinstance(obs, np.ndarray) and len(obs) == 8:
            return {
                "x_pos": float(obs[0]),
                "y_pos": float(obs[1]),
                "x_vel": float(obs[2]),
                "y_vel": float(obs[3]),
                "angle": float(obs[4]),
                "angular_vel": float(obs[5]),
                "leg1_contact": bool(obs[6] > 0.5),
                "leg2_contact": bool(obs[7] > 0.5),
                "readable_summary": f"Pos:({obs[0]:.2f}, {obs[1]:.2f}), Vel:({obs[2]:.2f}, {obs[3]:.2f}), Angle:{obs[4]:.2f}"
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
        self._record_step(obs, 0, 0, False, False, info, source="rl")

    def _record_step(self, obs, action, reward, terminated, truncated, info, source="rl"):
        frame = self.env.render()
        if frame is None and self.trajectory:
            # Try to inherit frame if render fails (e.g. during fast stepping)
            frame = self.trajectory[-1].get("frame_image")

        # Attempt to capture internal state for O(1) branching
        try:
            env_state = self.env.unwrapped.get_state()
        except AttributeError:
            env_state = None  

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
            "truncated": truncated,
            "source": source
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
                return np.allclose(obs1, obs2, atol=atol)
            else:
                return obs1 == obs2
        except Exception:
            return False

    def _restore_state(self, target_frame_idx):
        """Restores environment state to target_frame_idx."""
        if target_frame_idx < 0: return

        # 1. Try O(1) state restoration first
        saved_state = self.trajectory[target_frame_idx].get("env_state")
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
            print(f"[Timeline] Triggering O(N) Deterministic Replay to frame {target_frame_idx}...")
            self.env.reset(seed=self.current_seed)
            for i in range(1, target_frame_idx + 1):
                past_action = self.trajectory[i]["action"]
                self.env.step(past_action)
            print("✅ [Timeline] Fast-forward complete.")

    def _branch_timeline(self, source):
        """Prepares for branching by saving the future trajectory."""
        print(f"\n[Timeline] Preparing to branch at frame {self.current_frame_idx} (Source: {source})...")

        self.override_start_frame = self.current_frame_idx
        self.override_source = source

        # Save the future as discarded
        self.discarded_trajectory = self.trajectory[self.current_frame_idx + 1:]

        # Truncate trajectory
        self.trajectory = self.trajectory[:self.current_frame_idx + 1]
        self.notes = [n for n in self.notes if n["frame"] <= self.current_frame_idx]

        # Ensure env matches current_frame_idx
        self._restore_state(self.current_frame_idx)

    def _handle_decision(self, decision):
        """Processes Accept/Reject decision for an override."""
        if decision == "accept":
            print(f"✅ [Override] Accepted {self.override_source} segment.")
            if self.buffers:
                # 1. Push up to 100 steps of the rejected segment to anti-example buffer.
                max_anti_frames = 100
                prev_obs = self.trajectory[self.override_start_frame]['obs']
                for step in self.discarded_trajectory[:max_anti_frames]:
                    self.buffers['anti_example'].push(prev_obs, step['action'])
                    prev_obs = step['obs']

                # 2. Push human segment to example buffer (ONLY if source was realtime)
                if self.override_source == "realtime":
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

        self.override_source = None

    def step_forward(self, action, source="rl"):
        """Advances the environment if at the end of the buffer, or steps forward in history."""
        if self.current_frame_idx < len(self.trajectory) - 1:
            self.current_frame_idx += 1
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]
            return

        # Prevent stepping the environment if the episode is already over
        if self.trajectory and (self.trajectory[-1]["terminated"] or self.trajectory[-1]["truncated"]):
            return

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_step(obs, action, reward, terminated, truncated, info, source=source)
        self.current_obs = obs

        if terminated or truncated:
            # Only trigger a decision if we actively branched the timeline
            if self.override_source is not None:
                self.mode = "decision"

    def step_backward(self):
        """Steps backward through the saved trajectory history."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]

    def draw_overlay(self):
        """Draws UI elements (mode, frame index, text buffer, existing notes)."""
        if self.screen is None or not self.trajectory:
            return

        frame = self.trajectory[self.current_frame_idx].get("frame_image")
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (0, 0))
        else:
            self.screen.fill((50, 50, 50))
            msg = self.font.render("No Image Frame Saved", True, (255, 255, 255))
            self.screen.blit(msg, (self.screen.get_width() // 2 - msg.get_width() // 2, self.screen.get_height() // 2))

        # Mode display
        color = (255, 255, 0)
        if self.mode == "decision": color = (255, 0, 0)
        elif self.mode == "realtime": color = (0, 255, 0)
        elif self.mode == "agent": color = (0, 255, 255)

        mode_surf = self.font.render(f"MODE: {self.mode.upper()}", True, color)
        self.screen.blit(mode_surf, (10, 10))

        frame_surf = self.small_font.render(
            f"Frame: {self.current_frame_idx}/{len(self.trajectory)-1}", 
            True, (200, 200, 200)
        )
        self.screen.blit(frame_surf, (10, 40))

        source = self.trajectory[self.current_frame_idx].get("source", "rl")
        source_surf = self.small_font.render(f"SOURCE: {source.upper()}", True, (200, 200, 200))
        self.screen.blit(source_surf, (10, 60))

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
            
            # Add a fallback string if override_source is None
            source_name = self.override_source.title() if self.override_source else "Changes"
            msg = self.font.render(f"[A]ccept {source_name} or [R]eject?", True, (255, 255, 255))
            self.screen.blit(msg, (self.screen.get_width() // 2 - msg.get_width() // 2, self.screen.get_height() // 2 - 20))
        
        current_note = next((n["text"] for n in self.notes if n["frame"] == self.current_frame_idx), None)
        if current_note:
            note_display = self.small_font.render(f"NOTE: {current_note}", True, (255, 100, 100))
            self.screen.blit(note_display, (10, 80))

        pygame.display.flip()

    def run(self):
        self.running = True
        if not self.trajectory:
            self.reset_env()
        else:
            if self.screen is None:
                self.screen = pygame.display.set_mode((600, 400))
                pygame.display.set_caption("Interactive Replay Review")

        step_counter = 0

        while self.running:
            events = pygame.event.get()

            new_mode, self.text_buffer, submitted_note, step_dir, reset, branch_timeline, decision = process_events(
                events, self.mode, self.text_buffer
            )

            # Timer management
            if self.mode in ["realtime", "agent"] and new_mode not in ["realtime", "agent"]:
                if self.metrics: self.metrics.stop_timer("human_overriding")
            
            if new_mode in ["realtime", "agent"] and self.mode not in ["realtime", "agent"]:
                # Only start if we are in an override state or just branched
                if self.override_source is not None or branch_timeline:
                    if self.metrics: self.metrics.start_timer("human_overriding")

            if new_mode == "note" and self.mode != "note" and self.metrics:
                self.metrics.start_timer("human_annotating")
            elif self.mode == "note" and new_mode != "note" and self.metrics:
                self.metrics.stop_timer("human_annotating")

            if new_mode == "step" and self.mode != "step" and self.metrics:
                self.metrics.start_timer("human_reviewing")
            elif self.mode == "step" and new_mode != "step" and self.metrics:
                self.metrics.stop_timer("human_reviewing")

            self.mode = new_mode
            if self.mode in ["quit", "finish"]:
                self.running = False
                break

            # Timeline Branch Check
            if branch_timeline:
                self._branch_timeline(source=self.mode)

            if decision:
                self._handle_decision(decision)

            if reset:
                self.reset_env()

            if submitted_note:
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
                if step_dir != 0:
                    step_counter += 1
                    if step_dir == 1:
                        self.step_forward(action=0, source="rl")
                    elif step_dir == -1:
                        self.step_backward()
                else:
                    step_counter = 0

            elif self.mode == "realtime":
                keys = pygame.key.get_pressed()
                action = get_realtime_action(keys)
                self.step_forward(action, source="human")
                if self.metrics: 
                    self.metrics.log_frames(1, source="human")

            elif self.mode == "agent":
                if self.agent is not None:
                    # Note: We must use the current_obs from history or live env
                    action = self.agent.predict(self.current_obs) 
                else:
                    action = self.env.action_space.sample()
                self.step_forward(action, source="rl")
                if self.metrics: 
                    self.metrics.log_frames(1, source="rl")
            #print(self.current_obs)
            self.draw_overlay()
            self.clock.tick(self.fps)

        pygame.quit()
        return self.trajectory, self.notes, self.current_seed