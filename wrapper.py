import pygame
import numpy as np
import gymnasium as gym
from input_handler import process_events, get_realtime_action

class InteractiveGymWrapper:
    def __init__(self, env: gym.Env, agent=None, fps=30, buffers=None, metrics=None, initial_trajectory=None, initial_seed=None, env_name="LunarLander-v3", is_curriculum=False):
        self.env = env
        if env_name == "highway":
            env_name = "highway-v0"
        self.env_name = env_name
        self.agent = agent
        self.fps = fps
        self.buffers = buffers # Expected: dict with 'example', 'anti_example', 'llm', 'curriculum'
        self.metrics = metrics
        self.is_curriculum = is_curriculum
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
        # Ensure initial trajectory has sources and correct keys
        for step in self.trajectory:
            if "source" not in step: step["source"] = "rl"

        self.notes = []
        self.current_frame_idx = 0 if self.trajectory else -1
        self.current_obs = self.trajectory[0]["obs"] if self.trajectory else None
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
        """Formats observations for human readability."""
        if "LunarLander" in self.env_name and isinstance(obs, np.ndarray) and len(obs) == 8:
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
        elif "highway" in self.env_name:
            if isinstance(obs, np.ndarray):
                ego = obs[0] if len(obs.shape) > 1 else obs[:5]
                summary = f"Ego - Pos:({ego[1]:.2f}, {ego[2]:.2f}), Vel:({ego[3]:.2f}, {ego[4]:.2f})"
                formatted = {"readable_summary": summary}
                if len(obs.shape) > 1:
                    for i in range(len(obs)):
                        v = obs[i]
                        formatted[f"vehicle_{i}"] = {
                            "presence": bool(v[0] > 0.5), "x": float(v[1]), "y": float(v[2]),
                            "vx": float(v[3]), "vy": float(v[4])
                        }
                return formatted
        elif "football" in self.env_name or "gfootball" in self.env_name:
            if isinstance(obs, np.ndarray):
                return {
                    "readable_summary": f"Football State (dim: {obs.shape})",
                    "raw_vector": obs.tolist()
                }
        return str(obs)

    def reset_env(self):
        if self.current_seed is None:
            self.current_seed = np.random.randint(0, 1000000)

        res = self.env.reset(seed=self.current_seed)
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}

        self.trajectory = []
        self.notes = []
        self.current_frame_idx = 0
        self.current_obs = obs
        
        # Initial step
        frame = self.env.render()
        
        # Check for get_state
        env_state = None
        if hasattr(self.env, 'get_state'): env_state = self.env.get_state()
        elif hasattr(self.env.unwrapped, 'get_state'): env_state = self.env.unwrapped.get_state()

        self.trajectory.append({
            "obs": obs, "action": 0, "reward": 0.0, "next_obs": obs,
            "frame_image": frame, 
            "env_state": env_state,
            "terminated": False, "truncated": False, "source": "rl"
        })

    def _verify_observations(self, obs1, obs2, atol=1e-5):
        try:
            if isinstance(obs1, dict):
                if obs1.keys() != obs2.keys(): return False
                return all(self._verify_observations(obs1[k], obs2[k], atol) for k in obs1)
            elif isinstance(obs1, (tuple, list)):
                if len(obs1) != len(obs2): return False
                return all(self._verify_observations(o1, o2, atol) for o1, o2 in zip(obs1, obs2))
            elif isinstance(obs1, np.ndarray) or isinstance(obs2, np.ndarray):
                # Ensure they have same shape before np.allclose
                if np.array(obs1).shape != np.array(obs2).shape: return False
                return np.allclose(obs1, obs2, atol=atol)
            else: return obs1 == obs2
        except Exception: return False

    def _restore_state(self, target_frame_idx):
        if target_frame_idx < 0: return
        saved_state = self.trajectory[target_frame_idx].get("env_state")
        state_restored = False
        if saved_state is not None:
            try:
                if hasattr(self.env, 'set_state'):
                    self.env.set_state(saved_state)
                    state_restored = True
                elif hasattr(self.env.unwrapped, 'set_state'):
                    self.env.unwrapped.set_state(saved_state)
                    state_restored = True
            except Exception: pass 

        if not state_restored:
            self.env.reset(seed=self.current_seed)
            # Replay actions. Note: trajectory[i] has action taken in obs_i.
            # So to get to state at target_frame_idx, we take actions from 0 up to target-1
            for i in range(0, target_frame_idx):
                self.env.step(self.trajectory[i]["action"])

    def _branch_timeline(self, source):
        print(f"\n[Timeline] Branching at frame {self.current_frame_idx} (Source: {source})...")
        self.override_start_frame = self.current_frame_idx
        self.override_source = source
        self.discarded_trajectory = self.trajectory[self.current_frame_idx + 1:]
        self.trajectory = self.trajectory[:self.current_frame_idx + 1]
        self.notes = [n for n in self.notes if n["frame"] <= self.current_frame_idx]
        self._restore_state(self.current_frame_idx)

    def _handle_decision(self, decision):
        if decision == "accept":
            print(f"✅ [Override] Accepted {self.override_source} segment.")
            if self.buffers:
                for step in self.discarded_trajectory[:100]:
                    if step.get('next_obs') is not None:
                        self.buffers['anti_example'].push(step['obs'], step['action'])

                if self.override_source == "realtime":
                    # Correct Causal pairing: obs_t -> action_t
                    # The frames from override_start_frame to end are the new behavior.
                    new_part = self.trajectory[self.override_start_frame:]
                    for step in new_part:
                        if step.get('next_obs') is not None:
                            self.buffers['example'].push(
                                step['obs'], step['action'], reward=step['reward'],
                                next_obs=step['next_obs'], terminated=step['terminated'],
                                truncated=step['truncated']
                            )
            self.discarded_trajectory = []
        elif decision == "reject":
            print("❌ [Override] Rejected.")
            self.trajectory = self.trajectory[:self.override_start_frame + 1] + self.discarded_trajectory
            self.discarded_trajectory = []
            self.current_frame_idx = self.override_start_frame
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]
            self._restore_state(self.current_frame_idx)
        self.override_source = None

    def step_forward(self, action, source="rl"):
        if self.current_frame_idx < len(self.trajectory) - 1:
            self.current_frame_idx += 1
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]
            return

        if self.trajectory[-1]["terminated"] or self.trajectory[-1]["truncated"]:
            return

        # Current state before action
        obs = self.current_obs
        
        # Update the action in the current step
        self.trajectory[self.current_frame_idx]["action"] = action
        self.trajectory[self.current_frame_idx]["source"] = source
        
        res = self.env.step(action)
        if len(res) == 5:
            next_obs, reward, terminated, truncated, info = res
        else:
            next_obs, reward, terminated, info = res
            truncated = False

        frame = self.env.render()
        
        # Update current step's next_obs and results
        self.trajectory[self.current_frame_idx]["next_obs"] = next_obs
        self.trajectory[self.current_frame_idx]["reward"] = reward
        self.trajectory[self.current_frame_idx]["terminated"] = terminated
        self.trajectory[self.current_frame_idx]["truncated"] = truncated

        # Route to buffers if source is RL
        if source == "rl":
            if self.agent:
                self.agent.store_transition(obs, action, reward, next_obs, terminated, truncated)

        # Check for get_state
        env_state = None
        if hasattr(self.env, 'get_state'): env_state = self.env.get_state()
        elif hasattr(self.env.unwrapped, 'get_state'): env_state = self.env.unwrapped.get_state()

        # Create NEW step for next_obs
        step_data = {
            "obs": next_obs, "action": 0, "reward": 0.0, "next_obs": None,
            "frame_image": frame, 
            "env_state": env_state,
            "terminated": terminated, "truncated": truncated, "source": source
        }
        self.trajectory.append(step_data)
        self.current_frame_idx += 1
        self.current_obs = next_obs

        if terminated or truncated:
            self.trajectory[-1]["terminated"] = terminated
            self.trajectory[-1]["truncated"] = truncated
            if self.override_source is not None: self.mode = "decision"

    def step_backward(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.current_obs = self.trajectory[self.current_frame_idx]["obs"]

    def _switch_timer(self, target_timer):
        if not self.metrics: return
        current = getattr(self, "_active_timer", None)
        if target_timer != current:
            if current: self.metrics.stop_timer(current)
            self._active_timer = target_timer
            if target_timer: self.metrics.start_timer(target_timer)

    def draw_overlay(self, verification_phrase=None):
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
        elif self.mode == "verification": color = (255, 100, 255)

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

        if verification_phrase:
            msg = self.small_font.render(f"HEURISTIC: {verification_phrase}", True, (255, 255, 255))
            self.screen.blit(msg, (10, 90))

        if self.mode == "note":
            bg_rect = pygame.Rect(0, self.screen.get_height() - 50, self.screen.get_width(), 50)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            note_surf = self.font.render(f"> {self.text_buffer}_", True, (0, 255, 0))
            self.screen.blit(note_surf, (10, self.screen.get_height() - 40))

        if self.mode == "decision" or (self.mode == "step" and verification_phrase and self.current_frame_idx == len(self.trajectory) - 1):
            overlay = pygame.Surface((self.screen.get_width(), 120))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, self.screen.get_height() // 2 - 60))
            
            if verification_phrase:
                msg = self.font.render(f"[A]ccept Heuristic, [R]eject, or [P]rephrase?", True, (255, 255, 255))
            else:
                source_name = self.override_source.title() if self.override_source else "Changes"
                msg = self.font.render(f"[A]ccept {source_name} or [R]eject?", True, (255, 255, 255))

            self.screen.blit(msg, (self.screen.get_width() // 2 - msg.get_width() // 2, self.screen.get_height() // 2 - 20))
        
        current_note = next((n["text"] for n in self.notes if n["frame"] == self.current_frame_idx), None)
        if current_note:
            note_display = self.small_font.render(f"NOTE: {current_note}", True, (255, 100, 100))
            self.screen.blit(note_display, (10, 80))

        pygame.display.flip()

    def run_verification(self, start_frame, action, termination_rule, phrase):
        print(f"\n[Verification] Playback: '{phrase}'")
        self._restore_state(start_frame)
        self.mode = "verification"; self.running = True
        self._switch_timer("human_reviewing")
        verification_obs = self.trajectory[start_frame]["obs"]
        timeout_frames = 100; frames_run = 0; final_decision = None; rephrased_text = None

        while self.running:
            events = pygame.event.get()
            new_mode, self.text_buffer, submitted_note, step_dir, reset, branch_timeline, decision = process_events(events, self.mode, self.text_buffer)
            self._switch_timer("human_annotating" if new_mode == "note" else "human_reviewing")
            if submitted_note: rephrased_text = submitted_note; final_decision = "rephrase"; break
            if decision: final_decision = decision; break
            if new_mode == "quit": break
            if self.mode == "verification":
                if not termination_rule(verification_obs) and frames_run < timeout_frames:
                    obs, reward, term, trunc, info = self.env.step(action)
                    verification_obs = obs; frame = self.env.render()
                    self.trajectory.append({"obs": obs, "frame_image": frame, "source": "heuristic"})
                    self.current_frame_idx = len(self.trajectory) - 1; frames_run += 1
                else: self.mode = "decision"
            self.draw_overlay(verification_phrase=phrase); self.clock.tick(self.fps)
        return final_decision, rephrased_text

    def ensure_screen(self):
        if self.screen is not None: return
        frame = self.env.render()
        if frame is None and self.trajectory: frame = self.trajectory[0].get("frame_image")
        if frame is not None:
            h, w, _ = frame.shape
            self.screen = pygame.display.set_mode((w, h))

    def run(self):
        self.running = True
        if not self.trajectory: self.reset_env()
        else: self.ensure_screen()
        self._switch_timer("human_reviewing")
        step_counter = 0
        while self.running:
            events = pygame.event.get()
            new_mode, self.text_buffer, submitted_note, step_dir, reset, branch_timeline, decision = process_events(events, self.mode, self.text_buffer)
            t = "human_reviewing"
            if new_mode == "realtime": t = "human_overriding"
            elif new_mode == "note": t = "human_annotating"
            elif new_mode in ["quit", "finish"]: t = None
            self._switch_timer(t)
            self.mode = new_mode
            if self.mode in ["quit", "finish"]: break
            if branch_timeline: self._branch_timeline(source=self.mode)
            if decision: self._handle_decision(decision)
            if reset: self.reset_env()
            if submitted_note:
                note_data = {"frame": self.current_frame_idx, "text": submitted_note, "obs_context": self._format_obs(self.trajectory[self.current_frame_idx]["obs"])}
                self.notes.append(note_data)
                if self.buffers: self.buffers['llm'].push(self.trajectory, self.current_seed, submitted_note, self.current_frame_idx, note_data["obs_context"])
            if self.mode == "step":
                if step_dir != 0:
                    step_counter += 1
                    if step_dir == 1: self.step_forward(action=0, source="manual")
                    elif step_dir == -1: self.step_backward()
                else: step_counter = 0
            elif self.mode == "realtime":
                action = get_realtime_action(pygame.key.get_pressed(), env_name=self.env_name)
                self.step_forward(action, source="human")
                if self.metrics: self.metrics.log_frames(1, source="human")
            elif self.mode == "agent":
                action = self.agent.predict(self.current_obs) if self.agent else self.env.action_space.sample()
                self.step_forward(action, source="rl")
                if self.metrics: self.metrics.log_frames(1, source="rl")
            self.draw_overlay(); self.clock.tick(self.fps)
        return self.trajectory, self.notes, self.current_seed
