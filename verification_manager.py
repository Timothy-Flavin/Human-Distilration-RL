import pygame
import numpy as np
from wrapper import InteractiveGymWrapper

class VerificationManager:
    def __init__(self, env, agent, buffers, metrics):
        self.env = env
        self.agent = agent
        self.buffers = buffers
        self.metrics = metrics

    def verify_heuristic(self, item, classification, router):
        """Runs the interactive verification loop for a heuristic."""
        verified = False
        current_classification = classification
        current_item = item

        while not verified:
            current_text = current_item['note_text']
            current_phrase = current_classification.get('phrase', current_text)
            print(f"\n[Verification] Starting playback for: '{current_phrase}'")
            
            # 1. Setup Wrapper for Verification
            v_wrapper = InteractiveGymWrapper(
                self.env, 
                agent=self.agent, 
                buffers=self.buffers, 
                metrics=self.metrics,
                initial_trajectory=current_item['episode_trajectory'],
                initial_seed=current_item['seed']
            )
            v_wrapper.ensure_screen()
            
            # 2. Simulate Heuristic to generate verification trajectory
            simulation_trajectory = self._simulate_heuristic(
                v_wrapper, 
                current_item['note_frame_idx'], 
                current_classification,
                current_classification.get('termination_rule', lambda o: True)
            )
            
            # 3. Load simulation into wrapper for interactive review (skimming)
            v_wrapper.trajectory = simulation_trajectory
            v_wrapper.current_frame_idx = current_item['note_frame_idx'] # Start where heuristic starts
            v_wrapper.mode = "step" # Allow skimming
            v_wrapper.override_source = "heuristic"
            
            # Run the interactive review phase
            decision = None
            new_text = None
            v_wrapper.running = True
            
            while v_wrapper.running:
                events = pygame.event.get()
                from input_handler import process_events
                
                # Check if we are at the last frame to enable decision
                is_at_end = (v_wrapper.current_frame_idx == len(v_wrapper.trajectory) - 1)
                
                # Manual check for decision keys before process_events consumes them
                dec = None
                if is_at_end:
                    for event in events:
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_a: dec = "accept"
                            elif event.key == pygame.K_r: dec = "reject"
                            elif event.key == pygame.K_p: dec = "rephrase"
                
                new_mode, v_wrapper.text_buffer, submitted_note, step_dir, reset, branch, _ = process_events(
                    events, v_wrapper.mode, v_wrapper.text_buffer
                )
                
                if submitted_note:
                    new_text = submitted_note
                    decision = "rephrase"
                    v_wrapper.running = False
                elif dec and is_at_end:
                    decision = dec
                    v_wrapper.running = False
                
                v_wrapper.mode = new_mode
                if v_wrapper.mode == "quit":
                    v_wrapper.running = False
                    decision = "reject"
                
                if v_wrapper.mode == "step":
                    if step_dir == 1: 
                        if v_wrapper.current_frame_idx < len(v_wrapper.trajectory) - 1:
                            v_wrapper.current_frame_idx += 1
                            v_wrapper.current_obs = v_wrapper.trajectory[v_wrapper.current_frame_idx]["obs"]
                    elif step_dir == -1: 
                        v_wrapper.step_backward()

                v_wrapper.draw_overlay(verification_phrase=current_phrase)
                v_wrapper.clock.tick(v_wrapper.fps)

            # 4. Handle Decision
            if decision == "accept":
                router.commit(current_item, current_classification, verification_trajectory=v_wrapper.trajectory)
                verified = True
            elif decision == "reject":
                print(f"[Verification] Heuristic rejected: '{current_phrase}'")
                verified = True # Discard and move on
            elif decision == "rephrase" and new_text:
                print(f"[Verification] Rephrasing: '{current_text}' -> '{new_text}'")
                current_item['note_text'] = new_text
                current_classification = router.classify(current_item)
                if current_classification['type'] != 'HEURISTIC':
                    router.commit(current_item, current_classification)
                    verified = True
                # Else: loop again with new heuristic classification and ORIGINAL start frame
            else:
                verified = True

    def _simulate_heuristic(self, wrapper, start_frame, classification, termination_rule):
        """Simulates the heuristic action/policy to create a trajectory for review."""
        wrapper._restore_state(start_frame)
        
        # Build trajectory up to start_frame
        traj = wrapper.trajectory[:start_frame + 1]
        current_obs = traj[-1]["obs"]
        
        timeout_frames = 150
        action_fn = classification.get('action_fn')
        fixed_action = classification.get('action')

        for _ in range(timeout_frames):
            if termination_rule(current_obs):
                break
            
            # Determine action from procedural policy or fixed value
            if action_fn:
                action = action_fn(current_obs)
            else:
                action = fixed_action if fixed_action is not None else 0
                
            obs, reward, term, trunc, info = self.env.step(action)
            frame = self.env.render()
            traj.append({
                "obs": obs,
                "reward": reward,
                "frame_image": frame,
                "terminated": term,
                "truncated": trunc,
                "source": "heuristic",
                "action": action # Record the action taken by heuristic
            })
            current_obs = obs
            if term or trunc: break
            
        return traj
