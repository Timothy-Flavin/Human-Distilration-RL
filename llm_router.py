import json
import re
import numpy as np
from LunarLander_v3_heuristics import HEURISTICS, get_heuristic_by_text

class LLMRouter:
    def __init__(self, curriculum_buffer, ssl_buffer, global_buffer=None, example_buffer=None, metrics=None, noise_scale=0.1, num_noisy_samples=5):
        self.curriculum_buffer = curriculum_buffer
        self.ssl_buffer = ssl_buffer
        self.global_buffer = global_buffer
        self.example_buffer = example_buffer
        self.metrics = metrics
        self.heuristics_file = "LunarLander_v3_heuristics.py"
        self.noise_scale = noise_scale
        self.num_noisy_samples = num_noisy_samples

    def process(self, item):
        """Processes a single item from the LLMBuffer (Classify then Commit)."""
        classification = self.classify(item)
        if classification:
            return self.commit(item, classification)
        return None

    def classify(self, item):
        """Classifies a single item without pushing to buffers."""
        text = item['note_text']
        obs_context = item['current_obs_dict']
        
        # 1. Try matching against existing library first
        classification = self._mock_llm_classify(text, obs_context)
        
        # 2. If it's a generic classification but the text looks like a code block, try dynamic integration
        if classification['type'] == 'GENERIC' and ("```python" in text or "NEW_HEURISTIC" in text):
            dynamic_h = self.integrate_llm_heuristic(text)
            if dynamic_h:
                classification = dynamic_h
        
        # 3. Ensure all components are callable (for verification playback)
        if classification['type'] == 'HEURISTIC':
            classification = self._ensure_callable(classification)
            
        return classification

    def integrate_llm_heuristic(self, llm_string):
        """Extracts and executes a heuristic code block from LLM output."""
        # Extract code block using regex
        code_match = re.search(r"```python\n(.*?)\n```", llm_string, re.DOTALL)
        code = code_match.group(1) if code_match else llm_string
        
        try:
            # Create isolated namespace for execution
            from LunarLander_v3_heuristics import sign, extreme
            local_vars = {"np": np, "sign": sign, "extreme": extreme}
            exec(code, {"np": np, "sign": sign, "extreme": extreme}, local_vars)
            
            if "NEW_HEURISTIC" in local_vars:
                # Grab the first key from the dict
                key = list(local_vars["NEW_HEURISTIC"].keys())[0]
                h_data = local_vars["NEW_HEURISTIC"][key]
                
                # Ensure it's marked as a heuristic and has the raw code for persistence
                h_data["type"] = "HEURISTIC"
                h_data["name"] = key
                h_data["raw_code"] = code
                
                # Standardize naming: trigger_rule -> rule
                if "trigger_rule" in h_data:
                    h_data["rule"] = h_data["trigger_rule"]
                
                return h_data
        except Exception as e:
            print(f"[Router] Error integrating dynamic heuristic: {e}")
        return None

    def _ensure_callable(self, h_data):
        """Ensures action_fn and rules are callable objects."""
        # Handle stringified legacy action_fn (e.g. from SPIN_AND_FALL_RECOVERY)
        if isinstance(h_data.get('action_fn'), str):
            code = h_data['action_fn']
            try:
                from LunarLander_v3_heuristics import sign, extreme
                local_vars = {"np": np, "sign": sign, "extreme": extreme}
                exec(code, {"np": np, "sign": sign, "extreme": extreme}, local_vars)
                for val in local_vars.values():
                    if callable(val):
                        h_data['action_fn'] = val
                        break
            except Exception as e:
                print(f"[Router] Error compiling stringified action_fn: {e}")
        return h_data

    def commit(self, item, classification, verification_trajectory=None):
        """Commits a classification by pushing to curriculum, SSL, or example buffers."""
        classification = self._ensure_callable(classification)
        
        # Extract historical actions up to the note_frame_idx (exclude dummy action at index 0)
        historical_actions = [step['action'] for step in item['episode_trajectory'][1:item['note_frame_idx'] + 1]]
        
        if classification['type'] == 'GENERIC':
            self.curriculum_buffer.push(
                seed=item['seed'],
                start_frame=item['note_frame_idx'],
                trajectory_length=50, 
                reward_function_callable=lambda obs, next_obs, r: r,
                historical_actions=historical_actions
            )
        elif classification['type'] == 'GOAL':
            reward_fn = self._create_reward_fn(classification['code'])
            self.curriculum_buffer.push(
                seed=item['seed'],
                start_frame=item['note_frame_idx'],
                trajectory_length=100,
                reward_function_callable=reward_fn,
                historical_actions=historical_actions
            )
        elif classification['type'] == 'HEURISTIC':
            # R4.2 & Paper Telemetry: 
            # Only store the PRISTINE trajectory human just approved.
            # Only NEW frames generated here count as interactions.
            
            term_rule = classification.get('termination_rule')
            committed_count = 0
            
            # 1. Push only frames from the human-verified trajectory
            if verification_trajectory:
                start_idx = item['note_frame_idx']
                for step in verification_trajectory[start_idx:]:
                    if step.get('source') == 'heuristic':
                        self.ssl_buffer.push(
                            step['obs'], 
                            step['action'], 
                            classification['feature_mask'], 
                            termination_rule=term_rule
                        )
                        committed_count += 1
                        # Log ONLY these as new interactions
                        if self.metrics: self.metrics.log_frames(1, source="ssl")

            # Store metadata for dynamic mining during training
            classification['committed_frames'] = committed_count
            print(f"[SSL Commitment] Added {committed_count} pristine verified frames for heuristic: {classification.get('name', 'Unknown')}")
            
            # 2. Persistence: If this was a dynamic heuristic, save it to the library file
            if "raw_code" in classification:
                try:
                    with open(self.heuristics_file, "r") as f:
                        content = f.read()
                    
                    # Prevent duplicate appends
                    if classification["name"] not in content:
                        print(f"[Router] Persisting new heuristic '{classification['name']}' to {self.heuristics_file}")
                        with open(self.heuristics_file, "a") as f:
                            f.write(f"\n\n# Dynamically generated heuristic: {classification['name']}\n")
                            f.write(classification["raw_code"])
                            # Add to the live HEURISTICS dictionary in the file
                            f.write(f"\nHEURISTICS.update(NEW_HEURISTIC)\n")
                except Exception as e:
                    print(f"[Router] Error persisting heuristic: {e}")

        elif classification['type'] == 'NOISY_HUMAN':
            # Identify the specific contiguous human segment associated with this note
            note_idx = item['note_frame_idx']
            trajectory = item['episode_trajectory']
            
            # 1. Find the start and end of the human segment containing or preceding note_idx
            # If the note is on an RL frame, we look backwards for the closest human segment
            search_start = note_idx
            while search_start >= 0 and trajectory[search_start].get('source') != 'human':
                search_start -= 1
            
            if search_start < 0:
                print(f"[Router] Warning: No human segment found preceding frame {note_idx} for NOISY_HUMAN.")
                return classification

            # Find boundaries of this specific "island" of human control
            seg_start = search_start
            while seg_start > 0 and trajectory[seg_start - 1].get('source') == 'human':
                seg_start -= 1
            
            seg_end = search_start
            while seg_end < len(trajectory) - 1 and trajectory[seg_end + 1].get('source') == 'human':
                seg_end += 1
            
            human_segment = trajectory[seg_start : seg_end + 1]
            
            noise_specs = classification.get('noise_specs', [])
            committed_count = 0
            
            for step in human_segment:
                obs = step['obs']
                action = step['action']
                
                # Push the original human step
                self.example_buffer.push(obs, action)
                committed_count += 1
                
                # Generate noisy variations
                for _ in range(self.num_noisy_samples):
                    noisy_obs = obs.copy()
                    for spec in noise_specs:
                        idx = spec['feature']
                        dist = spec.get('dist', 'gaussian')
                        
                        if dist == 'uniform':
                            # Sample directly from the specified range (Uniform Support)
                            noisy_obs[idx] = np.random.uniform(spec['low'], spec['high'])
                        else:
                            # Add Gaussian noise with optional clipping
                            scale = spec.get('scale', self.noise_scale)
                            noise = np.random.normal(0, scale)
                            val = noisy_obs[idx] + noise
                            if 'clip_low' in spec: val = max(val, spec['clip_low'])
                            if 'clip_high' in spec: val = min(val, spec['clip_high'])
                            noisy_obs[idx] = val
                    
                    self.example_buffer.push(noisy_obs, action)
                    committed_count += 1
            
            print(f"[Noisy Human] Added {committed_count} augmented frames (Original + Advanced Noise) for annotation: '{item['note_text']}'")
            
        return classification

    def _mock_llm_classify(self, text, obs):
        """
        Mocks LLM classification logic with specific keywords.
        Reference LunarLander-V3.md for qualitative magnitudes.
        """
        text = text.lower()
        
        # --- 1. Check for Noisy Human (New Method) ---
        if any(kw in text for kw in ["ignore", "don't care", "unimportant", "doesn't matter"]):
            noise_specs = []
            
            # Helper to map names to indices
            feat_map = {
                "x_pos": 0, "horizontal position": 0,
                "y_pos": 1, "height": 1, "altitude": 1,
                "x_vel": 2, "horizontal velocity": 2,
                "y_vel": 3, "vertical velocity": 3,
                "angle": 4, "tilt": 4,
                "angular_vel": 5, "spin": 5
            }
            
            # Default bounds for LunarLander
            bounds = {
                0: (-1.0, 1.0), # x_pos
                1: (0.0, 1.5),  # y_pos
                2: (-1.0, 1.0), # x_vel
                3: (-1.0, 1.0), # y_vel
                4: (-1.0, 1.0), # angle
                5: (-1.0, 1.0)  # angular_vel
            }

            for name, idx in feat_map.items():
                if name in text:
                    spec = {"feature": idx}
                    
                    # 1. Check for Uniform Range (e.g. "between 0.4 and 1.0")
                    range_match = re.search(fr"{name}.*?between\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)", text)
                    if range_match:
                        spec.update({
                            "dist": "uniform",
                            "low": float(range_match.group(1)),
                            "high": float(range_match.group(2))
                        })
                    
                    # 2. Check for "above" or "below" (Uniform with one bound)
                    elif f"{name} above" in text:
                        val_match = re.search(fr"{name}\s+above\s+(-?[\d.]+)", text)
                        if val_match:
                            spec.update({
                                "dist": "uniform",
                                "low": float(val_match.group(1)),
                                "high": bounds[idx][1]
                            })
                    elif f"{name} below" in text:
                        val_match = re.search(fr"{name}\s+below\s+(-?[\d.]+)", text)
                        if val_match:
                            spec.update({
                                "dist": "uniform",
                                "low": bounds[idx][0],
                                "high": float(val_match.group(1))
                            })
                    
                    # 3. Check for specific Gaussian scale (e.g. "gaussian 0.2")
                    elif "gaussian" in text:
                        g_match = re.search(r"gaussian\s+([\d.]+)", text)
                        spec["dist"] = "gaussian"
                        if g_match: spec["scale"] = float(g_match.group(1))
                    
                    # 4. Default to standard Gaussian if no specific distribution mentioned
                    else:
                        spec["dist"] = "gaussian"
                        spec["scale"] = self.noise_scale

                    noise_specs.append(spec)
            
            if noise_specs:
                return {
                    "type": "NOISY_HUMAN",
                    "noise_specs": noise_specs
                }

        # --- 2. Check Heuristics Library First (SSL) ---
        h_name, h_data = get_heuristic_by_text(text)
        if h_data:
            action = h_data.get('action')
            # Dynamic action assignment for direction-dependent rules
            if action is None and h_data.get('action_fn') is None:
                if h_name == "UNRECOVERABLE_SPIN_PREVENTION":
                    ang_vel = obs['angular_vel']
                    action = 3 if ang_vel > 0 else 1 # If spinning left (+), rotate right (3)
                elif h_name == "DRIFT_CATCHER":
                    x_vel = obs['x_vel']
                    action = 1 if x_vel > 0 else 3 # If drifting right (+), rotate left (1)
            
            return {
                "type": "HEURISTIC",
                "name": h_name,
                "action": action,
                "action_fn": h_data.get('action_fn'),
                "feature_mask": h_data['feature_mask'],
                "rule": h_data['trigger_rule'],
                "termination_rule": h_data['termination_rule'],
                "phrase": h_data['phrase']
            }

        # --- 2. GOALS (Reward Functions) ---
        if "gain stability" in text or "gain control" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    # Penalize velocity and angular velocity to encourage stability\n    return base_r - 0.1 * (abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[5]))"
            }
        if "straighten out" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    # Penalize linear velocities (x, y), tilt angle, and rotation speed to kill drift\n    drift_penalty = 0.5 * (abs(next_obs[2]) + abs(next_obs[3]))\n    tilt_penalty = 0.5 * abs(next_obs[4]) + 0.1 * abs(next_obs[5])\n    return base_r - (drift_penalty + tilt_penalty)"
            }
        elif "hover down" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    target_vy = -0.3\n    stability_penalty = 0.5 * (abs(next_obs[2]) + abs(next_obs[4]) + abs(next_obs[5]))\n    speed_reward = -abs(next_obs[3] - target_vy)\n    return base_r + speed_reward - stability_penalty"
            }
        elif "hover left" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    target_vx = -0.3\n    stability_penalty = 0.5 * (abs(next_obs[3]) + abs(next_obs[5]))\n    speed_reward = -abs(next_obs[2] - target_vx)\n    return base_r + speed_reward - stability_penalty"
            }
        if "hover right" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    target_vx = 0.3\n    stability_penalty = 0.5 * (abs(next_obs[3]) + abs(next_obs[5]))\n    speed_reward = -abs(next_obs[2] - target_vx)\n    return base_r + speed_reward - stability_penalty"
            }
        elif "soft landing" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    # Reward low vertical velocity when close to ground\n    reward = base_r\n    if next_obs[1] < 0.2:\n        reward += 2.0 if abs(next_obs[3]) < 0.1 else -1.0\n    return reward"
            }
            
        # --- 3. Default Fallbacks ---
        if "reward" in text or "goal" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    return base_r + 10 if next_obs['y_vel'] > -0.1 else base_r - 1"
            }
        elif "look at" in text or "ignore" in text:
            return {
                "type": "HEURISTIC",
                "action": 0,
                "feature_mask": [0]
            }
        else:
            return {"type": "GENERIC"}

    def _create_reward_fn(self, code_string):
        """Safely evaluates a code string into a callable reward function."""
        # Extremely basic safety check
        if "import" in code_string or "eval" in code_string or "exec" in code_string:
            return lambda obs, next_obs, r: r
        
        try:
            local_vars = {}
            exec(code_string, {}, local_vars)
            # Find the first function defined in the code string
            for val in local_vars.values():
                if callable(val):
                    return val
        except Exception as e:
            print(f"Error evaluating reward function: {e}")
            
        return lambda obs, next_obs, r: r
