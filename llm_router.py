import json
import re
import numpy as np
import importlib

class LLMRouter:
    def __init__(self, curriculum_buffer, ssl_buffer, global_buffer=None, example_buffer=None, metrics=None, noise_scale=0.1, num_noisy_samples=5, env_name="LunarLander-v3"):
        self.curriculum_buffer = curriculum_buffer
        self.ssl_buffer = ssl_buffer
        self.global_buffer = global_buffer
        self.example_buffer = example_buffer
        self.metrics = metrics
        self.noise_scale = noise_scale
        self.num_noisy_samples = num_noisy_samples
        self.env_name = env_name

        if "LunarLander" in env_name:
            self.heuristics_module = importlib.import_module("LunarLander_v3_heuristics")
            self.heuristics_file = "LunarLander_v3_heuristics.py"
        elif "highway" in env_name:
            self.heuristics_module = importlib.import_module("Highway_heuristics")
            self.heuristics_file = "Highway_heuristics.py"
        else:
            self.heuristics_module = None
            self.heuristics_file = None

    def classify(self, item):
        text = item['note_text']
        obs_context = item['current_obs_dict']
        classification = self._mock_llm_classify(text, obs_context)
        if classification['type'] == 'GENERIC' and ("```python" in text or "NEW_HEURISTIC" in text):
            dynamic_h = self.integrate_llm_heuristic(text)
            if dynamic_h: classification = dynamic_h
        if classification['type'] == 'HEURISTIC':
            classification = self._ensure_callable(classification)
        return classification

    def integrate_llm_heuristic(self, llm_string):
        code_match = re.search(r"```python\n(.*?)\n```", llm_string, re.DOTALL)
        code = code_match.group(1) if code_match else llm_string
        try:
            local_vars = {"np": np}
            if "LunarLander" in self.env_name:
                from LunarLander_v3_heuristics import sign, extreme
                local_vars.update({"sign": sign, "extreme": extreme})
            exec(code, local_vars, local_vars)
            if "NEW_HEURISTIC" in local_vars:
                key = list(local_vars["NEW_HEURISTIC"].keys())[0]
                h_data = local_vars["NEW_HEURISTIC"][key]
                h_data["type"] = "HEURISTIC"; h_data["name"] = key; h_data["raw_code"] = code
                if "trigger_rule" in h_data: h_data["rule"] = h_data["trigger_rule"]
                if isinstance(h_data.get('feature_mask'), list):
                    spec_dict = {idx: {'dist': 'gaussian', 'scale': self.noise_scale} for idx in h_data['feature_mask']}
                    h_data['feature_mask'] = spec_dict
                return h_data
        except Exception as e: print(f"[Router] Error integrating: {e}")
        return None

    def _ensure_callable(self, h_data):
        if isinstance(h_data.get('action_fn'), str):
            code = h_data['action_fn']
            try:
                local_vars = {"np": np}
                if "LunarLander" in self.env_name:
                    from LunarLander_v3_heuristics import sign, extreme
                    local_vars.update({"sign": sign, "extreme": extreme})
                exec(code, local_vars, local_vars)
                for val in local_vars.values():
                    if callable(val): h_data['action_fn'] = val; break
            except Exception as e: print(f"[Router] Error: {e}")
        return h_data

    def commit(self, item, classification, verification_trajectory=None):
        classification = self._ensure_callable(classification)
        # Correctly capture all actions taken BEFORE the note frame to replay to that state
        historical_actions = [step['action'] for step in item['episode_trajectory'][:item['note_frame_idx']]]
        
        if classification['type'] == 'GENERIC':
            self.curriculum_buffer.push(
                seed=item['seed'], start_frame=item['note_frame_idx'], trajectory_length=50, 
                reward_function_callable=lambda obs, next_obs, r: r, historical_actions=historical_actions
            )
        elif classification['type'] == 'GOAL':
            reward_fn = self._create_reward_fn(classification['code'])
            self.curriculum_buffer.push(
                seed=item['seed'], start_frame=item['note_frame_idx'], trajectory_length=100,
                reward_function_callable=reward_fn, historical_actions=historical_actions
            )
        elif classification['type'] == 'HEURISTIC':
            term_rule = classification.get('termination_rule')
            if verification_trajectory:
                start_idx = item['note_frame_idx']
                # Invert mask: add noise to ALL features NOT in the important list
                important_features = classification.get('feature_mask', {})
                if isinstance(important_features, dict):
                    important_indices = set(important_features.keys())
                else:
                    important_indices = set()
                
                obs_dim = len(verification_trajectory[0]['obs'])
                unimportant_mask = {
                    idx: {'dist': 'gaussian', 'scale': self.noise_scale} 
                    for idx in range(obs_dim) if idx not in important_indices
                }

                for step in verification_trajectory[start_idx:]:
                    if step.get('source') == 'heuristic':
                        # Push pristine transition
                        self.ssl_buffer.push(step['obs'], step['action'], important_features, termination_rule=term_rule)
                        
                        # Augmentation: push noisy variations to example buffer for BC
                        for _ in range(self.num_noisy_samples):
                            # We use example_buffer for BC updates
                            self.example_buffer.push(
                                step['obs'], step['action'], reward=step.get('reward', 0.0),
                                next_obs=step.get('next_obs'), terminated=step.get('terminated', False),
                                truncated=step.get('truncated', False), mask=unimportant_mask
                            )
                        
                        if self.metrics: self.metrics.log_frames(1 + self.num_noisy_samples, source="ssl")

        elif classification['type'] == 'NOISY_HUMAN':
            note_idx = item['note_frame_idx']
            trajectory = item['episode_trajectory']
            search_start = note_idx
            while search_start >= 0 and trajectory[search_start].get('source') != 'human':
                search_start -= 1
            if search_start < 0: return classification
            seg_start = search_start
            while seg_start > 0 and trajectory[seg_start - 1].get('source') == 'human': seg_start -= 1
            seg_end = search_start
            while seg_end < len(trajectory) - 1 and trajectory[seg_end + 1].get('source') == 'human': seg_end += 1
            
            human_segment = trajectory[seg_start : seg_end + 1]
            noise_specs = classification.get('noise_specs', {})
            for step in human_segment:
                # 1. Push original human transition
                self.example_buffer.push(
                    step['obs'], step['action'], reward=step.get('reward', 0.0),
                    next_obs=step.get('next_obs'), terminated=step.get('terminated', False),
                    truncated=step.get('truncated', False), mask=None
                )
                
                # 2. Augmentation: Push N noisy variations
                for _ in range(self.num_noisy_samples):
                    self.example_buffer.push(
                        step['obs'], step['action'], reward=step.get('reward', 0.0),
                        next_obs=step.get('next_obs'), terminated=step.get('terminated', False),
                        truncated=step.get('truncated', False), mask=noise_specs
                    )
                
                if self.metrics: self.metrics.log_frames(1 + self.num_noisy_samples, source="human")
            print(f"[Noisy Human] Committed human segment with {self.num_noisy_samples}x augmentation.")
        return classification

    def _mock_llm_classify(self, text, obs):
        text = text.lower()
        if any(kw in text for kw in ["ignore", "don't care", "unimportant", "doesn't matter"]):
            noise_specs = {}
            if "LunarLander" in self.env_name:
                feat_map = {
                    "x_pos": 0, "horizontal position": 0,
                    "y_pos": 1, "height": 1, "altitude": 1,
                    "x_vel": 2, "horizontal velocity": 2,
                    "y_vel": 3, "vertical velocity": 3,
                    "angle": 4, "tilt": 4,
                    "angular_vel": 5, "spin": 5
                }
                bounds = {0: (-1.0, 1.0), 1: (0.0, 1.5), 2: (-1.0, 1.0), 3: (-1.0, 1.0), 4: (-1.0, 1.0), 5: (-1.0, 1.0)}
            elif "highway" in self.env_name:
                feat_map = {"presence": 0, "x": 1, "y": 2, "vx": 3, "vy": 4}
                bounds = {0: (0.0, 1.0), 1: (-1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, 1.0), 4: (-1.0, 1.0)}
            else: feat_map, bounds = {}, {}
            for name, idx in feat_map.items():
                if name in text:
                    spec = {}
                    range_match = re.search(fr"{name}.*?between\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)", text)
                    if range_match: spec.update({"dist": "uniform", "low": float(range_match.group(1)), "high": float(range_match.group(2))})
                    elif f"{name} above" in text:
                        val_match = re.search(fr"{name}\s+above\s+(-?[\d.]+)", text)
                        if val_match: spec.update({"dist": "uniform", "low": float(val_match.group(1)), "high": bounds.get(idx, (0, 1))[1]})
                    elif f"{name} below" in text:
                        val_match = re.search(fr"{name}\s+below\s+(-?[\d.]+)", text)
                        if val_match: spec.update({"dist": "uniform", "low": bounds.get(idx, (-1, 1))[0], "high": float(val_match.group(1))})
                    elif "gaussian" in text:
                        g_match = re.search(r"gaussian\s+([\d.]+)", text)
                        spec["dist"] = "gaussian"; spec["scale"] = float(g_match.group(1)) if g_match else self.noise_scale
                    else: spec["dist"] = "gaussian"; spec["scale"] = self.noise_scale
                    noise_specs[idx] = spec
            
            # If they just said "ignore unimportant" without naming them, default to x_pos and y_pos for LunarLander
            if not noise_specs and "LunarLander" in self.env_name:
                noise_specs = {0: {"dist": "gaussian", "scale": self.noise_scale}, 1: {"dist": "gaussian", "scale": self.noise_scale}}
                
            if noise_specs: return {"type": "NOISY_HUMAN", "noise_specs": noise_specs}
        if self.heuristics_module:
            h_name, h_data = self.heuristics_module.get_heuristic_by_text(text)
            if h_data:
                action = h_data.get('action')
                if action is None and h_data.get('action_fn') is None and "LunarLander" in self.env_name:
                    if h_name == "UNRECOVERABLE_SPIN_PREVENTION": action = 3 if obs.get('angular_vel', 0) > 0 else 1
                    elif h_name == "DRIFT_CATCHER": action = 1 if obs.get('x_vel', 0) > 0 else 3
                mask = h_data['feature_mask']
                if isinstance(mask, list): mask = {idx: {'dist': 'gaussian', 'scale': self.noise_scale} for idx in mask}
                return {"type": "HEURISTIC", "name": h_name, "action": action, "action_fn": h_data.get('action_fn'), "feature_mask": mask, "rule": h_data['trigger_rule'], "termination_rule": h_data['termination_rule'], "phrase": h_data['phrase']}
        if "LunarLander" in self.env_name:
            if "gain stability" in text or "straighten out" in text:
                return {
                    "type": "GOAL",
                    "code": "def custom_reward(obs, next_obs, base_r):\n    # Penalize linear velocities (x, y), tilt angle, and rotation speed to kill drift\n    drift_penalty = 0.5 * (abs(next_obs[2]) + abs(next_obs[3]))\n    tilt_penalty = 0.5 * abs(next_obs[4]) + 0.1 * abs(next_obs[5])\n    return base_r - (drift_penalty + tilt_penalty)"
                }
            elif "hover down" in text:
                return {
                    "type": "GOAL",
                    "code": "def custom_reward(obs, next_obs, base_r):\n    # target_vy is negative for downward movement\n    target_vy = -0.3\n    # Penalize horizontal velocity (2), angle (4), and angular velocity (5)\n    stability_penalty = 0.5 * (abs(next_obs[2]) + abs(next_obs[4]) + abs(next_obs[5]))\n    # Reward staying close to the slow downward target speed\n    speed_reward = -abs(next_obs[3] - target_vy)\n    return base_r + speed_reward - stability_penalty"
                }
            elif "hover left" in text:
                return {
                    "type": "GOAL",
                    "code": "def custom_reward(obs, next_obs, base_r):\n    # target_vx is negative for moving left\n    target_vx = -0.3\n    # Penalize vertical movement (3) and rotational velocity (5)\n    stability_penalty = 0.5 * (abs(next_obs[3]) + abs(next_obs[5]))\n    # Reward staying close to the slow leftward target speed\n    speed_reward = -abs(next_obs[2] - target_vx)\n    return base_r + speed_reward - stability_penalty"
                }
            elif "hover right" in text:
                return {
                    "type": "GOAL",
                    "code": "def custom_reward(obs, next_obs, base_r):\n    # target_vx is positive for moving right\n    target_vx = 0.3\n    # Penalize vertical movement (3) and rotational velocity (5)\n    stability_penalty = 0.5 * (abs(next_obs[3]) + abs(next_obs[5]))\n    # Reward staying close to the slow rightward target speed\n    speed_reward = -abs(next_obs[2] - target_vx)\n    return base_r + speed_reward - stability_penalty"
                }
            elif "soft landing" in text:
                return {
                    "type": "GOAL",
                    "code": "def custom_reward(obs, next_obs, base_r):\n    # Penalize vertical velocity when close to ground\n    land_penalty = 0.0\n    if next_obs[1] < 0.2: land_penalty = 2.0 * abs(next_obs[3])\n    return base_r - land_penalty"
                }
        elif "highway" in self.env_name:
            if "high speed" in text: return {"type": "GOAL", "code": "def custom_reward(obs, next_obs, base_r):\n    return base_r + next_obs[0, 3]"}
        return {"type": "GENERIC"}

    def _create_reward_fn(self, code_string):
        if any(kw in code_string for kw in ["import", "eval", "exec"]): return lambda obs, next_obs, r: r
        try:
            local_vars = {}
            exec(code_string, {}, local_vars)
            for val in local_vars.values():
                if callable(val): return val
        except Exception as e: print(f"Error evaluating reward: {e}")
        return lambda obs, next_obs, r: r
