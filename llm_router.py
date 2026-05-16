import json
import re
from LunarLander_v3_heuristics import HEURISTICS, get_heuristic_by_text

class LLMRouter:
    def __init__(self, curriculum_buffer, ssl_buffer, global_buffer=None, example_buffer=None):
        self.curriculum_buffer = curriculum_buffer
        self.ssl_buffer = ssl_buffer
        self.global_buffer = global_buffer
        self.example_buffer = example_buffer

    def process(self, item):
        """Processes a single item from the LLMBuffer."""
        note_text = item['note_text']
        obs_context = item['current_obs_dict']
        
        # Extract historical actions up to the note_frame_idx (exclude dummy action at index 0)
        historical_actions = [step['action'] for step in item['episode_trajectory'][1:item['note_frame_idx'] + 1]]
        
        classification = self._mock_llm_classify(note_text, obs_context)
        
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
            # R4.2: Pull all states meeting the rule from the buffers
            rule = classification.get('rule')
            if rule:
                mined_count = 0
                # Mine from global buffer
                if self.global_buffer:
                    for obs, action in self.global_buffer.buffer:
                        if rule(obs.numpy()):
                            self.ssl_buffer.push(obs, classification['action'], classification['feature_mask'])
                            mined_count += 1
                
                # Mine from example buffer
                if self.example_buffer:
                    for obs, action in self.example_buffer.buffer:
                        if rule(obs.numpy()):
                            self.ssl_buffer.push(obs, classification['action'], classification['feature_mask'])
                            mined_count += 1
                
                print(f"[SSL Mining] Found {mined_count} matching states for heuristic: {classification.get('name', 'Unknown')}")
            else:
                # Fallback to just the current frame
                self.ssl_buffer.push(
                    obs=item['episode_trajectory'][item['note_frame_idx']]['obs'],
                    action=classification['action'],
                    feature_mask=classification['feature_mask']
                )

    def _mock_llm_classify(self, text, obs):
        """
        Mocks LLM classification logic with specific keywords.
        Reference LunarLander-V3.md for qualitative magnitudes.
        """
        text = text.lower()
        
        # --- 1. Check Heuristics Library First (SSL) ---
        h_name, h_data = get_heuristic_by_text(text)
        if h_data:
            action = h_data['action']
            # Dynamic action assignment for direction-dependent rules
            if action is None:
                if h_name == "UNRECOVERABLE_SPIN_PREVENTION":
                    # obs is the dict formatted by wrapper
                    ang_vel = obs['angular_vel']
                    action = 1 if ang_vel > 0 else 3 # If spinning left (+), fire left (1) to rotate right
                elif h_name == "DRIFT_CATCHER":
                    x_vel = obs['x_vel']
                    action = 3 if x_vel > 0 else 1 # If drifting right (+), fire right (3) to rotate left
            
            return {
                "type": "HEURISTIC",
                "name": h_name,
                "action": action,
                "feature_mask": h_data['feature_mask'],
                "rule": h_data['rule']
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
