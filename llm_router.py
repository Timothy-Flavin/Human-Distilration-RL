import json
import re

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
                
                print(f"[SSL Mining] Found {mined_count} matching states for rule.")
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
        
        LunarLander State Context:
        [0] x_pos, [1] y_pos, [2] x_vel, [3] y_vel, [4] angle, [5] angular_vel, [6] leg1_contact, [7] leg2_contact
        
        Actions:
        0: None, 1: Left Engine, 2: Main Engine, 3: Right Engine
        """
        text = text.lower()
        
        # --- GOALS (Reward Functions) ---
        if "gain stability" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    # Penalize velocity and angular velocity to encourage stability\n    return base_r - 0.1 * (abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[5]))"
            }
        elif "soft landing" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    # Reward low vertical velocity when close to ground\n    reward = base_r\n    if next_obs[1] < 0.2:\n        reward += 2.0 if abs(next_obs[3]) < 0.1 else -1.0\n    return reward"
            }
            
        # --- HEURISTICS (SSL / Feature Selection) ---
        elif "unrecoverable spin" in text:
            ang_vel = obs['angular_vel']
            if abs(ang_vel) > 0.5:
                return {
                    "type": "HEURISTIC",
                    "action": 3 if ang_vel > 0 else 1,
                    "feature_mask": [5],
                    "rule": lambda o: abs(o[5]) > 0.4 # Slightly lower threshold for buffer mining
                }
        
        elif "catch drift" in text:
            x_vel = obs['x_vel']
            if abs(x_vel) > 0.5:
                return {
                    "type": "HEURISTIC",
                    "action": 3 if x_vel > 0 else 1,
                    "feature_mask": [2, 4, 5],
                    "rule": lambda o: abs(o[2]) > 0.4
                }

        elif "emergency thrust" in text:
            if obs['y_vel'] < -0.8:
                return {
                    "type": "HEURISTIC",
                    "action": 2,
                    "feature_mask": [3, 6, 7],
                    "rule": lambda o: o[3] < -0.7
                }
            
        # Default fallback
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
