import json
import re

class LLMRouter:
    def __init__(self, curriculum_buffer, ssl_buffer, llm_client=None):
        self.curriculum_buffer = curriculum_buffer
        self.ssl_buffer = ssl_buffer
        self.llm_client = llm_client # Placeholder for an actual LLM API client

    def process(self, item):
        """Processes a single item from the LLMBuffer."""
        note_text = item['note_text']
        obs_context = item['current_obs_dict']
        
        # In a real implementation, we would call the LLM here.
        # For now, we'll use a mock or simplified logic.
        classification = self._mock_llm_classify(note_text, obs_context)
        
        if classification['type'] == 'GENERIC':
            self.curriculum_buffer.push(
                seed=item['seed'],
                start_frame=item['note_frame_idx'],
                trajectory_length=50, # Default length
                reward_function_callable=lambda obs, next_obs, r: r
            )
        elif classification['type'] == 'GOAL':
            reward_fn = self._create_reward_fn(classification['code'])
            self.curriculum_buffer.push(
                seed=item['seed'],
                start_frame=item['note_frame_idx'],
                trajectory_length=100,
                reward_function_callable=reward_fn
            )
        elif classification['type'] == 'HEURISTIC':
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
        elif "fix spin" in text:
            # Mask: [5] is angular velocity
            # If spinning left (positive ang_vel), fire right engine (3). 
            # If spinning right (negative), fire left (1).
            action = 3 if obs['angular_vel'] > 0 else 1
            return {
                "type": "HEURISTIC",
                "action": action,
                "feature_mask": [5]
            }
        elif "center lander" in text:
            # Mask: [0] is x_pos
            action = 1 if obs['x_pos'] > 0 else 3
            return {
                "type": "HEURISTIC",
                "action": action,
                "feature_mask": [0]
            }
        elif "descend slow" in text:
            # Mask: [3] is y_vel
            return {
                "type": "HEURISTIC",
                "action": 2, # Main engine
                "feature_mask": [3]
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
