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
        """Mocks LLM classification logic."""
        text = text.lower()
        if "reward" in text or "goal" in text:
            return {
                "type": "GOAL",
                "code": "def custom_reward(obs, next_obs, base_r):\n    return base_r + 10 if next_obs['y_vel'] > -0.1 else base_r - 1"
            }
        elif "look at" in text or "ignore" in text:
            # Heuristic example: "Look at the x position"
            return {
                "type": "HEURISTIC",
                "action": 0, # NOOP
                "feature_mask": [0] # x_pos is index 0 in LunarLander
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
