import torch

class Agent():
    def __init__(self, name="CQL", save_dir="./default_environment", device_name="cpu"):
        self.name = name
        self.save_dir = save_dir
        self.device_name = device

    # take either a train action via epsilon greedy or from
    # an action distribution if deterministic = False, or take
    # the greedy max action if deterministic = True.
    def act(self, observations:torch.Tensor, deterministic:bool=False):
        pass

    # An agent should maintain it's own memory buffer and 
    # learning dynamics including batch-size etc so that
    # the runner can be agant agnostic but it should return
    # relevant stats like {"actor_loss":x, "critic_loss":y}
    # Sometimes the agent will be given a specific state 
    # to branch from repeatedly to fix a local part of the 
    # policy that is broken on a specific reward heuristic
    # During these updates we may not want to increment things
    # like the epsilon schedule or other counters and 
    # the agent likely will have a local_buffer instead of a
    # massive general purpose replay buffer.
    def rl_update(self, local:bool=False)->dict:
        pass

    # Should either behavior clone or do the opposite of 
    # behavior cloning based on 'anti' keyword. For example
    # If the agent took a bad set of actions and the human
    # overrode those actions, it should both push away from
    # it's historical actions and towards the human ones. 
    def supervised_update(self, obs:torch.Tensor, labels:torch.Tensor, anti:bool=False):
        pass

    # Save the model weights with a specific name appended 
    # optionally such as "final" or "english_only"
    def checkpoint_model(self, specific_name=None):
        pass

    # Load model checkpoint
    def load_model(self, specific_name=None):
        pass

