import numpy as np

def lander_precision_score(obs_sequence):
    """
    Measures compliance with 'hover-then-descend' behavior.
    - Rewards vertical stability (low y_vel) when |x| > 0.1 (not near center).
    - Rewards horizontal stability (low x_vel) when |x| < 0.1 (near center).
    """
    scores = []
    for obs in obs_sequence:
        x_pos = obs[0]
        x_vel = obs[2]
        y_vel = obs[3]
        
        if abs(x_pos) > 0.1:
            # Not centered: vertical stability is prioritized
            score = -abs(y_vel)
        else:
            # Centered: horizontal stability is prioritized (dropping straight down)
            score = -abs(x_vel)
        scores.append(score)
    return np.mean(scores)

def highway_safety_score(obs_sequence):
    """
    Measures tailgating penalty.
    - Standard highway-env observation is (5, 5) flattened to (25,).
    - obs[0:5] is ego [presence, x, y, vx, vy]
    - obs[5:10] is lead vehicle
    """
    scores = []
    for obs in obs_sequence:
        # Reshape if flattened
        if len(obs) == 25:
            obs = obs.reshape(5, 5)
        
        ego_x = obs[0, 1]
        lead_presence = obs[1, 0]
        lead_x = obs[1, 1]
        
        if lead_presence > 0.5:
            dist = lead_x - ego_x
            # Penalty if distance is below 'safe' threshold (e.g. 0.1 normalized)
            # and ego is behind (dist > 0)
            if 0 < dist < 0.15:
                penalty = - (0.15 - dist) * 10
            else:
                penalty = 0
        else:
            penalty = 0
        scores.append(penalty)
    return np.mean(scores)

def get_compliance_score(env_name, obs_sequence):
    if "LunarLander" in env_name:
        return lander_precision_score(obs_sequence)
    elif "highway" in env_name:
        return highway_safety_score(obs_sequence)
    return 0.0
