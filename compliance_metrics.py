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
    Measures tailgating penalty (Following Distance Compliance).
    - Row 0 (Ego): Absolute World Coordinates [presence, x, y, vx, vy]
    - Rows 1-4 (Others): Ego-Relative Coordinates [presence, dx, dy, dvx, dvy]
    - Threshold: 2 car lengths (~10m). In normalized units (100m scale), this is 0.1.
    """
    scores = []
    for obs in obs_sequence:
        # Reshape if flattened
        if len(obs) == 25:
            obs = obs.reshape(5, 5)
        
        min_penalty = 0
        # Rows 1-4 are other vehicles relative to ego
        for i in range(1, 5):
            presence = obs[i, 0]
            if presence < 0.5:
                continue
            
            rel_x = obs[i, 1]
            rel_y = obs[i, 2]
            
            # rel_x > 0 means the vehicle is IN FRONT of ego
            # abs(rel_y) < 0.1 means the vehicle is in the SAME LANE (lane width is 0.25)
            if rel_x > 0 and abs(rel_y) < 0.1:
                # Penalty if distance is below 'safe' threshold (0.1 normalized = 2 car lengths)
                if rel_x < 0.1:
                    # More negative the closer it gets
                    # At 0.1, penalty is -0.1. At 0.0, penalty is -1.1.
                    penalty = - ( (0.1 - rel_x) / 0.1 ) - 0.1
                    min_penalty = min(min_penalty, penalty)
        
        scores.append(min_penalty)
    
    return np.mean(scores) if scores else 0.0

def get_compliance_score(env_name, obs_sequence):
    if not obs_sequence or not isinstance(obs_sequence[0], np.ndarray):
        return 0.0
        
    # Check for image data (3D or more) and skip
    if obs_sequence[0].ndim >= 3:
        return 0.0

    if "LunarLander" in env_name:
        return lander_precision_score(obs_sequence)
    elif "highway" in env_name:
        return highway_safety_score(obs_sequence)
    return 0.0
