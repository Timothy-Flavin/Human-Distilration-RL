import numpy as np
# LunarLander-v3 Heuristics Library: "Rules of Thumb"
# Action Mapping: 0: NOOP, 1: Rotate LEFT, 2: Main, 3: Rotate RIGHT

def spin_correction(o):
    if o(5)>0: return 3
    else: return 1

def drift_catcher_policy(o):
    """
    User-confirmed working logic for catching drift.
    To slow down horizontal drift, rotate opposite to velocity.
    """
    # [2]x_vel, [3]y_vel, [4]angle, [5]angular_vel
    if o[3] < -0.05: return 2 # Priority: kill downward velocity
    
    # If moving Left (-), rotate Right (3) to lean right and kill velocity
    # If moving Right (+), rotate Left (1) to lean left and kill velocity
    if o[4] > -0.05 and o[2] < 0.00: return 3 # Moving Left -> Rotate Right
    if o[4] <  0.05 and o[2] > 0.00: return 1 # Moving Right -> Rotate Left
    
    if o[4] > 0.10 and o[5] > 0: return 3 # Prevent over-tilt left (rotate right)
    if o[4] < -0.10 and o[5] < 0: return 1 # Prevent over-tilt right (rotate left)
    
    return 0

def sign(x):
    return (x >= 0)

def extreme(o):
    return abs(o[2])>0.7 or abs(o[4])>0.7 

HEURISTICS = {
    "EXTREME_RIGHT_DRIFT_CORRECTION": {
        "phrase": "extreme right drift",
        "action": 1, # Rotate Left to lean against right drift
        "feature_mask": [2, 4, 5],
        "trigger_rule": lambda o: o[2] > 0.7 and o[1] > 0.8,
        "termination_rule": lambda o: o[4] > 0.1 
    },
    "RIGHT_DRIFT_LEAN_CORRECTION": {
        "phrase": "right drift lean",
        "action": 1, # Rotate Left
        "feature_mask": [0, 1, 2, 4],
        "trigger_rule": lambda o: o[2] > 0.2 and o[4] < -0.2 and o[1] > 0.8 and o[0] > 0.1,
        "termination_rule": lambda o: o[4] > -0.05
    },
    "LEFT_DRIFT_HIGH_CORRECTION": {
        "phrase": "left drift high",
        "action": 3, # Rotate Right
        "feature_mask": [0, 1, 2],
        "trigger_rule": lambda o: o[2] < -0.5 and o[1] > 1.0 and o[0] < -0.1,
        "termination_rule": lambda o: o[4] < 0.05
    },

    "EXTREME_SPIN_PREVENTION": {
        "phrase": "extreme spin",
        "action": spin_correction, 
        "feature_mask": [5],
        "trigger_rule": lambda o: abs(o[5]) > 0.4 and o[1]>0.5 and o[3]>-0.4,
        "termination_rule": lambda o: abs(o[5]) < 0.1 or o[1]<0.3
    },
    "DRIFT_CATCHER": {
        "phrase": "catch drift",
        "action_fn": drift_catcher_policy,
        "feature_mask": [0,2, 3, 4, 5], 
        "trigger_rule": lambda o: abs(o[2]) > 0.2 and sign(o[0])==sign(o[2]) and not extreme(o),# drifting away from center
        "termination_rule": lambda o: abs(o[2]) < 0.1
    },
    "EMERGENCY_LANDING_THRUST": {
        "phrase": "emergency thrust",
        "action": 2,
        "feature_mask": [3, 6, 7],
        "trigger_rule": lambda o: o[3] < -0.7,
        "termination_rule": lambda o: o[3] > -0.1 or o[6] or o[7]
    }
}

def get_heuristic_by_text(text):
    text = text.lower()
    for key, h in HEURISTICS.items():
        if h['phrase'].lower() in text or key.lower() in text:
            return key, h
    return None, None
