import numpy as np

# Highway-v0 Heuristics Library
# Action Mapping: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER

HEURISTICS = {
    "EMERGENCY_BRAKE": {
        "phrase": "emergency brake",
        "action": 4, # SLOWER
        "feature_mask": [0, 1, 2],
        "trigger_rule": lambda o: o[1, 0] > 0.5 and (o[1, 1] - o[0, 1]) < 0.1 and abs(o[1, 2] - o[0, 2]) < 0.05,
        "termination_rule": lambda o: (o[1, 1] - o[0, 1]) > 0.3
    },
    "FASTER_CLEAR_ROAD": {
        "phrase": "speed up",
        "action": 3, # FASTER
        "feature_mask": [0, 1, 2, 3],
        "trigger_rule": lambda o: o[1, 0] < 0.5 or (o[1, 1] - o[0, 1]) > 0.5,
        "termination_rule": lambda o: o[0, 3] > 0.8 or (o[1, 1] - o[0, 1]) < 0.2
    },
    "AVOID_AND_OVERTAKE": {
        "phrase": "overtake",
        "action": 0, # LANE_LEFT
        "feature_mask": [0, 1, 2],
        "trigger_rule": lambda o: (o[1, 1] - o[0, 1]) < 0.15 and abs(o[1, 2] - o[0, 2]) < 0.05,
        "termination_rule": lambda o: abs(o[1, 2] - o[0, 2]) > 0.1
    }
}

def get_heuristic_by_text(text):
    text = text.lower()
    for key, h in HEURISTICS.items():
        if h['phrase'].lower() in text or key.lower() in text:
            return key, h
    return None, None
