import numpy as np

# LunarLander-v3 Heuristics Library: "Rules of Thumb"
# Each heuristic maps a human "code-phrase" to a rule and a suggested action.

HEURISTICS = {
    "EXTREME_RIGHT_DRIFT_CORRECTION": {
        # Full: "The lander is facing right and moving right extremely quickly but it is still high up so it needs to fully prioritize rotating left"
        "phrase": "extreme right drift",
        "action": 3, # Fire Right Engine to rotate Left
        "feature_mask": [2, 4, 5], # x_vel, angle, angular_vel
        "rule": lambda o: o[2] > 0.7 and o[1] > 0.8
    },
    "RIGHT_DRIFT_LEAN_CORRECTION": {
        # Full: "Going right and leaning right while high up and right of the flag, you need to fire the right engine until rotating left slowely"
        "phrase": "right drift lean",
        "action": 3, # Fire Right Engine to rotate Left
        "feature_mask": [0, 1, 2, 4], # x_pos, y_pos, x_vel, angle
        "rule": lambda o: o[2] > 0.2 and o[4] < -0.2 and o[1] > 0.8 and o[0] > 0.1
    },
    "LEFT_DRIFT_HIGH_CORRECTION": {
        # Full: "you are moving left and you are 1.5x higher than you need to be to straighten out and your are left of center, you need to fire the left engine until slowely rotating right"
        "phrase": "left drift high",
        "action": 1, # Fire Left Engine to rotate Right
        "feature_mask": [0, 1, 2], # x_pos, y_pos, x_vel
        "rule": lambda o: o[2] < -0.5 and o[1] > 1.0 and o[0] < -0.1
    },
    "UNRECOVERABLE_SPIN_PREVENTION": {
        "phrase": "unrecoverable spin",
        "action": None, # Depends on direction
        "feature_mask": [5], # angular_vel
        "rule": lambda o: abs(o[5]) > 0.4
    },
    "DRIFT_CATCHER": {
        "phrase": "catch drift",
        "action": None, # Depends on direction
        "feature_mask": [2, 4, 5], # x_vel, angle, angular_vel
        "rule": lambda o: abs(o[2]) > 0.4
    },
    "EMERGENCY_LANDING_THRUST": {
        "phrase": "emergency thrust",
        "action": 2, # Main Engine
        "feature_mask": [3, 6, 7], # y_vel, leg contacts
        "rule": lambda o: o[3] < -0.7
    }
}

def get_heuristic_by_text(text):
    text = text.lower()
    for key, h in HEURISTICS.items():
        if h['phrase'].lower() in text or key.lower() in text:
            # Dynamic action assignment for direction-dependent rules
            action = h['action']
            if key == "UNRECOVERABLE_SPIN_PREVENTION":
                # If rotating left (pos), fire right engine (3) to counter
                # Wait, angular_vel positive is left. action 3 fires right engine (rotates left).
                # To rotate RIGHT, fire LEFT engine (1).
                # Let's check wrapper action mapping: 1: Left, 2: Main, 3: Right.
                # Fire Right (3) -> Rotates Left. Fire Left (1) -> Rotates Right.
                # If spinning left (pos), fire Left (1) to rotate right.
                pass # Will be handled in router for more flexibility
            return key, h
    return None, None
