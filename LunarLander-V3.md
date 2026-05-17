# LunarLander-v3 Meta-Information (Qualitative Magnitudes)

This document defines the qualitative thresholds for the LunarLander-v3 environment state vectors to ensure consistent interpretation of human instructions. You will recieve an instruction at the end of this text roughly following the structure "You must do 'A' because of 'B' until 'C'" or "Because of 'B' you must do 'A' until 'C'". Your job is to use this document and the connotation of the user's words to create a control function that meets their requirements where 'A' defines the action or action function, 'B' defines the set and magnitudes of relevant features and to base your simple controller on, and 'C' defines the feature states that should terminate this particular policy. 'B' and 'C' are very important because your proposed action or policy is going to be applied for all environment states where the features' requirements are met so be very careful about making the requirements too loose. You are to output your response in exactly the dictionary format shown in the example below so that it can be interpreted as python code.

## Example Command and resulting output

"You are currently spinning too fast and you have time to correct it, so fire the side engine until you are no longer spinning" {'x_pos': 0.12, 'y_pos': 1.40, 'x_vel': 0.2, 'y_vel': -0.1, 'angle': -0.10, 'angular_vel': -0.5, 'leg1_contact': False, 'leg2_contact': False, 'readable_summary': 'Pos:(0.12, 1.40), Vel:(0.2, -0.1), Angle:-0.1'}

"SPIN_PREVENTION": {
    "phrase": "high spin",
    "action_fn": "def spin_correction(o):\n    if o(5)>0: return 3\n    else: return 1", 
    "action": None, # blank because the action could be either direction based on the sign of the spin so it must be a function
    "feature_mask": [1,3,5], # We care about height, falling, and spin the rest is not currently relevant
    "trigger_rule": lambda o: abs(o[5]) > 0.4 and o[1]>0.5 and o[3]>-0.4, # you are spinning fast but not too low to correct yet
    "termination_rule": lambda o: abs(o[5]) < 0.1 or o[1]<0.3 # You stopped the spin or ran out of altitude
},

## 1. Velocity Magnitudes (x_vel, y_vel, angular_vel)
- **Barely**: `|v| < 0.05` (Almost stationary or stable)
- **Slowly**: `0.05 <= |v| < 0.2` (Controlled movement/rotation)
- **Moderately**: `0.2 <= |v| < 0.5` (Standard flight speed)
- **Fast**: `0.5 <= |v| < 1.0` (High energy, needs attention)
- **Extremely Fast**: `|v| >= 1.0` (Dangerously high, likely terminal)

## 2. Positional Magnitudes (x_pos, y_pos)
- **Centered**: `|x_pos| < 0.1`
- **Left/Right of Center**: `0.1 <= |x_pos| < 0.5`
- **Far Left/Right**: `|x_pos| >= 0.5` (Close to screen boundaries)
- **Low (Near Pad)**: `y_pos < 0.4`
- **Medium Height**: `0.4 <= y_pos < 1.0`
- **High Up**: `y_pos >= 1.0`

## 3. Orientation (angle)
- **Straight/Vertical**: `|angle| < 0.05`
- **Slightly Leaning**: `0.05 <= |angle| < 0.2`
- **Leaning**: `0.2 <= |angle| < 0.4`
- **Heavily Tilted**: `|angle| >= 0.4`
- **SIGN CONVENTION**:
    - **Positive Angle (+)**: Leaning **LEFT** (Counter-Clockwise)
    - **Negative Angle (-)**: Leaning **RIGHT** (Clockwise)

## 4. State Vector Mapping
- `obs[0]`: `x_pos` (Horizontal position, 0 is center)
- `obs[1]`: `y_pos` (Vertical position, 0 is landing pad)
- `obs[2]`: `x_vel` (Horizontal velocity. **Positive (+)** is moving **RIGHT**, **Negative (-)** is moving **LEFT**)
- `obs[3]`: `y_vel` (Vertical velocity. **Positive (+)** is moving **UP**, **Negative (-)** is moving **DOWN**)
- `obs[4]`: `angle` (Lander angle, 0 is vertical. **Positive (+)** is **LEFT**, **Negative (-)** is **RIGHT**)
- `obs[5]`: `angular_vel` (Rotational velocity. **Positive (+)** is rotating **LEFT/CCW**)
- `obs[6]`: `leg1_contact` (Boolean)
- `obs[7]`: `leg2_contact` (Boolean)

## 5. Action Mapping
- `0`: NOOP
- `1`: Rotate Lander **LEFT** (Counter-Clockwise)
- `2`: Fire Main Engine (Thrust Upwards)
- `3`: Rotate Lander **RIGHT** (Clockwise)


You are spinning out of control and going down fast, hit the correct rotation thruster until you start rotating slowely back towards the middle then try to thrust out of it with the main engine until you crash or have time to fully control the spin. {'x_pos': 0.12610730528831482, 'y_pos': 1.4390538930892944, 'x_vel': 0.3537416458129883, 'y_vel': -0.3047916293144226, 'angle': 0.519317090511322, 'angular_vel': 0.8656536340713501, 'leg1_contact': False, 'leg2_contact': False, 'readable_summary': 'Pos:(0.13, 1.44), Vel:(0.35, -0.30), Angle:0.52'}