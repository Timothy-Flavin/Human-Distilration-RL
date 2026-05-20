# Highway-v0 Meta-Information (Qualitative Magnitudes)

This document defines the qualitative thresholds for the Highway-v0 environment state vectors (Kinematic Observation). You will receive an instruction at the end of this text. Your job is to create a control function or classify the human intent.

## 1. Observation Space (Kinematics)
The observation is a $V \times F$ matrix (default $5 \times 5$). Each row represents a vehicle.
- `obs[0]`: Ego-vehicle
- `obs[1-4]`: Nearest neighbors

### Feature Mapping (per vehicle):
- `f[0]`: **Presence** (1 if present, 0 if empty)
- `f[1]`: **x** (Longitudinal position)
- `f[2]`: **y** (Lateral position/Lane)
- `f[3]`: **vx** (Longitudinal velocity)
- `f[4]`: **vy** (Lateral velocity)

## 2. Action Mapping
- `0`: **LANE_LEFT** (Switch to the lane on the left)
- `1`: **IDLE** (Maintain current lane and speed)
- `2`: **LANE_RIGHT** (Switch to the lane on the right)
- `3`: **FASTER** (Accelerate)
- `4`: **SLOWER** (Decelerate)

## 3. Qualitative Thresholds
- **Close**: Distance $< 0.1$
- **Far**: Distance $> 0.5$
- **Fast**: Velocity $> 0.7$
- **Slow**: Velocity $< 0.3$

## 4. Example
"Someone is right in front of me, move left"
{'type': 'HEURISTIC', 'name': 'AVOID_FRONT_COLLISION', 'action': 0, 'feature_mask': [1, 2], 'trigger_rule': lambda o: o[1, 1] - o[0, 1] < 0.1 and abs(o[1, 2] - o[0, 2]) < 0.05}
