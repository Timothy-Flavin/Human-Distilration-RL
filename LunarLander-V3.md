# LunarLander-v3 Meta-Prompt (Qualitative Magnitudes)

This document defines the qualitative thresholds for the LunarLander-v3 environment state vectors to ensure consistent interpretation of human instructions.

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
- **Low (Near Pad)**: `y_pos < 0.3`
- **Medium Height**: `0.3 <= y_pos < 0.8`
- **High Up**: `y_pos >= 0.8`

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
