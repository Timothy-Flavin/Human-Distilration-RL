# "Golden Path" Usage Examples: LunarLander-v3

To test the system's ability to process goals and heuristics reliably before introducing LLM stochasticity, use the following exact phrases in the Pygame annotation window (triggered by `Enter`).

## 1. Triggering Heuristics (Semi-Supervised Learning)
These phrases map directly to the `HEURISTICS` library in `LunarLander_v3_heuristics.py`. They will trigger buffer mining and consistency training on specific features.

| Instruction Phrase | Heuristic Key | Action | Focus Features |
| :--- | :--- | :--- | :--- |
| `extreme spin` | `EXTREME_SPIN_PREVENTION` | Dynamic (Counter-rotate) | Angular Velocity |
| `catch drift` | `DRIFT_CATCHER` | Dynamic (Counter-drift) | X-Vel, Angle, Ang-Vel |
| `emergency thrust` | `EMERGENCY_LANDING_THRUST` | Main Engine (2) | Y-Vel, Leg Contacts |
| `extreme right drift` | `EXTREME_RIGHT_DRIFT_CORRECTION` | Right Engine (3) | X-Vel, Angle, Ang-Vel |
| `right drift lean` | `RIGHT_DRIFT_LEAN_CORRECTION` | Right Engine (3) | Pos-X, Pos-Y, X-Vel, Angle |
| `left drift high` | `LEFT_DRIFT_HIGH_CORRECTION` | Left Engine (1) | Pos-X, Pos-Y, X-Vel |

## 2. Triggering Goals (Curriculum Learning)
These phrases trigger the creation of a `CurriculumBuffer` task with an auxiliary reward function.

| Instruction Phrase | Goal Description | Auxiliary Reward Logic |
| :--- | :--- | :--- |
| `gain stability` | Stability Task | Penalizes linear and angular velocities. |
| `straighten out` | Drift Killer | Penalizes horizontal drift and tilt angle. |
| `hover down` | Vertical Control | Rewards matching a slow downward target velocity (-0.3). |
| `hover left` | Horizontal Control | Rewards matching a slow leftward target velocity (-0.3). |
| `hover right` | Horizontal Control | Rewards matching a slow rightward target velocity (0.3). |
| `soft landing` | Touchdown Prep | Rewards low vertical velocity when near the pad (`y < 0.2`). |

## 3. Heuristic Verification Phase
After you close the interactive review window, the system will process any `HEURISTIC` annotations. For each one, a **Verification Playback** will start:

1. **Simulation**: The environment will jump to the frame you annotated and perform the suggested action until the termination condition is met (or a timeout occurs).
2. **Review**: Watch the playback to ensure the action (e.g., firing the engine) produces the desired result (e.g., killing rotation).
3. **Decision**:
   - **[A]ccept**: Commits the heuristic to the SSL pipeline for buffer mining.
   - **[R]eject**: Discards the heuristic entirely.
   - **[P]rephrase**: Opens a text input for you to rewrite the heuristic. The system will immediately re-classify and re-verify the new text.

## 5. Noisy Human Trajectories (Data Augmentation)
Instead of rule-based heuristics, you can take control and then tell the LLM which features are *not* important for your current behavior. The system will add noise to those features in your trajectory to help the agent generalize.

| Example Note | Identified Unimportant Features | Effect |
| :--- | :--- | :--- |
| `ignore x_pos between -0.5 and 0.5` | Horizontal Position (0) | Uniform noise in the specified range. |
| `don't care about height above 0.4` | Vertical Position (1) | Uniform noise from 0.4 up to max altitude. |
| `ignore spin (gaussian 0.2)` | Angular Velocity (5) | Increases Gaussian noise scale for rotation. |
| `x_vel doesn't matter (uniform)` | Horizontal Velocity (2) | Uniform noise across standard bounds [-1, 1]. |

### Scenario 1: Horizontal Hover (Height Unimportant)
**Context**: You are moving the lander from the left boundary toward the center pad at a constant altitude.
**Action**: Take control, use side thrusters to move right, then stabilize in the center.
**Annotation**: `ignore height between 0.4 and 1.2, focus on centering`
**Result**: Feature `1` is sampled uniformly from `[0.4, 1.2]`. The agent learns that the horizontal correction logic is valid regardless of altitude within that range.

### Scenario 2: Vertical Descent (Horizontal Position Unimportant)
**Context**: You are hovering straight down toward the pad.
**Action**: Take control, fire main engine to maintain a slow descent rate.
**Annotation**: `ignore x_pos between -0.3 and 0.3, just hover down`
**Result**: Feature `0` is sampled uniformly from `[-0.3, 0.3]`. The agent learns to manage its descent rate while ignoring centered-horizontal jitter.

### Scenario 3: Spin/Fall Recovery (Location Unimportant)
**Context**: The lander is spinning and falling rapidly after a collision.
**Action**: Take control, use counter-rotation and then main thrust to regain stable orientation.
**Annotation**: `ignore x_pos and height above 0.5, just gain control`
**Result**: Feature `1` is sampled uniformly from `[0.5, 1.5]`. The agent learns the recovery sequence as a general policy that applies anywhere high in the sky.

### Scenario 4: Positional Invariance Verification (Experimental)
**Goal**: Verify that noise augmentation on altitude allows a policy learned at $y=0.8$ to generalize to $y=0.2$.

**Setup**: 
- Run with `--ssl --noise_scale 0.2 --num_noisy_samples 10`
- Keep `--bc` turned off (to ensure only the augmented SSL data is driving the generalization).

**Workflow**:
1. **Fly**: Fly the lander to the far left at a medium altitude ($y \approx 0.6$).
2. **Demonstrate**: Take control (`Space`) and hover the lander back to the center pad.
3. **Annotate**: Hit `Enter` and type: 
   > `ignore height between 0.2 and 1.2, focus on horizontal centering`
4. **Iterate**: Repeat this for 3-5 episodes, always hovering back from the left at roughly the same height.
5. **Test**: In a new episode, fly the lander to the far left but at a *very low* altitude ($y \approx 0.2$) or *very high* ($y \approx 1.1$). Let the agent take control.
6. **Observation**: If the augmentation worked, the agent should successfully hover to the right even though it never "saw" a human demonstration at those specific altitudes.

**Workflow**:
1. Take control with `Space`.
2. Perform the desired behavior.
3. Press `Enter` and type your note (e.g., `ignore altitude`).
4. Accept the segment with `A`.
5. The system will generate $N$ noisy copies of your segment and add them to the training buffer.
