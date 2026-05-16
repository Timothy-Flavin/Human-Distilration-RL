# "Golden Path" Usage Examples: LunarLander-v3

To test the system's ability to process goals and heuristics reliably before introducing LLM stochasticity, use the following exact phrases in the Pygame annotation window (triggered by `Enter`).

## 1. Triggering Heuristics (Semi-Supervised Learning)
These phrases map directly to the `HEURISTICS` library in `LunarLander_v3_heuristics.py`. They will trigger buffer mining and consistency training on specific features.

| Instruction Phrase | Heuristic Key | Action | Focus Features |
| :--- | :--- | :--- | :--- |
| `unrecoverable spin` | `UNRECOVERABLE_SPIN_PREVENTION` | Dynamic (Counter-rotate) | Angular Velocity |
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

## 4. Workflow for Testing
1. **Interactive Review**: Run `main.py` and wait for the interactive window.
2. **Identify Moment**: Use arrow keys to find a frame where the agent is failing (e.g., drifting too fast).
3. **Annotate**: Press `Enter`, type a phrase from the "Golden Path" (e.g., `catch drift`), and press `Enter` again to submit.
4. **Finish**: Close the window or press `q`.
5. **Verify**: If you left a heuristic, the verification playback will start. Press `A` to accept if it looks correct.
6. **Unified Update**: Check the console output for `[SSL Mining] Found X matching states` or `[Curriculum] Replaying tasks`.
