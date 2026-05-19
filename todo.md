# Project TODO: Ground Truth Alignment
Use this virtual environment for locally compiled torch 2.13: `source ../../../opt/pytorch-build/venv/bin/activate`
Read the project goals in `project_goal.md` or the more compressed `requirements.md` to familiarize youreself with the purpose of this environment. 
The following gaps remain to achieve 100% alignment with `project_goal.md`.

## 1. Metrics & Telemetry (R5.2)
- [X] **Task**: Add `time_human_reviewing` to `MetricsLogger`.
- [X] **Task**: Update `InteractiveGymWrapper` to start/stop the `human_reviewing` timer when the interactive window is in reviewing mode. When the used is skimming through the playback footage the `human_reviewing` time should be incrementing to include time fiddling with the UI.
- [X] **Task**: Add `frames_generated_ssl` and `frames_generated_curriculum` to frame counters.

## 8. Noisy Human Trajectories (New Method)
- [X] **Task**: Update `LLMRouter.classify` to recognize `NOISY_HUMAN` annotations where the user specifies unimportant features (e.g., "ignore x_pos", "don't care about height").
- [X] **Task**: Implement noise addition logic in `LLMRouter.commit` for `NOISY_HUMAN` classifications. For each frame in the human-controlled segment of the trajectory, generate $N$ noisy variations of the observation by perturbing the unimportant features while keeping the action constant.
- [X] **Task**: Ensure these noisy transitions are pushed to the `example_buffer` for Behavior Cloning.
- [X] **Task**: Maintain the existing `HEURISTIC` (SSL Mining) method as a baseline for the paper's negative results. 
- [X] **Task**: Add a `noise_scale` parameter to `LLMRouter` to control the magnitude of feature perturbation.

## 2. SSL Buffer Mining (R4.2) [LEGACY - Negative Result]
- [X] **Task**: Review the annotation recorder...
- [X] **Task**: Update `llm_router.py` to provide a `rule_lambda`...
- [X] **Task**: Modify `agent.ssl_update` or `main.py` to search...

## 3. Unified Update Epochs (R5.1)
- [X] **Task**: Refactor the update section in `main.py` to perform a unified update loop where all buffers are sampled and updated over $N$ epochs, ensuring consistent convergence.

## 4. LunarLander Formatter Enhancement (R2.5)
- [X] **Task**: Ensure the `InteractiveGymWrapper._format_obs` provides a full, pretty-printed JSON-like context for the LLM as requested in the ground truth.

## 5. Curriculum Stability (R3.3)
Implement 3 methods for handling auxiliary curriculum learning selected by a command line arg.
- [X] **Task**: (Current) Perform RL updates on the auxiliary buffer using the main model and optimizer. 
- [X] **Task**: (Most stable) Keep a second Agent and copy the weights from the main agent, then train the second agent on the auxiliary reward curriculum stages. Train this agent on both the global and auxiliary buffers, and add the experience from the aux buffer to the `global_rl_buffer` with only the DEFAULT ENV REWARDS so that the auxiliary agent can focus on the annotations to show the main agent the desired behavior without skewing it's Q values towards the auxiliary tasks. (such as seeking instability to get the stabilize reward) 
- [X] **Task**: (Most Certain) In each iteration copy the main agent weights to the auxiliary agent, then train the auxilairy agent offline on the auxiliary buffer for some time, then while the main agent is training on the `global_rl_buffer`, add a KL-Divergence penalty pulling it towards the behavior of the auxiliary agent. 

## 6. Count Frames, not Episodes (R5.1)
- [X] **Task**: Collect data from each stage for a set number of frames instead of a set number of episodes so that the total experience per iteration and per learning-method is consistent from run to run. Iteration 5 does not mean the same thing when one had a human episode that took 800 frames (100 model updates) while the other tipped over at frame 90

## 7. Improved Cloning (R2.6)
- [X] **Task**: Add a command line arg and related implementation for advantage weighted behavior cloning to allow out-of-date human data to be ignored such as a trajectory where a human helped the Agent gain stability, then performed a clumsy landing. When the Agent gets better at landing, it should care less about the human's behavior.
