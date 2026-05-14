# Active RL Distillation

This project tests an interactive LLM-assisted human-in-the-loop pipeline for generating RL policies in both sample and real-time efficient ways.
The agents (Proximal Policy Optimization PPO and Conservative Q Learning CQL) are trained by a multi-stage interaction loop.

### Stage 1: Traditional RL loop
Each agent starts by collecting and learning from a handful of environment trajectories by a stander RL loop.
CQL keeps a large replay buffer as an offline algorithm and PPO keeps a small rollout buffer.
Both are called the `global_rl_buffer` for the purposes of this repository.

### Stage 2: Human Review
A human coach is given access to the lowest performing replay from the last iteration of Stage 1. 
The human can skim through the replay frames with the left and right arrows until they see a moment of interest. 

**Human Action Correction**
Pressing spacebar allows the human to take control from that frame onwards which creates a branched timeline from the point of interest.
When a human takes control, the frames that immediately followed in the old timeline are marked as bad behavior and saved to the `anti_example` buffer.
When the human is done controlling the agent by the episode ending or hitting spacebar, they can accept or reject their gameplay.
If accepted, the new human data is added to the `example_buffer` and the current episode will now contain the previous episode data up until the branch in the timeline, then the new human data.
If rejected, the episode data is rolled back as if the human had never chosen to override in the first place. 
The human can also press tab to let the Agent control the trajectory.
The human can then jump in again if needed for a final episode trajectory that may alternate between human and RL control multiple times. 
For online RL like PPO, only histories of (obs, action) are technically necessary to be saved, but for offline RL like CQL, the full `(obs,r,next_obs,term,trunc)` tuple can be added to the `global_rl_buffer`.

It is very important to note that an environment must support `set_state()` and `get_state()` or `rng_state()` so that an episode can be resumed from mid way through by either injecting a historical state, or replaying the episode from the start with the historical actions and rng_state for a deterministic way to resume from a replay. 


**Context-Aware Human Annotation**
When the human hits `Enter` to submit a note, the wrapper must capture the *exact* observation magnitudes at that frame.
The episode, note, and meta data such as frame data at that note and which frame the note was left as are appended to the `LLM_Buffer` which serves as a queue for tasks to be processed by an LLM.
Notes can be classified as `[Goal, Heuristic, Generic]`.
Notes classified as `Goal` specify a particular subtask that the agent needs to complete at that particular moment in order to progress.
`Goal` notes are processed into an auxiliary reward function.
An example `Goal` annotation may be "Halt movement" where all forms of velocity result in a negative reward.
Notes classified as `Heuristic` specify a corrective action or simple policy along with a reason or rule-of-thumb for performing that action.
An example `Heuristic` annotation may be "


* **Required:** An environment-specific formatter. For LunarLander, map the 8-dimensional vector to readable text: `{"x_pos": obs[0], "y_pos": obs[1], "x_vel": obs[2], ...}`. Append this dictionary string to the note object sent to the `LLM_Buffer` so the LLM knows *exactly* what "going too fast" means numerically.

---
**1. Buffer Implementations**
Create a `buffers.py` module to handle the five specific datasets:

* **`example_buffer` & `anti_example_buffer`:** Standard FIFO replay buffers storing `(obs, action)` tensors.
* **`LLM_Buffer`:** A queue of dictionaries containing: `{episode_trajectory, seed, note_text, note_frame_idx, current_obs_dict}`.
* **`curriculum_buffer`:** A localized RL buffer storing: `{seed, start_frame, trajectory_length, reward_function_callable}`.
* **`semi_supervised_buffer`:** Stores `{obs, action, feature_mask}`. The `feature_mask` is a boolean array (or list of dict keys) indicating which features the LLM deemed "important."


### Phase 1: Data Structures & Telemetry

Before writing the execution loops, the foundational data structures must be established.



**2. Telemetry Tracker**
Create a `MetricsLogger` class to record human and compute time to justify your ablation studies. It must track:

* `time_gathering_rl_experience`
* `time_human_overriding` (Wall-clock time spent in `realtime` mode)
* `time_human_annotating` (Wall-clock time spent in `note` mode)
* `time_llm_processing`
* `time_agent_updating` (Separated by BC, Anti-BC, Local RL, and SSL)
* `frames_generated_rl` vs. `frames_generated_human`

---

### Phase 2: Interactive Wrapper Upgrades

The Pygame wrapper requires significant upgrades to support Accept/Reject states and context-aware note taking.



**2. The Accept/Reject Override System**

* **Actionable:** Add an `override_cache` list to your wrapper.
* When a human hits `Spacebar`, log `override_start_frame = current_frame`. Record subsequent steps into the `override_cache` instead of directly overwriting the main trajectory.
* When the human hits `Spacebar` again to relinquish control, pause the Pygame environment and display an overlay: `[A]ccept Override or [R]eject Override`.
* **If Accept:** 1. Slice the original trajectory from `override_start_frame` to the end, and push the RL actions to the `anti_example_buffer`.
2. Push the human's `override_cache` actions to the `example_buffer`.
3. Replace the main trajectory with the `override_cache`.
* **If Reject:** Clear the `override_cache`, reload the environment state to `override_start_frame` using your deterministic re-seeding/fast-forward approach, and resume playback.

---

### Phase 3: The LLM Router

This is a standalone Python service (e.g., `llm_router.py`) that consumes the `LLM_Buffer` and dispatches data to the training buffers.

**1. The Routing Prompt Setup**
Pass the LLM the following system prompt architecture:

> "You are an RL routing agent. You will receive a Human Note, the Current State Array, and the Action Space. Classify the note into: [GENERIC], [GOAL], or [HEURISTIC]."

**2. Branch Execution Logic**

* **Route 3a (Generic):** If the LLM outputs `[GENERIC]`, wrap the episode in the `curriculum_buffer` with an identity reward function: `lambda obs, next_obs, r: r`.
* **Route 3b (Goal):** If the LLM outputs `[GOAL]`, prompt it to write a Python function.
* *Actionable:* Use a strict code-generation prompt. Safely evaluate it using Python's `exec()` within a restricted namespace. Example LLM output:
```python
def custom_reward(obs, next_obs, base_r):
    return base_r + 10 if next_obs['y_vel'] > -0.1 else base_r - 1

```


* Push this callable and the episode seed to the `curriculum_buffer`.


* **Route 3c (Heuristic/Action):** If the LLM outputs `[HEURISTIC]`, have it return a JSON list of important feature indices (e.g., `[2, 3]` for velocity) and the desired action integer. Push this to the `semi_supervised_buffer`.

---

### Phase 4: Agent API Integration

Extend your `Agent.py` class to handle the new data pipelines.

**1. Supervised Updates (BC & Anti-BC)**

* **Actionable:** Map your `example_buffer` to `agent.supervised_update(obs, labels, anti=False)`. Map the `anti_example_buffer` to `anti=True`.
* *Implementation detail:* For `anti=True`, use a loss function that maximizes the distance between the predicted action distribution and the rejected action (e.g., Negative Log Likelihood, or pushing the specific action's logit down while applying a small entropy bonus to the others).

**2. Localized RL (Curriculum)**

* **Actionable:** Update `agent.rl_update(local=True)`. When `local=True`, the agent must instantiate a fresh, temporary local replay buffer.
* It plays out the specific episode from the `curriculum_buffer` starting at the `note_frame`, using the generated `f(obs, next_obs, r)` to calculate returns, and updates its critic/actor purely on this local manifold.

**3. Semi-Supervised Learning (FixMatch/Consistency)**

* **Actionable:** Add an `ssl_update(obs, labels, important_features_mask)` method to your agent.
* During this update, create $N$ augmented versions of the `obs` tensor by adding heavy Gaussian noise to all indices *except* those flagged in the `important_features_mask`.
* Compute the loss to ensure the agent predicts the `label` action consistently across all augmented versions of the state.

---

### Phase 5: The Master Orchestration Loop

Tie it all together in `main.py`.

```python
# Pseudo-architecture for the orchestration loop
for iteration in range(TOTAL_ITERATIONS):
    # Step 1: Base RL Collection
    run_rl_collection(agent, env, num_episodes=50, save_seeds=True)
    
    # Step 2: Human Interactive Review
    sampled_episodes = sample_for_review(episodes, k=3)
    for ep in sampled_episodes:
        wrapper = InteractiveWrapper(ep)
        wrapper.run() # Handles Overrides (Accept/Reject) and Note taking
        
    # Step 3: LLM Routing
    while not LLM_Buffer.empty():
        item = LLM_Buffer.pop()
        llm_router.process(item) # Routes to Curriculum or SSL buffers
        
    # Step 4: Multi-Faceted Agent Update
    if example_buffer:
        agent.supervised_update(example_buffer.sample(), anti=False)
    if anti_example_buffer:
        agent.supervised_update(anti_example_buffer.sample(), anti=True)
    if curriculum_buffer:
        for task in curriculum_buffer:
             agent.rl_update(local=True, custom_reward=task.reward_fn, start_state=task.state)
    if semi_supervised_buffer:
        agent.ssl_update(semi_supervised_buffer.sample())
        
    # Step 5: Log Telemetry & Checkpoint
    metrics.log_iteration()
    agent.checkpoint_model(specific_name=f"iter_{iteration}")

```