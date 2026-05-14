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
Goals are added to the `curriculum_queue` to be accomplished one by one.
Notes classified as `Heuristic` specify a corrective action or simple policy along with a reason or rule-of-thumb for performing that action.
An example `Heuristic` annotation may be "If you are moving down this fast and facing up, fire the main engine" which would pinpoint a few features `[vertical velocity, current_heading]` and an action `engine on` implying that the other features are not very important here. 
Heuristics are added to the `semi_supervised_queue` to be used later.
Finally, feedback may not specify a rule-of-thumb or a goal such as "You flew badly here" where the only thing the Agent can do is practice that scenario with the built in reward function from that point of interest.
These `Generic` annotations are also added to the `curriculum_queue` but the reward function is the identity function.

* **Required:** An environment-specific formatter. For LunarLander, map the 8-dimensional vector to readable text: `{"x_pos": obs[0], "y_pos": obs[1], "x_vel": obs[2], ...}`. Append this dictionary string to the note object sent to the `LLM_Buffer` so the LLM knows *exactly* what "going too fast" means numerically.

**Curriculum Learning**
For each `Goal` or `Generic` the agent learns from a particular start state with an augmented reward function to overcome sparse rewards or to enforce a particular policy among the possible set of approximately optimal policies. 
This phase is an otherwise normal RL update, besides the memory/rollout buffers. 
In these "imagined" rollouts from a start-state of interest, the transitions alongside the original reward function rewards may be added to the `global_rl_buffer`, but the auxiliary rewards may not leak into the normal buffer. 
Otherwise, a particularly large auxiliar reward for recovering from a bad state would encourage the global agent to enter that state on purpose for the large auxiliary reward. 
In the future, these rollouts may train a clone of the current policy to then record the actions as a supervised learning objective instead of training the actual current policy to keep the reward scales completely isolated. 
Esdpecially for Online RL, the buffers cant be reused to form a policy gradient so they can really only be used as a classification dataset for the actor. 

**Semi-Supervised Learning**
For each Heuristic in the `semi_supervised_queue`, a semi-supervised dataset is created by pulling all states which meet the rule of thumb requirements from the `global_rl_buffer` and the `example_buffer`. Noises is added to all of the unspecified features and a semi-sipervised learning algorithm such as FixMatch is applied to push the actor towards the rule of thumb. These annotations can force the agent to emphasize certain features that the human coach wants to be in focus instead of learning spurrious correlation with noisy features. 

### Stage 3: Incorperating the update data

Learning from each kind of human review sequantially would cause the model to forget the previous task or oscillate. Instead, the curriculum rl buffes, supervised examples and semisupervised heuristics need to be gathered and then the model needs to update from all sources in a row for multiple epochs. The exact details need to be derived via experimentation but the model needs to attempt to meet all the requirements. 


**Experimental Requirement: Telemetry Tracker**
Create a `MetricsLogger` class to record human and compute time to justify your ablation studies. It must track:

* `time_gathering_rl_experience`
* `time_human_overriding` (Wall-clock time spent in `realtime` mode)
* `time_human_reviewing` (Wall-clock time spent skimming the episode for points of interest using arrow keys)
* `time_human_annotating` (Wall-clock time spent in `note` mode)
* `time_llm_processing`
* `time_agent_updating` (Separated by BC, Anti-BC, curriculum RL, and SSL)
* `frames_generated_online_rl` vs. `frames_generated_human` vs. `frames_generated_curriculum` vs. `frames_generated_ssl`

### Stage 4: Analysing the results for publication

Each of these phases needs to be ablated with performance graphed and total real-world time taken into account.