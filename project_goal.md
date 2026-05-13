# Freshman: Human-in-the-Loop RL with Dynamic Curriculum

This project implements a robust pipeline for data-efficient Reinforcement Learning (RL) using human intervention, behavior cloning, and LLM-assisted curriculum generation.

## Project Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Core RL Algorithms** | ✅ Complete | PPO and CQL (Conservative Q-Learning) are implemented and functional. |
| **Interactive Wrapper** | ✅ Complete | Pygame-based wrapper supports real-time play, step-by-step review, timeline branching, and note-taking. |
| **Behavior Cloning (BC)** | ✅ Complete | Standard BC from human demonstrations is fully integrated. |
| **Anti-BC** | ✅ Complete | Negative BC from rejected human/agent trajectories is implemented to avoid "bad" behaviors. |
| **LLM Router** | ⚠️ Partial | Basic routing logic exists; currently uses mock keyword-based classification. |
| **Semi-Supervised (SSL)** | ✅ Complete | Feature-masked noise augmentation for consistency training is implemented in both agents. |
| **Curriculum RL** | ⚠️ Partial | Buffer infrastructure exists; localized RL updates with LLM-generated reward functions are in progress. |

## Environment Context (LunarLander-v3)

The goal is to land a rocket on a landing pad at (0,0).

### Action Space (Discrete)
- `0`: Do nothing
- `1`: Fire left orientation engine
- `2`: Fire main engine
- `3`: Fire right orientation engine

### Observation Space (8D Vector)
1. `x_pos`: Horizontal coordinate
2. `y_pos`: Vertical coordinate
3. `x_vel`: Horizontal velocity
4. `y_vel`: Vertical velocity
5. `angle`: Lander angle (0 is vertical)
6. `angular_vel`: Angular velocity
7. `leg1_contact`: Boolean (0/1) - Left leg touching ground
8. `leg2_contact`: Boolean (0/1) - Right leg touching ground

---

## Technical Pipeline

1. **Experience Collection**: Agent (PPO/CQL) collects trajectories.
2. **Interactive Review**: Human reviews "worst" episodes, overrides actions (Accept/Reject), and provides natural language notes.
3. **LLM Routing**: Notes are classified into:
   - `GENERIC`: Replay with standard rewards.
   - `GOAL`: Generates a Python reward function for localized Curriculum RL.
   - `HEURISTIC`: Identifies key features for SSL (consistency training).
4. **Agent Update**: Multi-faceted update combining RL, BC, Anti-BC, SSL, and Localized RL.
