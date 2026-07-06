python main.py --env "LunarLander-v3" --algo cql --online_rl --offline_rl --awbc \
        --num_rl_frames 2000 \
        --num_unified_epochs 500 \
        --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" \
        --experiment_name "online_awbc_handsfree" --seed 1 

# old cpu times
# "timers": {
#     "rl_experience": 13.318066120147705,
#     "human_overriding": 0.0,
#     "human_reviewing": 0.0,
#     "human_annotating": 0.0,
#     "llm_processing": 0.0,
#     "agent_updating_bc": 19.914959192276,
#     "agent_updating_anti_bc": 0.0,
#     "agent_updating_local_rl": 0.0,
#     "agent_updating_ssl": 0.0,
#     "agent_updating_rl": 55.13386845588684,
#     "agent_updating_value": 18.432204246520996,
#     "expert_preload_effort": 250.19999999999996
# },
# "frames": {
#     "rl": 20000,
#     "human": 0,
#     "curriculum": 0,
#     "ssl": 0,
#     "expert_preload": 7506
# }
# New GPU times
# "timers": {
#     "rl_experience": 26.690226793289185,
#     "human_overriding": 0.0,
#     "human_reviewing": 0.0,
#     "human_annotating": 0.0,
#     "llm_processing": 0.0,
#     "agent_updating_bc": 27.906219720840454,
#     "agent_updating_anti_bc": 0.0,
#     "agent_updating_local_rl": 0.0,
#     "agent_updating_ssl": 0.0,
#     "agent_updating_rl": 71.76118493080139,
#     "agent_updating_value": 27.805789470672607,
#     "expert_preload_effort": 250.19999999999996
# },
# "frames": {
#     "rl": 20000,
#     "human": 0,
#     "curriculum": 0,
#     "ssl": 0,
#     "expert_preload": 7506
# }

python recurrent_main.py --env "crafter" --online_rl --offline_rl --awbc --bc \
        --num_rl_frames 2000 \
        --num_unified_epochs 200 \
        --preload_expert_data "cleaned_expert_demonstrations_crafter.pkl" \
        --experiment_name "online_awbc_handsfree" --seed 1