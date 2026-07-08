
#HIL
python main.py --env LunarLander-v3 --algo cql --offline_rl --online_rl --awbc --intervention --num_rl_frames 2000 --num_unified_epochs 500 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "2_interactive_hil" --num_iterations 20

#HIL + CUR
python main.py --env LunarLander-v3 --algo cql --online_rl --offline_rl --awbc --intervention --curriculum --curriculum_method kl --num_rl_frames 2000 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "3_curriculum_hil"

# full
python main.py --env LunarLander-v3 --algo cql --online_rl --offline_rl --awbc --intervention --curriculum --curriculum_method kl --ssl --num_rl_frames 2000 --num_unified_epochs 200 --preload_expert_data "expert_demonstrations_LunarLander-v3.pkl" --experiment_name "4_full_pipeline"




