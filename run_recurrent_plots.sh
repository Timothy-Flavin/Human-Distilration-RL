source .venv/bin/activate

for env in "crafter"
do
  for name in "online_dqn_n5" "r2d3_ne5" "online_offline_anneal" #"online_offline" "online_awbc" "online_dqn" "baseline_awrcql" "baseline_bc" "baseline_rcql" "online_awbc_handsfree" "online_offline_handsfree" "online_offline_awbc"
  do
    python plot_results_aggregate.py --env "$env" --experiment_name "$name"
  done
done