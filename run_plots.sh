source .venv/bin/activate

for env in "crafter" "LunarLander-v3" "highway-v0"
do
  for name in "baseline_awbc" "baseline_awcql" "baseline_bc" "baseline_cql" "online_awbc_handsfree" "online_bc_handsfree" "online_offline_awbc_cql" "online_offline_cql"
  do
    python plot_results_aggregate.py --env "$env" --experiment_name "$name"
  done
done