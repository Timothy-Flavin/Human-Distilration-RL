source venv/bin/activate

for env in "crafter" "LunarLander-v3" "highway-v0"
do
  for name in "baseline_awrcql" "baseline_bc" "baseline_rcql" "online_awbc_handsfree"
  do
    python plot_results_aggregate.py --env "$env" --experiment_name "$name"
  done
done