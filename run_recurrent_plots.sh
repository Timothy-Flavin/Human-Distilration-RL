source .venv/bin/activate

for env in "crafter"
do
  for name in "abl_imp_per" "abl_nat_noper" "abl_imp_elu" "online_demostart" "dqfd_demostart" "abl_nat_online_per" "abl_nat_per" "abl_impelu_noper"
  #"r2d3_ne5" "online_offline_anneal" "online_bc_anneal" "dqfd_lite" #"online_offline" "online_awbc" "online_dqn" "baseline_awrcql" "baseline_bc" "baseline_rcql" "online_awbc_handsfree" "online_offline_handsfree" "online_offline_awbc"
  do
    python plot_results_aggregate.py --env "$env" --experiment_name "$name"
  done
done