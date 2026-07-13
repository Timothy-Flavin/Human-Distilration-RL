#!/usr/bin/env bash
set -euo pipefail

folder="${1:-crafter}"
experiment_filter="${2:-*}"

source .venv/bin/activate

# Both results roots: this PC's runs and the lab computer's copied runs.
# Seeds of the same experiment split across machines are pooled into one
# aggregate plot (plot_results_aggregate.py --roots).
roots=("results" "lab-impala")

# Union of experiment names across the roots
shopt -s nullglob
declare -A seen
for root in "${roots[@]}"; do
  [[ -d "$root/$folder" ]] || continue
  for experiment_dir in "$root/$folder"/$experiment_filter; do
    [[ -d "$experiment_dir" ]] || continue
    seen["$(basename "$experiment_dir")"]=1
  done
done

if [[ ${#seen[@]} -eq 0 ]]; then
  echo "Error: no experiments matching '$experiment_filter' under ${roots[*]/%//$folder}"
  exit 1
fi

for experiment_name in "${!seen[@]}"; do
  python plot_results_aggregate.py --env "$folder" \
    --experiment_name "$experiment_name" --roots "${roots[@]}"
done
