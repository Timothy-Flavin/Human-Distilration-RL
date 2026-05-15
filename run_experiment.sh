#!/bin/bash

# Default values
ALGO="cql"
EXP_NAME="experiment_$(date +%Y%m%d_%H%M%S)"
FLAGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo) ALGO="$2"; shift ;;
        --name) EXP_NAME="$2"; shift ;;
        --rl|--bc|--anti_bc|--ssl|--curriculum) FLAGS="$FLAGS $1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting Experiment: $EXP_NAME using $ALGO"
echo "Flags: $FLAGS"

# Run the experiment using the local virtual environment
.venv/bin/python3 main.py --algo $ALGO --experiment_name $EXP_NAME $FLAGS

# Generate plots
echo "Generating plots..."
.venv/bin/python3 plot_results.py --algo $ALGO --experiment_name $EXP_NAME

echo "Experiment $EXP_NAME complete. Results are in results/$ALGO/LunarLander-v3/$EXP_NAME"
