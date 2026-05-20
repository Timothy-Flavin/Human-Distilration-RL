import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob

def aggregate_and_plot(experiment_dir, output_dir):
    """
    Finds all metrics_latest.json files in subdirectories of experiment_dir,
    aggregates them, and plots mean + individual runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all metrics_latest.json files recursively
    metrics_files = glob.glob(os.path.join(experiment_dir, "**", "metrics_latest.json"), recursive=True)
    
    if not metrics_files:
        print(f"No metrics files found in {experiment_dir}")
        return

    all_returns = []
    all_iterations = []
    
    for f_path in metrics_files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
                iters = [e["iteration"] for e in data["evaluations"]]
                rets = [e["return_mean"] for e in data["evaluations"]]
                all_returns.append(rets)
                all_iterations = iters # Assume same iteration count for all seeds
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    if not all_returns:
        return

    # Convert to numpy for easier aggregation
    # Note: May need to truncate if iterations differ
    min_len = min(len(r) for r in all_returns)
    all_returns = [r[:min_len] for r in all_returns]
    all_returns = np.array(all_returns)
    all_iterations = all_iterations[:min_len]

    # 1. Plot Returns with Transparency
    plt.figure(figsize=(12, 6))
    
    # Plot individual seeds
    for i in range(len(all_returns)):
        plt.plot(all_iterations, all_returns[i], color='blue', alpha=0.3, label=f"Seed {i+1}" if i==0 else "")
    
    # Plot Mean
    mean_return = np.mean(all_returns, axis=0)
    plt.plot(all_iterations, mean_return, color='blue', linewidth=3, label="Mean")
    
    # Add Shaded Area (Std Dev)
    std_return = np.std(all_returns, axis=0)
    plt.fill_between(all_iterations, mean_return - std_return, mean_return + std_return, color='blue', alpha=0.1)

    plt.title(f"Aggregated Performance: {os.path.basename(experiment_dir)}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "aggregated_returns.png"))
    plt.close()

    # 2. Plot Human Effort (Aggregate across seeds)
    plt.figure(figsize=(10, 6))
    all_human_times = []
    all_compute_times = []
    
    for f_path in metrics_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            timers = data["timers"]
            human = timers.get("human_overriding", 0) + timers.get("human_annotating", 0) + timers.get("human_reviewing", 0)
            compute = sum(v for k, v in timers.items() if "human" not in k)
            all_human_times.append(human)
            all_compute_times.append(compute)
            
    plt.bar(["Human Effort (Mean)", "Compute Time (Mean)"], 
            [np.mean(all_human_times), np.mean(all_compute_times)], 
            yerr=[np.std(all_human_times), np.std(all_compute_times)],
            color=['orange', 'blue'], capsize=10)
    
    plt.title("Mean Effort Across Seeds")
    plt.ylabel("Seconds")
    plt.savefig(os.path.join(output_dir, "aggregated_time_spent.png"))
    plt.close()

    print(f"Aggregated plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    args = parser.parse_args()
    
    # Correct path mapping: ./results/{env}/{experiment_name}/
    experiment_dir = os.path.join("results", args.env, args.experiment_name)
    output_dir = os.path.join(experiment_dir, "plots")
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found at {experiment_dir}")
    else:
        aggregate_and_plot(experiment_dir, output_dir)
