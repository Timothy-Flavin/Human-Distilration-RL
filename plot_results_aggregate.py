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
    plt.figure(figsize=(12, 7))
    all_human_times = []
    all_compute_times = []
    all_preload_times = []
    all_env_times = []
    
    # Track frames for the new chart
    mode_frames = {
        "Online RL": [],
        "Expert Intervention": [],
        "SSL Mining": [],
        "Expert Pre-recording": []
    }

    for f_path in metrics_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            timers = data.get("timers", {})
            f_counts = data.get("frames", {})
            
            # 1. Expert Pre-recording (Backwards compatible: use timer or estimate from frames at 30 FPS)
            preload_time = timers.get("expert_preload_effort", 0)
            if preload_time == 0:
                # If timer is missing, check if frames were logged to 'human' (legacy) or 'expert_preload'
                preload_frames = f_counts.get("expert_preload", 0)
                if preload_frames == 0 and f_counts.get("human", 0) > 0 and timers.get("human_overriding", 0) == 0:
                    # Legacy fallback: if HUMAN frames exist but no override time, they are preloaded
                    preload_frames = f_counts.get("human", 0)
                preload_time = preload_frames / 30.0
            
            # 2. Live Intervention Time
            human_live = timers.get("human_overriding", 0) + timers.get("human_annotating", 0) + timers.get("human_reviewing", 0)
            
            # 3. Agent Training Time (The actual compute cost)
            compute = sum(v for k, v in timers.items() if k.startswith("agent_updating") or k == "llm_processing")
            
            # 4. Environment Time (Experience gathering)
            env_time = timers.get("rl_experience", 0)
            
            all_preload_times.append(preload_time)
            all_human_times.append(human_live)
            all_compute_times.append(compute)
            all_env_times.append(env_time)

            # Collect frames for the mode chart
            mode_frames["Online RL"].append(f_counts.get("rl", 0))
            mode_frames["Expert Intervention"].append(f_counts.get("human", 0) if timers.get("human_overriding", 0) > 0 else 0)
            mode_frames["SSL Mining"].append(f_counts.get("ssl", 0) + f_counts.get("curriculum", 0))
            # Expert Pre-recording frames (Backwards compatible logic)
            ep_frames = f_counts.get("expert_preload", 0)
            if ep_frames == 0 and f_counts.get("human", 0) > 0 and timers.get("human_overriding", 0) == 0:
                ep_frames = f_counts.get("human", 0)
            mode_frames["Expert Pre-recording"].append(ep_frames)
            
    # Plot Time Summary
    plt.figure(figsize=(14, 8))
    labels = ["Expert Pre-recording", "Live Intervention", "Agent Training (Compute)", "RL Experience (Env)"]
    means = [np.mean(all_preload_times), np.mean(all_human_times), np.mean(all_compute_times), np.mean(all_env_times)]
    stds = [np.std(all_preload_times), np.std(all_human_times), np.std(all_compute_times), np.std(all_env_times)]
    
    plt.bar(labels, means, yerr=stds, color=['darkorange', 'orange', 'blue', 'lightblue'], capsize=10)
    plt.title("Aggregated Time Effort Across Seeds")
    plt.ylabel("Seconds")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_time_spent.png"))
    plt.close()

    # 3. Plot Total Frames (New Chart)
    plt.figure(figsize=(12, 7))
    f_labels = list(mode_frames.keys())
    f_means = [np.mean(mode_frames[m]) for m in f_labels]
    f_stds = [np.std(mode_frames[m]) for m in f_labels]
    
    plt.bar(f_labels, f_means, yerr=f_stds, color='green', capsize=10)
    plt.title("Total Environment Samples by Mode (Mean across Seeds)")
    plt.ylabel("Frames")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_frames.png"))
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
