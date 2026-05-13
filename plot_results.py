import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_metrics(metrics_path, output_dir):
    with open(metrics_path, "r") as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    iterations = [e["iteration"] for e in data["evaluations"]]
    returns = [e["return_mean"] for e in data["evaluations"]]
    returns_std = [e["return_std"] for e in data["evaluations"]]
    
    # 1. Plot Returns
    plt.figure(figsize=(10, 5))
    plt.errorbar(iterations, returns, yerr=returns_std, fmt='-o', capsize=5)
    plt.title("Episodic Return Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "returns.png"))
    plt.close()
    
    # 2. Plot Behavior Similarity (Inverse Loss)
    plt.figure(figsize=(10, 5))
    bc_losses = [e["bc_loss"] for e in data["evaluations"] if e["bc_loss"] is not None]
    anti_bc_losses = [e["anti_bc_loss"] for e in data["evaluations"] if e["anti_bc_loss"] is not None]
    
    if bc_losses:
        plt.plot(range(len(bc_losses)), [-l for l in bc_losses], label="BC Similarity (-Loss)")
    if anti_bc_losses:
        plt.plot(range(len(anti_bc_losses)), [-l for l in anti_bc_losses], label="Anti-BC Similarity (-Loss)")
    
    if bc_losses or anti_bc_losses:
        plt.title("Behavior Similarity Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Negative Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "behavior_similarity.png"))
    plt.close()
    
    # 3. Bar chart: Time Spent (Detailed Breakdown)
    plt.figure(figsize=(12, 7))
    timers = data["timers"]
    
    # Sort timers for better visualization
    # Separate Human vs Compute for different colors
    human_keys = ["human_overriding", "human_annotating"]
    compute_keys = [k for k in timers.keys() if k not in human_keys]
    
    keys = human_keys + compute_keys
    values = [timers[k] for k in keys]
    colors = ['orange'] * len(human_keys) + ['blue'] * len(compute_keys)
    
    # Clean up labels for display
    display_labels = [k.replace('_', ' ').title() for k in keys]
    
    plt.barh(display_labels, values, color=colors)
    plt.title("Detailed Time Allocation (Seconds)")
    plt.xlabel("Time (s)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_spent_detailed.png"))
    plt.close()

    # Also keep the summary bar chart but make it more robust
    plt.figure(figsize=(8, 6))
    human_total = sum(timers[k] for k in human_keys)
    compute_total = sum(timers[k] for k in compute_keys)
    
    plt.bar(["Human", "Compute"], [human_total, compute_total], color=['orange', 'blue'])
    plt.title("Human vs. Compute Time")
    plt.ylabel("Total Time (s)")
    plt.savefig(os.path.join(output_dir, "time_spent_summary.png"))
    plt.close()
    
    # 4. Bar chart: Env Samples
    plt.figure(figsize=(10, 6))
    frames = data["frames"]
    plt.bar(frames.keys(), frames.values(), color='green')
    plt.title("Environment Samples")
    plt.ylabel("Frames")
    plt.savefig(os.path.join(output_dir, "env_samples.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, required=True, help="Path to metrics_latest.json")
    parser.add_argument("--output_dir", type=str, default="./plots")
    args = parser.parse_args()
    
    plot_metrics(args.metrics_file, args.output_dir)
