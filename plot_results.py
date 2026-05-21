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
    plt.figure(figsize=(14, 8))
    timers = data.get("timers", {})
    f_counts = data.get("frames", {})

    # 1. Expert Pre-recording (Backwards compatible)
    preload_time = timers.get("expert_preload_effort", 0)
    if preload_time == 0:
        preload_frames = f_counts.get("expert_preload", 0)
        if preload_frames == 0 and f_counts.get("human", 0) > 0 and timers.get("human_overriding", 0) == 0:
            preload_frames = f_counts.get("human", 0)
        preload_time = preload_frames / 30.0

    # 2. Live Intervention
    human_live = timers.get("human_overriding", 0) + timers.get("human_annotating", 0) + timers.get("human_reviewing", 0)

    # 3. Agent Training (Actual Compute)
    compute = sum(v for k, v in timers.items() if k.startswith("agent_updating") or k == "llm_processing")

    # 4. Environment Time
    env_time = timers.get("rl_experience", 0)

    labels = ["Expert Pre-recording", "Live Intervention", "Agent Training (Compute)", "RL Experience (Env)"]
    values = [preload_time, human_live, compute, env_time]
    colors = ['darkorange', 'orange', 'blue', 'lightblue']

    plt.bar(labels, values, color=colors)
    plt.title("Time Allocation Summary")
    plt.ylabel("Seconds")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_spent_summary.png"))
    plt.close()

    # Detailed Breakdown (Horizontal)
    plt.figure(figsize=(12, 10))
    # Filter out keys with 0 to keep it clean
    display_timers = {k: v for k, v in timers.items() if v > 0}
    # Add the estimated preload if it was calculated manually
    if "expert_preload_effort" not in display_timers and preload_time > 0:
        display_timers["expert_preload_effort"] = preload_time

    keys = sorted(display_timers.keys())
    values = [display_timers[k] for k in keys]
    labels = [k.replace('_', ' ').title() for k in keys]
    
    plt.barh(labels, values, color='gray')
    plt.title("All Timers (Raw Allocation)")
    plt.xlabel("Seconds")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_spent_detailed.png"))
    plt.close()
    
    # 4. Bar chart: Env Samples
    plt.figure(figsize=(12, 7))
    # Map sources to more readable names
    source_map = {
        "rl": "Online RL",
        "human": "Expert Intervention",
        "curriculum": "Curriculum/SSL",
        "ssl": "SSL Mining",
        "expert_preload": "Expert Pre-recording"
    }
    
    labels = []
    values = []
    for k, v in f_counts.items():
        # Backwards compatible check: if human frames exist but it was a preload run
        if k == "human" and timers.get("human_overriding", 0) == 0:
            labels.append("Expert Pre-recording (Legacy)")
        else:
            labels.append(source_map.get(k, k.upper()))
        values.append(v)
        
    plt.bar(labels, values, color='green')
    plt.title("Total Environment Samples by Mode")
    plt.ylabel("Frames")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "env_samples.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "ppo"])
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    metrics_path = os.path.join("results", args.algo, args.env, args.experiment_name, "metrics_latest.json")
    output_dir = args.output_dir if args.output_dir else os.path.join("results", args.algo, args.env, args.experiment_name, "plots")
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
    else:
        plot_metrics(metrics_path, output_dir)
