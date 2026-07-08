import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob

def aggregate_and_plot(experiment_dir, output_dir):
    """
    Produces 6 paper-ready graphs by aggregating across multiple seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_files = glob.glob(os.path.join(experiment_dir, "**", "metrics_latest.json"), recursive=True)
    
    if not metrics_files:
        print(f"No metrics files found in {experiment_dir}")
        return

    all_data = []
    for f_path in metrics_files:
        with open(f_path, "r") as f:
            all_data.append(json.load(f))

    num_seeds = len(all_data)
    iters = set()
    for d in all_data:
        iters.update([e["iteration"] for e in d.get("evaluations", [])])
    iters = sorted(list(iters))
    num_iters = len(iters)

    # Helper to extract and aggregate a metric over iterations
    def get_iter_metric(data_list, key_path):
        # key_path: list of keys to traverse, or "evaluations:key"
        results = []
        for d in data_list:
            if ":" in key_path:
                root, key = key_path.split(":")
                vals = [e.get(key, 0) for e in d[root]]
                results.append(vals[:num_iters])
            else:
                # Timers/Frames are cumulative? No, they are usually snapshot at end of iter
                # Actually, our metrics_latest.json has the totals.
                # We need iteration-by-iteration totals if we want X-axis to be cumulative.
                # Since we only have 'latest', we assume uniform distribution or just final result.
                # TO BE RIGOROUS: We should use iteration-specific metrics files.
                pass
        return np.array(results)

    # For X-axes (Interactions, Time, Human Effort), we need iteration-by-iteration cumulative counts.
    # We find all metrics_{i}.json files for each seed.
    seed_dirs = [os.path.dirname(f) for f in metrics_files]
    
    all_seeds_interactions = []
    all_seeds_wallclock = []
    all_seeds_human_time = []
    all_seeds_scores = []
    all_seeds_likeness = [] # BC Loss (Cross Entropy)

    for s_dir in seed_dirs:
        seed_interactions = []
        seed_wallclock = []
        seed_human_time = []
        seed_scores = []
        seed_likeness = []
        
        for i in iters:
            m_path = os.path.join(s_dir, f"metrics_{i}.json")
            if not os.path.exists(m_path):
                seed_interactions.append(np.nan)
                seed_wallclock.append(np.nan)
                seed_human_time.append(np.nan)
                seed_scores.append(np.nan)
                seed_likeness.append(np.nan)
                continue
            with open(m_path, "r") as f:
                d = json.load(f)
                # Total Frames
                f_counts = d["frames"]
                total_f = sum(f_counts.values())
                seed_interactions.append(total_f)
                
                # Total Wallclock
                timers = d["timers"]
                total_t = sum(v for k, v in timers.items() if k != "training_throughput_fps")
                seed_wallclock.append(total_t)
                
                # Human Effort
                h_t = timers.get("human_overriding", 0) + timers.get("human_annotating", 0) + timers.get("human_reviewing", 0) + timers.get("expert_preload_effort", 0)
                seed_human_time.append(h_t)
                
                # Score
                eval_last = d["evaluations"][-1]
                seed_scores.append(eval_last["return_mean"])
                # Handle None in bc_loss
                bcl = eval_last.get("bc_loss", 0.0)
                seed_likeness.append(bcl if bcl is not None else 0.0)
        
        all_seeds_interactions.append(seed_interactions)
        all_seeds_wallclock.append(seed_wallclock)
        all_seeds_human_time.append(seed_human_time)
        all_seeds_scores.append(seed_scores)
        all_seeds_likeness.append(seed_likeness)

    # Convert to arrays [Seed, Iteration]
    all_seeds_interactions = np.array(all_seeds_interactions, dtype=float)
    all_seeds_wallclock = np.array(all_seeds_wallclock, dtype=float)
    all_seeds_human_time = np.array(all_seeds_human_time, dtype=float)
    all_seeds_scores = np.array(all_seeds_scores, dtype=float)
    all_seeds_likeness = np.array(all_seeds_likeness, dtype=float)

    def plot_interpolated_mean(ax, x_arrays, y_arrays, color, label="Mean"):
        # Gather all valid X data to find global min and max
        all_x = np.concatenate([x[~np.isnan(x)] for x in x_arrays if len(x[~np.isnan(x)]) > 0])
        if len(all_x) == 0: return
        min_x, max_x = np.min(all_x), np.max(all_x)
        
        # We need a monotonically increasing grid
        common_x = np.linspace(min_x, max_x, 300)
        interp_y = []
        
        for x, y in zip(x_arrays, y_arrays):
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() > 1:
                # Sort X to ensure it is strictly increasing for interpolation
                sort_idx = np.argsort(x[valid])
                x_val = x[valid][sort_idx]
                y_val = y[valid][sort_idx]
                
                # Remove duplicates in X
                x_val, unique_idx = np.unique(x_val, return_index=True)
                y_val = y_val[unique_idx]
                
                if len(x_val) > 1:
                    y_interp = np.interp(common_x, x_val, y_val, left=np.nan, right=np.nan)
                    interp_y.append(y_interp)
                    
        if interp_y:
            mean_y = np.nanmean(interp_y, axis=0)
            ax.plot(common_x, mean_y, color=color, linewidth=3, label=label)

    # --- Plot 1: Interactions vs Eval Score ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(all_seeds_interactions[s], all_seeds_scores[s], alpha=0.3, color='blue')
    plot_interpolated_mean(plt.gca(), all_seeds_interactions, all_seeds_scores, color='blue')
    plt.title("Sample Efficiency")
    plt.xlabel("Total Environment Interactions")
    plt.ylabel("Eval Return")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "1_sample_efficiency.png"))

    # --- Plot 2: Wall-clock Time vs Eval Score ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(all_seeds_wallclock[s], all_seeds_scores[s], alpha=0.3, color='red')
    plot_interpolated_mean(plt.gca(), all_seeds_wallclock, all_seeds_scores, color='red')
    plt.title("Real-Time Performance")
    plt.xlabel("Total Wall-clock Time (s)")
    plt.ylim(-400,400)
    plt.ylabel("Eval Return")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "2_realtime_performance.png"))

    # --- Plot 3: Human Likeness (Cross Entropy) ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(iters, all_seeds_likeness[s], alpha=0.3, color='green')
    # For iters which are perfectly aligned across seeds, nanmean is perfectly fine
    mean_likeness = np.nanmean(all_seeds_likeness, axis=0)
    plt.plot(iters, mean_likeness, color='green', linewidth=3, label="Mean")
    plt.title("Human Likeness (Policy Divergence)")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss (Expert Data)")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "3_human_likeness.png"))

    # --- Plot 4: Active Human Time vs Eval Score ---
    plt.figure(figsize=(10, 6))
    active_mask = np.nanmean(all_seeds_human_time, axis=0) > 0
    if active_mask.any():
        for s in range(num_seeds):
            plt.plot(all_seeds_human_time[s], all_seeds_scores[s], alpha=0.3, color='orange')
        plot_interpolated_mean(plt.gca(), all_seeds_human_time, all_seeds_scores, color='orange')
    plt.title("Human Effort Efficiency")
    plt.xlabel("Total Human Effort (Seconds)")
    plt.ylabel("Eval Return")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "4_human_effort_efficiency.png"))

    # --- Plot 5: Bar-graph of Frames by Category ---
    plt.figure(figsize=(12, 7))
    latest = all_data[0] # Use first seed as template for labels
    f_labels = list(latest["frames"].keys())
    f_means = [np.mean([d["frames"].get(l, 0) for d in all_data]) for l in f_labels]
    f_stds = [np.std([d["frames"].get(l, 0) for d in all_data]) for l in f_labels]
    plt.bar(f_labels, f_means, yerr=f_stds, color='green', alpha=0.7, capsize=10)
    plt.title("Distribution of Environment Samples")
    plt.ylabel("Total Frames")
    plt.savefig(os.path.join(output_dir, "5_frame_distribution.png"))

    # --- Plot 6: Bar-graph of Time by Category ---
    plt.figure(figsize=(12, 7))
    t_labels = [k for k in latest["timers"].keys() if k != "training_throughput_fps"]
    t_means = [np.mean([d["timers"].get(l, 0) for d in all_data]) for l in t_labels]
    t_stds = [np.std([d["timers"].get(l, 0) for d in all_data]) for l in t_labels]
    plt.barh(t_labels, t_means, xerr=t_stds, color='blue', alpha=0.7, capsize=10)
    plt.title("Time Allocation Breakdown")
    plt.xlabel("Seconds")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_time_breakdown.png"))

    print(f"Paper-ready graphs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()
    
    experiment_dir = os.path.join("results", args.env, args.experiment_name)
    output_dir = os.path.join(experiment_dir, "plots")
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found at {experiment_dir}")
    else:
        aggregate_and_plot(experiment_dir, output_dir)
