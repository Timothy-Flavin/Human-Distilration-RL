import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import glob


def aggregate_and_plot(experiment_dirs, output_dirs):
    """
    Produces 6 paper-ready graphs by aggregating across multiple seeds.
    experiment_dirs: one experiment's directory in each results root
    (seeds of the same experiment split across machines are pooled).
    The same plots are written to every output_dir, so each root's copy
    of the experiment carries current plots.
    """
    if isinstance(experiment_dirs, str):
        experiment_dirs = [experiment_dirs]
    if isinstance(output_dirs, str):
        output_dirs = [output_dirs]
    for od in output_dirs:
        os.makedirs(od, exist_ok=True)

    def save(fname):
        for od in output_dirs:
            plt.savefig(os.path.join(od, fname))
    metrics_files = []
    for d in experiment_dirs:
        metrics_files.extend(
            glob.glob(os.path.join(d, "**", "metrics_latest.json"), recursive=True)
        )

    # Same seed present in more than one root (copied results) would be
    # double-counted in the mean: keep the first root's copy.
    seen_seeds = set()
    unique_files = []
    for f_path in metrics_files:
        seed_key = os.path.basename(os.path.dirname(f_path))
        if seed_key in seen_seeds:
            print(f"Note: skipping duplicate seed dir {f_path}")
            continue
        seen_seeds.add(seed_key)
        unique_files.append(f_path)
    metrics_files = unique_files

    if not metrics_files:
        print(f"No metrics files found in {experiment_dirs}")
        return

    all_data = []
    for f_path in metrics_files:
        try:
            with open(f_path, "r") as f:
                all_data.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: skipping invalid metrics file {f_path}: {e}")

    if not all_data:
        print(f"No valid metrics files found in {experiment_dirs}")
        return

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
    all_seeds_likeness = []  # BC Loss (Cross Entropy)

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
            try:
                with open(m_path, "r") as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: skipping invalid metrics file {m_path}: {e}")
                seed_interactions.append(np.nan)
                seed_wallclock.append(np.nan)
                seed_human_time.append(np.nan)
                seed_scores.append(np.nan)
                seed_likeness.append(np.nan)
                continue

            # Total Frames
            f_counts = d["frames"]
            total_f = sum(f_counts.values())
            seed_interactions.append(total_f)

            # Total Wallclock
            timers = d["timers"]
            total_t = sum(
                v for k, v in timers.items() if k != "training_throughput_fps"
            )
            seed_wallclock.append(total_t)

            # Human Effort
            h_t = (
                timers.get("human_overriding", 0)
                + timers.get("human_annotating", 0)
                + timers.get("human_reviewing", 0)
                + timers.get("expert_preload_effort", 0)
            )
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
        all_x = np.concatenate(
            [x[~np.isnan(x)] for x in x_arrays if len(x[~np.isnan(x)]) > 0]
        )
        if len(all_x) == 0:
            return
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
                    y_interp = np.interp(
                        common_x, x_val, y_val, left=np.nan, right=np.nan
                    )
                    interp_y.append(y_interp)

        if interp_y:
            mean_y = np.nanmean(interp_y, axis=0)
            smoothed = np.convolve(mean_y, np.ones(10) / 10, mode="same")
            ax.plot(common_x, mean_y, color=color, linewidth=3, label=label)
            ax.plot(
                common_x,
                smoothed,
                color=(0.2, 0.6, 0.4),
                linewidth=3,
                label=label + " (smoothed)",
            )

    # --- Plot 1: Interactions vs Eval Score ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(
            all_seeds_interactions[s], all_seeds_scores[s], alpha=0.3, color="blue"
        )
    plot_interpolated_mean(
        plt.gca(), all_seeds_interactions, all_seeds_scores, color="blue"
    )
    plt.title("Sample Efficiency")
    plt.xlabel("Total Environment Interactions")
    plt.ylabel("Eval Return")
    plt.grid(True)
    plt.legend()
    save("1_sample_efficiency.png")

    # --- Plot 2: Wall-clock Time vs Eval Score ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(all_seeds_wallclock[s], all_seeds_scores[s], alpha=0.3, color="red")
    plot_interpolated_mean(
        plt.gca(), all_seeds_wallclock, all_seeds_scores, color="red"
    )
    plt.title("Real-Time Performance")
    plt.xlabel("Total Wall-clock Time (s)")
    # plt.ylim(-400,400)
    plt.ylabel("Eval Return")
    plt.grid(True)
    plt.legend()
    save("2_realtime_performance.png")

    # --- Plot 3: Human Likeness (Cross Entropy) ---
    plt.figure(figsize=(10, 6))
    for s in range(num_seeds):
        plt.plot(iters, all_seeds_likeness[s], alpha=0.3, color="green")
    # For iters which are perfectly aligned across seeds, nanmean is perfectly fine
    mean_likeness = np.nanmean(all_seeds_likeness, axis=0)
    plt.plot(iters, mean_likeness, color="green", linewidth=3, label="Mean")
    plt.title("Human Likeness (Policy Divergence)")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss (Expert Data)")
    plt.grid(True)
    plt.legend()
    save("3_human_likeness.png")

    # --- Plot 4: Active Human Time vs Eval Score ---
    plt.figure(figsize=(10, 6))
    active_mask = np.nanmean(all_seeds_human_time, axis=0) > 0
    if active_mask.any():
        for s in range(num_seeds):
            plt.plot(
                all_seeds_human_time[s], all_seeds_scores[s], alpha=0.3, color="orange"
            )
        plot_interpolated_mean(
            plt.gca(), all_seeds_human_time, all_seeds_scores, color="orange"
        )
    plt.title("Human Effort Efficiency")
    plt.xlabel("Total Human Effort (Seconds)")
    plt.ylabel("Eval Return")
    plt.grid(True)
    plt.legend()
    save("4_human_effort_efficiency.png")

    # --- Plot 5: Bar-graph of Frames by Category ---
    plt.figure(figsize=(12, 7))
    latest = all_data[0]  # Use first seed as template for labels
    f_labels = list(latest["frames"].keys())
    f_means = [np.mean([d["frames"].get(l, 0) for d in all_data]) for l in f_labels]
    f_stds = [np.std([d["frames"].get(l, 0) for d in all_data]) for l in f_labels]
    plt.bar(f_labels, f_means, yerr=f_stds, color="green", alpha=0.7, capsize=10)
    plt.title("Distribution of Environment Samples")
    plt.ylabel("Total Frames")
    save("5_frame_distribution.png")

    # --- Plot 6: Bar-graph of Time by Category ---
    plt.figure(figsize=(12, 7))
    t_labels = [k for k in latest["timers"].keys() if k != "training_throughput_fps"]
    t_means = [np.mean([d["timers"].get(l, 0) for d in all_data]) for l in t_labels]
    t_stds = [np.std([d["timers"].get(l, 0) for d in all_data]) for l in t_labels]
    plt.barh(t_labels, t_means, xerr=t_stds, color="blue", alpha=0.7, capsize=10)
    plt.title("Time Allocation Breakdown")
    plt.xlabel("Seconds")
    plt.tight_layout()
    save("6_time_breakdown.png")

    print(f"Paper-ready graphs saved to {output_dirs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--roots", type=str, nargs="+", default=["results"],
                        help="Results roots to search (e.g. results lab-impala); "
                             "seeds found in any root are pooled into one plot")
    args = parser.parse_args()

    experiment_dirs = [
        d for d in (os.path.join(r, args.env, args.experiment_name) for r in args.roots)
        if os.path.exists(d)
    ]
    if not experiment_dirs:
        print(f"Error: Experiment '{args.experiment_name}' not found under any of "
              f"{[os.path.join(r, args.env) for r in args.roots]}")
    else:
        # Plots land next to the first root that has the experiment
        output_dirs = [os.path.join(d, "plots") for d in experiment_dirs]
        aggregate_and_plot(experiment_dirs, output_dirs)
