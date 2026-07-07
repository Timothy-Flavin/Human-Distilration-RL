import json
import os
import glob
import numpy as np

experiment_dir = os.path.join("results", "crafter", "baseline_bc")
metrics_files = glob.glob(os.path.join(experiment_dir, "**", "metrics_latest.json"), recursive=True)

all_data = []
for f_path in metrics_files:
    with open(f_path, "r") as f:
        all_data.append(json.load(f))

num_seeds = len(all_data)
iters = set()
for d in all_data:
    iters.update([e["iteration"] for e in d.get("evaluations", [])])
iters = sorted(list(iters))

seed_dirs = [os.path.dirname(f) for f in metrics_files]
print("iters:", iters)

for s_dir in seed_dirs:
    print(f"Seed dir: {s_dir}")
    seed_wallclock = []
    for i in iters:
        m_path = os.path.join(s_dir, f"metrics_{i}.json")
        if not os.path.exists(m_path):
            seed_wallclock.append(np.nan)
            continue
        with open(m_path, "r") as f:
            d = json.load(f)
            timers = d["timers"]
            total_t = sum(timers.values())
            seed_wallclock.append(total_t)
    print("Wallclock:", seed_wallclock)
    
