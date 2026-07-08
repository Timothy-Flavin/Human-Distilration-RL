import json
import glob
import os

files = glob.glob("results/crafter/**/metrics_latest.json", recursive=True)
if not files:
    print("No metrics files found.")
else:
    # Get the most recently modified one
    latest_file = max(files, key=os.path.getmtime)
    print("Reading:", latest_file)
    with open(latest_file, "r") as f:
        data = json.load(f)
    print("Timers:")
    for k, v in data["timers"].items():
        print(f"  {k}: {v}")
