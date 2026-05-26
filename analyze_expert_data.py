import pickle
import os
import numpy as np
import argparse
from compliance_metrics import get_compliance_score
import collections 

def analyze_dataset(path, fps=30):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Infer env name from path if possible
    env_name = "LunarLander-v3"
    if "highway" in path.lower(): env_name = "highway-v0"
    elif "crafter" in path.lower(): env_name = "crafter"

    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    num_episodes = len(dataset)
    if num_episodes == 0:
        print(f"Dataset at {path} is empty.")
        return

    returns = []
    compliance_scores = []
    total_frames = 0
    total_duration_seconds = 0.0
    
    # Crafter specific tracking
    is_crafter = "crafter" in path or "crafter" in env_name.lower()
    crafter_achievements = collections.defaultdict(int)
    num_episodes_with_info = 0

    for item in dataset:
        # Handle both [transitions, ...] and [{'transitions': transitions}, ...]
        if isinstance(item, dict) and 'transitions' in item:
            episode = item['transitions']
            total_duration_seconds += item.get('duration', len(episode) / fps)
        else:
            episode = item
            total_duration_seconds += len(episode) / fps
            
        episode_return = sum(step['reward'] for step in episode)
        returns.append(episode_return)
        total_frames += len(episode)
        
        # Calculate Compliance Score (Arbitrary Behavior Score)
        obs_seq = [step['obs'] for step in episode]
        compliance_scores.append(get_compliance_score(env_name, obs_seq))

        # Crafter Achievement Tracking
        if is_crafter:
            # Check the last step of the episode for achievements
            # Crafter info['achievements'] contains cumulative counts/bools
            last_step = episode[-1]
            if 'info' in last_step and isinstance(last_step['info'], dict) and 'achievements' in last_step['info']:
                num_episodes_with_info += 1
                for ach, val in last_step['info']['achievements'].items():
                    if val > 0:
                        crafter_achievements[ach] += 1

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_compliance = np.mean(compliance_scores)
    std_compliance = np.std(compliance_scores)
    max_return = np.max(returns)
    min_return = np.min(returns)
    
    # Estimate time spent
    minutes = int(total_duration_seconds // 60)
    seconds = int(total_duration_seconds % 60)

    print(f"\n=== Analysis for {os.path.basename(path)} ===")
    print(f"Total Episodes:   {num_episodes}")
    print(f"Total Frames:     {total_frames}")
    print(f"Recorded Time:    {minutes}m {seconds}s")
    print(f"Return Mean:      {mean_return:.2f}")
    print(f"Return Std:       {std_return:.2f}")
    print(f"Return Range:     [{min_return:.2f}, {max_return:.2f}]")
    print(f"Compliance Score: {mean_compliance:.4f} +/- {std_compliance:.4f}")

    if is_crafter and num_episodes_with_info > 0:
        print("\n--- Crafter Achievements (Completion Rate) ---")
        rates = []
        # Sorted for consistent output
        for ach in sorted(crafter_achievements.keys()):
            rate = (crafter_achievements[ach] / num_episodes_with_info) * 100
            rates.append(rate / 100.0) # For geometric mean
            print(f"{ach:20}: {rate:6.1f}%")
        
        # Geometric Mean (Standard Crafter Metric)
        if rates:
            geo_mean = np.exp(np.mean(np.log([max(r, 1e-4) for r in rates]))) * 100
            print(f"\nGeometric Mean Score: {geo_mean:.2f}%")
            
    print("-" * 40)

def drop_bottom_episodes(path, ratio=0.1):
    """Sorts episodes by return and drops the lowest X%."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    if not dataset:
        print(f"Dataset at {path} is empty.")
        return
    
    # Calculate returns for sorting
    scored_episodes = []
    for item in dataset:
        if isinstance(item, dict) and 'transitions' in item:
            ret = sum(t['reward'] for t in item['transitions'])
        else:
            ret = sum(t['reward'] for t in item)
        scored_episodes.append((item, ret))
    
    # Sort by return ascending
    scored_episodes.sort(key=lambda x: x[1])
    
    num_to_drop = int(len(dataset) * ratio)
    # Ensure we drop at least one if ratio > 0 and we have more than one episode
    if num_to_drop == 0 and ratio > 0 and len(dataset) > 1:
        num_to_drop = 1
        
    cleaned_dataset = [x[0] for x in scored_episodes[num_to_drop:]]
    
    new_path = path.replace(".pkl", "_cleaned.pkl")
    with open(new_path, 'wb') as f:
        pickle.dump(cleaned_dataset, f)
    
    print(f"\n[*] Cleaning: {os.path.basename(path)}")
    print(f"[*] Original Episodes: {len(dataset)}")
    print(f"[*] Dropped bottom {num_to_drop} episodes ({ratio*100:.1f}%)")
    print(f"[*] Cleaned dataset saved to: {new_path}")

def main():
    import collections # Ensure collections is available for defaultdict
    parser = argparse.ArgumentParser(description="Analyze and clean expert demonstration pickle files.")
    parser.add_argument("--env", type=str, default=None, help="Environment name (e.g. crafter)")
    parser.add_argument("--path", type=str, default=None, help="Direct path to a .pkl file")
    parser.add_argument("--drop_bottom", type=float, default=0.0, help="Ratio of lowest-return episodes to drop (0.0 to 1.0)")
    args = parser.parse_args()

    target_files = []
    if args.path:
        target_files.append(args.path)
    elif args.env:
        target_files.append(f"expert_demonstrations_{args.env}.pkl")
    else:
        # Try to find all expert pkl files in current directory
        target_files = [f for f in os.listdir('.') if f.startswith('expert_demonstrations') and f.endswith('.pkl')]
    
    if not target_files:
        print("No expert demonstration files found. Use --path or --env.")
        return

    for f in target_files:
        if args.drop_bottom > 0:
            drop_bottom_episodes(f, args.drop_bottom)
        else:
            analyze_dataset(f)


if __name__ == "__main__":
    main()
