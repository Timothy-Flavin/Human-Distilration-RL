import pickle
import os
import numpy as np
import argparse
from compliance_metrics import get_compliance_score

def analyze_dataset(path, fps=30):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Infer env name from path if possible
    env_name = "LunarLander-v3"
    if "highway" in path: env_name = "highway-v0"

    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    num_episodes = len(dataset)
    if num_episodes == 0:
        print(f"Dataset at {path} is empty.")
        return

    returns = []
    compliance_scores = []
    total_frames = 0
    
    for episode in dataset:
        episode_return = sum(step['reward'] for step in episode)
        returns.append(episode_return)
        total_frames += len(episode)
        
        # Calculate Compliance Score (Arbitrary Behavior Score)
        obs_seq = [step['obs'] for step in episode]
        compliance_scores.append(get_compliance_score(env_name, obs_seq))

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_compliance = np.mean(compliance_scores)
    std_compliance = np.std(compliance_scores)
    max_return = np.max(returns)
    min_return = np.min(returns)
    
    # Estimate time spent based on frames and FPS
    # Note: This is simulation time, but for real-time recording it approximates wall-clock time
    estimated_time_seconds = total_frames / fps
    minutes = int(estimated_time_seconds // 60)
    seconds = int(estimated_time_seconds % 60)

    print(f"\n=== Analysis for {os.path.basename(path)} ===")
    print(f"Total Episodes:   {num_episodes}")
    print(f"Total Frames:     {total_frames}")
    print(f"Estimated Time:   {minutes}m {seconds}s (at {fps} FPS)")
    print(f"Return Mean:      {mean_return:.2f}")
    print(f"Return Std:       {std_return:.2f}")
    print(f"Return Range:     [{min_return:.2f}, {max_return:.2f}]")
    print(f"Compliance Score: {mean_compliance:.4f} +/- {std_compliance:.4f}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Analyze expert demonstration pickle files.")
    parser.add_argument("--env", type=str, default=None, help="Environment name to find default file (e.g. highway-v0)")
    parser.add_argument("--path", type=str, default=None, help="Direct path to a .pkl file")
    args = parser.parse_args()

    if args.path:
        analyze_dataset(args.path)
    elif args.env:
        path = f"expert_demonstrations_{args.env}.pkl"
        analyze_dataset(path)
    else:
        # Try to find all expert pkl files in current directory
        files = [f for f in os.listdir('.') if f.startswith('expert_demonstrations') and f.endswith('.pkl')]
        if not files:
            print("No expert demonstration files found. Use --path or --env.")
            return
        
        for f in files:
            analyze_dataset(f)

if __name__ == "__main__":
    main()
