import pickle
import os
import argparse
import numpy as np
from compliance_metrics import get_compliance_score

def get_episode_stats(ep, env_name):
    transitions = ep['transitions'] if isinstance(ep, dict) else ep
    frames = len(transitions)
    total_return = sum(t['reward'] for t in transitions)
    obs_seq = [t['obs'] for t in transitions]
    compliance = get_compliance_score(env_name, obs_seq)
    return {
        "frames": frames,
        "return": total_return,
        "compliance": compliance
    }

def interactive_editor(path, env_name):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return
    
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    if not isinstance(dataset, list):
        print("Error: Dataset is not a list of episodes.")
        return

    while True:
        print(f"\n=== Dataset Editor: {path} ===")
        print(f"{'ID':<4} | {'Frames':<6} | {'Return':<10} | {'Compliance':<10}")
        print("-" * 45)
        
        all_stats = []
        for i, ep in enumerate(dataset):
            stats = get_episode_stats(ep, env_name)
            all_stats.append(stats)
            print(f"{i:<4} | {stats['frames']:<6} | {stats['return']:<10.2f} | {stats['compliance']:<10.4f}")
        
        print("-" * 45)
        print(f"Total Episodes: {len(dataset)}")
        
        cmd = input("\nEnter IDs to remove (comma-separated), 's' to save and exit, or 'q' to quit without saving: ").strip().lower()
        
        if cmd == 'q':
            print("Exiting without saving.")
            break
        elif cmd == 's':
            save_path = input(f"Enter save path [default: {path}]: ").strip()
            if not save_path:
                save_path = path
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Dataset saved to {save_path}. Exiting.")
            break
        else:
            try:
                ids_to_remove = sorted([int(x.strip()) for x in cmd.split(',')], reverse=True)
                for idx in ids_to_remove:
                    if 0 <= idx < len(dataset):
                        removed = dataset.pop(idx)
                        print(f"Removed Episode {idx}")
                    else:
                        print(f"Warning: Invalid ID {idx}")
            except ValueError:
                print("Error: Please enter valid numeric IDs or 's'/'q'.")

def main():
    parser = argparse.ArgumentParser(description="Interactive Expert Dataset Editor")
    parser.add_argument("--path", type=str, required=True, help="Path to the .pkl dataset")
    parser.add_argument("--env", type=str, default="highway-v0", help="Environment name for compliance metrics")
    args = parser.parse_args()
    
    interactive_editor(args.path, args.env)

if __name__ == "__main__":
    main()
