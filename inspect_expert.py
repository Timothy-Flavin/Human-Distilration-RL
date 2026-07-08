import pickle
import collections

with open("expert_demonstrations_crafter.pkl", "rb") as f:
    data = pickle.load(f)

action_counts = collections.Counter()
total_rewards = 0
for item in data:
    if isinstance(item, dict) and 'transitions' in item:
        transitions = item['transitions']
    else:
        transitions = item
    
    for t in transitions:
        action_counts[t['action']] += 1
        total_rewards += t.get('reward', 0)

print("Total Transitions:", sum(action_counts.values()))
print("Total Reward:", total_rewards)
print("Action Distribution:", dict(action_counts.most_common()))
