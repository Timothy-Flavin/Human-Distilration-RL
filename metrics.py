import time
import json
import os

class MetricsLogger:
    def __init__(self):
        # Timers
        self.timers = {
            "rl_experience": 0.0,
            "human_overriding": 0.0,
            "human_annotating": 0.0,
            "llm_processing": 0.0,
            "agent_updating_bc": 0.0,
            "agent_updating_anti_bc": 0.0,
            "agent_updating_local_rl": 0.0,
            "agent_updating_ssl": 0.0,
        }
        
        # Frame counters
        self.frames = {
            "rl": 0,
            "human": 0,
            "curriculum": 0
        }
        
        # Evaluation results
        self.evaluations = [] # List of dicts: {"iteration": i, "return_mean": x, "return_std": y, "bc_loss": z, "anti_bc_loss": w}
        
        self._start_times = {}

    def start_timer(self, key):
        if key in self.timers:
            self._start_times[key] = time.time()

    def stop_timer(self, key):
        if key in self._start_times:
            elapsed = time.time() - self._start_times.pop(key)
            self.timers[key] += elapsed

    def log_frames(self, count, source="rl"):
        if source in self.frames:
            self.frames[source] += count

    def log_evaluation(self, iteration, return_mean, return_std, bc_loss=None, anti_bc_loss=None):
        self.evaluations.append({
            "iteration": iteration,
            "return_mean": return_mean,
            "return_std": return_std,
            "bc_loss": bc_loss,
            "anti_bc_loss": anti_bc_loss
        })

    def get_summary(self):
        return {
            "timers": self.timers,
            "frames": self.frames,
            "evaluations": self.evaluations
        }

    def save_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=4)
        print(f"[Metrics] Saved to {path}")

    def log_iteration(self):
        summary = self.get_summary()
        print("\n--- Iteration Metrics ---")
        for key, val in summary["timers"].items():
            print(f"{key.replace('_', ' ').capitalize()}: {val:.2f}s")
        for key, val in summary["frames"].items():
            print(f"Frames ({key.upper()}): {val}")
        if self.evaluations:
            last = self.evaluations[-1]
            print(f"Eval Return: {last['return_mean']:.2f} +/- {last['return_std']:.2f}")
        print("-------------------------\n")
