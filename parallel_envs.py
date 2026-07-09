"""Multiprocess Crafter environment workers for fast RL experience collection.

env.step dominates collection time (~2ms/step vs ~0.4ms for the batched policy
forward), so the envs run in worker processes while the main process batches
all policy forwards on the GPU.
"""

import multiprocessing as mp


def _worker(remote):
    import crafter

    env = crafter.Env()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            # Strip the large per-step arrays; downstream only reads achievements.
            info_lite = {
                "achievements": info.get("achievements"),
                "discount": info.get("discount"),
            }
            remote.send((obs, reward, done, info_lite))
        elif cmd == "reset":
            obs = env.reset()
            remote.send(obs)
        elif cmd == "close":
            remote.close()
            break


class ParallelCrafterEnvs:
    def __init__(self, num_envs):
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.procs = [
            ctx.Process(target=_worker, args=(wr,), daemon=True)
            for wr in work_remotes
        ]
        for p in self.procs:
            p.start()
        for wr in work_remotes:
            wr.close()

    def reset_all(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def reset_one(self, i):
        self.remotes[i].send(("reset", None))
        return self.remotes[i].recv()

    def reset_async(self, i):
        """Fire a reset without waiting; crafter world generation takes ~100ms+,
        so the other envs keep stepping while this one regenerates."""
        self.remotes[i].send(("reset", None))

    def poll_reset(self, i):
        """Returns the new obs if the async reset finished, else None."""
        if self.remotes[i].poll():
            return self.remotes[i].recv()
        return None

    def step(self, actions):
        """actions: dict {env_idx: action}. Steps only those envs, in parallel."""
        for i, a in actions.items():
            self.remotes[i].send(("step", int(a)))
        return {i: self.remotes[i].recv() for i in actions}

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
