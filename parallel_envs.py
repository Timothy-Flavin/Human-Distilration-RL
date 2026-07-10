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
            if data is None:
                obs = env.reset()
            else:
                # Demo-state restart: data is a DemoStartSampler payload.
                from demo_starts import restore_env_from_info

                obs = restore_env_from_info(env, data)
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

    def reset_all(self, payloads=None):
        """payloads: optional {env_idx: demo-start payload}; envs not in the
        dict (or with a None value) do a normal fresh-world reset."""
        payloads = payloads or {}
        for i, r in enumerate(self.remotes):
            r.send(("reset", payloads.get(i)))
        return [r.recv() for r in self.remotes]

    def reset_one(self, i, payload=None):
        self.remotes[i].send(("reset", payload))
        return self.remotes[i].recv()

    def reset_async(self, i, payload=None):
        """Fire a reset without waiting; crafter world generation takes ~100ms+,
        so the other envs keep stepping while this one regenerates. An optional
        payload restarts the env from a demo state instead of a fresh world."""
        self.remotes[i].send(("reset", payload))

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
