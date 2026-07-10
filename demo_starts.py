"""Demo-state episode starts (Backplay-style) for Crafter.

The recorded demos store everything needed to rebuild the env mid-episode:
info['semantic'] is the FULL world material map with object ids overlaid,
plus inventory / achievements / player_pos, and daylight is a pure function
of the step index. restore_env_from_info() rebuilds a live env from one
transition; DemoStartSampler picks restart points, optionally prioritized
toward states just before large TD errors on the demo transitions.

Approximations (all fine for exploration seeding): grass is backfilled under
object cells, mob internals and player facing take defaults, transient
arrows are dropped, and the world RNG is fresh.
"""

import numpy as np


def restore_env_from_info(env, payload):
    """Rebuild env._world / _player from a demo restart payload.

    payload: {'semantic': (64,64) uint8, 'inventory': dict,
              'achievements': dict, 'player_pos': (x, y), 'step': int}
    Returns the first observation, like env.reset().
    """
    from crafter import objects

    sem = np.asarray(payload["semantic"])
    world = env._world

    # Fresh RNG for future spawns; clears object/material maps and chunks.
    world.reset(seed=np.random.randint(2**31 - 1))

    n_mats = len(world._mat_ids)  # includes None at 0
    obj_types = [objects.Player, objects.Cow, objects.Zombie,
                 objects.Skeleton, objects.Arrow, objects.Plant]
    obj_id = {n_mats + i: cls for i, cls in enumerate(obj_types)}

    # Object cells hide the material underneath; backfill grass (mobs and
    # players only stand on walkable tiles).
    grass = world._mat_ids["grass"]
    mat_map = sem.copy()
    obj_cells = sem >= n_mats
    mat_map[obj_cells] = grass
    world._mat_map = mat_map.astype(np.uint8)

    pos = tuple(int(x) for x in payload["player_pos"])
    player = objects.Player(world, pos)
    player.inventory = dict(payload["inventory"])
    player._last_health = player.health
    world.add(player)

    # Restore achievement counters and the env unlock set so only NEW
    # achievements yield reward after the restart (Backplay semantics).
    for name, count in payload["achievements"].items():
        player.achievements[name] = count
    env._unlocked = {n for n, c in payload["achievements"].items() if c > 0}

    for cell in np.argwhere(obj_cells):
        cell = tuple(int(c) for c in cell)
        if cell == pos:
            continue
        cls = obj_id[int(sem[cell])]
        if cls in (objects.Player, objects.Arrow):
            continue
        if cls in (objects.Zombie, objects.Skeleton):
            world.add(cls(world, cell, player))
        else:
            world.add(cls(world, cell))

    env._player = player
    env._last_health = player.health
    env._step = int(payload["step"])
    env._update_time()
    return env._obs()


def _episode_transitions(item):
    if isinstance(item, dict) and "transitions" in item:
        return item["transitions"]
    return item


def _rolling_max(x, window):
    """max over x[i:i+window] for each i, same length as x (tail zero-padded)."""
    if len(x) == 0:
        return x
    pad = np.pad(x, (0, window - 1), constant_values=0.0)
    return np.lib.stride_tricks.sliding_window_view(pad, window).max(axis=1)


class DemoStartSampler:
    """Samples demo restart payloads, prioritized toward restart points whose
    following `lookahead` demo transitions contain large TD errors (i.e. start
    the agent just BEFORE the states its value function gets most wrong).

    Restart point (ep, t) = the state AFTER transition t (its info dict), so
    it enters the episode at demo step t+1. Points within `min_remaining`
    steps of the demo's end are excluded (nothing left to do there).
    """

    def __init__(self, expert_dataset, alpha=0.6, lookahead=50,
                 min_remaining=50, uniform_mix=0.2, seed=0):
        self.alpha = alpha
        self.lookahead = lookahead
        self.uniform_mix = uniform_mix
        self.rng = np.random.default_rng(seed)

        self.payloads = []      # flat list of restart payloads
        self.point_index = []   # (episode_idx, transition_idx) per payload
        self.ep_lengths = []
        for ep_i, item in enumerate(expert_dataset):
            transitions = _episode_transitions(item)
            T = len(transitions)
            self.ep_lengths.append(T)
            for t in range(0, T - 1 - min_remaining):
                info = transitions[t].get("info") or {}
                if "semantic" not in info:
                    continue
                self.payloads.append({
                    "semantic": np.asarray(info["semantic"], dtype=np.uint8),
                    "inventory": dict(info["inventory"]),
                    "achievements": dict(info["achievements"]),
                    "player_pos": tuple(int(x) for x in info["player_pos"]),
                    "step": t + 1,
                })
                self.point_index.append((ep_i, t))
        self.num_episodes = len(self.ep_lengths)
        self._probs = None  # uniform until TD errors arrive
        print(f"[DemoStarts] {len(self.payloads)} restart points from "
              f"{self.num_episodes} episodes (alpha={alpha}, "
              f"lookahead={lookahead}).")

    def set_td_errors(self, per_episode_abs_deltas):
        """per_episode_abs_deltas: list (len == num_episodes, insertion order
        matching the dataset) of 1-D arrays of |TD error| per transition."""
        if self.alpha <= 0 or not self.payloads:
            return
        assert len(per_episode_abs_deltas) == self.num_episodes, (
            f"got TD errors for {len(per_episode_abs_deltas)} episodes, "
            f"expected {self.num_episodes}")
        rolled = [_rolling_max(np.asarray(d, dtype=np.float64), self.lookahead)
                  for d in per_episode_abs_deltas]
        w = np.empty(len(self.payloads))
        for k, (ep_i, t) in enumerate(self.point_index):
            r = rolled[ep_i]
            # restart enters at demo step t+1; look ahead from there
            w[k] = r[t + 1] if t + 1 < len(r) else r[-1]
        p = (w + 1e-2) ** self.alpha
        p /= p.sum()
        n = len(p)
        self._probs = self.uniform_mix / n + (1.0 - self.uniform_mix) * p

    def sample(self):
        if not self.payloads:
            return None
        k = self.rng.choice(len(self.payloads), p=self._probs)
        return self.payloads[int(k)]
