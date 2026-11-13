"""
baselines.py
=============
Two non-learned baselines used in the comparison table/figures.

Round Robin: cycles blindly through the three Fog nodes and never
considers RAM, queue depth, or QoS priority -- this is intentional: it is
what makes RR a fair, weak baseline. (It also therefore never sends
anything to the Cloud, which matches "static, ignores congestion"
behaviour reported in real papers; if you want an RR variant that also
cycles the Cloud in, set `include_cloud=True`.)

FCFS (First-Come-First-Served): see the FCFSPolicy docstring below for
the exact, deliberately-disclosed definition used here -- "FCFS" is
overloaded in the scheduling literature, so the specific rule this
baseline implements is spelled out rather than left implicit.
"""
from __future__ import annotations
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from fog_cloud_env import CRASH_UTIL_THRESHOLD


class RoundRobinPolicy:
    def __init__(self, n_fog_nodes: int = 3, include_cloud: bool = False):
        self.n_actions = n_fog_nodes + (1 if include_cloud else 0)
        self._counter = 0

    def reset(self):
        self._counter = 0

    def predict(self, obs, deterministic=True):
        action = self._counter % self.n_actions
        self._counter += 1
        return action, None


class FCFSPolicy:
    """First-Come-First-Served baseline.

    Definition used here (stated explicitly since "FCFS" means different
    things in different scheduling papers): every policy in this harness,
    including Round Robin, DQN, and PPO, already processes tasks strictly
    in arrival order -- the environment feeds them one at a time, in
    sequence, to whichever policy is being evaluated. What FCFS
    specifically contributes as a *placement rule* on top of that shared
    arrival order is a fixed node-preference list with greedy first-fit:
    each task is assigned to the first node in the preference order
    [Fog-1, Fog-2, Fog-3, Cloud] whose current utilisation is below the
    environment's own crash-violation threshold (imported from
    fog_cloud_env so the two stay consistent), reading that utilisation
    directly off the last three elements of the observation vector
    (mu_F1, mu_F2, mu_F3). If all three Fog nodes are at or above the
    threshold, the task overflows to Cloud.

    Like RoundRobinPolicy, FCFS is deliberately QoS-blind (it never reads
    the priority bit obs[0]) and does not perform a full RAM-feasibility
    check against each node's absolute capacity -- it only checks
    utilisation. This is consistent with RR also being an intentionally
    "dumb" baseline, and means FCFS can still incur capacity/crash
    violations exactly like RR does; that is expected, not a bug, and is
    itself a useful comparison point in the Crash_Violation_Rate column.
    """

    def __init__(self, n_fog_nodes: int = 3, util_threshold: float = CRASH_UTIL_THRESHOLD):
        self.n_fog_nodes = n_fog_nodes
        self.util_threshold = util_threshold

    def reset(self):
        pass  # stateless -- FCFS has no counters or memory to reset

    def predict(self, obs, deterministic=True):
        # obs layout: [priority, ram_norm, payload_norm, lpred, mu_F1, mu_F2, mu_F3]
        utils = obs[-self.n_fog_nodes:]
        for node_idx in range(self.n_fog_nodes):
            if utils[node_idx] < self.util_threshold:
                return node_idx, None
        return self.n_fog_nodes, None  # all Fog nodes busy -> overflow to Cloud


def run_episode(env, policy, is_sb3=False):
    """Runs a full pass over env.tasks with the given policy and returns
    the per-step info log (list of dicts) produced by FogCloudEnv."""
    obs, _ = env.reset()
    if not is_sb3 and hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        if is_sb3:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.info_log
