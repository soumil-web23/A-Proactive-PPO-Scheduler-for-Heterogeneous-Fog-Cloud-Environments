"""
fog_cloud_env.py
=================
Section 4.3 / 4.4 of the report as a Gymnasium MDP.

4 discrete actions: 0=Fog-1 (high perf), 1=Fog-2 (eco), 2=Fog-3 (eco), 3=Cloud.
7-dim observation: [P, Rreq_norm, Sdata_norm, Lpred, mu_F1, mu_F2, mu_F3]
Reward: Algorithm 2 of the report -- negative weighted (latency, energy, cost),
with a hard +10 raw penalty for QoS violations (Priority-1 task sent to Cloud)
and for node-crash violations (RAM overflow or >90% utilisation).

Every constant below (MIPS, power draw, cost rates, normalisation bounds) is
declared once at the top so you can tune them without touching the MDP logic.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------- constants
NODES = [
    dict(name="Fog-1", mips=5000, ram_mb=8192, active_w=150.0, idle_w=15.0,
         cpu_cost_per_mi=0.0004, is_cloud=False, prop_delay_s=0.002),
    dict(name="Fog-2", mips=2000, ram_mb=4096, active_w=80.0, idle_w=8.0,
         cpu_cost_per_mi=0.0002, is_cloud=False, prop_delay_s=0.003),
    dict(name="Fog-3", mips=2000, ram_mb=4096, active_w=80.0, idle_w=8.0,
         cpu_cost_per_mi=0.0002, is_cloud=False, prop_delay_s=0.003),
    dict(name="Cloud", mips=100000, ram_mb=10**9, active_w=0.0, idle_w=0.0,
         cpu_cost_per_mi=0.00005, is_cloud=True, prop_delay_s=0.100,
         bandwidth_cost_per_kb=0.0015),
]
N_NODES = len(NODES)

COMPUTE_INTENSITY_MI_PER_KB = 0.8   # million-instructions represented per KB payload
UTIL_NORM_S = 5.0                   # backlog seconds treated as 100% utilised
QOS_VIOLATION_PENALTY = 10.0        # raw seconds added to latency penalty
CRASH_VIOLATION_PENALTY = 10.0
CRASH_UTIL_THRESHOLD = 0.90

# Cloud draws 0 local *compute* power in this sim (it's off-prem), which is
# realistic -- but earlier that made Cloud a free lunch for latency-tolerant
# tasks once you (correctly) asked to drop the cost term from the reward,
# since cost was the only thing discouraging cloud offload. WAN data
# transfer has a real, well-documented energy footprint though, so we now
# charge Cloud-routed tasks a small transfer-energy cost proportional to
# payload size instead. This is more physically honest than a pure
# software "cost" term anyway, and keeps the multi-node scheduling problem
# non-trivial (i.e. the agent still has a real reason to prefer Fog for
# most traffic, not just dump everything on Cloud). Tune if you disagree
# with the magnitude -- it's a placeholder like the other watt figures.
CLOUD_TRANSFER_W_PER_KB = 0.015

NORM_LATENCY_S = 2.0
NORM_ENERGY_W = 250.0
NORM_COST_USD = 5.0
NORM_RAM_MB = 4096.0
NORM_PAYLOAD_KB = 4000.0

# Cost is computed and logged per-step for informational/appendix purposes
# only -- it is NOT part of the reward. Latency and energy are now the only
# two optimisation objectives, per your instruction. Re-add W_COST > 0 and
# a pen_cost term in step() below if you ever want it back in the reward.
W_LATENCY, W_ENERGY = 0.6, 0.4

class FogCloudEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tasks_df, lpred_series, max_steps: int | None = None, seed: int | None = None,
                 w_latency: float | None = None, w_energy: float | None = None):
        """tasks_df must already contain columns:
             priority (0/1, from QoSClassifier), ram_mb, payload_kb
           lpred_series: array-like, same length as tasks_df, BiLSTM forecast
             of near-term network load in [0,1], aligned by row index.
           w_latency, w_energy: optional per-instance override of the reward
             weights (module-level W_LATENCY/W_ENERGY are used otherwise).
             Added for the reward-weight sensitivity sweep (see
             sweep_reward_weights.py) so multiple weight settings can be
             trained in one process without touching the module constants
             that every other script relies on."""
        super().__init__()
        assert len(tasks_df) == len(lpred_series), "tasks_df and lpred_series length mismatch"
        self.w_latency = W_LATENCY if w_latency is None else w_latency
        self.w_energy = W_ENERGY if w_energy is None else w_energy
        self.tasks = tasks_df.reset_index(drop=True)
        self.lpred = np.asarray(lpred_series, dtype=np.float32)
        self.max_steps = max_steps or len(self.tasks)

        self.action_space = spaces.Discrete(N_NODES)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self._np_random = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------ helpers
    def _reset_nodes(self):
        self.backlog_s = np.zeros(N_NODES, dtype=np.float64)  # queued work per node
        self.ram_used_mb = np.zeros(N_NODES, dtype=np.float64)

    def _utilisation(self):
        return np.clip(self.backlog_s / UTIL_NORM_S, 0.0, 1.0)

    def _build_obs(self):
        row = self.tasks.iloc[self.idx]
        mu = self._utilisation()
        obs = np.array([
            float(row["priority"]),
            np.clip(row["ram_mb"] / NORM_RAM_MB, 0, 1),
            np.clip(row["payload_kb"] / NORM_PAYLOAD_KB, 0, 1),
            float(np.clip(self.lpred[self.idx], 0, 1)),
            mu[0], mu[1], mu[2],
        ], dtype=np.float32)
        return obs

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_nodes()
        self.idx = 0
        self.step_count = 0
        self.info_log = []
        return self._build_obs(), {}

    def step(self, action: int):
        row = self.tasks.iloc[self.idx]
        node = NODES[action]
        priority = int(row["priority"])
        ram_req = float(row["ram_mb"])
        payload_kb = float(row["payload_kb"])
        length_mi = payload_kb * COMPUTE_INTENSITY_MI_PER_KB

        exec_time_s = length_mi / node["mips"]
        queue_wait_s = self.backlog_s[action] if not node["is_cloud"] else 0.0
        latency_raw = exec_time_s + queue_wait_s + node["prop_delay_s"]

        qos_violation = priority == 1 and node["is_cloud"]
        crash_violation = (
            (not node["is_cloud"]) and
            (ram_req > node["ram_mb"] or self._utilisation()[action] > CRASH_UTIL_THRESHOLD)
        )
        if qos_violation:
            latency_raw += QOS_VIOLATION_PENALTY
        if crash_violation:
            latency_raw += CRASH_VIOLATION_PENALTY

        # ---- energy: active node draws active_w for the exec duration;
        #      other *awake* fog nodes draw idle_w for this tick; a fog
        #      node with empty backlog and not chosen this tick sleeps (0 W).
        #      Cloud draws 0 local compute power but pays a WAN transfer cost.
        energy_raw = 0.0
        if not node["is_cloud"]:
            energy_raw += node["active_w"] * exec_time_s
        else:
            energy_raw += payload_kb * CLOUD_TRANSFER_W_PER_KB
        for i, n in enumerate(NODES):
            if n["is_cloud"] or i == action:
                continue
            was_awake = self.backlog_s[i] > 0
            if was_awake:
                energy_raw += n["idle_w"] * 1.0  # 1 tick of idle draw

        # ---- cost: computed + logged for reference only, NOT in the reward
        cost_raw = length_mi * node["cpu_cost_per_mi"]
        if node["is_cloud"]:
            cost_raw += payload_kb * node["bandwidth_cost_per_kb"]

        # ---- apply task to node state, then drain backlog by one tick
        if not node["is_cloud"]:
            self.backlog_s[action] += exec_time_s
        for i in range(N_NODES):
            self.backlog_s[i] = max(0.0, self.backlog_s[i] - 1.0)

        pen_lat = np.clip(latency_raw / NORM_LATENCY_S, 0, 1)
        pen_eng = np.clip(energy_raw / NORM_ENERGY_W, 0, 1)
        reward = -(self.w_latency * pen_lat + self.w_energy * pen_eng)

        self.info_log.append(dict(
            action=action, node=node["name"], priority=priority,
            latency_s=latency_raw, energy_w=energy_raw, cost_usd=cost_raw,
            qos_violation=qos_violation, crash_violation=crash_violation,
        ))

        self.idx += 1
        self.step_count += 1
        terminated = self.idx >= len(self.tasks)
        truncated = self.step_count >= self.max_steps
        obs = self._build_obs() if not (terminated or truncated) else np.zeros(7, dtype=np.float32)
        info = self.info_log[-1]
        return obs, float(reward), terminated, truncated, info
